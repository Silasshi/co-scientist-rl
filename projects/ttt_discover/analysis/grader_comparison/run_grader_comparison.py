#!/usr/bin/env python3
"""Grader comparison: grade existing plans with different-sized models,
compute correlation with 235B reference scores.

Usage:
    python projects/ttt_discover/analysis/grader_comparison/run_grader_comparison.py

No training, no plan generation — pure grading comparison.
"""

import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import tinker
import tinker.types as types
from tinker_cookbook import model_info, renderers

from co_scientist.shared.ten_signal_reward import (
    SIGNALS,
    SIGNAL_WEIGHTS,
    aggregate_reward,
    build_single_signal_prompt,
    normalize_score,
    parse_scores,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Config ──────────────────────────────────────────────────────────────────

BUFFER_PATH = PROJECT_ROOT / "projects/ttt_discover/runs/_archive/2026_04_buffer_ttt/cr_v5_paragraph_edit/buffer.jsonl"
GOAL_PATH = PROJECT_ROOT / "projects/ttt_discover/analysis/sanity_check/research_goal.txt"
OUTPUT_DIR = Path(__file__).parent
N_SAMPLE = 20           # plans to sample
GRADER_MAX_TOKENS = 2048
GRADER_TEMPERATURE = 0.0
WAVE_SIZE = 16          # parallel grader calls per wave

CANDIDATE_GRADERS = [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-32B",
]

SIGNAL_IDS = [s.id for s in SIGNALS]


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_goal() -> str:
    return GOAL_PATH.read_text().strip()


def sample_plans(n: int, seed: int = 42) -> list[dict]:
    """Sample n diverse plans from the CR-v5 buffer, spanning the reward range."""
    with open(BUFFER_PATH) as f:
        entries = [json.loads(line) for line in f]

    # Filter: must have valid signal_vector and non-zero reward
    entries = [e for e in entries if e.get("aggregate_reward", 0) > 0 and e.get("signal_vector")]

    # Stratified sample by reward bins
    bins = [(0.0, 0.45), (0.45, 0.6), (0.6, 0.75), (0.75, 0.85), (0.85, 1.01)]
    per_bin = max(1, n // len(bins))
    rng = random.Random(seed)

    sampled = []
    for lo, hi in bins:
        in_bin = [e for e in entries if lo <= e["aggregate_reward"] < hi]
        k = min(per_bin, len(in_bin))
        if k > 0:
            sampled.extend(rng.sample(in_bin, k))

    # If not enough, fill from remaining
    remaining_ids = {id(e) for e in sampled}
    pool = [e for e in entries if id(e) not in remaining_ids]
    while len(sampled) < n and pool:
        sampled.append(pool.pop(rng.randrange(len(pool))))

    logger.info(f"Sampled {len(sampled)} plans: rewards {[round(e['aggregate_reward'], 3) for e in sampled]}")
    return sampled


def grade_plans_with_model(
    model_name: str,
    plans: list[dict],
    goal: str,
) -> list[dict]:
    """Grade all plans with a given grader model. Returns list of {signal_id: score}."""
    from co_scientist.shared.api_profiles import create_service_client

    logger.info(f"=== Grading with {model_name} ===")

    service_client = create_service_client()
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    tokenizer = sampling_client.get_tokenizer()

    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    stop_seqs = renderer.get_stop_sequences()

    # Build all prompts
    all_tasks = []  # (plan_idx, signal_spec, prompt)
    for i, entry in enumerate(plans):
        plan_text = entry["plan_text"]
        for signal_spec in SIGNALS:
            prompt = build_single_signal_prompt(goal, plan_text, signal_spec)
            all_tasks.append((i, signal_spec, prompt))

    logger.info(f"Total grading calls: {len(all_tasks)} ({len(plans)} plans × {len(SIGNALS)} signals)")

    # Launch in waves
    results = {}  # (plan_idx, signal_id) -> score
    for wave_start in range(0, len(all_tasks), WAVE_SIZE):
        wave = all_tasks[wave_start : wave_start + WAVE_SIZE]
        wave_num = wave_start // WAVE_SIZE + 1
        total_waves = (len(all_tasks) + WAVE_SIZE - 1) // WAVE_SIZE
        logger.info(f"  Wave {wave_num}/{total_waves}: launching {len(wave)} calls...")

        futures = []
        for plan_idx, signal_spec, prompt in wave:
            convo = [{"role": "user", "content": prompt}]
            model_input = renderer.build_generation_prompt(convo)
            future = sampling_client.sample(
                model_input,
                num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=GRADER_MAX_TOKENS,
                    temperature=GRADER_TEMPERATURE,
                    stop=stop_seqs,
                ),
            )
            futures.append((plan_idx, signal_spec.id, future))

        # Collect
        for plan_idx, signal_id, future in futures:
            try:
                response = future.result(timeout=120)
                text = renderers.get_text_content(renderer.parse_response(response.sequences[0].tokens)[0])
                parsed = parse_scores(text)
                info = parsed.get(signal_id, {})
                score = info.get("score")
                results[(plan_idx, signal_id)] = score
            except Exception as e:
                logger.warning(f"  Failed: plan {plan_idx}, {signal_id}: {e}")
                results[(plan_idx, signal_id)] = None

    # Assemble per-plan score dicts
    plan_scores = []
    for i in range(len(plans)):
        scores = {sid: results.get((i, sid)) for sid in SIGNAL_IDS}
        plan_scores.append(scores)

    return plan_scores


def compute_correlations(
    ref_scores: list[dict],
    candidate_scores: list[dict],
) -> dict:
    """Compute Spearman and Pearson correlation between reference and candidate scores."""
    from scipy import stats

    # Per-signal correlation
    per_signal = {}
    for sid in SIGNAL_IDS:
        ref_vals = [s.get(sid) for s in ref_scores]
        cand_vals = [s.get(sid) for s in candidate_scores]
        # Filter out None pairs
        pairs = [(r, c) for r, c in zip(ref_vals, cand_vals) if r is not None and c is not None]
        if len(pairs) < 3:
            per_signal[sid] = {"spearman": None, "pearson": None, "n": len(pairs)}
            continue
        r_vals, c_vals = zip(*pairs)
        spearman = stats.spearmanr(r_vals, c_vals)
        pearson = stats.pearsonr(r_vals, c_vals)
        per_signal[sid] = {
            "spearman": round(spearman.statistic, 3),
            "pearson": round(pearson.statistic, 3),
            "n": len(pairs),
            "ref_mean": round(mean(r_vals), 2),
            "cand_mean": round(mean(c_vals), 2),
        }

    # Aggregate correlation
    ref_agg = [aggregate_reward(s) for s in ref_scores]
    cand_agg = [aggregate_reward(s) for s in candidate_scores]
    pairs = [(r, c) for r, c in zip(ref_agg, cand_agg) if r is not None and c is not None]
    if len(pairs) >= 3:
        r_vals, c_vals = zip(*pairs)
        sp = stats.spearmanr(r_vals, c_vals)
        pe = stats.pearsonr(r_vals, c_vals)
        agg_corr = {
            "spearman": round(sp.statistic, 3),
            "pearson": round(pe.statistic, 3),
            "n": len(pairs),
            "ref_mean": round(mean(r_vals), 3),
            "cand_mean": round(mean(c_vals), 3),
        }
    else:
        agg_corr = {"spearman": None, "pearson": None, "n": len(pairs)}

    return {"per_signal": per_signal, "aggregate": agg_corr}


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    goal = load_goal()
    plans = sample_plans(N_SAMPLE)

    # Extract 235B reference scores from buffer entries
    ref_scores = []
    for entry in plans:
        sv = entry["signal_vector"]
        # signal_vector is {signal_id: int_score}
        ref_scores.append(sv)

    logger.info(f"Reference (235B) aggregate scores: {[round(aggregate_reward(s), 3) for s in ref_scores]}")

    # Grade with each candidate model
    all_results = {}
    for model_name in CANDIDATE_GRADERS:
        t0 = time.time()
        try:
            cand_scores = grade_plans_with_model(model_name, plans, goal)
            elapsed = time.time() - t0
            logger.info(f"  {model_name}: completed in {elapsed:.1f}s")

            corr = compute_correlations(ref_scores, cand_scores)
            all_results[model_name] = {
                "scores": cand_scores,
                "correlations": corr,
                "elapsed_s": round(elapsed, 1),
            }
        except Exception as e:
            logger.error(f"  {model_name}: FAILED - {e}")
            all_results[model_name] = {"error": str(e)}

    # ── Print summary ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("GRADER COMPARISON RESULTS")
    print("=" * 80)
    print(f"Reference: Qwen3-235B-A22B-Instruct-2507 (scores from CR-v5 buffer)")
    print(f"Plans sampled: {N_SAMPLE}")
    print()

    # Aggregate correlation table
    print(f"{'Model':<30} {'Spearman':>10} {'Pearson':>10} {'Ref Mean':>10} {'Cand Mean':>10} {'Time(s)':>8}")
    print("-" * 80)
    for model_name, res in all_results.items():
        if "error" in res:
            print(f"{model_name:<30} {'ERROR':>10}")
            continue
        agg = res["correlations"]["aggregate"]
        sp = f"{agg['spearman']:.3f}" if agg["spearman"] is not None else "N/A"
        pe = f"{agg['pearson']:.3f}" if agg["pearson"] is not None else "N/A"
        rm = f"{agg['ref_mean']:.3f}" if agg.get("ref_mean") is not None else "N/A"
        cm = f"{agg['cand_mean']:.3f}" if agg.get("cand_mean") is not None else "N/A"
        print(f"{model_name:<30} {sp:>10} {pe:>10} {rm:>10} {cm:>10} {res['elapsed_s']:>8.1f}")

    print()
    # Per-signal detail
    print("Per-signal Spearman correlation:")
    header = f"{'Signal':<20}" + "".join(f"{m.split('/')[-1]:>15}" for m in CANDIDATE_GRADERS)
    print(header)
    print("-" * len(header))
    for sid in SIGNAL_IDS:
        row = f"{sid:<20}"
        for model_name in CANDIDATE_GRADERS:
            res = all_results.get(model_name, {})
            if "error" in res:
                row += f"{'ERR':>15}"
            else:
                val = res["correlations"]["per_signal"].get(sid, {}).get("spearman")
                row += f"{val:>15.3f}" if val is not None else f"{'N/A':>15}"
        print(row)

    # Per-signal mean score comparison (detect inflation)
    print()
    print("Per-signal mean score (detect inflation):")
    header = f"{'Signal':<20} {'235B ref':>10}" + "".join(f"{m.split('/')[-1]:>15}" for m in CANDIDATE_GRADERS)
    print(header)
    print("-" * len(header))
    for sid in SIGNAL_IDS:
        ref_vals = [s.get(sid) for s in ref_scores if s.get(sid) is not None]
        ref_m = mean(ref_vals) if ref_vals else 0
        row = f"{sid:<20} {ref_m:>10.2f}"
        for model_name in CANDIDATE_GRADERS:
            res = all_results.get(model_name, {})
            if "error" in res:
                row += f"{'ERR':>15}"
            else:
                cand_vals = [s.get(sid) for s in res["scores"] if s.get(sid) is not None]
                cm = mean(cand_vals) if cand_vals else 0
                diff = cm - ref_m
                row += f"{cm:>10.2f}({diff:+.1f})"
            pass
        print(row)

    # Save raw results
    out_path = OUTPUT_DIR / "results.json"
    # Convert scores to serializable format
    save_data = {}
    for model_name, res in all_results.items():
        if "error" in res:
            save_data[model_name] = {"error": res["error"]}
        else:
            save_data[model_name] = {
                "correlations": res["correlations"],
                "elapsed_s": res["elapsed_s"],
                "per_plan_scores": res["scores"],
            }
    save_data["_reference_235B"] = {
        "per_plan_scores": ref_scores,
        "per_plan_aggregate": [round(aggregate_reward(s), 4) for s in ref_scores],
    }
    save_data["_plans"] = [
        {"idx": i, "aggregate_reward": round(p["aggregate_reward"], 4), "word_count": p.get("word_count", 0)}
        for i, p in enumerate(plans)
    ]

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
