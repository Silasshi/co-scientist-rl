#!/usr/bin/env python3
"""Signal validity test: grade reference solutions with our signals.

Hypothesis: If our signals are valid, high-quality reference solutions should
score high (aggregate ≥ 0.7). If they don't, signals are miscalibrated.

Data sources:
  1. facebook/research-plan-gen: ml, arxiv, pubmed (sample N each)
  2. Papers we've read: extract Goal + Method from shared/papers/by_topic/*/*/analysis.md

Output: projects/ttt_discover/analysis/signal_validity/
  - references.jsonl            (all {goal, reference_solution, source} tuples)
  - grading_results.jsonl       (per-reference signal scores + aggregate)
  - summary.md                  (distribution + per-signal analysis)
"""

import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import tinker
import tinker.types as types
from datasets import load_dataset
from tinker_cookbook import model_info, renderers
from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.ten_signal_reward import (
    SIGNALS, SIGNAL_WEIGHTS,
    aggregate_reward, build_single_signal_prompt, parse_scores,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)
logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
BASE = Path(__file__).resolve().parents[2]
REFS_PATH = BASE / "data" / "refs" / "references.jsonl"
RESULTS_PATH = BASE / "archive" / "grading_results.jsonl"
SUMMARY_PATH = HERE / "summary.md"

GRADER_MODEL = "Qwen/Qwen3-30B-A3B"  # 30B: 5x faster than 235B, Spearman=0.827 vs 235B (validated)
N_PER_SUBSET = 50           # 50 each from ml/arxiv/pubmed = 150 (scale later if needed)
WAVE_SIZE = 4               # 4 refs * 9 signals = 36 concurrent; 30B handles this fine
SIGNAL_IDS = [s.id for s in SIGNALS]


# ── Data collection ────────────────────────────────────────────────────────

def load_facebook_references(config: str, n: int, seed: int = 42) -> list[dict]:
    """Load n references from facebook/research-plan-gen (config = ml/arxiv/pubmed)."""
    ds = load_dataset("facebook/research-plan-gen", config, split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    refs = []
    for idx in indices:
        ex = ds[idx]
        ref = {
            "source": f"facebook_{config}",
            "source_id": f"{config}_{ex.get('q_id', idx)}",
            "goal": ex.get("Goal", "").strip(),
            "reference_solution": ex.get("Reference solution", "").strip(),
            "subdomain": ex.get("Subdomain", "") or ex.get("Category", ""),
        }
        # Filter: must have non-empty goal and solution
        if len(ref["goal"]) > 100 and len(ref["reference_solution"]) > 200:
            refs.append(ref)
    logger.info(f"Loaded {len(refs)} references from facebook_{config}")
    return refs


def load_paper_references(papers_dir: Path) -> list[dict]:
    """Extract reference solutions from analysis.md files in shared/papers/by_topic."""
    refs = []
    for analysis_file in papers_dir.rglob("analysis.md"):
        content = analysis_file.read_text()
        # Extract frontmatter
        title_m = re.search(r'^title:\s*"([^"]+)"', content, re.MULTILINE)
        short_name_m = re.search(r'^short_name:\s*(\S+)', content, re.MULTILINE)
        name = short_name_m.group(1) if short_name_m else analysis_file.parent.name

        # Extract sections by markdown header
        sections = {}
        for m in re.finditer(r'^##\s+\d*\.?\s*([^\n]+)\n(.+?)(?=^##\s|\Z)', content, re.MULTILINE | re.DOTALL):
            header = m.group(1).strip().lower()
            body = m.group(2).strip()
            sections[header] = body

        # Find "Problem Solved" / "Core Idea" as goal; "Main Pipeline" + "Key Methodologies" as solution
        goal_parts = []
        for k, v in sections.items():
            if 'problem' in k or 'core idea' in k:
                goal_parts.append(v)
        solution_parts = []
        for k, v in sections.items():
            if 'pipeline' in k or 'methodolog' in k or 'method' in k or 'significan' in k:
                solution_parts.append(v)

        if not goal_parts or not solution_parts:
            continue

        goal = "\n\n".join(goal_parts)
        solution = "\n\n".join(solution_parts)

        # Format as a proper research plan
        ref = {
            "source": "papers_analysis",
            "source_id": name,
            "goal": goal,
            "reference_solution": solution,
            "subdomain": analysis_file.parent.parent.name,
        }
        if len(ref["goal"]) > 100 and len(ref["reference_solution"]) > 200:
            refs.append(ref)
    logger.info(f"Loaded {len(refs)} references from papers")
    return refs


# ── Grading ────────────────────────────────────────────────────────────────

def grade_one_plan_all_signals(
    ref: dict,
    grader_client,
    renderer,
    stop_seqs,
) -> dict:
    """Grade one reference against all 9 signals. Returns {signal_id: score}."""
    plan = ref["reference_solution"]
    goal = ref["goal"]
    futures = {}
    for sig in SIGNALS:
        prompt = build_single_signal_prompt(goal, plan, sig)
        convo = [{"role": "user", "content": prompt}]
        mi = renderer.build_generation_prompt(convo)
        future = grader_client.sample(
            mi, num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=2048, temperature=0.0, stop=stop_seqs,
            ),
        )
        futures[sig.id] = future

    scores = {}
    for sig_id, future in futures.items():
        try:
            resp = future.result(timeout=180)
            text = renderers.get_text_content(renderer.parse_response(resp.sequences[0].tokens)[0])
            parsed = parse_scores(text)
            scores[sig_id] = parsed.get(sig_id, {}).get("score")
        except Exception as e:
            logger.warning(f"Failed {ref['source_id']} / {sig_id}: {e}")
            scores[sig_id] = None
    return scores


def grade_all_references(refs: list[dict], grader_client, renderer, stop_seqs) -> list[dict]:
    """Grade all references, writing results to RESULTS_PATH as we go.

    Resumes from previous run if RESULTS_PATH exists.
    """
    done_ids = set()
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            for line in f:
                r = json.loads(line)
                done_ids.add(r["source_id"])
        logger.info(f"Resuming: {len(done_ids)} already graded")

    results = []
    pending = [r for r in refs if r["source_id"] not in done_ids]
    logger.info(f"Grading {len(pending)} remaining references (wave_size={WAVE_SIZE})")

    t_start = time.time()
    with open(RESULTS_PATH, "a") as fout:
        for i in range(0, len(pending), WAVE_SIZE):
            wave = pending[i:i + WAVE_SIZE]
            # Launch all signals for all refs in this wave in parallel
            wave_futures = []
            for ref in wave:
                plan = ref["reference_solution"]
                goal = ref["goal"]
                per_ref_futures = {}
                for sig in SIGNALS:
                    prompt = build_single_signal_prompt(goal, plan, sig)
                    convo = [{"role": "user", "content": prompt}]
                    mi = renderer.build_generation_prompt(convo)
                    future = grader_client.sample(
                        mi, num_samples=1,
                        sampling_params=types.SamplingParams(
                            max_tokens=2048, temperature=0.0, stop=stop_seqs,
                        ),
                    )
                    per_ref_futures[sig.id] = future
                wave_futures.append((ref, per_ref_futures))

            # Collect
            for ref, per_ref_futures in wave_futures:
                scores = {}
                for sig_id, future in per_ref_futures.items():
                    try:
                        resp = future.result(timeout=90)
                        text = renderers.get_text_content(renderer.parse_response(resp.sequences[0].tokens)[0])
                        parsed = parse_scores(text)
                        scores[sig_id] = parsed.get(sig_id, {}).get("score")
                    except Exception as e:
                        logger.warning(f"Failed {ref['source_id']} / {sig_id}: {type(e).__name__}: {str(e)[:80]}")
                        scores[sig_id] = None
                agg = aggregate_reward(scores)
                out = {
                    "source_id": ref["source_id"],
                    "source": ref["source"],
                    "signals": scores,
                    "aggregate": agg,
                    "word_count": len(ref["reference_solution"].split()),
                }
                fout.write(json.dumps(out) + "\n")
                fout.flush()
                results.append(out)

            elapsed = time.time() - t_start
            n_done = i + len(wave)
            eta = elapsed / n_done * (len(pending) - n_done) if n_done > 0 else 0
            logger.info(f"  [{n_done}/{len(pending)}] elapsed={elapsed:.0f}s ETA={eta:.0f}s last_agg={agg:.3f}")

    return results


# ── Analysis ────────────────────────────────────────────────────────────────

def analyze_results():
    """Read grading_results.jsonl and write summary.md."""
    results = []
    with open(RESULTS_PATH) as f:
        for line in f:
            results.append(json.loads(line))

    if not results:
        logger.warning("No results to analyze")
        return

    # Overall distribution
    aggs = [r["aggregate"] for r in results]
    n = len(aggs)

    # By source
    by_source = {}
    for r in results:
        by_source.setdefault(r["source"], []).append(r["aggregate"])

    # Per-signal distribution
    per_signal = {sid: [] for sid in SIGNAL_IDS}
    for r in results:
        for sid, s in r["signals"].items():
            if s is not None:
                per_signal[sid].append(s)

    # Write summary
    lines = [
        "# Signal Validity Test Results\n",
        f"Grader: {GRADER_MODEL}",
        f"Total references graded: {n}",
        "",
        "## Distribution of aggregate scores (higher is better)",
        f"- Mean: {mean(aggs):.3f}",
        f"- Median: {median(aggs):.3f}",
        f"- Std: {stdev(aggs) if len(aggs) > 1 else 0:.3f}",
        f"- Min: {min(aggs):.3f}",
        f"- Max: {max(aggs):.3f}",
        "",
        "## Pass-rate thresholds",
        f"- Score >= 0.9 : {sum(1 for a in aggs if a >= 0.9) / n:.1%} ({sum(1 for a in aggs if a >= 0.9)}/{n})",
        f"- Score >= 0.8 : {sum(1 for a in aggs if a >= 0.8) / n:.1%} ({sum(1 for a in aggs if a >= 0.8)}/{n})",
        f"- Score >= 0.7 : {sum(1 for a in aggs if a >= 0.7) / n:.1%} ({sum(1 for a in aggs if a >= 0.7)}/{n})",
        f"- Score >= 0.6 : {sum(1 for a in aggs if a >= 0.6) / n:.1%} ({sum(1 for a in aggs if a >= 0.6)}/{n})",
        f"- Score >= 0.5 : {sum(1 for a in aggs if a >= 0.5) / n:.1%} ({sum(1 for a in aggs if a >= 0.5)}/{n})",
        "",
        "## By source",
        "",
        "| Source | Count | Mean | Median | Std | ≥0.7 | ≥0.8 |",
        "|---|---|---|---|---|---|---|",
    ]
    for src, vals in sorted(by_source.items()):
        lines.append(
            f"| {src} | {len(vals)} | {mean(vals):.3f} | {median(vals):.3f} | "
            f"{stdev(vals) if len(vals) > 1 else 0:.3f} | "
            f"{sum(1 for a in vals if a >= 0.7)/len(vals):.0%} | "
            f"{sum(1 for a in vals if a >= 0.8)/len(vals):.0%} |"
        )

    lines += [
        "",
        "## Per-signal distribution",
        "",
        "| Signal | Weight | Mean | Median | ≥4 count | ≥3 count |",
        "|---|---|---|---|---|---|",
    ]
    for sid in SIGNAL_IDS:
        vals = per_signal[sid]
        if not vals: continue
        w = SIGNAL_WEIGHTS[sid]
        mv = mean(vals)
        md = median(vals)
        ge4 = sum(1 for v in vals if v >= 4)
        ge3 = sum(1 for v in vals if v >= 3)
        lines.append(f"| {sid} | {w:.2f} | {mv:.2f} | {md:.1f} | {ge4/len(vals):.0%} | {ge3/len(vals):.0%} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "**If signals are well-calibrated:** reference plans (which are human-written methodologies from accepted papers) should score >= 0.7 on aggregate.",
        "",
        "**Signals are systematically underrating references if:**",
        "- Median aggregate is << 0.7",
        "- Many individual signals have median score < 4",
        "- The distribution clusters in the 0.5-0.7 range",
        "",
    ]

    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Summary written to {SUMMARY_PATH}")

    # Print headline
    print("\n" + "=" * 70)
    print(f"HEADLINE: {n} references graded with {GRADER_MODEL.split('/')[-1]}")
    print(f"  Mean aggregate: {mean(aggs):.3f}")
    print(f"  % scoring >= 0.7: {sum(1 for a in aggs if a >= 0.7) / n:.0%}")
    print(f"  % scoring >= 0.8: {sum(1 for a in aggs if a >= 0.8) / n:.0%}")
    print("=" * 70)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    # Collect references
    refs_all = []
    if REFS_PATH.exists():
        logger.info(f"Loading existing references from {REFS_PATH}")
        with open(REFS_PATH) as f:
            refs_all = [json.loads(l) for l in f]
    else:
        for config in ["ml", "arxiv", "pubmed"]:
            refs_all.extend(load_facebook_references(config, N_PER_SUBSET))
        refs_all.extend(load_paper_references(PROJECT_ROOT / "shared" / "papers" / "by_topic"))

        with open(REFS_PATH, "w") as f:
            for r in refs_all:
                f.write(json.dumps(r) + "\n")
        logger.info(f"Saved {len(refs_all)} references to {REFS_PATH}")

    logger.info(f"Total references: {len(refs_all)}")
    logger.info(f"Sources: {sorted(set(r['source'] for r in refs_all))}")

    # Setup grader
    service_client = create_service_client()
    grader_client = service_client.create_sampling_client(base_model=GRADER_MODEL)
    tokenizer = grader_client.get_tokenizer()
    renderer_name = model_info.get_recommended_renderer_name(GRADER_MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    stop_seqs = renderer.get_stop_sequences()

    logger.info(f"Grader: {GRADER_MODEL}, renderer: {renderer_name}")

    # Grade
    grade_all_references(refs_all, grader_client, renderer, stop_seqs)

    # Analyze
    analyze_results()


if __name__ == "__main__":
    main()
