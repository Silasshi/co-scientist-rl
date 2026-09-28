#!/usr/bin/env python3
"""Phase 1.5 smoke test: CR-v6 locus path end-to-end on 10 buffer plans.

Covers verification gates 2+3:
  Gate 2: grader-cost regression (tokens/sec with locus ≤ 1.5× baseline)
  Gate 3: locus parse → revision → apply → re-grade cleanly

Uses Phase 0 sample (sample_s3_pilot.jsonl) as data source — these plans
already have full signal_vectors so we can identify real bottlenecks.

Output: prints summary to stdout; writes per-plan JSONL to data/locus_pilot/
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.ten_signal_reward import (
    SIGNALS,
    build_single_signal_prompt,
    parse_locus,
    parse_scores,
)
from co_scientist.ttt_discover.train_critique_revise import (
    apply_locus_revisions,
    build_locus_revision_prompt,
    identify_bottleneck,
    parse_locus_revisions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Config ---
MODEL = "Qwen/Qwen3-30B-A3B"
N_PLANS = 10
SKIP_SIGNALS = {"S3_positioning", "S4_significance"}
LOCUS_MAX_TOKENS = 6144
BASELINE_MAX_TOKENS = 4096
REVISION_MAX_TOKENS = 2048
GRADER_TEMP = 0.0
REVISION_TEMP = 1.0

BASE = Path(__file__).resolve().parents[1]  # signal_validity/
SAMPLE_PATH = BASE / "data" / "locus_pilot" / "sample_s3_pilot.jsonl"
GOAL_PATH = PROJECT_ROOT / "projects" / "ttt_discover" / "analysis" / "sanity_check" / "research_goal.txt"
OUT_PATH = BASE / "data" / "locus_pilot" / "smoke_test_cr_v6.jsonl"


def _grade_once(prompt: str, client, renderer, tokenizer, max_tokens: int) -> tuple[str, float, int]:
    """Grade once, return (response_text, elapsed_sec, n_output_tokens)."""
    model_input = renderer.build_generation_prompt(
        [{"role": "user", "content": prompt}]
    )
    t0 = time.time()
    result = client.sample(
        model_input, num_samples=1,
        sampling_params=tinker.types.SamplingParams(
            max_tokens=max_tokens, temperature=GRADER_TEMP,
            stop=renderer.get_stop_sequences(),
        ),
    ).result()
    elapsed = time.time() - t0
    tokens = result.sequences[0].tokens
    text = tokenizer.decode(tokens)
    return text, elapsed, len(tokens)


def main():
    goal = GOAL_PATH.read_text().strip()
    with SAMPLE_PATH.open() as f:
        all_plans = [json.loads(l) for l in f][:N_PLANS]
    logger.info(f"Loaded {len(all_plans)} plans from {SAMPLE_PATH.name}")

    client = create_service_client(api_profile="new")
    grader_client = client.create_sampling_client(base_model=MODEL)
    sampling_client = client.create_sampling_client(base_model=MODEL)
    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    results = []
    # Accumulators for summary
    locus_times, baseline_times = [], []
    locus_tok_counts, baseline_tok_counts = [], []
    locus_parse_ok, locus_parse_fail = 0, 0
    revision_parse_ok, revision_apply_ok = 0, 0
    deltas_target, deltas_agg = [], []

    for idx, plan_data in enumerate(all_plans):
        plan_text = plan_data["plan_text"]
        sv = plan_data["signal_vector"]
        bottleneck_id, bottleneck_score = identify_bottleneck(sv, skip_signals=SKIP_SIGNALS)
        spec = next(s for s in SIGNALS if s.id == bottleneck_id)

        logger.info(f"Plan {idx}: bottleneck={bottleneck_id} (score={bottleneck_score})")

        entry = {
            "plan_idx": idx,
            "bottleneck_id": bottleneck_id,
            "bottleneck_score": bottleneck_score,
        }

        # --- Gate 2: cost comparison ---
        # Locus grading (emit_locus=True)
        locus_prompt = build_single_signal_prompt(goal, plan_text, spec, emit_locus=True)
        locus_text, locus_t, locus_n = _grade_once(
            locus_prompt, grader_client, renderer, tokenizer, LOCUS_MAX_TOKENS,
        )
        locus_times.append(locus_t)
        locus_tok_counts.append(locus_n)

        # Baseline grading (emit_locus=False)
        baseline_prompt = build_single_signal_prompt(goal, plan_text, spec, emit_locus=False)
        baseline_text, baseline_t, baseline_n = _grade_once(
            baseline_prompt, grader_client, renderer, tokenizer, BASELINE_MAX_TOKENS,
        )
        baseline_times.append(baseline_t)
        baseline_tok_counts.append(baseline_n)

        entry["locus_time"] = locus_t
        entry["locus_tokens"] = locus_n
        entry["baseline_time"] = baseline_t
        entry["baseline_tokens"] = baseline_n
        entry["cost_ratio"] = locus_t / max(baseline_t, 0.01)

        # --- Gate 3: locus parse ---
        loci = parse_locus(locus_text, plan_text)
        if loci is not None and len(loci) > 0:
            locus_parse_ok += 1
            entry["locus_status"] = "ok"
            entry["n_loci"] = len(loci)
            entry["loci_quotes"] = [l.quote[:80] for l in loci]
        else:
            locus_parse_fail += 1
            entry["locus_status"] = "fail" if loci is None else "empty"
            entry["n_loci"] = 0
            results.append(entry)
            logger.info(f"  Locus parse: {entry['locus_status']}")
            continue  # Can't test revision without loci

        # --- Gate 3: revision ---
        revise_prompt = build_locus_revision_prompt(
            goal=goal, plan_text=plan_text,
            signal_vector=sv, bottleneck_id=bottleneck_id,
            bottleneck_score=bottleneck_score, loci=loci,
        )
        rev_input = renderer.build_generation_prompt(
            [{"role": "user", "content": revise_prompt}]
        )
        rev_result = sampling_client.sample(
            rev_input, num_samples=1,
            sampling_params=tinker.types.SamplingParams(
                max_tokens=REVISION_MAX_TOKENS, temperature=REVISION_TEMP,
                stop=renderer.get_stop_sequences(),
            ),
        ).result()
        raw_rev = tokenizer.decode(rev_result.sequences[0].tokens)

        revisions = parse_locus_revisions(raw_rev)
        if revisions:
            revision_parse_ok += 1
            revised_plan, success, status = apply_locus_revisions(plan_text, revisions)
            entry["revision_parse"] = "ok"
            entry["n_revisions"] = len(revisions)
            entry["apply_status"] = status
            entry["apply_success"] = success
            if success:
                revision_apply_ok += 1
                # Re-grade bottleneck only
                regrade_prompt = build_single_signal_prompt(goal, revised_plan, spec, emit_locus=False)
                regrade_text, _, _ = _grade_once(
                    regrade_prompt, grader_client, renderer, tokenizer, BASELINE_MAX_TOKENS,
                )
                parsed = parse_scores(regrade_text)
                new_score = (parsed.get(bottleneck_id) or {}).get("score")
                entry["new_score"] = new_score
                if new_score is not None:
                    dt = (new_score - bottleneck_score) / 4.0
                    da = 0.0  # would need full 8-signal regrade for aggregate; skip
                    deltas_target.append(dt)
                    entry["delta_target"] = dt
                    logger.info(f"  Revised: {bottleneck_id} {bottleneck_score}→{new_score} (Δ={dt:+.2f})")
        else:
            entry["revision_parse"] = "fail"
            logger.info(f"  Revision parse failed (no <revisions> block)")

        results.append(entry)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("CR-v6 SMOKE TEST SUMMARY")
    print("=" * 60)

    n = len(all_plans)
    mean_locus_t = sum(locus_times) / n
    mean_baseline_t = sum(baseline_times) / n
    cost_ratio = mean_locus_t / max(mean_baseline_t, 0.01)
    mean_locus_tok = sum(locus_tok_counts) / n
    mean_baseline_tok = sum(baseline_tok_counts) / n

    print(f"\nGate 2 — Grader Cost Regression:")
    print(f"  Baseline: {mean_baseline_t:.1f}s, {mean_baseline_tok:.0f} tokens/call")
    print(f"  Locus:    {mean_locus_t:.1f}s, {mean_locus_tok:.0f} tokens/call")
    print(f"  Ratio:    {cost_ratio:.2f}x {'✅ PASS' if cost_ratio <= 1.5 else '❌ FAIL'} (threshold ≤1.5×)")

    print(f"\nGate 3 — Smoke Test ({n} plans):")
    print(f"  Locus parse:     {locus_parse_ok}/{n} ({100*locus_parse_ok/n:.0f}%)")
    print(f"  Revision parse:  {revision_parse_ok}/{n} ({100*revision_parse_ok/n:.0f}%)")
    print(f"  Apply success:   {revision_apply_ok}/{n} ({100*revision_apply_ok/n:.0f}%)")
    if deltas_target:
        import statistics
        print(f"  Δ_target mean:   {statistics.mean(deltas_target):+.3f}")
        print(f"  Δ_target > 0:    {sum(1 for d in deltas_target if d > 0)}/{len(deltas_target)}")

    # Per-signal breakdown
    signal_counts: dict[str, int] = {}
    for r in results:
        sid = r["bottleneck_id"]
        signal_counts[sid] = signal_counts.get(sid, 0) + 1
    print(f"\n  Bottleneck distribution: {signal_counts}")

    print(f"\n  Gate 2: {'PASS' if cost_ratio <= 1.5 else 'FAIL'}")
    print(f"  Gate 3: {'PASS' if locus_parse_ok >= 7 and revision_apply_ok >= 3 else 'NEEDS REVIEW'}")

    # Save raw data
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nRaw data: {OUT_PATH}")


if __name__ == "__main__":
    main()
