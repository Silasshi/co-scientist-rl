#!/usr/bin/env python3
"""Sample ~35 plans for CR-v6 Phase 0 locus-accuracy validation.

Pool: 4 runs under `_corrupted_2048/` (MAIN, A3p_no_s8_scope, B4_no_training,
B1_zero_shot). Skip A8_no_revision to avoid fresh-only distribution bias.

Stratify by S3_positioning score to test grader locus ability across severity:
  - S3=1: 7 plans  (most broken — weakness concentrated)
  - S3=2: 7 plans
  - S3=3: 7 plans
  - S3=4: 7 plans  (mildly broken)
  - S3=5: 5 plans  (CONTROLS — check false-positive locus rate on good plans)
Total: up to 33 plans. If a stratum has fewer than quota, take all and note.

Filter: hard_gate_passed=True and signal_vector.S3_positioning is not None.
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[6]
RUNS_ROOT = PROJECT_ROOT / "projects/ttt_discover/runs/2026_04_30b_paper_experiments_v81/_corrupted_2048"
OUT_DIR = PROJECT_ROOT / "projects/ttt_discover/analysis/signal_validity/data/locus_pilot"
OUT_FILE = OUT_DIR / "sample_s3_pilot.jsonl"

SOURCE_RUNS = ["MAIN", "A3p_no_s8_scope", "B4_no_training", "B1_zero_shot"]
QUOTA = {1: 7, 2: 7, 3: 7, 4: 7, 5: 5}
SEED = 20260416  # today's date for reproducibility

# Filter out partial-plan revisions (from stitch failures in CR-v5 buffer).
# A full plan needs all core sections (Problem, Core Idea, Methodology,
# Evaluation, etc.). Use word_count as a cheap proxy — partial revisions
# that only contain one section are typically < 800 words; a full plan is
# typically 1000+. Manual inspection confirms wc≥800 eliminates the
# Risk-only / Training-Objective-only revision fragments.
MIN_WORDS = 800


def main() -> None:
    random.seed(SEED)

    # Load all eligible buffer entries
    pool: list[dict] = []
    n_skipped_partial = 0
    for run_name in SOURCE_RUNS:
        buffer_path = RUNS_ROOT / run_name / "buffer.jsonl"
        if not buffer_path.exists():
            print(f"WARN: missing {buffer_path}")
            continue
        with buffer_path.open() as f:
            for line in f:
                d = json.loads(line)
                if not d.get("hard_gate_passed"):
                    continue
                s3 = d.get("signal_vector", {}).get("S3_positioning")
                if s3 is None:
                    continue
                wc = d.get("word_count", 0)
                if wc < MIN_WORDS:
                    n_skipped_partial += 1
                    continue
                pool.append({
                    "source_run": run_name,
                    "iteration": d.get("iteration"),
                    "entry_type": d.get("entry_type"),
                    "s3_score": int(s3),
                    "aggregate_reward": d.get("aggregate_reward"),
                    "signal_vector": d.get("signal_vector"),
                    "plan_text": d.get("plan_text"),
                    "word_count": wc,
                })
    print(f"Skipped {n_skipped_partial} partial-plan entries (word_count < {MIN_WORDS})")

    # Distribution summary before sampling
    total_by_s3 = Counter(p["s3_score"] for p in pool)
    total_by_run = Counter(p["source_run"] for p in pool)
    print(f"Eligible pool: {len(pool)} plans (hard_gate_passed=True, S3 present)")
    print("  By S3 score:", dict(sorted(total_by_s3.items())))
    print("  By run:     ", dict(total_by_run))

    # Stratified sample
    by_s3: dict[int, list[dict]] = defaultdict(list)
    for p in pool:
        by_s3[p["s3_score"]].append(p)

    sampled: list[dict] = []
    for s3_val, quota in sorted(QUOTA.items()):
        available = by_s3.get(s3_val, [])
        k = min(quota, len(available))
        if k < quota:
            print(f"  NOTE: S3={s3_val} requested {quota}, only {k} available")
        sampled.extend(random.sample(available, k))

    # Sort by (s3_score, source_run, iteration) for deterministic output ordering
    sampled.sort(key=lambda p: (p["s3_score"], p["source_run"], p["iteration"] or -1))

    # Assign sample_id for downstream reference
    for i, s in enumerate(sampled):
        s["sample_id"] = f"locus_s3_{i:03d}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w") as f:
        for s in sampled:
            f.write(json.dumps(s) + "\n")

    print(f"\nWrote {len(sampled)} plans to {OUT_FILE}")
    sampled_by_s3 = Counter(s["s3_score"] for s in sampled)
    sampled_by_run = Counter(s["source_run"] for s in sampled)
    print("  Sampled by S3:", dict(sorted(sampled_by_s3.items())))
    print("  Sampled by run:", dict(sampled_by_run))


if __name__ == "__main__":
    main()
