"""Sample 15 FoundOpt test plans spanning quality range for grader panel test.

Sampling strategy (seed=42):
  3 ref-anchor: reference solution duplicated (grader noise floor)
  4 high-quality: best plans from different methods
  4 mid-quality: median plans from different runs
  4 low-quality: early fresh plans (no critique-revise)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

PROJ = Path(__file__).resolve().parents[5]
RUNS_DIR = PROJ / "projects/grant_proposal/runs"
OUT_DIR = Path(__file__).resolve().parent.parent / "test_plans"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE = (
    PROJ
    / "projects/grant_proposal/dataset/goals/01_foundopt/reference_proposal.md"
)

SEED = 42
rng = random.Random(SEED)


def load_buffer(run_name: str) -> list[dict]:
    buf = RUNS_DIR / run_name / "buffer.jsonl"
    if not buf.exists():
        return []
    return [json.loads(l) for l in open(buf)]


def pick_best(run_name: str) -> dict | None:
    entries = [e for e in load_buffer(run_name) if e.get("hard_gate_passed")]
    if not entries:
        return None
    return max(entries, key=lambda e: e["aggregate_reward"])


def pick_median(run_name: str, iter_range: tuple[int, int] | None = None) -> dict | None:
    entries = [e for e in load_buffer(run_name) if e.get("hard_gate_passed")]
    if iter_range is not None:
        entries = [
            e for e in entries
            if iter_range[0] <= e.get("iteration", 0) <= iter_range[1]
        ]
    if not entries:
        return None
    entries.sort(key=lambda e: e["aggregate_reward"])
    return entries[len(entries) // 2]


def pick_low(run_name: str, iter_range: tuple[int, int] = (0, 2)) -> dict | None:
    entries = [
        e for e in load_buffer(run_name)
        if e.get("hard_gate_passed") and e.get("entry_type") == "fresh"
        and iter_range[0] <= e.get("iteration", 0) <= iter_range[1]
    ]
    if not entries:
        return None
    return rng.choice(entries)


def main() -> None:
    samples: list[dict] = []

    # --- 3 reference-anchor (identical plan, 3 IDs for grader noise measurement) ---
    ref_text = REFERENCE.read_text().strip()
    for i in range(3):
        samples.append({
            "plan_id": f"plan_{len(samples):02d}",
            "quality_bucket": "reference_anchor",
            "source_run": "reference",
            "iteration": None,
            "entry_type": "reference",
            "aggregate_reward": None,
            "plan_text": ref_text,
            "word_count": len(ref_text.split()),
        })

    # --- 4 high-quality: best plans from diverse methods ---
    high_runs = [
        "2026_04_21_b4_paper_01_foundopt",
        "2026_04_20_d4v7_01_foundopt_B4",
        "2026_04_21_gapo_foundopt",
        "2026_04_22_sdpo_C3_sdpo_scores_only",
    ]
    for run in high_runs:
        e = pick_best(run)
        if e is None:
            print(f"WARN: no best plan for {run}")
            continue
        samples.append({
            "plan_id": f"plan_{len(samples):02d}",
            "quality_bucket": "high",
            "source_run": run,
            "iteration": e.get("iteration"),
            "entry_type": e.get("entry_type"),
            "aggregate_reward": round(e["aggregate_reward"], 3),
            "plan_text": e["plan_text"],
            "word_count": e.get("word_count", len(e["plan_text"].split())),
        })

    # --- 4 mid-quality: median plans from diverse runs ---
    mid_runs = [
        ("2026_04_21_b4_paper_01_foundopt", (5, 10)),
        ("2026_04_20_d4v7_01_foundopt_MAIN", (5, 15)),
        ("2026_04_21_rag_foundopt_B4", None),
        ("2026_04_22_sdpo_C4_aggregate_scores_only", (10, 25)),
    ]
    for run, ir in mid_runs:
        e = pick_median(run, ir)
        if e is None:
            print(f"WARN: no median plan for {run} iter_range={ir}")
            continue
        samples.append({
            "plan_id": f"plan_{len(samples):02d}",
            "quality_bucket": "mid",
            "source_run": run,
            "iteration": e.get("iteration"),
            "entry_type": e.get("entry_type"),
            "aggregate_reward": round(e["aggregate_reward"], 3),
            "plan_text": e["plan_text"],
            "word_count": e.get("word_count", len(e["plan_text"].split())),
        })

    # --- 4 low-quality: early fresh plans (iter 0-2) from diverse runs ---
    low_runs = [
        "2026_04_21_b4_paper_01_foundopt",
        "2026_04_20_d4v7_01_foundopt_MAIN",
        "2026_04_21_gapo_foundopt",
        "2026_04_22_sdpo_C2_B4_scores_only",
    ]
    for run in low_runs:
        e = pick_low(run)
        if e is None:
            print(f"WARN: no low plan for {run}")
            continue
        samples.append({
            "plan_id": f"plan_{len(samples):02d}",
            "quality_bucket": "low",
            "source_run": run,
            "iteration": e.get("iteration"),
            "entry_type": e.get("entry_type"),
            "aggregate_reward": round(e["aggregate_reward"], 3),
            "plan_text": e["plan_text"],
            "word_count": e.get("word_count", len(e["plan_text"].split())),
        })

    # Write each plan text to individual file + sources.jsonl metadata
    sources_lines = []
    for s in samples:
        plan_id = s["plan_id"]
        (OUT_DIR / f"{plan_id}.txt").write_text(s["plan_text"])
        meta = {k: v for k, v in s.items() if k != "plan_text"}
        sources_lines.append(json.dumps(meta, ensure_ascii=False))
    (OUT_DIR / "sources.jsonl").write_text("\n".join(sources_lines) + "\n")

    # Summary
    print(f"\nSampled {len(samples)} plans (seed={SEED})")
    print(f"{'plan_id':<10s} {'bucket':<18s} {'qwen':>6s} {'words':>6s} {'source'}")
    print("-" * 90)
    for s in samples:
        q = s["aggregate_reward"] if s["aggregate_reward"] is not None else "-"
        print(
            f"{s['plan_id']:<10s} {s['quality_bucket']:<18s} "
            f"{str(q):>6s} {s['word_count']:>6d} {s['source_run']}"
        )


if __name__ == "__main__":
    main()
