"""
Build 3 rubric-generation request files from existing v8 B4 FoundOpt rollouts.

Usage:
    PYTHONPATH=src python projects/grant_proposal_v2/analysis/rubric_evolution_pilot/build_pair_requests.py

Writes three request JSONs (one per contrast pair) that a Claude subagent can
consume to produce new rubric items. The subagent's responses will be dropped
back into `responses/{request_id}.json` and parsed by `ingest_responses.py`.

Pairs chosen:
  1. best_vs_ref:  highest-scoring iter-10 B4 plan  vs reference proposal
  2. mid_vs_ref:   median-scoring iter-10 B4 plan    vs reference proposal
  3. best_vs_worst: highest vs lowest iter-10 B4 plan (policy-internal contrast)

All three use the SAME v8 signal set as R_persist (so generator knows what's
already covered). R_active starts empty (first elicitation).
"""

import json
from pathlib import Path

from co_scientist.shared.grant_rubric_v8 import SIGNALS
from co_scientist.shared.rubric_buffer import RubricBuffer
from co_scientist.shared.rubric_gen_prompt import build_rubric_gen_messages


ROOT = Path("/home/silas/co-scientist-project/projects/grant_proposal_v2")
B4_BUFFER = ROOT / "runs/2026_04_22_v8_foundopt_B4/buffer.jsonl"
REF_PATH = ROOT / "dataset/goals/01_foundopt/reference_proposal.md"
GOAL_PATH = ROOT / "dataset/goals/01_foundopt/research_goal.md"
OUT_DIR = ROOT / "analysis/rubric_evolution_pilot/requests"

TARGET_ITER = 10  # v8 B4 ran to iter 13 before kill; iter 10 is a stable checkpoint


def load_buffer(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ref_text = REF_PATH.read_text().strip()
    goal_text = GOAL_PATH.read_text().strip()

    all_entries = load_buffer(B4_BUFFER)
    # only revised entries from target iter (revisions carry the final-pipeline output)
    iter_entries = [
        e
        for e in all_entries
        if e.get("iteration") == TARGET_ITER
        and e.get("entry_type") == "revision"
        and e.get("hard_gate_passed")
    ]
    if not iter_entries:
        # fall back: any entry from that iter
        iter_entries = [e for e in all_entries if e.get("iteration") == TARGET_ITER]

    # sort by aggregate_reward desc
    iter_entries.sort(key=lambda x: x.get("aggregate_reward", 0), reverse=True)
    print(f"iter {TARGET_ITER} entries: {len(iter_entries)}")
    for i, e in enumerate(iter_entries):
        print(
            f"  [{i}] reward={e.get('aggregate_reward', 0):.3f}  "
            f"words={e.get('word_count', 0)}  "
            f"gate={'✓' if e.get('hard_gate_passed') else '✗'}"
        )

    if len(iter_entries) < 2:
        raise RuntimeError(f"need ≥2 entries at iter {TARGET_ITER}, got {len(iter_entries)}")

    best = iter_entries[0]
    mid = iter_entries[len(iter_entries) // 2]
    worst = iter_entries[-1]

    persistent_names = [f"{s.id} ({s.name})" for s in SIGNALS]

    pairs = [
        (
            "pair_01_best_vs_ref",
            best["plan_text"],
            ref_text,
            f"B4 iter{TARGET_ITER} best (reward={best['aggregate_reward']:.3f})",
            "Reference proposal",
        ),
        (
            "pair_02_mid_vs_ref",
            mid["plan_text"],
            ref_text,
            f"B4 iter{TARGET_ITER} median (reward={mid['aggregate_reward']:.3f})",
            "Reference proposal",
        ),
        (
            "pair_03_best_vs_worst",
            best["plan_text"],
            worst["plan_text"],
            f"B4 iter{TARGET_ITER} best (reward={best['aggregate_reward']:.3f})",
            f"B4 iter{TARGET_ITER} worst (reward={worst['aggregate_reward']:.3f})",
        ),
    ]

    # R_active is empty for the pilot (first elicitation)
    empty_buf = RubricBuffer()

    for pair_id, plan_a, plan_b, label_a, label_b in pairs:
        messages = build_rubric_gen_messages(
            research_goal=goal_text,
            persistent_rubric_names=persistent_names,
            active_rubric_items=empty_buf.items.values(),
            plan_a=plan_a,
            plan_b=plan_b,
            plan_a_label=label_a,
            plan_b_label=label_b,
        )

        out = {
            "pair_id": pair_id,
            "plan_a_label": label_a,
            "plan_b_label": label_b,
            "system": messages[0]["content"],
            "user": messages[1]["content"],
            "meta": {
                "goal": "01_foundopt",
                "source_run": str(B4_BUFFER),
                "iter": TARGET_ITER,
                "plan_a_reward": best["aggregate_reward"] if "best" in pair_id else (
                    mid["aggregate_reward"] if "mid" in pair_id else best["aggregate_reward"]
                ),
            },
        }
        out_path = OUT_DIR / f"{pair_id}.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"wrote {out_path.name}  ({len(messages[1]['content'])} user chars)")


if __name__ == "__main__":
    main()
