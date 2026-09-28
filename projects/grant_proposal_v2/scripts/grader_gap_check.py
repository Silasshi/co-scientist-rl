"""
Pre-Phase-3 grader gap check.

Grades the 5 Phase-2 smoke milestone plans × 1 rubric item
("Avoids Fabricated Quantitative Precision") through two graders and
reports agreement against Opus 4.7 (which independently flagged fabrication
on all 5 milestones in depth_audit/).

Graders compared:
  * GPT-OSS-120B (Phase 2 default)  — expected to miss fabrication
  * Qwen3-235B-A22B (grader_model_alt) — expected to catch fabrication

Success criterion (for Phase 3 to proceed): Qwen3-235B gives "yes" on
≤1 of 5 plans, i.e. ≥80%% agreement with Opus.

Usage:
    source shared/tools/use_api_profile.sh new
    PYTHONPATH=src python projects/grant_proposal_v2/scripts/grader_gap_check.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC))

from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.rubric_buffer import RubricBuffer, RubricItem
from co_scientist.shared.rubric_grader import (
    launch_binary_grade, collect_binary_grade,
)


RUN = Path("projects/grant_proposal_v2/runs/2026_04_23_ger_cr_v1_smoke")
MILESTONES = [
    "iter0_fresh", "iter1_rev", "iter2_rev", "iter4_rev", "iter5_rev",
]

# Opus audit verdicts (from depth_audit/SUMMARY.json notes):
# fabrication flagged on every milestone.
OPUS_FAB_FLAGS = {ms: True for ms in MILESTONES}


def load_fab_item() -> RubricItem:
    """Reconstruct the surviving 'Avoids Fabricated Quantitative Precision' item."""
    rows = [json.loads(l) for l in open(RUN / "rubric_buffer.jsonl")]
    raw = next(e for e in rows if "Fabricated" in e["title"])
    # RubricItem expects specific fields; strip grade history (not needed for grading)
    return RubricItem(
        title=raw["title"],
        description=raw["description"],
        rubric_type=raw["rubric_type"],
        weight=raw["weight"],
        created_at_iter=raw["created_at_iter"],
    )


def run_one_grader(model_name: str, plans: dict[str, str], goal: str, item: RubricItem) -> dict[str, float]:
    """Grade all plans × item under one grader model. Returns {milestone: 0/1}."""
    client = create_service_client(api_profile="new")
    grader_client = client.create_sampling_client(base_model=model_name)
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Fire all futures in parallel
    futs = {
        ms: launch_binary_grade(
            plan=plan, goal=goal, item=item,
            grader_client=grader_client, renderer=renderer,
            max_tokens=512, temperature=0.0,
        )
        for ms, plan in plans.items()
    }
    return {
        ms: collect_binary_grade(fut, tokenizer, renderer=renderer, timeout=600)
        for ms, fut in futs.items()
    }


def main() -> None:
    item = load_fab_item()
    print(f"Rubric item: {item.title}")
    print(f"  type={item.rubric_type}  weight={item.weight}")
    print()

    goal = (Path("projects/grant_proposal_v2/dataset/goals/01_foundopt/research_goal.md")).read_text().strip()

    plans: dict[str, str] = {}
    for ms in MILESTONES:
        p = RUN / "depth_audit" / f"{ms}_input.json"
        plans[ms] = json.loads(p.read_text())["plan"]

    graders = [
        ("openai/gpt-oss-120b",                "GPT-OSS-120B (Phase 2 default)"),
        ("Qwen/Qwen3-235B-A22B-Instruct-2507", "Qwen3-235B-A22B-Instruct-2507 (alt)"),
    ]

    results: dict[str, dict[str, float]] = {}
    for model, label in graders:
        print(f"Grading via {label} ...")
        results[model] = run_one_grader(model, plans, goal, item)
        for ms in MILESTONES:
            v = results[model][ms]
            print(f"  {ms:<14} {'yes (avoids)' if v >= 0.5 else 'no (does not avoid)'}")
        print()

    # Report agreement with Opus
    print("=" * 72)
    print(f"{'milestone':<14} {'Opus fab?':<10} {'GPT-OSS':<10} {'Qwen-235B':<12}  agreement")
    opus_n = 0
    gpt_n = 0
    qwen_n = 0
    for ms in MILESTONES:
        # Opus says fabrication=yes ⇒ "avoids" verdict should be "no" (=0.0)
        opus_says_fab = OPUS_FAB_FLAGS[ms]
        expected = 0.0 if opus_says_fab else 1.0
        gpt_agree = "agree" if results["openai/gpt-oss-120b"][ms] == expected else "disagree"
        qwen_agree = "agree" if results["Qwen/Qwen3-235B-A22B-Instruct-2507"][ms] == expected else "disagree"
        print(
            f"{ms:<14} {'yes' if opus_says_fab else 'no':<10} "
            f"{('yes' if results['openai/gpt-oss-120b'][ms]>=0.5 else 'no'):<10} "
            f"{('yes' if results['Qwen/Qwen3-235B-A22B-Instruct-2507'][ms]>=0.5 else 'no'):<12} "
            f"gpt={gpt_agree}  qwen={qwen_agree}"
        )
        if opus_says_fab:
            opus_n += 1
        if gpt_agree == "agree":
            gpt_n += 1
        if qwen_agree == "agree":
            qwen_n += 1

    print()
    print(f"GPT-OSS agreement with Opus:    {gpt_n}/5  ({100*gpt_n/5:.0f}%)")
    print(f"Qwen-235B agreement with Opus:  {qwen_n}/5  ({100*qwen_n/5:.0f}%)")
    print()
    verdict = "PASS" if qwen_n >= 4 else "FAIL"
    print(f"Phase 3 grader-gap check: {verdict}  (needed ≥4/5)")

    out = RUN / "depth_audit" / "grader_gap.json"
    out.write_text(json.dumps({
        "item": item.title,
        "milestones": MILESTONES,
        "opus_fab_flags": OPUS_FAB_FLAGS,
        "gpt_oss_120b": results["openai/gpt-oss-120b"],
        "qwen_235b": results["Qwen/Qwen3-235B-A22B-Instruct-2507"],
        "gpt_agreement": gpt_n / 5,
        "qwen_agreement": qwen_n / 5,
        "verdict": verdict,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
