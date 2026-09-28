"""Build 5 Opus subagent bundles — each scores 3 plans × 5 changed signals = 15 grades.

Output: /tmp/opus_v9_group_{0..4}.md — one file per subagent, containing all
prompts the subagent needs to grade. Each bundle is self-contained.

Only the 5 CHANGED signals are re-graded: G4, G10, G11, G12, G13.
G1/G2/G6/G8/G9 Opus scores can be reused from grader_panel_v8/grades/opus.jsonl.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJ / "src"))

from co_scientist.shared.grant_rubric_v9 import SIGNALS as V9_SIGNALS
from co_scientist.shared.grant_signal_reward import build_single_signal_prompt

CHANGED = {"G4_focus", "G10_approach_coverage", "G11_evidence_rigor",
           "G12_formalism", "G13_risk_awareness"}

PANEL = Path(__file__).resolve().parent.parent
PLANS_DIR = PANEL / "test_plans"
GOAL_TEXT = (PROJ / "projects/grant_proposal/dataset/goals/01_foundopt/research_goal.md").read_text().strip()


def main() -> None:
    sources = [json.loads(l) for l in open(PLANS_DIR / "sources.jsonl")]
    plan_ids = [s["plan_id"] for s in sources]  # plan_00 ... plan_14
    plans = {pid: (PLANS_DIR / f"{pid}.txt").read_text() for pid in plan_ids}

    changed_signals = [s for s in V9_SIGNALS if s.id in CHANGED]
    assert len(changed_signals) == 5, f"expected 5 changed signals, got {len(changed_signals)}"

    # 5 groups × 3 plans each
    for group_idx in range(5):
        group_plan_ids = plan_ids[group_idx * 3: (group_idx + 1) * 3]
        bundle_lines = []
        bundle_lines.append(f"# Opus v9 grading bundle — group {group_idx}")
        bundle_lines.append(f"")
        bundle_lines.append(f"You will grade 3 research plans on 5 quality signals each (15 total grades).")
        bundle_lines.append(f"For each (plan, signal), produce the XML output as specified in the prompt.")
        bundle_lines.append(f"")
        bundle_lines.append(f"## Output format")
        bundle_lines.append(f"")
        bundle_lines.append(f"Return ALL 15 grades as a single JSONL block (one JSON object per line), wrapped in <results>...</results>.")
        bundle_lines.append(f"Each JSON object: `{{\"plan_id\": \"plan_XX\", \"signal_id\": \"G...\", \"score\": N, \"reasoning\": \"...\"}}`")
        bundle_lines.append(f"Reasoning field should be 1-3 sentences explaining the score under the v9 rubric.")
        bundle_lines.append(f"Scores must be integers 1-5.")
        bundle_lines.append(f"")
        bundle_lines.append(f"---")
        bundle_lines.append(f"")
        for pid in group_plan_ids:
            for sig in changed_signals:
                prompt = build_single_signal_prompt(
                    goal=GOAL_TEXT, plan=plans[pid], signal=sig,
                    emit_critique=True, include_cot=True,
                )
                bundle_lines.append(f"## TASK: plan={pid}, signal={sig.id}")
                bundle_lines.append(f"")
                bundle_lines.append(prompt)
                bundle_lines.append(f"")
                bundle_lines.append(f"---")
                bundle_lines.append(f"")
        bundle_lines.append(f"## FINAL OUTPUT")
        bundle_lines.append(f"")
        bundle_lines.append(f"Now output exactly 15 JSONL lines (3 plans × 5 signals) inside a single <results>...</results> block, covering every (plan, signal) task above.")

        out = Path(f"/tmp/opus_v9_group_{group_idx}.md")
        out.write_text("\n".join(bundle_lines))
        print(f"group {group_idx}: {len(group_plan_ids)} plans × {len(changed_signals)} signals = 15 grades; saved {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
