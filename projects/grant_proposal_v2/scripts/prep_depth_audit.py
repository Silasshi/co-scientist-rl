"""Prepare depth_audit inputs from buffer.jsonl for Opus subagent grading.

Reads a run's buffer.jsonl, picks the buffer_max plan at each target iter, and
writes one JSON file per (run, iter) with {goal, plan, prompt} that a
subagent can grade directly.

Usage:
    python prep_depth_audit.py <run_dir> <iter1,iter2,...>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


DEPTH_AUDIT_PROMPT = """\
You are an expert grant proposal reviewer (NIH study section level). Score the
following research proposal on 4 independent dimensions, each 1-10 (integer).
Evaluate SUBSTANCE, not surface structure. A proposal that names many methods
or cites many references but lacks depth should score LOW.

Use the FULL 1-10 range. A mediocre proposal should score 4-5, not 6-7.
Reserve 9-10 for proposals that would impress a domain expert.

# Research Goal
{goal}

# Research Proposal to Evaluate
{plan}

# Scoring Dimensions

## 1. Problem Depth (1-10)
- 1-2: Restates the goal without analysis; no independent thinking
- 3-4: Names the problem with generic background; sub-problems identified but
  at a textbook level without revealing why they are hard
- 5-6: Shows domain knowledge; identifies specific sub-problems with concrete
  examples; begins to articulate why existing approaches fall short
- 7-8: Demonstrates genuine understanding; reveals specific tensions or
  trade-offs between competing objectives; explains the mechanism behind
  each gap, not just the gap itself
- 9-10: Reveals non-obvious insights that a non-expert would miss; reframes
  the problem in a way that opens new solution directions

## 2. Methodological Substance (1-10)
- 1-2: Lists method names without explaining how they apply
- 3-4: Describes standard procedures generically; equations are textbook
  restated without adaptation to this specific problem
- 5-6: Explains how specific methods address specific aims; some justification
  for method choices but alternatives not seriously considered
- 7-8: Justifies method choices with domain-appropriate reasoning; addresses
  limitations; explains why alternatives were rejected; formulas are
  adapted or derived for this problem, not just cited
- 9-10: Demonstrates methodological innovation with clear rationale; methods
  are tightly coupled to the problem structure in a non-obvious way;
  a reviewer would learn something from reading the methodology

## 3. Feasibility & Specificity (1-10)
- 1-2: Vague scope with no timeline or resource plan
- 3-4: Has structure but commitments are unrealistic or contradictory;
  arbitrary quantitative targets without justification
- 5-6: Reasonable scope with basic timeline and named datasets; some
  quantitative targets that are plausible but not well-justified
- 7-8: Specific, achievable milestones with acknowledged limitations;
  contingency plans for identified risks; resource allocation justified
- 9-10: Detailed work plan with fallback strategies; success criteria are
  operationalized with specific thresholds and statistical justification;
  team composition and timeline are realistic given the scope

## 4. Scholarly Grounding (1-10)
- 1-2: No real citations; fabricated references or completely generic
- 3-4: A few real citations but used as decoration, not integrated into
  the argument; some citations appear fabricated or misattributed
- 5-6: Several real citations properly contextualized; demonstrates
  awareness of the field but gaps in coverage of key prior work
- 7-8: Strong command of relevant literature; citations are integrated
  to build the argument; identifies a genuine gap in prior work
- 9-10: Synthesizes literature to identify a non-obvious gap; all citations
  verifiable and precisely attributed; positions the work relative to
  the state of the art with specific comparisons

IMPORTANT: For Dimension 4, be skeptical of citations. Check whether cited
papers actually exist and whether the attributed claims are plausible.
Fabricated references should score 1-4 regardless of other merits.

# Output Format
Respond with ONLY this JSON (no other text):
{{"depth": <int>, "methods": <int>, "feasibility": <int>, "grounding": <int>}}
"""


def main() -> None:
    run_dir = Path(sys.argv[1])
    target_iters = {int(x) for x in sys.argv[2].split(",")}

    # Load goal
    goal_dir = Path("/home/silas/co-scientist-project/projects/grant_proposal/dataset/goals/01_foundopt_v10")
    goal = (goal_dir / "research_goal.md").read_text()

    buf = [json.loads(l) for l in (run_dir / "buffer.jsonl").read_text().splitlines()]
    by_iter: dict[int, dict] = {}
    for e in buf:
        it = e.get("iteration", e.get("iter"))
        if it is None or not e.get("hard_gate_passed", True):
            continue
        cur = by_iter.get(it)
        if cur is None or (e.get("aggregate_reward") or 0) > (cur.get("aggregate_reward") or 0):
            by_iter[it] = e

    out_dir = run_dir / "depth_audit_workdir"
    out_dir.mkdir(exist_ok=True)
    for it in sorted(target_iters):
        if it not in by_iter:
            print(f"  iter {it} not found in buffer — skipping")
            continue
        e = by_iter[it]
        prompt = DEPTH_AUDIT_PROMPT.format(goal=goal, plan=e["plan_text"])
        payload = {
            "run": run_dir.name,
            "iter": it,
            "plan_id_origin": f"iter{it}_bufmax_agg{e['aggregate_reward']:.3f}",
            "aggregate_reward": e["aggregate_reward"],
            "plan_text": e["plan_text"],
            "prompt": prompt,
        }
        out_path = out_dir / f"iter_{it:03d}_input.json"
        out_path.write_text(json.dumps(payload))
        print(f"  wrote {out_path}  (plan_len={len(e['plan_text'])}, agg={e['aggregate_reward']:.3f})")


if __name__ == "__main__":
    main()
