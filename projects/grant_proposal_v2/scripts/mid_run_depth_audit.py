"""Extract best-buffer-plan at iter N and produce an Opus depth_audit prompt.

Usage:
    python projects/grant_proposal_v2/scripts/mid_run_depth_audit.py <run_dir> --iter N

Output:
    <run_dir>/depth_audit_pending_iter_N.json  — contains plan text + prompt
    Prints prompt + plan to stdout.

The actual Opus subagent call is made separately by the main agent spawning
an Agent (claude-code-guide or general-purpose) with this prompt. Each
subagent returns one depth_audit grade; main agent appends to
<run_dir>/depth_audit_mid.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEPTH_AUDIT_PROMPT = """You are a rigorous reviewer evaluating a research plan on four dimensions of quality. You are NOT following any specific rubric — evaluate holistically with high standards.

**CRITICAL — length-normalized scoring**: Score the plan's *density* and *specificity*, NOT its verbosity. Imagine every plan is rewritten to exactly 1500 words; score based on the per-word quality that would result. A short dense plan and a long padded plan should receive identical scores if the underlying commitments are equivalent. Extra length without extra substance is NOT rigor.

# Research Goal
{goal}

# Research Plan
{plan}

# Evaluation

Score the plan on four dimensions, each 1-10. Apply the length-normalization rule above to every dimension.

1. **DEPTH** (1-10): Technical depth. Does the plan go beyond surface-level description? Are key methodological choices explained? Does it show mechanism-level reasoning about why proposed methods would work?

2. **METHODS** (1-10): Methodological clarity and specificity. Are the methods concrete (named algorithms, specific techniques, measurable procedures)? Or vague keywords?

3. **FEASIBILITY** (1-10): Resource-realistic execution plan. Is the scope achievable? Are compute/data/timeline constraints addressed? Would a PhD student know what to do on Monday?

4. **GROUNDING** (1-10): Literature connection and citations. Does the plan demonstrate awareness of related work? Does it identify gaps without misrepresenting existing work? Are the references plausible? (Do NOT reward a plan merely for having more citations; reward for citations that are *relevant and load-bearing* to the claims.)

For each dimension, give a score AND a 1-2 sentence rationale.

# Output Format (strict)

<depth_audit>
  <depth>{{1-10}}</depth>
  <depth_rationale>{{1-2 sentences}}</depth_rationale>
  <methods>{{1-10}}</methods>
  <methods_rationale>{{1-2 sentences}}</methods_rationale>
  <feasibility>{{1-10}}</feasibility>
  <feasibility_rationale>{{1-2 sentences}}</feasibility_rationale>
  <grounding>{{1-10}}</grounding>
  <grounding_rationale>{{1-2 sentences}}</grounding_rationale>
</depth_audit>
"""


def find_best_buffer_plan(run_dir: Path, iter_cutoff: int) -> tuple[int, dict]:
    """Return (buffer_idx, entry) for the highest-reward hard_gate_passed plan
    with iter <= iter_cutoff."""
    buffer_path = run_dir / "buffer.jsonl"
    if not buffer_path.exists():
        raise FileNotFoundError(f"No buffer.jsonl at {buffer_path}")
    entries = []
    for i, line in enumerate(open(buffer_path)):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Buffer uses "iteration" field (some older entries may lack it)
        it = e.get("iteration")
        if it is not None and it > iter_cutoff:
            continue
        if not e.get("hard_gate_passed", False):
            continue
        entries.append((i, e))
    if not entries:
        raise ValueError(f"No hard_gate_passed entries at iter <= {iter_cutoff}")
    entries.sort(key=lambda t: t[1].get("aggregate_reward", 0.0), reverse=True)
    return entries[0]


def load_goal_text(run_dir: Path) -> str:
    """Load the research goal from the run's config.json and goal_dir."""
    config = json.loads((run_dir / "config.json").read_text())
    goal_dir = Path(config["goal_dir"])
    if not goal_dir.is_absolute():
        # Resolve relative to project root
        goal_dir = Path(__file__).resolve().parents[3] / goal_dir
    return (goal_dir / "research_goal.md").read_text().strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="Path to run log dir")
    ap.add_argument("--iter", type=int, required=True, help="Iter cutoff (inclusive)")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    buf_idx, entry = find_best_buffer_plan(run_dir, args.iter)
    goal = load_goal_text(run_dir)
    plan = entry.get("plan_text", "")
    prompt = DEPTH_AUDIT_PROMPT.format(goal=goal, plan=plan)

    pending_path = run_dir / f"depth_audit_pending_iter_{args.iter}.json"
    pending = {
        "iter": args.iter,
        "buffer_idx": buf_idx,
        "buffer_iter": entry.get("iteration"),
        "aggregate_reward": entry.get("aggregate_reward"),
        "signal_vector": entry.get("signal_vector"),
        "plan_text": plan,
        "goal": goal,
        "prompt": prompt,
    }
    pending_path.write_text(json.dumps(pending, indent=2))

    print(f"=== Best plan at iter <= {args.iter} ===", file=sys.stderr)
    print(f"buffer_idx: {buf_idx}, buffer_iter: {entry.get('iteration')}", file=sys.stderr)
    print(f"aggregate_reward: {entry.get('aggregate_reward'):.3f}", file=sys.stderr)
    print(f"Saved prompt + plan to: {pending_path}", file=sys.stderr)
    print("=== Prompt (paste to Opus subagent) ===", file=sys.stderr)
    print()
    print(prompt)


if __name__ == "__main__":
    main()
