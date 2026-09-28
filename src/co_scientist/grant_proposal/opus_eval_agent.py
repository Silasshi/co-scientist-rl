"""Background Opus 4.7 depth evaluation agent for grant proposals.

Forked from ttt_discover/opus_eval_agent.py for D4 (grant_proposal).

Usage:
    python opus_eval_agent.py \\
        --buffer_path runs/MAIN/buffer.jsonl \\
        --eval_every 5 \\
        --goal_path projects/grant_proposal/pilots/nih_r21_pediatric_genomics/research_goal.md
"""
import argparse
import json
import logging
import os
import time
from pathlib import Path

import anthropic

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OPUS_MODEL = "claude-opus-4-7-20250430"

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


def get_buffer_plans(buffer_path: str) -> list[dict]:
    """Read buffer.jsonl and return all hard-gate-passed entries sorted by reward."""
    entries = []
    with open(buffer_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if not entry.get("hard_gate_passed", True):
                continue
            entries.append(entry)
    entries.sort(key=lambda e: e.get("aggregate_reward", 0), reverse=True)
    return entries


def pick_eval_plans(entries: list[dict]) -> list[tuple[str, str, int, float]]:
    """Pick best, median, and worst plans for evaluation.

    Returns list of (label, plan_text, iteration, reward).
    """
    if not entries:
        return []
    picks = []
    picks.append(("best", entries[0]["plan_text"], entries[0]["iteration"],
                   entries[0]["aggregate_reward"]))
    if len(entries) >= 3:
        mid = len(entries) // 2
        picks.append(("median", entries[mid]["plan_text"], entries[mid]["iteration"],
                       entries[mid]["aggregate_reward"]))
        picks.append(("worst", entries[-1]["plan_text"], entries[-1]["iteration"],
                       entries[-1]["aggregate_reward"]))
    elif len(entries) >= 2:
        picks.append(("worst", entries[-1]["plan_text"], entries[-1]["iteration"],
                       entries[-1]["aggregate_reward"]))
    return picks


def opus_depth_audit(plan: str, goal: str) -> dict:
    """Send plan to Opus 4.7 for 4-dimension depth audit."""
    client = anthropic.Anthropic()
    prompt = DEPTH_AUDIT_PROMPT.format(goal=goal, plan=plan)

    response = client.messages.create(
        model=OPUS_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    try:
        scores = json.loads(text)
        scores["total"] = sum(scores.get(k, 0) for k in ["depth", "methods", "feasibility", "grounding"])
        return scores
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse Opus response: {text[:200]}")
        return {"depth": None, "methods": None, "feasibility": None, "grounding": None, "total": None, "raw": text}


def get_latest_iter(buffer_path: str) -> int:
    """Get the highest iteration number in the buffer."""
    max_iter = -1
    with open(buffer_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            it = entry.get("iteration", -1)
            if it > max_iter:
                max_iter = it
    return max_iter


def main():
    parser = argparse.ArgumentParser(description="Opus depth eval agent")
    parser.add_argument("--buffer_path", required=True, help="Path to buffer.jsonl")
    parser.add_argument("--eval_every", type=int, default=5, help="Evaluate every N iters")
    parser.add_argument("--goal_path", required=True, help="Path to research_goal.txt")
    parser.add_argument("--max_wait_hours", type=float, default=4.0, help="Stop after N hours")
    args = parser.parse_args()

    goal = Path(args.goal_path).read_text().strip()
    log_path = Path(args.buffer_path).parent / "opus_eval_log.jsonl"
    evaluated_iters: set[int] = set()

    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                evaluated_iters.add(entry.get("iteration", -1))
        logger.info(f"Resuming: already evaluated iters {sorted(evaluated_iters)}")

    start_time = time.time()
    max_wait = args.max_wait_hours * 3600

    logger.info(f"Opus eval agent started. Watching {args.buffer_path}, eval_every={args.eval_every}")

    while time.time() - start_time < max_wait:
        if not Path(args.buffer_path).exists():
            time.sleep(30)
            continue

        current_iter = get_latest_iter(args.buffer_path)

        checkpoints = [i for i in range(args.eval_every, current_iter + 1, args.eval_every)
                       if i not in evaluated_iters]

        for checkpoint_iter in checkpoints:
            entries = get_buffer_plans(args.buffer_path)
            picks = pick_eval_plans(entries)

            if not picks:
                logger.warning("No valid plans in buffer")
                evaluated_iters.add(checkpoint_iter)
                continue

            logger.info(f"Evaluating at iter {checkpoint_iter}: {len(picks)} plans (buffer={len(entries)} total)")

            for label, plan_text, plan_iter, qwen_reward in picks:
                scores = opus_depth_audit(plan_text, goal)
                scores["checkpoint_iter"] = checkpoint_iter
                scores["label"] = label
                scores["plan_iter"] = plan_iter
                scores["qwen_reward"] = round(qwen_reward, 4)
                scores["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

                with open(log_path, "a") as f:
                    f.write(json.dumps(scores) + "\n")

                logger.info(
                    f"  [{label}] Opus={scores.get('total', '?')}/20 "
                    f"(depth={scores.get('depth')}, methods={scores.get('methods')}, "
                    f"feasibility={scores.get('feasibility')}, grounding={scores.get('grounding')}) "
                    f"| Qwen={qwen_reward:.3f}"
                )

            evaluated_iters.add(checkpoint_iter)

        time.sleep(60)

    logger.info("Opus eval agent finished (timeout reached)")


if __name__ == "__main__":
    main()
