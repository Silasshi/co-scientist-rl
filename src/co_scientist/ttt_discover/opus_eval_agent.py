"""Background Opus 4.7 depth evaluation agent.

Polls buffer.jsonl during training runs, extracts the buffer_max plan
every `eval_every` iterations, sends it to Claude Opus 4.7 for a
4-dimension depth audit, and logs scores to opus_eval_log.jsonl.

Runs as a separate process alongside train_cr_v7.py.
Cost: ~$1.50/run (5 evals × ~$0.30/eval).

Usage:
    python opus_eval_agent.py \
        --buffer_path runs/MAIN_v9/buffer.jsonl \
        --eval_every 5 \
        --goal_path projects/ttt_discover/analysis/sanity_check/research_goal.txt
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
You are an expert research methodology reviewer. Score the following research
plan on 4 independent dimensions, each 1-5 (integer). Use the reference
calibration below to anchor your scores.

# Research Goal
{goal}

# Research Plan to Evaluate
{plan}

# Scoring Dimensions

## 1. Mathematical Formalism (1-5)
- 1: No equations or formal statements
- 2: One standard formula without derivation
- 3: One adapted/novel formula for this plan
- 4: Full objective derivation with gradient/update rule
- 5: Novel objective + limit analysis or convergence sketch
  Reference anchor (5/5): J_β entropic objective, ∇J_β policy gradient,
  adaptive β via KL budget, MAX-PUCT with one-character deviation from AlphaZero

## 2. Algorithmic Novelty (1-5)
- 1: Standard toolkit combination (REINFORCE + LoRA + entropy)
- 2: Known techniques with minor adaptation
- 3: Non-obvious combination with clear justification
- 4: Substantive modification to existing algorithm with mechanistic insight
- 5: Novel algorithm design with rigorous justification (e.g., MAX not MEAN
  with limit proof)

## 3. Implementation Realism (1-5)
- 1: Multiple fatal errors (wrong parameter counts, infeasible algorithms)
- 2: 2+ arithmetic inconsistencies
- 3: 1 error or vague on key details
- 4: All numerical claims consistent, key parameters specified
- 5: All claims consistent, specific, and justified

## 4. Empirical Rigor (1-5)
- 1: No concrete baselines or metrics
- 2: Named baselines without prior numbers
- 3: Named baselines with some prior numbers
- 4: Specific benchmarks with published baseline numbers
- 5: Specific open problems with exact prior AI numbers and testable claims

# Output Format
Respond with ONLY this JSON (no other text):
{{"math": <int>, "novelty": <int>, "realism": <int>, "rigor": <int>}}
"""


def get_buffer_max_plan(buffer_path: str) -> tuple[str, int, float]:
    """Read buffer.jsonl and return (plan_text, iteration, aggregate_reward)
    for the entry with the highest aggregate_reward where hard_gate_passed."""
    best_plan = None
    best_reward = -1.0
    best_iter = -1

    with open(buffer_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if not entry.get("hard_gate_passed", True):
                continue
            reward = entry.get("aggregate_reward", 0.0)
            if reward > best_reward:
                best_reward = reward
                best_plan = entry.get("plan_text", "")
                best_iter = entry.get("iteration", -1)

    return best_plan, best_iter, best_reward


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
        scores["total"] = sum(scores.get(k, 0) for k in ["math", "novelty", "realism", "rigor"])
        return scores
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse Opus response: {text[:200]}")
        return {"math": None, "novelty": None, "realism": None, "rigor": None, "total": None, "raw": text}


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
            logger.info(f"Evaluating at iter {checkpoint_iter} (buffer at iter {current_iter})")
            plan_text, best_iter, best_reward = get_buffer_max_plan(args.buffer_path)

            if plan_text is None:
                logger.warning("No valid plan in buffer")
                continue

            scores = opus_depth_audit(plan_text, goal)
            scores["iteration"] = checkpoint_iter
            scores["buffer_max_iter"] = best_iter
            scores["buffer_max_reward"] = best_reward
            scores["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

            with open(log_path, "a") as f:
                f.write(json.dumps(scores) + "\n")

            evaluated_iters.add(checkpoint_iter)
            logger.info(
                f"Iter {checkpoint_iter}: Opus={scores.get('total', '?')}/20 "
                f"(math={scores.get('math')}, novelty={scores.get('novelty')}, "
                f"realism={scores.get('realism')}, rigor={scores.get('rigor')}), "
                f"Qwen={best_reward:.3f}"
            )

        time.sleep(60)

    logger.info("Opus eval agent finished (timeout reached)")


if __name__ == "__main__":
    main()
