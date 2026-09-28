"""
Generate multi-turn conversation curriculum for research plan training.

For each goal in the dataset:
  1. Use Sonnet (via openclaw agent gateway) to decompose goal into 3 key insights
  2. Build templated student messages from insights
  3. Save as JSONL

Usage:
  python generate_curriculum.py                          # Train split (default)
  python generate_curriculum.py config.split=test        # Test split
  python generate_curriculum.py config.num_goals=100     # Subset for testing
"""

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
from datasets import load_dataset

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    split: str = "train"
    output_dir: str = "/home/silas/co-scientist-project/data/curriculum"
    num_goals: int = 0  # 0 = all goals in split
    num_insights: int = 3
    sonnet_timeout: int = 60
    max_retries: int = 2
    concurrency: int = 8  # Parallel Sonnet calls
    resume: bool = True  # Skip already-generated goals


def _call_sonnet(message: str, session_id: str, timeout: int = 60) -> str:
    result = subprocess.run(
        ["openclaw", "agent",
         "--message", message,
         "--session-id", session_id,
         "--json", "--timeout", str(timeout)],
        capture_output=True, text=True, timeout=timeout + 15,
    )
    if result.returncode != 0:
        raise RuntimeError(f"openclaw agent failed: {result.stderr[:300]}")
    data = json.loads(result.stdout)
    return data["result"]["payloads"][0]["text"]


def decompose_into_insights(goal: str, num_insights: int, goal_id: int,
                            timeout: int, max_retries: int) -> list[str]:
    prompt = (
        f"Given this research scenario, identify {num_insights} key insights or fragments "
        f"that a researcher might develop before formulating a full plan. "
        f"Each insight should be a partial observation, hypothesis, or constraint — "
        f"NOT a complete solution, just a starting point for discussion.\n\n"
        f"Research scenario:\n{goal}\n\n"
        f'Return ONLY a JSON array of strings, e.g.: ["insight 1", "insight 2", "insight 3"]'
    )
    for attempt in range(max_retries + 1):
        try:
            text = _call_sonnet(prompt, session_id=f"curr-insights-{goal_id}", timeout=timeout)
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                if isinstance(parsed, list) and len(parsed) >= 1:
                    return [str(s) for s in parsed[:num_insights]]
            # Fallback: split by newlines
            lines = [l.strip().lstrip("-•0123456789.) ") for l in text.split("\n") if l.strip() and len(l.strip()) > 10]
            if lines:
                return lines[:num_insights]
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"[Goal {goal_id}] Attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(2)
            else:
                raise
    return [f"Key aspect {i+1} of the research problem" for i in range(num_insights)]


def build_student_messages(goal: str, insights: list[str]) -> list[str]:
    messages = []

    # Turn 0: Present first insight
    messages.append(
        f"I've been thinking about a research problem and have some initial ideas. "
        f"Here's one insight I've been considering:\n\n"
        f'"{insights[0]}"\n\n'
        f"What research directions could this lead to? What approaches would you suggest?"
    )

    # Turn 1: Present second insight (or rephrase first if only 1)
    if len(insights) >= 2:
        messages.append(
            f"That's really helpful. I also noticed something related:\n\n"
            f'"{insights[1]}"\n\n'
            f"How does this connect to the directions you mentioned? "
            f"Does it change which approach you'd recommend?"
        )
    else:
        messages.append(
            "That's really helpful. Can you go deeper on the methodology? "
            "What specific techniques or frameworks would be most appropriate, "
            "and what are the potential pitfalls?"
        )

    # Turn 2: Present full goal and request plan
    messages.append(
        f"Those are great points. Let me share the full research goal I'm working on:\n\n"
        f"{goal}\n\n"
        f"Based on our discussion, can you synthesize a complete research plan? "
        f"Please put your plan inside <solution></solution> tags and keep it under 750 words. "
        f"Make it self-contained, detailed, and explain HOW and WHY for each step."
    )

    # Turn 3: Request revision
    if len(insights) >= 3:
        messages.append(
            f'One more thing I want to make sure is addressed: "{insights[2]}". '
            f"Can you strengthen the plan's methodology to account for this, "
            f"and address any potential limitations? "
            f"Please provide the revised plan inside <solution></solution> tags, under 750 words."
        )
    else:
        messages.append(
            "Can you strengthen the weakest parts of this plan? "
            "Focus on making the methodology more concrete and addressing potential limitations. "
            "Please provide the revised plan inside <solution></solution> tags, under 750 words."
        )

    return messages


def main(config: Config):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load dataset
    logger.info(f"Loading dataset ({config.split} split)...")
    data = load_dataset("facebook/research-plan-gen", "ml")
    dataset = data[config.split]
    n_goals = config.num_goals if config.num_goals > 0 else len(dataset)
    n_goals = min(n_goals, len(dataset))
    logger.info(f"Processing {n_goals} goals from {config.split} split ({len(dataset)} total)")

    # Output path
    output_path = os.path.join(config.output_dir, f"ml_{config.split}_curriculum.jsonl")
    os.makedirs(config.output_dir, exist_ok=True)

    # Resume: load existing entries
    existing_ids = set()
    if config.resume and os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                entry = json.loads(line)
                existing_ids.add(entry["goal_id"])
        logger.info(f"Resuming: {len(existing_ids)} goals already generated")

    # Process goals in parallel
    t_start = time.time()
    generated = 0
    failed = 0
    pending_ids = [gid for gid in range(n_goals) if gid not in existing_ids]
    logger.info(f"Generating {len(pending_ids)} new goals with {config.concurrency} workers...")

    import threading
    write_lock = threading.Lock()

    def _process_goal(goal_id):
        goal = dataset[goal_id]["Goal"]
        rubric = dataset[goal_id]["Rubric"]
        ref_solution = dataset[goal_id]["Reference solution"]
        insights = decompose_into_insights(
            goal, config.num_insights, goal_id,
            config.sonnet_timeout, config.max_retries,
        )
        student_messages = build_student_messages(goal, insights)
        return {
            "goal_id": goal_id,
            "goal": goal,
            "rubric": rubric,
            "reference_solution": ref_solution,
            "insights": insights,
            "student_messages": student_messages,
        }

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with open(output_path, "a" if config.resume else "w") as out_f:
        with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
            futures = {executor.submit(_process_goal, gid): gid for gid in pending_ids}
            for future in as_completed(futures):
                gid = futures[future]
                try:
                    entry = future.result()
                    with write_lock:
                        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        out_f.flush()
                        generated += 1
                    if generated % 50 == 0:
                        elapsed = time.time() - t_start
                        rate = generated / elapsed * 60
                        remaining = (len(pending_ids) - generated) / rate if rate > 0 else 0
                        logger.info(
                            f"Progress: {generated + len(existing_ids)}/{n_goals} "
                            f"({generated} new, {failed} failed) "
                            f"| {rate:.1f} goals/min | ~{remaining:.0f} min remaining"
                        )
                except Exception as e:
                    logger.error(f"[Goal {gid}] Failed: {e}")
                    failed += 1

    elapsed = time.time() - t_start
    total = len(existing_ids) + generated
    logger.info(f"Done: {total}/{n_goals} goals ({generated} new, {failed} failed) in {elapsed:.0f}s")
    logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    chz.entrypoint(main)
