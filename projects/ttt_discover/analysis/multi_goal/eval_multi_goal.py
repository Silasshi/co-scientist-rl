"""P2: Multi-goal generalization test.

For each goal from the facebook/research-plan-gen dataset:
  1. Generate 2 plans (zero-shot from base model)
  2. Score on all 9 signals with 235B grader
  3. Pick the better plan, identify bottleneck signal
  4. Do 1 critique-revise cycle (2 candidates, pick best)
  5. Score the revision
  6. Report: did the revision improve? By how much?

This tests whether our signal set + critique-revise mechanism works
across diverse ML research goals, not just the TTT-Discover problem.

Usage:
    PYTHONPATH=src python3 projects/ttt_discover/analysis/multi_goal/eval_multi_goal.py
"""

import json
import logging
import os
import re
import sys
import textwrap
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import tinker
from datasets import load_dataset
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.ten_signal_reward import (
    SIGNALS, SIGNAL_WEIGHTS, aggregate_reward,
    build_single_signal_prompt, parse_scores, SignalSpec,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)
logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
RESULTS_PATH = HERE / "multi_goal_results.jsonl"

POLICY_MODEL = "Qwen/Qwen3-30B-A3B"
GRADER_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
N_GOALS = 5


def build_plan_prompt(goal: str) -> str:
    return textwrap.dedent(f"""
        I will provide you a research scenario. You have to provide me a concise
        yet thoughtful research plan with all details needed to execute it.

        Here is the research scenario.
        Scenario: {goal}

        # Instructions
        - Explain HOW you will do it and WHY it is needed.
        - Do NOT be verbose. Be in present tense.
        - Do not claim to have done experiments.
        - Do not add self-proclaimed praises.

        <think>
        ...your reasoning...
        </think>
        <solution>
        ...your detailed research plan (max 750 words)...
        </solution>
    """).strip()


def build_critique_revise_prompt(goal: str, plan: str, signals: dict, bottleneck_id: str, score: int) -> str:
    spec = next((s for s in SIGNALS if s.id == bottleneck_id), SIGNALS[0])
    sig_lines = "\n".join(f"  {k}: {v}/5" + (" ← WEAKEST" if k == bottleneck_id else "")
                          for k, v in signals.items() if v is not None)
    return textwrap.dedent(f"""
        You are improving a research plan. Its weakest dimension has been identified.

        # Research Goal
        {goal}

        # Current Plan
        {plan}

        # Signal Scores
        {sig_lines}

        # Weakest: {spec.name} (score: {score}/5)
        ## What it measures: {spec.question}
        ## Rubric: {spec.scoring_rubric}

        # Task
        1. <critique>: WHY did this score only {score}/5?
        2. <solution>: revised plan fixing this weakness (max 750 words)

        <critique>...</critique>
        <solution>...</solution>
    """).strip()


def grade_plan(plan_text: str, goal: str, grader_client, renderer, tokenizer) -> dict[str, int]:
    """Grade one plan on all 9 signals (single repeat for speed)."""
    futures = {}
    for spec in SIGNALS:
        prompt = build_single_signal_prompt(goal, plan_text, spec)
        convo = [{"role": "user", "content": prompt}]
        model_input = renderer.build_generation_prompt(convo)
        futures[spec.id] = grader_client.sample(
            model_input, num_samples=1,
            sampling_params=tinker.types.SamplingParams(max_tokens=2048, temperature=0.0),
        )

    scores = {}
    for sig_id, fut in futures.items():
        try:
            result = fut.result(timeout=300)
            raw = renderers.get_text_content(renderer.parse_response(result.sequences[0].tokens)[0])
            parsed = parse_scores(raw)
            if sig_id in parsed and parsed[sig_id]["score"] is not None:
                scores[sig_id] = parsed[sig_id]["score"]
        except Exception as e:
            logger.warning(f"Grading failed for {sig_id}: {type(e).__name__}")
    return scores


def main():
    logger.info(f"Loading dataset...")
    ds = load_dataset("facebook/research-plan-gen", "ml")
    test = ds["test"]
    logger.info(f"Dataset: {len(test)} goals")

    # Pick N diverse goals (spread across the dataset)
    indices = [0, 100, 200, 400, 600][:N_GOALS]
    goals = [(i, test[i]["Goal"], test[i].get("Reference solution", "")) for i in indices]

    logger.info(f"Selected {len(goals)} goals")

    # Setup clients
    service_client = create_service_client()

    policy_client = service_client.create_sampling_client(base_model=POLICY_MODEL)
    grader_client = service_client.create_sampling_client(base_model=GRADER_MODEL)

    tokenizer_policy = get_tokenizer(POLICY_MODEL)
    renderer_policy_name = model_info.get_recommended_renderer_name(POLICY_MODEL)
    renderer_policy = renderers.get_renderer(renderer_policy_name, tokenizer_policy)

    tokenizer_grader = get_tokenizer(GRADER_MODEL)
    renderer_grader_name = model_info.get_recommended_renderer_name(GRADER_MODEL)
    renderer_grader = renderers.get_renderer(renderer_grader_name, tokenizer_grader)

    policy_params = tinker.types.SamplingParams(
        max_tokens=2048, temperature=1.0,
        stop=renderer_policy.get_stop_sequences(),
    )

    results = []

    for goal_idx, goal_text, ref_solution in goals:
        t0 = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Goal [{goal_idx}]: {goal_text[:100]}...")

        # --- Step 1: Generate 2 plans ---
        prompt = build_plan_prompt(goal_text)
        convo = [{"role": "user", "content": prompt}]
        model_input = renderer_policy.build_generation_prompt(convo)

        gen_result = policy_client.sample(
            model_input, num_samples=2, sampling_params=policy_params,
        ).result()

        plans = []
        for seq in gen_result.sequences:
            text = renderers.get_text_content(renderer_policy.parse_response(seq.tokens)[0])
            if "<solution>" in text and "</solution>" not in text:
                text = text.rstrip() + "\n</solution>"
            plans.append(text)

        logger.info(f"Generated {len(plans)} plans")

        # --- Step 2: Score both plans ---
        plan_scores = []
        for p_idx, plan in enumerate(plans):
            scores = grade_plan(plan, goal_text, grader_client, renderer_grader, tokenizer_grader)
            agg = aggregate_reward(scores)
            plan_scores.append({"text": plan, "scores": scores, "aggregate": agg})
            logger.info(f"  Plan {p_idx}: agg={agg:.3f} scores={scores}")

        # Pick the better plan
        best = max(plan_scores, key=lambda p: p["aggregate"])
        logger.info(f"Best plan: agg={best['aggregate']:.3f}")

        # --- Step 3: Identify bottleneck ---
        valid_scores = {k: v for k, v in best["scores"].items() if v is not None}
        if not valid_scores:
            logger.warning("No valid scores, skipping")
            continue
        bottleneck_id = min(valid_scores, key=valid_scores.get)
        bottleneck_score = valid_scores[bottleneck_id]
        logger.info(f"Bottleneck: {bottleneck_id}={bottleneck_score}")

        # --- Step 4: Critique-revise (2 candidates) ---
        cr_prompt = build_critique_revise_prompt(
            goal_text, best["text"], best["scores"], bottleneck_id, bottleneck_score,
        )
        cr_input = renderer_policy.build_generation_prompt(
            [{"role": "user", "content": cr_prompt}]
        )
        cr_result = policy_client.sample(
            cr_input, num_samples=2, sampling_params=policy_params,
        ).result()

        revisions = []
        for seq in cr_result.sequences:
            raw = renderers.get_text_content(renderer_policy.parse_response(seq.tokens)[0])
            # Extract revised plan
            sol_match = re.search(r"<solution>(.*?)</solution>", raw, re.DOTALL)
            rev_text = sol_match.group(1).strip() if sol_match else raw
            if "<solution>" not in rev_text:
                rev_text = f"<solution>\n{rev_text}\n</solution>"
            revisions.append(rev_text)

        # --- Step 5: Score revisions, pick best ---
        rev_scores = []
        for r_idx, rev in enumerate(revisions):
            scores = grade_plan(rev, goal_text, grader_client, renderer_grader, tokenizer_grader)
            agg = aggregate_reward(scores)
            delta = agg - best["aggregate"]
            rev_scores.append({"text": rev, "scores": scores, "aggregate": agg, "delta": delta})
            logger.info(f"  Revision {r_idx}: agg={agg:.3f} delta={delta:+.3f}")

        best_rev = max(rev_scores, key=lambda r: r["delta"])
        elapsed = time.time() - t0

        result = {
            "goal_idx": goal_idx,
            "goal": goal_text[:200],
            "original_aggregate": best["aggregate"],
            "original_scores": best["scores"],
            "bottleneck": bottleneck_id,
            "best_revision_aggregate": best_rev["aggregate"],
            "best_revision_scores": best_rev["scores"],
            "best_revision_delta": best_rev["delta"],
            "revision_positive": best_rev["delta"] > 0,
            "elapsed_s": elapsed,
        }
        results.append(result)

        logger.info(
            f"RESULT: original={best['aggregate']:.3f} → revision={best_rev['aggregate']:.3f} "
            f"delta={best_rev['delta']:+.3f} {'✓' if best_rev['delta'] > 0 else '✗'} "
            f"({elapsed:.0f}s)"
        )

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    print("\n" + "=" * 70)
    print("MULTI-GOAL GENERALIZATION RESULTS")
    print("=" * 70)
    n_positive = sum(1 for r in results if r["revision_positive"])
    print(f"\nGoals tested: {len(results)}")
    print(f"Revision improved: {n_positive}/{len(results)} ({100*n_positive/max(1,len(results)):.0f}%)")
    print(f"\n{'Goal':<6} {'Original':<10} {'Revised':<10} {'Delta':<8} {'Bottleneck':<20} {'OK?'}")
    for r in results:
        ok = "✓" if r["revision_positive"] else "✗"
        print(f"[{r['goal_idx']}]  {r['original_aggregate']:<10.3f} {r['best_revision_aggregate']:<10.3f} "
              f"{r['best_revision_delta']:+8.3f} {r['bottleneck']:<20} {ok}")
    print("=" * 70)


if __name__ == "__main__":
    main()
