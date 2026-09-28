"""
Eval for IBT-trained checkpoints.

Two modes:
  - single_pass: Generate 1 (or N) plan(s) per test goal, grade with canonical
                  grader (no hint appendix). Directly comparable to bestversion
                  eval (0.69 rubric).
  - iterative:   Run K turns with hints per test goal (same loop as training).
                  Tests whether IBT generalizes to unseen goals.

Usage:
  # Single-pass eval (comparable to bestversion)
  python eval_ibt.py checkpoint_path=runs/2026/4/ibt/1/single_chain num_samples=1

  # Iterative eval on test set
  python eval_ibt.py checkpoint_path=runs/2026/4/ibt/1/single_chain mode=iterative num_turns=5

  # Base model eval (no checkpoint — raw 4B)
  python eval_ibt.py checkpoint_path=none
"""

import json
import logging
import os
import re
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
import datasets
import tinker
from co_scientist.shared.api_profiles import create_service_client
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from datasets import load_dataset

from co_scientist.ibt.train_ibt import (
    build_plan_prompt,
    build_grader_with_hint_prompt,
    compute_rubric_score,
    compute_reward,
    extract_hint,
    extract_solution_text,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


# ============================================================
# Canonical grader prompt (NO hint appendix — for fair eval)
# ============================================================

def build_canonical_grader_prompt(
    scenario: str,
    rubric_items: list[str],
    proposed_plan: str,
    reference_solution: str | None = None,
) -> str:
    """Canonical grader prompt identical to eval_only.py / best_ver.py.

    Does NOT include <improvement_hint> section — scores are directly
    comparable to bestversion eval numbers.
    """
    rubric_block = "\n".join(
        [f"Item {i+1}: {item}" for i, item in enumerate(rubric_items)]
    )

    desiderata_text = textwrap.dedent("""
    **DESIDERATA**
    1. HANDLES ALL CRITERIA: Does the plan satisfy all criteria mentioned in the rubric item? An exception is if the criteria says "such as", "for example", or "including", the response does not have to include the same examples listed to meet the criteria, but whatever is provided must be valid and reasonable.
    2. DETAILED, SPECIFIC SOLUTION: Does the part of the plan relevant to satisfying this rubric item include fully specified details on HOW to implement it? There should be no self-proclaimed claims of handling something without doing so. There should be no vague terms, ambiguity, or lack of clarity. It should be described in simple to understand language.
    3. NO OVERLOOKED FLAWS OR WEAKNESSES: Are there any important overlooked flaws or weaknesses in the part of the plan addressing this rubric item that invalidate its satisfaction of the rubric item?
    4. WELL JUSTIFIED RATIONALE: Is the part of the plan relevant to this grading item well-motivated and justified? For example, are there convincing arguments provided for how the plan handles this grading item is better than simpler solutions or alternate hypotheses?
    5. COST AND EFFORT EFFICIENT: Does the plan handle this item efficiently without unnecessary complexity?
    6. NO ETHICAL ISSUES: Does this part of the plan have any potential for negative consequences, or is it ethically problematic?
    7. CONSISTENT WITH OVERALL PLAN: Is this part of the plan consistent with the rest of the plan? Check if it contradicts any other parts of the plan.
    """).strip()

    prompt = textwrap.dedent(f"""
        Evaluate if the Proposed Research Plan satisfies the Research Scenario based on the provided evaluation criteria.

        # Research Scenario
        {scenario}

        You have to evaluate each of the rubric items provided below.

        # Rubric
        {rubric_block}
    """).strip()

    if reference_solution is not None:
        prompt += textwrap.dedent(f"""
            # Reference Solution
            Here is a reference solution written by an expert:
            {reference_solution}

            \u2022 It is just meant to demonstrate one possible approach that satisfies the scenario. It is not necessary for the proposed research plan you are grading to match all details in the reference solution.
            \u2022 The Research Plan you have to grade might have different design choices. This is okay, if the choices are valid, and supported with correct rationale.
        """).strip()

    prompt += textwrap.dedent(f"""
        # Proposed Research Plan
        {proposed_plan}

        # Instructions
        First, come up with weaknesses of the proposed plan specific to the scenario. Then, return the following nested XML block for each of the grading items (always close opened XML tags):

        <rubric>
            <item num=1>
                <criteria>Repeat the rubric item string you are checking here. </criteria>
                <reasoning>
                Analyze how well the proposed plan satisfies EACH of the following 7 GENERAL DESIDERATA with respect to the rubric item, using an integer satisfaction level.
                {desiderata_text}

                For EACH desideratum, assign a satisfaction level according to this scale:

                Level 0 — NOT SATISFIED:
                The plan does not meaningfully satisfy this desideratum.

                Level 1 — WEAKLY SATISFIED:
                The plan touches on this desideratum, but in a vague, superficial, or insufficient way.

                Level 2 — PARTIALLY SATISFIED:
                The plan satisfies this desideratum to a reasonable extent, but with notable gaps,
                weaknesses, or missing justifications.

                Level 3 — FULLY SATISFIED:
                The plan clearly, concretely, and convincingly satisfies this desideratum.
                No major issues are apparent.

                - Be skeptical, careful, and come up with valid criticisms. Be as strict as possible, while being unbiased and reasonable.
                - Note that the plan should not just say it satisfies these desiderata, don't be fooled by that. Check carefully WHETHER, HOW and WHY the proposed plan meets each desiderata for this rubric item one by one.
                - Based on the above analysis, list the satisfaction level for each desiderata. Don't be lazy and assign the same level to all desiderata. Be precise and careful.
                </reasoning>
                <desiderata num=1>
                Repeat for all 7 desiderata:
                    <level>[Satisfaction level for desiderata, put a single integer from 0, 1, 2, 3]</level>
                </desiderata>
            </item>

            ... Similarly, for all rubric items...
        </rubric>
    """).strip()
    return prompt


# ============================================================
# Config
# ============================================================

@chz.chz
class Config:
    base_url: str | None = None
    api_profile: str | None = "NEW"

    # Checkpoint to evaluate. "none" = base model without LoRA.
    checkpoint_path: str = "runs/2026/4/ibt/1/single_chain"

    policy_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    grader_model: str = "Qwen/Qwen3-30B-A3B"

    # Eval mode: "single_pass" or "iterative"
    mode: str = "single_pass"

    ml_data: bool = True
    arxiv_data: bool = False
    pubmed_data: bool = False

    num_goals: int = 50  # 0 = all test goals

    # single_pass: how many samples per goal (1 = mean-of-1, 8 = best-of-8)
    num_samples: int = 1

    # iterative: how many turns per goal
    num_turns: int = 5

    max_tokens: int = 2048
    temperature: float = 1.0
    grader_max_tokens: int = 8192
    grader_temperature: float = 0.0

    max_word_count: int = 750
    target_word_count: int = 600
    scale_length_bonus: float = 140.0
    scaling_factor: float = 0.08
    min_words: int = 30


# ============================================================
# Main
# ============================================================

def main(config: Config):
    assert config.mode in ("single_pass", "iterative")

    # ── Output dir ──
    eval_name = f"eval_{config.mode}"
    if config.mode == "single_pass" and config.num_samples > 1:
        eval_name += f"_n{config.num_samples}"
    eval_dir = os.path.join(config.checkpoint_path, eval_name)
    os.makedirs(eval_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s:%(lineno)d [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(eval_dir, "eval.log")),
        ],
    )

    logger.info(f"IBT Eval: mode={config.mode}, checkpoint={config.checkpoint_path}")
    logger.info(f"Policy: {config.policy_model}, Grader: {config.grader_model}")

    # ── Tokenizer & renderer ──
    tokenizer = get_tokenizer(config.policy_model)
    renderer_name = model_info.get_recommended_renderer_name(config.policy_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    grader_tokenizer = get_tokenizer(config.grader_model)
    grader_renderer_name = model_info.get_recommended_renderer_name(config.grader_model)
    grader_renderer = renderers.get_renderer(grader_renderer_name, grader_tokenizer)

    # ── Dataset (TEST set) ──
    if config.ml_data:
        data = load_dataset("facebook/research-plan-gen", "ml")
    elif config.arxiv_data:
        data = load_dataset("facebook/research-plan-gen", "arxiv")
    elif config.pubmed_data:
        data = load_dataset("facebook/research-plan-gen", "pubmed")
    assert isinstance(data, datasets.DatasetDict)
    dataset = data["test"]
    logger.info(f"Test set: {len(dataset)} goals")

    # ── Clients ──
    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )

    # Load checkpoint or use base model
    if config.checkpoint_path.lower() == "none":
        logger.info("Using base model (no checkpoint)")
        sampling_client = service_client.create_sampling_client(
            base_model=config.policy_model
        )
    else:
        last_ckpt = checkpoint_utils.get_last_checkpoint(config.checkpoint_path)
        if last_ckpt is None:
            logger.error(f"No checkpoint found at {config.checkpoint_path}")
            return
        logger.info(f"Loading checkpoint: {last_ckpt}")
        training_client = service_client.create_training_client_from_state_with_optimizer(
            last_ckpt["state_path"]
        )
        sr = training_client.save_weights_for_sampler(name="eval").result()
        sampling_client = service_client.create_sampling_client(model_path=sr.path)

    grader_client = service_client.create_sampling_client(
        base_model=config.grader_model
    )

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )

    grader_sampling_params = tinker.types.SamplingParams(
        max_tokens=config.grader_max_tokens,
        stop=grader_renderer.get_stop_sequences(),
        temperature=config.grader_temperature,
    )

    # ── Log files ──
    f_summary = open(os.path.join(eval_dir, "eval_summary.jsonl"), "w")
    f_logs = open(os.path.join(eval_dir, "eval_logs.jsonl"), "w")

    all_rubric_scores = []
    all_best_scores = []
    t_start = time.time()

    # ============================================================
    if config.mode == "single_pass":
        _eval_single_pass(
            config, dataset, renderer, grader_renderer,
            sampling_client, grader_client,
            sampling_params, grader_sampling_params,
            f_summary, f_logs,
            all_rubric_scores, all_best_scores,
        )
    else:
        _eval_iterative(
            config, dataset, renderer, grader_renderer,
            sampling_client, grader_client,
            sampling_params, grader_sampling_params,
            f_summary, f_logs,
            all_rubric_scores, all_best_scores,
        )

    # ── Final summary ──
    f_summary.close()
    f_logs.close()

    if all_rubric_scores:
        summary = {
            "mode": config.mode,
            "checkpoint": config.checkpoint_path,
            "n_goals": len(all_best_scores),
            "rubric/mean": float(np.mean(all_rubric_scores)),
            "rubric/std": float(np.std(all_rubric_scores)),
            "rubric/best_per_goal_mean": float(np.mean(all_best_scores)),
            "rubric/best_per_goal_std": float(np.std(all_best_scores)),
            "time": time.time() - t_start,
        }
        logger.info(f"\n{'='*60}")
        logger.info(f"EVAL COMPLETE")
        logger.info(f"  Mode: {config.mode}")
        logger.info(f"  Goals: {len(all_best_scores)}")
        logger.info(f"  Mean rubric: {summary['rubric/mean']:.3f}")
        logger.info(f"  Best-per-goal mean: {summary['rubric/best_per_goal_mean']:.3f}")
        logger.info(f"  Time: {summary['time']:.0f}s")

        with open(os.path.join(eval_dir, "eval_result.json"), "w") as f:
            json.dump(summary, f, indent=2)


# ============================================================
# Single-pass eval
# ============================================================

def _eval_single_pass(
    config, dataset, renderer, grader_renderer,
    sampling_client, grader_client,
    sampling_params, grader_sampling_params,
    f_summary, f_logs,
    all_rubric_scores, all_best_scores,
):
    """Generate N plans per goal, grade with canonical grader. No hints."""
    n_goals = config.num_goals if config.num_goals > 0 else len(dataset)
    n_goals = min(n_goals, len(dataset))
    for goal_idx in range(n_goals):
        goal_row = dataset[goal_idx]
        goal_text = goal_row["Goal"]
        rubric_items = goal_row["Rubric"]
        ref_solution = goal_row["Reference solution"]

        logger.info(f"Goal {goal_idx}/{len(dataset)}: {goal_text[:80]}...")

        # Generate
        prompt_text = build_plan_prompt(scenario=goal_text, hint=None)
        convo = [{"role": "user", "content": prompt_text}]
        model_input = renderer.build_generation_prompt(convo)

        result = sampling_client.sample(
            prompt=model_input,
            num_samples=config.num_samples,
            sampling_params=sampling_params,
        ).result()

        # Parse plans & launch graders
        plans = []
        grader_futures = []
        for seq in result.sequences:
            plan_text = renderers.get_text_content(
                renderer.parse_response(seq.tokens)[0]
            )
            if "<solution>" in plan_text and "</solution>" not in plan_text:
                plan_text = plan_text.rstrip() + "\n</solution>"
            plans.append(plan_text)

            # Canonical grader (no hint appendix)
            grader_prompt = build_canonical_grader_prompt(
                scenario=goal_text,
                rubric_items=rubric_items,
                proposed_plan=plan_text,
                reference_solution=ref_solution,
            )
            grader_input = grader_renderer.build_generation_prompt(
                [{"role": "user", "content": grader_prompt}]
            )
            grader_futures.append(
                grader_client.sample(
                    grader_input, num_samples=1,
                    sampling_params=grader_sampling_params,
                )
            )

        # Collect grades
        goal_scores = []
        for k, gf in enumerate(grader_futures):
            grader_result = gf.result()
            grader_text = renderers.get_text_content(
                grader_renderer.parse_response(grader_result.sequences[0].tokens)[0]
            )
            rubric_score = compute_rubric_score(grader_text)
            solution_text = extract_solution_text(plans[k])
            word_count = len(solution_text.split())

            goal_scores.append(rubric_score)
            all_rubric_scores.append(rubric_score)

            f_logs.write(json.dumps({
                "goal_idx": goal_idx,
                "sample_idx": k,
                "rubric_score": rubric_score,
                "word_count": word_count,
                "plan": plans[k][:500],
            }) + "\n")

        best_score = max(goal_scores) if goal_scores else 0.0
        mean_score = float(np.mean(goal_scores)) if goal_scores else 0.0
        all_best_scores.append(best_score)

        f_summary.write(json.dumps({
            "goal_idx": goal_idx,
            "rubric/mean": mean_score,
            "rubric/best": best_score,
            "rubric/scores": goal_scores,
            "n_samples": len(goal_scores),
        }) + "\n")

        logger.info(f"  mean={mean_score:.3f}, best={best_score:.3f} (n={len(goal_scores)})")

        if (goal_idx + 1) % 10 == 0:
            f_summary.flush()
            f_logs.flush()
            logger.info(
                f"  Running average: mean={np.mean(all_rubric_scores):.3f}, "
                f"best-per-goal={np.mean(all_best_scores):.3f}"
            )


# ============================================================
# Iterative eval (with hints, on test set)
# ============================================================

def _eval_iterative(
    config, dataset, renderer, grader_renderer,
    sampling_client, grader_client,
    sampling_params, grader_sampling_params,
    f_summary, f_logs,
    all_rubric_scores, all_best_scores,
):
    """Run K turns with hints per test goal. NO weight updates during eval."""
    n_goals = config.num_goals if config.num_goals > 0 else len(dataset)
    n_goals = min(n_goals, len(dataset))
    for goal_idx in range(n_goals):
        goal_row = dataset[goal_idx]
        goal_text = goal_row["Goal"]
        rubric_items = goal_row["Rubric"]
        ref_solution = goal_row["Reference solution"]

        logger.info(f"Goal {goal_idx}/{len(dataset)}: {goal_text[:80]}...")

        hint = None
        goal_turn_scores = []

        for turn_idx in range(config.num_turns):
            # Generate with hint
            prompt_text = build_plan_prompt(scenario=goal_text, hint=hint)
            convo = [{"role": "user", "content": prompt_text}]
            model_input = renderer.build_generation_prompt(convo)

            result = sampling_client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
            ).result()

            plan_text = renderers.get_text_content(
                renderer.parse_response(result.sequences[0].tokens)[0]
            )
            if "<solution>" in plan_text and "</solution>" not in plan_text:
                plan_text = plan_text.rstrip() + "\n</solution>"

            # Grade with hint prompt (to get hint for next turn)
            grader_prompt = build_grader_with_hint_prompt(
                scenario=goal_text,
                rubric_items=rubric_items,
                proposed_plan=plan_text,
                reference_solution=ref_solution,
            )
            grader_input = grader_renderer.build_generation_prompt(
                [{"role": "user", "content": grader_prompt}]
            )
            grader_result = grader_client.sample(
                grader_input, num_samples=1,
                sampling_params=grader_sampling_params,
            ).result()

            grader_text = renderers.get_text_content(
                grader_renderer.parse_response(grader_result.sequences[0].tokens)[0]
            )

            rubric_score = compute_rubric_score(grader_text)
            plan_hint = extract_hint(grader_text)
            solution_text = extract_solution_text(plan_text)
            word_count = len(solution_text.split())

            if plan_hint:
                hint = plan_hint

            goal_turn_scores.append(rubric_score)
            all_rubric_scores.append(rubric_score)

            f_logs.write(json.dumps({
                "goal_idx": goal_idx,
                "turn_idx": turn_idx,
                "rubric_score": rubric_score,
                "word_count": word_count,
                "hint_length": len(plan_hint),
                "plan": plan_text[:500],
            }) + "\n")

        best_score = max(goal_turn_scores) if goal_turn_scores else 0.0
        all_best_scores.append(best_score)

        f_summary.write(json.dumps({
            "goal_idx": goal_idx,
            "rubric_trajectory": goal_turn_scores,
            "rubric_first": goal_turn_scores[0] if goal_turn_scores else 0.0,
            "rubric_last": goal_turn_scores[-1] if goal_turn_scores else 0.0,
            "rubric_best": best_score,
            "improvement": (goal_turn_scores[-1] - goal_turn_scores[0]) if goal_turn_scores else 0.0,
        }) + "\n")

        logger.info(
            f"  trajectory={[f'{s:.3f}' for s in goal_turn_scores]}, best={best_score:.3f}"
        )

        if (goal_idx + 1) % 10 == 0:
            f_summary.flush()
            f_logs.flush()
            logger.info(
                f"  Running average: mean={np.mean(all_rubric_scores):.3f}, "
                f"best-per-goal={np.mean(all_best_scores):.3f}"
            )


if __name__ == "__main__":
    chz.nested_entrypoint(main)
