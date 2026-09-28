#!/usr/bin/env python3
"""
Offline pilot: test whether iterative refinement improves research plan scores.

For each of N goals from the ML split:
  1. Generate an initial plan (best_ver214 checkpoint)
  2. Grade it (SDPO detailed grader, base model)
  3. Extract structured feedback
  4. Generate a refined plan using the feedback
  5. Grade the refined plan
  6. Compare: delta_rubric, improvement rate, per-desiderata changes
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Imports from existing codebase
# ---------------------------------------------------------------------------
import tinker
import tinker.types
from datasets import load_dataset
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from co_scientist.trainers.api_profiles import create_service_client
from co_scientist.trainers.grpo.best_ver import (
    build_research_plan_prompt,
)
from co_scientist.trainers.sdpo.train_sdpo import (
    build_grader_prompt as build_detailed_grader_prompt,
    build_self_teacher_prompt,
    check_format_compliance,
    compute_rubric_reward_from_xml,
    extract_solution_text,
    extract_weaknesses_from_grader,
    is_valid_grader_xml,
)
from warmstart_bestver214_common import (
    SOURCE_BATCH,
    SOURCE_RUN,
    _load_checkpoint_entry,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-30B-A3B"
N_GOALS = 20
MAX_TOKENS = 2048
GRADER_MAX_TOKENS = 12288
TEMPERATURE = 1.0
GRADER_TEMPERATURE = 0.0
TARGET_WORD_COUNT = 600
MAX_WORD_COUNT = 750
OUTPUT_PATH = PROJECT_ROOT / "analysis/results/refinement_pilot.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_per_item_levels(xml_text: str) -> list[list[int]]:
    """Parse per-item, per-desiderata level vectors from grader XML."""
    item_blocks = re.findall(
        r"<item\s+num=.*?>.*?</item>", xml_text, flags=re.DOTALL
    )
    result = []
    for item_xml in item_blocks:
        levels = [
            int(l)
            for l in re.findall(r"<level>(\d+)</level>", item_xml)
            if l.isdigit()
        ]
        if len(levels) < 7:
            levels = levels + [0] * (7 - len(levels))
        levels = levels[:7]
        result.append(levels)
    return result


def grade_plan(
    renderer,
    grader_client,
    goal: str,
    rubric: list[str],
    ref_sol: str | None,
    plan_text: str,
) -> dict:
    """Grade a single plan using the SDPO detailed grader. Returns scoring dict."""
    grader_prompt_text = build_detailed_grader_prompt(
        scenario=goal,
        rubric_items=rubric,
        proposed_plan=plan_text,
        reference_solution=ref_sol,
        grader_prompt_mode="detailed",
        fast_grader_mode=True,
    )
    grader_input = renderer.build_generation_prompt(
        [{"role": "user", "content": grader_prompt_text}]
    )
    grader_result = grader_client.sample(
        prompt=grader_input,
        num_samples=1,
        sampling_params=tinker.types.SamplingParams(
            max_tokens=GRADER_MAX_TOKENS,
            temperature=GRADER_TEMPERATURE,
        ),
    ).result()

    parsed_msg, _ = renderer.parse_response(
        grader_result.sequences[0].tokens
    )
    xml_text = renderers.get_text_content(parsed_msg)

    rubric_score = compute_rubric_reward_from_xml(xml_text)
    valid_xml = is_valid_grader_xml(xml_text)
    sol_text = extract_solution_text(plan_text) or plan_text
    word_count = len(sol_text.split())
    compliant = check_format_compliance(plan_text, MAX_WORD_COUNT)

    return {
        "grader_xml": xml_text,
        "rubric_score": rubric_score,
        "is_valid_xml": valid_xml,
        "is_compliant": compliant,
        "word_count": word_count,
        "per_item_levels": extract_per_item_levels(xml_text),
    }


def print_summary(results: list[dict]) -> None:
    """Print aggregate statistics from pilot results."""
    n = len(results)
    valid = [
        r
        for r in results
        if r["original_valid_xml"] and r["refined_valid_xml"]
    ]
    n_valid = len(valid)

    if n_valid == 0:
        print("\nNo valid results to summarize.")
        return

    orig_scores = [r["original_rubric_score"] for r in valid]
    ref_scores = [r["refined_rubric_score"] for r in valid]
    deltas = [r["delta_rubric_score"] for r in valid]

    improved = sum(1 for d in deltas if d > 0)
    regressed = sum(1 for d in deltas if d < 0)
    unchanged = sum(1 for d in deltas if d == 0)

    print(f"\n{'=' * 60}")
    print(f"REFINEMENT PILOT RESULTS ({n} goals, {n_valid} valid)")
    print(f"{'=' * 60}")
    print(
        f"Original  mean rubric: {np.mean(orig_scores):.4f} "
        f"+/- {np.std(orig_scores):.4f}"
    )
    print(
        f"Refined   mean rubric: {np.mean(ref_scores):.4f} "
        f"+/- {np.std(ref_scores):.4f}"
    )
    print(f"Mean delta:            {np.mean(deltas):+.4f}")
    print(f"Median delta:          {np.median(deltas):+.4f}")
    print(f"Improved:  {improved}/{n_valid} ({100 * improved / n_valid:.1f}%)")
    print(
        f"Regressed: {regressed}/{n_valid} ({100 * regressed / n_valid:.1f}%)"
    )
    print(
        f"Unchanged: {unchanged}/{n_valid} ({100 * unchanged / n_valid:.1f}%)"
    )

    # Word count changes
    orig_wc = [r["original_word_count"] for r in valid]
    ref_wc = [r["refined_word_count"] for r in valid]
    print(f"\nWord count: {np.mean(orig_wc):.0f} -> {np.mean(ref_wc):.0f}")

    # Format compliance
    orig_comply = sum(r["original_compliant"] for r in valid)
    ref_comply = sum(r["refined_compliant"] for r in valid)
    print(f"Format compliant: {orig_comply}/{n_valid} -> {ref_comply}/{n_valid}")

    # Per-desiderata breakdown (D1-D7)
    desiderata_names = [
        "HANDLES ALL CRITERIA",
        "DETAILED, SPECIFIC",
        "NO OVERLOOKED FLAWS",
        "WELL JUSTIFIED",
        "COST EFFICIENT",
        "NO ETHICAL ISSUES",
        "CONSISTENT W/ PLAN",
    ]
    print(f"\nPer-desiderata mean level changes:")
    for d_idx in range(7):
        orig_d = []
        ref_d = []
        for r in valid:
            for item_levels in r["original_per_item_levels"]:
                if len(item_levels) > d_idx:
                    orig_d.append(item_levels[d_idx])
            for item_levels in r["refined_per_item_levels"]:
                if len(item_levels) > d_idx:
                    ref_d.append(item_levels[d_idx])
        if orig_d and ref_d:
            delta_d = np.mean(ref_d) - np.mean(orig_d)
            print(
                f"  D{d_idx + 1} ({desiderata_names[d_idx]:20s}): "
                f"{np.mean(orig_d):.2f} -> {np.mean(ref_d):.2f} "
                f"({delta_d:+.2f})"
            )

    # Correlation: does refinement help more for low-scoring originals?
    if len(valid) >= 5:
        from scipy import stats

        corr, pval = stats.pearsonr(orig_scores, deltas)
        print(f"\nCorrelation (orig_score vs delta): r={corr:.3f}, p={pval:.3f}")
        if corr < -0.3 and pval < 0.1:
            print("  -> Refinement helps MORE for low-scoring plans (good)")
        elif corr > 0.3 and pval < 0.1:
            print("  -> Refinement helps MORE for high-scoring plans (unexpected)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("Loading checkpoint from best_ver214...")
    ckpt = _load_checkpoint_entry(SOURCE_RUN, SOURCE_BATCH)
    logger.info(f"Checkpoint state_path: {ckpt.get('state_path')}")

    # Setup tokenizer and renderer
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Create service client
    service_client = create_service_client(base_url=None, api_profile=None)

    # Policy client: best_ver214 checkpoint with LoRA
    training_client = (
        service_client.create_training_client_from_state_with_optimizer(
            ckpt["state_path"]
        )
    )
    sampling_result = training_client.save_weights_for_sampler(
        name="refinement_pilot_bestver214"
    ).result()
    sampling_client = service_client.create_sampling_client(
        model_path=sampling_result.path
    )
    logger.info("Policy client ready (best_ver214 checkpoint)")

    # Grader client: base model, no LoRA
    grader_client = service_client.create_sampling_client(
        base_model=MODEL_NAME
    )
    logger.info("Grader client ready (base model)")

    sampling_params = tinker.types.SamplingParams(
        max_tokens=MAX_TOKENS,
        stop=renderer.get_stop_sequences(),
        temperature=TEMPERATURE,
    )

    # Load dataset
    logger.info("Loading ML dataset...")
    data = load_dataset("facebook/research-plan-gen", "ml")
    dataset = data["train"]
    n_goals = min(N_GOALS, len(dataset))
    logger.info(f"Running pilot on {n_goals} goals")

    # Main loop
    results = []
    for idx in range(n_goals):
        goal = dataset["Goal"][idx]
        rubric = dataset["Rubric"][idx]
        ref_sol = dataset["Reference solution"][idx]
        t0 = time.time()

        logger.info(f"[{idx + 1}/{n_goals}] {goal[:80]}...")

        # Phase 1: Generate initial plan
        prompt_text = build_research_plan_prompt(scenario=goal, examples=None)
        model_input = renderer.build_generation_prompt(
            [{"role": "user", "content": prompt_text}]
        )
        gen_result = sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        ).result()

        parsed_gen, _ = renderer.parse_response(
            gen_result.sequences[0].tokens
        )
        original_plan = renderers.get_text_content(parsed_gen)
        if "<solution>" in original_plan and "</solution>" not in original_plan:
            original_plan = original_plan.rstrip() + "\n</solution>"

        # Phase 2: Grade initial plan
        orig_grade = grade_plan(
            renderer, grader_client, goal, rubric, ref_sol, original_plan
        )

        # Phase 3: Extract feedback
        if orig_grade["is_valid_xml"]:
            feedback_text, bullet_count, low_level_count = (
                extract_weaknesses_from_grader(
                    orig_grade["grader_xml"],
                    max_bullets_per_item=3,
                    max_items=10,
                    focus_low_confidence_items_only=True,
                    include_sample_review=True,
                    include_item_reasoning_feedback=True,
                    include_item_desiderata_review=True,
                    include_global_desiderata_summary=True,
                )
            )
        else:
            feedback_text = (
                "Critical revision needed: strengthen implementation detail, "
                "rationale, and risk controls."
            )
            bullet_count, low_level_count = 1, 0

        # Phase 4: Generate refined plan
        refinement_prompt = build_self_teacher_prompt(
            scenario=goal,
            grader_feedback=feedback_text,
            successful_previous_rollout=None,
            policy_output_mode="solution_only",
            target_word_count=TARGET_WORD_COUNT,
            max_solution_words=MAX_WORD_COUNT,
        )
        refine_input = renderer.build_generation_prompt(
            [{"role": "user", "content": refinement_prompt}]
        )
        refine_result = sampling_client.sample(
            prompt=refine_input,
            num_samples=1,
            sampling_params=sampling_params,
        ).result()

        parsed_refine, _ = renderer.parse_response(
            refine_result.sequences[0].tokens
        )
        refined_plan = renderers.get_text_content(parsed_refine)
        if "<solution>" in refined_plan and "</solution>" not in refined_plan:
            refined_plan = refined_plan.rstrip() + "\n</solution>"

        # Phase 5: Grade refined plan
        refined_grade = grade_plan(
            renderer, grader_client, goal, rubric, ref_sol, refined_plan
        )

        # Phase 6: Record
        elapsed = time.time() - t0
        result = {
            "goal_idx": idx,
            "goal": goal,
            # Original
            "original_plan": original_plan,
            "original_grader_xml": orig_grade["grader_xml"],
            "original_rubric_score": orig_grade["rubric_score"],
            "original_compliant": orig_grade["is_compliant"],
            "original_word_count": orig_grade["word_count"],
            "original_per_item_levels": orig_grade["per_item_levels"],
            "original_valid_xml": orig_grade["is_valid_xml"],
            # Feedback
            "feedback_text": feedback_text,
            "feedback_bullet_count": bullet_count,
            "feedback_low_level_count": low_level_count,
            # Refined
            "refined_plan": refined_plan,
            "refined_grader_xml": refined_grade["grader_xml"],
            "refined_rubric_score": refined_grade["rubric_score"],
            "refined_compliant": refined_grade["is_compliant"],
            "refined_word_count": refined_grade["word_count"],
            "refined_per_item_levels": refined_grade["per_item_levels"],
            "refined_valid_xml": refined_grade["is_valid_xml"],
            # Delta
            "delta_rubric_score": (
                refined_grade["rubric_score"] - orig_grade["rubric_score"]
            ),
            "elapsed_seconds": elapsed,
        }
        results.append(result)
        logger.info(
            f"  orig={orig_grade['rubric_score']:.3f} -> "
            f"refined={refined_grade['rubric_score']:.3f} "
            f"(delta={result['delta_rubric_score']:+.3f}) "
            f"[{elapsed:.1f}s]"
        )

    # Save results
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    logger.info(f"Results saved to {OUTPUT_PATH}")

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
