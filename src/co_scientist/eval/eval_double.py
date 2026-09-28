"""
Two-stage evaluation script for doublegeneration checkpoints.

Pipeline per sample:
  Stage 1  — generate initial plan  (policy checkpoint)
  Feedback — grade with DETAILED grader → extract structured critique
  Stage 2  — generate refined plan  (same policy checkpoint, conditioned on critique)
  Scoring  — grade BOTH initial and refined plans with the SIMPLE grader
              (same prompt as eval_only.py / best_ver.py → scores are directly
               comparable to the bestversion 0.69 benchmark)

Output per run:
  eval_b{N}_double_logs.jsonl      per-sample details
  eval_b{N}_double_summary.jsonl   per-eval-batch aggregates
  eval_b{N}_double_aggregate.json  final numbers

Usage:
  # Evaluate checkpoint at batch 133
  python eval_only_double.py checkpoint_run_path=runs/2026/3/refinement/8 checkpoint_batch=133

  # Last checkpoint
  python eval_only_double.py checkpoint_run_path=runs/2026/3/refinement/8

  # Base model (no fine-tuning)
  python eval_only_double.py checkpoint_run_path=none
"""

import logging
import time
import re
import textwrap
import numpy as np
import json
import os
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
import datasets
import tinker
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from datasets import load_dataset

# Refinement pipeline (detailed grader prompt, feedback extraction, Stage 2 prompt)
from co_scientist.rubric_reward.refinement.train_double_generation import (
    build_research_plan_prompt,
    build_detailed_grader_prompt,
    build_self_teacher_prompt,
    extract_weaknesses_from_grader,
    extract_solution_text,
    sdpo_compute_rubric_reward_from_xml,
    sdpo_check_format_compliance,
    create_service_client,
)

# Simple grader prompt — same as best_ver.py / eval_only.py for comparable scores
from co_scientist.shared.eval_core import (
    build_grader_prompt as build_simple_grader_prompt,
    compute_rubric_reward_from_xml as compute_rubric_simple,
    check_format_compliance as check_format_simple,
    resolve_checkpoint,
)


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    # ---- Checkpoint selection ----
    checkpoint_run_path: str = "/home/silas/co-scientist-project/runs/2026/3/refinement/8"
    # -1 = last checkpoint, 0 = base model, N = exact batch number
    checkpoint_batch: int = -1

    # ---- Output ----
    # Defaults to <checkpoint_run_path>/eval/ when empty
    eval_output_path: str = ""

    # ---- Model / API ----
    base_url: str | None = None
    api_profile: str | None = None
    model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"

    # ---- Dataset ----
    ml_data: bool = True
    arxiv_data: bool = False
    pubmed_data: bool = False

    # ---- Sampling ----
    batch_size: int = 64
    group_size: int = 1        # Samples per goal (1 = single plan per goal for eval)
    max_tokens: int = 2048
    refinement_max_tokens: int = 4096
    grader_max_tokens: int = 12288
    temperature: float = 1.0
    grader_temperature: float = 0.0

    # ---- Reward / format (must match best_ver.py for comparable scores) ----
    max_word_count: int = 750
    target_word_count: int = 600
    scale_length_bonus: float = 120.0
    scaling_factor: float = 0.08
    min_words: int = 30

    # ---- Feedback extraction settings (mirrors train_double_generation.py) ----
    feedback_max_bullets_per_item: int = 3
    feedback_max_items: int = 10

    # ---- LoRA rank (needed to reconstruct the training client) ----
    lora_rank: int = 64


# ============================================================
# Reward helper (using best_ver scale for comparability)
# ============================================================

def _compute_reward(rubric_score: float, plan_text: str, config: Config) -> tuple[float, float, bool]:
    """Returns (final_reward, format_penalty, is_compliant)."""
    word_count = len(plan_text.strip().split())
    is_compliant = check_format_simple(plan_text, config.max_word_count)
    excess = max(0, word_count - config.max_word_count)
    format_penalty = 0.0 if is_compliant else 0.2 + 0.0005 * excess
    length_bonus = np.exp(
        -((word_count - config.target_word_count) / config.scale_length_bonus) ** 2
    )
    final_reward = rubric_score + config.scaling_factor * length_bonus - format_penalty
    return final_reward, format_penalty, is_compliant


# ============================================================
# Main
# ============================================================

def main(config: Config):
    # --- Output path ---
    if config.eval_output_path:
        output_dir = config.eval_output_path
    elif config.checkpoint_run_path.lower() != "none":
        output_dir = os.path.join(config.checkpoint_run_path, "eval")
    else:
        output_dir = os.path.join(os.getcwd(), "eval_base_model_double")
    os.makedirs(output_dir, exist_ok=True)

    ml_logger = ml_log.setup_logging(
        log_dir=output_dir,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    # --- Dataset (test split) ---
    logger.info("Loading dataset (test split)...")
    if config.ml_data:
        data = load_dataset("facebook/research-plan-gen", "ml")
    elif config.arxiv_data:
        data = load_dataset("facebook/research-plan-gen", "arxiv")
    elif config.pubmed_data:
        data = load_dataset("facebook/research-plan-gen", "pubmed")
    assert isinstance(data, datasets.DatasetDict)
    dataset = data["test"]

    n_eval_batches = len(dataset) // config.batch_size
    logger.info(f"Test set: {len(dataset)} examples → {n_eval_batches} batches of {config.batch_size}")

    # --- Tokenizer / renderer ---
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # --- Service clients ---
    from co_scientist.shared.api_profiles import create_service_client as _create_sc
    service_client = _create_sc(
        base_url=config.base_url,
        api_profile=getattr(config, "api_profile", None),
    )

    # --- Checkpoint ---
    resume_info, batch_label = resolve_checkpoint(
        config.checkpoint_run_path, config.checkpoint_batch
    )

    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
        loaded_batch = resume_info["batch"]
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        loaded_batch = -1

    # Freeze weights once for the whole eval
    logger.info(f"Saving weights for eval sampler (checkpoint=b{batch_label})...")
    sampling_result = training_client.save_weights_for_sampler(
        name=f"eval_double_b{batch_label}"
    ).result()
    policy_client = service_client.create_sampling_client(
        model_path=sampling_result.path
    )

    # Grader client (base model, no LoRA)
    grader_client = service_client.create_sampling_client(
        base_model=config.grader_model_name
    )

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )
    refinement_params = tinker.types.SamplingParams(
        max_tokens=config.refinement_max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )
    grader_params = tinker.types.SamplingParams(
        max_tokens=config.grader_max_tokens,
        temperature=config.grader_temperature,
    )

    # --- Output files ---
    logs_path    = os.path.join(output_dir, f"eval_b{batch_label}_double_logs.jsonl")
    summary_path = os.path.join(output_dir, f"eval_b{batch_label}_double_summary.jsonl")

    logger.info(
        f"Starting two-stage eval | checkpoint=b{batch_label} | "
        f"{n_eval_batches} batches | output → {output_dir}"
    )
    logger.info("Grader for SCORING : simple (best_ver compatible)")
    logger.info("Grader for FEEDBACK: detailed (train_double_generation)")

    # Accumulators across all batches
    all_initial_rubric: list[float] = []
    all_refined_rubric: list[float] = []
    all_initial_reward: list[float] = []
    all_refined_reward: list[float] = []
    all_deltas: list[float] = []

    # ============================================================
    # Eval loop
    # ============================================================
    for batch_idx in range(n_eval_batches):
        t_start = time.time()
        batch_start = batch_idx * config.batch_size
        batch_end   = min((batch_idx + 1) * config.batch_size, len(dataset))
        batch_rows  = dataset.select(range(batch_start, batch_end))

        logger.info(
            f"Eval batch {batch_idx + 1}/{n_eval_batches} "
            f"(goals {batch_start}–{batch_end - 1})"
        )

        # ── Stage 1: launch initial generation ───────────────────────
        s1_futures = []
        for goal in batch_rows["Goal"]:
            prompt_text = build_research_plan_prompt(scenario=goal)
            model_input = renderer.build_generation_prompt(
                [{"role": "user", "content": prompt_text}]
            )
            s1_futures.append(
                policy_client.sample(
                    prompt=model_input,
                    num_samples=config.group_size,
                    sampling_params=sampling_params,
                )
            )

        # ── Collect Stage 1 plans; launch DETAILED graders (for feedback) ─
        detailed_grader_futures = []   # (goal_idx, sample_idx) → future
        s1_plans: list[list[str]] = [] # [goal_idx][sample_idx]

        for i, fut in enumerate(s1_futures):
            result = fut.result()
            goal    = batch_rows["Goal"][i]
            rubric  = batch_rows["Rubric"][i]
            ref_sol = batch_rows["Reference solution"][i]

            goal_plans: list[str] = []
            goal_detailed_futures = []

            for seq in result.sequences:
                plan_text = renderers.get_text_content(
                    renderer.parse_response(seq.tokens)[0]
                )
                if "<solution>" in plan_text and "</solution>" not in plan_text:
                    plan_text = plan_text.rstrip() + "\n</solution>"
                goal_plans.append(plan_text)

                # Detailed grader — used only for feedback extraction
                detailed_prompt = build_detailed_grader_prompt(
                    scenario=goal,
                    rubric_items=rubric,
                    proposed_plan=plan_text,
                    reference_solution=ref_sol,
                )
                detailed_input = renderer.build_generation_prompt(
                    [{"role": "user", "content": detailed_prompt}]
                )
                goal_detailed_futures.append(
                    grader_client.sample(detailed_input, num_samples=1, sampling_params=grader_params)
                )

            s1_plans.append(goal_plans)
            detailed_grader_futures.append(goal_detailed_futures)

        # ── Launch SIMPLE graders for Stage 1 scoring (while detailed graders run) ──
        simple_s1_futures = []
        for i in range(len(batch_rows)):
            goal    = batch_rows["Goal"][i]
            rubric  = batch_rows["Rubric"][i]
            ref_sol = batch_rows["Reference solution"][i]
            goal_simple_futures = []
            for plan_text in s1_plans[i]:
                simple_prompt = build_simple_grader_prompt(
                    scenario=goal,
                    rubric_items=rubric,
                    proposed_plan=plan_text,
                    reference_solution=ref_sol,
                )
                simple_input = renderer.build_generation_prompt(
                    [{"role": "user", "content": simple_prompt}]
                )
                goal_simple_futures.append(
                    grader_client.sample(simple_input, num_samples=1, sampling_params=grader_params)
                )
            simple_s1_futures.append(goal_simple_futures)

        # ── Collect detailed grader results; extract feedback; launch Stage 2 ──
        s2_futures = []
        feedbacks: list[list[str]] = []  # [goal_idx][sample_idx]

        for i in range(len(batch_rows)):
            goal       = batch_rows["Goal"][i]
            rubric     = batch_rows["Rubric"][i]
            goal_s2_futures = []
            goal_feedbacks  = []

            for j, det_fut in enumerate(detailed_grader_futures[i]):
                det_result  = det_fut.result()
                det_msg, _  = renderer.parse_response(det_result.sequences[0].tokens)
                det_xml     = renderers.get_text_content(det_msg)

                feedback_text, _, _ = extract_weaknesses_from_grader(
                    det_xml,
                    max_bullets_per_item=config.feedback_max_bullets_per_item,
                    max_items=config.feedback_max_items,
                    focus_low_confidence_items_only=True,
                    include_sample_review=True,
                    include_item_reasoning_feedback=True,
                    include_item_desiderata_review=False,  # matches training
                )
                goal_feedbacks.append(feedback_text)

                # Extract solution text only (strip <think> blocks) — matches training exactly
                # (see train_double_generation.py line 1562)
                s1_draft = extract_solution_text(s1_plans[i][j]) or s1_plans[i][j]

                # Stage 2: refine using feedback
                # policy_output_mode="solution_only" — Stage 2 was trained with this mode
                # (see train_double_generation.py line 1570)
                refinement_prompt = build_self_teacher_prompt(
                    scenario=goal,
                    grader_feedback=feedback_text,
                    initial_draft=s1_draft,
                    rubric_items=rubric,
                    policy_output_mode="solution_only",
                    target_word_count=config.target_word_count,
                    max_solution_words=config.max_word_count,
                )
                refinement_input = renderer.build_generation_prompt(
                    [{"role": "user", "content": refinement_prompt}]
                )
                goal_s2_futures.append(
                    policy_client.sample(
                        prompt=refinement_input,
                        num_samples=1,
                        sampling_params=refinement_params,
                    )
                )

            s2_futures.append(goal_s2_futures)
            feedbacks.append(goal_feedbacks)

        # ── Collect Stage 2 plans; launch SIMPLE graders for Stage 2 scoring ──
        s2_plans: list[list[str]] = []
        simple_s2_futures = []

        for i in range(len(batch_rows)):
            goal    = batch_rows["Goal"][i]
            rubric  = batch_rows["Rubric"][i]
            ref_sol = batch_rows["Reference solution"][i]
            goal_s2_plans = []
            goal_simple_s2_futures = []

            for j, s2_fut in enumerate(s2_futures[i]):
                s2_result  = s2_fut.result()
                s2_plan    = renderers.get_text_content(
                    renderer.parse_response(s2_result.sequences[0].tokens)[0]
                )
                if "<solution>" in s2_plan and "</solution>" not in s2_plan:
                    s2_plan = s2_plan.rstrip() + "\n</solution>"
                goal_s2_plans.append(s2_plan)

                simple_prompt = build_simple_grader_prompt(
                    scenario=goal,
                    rubric_items=rubric,
                    proposed_plan=s2_plan,
                    reference_solution=ref_sol,
                )
                simple_input = renderer.build_generation_prompt(
                    [{"role": "user", "content": simple_prompt}]
                )
                goal_simple_s2_futures.append(
                    grader_client.sample(simple_input, num_samples=1, sampling_params=grader_params)
                )

            s2_plans.append(goal_s2_plans)
            simple_s2_futures.append(goal_simple_s2_futures)

        # ── Collect all simple grader results and compute metrics ──
        batch_initial_rubric: list[float] = []
        batch_refined_rubric: list[float] = []
        batch_initial_reward: list[float] = []
        batch_refined_reward: list[float] = []
        batch_deltas:         list[float] = []
        batch_logs: list[dict] = []
        dropped = 0
        num_total = 0

        for i in range(len(batch_rows)):
            goal = batch_rows["Goal"][i]

            for j in range(len(s1_plans[i])):
                num_total += 1
                s1_plan = s1_plans[i][j]
                s2_plan = s2_plans[i][j]

                # Skip degenerate Stage 1 outputs
                s1_words = len(s1_plan.strip().split())
                if s1_words < config.min_words:
                    dropped += 1
                    continue
                if s1_plan.strip() in {"<think>", "</think>", "<solution>", "</solution>"}:
                    dropped += 1
                    continue

                # --- Score Stage 1 with simple grader ---
                s1_grader_result = simple_s1_futures[i][j].result()
                s1_msg, _ = renderer.parse_response(s1_grader_result.sequences[0].tokens)
                s1_xml    = renderers.get_text_content(s1_msg)
                s1_rubric = compute_rubric_simple(s1_xml)
                s1_reward, s1_format_pen, s1_compliant = _compute_reward(s1_rubric, s1_plan, config)

                # --- Score Stage 2 with simple grader ---
                s2_grader_result = simple_s2_futures[i][j].result()
                s2_msg, _ = renderer.parse_response(s2_grader_result.sequences[0].tokens)
                s2_xml    = renderers.get_text_content(s2_msg)
                s2_rubric = compute_rubric_simple(s2_xml)
                s2_reward, s2_format_pen, s2_compliant = _compute_reward(s2_rubric, s2_plan, config)

                delta = s2_rubric - s1_rubric

                batch_initial_rubric.append(s1_rubric)
                batch_refined_rubric.append(s2_rubric)
                batch_initial_reward.append(s1_reward)
                batch_refined_reward.append(s2_reward)
                batch_deltas.append(delta)

                batch_logs.append({
                    "eval_batch_idx":    batch_idx,
                    "checkpoint_batch":  batch_label,
                    "goal_idx":          i,
                    "sample_idx":        j,
                    "goal":              goal,

                    # Stage 1
                    "initial_plan":        s1_plan,
                    "initial_rubric":      s1_rubric,
                    "initial_reward":      s1_reward,
                    "initial_format_pen":  s1_format_pen,
                    "initial_compliant":   s1_compliant,
                    "initial_word_count":  s1_words,

                    # Feedback
                    "feedback":            feedbacks[i][j],

                    # Stage 2
                    "refined_plan":        s2_plan,
                    "refined_rubric":      s2_rubric,
                    "refined_reward":      s2_reward,
                    "refined_format_pen":  s2_format_pen,
                    "refined_compliant":   s2_compliant,
                    "refined_word_count":  len(s2_plan.strip().split()),

                    # Delta
                    "rubric_delta":        delta,
                    "improved":            delta > 0,
                })

        # Accumulate
        all_initial_rubric.extend(batch_initial_rubric)
        all_refined_rubric.extend(batch_refined_rubric)
        all_initial_reward.extend(batch_initial_reward)
        all_refined_reward.extend(batch_refined_reward)
        all_deltas.extend(batch_deltas)

        n_valid = len(batch_initial_rubric)
        n_improved = sum(1 for d in batch_deltas if d > 0)

        batch_summary = {
            "eval_batch_idx":         batch_idx,
            "checkpoint_batch":       batch_label,

            "initial/rubric_mean":    float(np.mean(batch_initial_rubric)) if batch_initial_rubric else 0.0,
            "initial/rubric_std":     float(np.std(batch_initial_rubric))  if batch_initial_rubric else 0.0,
            "initial/reward_mean":    float(np.mean(batch_initial_reward)) if batch_initial_reward else 0.0,

            "refined/rubric_mean":    float(np.mean(batch_refined_rubric)) if batch_refined_rubric else 0.0,
            "refined/rubric_std":     float(np.std(batch_refined_rubric))  if batch_refined_rubric else 0.0,
            "refined/reward_mean":    float(np.mean(batch_refined_reward)) if batch_refined_reward else 0.0,

            "delta/rubric_mean":      float(np.mean(batch_deltas))         if batch_deltas else 0.0,
            "improvement_rate":       n_improved / n_valid                 if n_valid else 0.0,

            "samples/total":          num_total,
            "samples/valid":          n_valid,
            "samples/dropped":        dropped,
            "time_s":                 round(time.time() - t_start, 1),
        }

        logger.info(
            f"  initial  rubric={batch_summary['initial/rubric_mean']:.4f}  "
            f"reward={batch_summary['initial/reward_mean']:.4f}"
        )
        logger.info(
            f"  refined  rubric={batch_summary['refined/rubric_mean']:.4f}  "
            f"reward={batch_summary['refined/reward_mean']:.4f}  "
            f"delta={batch_summary['delta/rubric_mean']:+.4f}  "
            f"improved={n_improved}/{n_valid}"
        )

        with open(logs_path, "a") as f:
            for item in batch_logs:
                f.write(json.dumps(item) + "\n")
        with open(summary_path, "a") as f:
            f.write(json.dumps(batch_summary) + "\n")

    # --- Final aggregate ---
    n_valid_total  = len(all_initial_rubric)
    n_improved_all = sum(1 for d in all_deltas if d > 0)

    aggregate = {
        "checkpoint_batch":       batch_label,
        "loaded_batch":           loaded_batch,
        "n_eval_batches":         n_eval_batches,
        "n_samples_valid":        n_valid_total,

        "initial/rubric_mean":    float(np.mean(all_initial_rubric)) if all_initial_rubric else 0.0,
        "initial/rubric_std":     float(np.std(all_initial_rubric))  if all_initial_rubric else 0.0,
        "initial/reward_mean":    float(np.mean(all_initial_reward)) if all_initial_reward else 0.0,

        "refined/rubric_mean":    float(np.mean(all_refined_rubric)) if all_refined_rubric else 0.0,
        "refined/rubric_std":     float(np.std(all_refined_rubric))  if all_refined_rubric else 0.0,
        "refined/reward_mean":    float(np.mean(all_refined_reward)) if all_refined_reward else 0.0,

        "delta/rubric_mean":      float(np.mean(all_deltas))         if all_deltas else 0.0,
        "delta/rubric_std":       float(np.std(all_deltas))          if all_deltas else 0.0,
        "improvement_rate":       n_improved_all / n_valid_total      if n_valid_total else 0.0,

        "grader_for_scoring":     "simple (best_ver compatible)",
        "grader_for_feedback":    "detailed (train_double_generation)",
    }

    aggregate_path = os.path.join(output_dir, f"eval_b{batch_label}_double_aggregate.json")
    with open(aggregate_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"TWO-STAGE EVAL COMPLETE  checkpoint=b{batch_label}")
    logger.info(f"  initial rubric : {aggregate['initial/rubric_mean']:.4f}  (vs bestversion single-pass)")
    logger.info(f"  refined rubric : {aggregate['refined/rubric_mean']:.4f}  (two-stage ceiling)")
    logger.info(f"  delta          : {aggregate['delta/rubric_mean']:+.4f}")
    logger.info(f"  improved       : {n_improved_all}/{n_valid_total}  ({aggregate['improvement_rate']:.1%})")
    logger.info(f"  results        : {output_dir}")
    logger.info("=" * 60)

    ml_logger.close()


if __name__ == "__main__":
    chz.nested_entrypoint(main)
