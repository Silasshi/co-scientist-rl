"""
CPR (Contrastive Plan Ranking) evaluation script.

Generates N plans per test goal from the bestversion checkpoint, grades all plans
with the standard grader, then uses a trained CPR ranker to pairwise-rank plans
and select the top one.

Reports: mean-of-N, oracle best-of-N, CPR-selected, gap_closed%, rank correlation.

Usage:
  python eval_cpr.py
  python eval_cpr.py policy_checkpoint_run_path=/path/to/run cpr_checkpoint_run_path=/path/to/cpr
"""

import logging
import random
import time
import numpy as np
import json
import os
from itertools import combinations
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
import datasets
import tinker
from co_scientist.shared.api_profiles import create_service_client
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from datasets import load_dataset

from co_scientist.shared.eval_core import (
    build_research_plan_prompt,
    build_grader_prompt,
    compute_rubric_reward_from_xml,
    check_format_compliance,
    resolve_checkpoint,
)
from co_scientist.rubric_reward.grpo.train_cpr import build_cpr_prompt, extract_choice
from co_scientist.eval.eval_bon import spearman_rank_corr


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    # ---- Policy checkpoint (bestversion, for plan generation) ----
    policy_checkpoint_run_path: str = "runs/2026/2/withA1,A2/2(ml)"
    policy_checkpoint_batch: int = -1

    # ---- CPR checkpoint (for ranking) ----
    cpr_checkpoint_run_path: str = "runs/2026/4/cpr/1"
    cpr_checkpoint_batch: int = -1
    cpr_model_name: str = "Qwen/Qwen3-30B-A3B"
    cpr_lora_rank: int = 64

    # ---- Output ----
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
    group_size: int = 8        # N plans per goal
    max_tokens: int = 2048
    grader_max_tokens: int = 8192
    cpr_max_tokens: int = 32   # only need "A" or "B"
    temperature: float = 1.0
    grader_temperature: float = 0.0
    cpr_temperature: float = 0.0  # deterministic ranking

    # ---- Reward / format ----
    max_word_count: int = 750
    target_word_count: int = 600
    scale_length_bonus: float = 120.0
    scaling_factor: float = 0.08
    min_words: int = 30

    # ---- LoRA rank (for policy model) ----
    lora_rank: int = 64


# ============================================================
# Metrics
# ============================================================

def compute_goal_metrics(
    grader_scores: list[float],
    cpr_wins: list[int],
) -> dict:
    """Per-goal selection metrics using CPR win counts."""
    n = len(grader_scores)
    assert n == len(cpr_wins) and n > 0

    oracle_idx = int(np.argmax(grader_scores))
    cpr_idx = int(np.argmax(cpr_wins))

    # Break ties in cpr_wins by picking a random winner among tied plans
    max_wins = max(cpr_wins)
    tied_indices = [i for i, w in enumerate(cpr_wins) if w == max_wins]
    if len(tied_indices) > 1:
        cpr_idx = random.choice(tied_indices)

    return {
        "mean_of_n": float(np.mean(grader_scores)),
        "oracle_bon": float(np.max(grader_scores)),
        "oracle_idx": oracle_idx,
        "cpr_selected_rubric": grader_scores[cpr_idx],
        "cpr_selected_idx": cpr_idx,
        "cpr_picked_oracle": int(cpr_idx == oracle_idx),
        "rank_corr": spearman_rank_corr(grader_scores, [float(w) for w in cpr_wins]),
        "n_samples": n,
    }


# ============================================================
# Main
# ============================================================

def main(config: Config):
    # --- Resolve output path ---
    if config.eval_output_path:
        output_dir = config.eval_output_path
    elif config.cpr_checkpoint_run_path.lower() != "none":
        output_dir = os.path.join(config.cpr_checkpoint_run_path, "eval_cpr")
    else:
        output_dir = os.path.join(os.getcwd(), "eval_cpr")
    os.makedirs(output_dir, exist_ok=True)

    ml_logger = ml_log.setup_logging(
        log_dir=output_dir,
        wandb_project=None, wandb_name=None,
        config=config, do_configure_logging_module=True,
    )

    # --- Load dataset (test split) ---
    logger.info("Loading dataset (test split)...")
    if config.ml_data:
        data = load_dataset("facebook/research-plan-gen", "ml")
    elif config.arxiv_data:
        data = load_dataset("facebook/research-plan-gen", "arxiv")
    elif config.pubmed_data:
        data = load_dataset("facebook/research-plan-gen", "pubmed")
    else:
        raise ValueError("No dataset selected")
    assert isinstance(data, datasets.DatasetDict)
    dataset = data["test"]

    n_eval_batches = len(dataset) // config.batch_size
    logger.info(f"Test set: {len(dataset)} examples -> {n_eval_batches} batches of {config.batch_size}")

    # --- Tokenizer / renderer (for policy model) ---
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # --- Tokenizer / renderer (for CPR model) ---
    cpr_tokenizer = get_tokenizer(config.cpr_model_name)
    cpr_renderer_name = model_info.get_recommended_renderer_name(config.cpr_model_name)
    cpr_renderer = renderers.get_renderer(cpr_renderer_name, cpr_tokenizer)

    # --- Service client ---
    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )

    # --- Load policy checkpoint (bestversion) ---
    policy_resume, policy_batch_label = resolve_checkpoint(
        config.policy_checkpoint_run_path, config.policy_checkpoint_batch,
    )
    if policy_resume:
        policy_training_client = service_client.create_training_client_from_state_with_optimizer(
            policy_resume["state_path"]
        )
    else:
        policy_training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank,
        )

    logger.info(f"Saving policy weights (checkpoint=b{policy_batch_label})...")
    policy_sampling_result = policy_training_client.save_weights_for_sampler(
        name=f"eval_cpr_policy_b{policy_batch_label}"
    ).result()
    policy_sampling_client = service_client.create_sampling_client(
        model_path=policy_sampling_result.path,
    )

    # --- Load CPR checkpoint (ranker) ---
    cpr_resume, cpr_batch_label = resolve_checkpoint(
        config.cpr_checkpoint_run_path, config.cpr_checkpoint_batch,
    )
    if cpr_resume:
        cpr_training_client = service_client.create_training_client_from_state_with_optimizer(
            cpr_resume["state_path"]
        )
    else:
        cpr_training_client = service_client.create_lora_training_client(
            base_model=config.cpr_model_name, rank=config.cpr_lora_rank,
        )

    logger.info(f"Saving CPR weights (checkpoint=b{cpr_batch_label})...")
    cpr_sampling_result = cpr_training_client.save_weights_for_sampler(
        name=f"eval_cpr_ranker_b{cpr_batch_label}"
    ).result()
    cpr_sampling_client = service_client.create_sampling_client(
        model_path=cpr_sampling_result.path,
    )

    # --- Grader client (frozen base model) ---
    grader_client = service_client.create_sampling_client(
        base_model=config.grader_model_name,
    )

    # --- Sampling params ---
    policy_sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )
    grader_sampling_params = tinker.types.SamplingParams(
        max_tokens=config.grader_max_tokens,
        temperature=config.grader_temperature,
    )
    cpr_sampling_params = tinker.types.SamplingParams(
        max_tokens=config.cpr_max_tokens,
        stop=cpr_renderer.get_stop_sequences(),
        temperature=config.cpr_temperature,
    )

    # --- Output files ---
    logs_path = os.path.join(output_dir, f"eval_cpr_b{cpr_batch_label}_logs.jsonl")
    summary_path = os.path.join(output_dir, f"eval_cpr_b{cpr_batch_label}_summary.jsonl")

    logger.info(
        f"Starting CPR eval | policy=b{policy_batch_label} | cpr=b{cpr_batch_label} | "
        f"N={config.group_size} | {n_eval_batches} batches | output -> {output_dir}"
    )

    # --- Accumulators ---
    all_goal_metrics: list[dict] = []

    # ============================================================
    # Eval loop
    # ============================================================
    for batch_idx in range(n_eval_batches):
        t_start = time.time()
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(dataset))
        batch_rows = dataset.select(range(batch_start, batch_end))

        logger.info(
            f"Batch {batch_idx + 1}/{n_eval_batches} "
            f"(goals {batch_start}-{batch_end - 1})"
        )

        # === Phase 1: Generate N plans per goal ===
        policy_futures = []
        for goal in batch_rows["Goal"]:
            prompt_text = build_research_plan_prompt(scenario=goal)
            convo = [{"role": "user", "content": prompt_text}]
            model_input = renderer.build_generation_prompt(convo)
            policy_futures.append(
                policy_sampling_client.sample(
                    prompt=model_input,
                    num_samples=config.group_size,
                    sampling_params=policy_sampling_params,
                )
            )

        # === Phase 2: Collect plans, launch graders ===
        all_grader_futures: list[list] = []
        all_plan_texts: list[list[str]] = []

        for i, p_future in enumerate(policy_futures):
            result = p_future.result()
            goal = batch_rows["Goal"][i]
            rubric = batch_rows["Rubric"][i]
            ref_sol = batch_rows["Reference solution"][i]

            goal_grader_futures = []
            goal_plans = []

            for seq in result.sequences:
                plan_text = renderers.get_text_content(
                    renderer.parse_response(seq.tokens)[0]
                )
                if "<solution>" in plan_text and "</solution>" not in plan_text:
                    plan_text = plan_text.rstrip() + "\n</solution>"
                goal_plans.append(plan_text)

                # Launch grader
                grader_prompt = build_grader_prompt(
                    scenario=goal, rubric_items=rubric,
                    proposed_plan=plan_text, reference_solution=ref_sol,
                )
                grader_input = renderer.build_generation_prompt(
                    [{"role": "user", "content": grader_prompt}]
                )
                goal_grader_futures.append(
                    grader_client.sample(
                        grader_input, num_samples=1,
                        sampling_params=grader_sampling_params,
                    )
                )

            all_grader_futures.append(goal_grader_futures)
            all_plan_texts.append(goal_plans)

        # === Phase 3: Collect grader results ===
        # all_grader_scores[goal_idx][sample_idx] = rubric score
        all_grader_scores: list[list[float]] = []

        for goal_idx in range(len(batch_rows)):
            grader_futs = all_grader_futures[goal_idx]
            goal_scores = []
            for j, g_future in enumerate(grader_futs):
                g_result = g_future.result()
                g_parsed, _ = renderer.parse_response(g_result.sequences[0].tokens)
                g_xml = renderers.get_text_content(g_parsed)
                rubric_score = compute_rubric_reward_from_xml(g_xml)
                goal_scores.append(rubric_score)
            all_grader_scores.append(goal_scores)

        # === Phase 4: Pairwise CPR ranking ===
        logger.info(f"  Phase 4: launching pairwise CPR comparisons...")

        # For each goal, launch all pairwise comparison futures
        all_cpr_futures: list[list] = []  # [goal_idx] -> list of (i, j, future)

        for goal_idx in range(len(batch_rows)):
            goal = batch_rows["Goal"][goal_idx]
            plans = all_plan_texts[goal_idx]
            n_plans = len(plans)
            goal_cpr_futures = []

            for i, j in combinations(range(n_plans), 2):
                # Randomly assign to A/B to avoid position bias
                if random.random() < 0.5:
                    prompt = build_cpr_prompt(goal, plans[i], plans[j])
                    order = (i, j)  # A=i, B=j
                else:
                    prompt = build_cpr_prompt(goal, plans[j], plans[i])
                    order = (j, i)  # A=j, B=i

                cpr_input = cpr_renderer.build_generation_prompt(
                    [{"role": "user", "content": prompt}]
                )
                future = cpr_sampling_client.sample(
                    cpr_input, num_samples=1,
                    sampling_params=cpr_sampling_params,
                )
                goal_cpr_futures.append((order, future))

            all_cpr_futures.append(goal_cpr_futures)

        # === Phase 5: Collect CPR results, compute metrics ===
        batch_logs: list[dict] = []
        batch_goal_metrics: list[dict] = []

        for goal_idx in range(len(batch_rows)):
            goal = batch_rows["Goal"][goal_idx]
            plans = all_plan_texts[goal_idx]
            grader_scores = all_grader_scores[goal_idx]
            cpr_futures = all_cpr_futures[goal_idx]
            n_plans = len(plans)

            # Filter degenerate samples
            valid_mask = []
            for j in range(n_plans):
                word_count = len(plans[j].strip().split())
                is_degenerate = (
                    word_count < config.min_words
                    or plans[j].strip() in {"<think>", "</think>", "<solution>", "</solution>"}
                )
                valid_mask.append(not is_degenerate)

            valid_indices = [j for j in range(n_plans) if valid_mask[j]]
            if len(valid_indices) < 2:
                logger.warning(f"  Goal {goal_idx}: fewer than 2 valid samples, skipping")
                continue

            valid_scores = [grader_scores[j] for j in valid_indices]

            # Count wins from CPR comparisons
            wins = {j: 0 for j in valid_indices}
            n_comparisons = 0
            n_parsed = 0

            for (order_a, order_b), future in cpr_futures:
                # Skip comparisons involving degenerate samples
                if not valid_mask[order_a] or not valid_mask[order_b]:
                    continue

                n_comparisons += 1
                cpr_result = future.result()
                cpr_parsed, _ = cpr_renderer.parse_response(cpr_result.sequences[0].tokens)
                cpr_text = renderers.get_text_content(cpr_parsed)
                choice = extract_choice(cpr_text)

                if choice is not None:
                    n_parsed += 1
                    if choice == "A":
                        wins[order_a] += 1
                    else:
                        wins[order_b] += 1

            valid_wins = [wins[j] for j in valid_indices]

            # Compute goal-level metrics
            goal_met = compute_goal_metrics(valid_scores, valid_wins)
            batch_goal_metrics.append(goal_met)
            all_goal_metrics.append(goal_met)

            # Log sample details
            for j in valid_indices:
                word_count = len(plans[j].strip().split())
                is_compliant = check_format_compliance(plans[j], config.max_word_count)
                excess = max(0, word_count - config.max_word_count)
                format_penalty = 0.0 if is_compliant else 0.2 + 0.0005 * excess
                length_bonus = np.exp(
                    -((word_count - config.target_word_count) / config.scale_length_bonus) ** 2
                )
                final_reward = grader_scores[j] + config.scaling_factor * length_bonus - format_penalty

                batch_logs.append({
                    "eval_batch_idx": batch_idx,
                    "policy_batch": policy_batch_label,
                    "cpr_batch": cpr_batch_label,
                    "goal_idx": goal_idx,
                    "sample_idx": j,
                    "goal": goal,
                    "policy_output": plans[j],
                    "grader_rubric_score": grader_scores[j],
                    "cpr_wins": wins.get(j, 0),
                    "format_penalty": format_penalty,
                    "final_reward": final_reward,
                    "word_count": word_count,
                    "is_compliant": is_compliant,
                    "is_oracle_pick": j == goal_met["oracle_idx"],
                    "is_cpr_pick": j == goal_met["cpr_selected_idx"],
                })

        # --- Batch summary ---
        if batch_goal_metrics:
            mean_of_n = float(np.mean([m["mean_of_n"] for m in batch_goal_metrics]))
            oracle_bon = float(np.mean([m["oracle_bon"] for m in batch_goal_metrics]))
            cpr_selected = float(np.mean([m["cpr_selected_rubric"] for m in batch_goal_metrics]))
            oracle_gap = oracle_bon - mean_of_n
            cpr_gap = cpr_selected - mean_of_n

            batch_summary = {
                "eval_batch_idx": batch_idx,
                "policy_batch": policy_batch_label,
                "cpr_batch": cpr_batch_label,
                "n_goals": len(batch_goal_metrics),

                "rubric/mean_of_n": mean_of_n,
                "rubric/oracle_bon": oracle_bon,
                "rubric/cpr_selected": cpr_selected,

                "selection/oracle_gap": oracle_gap,
                "selection/cpr_gap": cpr_gap,
                "selection/gap_closed_pct": 100 * cpr_gap / oracle_gap if oracle_gap > 0 else 0.0,
                "selection/cpr_picked_oracle": float(np.mean([m["cpr_picked_oracle"] for m in batch_goal_metrics])),
                "selection/rank_corr": float(np.mean([m["rank_corr"] for m in batch_goal_metrics])),

                "time_s": round(time.time() - t_start, 1),
            }
        else:
            batch_summary = {"eval_batch_idx": batch_idx, "n_goals": 0}

        logger.info(
            f"  mean={batch_summary.get('rubric/mean_of_n', 0):.4f}  "
            f"oracle={batch_summary.get('rubric/oracle_bon', 0):.4f}  "
            f"cpr={batch_summary.get('rubric/cpr_selected', 0):.4f}  "
            f"gap_closed={batch_summary.get('selection/gap_closed_pct', 0):.1f}%  "
            f"rank_corr={batch_summary.get('selection/rank_corr', 0):.3f}"
        )

        # Write logs
        with open(logs_path, "a") as f:
            for item in batch_logs:
                f.write(json.dumps(item) + "\n")
        with open(summary_path, "a") as f:
            f.write(json.dumps(batch_summary) + "\n")

    # --- Final aggregate ---
    if all_goal_metrics:
        mean_of_n = float(np.mean([m["mean_of_n"] for m in all_goal_metrics]))
        oracle_bon = float(np.mean([m["oracle_bon"] for m in all_goal_metrics]))
        cpr_selected = float(np.mean([m["cpr_selected_rubric"] for m in all_goal_metrics]))
        oracle_gap = oracle_bon - mean_of_n
        cpr_gap = cpr_selected - mean_of_n

        aggregate = {
            "policy_batch": policy_batch_label,
            "cpr_batch": cpr_batch_label,
            "n_goals": len(all_goal_metrics),
            "group_size": config.group_size,

            "rubric/mean_of_n": mean_of_n,
            "rubric/mean_of_n_std": float(np.std([m["mean_of_n"] for m in all_goal_metrics])),
            "rubric/oracle_bon": oracle_bon,
            "rubric/oracle_bon_std": float(np.std([m["oracle_bon"] for m in all_goal_metrics])),
            "rubric/cpr_selected": cpr_selected,
            "rubric/cpr_selected_std": float(np.std([m["cpr_selected_rubric"] for m in all_goal_metrics])),

            "selection/oracle_gap": oracle_gap,
            "selection/cpr_gap": cpr_gap,
            "selection/gap_closed_pct": 100 * cpr_gap / oracle_gap if oracle_gap > 0 else 0.0,
            "selection/cpr_picked_oracle": float(np.mean([m["cpr_picked_oracle"] for m in all_goal_metrics])),
            "selection/rank_corr_mean": float(np.mean([m["rank_corr"] for m in all_goal_metrics])),
            "selection/rank_corr_std": float(np.std([m["rank_corr"] for m in all_goal_metrics])),
        }

        agg_path = os.path.join(output_dir, f"eval_cpr_b{cpr_batch_label}_aggregate.json")
        with open(agg_path, "w") as f:
            json.dump(aggregate, f, indent=2)

        logger.info("=" * 60)
        logger.info(f"CPR EVAL COMPLETE  policy=b{policy_batch_label}  cpr=b{cpr_batch_label}  N={config.group_size}")
        logger.info(f"  mean-of-N :  {mean_of_n:.4f}")
        logger.info(f"  oracle BoN:  {oracle_bon:.4f}  (+{oracle_gap:.4f})")
        logger.info(f"  cpr-select:  {cpr_selected:.4f}  (+{cpr_gap:.4f})")
        logger.info(f"  gap closed:  {aggregate['selection/gap_closed_pct']:.1f}%")
        logger.info(f"  rank corr :  {aggregate['selection/rank_corr_mean']:.3f}")
        logger.info(f"  results   :  {output_dir}")
        logger.info("=" * 60)

    ml_logger.close()


if __name__ == "__main__":
    chz.nested_entrypoint(main)
