"""
Self-Selector evaluation script.

Generates N plans per test goal from a policy checkpoint, grades all plans
with the standard grader, then uses a trained selector to pairwise-rank
plans and pick the top one.

Reports: mean-of-N, oracle best-of-N, selector-selected, gap_closed%,
rank correlation, position bias, oracle hit rate.

Supports position-debiased evaluation: each pair is compared in BOTH
orderings (A=i,B=j AND A=j,B=i). If the two comparisons disagree,
the pair is treated as a tie (0.5 wins each).

Usage:
  python eval_selector.py
  python eval_selector.py selector_checkpoint_run_path=runs/2026/4/selector/1
  python eval_selector.py position_debias=false  # faster, single comparison
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
from co_scientist.rubric_reward.selector.train_selector import (
    build_selector_prompt,
    extract_verdict,
)
from co_scientist.eval.eval_bon import spearman_rank_corr


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


# ============================================================
# Config
# ============================================================

@chz.chz
class Config:
    # ---- Policy checkpoint (bestversion, for plan generation) ----
    policy_checkpoint_run_path: str = "runs/2026/2/withA1,A2/2(ml)"
    policy_checkpoint_batch: int = -1

    # ---- Selector checkpoint (for ranking) ----
    selector_checkpoint_run_path: str = "runs/2026/4/selector/1"
    selector_checkpoint_batch: int = -1
    selector_model_name: str = "Qwen/Qwen3-30B-A3B"
    selector_lora_rank: int = 64

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
    group_size: int = 8          # N plans per goal
    max_tokens: int = 2048       # policy generation
    grader_max_tokens: int = 8192
    selector_max_tokens: int = 512  # selector reasoning
    temperature: float = 1.0     # policy
    grader_temperature: float = 0.0
    selector_temperature: float = 0.0  # deterministic ranking

    # ---- Position debiasing ----
    position_debias: bool = True  # compare in both orderings

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
    selector_wins: list[float],
) -> dict:
    """Per-goal selection metrics using selector win counts."""
    n = len(grader_scores)
    assert n == len(selector_wins) and n > 0

    oracle_idx = int(np.argmax(grader_scores))

    # Select plan with most wins; break ties randomly
    max_wins = max(selector_wins)
    tied_indices = [i for i, w in enumerate(selector_wins) if w == max_wins]
    selector_idx = random.choice(tied_indices)

    return {
        "mean_of_n": float(np.mean(grader_scores)),
        "oracle_bon": float(np.max(grader_scores)),
        "oracle_idx": oracle_idx,
        "selector_selected_rubric": grader_scores[selector_idx],
        "selector_selected_idx": selector_idx,
        "selector_picked_oracle": int(selector_idx == oracle_idx),
        "rank_corr": spearman_rank_corr(
            grader_scores, [float(w) for w in selector_wins]
        ),
        "n_samples": n,
    }


# ============================================================
# Main
# ============================================================

def main(config: Config):
    # --- Output path ---
    if config.eval_output_path:
        output_dir = config.eval_output_path
    elif config.selector_checkpoint_run_path.lower() != "none":
        output_dir = os.path.join(config.selector_checkpoint_run_path, "eval_selector")
    else:
        output_dir = os.path.join(os.getcwd(), "eval_selector")
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
    logger.info(
        f"Test set: {len(dataset)} examples -> "
        f"{n_eval_batches} batches of {config.batch_size}"
    )

    # --- Tokenizer / renderer (policy) ---
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # --- Tokenizer / renderer (selector — may differ if different model) ---
    sel_tokenizer = get_tokenizer(config.selector_model_name)
    sel_renderer_name = model_info.get_recommended_renderer_name(config.selector_model_name)
    sel_renderer = renderers.get_renderer(sel_renderer_name, sel_tokenizer)

    # --- Tokenizer / renderer (grader — may differ from policy) ---
    grader_tokenizer = get_tokenizer(config.grader_model_name)
    grader_renderer_name = model_info.get_recommended_renderer_name(config.grader_model_name)
    grader_renderer = renderers.get_renderer(grader_renderer_name, grader_tokenizer)

    # --- Service client ---
    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )

    # --- Load policy checkpoint (bestversion) ---
    policy_resume, policy_batch_label = resolve_checkpoint(
        config.policy_checkpoint_run_path, config.policy_checkpoint_batch,
    )
    if policy_resume:
        policy_training_client = (
            service_client.create_training_client_from_state_with_optimizer(
                policy_resume["state_path"]
            )
        )
    else:
        policy_training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank,
        )

    policy_sr = policy_training_client.save_weights_for_sampler(
        name=f"eval_sel_policy_b{policy_batch_label}"
    ).result()
    policy_sampling_client = service_client.create_sampling_client(
        model_path=policy_sr.path,
    )

    # --- Load selector checkpoint ---
    sel_resume, sel_batch_label = resolve_checkpoint(
        config.selector_checkpoint_run_path, config.selector_checkpoint_batch,
    )
    if sel_resume:
        sel_training_client = (
            service_client.create_training_client_from_state_with_optimizer(
                sel_resume["state_path"]
            )
        )
    else:
        sel_training_client = service_client.create_lora_training_client(
            base_model=config.selector_model_name, rank=config.selector_lora_rank,
        )

    sel_sr = sel_training_client.save_weights_for_sampler(
        name=f"eval_sel_ranker_b{sel_batch_label}"
    ).result()
    sel_sampling_client = service_client.create_sampling_client(
        model_path=sel_sr.path,
    )

    # --- Grader client (frozen base model) ---
    grader_client = service_client.create_sampling_client(
        base_model=config.grader_model_name,
    )

    # --- Sampling params ---
    policy_sp = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )
    grader_sp = tinker.types.SamplingParams(
        max_tokens=config.grader_max_tokens,
        temperature=config.grader_temperature,
    )
    sel_sp = tinker.types.SamplingParams(
        max_tokens=config.selector_max_tokens,
        stop=sel_renderer.get_stop_sequences(),
        temperature=config.selector_temperature,
    )

    # --- Output files ---
    logs_path = os.path.join(
        output_dir, f"eval_sel_b{sel_batch_label}_logs.jsonl"
    )
    summary_path = os.path.join(
        output_dir, f"eval_sel_b{sel_batch_label}_summary.jsonl"
    )

    logger.info(
        f"Starting selector eval | policy=b{policy_batch_label} | "
        f"selector=b{sel_batch_label} | N={config.group_size} | "
        f"position_debias={config.position_debias} | "
        f"{n_eval_batches} batches"
    )

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
                    sampling_params=policy_sp,
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

            goal_plans = []
            goal_grader_futures = []

            for seq in result.sequences:
                plan_text = renderers.get_text_content(
                    renderer.parse_response(seq.tokens)[0]
                )
                if "<solution>" in plan_text and "</solution>" not in plan_text:
                    plan_text = plan_text.rstrip() + "\n</solution>"
                goal_plans.append(plan_text)

                grader_prompt = build_grader_prompt(
                    scenario=goal, rubric_items=rubric,
                    proposed_plan=plan_text, reference_solution=ref_sol,
                )
                grader_input = grader_renderer.build_generation_prompt(
                    [{"role": "user", "content": grader_prompt}]
                )
                goal_grader_futures.append(
                    grader_client.sample(
                        grader_input, num_samples=1, sampling_params=grader_sp,
                    )
                )

            all_grader_futures.append(goal_grader_futures)
            all_plan_texts.append(goal_plans)

        # === Phase 3: Collect grader results ===
        all_grader_scores: list[list[float]] = []

        for goal_idx in range(len(batch_rows)):
            goal_scores = []
            for g_future in all_grader_futures[goal_idx]:
                g_result = g_future.result()
                g_parsed, _ = grader_renderer.parse_response(g_result.sequences[0].tokens)
                g_xml = renderers.get_text_content(g_parsed)
                goal_scores.append(compute_rubric_reward_from_xml(g_xml))
            all_grader_scores.append(goal_scores)

        # === Phase 4: Pairwise selector comparisons ===
        logger.info("  Phase 4: pairwise selector comparisons...")

        # Structure: all_sel_futures[goal_idx] = [(i, j, future_fwd, future_rev), ...]
        all_sel_futures: list[list] = []

        for goal_idx in range(len(batch_rows)):
            goal = batch_rows["Goal"][goal_idx]
            plans = all_plan_texts[goal_idx]
            n_plans = len(plans)
            goal_sel_futures = []

            for i, j in combinations(range(n_plans), 2):
                # Forward comparison: A=i, B=j
                prompt_fwd = build_selector_prompt(goal, plans[i], plans[j])
                input_fwd = sel_renderer.build_generation_prompt(
                    [{"role": "user", "content": prompt_fwd}]
                )
                future_fwd = sel_sampling_client.sample(
                    input_fwd, num_samples=1, sampling_params=sel_sp,
                )

                future_rev = None
                if config.position_debias:
                    # Reverse comparison: A=j, B=i
                    prompt_rev = build_selector_prompt(goal, plans[j], plans[i])
                    input_rev = sel_renderer.build_generation_prompt(
                        [{"role": "user", "content": prompt_rev}]
                    )
                    future_rev = sel_sampling_client.sample(
                        input_rev, num_samples=1, sampling_params=sel_sp,
                    )

                goal_sel_futures.append((i, j, future_fwd, future_rev))

            all_sel_futures.append(goal_sel_futures)

        # === Phase 5: Collect selector results, compute metrics ===
        batch_logs: list[dict] = []
        batch_goal_metrics: list[dict] = []
        batch_n_agreements = 0
        batch_n_disagreements = 0
        batch_n_comparisons = 0

        for goal_idx in range(len(batch_rows)):
            plans = all_plan_texts[goal_idx]
            grader_scores = all_grader_scores[goal_idx]
            n_plans = len(plans)

            # Filter degenerate samples (cache word counts for reuse in logging)
            word_counts = [len(plans[j].strip().split()) for j in range(n_plans)]
            valid_mask = []
            for j in range(n_plans):
                is_degen = (
                    word_counts[j] < config.min_words
                    or plans[j].strip() in {
                        "<think>", "</think>", "<solution>", "</solution>"
                    }
                )
                valid_mask.append(not is_degen)

            valid_indices = [j for j in range(n_plans) if valid_mask[j]]
            if len(valid_indices) < 2:
                continue

            valid_scores = [grader_scores[j] for j in valid_indices]
            wins: dict[int, float] = {j: 0.0 for j in valid_indices}

            for idx_i, idx_j, future_fwd, future_rev in all_sel_futures[goal_idx]:
                if not valid_mask[idx_i] or not valid_mask[idx_j]:
                    continue

                batch_n_comparisons += 1

                # Parse forward result
                fwd_result = future_fwd.result()
                fwd_parsed, _ = sel_renderer.parse_response(
                    fwd_result.sequences[0].tokens
                )
                fwd_text = renderers.get_text_content(fwd_parsed)
                fwd_verdict = extract_verdict(fwd_text)

                if config.position_debias and future_rev is not None:
                    # Parse reverse result
                    rev_result = future_rev.result()
                    rev_parsed, _ = sel_renderer.parse_response(
                        rev_result.sequences[0].tokens
                    )
                    rev_text = renderers.get_text_content(rev_parsed)
                    rev_verdict = extract_verdict(rev_text)

                    # Translate verdicts to plan indices
                    # Forward: A=idx_i, B=idx_j
                    fwd_winner = (
                        idx_i if fwd_verdict == "A"
                        else idx_j if fwd_verdict == "B"
                        else None
                    )
                    # Reverse: A=idx_j, B=idx_i
                    rev_winner = (
                        idx_j if rev_verdict == "A"
                        else idx_i if rev_verdict == "B"
                        else None
                    )

                    if fwd_winner is not None and rev_winner is not None:
                        if fwd_winner == rev_winner:
                            # Both orderings agree
                            wins[fwd_winner] += 1.0
                            batch_n_agreements += 1
                        else:
                            # Disagreement (position bias) → tie
                            wins[idx_i] += 0.5
                            wins[idx_j] += 0.5
                            batch_n_disagreements += 1
                    elif fwd_winner is not None:
                        wins[fwd_winner] += 0.5
                    elif rev_winner is not None:
                        wins[rev_winner] += 0.5
                else:
                    # Single comparison (no debiasing)
                    if fwd_verdict == "A":
                        wins[idx_i] += 1.0
                    elif fwd_verdict == "B":
                        wins[idx_j] += 1.0

            valid_wins = [wins[j] for j in valid_indices]

            goal_met = compute_goal_metrics(valid_scores, valid_wins)
            batch_goal_metrics.append(goal_met)
            all_goal_metrics.append(goal_met)

            # Per-sample logs
            for j in valid_indices:
                wc = word_counts[j]
                is_compliant = check_format_compliance(plans[j], config.max_word_count)
                excess = max(0, wc - config.max_word_count)
                fmt_pen = 0.0 if is_compliant else 0.2 + 0.0005 * excess
                len_bonus = np.exp(
                    -((wc - config.target_word_count) / config.scale_length_bonus) ** 2
                )
                final_reward = grader_scores[j] + config.scaling_factor * len_bonus - fmt_pen

                batch_logs.append({
                    "eval_batch_idx": batch_idx,
                    "policy_batch": policy_batch_label,
                    "selector_batch": sel_batch_label,
                    "goal_idx": goal_idx,
                    "sample_idx": j,
                    "grader_rubric_score": grader_scores[j],
                    "selector_wins": wins.get(j, 0),
                    "final_reward": final_reward,
                    "word_count": wc,
                    "is_oracle_pick": j == goal_met["oracle_idx"],
                    "is_selector_pick": j == goal_met["selector_selected_idx"],
                })

        # --- Batch summary ---
        if batch_goal_metrics:
            mean_of_n = float(
                np.mean([m["mean_of_n"] for m in batch_goal_metrics])
            )
            oracle_bon = float(
                np.mean([m["oracle_bon"] for m in batch_goal_metrics])
            )
            sel_selected = float(
                np.mean([m["selector_selected_rubric"] for m in batch_goal_metrics])
            )
            oracle_gap = oracle_bon - mean_of_n
            sel_gap = sel_selected - mean_of_n

            agreement_rate = (
                batch_n_agreements / (batch_n_agreements + batch_n_disagreements)
                if (batch_n_agreements + batch_n_disagreements) > 0
                else 0.0
            )

            batch_summary = {
                "eval_batch_idx": batch_idx,
                "policy_batch": policy_batch_label,
                "selector_batch": sel_batch_label,
                "n_goals": len(batch_goal_metrics),
                "rubric/mean_of_n": mean_of_n,
                "rubric/oracle_bon": oracle_bon,
                "rubric/selector_selected": sel_selected,
                "selection/oracle_gap": oracle_gap,
                "selection/selector_gap": sel_gap,
                "selection/gap_closed_pct": (
                    100 * sel_gap / oracle_gap if oracle_gap > 0 else 0.0
                ),
                "selection/selector_picked_oracle": float(
                    np.mean([m["selector_picked_oracle"] for m in batch_goal_metrics])
                ),
                "selection/rank_corr": float(
                    np.mean([m["rank_corr"] for m in batch_goal_metrics])
                ),
                "position_debias/agreement_rate": agreement_rate,
                "position_debias/n_comparisons": batch_n_comparisons,
                "time_s": round(time.time() - t_start, 1),
            }
        else:
            batch_summary = {"eval_batch_idx": batch_idx, "n_goals": 0}

        logger.info(
            f"  mean={batch_summary.get('rubric/mean_of_n', 0):.4f}  "
            f"oracle={batch_summary.get('rubric/oracle_bon', 0):.4f}  "
            f"selector={batch_summary.get('rubric/selector_selected', 0):.4f}  "
            f"gap_closed={batch_summary.get('selection/gap_closed_pct', 0):.1f}%  "
            f"rank_corr={batch_summary.get('selection/rank_corr', 0):.3f}  "
            f"agree={batch_summary.get('position_debias/agreement_rate', 0):.3f}"
        )

        with open(logs_path, "a") as f:
            for item in batch_logs:
                f.write(json.dumps(item) + "\n")
        with open(summary_path, "a") as f:
            f.write(json.dumps(batch_summary) + "\n")

    # --- Final aggregate ---
    if all_goal_metrics:
        mean_of_n = float(np.mean([m["mean_of_n"] for m in all_goal_metrics]))
        oracle_bon = float(np.mean([m["oracle_bon"] for m in all_goal_metrics]))
        sel_selected = float(
            np.mean([m["selector_selected_rubric"] for m in all_goal_metrics])
        )
        oracle_gap = oracle_bon - mean_of_n
        sel_gap = sel_selected - mean_of_n

        aggregate = {
            "policy_batch": policy_batch_label,
            "selector_batch": sel_batch_label,
            "n_goals": len(all_goal_metrics),
            "group_size": config.group_size,
            "position_debias": config.position_debias,
            "rubric/mean_of_n": mean_of_n,
            "rubric/mean_of_n_std": float(
                np.std([m["mean_of_n"] for m in all_goal_metrics])
            ),
            "rubric/oracle_bon": oracle_bon,
            "rubric/oracle_bon_std": float(
                np.std([m["oracle_bon"] for m in all_goal_metrics])
            ),
            "rubric/selector_selected": sel_selected,
            "rubric/selector_selected_std": float(
                np.std(
                    [m["selector_selected_rubric"] for m in all_goal_metrics]
                )
            ),
            "selection/oracle_gap": oracle_gap,
            "selection/selector_gap": sel_gap,
            "selection/gap_closed_pct": (
                100 * sel_gap / oracle_gap if oracle_gap > 0 else 0.0
            ),
            "selection/selector_picked_oracle": float(
                np.mean(
                    [m["selector_picked_oracle"] for m in all_goal_metrics]
                )
            ),
            "selection/rank_corr_mean": float(
                np.mean([m["rank_corr"] for m in all_goal_metrics])
            ),
            "selection/rank_corr_std": float(
                np.std([m["rank_corr"] for m in all_goal_metrics])
            ),
        }

        agg_path = os.path.join(
            output_dir, f"eval_sel_b{sel_batch_label}_aggregate.json"
        )
        with open(agg_path, "w") as f:
            json.dump(aggregate, f, indent=2)

        logger.info("=" * 60)
        logger.info(
            f"SELECTOR EVAL COMPLETE  "
            f"policy=b{policy_batch_label}  selector=b{sel_batch_label}  "
            f"N={config.group_size}"
        )
        logger.info(f"  mean-of-N  :  {mean_of_n:.4f}")
        logger.info(f"  oracle BoN :  {oracle_bon:.4f}  (+{oracle_gap:.4f})")
        logger.info(f"  selector   :  {sel_selected:.4f}  (+{sel_gap:.4f})")
        logger.info(
            f"  gap closed :  {aggregate['selection/gap_closed_pct']:.1f}%"
        )
        logger.info(
            f"  rank corr  :  {aggregate['selection/rank_corr_mean']:.3f}"
        )
        logger.info(f"  results    :  {output_dir}")
        logger.info("=" * 60)

    ml_logger.close()


if __name__ == "__main__":
    chz.nested_entrypoint(main)
