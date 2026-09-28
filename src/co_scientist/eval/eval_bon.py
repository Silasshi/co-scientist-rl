"""
Best-of-N evaluation with self-selection.

Generates N plans per goal, self-evaluates each (without rubric items),
selects the highest-rated plan, then grades all plans with the standard grader.

Reports three metrics:
  - mean-of-N:    average rubric score across all N samples (= current eval)
  - oracle-best:  best sample per goal by grader score (upper bound)
  - self-selected: plan chosen by self-evaluator's score

The self-evaluator infers likely evaluation criteria from the scenario alone
and scores the plan using the same 7 desiderata as the real grader. This tests
whether the model can close the selection gap (0.69 mean → 0.85 oracle).

Usage:
  python eval_bon.py checkpoint_run_path=/path/to/run
  python eval_bon.py checkpoint_run_path=/path/to/run self_eval_model=finetuned
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

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    # ---- Checkpoint selection ----
    checkpoint_run_path: str = "/home/silas/co-scientist-project/runs/2026/2/withA1,A2/2(ml)"
    checkpoint_batch: int = -1
    eval_output_path: str = ""

    # ---- Model / API ----
    base_url: str | None = None
    api_profile: str | None = None
    model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"

    # ---- Self-evaluation ----
    # "base" = grader model (no LoRA), "finetuned" = same checkpoint as policy
    self_eval_model: str = "base"
    self_eval_max_tokens: int = 8192
    self_eval_temperature: float = 0.0

    # ---- Dataset ----
    ml_data: bool = True
    arxiv_data: bool = False
    pubmed_data: bool = False

    # ---- Sampling ----
    batch_size: int = 64
    group_size: int = 8
    max_tokens: int = 2048
    grader_max_tokens: int = 8192
    temperature: float = 1.0
    grader_temperature: float = 0.0

    # ---- Reward / format ----
    max_word_count: int = 750
    target_word_count: int = 600
    scale_length_bonus: float = 120.0
    scaling_factor: float = 0.08
    min_words: int = 30

    # ---- LoRA rank ----
    lora_rank: int = 64


# ============================================================
# Self-evaluation prompt (no rubric items, no reference)
# ============================================================

def build_self_eval_prompt(scenario: str, proposed_plan: str) -> str:
    """
    Build a self-evaluation prompt that asks the model to:
    1. Infer what criteria an expert evaluator would check
    2. Score the plan against those inferred criteria using the standard 7 desiderata

    The output XML format matches the real grader's format so that
    compute_rubric_reward_from_xml() can parse it directly.
    """
    desiderata_text = textwrap.dedent("""
    1. HANDLES ALL CRITERIA: Does the plan satisfy what the criterion requires?
    2. DETAILED, SPECIFIC SOLUTION: Are implementation details concrete, not vague?
    3. NO OVERLOOKED FLAWS: Are there important weaknesses that undermine this?
    4. WELL JUSTIFIED RATIONALE: Is the approach motivated and justified?
    5. COST AND EFFORT EFFICIENT: Is the approach efficient, without unnecessary complexity?
    6. NO ETHICAL ISSUES: Are there potential negative consequences?
    7. CONSISTENT WITH OVERALL PLAN: Does this part cohere with the rest?
    """).strip()

    prompt = textwrap.dedent(f"""
        You are evaluating a research plan. You do NOT have access to the specific grading rubric. Your job is to infer what criteria an expert evaluator would check, then assess the plan.

        # Research Scenario
        {scenario}

        # Proposed Research Plan
        {proposed_plan}

        # Instructions
        Step 1 — Infer evaluation criteria:
        Read the scenario carefully. Identify the key technical criteria an expert evaluator would use to grade a plan for THIS scenario. Consider:
        - What specific goals and constraints does the scenario state or imply?
        - What methodological choices would distinguish a thorough plan from a superficial one?
        - What would an expert in this specific area check for?

        Step 2 — Evaluate the plan:
        For each inferred criterion, assess the plan using these 7 desiderata:
        {desiderata_text}

        Assign a satisfaction level for each desideratum:
          Level 0 — NOT SATISFIED
          Level 1 — WEAKLY SATISFIED
          Level 2 — PARTIALLY SATISFIED
          Level 3 — FULLY SATISFIED

        Be strict and evidence-based.

        Step 3 — Output your evaluation in this XML format:

        <rubric>
            <item num=1>
                <criteria>Your first inferred criterion</criteria>
                <reasoning>
                Brief assessment of the plan against the 7 desiderata for this criterion.
                </reasoning>
                <desiderata num=1><level>[0-3]</level></desiderata>
                <desiderata num=2><level>[0-3]</level></desiderata>
                <desiderata num=3><level>[0-3]</level></desiderata>
                <desiderata num=4><level>[0-3]</level></desiderata>
                <desiderata num=5><level>[0-3]</level></desiderata>
                <desiderata num=6><level>[0-3]</level></desiderata>
                <desiderata num=7><level>[0-3]</level></desiderata>
            </item>
            ... for all inferred criteria ...
        </rubric>
    """).strip()

    return prompt


# ============================================================
# Metrics
# ============================================================

def spearman_rank_corr(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation. Returns 0.0 for degenerate cases."""
    n = len(x)
    if n < 2:
        return 0.0
    arr_x = np.array(x)
    arr_y = np.array(y)
    if np.std(arr_x) == 0 or np.std(arr_y) == 0:
        return 0.0

    def _rank(vals):
        order = vals.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(vals) + 1, dtype=float)
        # Handle ties: average rank
        for v in np.unique(vals):
            mask = vals == v
            ranks[mask] = ranks[mask].mean()
        return ranks

    rx = _rank(arr_x)
    ry = _rank(arr_y)
    return float(np.corrcoef(rx, ry)[0, 1])


def compute_goal_metrics(
    grader_scores: list[float],
    self_eval_scores: list[float],
) -> dict:
    """Per-goal selection metrics."""
    n = len(grader_scores)
    assert n == len(self_eval_scores) and n > 0

    oracle_idx = int(np.argmax(grader_scores))
    self_idx = int(np.argmax(self_eval_scores))

    return {
        "mean_of_n": float(np.mean(grader_scores)),
        "oracle_bon": float(np.max(grader_scores)),
        "oracle_idx": oracle_idx,
        "self_selected_rubric": grader_scores[self_idx],
        "self_selected_idx": self_idx,
        "self_picked_oracle": int(self_idx == oracle_idx),
        "rank_corr": spearman_rank_corr(grader_scores, self_eval_scores),
        "n_samples": n,
    }


# ============================================================
# Main
# ============================================================

def main(config: Config):
    # --- Resolve output path ---
    if config.eval_output_path:
        output_dir = config.eval_output_path
    elif config.checkpoint_run_path.lower() != "none":
        output_dir = os.path.join(config.checkpoint_run_path, "eval_bon")
    else:
        output_dir = os.path.join(os.getcwd(), "eval_bon_base_model")
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
    logger.info(f"Test set: {len(dataset)} examples → {n_eval_batches} batches of {config.batch_size}")

    # --- Tokenizer / renderer ---
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # --- Service client ---
    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )

    # --- Checkpoint ---
    resume_info, batch_label = resolve_checkpoint(
        config.checkpoint_run_path, config.checkpoint_batch
    )
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )

    # --- Create sampling clients ---
    logger.info(f"Saving weights for eval sampler (checkpoint=b{batch_label})...")
    sampling_result = training_client.save_weights_for_sampler(
        name=f"eval_bon_b{batch_label}"
    ).result()
    sampling_client = service_client.create_sampling_client(
        model_path=sampling_result.path
    )
    grader_client = service_client.create_sampling_client(
        base_model=config.grader_model_name
    )

    # Self-eval client: base model or fine-tuned
    if config.self_eval_model == "finetuned":
        self_eval_client = sampling_client
        logger.info("Self-eval using fine-tuned model")
    else:
        self_eval_client = grader_client
        logger.info("Self-eval using base model")

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )
    grader_sampling_params = tinker.types.SamplingParams(
        max_tokens=config.grader_max_tokens,
        temperature=config.grader_temperature,
    )
    self_eval_sampling_params = tinker.types.SamplingParams(
        max_tokens=config.self_eval_max_tokens,
        temperature=config.self_eval_temperature,
    )

    # --- Output files ---
    logs_path = os.path.join(output_dir, f"eval_b{batch_label}_bon_logs.jsonl")
    summary_path = os.path.join(output_dir, f"eval_b{batch_label}_bon_summary.jsonl")

    logger.info(
        f"Starting BoN eval | checkpoint=b{batch_label} | "
        f"N={config.group_size} | {n_eval_batches} batches | "
        f"self_eval_model={config.self_eval_model} | output → {output_dir}"
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
            f"(goals {batch_start}–{batch_end - 1})"
        )

        # --- Phase 1: Launch policy generations ---
        policy_futures = []
        for goal in batch_rows["Goal"]:
            prompt_text = build_research_plan_prompt(scenario=goal)
            convo = [{"role": "user", "content": prompt_text}]
            model_input = renderer.build_generation_prompt(convo)
            policy_futures.append(
                sampling_client.sample(
                    prompt=model_input,
                    num_samples=config.group_size,
                    sampling_params=sampling_params,
                )
            )

        # --- Phase 2: Collect plans, launch graders + self-evals ---
        # Indexed [goal_idx][sample_idx]
        all_grader_futures: list[list] = []
        all_self_eval_futures: list[list] = []
        all_plan_texts: list[list[str]] = []

        for i, p_future in enumerate(policy_futures):
            result = p_future.result()
            goal = batch_rows["Goal"][i]
            rubric = batch_rows["Rubric"][i]
            ref_sol = batch_rows["Reference solution"][i]

            goal_grader_futures = []
            goal_self_eval_futures = []
            goal_plans = []

            for seq in result.sequences:
                plan_text = renderers.get_text_content(
                    renderer.parse_response(seq.tokens)[0]
                )
                if "<solution>" in plan_text and "</solution>" not in plan_text:
                    plan_text = plan_text.rstrip() + "\n</solution>"
                goal_plans.append(plan_text)

                # Launch grader (with rubric + reference)
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

                # Launch self-eval (scenario + plan only, no rubric)
                self_eval_prompt = build_self_eval_prompt(
                    scenario=goal, proposed_plan=plan_text,
                )
                self_eval_input = renderer.build_generation_prompt(
                    [{"role": "user", "content": self_eval_prompt}]
                )
                goal_self_eval_futures.append(
                    self_eval_client.sample(
                        self_eval_input, num_samples=1,
                        sampling_params=self_eval_sampling_params,
                    )
                )

            all_grader_futures.append(goal_grader_futures)
            all_self_eval_futures.append(goal_self_eval_futures)
            all_plan_texts.append(goal_plans)

        # --- Phase 3: Collect results, compute per-goal metrics ---
        batch_logs: list[dict] = []
        batch_goal_metrics: list[dict] = []

        for goal_idx in range(len(batch_rows)):
            goal = batch_rows["Goal"][goal_idx]
            plans = all_plan_texts[goal_idx]
            grader_futs = all_grader_futures[goal_idx]
            self_eval_futs = all_self_eval_futures[goal_idx]

            grader_scores: list[float] = []
            self_eval_scores: list[float] = []
            sample_details: list[dict] = []

            for j in range(len(plans)):
                plan_text = plans[j]
                word_count = len(plan_text.strip().split())

                # Grader result
                g_result = grader_futs[j].result()
                g_parsed, _ = renderer.parse_response(g_result.sequences[0].tokens)
                g_xml = renderers.get_text_content(g_parsed)
                grader_rubric = compute_rubric_reward_from_xml(g_xml)

                # Self-eval result
                se_result = self_eval_futs[j].result()
                se_parsed, _ = renderer.parse_response(se_result.sequences[0].tokens)
                se_xml = renderers.get_text_content(se_parsed)
                self_eval_score = compute_rubric_reward_from_xml(se_xml)

                # Reward (same formula as eval_only.py)
                is_compliant = check_format_compliance(plan_text, config.max_word_count)
                excess = max(0, word_count - config.max_word_count)
                format_penalty = 0.0 if is_compliant else 0.2 + 0.0005 * excess
                length_bonus = np.exp(
                    -((word_count - config.target_word_count) / config.scale_length_bonus) ** 2
                )
                final_reward = grader_rubric + config.scaling_factor * length_bonus - format_penalty

                # Filter degenerate samples
                is_degenerate = (
                    word_count < config.min_words
                    or plan_text.strip() in {"<think>", "</think>", "<solution>", "</solution>"}
                )

                if not is_degenerate:
                    grader_scores.append(grader_rubric)
                    self_eval_scores.append(self_eval_score)

                sample_details.append({
                    "eval_batch_idx": batch_idx,
                    "checkpoint_batch": batch_label,
                    "goal_idx": goal_idx,
                    "sample_idx": j,
                    "goal": goal,
                    "policy_output": plan_text,
                    "grader_output": g_xml,
                    "grader_rubric_score": grader_rubric,
                    "self_eval_output": se_xml,
                    "self_eval_score": self_eval_score,
                    "format_penalty": format_penalty,
                    "final_reward": final_reward,
                    "word_count": word_count,
                    "is_compliant": is_compliant,
                    "is_degenerate": is_degenerate,
                })

            # Skip goal if no valid samples
            if not grader_scores:
                logger.warning(f"  Goal {goal_idx}: all samples degenerate, skipping")
                batch_logs.extend(sample_details)
                continue

            # Compute goal-level metrics
            goal_met = compute_goal_metrics(grader_scores, self_eval_scores)
            batch_goal_metrics.append(goal_met)
            all_goal_metrics.append(goal_met)

            # Mark oracle and self-selected in sample details
            valid_idx = 0
            for sd in sample_details:
                if not sd["is_degenerate"]:
                    sd["is_oracle_pick"] = (valid_idx == goal_met["oracle_idx"])
                    sd["is_self_pick"] = (valid_idx == goal_met["self_selected_idx"])
                    valid_idx += 1
                else:
                    sd["is_oracle_pick"] = False
                    sd["is_self_pick"] = False
            batch_logs.extend(sample_details)

        # --- Phase 4: Batch summary ---
        if batch_goal_metrics:
            mean_of_n = float(np.mean([m["mean_of_n"] for m in batch_goal_metrics]))
            oracle_bon = float(np.mean([m["oracle_bon"] for m in batch_goal_metrics]))
            self_selected = float(np.mean([m["self_selected_rubric"] for m in batch_goal_metrics]))
            oracle_gap = oracle_bon - mean_of_n
            self_gap = self_selected - mean_of_n

            batch_summary = {
                "eval_batch_idx": batch_idx,
                "checkpoint_batch": batch_label,
                "n_goals": len(batch_goal_metrics),

                "rubric/mean_of_n": mean_of_n,
                "rubric/oracle_bon": oracle_bon,
                "rubric/self_selected": self_selected,

                "selection/oracle_gap": oracle_gap,
                "selection/self_gap": self_gap,
                "selection/gap_closed_pct": 100 * self_gap / oracle_gap if oracle_gap > 0 else 0.0,
                "selection/self_picked_oracle": float(np.mean([m["self_picked_oracle"] for m in batch_goal_metrics])),
                "selection/rank_corr": float(np.mean([m["rank_corr"] for m in batch_goal_metrics])),

                "time_s": round(time.time() - t_start, 1),
            }
        else:
            batch_summary = {"eval_batch_idx": batch_idx, "n_goals": 0}

        logger.info(
            f"  mean={batch_summary.get('rubric/mean_of_n', 0):.4f}  "
            f"oracle={batch_summary.get('rubric/oracle_bon', 0):.4f}  "
            f"self={batch_summary.get('rubric/self_selected', 0):.4f}  "
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
        self_selected = float(np.mean([m["self_selected_rubric"] for m in all_goal_metrics]))
        oracle_gap = oracle_bon - mean_of_n
        self_gap = self_selected - mean_of_n

        aggregate = {
            "checkpoint_batch": batch_label,
            "self_eval_model": config.self_eval_model,
            "n_goals": len(all_goal_metrics),
            "group_size": config.group_size,

            "rubric/mean_of_n": mean_of_n,
            "rubric/mean_of_n_std": float(np.std([m["mean_of_n"] for m in all_goal_metrics])),
            "rubric/oracle_bon": oracle_bon,
            "rubric/oracle_bon_std": float(np.std([m["oracle_bon"] for m in all_goal_metrics])),
            "rubric/self_selected": self_selected,
            "rubric/self_selected_std": float(np.std([m["self_selected_rubric"] for m in all_goal_metrics])),

            "selection/oracle_gap": oracle_gap,
            "selection/self_gap": self_gap,
            "selection/gap_closed_pct": 100 * self_gap / oracle_gap if oracle_gap > 0 else 0.0,
            "selection/self_picked_oracle": float(np.mean([m["self_picked_oracle"] for m in all_goal_metrics])),
            "selection/rank_corr_mean": float(np.mean([m["rank_corr"] for m in all_goal_metrics])),
            "selection/rank_corr_std": float(np.std([m["rank_corr"] for m in all_goal_metrics])),
        }

        agg_path = os.path.join(output_dir, f"eval_b{batch_label}_bon_aggregate.json")
        with open(agg_path, "w") as f:
            json.dump(aggregate, f, indent=2)

        logger.info("=" * 60)
        logger.info(f"BON EVAL COMPLETE  checkpoint=b{batch_label}  N={config.group_size}")
        logger.info(f"  mean-of-N :  {mean_of_n:.4f}")
        logger.info(f"  oracle BoN:  {oracle_bon:.4f}  (+{oracle_gap:.4f})")
        logger.info(f"  self-select: {self_selected:.4f}  (+{self_gap:.4f})")
        logger.info(f"  gap closed:  {aggregate['selection/gap_closed_pct']:.1f}%")
        logger.info(f"  rank corr :  {aggregate['selection/rank_corr_mean']:.3f}")
        logger.info(f"  results   :  {output_dir}")
        logger.info("=" * 60)

    ml_logger.close()


if __name__ == "__main__":
    chz.nested_entrypoint(main)
