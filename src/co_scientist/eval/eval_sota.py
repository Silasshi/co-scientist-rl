"""
Evaluate a SOTA LLM on the research plan generation task using the canonical grader.

Generates plans from an external model via OpenRouter, then grades with the
canonical grader (Qwen3-30B-A3B on tinker). Results are directly comparable
to bestversion eval (0.693).

Usage:
  python eval_sota.py
  python eval_sota.py sota_model=openai/gpt-5.3-thinking
  python eval_sota.py sota_model=openai/gpt-5.3-thinking group_size=4
  python eval_sota.py batch_size=5  # quick test on 5 goals
"""

import asyncio
import logging
import time
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
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from datasets import load_dataset

from co_scientist.shared.eval_core import (
    build_research_plan_prompt,
    build_grader_prompt,
    compute_rubric_reward_from_xml,
    check_format_compliance,
)
from co_scientist.shared.openrouter_client import OpenRouterClient


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


def strip_thinking(text: str) -> str:
    """Strip <think>...</think> blocks from plan text before grading.

    Qwen3's tokenizer treats <think> as a special token. If raw text with
    <think> tags is embedded in the grader prompt, it triggers thinking mode
    and the grader produces truncated output. Stripping matches what
    eval_only.py does implicitly via renderer.parse_response().
    """
    import re
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


@chz.chz
class Config:
    # API
    base_url: str | None = None
    api_profile: str | None = None

    # SOTA model (via OpenRouter)
    sota_model: str = "openai/gpt-5.4"
    sota_temperature: float = 1.0
    sota_max_tokens: int = 4096
    group_size: int = 1

    # Grader (on tinker)
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_max_tokens: int = 8192
    grader_temperature: float = 0.0
    grader_max_retries: int = 3

    # Dataset
    ml_data: bool = True
    batch_size: int = 64

    # Reward / format
    max_word_count: int = 750
    target_word_count: int = 600
    scale_length_bonus: float = 120.0
    scaling_factor: float = 0.08
    min_words: int = 30

    # Output
    eval_output_path: str = "runs/2026/4/eval_sota/gpt53"


# ============================================================
# Async generation via OpenRouter
# ============================================================

async def generate_plans_openrouter(
    client: OpenRouterClient,
    goals: list[str],
    model: str,
    temperature: float,
    max_tokens: int,
    group_size: int,
) -> list[list[str]]:
    """Generate plans for all goals via OpenRouter. Returns list of list of plan texts."""

    async def _generate_one(goal: str) -> str:
        prompt_text = build_research_plan_prompt(scenario=goal)
        response = await client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Handle <solution> tag closure
        if "<solution>" in response and "</solution>" not in response:
            response = response.rstrip() + "\n</solution>"
        return response

    all_plans = []
    for goal in goals:
        goal_plans = []
        tasks = [_generate_one(goal) for _ in range(group_size)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Generation failed: {r}")
                goal_plans.append("")
            else:
                goal_plans.append(r)
        all_plans.append(goal_plans)

    return all_plans


# ============================================================
# Main
# ============================================================

def main(config: Config):
    output_dir = config.eval_output_path
    os.makedirs(output_dir, exist_ok=True)

    ml_logger = ml_log.setup_logging(
        log_dir=output_dir,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    # --- Load dataset (test split) ---
    logger.info("Loading dataset (test split)...")
    if config.ml_data:
        data = load_dataset("facebook/research-plan-gen", "ml")
    else:
        raise ValueError("Only ml_data supported")
    assert isinstance(data, datasets.DatasetDict)
    dataset = data["test"]

    n_eval_batches = len(dataset) // config.batch_size
    logger.info(
        f"Test set: {len(dataset)} examples -> {n_eval_batches} batches of {config.batch_size}"
    )

    # --- Tokenizer / renderer for grader ---
    tokenizer = get_tokenizer(config.grader_model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.grader_model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Grader renderer: {renderer_name}")

    # --- Service client for grader ---
    service_client = create_service_client(
        base_url=config.base_url,
        api_profile=config.api_profile,
    )
    grader_client = service_client.create_sampling_client(
        base_model=config.grader_model_name
    )
    grader_sampling_params = tinker.types.SamplingParams(
        max_tokens=config.grader_max_tokens,
        temperature=config.grader_temperature,
    )

    # --- Output files ---
    model_label = config.sota_model.replace("/", "_")
    logs_path = os.path.join(output_dir, f"eval_{model_label}_logs.jsonl")
    summary_path = os.path.join(output_dir, f"eval_{model_label}_summary.jsonl")

    logger.info(
        f"Starting SOTA eval | model={config.sota_model} | "
        f"group_size={config.group_size} | {n_eval_batches} batches | "
        f"output -> {output_dir}"
    )

    all_rubric_scores: list[float] = []
    all_rewards: list[float] = []
    all_format_penalties: list[float] = []

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

        # --- Phase 1: Generate plans from SOTA model via OpenRouter ---
        logger.info(f"  Phase 1: Generating {config.group_size} plan(s) per goal via {config.sota_model}...")
        t0 = time.time()

        async def _generate_batch():
            async with OpenRouterClient() as client:
                plans = await generate_plans_openrouter(
                    client=client,
                    goals=list(batch_rows["Goal"]),
                    model=config.sota_model,
                    temperature=config.sota_temperature,
                    max_tokens=config.sota_max_tokens,
                    group_size=config.group_size,
                )
                return plans, client.total_tokens

        all_plan_texts, batch_tokens = asyncio.run(_generate_batch())

        gen_time = time.time() - t0
        logger.info(f"  Generation done ({gen_time:.1f}s, {batch_tokens} tokens)")

        # --- Phase 2: Grade with canonical grader on tinker (with retry) ---
        logger.info("  Phase 2: Grading with canonical grader...")

        def grade_single_plan(plan_text, goal, rubric, ref_sol):
            """Grade a single plan, retrying if grader returns truncated output."""
            grader_prompt_text = build_grader_prompt(
                scenario=goal,
                rubric_items=rubric,
                proposed_plan=strip_thinking(plan_text),
                reference_solution=ref_sol,
            )
            grader_input = renderer.build_generation_prompt(
                [{"role": "user", "content": grader_prompt_text}]
            )
            for attempt in range(config.grader_max_retries):
                g_future = grader_client.sample(
                    grader_input,
                    num_samples=1,
                    sampling_params=grader_sampling_params,
                )
                g_result = g_future.result()
                parsed_msg, _ = renderer.parse_response(g_result.sequences[0].tokens)
                xml_text = renderers.get_text_content(parsed_msg)
                # Retry if grader returned truncated output (e.g., just "<think>")
                if len(xml_text) > 100:
                    return xml_text
                logger.warning(
                    f"  Grader returned truncated output ({len(xml_text)} chars), "
                    f"retry {attempt + 1}/{config.grader_max_retries}"
                )
            return xml_text  # return last attempt even if still short

        # Launch initial grader calls (async)
        all_grader_futures = []
        for goal_idx in range(len(batch_rows)):
            goal = batch_rows["Goal"][goal_idx]
            rubric = batch_rows["Rubric"][goal_idx]
            ref_sol = batch_rows["Reference solution"][goal_idx]

            goal_futures = []
            for plan_text in all_plan_texts[goal_idx]:
                grader_prompt_text = build_grader_prompt(
                    scenario=goal,
                    rubric_items=rubric,
                    proposed_plan=strip_thinking(plan_text),
                    reference_solution=ref_sol,
                )
                grader_input = renderer.build_generation_prompt(
                    [{"role": "user", "content": grader_prompt_text}]
                )
                g_future = grader_client.sample(
                    grader_input,
                    num_samples=1,
                    sampling_params=grader_sampling_params,
                )
                goal_futures.append(g_future)
            all_grader_futures.append(goal_futures)

        # --- Phase 3: Collect grades, retry failures, compute metrics ---
        batch_rubric_scores: list[float] = []
        batch_rewards: list[float] = []
        batch_format_penalties: list[float] = []
        batch_logs: list[dict] = []
        dropped = 0
        num_total = 0
        num_valid = 0
        grader_retries = 0

        for goal_idx in range(len(batch_rows)):
            goal = batch_rows["Goal"][goal_idx]
            rubric = batch_rows["Rubric"][goal_idx]
            ref_sol = batch_rows["Reference solution"][goal_idx]

            for sample_idx, g_future in enumerate(all_grader_futures[goal_idx]):
                num_total += 1
                plan_text = all_plan_texts[goal_idx][sample_idx]

                # Skip empty generations
                if not plan_text.strip():
                    dropped += 1
                    continue

                g_result = g_future.result()
                parsed_msg, _ = renderer.parse_response(g_result.sequences[0].tokens)
                xml_text = renderers.get_text_content(parsed_msg)

                # Retry if grader returned truncated output
                if len(xml_text) <= 100:
                    grader_retries += 1
                    xml_text = grade_single_plan(plan_text, goal, rubric, ref_sol)

                rubric_score = compute_rubric_reward_from_xml(xml_text)
                word_count = len(plan_text.strip().split())

                is_compliant = check_format_compliance(plan_text, config.max_word_count)
                excess = max(0, word_count - config.max_word_count)
                format_penalty = 0.0 if is_compliant else 0.2 + 0.0005 * excess
                length_bonus = np.exp(
                    -((word_count - config.target_word_count) / config.scale_length_bonus) ** 2
                )
                final_reward = rubric_score + config.scaling_factor * length_bonus - format_penalty

                batch_rubric_scores.append(rubric_score)
                batch_format_penalties.append(format_penalty)

                if word_count < config.min_words:
                    dropped += 1
                    continue
                if plan_text.strip() in {"<think>", "</think>", "<solution>", "</solution>"}:
                    dropped += 1
                    continue

                batch_rewards.append(final_reward)
                num_valid += 1

                batch_logs.append({
                    "eval_batch_idx": batch_idx,
                    "sota_model": config.sota_model,
                    "goal_idx": goal_idx,
                    "sample_idx": sample_idx,
                    "goal": goal,
                    "policy_output": plan_text,
                    "grader_output": xml_text,
                    "rubric_score": rubric_score,
                    "format_penalty": format_penalty,
                    "final_reward": final_reward,
                    "word_count": word_count,
                    "is_compliant": is_compliant,
                })

        # Accumulate across batches
        all_rubric_scores.extend(batch_rubric_scores)
        all_rewards.extend(batch_rewards)
        all_format_penalties.extend(batch_format_penalties)

        # Per-batch summary
        batch_summary = {
            "eval_batch_idx": batch_idx,
            "sota_model": config.sota_model,
            "rubric/mean": float(np.mean(batch_rubric_scores)) if batch_rubric_scores else 0.0,
            "rubric/std": float(np.std(batch_rubric_scores)) if batch_rubric_scores else 0.0,
            "reward/mean": float(np.mean(batch_rewards)) if batch_rewards else 0.0,
            "format_penalty/mean": float(np.mean(batch_format_penalties)) if batch_format_penalties else 0.0,
            "samples/total": num_total,
            "samples/valid": num_valid,
            "samples/dropped": dropped,
            "grader_retries": grader_retries,
            "time_s": round(time.time() - t_start, 1),
        }
        logger.info(
            f"  rubric={batch_summary['rubric/mean']:.4f}  "
            f"reward={batch_summary['reward/mean']:.4f}  "
            f"format_pen={batch_summary['format_penalty/mean']:.4f}  "
            f"valid={num_valid}/{num_total}  "
            f"time={batch_summary['time_s']}s"
        )

        # Write logs
        with open(logs_path, "a") as f:
            for item in batch_logs:
                f.write(json.dumps(item) + "\n")
        with open(summary_path, "a") as f:
            f.write(json.dumps(batch_summary) + "\n")

    # --- Final aggregate ---
    aggregate = {
        "sota_model": config.sota_model,
        "group_size": config.group_size,
        "n_eval_batches": n_eval_batches,
        "n_samples_total": len(all_rubric_scores),
        "n_samples_valid": len(all_rewards),
        "rubric/mean": float(np.mean(all_rubric_scores)) if all_rubric_scores else 0.0,
        "rubric/std": float(np.std(all_rubric_scores)) if all_rubric_scores else 0.0,
        "reward/mean": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "reward/std": float(np.std(all_rewards)) if all_rewards else 0.0,
        "format_penalty/mean": float(np.mean(all_format_penalties)) if all_format_penalties else 0.0,
        "comparison": {
            "bestversion_rubric": 0.693,
            "reference_rubric": 0.867,
        },
    }

    agg_path = os.path.join(output_dir, f"eval_{model_label}_aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"SOTA EVAL COMPLETE  model={config.sota_model}")
    logger.info(f"  rubric mean : {aggregate['rubric/mean']:.4f}")
    logger.info(f"  reward mean : {aggregate['reward/mean']:.4f}")
    logger.info(f"  format pen  : {aggregate['format_penalty/mean']:.4f}")
    logger.info(f"  valid/total : {aggregate['n_samples_valid']}/{aggregate['n_samples_total']}")
    logger.info(f"  vs bestversion: {aggregate['rubric/mean']:.4f} vs 0.693")
    logger.info(f"  vs reference : {aggregate['rubric/mean']:.4f} vs 0.867")
    logger.info(f"  results     : {output_dir}")
    logger.info("=" * 60)

    ml_logger.close()


if __name__ == "__main__":
    chz.nested_entrypoint(main)
