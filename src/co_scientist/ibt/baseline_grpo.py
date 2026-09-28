"""
Baseline B: Standard GRPO (single-pass, no iteration).

For each goal: generate G=8 samples, grade all, compute group-relative
advantages, one PPO update. No hints, no iteration. Move to next goal
with updated weights.

This is the bestversion approach applied to the 4B model on 25 goals
sequentially.  Tests: does iterating with hints beat standard GRPO?

Usage:
  python baseline_grpo.py num_goals=25 group_size=8 api_profile=NEW
"""

import json
import logging
import os
import sys
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
from tinker import types
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from datasets import load_dataset

# Reuse from IBT (same grader prompt with hint for fair comparison)
from co_scientist.ibt.train_ibt import (
    build_grader_with_hint_prompt,
    build_plan_prompt,
    compute_rubric_score,
    compute_reward,
    create_training_datum,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    api_profile: str | None = "NEW"
    log_path: str = "/home/silas/co-scientist-project/runs/2026/4/ibt/1"

    policy_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    grader_model: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 32

    # Load shared init checkpoint (empty string = fresh LoRA)
    init_checkpoint: str = ""

    ml_data: bool = True
    arxiv_data: bool = False
    pubmed_data: bool = False

    num_goals: int = 25
    start_goal: int = 0
    group_size: int = 8

    learning_rate: float = 1e-5
    clip_eps: float = 0.2

    max_tokens: int = 2048
    temperature: float = 1.0
    grader_max_tokens: int = 8192
    grader_temperature: float = 0.0

    max_word_count: int = 750
    target_word_count: int = 600
    scale_length_bonus: float = 140.0
    scaling_factor: float = 0.08
    min_words: int = 30

    today_date: str = time.strftime("%Y-%m-%d", time.localtime())


def main(config: Config):
    run_dir = os.path.join(config.log_path, "baseline_grpo")
    os.makedirs(run_dir, exist_ok=True)
    train_dir = os.path.join(run_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    ml_logger = ml_log.setup_logging(
        log_dir=run_dir, wandb_project=None, wandb_name=None,
        config=config, do_configure_logging_module=True,
    )

    logger.info(f"Baseline: Standard GRPO (G={config.group_size})")
    logger.info(f"Policy: {config.policy_model}")
    logger.info(f"Grader: {config.grader_model}")
    logger.info(f"Goals: {config.num_goals}, group_size: {config.group_size}")

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({
            "baseline": "grpo",
            "policy_model": config.policy_model,
            "grader_model": config.grader_model,
            "num_goals": config.num_goals,
            "group_size": config.group_size,
            "learning_rate": config.learning_rate,
            "lora_rank": config.lora_rank,
        }, f, indent=2)

    # ── Tokenizer & renderer ──
    tokenizer = get_tokenizer(config.policy_model)
    renderer_name = model_info.get_recommended_renderer_name(config.policy_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    grader_tokenizer = get_tokenizer(config.grader_model)
    grader_renderer_name = model_info.get_recommended_renderer_name(config.grader_model)
    grader_renderer = renderers.get_renderer(grader_renderer_name, grader_tokenizer)

    # ── Dataset ──
    if config.ml_data:
        data = load_dataset("facebook/research-plan-gen", "ml")
    elif config.arxiv_data:
        data = load_dataset("facebook/research-plan-gen", "arxiv")
    elif config.pubmed_data:
        data = load_dataset("facebook/research-plan-gen", "pubmed")
    assert isinstance(data, datasets.DatasetDict)
    dataset = data["train"]
    num_goals = min(config.num_goals, len(dataset))

    # ── Clients ──
    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )

    if config.init_checkpoint:
        logger.info(f"Loading shared init checkpoint: {config.init_checkpoint}")
        ckpt = checkpoint_utils.get_last_checkpoint(config.init_checkpoint)
        assert ckpt is not None, f"No checkpoint found at {config.init_checkpoint}"
        training_client = service_client.create_training_client_from_state_with_optimizer(
            ckpt["state_path"]
        )
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.policy_model, rank=config.lora_rank
        )

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

    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    # ── Log files ──
    f_summary = open(os.path.join(train_dir, "batch_summary.jsonl"), "a")
    f_logs = open(os.path.join(train_dir, "training_logs.jsonl"), "a")

    t_start = time.time()

    if config.start_goal > 0:
        logger.info(f"Resuming from goal {config.start_goal}")

    for goal_idx in range(config.start_goal, num_goals):
        goal_t0 = time.time()
        goal_row = dataset[goal_idx]
        goal_text = goal_row["Goal"]
        rubric_items = goal_row["Rubric"]
        ref_solution = goal_row["Reference solution"]

        logger.info(f"\n{'='*60}")
        logger.info(f"GOAL {goal_idx}/{num_goals}: {goal_text[:100]}...")

        # ── 1. Save weights for sampling ──
        sr = training_client.save_weights_for_sampler(
            name=f"g{goal_idx:04d}"
        ).result()
        sampling_client = service_client.create_sampling_client(
            model_path=sr.path
        )

        # ── 2. Build prompt (no hint — single pass) ──
        prompt_text = build_plan_prompt(scenario=goal_text, hint=None)
        convo = [{"role": "user", "content": prompt_text}]
        model_input = renderer.build_generation_prompt(convo)
        prompt_tokens = [int(t) for t in model_input.to_ints()]

        # ── 3. Generate G samples ──
        sample_result = sampling_client.sample(
            prompt=model_input,
            num_samples=config.group_size,
            sampling_params=sampling_params,
        ).result()

        # ── 4. Parse plans & launch grading ──
        plans = []
        grader_futures = []

        for seq in sample_result.sequences:
            plan_text = renderers.get_text_content(
                renderer.parse_response(seq.tokens)[0]
            )
            if "<solution>" in plan_text and "</solution>" not in plan_text:
                plan_text = plan_text.rstrip() + "\n</solution>"

            plans.append({
                "text": plan_text,
                "tokens": seq.tokens,
                "logprobs": seq.logprobs,
            })

            grader_prompt = build_grader_with_hint_prompt(
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

        # ── 5. Collect grades ──
        valid_plans = []

        for k, gf in enumerate(grader_futures):
            grader_result = gf.result()
            grader_text = renderers.get_text_content(
                grader_renderer.parse_response(grader_result.sequences[0].tokens)[0]
            )

            rubric_score = compute_rubric_score(grader_text)
            plan_text = plans[k]["text"]
            reward, word_count = compute_reward(plan_text, rubric_score, config)

            if word_count < config.min_words:
                logger.warning(f"  Dropping sample {k}: too short ({word_count} words)")
                continue

            valid_plans.append({
                **plans[k],
                "reward": reward,
                "rubric_score": rubric_score,
                "word_count": word_count,
            })

        if not valid_plans:
            logger.warning(f"  No valid plans, skipping goal {goal_idx}")
            continue

        # ── 6. GRPO advantages ──
        rewards = [vp["reward"] for vp in valid_plans]
        rubric_scores = [vp["rubric_score"] for vp in valid_plans]
        mean_reward = np.mean(rewards)
        advantages = [r - mean_reward for r in rewards]

        logger.info(
            f"  G={len(valid_plans)}, rubric mean={np.mean(rubric_scores):.3f}, "
            f"best={max(rubric_scores):.3f}, reward mean={mean_reward:.3f}"
        )

        # ── 7. Create datums & PPO update ──
        if not all(a == 0.0 for a in advantages) and len(valid_plans) >= 2:
            training_datums = []
            for k, vp in enumerate(valid_plans):
                datum = create_training_datum(
                    prompt_tokens=prompt_tokens,
                    generated_tokens=[int(t) for t in vp["tokens"]],
                    logprobs=vp["logprobs"],
                    advantages=advantages[k],
                )
                training_datums.append(datum)

            try:
                fwd_bwd = training_client.forward_backward(
                    training_datums,
                    loss_fn="ppo",
                    loss_fn_config={
                        "clip_low_threshold": 1 - config.clip_eps,
                        "clip_high_threshold": 1 + config.clip_eps,
                    },
                )
                optim = training_client.optim_step(adam_params)
                fwd_bwd.result()
                optim.result()
                logger.info(f"  PPO update done ({len(training_datums)} datums)")
            except Exception as e:
                logger.exception(f"  Training step failed: {e}")

        # ── 8. Log ──
        goal_summary = {
            "type": "goal_summary",
            "goal_idx": goal_idx,
            "mode": "grpo",
            "rubric/mean": float(np.mean(rubric_scores)),
            "rubric/best": float(max(rubric_scores)),
            "rubric/worst": float(min(rubric_scores)),
            "rubric/std": float(np.std(rubric_scores)),
            "reward/mean": float(mean_reward),
            "n_valid": len(valid_plans),
            "n_total": config.group_size,
            "time": time.time() - goal_t0,
            # Single-pass: no trajectory. Use per-sample scores instead.
            "rubric_scores": [float(s) for s in rubric_scores],
            "rubric_first": float(np.mean(rubric_scores)),
            "rubric_last": float(np.mean(rubric_scores)),
            "rubric_best": float(max(rubric_scores)),
            "improvement": 0.0,
            "total_steps": 1,
        }
        f_summary.write(json.dumps(goal_summary) + "\n")

        for k, vp in enumerate(valid_plans):
            sample_log = {
                "goal_idx": goal_idx,
                "sample_idx": k,
                "rubric_score": vp["rubric_score"],
                "reward": vp["reward"],
                "word_count": vp["word_count"],
                "advantage": advantages[k] if k < len(advantages) else 0.0,
                "policy_output": vp["text"][:500],
            }
            f_logs.write(json.dumps(sample_log) + "\n")

        f_summary.flush()

        ml_logger.log_metrics({
            "progress/goal": goal_idx,
            "rubric/mean": float(np.mean(rubric_scores)),
            "rubric/best": float(max(rubric_scores)),
            "reward/mean": float(mean_reward),
            "time/total": time.time() - t_start,
        }, step=goal_idx)

    f_summary.close()
    f_logs.close()

    # ── Save final checkpoint ──
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=f"final_{num_goals}goals_{config.today_date}",
        log_path=run_dir,
        kind="state",
        loop_state={"goal": num_goals - 1},
    )
    logger.info(f"Done. {num_goals} goals, {num_goals * config.group_size} total rollouts.")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
