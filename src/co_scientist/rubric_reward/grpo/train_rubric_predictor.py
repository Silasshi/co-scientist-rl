"""
Rubric predictor training using GRPO with token-F1 reward.

The model learns to predict rubric items from goal text. Reward is computed as
token-level F1 between predicted and actual rubric items (a simplified ROUGE-like
metric). Training uses the same GRPO loop as best_ver.py with loss_fn="ppo".

Usage:
  python src/co_scientist/trainers/grpo/train_rubric_predictor.py

  # Resume from checkpoint
  python src/co_scientist/trainers/grpo/train_rubric_predictor.py \\
      log_path=runs/2026/4/rubric_predictor/1
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

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
import datasets
import tinker
import torch
from co_scientist.shared.api_profiles import create_service_client
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from datasets import load_dataset


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    api_profile: str | None = None
    model_name: str = "Qwen/Qwen3-8B-Base"
    lora_rank: int = 32
    learning_rate: float = 1e-5
    clip_eps: float = 0.2
    batch_size: int = 32       # goals per batch
    group_size: int = 4        # predictions per goal
    max_tokens: int = 1024     # rubric items can be long
    temperature: float = 0.7
    max_length: int = 32768
    save_every: int = 50
    log_path: str = "runs/2026/4/rubric_predictor/1"
    ml_data: bool = True
    today_date = time.strftime("%Y-%m-%d", time.localtime())


# ============================================================
# Prompt builder (shared with train_rcgrpo.py)
# ============================================================

def build_predictor_prompt(goal_text: str) -> str:
    """Build prompt for rubric item prediction from a research goal."""
    return textwrap.dedent(f"""
        You are an expert research evaluator. Given a research goal, predict the specific evaluation criteria (rubric items) that would be used to assess a research plan for this goal.

        Be concrete and methodology-specific. List each criterion on a separate line.

        Research Goal:
        {goal_text}

        Predicted evaluation criteria:
    """).strip()


# ============================================================
# Reward function
# ============================================================

def compute_rubric_prediction_reward(predicted: str, actual_items: list[str]) -> float:
    """
    Compute reward based on token-level F1 between predicted and actual rubric items.
    This is a simplified ROUGE-like metric.
    """
    actual_text = " ".join(actual_items).lower()
    predicted_lower = predicted.lower()

    # Token-level F1
    actual_tokens = set(re.findall(r"[a-z0-9]+", actual_text))
    predicted_tokens = set(re.findall(r"[a-z0-9]+", predicted_lower))

    if not predicted_tokens or not actual_tokens:
        return 0.0

    overlap = actual_tokens & predicted_tokens
    precision = len(overlap) / len(predicted_tokens)
    recall = len(overlap) / len(actual_tokens)

    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ============================================================
# Main
# ============================================================

def main(config: Config):
    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    os.makedirs(config.log_path, exist_ok=True)

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load dataset
    logger.info("Loading dataset...")
    if config.ml_data:
        data = load_dataset("facebook/research-plan-gen", "ml")
    assert isinstance(data, datasets.DatasetDict)
    dataset = data["train"]

    n_train_batches = len(dataset) // config.batch_size
    logger.info(f"Dataset: {len(dataset)} goals, {n_train_batches} batches of {config.batch_size}")

    # Setup service client
    service_client = create_service_client(
        base_url=config.base_url,
        api_profile=config.api_profile,
    )

    # Load checkpoint or start fresh
    last_checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path)
    if last_checkpoint:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            last_checkpoint["state_path"]
        )
        if last_checkpoint["batch"] + 1 // n_train_batches > 0:
            logger.info(f"Training for epoch: {last_checkpoint['batch'] + 1 // n_train_batches}")
            actual_batch = last_checkpoint["batch"] + 1
            start_batch = 0
        else:
            start_batch = last_checkpoint["batch"] + 1
            actual_batch = start_batch
        logger.info(f"Resuming from batch {last_checkpoint['batch']}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0
        actual_batch = 0

    # Sampling parameters
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )

    # Optimization parameters
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    logger.info(f"Training for {n_train_batches} batches")

    # Main training loop
    for batch_idx in range(start_batch, n_train_batches):
        t_start = time.time()
        real_batch = actual_batch + (batch_idx - start_batch)

        # Save checkpoint periodically
        if config.save_every > 0 and batch_idx % config.save_every == 0 and batch_idx > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{real_batch:06d}_{config.today_date}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": real_batch},
            )

        # Get training batch
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(dataset))
        batch_rows = dataset.select(range(batch_start, batch_end))

        # Save weights for sampler
        sampling_result = training_client.save_weights_for_sampler(
            name=f"{real_batch:06d}"
        ).result()
        sampling_path = sampling_result.path
        sampling_client = service_client.create_sampling_client(
            model_path=sampling_path
        )

        logger.info(f"Batch {real_batch}: Launching {config.group_size} predictions for {len(batch_rows)} goals")

        # --- PHASE 1: LAUNCH PREDICTIONS (ASYNC) ---
        prediction_futures = []
        prediction_prompts_tokens = []

        for goal in batch_rows["Goal"]:
            prompt_text = build_predictor_prompt(goal_text=goal)
            convo = [{"role": "user", "content": prompt_text}]
            model_input = renderer.build_generation_prompt(convo)
            prediction_prompts_tokens.append(model_input.to_ints())

            prediction_futures.append(
                sampling_client.sample(
                    prompt=model_input,
                    num_samples=config.group_size,
                    sampling_params=sampling_params,
                )
            )

        # --- PHASE 2: COLLECT PREDICTIONS & COMPUTE REWARDS ---
        logger.info(f"Batch {real_batch}: Collecting predictions and computing rewards...")

        training_datums = []
        batch_rewards = []
        batch_sample_rewards = []
        batch_logs_to_save = []
        batch_advantages = []
        num_total_samples = 0
        num_valid_samples = 0
        dropped_samples = 0

        for group_idx, p_future in enumerate(prediction_futures):
            result = p_future.result()
            goal = batch_rows["Goal"][group_idx]
            actual_rubric_items = batch_rows["Rubric"][group_idx]
            prompt_tokens = prediction_prompts_tokens[group_idx]

            group_rewards = []
            valid_samples = []

            for j, seq in enumerate(result.sequences):
                num_total_samples += 1

                predicted_text = renderers.get_text_content(
                    renderer.parse_response(seq.tokens)[0]
                )

                # Compute token-F1 reward
                reward = compute_rubric_prediction_reward(predicted_text, actual_rubric_items)

                word_count = len(predicted_text.strip().split())

                # Drop degenerate samples
                if word_count < 5:
                    dropped_samples += 1
                    continue

                group_rewards.append(reward)
                batch_sample_rewards.append(reward)

                valid_samples.append({
                    "tokens": seq.tokens,
                    "logprobs": seq.logprobs,
                    "text": predicted_text,
                    "reward": reward,
                })

                num_valid_samples += 1

                batch_logs_to_save.append({
                    "batch_idx": real_batch,
                    "group_idx": group_idx,
                    "sample_idx": j,
                    "predicted_rubric": predicted_text,
                    "actual_rubric": " | ".join(actual_rubric_items),
                    "reward": reward,
                    "word_count": word_count,
                })

            # GRPO advantage calculation
            if not group_rewards:
                continue

            mean_reward = np.mean(group_rewards)
            advantages = [(r - mean_reward) for r in group_rewards]
            batch_rewards.append(mean_reward)
            batch_advantages.extend(advantages)

            # Skip if no learning signal
            if all(a == 0.0 for a in advantages):
                continue
            if len(valid_samples) < 2:
                continue

            # Create training datums
            for k, sample in enumerate(valid_samples):
                advantage = advantages[k]

                prompt_tokens_int = [int(t) for t in prompt_tokens]
                generated_tokens = [int(t) for t in sample["tokens"]]
                full_seq = prompt_tokens_int + generated_tokens

                ob_len = len(prompt_tokens_int) - 1
                input_tokens = full_seq[:-1]
                target_tokens = full_seq[1:]

                all_logprobs = [0.0] * ob_len + sample["logprobs"]
                all_advantages = [0.0] * ob_len + [advantage] * len(sample["logprobs"])

                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
                        "logprobs": TensorData.from_torch(torch.tensor(all_logprobs, dtype=torch.float)),
                        "advantages": TensorData.from_torch(torch.tensor(all_advantages, dtype=torch.float)),
                    },
                )
                training_datums.append(datum)

        # --- Batch summary ---
        batch_summary = {
            "batch_idx": real_batch,
            "reward/sample_mean": float(np.mean(batch_sample_rewards)) if batch_sample_rewards else 0.0,
            "reward/sample_std": float(np.std(batch_sample_rewards)) if batch_sample_rewards else 0.0,
            "reward/group_mean": float(np.mean(batch_rewards)) if batch_rewards else 0.0,
            "reward/group_std": float(np.std(batch_rewards)) if batch_rewards else 0.0,
            "advantage_mean": float(np.mean(batch_advantages)) if batch_advantages else 0.0,
            "advantage_std": float(np.std(batch_advantages)) if batch_advantages else 0.0,
            "samples/generated": num_total_samples,
            "samples/valid": num_valid_samples,
            "samples/dropped": dropped_samples,
        }

        # Write logs
        if batch_logs_to_save:
            logs_path = os.path.join(config.log_path, "train/training_logs.jsonl")
            os.makedirs(os.path.dirname(logs_path), exist_ok=True)
            with open(logs_path, "a") as f:
                for log_item in batch_logs_to_save:
                    f.write(json.dumps(log_item) + "\n")

        summary_path = os.path.join(config.log_path, "train/batch_summary.jsonl")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "a") as f:
            f.write(json.dumps(batch_summary) + "\n")

        # --- PHASE 3: OPTIMIZATION STEP ---
        if not training_datums:
            logger.warning("No valid datums this batch. Skipping optimization.")
            continue

        try:
            fwd_bwd_future = training_client.forward_backward(
                training_datums,
                loss_fn="ppo",
                loss_fn_config={
                    "clip_low_threshold": 1 - config.clip_eps,
                    "clip_high_threshold": 1 + config.clip_eps,
                }
            )
            optim_step_future = training_client.optim_step(adam_params)

            t0 = time.time()
            _fwd_bwd_result = fwd_bwd_future.result()
            logger.info(f"Forward/Backward took {time.time()-t0:.2f}s")

            t1 = time.time()
            _optim_result = optim_step_future.result()
            logger.info(f"Optim step took {time.time()-t1:.2f}s")
        except Exception as e:
            logger.exception("Training step failed")
            continue

        # Log metrics
        metrics: dict[str, float] = {
            "progress/batch": real_batch,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (real_batch + 1) / n_train_batches,
            "time/total": time.time() - t_start,
            "reward/total": sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0,
            "dropped_samples": dropped_samples,
        }
        ml_logger.log_metrics(metrics, step=real_batch)

    # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=f"{real_batch:06d}_final_{config.today_date}",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": real_batch},
    )
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
