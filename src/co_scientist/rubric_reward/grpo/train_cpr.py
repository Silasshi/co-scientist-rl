"""
CPR (Contrastive Plan Ranking) training script.

Trains a pairwise plan ranker using GRPO with binary classification reward.
For each preference pair (goal, plan_a, plan_b, label), the model predicts
which plan is better ("A" or "B"). Reward is 1.0 if correct, 0.0 if wrong.

Uses the same GRPO training loop as best_ver.py but for a classification task.

Usage:
  python train_cpr.py
  python train_cpr.py pairs_file=data/cpr_pairs_train.jsonl
  python train_cpr.py learning_rate=5e-6 batch_size=16
"""

import logging
import time
import numpy as np
import json
import os
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
import tinker
import torch
from co_scientist.shared.api_profiles import create_service_client
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    api_profile: str | None = None
    model_name: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 64
    learning_rate: float = 1e-5
    clip_eps: float = 0.2
    batch_size: int = 32       # pairs per batch
    group_size: int = 8        # samples per pair — needs to be large for binary tasks
    max_tokens: int = 32       # only need "A" or "B" + short reasoning
    temperature: float = 1.5   # high temp needed: binary choice collapses GRPO at low temp
    max_length: int = 32768
    save_every: int = 50
    pairs_file: str = "data/cpr_pairs_train.jsonl"
    val_pairs_file: str = "data/cpr_pairs_val.jsonl"
    log_path: str = "runs/2026/4/cpr/1"
    today_date = time.strftime("%Y-%m-%d", time.localtime())


# ============================================================
# CPR Prompt
# ============================================================

def build_cpr_prompt(goal: str, plan_a: str, plan_b: str) -> str:
    """Build a pairwise comparison prompt for the CPR ranker."""
    return (
        "You are an expert research plan evaluator. Given a research goal and "
        "two candidate plans, determine which plan better addresses the goal. "
        'Answer with just "A" or "B".\n\n'
        f"Research Goal:\n{goal}\n\n"
        f"Plan A:\n{plan_a}\n\n"
        f"Plan B:\n{plan_b}\n\n"
        "The better plan is: Plan"
    )


def extract_choice(text: str) -> str | None:
    """Extract A or B from model response."""
    text = text.strip()
    for char in text:
        if char in ('A', 'B'):
            return char
    return None


# ============================================================
# Main
# ============================================================

def main(config: Config):
    # Setup logging
    os.makedirs(config.log_path, exist_ok=True)
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load pairs data
    PROJECT_ROOT = Path(SRC_ROOT).parent
    pairs_path = Path(config.pairs_file) if Path(config.pairs_file).is_absolute() else PROJECT_ROOT / config.pairs_file
    val_pairs_path = Path(config.val_pairs_file) if Path(config.val_pairs_file).is_absolute() else PROJECT_ROOT / config.val_pairs_file
    logger.info(f"Loading training pairs from: {pairs_path}")

    train_pairs = []
    with open(pairs_path) as f:
        for line in f:
            train_pairs.append(json.loads(line))
    logger.info(f"Loaded {len(train_pairs)} training pairs")

    val_pairs = []
    if val_pairs_path.exists():
        with open(val_pairs_path) as f:
            for line in f:
                val_pairs.append(json.loads(line))
        logger.info(f"Loaded {len(val_pairs)} validation pairs")

    # Setup service client
    service_client = create_service_client(
        base_url=config.base_url,
        api_profile=config.api_profile,
    )

    # Check for existing checkpoint to resume
    last_checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path)
    if last_checkpoint:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            last_checkpoint["state_path"]
        )
        start_batch = last_checkpoint["batch"] + 1
        logger.info(f"Resuming from batch {last_checkpoint['batch']}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0

    # Adam optimizer params
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    # Sampling params
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )

    # Shuffle and batch pairs
    np.random.seed(42)
    indices = np.random.permutation(len(train_pairs))
    n_batches = len(train_pairs) // config.batch_size

    # Output log files
    train_log_dir = os.path.join(config.log_path, "train")
    os.makedirs(train_log_dir, exist_ok=True)
    training_logs_path = os.path.join(train_log_dir, "training_logs.jsonl")
    batch_summary_path = os.path.join(train_log_dir, "batch_summary.jsonl")

    logger.info(f"Training for {n_batches} batches ({len(train_pairs)} pairs, batch_size={config.batch_size})")

    # ============================================================
    # Training loop
    # ============================================================
    for batch_idx in range(start_batch, n_batches):
        t_start = time.time()

        # Save checkpoint periodically
        if config.save_every > 0 and batch_idx % config.save_every == 0 and batch_idx > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{batch_idx:06d}_{config.today_date}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
            )

        # Get batch of pairs
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_pairs))
        batch_indices = indices[batch_start:batch_end]
        batch_pairs = [train_pairs[i] for i in batch_indices]

        # Save weights for sampling
        sampling_result = training_client.save_weights_for_sampler(
            name=f"{batch_idx:06d}"
        ).result()
        sampling_client = service_client.create_sampling_client(
            model_path=sampling_result.path
        )

        logger.info(
            f"Batch {batch_idx}/{n_batches}: "
            f"launching {config.group_size} samples for {len(batch_pairs)} pairs"
        )

        # --- Phase 1: Build prompts and launch sampling ---
        prompt_futures = []
        prompt_tokens_list = []

        for pair in batch_pairs:
            prompt_text = build_cpr_prompt(
                goal=pair["goal"],
                plan_a=pair["plan_a"],
                plan_b=pair["plan_b"],
            )
            convo = [{"role": "user", "content": prompt_text}]
            model_input = renderer.build_generation_prompt(convo)
            prompt_tokens_list.append(model_input.to_ints())

            prompt_futures.append(
                sampling_client.sample(
                    prompt=model_input,
                    num_samples=config.group_size,
                    sampling_params=sampling_params,
                )
            )

        # --- Phase 2: Collect responses, compute rewards and advantages ---
        training_datums = []
        batch_rewards = []
        batch_advantages = []
        batch_logs = []
        num_total = 0
        num_correct = 0
        num_parsed = 0

        for pair_idx, p_future in enumerate(prompt_futures):
            result = p_future.result()
            pair = batch_pairs[pair_idx]
            label = pair["label"]
            prompt_tokens = prompt_tokens_list[pair_idx]

            group_rewards = []
            valid_samples = []

            for seq in result.sequences:
                num_total += 1
                parsed_msg = renderer.parse_response(seq.tokens)[0]
                response_text = renderers.get_text_content(parsed_msg)

                choice = extract_choice(response_text)
                if choice is None:
                    # Unparseable response gets 0 reward
                    reward = 0.0
                else:
                    num_parsed += 1
                    reward = 1.0 if choice == label else 0.0
                    if choice == label:
                        num_correct += 1

                group_rewards.append(reward)
                valid_samples.append({
                    "tokens": seq.tokens,
                    "logprobs": seq.logprobs,
                    "reward": reward,
                    "choice": choice,
                    "response_text": response_text,
                })

            # GRPO advantage calculation within group
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

                p_tokens = [int(t) for t in prompt_tokens]
                g_tokens = [int(t) for t in sample["tokens"]]
                full_seq = p_tokens + g_tokens

                ob_len = len(p_tokens) - 1

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

            # Accuracy for this pair
            pair_accuracy = sum(1 for s in valid_samples if s["choice"] == label) / len(valid_samples)

            # Log first sample per pair
            batch_logs.append({
                "batch_idx": batch_idx,
                "pair_idx": pair_idx,
                "label": label,
                "score_gap": pair["score_gap"],
                "pair_accuracy": pair_accuracy,
                "group_mean_reward": float(mean_reward),
                "response_sample": valid_samples[0]["response_text"][:200] if valid_samples else "",
            })

        # --- Phase 3: Optimization step ---
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
                },
            )
            optim_step_future = training_client.optim_step(adam_params)

            t0 = time.time()
            _fwd_bwd_result = fwd_bwd_future.result()
            logger.info(f"Forward/Backward took {time.time() - t0:.2f}s")

            t1 = time.time()
            _optim_result = optim_step_future.result()
            logger.info(f"Optim step took {time.time() - t1:.2f}s")
        except Exception as e:
            logger.exception("Training step failed")
            continue

        # --- Batch summary ---
        accuracy = num_correct / num_parsed if num_parsed > 0 else 0.0
        parse_rate = num_parsed / num_total if num_total > 0 else 0.0

        batch_summary = {
            "batch_idx": batch_idx,
            "accuracy": accuracy,
            "parse_rate": parse_rate,
            "reward/mean": float(np.mean(batch_rewards)) if batch_rewards else 0.0,
            "reward/std": float(np.std(batch_rewards)) if batch_rewards else 0.0,
            "advantage/mean": float(np.mean(batch_advantages)) if batch_advantages else 0.0,
            "advantage/std": float(np.std(batch_advantages)) if batch_advantages else 0.0,
            "samples/total": num_total,
            "samples/parsed": num_parsed,
            "samples/correct": num_correct,
            "n_datums": len(training_datums),
            "time_s": round(time.time() - t_start, 1),
        }

        logger.info(
            f"  accuracy={accuracy:.3f}  "
            f"parse_rate={parse_rate:.3f}  "
            f"reward={batch_summary['reward/mean']:.3f}  "
            f"adv_std={batch_summary['advantage/std']:.3f}  "
            f"datums={len(training_datums)}  "
            f"time={batch_summary['time_s']}s"
        )

        # Write logs
        with open(training_logs_path, "a") as f:
            for item in batch_logs:
                f.write(json.dumps(item) + "\n")
        with open(batch_summary_path, "a") as f:
            f.write(json.dumps(batch_summary) + "\n")

        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / n_batches,
            "accuracy": accuracy,
            "time/total": time.time() - t_start,
        }
        ml_logger.log_metrics(metrics, step=batch_idx)

    # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=f"{batch_idx:06d}_final_{config.today_date}",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": batch_idx},
    )

    ml_logger.close()
    logger.info("CPR training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
