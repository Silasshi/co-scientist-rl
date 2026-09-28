"""
Self-Selector training: pairwise plan ranking via GRPO.

Trains the model to compare two research plans and select the better one,
WITHOUT access to rubric items. The selector learns to internalize quality
criteria from pairwise preference labels derived from grader scores.

Improvements over train_cpr.py:
  - Structured prompt with 7 quality dimensions (generic, not rubric-specific)
  - Chain-of-thought reasoning (configurable max_tokens, default 512)
  - Periodic validation on held-out pairs
  - Position bias tracking

Usage:
  python train_selector.py
  python train_selector.py learning_rate=5e-6 max_tokens=256
  python train_selector.py val_every=25 val_sample_size=100
"""

import logging
import re
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
from co_scientist.shared.api_profiles import create_service_client
from co_scientist.ibt.train_ibt import create_training_datum
from tinker import types
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


# ============================================================
# Config
# ============================================================

@chz.chz
class Config:
    base_url: str | None = None
    api_profile: str | None = None
    model_name: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 64
    learning_rate: float = 1e-5
    clip_eps: float = 0.2

    batch_size: int = 32        # pairs per batch
    group_size: int = 8         # GRPO samples per pair
    max_tokens: int = 512       # allow chain-of-thought reasoning
    temperature: float = 1.0    # lower than CPR (1.5) since reasoning adds diversity
    max_length: int = 32768

    # Validation
    val_every: int = 50         # validate every N batches
    val_sample_size: int = 200  # number of val pairs per validation run
    val_temperature: float = 0.0  # deterministic for validation

    save_every: int = 50

    pairs_file: str = "data/cpr_pairs_train.jsonl"
    val_pairs_file: str = "data/cpr_pairs_val.jsonl"
    log_path: str = "runs/2026/4/selector/1"
    today_date = time.strftime("%Y-%m-%d", time.localtime())


# ============================================================
# Prompt
# ============================================================

def build_selector_prompt(goal: str, plan_a: str, plan_b: str) -> str:
    """Build a pairwise comparison prompt with quality dimension guidance.

    Unlike CPR's minimal prompt, this provides structured evaluation criteria
    (generic quality dimensions, NOT rubric items) and allows reasoning.
    """
    return (
        "You are an expert research plan evaluator. Given a research goal and "
        "two candidate plans, determine which plan better addresses the goal.\n\n"
        "Consider these quality dimensions:\n"
        "1. Thoroughness: Does the plan cover all aspects of the research goal?\n"
        "2. Specificity: Are the proposed methods concrete, detailed, and actionable?\n"
        "3. Soundness: Is the methodology scientifically rigorous?\n"
        "4. Justification: Are design choices well-motivated?\n"
        "5. Efficiency: Is the approach practical and not unnecessarily complex?\n"
        "6. Ethics: Are ethical considerations properly addressed?\n"
        "7. Coherence: Is the plan well-organized and internally consistent?\n\n"
        f"Research Goal:\n{goal}\n\n"
        f"Plan A:\n{plan_a}\n\n"
        f"Plan B:\n{plan_b}\n\n"
        "Compare both plans on the quality dimensions above. "
        "End your response with your verdict on a new line: VERDICT: A or VERDICT: B"
    )


def extract_verdict(text: str) -> str | None:
    """Extract A or B verdict from model response.

    Tries patterns in order of specificity:
    1. "VERDICT: A" or "VERDICT: B"
    2. "Plan A/B is better"
    3. "Better plan: A/B"
    4. Last standalone A or B in the final 100 characters
    """
    # Pattern 1: explicit VERDICT
    m = re.search(r"VERDICT:\s*([AB])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Pattern 2: "Plan X is better" / "Plan X is the better"
    m = re.search(r"Plan\s+([AB])\s+is\s+(?:the\s+)?better", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Pattern 3: "Better plan: X"
    m = re.search(r"[Bb]etter\s+plan:\s*(?:Plan\s+)?([AB])\b", text)
    if m:
        return m.group(1).upper()

    # Pattern 4: last A or B in the tail of the response
    tail = text[-100:] if len(text) > 100 else text
    for char in reversed(tail):
        if char in ("A", "B"):
            return char

    return None


# ============================================================
# Validation
# ============================================================

def run_validation(
    sampling_client,
    renderer,
    val_pairs: list[dict],
    sample_size: int,
    sampling_params,
) -> dict:
    """Run validation on a random subset of held-out pairs.

    Returns dict with accuracy, position bias, and parse rate.
    """
    if not val_pairs:
        return {"val_accuracy": 0.0, "val_parse_rate": 0.0, "val_position_bias": 0.5}

    subset = np.random.choice(len(val_pairs), size=min(sample_size, len(val_pairs)), replace=False)
    selected = [val_pairs[i] for i in subset]

    # Launch all comparisons
    futures = []
    for pair in selected:
        prompt_text = build_selector_prompt(
            goal=pair["goal"], plan_a=pair["plan_a"], plan_b=pair["plan_b"],
        )
        convo = [{"role": "user", "content": prompt_text}]
        model_input = renderer.build_generation_prompt(convo)
        futures.append(sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=sampling_params,
        ))

    # Collect results
    n_correct = 0
    n_parsed = 0
    n_picked_a = 0

    for i, future in enumerate(futures):
        result = future.result()
        parsed_msg = renderer.parse_response(result.sequences[0].tokens)[0]
        response_text = renderers.get_text_content(parsed_msg)

        verdict = extract_verdict(response_text)
        if verdict is None:
            continue

        n_parsed += 1
        if verdict == "A":
            n_picked_a += 1
        if verdict == selected[i]["label"]:
            n_correct += 1

    return {
        "val_accuracy": n_correct / n_parsed if n_parsed > 0 else 0.0,
        "val_parse_rate": n_parsed / len(selected),
        "val_position_bias": n_picked_a / n_parsed if n_parsed > 0 else 0.5,
        "val_n_evaluated": n_parsed,
    }


# ============================================================
# Main
# ============================================================

def main(config: Config):
    os.makedirs(config.log_path, exist_ok=True)
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None, wandb_name=None,
        config=config, do_configure_logging_module=True,
    )

    # Tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load pairs data
    PROJECT_ROOT = Path(SRC_ROOT).parent
    pairs_path = Path(config.pairs_file) if Path(config.pairs_file).is_absolute() else PROJECT_ROOT / config.pairs_file
    val_pairs_path = Path(config.val_pairs_file) if Path(config.val_pairs_file).is_absolute() else PROJECT_ROOT / config.val_pairs_file

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

    # Service client and training setup
    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )

    last_checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path)
    if last_checkpoint:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            last_checkpoint["state_path"]
        )
        start_batch = last_checkpoint["batch"] + 1
        logger.info(f"Resuming from batch {last_checkpoint['batch']}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank,
        )
        start_batch = 0

    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )
    val_sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.val_temperature,
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

    logger.info(
        f"Training for {n_batches} batches "
        f"({len(train_pairs)} pairs, batch_size={config.batch_size}, "
        f"group_size={config.group_size}, max_tokens={config.max_tokens})"
    )

    # ============================================================
    # Training loop
    # ============================================================
    for batch_idx in range(start_batch, n_batches):
        t_start = time.time()

        # --- Checkpoint ---
        if config.save_every > 0 and batch_idx % config.save_every == 0 and batch_idx > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{batch_idx:06d}_{config.today_date}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
            )

        # --- Get batch ---
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_pairs))
        batch_indices = indices[batch_start:batch_end]
        batch_pairs = [train_pairs[i] for i in batch_indices]

        # --- Save weights for sampling ---
        sampling_result = training_client.save_weights_for_sampler(
            name=f"{batch_idx:06d}"
        ).result()
        sampling_client = service_client.create_sampling_client(
            model_path=sampling_result.path,
        )

        logger.info(
            f"Batch {batch_idx}/{n_batches}: "
            f"{config.group_size} samples × {len(batch_pairs)} pairs"
        )

        # --- Phase 1: Build prompts and launch sampling ---
        prompt_futures = []
        prompt_tokens_list = []

        for pair in batch_pairs:
            prompt_text = build_selector_prompt(
                goal=pair["goal"], plan_a=pair["plan_a"], plan_b=pair["plan_b"],
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
        num_picked_a = 0

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

                verdict = extract_verdict(response_text)
                if verdict is None:
                    reward = 0.0
                else:
                    num_parsed += 1
                    if verdict == "A":
                        num_picked_a += 1
                    reward = 1.0 if verdict == label else 0.0
                    if verdict == label:
                        num_correct += 1

                group_rewards.append(reward)
                valid_samples.append({
                    "tokens": seq.tokens,
                    "logprobs": seq.logprobs,
                    "reward": reward,
                    "verdict": verdict,
                })

            if not group_rewards:
                continue

            # GRPO advantages
            mean_reward = np.mean(group_rewards)
            advantages = [(r - mean_reward) for r in group_rewards]

            batch_rewards.append(mean_reward)
            batch_advantages.extend(advantages)

            # Skip if no learning signal (all same answer)
            if all(a == 0.0 for a in advantages):
                continue
            if len(valid_samples) < 2:
                continue

            # Create training datums
            p_tokens = [int(t) for t in prompt_tokens]
            for k, sample in enumerate(valid_samples):
                datum = create_training_datum(
                    prompt_tokens=p_tokens,
                    generated_tokens=[int(t) for t in sample["tokens"]],
                    logprobs=sample["logprobs"],
                    advantages=advantages[k],
                )
                training_datums.append(datum)

            # Per-pair log
            pair_accuracy = (
                sum(1 for s in valid_samples if s["verdict"] == label)
                / max(1, sum(1 for s in valid_samples if s["verdict"] is not None))
            )
            batch_logs.append({
                "batch_idx": batch_idx,
                "pair_idx": pair_idx,
                "label": label,
                "score_gap": pair["score_gap"],
                "pair_accuracy": pair_accuracy,
                "group_mean_reward": float(mean_reward),
            })

        # --- Phase 3: Optimization step ---
        if not training_datums:
            logger.warning("No valid datums this batch. Skipping.")
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
            _fwd_bwd_result = fwd_bwd_future.result()
            _optim_result = optim_step_future.result()
        except Exception as e:
            logger.exception("Training step failed")
            continue

        # --- Batch summary ---
        accuracy = num_correct / num_parsed if num_parsed > 0 else 0.0
        parse_rate = num_parsed / num_total if num_total > 0 else 0.0
        position_bias = num_picked_a / num_parsed if num_parsed > 0 else 0.5

        batch_summary = {
            "batch_idx": batch_idx,
            "accuracy": accuracy,
            "parse_rate": parse_rate,
            "position_bias_a": position_bias,
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
            f"  acc={accuracy:.3f}  parse={parse_rate:.3f}  "
            f"pos_bias_A={position_bias:.3f}  "
            f"adv_std={batch_summary['advantage/std']:.3f}  "
            f"datums={len(training_datums)}  time={batch_summary['time_s']}s"
        )

        # Write logs
        with open(training_logs_path, "a") as f:
            for item in batch_logs:
                f.write(json.dumps(item) + "\n")
        with open(batch_summary_path, "a") as f:
            f.write(json.dumps(batch_summary) + "\n")

        ml_logger.log_metrics(
            {
                "progress/batch": batch_idx,
                "accuracy": accuracy,
                "position_bias_a": position_bias,
                "time/total": time.time() - t_start,
            },
            step=batch_idx,
        )

        # --- Periodic validation ---
        if config.val_every > 0 and (batch_idx + 1) % config.val_every == 0 and val_pairs:
            logger.info(f"  Running validation ({config.val_sample_size} pairs)...")
            t_val = time.time()
            val_metrics = run_validation(
                sampling_client=sampling_client,
                renderer=renderer,
                val_pairs=val_pairs,
                sample_size=config.val_sample_size,
                sampling_params=val_sampling_params,
            )
            val_metrics["batch_idx"] = batch_idx
            val_metrics["val_time_s"] = round(time.time() - t_val, 1)

            logger.info(
                f"  VAL acc={val_metrics['val_accuracy']:.3f}  "
                f"parse={val_metrics['val_parse_rate']:.3f}  "
                f"pos_bias_A={val_metrics['val_position_bias']:.3f}  "
                f"time={val_metrics['val_time_s']}s"
            )

            # Append to batch summary with val_ prefix
            val_summary = {**batch_summary, **val_metrics}
            with open(batch_summary_path, "a") as f:
                f.write(json.dumps({"validation": val_metrics}) + "\n")

            ml_logger.log_metrics(
                {
                    "val/accuracy": val_metrics["val_accuracy"],
                    "val/position_bias_a": val_metrics["val_position_bias"],
                },
                step=batch_idx,
            )

    # --- Final checkpoint ---
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=f"{n_batches - 1:06d}_final_{config.today_date}",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_batches - 1},
    )

    ml_logger.close()
    logger.info("Selector training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
