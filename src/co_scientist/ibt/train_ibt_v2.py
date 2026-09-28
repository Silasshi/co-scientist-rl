"""
IBT v2: Three new training setups for isolating RL vs hint effects.

All setups: NO hints in the model's input prompt. Hints used only for
supervision signals (reward computation, self-distillation targets).

Setup 1 (self_distill_ratio): Modified PPO ratio
  - Per-token advantage = clip(π(t|goal+hint) / π(t|goal)) × scalar_advantage
  - Hint signal acts as multiplicative weight on RL advantage
  - Zero signal when advantage is zero (safe)

Setup 2 (self_distill_additive): OPD-style additive
  - Per-token advantage = w_rl × scalar_advantage + w_sd × clip(log_hint - log_base)
  - Same as OPD but using same model (self-distillation)

Setup 3 (dense_grpo): Multi-round sample accumulation
  - Each turn generates G samples, stored in buffer
  - GRPO advantage computed over ALL accumulated samples
  - Old samples reweighted by importance ratio; dropped if too stale

Usage:
  python train_ibt_v2.py mode=self_distill_ratio num_goals=50 api_profile=NEW
  python train_ibt_v2.py mode=self_distill_additive num_goals=50 api_profile=NEW
  python train_ibt_v2.py mode=dense_grpo group_size=4 num_goals=50 api_profile=NEW
"""

import json
import logging
import os
import re
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
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

from co_scientist.ibt.train_ibt import (
    build_plan_prompt,
    build_grader_with_hint_prompt,
    compute_rubric_score,
    compute_reward,
    create_training_datum as create_datum,
    extract_hint,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    api_profile: str | None = "NEW"
    log_path: str = "/home/silas/co-scientist-project/runs/2026/4/ibt/6"

    policy_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    grader_model: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 32
    init_checkpoint: str = ""

    # Mode: "self_distill_ratio", "self_distill_additive", "dense_grpo"
    mode: str = "self_distill_ratio"

    ml_data: bool = True
    arxiv_data: bool = False
    pubmed_data: bool = False

    num_goals: int = 50
    start_goal: int = 0
    num_turns: int = 5
    group_size: int = 4  # samples per turn (dense_grpo only)

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

    # Self-distillation (setups 1 & 2)
    w_rl: float = 1.0
    w_sd: float = 1.0     # weight for self-distillation signal (setup 2 only)
    sd_clip: float = 5.0   # clip range for log-ratio

    # Running baseline (setups 1 & 2)
    baseline_decay: float = 0.9

    # Dense GRPO (setup 3)
    max_buffer_age: int = 3  # drop samples older than N turns
    staleness_threshold: float = 2.0  # drop if |log(π_new/π_old)| > this

    today_date: str = time.strftime("%Y-%m-%d", time.localtime())


def create_datum_custom(prompt_tokens, gen_tokens, hint_logprobs, base_logprobs, advantage):
    """Create training datum for custom self-distill ratio loss.

    Passes both hint-conditioned and base logprobs so the custom loss can
    compute the proper PPO ratio: π(t|goal+hint) / π(t|goal).
    """
    full_seq = prompt_tokens + gen_tokens
    ob_len = len(prompt_tokens) - 1
    input_tokens = full_seq[:-1]
    target_tokens = full_seq[1:]

    # Pad prompt portion
    all_hint_lps = [0.0] * ob_len + hint_logprobs
    all_base_lps = [0.0] * ob_len + base_logprobs
    all_advantages = [0.0] * ob_len + [advantage] * len(base_logprobs)
    # Mask: 1 for generated tokens, 0 for prompt
    mask = [0.0] * ob_len + [1.0] * len(base_logprobs)

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "hint_logprobs": TensorData.from_torch(torch.tensor(all_hint_lps, dtype=torch.float)),
            "base_logprobs": TensorData.from_torch(torch.tensor(all_base_lps, dtype=torch.float)),
            "advantages": TensorData.from_torch(torch.tensor(all_advantages, dtype=torch.float)),
            "mask": TensorData.from_torch(torch.tensor(mask, dtype=torch.float)),
        },
    )


def self_distill_ratio_loss(data, logprobs_list, clip_eps=0.2):
    """Custom loss: PPO with ratio = π(t|goal+hint) / π(t|goal).

    For each token:
      ratio = exp(hint_logprob - base_logprob)
      clipped_ratio = clip(ratio, 1-eps, 1+eps)
      loss = -min(ratio * advantage, clipped_ratio * advantage)

    This makes the model internalize hint-following behavior without
    seeing hints at inference time.
    """
    total_loss = torch.tensor(0.0)
    n_tokens = 0

    for datum, current_lps in zip(data, logprobs_list):
        hint_lps = datum.loss_fn_inputs["hint_logprobs"].to_torch()
        base_lps = datum.loss_fn_inputs["base_logprobs"].to_torch()
        advantages = datum.loss_fn_inputs["advantages"].to_torch()
        mask = datum.loss_fn_inputs["mask"].to_torch()

        # Ratio: π(t|goal+hint) / π(t|goal)
        log_ratio = hint_lps - base_lps
        ratio = torch.exp(log_ratio)

        # Clipped ratio (standard PPO clipping)
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

        # PPO surrogate loss
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        token_loss = -torch.min(surr1, surr2)

        # Apply mask (only generated tokens)
        masked_loss = (token_loss * mask).sum()
        total_loss = total_loss + masked_loss
        n_tokens += mask.sum().item()

    mean_loss = total_loss / max(n_tokens, 1)
    return mean_loss, {"self_distill_ratio_loss": mean_loss.item(), "n_tokens": n_tokens}


def main(config: Config):
    assert config.mode in ("self_distill_ratio", "self_distill_additive", "dense_grpo")

    run_dir = os.path.join(config.log_path, config.mode)
    os.makedirs(run_dir, exist_ok=True)
    train_dir = os.path.join(run_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    ml_log.setup_logging(log_dir=run_dir, wandb_project=None, wandb_name=None,
                         config=config, do_configure_logging_module=True)

    logger.info(f"IBT v2 mode: {config.mode}")
    logger.info(f"Policy: {config.policy_model}, Grader: {config.grader_model}")
    logger.info(f"Goals: {config.num_goals}, Turns: {config.num_turns}")
    logger.info(f"NO hints in model input — hints used for signals only")

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({k: getattr(config, k) for k in [
            "mode", "policy_model", "grader_model", "num_goals", "num_turns",
            "group_size", "learning_rate", "lora_rank", "w_rl", "w_sd", "sd_clip",
            "baseline_decay", "max_buffer_age",
        ]}, f, indent=2)

    # Tokenizer & renderer
    tokenizer = get_tokenizer(config.policy_model)
    renderer_name = model_info.get_recommended_renderer_name(config.policy_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    grader_tokenizer = get_tokenizer(config.grader_model)
    grader_renderer_name = model_info.get_recommended_renderer_name(config.grader_model)
    grader_renderer = renderers.get_renderer(grader_renderer_name, grader_tokenizer)

    # Dataset
    if config.ml_data:
        data = load_dataset("facebook/research-plan-gen", "ml")
    elif config.arxiv_data:
        data = load_dataset("facebook/research-plan-gen", "arxiv")
    elif config.pubmed_data:
        data = load_dataset("facebook/research-plan-gen", "pubmed")
    assert isinstance(data, datasets.DatasetDict)
    dataset = data["train"]
    num_goals = min(config.num_goals, len(dataset))

    # Clients
    service_client = create_service_client(base_url=config.base_url, api_profile=config.api_profile)

    if config.init_checkpoint:
        ckpt = checkpoint_utils.get_last_checkpoint(config.init_checkpoint)
        assert ckpt, f"No checkpoint at {config.init_checkpoint}"
        training_client = service_client.create_training_client_from_state_with_optimizer(ckpt["state_path"])
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.policy_model, rank=config.lora_rank)

    grader_client = service_client.create_sampling_client(base_model=config.grader_model)

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens, stop=renderer.get_stop_sequences(), temperature=config.temperature)
    grader_params = tinker.types.SamplingParams(
        max_tokens=config.grader_max_tokens, stop=grader_renderer.get_stop_sequences(),
        temperature=config.grader_temperature)
    adam_params = types.AdamParams(learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)

    # State
    running_baseline = None
    f_summary = open(os.path.join(train_dir, "batch_summary.jsonl"), "a")
    f_logs = open(os.path.join(train_dir, "training_logs.jsonl"), "a")
    global_step = config.start_goal * config.num_turns
    t_start = time.time()

    for goal_idx in range(config.start_goal, num_goals):
        goal_row = dataset[goal_idx]
        goal_text = goal_row["Goal"]
        rubric_items = goal_row["Rubric"]
        ref_solution = goal_row["Reference solution"]

        logger.info(f"\n{'='*60}")
        logger.info(f"GOAL {goal_idx}/{num_goals}: {goal_text[:100]}...")

        goal_turn_scores = []
        sample_buffer_goal = []  # for dense_grpo: this goal's buffer

        for turn_idx in range(config.num_turns):
            turn_t0 = time.time()
            logger.info(f"  Turn {turn_idx}/{config.num_turns}")

            # Save weights for sampling
            sr = training_client.save_weights_for_sampler(name=f"g{goal_idx:04d}_t{turn_idx:02d}").result()
            sampling_client = service_client.create_sampling_client(model_path=sr.path)

            # Build prompt — NO hints
            prompt_text = build_plan_prompt(scenario=goal_text, hint=None)
            convo = [{"role": "user", "content": prompt_text}]
            model_input = renderer.build_generation_prompt(convo)
            prompt_tokens = [int(t) for t in model_input.to_ints()]

            # Generate samples
            n_samples = config.group_size if config.mode == "dense_grpo" else 1
            sample_result = sampling_client.sample(
                prompt=model_input, num_samples=n_samples, sampling_params=sampling_params).result()

            # Parse plans & launch grading
            plans = []
            grader_futures = []
            for seq in sample_result.sequences:
                plan_text = renderers.get_text_content(renderer.parse_response(seq.tokens)[0])
                if "<solution>" in plan_text and "</solution>" not in plan_text:
                    plan_text = plan_text.rstrip() + "\n</solution>"
                plans.append({"text": plan_text, "tokens": seq.tokens, "logprobs": seq.logprobs})

                gp = build_grader_with_hint_prompt(
                    scenario=goal_text, rubric_items=rubric_items,
                    proposed_plan=plan_text, reference_solution=ref_solution)
                gi = grader_renderer.build_generation_prompt([{"role": "user", "content": gp}])
                grader_futures.append(grader_client.sample(gi, num_samples=1, sampling_params=grader_params))

            # Collect grades
            valid_plans = []
            for k, gf in enumerate(grader_futures):
                gr = gf.result()
                gt = renderers.get_text_content(grader_renderer.parse_response(gr.sequences[0].tokens)[0])
                rubric_score = compute_rubric_score(gt)
                reward, word_count = compute_reward(plans[k]["text"], rubric_score, config)
                hint = extract_hint(gt)

                if word_count < config.min_words:
                    continue

                valid_plans.append({
                    **plans[k], "reward": reward, "rubric_score": rubric_score,
                    "hint": hint, "word_count": word_count,
                })

            if not valid_plans:
                logger.warning(f"    No valid plans, skipping")
                continue

            rewards = [vp["reward"] for vp in valid_plans]
            rubric_scores = [vp["rubric_score"] for vp in valid_plans]
            turn_mean_rubric = float(np.mean(rubric_scores))
            goal_turn_scores.append(turn_mean_rubric)

            # ============================================================
            # MODE-SPECIFIC TRAINING
            # ============================================================
            training_datums = []
            turn_log = {"global_step": global_step, "goal_idx": goal_idx, "turn_idx": turn_idx,
                        "mode": config.mode, "rubric/mean": turn_mean_rubric,
                        "rubric/best": float(max(rubric_scores)),
                        "reward/mean": float(np.mean(rewards)),
                        "n_samples": len(valid_plans)}

            if config.mode in ("self_distill_ratio", "self_distill_additive"):
                # --- Self-distillation (1 sample, per-token advantages) ---
                vp = valid_plans[0]
                r = rewards[0]
                use_custom_loss = (config.mode == "self_distill_ratio")

                # Running baseline advantage
                if running_baseline is None:
                    running_baseline = r
                    scalar_adv = 0.0
                else:
                    scalar_adv = r - running_baseline
                running_baseline = config.baseline_decay * running_baseline + (1 - config.baseline_decay) * r

                # Compute hint-conditioned logprobs from SAME model
                hint = vp["hint"]
                sd_signal = None
                if hint and scalar_adv != 0.0:
                    hint_prompt = build_plan_prompt(scenario=goal_text, hint=hint)
                    hint_convo = [{"role": "user", "content": hint_prompt}]
                    hint_input = renderer.build_generation_prompt(hint_convo)
                    hint_prompt_tokens = [int(t) for t in hint_input.to_ints()]

                    full_seq = hint_prompt_tokens + [int(t) for t in vp["tokens"]]
                    try:
                        all_lps = sampling_client.compute_logprobs(
                            types.ModelInput.from_ints(tokens=full_seq)).result()
                        hint_lps = list(all_lps[len(hint_prompt_tokens):len(hint_prompt_tokens) + len(vp["tokens"])])
                        if len(hint_lps) < len(vp["tokens"]):
                            hint_lps.extend([0.0] * (len(vp["tokens"]) - len(hint_lps)))

                        base_lps = list(vp["logprobs"])
                        n = min(len(base_lps), len(hint_lps))
                        log_ratios = [hint_lps[i] - base_lps[i] for i in range(n)]
                        sd_signal = float(np.mean(np.abs(log_ratios)))

                        if use_custom_loss:
                            # Setup 1: custom loss with proper PPO ratio
                            datum = create_datum_custom(
                                prompt_tokens, [int(t) for t in vp["tokens"]],
                                hint_lps[:len(vp["tokens"])], base_lps, scalar_adv)
                        else:
                            # Setup 2: additive — w_rl × advantage + w_sd × signal
                            per_token_adv = []
                            for i in range(n):
                                sig = np.clip(log_ratios[i], -config.sd_clip, config.sd_clip)
                                per_token_adv.append(config.w_rl * scalar_adv + config.w_sd * sig)
                            for _ in range(len(base_lps) - n):
                                per_token_adv.append(config.w_rl * scalar_adv)
                            datum = create_datum(prompt_tokens, [int(t) for t in vp["tokens"]],
                                                base_lps, per_token_adv)

                        training_datums.append(datum)
                    except Exception as e:
                        logger.warning(f"    Self-distill logprobs failed: {e}, falling back to scalar PPO")
                        datum = create_datum(prompt_tokens, [int(t) for t in vp["tokens"]],
                                            vp["logprobs"], scalar_adv)
                        training_datums.append(datum)
                        use_custom_loss = False  # use standard PPO for this turn
                else:
                    # No hint available or zero advantage — use scalar PPO
                    datum = create_datum(prompt_tokens, [int(t) for t in vp["tokens"]],
                                        vp["logprobs"], scalar_adv)
                    training_datums.append(datum)
                    use_custom_loss = False

                turn_log["baseline"] = running_baseline
                turn_log["advantage"] = scalar_adv
                if sd_signal is not None:
                    turn_log["sd/mean_signal"] = sd_signal

                logger.info(f"    reward={r:.3f}, baseline={running_baseline:.3f}, "
                           f"adv={scalar_adv:.3f}" + (f", sd_signal={sd_signal:.3f}" if sd_signal else ""))

            elif config.mode == "dense_grpo":
                # --- Dense GRPO with sample accumulation ---
                # Add new samples to buffer
                for vp in valid_plans:
                    sample_buffer_goal.append({
                        "tokens": vp["tokens"],
                        "logprobs": vp["logprobs"],
                        "reward": vp["reward"],
                        "rubric_score": vp["rubric_score"],
                        "turn_idx": turn_idx,
                        "prompt_tokens": prompt_tokens,
                    })

                # Filter buffer: remove stale samples
                active_buffer = [s for s in sample_buffer_goal
                                 if turn_idx - s["turn_idx"] <= config.max_buffer_age]

                if len(active_buffer) < 2:
                    logger.info(f"    Buffer too small ({len(active_buffer)}), skipping update")
                    global_step += 1
                    continue

                # GRPO advantage over entire buffer
                buffer_rewards = [s["reward"] for s in active_buffer]
                mean_reward = np.mean(buffer_rewards)
                advantages = [r - mean_reward for r in buffer_rewards]

                if all(a == 0.0 for a in advantages):
                    logger.info(f"    All advantages zero, skipping")
                    global_step += 1
                    continue

                # Separate on-policy (current turn) and off-policy (old) samples
                on_policy = [(k, s) for k, s in enumerate(active_buffer) if turn_idx - s["turn_idx"] == 0]
                off_policy = [(k, s) for k, s in enumerate(active_buffer) if turn_idx - s["turn_idx"] > 0]

                # On-policy: use directly
                for k, sample in on_policy:
                    datum = create_datum(sample["prompt_tokens"],
                                       [int(t) for t in sample["tokens"]],
                                       sample["logprobs"], advantages[k])
                    training_datums.append(datum)

                # Off-policy: launch all logprob recomputations in parallel
                if off_policy:
                    lp_futures = {}
                    for k, sample in off_policy:
                        full_seq = sample["prompt_tokens"] + [int(t) for t in sample["tokens"]]
                        lp_futures[k] = (
                            sampling_client.compute_logprobs(types.ModelInput.from_ints(tokens=full_seq)),
                            sample,
                        )

                    for k, (future, sample) in lp_futures.items():
                        try:
                            current_lps = future.result()
                            plen = len(sample["prompt_tokens"])
                            new_lps = list(current_lps[plen:plen + len(sample["tokens"])])
                            old_lps = sample["logprobs"]
                            n = min(len(new_lps), len(old_lps))

                            mean_log_ratio = np.mean([abs(new_lps[i] - old_lps[i]) for i in range(n)])
                            if mean_log_ratio > config.staleness_threshold:
                                logger.debug(f"    Dropping stale sample (ratio={mean_log_ratio:.2f})")
                                continue

                            datum = create_datum(sample["prompt_tokens"],
                                               [int(t) for t in sample["tokens"]],
                                               new_lps[:len(sample["tokens"])], advantages[k])
                            training_datums.append(datum)
                        except Exception as e:
                            logger.debug(f"    Recompute failed: {e}")

                turn_log["buffer_size"] = len(active_buffer)
                turn_log["datums_used"] = len(training_datums)
                turn_log["advantage/mean"] = float(np.mean(advantages))

                logger.info(f"    Buffer={len(active_buffer)}, datums={len(training_datums)}, "
                           f"mean_reward={mean_reward:.3f}")

            # Training update
            if training_datums:
                try:
                    if config.mode == "self_distill_ratio":
                        # Setup 1: custom loss with proper PPO ratio
                        clip_eps = config.clip_eps
                        def _loss_fn(data, logprobs_list):
                            return self_distill_ratio_loss(data, logprobs_list, clip_eps)
                        training_client.forward_backward_custom(
                            training_datums, _loss_fn).result()
                    else:
                        # Setups 2 & 3: standard PPO with precomputed advantages
                        training_client.forward_backward(
                            training_datums, loss_fn="ppo",
                            loss_fn_config={"clip_low_threshold": 1 - config.clip_eps,
                                           "clip_high_threshold": 1 + config.clip_eps}).result()

                    training_client.optim_step(adam_params).result()
                    logger.info(f"    Update done ({len(training_datums)} datums, "
                               f"{'custom' if config.mode == 'self_distill_ratio' else 'ppo'})")
                except Exception as e:
                    logger.exception(f"    Training failed: {e}")

            turn_log["time"] = time.time() - turn_t0
            f_summary.write(json.dumps(turn_log) + "\n")

            for k, vp in enumerate(valid_plans):
                f_logs.write(json.dumps({
                    "global_step": global_step, "goal_idx": goal_idx, "turn_idx": turn_idx,
                    "sample_idx": k, "rubric_score": vp["rubric_score"], "reward": vp["reward"],
                    "word_count": vp["word_count"], "policy_output": vp["text"][:500],
                }) + "\n")

            global_step += 1

        # Goal summary
        if goal_turn_scores:
            logger.info(f"  Goal {goal_idx} done: {[f'{s:.3f}' for s in goal_turn_scores]}")
            goal_summary = {
                "type": "goal_summary", "goal_idx": goal_idx,
                "rubric_trajectory": goal_turn_scores,
                "rubric_first": goal_turn_scores[0],
                "rubric_last": goal_turn_scores[-1],
                "rubric_best": max(goal_turn_scores),
                "improvement": goal_turn_scores[-1] - goal_turn_scores[0],
                "total_steps": len(goal_turn_scores),
            }
            f_summary.write(json.dumps(goal_summary) + "\n")
            f_summary.flush()

    f_summary.close()
    f_logs.close()

    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=f"final_{num_goals}goals_{config.today_date}",
        log_path=run_dir, kind="state",
        loop_state={"goal": num_goals - 1, "global_step": global_step})
    logger.info(f"Done. {global_step} steps across {num_goals} goals.")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
