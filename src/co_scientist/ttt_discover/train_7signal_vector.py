"""Vector-advantage GRPO trainer with the 7-signal reward.

Phase A.1 validation trainer. Differences from best_ver.py (the rubric baseline):

  1. REWARD: uses src/co_scientist/trainers/_shared/seven_signal_reward.py
     instead of the rubric grader. Each rollout incurs 9 grader calls (3 for
     Signal 1.1 + 6 for the other signals).

  2. ADVANTAGE: per-signal group-relative advantage. For each signal k, compute
     advantage_k = (per_signal[k] - mean_k) / (std_k + eps) within the group,
     then total advantage = sum(weight_k * advantage_k). Hard gates zero out
     the per_signal vector for the affected sample.

  3. SCOPE: per-goal optimization. We train on ONE hardcoded research goal
     (matching the Phase A.0 sanity check goal) so that the validation result
     can be directly compared against bestversion on the same goal.

  4. CONFIG: batch_size and group_size are smaller because each rollout costs
     ~9 grader calls instead of 1. batch_size=4 / group_size=8 = 32 rollouts =
     288 grader calls per batch, comparable to bestversion's per-batch grader
     volume.

Usage:
    source tools/use_api_profile.sh new
    python src/co_scientist/trainers/grpo/train_7signal_vector.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.seven_signal_reward import (
    SIGNAL_1_1_SCORING_PROMPT,
    SIGNAL_1_2_EXTRACTION_PROMPT,
    SIGNAL_1_3_PROMPT,
    SIGNAL_2_1_PROMPT,
    SIGNAL_2_2_PROMPT,
    SIGNAL_2_3_PROMPT,
    SIGNAL_3_1_PROMPT,
    SIGNAL_NAMES,
    SIGNAL_WEIGHTS,
    HARD_GATES,
    SignalReward,
    aggregate_reward,
    extract_json,
    programmatic_claim_verification,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)


# =============================================================================
# Hardcoded per-goal training target (matches Phase A.0 sanity check)
# =============================================================================

TARGET_GOAL = (
    "Develop a multi-signal reward function for training an LLM co-scientist "
    "to generate research plans, where no ground truth exists and "
    "self-rewarding loops are vulnerable to collapse"
)

TARGET_TARGET = (
    "A reward function composed of multiple orthogonal signals, resistant to "
    "known hacking templates (generic restatement, reviewer cosplay, "
    "fabricated specifics), that can serve as a stable training signal for "
    "TTT-Discover-style per-goal optimization on Qwen3-30B-scale models"
)

ALT_GOALS = [
    (
        "Design an RLHF training method that reduces the reward hacking failure "
        "mode in code generation tasks using execution feedback as ground truth",
        "A training procedure that uses program execution outcomes as a verifiable "
        "reward signal to prevent reward hacking in LLM code generation, with "
        "demonstrated improvements on HumanEval and MBPP benchmarks",
    ),
    (
        "Develop a prompt engineering technique that improves zero-shot "
        "mathematical reasoning in small language models without fine-tuning",
        "A prompt construction method that lifts zero-shot accuracy on GSM8K and "
        "MATH benchmarks by >= 10 points on models under 7B parameters, using no "
        "training data beyond the prompt itself",
    ),
]


# =============================================================================
# Config
# =============================================================================


@chz.chz
class Config:
    base_url: str | None = None
    api_profile: str | None = "new"
    log_path: str = "/home/silas/co-scientist-project/runs/2026/4/grpo_7signal_vector/v1"
    model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"

    # Per-goal optimization: train on ONE goal repeatedly
    n_batches: int = 50

    # Reduced because each rollout costs 9 grader calls instead of 1
    batch_size: int = 4   # 4 "duplicate" goals per batch (all the same TARGET_GOAL)
    group_size: int = 8   # 8 rollouts per duplicate
    learning_rate: float = 1e-5
    clip_eps: float = 0.2

    max_length: int = 32768
    lora_rank: int = 64
    save_every: int = 10
    max_tokens: int = 2048
    grader_max_tokens: int = 8192
    temperature: float = 1.0
    grader_temperature: float = 0.0

    # Vector advantage normalization
    advantage_eps: float = 1e-6  # epsilon for std normalization

    # Length / format constraints (kept for parity with best_ver)
    max_word_count: int = 750
    target_word_count: int = 600
    min_words: int = 30

    today_date: str = time.strftime("%Y-%m-%d", time.localtime())


# =============================================================================
# Plan generation prompt (parity with best_ver.py)
# =============================================================================


def build_research_plan_prompt(scenario: str) -> str:
    prompt = textwrap.dedent(f"""
        I will provide you a research scenario. You have to provide me a concise yet thoughtful research plan with all details needed to execute it.
    """).strip()

    prompt += f"""
            Here is the research scenario.
            Scenario: {scenario}
            """

    prompt += textwrap.dedent(f"""
        Here is the research scenario.
        Scenario: {scenario}

        # Instructions
        First, come up with a detailed research plan to address the scenario based on the following overall solution guidelines:
        - The plan should address the goals of the scenario, and account for all constraints and confounders.
        - Do NOT just say WHAT you will do. Explain HOW you will do it and WHY it is needed. Provide clear explanation and justification for each proposed step. The solution inside <solution></solution> tags should be readable for humans, and not in XML itself.
        - The phrasing should NOT be verbose, and NOT be in past tense, as in "the author's approach" but rather in present tense, as how you would approach the problem.
        - Do not claim to have done any experiments or have results, just provide the plan.
        - Do not add self-proclaimed praises of your solution. For example do NOT say yourself it satisfies some desiderata, we will let the evaluator decide that.

        Then, return the following nested XML block as your output (always close opened XML tags):
        <think>
        ...your reasoning...
        </think>
        <solution>
        ...your detailed research plan...
        </solution>
    """).strip()

    return prompt


# =============================================================================
# Grader call helpers (async, launch-then-collect pattern)
# =============================================================================


def launch_signal_call(grader_client, renderer, prompt: str, max_tokens: int, temperature: float):
    """Launch one grader sample call. Returns the future without waiting."""
    grader_input = renderer.build_generation_prompt([{"role": "user", "content": prompt}])
    return grader_client.sample(
        grader_input,
        num_samples=1,
        sampling_params=tinker.types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=renderer.get_stop_sequences(),
        ),
    )


def collect_decode(future, tokenizer) -> str:
    """Wait for one grader future and decode tokens to text."""
    result = future.result()
    tokens = result.sequences[0].tokens
    return tokenizer.decode(tokens)


def parse_score(text: str) -> tuple[float, dict | None]:
    """Extract JSON and pull score. Returns (score, parsed_dict_or_None)."""
    parsed = extract_json(text)
    if parsed is None:
        return 0.0, None
    raw_score = parsed.get("score", 0.0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(1.0, score)), parsed


def launch_seven_signal_grader_calls(
    grader_client,
    renderer,
    plan: str,
    target_goal: str,
    target_target: str,
    alt_goals: list[tuple[str, str]],
    max_tokens: int,
    temperature: float,
) -> dict:
    """Launch all 9 grader futures for a single plan. Does NOT wait."""
    futures = {
        "1.1_target": launch_signal_call(grader_client, renderer,
            SIGNAL_1_1_SCORING_PROMPT.format(goal=target_goal, target=target_target, plan=plan),
            max_tokens, temperature),
        "1.1_alt1": launch_signal_call(grader_client, renderer,
            SIGNAL_1_1_SCORING_PROMPT.format(goal=alt_goals[0][0], target=alt_goals[0][1], plan=plan),
            max_tokens, temperature),
        "1.1_alt2": launch_signal_call(grader_client, renderer,
            SIGNAL_1_1_SCORING_PROMPT.format(goal=alt_goals[1][0], target=alt_goals[1][1], plan=plan),
            max_tokens, temperature),
        "1.2": launch_signal_call(grader_client, renderer,
            SIGNAL_1_2_EXTRACTION_PROMPT.format(goal=target_goal, plan=plan),
            max_tokens, temperature),
        "1.3": launch_signal_call(grader_client, renderer,
            SIGNAL_1_3_PROMPT.format(plan=plan),
            max_tokens, temperature),
        "2.1": launch_signal_call(grader_client, renderer,
            SIGNAL_2_1_PROMPT.format(goal=target_goal, target=target_target, plan=plan),
            max_tokens, temperature),
        "2.2": launch_signal_call(grader_client, renderer,
            SIGNAL_2_2_PROMPT.format(goal=target_goal, plan=plan),
            max_tokens, temperature),
        "2.3": launch_signal_call(grader_client, renderer,
            SIGNAL_2_3_PROMPT.format(plan=plan),
            max_tokens, temperature),
        "3.1": launch_signal_call(grader_client, renderer,
            SIGNAL_3_1_PROMPT.format(goal=target_goal, plan=plan),
            max_tokens, temperature),
    }
    return futures


def collect_seven_signal_reward(futures: dict, tokenizer) -> SignalReward:
    """Wait for all 9 futures, parse, and produce a SignalReward."""
    per_signal: dict[str, float] = {}
    diagnostic: dict = {}
    raw_outputs: dict = {}

    # Signal 1.1: Goal-Contrast Margin
    target_text = collect_decode(futures["1.1_target"], tokenizer)
    alt1_text = collect_decode(futures["1.1_alt1"], tokenizer)
    alt2_text = collect_decode(futures["1.1_alt2"], tokenizer)
    target_score, _ = parse_score(target_text)
    alt1_score, _ = parse_score(alt1_text)
    alt2_score, _ = parse_score(alt2_text)
    margin = target_score - 0.5 * (alt1_score + alt2_score)
    if margin <= 0:
        signal_1_1 = 0.0
    else:
        signal_1_1 = max(0.0, min(1.0, min(target_score, 2.0 * margin)))
    per_signal["goal_contrast_margin"] = signal_1_1
    diagnostic["1.1"] = {"target": target_score, "alt1": alt1_score, "alt2": alt2_score, "margin": margin}

    # Signal 1.2: Claim Verification
    cv_text = collect_decode(futures["1.2"], tokenizer)
    cv_parsed = extract_json(cv_text)
    cv_score, cv_diag = programmatic_claim_verification(cv_parsed)
    per_signal["claim_verification"] = cv_score
    diagnostic["1.2"] = cv_diag

    # Signal 1.3: Internal Consistency
    ic_text = collect_decode(futures["1.3"], tokenizer)
    ic_score, ic_parsed = parse_score(ic_text)
    per_signal["internal_consistency"] = ic_score
    diagnostic["1.3"] = {"score": ic_score, "parsed": (ic_parsed or {})}

    # Signal 2.1: Methodological Soundness
    ms_text = collect_decode(futures["2.1"], tokenizer)
    ms_score, ms_parsed = parse_score(ms_text)
    per_signal["soundness"] = ms_score
    diagnostic["2.1"] = {"score": ms_score, "classification": (ms_parsed or {}).get("classification")}

    # Signal 2.2: Feasibility
    fe_text = collect_decode(futures["2.2"], tokenizer)
    fe_score, _ = parse_score(fe_text)
    per_signal["feasibility"] = fe_score

    # Signal 2.3: Specificity
    sp_text = collect_decode(futures["2.3"], tokenizer)
    sp_score, _ = parse_score(sp_text)
    per_signal["specificity"] = sp_score

    # Signal 3.1: Coherence
    co_text = collect_decode(futures["3.1"], tokenizer)
    co_score, _ = parse_score(co_text)
    per_signal["coherence"] = co_score

    aggregate, agg_diag = aggregate_reward(per_signal)
    hard_gate_triggered = "hard_gate_failed" in agg_diag
    hard_gate_name = agg_diag.get("hard_gate_failed")

    return SignalReward(
        aggregate=aggregate,
        per_signal=per_signal,
        hard_gate_triggered=hard_gate_triggered,
        hard_gate_name=hard_gate_name,
        diagnostic=diagnostic,
        raw_outputs=raw_outputs,
    )


# =============================================================================
# Vector advantage computation
# =============================================================================


def compute_vector_advantages(
    group_rewards: list[SignalReward],
    eps: float,
) -> tuple[list[float], dict]:
    """Per-signal group-relative advantage, weighted sum.

    For each signal k:
        adv_k_i = (s_k_i - mean_k) / (std_k + eps)
    Total advantage for sample i:
        A_i = sum_k weight_k * adv_k_i

    Hard-gate-triggered samples have their per_signal vector zeroed before
    computing group statistics, which makes them get the worst possible
    advantage in every signal where the others scored higher.

    Returns (per_sample_total_advantage, signal_diagnostic).
    """
    n = len(group_rewards)

    # Stack per-signal scores into a (n, 7) matrix, applying hard-gate zeroing
    score_matrix = np.zeros((n, len(SIGNAL_NAMES)), dtype=np.float64)
    for i, reward in enumerate(group_rewards):
        if reward.hard_gate_triggered:
            score_matrix[i, :] = 0.0
        else:
            for k, name in enumerate(SIGNAL_NAMES):
                score_matrix[i, k] = reward.per_signal[name]

    # Per-signal group statistics
    means = score_matrix.mean(axis=0)
    stds = score_matrix.std(axis=0)

    # Per-signal advantages: (n, 7)
    advantage_matrix = (score_matrix - means[None, :]) / (stds[None, :] + eps)

    # Weighted sum across signals
    weights = np.array([SIGNAL_WEIGHTS[name] for name in SIGNAL_NAMES])
    total_advantages = (advantage_matrix * weights[None, :]).sum(axis=1).tolist()

    diagnostic = {
        "per_signal_means": dict(zip(SIGNAL_NAMES, means.tolist())),
        "per_signal_stds": dict(zip(SIGNAL_NAMES, stds.tolist())),
        "per_signal_advantage_mean": dict(
            zip(SIGNAL_NAMES, advantage_matrix.mean(axis=0).tolist())
        ),
        "per_signal_advantage_std": dict(
            zip(SIGNAL_NAMES, advantage_matrix.std(axis=0).tolist())
        ),
        "n_hard_gate_triggered": sum(1 for r in group_rewards if r.hard_gate_triggered),
    }

    return total_advantages, diagnostic


# =============================================================================
# Format check helper (parity with best_ver.py)
# =============================================================================


def check_format_compliance(text: str, max_words: int) -> bool:
    """Check whether the generated text has <solution> tags and is within length."""
    has_open = "<solution>" in text
    has_close = "</solution>" in text
    word_count = len(text.strip().split())
    return has_open and has_close and word_count <= max_words


# =============================================================================
# Main training loop
# =============================================================================


def main(config: Config):
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    os.makedirs(config.log_path, exist_ok=True)

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    service_client = create_service_client(
        base_url=config.base_url,
        api_profile=config.api_profile,
    )

    # Load checkpoint if exists
    last_checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path)
    if last_checkpoint is None:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0
    else:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            last_checkpoint["state_path"]
        )
        start_batch = last_checkpoint["batch"] + 1
        logger.info(f"Resuming from batch {start_batch}")

    grader_client = service_client.create_sampling_client(
        base_model=config.grader_model_name
    )

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )

    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    logger.info(f"Per-goal training: {TARGET_GOAL[:80]}...")
    logger.info(f"Training for {config.n_batches} batches, batch_size={config.batch_size}, group_size={config.group_size}")

    for batch_idx in range(start_batch, config.n_batches):
        t_start = time.time()
        real_batch = batch_idx

        # Save checkpoint periodically
        if config.save_every > 0 and batch_idx % config.save_every == 0 and batch_idx > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{real_batch:06d}_{config.today_date}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": real_batch},
            )

        # Save current weights for sampler
        sampling_result = training_client.save_weights_for_sampler(
            name=f"{real_batch:06d}"
        ).result()
        sampling_path = sampling_result.path
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)

        # === PHASE 1: launch policy generations ===
        # All batch_size "duplicates" generate from the same goal
        prompt_text = build_research_plan_prompt(scenario=TARGET_GOAL)
        convo = [{"role": "user", "content": prompt_text}]
        model_input = renderer.build_generation_prompt(convo)
        prompt_tokens_ints = model_input.to_ints()

        policy_futures = []
        for goal_idx in range(config.batch_size):
            policy_futures.append(
                sampling_client.sample(
                    prompt=model_input,
                    num_samples=config.group_size,
                    sampling_params=sampling_params,
                )
            )

        logger.info(f"Batch {real_batch}: launched {config.batch_size} policy futures")

        # === PHASE 2: collect plans, launch all 9 grader futures per sample ===
        # We launch ALL grader calls before collecting any, for maximum parallelism.
        all_groups = []  # list of {"samples_info": [...], "grader_futures": [...]}

        for i, p_future in enumerate(policy_futures):
            result = p_future.result()
            group_samples_info = []
            group_grader_futures = []

            for group_result in result.sequences:
                proposed_plan = renderer.parse_response(group_result.tokens)[0]["content"]
                if "<solution>" in proposed_plan and "</solution>" not in proposed_plan:
                    proposed_plan = proposed_plan.rstrip() + "\n</solution>"

                # Launch all 9 grader calls for this sample
                grader_futures = launch_seven_signal_grader_calls(
                    grader_client=grader_client,
                    renderer=renderer,
                    plan=proposed_plan,
                    target_goal=TARGET_GOAL,
                    target_target=TARGET_TARGET,
                    alt_goals=ALT_GOALS,
                    max_tokens=config.grader_max_tokens,
                    temperature=config.grader_temperature,
                )

                group_samples_info.append({
                    "tokens": group_result.tokens,
                    "logprobs": group_result.logprobs,
                    "text": proposed_plan,
                })
                group_grader_futures.append(grader_futures)

            all_groups.append({
                "samples_info": group_samples_info,
                "grader_futures": group_grader_futures,
            })

        total_grader_calls = sum(len(g["grader_futures"]) * 9 for g in all_groups)
        logger.info(f"Batch {real_batch}: launched {total_grader_calls} grader futures")

        # === PHASE 3: collect grader results, compute SignalReward, compute vector advantages ===
        training_datums = []
        batch_logs = []
        batch_aggregate_scores = []
        batch_per_signal_scores: dict[str, list[float]] = {name: [] for name in SIGNAL_NAMES}
        batch_hard_gate_count = 0
        batch_advantages = []
        n_total_samples = 0
        n_valid_samples = 0
        n_dropped_samples = 0

        for group_idx, group in enumerate(all_groups):
            group_rewards: list[SignalReward] = []
            group_valid_indices = []

            for j, sample_info in enumerate(group["samples_info"]):
                n_total_samples += 1
                plan_text = sample_info["text"]
                word_count = len(plan_text.strip().split())

                # Drop degenerate samples (too short or only tags)
                if word_count < config.min_words:
                    n_dropped_samples += 1
                    group_rewards.append(None)
                    continue
                if plan_text.strip() in {"<think>", "</think>", "<solution>", "</solution>"}:
                    n_dropped_samples += 1
                    group_rewards.append(None)
                    continue

                # Collect grader results and compute SignalReward
                reward = collect_seven_signal_reward(group["grader_futures"][j], tokenizer)
                group_rewards.append(reward)
                group_valid_indices.append(j)
                n_valid_samples += 1

                # Track batch-level stats
                batch_aggregate_scores.append(reward.aggregate)
                for name in SIGNAL_NAMES:
                    batch_per_signal_scores[name].append(reward.per_signal[name])
                if reward.hard_gate_triggered:
                    batch_hard_gate_count += 1

            # Need at least 2 valid samples in the group for advantage normalization
            valid_rewards = [r for r in group_rewards if r is not None]
            if len(valid_rewards) < 2:
                logger.debug(f"Group {group_idx}: skipping (only {len(valid_rewards)} valid samples)")
                continue

            # Compute per-signal vector advantages
            valid_advantages, signal_diag = compute_vector_advantages(valid_rewards, config.advantage_eps)
            batch_advantages.extend(valid_advantages)

            # Skip if no learning signal (all advantages zero)
            if all(abs(a) < 1e-9 for a in valid_advantages):
                logger.debug(f"Group {group_idx}: skipping (no advantage signal)")
                continue

            # Build training datums for valid samples
            for k, sample_idx in enumerate(group_valid_indices):
                sample_info = group["samples_info"][sample_idx]
                advantage = valid_advantages[k]

                prompt_tokens = [int(t) for t in prompt_tokens_ints]
                generated_tokens = [int(t) for t in sample_info["tokens"]]
                full_seq = prompt_tokens + generated_tokens
                ob_len = len(prompt_tokens) - 1

                input_tokens = full_seq[:-1]
                target_tokens = full_seq[1:]
                all_logprobs = [0.0] * ob_len + sample_info["logprobs"]
                all_advantages = [0.0] * ob_len + [advantage] * len(sample_info["logprobs"])

                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
                        "logprobs": TensorData.from_torch(torch.tensor(all_logprobs, dtype=torch.float)),
                        "advantages": TensorData.from_torch(torch.tensor(all_advantages, dtype=torch.float)),
                    },
                )
                training_datums.append(datum)

                # Per-sample log
                reward = valid_rewards[k]
                batch_logs.append({
                    "batch_idx": real_batch,
                    "group_idx": group_idx,
                    "sample_idx": sample_idx,
                    "policy_output": plan_text,
                    "aggregate": reward.aggregate,
                    "per_signal": reward.per_signal,
                    "hard_gate_triggered": reward.hard_gate_triggered,
                    "hard_gate_name": reward.hard_gate_name,
                    "advantage": advantage,
                    "word_count": len(plan_text.strip().split()),
                })

        # === PHASE 4: optimization step ===
        if not training_datums:
            logger.warning(f"Batch {real_batch}: no valid datums, skipping optimization")
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
            _ = fwd_bwd_future.result()
            _ = optim_step_future.result()
        except Exception:
            logger.exception(f"Batch {real_batch}: training step failed")
            continue

        # === PHASE 5: log batch summary ===
        batch_summary = {
            "batch_idx": real_batch,
            "time/total": time.time() - t_start,
            "samples/total": n_total_samples,
            "samples/valid": n_valid_samples,
            "samples/dropped": n_dropped_samples,
            "samples/hard_gate_triggered": batch_hard_gate_count,
            "aggregate/mean": float(np.mean(batch_aggregate_scores)) if batch_aggregate_scores else 0.0,
            "aggregate/std": float(np.std(batch_aggregate_scores)) if batch_aggregate_scores else 0.0,
            "advantage/mean": float(np.mean(batch_advantages)) if batch_advantages else 0.0,
            "advantage/std": float(np.std(batch_advantages)) if batch_advantages else 0.0,
        }
        for name in SIGNAL_NAMES:
            scores = batch_per_signal_scores[name]
            if scores:
                batch_summary[f"signal/{name}/mean"] = float(np.mean(scores))
                batch_summary[f"signal/{name}/std"] = float(np.std(scores))

        # Persist logs
        train_log_path = os.path.join(config.log_path, "train", "training_logs.jsonl")
        os.makedirs(os.path.dirname(train_log_path), exist_ok=True)
        with open(train_log_path, "a") as f:
            for log_item in batch_logs:
                f.write(json.dumps(log_item) + "\n")

        summary_path = os.path.join(config.log_path, "train", "batch_summary.jsonl")
        with open(summary_path, "a") as f:
            f.write(json.dumps(batch_summary) + "\n")

        ml_logger.log_metrics(batch_summary, step=real_batch)
        logger.info(
            f"Batch {real_batch}: agg={batch_summary['aggregate/mean']:.3f} "
            f"(std {batch_summary['aggregate/std']:.3f}), "
            f"valid={n_valid_samples}/{n_total_samples}, "
            f"hard_gate={batch_hard_gate_count}, "
            f"time={batch_summary['time/total']:.1f}s"
        )

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
    chz.entrypoint(main)
