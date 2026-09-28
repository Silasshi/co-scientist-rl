"""D6 GRPO trainer v1 — group-relative policy optimization for grant proposals.

Reward-only training (no privileged Opus reviewer). Uses D4's 12-signal grant
reward as the group-level reward signal. Advantage = (R_i - mean(R)) / (std(R) + 1e-8).

Scientific purpose: isolates the value of the Opus reviewer signal in OPD.
  GRPO ≪ OPD → reviewer signal is load-bearing for D6 quality improvement
  GRPO ≈ OPD → reward-only optimization is sufficient; reviewer adds overhead

Architecture differences from OPD:
  - No teacher model (no reference distribution, no critic conditioning)
  - No frozen base client (no trust-region)
  - No Opus reviewer (no build_critic_request_payload)
  - Reward: 12-signal grant weighted mean via frozen Qwen3-30B-A3B grader
  - Advantage: group-relative scalar per plan (not per-token teacher_lp - student_lp)
  - With solution_only_mask=True for comparability with OPD

Adapted from:
  D1: src/co_scientist/rubric_reward/grpo/best_ver.py (advantage + datum building)
  D6: src/co_scientist/d6_grant_proposal/train_opd.py (infrastructure)

Usage:
    source shared/tools/use_api_profile.sh new
    PYTHONPATH=src python -m co_scientist.d6_grant_proposal.train_grpo \\
        goal_domain=ai goal_name=02_foundational_rl \\
        log_path=projects/d6_grant_proposal/runs/2026_04_30_pilot_foundrl_grpo \\
        n_iter=16
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
import tinker
from tinker import types
from tinker.types import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.grant_signal_reward import (
    SIGNALS, SIGNAL_VARIANTS, aggregate_reward, build_single_call_prompt, parse_scores,
)
from co_scientist.d6_grant_proposal.prompts import (
    build_student_prompt, extract_solution,
)
from co_scientist.d6_grant_proposal.train_opd import (
    find_solution_content_span,
    build_solution_token_mask_from_tokens,
    _safe_pos_frac,
    _safe_mean_abs,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _goal_dir(config) -> Path:
    return _resolve(f"{config.dataset_base}/{config.goal_domain}/{config.goal_name}")


def _oracle_path(config) -> Path:
    return _resolve(
        f"{config.data_base}/{config.goal_domain}/{config.goal_name}/oracle/slim.md"
    )


def _active_signals(goal_domain: str) -> list:
    non_stem = {"social_science"}
    if goal_domain in non_stem:
        return [
            SIGNAL_VARIANTS["G12a_analytical_framework"]
            if s.id == "G12_formalism" else s
            for s in SIGNALS
        ]
    return list(SIGNALS)


@chz.chz
class Config:
    api_profile: str | None = "new"
    base_url: str | None = None

    goal_domain: str = "ai"
    goal_name: str = "02_foundational_rl"

    log_path: str = "projects/d6_grant_proposal/runs/2026_04_30_pilot_foundrl_grpo"
    dataset_base: str = "projects/d6_grant_proposal/dataset"
    data_base: str = "projects/d6_grant_proposal/data"

    model_name: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 64
    learning_rate: float = 5e-5

    # GRPO group size
    n_rollouts_per_group: int = 4
    max_tokens: int = 8192
    temperature: float = 1.0
    top_p: float = 0.95

    # Grader settings (frozen Qwen3-30B-A3B)
    grader_max_tokens: int = 8192
    grader_temperature: float = 0.0

    # Advantage clipping
    reward_clip: float = 5.0
    adv_norm_eps: float = 1e-8

    # Training
    n_iter: int = 16
    eval_every: int = 1
    n_eval_plans: int = 8
    save_every: int = 1
    n_grad_steps_per_iter: int = 4

    # OPD-compat knobs (kept for comparability)
    solution_only_mask: bool = True
    ppo_clip_eps: float = 0.2
    loss_fn_name: Literal["importance_sampling", "ppo"] = "ppo"

    # Continual / resume
    init_state_path: str = ""
    reset_optimizer_state: bool = False

    today_date: str = "2026_04_30"
    seed: int = 42


def _grade_plan(
    plan_text: str,
    goal: str,
    active_signals: list,
    grader_client,
    renderer,
    tokenizer,
    grader_params,
) -> float:
    """Call frozen Qwen3 grader and return aggregate_reward (0-1)."""
    grader_prompt_str = build_single_call_prompt(goal, plan_text[:15000], active_signals)
    grader_convo = [{"role": "user", "content": grader_prompt_str}]
    grader_input = renderer.build_generation_prompt(grader_convo)
    try:
        grader_result = grader_client.sample(
            grader_input, num_samples=1, sampling_params=grader_params,
        ).result()
        grader_seq = grader_result.sequences[0]
        grader_parsed = renderer.parse_response(grader_seq.tokens)
        grader_text = grader_parsed[0].get("content", "") if grader_parsed else tokenizer.decode(grader_seq.tokens)
        scores_parsed = parse_scores(grader_text)
        signal_scores = {sid: d["score"] for sid, d in scores_parsed.items()}
        return aggregate_reward(signal_scores)
    except Exception as e:
        logger.warning("Grading failed: %s", e)
        return 0.0


def main(config: Config):
    log_dir = _resolve(config.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    ml_log.setup_logging(
        log_dir=str(log_dir), wandb_project=None, wandb_name=None,
        config=config, do_configure_logging_module=True,
    )

    goal_dir = _goal_dir(config)
    goal = (goal_dir / "research_goal.md").read_text().strip()

    oracle_path = _oracle_path(config)
    if not oracle_path.exists():
        raise FileNotFoundError(
            f"Slim oracle not found at {oracle_path}. "
            "Run extract_oracle.py and extract_oracle_slim.py first."
        )
    oracle = oracle_path.read_text().strip()

    logger.info("D6 GRPO-v1 training")
    logger.info("  goal_domain: %s, goal_name: %s", config.goal_domain, config.goal_name)
    logger.info("  n_rollouts_per_group: %d, n_iter: %d", config.n_rollouts_per_group, config.n_iter)
    logger.info("  solution_only_mask: %s, ppo_clip_eps: %.2f", config.solution_only_mask, config.ppo_clip_eps)
    logger.info("  [NO Opus reviewer — reward-only GRPO]")

    (log_dir / "config.json").write_text(json.dumps(
        {k: getattr(config, k) for k in dir(config)
         if not k.startswith("_") and not callable(getattr(config, k))},
        default=str, indent=2
    ))

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )

    # Frozen grader client (base model, no LoRA)
    grader_client = service_client.create_sampling_client(base_model=config.model_name)

    # Training client
    last_checkpoint = checkpoint_utils.get_last_checkpoint(str(log_dir))
    if last_checkpoint is not None:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            last_checkpoint["state_path"]
        )
        start_iter = last_checkpoint.get("batch", 0) + 1
        logger.info("Resuming from iter %d", start_iter)
    elif config.init_state_path:
        if config.reset_optimizer_state:
            training_client = service_client.create_training_client_from_state(config.init_state_path)
        else:
            training_client = service_client.create_training_client_from_state_with_optimizer(
                config.init_state_path
            )
        start_iter = 0
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank,
        )
        start_iter = 0
        logger.info("Fresh LoRA training client (rank=%d)", config.lora_rank)

    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8,
    )
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=renderer.get_stop_sequences(),
    )
    grader_params = tinker.types.SamplingParams(
        max_tokens=config.grader_max_tokens,
        temperature=config.grader_temperature,
        stop=renderer.get_stop_sequences(),
    )

    active_signals = _active_signals(config.goal_domain)

    student_text = build_student_prompt(goal, oracle)
    student_convo = [{"role": "user", "content": student_text}]
    student_input = renderer.build_generation_prompt(student_convo)
    student_tokens_prefix = student_input.to_ints()

    buffer_path = log_dir / "buffer.jsonl"
    metrics_path = log_dir / "metrics.jsonl"
    checkpoints_path = log_dir / "checkpoints.jsonl"
    eval_path = log_dir / "eval_rollouts.jsonl"

    # ===========================================================================
    # Main training loop
    # ===========================================================================
    for iter_idx in range(start_iter, config.n_iter):
        t0 = time.time()

        if config.save_every > 0 and iter_idx % config.save_every == 0 and iter_idx > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{iter_idx:06d}_{config.today_date}",
                log_path=str(log_dir),
                kind="state",
                loop_state={"batch": iter_idx},
            )
            logger.info("Saved checkpoint at iter %d", iter_idx)

        sp_path = training_client.save_weights_for_sampler(
            name=f"iter_{iter_idx:04d}"
        ).result().path
        sampling_client = service_client.create_sampling_client(model_path=sp_path)
        with open(checkpoints_path, "a") as f:
            f.write(json.dumps({"iter": iter_idx, "sampler_path": sp_path,
                                 "ts": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")

        # ----- SAMPLE n_rollouts_per_group plans -----
        t_sample0 = time.time()
        result = sampling_client.sample(
            prompt=student_input,
            num_samples=config.n_rollouts_per_group,
            sampling_params=sampling_params,
        ).result()
        t_sample = time.time() - t_sample0
        logger.info("Iter %d: sampled %d plans in %.1fs",
                     iter_idx, len(result.sequences), t_sample)

        # ----- DECODE plans -----
        decoded_plans = []
        for k, seq in enumerate(result.sequences):
            try:
                parsed = renderer.parse_response(seq.tokens)
                raw = parsed[0].get("content", "") if parsed else tokenizer.decode(seq.tokens)
            except Exception:
                raw = tokenizer.decode(seq.tokens)
            decoded_plans.append(extract_solution(raw))

        # ----- GRADE plans → rewards -----
        t_grade0 = time.time()
        rewards = []
        for k, plan_text in enumerate(decoded_plans):
            if not plan_text:
                rewards.append(0.0)
                continue
            r = _grade_plan(plan_text, goal, active_signals, grader_client, renderer, tokenizer, grader_params)
            rewards.append(r)
            logger.debug("Iter %d plan %d: reward=%.4f", iter_idx, k, r)
        t_grade = time.time() - t_grade0
        logger.info("Iter %d: graded %d plans in %.1fs, mean_reward=%.4f",
                     iter_idx, len(rewards), t_grade, float(np.mean(rewards)) if rewards else 0.0)

        # ----- GRPO ADVANTAGE (from D1 best_ver.py:692-709) -----
        if not rewards or all(r == 0.0 for r in rewards):
            logger.warning("Iter %d: all rewards 0 or empty; skipping update", iter_idx)
            continue

        mean_r = float(np.mean(rewards))
        std_r = float(np.std(rewards)) + config.adv_norm_eps
        advantages_scalar = [(r - mean_r) / std_r for r in rewards]

        if all(a == 0.0 for a in advantages_scalar):
            logger.info("Iter %d: all advantages 0 (identical rewards); skipping update", iter_idx)
            continue

        # ----- BUILD DATUMS -----
        datums = []
        all_advs_flat = []
        per_plan_records = []

        for k, seq in enumerate(result.sequences):
            gen = list(seq.tokens)
            s_lp = list(seq.logprobs) if seq.logprobs else []
            adv_scalar = advantages_scalar[k]

            if not gen or len(s_lp) != len(gen):
                logger.warning(
                    "Iter %d plan %d: len mismatch gen=%d s_lp=%d; skipping",
                    iter_idx, k, len(gen), len(s_lp),
                )
                continue

            # Per-token advantage: same scalar for all generated tokens (GRPO standard)
            # Clipped to reward_clip for stability
            clipped_adv = max(-config.reward_clip, min(config.reward_clip, adv_scalar))
            per_tok_adv = [clipped_adv] * len(gen)

            # Solution-only mask (same as OPD, for comparability)
            n_mask_active = len(per_tok_adv)
            mask_density = 1.0
            if config.solution_only_mask:
                sol_mask = build_solution_token_mask_from_tokens(
                    tokenizer=tokenizer, sample_tokens=gen,
                )
                if sol_mask.size == len(per_tok_adv):
                    per_tok_adv = [a * float(m) for a, m in zip(per_tok_adv, sol_mask.tolist())]
                    n_mask_active = int(sol_mask.sum())
                    mask_density = float(n_mask_active) / max(1, len(gen))

            all_advs_flat.extend(per_tok_adv)

            # Build datum (same structure as D1 best_ver.py + D6 OPD)
            prompt_len = len(student_tokens_prefix)
            ob_len = prompt_len - 1
            input_tokens = student_tokens_prefix[:-1] + gen[:-1]
            target_tokens = student_tokens_prefix[1:] + gen[1:]  # not actually needed but matches D1 pattern
            all_logprobs = [0.0] * ob_len + s_lp
            all_advantages_tok = [0.0] * ob_len + per_tok_adv

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens=student_tokens_prefix + gen[:-1]),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(
                        __import__("torch").tensor(student_tokens_prefix[1:] + gen, dtype=__import__("torch").long)
                    ),
                    "logprobs": TensorData.from_torch(
                        __import__("torch").tensor([0.0] * len(student_tokens_prefix) + s_lp, dtype=__import__("torch").float)
                    ),
                    "advantages": TensorData.from_torch(
                        __import__("torch").tensor([0.0] * len(student_tokens_prefix) + per_tok_adv, dtype=__import__("torch").float)
                    ),
                },
            )
            datums.append(datum)
            per_plan_records.append({
                "iter": iter_idx, "plan_idx": k,
                "n_gen_tokens": len(gen),
                "reward": rewards[k],
                "adv_scalar": adv_scalar,
                "mean_adv": float(np.mean(per_tok_adv)),
                "stop_reason": seq.stop_reason,
                "plan_text": decoded_plans[k] if k < len(decoded_plans) else "",
                "solution_mask_active_tokens": n_mask_active,
                "solution_mask_density": mask_density,
                "sampled_by": "grpo",
            })

        # ----- TRAIN STEP -----
        t_step0 = time.time()
        if datums:
            n_steps = max(1, config.n_grad_steps_per_iter)
            chunk_size = max(1, len(datums) // n_steps)
            chunks = [datums[i:i + chunk_size] for i in range(0, len(datums), chunk_size)]
            loss_fn_kwargs: dict = {"loss_fn": config.loss_fn_name}
            if config.loss_fn_name == "ppo":
                loss_fn_kwargs["loss_fn_config"] = {
                    "clip_low_threshold": 1.0 - config.ppo_clip_eps,
                    "clip_high_threshold": 1.0 + config.ppo_clip_eps,
                }
            for chunk in chunks[:n_steps]:
                if not chunk:
                    continue
                training_client.forward_backward(chunk, **loss_fn_kwargs).result()
                training_client.optim_step(adam_params).result()
        t_step = time.time() - t_step0

        # ----- EVAL ROLLOUT -----
        if iter_idx % config.eval_every == 0:
            eval_result = sampling_client.sample(
                prompt=student_input,
                num_samples=config.n_eval_plans,
                sampling_params=sampling_params,
            ).result()
            with open(eval_path, "a") as f:
                for ek, eseq in enumerate(eval_result.sequences):
                    try:
                        eparsed = renderer.parse_response(eseq.tokens)
                        eraw = eparsed[0].get("content", "") if eparsed else tokenizer.decode(eseq.tokens)
                    except Exception:
                        eraw = tokenizer.decode(eseq.tokens)
                    f.write(json.dumps({
                        "iter": iter_idx,
                        "plan_id": f"iter_{iter_idx:03d}_eval_{ek}",
                        "text": extract_solution(eraw),
                    }) + "\n")

        # ----- WRITE BUFFER + METRICS -----
        with open(buffer_path, "a") as f:
            for rec in per_plan_records:
                f.write(json.dumps(rec) + "\n")

        mean_adv = float(np.mean(all_advs_flat)) if all_advs_flat else 0.0
        pos_frac = _safe_pos_frac(all_advs_flat)
        mean_r_this_iter = float(np.mean(rewards)) if rewards else 0.0
        with open(metrics_path, "a") as f:
            f.write(json.dumps({
                "iter": iter_idx,
                "n_datums": len(datums),
                "mean_reward": mean_r_this_iter,
                "mean_adv": mean_adv,
                "pos_frac": pos_frac,
                "mean_abs_adv": _safe_mean_abs(all_advs_flat),
                "wall_total_sec": time.time() - t0,
                "wall_sample_sec": t_sample,
                "wall_grade_sec": t_grade,
                "wall_train_sec": t_step,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "solution_only_mask": bool(config.solution_only_mask),
                "loss_fn_name": config.loss_fn_name,
            }) + "\n")

        logger.info(
            "Iter %d GRPO: %d datums, mean_reward=%.4f, mean_adv=%.4f, pos_frac=%.3f, train=%.1fs",
            iter_idx, len(datums), mean_r_this_iter, mean_adv, pos_frac, t_step,
        )
        logger.info("Iter %d done in %.1fs", iter_idx, time.time() - t0)

    # Final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=f"final_{config.today_date}",
        log_path=str(log_dir),
        kind="both",
        loop_state={"batch": config.n_iter},
    )
    logger.info("D6 GRPO-v1 complete. Final checkpoint saved.")


if __name__ == "__main__":
    config = chz.entrypoint(Config)
    main(config)
