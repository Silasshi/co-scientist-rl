"""Pure-Revision trainer with UCB plan selection.

Evolution of train_critique_revise.py. Key differences:
  - Iter 0: generate M_seed fresh plans (cold start population)
  - Iter 1+: NO fresh generation. All plans come from revising buffer entries.
  - Plan selection uses UCB (Upper Confidence Bound) instead of top-K,
    balancing exploitation (high-reward plans) with exploration (under-revised plans).
  - RL trains ONLY the revision ability (delta-as-advantage on positive revisions).

Pipeline:
  Iter 0 (seed):
    → Generate M_seed fresh plans
    → Score all (hard gates + 8 signals × N repeats)
    → Add to buffer (no RL update)

  Iter 1+ (pure revision):
    → UCB-select M_revise plans from buffer
    → For each: identify bottleneck signal → critique → revise
    → Score all revisions (hard gates + 8 signals × N repeats)
    → Add revisions to buffer (with parent link, delta, critique)
    → RL update: importance_sampling, advantage = max(0, delta) * scale

Inspired by:
  - Learning to Self-Evolve (LSE, Chen et al. 2026, arXiv:2603.18620):
    delta-as-reward for self-improvement via RL
  - SCoRe (ICLR 2025): multi-turn RL for self-correction
  - TTT-Discover (Yuksekgonul et al. 2026): buffer-based test-time training

Usage:
    source tools/use_api_profile.sh new
    python src/co_scientist/ttt_discover/train_pure_revision.py
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field
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

from co_scientist.ttt_discover.train_buffer_ttt import (
    TARGET_GOAL, TARGET_TARGET, ALT_GOALS,
    TenSignalReward, PlanFutures,
    build_research_plan_prompt,
    launch_plan_reward, collect_plan_reward,
)
from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.ten_signal_reward import (
    SIGNALS as GRADIENT_SIGNALS,
    SIGNAL_WEIGHTS as GRADIENT_WEIGHTS,
    aggregate_reward as gradient_aggregate,
    SignalSpec,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)


# =============================================================================
# Config
# =============================================================================


@chz.chz
class Config:
    base_url: str | None = None
    api_profile: str | None = "new"

    log_path: str = "/home/silas/co-scientist-project/projects/ttt_discover/runs/_archive/2026_04_buffer_ttt/pure_revision_v1"
    model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"

    # Loop
    n_iterations: int = 50
    M_seed: int = 8                # fresh plans at iter 0 (seed population)
    M_revise: int = 4              # revisions per iter (iter 1+)
    grader_repeats: int = 2        # N: grader repeats per signal

    # UCB plan selection
    ucb_c: float = 1.0             # exploration coefficient for UCB
    min_buffer_for_revision: int = 4  # need at least this many entries to start revising

    # RL
    delta_scale: float = 5.0       # scale delta to make advantage meaningful
    learning_rate: float = 4e-5
    lora_rank: int = 32

    # Sampling
    max_length: int = 32768
    max_tokens: int = 2048
    grader_max_tokens: int = 2048
    grader_hard_gate_max_tokens: int = 8192
    temperature: float = 1.0
    grader_temperature: float = 0.0

    # Format
    max_word_count: int = 750
    min_words: int = 30

    # Logging
    save_every: int = 5
    today_date: str = time.strftime("%Y-%m-%d", time.localtime())
    seed: int = 0


# =============================================================================
# UCB plan selection
# =============================================================================


def ucb_select(
    buffer: list[dict],
    n_select: int,
    total_selections: int,
    c: float,
    rng: random.Random,
) -> list[tuple[int, dict]]:
    """Select plans from buffer using Upper Confidence Bound.

    UCB(plan) = aggregate_reward + c * sqrt(ln(N) / (1 + n_selected))

    Where N = total_selections across all plans, n_selected = how many times
    this specific plan has been selected for revision.

    Returns list of (buffer_index, entry) tuples.
    """
    valid = [(i, e) for i, e in enumerate(buffer) if e.get("hard_gate_passed")]
    if len(valid) <= n_select:
        return valid

    # Compute UCB scores
    scored = []
    for buf_idx, entry in valid:
        reward = entry["aggregate_reward"]
        n_sel = entry.get("n_selected", 0)
        if total_selections > 0:
            exploration = c * math.sqrt(math.log(total_selections + 1) / (1 + n_sel))
        else:
            exploration = c  # initial: everyone gets full exploration bonus
        ucb_score = reward + exploration
        scored.append((buf_idx, entry, ucb_score))

    # Sort by UCB score descending, take top n_select
    scored.sort(key=lambda x: x[2], reverse=True)
    selected = [(idx, entry) for idx, entry, _ in scored[:n_select]]
    return selected


# =============================================================================
# Critique-revise prompt (same as train_critique_revise.py)
# =============================================================================


def identify_bottleneck(signal_vector: dict[str, int | None]) -> tuple[str, int]:
    """Find the signal with the lowest score."""
    valid = {k: v for k, v in signal_vector.items() if v is not None}
    if not valid:
        return GRADIENT_SIGNALS[0].id, 1
    bottleneck_id = min(valid, key=valid.get)
    return bottleneck_id, valid[bottleneck_id]


def get_signal_spec(signal_id: str) -> SignalSpec:
    for spec in GRADIENT_SIGNALS:
        if spec.id == signal_id:
            return spec
    return GRADIENT_SIGNALS[0]


def build_critique_revise_prompt(
    goal: str,
    plan_text: str,
    signal_vector: dict[str, int | None],
    aggregate_reward: float,
    bottleneck_id: str,
    bottleneck_score: int,
) -> str:
    spec = get_signal_spec(bottleneck_id)
    signal_lines = "\n".join(
        f"  {sid}: {score}/5" + (" ← WEAKEST" if sid == bottleneck_id else "")
        for sid, score in signal_vector.items()
        if score is not None
    )

    return textwrap.dedent(f"""
        You are improving a research plan. The plan has been evaluated on several
        quality dimensions and its WEAKEST dimension has been identified.

        # Research Goal
        {goal}

        # Current Plan (aggregate score: {aggregate_reward:.3f})
        {plan_text.strip()}

        # Signal Scores
        {signal_lines}

        # Weakest Signal: {spec.name} (score: {bottleneck_score}/5)

        ## What this signal measures:
        {spec.question}

        ## Scoring rubric for this signal:
        {spec.scoring_rubric}

        # Your Task

        1. In <critique> tags, analyze in 2-3 sentences WHY the plan scored only
           {bottleneck_score}/5 on **{spec.name}**. Be specific about what is
           missing, weak, or unconvincing. Refer to the scoring rubric above.

        2. In <solution> tags, write a REVISED version of the plan that specifically
           addresses the weakness you identified in step 1. Keep all parts of the
           plan that are already strong — only modify what needs to improve on
           {spec.name}. Stay within 750 words.

        Important:
        - Do NOT just add vague claims. Add CONCRETE content that would satisfy
          the scoring rubric at a higher level.
        - Do NOT remove or weaken parts of the plan that score well on other signals.
        - Focus your changes on improving {spec.name} from {bottleneck_score} to
          {bottleneck_score + 1} or higher.

        <critique>
        ...your analysis of why {spec.name} scored {bottleneck_score}...
        </critique>
        <solution>
        ...your revised plan (max 750 words)...
        </solution>
    """).strip()


def parse_critique_and_plan(text: str) -> tuple[str, str]:
    critique = ""
    plan = ""
    critique_match = re.search(r"<critique>(.*?)</critique>", text, re.DOTALL)
    if critique_match:
        critique = critique_match.group(1).strip()
    solution_match = re.search(r"<solution>(.*?)</solution>", text, re.DOTALL)
    if solution_match:
        plan = solution_match.group(1).strip()
    if not plan and critique_match:
        plan = text[critique_match.end():].strip()
    return critique, plan


# =============================================================================
# Main training loop
# =============================================================================


def main(config: Config):
    os.makedirs(config.log_path, exist_ok=True)
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    logger.info(f"Paradigm: Pure-Revision with UCB Plan Selection")
    logger.info(f"Iter 0: {config.M_seed} seed plans. Iter 1+: {config.M_revise} revisions/iter")
    logger.info(f"UCB exploration coefficient c={config.ucb_c}")

    rng = random.Random(config.seed)

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Renderer: {renderer_name}")

    service_client = create_service_client(
        base_url=config.base_url,
        api_profile=config.api_profile,
    )

    last_checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path)
    if last_checkpoint is None:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_iter = 0
        logger.info(f"Fresh start: LoRA rank={config.lora_rank}")
    else:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            last_checkpoint["state_path"]
        )
        start_iter = last_checkpoint["batch"] + 1
        logger.info(f"Resuming from iteration {start_iter}")

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

    # Buffer
    buffer: list[dict] = []
    buffer_path = os.path.join(config.log_path, "buffer.jsonl")
    if os.path.exists(buffer_path):
        with open(buffer_path) as f:
            for line in f:
                buffer.append(json.loads(line))
        logger.info(f"Resumed buffer with {len(buffer)} entries")

    total_selections = sum(e.get("n_selected", 0) for e in buffer)

    train_log_path = os.path.join(config.log_path, "train", "training_logs.jsonl")
    os.makedirs(os.path.dirname(train_log_path), exist_ok=True)
    iter_summary_path = os.path.join(config.log_path, "train", "iter_summary.jsonl")

    # =========================================================================
    for iter_idx in range(start_iter, config.n_iterations):
        t_start = time.time()

        # Checkpoint
        if config.save_every > 0 and iter_idx > 0 and iter_idx % config.save_every == 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{iter_idx:04d}_{config.today_date}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": iter_idx},
            )

        # Save weights for sampler
        sampling_result = training_client.save_weights_for_sampler(
            name=f"iter_{iter_idx:04d}"
        ).result()
        sampling_client = service_client.create_sampling_client(
            model_path=sampling_result.path
        )

        is_seed_iter = (iter_idx == 0 and len(buffer) < config.min_buffer_for_revision)

        if is_seed_iter:
            # =============================================================
            # SEED ITERATION: Generate fresh plans to populate buffer
            # =============================================================
            prompt_text = build_research_plan_prompt(goal=TARGET_GOAL)
            model_input = renderer.build_generation_prompt(
                [{"role": "user", "content": prompt_text}]
            )

            seed_future = sampling_client.sample(
                prompt=model_input,
                num_samples=config.M_seed,
                sampling_params=sampling_params,
            )
            seed_result = seed_future.result()
            seed_samples = []
            for seq in seed_result.sequences:
                text = renderer.parse_response(seq.tokens)[0]["content"]
                if "<solution>" in text and "</solution>" not in text:
                    text = text.rstrip() + "\n</solution>"
                seed_samples.append({"text": text})

            logger.info(f"Iter {iter_idx}: generated {len(seed_samples)} seed plans")

            # Score all seeds
            seed_plan_futures = []
            for sample in seed_samples:
                if len(sample["text"].strip().split()) < config.min_words:
                    seed_plan_futures.append(None)
                    continue
                seed_plan_futures.append(
                    launch_plan_reward(
                        plan=sample["text"], goal=TARGET_GOAL,
                        grader_client=grader_client, renderer=renderer,
                        n_repeats=config.grader_repeats,
                        grader_max_tokens=config.grader_max_tokens,
                        hg_max_tokens=config.grader_hard_gate_max_tokens,
                        temperature=config.grader_temperature,
                    )
                )

            logger.info(f"Iter {iter_idx}: collecting seed grading...")
            t_collect = time.time()
            for i, (sample, fut) in enumerate(zip(seed_samples, seed_plan_futures)):
                if fut is None:
                    continue
                try:
                    reward = collect_plan_reward(fut, tokenizer)
                except Exception as e:
                    logger.error(f"Seed {i} grading failed: {type(e).__name__}")
                    continue
                entry = {
                    "iteration": iter_idx,
                    "plan_text": sample["text"],
                    "signal_vector": {k: v for k, v in reward.signal_vector.items() if v is not None},
                    "hard_gate_passed": reward.hard_gate_passed,
                    "hard_gate_failed_name": reward.hard_gate_failed_name,
                    "aggregate_reward": reward.aggregate_reward,
                    "word_count": len(sample["text"].strip().split()),
                    "entry_type": "seed",
                    "parent_idx": None,
                    "targeted_signal": None,
                    "critique_text": None,
                    "delta_reward": None,
                    "n_selected": 0,
                }
                buffer.append(entry)
                with open(buffer_path, "a") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            buffer_max = max((e["aggregate_reward"] for e in buffer if e.get("hard_gate_passed")), default=0.0)
            elapsed = time.time() - t_start
            logger.info(
                f"Iter {iter_idx}: SEED complete. buffer_size={len(buffer)} "
                f"buffer_max={buffer_max:.3f} | time={elapsed:.0f}s"
            )

            iter_summary = {
                "iter": iter_idx, "phase": "seed",
                "time/total": elapsed,
                "buffer/size": len(buffer),
                "reward/buffer_max": buffer_max,
                "seed/count": len(seed_samples),
            }
            with open(iter_summary_path, "a") as f:
                f.write(json.dumps(iter_summary) + "\n")
            ml_logger.log_metrics(iter_summary, step=iter_idx)
            continue  # no RL update on seed iter

        # =================================================================
        # REVISION ITERATION: Select plans via UCB → critique → revise
        # =================================================================

        # UCB selection
        selected = ucb_select(buffer, config.M_revise, total_selections, config.ucb_c, rng)
        if len(selected) < 2:
            logger.warning(f"Iter {iter_idx}: only {len(selected)} plans in buffer, skipping")
            continue

        # Update selection counts
        for buf_idx, _ in selected:
            buffer[buf_idx]["n_selected"] = buffer[buf_idx].get("n_selected", 0) + 1
            total_selections += 1

        targets_info = []
        for buf_idx, entry in selected:
            bid, bscore = identify_bottleneck(entry["signal_vector"])
            targets_info.append((buf_idx, entry, bid, bscore))

        logger.info(
            f"Iter {iter_idx}: UCB selected {len(selected)} plans. "
            f"Targets: {[t[2] for t in targets_info]}"
        )

        # Generate revisions
        revise_samples = []
        revise_plan_futures = []

        for buf_idx, parent, bottleneck_id, bottleneck_score in targets_info:
            revise_prompt = build_critique_revise_prompt(
                goal=TARGET_GOAL,
                plan_text=parent["plan_text"],
                signal_vector=parent["signal_vector"],
                aggregate_reward=parent["aggregate_reward"],
                bottleneck_id=bottleneck_id,
                bottleneck_score=bottleneck_score,
            )

            revise_input = renderer.build_generation_prompt(
                [{"role": "user", "content": revise_prompt}]
            )
            revise_tokens_ints = revise_input.to_ints()

            revise_future = sampling_client.sample(
                prompt=revise_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
            revise_result = revise_future.result()
            seq = revise_result.sequences[0]
            raw_text = renderer.parse_response(seq.tokens)[0]["content"]

            critique, revised_plan = parse_critique_and_plan(raw_text)
            if not revised_plan:
                revised_plan = raw_text

            plan_for_grading = revised_plan
            if "<solution>" not in plan_for_grading:
                plan_for_grading = f"<solution>\n{plan_for_grading}\n</solution>"
            if "<solution>" in plan_for_grading and "</solution>" not in plan_for_grading:
                plan_for_grading = plan_for_grading.rstrip() + "\n</solution>"

            revise_samples.append({
                "parent_idx": buf_idx,
                "parent_reward": parent["aggregate_reward"],
                "bottleneck_id": bottleneck_id,
                "bottleneck_score": bottleneck_score,
                "critique": critique,
                "text": plan_for_grading,
                "tokens": seq.tokens,
                "logprobs": seq.logprobs,
                "revise_tokens_ints": revise_tokens_ints,
            })

            # Launch grading
            if len(plan_for_grading.strip().split()) >= config.min_words:
                revise_plan_futures.append(
                    launch_plan_reward(
                        plan=plan_for_grading, goal=TARGET_GOAL,
                        grader_client=grader_client, renderer=renderer,
                        n_repeats=config.grader_repeats,
                        grader_max_tokens=config.grader_max_tokens,
                        hg_max_tokens=config.grader_hard_gate_max_tokens,
                        temperature=config.grader_temperature,
                    )
                )
            else:
                revise_plan_futures.append(None)

        # Collect revision scores
        logger.info(f"Iter {iter_idx}: collecting revision grading...")
        t_collect = time.time()
        revise_rewards: list[TenSignalReward | None] = []
        for fut in revise_plan_futures:
            if fut is None:
                revise_rewards.append(None)
                continue
            try:
                revise_rewards.append(collect_plan_reward(fut, tokenizer))
            except Exception as e:
                logger.error(f"Revision grading failed: {type(e).__name__}")
                revise_rewards.append(None)
        logger.info(f"Iter {iter_idx}: collection complete in {time.time() - t_collect:.0f}s")

        # Add revisions to buffer, compute deltas
        revision_deltas = []
        n_positive = 0
        for rsample, rreward in zip(revise_samples, revise_rewards):
            if rreward is None:
                continue
            delta = rreward.aggregate_reward - rsample["parent_reward"]
            revision_deltas.append(delta)
            if delta > 0:
                n_positive += 1
            entry = {
                "iteration": iter_idx,
                "plan_text": rsample["text"],
                "signal_vector": {k: v for k, v in rreward.signal_vector.items() if v is not None},
                "hard_gate_passed": rreward.hard_gate_passed,
                "hard_gate_failed_name": rreward.hard_gate_failed_name,
                "aggregate_reward": rreward.aggregate_reward,
                "word_count": len(rsample["text"].strip().split()),
                "entry_type": "revision",
                "parent_idx": rsample["parent_idx"],
                "targeted_signal": rsample["bottleneck_id"],
                "critique_text": rsample["critique"],
                "delta_reward": delta,
                "n_selected": 0,
            }
            buffer.append(entry)
            with open(buffer_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # =================================================================
        # RL Update: only positive-delta revisions
        # =================================================================
        training_datums = []
        for rsample, rreward in zip(revise_samples, revise_rewards):
            if rreward is None:
                continue
            delta = rreward.aggregate_reward - rsample["parent_reward"]
            if delta <= 0:
                continue

            advantage = delta * config.delta_scale
            rev_prompt_tokens = [int(t) for t in rsample["revise_tokens_ints"]]
            gen_tokens = [int(t) for t in rsample["tokens"]]
            full_seq = rev_prompt_tokens + gen_tokens
            ob_len = len(rev_prompt_tokens) - 1

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens=full_seq[:-1]),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(
                        torch.tensor(full_seq[1:], dtype=torch.long)
                    ),
                    "logprobs": TensorData.from_torch(
                        torch.tensor(
                            [0.0] * ob_len + rsample["logprobs"], dtype=torch.float
                        )
                    ),
                    "advantages": TensorData.from_torch(
                        torch.tensor(
                            [0.0] * ob_len + [advantage] * len(rsample["logprobs"]),
                            dtype=torch.float,
                        )
                    ),
                },
            )
            training_datums.append(datum)

        if training_datums:
            try:
                fwd_bwd_future = training_client.forward_backward(
                    training_datums,
                    loss_fn="importance_sampling",
                )
                optim_step_future = training_client.optim_step(adam_params)
                _ = fwd_bwd_future.result()
                _ = optim_step_future.result()
            except Exception:
                logger.exception(f"Iter {iter_idx}: training step failed")

        # =================================================================
        # Logging
        # =================================================================
        buffer_max = max(
            (e["aggregate_reward"] for e in buffer if e.get("hard_gate_passed")),
            default=0.0,
        )
        mean_delta = float(np.mean(revision_deltas)) if revision_deltas else 0.0
        max_delta = float(max(revision_deltas)) if revision_deltas else 0.0

        # Per-signal means
        per_signal = {}
        for spec in GRADIENT_SIGNALS:
            scores = [
                rr.signal_vector.get(spec.id)
                for rr in revise_rewards
                if rr is not None and rr.hard_gate_passed and rr.signal_vector.get(spec.id) is not None
            ]
            if scores:
                per_signal[f"signal/{spec.id}/mean"] = float(np.mean(scores))

        all_rewards = [rr.aggregate_reward for rr in revise_rewards if rr is not None]

        iter_summary = {
            "iter": iter_idx,
            "phase": "revision",
            "time/total": time.time() - t_start,
            "revise/count": len(revise_samples),
            "revise/n_positive": n_positive,
            "revise/mean_delta": mean_delta,
            "revise/max_delta": max_delta,
            "reward/mean": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "reward/max": float(max(all_rewards)) if all_rewards else 0.0,
            "reward/buffer_max": buffer_max,
            "buffer/size": len(buffer),
            "datums/count": len(training_datums),
            "ucb/total_selections": total_selections,
            **per_signal,
        }

        with open(iter_summary_path, "a") as f:
            f.write(json.dumps(iter_summary) + "\n")

        # Per-sample logs
        with open(train_log_path, "a") as f:
            for i, (rsample, rreward) in enumerate(zip(revise_samples, revise_rewards)):
                if rreward is None:
                    continue
                f.write(json.dumps({
                    "iter": iter_idx, "type": "revision", "sample_idx": i,
                    "text": rsample["text"],
                    "aggregate_reward": rreward.aggregate_reward,
                    "signal_vector": rreward.signal_vector,
                    "hard_gate_passed": rreward.hard_gate_passed,
                    "parent_idx": rsample["parent_idx"],
                    "parent_reward": rsample["parent_reward"],
                    "targeted_signal": rsample["bottleneck_id"],
                    "critique": rsample["critique"],
                    "delta": rreward.aggregate_reward - rsample["parent_reward"],
                }, ensure_ascii=False) + "\n")

        ml_logger.log_metrics(iter_summary, step=iter_idx)
        logger.info(
            f"Iter {iter_idx}: reward mean={iter_summary['reward/mean']:.3f} "
            f"max={iter_summary['reward/max']:.3f} buffer_max={buffer_max:.3f} "
            f"| revisions: {n_positive}/{len(revise_samples)} positive "
            f"(mean_delta={mean_delta:.3f}, max_delta={max_delta:.3f}) "
            f"| UCB selections={total_selections} "
            f"| time={iter_summary['time/total']:.0f}s"
        )

        # Persist updated n_selected counts (rewrite buffer file)
        if iter_idx % 10 == 0:
            with open(buffer_path, "w") as f:
                for entry in buffer:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=f"{iter_idx:04d}_final_{config.today_date}",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": iter_idx},
    )
    ml_logger.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    chz.entrypoint(main)
