"""D5 μ baseline trainer — plan-level SDPO with oracle abstraction (v3).

What this trains: Qwen3-30B-A3B LoRA, plan-level SDPO between
  TEACHER context = goal + oracle_v3 + prev_critique
  STUDENT context = goal + oracle_v3 (no critique)

Per iter:
  1. sample N plans under TEACHER context
  2. compute student logprobs of same plan tokens under STUDENT context
  3. SDPO advantage A_t = clamp(scale * (teacher_lp - student_lp), [-clip, +clip])
  4. forward_backward(loss_fn="importance_sampling") + optim_step
  5. pick 1 plan, ship to plan-critic subagent, block on response, parse
     critique XML for next iter
  6. iter 0 uses COLD_START_CRITIQUE_NONE (handcrafted neutral block)

Eval cadence K=2 (iters 0, 2, 4, 6, 8, 10): sample 8 from STUDENT context,
fire async audit subagent request (non-blocking). Audit results are read
after the run completes.

Decision matrix is applied MANUALLY post-run (see plan steady-tinkering-wave.md).

Usage:
    source shared/tools/use_api_profile.sh new
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.train_mu_baseline_v1 \
        config.log_path=projects/d5_abstract_retrieve_refine/runs/2026_04_25_mu_baseline_v1 \
        config.n_iter=10 config.eval_every=2

Daemon must be running in a separate Claude Code window with the prompt at
  projects/d5_abstract_retrieve_refine/scripts/opus_critic_audit_daemon.md
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
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
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.opus_critic_subagent import OpusCriticClient
from co_scientist.shared.opus_audit_subagent import OpusAuditClient
from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v1 import (
    COLD_START_CRITIQUE_NONE,
    build_audit_request_payload,
    build_critic_request_payload,
    build_sdpo_datum,
    build_student_prompt,
    build_teacher_prompt,
    extract_solution,
    parse_critique_xml,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)


REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    api_profile: str | None = "new"
    base_url: str | None = None

    # Paths
    log_path: str = (
        "projects/d5_abstract_retrieve_refine/runs/2026_04_25_mu_baseline_v1"
    )
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"
    reference_plan_path: str = (
        "projects/d5_abstract_retrieve_refine/dataset/reference_solution.txt"
    )
    oracle_abstraction_path: str = (
        "projects/d5_abstract_retrieve_refine/data/oracles/smoke/v3.md"
    )
    source_paper_path: str = (
        "projects/d5_abstract_retrieve_refine/data/source_paper/v1.md"
    )

    # Model
    model_name: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 64
    learning_rate: float = 1e-5

    # Sampling (Phase 0.5c sampling fix: temp=1.0 + top_p, NO seed)
    n_plans: int = 8
    max_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 0.95

    # SDPO
    sdpo_scale: float = 1.0
    sdpo_clip_advantage: float = 5.0
    anchor_ce_weight: float = 0.0  # off per user decision (pure SDPO ablation)

    # Loop
    n_iter: int = 10
    eval_every: int = 2  # eval at iters 0, 2, 4, ..., 10 (modulo)
    n_eval_plans: int = 8
    save_every: int = 5

    # Critic
    critic_timeout_sec: float = 900.0  # 15 min per critic call
    skip_critic_at_iter0: bool = False  # if True, do not call critic in iter 0
                                         # (since cold-start is already in use)

    # Misc
    today_date: str = "2026_04_25"
    seed: int = 42  # for plan-pick randomness (NOT for sampling)


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _decode_plan(seq, tokenizer, renderer) -> str:
    """Decode a sampled sequence to plan text. Strips Qwen3 chat scaffolding via renderer."""
    parsed = renderer.parse_response(seq.tokens)
    content = parsed[0].get("content", "") if parsed else ""
    # Best-effort solution-tag extraction; falls back to raw content
    return extract_solution(content) if content else tokenizer.decode(seq.tokens)


def _safe_pos_frac(advs: list[float]) -> float:
    if not advs:
        return 0.0
    return float(np.mean([1.0 if a > 0 else 0.0 for a in advs]))


def _safe_mean_abs(advs: list[float]) -> float:
    if not advs:
        return 0.0
    return float(np.mean([abs(a) for a in advs]))


def main(config: Config):
    rng = random.Random(config.seed)

    log_dir = _resolve(config.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    ml_log.setup_logging(
        log_dir=str(log_dir),
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    # Inputs
    goal = _resolve(config.goal_path).read_text().strip()
    oracle = _resolve(config.oracle_abstraction_path).read_text().strip()
    ref_plan = _resolve(config.reference_plan_path).read_text().strip()
    source_paper_md = _resolve(config.source_paper_path).read_text().strip()
    logger.info("Inputs loaded:")
    logger.info("  goal:           %d chars", len(goal))
    logger.info("  oracle (v3):    %d chars", len(oracle))
    logger.info("  reference plan: %d chars (anchor; not used unless anchor_ce_weight>0)", len(ref_plan))
    logger.info("  source paper:   %d chars (privileged info for critic)", len(source_paper_md))
    logger.info("  log_dir:        %s", log_dir)

    # Save config snapshot
    cfg_path = log_dir / "config.json"
    cfg_path.write_text(json.dumps({k: getattr(config, k) for k in dir(config)
                                     if not k.startswith("_") and not callable(getattr(config, k))},
                                    default=str, indent=2))

    # Tokenizer + renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info("Renderer: %s", renderer_name)

    # Tinker service + LoRA training client (resume if checkpoint exists)
    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )
    last_checkpoint = checkpoint_utils.get_last_checkpoint(str(log_dir))
    if last_checkpoint is None:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank,
        )
        start_iter = 0
        logger.info("Fresh training client (LoRA rank=%d)", config.lora_rank)
    else:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            last_checkpoint["state_path"]
        )
        start_iter = last_checkpoint.get("batch", 0) + 1
        logger.info("Resuming from iter %d (state=%s)", start_iter, last_checkpoint["state_path"])

    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=renderer.get_stop_sequences(),
    )

    # Subagent clients
    critic_client = OpusCriticClient(
        log_path=log_dir, timeout_sec=config.critic_timeout_sec,
    )
    audit_client = OpusAuditClient(log_path=log_dir)

    # Output files
    buffer_path = log_dir / "buffer.jsonl"
    metrics_path = log_dir / "metrics.jsonl"
    checkpoints_path = log_dir / "checkpoints.jsonl"

    # If resuming, reload prev_critique from last critic_responses file
    prev_critique = COLD_START_CRITIQUE_NONE
    if start_iter > 0:
        prev_iter = start_iter - 1
        rp = log_dir / "critic_responses" / f"iter_{prev_iter:03d}.json"
        if rp.exists():
            try:
                resp = json.loads(rp.read_text())
                judgments = resp.get("judgments", {})
                if judgments:
                    pid = list(judgments.keys())[0]
                    prev_critique = parse_critique_xml(
                        judgments[pid].get("critique_xml", "")
                    )
                    logger.info("Resumed prev_critique from iter %d", prev_iter)
            except Exception as e:
                logger.warning("Failed to resume prev_critique: %s; using cold-start", e)

    # ===========================================================================
    # Main loop
    # ===========================================================================
    for iter_idx in range(start_iter, config.n_iter):
        t0 = time.time()

        # Periodic checkpoint
        if config.save_every > 0 and iter_idx % config.save_every == 0 and iter_idx > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{iter_idx:06d}_{config.today_date}",
                log_path=str(log_dir),
                kind="state",
                loop_state={"batch": iter_idx},
            )
            logger.info("Saved checkpoint at iter %d", iter_idx)

        # Sampling client = current LoRA weights
        sp_path = training_client.save_weights_for_sampler(
            name=f"iter_{iter_idx:04d}"
        ).result().path
        sampling_client = service_client.create_sampling_client(model_path=sp_path)
        with open(checkpoints_path, "a") as f:
            f.write(json.dumps({"iter": iter_idx, "sampler_path": sp_path,
                                 "ts": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")

        # ----- TEACHER ROLLOUT -----
        teacher_text = build_teacher_prompt(goal, oracle, prev_critique)
        teacher_convo = [{"role": "user", "content": teacher_text}]
        teacher_input = renderer.build_generation_prompt(teacher_convo)
        t_sample0 = time.time()
        teacher_future = sampling_client.sample(
            prompt=teacher_input,
            num_samples=config.n_plans,
            sampling_params=sampling_params,
        )
        teacher_result = teacher_future.result()
        t_sample = time.time() - t_sample0
        logger.info("Iter %d: teacher sampled %d plans in %.1fs",
                     iter_idx, len(teacher_result.sequences), t_sample)

        # ----- STUDENT LOGPROBS -----
        student_text = build_student_prompt(goal, oracle)
        student_convo = [{"role": "user", "content": student_text}]
        student_input = renderer.build_generation_prompt(student_convo)
        student_tokens = student_input.to_ints()

        t_lp0 = time.time()
        # Launch all logprob futures first, then collect — server-side parallel
        student_lp_futures = [
            sampling_client.compute_logprobs(
                types.ModelInput.from_ints(tokens=student_tokens + list(seq.tokens))
            )
            for seq in teacher_result.sequences
        ]
        student_lps_full = [f.result() for f in student_lp_futures]
        t_lp = time.time() - t_lp0
        logger.info("Iter %d: student logprobs in %.1fs", iter_idx, t_lp)

        # ----- BUILD SDPO DATUMS -----
        datums = []
        all_advs = []
        per_plan_records = []
        for k, (seq, full_s_lp) in enumerate(zip(teacher_result.sequences, student_lps_full)):
            gen = list(seq.tokens)
            t_lp_seq = list(seq.logprobs) if seq.logprobs else []
            # full_s_lp has length len(student_tokens) + len(gen). Slice to gen-only.
            s_lp = list(full_s_lp[len(student_tokens):len(student_tokens) + len(gen)])
            if not gen or len(t_lp_seq) != len(gen) or len(s_lp) != len(gen):
                logger.warning("Iter %d plan %d: len mismatch gen=%d t_lp=%d s_lp=%d; skipping",
                                iter_idx, k, len(gen), len(t_lp_seq), len(s_lp))
                continue
            per_tok_adv = []
            for t_, s_ in zip(t_lp_seq, s_lp):
                a = float(t_ - s_) * config.sdpo_scale
                a = max(-config.sdpo_clip_advantage,
                          min(config.sdpo_clip_advantage, a))
                per_tok_adv.append(a)
            all_advs.extend(per_tok_adv)
            datum = build_sdpo_datum(
                student_prompt_tokens=student_tokens,
                gen_tokens=gen,
                student_logprobs=s_lp,
                advantages=per_tok_adv,
            )
            datums.append(datum)
            per_plan_records.append({
                "iter": iter_idx, "plan_idx": k,
                "n_gen_tokens": len(gen),
                "mean_t_lp": float(np.mean(t_lp_seq)),
                "mean_s_lp": float(np.mean(s_lp)),
                "mean_adv": float(np.mean(per_tok_adv)),
                "stop_reason": seq.stop_reason,
                "plan_text": _decode_plan(seq, tokenizer, renderer),
                "raw_tokens_text": tokenizer.decode(gen),
            })

        # ----- TRAIN STEP -----
        t_step0 = time.time()
        if datums:
            fb_fut = training_client.forward_backward(datums, loss_fn="importance_sampling")
            os_fut = training_client.optim_step(adam_params)
            fb_fut.result()
            os_fut.result()
        t_step = time.time() - t_step0

        mean_adv = float(np.mean(all_advs)) if all_advs else 0.0
        pos_frac = _safe_pos_frac(all_advs)
        mean_abs = _safe_mean_abs(all_advs)
        logger.info(
            "Iter %d SDPO: %d datums, mean_adv=%.4f, pos_frac=%.3f, mean|A|=%.4f, train_step=%.1fs",
            iter_idx, len(datums), mean_adv, pos_frac, mean_abs, t_step,
        )

        # ----- WRITE BUFFER (per-plan) -----
        with open(buffer_path, "a") as f:
            for rec in per_plan_records:
                f.write(json.dumps(rec) + "\n")

        # ----- CRITIC CALL (BLOCKING) -----
        if not (config.skip_critic_at_iter0 and iter_idx == 0):
            chosen_idx = rng.randrange(len(per_plan_records)) if per_plan_records else 0
            chosen_text = (per_plan_records[chosen_idx]["plan_text"]
                           if per_plan_records else "")
            chosen_plan_id = f"iter_{iter_idx:03d}_pick_{chosen_idx}"
            payload = build_critic_request_payload(
                iter_idx=iter_idx, goal=goal,
                plan_text=chosen_text,
                source_paper_md=source_paper_md,
                plan_id=chosen_plan_id,
            )
            t_crit0 = time.time()
            critic_client.submit(iter_idx, payload)
            try:
                resp = critic_client.wait(iter_idx)
                prev_critique = critic_client.critique_for(
                    resp, chosen_plan_id, fallback=COLD_START_CRITIQUE_NONE,
                )
            except TimeoutError as e:
                logger.error("Iter %d: critic timeout: %s. Falling back to cold-start critique.",
                              iter_idx, e)
                prev_critique = COLD_START_CRITIQUE_NONE
            t_crit = time.time() - t_crit0
            logger.info("Iter %d: critic round-trip %.1fs, critique len=%d chars",
                         iter_idx, t_crit, len(prev_critique))

        # ----- EVAL AUDIT (ASYNC, non-blocking) -----
        if iter_idx % config.eval_every == 0:
            t_eval0 = time.time()
            eval_future = sampling_client.sample(
                prompt=student_input,
                num_samples=config.n_eval_plans,
                sampling_params=sampling_params,
            )
            eval_result = eval_future.result()
            eval_plans = []
            for k, seq in enumerate(eval_result.sequences):
                pid = f"iter_{iter_idx:03d}_eval_{k}"
                ptext = _decode_plan(seq, tokenizer, renderer)
                eval_plans.append({"plan_id": pid, "text": ptext})
            audit_payload = build_audit_request_payload(
                iter_idx=iter_idx, goal=goal, plans=eval_plans,
            )
            audit_client.submit(iter_idx, audit_payload)
            t_eval = time.time() - t_eval0
            # Persist eval rollouts to a sidecar file for traceability
            eval_path = log_dir / "eval_rollouts.jsonl"
            with open(eval_path, "a") as f:
                for ep in eval_plans:
                    f.write(json.dumps({"iter": iter_idx, **ep}) + "\n")
            logger.info("Iter %d: eval audit submitted (%d plans, %.1fs sample, async)",
                         iter_idx, len(eval_plans), t_eval)

        # ----- METRICS -----
        with open(metrics_path, "a") as f:
            f.write(json.dumps({
                "iter": iter_idx,
                "n_datums": len(datums),
                "mean_adv": mean_adv,
                "pos_frac": pos_frac,
                "mean_abs_adv": mean_abs,
                "wall_total_sec": time.time() - t0,
                "wall_sample_sec": t_sample,
                "wall_logprob_sec": t_lp,
                "wall_train_sec": t_step,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "prev_critique_len": len(prev_critique),
            }) + "\n")

        # ----- SAFETY RAIL: NaN advantages or trivially zero -----
        if not all_advs:
            logger.warning("Iter %d: no advantages computed; check len mismatches.", iter_idx)

        logger.info("Iter %d done in %.1fs", iter_idx, time.time() - t0)

    # ===========================================================================
    # Final checkpoint + audit collection
    # ===========================================================================
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=f"final_{config.today_date}",
        log_path=str(log_dir),
        kind="both",
        loop_state={"batch": config.n_iter},
    )
    logger.info("Saved final checkpoint")

    # Pull all available audit responses; wait briefly for last submitted iter
    logger.info("Collecting audit responses (waiting up to 30 min for final iters)...")
    audit_summary = audit_client.collect_all(
        out_path=log_dir / "audit_log.jsonl",
        timeout_sec=1800.0,
        poll_interval_sec=15.0,
    )

    # Decision-matrix preview (last 3 evals)
    valid = [s for s in audit_summary if s.get("mean_total") is not None]
    if valid:
        last3 = valid[-3:]
        mu_proxy = float(np.mean([s["mean_total"] for s in last3]))
        logger.info("==================== DECISION MATRIX PREVIEW ====================")
        logger.info("μ proxy (mean of last %d audit means): %.2f /20", len(last3), mu_proxy)
        logger.info("δ baseline (frozen 30B + ref in context):     6.75")
        logger.info("ε baseline (frozen 235B + ref in context):  12.13")
        logger.info("Decision threshold: μ ≤ δ+1 (≤ 7.75) → STOP D5")
        logger.info("                     μ ≥ 9.00            → continue Phase 0b/1")
        logger.info("================================================================")
    else:
        logger.warning("No valid audit responses yet — run `audit_client.collect_all()` later.")

    logger.info("μ baseline run complete.")


if __name__ == "__main__":
    config = chz.entrypoint(Config)
    main(config)
