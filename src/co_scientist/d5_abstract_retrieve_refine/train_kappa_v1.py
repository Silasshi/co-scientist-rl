"""D5 κ trainer v1 — distillation-level option (c) HER (mislabeled "SDPO").

⚠ TERMINOLOGY: same option (c) HER pattern as μ-v4 (TEACHER samples,
STUDENT supervises) but applied to distillation-level training (3-round
cumulative critique). Inherits same off-manifold gradient issue.
See `knowledge/current/CANONICAL_NAMING_REFERENCE.md`.

Architecture (per RETRIEVAL_DESIGN_v1.md):
- Per iter, per instance i ∈ 1..8:
  - Per round j ∈ 1..3 (independent across A_i^j):
    Teacher prompt = goal + oracle_batch_j + critique_{j-1}  (* round 1 cold-start)
    Student prompt = goal + oracle_batch_j  (no critique)
    Sample teacher distillation A_i^j; compute student logprobs on same tokens
    Build SDPO datum (advantages = clamp(scale * (teacher_lp - student_lp), [-clip, clip]))
  - After 3 rounds:
    Plan generation: model(goal + [A_i^1, A_i^2, A_i^3]) → plan_text  (no SDPO/CE in smoke)
    Plan prompt: build_kappa_plan_prompt(version="v4") — locked by τ_v4 PASS

Per iter:
  - 8 instances × 3 rounds = 24 SDPO datums on distillation tokens
  - 8 plan_texts for downstream audit (NOT in loss)
  - Microbatch grad steps (n_grad_steps_per_iter=4 like μ-v4)
  - Save sampler weights checkpoint

**Smoke variant simplifications** (vs full κ-v1 RETRIEVAL_DESIGN_v1):
  - critique_{1,2,3} use COLD-START for ALL rounds (no real Opus distillation
    critic dispatch). This isolates the "SDPO infrastructure + signal health"
    test from "does critic feedback drive learning". Real critic added in
    full κ-v1 (next plan after smoke verdict).
  - n_iter=2 (vs 20)
  - No CE on plan tokens (vs full = CE against reference plan)
  - Audit dispatched OUT-OF-BAND from chat (not via daemon)

Smoke pass criteria (per peaceful-tickling-wolf.md):
  1. Infrastructure: trainer completes 2 iter without crash, all 24 datums/iter built
  2. SDPO signal: mean_adv ∈ [0.05, 0.6] (cold-start gives smaller magnitude than real
     critic; healthy means non-zero non-NaN, sign positive most plans)
  3. Trajectory: after-iter-0 OR after-iter-1 audit ≥ 27 over τ_v4 baseline 26.50

Usage:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.train_kappa_v1 \
        log_path=projects/d5_abstract_retrieve_refine/runs/2026_04_28_kappa_v1_smoke
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
import tinker
from tinker import types
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

from co_scientist.shared.api_profiles import create_service_client
from co_scientist.d5_abstract_retrieve_refine.kappa_prompts_v1 import (
    DISTILLATION_COLD_START_CRITIQUE,
    build_distillation_student_prompt,
    build_distillation_teacher_prompt,
    build_kappa_plan_prompt,
    build_sdpo_datum,
    extract_distillation,
)
from co_scientist.d5_abstract_retrieve_refine.oracle_batch_helper_v1 import (
    ORACLE_V2_DEFAULT_PATH,
    make_batches,
    parse_oracle,
    render_batch,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)


REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    api_profile: str | None = "new"
    base_url: str | None = None

    log_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_kappa_v1_smoke"
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"
    oracle_path: str = ORACLE_V2_DEFAULT_PATH  # medium.md per Q3 reframe

    model_name: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 64
    learning_rate: float = 5e-5  # μ-v4 lock

    n_instances: int = 8
    n_rounds: int = 3
    batch_sizes: tuple[int, ...] = (60, 60, 61)  # medium 181 / 3
    shuffle_seed: int = 43  # κ_seed (distinct from τ_seed=42)

    distillation_max_tokens: int = 3500  # τ_v2 lock
    plan_max_tokens: int = 4096  # μ-v4 lock

    plan_prompt_version: str = "v4"  # τ_v4 lock

    sdpo_scale: float = 1.0
    sdpo_clip_advantage: float = 5.0

    n_iter: int = 2  # SMOKE
    save_every: int = 1
    n_grad_steps_per_iter: int = 4  # μ-v4 lock

    today_date: str = "2026_04_28"

    temperature: float = 1.0
    top_p: float = 0.95


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _safe_pos_frac(advs):
    if not advs:
        return 0.0
    return float(np.mean([1.0 if a > 0 else 0.0 for a in advs]))


def _safe_mean_abs(advs):
    if not advs:
        return 0.0
    return float(np.mean([abs(a) for a in advs]))


def main(config: Config):
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
    oracle_items = parse_oracle(_resolve(config.oracle_path))
    if sum(config.batch_sizes) != len(oracle_items):
        logger.warning("batch_sizes sum=%d != oracle items=%d",
                       sum(config.batch_sizes), len(oracle_items))
    batches = make_batches(oracle_items, sizes=config.batch_sizes,
                            seed=config.shuffle_seed)
    logger.info("Inputs loaded:")
    logger.info("  goal:               %d chars", len(goal))
    logger.info("  oracle (%s): %d items in %s batches (seed=%d)",
                Path(config.oracle_path).name, len(oracle_items),
                [len(b) for b in batches], config.shuffle_seed)
    logger.info("  log_dir:            %s", log_dir)

    # Save config snapshot
    cfg_dump = {k: getattr(config, k) for k in dir(config)
                if not k.startswith("_") and not callable(getattr(config, k))}
    (log_dir / "config.json").write_text(json.dumps(cfg_dump, default=str, indent=2))

    # Tokenizer + renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info("Renderer: %s", renderer_name)

    # Tinker service + LoRA training client
    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )
    last_checkpoint = checkpoint_utils.get_last_checkpoint(str(log_dir))
    if last_checkpoint is not None:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            last_checkpoint["state_path"]
        )
        start_iter = last_checkpoint.get("batch", 0) + 1
        logger.info("Resuming from iter %d", start_iter)
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank,
        )
        start_iter = 0
        logger.info("Fresh LoRA training client (rank=%d)", config.lora_rank)

    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    distill_sampling_params = tinker.types.SamplingParams(
        max_tokens=config.distillation_max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=renderer.get_stop_sequences(),
    )
    plan_sampling_params = tinker.types.SamplingParams(
        max_tokens=config.plan_max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=renderer.get_stop_sequences(),
    )

    # Output files
    metrics_path = log_dir / "metrics.jsonl"
    checkpoints_path = log_dir / "checkpoints.jsonl"
    eval_rollouts_path = log_dir / "eval_rollouts.jsonl"

    # ===========================================================================
    # Main loop
    # ===========================================================================
    for iter_idx in range(start_iter, config.n_iter):
        t0 = time.time()

        # Periodic checkpoint (skip iter 0 since no training has happened)
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

        # ===== 3-ROUND DISTILLATION + SDPO DATUM CONSTRUCTION =====
        # SMOKE: cold-start critique for ALL rounds (no real critic dispatch).
        prev_critique = DISTILLATION_COLD_START_CRITIQUE
        all_datums = []
        all_advs = []
        distillations_per_instance: list[list[str]] = [[] for _ in range(config.n_instances)]

        t_distill_total0 = time.time()
        for round_idx in range(1, config.n_rounds + 1):
            batch = batches[round_idx - 1]
            batch_text = render_batch(batch)

            teacher_text = build_distillation_teacher_prompt(
                goal=goal, oracle_batch_text=batch_text,
                critique_xml=prev_critique, round_idx=round_idx,
            )
            student_text = build_distillation_student_prompt(
                goal=goal, oracle_batch_text=batch_text, round_idx=round_idx,
            )
            teacher_input = renderer.build_generation_prompt(
                [{"role": "user", "content": teacher_text}]
            )
            student_input = renderer.build_generation_prompt(
                [{"role": "user", "content": student_text}]
            )
            student_tokens = student_input.to_ints()

            logger.info("Iter %d round %d: teacher prompt=%d chars, student prompt=%d chars",
                        iter_idx, round_idx, len(teacher_text), len(student_text))

            # Sample n_instances teacher distillations
            t_sample0 = time.time()
            teacher_future = sampling_client.sample(
                prompt=teacher_input,
                num_samples=config.n_instances,
                sampling_params=distill_sampling_params,
            )
            teacher_result = teacher_future.result()
            t_sample = time.time() - t_sample0

            # Compute student logprobs for all instances (parallel via futures)
            t_lp0 = time.time()
            student_lp_futures = [
                sampling_client.compute_logprobs(
                    types.ModelInput.from_ints(tokens=student_tokens + list(seq.tokens))
                )
                for seq in teacher_result.sequences
            ]
            student_lps_full = [f.result() for f in student_lp_futures]
            t_lp = time.time() - t_lp0
            logger.info("Iter %d round %d: %d distill samples in %.1fs, student lp in %.1fs",
                        iter_idx, round_idx, len(teacher_result.sequences), t_sample, t_lp)

            # Build SDPO datums + extract distillations per instance
            for k, (seq, full_s_lp) in enumerate(zip(teacher_result.sequences, student_lps_full)):
                gen = list(seq.tokens)
                t_lp_seq = list(seq.logprobs) if seq.logprobs else []
                s_lp = list(full_s_lp[len(student_tokens):len(student_tokens) + len(gen)])
                if not gen or len(t_lp_seq) != len(gen) or len(s_lp) != len(gen):
                    logger.warning("Iter %d round %d instance %d: len mismatch gen=%d t_lp=%d s_lp=%d; skipping",
                                   iter_idx, round_idx, k, len(gen), len(t_lp_seq), len(s_lp))
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
                all_datums.append(datum)
                # Extract distillation A_k^{round_idx}
                A_text = extract_distillation(tokenizer.decode(gen))
                distillations_per_instance[k].append(A_text)

        t_distill_total = time.time() - t_distill_total0
        logger.info("Iter %d: distillation phase done in %.1fs (%d datums, %d total adv tokens)",
                    iter_idx, t_distill_total, len(all_datums), len(all_advs))

        # ===== TRAIN STEP (microbatch like μ-v4) =====
        t_step0 = time.time()
        if all_datums:
            n_steps = max(1, config.n_grad_steps_per_iter)
            chunk_size = max(1, len(all_datums) // n_steps)
            chunks = [all_datums[i:i + chunk_size]
                      for i in range(0, len(all_datums), chunk_size)]
            n_done = 0
            for chunk in chunks[:n_steps]:
                if not chunk:
                    continue
                fb_fut = training_client.forward_backward(
                    chunk, loss_fn="importance_sampling",
                )
                os_fut = training_client.optim_step(adam_params)
                fb_fut.result()
                os_fut.result()
                n_done += 1
            logger.info("Iter %d: ran %d microbatch grad steps (chunk_size=%d)",
                        iter_idx, n_done, chunk_size)
        t_step = time.time() - t_step0

        mean_adv = float(np.mean(all_advs)) if all_advs else 0.0
        pos_frac = _safe_pos_frac(all_advs)
        mean_abs = _safe_mean_abs(all_advs)
        logger.info(
            "Iter %d SDPO: %d datums, mean_adv=%.4f, pos_frac=%.3f, mean|A|=%.4f, train=%.1fs",
            iter_idx, len(all_datums), mean_adv, pos_frac, mean_abs, t_step,
        )

        # ===== PLAN GENERATION (post-train, for audit) =====
        # Updated sampler reflects post-grad-step LoRA weights
        sp_path_post = training_client.save_weights_for_sampler(
            name=f"iter_{iter_idx:04d}_post"
        ).result().path
        sampling_client_post = service_client.create_sampling_client(model_path=sp_path_post)

        t_plan0 = time.time()
        plan_records = []
        for k in range(config.n_instances):
            A_list = distillations_per_instance[k]
            if len(A_list) != config.n_rounds:
                logger.warning("Iter %d instance %d: only %d distillations (expected %d), padding empty",
                               iter_idx, k, len(A_list), config.n_rounds)
                while len(A_list) < config.n_rounds:
                    A_list.append("")
            plan_prompt = build_kappa_plan_prompt(
                goal=goal, A_list=A_list, version=config.plan_prompt_version,
            )
            plan_input = renderer.build_generation_prompt(
                [{"role": "user", "content": plan_prompt}]
            )
            plan_result = sampling_client_post.sample(
                prompt=plan_input, num_samples=1, sampling_params=plan_sampling_params,
            ).result()
            plan_seq = plan_result.sequences[0]
            plan_text = tokenizer.decode(plan_seq.tokens)
            plan_records.append({
                "iter": iter_idx,
                "plan_id": f"iter_{iter_idx:03d}_inst_{k}",
                "instance_idx": k,
                "text": plan_text,
                "stop_reason": plan_seq.stop_reason,
                "n_tokens": len(plan_seq.tokens),
            })

        with open(eval_rollouts_path, "a") as f:
            for rec in plan_records:
                f.write(json.dumps(rec) + "\n")

        t_plan = time.time() - t_plan0
        logger.info("Iter %d: plan generation %d plans in %.1fs (post-train sampler %s)",
                    iter_idx, len(plan_records), t_plan, sp_path_post)

        # ===== METRICS =====
        with open(metrics_path, "a") as f:
            f.write(json.dumps({
                "iter": iter_idx,
                "n_datums": len(all_datums),
                "n_adv_tokens": len(all_advs),
                "mean_adv": mean_adv,
                "pos_frac": pos_frac,
                "mean_abs_adv": mean_abs,
                "wall_total_sec": time.time() - t0,
                "wall_distill_sec": t_distill_total,
                "wall_train_sec": t_step,
                "wall_plan_sec": t_plan,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "n_plans": len(plan_records),
                "post_train_sampler": sp_path_post,
            }) + "\n")

        if not all_advs:
            logger.warning("Iter %d: no advantages computed; check len mismatches.", iter_idx)

        logger.info("Iter %d done in %.1fs", iter_idx, time.time() - t0)

    # ===========================================================================
    # Final checkpoint
    # ===========================================================================
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=f"final_{config.today_date}",
        log_path=str(log_dir),
        kind="both",
        loop_state={"batch": config.n_iter},
    )
    logger.info("Saved final checkpoint")
    logger.info("κ smoke run complete: %d iter, %d plans/iter eval ready for audit dispatch",
                config.n_iter, config.n_instances)


if __name__ == "__main__":
    config = chz.entrypoint(Config)
    main(config)
