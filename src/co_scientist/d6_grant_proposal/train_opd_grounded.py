"""D6 OPD-grounded trainer (DEFERRED, out of paper scope) — OPD + per-token grounding bonus.

Extends train_opd.py with verifier-grounded reward:
  A_total[t] = clamp(A_critique[t] + grounding_bonus[t], ±opd_anchor_clip)

Grounding bonus fires on tokens within matched equation/citation spans found
in the generated proposal vs. the gold content extracted from reference_proposal.md.

Domain rules:
  social_science → citations-only grounding (G12a mode, no equation matching)
  ai / natural_science → equations + citations

D5 finding (F18): grounding alone Goodharts — equation score inflates,
citation/empirical scores collapse. Cliff guard (halve bonuses if >85% of
solution tokens receive bonus) is included. Recommend running this only AFTER
the OPD pilot shows improvement over the frozen baseline.

Prerequisite: build gold file first:
  PYTHONPATH=src python -m co_scientist.d6_grant_proposal.extract_gold \\
      goal_domain=ai goal_name=02_foundational_rl

Usage:
    source shared/tools/use_api_profile.sh new
    PYTHONPATH=src python -m co_scientist.d6_grant_proposal.train_opd_grounded \\
        goal_domain=ai goal_name=02_foundational_rl \\
        log_path=projects/d6_grant_proposal/runs/2026_04_30_pilot_foundrl_opd_grounded \\
        n_iter=16
"""
from __future__ import annotations

import json
import logging
import random
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
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.opus_critic_subagent import OpusCriticClient
from co_scientist.shared.opus_audit_subagent import OpusAuditClient
from co_scientist.d6_grant_proposal.prompts import (
    COLD_START_CRITIQUE_NONE,
    build_student_prompt,
    build_teacher_prompt,
    build_critic_request_payload,
    build_audit_request_payload,
    extract_solution,
    parse_critique_xml,
    build_sdpo_datum,
)
from co_scientist.d6_grant_proposal.train_opd import (
    find_solution_content_span,
    build_solution_token_mask_from_tokens,
    _safe_pos_frac,
    _safe_mean_abs,
    _decode_plan,
)
from co_scientist.d6_grant_proposal.grounding import (
    load_gold,
    compute_grounding_bonus_spans,
    check_grounding_saturation,
    build_grounding_token_mask,
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


def _gold_path(config) -> Path:
    if config.gold_path:
        return Path(config.gold_path)
    return _resolve(
        f"{config.data_base}/{config.goal_domain}/{config.goal_name}/gold/gold.json"
    )


def _active_grounding_signals(goal_domain: str) -> tuple[str, ...]:
    if goal_domain == "social_science":
        return ("citations",)
    return ("equations", "citations")


@chz.chz
class Config:
    api_profile: str | None = "new"
    base_url: str | None = None

    goal_domain: str = "ai"
    goal_name: str = "02_foundational_rl"

    log_path: str = "projects/d6_grant_proposal/runs/2026_04_30_pilot_foundrl_opd_grounded"
    dataset_base: str = "projects/d6_grant_proposal/dataset"
    data_base: str = "projects/d6_grant_proposal/data"

    model_name: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 64
    learning_rate: float = 5e-5

    n_plans: int = 8
    max_tokens: int = 8192
    temperature: float = 1.0
    top_p: float = 0.95

    sdpo_scale: float = 1.0
    opd_anchor_clip: float = 5.0

    n_iter: int = 16
    eval_every: int = 1
    n_eval_plans: int = 8
    save_every: int = 1
    n_grad_steps_per_iter: int = 4
    audit_drop_threshold: float = 5.0

    critic_timeout_sec: float = 900.0

    init_state_path: str = ""
    reset_optimizer_state: bool = False

    today_date: str = "2026_04_30"
    seed: int = 42

    # OPD knobs (unchanged)
    solution_only_mask: bool = True
    trust_region_alpha: float = 0.05
    loss_fn_name: Literal["importance_sampling", "ppo"] = "ppo"
    ppo_clip_eps: float = 0.2

    # ===== OPD-grounded bonus knobs =====
    use_grounding: bool = True
    grounding_per_eq_bonus: float = 0.1
    grounding_per_citation_bonus: float = 0.05
    gold_path: str = ""           # auto-resolved from domain/goal if empty
    cliff_threshold: float = 0.85  # Goodhart guard: halve bonuses if saturated


def main(config: Config):
    rng = random.Random(config.seed)

    log_dir = _resolve(config.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    ml_log.setup_logging(
        log_dir=str(log_dir), wandb_project=None, wandb_name=None,
        config=config, do_configure_logging_module=True,
    )

    goal_dir = _goal_dir(config)
    goal = (goal_dir / "research_goal.md").read_text().strip()
    reference_proposal_md = (goal_dir / "reference_proposal.md").read_text().strip()

    oracle_path = _oracle_path(config)
    if not oracle_path.exists():
        raise FileNotFoundError(
            f"Slim oracle not found at {oracle_path}. "
            f"Run extract_oracle.py and extract_oracle_slim.py first."
        )
    oracle = oracle_path.read_text().strip()

    # Load gold file for grounding
    gold = {}
    grounding_signals: tuple[str, ...] = ()
    if config.use_grounding:
        gp = _gold_path(config)
        gold = load_gold(gp)
        grounding_signals = _active_grounding_signals(config.goal_domain)
        logger.info(
            "OPD-grounded: signals=%s, eq_bonus=%.3f, cite_bonus=%.3f, gold_path=%s",
            grounding_signals, config.grounding_per_eq_bonus,
            config.grounding_per_citation_bonus, gp,
        )
        logger.info(
            "Gold: %d equations, %d citations",
            len(gold.get("equations", [])), len(gold.get("citations", [])),
        )
    else:
        logger.info("OPD-grounded: DISABLED (use_grounding=False)")

    logger.info("D6 OPD-grounded inputs: %s/%s", config.goal_domain, config.goal_name)
    logger.info("OPD knobs: solution_only_mask=%s trust_region_alpha=%.3f loss_fn=%s ppo_clip_eps=%.2f",
                config.solution_only_mask, config.trust_region_alpha,
                config.loss_fn_name, config.ppo_clip_eps)

    (log_dir / "config.json").write_text(json.dumps(
        {k: getattr(config, k) for k in dir(config)
         if not k.startswith("_") and not callable(getattr(config, k))},
        default=str, indent=2
    ))

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info("Renderer: %s", renderer_name)

    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )

    initial_teacher_client = None
    if config.trust_region_alpha > 0.0:
        initial_teacher_client = service_client.create_sampling_client(
            base_model=config.model_name,
        )
        logger.info("Trust-region: frozen base client created (alpha=%.3f)", config.trust_region_alpha)

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

    critic_client = OpusCriticClient(log_path=log_dir, timeout_sec=config.critic_timeout_sec)
    audit_client = OpusAuditClient(log_path=log_dir)

    buffer_path = log_dir / "buffer.jsonl"
    metrics_path = log_dir / "metrics.jsonl"
    checkpoints_path = log_dir / "checkpoints.jsonl"

    # Per-iter grounding bonus scale (halved if cliff guard fires)
    iter_eq_bonus = config.grounding_per_eq_bonus
    iter_cite_bonus = config.grounding_per_citation_bonus

    # ===========================================================================
    # Main training loop
    # ===========================================================================
    for iter_idx in range(start_iter, config.n_iter):
        t0 = time.time()
        t_crit = 0.0

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

        # ----- STUDENT ROLLOUT -----
        student_text = build_student_prompt(goal, oracle)
        student_convo = [{"role": "user", "content": student_text}]
        student_input = renderer.build_generation_prompt(student_convo)
        student_tokens = student_input.to_ints()

        t_sample0 = time.time()
        student_future = sampling_client.sample(
            prompt=student_input, num_samples=config.n_plans, sampling_params=sampling_params,
        )
        student_result = student_future.result()
        t_sample = time.time() - t_sample0
        logger.info("Iter %d: STUDENT sampled %d plans in %.1fs",
                     iter_idx, len(student_result.sequences), t_sample)

        decoded_plans = []
        for k, seq in enumerate(student_result.sequences):
            try:
                decoded_plans.append(_decode_plan(seq, tokenizer, renderer))
            except Exception as e:
                logger.warning("Iter %d plan %d: decode failed: %s", iter_idx, k, e)
                decoded_plans.append("")

        # ----- CRITIC ON CURRENT ROLLOUT -----
        critique_current = COLD_START_CRITIQUE_NONE
        if decoded_plans and any(decoded_plans):
            valid_idxs = [i for i, t in enumerate(decoded_plans) if t]
            chosen_idx = rng.choice(valid_idxs)
            chosen_plan_id = f"iter_{iter_idx:03d}_pick_{chosen_idx}"
            payload = build_critic_request_payload(
                iter_idx=iter_idx, goal=goal,
                plan_text=decoded_plans[chosen_idx],
                reference_proposal_md=reference_proposal_md,
                plan_id=chosen_plan_id,
            )
            t_crit0 = time.time()
            critic_client.submit(iter_idx, payload)
            try:
                resp = critic_client.wait(iter_idx)
                critique_current = critic_client.critique_for(
                    resp, chosen_plan_id, fallback=COLD_START_CRITIQUE_NONE,
                )
            except TimeoutError as e:
                logger.error("Iter %d: critic timeout: %s. Using cold-start.", iter_idx, e)
                critique_current = COLD_START_CRITIQUE_NONE
            t_crit = time.time() - t_crit0
            logger.info("Iter %d: critic %.1fs, pick=%d", iter_idx, t_crit, chosen_idx)

        # ----- TEACHER LOGPROBS -----
        teacher_text = build_teacher_prompt(goal, oracle, critique_current)
        teacher_convo = [{"role": "user", "content": teacher_text}]
        teacher_input = renderer.build_generation_prompt(teacher_convo)
        teacher_tokens_prefix = teacher_input.to_ints()

        t_lp0 = time.time()
        teacher_lp_futures = [
            sampling_client.compute_logprobs(
                types.ModelInput.from_ints(tokens=teacher_tokens_prefix + list(seq.tokens))
            )
            for seq in student_result.sequences
        ]
        teacher_lps_full = [f.result() for f in teacher_lp_futures]
        t_lp = time.time() - t_lp0
        logger.info("Iter %d: TEACHER logprobs in %.1fs", iter_idx, t_lp)

        # ----- TRUST-REGION FROZEN BASE -----
        frozen_lps_full = None
        t_frozen_lp = 0.0
        if config.trust_region_alpha > 0.0 and initial_teacher_client is not None:
            t_fr0 = time.time()
            frozen_lp_futures = [
                initial_teacher_client.compute_logprobs(
                    types.ModelInput.from_ints(tokens=teacher_tokens_prefix + list(seq.tokens))
                )
                for seq in student_result.sequences
            ]
            frozen_lps_full = [f.result() for f in frozen_lp_futures]
            t_frozen_lp = time.time() - t_fr0
            logger.info("Iter %d: FROZEN-BASE logprobs (α=%.3f) in %.1fs",
                         iter_idx, config.trust_region_alpha, t_frozen_lp)

        # ----- BUILD DATUMS (with OPD-grounded bonus) -----
        datums = []
        all_advs = []
        per_plan_records = []
        iter_eq_hits_total = 0.0
        iter_cite_hits_total = 0.0
        iter_saturated_count = 0

        for k, seq in enumerate(student_result.sequences):
            gen = list(seq.tokens)
            sampler_lp_seq = list(seq.logprobs) if seq.logprobs else []
            full_t_lp = teacher_lps_full[k]
            t_lp_seq = list(
                full_t_lp[len(teacher_tokens_prefix):len(teacher_tokens_prefix) + len(gen)]
            )
            if not gen or len(sampler_lp_seq) != len(gen) or len(t_lp_seq) != len(gen):
                logger.warning(
                    "Iter %d plan %d: len mismatch; skipping", iter_idx, k,
                )
                continue

            s_lp = sampler_lp_seq

            # Trust-region interpolation
            if frozen_lps_full is not None:
                full_f_lp = frozen_lps_full[k]
                f_lp_seq = list(
                    full_f_lp[len(teacher_tokens_prefix):len(teacher_tokens_prefix) + len(gen)]
                )
                if len(f_lp_seq) == len(t_lp_seq):
                    a = config.trust_region_alpha
                    t_lp_eff = [(1 - a) * t + a * f for t, f in zip(t_lp_seq, f_lp_seq)]
                else:
                    t_lp_eff = t_lp_seq
            else:
                t_lp_eff = t_lp_seq

            clip_v = config.opd_anchor_clip
            per_tok_adv = []
            for t_, s_ in zip(t_lp_eff, s_lp):
                adv_base = float(t_ - s_) * config.sdpo_scale
                adv_base = max(-clip_v, min(clip_v, adv_base))
                per_tok_adv.append(adv_base)

            # Solution-only mask
            n_mask_active = len(per_tok_adv)
            mask_density = 1.0
            sol_mask_arr = None
            decoded_full = None
            sol_span = None

            if config.solution_only_mask:
                try:
                    decoded_full = tokenizer.decode(
                        gen, skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    sol_span = find_solution_content_span(decoded_full)
                except Exception:
                    decoded_full = None
                    sol_span = None

                sol_mask_arr_np = build_solution_token_mask_from_tokens(
                    tokenizer=tokenizer, sample_tokens=gen,
                )
                if sol_mask_arr_np.size == len(per_tok_adv):
                    per_tok_adv = [a * float(m) for a, m in zip(per_tok_adv, sol_mask_arr_np.tolist())]
                    n_mask_active = int(sol_mask_arr_np.sum())
                    mask_density = float(n_mask_active) / max(1, len(gen))

            # OPD-grounded: grounding bonus
            eq_hits_k = 0
            cite_hits_k = 0
            saturated_k = False
            if config.use_grounding and grounding_signals and decoded_plans[k] and decoded_full and sol_span:
                plan_text = decoded_plans[k]
                sol_start_char, sol_end_char = sol_span
                spans, diag = compute_grounding_bonus_spans(
                    plan_text=plan_text,
                    gold=gold,
                    signal_set=grounding_signals,
                    per_eq_bonus=iter_eq_bonus,
                    per_citation_bonus=iter_cite_bonus,
                )
                eq_hits_k = diag.get("eq_hits", 0)
                cite_hits_k = diag.get("cite_hits", 0)

                if spans:
                    bonus_arr, bonus_token_count = build_grounding_token_mask(
                        tokenizer=tokenizer,
                        decoded_full=decoded_full,
                        sol_start_char=sol_start_char,
                        plan_text=plan_text,
                        spans=spans,
                        sample_tokens=gen,
                    )
                    # Cliff guard
                    if check_grounding_saturation(
                        bonus_token_count, max(1, n_mask_active), config.cliff_threshold
                    ):
                        saturated_k = True
                        logger.warning(
                            "Iter %d plan %d: grounding SATURATED (%d/%d tokens, %.1f%%); "
                            "halving bonuses for this iter",
                            iter_idx, k, bonus_token_count, n_mask_active,
                            100.0 * bonus_token_count / max(1, n_mask_active),
                        )
                        # Halve the bonus array
                        bonus_arr = bonus_arr * 0.5

                    # Add grounding bonus and re-clip
                    for t_idx in range(len(per_tok_adv)):
                        if t_idx < len(bonus_arr):
                            combined = per_tok_adv[t_idx] + float(bonus_arr[t_idx])
                            per_tok_adv[t_idx] = max(-clip_v, min(clip_v, combined))

                iter_eq_hits_total += eq_hits_k
                iter_cite_hits_total += cite_hits_k
                if saturated_k:
                    iter_saturated_count += 1

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
                "mean_t_lp_raw": float(np.mean(t_lp_seq)),
                "mean_t_lp_eff": float(np.mean(t_lp_eff)),
                "mean_s_lp": float(np.mean(s_lp)),
                "mean_adv": float(np.mean(per_tok_adv)),
                "stop_reason": seq.stop_reason,
                "plan_text": decoded_plans[k] if k < len(decoded_plans) else "",
                "solution_mask_active_tokens": n_mask_active,
                "solution_mask_density": mask_density,
                "trust_region_alpha": config.trust_region_alpha,
                "grounding_eq_hits": eq_hits_k,
                "grounding_cite_hits": cite_hits_k,
                "grounding_saturated": saturated_k,
                "sampled_by": "student",
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

        mean_adv = float(np.mean(all_advs)) if all_advs else 0.0
        pos_frac = _safe_pos_frac(all_advs)
        mean_abs = _safe_mean_abs(all_advs)
        n_plans = len(datums)
        logger.info(
            "Iter %d V9g: %d datums, mean_adv=%.4f, pos_frac=%.3f, "
            "eq_hits=%.1f, cite_hits=%.1f, saturated=%d, train=%.1fs",
            iter_idx, n_plans, mean_adv, pos_frac,
            iter_eq_hits_total / max(1, n_plans),
            iter_cite_hits_total / max(1, n_plans),
            iter_saturated_count, t_step,
        )

        # ----- WRITE BUFFER -----
        with open(buffer_path, "a") as f:
            for rec in per_plan_records:
                f.write(json.dumps(rec) + "\n")

        # ----- EVAL AUDIT (ASYNC) -----
        if iter_idx % config.eval_every == 0:
            eval_future = sampling_client.sample(
                prompt=student_input, num_samples=config.n_eval_plans,
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
                goal_id=f"{config.goal_domain}/{config.goal_name}",
            )
            audit_client.submit(iter_idx, audit_payload)
            eval_path = log_dir / "eval_rollouts.jsonl"
            with open(eval_path, "a") as f:
                for ep in eval_plans:
                    f.write(json.dumps({"iter": iter_idx, **ep}) + "\n")

        # ----- METRICS -----
        n_plans_with_sol = sum(
            1 for r in per_plan_records if r["solution_mask_density"] > 0.0
        )
        with open(metrics_path, "a") as f:
            f.write(json.dumps({
                "iter": iter_idx,
                "n_datums": len(datums),
                "mean_adv": mean_adv,
                "pos_frac": pos_frac,
                "mean_abs_adv": mean_abs,
                "wall_total_sec": time.time() - t0,
                "wall_sample_sec": t_sample,
                "wall_critic_sec": t_crit,
                "wall_logprob_sec": t_lp,
                "wall_frozen_logprob_sec": t_frozen_lp,
                "wall_train_sec": t_step,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "critique_was_cold_start": critique_current == COLD_START_CRITIQUE_NONE,
                "solution_only_mask": bool(config.solution_only_mask),
                "trust_region_alpha": config.trust_region_alpha,
                "loss_fn_name": config.loss_fn_name,
                "ppo_clip_eps": config.ppo_clip_eps,
                "use_grounding": bool(config.use_grounding),
                "grounding_eq_bonus": iter_eq_bonus,
                "grounding_cite_bonus": iter_cite_bonus,
                "grounding_eq_hits_mean": iter_eq_hits_total / max(1, n_plans),
                "grounding_cite_hits_mean": iter_cite_hits_total / max(1, n_plans),
                "grounding_saturated_count": iter_saturated_count,
                "n_plans_with_solution_tag": n_plans_with_sol,
            }) + "\n")

        # ----- EARLY-STOP GUARD -----
        try:
            audit_resp_dir = log_dir / "audit_responses"
            if audit_resp_dir.exists():
                done_iters = sorted(
                    int(p.stem.replace("iter_", ""))
                    for p in audit_resp_dir.glob("iter_*.json")
                    if "_plan_" not in p.name and ".partial" not in p.name
                )
                if len(done_iters) >= 2:
                    means = []
                    for it_done in done_iters[-2:]:
                        rp = audit_resp_dir / f"iter_{it_done:03d}.json"
                        try:
                            d = json.loads(rp.read_text())
                            judg = d.get("judgments", {})
                            aggs = [
                                j.get("aggregate_reward")
                                for j in judg.values()
                                if j.get("aggregate_reward") is not None
                            ]
                            if aggs:
                                means.append(float(np.mean(aggs)))
                        except Exception:
                            pass
                    if len(means) == 2 and (means[0] - means[1]) >= config.audit_drop_threshold:
                        logger.error(
                            "Iter %d: audit drop %.4f→%.4f exceeds threshold; halting",
                            iter_idx, means[0], means[1],
                        )
                        checkpoint_utils.save_checkpoint(
                            training_client=training_client,
                            name=f"early_stopped_iter_{iter_idx:03d}_{config.today_date}",
                            log_path=str(log_dir), kind="both",
                            loop_state={"batch": iter_idx + 1, "early_stopped": True},
                        )
                        return
        except Exception as e:
            logger.warning("Iter %d: early-stop check failed: %s", iter_idx, e)

        logger.info("Iter %d done in %.1fs", iter_idx, time.time() - t0)

    # Final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=f"final_{config.today_date}",
        log_path=str(log_dir),
        kind="both",
        loop_state={"batch": config.n_iter},
    )
    logger.info("Collecting audit responses...")
    audit_client.collect_all(
        out_path=log_dir / "audit_log.jsonl", timeout_sec=1800.0, poll_interval_sec=15.0,
    )
    logger.info("D6 OPD-grounded run complete.")


if __name__ == "__main__":
    config = chz.entrypoint(Config)
    main(config)
