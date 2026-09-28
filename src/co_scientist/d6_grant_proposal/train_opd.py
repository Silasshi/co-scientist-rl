"""D6 OPD trainer — on-policy distillation pipeline for grant proposal generation.

On-policy distillation (OPD) with privileged-Opus reviewer. The simplified
recipe shipped for the ICLR-workshop paper has three components:

  (1) Solution-only token mask (zeroes advantage outside <solution>...</solution>)
  (2) Privileged-reviewer teacher–student logprob delta:
        - Student rollout under context = (goal + slim_oracle)
        - Reviewer (Opus 4.7) sees the rollout PLUS reference_proposal.md
          as privileged ground truth and emits a structured <critique>
        - Teacher logprobs computed on student tokens under context =
          (goal + slim_oracle + critique)
        - Per-token advantage = clip(teacher_lp - student_lp, ±opd_anchor_clip)
  (3) PPO-clip loss with clip_eps=0.2

The earlier V8 cell included a trust-region interpolation toward the
frozen base policy (α=0.05). Removed 2026-04-30 — it was the best cell
in D5's TTT-Discover ablation grid, not validated on grant proposals,
and added a hard-to-motivate regularizer to the methods section. See
DECISIONS.md for the removal rationale.

Grant-proposal-specific adaptations:
  - Goal/oracle loaded from domain-organized paths: dataset/{domain}/{goal_name}/
  - Reviewer (Opus critic) receives reference_proposal.md as privileged context
  - Plan footer: 1500 target / 2000 max words (OPD standard)
  - Max tokens: 8192 (longer plans than V4's 4096)
  - Score scale: D4 12-signal weighted mean (NOT D5's /45)

Daemon must be running in a separate Claude Code window:
  projects/d6_grant_proposal/scripts/reviewer_subagent.md

Usage:
    source shared/tools/use_api_profile.sh new
    PYTHONPATH=src python -m co_scientist.d6_grant_proposal.train_opd \\
        config.goal_domain=ai \\
        config.goal_name=02_foundational_rl \\
        config.log_path=projects/d6_grant_proposal/runs/2026_04_30_pilot_foundrl_opd \\
        config.n_iter=16
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)

REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    api_profile: str | None = "new"
    base_url: str | None = None

    # Goal selection (domain-organized paths)
    goal_domain: str = "ai"          # "ai" | "natural_science" | "social_science"
    goal_name: str = "02_foundational_rl"

    # Base paths
    log_path: str = "projects/d6_grant_proposal/runs/2026_04_30_pilot_foundrl_opd"
    dataset_base: str = "projects/d6_grant_proposal/dataset"
    data_base: str = "projects/d6_grant_proposal/data"

    # Model
    model_name: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 64
    learning_rate: float = 5e-5

    # Sampling — 8192 because OPD's 1500/2000-word footer produces longer plans
    n_plans: int = 8
    max_tokens: int = 8192
    temperature: float = 1.0
    top_p: float = 0.95

    # SDPO advantage (legacy knob names preserved for config compat)
    sdpo_scale: float = 1.0
    opd_anchor_clip: float = 5.0

    # Loop
    n_iter: int = 16
    eval_every: int = 1
    n_eval_plans: int = 8
    save_every: int = 1
    n_grad_steps_per_iter: int = 4
    audit_drop_threshold: float = 5.0

    # Critic
    critic_timeout_sec: float = 900.0

    # Continual SDPO (future use)
    init_state_path: str = ""
    reset_optimizer_state: bool = False

    # Misc
    today_date: str = "2026_04_29"
    seed: int = 42

    # ===== OPD components (paper recipe) =====
    solution_only_mask: bool = True
    loss_fn_name: Literal["importance_sampling", "ppo"] = "ppo"
    ppo_clip_eps: float = 0.2


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _goal_dir(config: Config) -> Path:
    return _resolve(f"{config.dataset_base}/{config.goal_domain}/{config.goal_name}")


def _oracle_path(config: Config) -> Path:
    return _resolve(f"{config.data_base}/{config.goal_domain}/{config.goal_name}/oracle/slim.md")


def _decode_plan(seq, tokenizer, renderer) -> str:
    parsed = renderer.parse_response(seq.tokens)
    content = parsed[0].get("content", "") if parsed else ""
    return extract_solution(content) if content else tokenizer.decode(seq.tokens)


def _safe_pos_frac(advs: list[float]) -> float:
    if not advs:
        return 0.0
    return float(np.mean([1.0 if a > 0 else 0.0 for a in advs]))


def _safe_mean_abs(advs: list[float]) -> float:
    if not advs:
        return 0.0
    return float(np.mean([abs(a) for a in advs]))


# =============================================================================
# Solution-only token mask — port from D5 train_mu_v8_d5sdpo.py
# =============================================================================

def _span_overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def find_solution_content_span(text: str) -> tuple[int, int] | None:
    """Return char-span of last <solution> body not overlapping closed <think>."""
    think_spans = [
        (m.start(), m.end())
        for m in re.finditer(r"<think\b[^>]*>.*?</think>", text,
                             flags=re.DOTALL | re.IGNORECASE)
    ]
    solution_matches = list(
        re.finditer(r"<solution\b[^>]*>(.*?)</solution>", text,
                    flags=re.DOTALL | re.IGNORECASE)
    )
    if not solution_matches:
        return None
    candidate_spans: list[tuple[int, int]] = []
    for match in solution_matches:
        span = (match.start(1), match.end(1))
        if any(_span_overlaps(span, ts) for ts in think_spans):
            continue
        start, end = span
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            candidate_spans.append((start, end))
    if not candidate_spans:
        return None
    return candidate_spans[-1]


def build_solution_token_mask_from_tokens(
    *, tokenizer, sample_tokens: list[int],
) -> "np.ndarray":
    """Mask of 1.0 for tokens inside the last unclosed-by-think <solution>."""
    if not sample_tokens:
        return np.zeros(0, dtype=np.float32)
    try:
        decoded = tokenizer.decode(
            sample_tokens, skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except Exception:
        return np.zeros(len(sample_tokens), dtype=np.float32)
    span = find_solution_content_span(decoded)
    if span is None:
        return np.zeros(len(sample_tokens), dtype=np.float32)
    try:
        start_char, end_char = span
        start_tok = len(tokenizer.encode(decoded[:start_char], add_special_tokens=False))
        end_tok = len(tokenizer.encode(decoded[:end_char], add_special_tokens=False))
    except Exception:
        return np.zeros(len(sample_tokens), dtype=np.float32)
    start_tok = max(0, min(start_tok, len(sample_tokens)))
    end_tok = max(start_tok, min(end_tok, len(sample_tokens)))
    mask = np.zeros(len(sample_tokens), dtype=np.float32)
    if end_tok > start_tok:
        mask[start_tok:end_tok] = 1.0
    return mask


# =============================================================================
# Main loop
# =============================================================================

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

    # Load goal-specific inputs
    goal_dir = _goal_dir(config)
    goal = (goal_dir / "research_goal.md").read_text().strip()
    reference_proposal_md = (goal_dir / "reference_proposal.md").read_text().strip()

    oracle_path = _oracle_path(config)
    if not oracle_path.exists():
        raise FileNotFoundError(
            f"Slim oracle not found at {oracle_path}. "
            f"Run extract_oracle.py and extract_oracle_slim.py first:\n"
            f"  PYTHONPATH=src python -m co_scientist.d6_grant_proposal.extract_oracle "
            f"goal_domain={config.goal_domain} goal_name={config.goal_name}\n"
            f"  PYTHONPATH=src python -m co_scientist.d6_grant_proposal.extract_oracle_slim "
            f"goal_domain={config.goal_domain} goal_name={config.goal_name}"
        )
    oracle = oracle_path.read_text().strip()

    logger.info("D6 OPD inputs:")
    logger.info("  goal_domain:    %s", config.goal_domain)
    logger.info("  goal_name:      %s", config.goal_name)
    logger.info("  goal:           %d chars", len(goal))
    logger.info("  oracle (slim):  %d chars", len(oracle))
    logger.info("  ref proposal:   %d chars (privileged for critic)", len(reference_proposal_md))
    logger.info("  log_dir:        %s", log_dir)
    logger.info(
        "OPD components: solution_only_mask=%s loss_fn=%s ppo_clip_eps=%.2f",
        config.solution_only_mask, config.loss_fn_name, config.ppo_clip_eps,
    )

    cfg_path = log_dir / "config.json"
    cfg_path.write_text(json.dumps(
        {k: getattr(config, k) for k in dir(config)
         if not k.startswith("_") and not callable(getattr(config, k))},
        default=str, indent=2
    ))

    # Tokenizer + renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info("Renderer: %s", renderer_name)

    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )

    # Training client
    last_checkpoint = checkpoint_utils.get_last_checkpoint(str(log_dir))
    if last_checkpoint is not None:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            last_checkpoint["state_path"]
        )
        start_iter = last_checkpoint.get("batch", 0) + 1
        logger.info("Resuming from iter %d (state=%s)", start_iter, last_checkpoint["state_path"])
    elif config.init_state_path:
        if config.reset_optimizer_state:
            training_client = service_client.create_training_client_from_state(config.init_state_path)
            logger.info("Init from external state, FRESH Adam: %s", config.init_state_path)
        else:
            training_client = service_client.create_training_client_from_state_with_optimizer(
                config.init_state_path
            )
            logger.info("Init from external state: %s", config.init_state_path)
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

        # Current LoRA sampling client
        sp_path = training_client.save_weights_for_sampler(
            name=f"iter_{iter_idx:04d}"
        ).result().path
        sampling_client = service_client.create_sampling_client(model_path=sp_path)
        with open(checkpoints_path, "a") as f:
            f.write(json.dumps({"iter": iter_idx, "sampler_path": sp_path,
                                 "ts": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")

        # ----- STUDENT ROLLOUT (on-policy under student context) -----
        student_text = build_student_prompt(goal, oracle)
        student_convo = [{"role": "user", "content": student_text}]
        student_input = renderer.build_generation_prompt(student_convo)
        student_tokens = student_input.to_ints()

        t_sample0 = time.time()
        student_future = sampling_client.sample(
            prompt=student_input,
            num_samples=config.n_plans,
            sampling_params=sampling_params,
        )
        student_result = student_future.result()
        t_sample = time.time() - t_sample0
        logger.info("Iter %d: STUDENT sampled %d plans in %.1fs",
                     iter_idx, len(student_result.sequences), t_sample)

        # Decode plans now (needed for critic)
        decoded_plans = []
        for k, seq in enumerate(student_result.sequences):
            try:
                decoded_plans.append(_decode_plan(seq, tokenizer, renderer))
            except Exception as e:
                logger.warning("Iter %d plan %d: decode failed: %s", iter_idx, k, e)
                decoded_plans.append("")

        # ----- CRITIC ON CURRENT ROLLOUT (BLOCKING) -----
        # Paper-faithful (Hübotter SDPO §Algorithm 1): critic fires BEFORE teacher
        # logprob compute, on THIS iter's student rollout (not stale prev-iter critique).
        critique_current = COLD_START_CRITIQUE_NONE
        if decoded_plans and any(decoded_plans):
            valid_idxs = [i for i, t in enumerate(decoded_plans) if t]
            chosen_idx = rng.choice(valid_idxs)
            chosen_plan_id = f"iter_{iter_idx:03d}_pick_{chosen_idx}"
            payload = build_critic_request_payload(
                iter_idx=iter_idx,
                goal=goal,
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
            logger.info("Iter %d: critic (CURRENT rollout) %.1fs, len=%d chars, pick=%d",
                         iter_idx, t_crit, len(critique_current), chosen_idx)
        else:
            logger.warning("Iter %d: no decoded plans; cold-start critique", iter_idx)

        # ----- TEACHER CONTEXT WITH CURRENT-ROLLOUT CRITIQUE -----
        teacher_text = build_teacher_prompt(goal, oracle, critique_current)
        teacher_convo = [{"role": "user", "content": teacher_text}]
        teacher_input = renderer.build_generation_prompt(teacher_convo)
        teacher_tokens_prefix = teacher_input.to_ints()

        # ----- TEACHER LOGPROBS on student tokens under teacher context -----
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

        # ----- BUILD DATUMS -----
        datums = []
        all_advs = []
        per_plan_records = []
        for k, seq in enumerate(student_result.sequences):
            gen = list(seq.tokens)
            sampler_lp_seq = list(seq.logprobs) if seq.logprobs else []
            full_t_lp = teacher_lps_full[k]
            t_lp_seq = list(
                full_t_lp[len(teacher_tokens_prefix):len(teacher_tokens_prefix) + len(gen)]
            )
            if not gen or len(sampler_lp_seq) != len(gen) or len(t_lp_seq) != len(gen):
                logger.warning(
                    "Iter %d plan %d: len mismatch gen=%d sampler_lp=%d t_lp=%d; skipping",
                    iter_idx, k, len(gen), len(sampler_lp_seq), len(t_lp_seq),
                )
                continue

            s_lp = sampler_lp_seq

            clip_v = config.opd_anchor_clip
            per_tok_adv = []
            for t_, s_ in zip(t_lp_seq, s_lp):
                a = float(t_ - s_) * config.sdpo_scale
                a = max(-clip_v, min(clip_v, a))
                per_tok_adv.append(a)

            # solution-only token mask
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
                else:
                    logger.warning(
                        "Iter %d plan %d: solution mask shape %d != gen %d; skipping mask",
                        iter_idx, k, sol_mask.size, len(gen),
                    )

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
                "plan_text": decoded_plans[k] if k < len(decoded_plans) else "",
                "solution_mask_active_tokens": n_mask_active,
                "solution_mask_density": mask_density,
                "sampled_by": "student",
            })

        # ----- TRAIN STEP (PPO-clip or IS loss) -----
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
            n_done = 0
            for chunk in chunks[:n_steps]:
                if not chunk:
                    continue
                fb_fut = training_client.forward_backward(chunk, **loss_fn_kwargs)
                os_fut = training_client.optim_step(adam_params)
                fb_fut.result()
                os_fut.result()
                n_done += 1
            logger.info("Iter %d: ran %d microbatch grad steps (loss_fn=%s, chunk_size=%d)",
                         iter_idx, n_done, config.loss_fn_name, chunk_size)
        t_step = time.time() - t_step0

        mean_adv = float(np.mean(all_advs)) if all_advs else 0.0
        pos_frac = _safe_pos_frac(all_advs)
        mean_abs = _safe_mean_abs(all_advs)
        logger.info(
            "Iter %d OPD: %d datums, mean_adv=%.4f, pos_frac=%.3f, mean|A|=%.4f, train=%.1fs",
            iter_idx, len(datums), mean_adv, pos_frac, mean_abs, t_step,
        )

        # ----- WRITE BUFFER -----
        with open(buffer_path, "a") as f:
            for rec in per_plan_records:
                f.write(json.dumps(rec) + "\n")

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
                iter_idx=iter_idx,
                goal=goal,
                plans=eval_plans,
                goal_id=f"{config.goal_domain}/{config.goal_name}",
            )
            audit_client.submit(iter_idx, audit_payload)
            t_eval = time.time() - t_eval0
            eval_path = log_dir / "eval_rollouts.jsonl"
            with open(eval_path, "a") as f:
                for ep in eval_plans:
                    f.write(json.dumps({"iter": iter_idx, **ep}) + "\n")
            logger.info("Iter %d: eval audit submitted (%d plans, %.1fs sample, async)",
                         iter_idx, len(eval_plans), t_eval)

        # ----- METRICS -----
        n_plans_with_solution = sum(
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
                "wall_train_sec": t_step,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "critique_current_len": len(critique_current),
                "critique_was_cold_start": critique_current == COLD_START_CRITIQUE_NONE,
                "solution_only_mask": bool(config.solution_only_mask),
                "loss_fn_name": config.loss_fn_name,
                "ppo_clip_eps": config.ppo_clip_eps,
                "n_plans_with_solution_tag": n_plans_with_solution,
                "mean_solution_mask_density": (
                    float(np.mean([r["solution_mask_density"] for r in per_plan_records]))
                    if per_plan_records else 0.0
                ),
            }) + "\n")

        if not all_advs:
            logger.warning("Iter %d: no advantages computed; check len mismatches.", iter_idx)

        # ----- EARLY-STOP GUARD (audit aggregate drop) -----
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
                            "Iter %d: audit drop %.4f → %.4f exceeds threshold %.2f; halting",
                            iter_idx, means[0], means[1], config.audit_drop_threshold,
                        )
                        checkpoint_utils.save_checkpoint(
                            training_client=training_client,
                            name=f"early_stopped_iter_{iter_idx:03d}_{config.today_date}",
                            log_path=str(log_dir),
                            kind="both",
                            loop_state={"batch": iter_idx + 1, "early_stopped": True},
                        )
                        logger.info("Early-stopped checkpoint saved.")
                        return
        except Exception as e:
            logger.warning("Iter %d: early-stop check failed: %s", iter_idx, e)

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

    logger.info("Collecting audit responses (waiting up to 30 min for final iters)...")
    audit_summary = audit_client.collect_all(
        out_path=log_dir / "audit_log.jsonl",
        timeout_sec=1800.0,
        poll_interval_sec=15.0,
    )

    valid = [s for s in audit_summary if s.get("mean_aggregate") is not None]
    if valid:
        last3 = valid[-3:]
        mu_proxy = float(np.mean([s["mean_aggregate"] for s in last3]))
        logger.info("=== D6 OPD DECISION MATRIX PREVIEW ===")
        logger.info("OPD proxy (mean of last %d audit means): %.4f (D4 aggregate, 0-1)", len(last3), mu_proxy)
        logger.info("baseline (frozen 30B + oracle): TBD after baseline run")
        logger.info("Decision: OPD > baseline + 0.02 → continue; OPD ≤ baseline → diagnose")
        logger.info("=====================================")
    else:
        logger.warning("No valid audit responses yet — run audit.py later.")

    logger.info("D6 OPD run complete.")


if __name__ == "__main__":
    config = chz.entrypoint(Config)
    main(config)
