"""D5 μ-v7 OPD trainer — D5 in-house on-policy IS-loss variant (closest to OPSD Zhao 2601.18734 sampled-token PG).

ALGORITHM TAXONOMY (verified 2026-04-27 PM via paper-grade WebFetch; see
`projects/d5_abstract_retrieve_refine/knowledge/current/CANONICAL_NAMING_REFERENCE.md`):

  - This trainer (opd_mode=True): STUDENT samples on-policy under no-critique
    context; TEACHER computes logprobs on student tokens under critique context;
    A_t = clamp(scale * (teacher_lp - student_lp), [-clip, +clip]);
    importance_sampling loss on student_input + student-sampled tokens.
    → **D5 in-house on-policy IS-loss variant**.
    → Closest published comparable: OPSD (Zhao "Self-Distilled Reasoner",
      arxiv 2601.18734) Table 3 sampled-token policy-gradient variant
      (~2% gap from full-vocab JS in Zhao's reported results). Not a faithful
      reimplementation: OPSD's canonical default is full-vocab JS divergence;
      we use scalar-logprob IS-loss instead. Gradient direction is equivalent
      up to clamp/scale.
    → NOT canonical Hübotter SDPO (arxiv 2601.20802), which uses full-vocab
      / top-K vocab KL loss with student on-policy.

  - μ-v2/v3/v4/v6/κ-v1 (the prior trainer family): TEACHER samples + STUDENT
    lp recomputed + IS-loss → **D5 in-house off-policy IS-loss variant**.
    → Historically self-labeled "SDPO" / "option (c) HER from Hübotter 2026";
      the Hübotter paper has no such variant — that label was D5 internal
      speculation. NOT canonical SDPO and NOT canonical OPSD.

This trainer was developed to address the architectural bug isolated by Stage A +
Stage B (2026-04-27): the off-policy variant samples plans from the TEACHER
(with critique context) and computes STUDENT logprobs on those off-trajectory
tokens. It transfers style but not specific content (Stage B: μ-v4 EVAL is
0/8 PUCT; Arm B critique-cued EVAL still 0/8 PUCT — defers to oracle prefix
tokens like SIFT/GRPO). The on-policy variant samples STUDENT under no-critique
context and computes TEACHER logprobs on student tokens under critique context.
Gradient flows through tokens student actually produces → on-manifold updates
that fix F1 (prefix-prior lock-in) + F2 (off-manifold gradient). Combined
pairwise 45/48 = 93.75% wins anonymized over μ-v4 production (F13).

Per iter:
  1. sample N plans under STUDENT context (goal + oracle, NO critique) — ON-POLICY
  2. compute teacher logprobs of those tokens under TEACHER context (goal + oracle + critique)
  3. advantage A_t = clamp(scale * (teacher_lp - student_lp), [-clip, +clip])
     (same formula as the off-policy variant, but student_lp is now from
     on-policy seq.logprobs and teacher_lp is from compute_logprobs under
     teacher_input)
  4. forward_backward(loss_fn="importance_sampling") + optim_step
     (datum is on student_input + student-sampled tokens; on-manifold)
  5. pick 1 plan, ship to plan-critic subagent, parse critique XML for next iter
  6. iter 0 uses COLD_START_CRITIQUE_NONE

Tinker API constraint: full-vocab logits NOT exposed. We use per-token scalar
logprob from compute_logprobs (the IS-loss form). To faithfully implement
canonical OPSD's default full-vocab JS or canonical Hübotter SDPO's full-vocab
top-K KL would require a different inference backend.

Reference plan: /home/silas/.claude/plans/snug-dreaming-yao.md (μ-OPD redesign).

Usage:
    source shared/tools/use_api_profile.sh new
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.train_mu_v7_opd \
        log_path=projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v7_opd_smoke \
        n_iter=4 eval_every=1 save_every=1

Daemon must be running in a separate Claude Code window with the prompt at
  projects/d5_abstract_retrieve_refine/scripts/opus_critic_audit_daemon.md

Compatibility flag: set `opd_mode=False` to fall back to legacy option (c) HER
behavior for ablation. Default True for this trainer.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
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
from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v2 import (
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
        "projects/d5_abstract_retrieve_refine/runs/2026_04_27_mu_v4"
    )
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"
    reference_plan_path: str = (
        "projects/d5_abstract_retrieve_refine/dataset/reference_solution.txt"
    )
    # Phase 5α: anchor goal for anchor_ce regularization. When anchor_ce_weight > 0,
    # CE loss is computed on (anchor_goal + oracle | reference_plan) — anchors the
    # student distribution toward TTT-D's reference plan style while learning the
    # current task (config.goal_path) via SDPO. Empty = use config.goal_path
    # (degenerate; no continual-anchoring effect).
    anchor_goal_path: str = ""
    oracle_abstraction_path: str = (
        "projects/d5_abstract_retrieve_refine/data/oracles/oracle_v2_2026_04_26_build/slim.md"
    )
    source_paper_path: str = (
        "projects/d5_abstract_retrieve_refine/data/source_paper/v2.md"
    )

    # Model
    model_name: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 64
    learning_rate: float = 5e-5

    # Sampling (Phase 0.5c sampling fix: temp=1.0 + top_p, NO seed)
    n_plans: int = 8
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.95

    # SDPO
    sdpo_scale: float = 1.0
    sdpo_clip_advantage: float = 5.0
    anchor_ce_weight: float = 0.0  # off per user decision (pure SDPO ablation)

    # Loop
    n_iter: int = 20
    eval_every: int = 1  # v4: eval EVERY iter for full learning curve
    n_eval_plans: int = 8
    save_every: int = 1
    # v3-specific: split 8 datums into n_grad_steps_per_iter microbatches for finer-grained Adam updates
    n_grad_steps_per_iter: int = 4
    # v4: early-stop guard. If rolling mean of last 2 audits drops by >= this many points, halt.
    audit_drop_threshold: float = 3.0

    # Critic
    critic_timeout_sec: float = 900.0  # 15 min per critic call
    skip_critic_at_iter0: bool = False  # if True, do not call critic in iter 0
                                         # (since cold-start is already in use)
    skip_critic_always: bool = False  # if True, NEVER call critic; prev_critique
                                       # stays at cold-start every iter. Used for
                                       # F4 falsifier probe (no daemon needed).

    # Misc
    today_date: str = "2026_04_25"
    seed: int = 42  # for plan-pick randomness (NOT for sampling)

    # Continual SDPO (Phase 5): tinker:// LoRA-state URL to seed from when log_dir
    # has no resume checkpoint. Empty = fresh LoRA. On a partial-run resume, the
    # existing checkpoint takes precedence over this field (correct behavior).
    init_state_path: str = ""
    # B-plan when init_state_path triggers cliff (μ-v5_v1 cliff at iter 1):
    # if True, load WEIGHTS ONLY from init_state_path (drop Adam moments) so
    # Adam state is fresh on the new loss surface. Default False = inherit
    # optimizer state (matches μ-v4's recipe).
    reset_optimizer_state: bool = False

    # F4 falsifier probe (inherited from train_mu_v4 fork; left dormant in v7).
    # When True: zero per-token advantage on STYLISTIC tokens. Documented in F11.
    f4_probe_content_only_mask: bool = False

    # ===== μ-OPD specific config =====
    # opd_mode: when True, swap teacher↔student sampling (student samples on-policy
    # under student_input; teacher computes logprobs on student tokens under
    # teacher_input). When False, fall back to legacy option (c) HER (teacher
    # samples, student supervised) for ablation / sanity comparison.
    # See plan: /home/silas/.claude/plans/snug-dreaming-yao.md (μ-OPD).
    opd_mode: bool = True
    # Advantage clip in OPD mode (Option A: A_t = clamp(scale*(t_lp - s_lp), [-clip, +clip]))
    opd_anchor_clip: float = 5.0


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


# F4 falsifier probe (Day 1, temporary; revert when probe done).
# Coarse content-token regex used to test whether SDPO advantage flowing only
# through "content" tokens lifts EVAL-plan PUCT/J_β presence above F4's
# documented 0/8. Superset (false-positive direction conservative) — catches
# digits, math operators, Greek glyphs (full alphabet), Greek-letter names,
# and callable forms like Q(s)/J(θ).
_F4_CONTENT_CHARS_RE = re.compile(r"[0-9=()\[\]/\\*+\-_·∇√]")
_F4_CONTENT_GREEK_RE = re.compile(
    r"[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]"
)
_F4_CONTENT_CALL_RE = re.compile(r"[A-Z][a-z]?\(")
_F4_CONTENT_WORDS_RE = re.compile(
    r"\b(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|"
    r"omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)\b",
    re.IGNORECASE,
)


def is_content_token(tok_id: int, tokenizer) -> bool:
    """F4 probe coarse content-token classifier. See module-level comment."""
    try:
        text = tokenizer.decode([tok_id])
    except Exception:
        return False
    if not text:
        return False
    if _F4_CONTENT_CHARS_RE.search(text):
        return True
    if _F4_CONTENT_GREEK_RE.search(text):
        return True
    if _F4_CONTENT_CALL_RE.search(text):
        return True
    if _F4_CONTENT_WORDS_RE.search(text):
        return True
    return False


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

    # Phase 5α: precompute anchor CE prompt + ref_plan tokens (used per-iter when anchor_ce_weight > 0)
    # CE term anchors model distribution to (anchor_goal + oracle → ref_plan) — typically
    # anchor_goal = TTT-D goal during continual SDPO from a μ-v4 anchor.
    anchor_prompt_tokens: list = []
    ref_plan_tokens: list = []
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
    if last_checkpoint is not None:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            last_checkpoint["state_path"]
        )
        start_iter = last_checkpoint.get("batch", 0) + 1
        logger.info("Resuming from iter %d (state=%s)", start_iter, last_checkpoint["state_path"])
    elif config.init_state_path:
        # Phase 5 continual SDPO: seed from prior μ-v* run's LoRA-state path.
        if config.reset_optimizer_state:
            # Weights-only init (B-plan): drop Adam moments to avoid stale-moment
            # cliff dynamics on the new loss surface (μ-v5_v1 cliff diagnosis).
            training_client = service_client.create_training_client_from_state(
                config.init_state_path
            )
            logger.info("Initializing from external state, FRESH Adam (continual SDPO B-plan): %s",
                         config.init_state_path)
        else:
            training_client = service_client.create_training_client_from_state_with_optimizer(
                config.init_state_path
            )
            logger.info("Initializing from external state (continual SDPO): %s",
                         config.init_state_path)
        start_iter = 0
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank,
        )
        start_iter = 0
        logger.info("Fresh training client (LoRA rank=%d)", config.lora_rank)

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
    # Phase 5α: pre-tokenize anchor_ce prompt + ref_plan (when feature enabled)
    # ===========================================================================
    if config.anchor_ce_weight > 0:
        anchor_goal_text = (
            _resolve(config.anchor_goal_path).read_text().strip()
            if config.anchor_goal_path else goal
        )
        anchor_text = build_student_prompt(anchor_goal_text, oracle)
        anchor_convo = [{"role": "user", "content": anchor_text}]
        anchor_input = renderer.build_generation_prompt(anchor_convo)
        anchor_prompt_tokens = anchor_input.to_ints()
        ref_plan_tokens = tokenizer.encode(ref_plan)
        logger.info(
            "Phase 5α anchor_ce: prompt=%d tokens, ref_plan=%d tokens, weight=%.4f, "
            "anchor_goal_path=%s (%s)",
            len(anchor_prompt_tokens), len(ref_plan_tokens), config.anchor_ce_weight,
            config.anchor_goal_path or "(falling back to goal_path)",
            "TTT-D anchor" if config.anchor_goal_path else "DEGENERATE same-goal",
        )

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

        # ----- ROLLOUT + COMPLEMENT-CONTEXT LOGPROB COMPUTE -----
        # OPD mode (default): student samples on-policy under student_input;
        #   teacher provides per-token logprob supervision under teacher_input.
        # Legacy (c) HER mode (opd_mode=False): teacher samples; student
        #   supervised. Identical to old train_mu_v4 behavior.
        student_text = build_student_prompt(goal, oracle)
        student_convo = [{"role": "user", "content": student_text}]
        student_input = renderer.build_generation_prompt(student_convo)
        student_tokens = student_input.to_ints()

        teacher_text = build_teacher_prompt(goal, oracle, prev_critique)
        teacher_convo = [{"role": "user", "content": teacher_text}]
        teacher_input = renderer.build_generation_prompt(teacher_convo)
        teacher_tokens_prefix = teacher_input.to_ints()

        t_sample0 = time.time()
        if config.opd_mode:
            # On-policy: student samples
            sampler_future = sampling_client.sample(
                prompt=student_input,
                num_samples=config.n_plans,
                sampling_params=sampling_params,
            )
            sampler_result = sampler_future.result()
            t_sample = time.time() - t_sample0
            logger.info("Iter %d: STUDENT (on-policy) sampled %d plans in %.1fs",
                         iter_idx, len(sampler_result.sequences), t_sample)
        else:
            # Legacy (c) HER: teacher samples
            sampler_future = sampling_client.sample(
                prompt=teacher_input,
                num_samples=config.n_plans,
                sampling_params=sampling_params,
            )
            sampler_result = sampler_future.result()
            t_sample = time.time() - t_sample0
            logger.info("Iter %d: TEACHER (off-policy HER) sampled %d plans in %.1fs",
                         iter_idx, len(sampler_result.sequences), t_sample)

        # Compute the COMPLEMENT-context logprob (whichever side did NOT sample).
        # In OPD mode: teacher_lp on student-sampled tokens.
        # In legacy (c): student_lp on teacher-sampled tokens.
        complement_prefix = teacher_tokens_prefix if config.opd_mode else student_tokens
        t_lp0 = time.time()
        complement_lp_futures = [
            sampling_client.compute_logprobs(
                types.ModelInput.from_ints(tokens=complement_prefix + list(seq.tokens))
            )
            for seq in sampler_result.sequences
        ]
        complement_lps_full = [f.result() for f in complement_lp_futures]
        t_lp = time.time() - t_lp0
        if config.opd_mode:
            logger.info("Iter %d: TEACHER logprobs (under teacher_input) on student tokens in %.1fs",
                         iter_idx, t_lp)
        else:
            logger.info("Iter %d: STUDENT logprobs (under student_input) on teacher tokens in %.1fs",
                         iter_idx, t_lp)

        # ----- BUILD OPD/SDPO DATUMS -----
        # In OPD mode: gen tokens are student-sampled; sampler.seq.logprobs IS
        #   the on-policy student_lp; complement_lp is teacher_lp.
        # In legacy (c): gen tokens are teacher-sampled; sampler.seq.logprobs IS
        #   teacher_lp; complement_lp is student_lp.
        # Advantage formula identical: A_t = clamp(scale·(teacher_lp − student_lp), [−clip, +clip])
        # Datum is built on the STUDENT prefix + sampled tokens (so the IS-loss
        # update is interpreted as "student_pi(token | student_prefix)"). In OPD
        # mode this is on-manifold; in (c) mode this is off-manifold (the bug).
        datums = []
        all_advs = []
        per_plan_records = []
        for k, (seq, full_complement_lp) in enumerate(
            zip(sampler_result.sequences, complement_lps_full)
        ):
            gen = list(seq.tokens)
            sampler_lp_seq = list(seq.logprobs) if seq.logprobs else []
            # full_complement_lp has length len(complement_prefix) + len(gen). Slice to gen-only.
            complement_lp_seq = list(
                full_complement_lp[len(complement_prefix):len(complement_prefix) + len(gen)]
            )
            if (not gen
                or len(sampler_lp_seq) != len(gen)
                or len(complement_lp_seq) != len(gen)):
                logger.warning(
                    "Iter %d plan %d: len mismatch gen=%d sampler_lp=%d complement_lp=%d; skipping",
                    iter_idx, k, len(gen), len(sampler_lp_seq), len(complement_lp_seq),
                )
                continue
            # Resolve teacher_lp / student_lp depending on which side sampled.
            if config.opd_mode:
                t_lp_seq = complement_lp_seq  # teacher under teacher_input
                s_lp = sampler_lp_seq          # on-policy student
            else:
                t_lp_seq = sampler_lp_seq      # teacher under teacher_input (sampled)
                s_lp = complement_lp_seq       # student under student_input (recomputed)

            content_mask: list[bool] | None = None
            if config.f4_probe_content_only_mask:
                content_mask = [is_content_token(tid, tokenizer) for tid in gen]
            per_tok_adv = []
            clip_v = (config.opd_anchor_clip if config.opd_mode
                      else config.sdpo_clip_advantage)
            for t_idx, (t_, s_) in enumerate(zip(t_lp_seq, s_lp)):
                a = float(t_ - s_) * config.sdpo_scale
                a = max(-clip_v, min(clip_v, a))
                if content_mask is not None and not content_mask[t_idx]:
                    a = 0.0
                per_tok_adv.append(a)
            all_advs.extend(per_tok_adv)
            n_content = int(sum(content_mask)) if content_mask is not None else 0
            # Datum is on STUDENT prefix + sampled tokens (on-manifold in OPD mode).
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
                "f4_probe_n_content": n_content,
                "f4_probe_mask_density": (n_content / len(gen)) if gen else 0.0,
                "sampled_by": "student" if config.opd_mode else "teacher",
            })

        # ----- TRAIN STEP (v3: n_grad_steps_per_iter microbatches per iter) -----
        t_step0 = time.time()
        if datums:
            n_steps = max(1, config.n_grad_steps_per_iter)
            chunk_size = max(1, len(datums) // n_steps)
            chunks = [datums[i:i + chunk_size] for i in range(0, len(datums), chunk_size)]
            n_done = 0
            for chunk in chunks[:n_steps]:
                if not chunk:
                    continue
                fb_fut = training_client.forward_backward(chunk, loss_fn="importance_sampling")
                os_fut = training_client.optim_step(adam_params)
                fb_fut.result()
                os_fut.result()
                n_done += 1
            logger.info("Iter %d: ran %d microbatch grad steps (chunk_size=%d)",
                         iter_idx, n_done, chunk_size)
        t_step = time.time() - t_step0

        # ----- Phase 5α: ANCHOR_CE REGULARIZATION STEP -----
        # Constant +anchor_ce_weight advantage on (anchor_prompt | ref_plan) gives
        # importance_sampling loss = -mean(w · log_pi) = w · CE on ref_plan tokens.
        if config.anchor_ce_weight > 0 and anchor_prompt_tokens and ref_plan_tokens:
            t_anch0 = time.time()
            try:
                anchor_lp_fut = sampling_client.compute_logprobs(
                    types.ModelInput.from_ints(
                        tokens=anchor_prompt_tokens + ref_plan_tokens
                    )
                )
                anchor_lps_full = anchor_lp_fut.result()
                anchor_lps = list(anchor_lps_full[
                    len(anchor_prompt_tokens):
                    len(anchor_prompt_tokens) + len(ref_plan_tokens)
                ])
                if len(anchor_lps) == len(ref_plan_tokens):
                    anchor_datum = build_sdpo_datum(
                        student_prompt_tokens=anchor_prompt_tokens,
                        gen_tokens=ref_plan_tokens,
                        student_logprobs=anchor_lps,
                        advantages=[config.anchor_ce_weight] * len(ref_plan_tokens),
                    )
                    fb_fut = training_client.forward_backward(
                        [anchor_datum], loss_fn="importance_sampling"
                    )
                    os_fut = training_client.optim_step(adam_params)
                    fb_fut.result()
                    os_fut.result()
                    t_anch = time.time() - t_anch0
                    logger.info(
                        "Iter %d: anchor_ce step done (w=%.3f, ref_lp_mean=%.3f, %.1fs)",
                        iter_idx, config.anchor_ce_weight,
                        float(np.mean(anchor_lps)), t_anch,
                    )
                else:
                    logger.warning(
                        "Iter %d: anchor_ce skipped — logprob length mismatch (%d vs %d)",
                        iter_idx, len(anchor_lps), len(ref_plan_tokens),
                    )
            except Exception as e:
                logger.warning("Iter %d: anchor_ce step failed: %s", iter_idx, e)

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
        if config.skip_critic_always:
            pass  # F4 falsifier probe: cold-start critique stays put across iters
        elif not (config.skip_critic_at_iter0 and iter_idx == 0):
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
        # F4 probe iter-aggregate mask density (only meaningful when flag is on)
        if config.f4_probe_content_only_mask and per_plan_records:
            f4_mask_density_iter = float(np.mean(
                [r["f4_probe_mask_density"] for r in per_plan_records]
            ))
            logger.info(
                "Iter %d F4 PROBE: mean mask_density=%.3f across %d plans (content tokens active in IS-loss)",
                iter_idx, f4_mask_density_iter, len(per_plan_records),
            )
        else:
            f4_mask_density_iter = None
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
                "f4_probe_active": bool(config.f4_probe_content_only_mask),
                "f4_probe_mask_density": f4_mask_density_iter,
                "opd_mode": bool(config.opd_mode),
                "opd_anchor_clip": config.opd_anchor_clip,
            }) + "\n")

        # ----- SAFETY RAIL: NaN advantages or trivially zero -----
        if not all_advs:
            logger.warning("Iter %d: no advantages computed; check len mismatches.", iter_idx)

        # ----- v4 EARLY-STOP GUARD: detect catastrophic audit drop -----
        try:
            audit_resp_dir = log_dir / "audit_responses"
            if audit_resp_dir.exists():
                done_iters = sorted(int(p.stem.replace("iter_", "")) for p in audit_resp_dir.glob("iter_*.json"))
                if len(done_iters) >= 2:
                    means = []
                    for it_done in done_iters[-2:]:
                        rp = audit_resp_dir / f"iter_{it_done:03d}.json"
                        try:
                            d = json.loads(rp.read_text())
                            judg = d.get("judgments", {})
                            totals = [j.get("total") or sum((j.get("scores") or {}).values()) for j in judg.values()]
                            totals = [t for t in totals if isinstance(t, (int, float))]
                            if totals:
                                means.append(sum(totals) / len(totals))
                        except Exception:
                            pass
                    if len(means) == 2:
                        drop = means[0] - means[1]
                        if drop >= config.audit_drop_threshold:
                            logger.error(
                                "Iter %d: EARLY STOP — audit rolling mean dropped %.2f (>= threshold %.2f). "
                                "Mean(prev)=%.2f, Mean(last)=%.2f. Saving final checkpoint and exiting.",
                                iter_idx, drop, config.audit_drop_threshold, means[0], means[1],
                            )
                            checkpoint_utils.save_checkpoint(
                                training_client=training_client,
                                name=f"early_stopped_iter_{iter_idx:03d}_{config.today_date}",
                                log_path=str(log_dir),
                                kind="both",
                                loop_state={"batch": iter_idx + 1, "early_stopped": True},
                            )
                            logger.info("Iter %d: early-stopped checkpoint saved. Done.", iter_idx)
                            return
        except Exception as e:
            logger.warning("Iter %d: audit early-stop check failed: %s", iter_idx, e)

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
