"""D5 μ-v9 KL-anchor trainer — Layer 2 (fork of v9-grounded).

Adds KL-anchor toward π_oracle on top of v9-grounded's verifier-grounded reward:
    A_total[t] = clamp(
        A_critique[t] + grounding_bonus[t] − kl_beta · (lp_θ[t] − lp_oracle[t]),
        ±opd_anchor_clip
    )

π_oracle is a frozen LoRA on top of base Qwen3-30B-A3B, SFT'd on 30
oracle-grounded plans (3 anchor groups × 10 each — see
`projects/d5_abstract_retrieve_refine/dataset/oracle_grounded_sft.jsonl` and
`train_pi_oracle_sft.py` for SFT recipe).

Layer 2 motivation: Layer 1 (sparse verifier bonus) only patches reward at
the token-cluster level. If verifier signal is brittle (sympy parse fails on
measure-subscript equations, fuzzy match Goodhart-prone), the gradient never
gets a distribution-level oracle prior. Layer 2 adds a KL term that pulls the
entire output distribution toward π_oracle — providing a soft "this is what
oracle-grounded plans look like" signal independent of verifier sparseness.

Lp(π_oracle) is computed via a separate sampling_client loaded from the SFT'd
LoRA's tinker:// uri. Same pattern as v8 trust-region frozen-base lp at L575-587.

For NeurIPS paper Section 4 ablation: Layer 1 (v9_grounded) vs Layer 2 (v9_kl_anchor)
vs G+ baseline. Layer 1 framed as lightweight verifier baseline; Layer 2 as
distribution-level oracle anchor (the main method).

---

Inherited v9-grounded docstring follows:

D5 μ-v9 grounded trainer — Layer 1 verifier-grounded reward (fork of v8-d5sdpo).

Adds sparse per-token grounding bonus to v8's critique-conditioned advantage:
    A_total[t] = clamp(A_critique[t] + bonus_if_in_grounded_span, ±opd_anchor_clip)

Bonus fires on tokens within plan-text spans where verifier detects gold
equation or citation match (tiered: sympy strict → sympy normalized → fuzzy
key-token match). Bonus magnitude (per_eq=0.1, per_cite=0.05) is calibrated
sparse — ~5-15% of solution tokens get bonus on G+ peak plans, matching
G+ mean|A|=0.15 at the token level so grounding doesn't blow PPO clip.

Per ABC verdict (committed d0d94ef):
- H16-1 FALSIFIED: Qwen3-30B can retrieve 4/4 verbatim
- H16-2 CORROBORATED: critique-conditioned advantage rewards critique-conformity,
  not oracle-fidelity. SDPO loss never has oracle as gradient target.
- H16-3 INDETERMINATE: RAG peak ≈ G+ peak within noise (Δ=-1.12)
- F17 NEW: RAG accelerates cliff by ~5 iter (oracle breadth = anti-collapse anchor)

v9 attacks H16-2 directly by giving SDPO a reward channel that DOES depend on
oracle content (verifier reward bypasses critique→audit indirection).

Designed as Layer 1 baseline for paper Section 4.1 ablation; if pilot
audit lift <2pt, escalate to Layer 2 (KL toward π_oracle).

---

Inherited v8 docstring follows:

D5 μ-v8 d5sdpo trainer — D5 in-house SDPO-flavor variant with 4 design imports.

ALGORITHM TAXONOMY (verified 2026-04-28; see CANONICAL_NAMING_REFERENCE.md):
This trainer is a **D5 in-house on-policy IS-loss / PPO-clip variant**, fork of
`train_mu_v7_opd.py` (`opd_mode=True`). Closest published comparable: OPSD
(Zhao 2601.18734) Table 3 sampled-token policy-gradient variant + Hübotter SDPO
(2601.20802) Appendix A.2 trust-region anchor. NOT a faithful reimplementation
of either canonical algorithm — Tinker exposes only scalar logprob, not
full-vocab/top-K KL or full-vocab JS.

Four changes vs `train_mu_v7_opd.py` (planned 2026-04-28):

(1) **Critique-on-CURRENT-rollout** (paper-fidelity fix; primary reason for v8)
    Old (v7-opd): teacher_input uses `prev_critique` from previous iter, critic
    fires AFTER training to update for next iter. iter-N teacher_lp is conditioned
    on critique of iter-(N-1) plan; iter 0 uses cold-start (no real feedback).
    New (v8): critic fires BEFORE teacher_lp compute, on a plan from THIS iter's
    student rollout. Matches Hübotter SDPO Algorithm 1: f = environment(x, y)
    where y is the current rollout. Cost: +60-180s per iter blocking on Opus.

(2) **Solution-only token mask** (knob: solution_only_mask, default True)
    Port of `rubric_reward/sdpo/train_sdpo.py:807-844`. Zeros per-token advantage
    on tokens outside <solution>...</solution>. Treats F11 finding (15.83% content
    density). D5 plans already use <solution> wrapper (mu_prompts_v1.py:30).

(3) **Trust-region interpolation** (knob: trust_region_alpha, default 0.05)
    Hübotter SDPO Appendix A.2 mitigation. teacher_lp_eff = (1-α)·teacher_lp_current
    + α·frozen_base_lp where frozen_base = base Qwen3-30B-A3B. Targets F3
    multi-round cliff (iter 7 in v7-opd-full).

(4) **PPO-clip loss** (knobs: loss_fn_name "ppo"|"importance_sampling", ppo_clip_eps)
    Default loss_fn="ppo" with clip_eps=0.2. Within-iter (microbatch 2-4) IS-ratio
    control. Equivalent to importance_sampling at microbatch 1.

Plan footer changed from v1's 600/750-word to v8's 900/1100-word
(`mu_prompts_v8._PLAN_FOOTER_V8`) — matches reference plan's 962 words.
σ_v8 baseline must be re-run at v8 footer for length-matched comparison.

Per iter:
  1. STUDENT samples N plans on-policy under student_input (goal+oracle, NO critique)
  2. PICK 1 plan; submit to critic; BLOCK on Opus subagent → critique_current
  3. teacher_input = goal + oracle + critique_current  ← rollout-specific feedback
  4. teacher_lp = compute_logprobs(teacher_input + student_seq.tokens)
  5. (if α > 0) frozen_lp = initial_teacher_client.compute_logprobs(...)
                 teacher_lp_eff = (1-α)·teacher_lp + α·frozen_lp
  6. A_t = clamp((teacher_lp_eff - student_lp) · scale, ±clip)
  7. (if solution_only_mask) A_t *= build_solution_token_mask(student_seq.tokens)
  8. forward_backward(loss_fn=ppo or importance_sampling, with clip if ppo)
  9. EVAL audit (async, non-blocking)

Reference plan: /home/silas/.claude/plans/snug-dreaming-yao.md (μ-v8 d5sdpo).
Run dirs: 2026_04_29_mu_v8_d5sdpo_<cell>/ where cell ∈ {base, A, B, C, D, E, F, G}.

Usage:
    source shared/tools/use_api_profile.sh new
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.train_mu_v8_d5sdpo \
        log_path=projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v8_d5sdpo_G \
        solution_only_mask=True trust_region_alpha=0.05 loss_fn_name=ppo ppo_clip_eps=0.2 \
        n_iter=16 eval_every=1 save_every=1
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
import torch
from tinker import types
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.opus_critic_subagent import OpusCriticClient
from co_scientist.shared.opus_audit_subagent import OpusAuditClient
from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v8 import (
    COLD_START_CRITIQUE_NONE,
    build_audit_request_payload,
    build_critic_request_payload,
    build_sdpo_datum,
    build_student_prompt_v8,
    build_teacher_prompt_v8,
    extract_solution,
    parse_critique_xml,
)
from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v8_rag import (
    build_student_prompt_v8_rag,
    build_teacher_prompt_v8_rag,
)
from co_scientist.d5_abstract_retrieve_refine.oracle_retriever_v1 import (
    OracleRetrieverV1,
)
from co_scientist.d5_abstract_retrieve_refine.verifier_grounded_reward_v1 import (
    char_spans_to_token_spans,
    compute_grounding_reward,
    load_gold,
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
        "projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v8_d5sdpo_smoke"
    )
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"
    reference_plan_path: str = (
        "projects/d5_abstract_retrieve_refine/dataset/reference_solution.txt"
    )
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

    # Sampling
    n_plans: int = 8
    # max_tokens=8192 (NOT 4096 like v7-opd) because v8's 900/1100-word footer
    # produces longer plans. Per base run iter 3-5 data: 4096 truncates 2/8 →
    # 7/8 → 8/8 plans as model learns longer outputs, contaminating audit /45
    # trajectory. 8192 gives ~2× headroom for "well-formed plan + Qwen3 thinking
    # preamble + <solution> XML wrapper".
    max_tokens: int = 8192
    temperature: float = 1.0
    top_p: float = 0.95

    # SDPO advantage (legacy knobs preserved for checkpoint config compat)
    sdpo_scale: float = 1.0
    sdpo_clip_advantage: float = 5.0
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

    # Continual SDPO (Phase 5 compatibility — unused by default)
    init_state_path: str = ""
    reset_optimizer_state: bool = False

    # Misc
    today_date: str = "2026_04_29"
    seed: int = 42

    # ===== v8 ablation knobs (cell-specific) =====
    # Cell config: each cell sets these knobs via chz CLI override.
    # base = (False, 0.0, "importance_sampling")
    # A = (True,  0.0, "importance_sampling")
    # B = (False, 0.05,"importance_sampling")
    # C = (True,  0.05,"importance_sampling")
    # D = (False, 0.0, "ppo")
    # E = (True,  0.0, "ppo")
    # F = (False, 0.05,"ppo")
    # G = (True,  0.05,"ppo")
    solution_only_mask: bool = True
    trust_region_alpha: float = 0.05
    loss_fn_name: Literal["importance_sampling", "ppo"] = "ppo"
    ppo_clip_eps: float = 0.2

    # Exp C — RAG knobs (oracle-transfer ABC, F16 H16-1 falsifier)
    use_rag: bool = False
    rag_k: int = 5

    # v9 Layer 1 — verifier-grounded reward knobs (H16-2 attack)
    use_grounding: bool = False
    grounding_signal_set: tuple[str, ...] = ("equations", "citations")
    grounding_per_eq_bonus: float = 0.1
    grounding_per_citation_bonus: float = 0.05
    gold_equations_path: str = (
        "projects/d5_abstract_retrieve_refine/dataset/v9_gold_equations.json"
    )

    # v9 Layer 2 — KL-anchor toward π_oracle (distribution-level grounding)
    use_kl_anchor: bool = False
    kl_beta: float = 0.05
    pi_oracle_path: str = ""  # tinker:// uri from train_pi_oracle_sft.py output


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


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
# Solution-only mask helpers (port from rubric_reward/sdpo/train_sdpo.py:761-844)
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
) -> np.ndarray:
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

    goal = _resolve(config.goal_path).read_text().strip()
    oracle = _resolve(config.oracle_abstraction_path).read_text().strip()
    ref_plan = _resolve(config.reference_plan_path).read_text().strip()
    source_paper_md = _resolve(config.source_paper_path).read_text().strip()
    logger.info("v8-d5sdpo inputs:")
    logger.info("  goal:          %d chars", len(goal))
    logger.info("  oracle (slim): %d chars", len(oracle))
    logger.info("  ref plan:      %d chars (NOT used in loss; held for σ comparison)", len(ref_plan))
    logger.info("  source paper:  %d chars (privileged for critic)", len(source_paper_md))
    logger.info("  log_dir:       %s", log_dir)

    # Exp C — RAG retriever + cached student-side retrieval (goal is fixed)
    retriever: OracleRetrieverV1 | None = None
    oracle_for_student: str = oracle
    if config.use_rag:
        retriever = OracleRetrieverV1.load_default(
            slim_path=_resolve(config.oracle_abstraction_path)
        )
        oracle_for_student = retriever.retrieve(goal, k=config.rag_k)
        logger.info(
            "  RAG: use_rag=True, k=%d, student oracle %d chars (%.1f%% of full)",
            config.rag_k,
            len(oracle_for_student),
            100.0 * len(oracle_for_student) / max(len(oracle), 1),
        )

    # v9 — verifier-grounded reward setup (Layer 1)
    gold: dict | None = None
    if config.use_grounding:
        gold = load_gold(_resolve(config.gold_equations_path))
        logger.info(
            "  GROUNDING: use_grounding=True, signals=%s, gold=%d eqs+%d citations, "
            "per_eq=%.2f, per_cite=%.2f",
            config.grounding_signal_set,
            len(gold["equations"]),
            len(gold["citations"]),
            config.grounding_per_eq_bonus,
            config.grounding_per_citation_bonus,
        )

    # v9 Layer 2 — KL-anchor π_oracle client setup
    pi_oracle_client = None
    if config.use_kl_anchor:
        assert config.pi_oracle_path, "use_kl_anchor=True requires pi_oracle_path (tinker:// uri)"
        # Created later via service_client (after service_client init)
        logger.info(
            "  KL_ANCHOR: use_kl_anchor=True, beta=%.4f, pi_oracle=%s",
            config.kl_beta, config.pi_oracle_path,
        )
    logger.info(
        "v8 ablation knobs: solution_only_mask=%s trust_region_alpha=%.3f loss_fn=%s ppo_clip_eps=%.2f",
        config.solution_only_mask, config.trust_region_alpha,
        config.loss_fn_name, config.ppo_clip_eps,
    )

    cfg_path = log_dir / "config.json"
    cfg_path.write_text(json.dumps({k: getattr(config, k) for k in dir(config)
                                     if not k.startswith("_") and not callable(getattr(config, k))},
                                    default=str, indent=2))

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info("Renderer: %s", renderer_name)

    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )

    # Frozen base sampling client (for trust-region interp). Created lazily to
    # avoid the cost on cells where alpha=0 disables it. Per Hübotter A.2:
    # the "initial teacher" stays at base-model weights throughout training.
    initial_teacher_client = None
    if config.trust_region_alpha > 0.0:
        initial_teacher_client = service_client.create_sampling_client(
            base_model=config.model_name,
        )
        logger.info(
            "Trust-region: created frozen base sampling client (alpha=%.3f)",
            config.trust_region_alpha,
        )

    # v9 Layer 2 — load π_oracle sampling client from SFT'd LoRA tinker:// uri
    pi_oracle_client = None
    if config.use_kl_anchor:
        pi_oracle_client = service_client.create_sampling_client(
            model_path=config.pi_oracle_path,
        )
        logger.info(
            "KL-anchor: created π_oracle sampling client from %s (beta=%.4f)",
            config.pi_oracle_path, config.kl_beta,
        )

    # Training client (resume / init / fresh)
    last_checkpoint = checkpoint_utils.get_last_checkpoint(str(log_dir))
    if last_checkpoint is not None:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            last_checkpoint["state_path"]
        )
        start_iter = last_checkpoint.get("batch", 0) + 1
        logger.info("Resuming from iter %d (state=%s)", start_iter, last_checkpoint["state_path"])
    elif config.init_state_path:
        if config.reset_optimizer_state:
            training_client = service_client.create_training_client_from_state(
                config.init_state_path
            )
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

        # Sampling client at current LoRA weights
        sp_path = training_client.save_weights_for_sampler(
            name=f"iter_{iter_idx:04d}"
        ).result().path
        sampling_client = service_client.create_sampling_client(model_path=sp_path)
        with open(checkpoints_path, "a") as f:
            f.write(json.dumps({"iter": iter_idx, "sampler_path": sp_path,
                                 "ts": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")

        # ----- STUDENT ROLLOUT (on-policy under student_input) -----
        student_text = (
            build_student_prompt_v8_rag(goal, oracle_for_student)
            if config.use_rag
            else build_student_prompt_v8(goal, oracle)
        )
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
        logger.info("Iter %d: STUDENT (on-policy) sampled %d plans in %.1fs",
                     iter_idx, len(student_result.sequences), t_sample)

        # Decode plan texts now (used for critic + per_plan_records)
        decoded_plans = []
        for k, seq in enumerate(student_result.sequences):
            try:
                decoded_plans.append(_decode_plan(seq, tokenizer, renderer))
            except Exception as e:
                logger.warning("Iter %d plan %d: decode failed: %s", iter_idx, k, e)
                decoded_plans.append("")

        # ----- CHANGE 1: CRITIC CALL ON CURRENT ROLLOUT (BLOCKING) -----
        # Paper-faithful (Hübotter SDPO §Algorithm 1): teacher_input must condition
        # on environmental feedback `f` produced FOR THIS rollout y, not a stale
        # critique on a prior rollout. Move critic to BEFORE teacher_lp compute.
        critique_current = COLD_START_CRITIQUE_NONE
        if not decoded_plans or all(not t for t in decoded_plans):
            logger.warning("Iter %d: no decoded plans; using cold-start critique", iter_idx)
        else:
            valid_idxs = [i for i, t in enumerate(decoded_plans) if t]
            chosen_idx = rng.choice(valid_idxs)
            chosen_plan_id = f"iter_{iter_idx:03d}_pick_{chosen_idx}"
            payload = build_critic_request_payload(
                iter_idx=iter_idx, goal=goal,
                plan_text=decoded_plans[chosen_idx],
                source_paper_md=source_paper_md,
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
                logger.error("Iter %d: critic timeout: %s. Falling back to cold-start.",
                              iter_idx, e)
                critique_current = COLD_START_CRITIQUE_NONE
            t_crit = time.time() - t_crit0
            logger.info("Iter %d: critic (CURRENT rollout) round-trip %.1fs, len=%d chars, pick=%d",
                         iter_idx, t_crit, len(critique_current), chosen_idx)

        # ----- TEACHER CONTEXT WITH CURRENT-ROLLOUT CRITIQUE -----
        if config.use_rag:
            assert retriever is not None
            oracle_for_teacher = retriever.retrieve(
                f"{goal}\n\n{critique_current}", k=config.rag_k
            )
            teacher_text = build_teacher_prompt_v8_rag(
                goal, oracle_for_teacher, critique_current
            )
        else:
            teacher_text = build_teacher_prompt_v8(goal, oracle, critique_current)
        teacher_convo = [{"role": "user", "content": teacher_text}]
        teacher_input = renderer.build_generation_prompt(teacher_convo)
        teacher_tokens_prefix = teacher_input.to_ints()

        # ----- TEACHER LOGPROBS on STUDENT tokens under teacher_input -----
        t_lp0 = time.time()
        teacher_lp_futures = [
            sampling_client.compute_logprobs(
                types.ModelInput.from_ints(tokens=teacher_tokens_prefix + list(seq.tokens))
            )
            for seq in student_result.sequences
        ]
        teacher_lps_full = [f.result() for f in teacher_lp_futures]
        t_lp = time.time() - t_lp0
        logger.info("Iter %d: TEACHER logprobs on student tokens in %.1fs", iter_idx, t_lp)

        # ----- CHANGE 3: TRUST-REGION INTERP (frozen base lp) -----
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
            logger.info(
                "Iter %d: FROZEN-BASE logprobs (trust-region α=%.3f) in %.1fs",
                iter_idx, config.trust_region_alpha, t_frozen_lp,
            )

        # ----- v9 Layer 2: π_oracle lp under student prompt (for KL anchor) -----
        # Note: π_oracle is conditioned on STUDENT prompt (no critique), since we want
        # to anchor the policy's distribution over plans given (goal+oracle), not given
        # critique-conditioned prompt. Same pattern as student logprobs.
        oracle_lps_full = None
        t_oracle_lp = 0.0
        if config.use_kl_anchor and pi_oracle_client is not None:
            t_or0 = time.time()
            oracle_lp_futures = [
                pi_oracle_client.compute_logprobs(
                    types.ModelInput.from_ints(tokens=student_tokens + list(seq.tokens))
                )
                for seq in student_result.sequences
            ]
            oracle_lps_full = [f.result() for f in oracle_lp_futures]
            t_oracle_lp = time.time() - t_or0
            logger.info(
                "Iter %d: π_ORACLE logprobs (KL-anchor β=%.4f) in %.1fs",
                iter_idx, config.kl_beta, t_oracle_lp,
            )

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
            if (not gen
                or len(sampler_lp_seq) != len(gen)
                or len(t_lp_seq) != len(gen)):
                logger.warning(
                    "Iter %d plan %d: len mismatch gen=%d sampler_lp=%d t_lp=%d; skipping",
                    iter_idx, k, len(gen), len(sampler_lp_seq), len(t_lp_seq),
                )
                continue

            s_lp = sampler_lp_seq  # on-policy student lp from .sample().logprobs

            # CHANGE 3: trust-region interp on teacher_lp
            if frozen_lps_full is not None:
                full_f_lp = frozen_lps_full[k]
                f_lp_seq = list(
                    full_f_lp[len(teacher_tokens_prefix):len(teacher_tokens_prefix) + len(gen)]
                )
                if len(f_lp_seq) == len(t_lp_seq):
                    a = config.trust_region_alpha
                    t_lp_eff = [(1 - a) * t + a * f for t, f in zip(t_lp_seq, f_lp_seq)]
                else:
                    logger.warning(
                        "Iter %d plan %d: frozen_lp len mismatch (%d vs %d); falling back to teacher_lp only",
                        iter_idx, k, len(f_lp_seq), len(t_lp_seq),
                    )
                    t_lp_eff = t_lp_seq
            else:
                t_lp_eff = t_lp_seq

            # v9 Layer 2 — extract π_oracle per-token lp slice for KL anchor
            oracle_lp_seq = None
            kl_per_token_mean = 0.0
            if config.use_kl_anchor and oracle_lps_full is not None:
                full_o_lp = oracle_lps_full[k]
                # Note: oracle lp was computed under STUDENT prompt prefix
                oracle_lp_seq = list(
                    full_o_lp[len(student_tokens):len(student_tokens) + len(gen)]
                )
                if len(oracle_lp_seq) != len(gen):
                    logger.warning(
                        "Iter %d plan %d: oracle_lp len mismatch %d vs gen %d; KL skipped",
                        iter_idx, k, len(oracle_lp_seq), len(gen),
                    )
                    oracle_lp_seq = None

            # Per-token advantage: A_t = clamp((t_lp_eff - s_lp) * scale - β·KL_t, ±clip)
            # KL_t = lp_θ - lp_oracle (positive if θ diverges from oracle); subtract β·KL
            clip_v = config.opd_anchor_clip
            per_tok_adv = []
            kl_terms = []
            for ti, (t_, s_) in enumerate(zip(t_lp_eff, s_lp)):
                a = float(t_ - s_) * config.sdpo_scale
                if oracle_lp_seq is not None and ti < len(oracle_lp_seq):
                    kl_t = float(s_ - oracle_lp_seq[ti])  # lp_θ ≈ s_lp under student prompt
                    kl_terms.append(kl_t)
                    a -= config.kl_beta * kl_t
                a = max(-clip_v, min(clip_v, a))
                per_tok_adv.append(a)
            if kl_terms:
                kl_per_token_mean = float(np.mean(kl_terms))

            # v9 — sparse per-token grounding bonus BEFORE solution-only mask
            # (mask zeros bonuses on non-solution tokens — desired behavior)
            grounding_result: dict | None = None
            grounding_n_bonus_tokens = 0
            if config.use_grounding and gold is not None:
                plan_text_k = decoded_plans[k] if k < len(decoded_plans) else ""
                if plan_text_k:
                    grounding_result = compute_grounding_reward(
                        plan_text_k, gold,
                        signal_set=config.grounding_signal_set,
                        per_eq_bonus=config.grounding_per_eq_bonus,
                        per_citation_bonus=config.grounding_per_citation_bonus,
                        claims_extractor_fn=None,  # Layer 1 no LLM extractor (cost control)
                    )
                    # Map char spans to token positions in `gen` (gen is generated tokens
                    # from sample, plan_text_k is its decoded form via extract_solution).
                    # We approximate by encoding the plan_text prefix and treating as offset
                    # into `gen` directly; for v9 Layer 1 this is acceptable noise.
                    char_spans = [(cs, ce) for (cs, ce, _b) in grounding_result["char_bonus_spans"]]
                    token_spans = char_spans_to_token_spans(
                        plan_text_k, char_spans, tokenizer, plan_token_offset=0,
                    )
                    bonuses = [b for (_cs, _ce, b) in grounding_result["char_bonus_spans"]]
                    for (t_start, t_end), bonus in zip(token_spans, bonuses):
                        t_start = max(0, min(t_start, len(per_tok_adv)))
                        t_end = max(t_start, min(t_end, len(per_tok_adv)))
                        for ti in range(t_start, t_end):
                            per_tok_adv[ti] = max(-clip_v, min(clip_v, per_tok_adv[ti] + bonus))
                            grounding_n_bonus_tokens += 1

            # CHANGE 2: solution-only token mask
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
                "mean_t_lp_raw": float(np.mean(t_lp_seq)),
                "mean_t_lp_eff": float(np.mean(t_lp_eff)),
                "mean_s_lp": float(np.mean(s_lp)),
                "mean_adv": float(np.mean(per_tok_adv)),
                "stop_reason": seq.stop_reason,
                "plan_text": decoded_plans[k] if k < len(decoded_plans) else "",
                "raw_tokens_text": tokenizer.decode(gen),
                "solution_mask_active_tokens": n_mask_active,
                "solution_mask_density": mask_density,
                "trust_region_alpha": config.trust_region_alpha,
                "sampled_by": "student",
                # v9 grounding diagnostic fields (None when use_grounding=False)
                "grounding_score": (
                    grounding_result["score"] if grounding_result else None
                ),
                "grounding_per_signal": (
                    grounding_result["per_signal"] if grounding_result else None
                ),
                "grounding_n_bonus_tokens": grounding_n_bonus_tokens,
                "grounding_diag": (
                    grounding_result["diag"] if grounding_result else None
                ),
                # v9 Layer 2 KL-anchor diagnostic fields (None when use_kl_anchor=False)
                "kl_per_token_mean": (
                    kl_per_token_mean if config.use_kl_anchor else None
                ),
                "kl_beta": config.kl_beta if config.use_kl_anchor else None,
                "pi_oracle_path": (
                    config.pi_oracle_path if config.use_kl_anchor else None
                ),
            })

        # ----- CHANGE 4: TRAIN STEP (loss_fn = ppo or importance_sampling) -----
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
            "Iter %d v8 stats: %d datums, mean_adv=%.4f, pos_frac=%.3f, mean|A|=%.4f, train=%.1fs",
            iter_idx, len(datums), mean_adv, pos_frac, mean_abs, t_step,
        )

        # v9 grounding stats (iter-level mean across plans)
        if config.use_grounding:
            grounding_records = [r for r in per_plan_records if r.get("grounding_score") is not None]
            if grounding_records:
                gs = [r["grounding_score"] for r in grounding_records]
                eq_hits = [r["grounding_per_signal"].get("equations", 0) for r in grounding_records]
                cite_hits = [r["grounding_per_signal"].get("citations", 0) for r in grounding_records]
                bonus_tokens = [r["grounding_n_bonus_tokens"] for r in grounding_records]
                logger.info(
                    "Iter %d v9 grounding: mean_score=%.4f, mean_eq_hits=%.2f, "
                    "mean_cite_hits=%.2f, mean_bonus_tokens=%.1f",
                    iter_idx, float(np.mean(gs)), float(np.mean(eq_hits)),
                    float(np.mean(cite_hits)), float(np.mean(bonus_tokens)),
                )

        # v9 Layer 2 KL-anchor stats (iter-level mean across plans)
        if config.use_kl_anchor:
            kl_records = [r for r in per_plan_records if r.get("kl_per_token_mean") is not None]
            if kl_records:
                kls = [r["kl_per_token_mean"] for r in kl_records]
                logger.info(
                    "Iter %d v9 KL-anchor: mean_kl_per_token=%.4f, beta=%.4f, "
                    "n_plans_with_kl=%d",
                    iter_idx, float(np.mean(kls)), config.kl_beta, len(kl_records),
                )

        # ----- WRITE BUFFER -----
        with open(buffer_path, "a") as f:
            for rec in per_plan_records:
                f.write(json.dumps(rec) + "\n")

        # ----- EVAL AUDIT (ASYNC) -----
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
                "wall_critic_sec": float(t_crit) if 't_crit' in locals() else 0.0,
                "wall_logprob_sec": t_lp,
                "wall_frozen_logprob_sec": t_frozen_lp,
                "wall_train_sec": t_step,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "critique_current_len": len(critique_current),
                "critique_was_cold_start": critique_current == COLD_START_CRITIQUE_NONE,
                "solution_only_mask": bool(config.solution_only_mask),
                "trust_region_alpha": config.trust_region_alpha,
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

        # ----- v4 EARLY-STOP GUARD -----
        try:
            audit_resp_dir = log_dir / "audit_responses"
            if audit_resp_dir.exists():
                done_iters = sorted(int(p.stem.replace("iter_", ""))
                                    for p in audit_resp_dir.glob("iter_*.json")
                                    if "_plan_" not in p.name and ".partial" not in p.name)
                if len(done_iters) >= 2:
                    means = []
                    for it_done in done_iters[-2:]:
                        rp = audit_resp_dir / f"iter_{it_done:03d}.json"
                        try:
                            d = json.loads(rp.read_text())
                            judg = d.get("judgments", {})
                            scores = []
                            for pid, jd in judg.items():
                                tot = jd.get("total")
                                if tot is not None:
                                    scores.append(float(tot))
                            if scores:
                                means.append(float(np.mean(scores)))
                        except Exception:
                            pass
                    if len(means) == 2 and (means[0] - means[1]) >= config.audit_drop_threshold:
                        logger.error(
                            "Iter %d: audit drop %.2f → %.2f exceeds threshold %.2f; halting",
                            iter_idx, means[0], means[1], config.audit_drop_threshold,
                        )
                        break
        except Exception as e:
            logger.warning("Iter %d: early-stop check failed: %s", iter_idx, e)

    logger.info("v8-d5sdpo run complete: %s", log_dir)


if __name__ == "__main__":
    config = chz.entrypoint(Config)
    main(config)
