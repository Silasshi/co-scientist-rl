"""CR-v7: Per-signal context REINFORCE trainer for grant proposal generation.

Forked from ttt_discover/train_cr_v7.py for D4 (grant_proposal).
Contains the active CR-v7 pipeline adapted for grant proposal generation
with forked signal definitions.

Usage:
    source shared/tools/use_api_profile.sh new
    python src/co_scientist/grant_proposal/train_cr_v7.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

import httpx
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
from co_scientist.shared.opus_subagent_grader import OpusSubagentGrader
from co_scientist.shared.grant_signal_reward import (
    SIGNALS as GRADIENT_SIGNALS,
    SIGNAL_WEIGHTS as GRADIENT_WEIGHTS,
    SCORE_MAX as GRADIENT_SCORE_MAX,
    aggregate_reward as gradient_aggregate,
    normalize_score,
)
from co_scientist.grant_proposal.train_buffer_ttt import (
    BufferEntry,
    PlanFutures,
    TenSignalReward,
    build_research_plan_prompt,
    collect_plan_reward,
    launch_plan_reward,
    select_context,
)
from co_scientist.grant_proposal.train_critique_revise import (
    MAX_PLAN_CHARS_IN_PROMPT,
    detect_duplicated_content,
    parse_critique_and_plan,
    ucb_select,
)
from co_scientist.shared.paper_retrieval import (
    PaperRetriever,
    RetrievalResult,
    QUERY_GENERATION_PROMPT,
    REVISION_QUERY_GENERATION_PROMPT,
    extract_search_queries,
    format_papers_for_prompt,
    goal_to_queries,
    citation_match_rate,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)

_SOLUTION_RE = re.compile(r"<solution>(.*?)(?:</solution>|$)", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,3}\s|\*\*[^\s*])", re.MULTILINE)


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)} if len(words) >= n else set()


def compute_novelty(plan: str, buffer: list[dict], top_k: int = 5) -> float:
    """Compute novelty of a plan relative to buffer entries.

    Returns a value in [0, 1] where 1 = maximally novel (no overlap with buffer).
    Uses mixed unigram + trigram Jaccard distance against the top-K highest-reward
    buffer entries. Trigrams resist simple word-substitution hacking.
    """
    if not buffer or not plan.strip():
        return 1.0
    plan_words = plan.lower().split()
    if not plan_words:
        return 1.0
    plan_uni = set(plan_words)
    plan_tri = _ngrams(plan_words, 3)
    valid = [e for e in buffer if e.get("hard_gate_passed", True)]
    valid.sort(key=lambda e: e.get("aggregate_reward", 0), reverse=True)
    candidates = valid[:top_k]
    if not candidates:
        return 1.0
    similarities = []
    for entry in candidates:
        buf_words = entry.get("plan_text", "").lower().split()
        if not buf_words:
            continue
        buf_uni = set(buf_words)
        buf_tri = _ngrams(buf_words, 3)
        # Unigram Jaccard
        uni_inter = len(plan_uni & buf_uni)
        uni_union = len(plan_uni | buf_uni)
        sim_uni = uni_inter / uni_union if uni_union > 0 else 0
        # Trigram Jaccard
        tri_inter = len(plan_tri & buf_tri)
        tri_union = len(plan_tri | buf_tri)
        sim_tri = tri_inter / tri_union if tri_union > 0 else 0
        # Weighted average: trigram captures structure, unigram captures vocabulary
        similarities.append(0.4 * sim_uni + 0.6 * sim_tri)
    if not similarities:
        return 1.0
    return 1.0 - max(similarities)


def gapo_diversity_adjusted_rewards(
    fresh_rewards: list[TenSignalReward],
    bonus_scale: float = 1.0,
) -> np.ndarray:
    """GAPO diversity reward: quality * nearest-neighbor diversity bonus in signal space."""
    n = len(fresh_rewards)
    quality = np.array([r.aggregate_reward for r in fresh_rewards])
    if n < 2:
        return quality

    signal_ids = [s.id for s in GRADIENT_SIGNALS]
    signal_vecs = np.array([
        [normalize_score(r.signal_vector.get(sid), score_max=GRADIENT_SCORE_MAX.get(sid, 5))
         for sid in signal_ids]
        for r in fresh_rewards
    ])

    if (signal_vecs.std(axis=0) < 1e-9).all():
        return quality

    nn_dists = np.zeros(n)
    for i in range(n):
        dists = [np.linalg.norm(signal_vecs[i] - signal_vecs[j]) for j in range(n) if j != i]
        nn_dists[i] = min(dists)

    median_nn = np.median(nn_dists)
    if median_nn < 1e-9:
        return quality

    diversity_bonus = np.clip(nn_dists / median_nn * bonus_scale, 0.1, 2.0)
    adjusted = quality * diversity_bonus
    logger.info(
        f"GAPO: nn_dists={[f'{d:.3f}' for d in nn_dists]}, "
        f"bonus={[f'{b:.2f}' for b in diversity_bonus]}, "
        f"quality={[f'{q:.3f}' for q in quality]} → adjusted={[f'{a:.3f}' for a in adjusted]}"
    )
    return adjusted


def _confidence_stats(valid_fresh, revise_rewards) -> dict:
    """Aggregate grader confidence stats across fresh + revision graded plans.

    Returns dict with keys grader_confidence/mean, grader_confidence/low_frac
    (fraction of valid signal scores with confidence < 0.6), and per-signal
    mean confidence. Empty dict if no confidence data.
    """
    rewards = [r for _, r, _ in valid_fresh] + [
        rr for rr in revise_rewards if rr is not None
    ]
    all_conf: list[float] = []
    per_sig: dict[str, list[float]] = {}
    for r in rewards:
        psc = getattr(r, "per_signal_confidence", {}) or {}
        for sid, v in psc.items():
            if v is None:
                continue
            all_conf.append(float(v))
            per_sig.setdefault(sid, []).append(float(v))
    if not all_conf:
        return {}
    low_frac = sum(1 for v in all_conf if v < 0.6) / len(all_conf)
    out = {
        "grader_confidence/mean": float(np.mean(all_conf)),
        "grader_confidence/low_frac": float(low_frac),
    }
    for sid, vs in per_sig.items():
        out[f"grader_confidence/{sid}"] = float(np.mean(vs))
    return out


def _to_str(content) -> str:
    """Coerce renderer content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", content.get("content", str(content)))
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", item.get("content", str(item))))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def extract_solution(text) -> str:
    """Extract plan content, stripping <think> tags and in-solution reasoning.

    Qwen3 often puts "Okay, I need to..." thinking INSIDE <solution> tags.
    We strip the <solution> wrapper first, then find the first markdown
    heading (# or **Title) and discard the preamble before it.
    """
    if not isinstance(text, str):
        text = str(text) if not isinstance(text, list) else text[0] if text else ""
        if not isinstance(text, str):
            text = str(text)
    m = _SOLUTION_RE.search(text)
    if m:
        text = m.group(1).strip()
    heading = _HEADING_RE.search(text)
    if heading and heading.start() > 0:
        text = text[heading.start():]
    return text.strip()


def _run_retrieval_pipeline(
    goal: str,
    retriever: PaperRetriever,
    config,
) -> RetrievalResult | None:
    """Pipeline-driven retrieval: extract keywords from goal, search directly."""
    queries = goal_to_queries(goal, n_queries=config.retrieval_max_queries)
    if not queries:
        return None
    return retriever.search_multiple(
        queries, limit_per_query=config.retrieval_papers_per_query,
    )


def _run_retrieval_model(
    query_prompt_text: str,
    retriever: PaperRetriever,
    sampling_client,
    renderer,
    config,
) -> RetrievalResult | None:
    """Model-driven retrieval: LLM generates search queries, then search."""
    query_input = renderer.build_generation_prompt(
        [{"role": "user", "content": query_prompt_text}]
    )
    query_params = tinker.types.SamplingParams(
        max_tokens=config.retrieval_query_max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=0.7,
    )
    query_future = sampling_client.sample(
        prompt=query_input, num_samples=1, sampling_params=query_params,
    )
    query_result = query_future.result()
    parsed = renderer.parse_response(query_result.sequences[0].tokens)
    first_item = parsed[0]["content"] if isinstance(parsed[0], dict) else parsed[0]
    query_raw = _to_str(first_item)
    queries = extract_search_queries(query_raw, max_queries=config.retrieval_max_queries)
    if not queries:
        logger.warning("RAG: no <search> tags found. Raw output (200 chars): %s", query_raw[:200])
        return None
    return retriever.search_multiple(
        queries, limit_per_query=config.retrieval_papers_per_query,
    )


# =============================================================================
# Active research goal (loaded from disk)
# =============================================================================

from co_scientist.grant_proposal.train_buffer_ttt import TARGET_GOAL, load_goal_config


# =============================================================================
# Config (CR-v7 only; no locus / paragraph / c3_* fields)
# =============================================================================


@chz.chz
class Config:
    # --- Infrastructure ---
    base_url: str | None = None
    api_profile: str | None = "new"
    goal_dir: str = "projects/grant_proposal/dataset/goals/01_foundopt"
    log_path: str = "/home/silas/co-scientist-project/projects/grant_proposal/runs/default"
    model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"  # policy = grader

    # --- Training loop ---
    n_iterations: int = 25
    n_fresh: int = 4                  # fresh plans per iter (exploration)
    n_revise: int = 4                 # parents selected for revision (exploitation)
    n_revision_candidates: int = 1    # BoN per parent
    grader_repeats: int = 1           # grader calls per signal per plan (temperature=0 → 1 is sufficient)

    # --- Buffer / context selection ---
    K_exploit: int = 3
    K_explore: int = 2
    cold_start_iters: int = 1         # first N iters: fresh-only, no revisions
    use_ucb: bool = True
    ucb_c: float = 1.0

    # --- Per-signal REINFORCE ---
    delta_threshold: float = 1e-3     # skip per-signal datum if |Δ_i| below
    delta_scale: float = 5.0          # per-signal advantage multiplier

    # --- Baselines ---
    skip_rl_update: bool = False      # B4: in-context only, no gradient
    skip_hard_gates: bool = False     # A1: ablate goal-contrast + claim verif
    strip_critiques: bool = False     # B4_stripped: revision prompt shows only aggregate score
    scores_only: bool = False         # Show signal names + scores, no critique text (reduced leakage)
    include_cot_scaffolding: bool = True  # Ablation: set False to remove CoT scaffolding from grader prompts

    # --- Signal ablation ---
    disabled_signals: str = ""        # comma-separated IDs (e.g. "S4_significance")

    # --- Fresh-plan training (optional) ---
    train_on_fresh: bool = False      # B3/A8: entropic training on fresh plans
    fresh_use_context: bool = False   # When true, fresh plans see buffer context (D3-style)
    kl_budget: float = 0.693          # ln 2 (for train_on_fresh)
    beta_max: float = 20.0
    novelty_weight: float = 0.0       # λ in R_adj = R * (1 + λ * novelty). 0 = disabled.
    skip_revision_rl: bool = False    # When true, skip per-signal REINFORCE on revisions (A_fresh ablation)

    # --- GAPO diversity RL (2026-04-21) ---
    diversity_method: str = "none"    # "gapo" | "none". GAPO: NN-distance diversity bonus in signal space.
    diversity_bonus_scale: float = 1.0  # multiplier on diversity bonus (1.0 = standard)

    # --- SDPO: Self-Distillation Policy Optimization (Hübotter et al. 2026) ---
    sdpo: bool = False                # Per-token advantage from self-teacher (full critique context)
    sdpo_scale: float = 1.0           # Multiplier on SDPO advantages
    sdpo_clip_advantage: float = 5.0  # Clip |A^SDPO_t| to avoid extreme logprob-ratio outliers
    revision_rl_mode: str = "per_signal"  # "per_signal" (HER) | "aggregate" | "sdpo" (auto-set when sdpo=True)

    # --- D4v2: Multi-sample grader confidence filter (2026-04-22) ---
    # Requires grader_repeats >= 2 AND grader_temperature > 0 (e.g., 0.3) for variance.
    # Confidence = 1 - range/(score_max-1) across samples.
    # Literature: Cycles of Thought (2024), Confidence Improves Self-Consistency (2025).
    # For v8 recommended: grader_repeats=3, grader_temperature=0.3, min_grader_confidence=0.6.
    min_grader_confidence: float = 0.0  # Skip per-signal RL datum if confidence < threshold (0.0 disables)

    # --- Retrieval-Augmented Generation (RAG) ---
    use_retrieval: bool = False           # toggle RAG for ablation
    retrieval_mode: str = "model"         # "model" (LLM generates queries) | "pipeline" (keyword extraction)
    retrieval_api_key: str = ""           # unused (OpenAlex is free), kept for compat
    retrieval_phase1: bool = True         # retrieve for fresh plan generation
    retrieval_phase2: bool = True         # retrieve for revision (conditional on G3)
    retrieval_g3_threshold: int = 3       # trigger revision retrieval when G3 <= this
    retrieval_max_queries: int = 3        # max search queries per retrieval step
    retrieval_papers_per_query: int = 5   # papers per S2 API call
    retrieval_max_papers: int = 10        # papers injected into prompt after dedup
    retrieval_max_chars: int = 2000       # max chars for formatted paper block
    retrieval_query_max_tokens: int = 1024 # max tokens for query generation call (needs headroom for Qwen3 thinking)

    # --- Grader critique emission (required for CR-v7) ---
    emit_signal_critique: bool = True

    # --- Alt grader for signals with grader_model_override (v9: SA_arithmetic) ---
    grader_model_alt: str = ""        # e.g. "Qwen/Qwen3-235B-A22B"; empty = no alt

    # --- Optimization ---
    learning_rate: float = 4e-5
    clip_eps: float = 0.2
    lora_rank: int = 32

    # --- Sampling ---
    max_length: int = 32768
    max_tokens: int = 4096            # whole-plan rewrite (CR-v7 default = 4096)
    grader_max_tokens: int = 8192
    grader_hard_gate_max_tokens: int = 8192
    temperature: float = 1.0
    grader_temperature: float = 0.0

    # --- Format constraints ---
    max_word_count: int = 750
    min_words: int = 30

    # --- Logging / checkpoint ---
    save_every: int = 5
    today_date: str = time.strftime("%Y-%m-%d", time.localtime())
    seed: int = 0

    # --- Opus subagent grader backend (2026-04-23) ---
    # "tinker"        = default; grader_model_name drives a tinker sampling client
    # "opus_subagent" = route main gradient + hard-gate grading through a Claude Code
    #                   subagent daemon via file-bus JSON (see
    #                   projects/grant_proposal_v2/scripts/opus_grader_subagent.md).
    grader_backend: str = "tinker"
    grader_long_critique: bool = False            # Opus: request ~300-token critiques (vs V10 template default 1-3 sentences)
    grader_batch_timeout_sec: float = 900.0       # wait this long per iter for daemon response


# =============================================================================
# CR-v7 prompts
# =============================================================================


def build_whole_plan_revision_prompt(
    goal: str,
    plan_text: str,
    signal_vector: dict[str, int | None],
    critiques: dict[str, str],
    strip_critiques: bool = False,
    scores_only: bool = False,
    retrieved_papers: str | None = None,
) -> str:
    """Whole-plan rewrite conditioned on per-signal feedback.

    Three modes:
    - Default: signal name + score + critique text (full info)
    - scores_only: signal name + score only (reduced leakage)
    - strip_critiques: aggregate score only (minimal info)
    """
    if len(plan_text) > MAX_PLAN_CHARS_IN_PROMPT:
        plan_text = plan_text[:MAX_PLAN_CHARS_IN_PROMPT] + "\n\n[... rest truncated ...]"

    if strip_critiques and scores_only:
        raise ValueError("strip_critiques and scores_only are mutually exclusive")

    if strip_critiques:
        agg = gradient_aggregate(signal_vector)
        feedback_block = f"- Overall score: {agg:.3f}/1.0"
    elif scores_only:
        feedback_lines = []
        for spec in GRADIENT_SIGNALS:
            score = signal_vector.get(spec.id)
            score_str = f"{score}/{spec.score_max}" if score is not None else f"?/{spec.score_max}"
            desc = spec.question.split("?")[0] + "?" if "?" in spec.question else spec.question.split(".")[0] + "."
            feedback_lines.append(f"- {spec.name} ({score_str}): {desc}")
        feedback_block = "\n".join(feedback_lines)
    else:
        feedback_lines = []
        for spec in GRADIENT_SIGNALS:
            score = signal_vector.get(spec.id)
            critique = (critiques or {}).get(spec.id, "") or ""
            score_str = f"{score}/{spec.score_max}" if score is not None else f"?/{spec.score_max}"
            if critique:
                feedback_lines.append(f"- {spec.name} ({score_str}): {critique}")
            else:
                feedback_lines.append(f"- {spec.name} ({score_str}): [no critique]")
        feedback_block = "\n".join(feedback_lines)

    papers_section = ""
    if retrieved_papers:
        papers_section = (
            f"# Relevant Published Papers\n"
            f"Use these real papers to improve grounding and citations.\n\n"
            f"{retrieved_papers}\n\n"
        )

    return (
        f"# Research Goal\n{goal}\n\n"
        f"{papers_section}"
        f"# Current Plan\n{plan_text.strip()}\n\n"
        f"# Per-signal feedback\n{feedback_block}\n\n"
        f"# Output Format\n"
        f"Rewrite the plan addressing the feedback above.\n\n"
        f"<think>\n"
        f"...your reasoning...\n"
        f"</think>\n"
        f"<solution>\n"
        f"...your rewritten plan...\n"
        f"</solution>"
    )


def build_per_signal_context(
    plan_text: str,
    critique_i: str,
    signal_name: str,
    score: int | None,
    goal: str,
    score_max: int = 5,
    scores_only: bool = False,
    signal_question: str = "",
) -> str:
    """Single-signal context used at LOSS time (never sampled).

    The policy action `a` was sampled under `build_whole_plan_revision_prompt`
    (all per-signal feedback). At loss time we recompute `logπ(a | context_i)` under
    this single-signal context — HER-style relabelling that gives per-signal
    gradient decoupling.
    """
    if len(plan_text) > MAX_PLAN_CHARS_IN_PROMPT:
        plan_text = plan_text[:MAX_PLAN_CHARS_IN_PROMPT] + "\n\n[... rest truncated ...]"
    score_str = f"{score}/{score_max}" if score is not None else f"?/{score_max}"
    if scores_only:
        desc = signal_question.split("?")[0] + "?" if "?" in signal_question else signal_question.split(".")[0] + "."
        feedback = f"- {signal_name} ({score_str}): {desc}"
    else:
        critique = critique_i.strip() if critique_i else "[no critique]"
        feedback = f"- {signal_name} ({score_str}): {critique}"
    return (
        f"# Research Goal\n{goal}\n\n"
        f"# Current Plan\n{plan_text.strip()}\n\n"
        f"# Per-signal feedback\n{feedback}\n\n"
        f"# Output Format\n"
        f"Rewrite the plan addressing the feedback above.\n\n"
        f"<think>\n"
        f"...your reasoning...\n"
        f"</think>\n"
        f"<solution>\n"
        f"...your rewritten plan...\n"
        f"</solution>"
    )


# =============================================================================
# Main training loop
# =============================================================================


def main(config: Config):
    load_goal_config(config.goal_dir)
    import co_scientist.grant_proposal.train_buffer_ttt as _tbt
    TARGET_GOAL = _tbt.TARGET_GOAL

    os.makedirs(config.log_path, exist_ok=True)
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    # Signal ablation — must propagate to BOTH the shared reward module AND
    # train_buffer_ttt, which binds `GRADIENT_SIGNALS = ten_signal_reward.SIGNALS`
    # at import time. In-place list mutation (`[:] = ...`) updates every
    # module that imported the list by reference; plain rebinding would
    # miss import-time aliases.
    global GRADIENT_SIGNALS, GRADIENT_WEIGHTS
    if config.disabled_signals:
        disabled = {s.strip() for s in config.disabled_signals.split(",") if s.strip()}
        from co_scientist.shared import grant_signal_reward as _tsr
        from co_scientist.grant_proposal import train_buffer_ttt as _tbt
        active_signals = [s for s in _tsr.SIGNALS if s.id not in disabled]
        active_weights = {s.id: _tsr.SIGNAL_WEIGHTS[s.id] for s in active_signals}
        total_w = sum(active_weights.values())
        if total_w > 0:
            active_weights = {k: v / total_w for k, v in active_weights.items()}
        # Mutate in place so previously-imported aliases (in train_buffer_ttt
        # and elsewhere) see the ablated list.
        _tsr.SIGNALS[:] = active_signals
        _tsr.SIGNAL_WEIGHTS.clear()
        _tsr.SIGNAL_WEIGHTS.update(active_weights)
        # Explicitly rebind the alias in train_buffer_ttt (it used
        # `from ... import SIGNALS as GRADIENT_SIGNALS` which is a
        # separate name binding, not a mutation target).
        _tbt.GRADIENT_SIGNALS = _tsr.SIGNALS
        _tbt.GRADIENT_WEIGHTS = _tsr.SIGNAL_WEIGHTS
        GRADIENT_SIGNALS = _tsr.SIGNALS
        GRADIENT_WEIGHTS = _tsr.SIGNAL_WEIGHTS
        logger.info(f"Signal ablation: disabled {disabled}, {len(active_signals)} active")
        logger.info(f"Reweighted: {active_weights}")

    logger.info(f"TARGET GOAL ({len(TARGET_GOAL)} chars)")
    logger.info(
        f"CR-v7 plan: n_iter={config.n_iterations}, n_fresh={config.n_fresh}, "
        f"n_revise={config.n_revise}, BoN={config.n_revision_candidates}, "
        f"grader_repeats={config.grader_repeats}"
    )
    logger.info(
        f"Paradigm: Per-signal Context REINFORCE | "
        f"Model: {config.model_name} | Grader: {config.grader_model_name} | "
        f"UCB: {config.use_ucb} | HardGates: {not config.skip_hard_gates} | "
        f"SkipRL: {config.skip_rl_update} | TrainFresh: {config.train_on_fresh}"
    )

    rng = random.Random(config.seed)
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Renderer: {renderer_name}")

    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )

    last_checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path)
    if last_checkpoint is None:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank,
        )
        start_iter = 0
        logger.info(f"Fresh start: LoRA rank={config.lora_rank}")
    else:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            last_checkpoint["state_path"]
        )
        start_iter = last_checkpoint["batch"] + 1
        logger.info(f"Resuming from iteration {start_iter}")

    # Grader tokenizer/renderer always derived from grader_model_name — the
    # Opus subagent shim still uses grader_tokenizer for XML encode/decode.
    if config.grader_model_name == config.model_name:
        grader_tokenizer = tokenizer
        grader_renderer = renderer
    else:
        grader_tokenizer = get_tokenizer(config.grader_model_name)
        grader_renderer_name = model_info.get_recommended_renderer_name(config.grader_model_name)
        grader_renderer = renderers.get_renderer(grader_renderer_name, grader_tokenizer)
        logger.info(f"Grader renderer: {grader_renderer_name} (policy: {renderer_name})")

    if config.grader_backend == "opus_subagent":
        grader_client = OpusSubagentGrader(
            log_path=config.log_path,
            tokenizer=grader_tokenizer,
            long_critique=config.grader_long_critique,
            timeout_sec=config.grader_batch_timeout_sec,
        )
        # IMPORTANT: keep grader_renderer (not None) as main_grader_renderer.
        # launch_hard_gate_futures / launch_gradient_futures need it to build
        # ModelInput from plain text prompts. Downstream _decode_grader_output
        # calls renderer.parse_response(xml_tokens) which Qwen3 renderer
        # correctly returns as `{'role':'assistant', 'content': xml_str}` —
        # roundtrip verified. No chat-template mismatch because tokens are
        # encoded via the same tokenizer and the renderer treats raw content
        # as assistant text.
        main_grader_renderer = grader_renderer
        logger.info(
            f"Grader backend=opus_subagent (log_path={config.log_path}, "
            f"long_critique={config.grader_long_critique}, "
            f"timeout={config.grader_batch_timeout_sec}s)"
        )
    else:
        grader_client = service_client.create_sampling_client(
            base_model=config.grader_model_name,
        )
        main_grader_renderer = grader_renderer
    grader_client_alt = None
    renderer_alt = None
    if config.grader_model_alt:
        grader_client_alt = service_client.create_sampling_client(
            base_model=config.grader_model_alt,
        )
        alt_tokenizer = get_tokenizer(config.grader_model_alt)
        alt_renderer_name = model_info.get_recommended_renderer_name(config.grader_model_alt)
        renderer_alt = renderers.get_renderer(alt_renderer_name, alt_tokenizer)
        logger.info(f"Alt grader: {config.grader_model_alt} (for signals with grader_model_override)")
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    retriever = None
    if config.use_retrieval:
        retriever = PaperRetriever()
        logger.info("RAG enabled: phase1=%s, phase2=%s, g3_threshold=%d",
                     config.retrieval_phase1, config.retrieval_phase2, config.retrieval_g3_threshold)

    buffer: list[dict] = []
    buffer_path = os.path.join(config.log_path, "buffer.jsonl")
    if os.path.exists(buffer_path):
        with open(buffer_path) as f:
            for line in f:
                buffer.append(json.loads(line))
        logger.info(f"Resumed buffer with {len(buffer)} entries")

    train_log_path = os.path.join(config.log_path, "train", "training_logs.jsonl")
    os.makedirs(os.path.dirname(train_log_path), exist_ok=True)
    iter_summary_path = os.path.join(config.log_path, "train", "iter_summary.jsonl")

    # =========================================================================
    for iter_idx in range(start_iter, config.n_iterations):
        t_start = time.time()

        # Reset per-iter batch state for the Opus subagent shim (no-op under tinker).
        # Must precede any launch_plan_reward() call this iter.
        if isinstance(grader_client, OpusSubagentGrader):
            grader_client.begin_iter(iter_idx)

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
            name=f"iter_{iter_idx:04d}",
        ).result()
        sampling_client = service_client.create_sampling_client(
            model_path=sampling_result.path,
        )

        # =====================================================================
        # PHASE 1: Generate n_fresh plans
        # =====================================================================
        fresh_samples = []
        fresh_prompt_tokens = []
        fresh_context = None
        if config.fresh_use_context and len(buffer) > 0:
            fresh_context = select_context(buffer, config.K_exploit, config.K_explore, rng)

        # --- RAG: model-driven paper retrieval for fresh generation ---
        fresh_retrieval: RetrievalResult | None = None
        if retriever and config.retrieval_phase1:
            try:
                if config.retrieval_mode == "pipeline":
                    fresh_retrieval = _run_retrieval_pipeline(TARGET_GOAL, retriever, config)
                else:
                    query_prompt = QUERY_GENERATION_PROMPT.format(goal=TARGET_GOAL)
                    fresh_retrieval = _run_retrieval_model(
                        query_prompt, retriever, sampling_client, renderer, config,
                    )
                if fresh_retrieval:
                    logger.info(
                        "Iter %d PHASE1 RAG (%s): %d queries -> %d papers (%.1fs)",
                        iter_idx, config.retrieval_mode,
                        len(fresh_retrieval.queries_used),
                        len(fresh_retrieval.papers), fresh_retrieval.latency_s,
                    )
                else:
                    logger.warning("Iter %d PHASE1 RAG: retrieval returned no results", iter_idx)
            except (httpx.HTTPError, ValueError) as e:
                logger.warning("Iter %d PHASE1 RAG failed: %s: %s", iter_idx, type(e).__name__, e)

        fresh_papers_str = (
            format_papers_for_prompt(
                fresh_retrieval.papers, max_papers=config.retrieval_max_papers,
                max_chars=config.retrieval_max_chars,
            ) if fresh_retrieval else ""
        )
        prompt_text = build_research_plan_prompt(
            goal=TARGET_GOAL, context=fresh_context,
            retrieved_papers=fresh_papers_str or None,
        )
        model_input = renderer.build_generation_prompt(
            [{"role": "user", "content": prompt_text}]
        )
        shared_prompt_ints = model_input.to_ints()
        fresh_future = sampling_client.sample(
            prompt=model_input,
            num_samples=config.n_fresh,
            sampling_params=sampling_params,
        )
        fresh_result = fresh_future.result()
        for seq in fresh_result.sequences:
            parsed = renderer.parse_response(seq.tokens)
            raw_text = _to_str(parsed[0]["content"] if isinstance(parsed[0], dict) else parsed[0])
            text = extract_solution(raw_text)
            fresh_samples.append({"tokens": seq.tokens, "logprobs": seq.logprobs, "text": text})
            fresh_prompt_tokens.append(shared_prompt_ints)

        logger.info(f"Iter {iter_idx}: generated {len(fresh_samples)} fresh plans")

        # Launch grading for fresh plans (parallel, fire-and-forget)
        fresh_futures: list[PlanFutures | None] = []
        for k, sample in enumerate(fresh_samples):
            if len(sample["text"].strip().split()) < config.min_words:
                fresh_futures.append(None)
                continue
            if isinstance(grader_client, OpusSubagentGrader):
                grader_client.set_plan_context(f"fresh_{k}", sample["text"])
            fresh_futures.append(
                launch_plan_reward(
                    plan=sample["text"], goal=TARGET_GOAL,
                    grader_client=grader_client, renderer=main_grader_renderer,
                    n_repeats=config.grader_repeats,
                    grader_max_tokens=config.grader_max_tokens,
                    hg_max_tokens=config.grader_hard_gate_max_tokens,
                    temperature=config.grader_temperature,
                    skip_hard_gates=config.skip_hard_gates,
                    emit_critique=config.emit_signal_critique,
                    include_cot=config.include_cot_scaffolding,
                    grader_client_alt=grader_client_alt, renderer_alt=renderer_alt,
                )
            )

        # =====================================================================
        # PHASE 2: Critique-Revise from buffer
        # =====================================================================
        revise_samples = []
        revise_futures: list[PlanFutures | None] = []

        if iter_idx >= config.cold_start_iters and len(buffer) >= config.n_revise:
            if config.use_ucb:
                total_selections = sum(e.get("n_selected", 0) for e in buffer)
                candidates = ucb_select(buffer, config.n_revise, total_selections, config.ucb_c)
                for buf_idx, _ in candidates:
                    buffer[buf_idx]["n_selected"] = buffer[buf_idx].get("n_selected", 0) + 1
            else:
                valid_entries = [(i, e) for i, e in enumerate(buffer) if e.get("hard_gate_passed")]
                valid_entries.sort(key=lambda x: x[1]["aggregate_reward"], reverse=True)
                candidates = valid_entries[:config.n_revise]

            # Pass 1 — fire all revise sampling futures concurrently
            v7_pending: list[tuple[int, dict, dict, list[int], "tinker.Future", RetrievalResult | None]] = []
            for buf_idx, parent in candidates:
                parent_critiques = parent.get("per_signal_critiques", {}) or {}
                parent_confidence = parent.get("per_signal_confidence", {}) or {}

                # --- RAG: conditional retrieval when G3 is weak ---
                revise_retrieval: RetrievalResult | None = None
                revise_papers_str = ""
                if retriever and config.retrieval_phase2:
                    g3_score = (parent.get("signal_vector") or {}).get("G3_technical_evidence")
                    if g3_score is not None and g3_score <= config.retrieval_g3_threshold:
                        try:
                            g3_critique = (parent_critiques or {}).get("G3_technical_evidence", "")
                            if config.retrieval_mode == "pipeline":
                                revise_retrieval = _run_retrieval_pipeline(TARGET_GOAL, retriever, config)
                            else:
                                rq_prompt = REVISION_QUERY_GENERATION_PROMPT.format(
                                    goal=TARGET_GOAL, g3_score=g3_score, g3_critique=g3_critique,
                                )
                                revise_retrieval = _run_retrieval_model(
                                    rq_prompt, retriever, sampling_client, renderer, config,
                                )
                            if revise_retrieval:
                                revise_papers_str = format_papers_for_prompt(
                                    revise_retrieval.papers,
                                    max_papers=config.retrieval_max_papers,
                                    max_chars=config.retrieval_max_chars,
                                )
                                logger.info(
                                    "Iter %d PHASE2 RAG (parent %d, G3=%s): %d queries -> %d papers",
                                    iter_idx, buf_idx, g3_score,
                                    len(revise_retrieval.queries_used), len(revise_retrieval.papers),
                                )
                        except (httpx.HTTPError, ValueError) as e:
                            logger.warning("Iter %d PHASE2 RAG failed for parent %d: %s", iter_idx, buf_idx, e)

                revise_prompt = build_whole_plan_revision_prompt(
                    goal=TARGET_GOAL,
                    plan_text=parent["plan_text"],
                    signal_vector=parent["signal_vector"],
                    critiques=parent_critiques,
                    strip_critiques=config.strip_critiques,
                    scores_only=config.scores_only,
                    retrieved_papers=revise_papers_str or None,
                )
                revise_input = renderer.build_generation_prompt(
                    [{"role": "user", "content": revise_prompt}]
                )
                revise_tokens_ints = revise_input.to_ints()

                # SDPO: build teacher prompt with full critique (same plan, same scores, + critique text)
                teacher_tokens_ints: list[int] | None = None
                if config.sdpo and config.scores_only:
                    teacher_prompt = build_whole_plan_revision_prompt(
                        goal=TARGET_GOAL,
                        plan_text=parent["plan_text"],
                        signal_vector=parent["signal_vector"],
                        critiques=parent_critiques,
                        strip_critiques=False,
                        scores_only=False,
                        retrieved_papers=revise_papers_str or None,
                    )
                    teacher_input = renderer.build_generation_prompt(
                        [{"role": "user", "content": teacher_prompt}]
                    )
                    teacher_tokens_ints = teacher_input.to_ints()

                future = sampling_client.sample(
                    prompt=revise_input,
                    num_samples=config.n_revision_candidates,
                    sampling_params=sampling_params,
                )
                v7_pending.append((buf_idx, parent, parent_critiques, parent_confidence, revise_tokens_ints, teacher_tokens_ints, future, revise_retrieval))

            # Pass 2 — await all, process candidates
            for buf_idx, parent, parent_critiques, parent_confidence, revise_tokens_ints, teacher_tokens_ints, future, rev_retrieval in v7_pending:
                revise_result = future.result()
                for cand_idx, seq in enumerate(revise_result.sequences):
                    _parsed = renderer.parse_response(seq.tokens)
                    raw_text = _to_str(_parsed[0]["content"] if isinstance(_parsed[0], dict) else _parsed[0])
                    critique_str, revised_plan = parse_critique_and_plan(raw_text)
                    if not revised_plan:
                        revised_plan = extract_solution(raw_text)
                    if detect_duplicated_content(revised_plan):
                        logger.warning(
                            f"Iter {iter_idx} cand {cand_idx}: duplicated content, "
                            "reverting to parent"
                        )
                        revised_plan = parent["plan_text"]
                        critique_str = (critique_str or "") + " [REJECTED: dup]"
                    plan_for_grading = revised_plan

                    revise_samples.append({
                        "parent_idx": buf_idx,
                        "parent_reward": parent["aggregate_reward"],
                        "parent_plan_text": parent["plan_text"],
                        "parent_signal_vector": dict(parent["signal_vector"]),
                        "parent_critiques": dict(parent_critiques),
                        "parent_confidence": dict(parent_confidence),
                        "critique": critique_str or "",
                        "text": plan_for_grading,
                        "tokens": seq.tokens,
                        "logprobs": seq.logprobs,
                        "revise_tokens_ints": revise_tokens_ints,
                        "teacher_tokens_ints": teacher_tokens_ints,
                        "candidate_idx": cand_idx,
                        "retrieved_papers": [p.to_dict() for p in rev_retrieval.papers] if rev_retrieval else None,
                        "retrieval_queries": rev_retrieval.queries_used if rev_retrieval else None,
                    })
                    if len(plan_for_grading.strip().split()) >= config.min_words:
                        if isinstance(grader_client, OpusSubagentGrader):
                            grader_client.set_plan_context(
                                f"rev_{buf_idx}_cand_{cand_idx}", plan_for_grading,
                            )
                        revise_futures.append(
                            launch_plan_reward(
                                plan=plan_for_grading, goal=TARGET_GOAL,
                                grader_client=grader_client, renderer=main_grader_renderer,
                                n_repeats=config.grader_repeats,
                                grader_max_tokens=config.grader_max_tokens,
                                hg_max_tokens=config.grader_hard_gate_max_tokens,
                                temperature=config.grader_temperature,
                                skip_hard_gates=config.skip_hard_gates,
                                emit_critique=config.emit_signal_critique,
                                include_cot=config.include_cot_scaffolding,
                                grader_client_alt=grader_client_alt, renderer_alt=renderer_alt,
                            )
                        )
                    else:
                        revise_futures.append(None)

            logger.info(
                f"Iter {iter_idx}: generated {len(revise_samples)} revision candidates "
                f"for {len(candidates)} parents (best-of-{config.n_revision_candidates})"
            )

        # =====================================================================
        # Collect all grading results
        # =====================================================================
        logger.info(f"Iter {iter_idx}: collecting grading results...")
        t_collect = time.time()

        fresh_rewards: list[TenSignalReward | None] = []
        for fut in fresh_futures:
            if fut is None:
                fresh_rewards.append(None)
                continue
            try:
                fresh_rewards.append(collect_plan_reward(fut, grader_tokenizer, renderer=main_grader_renderer, skip_hard_gates=config.skip_hard_gates))
            except Exception as e:
                logger.error(f"Fresh plan grading failed: {type(e).__name__}")
                fresh_rewards.append(None)

        revise_rewards: list[TenSignalReward | None] = []
        for fut in revise_futures:
            if fut is None:
                revise_rewards.append(None)
                continue
            try:
                revise_rewards.append(collect_plan_reward(fut, grader_tokenizer, renderer=main_grader_renderer, skip_hard_gates=config.skip_hard_gates))
            except Exception as e:
                logger.error(f"Revision grading failed: {type(e).__name__}")
                revise_rewards.append(None)

        logger.info(f"Iter {iter_idx}: collection complete in {time.time() - t_collect:.0f}s")

        # =====================================================================
        # Add to buffer (fresh + best-of-N revisions)
        # =====================================================================
        for i, (sample, reward) in enumerate(zip(fresh_samples, fresh_rewards)):
            if reward is None:
                continue
            entry = {
                "iteration": iter_idx,
                "plan_text": sample["text"],
                "signal_vector": {k: v for k, v in reward.signal_vector.items() if v is not None},
                "hard_gate_passed": reward.hard_gate_passed,
                "hard_gate_failed_name": reward.hard_gate_failed_name,
                "aggregate_reward": reward.aggregate_reward,
                "goal_contrast_diag": reward.goal_contrast_diag,
                "claim_verification_diag": reward.claim_verification_diag,
                "per_repeat_scores": reward.per_repeat_scores,
                "per_signal_critiques": dict(reward.per_signal_critiques),
                "per_signal_confidence": dict(getattr(reward, "per_signal_confidence", {}) or {}),
                "word_count": len(sample["text"].strip().split()),
                "entry_type": "fresh",
                "parent_idx": None,
                "delta_reward": None,
                "retrieved_papers": [p.to_dict() for p in fresh_retrieval.papers] if fresh_retrieval else None,
                "retrieval_queries": fresh_retrieval.queries_used if fresh_retrieval else None,
                "citation_match": citation_match_rate(sample["text"], fresh_retrieval.papers) if fresh_retrieval else None,
            }
            buffer.append(entry)
            with open(buffer_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Best-of-N filter per parent: keep the candidate with highest delta
        from collections import defaultdict
        parent_candidates: dict[int, list[tuple[dict, TenSignalReward, float]]] = defaultdict(list)
        for rsample, rreward in zip(revise_samples, revise_rewards):
            if rreward is None:
                continue
            delta = rreward.aggregate_reward - rsample["parent_reward"]
            parent_candidates[rsample["parent_idx"]].append((rsample, rreward, delta))

        revision_deltas = []
        best_revisions: list[tuple[dict, TenSignalReward, float]] = []
        for parent_idx, cands in parent_candidates.items():
            best_cand = max(cands, key=lambda x: x[2])
            best_revisions.append(best_cand)
            rsample, rreward, delta = best_cand
            revision_deltas.append(delta)

            entry = {
                "iteration": iter_idx,
                "plan_text": rsample["text"],
                "signal_vector": {k: v for k, v in rreward.signal_vector.items() if v is not None},
                "hard_gate_passed": rreward.hard_gate_passed,
                "hard_gate_failed_name": rreward.hard_gate_failed_name,
                "aggregate_reward": rreward.aggregate_reward,
                "goal_contrast_diag": rreward.goal_contrast_diag,
                "claim_verification_diag": rreward.claim_verification_diag,
                "per_repeat_scores": rreward.per_repeat_scores,
                "per_signal_critiques": dict(rreward.per_signal_critiques),
                "per_signal_confidence": dict(getattr(rreward, "per_signal_confidence", {}) or {}),
                "word_count": len(rsample["text"].strip().split()),
                "entry_type": "revision",
                "parent_idx": rsample["parent_idx"],
                "critique_text": rsample.get("critique"),
                "delta_reward": delta,
                "n_candidates": len(cands),
                "best_of_n_deltas": sorted([d for _, _, d in cands], reverse=True),
                "retrieved_papers": rsample.get("retrieved_papers"),
                "retrieval_queries": rsample.get("retrieval_queries"),
            }
            buffer.append(entry)
            with open(buffer_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Count positive-delta revisions across ALL candidates (not just
        # the best-of-N survivors) — reported in iter_summary for
        # visibility into how often the revise step actually helps.
        all_valid_revisions = [
            (rs, rr, rr.aggregate_reward - rs["parent_reward"])
            for rs, rr in zip(revise_samples, revise_rewards)
            if rr is not None
        ]
        n_positive_revisions = sum(1 for _, _, d in all_valid_revisions if d > 0)

        # =====================================================================
        # PHASE 3: Per-signal REINFORCE loss construction
        # =====================================================================
        # Option (c) HER-style relabelling: stored logprob = recomputed
        # under context_i, not the sampling context. IS ratio = exp(new -
        # stored) = 1 → importance_sampling loss reduces to
        # L_i = -A_i · logπ(a | context_i) — per-signal REINFORCE on the
        # context_i distribution. Bias is intentional: we want the
        # single-critique gradient direction driven by the outcome of an
        # all-hints rewrite.
        training_datums = []
        beta = 0.0

        valid_fresh = [
            (s, r, i) for i, (s, r) in enumerate(zip(fresh_samples, fresh_rewards)) if r is not None
        ]

        # Ablation: train_on_fresh (entropic weights on fresh plans)
        if config.train_on_fresh and len(valid_fresh) >= 2:
            from co_scientist.grant_proposal.train_buffer_ttt import entropic_weights
            if config.diversity_method == "gapo":
                fresh_rewards_arr = gapo_diversity_adjusted_rewards(
                    [r for _, r, _ in valid_fresh],
                    bonus_scale=config.diversity_bonus_scale,
                )
            else:
                fresh_rewards_arr = np.array([r.aggregate_reward for _, r, _ in valid_fresh])
                if config.novelty_weight > 0:
                    for idx_f, (sample, _, _) in enumerate(valid_fresh):
                        nov = compute_novelty(sample["text"], buffer)
                        fresh_rewards_arr[idx_f] *= (1.0 + config.novelty_weight * nov)
            fresh_weights, beta = entropic_weights(
                fresh_rewards_arr, config.kl_budget, config.beta_max,
            )
            for (sample, reward, orig_idx), weight in zip(valid_fresh, fresh_weights):
                prompt_tokens = [int(t) for t in fresh_prompt_tokens[orig_idx]]
                gen_tokens = [int(t) for t in sample["tokens"]]
                full_seq = prompt_tokens + gen_tokens
                ob_len = len(prompt_tokens) - 1
                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=full_seq[:-1]),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(
                            torch.tensor(full_seq[1:], dtype=torch.long)
                        ),
                        "logprobs": TensorData.from_torch(
                            torch.tensor(
                                [0.0] * ob_len + sample["logprobs"], dtype=torch.float,
                            )
                        ),
                        "advantages": TensorData.from_torch(
                            torch.tensor(
                                [0.0] * ob_len + [float(weight)] * len(sample["logprobs"]),
                                dtype=torch.float,
                            )
                        ),
                    },
                )
                training_datums.append(datum)

        # Per-revision RL on revision candidates (three modes: sdpo, aggregate, per_signal)
        sdpo_stats = {"mean_adv": 0.0, "pos_frac": 0.0, "n_datums": 0}
        rl_mode = "sdpo" if config.sdpo else config.revision_rl_mode
        n_datums_before_revision = len(training_datums)
        if all_valid_revisions and not config.skip_revision_rl:

            if rl_mode == "sdpo":
                # =============================================================
                # SDPO: per-token advantage from self-teacher (Hübotter et al. 2026)
                # A^SDPO_t = log[π(y_t | x,f,y_{<t}) / π(y_t | x,y_{<t})]
                # =============================================================
                sdpo_work = []
                for rsample, rreward, _ in all_valid_revisions:
                    if not rreward.hard_gate_passed:
                        continue
                    gen_tokens = list(rsample["tokens"])
                    t_tokens = rsample.get("teacher_tokens_ints")
                    if not gen_tokens or not t_tokens:
                        continue
                    sdpo_work.append((
                        list(rsample["revise_tokens_ints"]),
                        t_tokens,
                        gen_tokens,
                        list(rsample["logprobs"]),
                    ))

                if sdpo_work:
                    teacher_inputs = [
                        types.ModelInput.from_ints(tokens=t_prompt + gen)
                        for (_, t_prompt, gen, _) in sdpo_work
                    ]
                    teacher_logprobs_all = [
                        sampling_client.compute_logprobs(mi).result()
                        for mi in teacher_inputs
                    ]

                    all_advs = []
                    for (s_prompt, t_prompt, gen_tokens, s_logprobs), t_full_logprobs in zip(
                        sdpo_work, teacher_logprobs_all,
                    ):
                        t_prompt_len = len(t_prompt)
                        t_gen_logprobs = t_full_logprobs[t_prompt_len : t_prompt_len + len(gen_tokens)]
                        if len(t_gen_logprobs) != len(gen_tokens):
                            logger.warning(
                                "SDPO: teacher logprob length mismatch: "
                                f"got {len(t_gen_logprobs)}, expected {len(gen_tokens)}; skipping."
                            )
                            continue
                        if len(s_logprobs) != len(gen_tokens):
                            logger.warning(
                                "SDPO: student logprob length mismatch: "
                                f"got {len(s_logprobs)}, expected {len(gen_tokens)}; skipping."
                            )
                            continue

                        per_token_adv = []
                        for t_lp, s_lp in zip(t_gen_logprobs, s_logprobs):
                            a = float(t_lp - s_lp) * config.sdpo_scale
                            a = max(-config.sdpo_clip_advantage, min(config.sdpo_clip_advantage, a))
                            per_token_adv.append(a)

                        all_advs.extend(per_token_adv)

                        full_seq = s_prompt + gen_tokens
                        ob_len = len(s_prompt) - 1
                        datum = types.Datum(
                            model_input=types.ModelInput.from_ints(tokens=full_seq[:-1]),
                            loss_fn_inputs={
                                "target_tokens": TensorData.from_torch(
                                    torch.tensor(full_seq[1:], dtype=torch.long)),
                                "logprobs": TensorData.from_torch(
                                    torch.tensor([0.0] * ob_len + s_logprobs, dtype=torch.float)),
                                "advantages": TensorData.from_torch(
                                    torch.tensor([0.0] * ob_len + per_token_adv, dtype=torch.float)),
                            },
                        )
                        training_datums.append(datum)

                    if all_advs:
                        sdpo_stats["mean_adv"] = float(np.mean(all_advs))
                        sdpo_stats["pos_frac"] = float(np.mean([1.0 if a > 0 else 0.0 for a in all_advs]))
                    sdpo_stats["n_datums"] = len(training_datums) - n_datums_before_revision
                    logger.info(
                        f"Iter {iter_idx} SDPO: {sdpo_stats['n_datums']} datums, "
                        f"mean_adv={sdpo_stats['mean_adv']:.4f}, pos_frac={sdpo_stats['pos_frac']:.3f}"
                    )

            elif rl_mode == "aggregate":
                # =============================================================
                # Aggregate REINFORCE: full-context + scalar aggregate delta
                # =============================================================
                for rsample, rreward, agg_delta in all_valid_revisions:
                    if not rreward.hard_gate_passed:
                        continue
                    gen_tokens = list(rsample["tokens"])
                    if not gen_tokens:
                        continue
                    if abs(agg_delta) < config.delta_threshold:
                        continue
                    s_prompt = list(rsample["revise_tokens_ints"])
                    s_logprobs = list(rsample["logprobs"])
                    if len(s_logprobs) != len(gen_tokens):
                        continue
                    A_agg = float(agg_delta) * config.delta_scale
                    full_seq = s_prompt + gen_tokens
                    ob_len = len(s_prompt) - 1
                    datum = types.Datum(
                        model_input=types.ModelInput.from_ints(tokens=full_seq[:-1]),
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(
                                torch.tensor(full_seq[1:], dtype=torch.long)),
                            "logprobs": TensorData.from_torch(
                                torch.tensor([0.0] * ob_len + s_logprobs, dtype=torch.float)),
                            "advantages": TensorData.from_torch(
                                torch.tensor([0.0] * ob_len + [A_agg] * len(gen_tokens), dtype=torch.float)),
                        },
                    )
                    training_datums.append(datum)

            else:
                # =============================================================
                # Per-signal HER REINFORCE (existing default)
                # =============================================================
                per_datum_work: list[tuple[str, float, list[int], list[int]]] = []
                for rsample, rreward, _ in all_valid_revisions:
                    if not rreward.hard_gate_passed:
                        continue
                    parent_sv = rsample["parent_signal_vector"]
                    parent_critiques = rsample["parent_critiques"]
                    parent_plan = rsample["parent_plan_text"]
                    if len(parent_plan) > MAX_PLAN_CHARS_IN_PROMPT:
                        parent_plan = parent_plan[:MAX_PLAN_CHARS_IN_PROMPT] + "\n\n[... rest truncated ...]"
                    gen_tokens = list(rsample["tokens"])
                    if not gen_tokens:
                        continue
                    parent_conf = rsample.get("parent_confidence", {}) or {}
                    rev_conf = getattr(rreward, "per_signal_confidence", {}) or {}
                    for spec in GRADIENT_SIGNALS:
                        sid = spec.id
                        s_parent = parent_sv.get(sid)
                        s_rev = rreward.signal_vector.get(sid)
                        if s_parent is None or s_rev is None:
                            continue
                        delta_i = (s_rev - s_parent) / (spec.score_max - 1)
                        if abs(delta_i) < config.delta_threshold:
                            continue
                        # D4v2: confidence filter. Skip if either parent or
                        # revision grading on this signal has low confidence.
                        if config.min_grader_confidence > 0.0:
                            pc = parent_conf.get(sid)
                            rc = rev_conf.get(sid)
                            min_conf = min(
                                v for v in (pc, rc) if v is not None
                            ) if (pc is not None or rc is not None) else None
                            if min_conf is not None and min_conf < config.min_grader_confidence:
                                continue
                        ctx_text = build_per_signal_context(
                            plan_text=parent_plan,
                            critique_i=parent_critiques.get(sid, ""),
                            signal_name=spec.name,
                            score=s_parent,
                            goal=TARGET_GOAL,
                            score_max=spec.score_max,
                            scores_only=config.scores_only,
                            signal_question=spec.question,
                        )
                        ctx_tokens = renderer.build_generation_prompt(
                            [{"role": "user", "content": ctx_text}]
                        ).to_ints()
                        if len(ctx_tokens) < 1:
                            continue
                        per_datum_work.append((sid, delta_i, ctx_tokens, gen_tokens))

                if per_datum_work:
                    model_inputs = [
                        types.ModelInput.from_ints(tokens=ctx + gen)
                        for (_, _, ctx, gen) in per_datum_work
                    ]

                    async def _batch() -> list[list[float]]:
                        return await asyncio.gather(
                            *[sampling_client.compute_logprobs_async(s) for s in model_inputs]
                        )

                    try:
                        logprobs_all = asyncio.run(_batch())
                    except RuntimeError:
                        logprobs_all = [
                            sampling_client.compute_logprobs(mi) for mi in model_inputs
                        ]

                    for (sid, delta_i, ctx_tokens, gen_tokens), full_logprobs in zip(
                        per_datum_work, logprobs_all,
                    ):
                        prompt_len = len(ctx_tokens)
                        gen_logprobs = full_logprobs[prompt_len : prompt_len + len(gen_tokens)]
                        if len(gen_logprobs) != len(gen_tokens):
                            logger.warning(
                                f"CR-v7: logprob length mismatch for {sid}: "
                                f"got {len(gen_logprobs)}, expected {len(gen_tokens)}; skipping."
                            )
                            continue
                        mean_logprob = sum(gen_logprobs) / len(gen_logprobs)
                        if mean_logprob < -10.0:
                            continue

                        A_i = float(delta_i) * config.delta_scale
                        full_seq = ctx_tokens + gen_tokens
                        ob_len = prompt_len - 1
                        datum = types.Datum(
                            model_input=types.ModelInput.from_ints(tokens=full_seq[:-1]),
                            loss_fn_inputs={
                                "target_tokens": TensorData.from_torch(
                                    torch.tensor(full_seq[1:], dtype=torch.long)
                                ),
                                "logprobs": TensorData.from_torch(
                                    torch.tensor(
                                        [0.0] * ob_len + list(gen_logprobs),
                                        dtype=torch.float,
                                    )
                                ),
                                "advantages": TensorData.from_torch(
                                    torch.tensor(
                                        [0.0] * ob_len + [A_i] * len(gen_tokens),
                                        dtype=torch.float,
                                    )
                                ),
                            },
                        )
                        training_datums.append(datum)

        # Apply update
        if training_datums and not config.skip_rl_update:
            try:
                fwd_bwd_future = training_client.forward_backward(
                    training_datums, loss_fn="importance_sampling",
                )
                optim_step_future = training_client.optim_step(adam_params)
                _ = fwd_bwd_future.result()
                _ = optim_step_future.result()
            except Exception:
                logger.exception(f"Iter {iter_idx}: training step failed")

        # =====================================================================
        # Logging
        # =====================================================================
        all_rewards = (
            [r.aggregate_reward for _, r, _ in valid_fresh]
            + [rr.aggregate_reward for rs, rr in zip(revise_samples, revise_rewards) if rr is not None]
        )
        buffer_max = max((e["aggregate_reward"] for e in buffer if e.get("hard_gate_passed")), default=0.0)

        per_signal_means = {}
        all_valid_rewards = (
            [r for _, r, _ in valid_fresh]
            + [rr for rr in revise_rewards if rr is not None]
        )
        for spec in GRADIENT_SIGNALS:
            scores = [r.signal_vector.get(spec.id) for r in all_valid_rewards
                      if r.hard_gate_passed and r.signal_vector.get(spec.id) is not None]
            if scores:
                per_signal_means[f"signal/{spec.id}/mean"] = float(np.mean(scores))

        fresh_novelties = [compute_novelty(s["text"], buffer) for s, _, _ in valid_fresh] if valid_fresh else []
        iter_summary = {
            "iter": iter_idx,
            "time/total": time.time() - t_start,
            "fresh/count": len(valid_fresh),
            "fresh/reward_mean": float(np.mean([r.aggregate_reward for _, r, _ in valid_fresh])) if valid_fresh else 0,
            "fresh/novelty_mean": float(np.mean(fresh_novelties)) if fresh_novelties else 0,
            "revise/count": len(revise_samples),
            "revise/n_positive": n_positive_revisions,
            "revise/mean_delta": float(np.mean(revision_deltas)) if revision_deltas else 0,
            "revise/max_delta": float(max(revision_deltas)) if revision_deltas else 0,
            "reward/mean": float(np.mean(all_rewards)) if all_rewards else 0,
            "reward/max": float(max(all_rewards)) if all_rewards else 0,
            "reward/buffer_max": buffer_max,
            "beta": beta,
            "buffer/size": len(buffer),
            "datums/count": len(training_datums),
            "rl_mode": rl_mode,
            **({"sdpo/mean_advantage": sdpo_stats["mean_adv"],
                "sdpo/positive_frac": sdpo_stats["pos_frac"],
                "sdpo/n_datums": sdpo_stats["n_datums"]} if config.sdpo else {}),
            # D4v2: grader confidence stats
            **_confidence_stats(valid_fresh, revise_rewards),
            **per_signal_means,
        }
        with open(iter_summary_path, "a") as f:
            f.write(json.dumps(iter_summary) + "\n")

        with open(train_log_path, "a") as f:
            for i, (sample, reward) in enumerate(zip(fresh_samples, fresh_rewards)):
                if reward is None:
                    continue
                f.write(json.dumps({
                    "iter": iter_idx, "type": "fresh", "sample_idx": i,
                    "text": sample["text"],
                    "aggregate_reward": reward.aggregate_reward,
                    "signal_vector": reward.signal_vector,
                    "hard_gate_passed": reward.hard_gate_passed,
                }, ensure_ascii=False) + "\n")
            for i, (rsample, rreward) in enumerate(zip(revise_samples, revise_rewards)):
                if rreward is None:
                    continue
                f.write(json.dumps({
                    "iter": iter_idx, "type": "revision", "sample_idx": i,
                    "text": rsample["text"],
                    "aggregate_reward": rreward.aggregate_reward,
                    "signal_vector": rreward.signal_vector,
                    "hard_gate_passed": rreward.hard_gate_passed,
                    "parent_reward": rsample["parent_reward"],
                    "critique": rsample.get("critique", ""),
                    "delta": rreward.aggregate_reward - rsample["parent_reward"],
                }, ensure_ascii=False) + "\n")

        ml_logger.log_metrics(iter_summary, step=iter_idx)
        logger.info(
            f"Iter {iter_idx}: reward mean={iter_summary['reward/mean']:.3f} "
            f"max={iter_summary['reward/max']:.3f} buffer_max={buffer_max:.3f} "
            f"| revisions: {n_positive_revisions}/{len(revise_samples)} positive "
            f"(mean_delta={iter_summary['revise/mean_delta']:.3f}) "
            f"| time={iter_summary['time/total']:.0f}s"
        )

    # Final checkpoint (guard against resume-complete case where the
    # for-loop never executes and iter_idx is unbound).
    if start_iter < config.n_iterations:
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
