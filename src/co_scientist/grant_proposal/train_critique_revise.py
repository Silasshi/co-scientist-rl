"""Signal-Targeted Critique-Revise trainer for grant proposal generation.

Forked from ttt_discover/train_critique_revise.py for D4 (grant_proposal).
Contains shared utilities (ucb_select, parse_critique_and_plan, etc.) and
legacy CR-v4/v5/v6 paths.

Usage:
    source tools/use_api_profile.sh new
    python src/co_scientist/grant_proposal/train_critique_revise.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import sys
import textwrap
import time
import re
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

# Import shared components from the buffer-TTT trainer
from co_scientist.grant_proposal.train_buffer_ttt import (
    TARGET_GOAL, TARGET_TARGET, ALT_GOALS,
    BufferEntry, TenSignalReward, PlanFutures,
    build_research_plan_prompt,
    select_context,
    launch_plan_reward, collect_plan_reward,
    adaptive_beta, entropic_weights,
    check_format_compliance,
    _softmax, _entropy,
)
from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.grant_signal_reward import (
    SIGNALS as GRADIENT_SIGNALS,
    SIGNAL_WEIGHTS as GRADIENT_WEIGHTS,
    aggregate_reward as gradient_aggregate,
    build_single_signal_prompt,
    LocusEntry,
    parse_locus,
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

    log_path: str = "/home/silas/co-scientist-project/projects/grant_proposal/runs/default"
    model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"  # policy = grader = same model for TTT

    # Loop
    n_iterations: int = 25
    n_fresh: int = 4               # fresh plans per iter (exploration)
    n_revise: int = 4              # critique-revise plans per iter (exploitation)
    n_revision_candidates: int = 2 # P3.1: Best-of-N — generate N candidates per revision, keep best
    grader_repeats: int = 2        # N: grader calls per signal per plan

    # Buffer context selection
    K_exploit: int = 3
    K_explore: int = 2
    cold_start_iters: int = 1      # first N iters: only fresh, no revisions

    # P3.2: Skip signals with historically low revision success rate
    min_revision_success_rate: float = 0.05  # skip signal if < 5% historical success
    min_revision_attempts: int = 8           # need at least N attempts before skipping

    # Ablation: skip RL update (for no-training baseline)
    skip_rl_update: bool = False             # if True, no gradient updates (pure in-context learning)

    # Ablation: skip hard gates (for Layer 0 ablation)
    skip_hard_gates: bool = False            # if True, skip Goal-Contrast + Claim Verification

    # Buffer selection: UCB vs top-K
    use_ucb: bool = True                     # if True, use UCB selection; if False, use top-K
    ucb_c: float = 1.0                       # UCB exploration coefficient

    # Ablation: disable specific signals (comma-separated IDs, e.g. "S9_focus" or "S1_depth")
    disabled_signals: str = ""               # empty = all signals active

    # Ablation: train on fresh plans with entropic weights (for B3/A8)
    train_on_fresh: bool = False             # if True, re-enable entropic training on fresh plans

    # Entropic objective (for fresh plans, only used when train_on_fresh=True)
    kl_budget: float = 0.693
    beta_max: float = 20.0

    # Diversity seeding: approach seed + negative conditioning on fresh plan generation
    diversity_seeding: bool = False  # A9 ablation: assign methodological seeds to fresh plans

    # Revision delta scaling
    delta_scale: float = 5.0       # scale delta to make it comparable to entropic weights

    # CR-v6 C3 mixed advantage: advantage = α·Δ_target + β·Δ_aggregate
    # Δ_target = per-signal delta on the bottleneck, normalized [-1, 1].
    # Δ_aggregate = whole-reward delta, already in [-1, 1].
    # Applies when locus_based_edit=True OR paragraph_level_edit=True.
    c3_alpha: float = 0.7  # weight on targeted signal delta
    c3_beta: float = 0.3   # weight on aggregate delta

    # Paragraph-level editing: only rewrite the section related to bottleneck signal
    paragraph_level_edit: bool = True  # CR-v5: paragraph-level editing is default

    # CR-v6: locus-based revision (grader emits verbatim spans → exact-string replace).
    # Takes priority over paragraph_level_edit when True. Falls back to paragraph/
    # whole-plan path when locus grading returns None for a given parent.
    locus_based_edit: bool = False  # Phase 1: default False; flip after pilot

    # CR-v6: signals always excluded from bottleneck targeting (comma-separated IDs).
    # S3_positioning: goal-specific Qwen3 grader prior hallucination on TTT-Discover
    # (8% true-CORRECT locus validity — Phase 0 report). S3 is still GRADED
    # (weight=0.12) and contributes to aggregate reward; it's only excluded from
    # locus revision targeting.
    revision_skip_signals: str = "S3_positioning"

    # CR-v6 locus pass uses more output tokens (locus block adds ~500 tokens)
    locus_grader_max_tokens: int = 6144

    # CR-v7 (2026-04-17): per-signal context REINFORCE mode.
    # When revision_mode == "whole_plan_per_signal":
    #   - grader emits <critique> per signal alongside <score>; critiques
    #     are stored per buffer entry (per_signal_critiques)
    #   - revise phase presents all per-signal critiques to policy in one prompt,
    #     samples one whole-plan rewrite
    #   - loss construction: N datums per revision (one per signal), each with context_i
    #     containing only critique_i; advantage A_i = Δ_i/4 · delta_scale
    #   - legacy locus_based_edit / paragraph_level_edit / c3_* ignored
    # Other values: "locus" (CR-v6), "paragraph" (CR-v5), "whole" (CR-v4).
    revision_mode: str = "locus"

    # CR-v7: grader emits <critique> block in evaluation output.
    # Cheap to leave on; extra ~60-100 tokens of grader output per signal.
    # If revision_mode != "whole_plan_per_signal", the critiques are
    # harmlessly stored but not consumed.
    emit_signal_critique: bool = True

    # CR-v7: skip per-signal datum when |Δ_i| below threshold (saves
    # forward-backward compute on signals that didn't meaningfully move).
    # 1e-3 ~ sub-0.5% of one raw-score step; use 0.25 (= one full point
    # on 4-scale) to keep only signals with at least one integer score
    # change.
    delta_threshold: float = 1e-3

    # Optimization
    learning_rate: float = 4e-5
    clip_eps: float = 0.2
    lora_rank: int = 64

    # Sampling
    max_length: int = 32768
    max_tokens: int = 2048
    grader_max_tokens: int = 4096  # v8.1 rubrics use explicit counting CoT, need more tokens
    grader_hard_gate_max_tokens: int = 8192
    temperature: float = 1.0
    grader_temperature: float = 0.0

    # Format constraints
    max_word_count: int = 750
    min_words: int = 30

    # Logging / checkpoint
    save_every: int = 5
    today_date: str = time.strftime("%Y-%m-%d", time.localtime())
    seed: int = 0


# =============================================================================
# Critique-Revise prompt
# =============================================================================


# =============================================================================
# Diversity seeding: approach seeds for brainstorming-style fresh plan generation
# =============================================================================

APPROACH_SEEDS = [
    "evolutionary and population-based methods (e.g., genetic programming, MAP-Elites, quality-diversity)",
    "information-theoretic objectives (e.g., entropy maximization, mutual information, curiosity-driven exploration)",
    "Bayesian and probabilistic methods (e.g., Bayesian optimization, Thompson sampling, posterior inference)",
    "meta-learning and learning-to-learn (e.g., MAML, hypernetworks, learned optimizers)",
    "game-theoretic and adversarial methods (e.g., self-play, minimax, adversarial training)",
    "hierarchical planning and decomposition (e.g., subgoal discovery, options framework, divide-and-conquer)",
    "neuro-symbolic and program synthesis approaches (e.g., DSL-guided search, neural program induction)",
    "reward shaping and intrinsic motivation (e.g., novelty search, empowerment, surprise minimization)",
]


def build_diverse_fresh_prompt(
    goal: str,
    approach_seed: str,
    context: "list[BufferEntry] | None" = None,
    existing_summaries: list[str] | None = None,
) -> str:
    """Build fresh plan prompt with approach seeding + negative conditioning.

    Same structure as build_research_plan_prompt but with:
    1. A methodological focus seed (forces different approach per plan)
    2. Negative conditioning (summaries of plans to avoid repeating)
    """
    prompt = textwrap.dedent(f"""
        I have a research goal. Write a research plan for it.

        Goal: {goal}

        # Methodological Focus
        Focus on {approach_seed}.
    """).strip()

    if existing_summaries:
        summaries_text = "\n".join(f"  - {s}" for s in existing_summaries)
        prompt += "\n\n" + textwrap.dedent(f"""
        # Approaches Already Proposed
        The following approaches were proposed in earlier plans. Choose a different methodology.

{summaries_text}
        """).strip()

    if context:
        prompt += "\n\n" + textwrap.dedent(f"""
        # Past Attempts
        Below are {len(context)} past attempts at this goal with their evaluation scores (1-5 per dimension).
        """).strip()
        for i, entry in enumerate(context, 1):
            sig_str = " ".join(f"{k.split('_', 1)[0]}={v}" for k, v in entry.signal_vector.items())
            prompt += f"\n\n## Past Attempt {i}  (aggregate: {entry.aggregate_reward:.3f})\n"
            prompt += f"Signal scores: {sig_str}\n\n"
            prompt += entry.plan_text.strip()[:8000]
        prompt += "\n\n---\n"

    prompt += "\n\n" + textwrap.dedent("""
        # Output Format
        <think>
        ...your reasoning...
        </think>
        <solution>
        ...your research plan...
        </solution>
    """).strip()

    return prompt


def _one_line_summary(plan_text: str, max_len: int = 120) -> str:
    """Extract a one-line summary for negative conditioning."""
    for line in plan_text.split("\n"):
        line = line.strip()
        if len(line) > 30 and not line.startswith("#") and not line.startswith("**"):
            return line[:max_len]
    return plan_text[:max_len]


def ucb_select(
    buffer: list[dict],
    n_select: int,
    total_selections: int,
    c: float,
) -> list[tuple[int, dict]]:
    """Select plans from buffer using Upper Confidence Bound.

    UCB(plan) = aggregate_reward + c * sqrt(ln(N) / (1 + n_selected))
    """
    import math
    valid = [(i, e) for i, e in enumerate(buffer) if e.get("hard_gate_passed")]
    if len(valid) <= n_select:
        return valid
    scored = []
    for buf_idx, entry in valid:
        reward = entry["aggregate_reward"]
        n_sel = entry.get("n_selected", 0)
        if total_selections > 0:
            exploration = c * math.sqrt(math.log(total_selections + 1) / (1 + n_sel))
        else:
            exploration = c
        scored.append((buf_idx, entry, reward + exploration))
    scored.sort(key=lambda x: x[2], reverse=True)
    return [(idx, entry) for idx, entry, _ in scored[:n_select]]


def identify_bottleneck(
    signal_vector: dict[str, int | None],
    skip_signals: set[str] | None = None,
) -> tuple[str, int]:
    """Find the signal with the lowest score, optionally skipping certain signals.

    P3.2: If skip_signals is provided, those signals are excluded from
    bottleneck consideration (because they have historically low revision
    success rates). Falls back to any signal if all are skipped.
    """
    skip = skip_signals or set()
    valid = {k: v for k, v in signal_vector.items() if v is not None and k not in skip}
    if not valid:
        # Fallback: use all signals (don't skip if everything would be skipped)
        valid = {k: v for k, v in signal_vector.items() if v is not None}
    if not valid:
        return GRADIENT_SIGNALS[0].id, 1
    bottleneck_id = min(valid, key=valid.get)
    return bottleneck_id, valid[bottleneck_id]


def compute_signal_revision_stats(buffer: list[dict]) -> tuple[dict[str, float], dict[str, int]]:
    """Compute per-signal revision success rate from buffer history.

    Returns (success_rates, attempt_counts).
    """
    from collections import defaultdict
    attempts: dict[str, int] = defaultdict(int)
    successes: dict[str, int] = defaultdict(int)
    for entry in buffer:
        if entry.get("entry_type") == "revision" and entry.get("targeted_signal"):
            sig = entry["targeted_signal"]
            attempts[sig] += 1
            if (entry.get("delta_reward") or 0) > 0:
                successes[sig] += 1
    rates = {sig: successes[sig] / attempts[sig] for sig in attempts if attempts[sig] > 0}
    return rates, dict(attempts)


def get_signal_spec(signal_id: str) -> SignalSpec:
    """Look up a signal spec by ID."""
    for spec in GRADIENT_SIGNALS:
        if spec.id == signal_id:
            return spec
    return GRADIENT_SIGNALS[0]


# =============================================================================
# CR-v6: Locus-based revision helpers
# =============================================================================


def grade_bottleneck_with_locus(
    parent_plan: str,
    bottleneck_id: str,
    goal: str,
    grader_client,
    renderer,
    tokenizer,
    grader_max_tokens: int,
    temperature: float,
) -> list[LocusEntry] | None:
    """CR-v6 Pass 2: re-grade the bottleneck signal with locus directive.

    Makes one synchronous grader call (N=1) on the PARENT plan's bottleneck
    signal, requesting a `<locus>` block of verbatim weak-span quotes.

    Returns parsed + verbatim-validated loci, or None if:
    - The signal has no locus_directive (e.g., S9_focus)
    - Grading call fails
    - `parse_locus` rejects (malformed JSON, hallucinated quote)

    Callers treat None as "fall back to legacy paragraph/whole-plan path."
    Cost: +1 grader call per parent, +6% vs Pass 1.
    """
    spec = get_signal_spec(bottleneck_id)
    if not spec.locus_directive:
        return None

    prompt = build_single_signal_prompt(goal, parent_plan, spec, emit_locus=True)
    model_input = renderer.build_generation_prompt(
        [{"role": "user", "content": prompt}]
    )
    try:
        future = grader_client.sample(
            model_input,
            num_samples=1,
            sampling_params=tinker.types.SamplingParams(
                max_tokens=grader_max_tokens,
                temperature=temperature,
                stop=renderer.get_stop_sequences(),
            ),
        )
        result = future.result()
        response_text = tokenizer.decode(result.sequences[0].tokens)
    except Exception as e:
        logger.warning(f"Locus grading failed for {bottleneck_id}: {type(e).__name__}: {e}")
        return None

    loci = parse_locus(response_text, parent_plan)
    if loci is None:
        logger.info(f"Locus parse failed for {bottleneck_id} — falling back to legacy path")
    elif not loci:
        logger.info(f"Locus returned empty list for {bottleneck_id} (score=5?) — falling back")
        return None  # No weak spans → nothing to revise via locus
    else:
        logger.info(f"Locus grading for {bottleneck_id}: {len(loci)} valid entries")
    return loci


def build_whole_plan_revision_prompt(
    goal: str,
    plan_text: str,
    signal_vector: dict[str, int | None],
    critiques: dict[str, str],
) -> str:
    """CR-v7: whole-plan rewrite conditioned on all per-signal critiques.

    The policy sees the complete parent plan, each signal's current score,
    and each signal's prose critique. It emits a whole rewritten plan in
    <solution>...</solution>. No locus, no bottleneck targeting — the
    per-signal credit flows back through the LOSS (see
    `build_per_signal_context` + 8-datum construction in the revise loop).

    Minimal-mode wording per 2026-04-17 preference (no directive sentences
    beyond structural scaffolding).
    """
    if len(plan_text) > MAX_PLAN_CHARS_IN_PROMPT:
        plan_text = plan_text[:MAX_PLAN_CHARS_IN_PROMPT] + "\n\n[... rest truncated ...]"

    # Ordered feedback block, one line per signal, in SIGNALS order.
    feedback_lines = []
    for spec in GRADIENT_SIGNALS:
        score = signal_vector.get(spec.id)
        critique = (critiques or {}).get(spec.id, "") or ""
        score_str = f"{score}/5" if score is not None else "?/5"
        if critique:
            feedback_lines.append(f"- {spec.name} ({score_str}): {critique}")
        else:
            feedback_lines.append(f"- {spec.name} ({score_str}): [no critique]")
    feedback_block = "\n".join(feedback_lines)

    return (
        f"# Research Goal\n{goal}\n\n"
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
) -> str:
    """CR-v7: single-critique context used at LOSS time (never sampled).

    Produces the `context_i` under which we recompute `logπ(a | context_i)`
    for per-signal REINFORCE. Mirrors `build_whole_plan_revision_prompt`
    EXACTLY except the Per-signal feedback block contains ONLY critique_i
    (1 line instead of 8). This keeps the HER-style bias assumption clean:
    the only difference between sampling context and loss context is
    critique content, not scaffolding.
    """
    if len(plan_text) > MAX_PLAN_CHARS_IN_PROMPT:
        plan_text = plan_text[:MAX_PLAN_CHARS_IN_PROMPT] + "\n\n[... rest truncated ...]"
    critique = critique_i.strip() if critique_i else "[no critique]"
    score_str = f"{score}/5" if score is not None else "?/5"
    return (
        f"# Research Goal\n{goal}\n\n"
        f"# Current Plan\n{plan_text.strip()}\n\n"
        f"# Per-signal feedback\n- {signal_name} ({score_str}): {critique}\n\n"
        f"# Output Format\n"
        f"Rewrite the plan addressing the feedback above.\n\n"
        f"<think>\n"
        f"...your reasoning...\n"
        f"</think>\n"
        f"<solution>\n"
        f"...your rewritten plan...\n"
        f"</solution>"
    )


def build_locus_revision_prompt(
    goal: str,
    plan_text: str,
    signal_vector: dict[str, int | None],
    bottleneck_id: str,
    bottleneck_score: int,
    loci: list[LocusEntry],
) -> str:
    """CR-v6: Build a revision prompt using grader-emitted loci.

    Shows the full plan + signal scores + bottleneck rubric + identified weak
    spans. Asks the policy to emit `<revisions>` with JSON array of
    `{quote_original, quote_replacement}` pairs. Three repair modes
    (REWRITE/EXPAND/REMOVE) are expressed through prompt affordance, not schema.

    See plan file Change #5 for design rationale. Rejected alternatives:
    repair_mode enum, structured output tags.
    """
    spec = get_signal_spec(bottleneck_id)

    if len(plan_text) > MAX_PLAN_CHARS_IN_PROMPT:
        plan_text = plan_text[:MAX_PLAN_CHARS_IN_PROMPT] + "\n\n[... rest of plan truncated for context ...]"

    signal_lines = "\n".join(
        f"  {sid}: {score}/5" + (" ← WEAKEST" if sid == bottleneck_id else "")
        for sid, score in signal_vector.items()
        if score is not None
    )

    loci_lines = "\n".join(
        f"  {i+1}. Section: {loc.section_hint or '(unspecified)'}\n"
        f"     Quote: \"{loc.quote}\"\n"
        f"     Why weak: {loc.why}"
        for i, loc in enumerate(loci)
    )

    return textwrap.dedent(f"""
        You are improving a research plan by fixing specific weak spans.
        The plan has been evaluated and its weakest signal identified.
        A grader has located the specific parts of the plan that need improvement.

        # Research Goal
        {goal}

        # Current Plan
        {plan_text.strip()}

        # Signal Scores
        {signal_lines}

        # Weakest Signal: {spec.name} (score: {bottleneck_score}/5)
        ## What this signal measures: {spec.question}
        ## Scoring rubric: {spec.scoring_rubric}

        # Weak Spans Identified by Grader
        {loci_lines}

        # Your Task

        For each weak span above, choose the best repair:
        - **REWRITE**: Replace the span with a stronger version.
        - **EXPAND**: Keep the span but add content before/after it. Set
          quote_replacement to the original span text PLUS your additions.
        - **REMOVE**: Delete the span entirely. Set quote_replacement to "".

        Emit your revisions in <revisions> tags as a JSON array:

        <revisions>
        [
          {{"quote_original": "exact verbatim text from the plan", "quote_replacement": "your improved text"}}
        ]
        </revisions>

        RULES:
        - Each "quote_original" MUST be a VERBATIM copy of text from the Current
          Plan above. If you cannot find the exact text, skip that entry.
        - Preserve all plan content that is NOT in the weak spans.
        - Focus on improving {spec.name} from {bottleneck_score} to
          {bottleneck_score + 1} or higher.
        - Do NOT weaken parts of the plan that score well on other signals.
    """).strip()


def parse_locus_revisions(text: str) -> list[dict] | None:
    """Parse `<revisions>...</revisions>` JSON array from policy output.

    Returns list of {quote_original, quote_replacement} dicts, or None if
    parsing fails (no tag, malformed JSON, non-list).
    """
    match = re.search(r"<revisions>\s*(.*?)\s*</revisions>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    try:
        raw = json.loads(match.group(1).strip())
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(raw, list):
        return None
    return raw


_BULLET_RE = re.compile(r'^\s*(?:[-*]|\d+\.)\s')


def _is_block_marker_line(line: str) -> bool:
    """True if `line` starts with a markdown block marker: header, bullet,
    numbered list item, or bold-inline header (``**Section**:``)."""
    stripped = line.lstrip()
    if not stripped:
        return False
    if stripped.startswith('#'):
        return True
    if stripped.startswith('**'):
        return True
    if _BULLET_RE.match(line):
        return True
    return False


def _is_block_start(plan: str, i: int, quote: str) -> bool:
    """True if position `i` in `plan` begins a markdown block.

    A block start occurs at one of:
    - start-of-plan
    - immediately after a blank line (``plan[i-2:i] == '\\n\\n'``)
    - immediately after a header or bullet line
    - the quote itself starts with a block marker (e.g., its first line
      is a header or bullet)
    """
    if i == 0:
        return True
    if plan[i-1] != '\n':
        return False
    if i >= 2 and plan[i-2] == '\n':
        return True
    prev_line_start = plan.rfind('\n', 0, i-1) + 1
    prev_line = plan[prev_line_start:i-1]
    if _is_block_marker_line(prev_line):
        return True
    first_line = quote.split('\n', 1)[0]
    if _is_block_marker_line(first_line):
        return True
    return False


def _is_block_end(plan: str, end_pos: int) -> bool:
    """True if position `end_pos` (char after quote) ends a markdown block.

    A block end occurs at one of:
    - end-of-plan
    - quote ends with ``\\n`` AND followed by another ``\\n`` (blank line)
    - quote ends with ``\\n`` AND the following line is a block marker
    """
    if end_pos >= len(plan):
        return True
    if end_pos == 0 or plan[end_pos-1] != '\n':
        return False
    if plan[end_pos] == '\n':
        return True
    next_newline = plan.find('\n', end_pos)
    next_line = plan[end_pos:] if next_newline < 0 else plan[end_pos:next_newline]
    if _is_block_marker_line(next_line):
        return True
    return False


def _spans_complete_block(plan: str, quote: str) -> bool:
    """True if `quote` occupies one or more complete markdown blocks in `plan`.

    A "complete block" is bounded by markdown structure (header line, bullet
    line, blank line, or plan boundary). Substring quotes that fall inside a
    paragraph or cover only part of a multi-line block return False.

    Prevents the stitching bug where a substring replacement leaves the
    remainder of the original block intact, producing duplicated content.
    """
    if not quote:
        return False
    i = plan.find(quote)
    if i < 0:
        return False
    return _is_block_start(plan, i, quote) and _is_block_end(plan, i + len(quote))


def apply_locus_revisions(
    plan: str, revisions: list[dict],
) -> tuple[str, bool, str]:
    """CR-v6: Apply locus-based revisions via exact-string replacement.

    Returns (new_plan, success, status):
    - ("ok"): all revisions applied cleanly
    - ("no_match"): a quote_original was not found in plan → return original
    - ("ambiguous"): a quote_original appeared 2+ times → return original
    - ("partial_span"): a quote_original did not span complete markdown
      blocks (e.g., mid-paragraph substring, first 2 lines of a 5-line
      section) → return original. Prevents the stitching bug where a
      substring replace leaves the remainder of the original block
      intact, producing duplicated sections.
    - ("parse_error"): revisions are malformed → return original
    - ("empty_revisions"): nothing to apply → return original

    v1 strategy: reject on ANY ambiguity. Phase 1.5 pilot measures the
    ambiguous-reject rate; if >30%, upgrade to anchor-aware matching (v2).
    """
    if not revisions:
        return plan, False, "empty_revisions"

    new_plan = plan
    for rev in revisions:
        if not isinstance(rev, dict):
            return plan, False, "parse_error"
        orig = rev.get("quote_original", "")
        repl = rev.get("quote_replacement")
        if not isinstance(orig, str) or not orig:
            return plan, False, "parse_error"
        if repl is None:
            repl = ""
        if not isinstance(repl, str):
            return plan, False, "parse_error"
        n_matches = new_plan.count(orig)
        if n_matches == 0:
            return plan, False, "no_match"
        if n_matches > 1:
            return plan, False, "ambiguous"
        if not _spans_complete_block(new_plan, orig):
            return plan, False, "partial_span"
        new_plan = new_plan.replace(orig, repl, 1)
    return new_plan, True, "ok"


# =============================================================================
# Paragraph-level editing
# =============================================================================

# Map each signal to keywords likely found in section headers.
# The FIRST matching section in the plan is the one to edit.
SIGNAL_SECTION_KEYWORDS: dict[str, list[str]] = {
    "S1_depth":        ["Hypothesis", "Training Objective", "Methodology", "Core"],
    "S2_rigor":        ["Evaluation", "Baselines", "Ablation", "Metrics"],
    "S3_positioning":  ["Background", "Related Work", "Prior", "Problem Statement"],
    "S4_significance": ["Significance", "Expected Outcomes", "Expected Results", "Impact"],
    "S5_feasibility":  ["Methodology", "Algorithm", "Implementation", "Training"],
    "S6_risk_awareness": ["Limitation", "Risk", "Boundary", "Failure"],
    "S7_specificity":  ["Implementation", "Concrete", "Algorithm", "Training Loop"],
    "S8_scope":        ["Evaluation Domains", "Scope", "Generalization", "Domains"],
    "S9_focus":        [],  # whole-plan issue — falls back to full rewrite
}


def parse_plan_sections(plan_text: str) -> list[tuple[str, str]]:
    """Split a plan into (header, body) sections by markdown bold headers.

    Handles patterns like:
      **Training Objective**: ...
      **Background**\n...

    Returns [(header_str, section_body), ...]. The first entry may have
    header="" if the plan starts without a header.
    """
    # Find all **Header** patterns
    header_pattern = re.compile(r'\*\*([^*]+)\*\*[:\s]*')
    matches = list(header_pattern.finditer(plan_text))

    if not matches:
        return [("", plan_text)]

    sections = []
    # Text before the first header
    if matches[0].start() > 0:
        sections.append(("", plan_text[:matches[0].start()].strip()))

    for i, match in enumerate(matches):
        header = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(plan_text)
        body = plan_text[start:end].strip()
        sections.append((header, body))

    return sections


def find_section_for_signal(
    sections: list[tuple[str, str]],
    signal_id: str,
) -> int | None:
    """Find the index of the section most relevant to the given signal.

    Returns section index, or None if no match (falls back to full rewrite).
    """
    keywords = SIGNAL_SECTION_KEYWORDS.get(signal_id, [])
    if not keywords:
        return None

    for i, (header, _) in enumerate(sections):
        for kw in keywords:
            if kw.lower() in header.lower():
                return i
    return None


MAX_PLAN_CHARS_IN_PROMPT = 12000  # ~3000 tokens, keeps prompt under 32K context


def build_paragraph_edit_prompt(
    goal: str,
    plan_text: str,
    section_header: str,
    section_body: str,
    signal_vector: dict[str, int | None],
    bottleneck_id: str,
    bottleneck_score: int,
) -> str:
    """Build a prompt that asks the model to rewrite ONE section of the plan."""
    spec = get_signal_spec(bottleneck_id)

    # Truncate plan to fit context window (plans grow long without word limit)
    if len(plan_text) > MAX_PLAN_CHARS_IN_PROMPT:
        plan_text = plan_text[:MAX_PLAN_CHARS_IN_PROMPT] + "\n\n[... rest of plan truncated for context ...]"

    signal_lines = "\n".join(
        f"  {sid}: {score}/5" + (" ← WEAKEST" if sid == bottleneck_id else "")
        for sid, score in signal_vector.items()
        if score is not None
    )

    return textwrap.dedent(f"""
        You are improving ONE SECTION of a research plan. The plan has been
        evaluated and its weakest dimension has been identified. You must rewrite
        ONLY the specified section to improve that dimension.

        # Research Goal
        {goal}

        # Full Plan (for context — DO NOT rewrite the whole plan)
        {plan_text.strip()}

        # Signal Scores
        {signal_lines}

        # Weakest Signal: {spec.name} (score: {bottleneck_score}/5)
        ## What this signal measures: {spec.question}
        ## Scoring rubric: {spec.scoring_rubric}

        # Section to Rewrite: **{section_header}**
        Current content:
        {section_body}

        # Your Task
        1. In <critique> tags: explain WHY {spec.name} scored only {bottleneck_score}/5,
           specifically in relation to the **{section_header}** section.
        2. In <revised_section> tags: write an improved version of ONLY the
           **{section_header}** section. Keep the same header. Prioritize depth
           and quality. Do NOT write any other sections.

        <critique>
        ...why {spec.name} is low in this section...
        </critique>
        <revised_section>
        **{section_header}**: ...improved content for this section only...
        </revised_section>
    """).strip()


def detect_duplicated_content(plan_text: str, min_block_words: int = 20, dup_threshold: int = 2) -> bool:
    """Detect if plan contains duplicated paragraphs (stitch/revision artifact).

    Breaks text into paragraphs (double-newline separated), normalizes whitespace,
    and flags if any paragraph of >=min_block_words appears >=dup_threshold times.
    """
    # Normalize whitespace within each paragraph, then compare
    paragraphs = [re.sub(r'\s+', ' ', p).strip() for p in plan_text.split('\n\n')]
    paragraphs = [p for p in paragraphs if len(p.split()) >= min_block_words]
    if not paragraphs:
        return False
    from collections import Counter
    counts = Counter(paragraphs)
    for p, count in counts.items():
        if count >= dup_threshold:
            return True
    # Also check for near-duplicate section headers (e.g., same bold header 3+ times)
    headers = re.findall(r'\*\*([^*]+)\*\*', plan_text)
    # Normalize headers
    headers = [re.sub(r'\s+', ' ', h.strip().lower()) for h in headers]
    header_counts = Counter(headers)
    for h, count in header_counts.items():
        if count >= 5 and len(h) > 10:  # 5+ occurrences of substantial header
            return True
    return False


def stitch_revised_section(
    original_plan: str,
    sections: list[tuple[str, str]],
    section_idx: int,
    revised_section_text: str,
) -> str:
    """Replace one section in the plan with the revised version."""
    header_pattern = re.compile(r'\*\*([^*]+)\*\*[:\s]*')
    matches = list(header_pattern.finditer(original_plan))

    if not matches or section_idx < 0:
        return revised_section_text  # fallback: return revised as-is

    # Account for possible "" header at index 0
    has_preamble = matches[0].start() > 0
    match_idx = section_idx - (1 if has_preamble else 0)

    if match_idx < 0:
        # Replacing the preamble (before first header)
        end = matches[0].start()
        return revised_section_text.strip() + "\n\n" + original_plan[end:]

    if match_idx >= len(matches):
        logger.warning(f"stitch_revised_section: match_idx={match_idx} >= {len(matches)} matches, returning original unchanged")
        return original_plan  # safety: return unchanged

    start = matches[match_idx].start()
    end = matches[match_idx + 1].start() if match_idx + 1 < len(matches) else len(original_plan)

    return original_plan[:start] + revised_section_text.strip() + "\n\n" + original_plan[end:]


def parse_revised_section(text: str) -> tuple[str, str]:
    """Extract critique and revised section from paragraph-edit output."""
    critique = ""
    revised = ""
    cm = re.search(r"<critique>(.*?)</critique>", text, re.DOTALL)
    if cm:
        critique = cm.group(1).strip()
    rm = re.search(r"<revised_section>(.*?)</revised_section>", text, re.DOTALL)
    if rm:
        revised = rm.group(1).strip()
    if not revised:
        # Fallback: try <solution> tags
        sm = re.search(r"<solution>(.*?)</solution>", text, re.DOTALL)
        if sm:
            revised = sm.group(1).strip()
    return critique, revised


def build_critique_revise_prompt(
    goal: str,
    plan_text: str,
    signal_vector: dict[str, int | None],
    aggregate_reward: float,
    bottleneck_id: str,
    bottleneck_score: int,
) -> str:
    """Build a combined critique + revision prompt targeting the weakest signal."""
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
           {spec.name}. Prioritize depth and quality over brevity.

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
        ...your revised plan (prioritize depth and quality)...
        </solution>
    """).strip()


def parse_critique_and_plan(text: str) -> tuple[str, str]:
    """Extract critique and revised plan from the combined output."""
    import re

    critique = ""
    plan = ""

    critique_match = re.search(r"<critique>(.*?)</critique>", text, re.DOTALL)
    if critique_match:
        critique = critique_match.group(1).strip()

    solution_match = re.search(r"<solution>(.*?)</solution>", text, re.DOTALL)
    if solution_match:
        plan = solution_match.group(1).strip()

    # If no <solution> tags, try to use everything after </critique>
    if not plan and critique_match:
        remainder = text[critique_match.end():]
        plan = remainder.strip()

    return critique, plan


# =============================================================================
# Extended buffer entry
# =============================================================================


@dataclass
class CritiqueReviseEntry(BufferEntry):
    """Extends BufferEntry with critique-revise metadata."""
    entry_type: str = "fresh"            # "fresh" or "revision"
    parent_idx: int | None = None        # buffer index of the plan being revised
    targeted_signal: str | None = None   # which signal was targeted
    critique_text: str | None = None     # the critique reasoning
    delta_reward: float | None = None    # revised_aggregate - original_aggregate


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

    # --- Apply signal ablations (disabled_signals) ---
    # Must update both the module-level lists AND the local import aliases
    global GRADIENT_SIGNALS, GRADIENT_WEIGHTS
    if config.disabled_signals:
        disabled = {s.strip() for s in config.disabled_signals.split(",") if s.strip()}
        # Filter SIGNALS and reweight
        from co_scientist.shared import grant_signal_reward as _tsr
        active_signals = [s for s in _tsr.SIGNALS if s.id not in disabled]
        active_weights = {s.id: _tsr.SIGNAL_WEIGHTS[s.id] for s in active_signals}
        # Renormalize weights to sum to 1.0
        total_w = sum(active_weights.values())
        if total_w > 0:
            active_weights = {k: v / total_w for k, v in active_weights.items()}
        # Monkey-patch the module AND local aliases
        _tsr.SIGNALS = active_signals
        _tsr.SIGNAL_WEIGHTS = active_weights
        GRADIENT_SIGNALS = active_signals
        GRADIENT_WEIGHTS = active_weights
        logger.info(f"Signal ablation: disabled {disabled}, {len(active_signals)} active signals")
        logger.info(f"Reweighted: {active_weights}")
    else:
        disabled = set()

    logger.info(f"TARGET GOAL ({len(TARGET_GOAL)} chars)")
    logger.info(
        f"Training plan: n_iter={config.n_iterations}, "
        f"n_fresh={config.n_fresh}, n_revise={config.n_revise}, "
        f"N_repeat={config.grader_repeats}"
    )
    logger.info(
        f"Paradigm: Signal-Targeted Critique-Revise Loop | "
        f"Model: {config.model_name} | Grader: {config.grader_model_name} | "
        f"UCB: {config.use_ucb} | HardGates: {not config.skip_hard_gates} | "
        f"TrainFresh: {config.train_on_fresh} | BoN: {config.n_revision_candidates} | "
        f"Locus: {config.locus_based_edit}"
    )

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

        # =================================================================
        # PHASE 1: Generate n_fresh plans (exploration)
        # =================================================================
        if iter_idx < config.cold_start_iters:
            context = []
        else:
            # Filter to BufferEntry-compatible fields (buffer dicts have extra
            # critique-revise fields that BufferEntry doesn't accept)
            import dataclasses as _dc
            _be_fields = {f.name for f in _dc.fields(BufferEntry)}
            valid_buf = [
                BufferEntry(**{k: v for k, v in e.items() if k in _be_fields})
                for e in buffer if e.get("hard_gate_passed")
            ]
            context = select_context(valid_buf, config.K_exploit, config.K_explore, rng)

        fresh_samples = []
        fresh_prompt_tokens = []  # per-plan prompt tokens (needed when diversity_seeding)

        if config.diversity_seeding:
            # A9: Generate each plan with a different approach seed
            seeds = rng.sample(APPROACH_SEEDS, min(config.n_fresh, len(APPROACH_SEEDS)))
            existing_summaries: list[str] = []
            for seed_idx, seed in enumerate(seeds):
                neg_cond = existing_summaries[-3:] if existing_summaries else None
                prompt_text = build_diverse_fresh_prompt(
                    goal=TARGET_GOAL, approach_seed=seed,
                    context=context, existing_summaries=neg_cond,
                )
                model_input = renderer.build_generation_prompt(
                    [{"role": "user", "content": prompt_text}]
                )
                future = sampling_client.sample(
                    prompt=model_input, num_samples=1,
                    sampling_params=sampling_params,
                )
                result = future.result()
                seq = result.sequences[0]
                text = renderer.parse_response(seq.tokens)[0]["content"]
                if "<solution>" in text and "</solution>" not in text:
                    text = text.rstrip() + "\n</solution>"
                fresh_samples.append({"tokens": seq.tokens, "logprobs": seq.logprobs, "text": text})
                fresh_prompt_tokens.append(model_input.to_ints())
                # Add summary for negative conditioning of next plan
                existing_summaries.append(_one_line_summary(text))
                logger.info(f"  Fresh {seed_idx} [{seed.split('(')[0].strip()[:25]}]: {len(text.split())} words")
        else:
            prompt_text = build_research_plan_prompt(goal=TARGET_GOAL, context=context)
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
                text = renderer.parse_response(seq.tokens)[0]["content"]
                if "<solution>" in text and "</solution>" not in text:
                    text = text.rstrip() + "\n</solution>"
                fresh_samples.append({"tokens": seq.tokens, "logprobs": seq.logprobs, "text": text})
                fresh_prompt_tokens.append(shared_prompt_ints)

        logger.info(f"Iter {iter_idx}: generated {len(fresh_samples)} fresh plans")

        # Launch grading for fresh plans
        fresh_futures: list[PlanFutures | None] = []
        for sample in fresh_samples:
            if len(sample["text"].strip().split()) < config.min_words:
                fresh_futures.append(None)
                continue
            fresh_futures.append(
                launch_plan_reward(
                    plan=sample["text"], goal=TARGET_GOAL,
                    grader_client=grader_client, renderer=renderer,
                    n_repeats=config.grader_repeats,
                    grader_max_tokens=config.grader_max_tokens,
                    hg_max_tokens=config.grader_hard_gate_max_tokens,
                    temperature=config.grader_temperature,
                    emit_critique=config.emit_signal_critique,
                )
            )

        # =================================================================
        # PHASE 2: Critique-Revise n_revise plans from buffer (exploitation)
        # =================================================================
        revise_samples = []  # list of (parent_entry, critique, revised_text, tokens, logprobs)
        revise_futures: list[PlanFutures | None] = []

        if iter_idx >= config.cold_start_iters and len(buffer) >= config.n_revise:
            # --- P3.2: Compute signals to skip (historically low revision success) ---
            signal_rates, signal_attempts = compute_signal_revision_stats(buffer)
            skip_signals: set[str] = set()
            # CR-v6: merge config-based static skip (e.g., S3 grader prior) with dynamic skip
            if config.revision_skip_signals:
                skip_signals.update(s.strip() for s in config.revision_skip_signals.split(",") if s.strip())
            for sig, rate in signal_rates.items():
                if (signal_attempts.get(sig, 0) >= config.min_revision_attempts
                        and rate < config.min_revision_success_rate):
                    skip_signals.add(sig)
            if skip_signals:
                logger.info(f"Iter {iter_idx}: skipping signals: {skip_signals} "
                            f"(config: {config.revision_skip_signals}, rates: {signal_rates})")

            # Pick plans from buffer for revision (UCB or top-K)
            if config.use_ucb:
                total_selections = sum(e.get("n_selected", 0) for e in buffer)
                candidates = ucb_select(buffer, config.n_revise, total_selections, config.ucb_c)
                # Update selection counts
                for buf_idx, _ in candidates:
                    buffer[buf_idx]["n_selected"] = buffer[buf_idx].get("n_selected", 0) + 1
            else:
                valid_entries = [(i, e) for i, e in enumerate(buffer) if e.get("hard_gate_passed")]
                valid_entries.sort(key=lambda x: x[1]["aggregate_reward"], reverse=True)
                candidates = valid_entries[:config.n_revise]

            # CR-v7: whole-plan per-signal path (skips locus/paragraph/
            # whole-plan-bottleneck dispatch entirely). All per-signal
            # critiques from the parent grading are handed to the policy
            # in a single sampling context; per-signal credit assignment
            # is recovered at loss construction time.
            #
            # Two-pass structure for parallelism: Pass 1 fires all N parent
            # sampling futures concurrently; Pass 2 awaits each and
            # processes the resulting candidates. Previously serial, so N
            # full rollouts stacked → 60-120s/iter saved.
            if config.revision_mode == "whole_plan_per_signal":
                v7_pending: list[tuple[int, dict, dict, list[int], "tinker.Future"]] = []
                for buf_idx, parent in candidates:
                    parent_critiques = parent.get("per_signal_critiques", {}) or {}
                    revise_prompt = build_whole_plan_revision_prompt(
                        goal=TARGET_GOAL,
                        plan_text=parent["plan_text"],
                        signal_vector=parent["signal_vector"],
                        critiques=parent_critiques,
                    )
                    revise_input = renderer.build_generation_prompt(
                        [{"role": "user", "content": revise_prompt}]
                    )
                    revise_tokens_ints = revise_input.to_ints()
                    future = sampling_client.sample(
                        prompt=revise_input,
                        num_samples=config.n_revision_candidates,
                        sampling_params=sampling_params,
                    )
                    v7_pending.append((buf_idx, parent, parent_critiques, revise_tokens_ints, future))

                for buf_idx, parent, parent_critiques, revise_tokens_ints, future in v7_pending:
                    revise_result = future.result()
                    for cand_idx, seq in enumerate(revise_result.sequences):
                        raw_text = renderer.parse_response(seq.tokens)[0]["content"]
                        critique_str, revised_plan = parse_critique_and_plan(raw_text)
                        if not revised_plan:
                            revised_plan = raw_text
                        if detect_duplicated_content(revised_plan):
                            logger.warning(
                                f"Iter {iter_idx} cand {cand_idx}: duplicated "
                                "content in whole_plan_per_signal output, "
                                "reverting to parent"
                            )
                            revised_plan = parent["plan_text"]
                            critique_str = (critique_str or "") + " [REJECTED: dup]"
                        if "<solution>" in revised_plan and "</solution>" not in revised_plan:
                            revised_plan = revised_plan.rstrip() + "\n</solution>"
                        plan_for_grading = revised_plan
                        if "<solution>" not in plan_for_grading:
                            plan_for_grading = f"<solution>\n{plan_for_grading}\n</solution>"

                        revise_samples.append({
                            "parent_idx": buf_idx,
                            "parent_reward": parent["aggregate_reward"],
                            "parent_plan_text": parent["plan_text"],
                            "parent_signal_vector": dict(parent["signal_vector"]),
                            "parent_critiques": dict(parent_critiques),
                            "bottleneck_id": None,
                            "bottleneck_score": None,
                            "critique": critique_str or "",
                            "text": plan_for_grading,
                            "tokens": seq.tokens,
                            "logprobs": seq.logprobs,
                            "revise_tokens_ints": revise_tokens_ints,
                            "candidate_idx": cand_idx,
                            "revision_mode_used": "whole_plan_per_signal",
                        })
                        if len(plan_for_grading.strip().split()) >= config.min_words:
                            revise_futures.append(
                                launch_plan_reward(
                                    plan=plan_for_grading, goal=TARGET_GOAL,
                                    grader_client=grader_client, renderer=renderer,
                                    n_repeats=config.grader_repeats,
                                    grader_max_tokens=config.grader_max_tokens,
                                    hg_max_tokens=config.grader_hard_gate_max_tokens,
                                    temperature=config.grader_temperature,
                                    emit_critique=config.emit_signal_critique,
                                )
                            )
                        else:
                            revise_futures.append(None)
                # Skip the legacy per-parent loop below when in CR-v7 mode.
                candidates = []

            for buf_idx, parent in candidates:
                # --- Legacy paths (revision_mode in {locus, paragraph, whole}) ---
                # P3.2: identify bottleneck with skip
                bottleneck_id, bottleneck_score = identify_bottleneck(
                    parent["signal_vector"], skip_signals=skip_signals,
                )

                # --- Determine revision mode: locus > paragraph > whole_plan ---
                revision_mode = "whole_plan"  # default fallback
                locus_entries = None
                target_section_idx = None
                plan_sections = None

                if config.locus_based_edit:
                    # CR-v6 Pass 2: re-grade bottleneck with locus directive
                    locus_entries = grade_bottleneck_with_locus(
                        parent_plan=parent["plan_text"],
                        bottleneck_id=bottleneck_id,
                        goal=TARGET_GOAL,
                        grader_client=grader_client,
                        renderer=renderer,
                        tokenizer=tokenizer,
                        grader_max_tokens=config.locus_grader_max_tokens,
                        temperature=config.grader_temperature,
                    )
                    if locus_entries:
                        revision_mode = "locus"
                        revise_prompt = build_locus_revision_prompt(
                            goal=TARGET_GOAL,
                            plan_text=parent["plan_text"],
                            signal_vector=parent["signal_vector"],
                            bottleneck_id=bottleneck_id,
                            bottleneck_score=bottleneck_score,
                            loci=locus_entries,
                        )
                    # else: locus failed → fall through to paragraph/whole-plan

                if revision_mode != "locus" and config.paragraph_level_edit:
                    plan_sections = parse_plan_sections(parent["plan_text"])
                    target_section_idx = find_section_for_signal(plan_sections, bottleneck_id)
                    if target_section_idx is not None and target_section_idx < len(plan_sections):
                        revision_mode = "paragraph"
                        sec_header, sec_body = plan_sections[target_section_idx]
                        revise_prompt = build_paragraph_edit_prompt(
                            goal=TARGET_GOAL,
                            plan_text=parent["plan_text"],
                            section_header=sec_header or "Introduction",
                            section_body=sec_body,
                            signal_vector=parent["signal_vector"],
                            bottleneck_id=bottleneck_id,
                            bottleneck_score=bottleneck_score,
                        )

                if revision_mode == "whole_plan":
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

                # P3.1: Generate N candidates per revision (Best-of-N)
                revise_future = sampling_client.sample(
                    prompt=revise_input,
                    num_samples=config.n_revision_candidates,
                    sampling_params=sampling_params,
                )
                revise_result = revise_future.result()

                # Process each candidate
                for cand_idx, seq in enumerate(revise_result.sequences):
                    raw_text = renderer.parse_response(seq.tokens)[0]["content"]

                    if revision_mode == "locus":
                        # CR-v6: parse revision JSON, apply exact-string replacements
                        revisions_parsed = parse_locus_revisions(raw_text)
                        if revisions_parsed:
                            revised_plan, success, locus_status = apply_locus_revisions(
                                parent["plan_text"], revisions_parsed,
                            )
                            critique = f"[locus:{locus_status}]"
                            if not success:
                                revised_plan = parent["plan_text"]
                                critique += " [REJECTED]"
                        else:
                            revised_plan = parent["plan_text"]
                            critique = "[locus:no_revisions_block]"
                    elif revision_mode == "paragraph" and target_section_idx is not None:
                        # CR-v5: extract revised section, stitch back
                        critique, revised_section = parse_revised_section(raw_text)
                        if revised_section:
                            revised_plan = stitch_revised_section(
                                parent["plan_text"], plan_sections,
                                target_section_idx, revised_section,
                            )
                        else:
                            revised_plan = raw_text
                    else:
                        # Whole-plan: extract full revised plan
                        critique, revised_plan = parse_critique_and_plan(raw_text)
                        if not revised_plan:
                            revised_plan = raw_text

                    # Reject revisions with duplicated content (stitch/model artifact)
                    if detect_duplicated_content(revised_plan):
                        logger.warning(
                            f"Iter {iter_idx} revision cand {cand_idx}: detected duplicated content, "
                            f"reverting to parent plan (revision treated as no-op)"
                        )
                        revised_plan = parent["plan_text"]
                        critique = (critique or "") + " [REJECTED: duplicated content]"

                    if "<solution>" in revised_plan and "</solution>" not in revised_plan:
                        revised_plan = revised_plan.rstrip() + "\n</solution>"

                    plan_for_grading = revised_plan
                    if "<solution>" not in plan_for_grading:
                        plan_for_grading = f"<solution>\n{plan_for_grading}\n</solution>"

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
                        "candidate_idx": cand_idx,
                    })

                    # Launch grading for each candidate
                    if len(plan_for_grading.strip().split()) >= config.min_words:
                        revise_futures.append(
                            launch_plan_reward(
                                plan=plan_for_grading, goal=TARGET_GOAL,
                                grader_client=grader_client, renderer=renderer,
                                n_repeats=config.grader_repeats,
                                grader_max_tokens=config.grader_max_tokens,
                                hg_max_tokens=config.grader_hard_gate_max_tokens,
                                temperature=config.grader_temperature,
                                emit_critique=config.emit_signal_critique,
                            )
                        )
                    else:
                        revise_futures.append(None)

            n_candidates_total = len(revise_samples)
            n_parents = len(candidates)
            logger.info(
                f"Iter {iter_idx}: generated {n_candidates_total} revision candidates "
                f"for {n_parents} parents (best-of-{config.n_revision_candidates}). "
                f"Targets: {list(dict.fromkeys(s['bottleneck_id'] for s in revise_samples))}"
            )

        # =================================================================
        # Collect all grading results
        # =================================================================
        logger.info(f"Iter {iter_idx}: collecting grading results...")
        t_collect = time.time()

        # Collect fresh plan rewards
        fresh_rewards: list[TenSignalReward | None] = []
        for fut in fresh_futures:
            if fut is None:
                fresh_rewards.append(None)
                continue
            try:
                fresh_rewards.append(collect_plan_reward(fut, tokenizer, skip_hard_gates=config.skip_hard_gates))
            except Exception as e:
                logger.error(f"Fresh plan grading failed: {type(e).__name__}")
                fresh_rewards.append(None)

        # Collect revision rewards
        revise_rewards: list[TenSignalReward | None] = []
        for fut in revise_futures:
            if fut is None:
                revise_rewards.append(None)
                continue
            try:
                revise_rewards.append(collect_plan_reward(fut, tokenizer, skip_hard_gates=config.skip_hard_gates))
            except Exception as e:
                logger.error(f"Revision grading failed: {type(e).__name__}")
                revise_rewards.append(None)

        logger.info(f"Iter {iter_idx}: collection complete in {time.time() - t_collect:.0f}s")

        # =================================================================
        # Add to buffer
        # =================================================================
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
                # CR-v7: per-signal prose critique emitted by grader.
                # Empty dict for pre-v7 runs (emit_signal_critique=False).
                "per_signal_critiques": dict(reward.per_signal_critiques),
                "word_count": len(sample["text"].strip().split()),
                "entry_type": "fresh",
                "parent_idx": None,
                "targeted_signal": None,
                "critique_text": None,
                "delta_reward": None,
            }
            buffer.append(entry)
            with open(buffer_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # --- P3.1: Best-of-N selection per parent ---
        # Group candidates by parent, pick the one with highest delta.
        # Only add the BEST candidate per parent to the buffer.
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
            # Pick candidate with highest delta
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
                # CR-v7: per-signal critique from the REVISED-plan grading
                # (so future revisions of this plan get fresh critiques).
                "per_signal_critiques": dict(rreward.per_signal_critiques),
                "word_count": len(rsample["text"].strip().split()),
                "entry_type": "revision",
                "parent_idx": rsample["parent_idx"],
                "targeted_signal": rsample.get("bottleneck_id"),
                "critique_text": rsample.get("critique"),
                "delta_reward": delta,
                "n_candidates": len(cands),
                "best_of_n_deltas": sorted([d for _, _, d in cands], reverse=True),
            }
            buffer.append(entry)
            with open(buffer_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        n_total_candidates = len(revise_samples)
        n_best_selected = len(best_revisions)

        # =================================================================
        # PHASE 3: RL Update
        # =================================================================
        # STABILITY FIX (v2): Only train on positive-delta revisions.
        # Fresh plans populate the buffer for exploration but do NOT contribute
        # to the RL gradient. This prevents the β=20 cap problem (4 fresh
        # samples → over-concentrated entropic weights → policy degradation).
        #
        # Previous version trained on both fresh (entropic weights) and
        # revisions (delta weights). This caused mean reward to collapse from
        # 0.405 → 0.141 over 10 iters while buffer_max still improved.
        training_datums = []
        beta = 0.0

        # Compute valid_fresh for logging (and optionally for training if train_on_fresh)
        # Keep original index for prompt token lookup
        valid_fresh = [
            (s, r, i) for i, (s, r) in enumerate(zip(fresh_samples, fresh_rewards)) if r is not None
        ]

        # Ablation: train_on_fresh — re-enable entropic training on fresh plans (for B3/A8)
        if config.train_on_fresh and len(valid_fresh) >= 2:
            from co_scientist.grant_proposal.train_buffer_ttt import entropic_weights
            fresh_rewards_arr = np.array([r.aggregate_reward for _, r, _ in valid_fresh])
            fresh_weights, beta = entropic_weights(
                fresh_rewards_arr, config.kl_budget, config.beta_max
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
                                [0.0] * ob_len + sample["logprobs"], dtype=torch.float
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

        # Compute advantages for ALL revision candidates.
        # Three modes:
        #
        # (C3) locus_based_edit=True OR paragraph_level_edit=True:
        #   CR-v6 mixed advantage:
        #     advantage = α·normalize(Δ_target) + β·Δ_aggregate
        #   Δ_target = per-signal integer delta on bottleneck, / 4.0 → [-1, 1].
        #   Δ_aggregate = aggregate reward delta, already in [-1, 1].
        #   Reframes the "sparse signal" problem: aggregate delta is noisy
        #   across all signals; targeted delta isolates the revision's effect.
        #
        # (GRPO) neither flag set (whole-plan rewrites):
        #   Group-relative centering: advantage_i = (delta_i - mean) / (std + eps)
        #   Handles noisy whole-plan deltas where collateral damage dominates.
        all_valid_revisions = [
            (rs, rr, rr.aggregate_reward - rs["parent_reward"])
            for rs, rr in zip(revise_samples, revise_rewards)
            if rr is not None
        ]
        n_positive_revisions = sum(1 for _, _, d in all_valid_revisions if d > 0)

        # CR-v7 branch: per-signal REINFORCE on all signals via
        # recomputed logprob under context_i (HER-style relabelling).
        #
        # Bias note (Option c): logprobs stored = recomputed under
        # context_i, not the sampling-context. IS ratio = exp(new - stored)
        # = 1 → importance_sampling loss degenerates to
        # L_i = -A_i · logπ(a | context_i) — a biased policy gradient for
        # the context_i distribution (action a was sampled from the full
        # all-hints context). Bias is intentional: we WANT the
        # single-critique gradient direction driven by the outcome of an
        # all-hints rewrite. Without per-context recomputation the loss
        # collapses to aggregate — 8 Δ scalars on one gradient direction.
        # Deferred alternative: cache logπ_old per context_i and do
        # multi-epoch PPO (v7.1).
        if config.revision_mode == "whole_plan_per_signal" and all_valid_revisions:
            # Build flat work list across ALL revisions first, then issue
            # one asyncio.gather for maximum concurrency (previously we
            # ran one event loop per revision).
            per_datum_work: list[tuple[str, float, list[int], list[int]]] = []
            for rsample, rreward, _ in all_valid_revisions:
                if not rreward.hard_gate_passed:
                    continue
                parent_sv = rsample["parent_signal_vector"]
                parent_critiques = rsample["parent_critiques"]
                # Truncate once per revision; build_per_signal_context
                # would otherwise run the same check 8× on the same plan.
                parent_plan = rsample["parent_plan_text"]
                if len(parent_plan) > MAX_PLAN_CHARS_IN_PROMPT:
                    parent_plan = parent_plan[:MAX_PLAN_CHARS_IN_PROMPT] + "\n\n[... rest truncated ...]"
                gen_tokens = list(rsample["tokens"])
                if not gen_tokens:
                    continue
                for spec in GRADIENT_SIGNALS:
                    sid = spec.id
                    s_parent = parent_sv.get(sid)
                    s_rev = rreward.signal_vector.get(sid)
                    if s_parent is None or s_rev is None:
                        continue
                    delta_i = (s_rev - s_parent) / 4.0
                    if abs(delta_i) < config.delta_threshold:
                        continue
                    ctx_text = build_per_signal_context(
                        plan_text=parent_plan,
                        critique_i=parent_critiques.get(sid, ""),
                        signal_name=spec.name,
                        score=s_parent,
                        goal=TARGET_GOAL,
                    )
                    ctx_tokens = renderer.build_generation_prompt(
                        [{"role": "user", "content": ctx_text}]
                    ).to_ints()
                    if len(ctx_tokens) < 1:
                        # Defensive: an empty context would produce a
                        # negative prefix length → silent tensor mis-alignment.
                        continue
                    per_datum_work.append((sid, delta_i, ctx_tokens, gen_tokens))

            if per_datum_work:
                # Issue one gather over all contexts × revisions at once.
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
                    # Outer event loop running; fall back to sync.
                    logprobs_all = [
                        sampling_client.compute_logprobs(mi) for mi in model_inputs
                    ]

                for (sid, delta_i, ctx_tokens, gen_tokens), full_logprobs in zip(
                    per_datum_work, logprobs_all,
                ):
                    # compute_logprobs returns one logprob per input token
                    # (same length as input). Slice to the generation
                    # portion: positions [len(ctx) .. len(ctx)+len(gen)-1]
                    # each hold logπ(gen_tokens[k] | ctx + gen_tokens[:k]).
                    # Matches SDPO convention at train_sdpo.py:3460-3463.
                    prompt_len = len(ctx_tokens)
                    gen_logprobs = full_logprobs[prompt_len : prompt_len + len(gen_tokens)]
                    if len(gen_logprobs) != len(gen_tokens):
                        logger.warning(
                            f"CR-v7: logprob length mismatch for {sid}: "
                            f"got {len(gen_logprobs)}, expected {len(gen_tokens)}; skipping."
                        )
                        continue
                    # Length-normalized clamp: skip datum when the mean
                    # per-token logprob is catastrophically low. mean < -10
                    # ≡ per-token prob < 4.5e-5 on average, a strong signal
                    # that context_i is off-distribution.
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
        elif len(all_valid_revisions) >= 1:
            deltas = np.array([d for _, _, d in all_valid_revisions])

            if config.locus_based_edit or config.paragraph_level_edit:
                # C3 mixed advantage (targeted revision modes)
                target_deltas = np.array([
                    (
                        (rr.signal_vector.get(rs["bottleneck_id"]) or rs["bottleneck_score"])
                        - rs["bottleneck_score"]
                    ) / 4.0
                    for rs, rr, _ in all_valid_revisions
                ])
                advantages = (
                    config.c3_alpha * target_deltas
                    + config.c3_beta * deltas
                ) * config.delta_scale
            elif len(all_valid_revisions) >= 2:
                # GRPO centered (whole-plan mode)
                mean_d = deltas.mean()
                std_d = deltas.std() + 1e-6
                advantages = (deltas - mean_d) / std_d * config.delta_scale
            else:
                advantages = np.zeros_like(deltas)

            for (rsample, rreward, delta), advantage in zip(all_valid_revisions, advantages):
                if abs(advantage) < 1e-6:
                    continue

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
                                [0.0] * ob_len + [float(advantage)] * len(rsample["logprobs"]),
                                dtype=torch.float,
                            )
                        ),
                    },
                )
                training_datums.append(datum)

        # Apply update
        if training_datums and not config.skip_rl_update:
            try:
                # importance_sampling = weighted log-likelihood (REINFORCE).
                # Cleaner than PPO for on-policy data where ratio ≈ 1 and
                # clipping never binds.
                # CR-v6: advantage = C3 mixed (α·Δ_target + β·Δ_aggregate) for
                # targeted modes; GRPO centered for whole-plan. Both signs used.
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

        iter_summary = {
            "iter": iter_idx,
            "time/total": time.time() - t_start,
            "fresh/count": len(valid_fresh),
            "fresh/reward_mean": float(np.mean([r.aggregate_reward for _, r, _ in valid_fresh])) if valid_fresh else 0,
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
            **per_signal_means,
        }

        with open(iter_summary_path, "a") as f:
            f.write(json.dumps(iter_summary) + "\n")

        # Per-sample logs
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
                    "targeted_signal": rsample["bottleneck_id"],
                    "critique": rsample["critique"],
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
