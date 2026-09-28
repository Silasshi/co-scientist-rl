"""D5 κ Phase 3 prompt builders — distillation pipeline (走线 A, Path Y).

Pure helper module. No I/O, no Tinker, no Anthropic.

Phase 3 architecture (per RETRIEVAL_DESIGN_v1.md):
  Per goal G, 3 INDEPENDENT distillation rounds (round j does NOT see A_1..A_{j-1}):
    Round j:
      Student (no critique): G + oracle_batch_j → A_j
      Teacher (with critique_{j-1}): G + oracle_batch_j + critique → A_j (teacher samples)
    Critic (Opus subagent + privileged source paper): A_j → critique_j
  After 3 rounds:
    Plan: G + [A_1, A_2, A_3] → research plan (final <solution>)
    Plan critic: NONE (Phase 2 didn't have plan critic in distillation pathway either)

This module provides:
  - build_distillation_student_prompt: STUDENT distillation context (no critique)
  - build_distillation_teacher_prompt: TEACHER distillation context (with prev critique)
  - build_kappa_plan_prompt: final plan-generation prompt from concat of 3 A_i
  - build_distillation_critic_request_payload: Opus subagent request per A_i
  - DISTILLATION_COLD_START_CRITIQUE: iter-0 / round-1 placeholder critique
  - parse_distillation_critique_xml: extract <distill_critique> from raw Opus
  - re-exports: extract_solution, build_sdpo_datum, COLD_START_CRITIQUE_NONE

Critic shape is different from Phase 2 μ-v4 plan-level critique:
  - <missing_critical>: oracle items the source paper says are core that A_j missed
  - <noise>: items A_j included that source paper deems peripheral
  - <faithfulness>: items A_j paraphrased incorrectly vs original oracle item
  - <improvement_directive>: free-text guidance for next round
"""
from __future__ import annotations

import re
from typing import Any

# Re-export shared helpers from v1 (Phase 2 infrastructure unchanged)
from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v1 import (  # noqa: F401
    COLD_START_CRITIQUE_NONE,
    build_sdpo_datum,
    extract_solution,
)


def _strip_think(text: str) -> str:
    """Strip Qwen3 `<think>...</think>` thinking preamble.

    Distillation outputs frequently include thinking preamble even when wrapped
    in <solution>. Worse, raw thinking tokens like </think> in distillation
    text downstream-poison plan-generation prompts (Qwen3 sees </think> in
    user message and triggers immediate `<|im_end|>` EOS, producing a
    17-char `<think><|im_end|>` plan output). Always strip thinking before
    feeding A_i into plan_generation prompt.

    Implementation mirrors `train_buffer_ttt._strip_think`. Returns text
    after the LAST `</think>` (handles cases where model nests or repeats
    thinking blocks).
    """
    idx = text.rfind("</think>")
    if idx != -1:
        return text[idx + len("</think>"):].strip()
    return text


def extract_distillation(raw_text: str) -> str:
    """Robust distillation-output extractor.

    Layered cleanup:
      1. Strip Qwen3 thinking preamble (everything up to and including the
         LAST </think>) — prevents downstream prompt poisoning
      2. Apply extract_solution() to find <solution>...</solution> body if
         present, else first markdown heading

    Use this for ALL distillation A_i extraction in Phase 3 (τ + κ). Do NOT
    use the bare extract_solution() — it lacks the thinking-strip step and
    leaks </think> tokens into plan_generation prompts.
    """
    return extract_solution(_strip_think(raw_text))


# =============================================================================
# Distillation prompts (per A_j, 3 rounds independent)
# =============================================================================

_DISTILL_FOOTER = (
    "\n\nWrite your distilled item list inside <solution>...</solution> tags."
    " Target 800-1200 tokens. Use the SAME 6 categories as the input oracle"
    " (Insights / Methodology / Theory / Math / Empirical / Failure_modes)."
    " For each item kept: write 1-3 sentences (longer if the item contains"
    " concrete equations or numerical content). **PRESERVE equations,"
    " hyperparameter values, and named benchmark numbers VERBATIM in"
    " backticks** (e.g. `β=8`, `LoRA rank 64`, `AIME pass@1 = 53.4%`,"
    " `J = log E[exp(β·R)]/β`); only rewrite surrounding prose. Aim for"
    " 40-50 distilled items across all categories combined (this batch);"
    " **PRIORITIZE items containing equations or numerical specifics over"
    " items containing only abstract principles**. Drop oracle items that"
    " are clearly orthogonal to the research goal."
    "\n\n# Example of GOOD vs BAD distillation"
    "\n## GOOD (preserves substance):"
    "\n- **RS-GRPO** (Math): `J_RS = log E[exp(β·R)] / β` with `β=8`"
    " stabilizes test-time policy updates by emphasizing high-reward rollouts"
    " (Hubotter et al. 2025). Applies to TTT-Discover's per-problem RL loop."
    "\n## BAD (substance lost):"
    "\n- Use risk-sensitive RL methods to focus on high-reward solutions."
)


def build_distillation_student_prompt(
    goal: str, oracle_batch_text: str, round_idx: int
) -> str:
    """STUDENT distillation context: goal + this round's oracle batch, NO critique.

    Used as the SDPO `student` (without-review) channel for the distillation
    step at round `round_idx` (1-indexed).
    """
    return (
        "You are extracting goal-relevant methodological abstractions from a"
        " curated oracle of insights drawn from prior work. This is round"
        f" {round_idx} of 3 independent distillation rounds (rounds do not"
        " share state — focus only on this batch)."
        f"\n\n# Research goal\n{goal}"
        f"\n\n# Oracle batch (round {round_idx})\n\n{oracle_batch_text}"
        + _DISTILL_FOOTER
    )


def build_distillation_teacher_prompt(
    goal: str, oracle_batch_text: str, critique_xml: str, round_idx: int
) -> str:
    """TEACHER distillation context: goal + this round's batch + prev critique.

    Used as the SDPO `teacher` (with-review) channel. The critique XML wraps
    structured fields (missing_critical / noise / faithfulness /
    improvement_directive). Round 1 receives DISTILLATION_COLD_START_CRITIQUE.
    """
    return (
        "You are extracting goal-relevant methodological abstractions from a"
        " curated oracle of insights drawn from prior work. This is round"
        f" {round_idx} of 3 independent distillation rounds. A reviewer with"
        " privileged access to the source paper has critiqued a prior"
        " distillation attempt; address the improvement_directive while"
        " selecting and condensing items from this batch."
        f"\n\n# Research goal\n{goal}"
        f"\n\n# Oracle batch (round {round_idx})\n\n{oracle_batch_text}"
        f"\n\n# Critique of prior attempt (to address)\n\n{critique_xml}"
        + _DISTILL_FOOTER
    )


# =============================================================================
# Plan generation prompt (after 3 distillation rounds, sees concat)
# =============================================================================

_KAPPA_PLAN_FOOTER = (
    "\n\nWrite your research plan inside <solution>...</solution> tags."
    " Target 600 words, max 750 words. Use structured sections (Problem"
    " Statement, Background, Hypothesis, Methodology, Evaluation, Limitations)."
    " The three distilled abstraction lists above were produced independently"
    " from disjoint oracle batches; expect overlap between them. INTEGRATE"
    " the lists, DEDUPLICATE shared items, and prioritize items most relevant"
    " to the research goal. Do not list the abstractions in your plan — use"
    " them as grounding."
)


# τ_v4 (added 2026-04-27): adds explicit T2 (disentanglement) + T3 (compute) instructions
# that the oracle does NOT contain — these need plan-prompt enforcement, not distillation.
_KAPPA_PLAN_FOOTER_V4 = (
    "\n\nWrite your research plan inside <solution>...</solution> tags."
    " Target 600 words, max 750 words. Use structured sections (Problem"
    " Statement, Background, Hypothesis, Methodology, Evaluation, Limitations)."
    " The three distilled abstraction lists above were produced independently"
    " from disjoint oracle batches; expect overlap between them. INTEGRATE"
    " the lists, DEDUPLICATE shared items, and prioritize items most relevant"
    " to the research goal. Do not list the abstractions in your plan — use"
    " them as grounding."
    "\n\n**Citation hygiene**: The distilled abstractions above use internal"
    " index labels like \"Methodology 17\", \"Theory 5\", \"Math 11\","
    " \"Algorithm 3\", \"Background 2\" — these are bookkeeping tags from the"
    " source organization, NOT real citations. Use the conceptual content but"
    " DO NOT reproduce these tags in your plan. Cite real prior work by"
    " first-author + year (e.g., \"Hübotter 2026\") or by descriptive name"
    " only."
    "\n\n**MANDATORY plan-quality elements** (these are not in the distilled"
    " abstractions; you must compose them yourself):"
    "\n- **Disentanglement ablation** (Methodology or Evaluation section):"
    " name 1-2 specific ablations that separate the LLM-prior contribution"
    " from the method's contribution (e.g., 'frozen-LLM same-prompt baseline',"
    " 'method without per-problem updates', 'random-init LoRA'). Cite which"
    " plan component each ablation isolates."
    "\n- **Compute accounting in operational units** (Evaluation or"
    " Limitations section): state compute as GPU-hours per problem (or"
    " wall-clock minutes per problem), NOT as FLOP-only or memory-only."
    " Give a concrete number range (e.g., '2-4 H100-hours per problem')."
    "\n- **Quantified frozen-LLM baseline gap** (Background or Evaluation):"
    " state the prior-art frozen-LLM number on at least one named benchmark"
    " (e.g., 'frozen GPT-4 scores 50.4% on MATH-500') AND your method's"
    " expected delta with rationale."
)


def build_kappa_plan_prompt(goal: str, A_list: list[str], version: str = "v3") -> str:
    """Final plan-generation prompt. Sees concat of A_1, A_2, A_3 (Q4.3 lock).

    Each A_i is the post-extract_solution() text from round i's distillation.
    The plan is the artifact graded by audit_v3 ISOLATED.

    `version`:
      - "v3" (default, Q4.3 lock + Phase 3 commits before 2026-04-27):
        original footer, no T2/T3 enforcement
      - "v4" (added 2026-04-27): adds mandatory disentanglement + compute +
        baseline-gap instructions to recover T2/T3/T1 dims that the oracle
        cannot supply
    """
    if len(A_list) != 3:
        raise ValueError(f"expected 3 distillations, got {len(A_list)}")
    sections = "\n\n".join(
        f"## Distillation round {i} (independent)\n\n{a}"
        for i, a in enumerate(A_list, 1)
    )
    base = (
        "You will write a research plan for the scenario below, grounded in"
        " three distilled abstraction lists. Each list was produced"
        " independently from a disjoint subset of the methodological oracle,"
        " so they may overlap or partially contradict. Integrate carefully."
        f"\n\n# Research goal\n{goal}"
        f"\n\n# Distilled abstractions (3 independent rounds)\n\n{sections}"
    )
    if version == "v4":
        return base + _KAPPA_PLAN_FOOTER_V4
    return base + _KAPPA_PLAN_FOOTER


# =============================================================================
# Cold-start distillation critique (round 1, before any real critique)
# =============================================================================

DISTILLATION_COLD_START_CRITIQUE = """\
<distill_critique>
<missing_critical>This is round 1; no prior distillation exists. Focus on extracting items that name concrete mechanisms (loss forms, search procedures, stability bounds) and items whose Applicability section directly engages the research goal. Drop items whose Source field cites work obviously orthogonal to the goal.</missing_critical>
<noise>To be evaluated against the source paper after this round. Avoid items whose Content section is purely empirical (numbers without mechanism) unless they set a critical baseline number.</noise>
<faithfulness>To be evaluated against the source paper after this round. When condensing, preserve key technical terms (loss names, algorithm names, hyperparameter symbols) verbatim; rewrite surrounding prose only.</faithfulness>
<improvement_directive>For each kept item, write 1-2 sentences condensing both the mechanism AND the applicability to the research goal. Prefer items that name a specific algorithm or equation over items that describe a general principle. Avoid hand-waving phrasings like "this approach has been shown to work well" — name what it produces.</improvement_directive>
</distill_critique>"""


# =============================================================================
# Critic request payload (Opus subagent, privileged source paper)
# =============================================================================

def build_distillation_critic_request_payload(
    *,
    iter_idx: int,
    round_idx: int,
    goal: str,
    distillation_text: str,
    source_paper_md: str,
    plan_id: str,
) -> dict[str, Any]:
    """JSON request body for distillation-critic subagent.

    Critic sees the full source paper as privileged info. Output is a single
    <distill_critique>...</distill_critique> block per A_i, with 4 children:
    missing_critical / noise / faithfulness / improvement_directive.

    This is critique on the DISTILLATION step (per A_j), NOT on the final
    plan. Phase 2 had plan-level critique; Phase 3 moves critique upstream
    to the learnable distillation action (Q5 option A lock).
    """
    return {
        "kind": "distill_critic",
        "iter": iter_idx,
        "round": round_idx,
        "goal": goal,
        "source_paper_md": source_paper_md,
        "plans": [{"plan_id": plan_id, "text": distillation_text}],
        "instructions": (
            "You are an expert research-methodology reviewer with privileged"
            " access to the source paper. The text below is a DISTILLED"
            " abstraction list — a subset of methodological items selected"
            " from a larger oracle. Judge whether this distillation captures"
            " what the source paper says is core for the research goal."
            " DO NOT mention the source paper by name, by author, or by"
            " arxiv ID in your output. Write a structured <distill_critique>"
            " block with the following children:"
            " <missing_critical> (≤3 sentences naming items the source paper"
            " emphasizes that the distillation missed),"
            " <noise> (≤3 sentences naming items the distillation included"
            " that the source paper does not consider central),"
            " <faithfulness> (≤3 sentences naming any item that the"
            " distillation paraphrased incorrectly relative to the original"
            " oracle item's content),"
            " <improvement_directive> (≤120 words, free-text guidance for"
            " the next distillation attempt). Wrap the entire response inside"
            " <distill_critique>...</distill_critique>. Output nothing else."
        ),
    }


# =============================================================================
# Critique parsing
# =============================================================================

_DISTILL_CRITIQUE_RE = re.compile(r"<distill_critique>.*?</distill_critique>", re.DOTALL)


def parse_distillation_critique_xml(raw: str) -> str:
    """Extract first <distill_critique>...</distill_critique> block from raw text.

    Returns the matched block including outer tags. Falls back to
    DISTILLATION_COLD_START_CRITIQUE if no block found.
    """
    if not raw:
        return DISTILLATION_COLD_START_CRITIQUE
    m = _DISTILL_CRITIQUE_RE.search(raw)
    if not m:
        return DISTILLATION_COLD_START_CRITIQUE
    return m.group(0).strip()


# =============================================================================
# Self-review critic prompt (added 2026-04-27 for κ_self variant)
# =============================================================================

def build_self_review_critic_prompt(
    *,
    goal: str,
    distillation_text: str,
    source_paper: str,
    round_idx: int,
) -> str:
    """Prompt for Qwen3-30B base model acting as self-critic.

    Structurally mirrors the Opus distillation critic prompt
    (see build_distillation_critic_request_payload `instructions` field) but
    formatted as a single user-message prompt for direct sampling. Returns
    `<distill_critique>...</distill_critique>` XML.

    Privileged info: source_paper (model can read but must not name it).
    """
    return (
        "You are an expert research-methodology reviewer with privileged"
        " access to the source paper for the research goal below. The text"
        " after `# Distilled abstraction list` is a candidate distillation"
        " produced by another model — your job is to critique it against the"
        " source paper as ground truth. Round " + str(round_idx) + " of 3"
        " (the next round will use your critique to revise)."
        "\n\n# Research goal\n" + goal +
        "\n\n# Source paper (privileged ground truth — for your reasoning"
        " only; DO NOT mention the paper by name, author, or arxiv ID in"
        " your critique)\n" + source_paper +
        "\n\n# Distilled abstraction list to critique\n" + distillation_text +
        "\n\n# Output format"
        "\nWrite a structured <distill_critique> block with these children:"
        "\n- <missing_critical>: ≤3 sentences naming items the source paper"
        " emphasizes that the distillation missed"
        "\n- <noise>: ≤3 sentences naming items the distillation included"
        " that the source paper does not consider central"
        "\n- <faithfulness>: ≤3 sentences naming any item the distillation"
        " paraphrased incorrectly relative to the original oracle item"
        "\n- <improvement_directive>: ≤120 words concrete guidance for the"
        " next distillation attempt (specific items to add, items to drop,"
        " preservation rules)"
        "\n\nWrap the entire response inside <distill_critique>...</distill_critique>."
        " Output nothing else."
    )
