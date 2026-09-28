"""Prompt templates for D5 oracle abstraction v2 build (multi-round Opus extraction).

Per ORACLE_DESIGN_v1.md (locked 2026-04-25): 6 categories — Insights / Methodology /
Theory / Math / Empirical / Failure-modes. 3 rounds with cumulative memory. Round 0
relevance filter. Per-item full citation on first occurrence.
"""
from __future__ import annotations

ORACLE_CATEGORIES = ["Insights", "Methodology", "Theory", "Math", "Empirical", "Failure_modes"]


CATEGORY_DESCRIPTIONS = {
    "Insights": (
        "High-level conceptual moves: 'X works only when Y'; 'self-improvement saturates because Z'; "
        "'AlphaZero's MCTS works because of W'. Used by plan-gen for Problem-statement / Background framing."
    ),
    "Methodology": (
        "Concrete reusable mechanisms: 'MCTS with PUCT selection'; 'REST-style self-distillation loop'. "
        "Each item names the mechanism + cites paper + states what's specifically reusable for this goal."
    ),
    "Theory": (
        "Algorithm reductions and theoretical guarantees: 'MCTS as variational inference'; "
        "'self-distillation as policy improvement'. Connects this work to known primitives."
    ),
    "Math": (
        "Specific objective functions, gradient forms, update rules with full RHS. "
        "'REINFORCE: ∇J = E[R · ∇log π]'; 'Entropy-regularized: J_β = E[R - β · H(π)]'. "
        "Used to write proper formal sections."
    ),
    "Empirical": (
        "Concrete benchmarks, baselines, prior numbers, eval protocols. "
        "'AHC039: top human 566,997'; 'Erdős minimum-overlap prior best 0.380924'. "
        "Used to write specific Empirical sections with cited numbers."
    ),
    "Failure_modes": (
        "What prior work didn't solve, with attribution. 'Prior self-training methods saturate after 3-5 iters'; "
        "'MCTS for math is brittle to reward noise'. Used for Hypothesis (motivation) and Limitations sections."
    ),
}


# =============================================================================
# Round 0 — Relevance filter
# =============================================================================

ROUND0_RELEVANCE_PROMPT = """\
You are evaluating whether a research paper is relevant for grounding a research plan
on the following research goal.

# Research Goal
{goal}

# Candidate Paper
- Title: {title}
- Authors: {authors_short}
- Year: {year}
- Venue: {venue}
- Abstract: {abstract}
- TLDR: {tldr}

# Task

Score this paper's relevance to the research goal on a 0-3 scale, then identify which of
the 6 oracle categories (Insights, Methodology, Theory, Math, Empirical, Failure_modes)
this paper would contribute to.

Score meaning:
- 0 = baseline citation only (e.g. classic textbook reference, citation for a foundational
  concept). Provides no methodological/conceptual contribution to a plan on THIS goal.
- 1 = relevant context but secondary (provides background but doesn't suggest specific
  reusable mechanisms for this plan).
- 2 = primary methodology / theory / empirical source for at least one category.
- 3 = core source — Opus would recommend reading the full paper to extract specific
  formulas, operators, or numerical results.

# Output Format

Respond with ONLY this JSON (no other text, no markdown fence):

{{"relevance_score": <0|1|2|3>,
 "rationale": "<1-2 sentences explaining the score>",
 "expected_categories": [<subset of "Insights"|"Methodology"|"Theory"|"Math"|"Empirical"|"Failure_modes">]}}
"""


# =============================================================================
# Round 1 — Cold extraction (per kept paper)
# =============================================================================

ROUND1_EXTRACT_PROMPT = """\
You are extracting reusable insights from a research paper to ground a research plan
on the following goal.

# Research Goal
{goal}

# Source Paper
- Title: {title}
- Authors: {authors_short}
- Year: {year}
- Venue: {venue}
- Content: {content}

# Expected Categories (from earlier relevance filter)
{expected_categories_csv}

# Categories Reference

You will extract 0-3 items per category. Categories:

{category_descriptions}

# Task

For EACH category in `expected_categories` (and any other category if you find genuinely
reusable content), extract 0-3 items. Each item is a specific, concrete reusable element.

Constraints:
- "Math" items MUST contain a full RHS equation (e.g. `L = -E[r · log π]`) when present in
  the paper. If no equation, leave Math empty.
- "Empirical" items MUST cite specific numerical results from the paper (e.g. "60.4%
  accuracy on MATH-500"). If no concrete numbers, leave Empirical empty.
- "Methodology" items name a SPECIFIC mechanism (e.g. "PUCT selection rule",
  "rejection sampling fine-tuning") with the deterministic operator if available.
- Avoid vague entries like "use reinforcement learning" — be specific.

# Output Format

Respond with ONLY this JSON (no other text, no markdown fence):

{{
  "Insights": [
    {{"summary": "<≤10 word title>", "content": "<2-4 sentences>", "applicability": "<1-2 sentences on how it's reusable for the goal>"}}
  ],
  "Methodology": [
    {{"summary": "...", "content": "...", "applicability": "..."}}
  ],
  "Theory": [...],
  "Math": [
    {{"summary": "<formula name>", "content": "<formula with RHS + variables>", "applicability": "..."}}
  ],
  "Empirical": [
    {{"summary": "<benchmark or eval setting>", "content": "<concrete numbers>", "applicability": "..."}}
  ],
  "Failure_modes": [
    {{"summary": "<what failed>", "content": "<why + attribution>", "applicability": "<how this informs Limitations or Hypothesis>"}}
  ]
}}

If a category has NO content from this paper, set it to an empty list `[]`.
"""


# =============================================================================
# Round 2/3 — Refine extraction (cumulative memory)
# =============================================================================

ROUND_REFINE_PROMPT = """\
You are refining the oracle abstraction. You see (a) the accumulated state from prior
rounds (other papers' extractions), (b) one paper's content, and (c) your prior-round
extraction for this paper.

The goal of refinement is to:
- Add new items if this paper contributes things not yet captured
- Sharpen existing items (more concrete content, better attribution)
- Mark obsolete items if a different paper covered the same point better

# Research Goal
{goal}

# Accumulated State (from prior round, all papers, organized by category)
{accumulated_state}

# This Paper
- Title: {title}
- Authors: {authors_short}
- Content: {content}

# Your Prior-Round Extraction for This Paper
{prior_extraction}

# Categories Reference

{category_descriptions}

# Task

Output the FINAL extraction for this paper after considering the accumulated state. May:
- Keep an item from prior extraction unchanged
- Refine an item (better content, deduplicate)
- Drop an item if already covered by another paper better
- Add a new item discovered on re-read

Same constraints as Round 1: Math items have full RHS; Empirical items have concrete
numbers; Methodology items name specific mechanisms.

# Output Format

Respond with ONLY this JSON (no other text, no markdown fence):

{{
  "Insights": [
    {{"summary": "...", "content": "...", "applicability": "...", "action": "keep"|"refine"|"new"}}
  ],
  "Methodology": [...],
  "Theory": [...],
  "Math": [...],
  "Empirical": [...],
  "Failure_modes": [...]
}}

Empty list `[]` if no items in a category.
"""


# =============================================================================
# Helpers
# =============================================================================

def render_category_descriptions() -> str:
    """Render the 6-category description block for prompts."""
    lines = []
    for cat in ORACLE_CATEGORIES:
        lines.append(f"- **{cat}**: {CATEGORY_DESCRIPTIONS[cat]}")
    return "\n".join(lines)


def render_accumulated_state(state: dict) -> str:
    """Render the cumulative state across all papers, organized by category, for refine rounds."""
    out = []
    for cat in ORACLE_CATEGORIES:
        items = state.get(cat, [])
        if not items:
            continue
        out.append(f"## {cat}")
        for it in items:
            ref = it.get("source_paper", "[?]")
            summary = it.get("summary", "")
            out.append(f"- {summary} (source: {ref})")
    return "\n".join(out) if out else "(empty — first paper)"


def render_prior_extraction(extraction: dict) -> str:
    """Render this paper's prior-round extraction (full detail)."""
    out = []
    for cat in ORACLE_CATEGORIES:
        items = extraction.get(cat, [])
        if not items:
            continue
        out.append(f"## {cat}")
        for it in items:
            out.append(f"- summary: {it.get('summary', '')}")
            out.append(f"  content: {it.get('content', '')}")
            out.append(f"  applicability: {it.get('applicability', '')}")
    return "\n".join(out) if out else "(empty)"
