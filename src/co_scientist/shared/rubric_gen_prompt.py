"""
Rubric-generator prompt module for GER-CR-v1.

Produces the system+user messages sent to the rubric generator (Opus 4.7 via
subagent) to elicit new R_active items from a pairwise contrast of rollouts.
The prompt combines two peer-reviewed / peer-reviewed-in-flight designs:

  * DR Tulu (arXiv 2511.19399) Appendix A.1 — core discriminative-power and
    non-redundancy guidelines, positive/negative split, quantity ceiling.
  * OnlineRubrics (arXiv 2510.07284, ICLR 2026 submit) Figure 2 — the pairwise
    contrast framing and the "based on ONE of the responses" grounding
    constraint that blocks the generator from hallucinating criteria from its
    own priors.

A few adaptations were made for our grant-proposal task:

  * Task-specific preamble (plans are long-form research proposals, not QA or
    survey answers) so Opus doesn't fall back to deep-research-style criteria.
  * Existing rubric block passes BOTH R_persist (v8 signals; Likert rubric
    names) AND R_active (current evolving items) so the generator knows what is
    already covered.
  * Negative rubric wording standardized as "avoids X" (positive phrasing of a
    pitfall) so scoring is consistently "higher = better" and we don't have
    sign bugs downstream.
  * Explicit targets on what to hunt for first: length-correlated hacks,
    fabricated citations, template repetition, over-claiming — the three
    Goodhart modes confirmed by `analysis/attack_surfaces_2026_04_22.ipynb`.

Output contract: Opus must return a single JSON object (in a ```json block or
raw) with keys `positive_rubrics` and `negative_rubrics`. Each item has
`title`, `description`, and optional `weight` (defaults to 1.0).
"""

from __future__ import annotations

import json
import re
from typing import Iterable

from co_scientist.shared.rubric_buffer import NEGATIVE, POSITIVE, RubricItem


SYSTEM_PROMPT = """You are an expert evaluator generating adaptive rubric items for assessing
long-form research grant proposals. You will be shown a research goal, the
current rubric (criteria already being applied), and two candidate research
plans. Your job is to propose NEW criteria that distinguish the two plans from
each other and are NOT already covered by the existing rubric.

## Output format (STRICT)

Return a single JSON object. No prose outside the JSON. Use this schema:

```json
{
  "positive_rubrics": [
    {"title": "<short abstract label>", "description": "<specific actionable criterion>"}
  ],
  "negative_rubrics": [
    {"title": "<short abstract label>", "description": "<specific failure pattern, phrased as 'avoids X' so higher score = better>"}
  ]
}
```

Return `{"positive_rubrics": [], "negative_rubrics": []}` if no meaningful new
criterion is warranted. Quality over quantity: 0-3 total new items is fine.

## Core guidelines

### 1. Discriminative power
- A new item must clearly separate the two plans you see. If both satisfy it
  equally (or both fail it equally), it has no gradient value — omit it.
- Exclude generic criteria that would apply equally to any research plan.

### 2. Grounded in the observed plans (CRITICAL)
- Every criterion must be **based on a difference you actually observe between
  Plan A and Plan B**. Do NOT introduce criteria from your own general
  knowledge of what good grant proposals should contain. The existing rubric
  already covers general quality; your job is to surface emergent differences.
- If you cannot point to a specific passage in one plan that the other lacks
  (or vice versa), the criterion is ungrounded — omit it.

### 3. Non-redundancy
- If the difference is already captured by any item in the existing rubric
  (even partially), do not re-introduce it. Prefer zero new items over
  duplicates.

### 4. No mirror pairs
- Never create a positive+negative pair that test the same dimension in
  opposite directions (e.g. "cites rigorously" as positive AND "fails to cite"
  as negative). Choose the more discriminative direction only.

### 5. Negative rubrics are for active failure modes
- Phrase them as "avoids X" / "does not X" so that higher grade = better.
- Reserve them for concrete, observable hacks (padding, fabrication, template
  repetition, over-claiming, etc.), not for "absence of a positive feature".

## Priority targets (from this project's known Goodhart modes)

You may weight these higher if you see them in the plans:

- **Length padding**: one plan uses more words than the other to make the same
  point; fewer distinct ideas per section; filler sentences that restate the
  title; long lists of qualitative adjectives without substantiation.
- **Fabricated rigor**: citations to non-existent papers, made-up theorem
  names, gratuitous equations that do not bind the research steps, named
  datasets without procedure.
- **Template repetition**: plans that share an identical section order and
  section-by-section style (a sign of RL collapsing to one template).

If the two plans differ on one of these axes, a criterion targeting that axis
has high value — introduce it.
"""


def _format_existing_rubric_block(persistent_names: list[str], active_items: Iterable[RubricItem]) -> str:
    """Render the current rubric for the generator's context."""
    lines: list[str] = []
    lines.append("### Persistent rubric (already applied at every iteration)")
    if persistent_names:
        for n in persistent_names:
            lines.append(f"- {n}")
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("### Active evolving rubric (built up during training so far)")
    active = list(active_items)
    if active:
        for it in active:
            tag = "POS" if it.rubric_type == POSITIVE else "NEG"
            lines.append(f"- [{tag}] **{it.title}** — {it.description}")
    else:
        lines.append("- (empty — this is the first evolving-rubric elicitation)")

    return "\n".join(lines)


def build_rubric_gen_messages(
    research_goal: str,
    persistent_rubric_names: list[str],
    active_rubric_items: Iterable[RubricItem],
    plan_a: str,
    plan_b: str,
    plan_a_label: str = "Plan A (current policy)",
    plan_b_label: str = "Plan B (control)",
) -> list[dict]:
    """
    Build the chat messages for the rubric generator.

    Parameters
    ----------
    research_goal: The research goal / prompt the plans address.
    persistent_rubric_names: Human-readable names of R_persist signals (so the
        generator knows what's already covered). Pass signal `.name` or
        `.title` fields from grant_rubric_v8.SIGNALS.
    active_rubric_items: Current R_active buffer items.
    plan_a, plan_b: The two plan texts to contrast. Following OnlineRubrics,
        `plan_a` should be from the current policy and `plan_b` from a control
        (initial policy, reference proposal, or B4 baseline).
    plan_a_label, plan_b_label: Labels shown to the generator to help it frame
        the contrast.

    Returns
    -------
    List of {"role": ..., "content": ...} suitable for an OpenAI-compatible
    chat completions API call.
    """
    user_lines: list[str] = []
    user_lines.append("## Research goal")
    user_lines.append(research_goal.strip())
    user_lines.append("")
    user_lines.append("## Existing rubric")
    user_lines.append(
        _format_existing_rubric_block(persistent_rubric_names, active_rubric_items)
    )
    user_lines.append("")
    user_lines.append(f"## {plan_a_label}")
    user_lines.append(plan_a.strip())
    user_lines.append("")
    user_lines.append(f"## {plan_b_label}")
    user_lines.append(plan_b.strip())
    user_lines.append("")
    user_lines.append(
        "Analyse the two plans, identify differences not already covered by "
        "the existing rubric, and return the JSON object specified in your "
        "system prompt. Remember: every new criterion must be grounded in a "
        "concrete difference you observe between the two plans."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


# ---------------------------------------------------------------- output parser

_JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"(\{.*\})", re.DOTALL)


def _extract_json_blob(text: str) -> dict:
    """Find and parse the JSON object in the generator's response."""
    text = text.strip()
    m = _JSON_FENCE_RE.search(text)
    if m:
        return json.loads(m.group(1))
    # fall back to first { ... } block
    m = _BARE_JSON_RE.search(text)
    if not m:
        raise ValueError("no JSON object found in rubric-generator output")
    return json.loads(m.group(1))


def parse_rubric_response(raw_text: str, created_at_iter: int) -> list[RubricItem]:
    """
    Parse the generator's JSON into RubricItem instances.

    Malformed individual items are skipped (logged via ValueError message in
    the caller's logs); a top-level JSON parse error raises so the caller can
    retry or surface the failure.
    """
    blob = _extract_json_blob(raw_text)
    items: list[RubricItem] = []
    for kind, polarity in [("positive_rubrics", POSITIVE), ("negative_rubrics", NEGATIVE)]:
        for raw in blob.get(kind, []) or []:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title", "")).strip()
            description = str(raw.get("description", "")).strip()
            if not title or not description:
                continue
            weight_raw = raw.get("weight", 1.0)
            try:
                weight = float(weight_raw) if weight_raw is not None else 1.0
            except (TypeError, ValueError):
                weight = 1.0
            if weight <= 0:
                continue
            items.append(
                RubricItem(
                    title=title,
                    description=description,
                    rubric_type=polarity,
                    weight=weight,
                    created_at_iter=created_at_iter,
                )
            )
    return items


# ------------------------------------------------------------------- self-test

def _run_sanity_checks() -> None:
    msgs = build_rubric_gen_messages(
        research_goal="Develop a foundational optimizer for deep learning.",
        persistent_rubric_names=["G1 Goal-Contrast Margin", "G4 Focus"],
        active_rubric_items=[],
        plan_a="## Methods\nWe propose Bayesian optimization with Gaussian processes...",
        plan_b="## Methods\nWe use random search as baseline...",
    )
    assert msgs[0]["role"] == "system"
    assert "JSON" in msgs[0]["content"]
    assert "Research goal" in msgs[1]["content"]
    assert "Plan A" in msgs[1]["content"]
    assert "(empty" in msgs[1]["content"], "active rubric block should show empty"

    # With R_active non-empty
    active = [
        RubricItem(
            title="Reproducibility",
            description="Names concrete datasets and their access procedure",
            rubric_type=POSITIVE,
        )
    ]
    msgs2 = build_rubric_gen_messages(
        research_goal="x", persistent_rubric_names=[], active_rubric_items=active,
        plan_a="a", plan_b="b",
    )
    assert "Reproducibility" in msgs2[1]["content"]

    # Parse a fenced JSON
    fenced = """
    Here's my analysis.
    ```json
    {
      "positive_rubrics": [
        {"title": "Named datasets", "description": "Plan names at least two concrete datasets with access procedure"}
      ],
      "negative_rubrics": [
        {"title": "Padding", "description": "Avoids paragraph-long restatements of the title without adding substance"}
      ]
    }
    ```
    Done.
    """
    items = parse_rubric_response(fenced, created_at_iter=3)
    assert len(items) == 2
    assert items[0].rubric_type == POSITIVE
    assert items[1].rubric_type == NEGATIVE
    assert items[0].created_at_iter == 3

    # Empty arrays should parse cleanly
    empty = '{"positive_rubrics": [], "negative_rubrics": []}'
    items = parse_rubric_response(empty, created_at_iter=5)
    assert items == []

    # Malformed items dropped silently
    mixed = """{"positive_rubrics": [
        {"title": "", "description": "skip"},
        {"title": "ok", "description": "keep this", "weight": 2.0},
        "bad"
    ], "negative_rubrics": []}"""
    items = parse_rubric_response(mixed, created_at_iter=0)
    assert len(items) == 1
    assert items[0].title == "ok"
    assert items[0].weight == 2.0

    print("rubric_gen_prompt self-tests passed.")


if __name__ == "__main__":
    _run_sanity_checks()
