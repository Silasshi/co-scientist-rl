"""Unit tests for CR-v6 locus-attribution parts of ten_signal_reward.py.

Covers:
- `SignalSpec` accepts `locus_directive` and defaults to None
- `build_single_signal_prompt` appends `<locus>` block only when directive is set
- `parse_locus` strict rejection of malformed / hallucinated / missing-field cases
- `parse_locus` happy path with verbatim-substring validation

Does NOT exercise the grader (no API calls). For end-to-end locus behavior see
`projects/ttt_discover/analysis/signal_validity/scripts/grade/grade_locus_s3_pilot.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.shared.ten_signal_reward import (  # noqa: E402
    SIGNALS,
    LocusEntry,
    SignalSpec,
    build_single_signal_prompt,
    parse_locus,
)


# =============================================================================
# SignalSpec: locus_directive field presence & defaults
# =============================================================================

def test_signalspec_locus_directive_defaults_to_none():
    spec = SignalSpec(id="S_test", name="Test", question="q?")
    assert spec.locus_directive is None


def test_signalspec_locus_directive_accepts_string():
    spec = SignalSpec(
        id="S_test",
        name="Test",
        question="q?",
        locus_directive="Quote vague spans.",
    )
    assert spec.locus_directive == "Quote vague spans."


def test_s9_focus_has_no_locus_directive():
    """S9_focus is LOCAL (whole Core Idea); locus targeting is unnecessary."""
    s9 = next(s for s in SIGNALS if s.id == "S9_focus")
    assert s9.locus_directive is None


def test_all_non_s9_signals_have_locus_directive():
    """CR-v6: all active non-S9 signals ship with a drafted directive.

    Even S4_significance (disabled via config) has one for consistency if
    later re-enabled.
    """
    for spec in SIGNALS:
        if spec.id == "S9_focus":
            continue
        assert spec.locus_directive is not None, (
            f"{spec.id} missing locus_directive — CR-v6 requires all non-S9 "
            f"signals to ship a draft directive"
        )
        assert len(spec.locus_directive) >= 40, (
            f"{spec.id} locus_directive unexpectedly short"
        )


# =============================================================================
# build_single_signal_prompt: conditional locus block
# =============================================================================

GOAL = "Discover a new training paradigm for LLMs."
PLAN = "## Problem\nX is important.\n\n## Methodology\nWe will use standard techniques."


def test_build_prompt_emit_locus_false_suppresses_block():
    """Default emit_locus=False: even a signal WITH directive gets no locus block."""
    spec = next(s for s in SIGNALS if s.id == "S7_specificity")
    prompt = build_single_signal_prompt(GOAL, PLAN, spec)  # default emit_locus=False
    assert "<locus>" not in prompt
    assert "Locus Attribution" not in prompt


def test_build_prompt_with_locus_directive_appends_locus_block():
    spec = next(s for s in SIGNALS if s.id == "S7_specificity")
    prompt = build_single_signal_prompt(GOAL, PLAN, spec, emit_locus=True)
    assert "<locus>" in prompt
    assert "Additional Output: Locus Attribution" in prompt
    assert "VERBATIM RULE" in prompt
    # Signal name parameterization
    assert "**Implementation Specificity**" in prompt
    # Signal-specific directive content
    assert "VAGUE MARKER" in prompt


def test_build_prompt_without_locus_directive_has_no_locus_block():
    s9 = next(s for s in SIGNALS if s.id == "S9_focus")
    prompt = build_single_signal_prompt(GOAL, PLAN, s9)
    assert "<locus>" not in prompt
    assert "Locus Attribution" not in prompt


def test_locus_block_uses_signal_specific_name():
    """Each signal's locus prompt should mention that signal's name, not S3's."""
    s7 = next(s for s in SIGNALS if s.id == "S7_specificity")
    prompt = build_single_signal_prompt(GOAL, PLAN, s7, emit_locus=True)
    # Parameterized in at least 2 places (intro + format)
    assert prompt.count("Implementation Specificity") >= 2
    # Not leaking S3's name
    assert "Positioning weakness" not in prompt


def test_locus_block_overhead_is_nontrivial():
    """Sanity: turning locus on adds meaningful prompt length (for cost accounting)."""
    s7 = next(s for s in SIGNALS if s.id == "S7_specificity")
    s9 = next(s for s in SIGNALS if s.id == "S9_focus")
    len_with_locus = len(build_single_signal_prompt(GOAL, PLAN, s7, emit_locus=True))
    len_without = len(build_single_signal_prompt(GOAL, PLAN, s9, emit_locus=True))
    assert len_with_locus - len_without > 500, (
        "Locus block should add at least 500 chars of overhead"
    )


# =============================================================================
# parse_locus: happy path
# =============================================================================

PLAN_BODY = (
    "## Problem\n"
    "We will use standard techniques to tune hyperparameters appropriately.\n"
    "\n"
    "## Methodology\n"
    "Various baselines will be explored. The approach is suitable for the task.\n"
)


def test_parse_locus_happy_path_single_entry():
    response = """
<evaluation>
    <dim id="S7_specificity"><reasoning>...</reasoning><score>2</score></dim>
</evaluation>
<locus>
[{"section_hint": "Problem", "quote": "standard techniques", "why": "Vague marker without named method."}]
</locus>
"""
    result = parse_locus(response, PLAN_BODY)
    assert result is not None
    assert len(result) == 1
    entry = result[0]
    assert isinstance(entry, LocusEntry)
    assert entry.section_hint == "Problem"
    assert entry.quote == "standard techniques"
    assert entry.why.startswith("Vague marker")


def test_parse_locus_multiple_entries():
    response = """<locus>
[
  {"section_hint": "Problem", "quote": "standard techniques", "why": "Vague."},
  {"section_hint": "Methodology", "quote": "Various baselines", "why": "Unnamed."},
  {"section_hint": "Methodology", "quote": "suitable for the task", "why": "Vague."}
]
</locus>"""
    result = parse_locus(response, PLAN_BODY)
    assert result is not None
    assert len(result) == 3
    assert all(r.quote in PLAN_BODY for r in result)


def test_parse_locus_empty_array_returns_empty_list():
    """Grader emitted `[]` meaning 'no weak spans' — valid, distinct from None."""
    response = "<locus>[]</locus>"
    result = parse_locus(response, PLAN_BODY)
    assert result == []


# =============================================================================
# parse_locus: rejection paths (return None)
# =============================================================================

def test_parse_locus_returns_none_when_no_block():
    response = "<evaluation><dim id='X'><score>3</score></dim></evaluation>"
    assert parse_locus(response, PLAN_BODY) is None


def test_parse_locus_returns_none_on_malformed_json():
    response = "<locus>[{not valid json]</locus>"
    assert parse_locus(response, PLAN_BODY) is None


def test_parse_locus_returns_none_when_not_a_list():
    response = '<locus>{"section_hint": "X", "quote": "standard techniques", "why": "..."}</locus>'
    assert parse_locus(response, PLAN_BODY) is None


def test_parse_locus_returns_none_on_hallucinated_quote():
    """If ANY quote is not verbatim in plan, reject whole list."""
    response = """<locus>
[
  {"section_hint": "Problem", "quote": "standard techniques", "why": "ok"},
  {"section_hint": "Methodology", "quote": "a span that is not in the plan", "why": "hallucinated"}
]
</locus>"""
    assert parse_locus(response, PLAN_BODY) is None


def test_parse_locus_returns_none_when_quote_missing():
    response = '<locus>[{"section_hint": "Problem", "why": "no quote key"}]</locus>'
    assert parse_locus(response, PLAN_BODY) is None


def test_parse_locus_returns_none_when_quote_empty_string():
    response = '<locus>[{"section_hint": "Problem", "quote": "", "why": "empty"}]</locus>'
    assert parse_locus(response, PLAN_BODY) is None


def test_parse_locus_returns_none_when_entry_not_dict():
    response = '<locus>["just a string"]</locus>'
    assert parse_locus(response, PLAN_BODY) is None


def test_parse_locus_tolerates_missing_section_hint_via_default():
    """section_hint defaults to ''; quote and why are the load-bearing fields."""
    response = '<locus>[{"quote": "standard techniques", "why": "vague"}]</locus>'
    result = parse_locus(response, PLAN_BODY)
    assert result is not None
    assert len(result) == 1
    assert result[0].section_hint == ""


def test_parse_locus_case_insensitive_tag_matching():
    """Grader sometimes emits <LOCUS>...</LOCUS> — still parse."""
    response = '<LOCUS>[{"section_hint": "Problem", "quote": "standard techniques", "why": "vague"}]</LOCUS>'
    result = parse_locus(response, PLAN_BODY)
    assert result is not None
    assert len(result) == 1


def test_parse_locus_survives_whitespace_in_block():
    """Leading/trailing whitespace inside <locus> tags is common."""
    response = """<locus>

    [{"section_hint": "Problem", "quote": "standard techniques", "why": "vague"}]

    </locus>"""
    result = parse_locus(response, PLAN_BODY)
    assert result is not None


# =============================================================================
# parse_locus: fuzzy matching (Phase 1.5 smoke test finding)
# =============================================================================

PLAN_BODY_MD = (
    "## Problem\n"
    "Current methods face challenges.\n"
    "\n"
    "## Evaluation\n"
    "**Baselines**:\n"
    "  - **AlphaEvolve**: Evolutionary search over a frozen LLM.\n"
    "  - **Standard RL**: Policy optimization baseline.\n"
    "  - **Random search**: Baseline for comparison.\n"
)


def test_parse_locus_fuzzy_match_condensed_list():
    """Grader condenses markdown list into single sentence — fuzzy match should recover."""
    condensed = "Baselines: AlphaEvolve, Standard RL, Random search."
    response = f'<locus>[{{"section_hint": "Evaluation", "quote": "{condensed}", "why": "strawman"}}]</locus>'
    result = parse_locus(response, PLAN_BODY_MD)
    assert result is not None, "Fuzzy match should find the baselines section"
    assert len(result) == 1
    # The returned quote must be a VERBATIM substring of PLAN_BODY_MD
    assert result[0].quote in PLAN_BODY_MD, (
        f"Corrected quote must be verbatim in plan. Got: {result[0].quote!r}"
    )


def test_parse_locus_fuzzy_match_still_rejects_total_hallucination():
    """If the grader fabricates content that's nowhere near the plan, reject."""
    response = '<locus>[{"section_hint": "X", "quote": "quantum entanglement in deep learning", "why": "bad"}]</locus>'
    result = parse_locus(response, PLAN_BODY_MD)
    assert result is None, "Total hallucination should still be rejected"


def test_parse_locus_exact_match_still_preferred():
    """When the grader quotes exactly, no fuzzy matching needed."""
    exact = "Current methods face challenges."
    response = f'<locus>[{{"section_hint": "Problem", "quote": "{exact}", "why": "vague"}}]</locus>'
    result = parse_locus(response, PLAN_BODY_MD)
    assert result is not None
    assert result[0].quote == exact  # Exact, not fuzzy-modified
