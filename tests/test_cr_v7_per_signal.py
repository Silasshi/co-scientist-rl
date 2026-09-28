"""Unit tests for CR-v7 per-signal context REINFORCE pipeline.

Covers:
- `build_single_signal_prompt(..., emit_critique=True)` adds the XML block
- `parse_scores` extracts the `<critique>` field (empty when missing)
- `aggregate_critique_across_repeats` picks median-score repeat's critique
- `build_whole_plan_revision_prompt` constructs sampling prompt with all 8 critiques
- `build_per_signal_context` isolates one critique

Does NOT exercise the grader or policy (no API calls).
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.shared.ten_signal_reward import (  # noqa: E402
    SIGNALS,
    aggregate_critique_across_repeats,
    build_single_signal_prompt,
    parse_scores,
)
from co_scientist.ttt_discover.train_cr_v7 import (  # noqa: E402
    build_per_signal_context,
    build_whole_plan_revision_prompt,
)


# =============================================================================
# build_single_signal_prompt(emit_critique=...)
# =============================================================================

def test_build_single_signal_prompt_without_critique():
    """Default call omits the <critique> XML block."""
    spec = SIGNALS[0]
    prompt = build_single_signal_prompt("goal", "plan", spec)
    assert "<critique>" not in prompt
    assert "<score>INTEGER 1-5</score>" in prompt


def test_build_single_signal_prompt_with_critique():
    """emit_critique=True adds the <critique> block with actionable instructions."""
    spec = SIGNALS[0]
    prompt = build_single_signal_prompt("goal", "plan", spec, emit_critique=True)
    assert "<critique>" in prompt
    assert "</critique>" in prompt
    assert "missing, weak, or unconvincing" in prompt
    # Must reference the signal's name in the critique instruction.
    assert spec.name in prompt


# =============================================================================
# parse_scores — <critique> extraction
# =============================================================================

def test_parse_scores_includes_critique_when_present():
    xml = """
    <evaluation>
        <dim id="S2_rigor">
            <reasoning>I analyzed the rigor.</reasoning>
            <score>3</score>
            <critique>Plan needs named baselines with prior numbers.</critique>
        </dim>
    </evaluation>
    """
    result = parse_scores(xml)
    assert "S2_rigor" in result
    assert result["S2_rigor"]["score"] == 3
    assert result["S2_rigor"]["critique"] == "Plan needs named baselines with prior numbers."


def test_parse_scores_empty_critique_when_missing():
    """Grader may omit <critique>; should default to empty string without crashing."""
    xml = """
    <evaluation>
        <dim id="S1_depth">
            <reasoning>analysis</reasoning>
            <score>4</score>
        </dim>
    </evaluation>
    """
    result = parse_scores(xml)
    assert result["S1_depth"]["critique"] == ""
    assert result["S1_depth"]["score"] == 4


def test_parse_scores_malformed_score_still_captures_critique():
    """If score is unparseable, critique should still be recovered."""
    xml = """
    <evaluation>
        <dim id="S7_specificity">
            <reasoning>r</reasoning>
            <score>not a number</score>
            <critique>Missing implementation specifics.</critique>
        </dim>
    </evaluation>
    """
    result = parse_scores(xml)
    assert result["S7_specificity"]["score"] is None
    assert result["S7_specificity"]["critique"] == "Missing implementation specifics."


# =============================================================================
# aggregate_critique_across_repeats
# =============================================================================

def test_aggregate_critique_picks_median_repeat():
    """When 3 repeats have scores [3, 4, 3], pick critique from a score-3 repeat."""
    parsed = [
        {"S2_rigor": {"score": 3, "reasoning": "", "critique": "critique_A"}},
        {"S2_rigor": {"score": 4, "reasoning": "", "critique": "critique_B"}},
        {"S2_rigor": {"score": 3, "reasoning": "", "critique": "critique_C"}},
    ]
    median = {"S2_rigor": 3}
    out = aggregate_critique_across_repeats(parsed, median)
    # Median is 3; critique_A or critique_C both valid; first median-match wins
    assert out["S2_rigor"] in {"critique_A", "critique_C"}


def test_aggregate_critique_falls_back_to_first_nonempty():
    """When no repeat matches the median score, pick first non-empty critique."""
    parsed = [
        {"S3_positioning": {"score": 2, "reasoning": "", "critique": ""}},
        {"S3_positioning": {"score": 5, "reasoning": "", "critique": "non-empty critique"}},
    ]
    # Suppose median wasn't computed for S3 (None), aggregator should still
    # return the first non-empty critique.
    median: dict[str, int | None] = {"S3_positioning": None}
    out = aggregate_critique_across_repeats(parsed, median)
    assert out["S3_positioning"] == "non-empty critique"


def test_aggregate_critique_all_empty_returns_empty():
    parsed = [
        {"S5_feasibility": {"score": 3, "reasoning": "", "critique": ""}},
        {"S5_feasibility": {"score": 3, "reasoning": "", "critique": ""}},
    ]
    median = {"S5_feasibility": 3}
    out = aggregate_critique_across_repeats(parsed, median)
    assert out["S5_feasibility"] == ""


# =============================================================================
# build_whole_plan_revision_prompt (CR-v7 sampling prompt)
# =============================================================================

def test_whole_plan_prompt_contains_every_signal_critique():
    """All 8 critiques must appear in the sampling prompt (minimal-mode)."""
    signal_vector = {s.id: 3 for s in SIGNALS}
    critiques = {s.id: f"critique_for_{s.id}" for s in SIGNALS}
    prompt = build_whole_plan_revision_prompt(
        goal="Test goal",
        plan_text="Test plan body.",
        signal_vector=signal_vector,
        critiques=critiques,
    )
    for s in SIGNALS:
        assert f"critique_for_{s.id}" in prompt, f"Missing critique for {s.id}"
        assert s.name in prompt, f"Missing signal name {s.name}"
    assert "Test plan body." in prompt
    assert "<solution>" in prompt


def test_whole_plan_prompt_handles_missing_critique():
    """Missing critique entries should render with '[no critique]' placeholder."""
    signal_vector = {s.id: 2 for s in SIGNALS}
    critiques: dict[str, str] = {}  # none provided
    prompt = build_whole_plan_revision_prompt(
        goal="g", plan_text="p", signal_vector=signal_vector, critiques=critiques,
    )
    assert "[no critique]" in prompt


# =============================================================================
# build_per_signal_context (CR-v7 loss-only context)
# =============================================================================

def test_per_signal_context_isolates_single_critique():
    """Context for signal_i must contain ONLY critique_i, not other critiques."""
    critiques = {
        "S1_depth": "depth critique text",
        "S2_rigor": "rigor critique text",
        "S7_specificity": "specificity critique text",
    }
    ctx_rigor = build_per_signal_context(
        plan_text="plan body",
        critique_i=critiques["S2_rigor"],
        signal_name="Scientific Rigor",
        score=2,
        goal="test goal",
    )
    assert "rigor critique text" in ctx_rigor
    assert "depth critique text" not in ctx_rigor
    assert "specificity critique text" not in ctx_rigor
    assert "Scientific Rigor" in ctx_rigor
    assert "plan body" in ctx_rigor


def test_per_signal_context_mirrors_sampling_context_structure():
    """Loss-time context MUST use the same headers as sampling context
    (build_whole_plan_revision_prompt) so logπ(a|context_i) differs from
    logπ(a|full-context) only by critique content, not scaffolding."""
    ctx = build_per_signal_context(
        plan_text="P", critique_i="c", signal_name="S", score=3, goal="G",
    )
    assert "# Research Goal" in ctx
    assert "# Current Plan" in ctx
    assert "# Per-signal feedback" in ctx
    assert "# Output Format" in ctx
    assert "<solution>" in ctx


def test_per_signal_context_handles_empty_critique():
    """Empty critique should render with a placeholder (no crash)."""
    ctx = build_per_signal_context(
        plan_text="plan", critique_i="", signal_name="S1", score=None, goal="g",
    )
    assert "[no critique]" in ctx


def test_per_signal_context_truncates_long_plan():
    """Plans over MAX_PLAN_CHARS_IN_PROMPT are truncated to keep prompt under context limit."""
    long_plan = "x" * 20000
    ctx = build_per_signal_context(
        plan_text=long_plan, critique_i="c", signal_name="S", score=3, goal="g",
    )
    assert len(ctx) < 15000
    assert "[... rest truncated ...]" in ctx


# =============================================================================
# v9 signal set: S2a_formalism + SA_arithmetic
# =============================================================================

def test_v9_signals_in_signal_list():
    """v9 signals must be present in the SIGNALS list."""
    ids = {s.id for s in SIGNALS}
    assert "S2a_formalism" in ids, "S2a_formalism missing from SIGNALS"
    assert "SA_arithmetic" in ids, "SA_arithmetic missing from SIGNALS"


def test_v9_signals_have_weights():
    """v9 signals must have entries in SIGNAL_WEIGHTS."""
    from co_scientist.shared.ten_signal_reward import SIGNAL_WEIGHTS
    assert "S2a_formalism" in SIGNAL_WEIGHTS
    assert "SA_arithmetic" in SIGNAL_WEIGHTS
    assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 1e-6


def test_v9_sa_arithmetic_has_grader_override():
    """SA_arithmetic must route to Qwen3-235B via grader_model_override."""
    sa = [s for s in SIGNALS if s.id == "SA_arithmetic"][0]
    assert sa.grader_model_override is not None
    assert "235B" in sa.grader_model_override
    assert "Instruct" in sa.grader_model_override


def test_v9_s2a_formalism_has_no_grader_override():
    """S2a_formalism uses the default grader (Qwen3-30B)."""
    s2a = [s for s in SIGNALS if s.id == "S2a_formalism"][0]
    assert s2a.grader_model_override is None


def test_v9_s2a_formalism_rubric_mentions_formula_counting():
    """S2a rubric must be about counting non-trivial formulas."""
    s2a = [s for s in SIGNALS if s.id == "S2a_formalism"][0]
    assert "NON-TRIVIAL" in s2a.cot_scaffolding
    assert "N_nontrivial" in s2a.cot_scaffolding
    assert "TAUTOLOGICAL" in s2a.cot_scaffolding


def test_v9_sa_arithmetic_rubric_mentions_consistency():
    """SA rubric must be about checking arithmetic consistency."""
    sa = [s for s in SIGNALS if s.id == "SA_arithmetic"][0]
    assert "INCONSISTENT" in sa.cot_scaffolding
    assert "CONSISTENT" in sa.cot_scaffolding
    assert "order of magnitude" in sa.cot_scaffolding.lower()


# =============================================================================
# strip_critiques (B4_stripped baseline)
# =============================================================================

def test_strip_critiques_shows_aggregate_only():
    """strip_critiques=True should hide per-signal critiques, show aggregate."""
    signal_vector = {s.id: 3 for s in SIGNALS}
    critiques = {s.id: f"critique_for_{s.id}" for s in SIGNALS}
    prompt = build_whole_plan_revision_prompt(
        goal="Test goal",
        plan_text="Test plan body.",
        signal_vector=signal_vector,
        critiques=critiques,
        strip_critiques=True,
    )
    assert "Overall score:" in prompt
    for s in SIGNALS:
        assert f"critique_for_{s.id}" not in prompt, f"Critique for {s.id} should be stripped"


def test_strip_critiques_false_shows_all_critiques():
    """strip_critiques=False (default) should show all per-signal critiques."""
    signal_vector = {s.id: 4 for s in SIGNALS}
    critiques = {s.id: f"feedback_{s.id}" for s in SIGNALS}
    prompt = build_whole_plan_revision_prompt(
        goal="g", plan_text="p",
        signal_vector=signal_vector,
        critiques=critiques,
        strip_critiques=False,
    )
    assert "Overall score:" not in prompt
    for s in SIGNALS:
        assert f"feedback_{s.id}" in prompt
