"""Unit tests for CR-v6 locus-revision functions in train_critique_revise.py.

Covers:
- `apply_locus_revisions`: exact-string replacement with reject semantics
- `parse_locus_revisions`: <revisions> block extraction
- `build_locus_revision_prompt`: prompt structure and parameterization

Does NOT exercise the grader or policy (no API calls).
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.ttt_discover.train_critique_revise import (  # noqa: E402
    _spans_complete_block,
    apply_locus_revisions,
    parse_locus_revisions,
    build_locus_revision_prompt,
)
from co_scientist.shared.ten_signal_reward import LocusEntry  # noqa: E402


# =============================================================================
# apply_locus_revisions
# =============================================================================

# PLAN exercises every block structure the validator must recognise:
# header-bounded paragraph, blank-line-bounded paragraph, bullet list items,
# and end-of-plan boundary.
PLAN = (
    "## Problem\n"
    "Current methods use standard techniques to tune hyperparameters.\n"
    "\n"
    "## Methodology\n"
    "- Apply various baselines to our task.\n"
    "- The approach is suitable.\n"
    "\n"
    "## Discussion\n"
    "We observe that results vary.\n"
)

FULL_PARAGRAPH = "Current methods use standard techniques to tune hyperparameters.\n"
FULL_BULLET_1 = "- Apply various baselines to our task.\n"
FULL_BULLET_2 = "- The approach is suitable.\n"


def test_apply_single_revision_ok():
    revisions = [
        {"quote_original": FULL_PARAGRAPH,
         "quote_replacement": "Current methods use Adam with lr=3e-4.\n"},
    ]
    new_plan, success, status = apply_locus_revisions(PLAN, revisions)
    assert success is True
    assert status == "ok"
    assert "Adam with lr=3e-4" in new_plan
    assert "standard techniques" not in new_plan


def test_apply_multiple_revisions_ok():
    revisions = [
        {"quote_original": FULL_PARAGRAPH,
         "quote_replacement": "Current methods use Adam with lr=3e-4.\n"},
        {"quote_original": FULL_BULLET_1,
         "quote_replacement": "- Apply GPT-4o and LLaMA-3-8B baselines.\n"},
    ]
    new_plan, success, status = apply_locus_revisions(PLAN, revisions)
    assert success is True
    assert status == "ok"
    assert "Adam with lr=3e-4" in new_plan
    assert "GPT-4o and LLaMA-3-8B" in new_plan


def test_apply_remove_revision():
    """quote_replacement="" means REMOVE (bullet line is a complete block)."""
    revisions = [{"quote_original": FULL_BULLET_2, "quote_replacement": ""}]
    new_plan, success, status = apply_locus_revisions(PLAN, revisions)
    assert success is True
    assert "suitable" not in new_plan


def test_apply_expand_revision():
    """EXPAND = keep original content + add new content, within a full block."""
    expanded = (
        "Current methods use standard techniques to tune hyperparameters, "
        "specifically Adam with cosine warmup.\n"
    )
    revisions = [{"quote_original": FULL_PARAGRAPH, "quote_replacement": expanded}]
    new_plan, success, status = apply_locus_revisions(PLAN, revisions)
    assert success is True
    assert "cosine warmup" in new_plan


def test_apply_returns_original_on_no_match():
    revisions = [{"quote_original": "text that does not exist", "quote_replacement": "replacement"}]
    new_plan, success, status = apply_locus_revisions(PLAN, revisions)
    assert success is False
    assert status == "no_match"
    assert new_plan == PLAN


def test_apply_returns_original_on_ambiguous():
    """Plan with duplicated text → ambiguous match (fires before partial_span)."""
    dup_plan = "use standard. use standard."
    revisions = [{"quote_original": "standard", "quote_replacement": "Adam"}]
    new_plan, success, status = apply_locus_revisions(dup_plan, revisions)
    assert success is False
    assert status == "ambiguous"
    assert new_plan == dup_plan


def test_apply_returns_original_on_empty_revisions():
    new_plan, success, status = apply_locus_revisions(PLAN, [])
    assert success is False
    assert status == "empty_revisions"
    assert new_plan == PLAN


def test_apply_returns_original_on_parse_error_missing_key():
    revisions = [{"wrong_key": "value"}]
    new_plan, success, status = apply_locus_revisions(PLAN, revisions)
    assert success is False
    assert status == "parse_error"


def test_apply_returns_original_on_parse_error_non_dict():
    revisions = ["not a dict"]
    new_plan, success, status = apply_locus_revisions(PLAN, revisions)
    assert success is False
    assert status == "parse_error"


def test_apply_tolerates_none_replacement_as_empty():
    """quote_replacement=None should be treated as "" (REMOVE) for a full block."""
    revisions = [{"quote_original": FULL_BULLET_2, "quote_replacement": None}]
    new_plan, success, status = apply_locus_revisions(PLAN, revisions)
    assert success is True
    assert "suitable" not in new_plan


def test_apply_first_failure_aborts_all():
    """If second revision fails (no_match), first one is also rolled back."""
    revisions = [
        {"quote_original": FULL_PARAGRAPH,
         "quote_replacement": "Current methods use Adam.\n"},
        {"quote_original": "nonexistent text", "quote_replacement": "X"},
    ]
    new_plan, success, status = apply_locus_revisions(PLAN, revisions)
    assert success is False
    assert status == "no_match"
    assert new_plan == PLAN  # Original returned, first revision NOT kept


# =============================================================================
# Partial-span rejection (stitching bug fix, 2026-04-17)
# =============================================================================

def test_complete_section_quote_applied():
    """A quote spanning a full ## section (header + body) is accepted."""
    full_section = (
        "## Methodology\n"
        "- Apply various baselines to our task.\n"
        "- The approach is suitable.\n"
    )
    replacement = (
        "## Methodology\n"
        "- Train with entropic objective and ES-grad hybrid.\n"
        "- Measure buffer_max vs reference.\n"
    )
    revisions = [{"quote_original": full_section, "quote_replacement": replacement}]
    new_plan, success, status = apply_locus_revisions(PLAN, revisions)
    assert success is True
    assert status == "ok"
    assert "entropic objective" in new_plan
    assert "Apply various baselines" not in new_plan
    assert "approach is suitable" not in new_plan


def test_partial_lines_rejected_as_partial_span():
    """First 2 of 3 body lines under a header → partial_span (stitching bug repro).

    Under the old code this would replace the first 2 lines but leave
    "Line 3 of approach." intact, creating duplicated content when the
    replacement text re-generates all three lines.
    """
    multi_line_plan = (
        "## Methodology\n"
        "Line 1 of approach.\n"
        "Line 2 of approach.\n"
        "Line 3 of approach.\n"
        "\n"
        "## Next\n"
    )
    partial = "Line 1 of approach.\nLine 2 of approach.\n"
    revisions = [{"quote_original": partial, "quote_replacement": "Rewritten.\n"}]
    new_plan, success, status = apply_locus_revisions(multi_line_plan, revisions)
    assert success is False
    assert status == "partial_span"
    assert new_plan == multi_line_plan


def test_mid_paragraph_substring_rejected_as_partial_span():
    """A mid-paragraph substring ('standard techniques') is rejected.

    Primary bug repro: policy emits a SUBSTRING of a paragraph as
    quote_original. Under old semantics this replaced the phrase but
    left the surrounding sentence fragments orphaned.
    """
    revisions = [{"quote_original": "standard techniques", "quote_replacement": "Adam"}]
    new_plan, success, status = apply_locus_revisions(PLAN, revisions)
    assert success is False
    assert status == "partial_span"
    assert new_plan == PLAN


def test_full_bullet_item_applied():
    """A single full bullet line (``- ... \\n``) is a complete block."""
    revisions = [
        {"quote_original": FULL_BULLET_1,
         "quote_replacement": "- Run CR-v6 minimal mode for 25 iterations.\n"},
    ]
    new_plan, success, status = apply_locus_revisions(PLAN, revisions)
    assert success is True
    assert status == "ok"
    assert "CR-v6 minimal mode" in new_plan


# =============================================================================
# _spans_complete_block unit tests (helper used by apply_locus_revisions)
# =============================================================================

def test_spans_complete_block_full_paragraph_after_header():
    assert _spans_complete_block(PLAN, FULL_PARAGRAPH) is True


def test_spans_complete_block_full_bullet():
    assert _spans_complete_block(PLAN, FULL_BULLET_1) is True


def test_spans_complete_block_mid_paragraph_is_false():
    assert _spans_complete_block(PLAN, "standard techniques") is False


def test_spans_complete_block_line_without_trailing_newline_is_false():
    """Quote without a trailing newline that isn't at EOF is mid-line."""
    quote = "Current methods use standard techniques to tune hyperparameters."
    assert _spans_complete_block(PLAN, quote) is False


def test_spans_complete_block_empty_quote_is_false():
    assert _spans_complete_block(PLAN, "") is False


def test_spans_complete_block_not_in_plan_is_false():
    assert _spans_complete_block(PLAN, "nonexistent text") is False


def test_spans_complete_block_full_section_with_header():
    section = (
        "## Methodology\n"
        "- Apply various baselines to our task.\n"
        "- The approach is suitable.\n"
    )
    assert _spans_complete_block(PLAN, section) is True


def test_spans_complete_block_eof_line_without_trailing_newline():
    """Last line of a plan that lacks a trailing \\n still counts as EOF-bounded."""
    tail_plan = "## Heading\nSingle line."
    assert _spans_complete_block(tail_plan, "Single line.") is True


def test_spans_complete_block_numbered_list_item():
    num_plan = (
        "## Steps\n"
        "1. First step.\n"
        "2. Second step.\n"
    )
    assert _spans_complete_block(num_plan, "1. First step.\n") is True


# =============================================================================
# parse_locus_revisions
# =============================================================================

def test_parse_revisions_happy_path():
    text = """
<revisions>
[{"quote_original": "foo", "quote_replacement": "bar"}]
</revisions>
"""
    result = parse_locus_revisions(text)
    assert result is not None
    assert len(result) == 1
    assert result[0]["quote_original"] == "foo"


def test_parse_revisions_multiple():
    text = '<revisions>[{"quote_original": "a", "quote_replacement": "b"}, {"quote_original": "c", "quote_replacement": "d"}]</revisions>'
    result = parse_locus_revisions(text)
    assert result is not None
    assert len(result) == 2


def test_parse_revisions_none_on_missing_tag():
    assert parse_locus_revisions("no revisions here") is None


def test_parse_revisions_none_on_malformed_json():
    assert parse_locus_revisions("<revisions>[{bad json]</revisions>") is None


def test_parse_revisions_none_on_non_list():
    assert parse_locus_revisions('<revisions>{"not": "a list"}</revisions>') is None


def test_parse_revisions_case_insensitive():
    text = '<REVISIONS>[{"quote_original": "x", "quote_replacement": "y"}]</REVISIONS>'
    result = parse_locus_revisions(text)
    assert result is not None
    assert len(result) == 1


def test_parse_revisions_tolerates_surrounding_text():
    """Policy output may have <think> blocks, critique, etc. around <revisions>."""
    text = """
<think>I need to fix the vague language.</think>
The main issue is...
<revisions>
[{"quote_original": "standard techniques", "quote_replacement": "Adam optimizer"}]
</revisions>
Done.
"""
    result = parse_locus_revisions(text)
    assert result is not None
    assert len(result) == 1


# =============================================================================
# build_locus_revision_prompt
# =============================================================================

def test_build_locus_revision_prompt_contains_key_elements():
    loci = [
        LocusEntry(section_hint="Problem", quote="standard techniques", why="Vague marker."),
    ]
    prompt = build_locus_revision_prompt(
        goal="Discover X",
        plan_text=PLAN,
        signal_vector={"S7_specificity": 2, "S1_depth": 4},
        bottleneck_id="S7_specificity",
        bottleneck_score=2,
        loci=loci,
    )
    # Contains the plan
    assert "standard techniques" in prompt
    # Contains the locus
    assert "Vague marker." in prompt
    # Contains the signal info
    assert "Implementation Specificity" in prompt
    assert "score: 2/5" in prompt
    # Contains revision format instructions
    assert "<revisions>" in prompt
    assert "quote_original" in prompt
    assert "REWRITE" in prompt
    assert "EXPAND" in prompt
    assert "REMOVE" in prompt


def test_build_locus_revision_prompt_multiple_loci():
    loci = [
        LocusEntry(section_hint="Problem", quote="standard techniques", why="Vague."),
        LocusEntry(section_hint="Methodology", quote="various baselines", why="Unnamed."),
    ]
    prompt = build_locus_revision_prompt(
        goal="X", plan_text=PLAN,
        signal_vector={"S7_specificity": 2},
        bottleneck_id="S7_specificity", bottleneck_score=2, loci=loci,
    )
    assert "1." in prompt and "2." in prompt
    assert "standard techniques" in prompt
    assert "various baselines" in prompt
