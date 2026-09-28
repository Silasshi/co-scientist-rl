"""Tests for D6 prompt loader and prompt content invariants.

Two layers:
1. Loader mechanics — `_strip_doc_header`, caching, missing-file errors.
2. Content shape — every production prompt loads, has the variables the
   callers pass, and matches the structural invariants documented in
   DECISIONS.md (length / 8-heading template / 8-child critique schema /
   12-dim pairwise schema). These are deliberately structural, not
   byte-equal — the .md files are now the source of truth and content
   edits should be possible without rewriting test fixtures.

Reference distribution facts (locked in for regression detection):
- 15 reference proposals across D4 and D6, 1318–2038 words (mean ~1639).
- All 15 use the 8-heading template:
    Specific Aims / Research Strategy → Significance / Approach /
    Evaluation Plan / Timeline / Deliverables / Risk Mitigation.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pytest

from co_scientist.d6_grant_proposal.prompt_loader import (
    PROMPTS_ROOT,
    _strip_doc_header,
    load_prompt,
    render_prompt,
    reload_prompts,
)
from co_scientist.d6_grant_proposal.prompts import (
    COLD_START_CRITIQUE_NONE,
    build_critic_request_payload,
    build_student_prompt,
    build_teacher_prompt,
)


REFERENCE_HEADINGS = [
    "## Specific Aims",
    "## Research Strategy",
    "### Significance",
    "### Approach",
    "### Evaluation Plan",
    "### Timeline",
    "### Deliverables",
    "### Risk Mitigation",
]

CRITIQUE_CHILDREN = [
    "<idea_alignment>",
    "<missing_components>",
    "<incorrect_assumptions>",
    "<feasibility>",
    "<clarity>",
    "<strengths>",
    "<weaknesses>",
    "<improvement_directive>",
]


# ---------------------------------------------------------------------------
# Loader mechanics
# ---------------------------------------------------------------------------

def test_strip_doc_header_strips_until_first_rule():
    raw = (
        "<!-- meta -->\n"
        "\n"
        "# Title\n"
        "\n"
        "**Role**: foo\n"
        "\n"
        "---\n"
        "\n"
        "actual body line 1\n"
        "actual body line 2\n"
    )
    assert _strip_doc_header(raw) == "actual body line 1\nactual body line 2"


def test_strip_doc_header_no_rule_returns_raw_minus_trailing_newlines():
    assert _strip_doc_header("body\n\n") == "body"
    assert _strip_doc_header("body without trailing newline") == "body without trailing newline"


def test_strip_doc_header_does_not_strip_leading_blank_lines_inside_body():
    raw = "---\n\nbody starts here\n\nstill body\n"
    assert _strip_doc_header(raw) == "body starts here\n\nstill body"


def test_load_prompt_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        reload_prompts()
        load_prompt("does/not/exist/__never__.md")


def test_reload_prompts_invalidates_cache(tmp_path, monkeypatch):
    """Loader's lru_cache must be invalidatable when the file on disk changes."""
    import co_scientist.d6_grant_proposal.prompt_loader as pl
    monkeypatch.setattr(pl, "PROMPTS_ROOT", tmp_path)
    test_path = tmp_path / "test.md"
    test_path.write_text("---\nv1\n")
    pl.reload_prompts()
    assert pl.load_prompt("test.md") == "v1"
    test_path.write_text("---\nv2\n")
    pl.reload_prompts()
    assert pl.load_prompt("test.md") == "v2"


# ---------------------------------------------------------------------------
# Folder layout
# ---------------------------------------------------------------------------

def test_prompts_root_exists_and_has_role_subfolders():
    assert PROMPTS_ROOT.is_dir()
    for sub in ["generation", "review", "audit", "pairwise", "oracle"]:
        assert (PROMPTS_ROOT / sub).is_dir(), f"missing {sub}/"


def test_no_top_level_prompts_files_outside_subfolders():
    """All .md files (other than README) must live under a role subfolder."""
    top_level_md = sorted(p.name for p in PROMPTS_ROOT.glob("*.md"))
    assert top_level_md == ["README.md"], f"unexpected top-level .md: {top_level_md}"


def test_no_legacy_directory():
    """legacy/ was removed in Phase 0 cleanup (2026-04-30); D4 archives are dead weight."""
    assert not (PROMPTS_ROOT / "legacy").exists()


# ---------------------------------------------------------------------------
# Generation prompts (student / teacher)
# ---------------------------------------------------------------------------

def test_student_prompt_renders_all_substitutions():
    rendered = build_student_prompt("MY_GOAL", "MY_ORACLE", external_knowledge="MY_EXT")
    assert "MY_GOAL" in rendered
    assert "MY_ORACLE" in rendered
    assert "MY_EXT" in rendered
    assert "<solution>" in rendered


def test_student_prompt_default_external_knowledge_placeholder():
    rendered = build_student_prompt("G", "O")
    assert "(none provided for this run)" in rendered


def test_student_prompt_uses_8_heading_reference_template():
    rendered = build_student_prompt("G", "O")
    for h in REFERENCE_HEADINGS:
        assert h in rendered, f"missing reference heading: {h}"


def test_student_prompt_length_target_matches_reference_distribution():
    rendered = build_student_prompt("G", "O")
    assert "1500" in rendered
    assert "2000" in rendered
    # Negative assertions — old 900/1100 must NOT be present.
    assert "900 words" not in rendered
    assert "1100 words" not in rendered


def test_teacher_prompt_includes_critique_xml():
    rendered = build_teacher_prompt("G", "O", "<critique>STUFF</critique>", external_knowledge="EXT")
    assert "STUFF" in rendered
    assert "EXT" in rendered
    assert "Critique of prior attempt" in rendered


def test_teacher_prompt_same_8_headings_as_student():
    rendered = build_teacher_prompt("G", "O", "<critique/>", external_knowledge="")
    for h in REFERENCE_HEADINGS:
        assert h in rendered


def test_student_prompt_handles_curly_braces_in_user_input():
    """Substituted values must not be re-interpreted by str.format."""
    weird = "Goal with {literal braces} in it."
    rendered = build_student_prompt(weird, "Oracle with {more braces}.")
    assert weird in rendered


# ---------------------------------------------------------------------------
# Cold-start critique (8-child schema)
# ---------------------------------------------------------------------------

def test_cold_start_critique_has_eight_children():
    for child in CRITIQUE_CHILDREN:
        assert child in COLD_START_CRITIQUE_NONE, f"missing {child}"
    assert COLD_START_CRITIQUE_NONE.startswith("<critique>")
    assert COLD_START_CRITIQUE_NONE.endswith("</critique>")


# ---------------------------------------------------------------------------
# Review (critic) instructions
# ---------------------------------------------------------------------------

def test_critic_instructions_mentions_eight_children():
    payload = build_critic_request_payload(
        iter_idx=0,
        goal="G",
        plan_text="plan",
        reference_proposal_md="ref",
        plan_id="p_0",
    )
    instructions = payload["instructions"]
    for child in CRITIQUE_CHILDREN:
        assert child in instructions, f"critic instructions missing {child}"


def test_critic_instructions_keep_privileged_observation_rule():
    instructions = load_prompt("review/critic_instructions.md")
    assert "DO NOT mention the reference proposal" in instructions
    assert "<critique>" in instructions and "</critique>" in instructions


def test_critic_payload_structural_fields_unchanged():
    payload = build_critic_request_payload(
        iter_idx=3,
        goal="G",
        plan_text="plan",
        reference_proposal_md="ref",
        plan_id="p_3",
    )
    assert payload["kind"] == "critic"
    assert payload["iter"] == 3
    assert payload["source_paper_md"] == "ref"
    assert payload["plans"] == [{"plan_id": "p_3", "text": "plan"}]


# ---------------------------------------------------------------------------
# Audit (multi-plan absolute scoring)
# ---------------------------------------------------------------------------

def test_audit_prompt_includes_active_signals_and_xml_schema():
    from co_scientist.d6_grant_proposal.audit import _build_audit_prompt
    from co_scientist.d6_grant_proposal.signals import SIGNALS

    plans = [{"anon_id": "A", "text": "..."}, {"anon_id": "B", "text": "..."}]
    rendered = _build_audit_prompt("MY_GOAL", plans, domain="ai")

    assert "MY_GOAL" in rendered
    for s in SIGNALS:
        assert s.id in rendered, f"missing signal id: {s.id}"
    assert "<plan id=" in rendered
    assert "<aggregate_reward>" in rendered
    assert "<key_differentiator>" in rendered


def test_audit_prompt_swaps_g12_for_social_science():
    from co_scientist.d6_grant_proposal.audit import _build_audit_prompt

    plans = [{"anon_id": "A", "text": "..."}]
    soc = _build_audit_prompt("G", plans, domain="social_science")
    assert "G12a_analytical_framework" in soc
    assert "G12_formalism" not in soc


# ---------------------------------------------------------------------------
# Pairwise (head-to-head)
# ---------------------------------------------------------------------------

def test_pairwise_prompt_renders_with_dim_lines():
    from co_scientist.d6_grant_proposal.signals import SIGNALS

    dim_lines = "".join(
        f'  <dim id="{s.id}">A | B | TIE — 1-sentence reason</dim>\n'
        for s in SIGNALS
    )
    rendered = render_prompt(
        "pairwise/head_to_head.md",
        goal="MY_GOAL",
        plan_a_text="PROPOSAL_A_BODY",
        plan_b_text="PROPOSAL_B_BODY",
        reference_proposal="REF_PROP_BODY",
        dim_lines=dim_lines,
    )
    assert "MY_GOAL" in rendered
    assert "PROPOSAL_A_BODY" in rendered
    assert "PROPOSAL_B_BODY" in rendered
    assert "REF_PROP_BODY" in rendered
    for s in SIGNALS:
        assert s.id in rendered, f"pairwise missing signal: {s.id}"
    assert "<comparison>" in rendered and "</comparison>" in rendered
    assert "<verdict>" in rendered
    assert "<margin>" in rendered
    assert "<justification>" in rendered


def test_pairwise_prompt_handles_no_reference_variant():
    """For the no-privileged-reference variant, the caller passes an explicit placeholder."""
    rendered = render_prompt(
        "pairwise/head_to_head.md",
        goal="G",
        plan_a_text="A",
        plan_b_text="B",
        reference_proposal="(no reference proposal available; judge on internal merit only)",
        dim_lines="  <dim id=\"X\">A | B | TIE — reason</dim>\n",
    )
    assert "(no reference proposal available" in rendered


# ---------------------------------------------------------------------------
# Oracle extraction
# ---------------------------------------------------------------------------

def test_oracle_extract_prompt_has_six_categories_and_placeholders():
    from co_scientist.d6_grant_proposal.extract_oracle import _EXTRACT_PROMPT
    for category in ["Insights", "Methodology", "Evidence", "Structure", "Deliverables", "Risk"]:
        assert f"**{category}**" in _EXTRACT_PROMPT
    # `_EXTRACT_PROMPT` is the raw template (placeholders intact); caller does .format(...).
    assert "{goal}" in _EXTRACT_PROMPT
    assert "{reference_proposal}" in _EXTRACT_PROMPT
    assert "<item>" in _EXTRACT_PROMPT
    assert "<category>" in _EXTRACT_PROMPT
