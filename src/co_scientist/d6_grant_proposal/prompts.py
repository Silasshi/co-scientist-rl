"""D6 prompt builders — grant proposal ERR pipeline.

All prompt text is loaded from `projects/d6_grant_proposal/prompts/` via
`prompt_loader`. The functions here are thin wrappers that pass kwargs
to the loader; public signatures are stable so callers don't change.

- Policy prompts: `generation/student.md`, `generation/teacher.md`
- Cold-start critique placeholder: `generation/cold_start_critique.md`
- Reviewer instructions: `review/critic_instructions.md`
- Audit metadata payload: structural only (no prompt text — the audit
  prompt itself lives in `audit/absolute_score.md` and is rendered by
  `audit._build_audit_prompt`).

`extract_solution` and `parse_critique_xml` are re-exported from D5
(identical implementation).
"""
from __future__ import annotations

from typing import Any

from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v1 import (  # noqa: F401
    parse_critique_xml,
    build_sdpo_datum,
    extract_solution,
)
from co_scientist.d6_grant_proposal.prompt_loader import load_prompt, render_prompt


# =============================================================================
# Grant proposal generation prompts
# =============================================================================

_NO_EXTERNAL = "(none provided for this run)"


def build_student_prompt(
    goal: str,
    oracle_abstraction: str,
    external_knowledge: str = "",
) -> str:
    """STUDENT context: goal + oracle abstraction (+ optional external evidence), NO critique."""
    return render_prompt(
        "generation/student.md",
        goal=goal,
        oracle_abstraction=oracle_abstraction,
        external_knowledge=external_knowledge or _NO_EXTERNAL,
    )


def build_teacher_prompt(
    goal: str,
    oracle_abstraction: str,
    critique_xml: str,
    external_knowledge: str = "",
) -> str:
    """TEACHER context: goal + oracle (+ optional external evidence) + previous-iter critique."""
    return render_prompt(
        "generation/teacher.md",
        goal=goal,
        oracle_abstraction=oracle_abstraction,
        external_knowledge=external_knowledge or _NO_EXTERNAL,
        critique_xml=critique_xml,
    )


# =============================================================================
# Cold-start critique (iter 0) — grant proposal framing
# =============================================================================

COLD_START_CRITIQUE_NONE = load_prompt("generation/cold_start_critique.md")


# =============================================================================
# Subagent request payloads
# =============================================================================

def build_critic_request_payload(
    *,
    iter_idx: int,
    goal: str,
    plan_text: str,
    reference_proposal_md: str,
    plan_id: str,
) -> dict[str, Any]:
    """JSON request body for plan-critic subagent.

    The subagent sees the reference proposal as privileged info. The reference
    proposal plays the same role as the source paper in D5.
    """
    return {
        "kind": "critic",
        "iter": iter_idx,
        "goal": goal,
        "source_paper_md": reference_proposal_md,
        "plans": [{"plan_id": plan_id, "text": plan_text}],
        "instructions": load_prompt("review/critic_instructions.md"),
    }


def build_audit_request_payload(
    *,
    iter_idx: int,
    goal: str,
    plans: list[dict[str, str]],
    goal_id: str = "",
) -> dict[str, Any]:
    """JSON request body for D6 12-signal audit subagent.

    Uses D4's 12-signal rubric. NO reference proposal here — audit must
    remain independent of the privileged reviewer signal.
    """
    return {
        "kind": "audit",
        "iter": iter_idx,
        "goal": goal,
        "goal_id": goal_id,
        "plans": plans,
        "prompt_version": "d6_v1",
    }
