"""D5 v2 prompt builders — re-exports v1 + α-v2 regen prompt + audit-v3 wrapper.

The v2 trainers all consume `oracle_v2.md` (not the old smoke-v3 oracle), but
the prompt SHAPES are identical to v1. So we re-export build_student_prompt /
build_teacher_prompt / cold-start critique / sdpo datum / extract_solution /
audit-payload from v1 unchanged.

What's new in v2:
- `build_alpha_v2_distill_prompt(goal, oracle)`: prompt fed to Opus when
  generating the α-v2 distillation training data — Opus sees goal + oracle
  (instead of goal-only) so the distillation target is conditioned on the
  same scaffolding the student/eval pipeline uses.
- `build_audit_v3_request_payload(...)`: like v1 audit payload but flagged
  with `prompt_version: "v3"` so the daemon dispatches v3 audit
"""
from __future__ import annotations

from typing import Any

# Re-export everything from v1 unchanged; v2 just swaps oracle PATH at the call site
from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v1 import (  # noqa: F401
    COLD_START_CRITIQUE_NONE,
    build_audit_request_payload,
    build_critic_request_payload,
    build_sdpo_datum,
    build_student_prompt,
    build_teacher_prompt,
    extract_solution,
    parse_critique_xml,
)


_PLAN_FOOTER = (
    "\n\nWrite your research plan inside <solution>...</solution> tags."
    " Target 600 words, max 750 words. Use structured sections (Problem"
    " Statement, Background, Hypothesis, Methodology, Evaluation, Limitations)."
)


def build_alpha_v2_distill_prompt(goal: str, oracle: str) -> str:
    """Prompt fed to Opus 4.7 when regenerating α-v2 distillation training data.

    Conditions Opus on goal + oracle_v2, matching what the eval-time student
    will see. Removes the "prompt-mismatch confound" of α-v1 (which had Opus
    write plans from goal-only but eval the student under goal+oracle).
    """
    return (
        "You are an expert ML/AI research methodologist. Write a complete"
        " research plan for the scenario below, drawing on the methodological"
        " patterns provided. Your plan will be used as a distillation target"
        " for a smaller open-weight model, so be specific: name mechanisms,"
        " include equations with full RHS where relevant, and cite concrete"
        " numerical baselines for any claims about prior work."
        f"\n\nScenario: {goal}"
        f"\n\n# Methodological patterns to apply\n\n{oracle}"
        + _PLAN_FOOTER
    )


def build_audit_v3_request_payload(
    *,
    iter_idx: int,
    goal: str,
    plans: list[dict[str, str]],
) -> dict[str, Any]:
    """v3-flagged audit request. Daemon should branch on `prompt_version` to
    select DEPTH_AUDIT_PROMPT_V3 (9-dim hybrid) instead of v2 (D3 canonical).
    """
    return {
        "kind": "audit",
        "iter": iter_idx,
        "goal": goal,
        "plans": plans,
        "prompt_version": "v3",
    }
