"""D5 μ-v8 prompt builders — 900-word footer (matches reference plan length).

Pure helper module, fork of `mu_prompts_v1.py`. Footer changed from
"target 600, max 750 words" to "target 900, max 1100 words" — verified against
`dataset/reference_solution.txt` which is 962 words. σ baseline at v1 footer
generated mean 618 words (footer-cap-bound at 750), so the v1 footer was
suppressing length below reference. v8 footer lets policy generate at
reference-length scale.

Used by `train_mu_v8_d5sdpo.py`.

Reuses unchanged from v1:
- COLD_START_CRITIQUE_NONE
- parse_critique_xml
- build_critic_request_payload
- build_audit_request_payload
- build_sdpo_datum
- extract_solution
- _SOLUTION_RE, _HEADING_RE, _THINK_RE

Diverges from v1 ONLY in the prompt footer + builder functions that use it.
This preserves v7-opd / v4 / v6 / kappa-v1 trainer comparability — those
trainers continue to import from `mu_prompts_v1`.
"""
from __future__ import annotations


from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v1 import (  # noqa: F401
    COLD_START_CRITIQUE_NONE,
    _HEADING_RE,
    _SOLUTION_RE,
    _THINK_RE,
    build_audit_request_payload,
    build_critic_request_payload,
    build_sdpo_datum,
    extract_solution,
    parse_critique_xml,
)


_PLAN_FOOTER_V8 = (
    "\n\nWrite your research plan inside <solution>...</solution> tags."
    " Target 900 words, max 1100 words. Use structured sections (Problem"
    " Statement, Background, Hypothesis, Methodology, Evaluation, Limitations)."
)


def build_student_prompt_v8(goal: str, oracle_abstraction: str) -> str:
    """STUDENT context for v8: goal + oracle abstraction, NO critique.

    Same body phrasing as v1 build_student_prompt; only footer changed
    (900 / 1100 instead of 600 / 750).
    """
    return (
        "I will provide you a research scenario and a set of methodological"
        " patterns extracted from a high-quality research plan on this same"
        " scenario. Use the patterns to guide the structure and reasoning of"
        " your plan, but do not copy the phrasing literally."
        f"\n\nScenario: {goal}"
        f"\n\n# Methodological patterns to apply\n\n{oracle_abstraction}"
        + _PLAN_FOOTER_V8
    )


def build_teacher_prompt_v8(goal: str, oracle_abstraction: str, critique_xml: str) -> str:
    """TEACHER context for v8: goal + oracle + critique on CURRENT rollout.

    Important: in v8, this critique is the critic's response to the CURRENT
    iter's student rollout (paper-faithful per Hübotter 2601.20802 Algorithm 1
    `f = environment(x, y)`), NOT the previous iter's critique as in v7-opd.
    The trainer is responsible for ordering the critic call BEFORE this
    teacher_lp compute, not after training.

    Same body phrasing as v1 build_teacher_prompt; only footer changed.
    """
    return (
        "I will provide you a research scenario, a set of methodological"
        " patterns extracted from a high-quality research plan on this same"
        " scenario, and a structured critique of a prior plan attempt. Use the"
        " patterns to guide structure and the critique's improvement_directive"
        " to address concrete weaknesses. Do not copy phrasing literally."
        f"\n\nScenario: {goal}"
        f"\n\n# Methodological patterns to apply\n\n{oracle_abstraction}"
        f"\n\n# Critique of prior attempt (to address)\n\n{critique_xml}"
        + _PLAN_FOOTER_V8
    )
