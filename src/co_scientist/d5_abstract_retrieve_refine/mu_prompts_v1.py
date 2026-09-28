"""D5 μ baseline — prompt builders, cold-start critique, datum helper.

Pure helper module. No I/O, no Tinker, no Anthropic.

Used by `train_mu_baseline_v1.py`:
    - build_student_prompt: no-critique context (the SDPO student)
    - build_teacher_prompt: with-critique context (the SDPO teacher)
    - COLD_START_CRITIQUE_NONE: handcrafted neutral critique for iter 0
    - parse_critique_xml: extract <critique> block from raw Opus output
    - build_critic_request_payload: JSON for plan-critic subagent
    - build_audit_request_payload: JSON for D3-canonical audit subagent
    - build_sdpo_datum: Tinker Datum constructor for plan-level SDPO
"""
from __future__ import annotations

import re
from typing import Any

import torch
import tinker
from tinker import types
from tinker.types import TensorData


# =============================================================================
# Plan generation prompts
# =============================================================================

_PLAN_FOOTER = (
    "\n\nWrite your research plan inside <solution>...</solution> tags."
    " Target 600 words, max 750 words. Use structured sections (Problem"
    " Statement, Background, Hypothesis, Methodology, Evaluation, Limitations)."
)


def build_student_prompt(goal: str, oracle_abstraction: str) -> str:
    """STUDENT context: goal + oracle abstraction, NO critique.

    Identical phrasing to smoke_pathway_v1.build_prompt_with_abstraction so
    that iter-0 student-context audit matches the smoke v3-rerun A baseline.
    """
    return (
        "I will provide you a research scenario and a set of methodological"
        " patterns extracted from a high-quality research plan on this same"
        " scenario. Use the patterns to guide the structure and reasoning of"
        " your plan, but do not copy the phrasing literally."
        f"\n\nScenario: {goal}"
        f"\n\n# Methodological patterns to apply\n\n{oracle_abstraction}"
        + _PLAN_FOOTER
    )


def build_teacher_prompt(goal: str, oracle_abstraction: str, critique_xml: str) -> str:
    """TEACHER context: goal + oracle abstraction + previous-iter critique.

    The critique is wrapped in its <critique>...</critique> block as written
    by the plan-critic subagent. Iter 0 uses COLD_START_CRITIQUE_NONE so
    structural shape is identical to iter N>0 prompts.
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
        + _PLAN_FOOTER
    )


# =============================================================================
# Cold-start critique (iter 0)
# =============================================================================

# Handcrafted neutral critique. Same XML shape as a real Opus response, so the
# iter-0 teacher prompt is structurally identical to iter-N>0 teacher prompts.
# Content: structural directives only — NO leakage of TTT-Discover specifics
# (no J_β, no MAX-PUCT, no Erdős/AHC039 numbers).
COLD_START_CRITIQUE_NONE = """\
<critique>
<idea_alignment>This is the first iteration; no prior plan exists to align against. Focus on producing a well-derived plan that addresses the methodological patterns provided.</idea_alignment>
<missing_components>To be evaluated against the source paper after generation. Ensure each pattern's derivation steps are followed, not skipped.</missing_components>
<incorrect_assumptions>To be evaluated against the source paper after generation. State assumptions explicitly so they can be checked.</incorrect_assumptions>
<feasibility>State a concrete compute budget, a concrete target hardware, and a concrete evaluation-baseline number even if estimated. Avoid hand-waving on numerical claims.</feasibility>
<improvement_directive>Derive each pattern's training/search objective from first principles (do not skip the derivation Steps). Name a concrete compute budget. State 1-3 specific evaluation baselines with prior numerical results. Name 3-5 mechanism-level limitations (not implementation limitations). Avoid generic phrases like "we will use standard reinforcement learning" — specify the loss form, the search procedure, and the stability mechanism.</improvement_directive>
</critique>"""


# =============================================================================
# Subagent request payloads
# =============================================================================

def build_critic_request_payload(
    *,
    iter_idx: int,
    goal: str,
    plan_text: str,
    source_paper_md: str,
    plan_id: str,
) -> dict[str, Any]:
    """JSON request body for plan-critic subagent (privileged-info reviewer).

    The subagent sees the full source paper as privileged info. Output is a
    single <critique>...</critique> block per plan.
    """
    return {
        "kind": "critic",
        "iter": iter_idx,
        "goal": goal,
        "source_paper_md": source_paper_md,
        "plans": [{"plan_id": plan_id, "text": plan_text}],
        "instructions": (
            "You are an expert research-methodology reviewer. You have"
            " privileged access to the source paper. Use it as ground truth"
            " when judging idea alignment, missing components, incorrect"
            " assumptions, and feasibility. DO NOT mention the source paper"
            " by name, by author, or by arxiv ID in your output. Write a"
            " structured <critique> block with the following children:"
            " <idea_alignment>, <missing_components>, <incorrect_assumptions>,"
            " <feasibility>, <improvement_directive>. Each is 1-3 sentences"
            " except improvement_directive (≤120 words). Wrap the entire"
            " response inside <critique>...</critique>. Output nothing else."
        ),
    }


def build_audit_request_payload(
    *,
    iter_idx: int,
    goal: str,
    plans: list[dict[str, str]],
) -> dict[str, Any]:
    """JSON request body for D3-canonical depth audit subagent.

    NO source paper here — audit is the ground-truth metric and must remain
    independent of paper-leakage signal in the critique loop.
    """
    return {
        "kind": "audit",
        "iter": iter_idx,
        "goal": goal,
        "plans": plans,  # list of {"plan_id": str, "text": str}
    }


# =============================================================================
# Critique parsing
# =============================================================================

_CRITIQUE_RE = re.compile(r"<critique>.*?</critique>", re.DOTALL)


def parse_critique_xml(raw: str) -> str:
    """Extract first <critique>...</critique> block from raw Opus text.

    Returns the matched block including outer tags. Falls back to
    COLD_START_CRITIQUE_NONE if no block is found (so training continues).
    """
    if not raw:
        return COLD_START_CRITIQUE_NONE
    m = _CRITIQUE_RE.search(raw)
    if not m:
        return COLD_START_CRITIQUE_NONE
    return m.group(0).strip()


# =============================================================================
# Tinker datum builder for plan-level SDPO
# =============================================================================

def build_sdpo_datum(
    *,
    student_prompt_tokens: list[int],
    gen_tokens: list[int],
    student_logprobs: list[float],
    advantages: list[float],
) -> types.Datum:
    """Build a Tinker Datum for plan-level SDPO via importance_sampling loss.

    Layout (matches train_cr_v7.py:1184-1196):
        full_seq         = student_prompt_tokens ++ gen_tokens
        model_input      = full_seq[:-1]                    # length L-1
        target_tokens    = full_seq[1:]                     # length L-1
        logprobs         = [0.0]·(S-1) ++ student_logprobs  # stored=student → IS ratio = 1 at iter 0
        advantages       = [0.0]·(S-1) ++ A_t               # gen-only

    where S = len(student_prompt_tokens), L = S + len(gen_tokens),
    and len(student_logprobs) == len(advantages) == len(gen_tokens).
    """
    assert len(student_logprobs) == len(gen_tokens), (
        f"student_logprobs len {len(student_logprobs)} != gen_tokens len {len(gen_tokens)}"
    )
    assert len(advantages) == len(gen_tokens), (
        f"advantages len {len(advantages)} != gen_tokens len {len(gen_tokens)}"
    )
    full_seq = list(student_prompt_tokens) + list(gen_tokens)
    ob_len = len(student_prompt_tokens) - 1
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=full_seq[:-1]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(
                torch.tensor(full_seq[1:], dtype=torch.long)
            ),
            "logprobs": TensorData.from_torch(
                torch.tensor([0.0] * ob_len + list(student_logprobs), dtype=torch.float)
            ),
            "advantages": TensorData.from_torch(
                torch.tensor([0.0] * ob_len + list(advantages), dtype=torch.float)
            ),
        },
    )


# =============================================================================
# Solution-tag extraction (Qwen3 thinking mode produces a preamble)
# =============================================================================

_SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.DOTALL)
_HEADING_RE = re.compile(r"^(#+\s|\*\*[^\n]{1,80}\*\*\s*$)", re.MULTILINE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _to_str(content) -> str:
    """Coerce renderer content (str | dict | list) to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", content.get("content", str(content)))
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", item.get("content", str(item))))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def extract_solution(text) -> str:
    """Extract <solution>...</solution> body, stripping Qwen3 thinking preamble.

    Qwen3 often puts "Okay, I need to..." reasoning INSIDE the solution tags.
    Strip the <solution> wrapper first, then find the first markdown heading
    (# or **Title) and discard everything before it.

    The leading <think>...</think> block is removed BEFORE solution extraction
    because some Qwen3 outputs include the literal string `<solution>...</solution>`
    inside the think preamble (when the model describes the format), and the
    non-greedy _SOLUTION_RE would otherwise match that 3-character placeholder
    instead of the real plan body. See M8 methodology doc for the regression case.
    """
    text = _to_str(text)
    text = _THINK_RE.sub("", text)
    m = _SOLUTION_RE.search(text)
    if m:
        text = m.group(1).strip()
    heading = _HEADING_RE.search(text)
    if heading and heading.start() > 0:
        text = text[heading.start():]
    return text.strip()
