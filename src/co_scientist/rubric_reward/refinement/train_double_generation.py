"""
Double Generation Trainer (async GRPO)

Trains Qwen3-30B-A3B to generate research plans via a two-stage pipeline:

  Stage 1 — Initial Generation:
    For each research goal, generate `group_size` plans from scratch using
    `build_research_plan_prompt`. Grade each plan with the detailed grader,
    then extract structured feedback (weaknesses, fixes, desiderata scores).

  Stage 2 — Refinement:
    For each initial plan, build a refinement prompt that includes the scenario,
    rubric items, the initial draft, and the expert critique. The model generates
    a revised plan. Grade the revised plan with the same detailed grader.

  Training:
    Both stages produce GRPO groups with INDEPENDENT advantage normalization:
      - Group A (initial):  advantages computed within initial plans only
      - Group B (refined):  advantages computed within refined plans only
    Both groups contribute training datums to the SAME optimization step.

  Compute-matched design:
    group_size=4 per stage → 4 initial + 4 refined = 8 total per goal,
    matching the baseline's group_size=8 for fair comparison.
"""
import asyncio
import logging
import re
import textwrap
import time
import numpy as np
import json
import os
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
import datasets
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from datasets import load_dataset


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


# ============================================================
# 1. CONFIG — All tunable hyperparameters in one place.
#    Edit these values before launching a run.
# ============================================================

@chz.chz
class Config:
    # --- Connection (set TINKER_API_KEY / TINKER_BASE_URL in your terminal) ---
    base_url: str | None = None
    log_path: str = "/home/silas/co-scientist-project/runs/2026/3/refinement/8"
    model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"

    # --- Evaluation ---
    run_eval: bool = False
    eval_epoch: int = -1

    # --- Dataset ---
    ml_data: bool = True
    arxiv_data: bool = False
    pubmed_data: bool = False

    # --- Batch & sampling ---
    batch_size: int = 64     # Number of research goals per batch
    group_size: int = 4      # Samples per goal per group (initial AND refined → 8 total per goal)
    learning_rate: float = 1e-5
    clip_eps: float = 0.2

    # --- Model generation ---
    max_length: int = 32768
    lora_rank: int = 64
    save_every: int = 5
    max_tokens: int = 2048
    refinement_max_tokens: int = 4096  # Refinement prompt is longer → model needs more generation budget
    grader_max_tokens: int = 12288  # Detailed grader needs more tokens for structured XML output
    temperature: float = 1.0
    grader_temperature: float = 0.0

    # --- Timeouts (None = no timeout) ---
    policy_timeout_sec: float = None
    grader_timeout_sec: float = None
    refinement_timeout_sec: float = None
    train_timeout_sec: float = None

    # --- Server capability limits ---
    use_server_cap_limits: bool = True

    # --- LoRA training targets ---
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True

    # --- Reward shaping ---
    # Composite reward = rubric_score + scaling_factor * length_bonus - format_penalty
    # length_bonus = exp(-((solution_words - target_word_count) / scale_length_bonus)^2)
    # format_penalty = 0 if compliant, else 0.2 + 0.0005 * excess_words
    max_word_count: int = 750           # Hard cap on solution word count
    target_word_count: int = 600        # Sweet spot for length bonus (Gaussian center)
    scale_length_bonus: float = 120.0   # Width of Gaussian length bonus
    scaling_factor: float = 0.08        # Weight of length bonus in composite reward
    min_words: int = 30                 # Below this → degenerate sample, dropped

    # --- Refinement settings ---
    use_refinement: bool = True
    refinement_group_weight: float = 1.0  # Loss weight for refined group (1.0 = same as initial)
    normalize_advantages: bool = True     # Divide GRPO advantages by within-group reward std
    feedback_max_bullets_per_item: int = 3
    feedback_max_items: int = 10

    today_date = time.strftime("%Y-%m-%d", time.localtime())


# ============================================================
# 2. SERVICE CLIENT
#    Set TINKER_API_KEY and TINKER_BASE_URL in your terminal.
# ============================================================

def create_service_client(base_url: str | None = None) -> tinker.ServiceClient:
    return tinker.ServiceClient(base_url=base_url)


# ============================================================
# 3. SCORING / FORMAT COMPLIANCE
#    These functions parse grader XML output to compute rewards.
#
#    Grader XML structure (simplified):
#      <rubric>
#        <item num=1>
#          <criteria>...</criteria>
#          <reasoning>...</reasoning>
#          <desiderata num=1><level>0-3</level></desiderata>
#          ... (7 desiderata per item)
#        </item>
#        ... (one item per rubric criterion)
#      </rubric>
#
#    Level mapping (non-linear, rewards partial satisfaction):
#      0 → 0.0,  1 → 0.2,  2 → 0.6,  3 → 1.0
#    Rubric score = mean of per-item scores (each = mean of 7 mapped levels)
# ============================================================

def _strip_think_blocks(text: str) -> str:
    """Remove closed <think>...</think> spans so nested/quoted <solution> tags
    inside thinking content are never treated as the final plan."""
    return re.sub(
        r"<think\b[^>]*>.*?</think>",
        " ",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )


def extract_solution_text(plan_text: str, require_tags: bool = False) -> str | None:
    """Extract plain content from <solution> tags; optionally require tags.

    Uses the LAST <solution> match (after stripping <think> blocks) to be robust
    to stray XML fragments earlier in the output.
    """
    pattern = r"<solution>\s*(.*?)\s*</solution>"
    candidate_text = _strip_think_blocks(plan_text)
    matches = re.findall(pattern, candidate_text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    if require_tags:
        return None
    return plan_text.strip()


def sdpo_check_format_compliance(text: str, max_words: int) -> bool:
    """Check format compliance: content must be inside <solution> tags and ≤ max_words."""
    content = extract_solution_text(text, require_tags=True)
    if content is None:
        return False
    word_count = len(content.split())
    if word_count > max_words:
        return False
    return True


def sdpo_compute_rubric_reward_from_xml(xml_text: str) -> float:
    """Compute rubric score from grader XML. Returns float in [0, 1].

    Non-linear level mapping: {0: 0.0, 1: 0.2, 2: 0.6, 3: 1.0}
    Score = mean over items of (mean over 7 desiderata of mapped level).
    """
    level_map = {0: 0.0, 1: 0.2, 2: 0.6, 3: 1.0}
    item_blocks = re.findall(
        r"<item\s+num=.*?>.*?</item>", xml_text, flags=re.DOTALL | re.IGNORECASE
    )
    if not item_blocks:
        return 0.0
    item_scores = []
    for item_xml in item_blocks:
        levels = re.findall(r"<level>(\d+)</level>", item_xml)
        levels = [int(l) for l in levels if l.isdigit()]
        if not levels:
            continue
        if len(levels) < 7:
            levels = levels + [0] * (7 - len(levels))
        levels = levels[:7]
        mapped = [level_map.get(l, 0.0) for l in levels]
        item_scores.append(sum(mapped) / len(mapped))
    if not item_scores:
        return 0.0
    return sum(item_scores) / len(item_scores)


def is_valid_grader_xml(xml_text: str) -> bool:
    """Basic structural validation: has <rubric>, has <item> blocks, has valid <level> values."""
    if re.search(r"<rubric\b", xml_text, flags=re.IGNORECASE) is None:
        return False
    item_blocks = re.findall(
        r"<item\s+num=.*?>.*?</item>", xml_text, flags=re.DOTALL | re.IGNORECASE,
    )
    if not item_blocks:
        return False
    return any(
        re.search(r"<level>\s*[0-3]\s*</level>", item_xml, flags=re.IGNORECASE) is not None
        for item_xml in item_blocks
    )


# ============================================================
# 4. FEEDBACK EXTRACTION FROM GRADER XML
#    Parses the detailed grader XML to produce a compact textual
#    critique that gets inserted into the refinement prompt.
#
#    Output structure (fed to the model as "Expert Critique"):
#      # Weaknesses
#      - [rubric item] concrete weakness text
#      # Fixes
#      - [rubric item] concrete fix text
#      # Desiderata Scores by Rubric Item
#      Key: D1: ...; D2: ...; ...  (printed once)
#      - Rubric item: "X". Behavior: ... Profile: D1=2, D2=1, ...
# ============================================================

_DESIDERATA_LABELS = {
    1: "handling all rubric criteria",
    2: "specific implementation detail",
    3: "addressing critical flaws and risks",
    4: "strong justification and rationale",
    5: "cost and effort efficiency",
    6: "ethical and safety safeguards",
    7: "consistency with the overall plan",
}

def _clean_reasoning_text(reasoning_raw: str) -> str:
    """Remove XML artifacts and normalize whitespace from grader reasoning."""
    no_tags = re.sub(r"<[^>]+>", " ", reasoning_raw)
    return re.sub(r"\s+", " ", no_tags).strip()


def _extract_bullets(section_text: str) -> list[str]:
    """Extract bullet points from a text section. Falls back to sentence splitting."""
    lines: list[str] = []
    for raw_line in section_text.splitlines():
        line = re.sub(r"^\s*[-*•]\s*", "", raw_line).strip()
        if not line:
            continue
        lines.append(re.sub(r"\s+", " ", line))
    if lines:
        return lines
    fallback = _clean_reasoning_text(section_text)
    if not fallback:
        return []
    return [p.strip() for p in re.split(r"[.;]\s+", fallback) if p.strip()]


def _extract_item_weaknesses_and_fixes(item_xml: str) -> tuple[list[str], list[str]]:
    """Extract per-item weaknesses and fixes from structured XML tags.

    Tries <weaknesses> and <actionable_fixes> XML tags first (the schema we ask
    the grader to produce). Falls back to regex-matching 'Weaknesses:' /
    'Actionable fixes:' headings inside <reasoning> for robustness.
    """
    # Primary: structured XML tags
    w_text = _extract_xml_tag_text(item_xml, "weaknesses")
    f_text = _extract_xml_tag_text(item_xml, "actionable_fixes")
    weakness_lines = _extract_bullets(w_text) if w_text else []
    fix_lines = _extract_bullets(f_text) if f_text else []
    if weakness_lines or fix_lines:
        return weakness_lines, fix_lines

    # Fallback: regex inside <reasoning> for older grader outputs
    reasoning_match = re.search(
        r"<reasoning>\s*(.*?)\s*</reasoning>",
        item_xml, flags=re.DOTALL | re.IGNORECASE,
    )
    reasoning_raw = reasoning_match.group(1) if reasoning_match else ""
    weak_match = re.search(
        r"Weaknesses:\s*(.*?)(?:Actionable\s+fixes:|$)",
        reasoning_raw, flags=re.DOTALL | re.IGNORECASE,
    )
    fix_match = re.search(
        r"Actionable\s+fixes:\s*(.*)$",
        reasoning_raw, flags=re.DOTALL | re.IGNORECASE,
    )
    weakness_lines = _extract_bullets(weak_match.group(1)) if weak_match else []
    fix_lines = _extract_bullets(fix_match.group(1)) if fix_match else []
    if weakness_lines or fix_lines:
        return weakness_lines, fix_lines
    fallback_line = _clean_reasoning_text(reasoning_raw)
    return ([fallback_line] if fallback_line else []), []


def _extract_xml_tag_text(text: str, tag: str) -> str:
    """Extract text content from a single XML tag, cleaned of nested tags."""
    match = re.search(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        text, flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return ""
    return _clean_reasoning_text(match.group(1))


def _is_generic_feedback_line(text: str) -> bool:
    """Filter out low-signal feedback like 'needs more detail' without concrete hints."""
    lower = text.lower()
    if len(lower) < 24:
        return True
    generic_phrases = (
        "needs more detail", "be more specific", "needs improvement",
        "improve clarity", "unclear", "insufficient detail", "not detailed enough",
    )
    concrete_hints = (
        "dataset", "baseline", "metric", "ablation", "evaluation", "control",
        "confound", "power", "random", "risk", "failure", "implementation",
        "resource", "timeline", "cost", "criterion", "token", "format", "compliance",
    )
    if any(phrase in lower for phrase in generic_phrases) and not any(
        hint in lower for hint in concrete_hints
    ):
        return True
    return False


def _build_fallback_weakness_from_levels(criteria: str, levels: list[int]) -> str:
    """Synthesize actionable weaknesses when grader reasoning is malformed or empty."""
    if not levels:
        return ""
    low_idxs = [idx + 1 for idx, level in enumerate(levels[:7]) if level <= 2]
    if not low_idxs:
        return ""
    low_dims = ", ".join(_DESIDERATA_LABELS.get(i, f"desiderata {i}") for i in low_idxs[:4])
    extra = f", and {len(low_idxs) - 4} more areas" if len(low_idxs) > 4 else ""
    criteria_prefix = f'For rubric item "{criteria}", ' if criteria else ""
    return (
        f"{criteria_prefix}the plan is weak on {low_dims}{extra}. "
        "Add concrete implementation steps, explicit rationale for design choices, "
        "risk/failure analysis, and clear evaluation criteria tied to this rubric item."
    )


def _extract_sample_review_sections(xml_text: str) -> tuple[list[str], list[str]]:
    """Return (weaknesses, potential_fixes) from sample-level <sample_review>."""
    review_match = re.search(
        r"<sample_review>\s*(.*?)\s*</sample_review>",
        xml_text, flags=re.DOTALL | re.IGNORECASE,
    )
    if review_match is None:
        return [], []
    review_body = review_match.group(1)
    weaknesses_raw = _extract_xml_tag_text(review_body, "weaknesses")
    fixes_raw = _extract_xml_tag_text(review_body, "potential_fixes")
    return _extract_bullets(weaknesses_raw), _extract_bullets(fixes_raw)


def _extract_item_desiderata_review_parts(item_xml: str) -> str:
    """Return the summary text from a per-item <desiderata_review> block."""
    review_match = re.search(
        r"<desiderata_review>\s*(.*?)\s*</desiderata_review>",
        item_xml, flags=re.DOTALL | re.IGNORECASE,
    )
    if review_match is None:
        return ""
    return _extract_xml_tag_text(review_match.group(1), "summary")


def _build_desiderata_reference_text() -> str:
    """One-line legend mapping D1..D7 to short labels (printed once in feedback header)."""
    entries = [
        f"D{i}: {_DESIDERATA_LABELS.get(i, f'desiderata {i}')}"
        for i in range(1, 8)
    ]
    return "; ".join(entries)


def _build_desiderata_profile_text(levels: list[int]) -> str:
    """Compact D1=2, D2=1, ... notation for a single rubric item's 7 desiderata levels."""
    normalized: list[str] = []
    for idx in range(1, 8):
        if idx - 1 < len(levels) and levels[idx - 1] in {0, 1, 2, 3}:
            normalized.append(f"D{idx}={levels[idx - 1]}")
        else:
            normalized.append(f"D{idx}=NA")
    return ", ".join(normalized)


def _infer_desiderata_behavior_from_levels(levels: list[int]) -> str:
    """Produce a natural-language summary of which desiderata are weak/strong."""
    low_dims = [
        _DESIDERATA_LABELS.get(i + 1, f"desiderata {i + 1}")
        for i, lvl in enumerate(levels[:7])
        if lvl in {0, 1, 2}
    ]
    strong_dims = [
        _DESIDERATA_LABELS.get(i + 1, f"desiderata {i + 1}")
        for i, lvl in enumerate(levels[:7])
        if lvl == 3
    ]
    if low_dims:
        weak_part = ", ".join(low_dims[:3])
        if len(low_dims) > 3:
            weak_part += f", and {len(low_dims) - 3} more areas"
        if strong_dims:
            strong_part = ", ".join(strong_dims[:2])
            return (
                f"The item underperforms on {weak_part}, while showing relative strength on {strong_part}."
            )
        return f"The item underperforms on {weak_part}."
    return "The item is consistently strong across all seven desiderata."


def extract_weaknesses_from_grader(
    xml_text: str,
    *,
    max_bullets_per_item: int = 3,
    max_items: int = 10,
    focus_low_confidence_items_only: bool = True,
    include_sample_review: bool = True,
    include_item_reasoning_feedback: bool = True,
    include_item_desiderata_review: bool = True,
) -> tuple[str, int, int]:
    """Build a compact critique from grader XML for the refinement prompt.

    Returns (feedback_text, bullet_count, low_level_item_count).
    Only rubric items with at least one desiderata level ≤ 2 are included
    when focus_low_confidence_items_only=True (default).
    """
    default_feedback = (
        "Critical revision needed: strengthen implementation detail, rationale, and risk controls."
    )
    if not xml_text:
        return default_feedback, 1, 0

    item_blocks = re.findall(
        r"<item\b[^>]*>.*?</item>", xml_text, flags=re.DOTALL | re.IGNORECASE,
    )
    if not item_blocks:
        return default_feedback, 1, 0

    max_lines = max(1, int(max_bullets_per_item))
    sample_review_weaknesses, sample_review_fixes = (
        _extract_sample_review_sections(xml_text)
        if include_sample_review else ([], [])
    )
    critical_weaknesses: list[str] = []
    required_fixes: list[str] = []
    rubric_desiderata_behavior: list[str] = []
    seen: set[str] = set()
    low_level_item_count = 0

    # Collect sample-level review bullets first (highest signal)
    for weakness in sample_review_weaknesses[:max_lines]:
        compact = re.sub(r"\s+", " ", weakness).strip()
        if compact and compact not in seen:
            critical_weaknesses.append(compact)
            seen.add(compact)
    for fix in sample_review_fixes[:max_lines]:
        compact = re.sub(r"\s+", " ", fix).strip()
        if compact and compact not in seen:
            required_fixes.append(compact)
            seen.add(compact)

    # Then collect per-rubric-item feedback
    processed_items = 0
    max_items_effective = max(1, int(max_items))
    for item_xml in item_blocks:
        all_numeric_levels = [
            int(level)
            for level in re.findall(
                r"<level>\s*([0-3])\s*</level>", item_xml, flags=re.IGNORECASE,
            )
        ]
        numeric_levels = all_numeric_levels[-7:] if len(all_numeric_levels) >= 7 else all_numeric_levels
        has_low_level = any(level <= 2 for level in numeric_levels)
        if focus_low_confidence_items_only and not has_low_level:
            continue
        if processed_items >= max_items_effective:
            break
        processed_items += 1
        if has_low_level:
            low_level_item_count += 1

        criteria_match = re.search(
            r"<criteria>\s*(.*?)\s*</criteria>",
            item_xml, flags=re.DOTALL | re.IGNORECASE,
        )
        criteria = ""
        if criteria_match is not None:
            criteria = re.sub(r"\s+", " ", criteria_match.group(1)).strip()

        item_weaknesses: list[str] = []
        item_fixes: list[str] = []
        if include_item_reasoning_feedback:
            item_weaknesses, item_fixes = _extract_item_weaknesses_and_fixes(item_xml)
            item_weaknesses = [
                line for line in item_weaknesses if line and not _is_generic_feedback_line(line)
            ]
            item_fixes = [line for line in item_fixes if line and not _is_generic_feedback_line(line)]
        if include_item_reasoning_feedback and not item_weaknesses:
            fallback = _build_fallback_weakness_from_levels(criteria, numeric_levels[:7])
            if fallback:
                item_weaknesses = [fallback]

        for weakness in item_weaknesses[:max_lines]:
            compact = re.sub(r"\s+", " ", weakness).strip()
            if not compact:
                continue
            line = f"[{criteria}] {compact}" if criteria else compact
            if line not in seen:
                critical_weaknesses.append(line)
                seen.add(line)
        for fix in item_fixes[:max_lines]:
            compact = re.sub(r"\s+", " ", fix).strip()
            if not compact:
                continue
            line = f"[{criteria}] {compact}" if criteria else compact
            if line not in seen:
                required_fixes.append(line)
                seen.add(line)

        if include_item_desiderata_review:
            review_summary = _extract_item_desiderata_review_parts(item_xml)
            inferred_behavior = _infer_desiderata_behavior_from_levels(numeric_levels[:7])
            summary_text = review_summary or inferred_behavior
            profile_text = _build_desiderata_profile_text(numeric_levels[:7])
            behavior_line = (
                f'Rubric item: "{criteria}". '
                f"Behavior: {summary_text} "
                f"Profile: {profile_text}."
            )
            behavior_line = re.sub(r"\s+", " ", behavior_line).strip()
            if behavior_line and behavior_line not in seen:
                rubric_desiderata_behavior.append(behavior_line)
                seen.add(behavior_line)

    # Assemble final feedback text
    feedback_sections: list[str] = []
    if critical_weaknesses:
        weakness_lines = [f"- {line}" for line in critical_weaknesses]
        feedback_sections.append("# Weaknesses\n" + "\n".join(weakness_lines))
    if required_fixes:
        fix_lines = [f"- {line}" for line in required_fixes]
        feedback_sections.append("# Fixes\n" + "\n".join(fix_lines))
    if rubric_desiderata_behavior:
        behavior_lines = [f"- {line}" for line in rubric_desiderata_behavior]
        # Desiderata reference legend printed once at the top, not per-item
        header = f"# Desiderata Scores by Rubric Item\nKey: {_build_desiderata_reference_text()}"
        feedback_sections.append(header + "\n" + "\n".join(behavior_lines))
    if not feedback_sections:
        return default_feedback, 1, low_level_item_count

    feedback_text = "\n\n".join(feedback_sections).strip()
    bullet_count = (
        len(critical_weaknesses)
        + len(required_fixes)
        + len(rubric_desiderata_behavior)
    )
    return feedback_text, max(1, bullet_count), low_level_item_count


# ============================================================
# 5. SERVER CAPABILITY HELPERS
#    Auto-reduce batch/group sizes if the Tinker server has
#    lower caps than configured.
# ============================================================

def _get_capability_int(capabilities: object, field_names: list[str]) -> int | None:
    for field_name in field_names:
        value = getattr(capabilities, field_name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _apply_server_cap_limits(
    batch_size: int,
    group_size: int,
    capabilities: object,
) -> tuple[int, int]:
    effective_batch_size = batch_size
    effective_group_size = group_size

    max_batch_size = _get_capability_int(capabilities, ["max_batch_size"])
    if max_batch_size is not None and effective_batch_size > max_batch_size:
        logger.warning(
            "Reducing batch_size from %s to server max_batch_size=%s",
            effective_batch_size,
            max_batch_size,
        )
        effective_batch_size = max_batch_size

    max_samples = _get_capability_int(
        capabilities,
        ["max_num_samples", "max_samples_per_request", "max_group_size", "max_n"],
    )
    if max_samples is not None and effective_group_size > max_samples:
        logger.warning(
            "Reducing group_size from %s to server sample cap=%s",
            effective_group_size,
            max_samples,
        )
        effective_group_size = max_samples

    return max(1, effective_batch_size), max(1, effective_group_size)


# ============================================================
# 6. PROMPT BUILDERS
#    Three prompts used in the pipeline. Edit these directly
#    to change what the model and grader see.
# ============================================================

def build_research_plan_prompt(
    scenario: str,
    examples: list[dict] | None = None,
) -> str:
    """Build the initial plan generation prompt.

    Output contract: model must produce <think>...</think><solution>...</solution>.
    Only content inside <solution> tags is graded.
    """

    prompt = textwrap.dedent(f"""
        I will provide you a research scenario. You have to provide me a concise yet thoughtful research plan with all details needed to execute it.
    """).strip()

    # Few-shot examples (optional)
    if examples:
        prompt += "\n\nFirst, I will show you some examples of research scenarios and how the researchers approached it."
        for i, ex in enumerate(examples):
            prompt += textwrap.dedent(f"""
                **Example {i+1}:**
                Scenario: {ex["scenario"]}

                Researcher's Plan:
                {ex["solution"]}
            """).strip()

    # Scenario + instructions
    prompt += textwrap.dedent(f"""
        Here is the research scenario.
        Scenario: {scenario}

        # Instructions
        First, come up with a detailed research plan to address the scenario based on the following overall solution guidelines:
        - The plan should address the goals of the scenario, and account for all constraints and confounders.
        - Do NOT just say WHAT you will do. Explain HOW you will do it and WHY it is needed. Provide clear explanation and justification for each proposed step. The solution inside <solution></solution> tags should be readable for humans, and not in XML itself.
        - The phrasing should NOT be verbose, and NOT be in past tense, as in "the author's approach" but rather in present tense, as how you would approach the problem.
        - Do not claim to have done any experiments or have results, just provide the plan.
        - Do not add self-proclaimed praises of your solution. For example do NOT say yourself it satisfies some desiderata, we will let the evaluator decide that.

        Then, return the following nested XML block as your output (always close opened XML tags):
        <think>
        Here is your thinking process. You can use this to reason about the problem before giving the final solution. But only the content within <solution></solution> tags will be judged so make sure to include (potentially repeat) all details in it.
        </think>
        <solution>
        Here is your final research plan. Make sure it is complete and self-contained. And it should not exceed 750 words.
        ... Your detailed research plan goes here ...
        </solution>
    """).strip()

    return prompt


def build_detailed_grader_prompt(
    scenario: str,
    rubric_items: list[str],
    proposed_plan: str,
    reference_solution: str | None = None,
    fast_grader_mode: bool = True,
    fast_reasoning_max_words_per_item: int = 90,
) -> str:
    """Build the detailed grader prompt.

    Produces structured XML with:
      - <sample_review>: overall weaknesses + fixes (used for feedback extraction)
      - <item num=N>: per-rubric-item scoring with 7 desiderata levels 0-3,
        plus <weaknesses>, <actionable_fixes>, and per-item <desiderata_review>
    """
    rubric_block = "\n".join(
        [f"Item {i+1}: {item}" for i, item in enumerate(rubric_items)]
    )

    desiderata_text = textwrap.dedent("""
    **DESIDERATA**
    1. HANDLES ALL CRITERIA: Does the plan satisfy all criteria mentioned in the rubric item? An exception is if the criteria says "such as", "for example", or "including", the response does not have to include the same examples listed to meet the criteria, but whatever is provided must be valid and reasonable.
    2. DETAILED, SPECIFIC SOLUTION: Does the part of the plan relevant to satisfying this rubric item include fully specified details on HOW to implement it? There should be no self-proclaimed claims of handling something without doing so. There should be no vague terms, ambiguity, or lack of clarity. It should be described in simple to understand language.
    3. NO OVERLOOKED FLAWS OR WEAKNESSES: Are there any important overlooked flaws or weaknesses in the part of the plan addressing this rubric item that invalidate its satisfaction of the rubric item?
    4. WELL JUSTIFIED RATIONALE: Is the part of the plan relevant to this grading item well-motivated and justified? For example, are there convincing arguments provided for how the plan handles this grading item is better than simpler solutions or alternate hypotheses?
    5. COST AND EFFORT EFFICIENT: Does the plan handle this item efficiently without unnecessary complexity?
    6. NO ETHICAL ISSUES: Does this part of the plan have any potential for negative consequences, or is it ethically problematic?
    7. CONSISTENT WITH OVERALL PLAN: Is this part of the plan consistent with the rest of the plan? Check if it contradicts any other parts of the plan.
    """).strip()

    prompt = textwrap.dedent(f"""
        Evaluate if the Proposed Research Plan satisfies the Research Scenario based on the provided evaluation criteria.

        # Research Scenario
        {scenario}

        You have to evaluate each of the rubric items provided below.

        # Rubric
        {rubric_block}
    """).strip()

    if reference_solution is not None:
        prompt += textwrap.dedent(f"""
            # Reference Solution
            Here is a reference solution written by an expert:
            {reference_solution}

            • It is just meant to demonstrate one possible approach that satisfies the scenario. It is not necessary for the proposed research plan you are grading to match all details in the reference solution.
            • The Research Plan you have to grade might have different design choices. This is okay, if the choices are valid, and supported with correct rationale.
        """).strip()

    min_weakness_bullets = 1 if fast_grader_mode else 2
    min_fix_bullets = 1 if fast_grader_mode else 2
    reasoning_budget = max(40, int(fast_reasoning_max_words_per_item))

    prompt += textwrap.dedent(f"""
        # Proposed Research Plan
        {proposed_plan}

        # Instructions
        Evaluate in this strict order:
        1) Sample-level critique first: identify the main weaknesses of this plan and concrete fixes.
        2) Rubric scoring second: review each rubric item (typically 10 items) against all 7 desiderata.

        General desiderata used for each rubric item:
        {desiderata_text}

        For EACH desideratum, assign a satisfaction level according to this scale:

                Level 0 — NOT SATISFIED:
                The plan does not meaningfully satisfy this desideratum.

                Level 1 — WEAKLY SATISFIED:
                The plan touches on this desideratum, but in a vague, superficial, or insufficient way.

                Level 2 — PARTIALLY SATISFIED:
                The plan satisfies this desideratum to a reasonable extent, but with notable gaps,
                weaknesses, or missing justifications.

                Level 3 — FULLY SATISFIED:
                The plan clearly, concretely, and convincingly satisfies this desideratum.
                No major issues are apparent.

        CRITICAL OUTPUT CONTRACT (XML only):
        - Return exactly one <rubric> root and close all tags.
        - Inside <rubric>, output exactly one <sample_review> block first:
          <sample_review>
            <weaknesses>Brief high-signal weaknesses, include at least {min_weakness_bullets} bullet-style points.</weaknesses>
            <potential_fixes>Brief concrete fixes, include at least {min_fix_bullets} bullet-style points.</potential_fixes>
          </sample_review>
        - Then output one <item num=...> per rubric item in the exact same order as provided.
        - For each <item>:
          1) <criteria> must repeat the exact rubric item string.
          2) <reasoning> must be brief (about {reasoning_budget} words max), concrete, and explain whether this rubric item satisfies each desideratum.
          3) <weaknesses> list concrete weaknesses of the plan for this rubric item (bullet-style, 1-3 points). Focus on what is missing, wrong, or vague.
          4) <actionable_fixes> list concrete fixes for the weaknesses above (bullet-style, 1-3 points). Each fix should be specific enough to implement.
          5) output exactly 7 <desiderata num=1..7> blocks.
          6) each desiderata block must contain exactly one integer <level> in {{0,1,2,3}}.
          7) after the 7 desiderata blocks, add one per-item <desiderata_review> block:
             <desiderata_review>
               <summary>Briefly explain how the 7 desiderata behave for this rubric item.</summary>
             </desiderata_review>
        - No extra text outside XML tags.
        - Be strict, evidence-based, and avoid repetitive wording.

        <rubric>
          <sample_review>
            <weaknesses>- weakness 1 ... - weakness 2 ...</weaknesses>
            <potential_fixes>- fix 1 ... - fix 2 ...</potential_fixes>
          </sample_review>
          <item num=1>
            <criteria>[exact rubric item text]</criteria>
            <reasoning>[brief reasoning against all 7 desiderata]</reasoning>
            <weaknesses>- concrete weakness 1\n- concrete weakness 2</weaknesses>
            <actionable_fixes>- concrete fix 1\n- concrete fix 2</actionable_fixes>
            <desiderata num=1><level>0|1|2|3</level></desiderata>
            <desiderata num=2><level>0|1|2|3</level></desiderata>
            <desiderata num=3><level>0|1|2|3</level></desiderata>
            <desiderata num=4><level>0|1|2|3</level></desiderata>
            <desiderata num=5><level>0|1|2|3</level></desiderata>
            <desiderata num=6><level>0|1|2|3</level></desiderata>
            <desiderata num=7><level>0|1|2|3</level></desiderata>
            <desiderata_review>
              <summary>[how the 7 desiderata behave for this item]</summary>
            </desiderata_review>
          </item>
          ...
        </rubric>
    """).strip()
    return prompt


def build_self_teacher_prompt(
    scenario: str,
    grader_feedback: str | None,
    initial_draft: str | None,
    rubric_items: list[str] | None = None,
    policy_output_mode: str = "think_solution",
    target_word_count: int = 600,
    max_solution_words: int = 750,
    max_think_words: int = 60,
) -> str:
    """Build the refinement prompt: scenario + rubric + initial draft + critique → revised plan.

    The model sees its own initial draft alongside the grader critique and the
    rubric items it is being evaluated against, so it can make targeted revisions.

    Prompt structure:
      1. System instruction (you are revising...)
      2. Research scenario
      3. Evaluation criteria (rubric items)
      4. Your initial draft
      5. Expert critique (extracted weaknesses/fixes/desiderata)
      6. Revision requirements
      7. Output contract (<think> + <solution>)
    """
    target_low = max(120, int(target_word_count - 80))
    target_high = min(int(max_solution_words), int(target_word_count + 100))

    sections: list[str] = [
        "You are revising a research plan based on expert critique. "
        "Improve the plan by addressing the most important issues while preserving what works well.",
        f"# Research Scenario\n{scenario}",
    ]

    if rubric_items:
        rubric_block = "\n".join(f"  {i+1}. {item}" for i, item in enumerate(rubric_items))
        sections.append(
            "# Evaluation Criteria\n"
            "Your revised plan will be graded against these rubric items. "
            "Make sure every item is addressed concretely:\n"
            f"{rubric_block}"
        )

    if initial_draft:
        sections.append(
            "# Your Initial Draft\n"
            "This is your previous attempt. Revise it to address the critique below "
            "while keeping what already works well.\n"
            f"{initial_draft}"
        )

    if grader_feedback:
        sections.append(
            "# Expert Critique\n"
            "Use the critique below to guide your revision. Prioritize the most impactful issues. "
            "Some points may overlap or be less critical — apply your judgment.\n"
            f"{grader_feedback}"
        )
    else:
        sections.append(
            "# Expert Critique\n"
            "No specific critique was provided. Focus on:\n"
            "- Strengthening concrete implementation details and rationale.\n"
            "- Adding explicit risk checks, evaluation criteria, and failure handling."
        )

    sections.append(
        textwrap.dedent(f"""
            # Revision Requirements
            - Address scenario goals, constraints, and confounders explicitly.
            - Explain HOW and WHY for each major step, not only WHAT to do.
            - Aim for {target_low}-{target_high} words.
        """).strip()
    )

    if policy_output_mode == "think_solution":
        sections.append(
            textwrap.dedent(f"""
                # Output Contract (strict)
                Return exactly two XML blocks and close all tags:
                <think>
                Brief reasoning about what to change, at most {max_think_words} words.
                </think>
                <solution>
                Complete and self-contained revised research plan.
                </solution>

                Important:
                - Put all substantive details in <solution>.
                - First output token must be <think>, and final output token must be </solution>.
                - Do not output any text before/after these two blocks.
            """).strip()
        )
    elif policy_output_mode == "solution_only":
        sections.append(
            "# Output Format\n"
            "Wrap your complete revised research plan in <solution></solution> tags."
        )
    else:
        raise ValueError(
            f"Invalid policy_output_mode={policy_output_mode!r}. "
            "Expected 'solution_only' or 'think_solution'."
        )
    return "\n\n".join(sections).strip()


# ============================================================
# 7. ASYNC SAMPLING HELPERS
#    Thin wrappers that add timeout + error handling around
#    Tinker's async sampling API.
# ============================================================

async def _sample_policy_with_timeout(
    sampling_client: tinker.SamplingClient,
    model_input: types.ModelInput,
    goal_idx: int,
    prompt_tokens: list[int],
    group_size: int,
    sampling_params: types.SamplingParams,
    timeout_sec: float,
) -> tuple[int, list[int], types.SampleResponse] | None:
    """Sample `group_size` initial plans for one goal. Returns (goal_idx, prompt_tokens, response)."""
    try:
        response = await asyncio.wait_for(
            sampling_client.sample_async(
                prompt=model_input,
                num_samples=group_size,
                sampling_params=sampling_params,
            ),
            timeout=timeout_sec,
        )
        return goal_idx, prompt_tokens, response
    except asyncio.TimeoutError:
        logger.warning("Policy sampling timed out for goal_idx=%s", goal_idx)
    except Exception:
        logger.exception("Policy sampling failed for goal_idx=%s", goal_idx)
    return None


async def _sample_grader_with_timeout(
    grader_client: tinker.SamplingClient,
    grader_input: types.ModelInput,
    group_idx: int,
    sample_idx: int,
    grader_sampling_params: types.SamplingParams,
    timeout_sec: float,
) -> tuple[int, int, types.SampleResponse | None]:
    """Grade one plan. Returns (group_idx, sample_idx, response_or_None)."""
    try:
        response = await asyncio.wait_for(
            grader_client.sample_async(
                prompt=grader_input,
                num_samples=1,
                sampling_params=grader_sampling_params,
            ),
            timeout=timeout_sec,
        )
        return group_idx, sample_idx, response
    except asyncio.TimeoutError:
        logger.warning(
            "Grader sampling timed out for group_idx=%s sample_idx=%s",
            group_idx, sample_idx,
        )
    except Exception:
        logger.exception(
            "Grader sampling failed for group_idx=%s sample_idx=%s",
            group_idx, sample_idx,
        )
    return group_idx, sample_idx, None


async def _sample_refinement_with_timeout(
    sampling_client: tinker.SamplingClient,
    refine_input: types.ModelInput,
    group_idx: int,
    sample_idx: int,
    sampling_params: types.SamplingParams,
    timeout_sec: float,
) -> tuple[int, int, types.SampleResponse | None]:
    """Generate one refined plan. Returns (group_idx, sample_idx, response_or_None)."""
    try:
        response = await asyncio.wait_for(
            sampling_client.sample_async(
                prompt=refine_input,
                num_samples=1,
                sampling_params=sampling_params,
            ),
            timeout=timeout_sec,
        )
        return group_idx, sample_idx, response
    except asyncio.TimeoutError:
        logger.warning(
            "Refinement sampling timed out for group_idx=%s sample_idx=%s",
            group_idx, sample_idx,
        )
    except Exception:
        logger.exception(
            "Refinement sampling failed for group_idx=%s sample_idx=%s",
            group_idx, sample_idx,
        )
    return group_idx, sample_idx, None


# ============================================================
# 8. REWARD COMPUTATION & TRAINING DATUM HELPERS
# ============================================================

def compute_reward(plan_text: str, xml_text: str, config: Config, raw_plan_text: str | None = None) -> dict:
    """Compute composite reward for a single plan.

    reward = rubric_score + scaling_factor * length_bonus - format_penalty

    Length bonus and excess penalty use solution-only word count (excluding <think>
    blocks) so the model is incentivized for the right solution length regardless
    of think-block verbosity.

    Compliance is checked on raw (pre-repair) text when available, so that plans
    missing </solution> are correctly penalized.
    """
    rubric_score = sdpo_compute_rubric_reward_from_xml(xml_text)
    word_count = len(plan_text.strip().split())

    # Use solution-only text for length-related signals
    solution_content = extract_solution_text(plan_text)
    solution_word_count = len(solution_content.split()) if solution_content else word_count

    compliance_text = raw_plan_text if raw_plan_text is not None else plan_text
    is_compliant = sdpo_check_format_compliance(compliance_text, config.max_word_count)
    excess = max(0, solution_word_count - config.max_word_count)
    format_penalty = 0.0 if is_compliant else 0.2 + 0.0005 * excess
    length_bonus = np.exp(
        -((solution_word_count - config.target_word_count) / config.scale_length_bonus) ** 2
    )
    final_reward = rubric_score + config.scaling_factor * length_bonus - format_penalty

    return {
        "rubric_score": rubric_score,
        "word_count": word_count,
        "solution_word_count": solution_word_count,
        "is_compliant": is_compliant,
        "format_penalty": format_penalty,
        "length_bonus": length_bonus,
        "final_reward": final_reward,
    }


def is_valid_sample(plan_text: str, logprobs, config: Config) -> tuple[bool, str]:
    """Check if a plan is valid for training (not degenerate).

    Returns (is_valid, reason) where reason is 'ok' or a drop category.
    """
    word_count = len(plan_text.strip().split())
    if word_count < config.min_words:
        return False, "too_short"
    if plan_text.strip() in {"<think>", "</think>", "<solution>", "</solution>"}:
        return False, "bare_tag"
    if logprobs is None:
        return False, "no_logprobs"
    return True, "ok"


def close_solution_tag(plan_text: str) -> str:
    """Hard-close solution tag if truncated (case-insensitive).

    Needed because the grader requires well-formed <solution>...</solution> to parse,
    but the model may get cut off by max_tokens before closing the tag.
    """
    has_open = re.search(r"<solution>", plan_text, re.IGNORECASE)
    has_close = re.search(r"</solution>", plan_text, re.IGNORECASE)
    if has_open and not has_close:
        return plan_text.rstrip() + "\n</solution>"
    return plan_text


def create_training_datum(
    prompt_tokens: list[int],
    sample_info: dict,
    advantage: float,
) -> types.Datum | None:
    """Create a single GRPO training datum from prompt + generated tokens + advantage.

    The datum contains:
      - input_tokens / target_tokens: full sequence shifted by 1
      - logprobs: 0.0 for prompt tokens, actual logprobs for generated tokens
      - advantages: 0.0 for prompt tokens, the GRPO advantage for generated tokens
    """
    generated_tokens = [int(t) for t in sample_info["tokens"]]
    full_seq = prompt_tokens + generated_tokens
    ob_len = len(prompt_tokens) - 1

    input_tokens = full_seq[:-1]
    target_tokens = full_seq[1:]

    sample_logprobs = sample_info["logprobs"]
    all_logprobs = [0.0] * ob_len + sample_logprobs
    all_advantages = [0.0] * ob_len + [advantage] * len(sample_logprobs)

    if not (
        len(input_tokens)
        == len(target_tokens)
        == len(all_logprobs)
        == len(all_advantages)
    ):
        logger.warning(
            "Skipping malformed sample lengths: input=%s target=%s logprobs=%s advantages=%s",
            len(input_tokens), len(target_tokens), len(all_logprobs), len(all_advantages),
        )
        return None

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(
                torch.tensor(target_tokens, dtype=torch.long)
            ),
            "logprobs": TensorData.from_torch(
                torch.tensor(all_logprobs, dtype=torch.float)
            ),
            "advantages": TensorData.from_torch(
                torch.tensor(all_advantages, dtype=torch.float)
            ),
        },
    )


# ============================================================
# 9. MAIN TRAINING LOOP
#
#    Each batch processes `batch_size` research goals through
#    6 phases, then runs one PPO-clipped optimization step.
#
#    Phase 1: Generate `group_size` initial plans per goal (async)
#    Phase 2: Grade each initial plan with detailed grader (async)
#    Phase 3: Parse grades, extract feedback for refinement
#    Phase 4: Generate refined plans using feedback (async)
#    Phase 5: Grade each refined plan (async)
#    Phase 6: Compute GRPO advantages (separate per group),
#             build training datums, run forward/backward + optim
# ============================================================

async def main(config: Config):
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    is_eval = config.run_eval
    os.makedirs(config.log_path, exist_ok=True)

    # --- Setup tokenizer, renderer, dataset ---
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    logger.info("Loading dataset...")
    if config.ml_data:
        data = load_dataset("facebook/research-plan-gen", "ml")
    elif config.arxiv_data:
        data = load_dataset("facebook/research-plan-gen", "arxiv")
    elif config.pubmed_data:
        data = load_dataset("facebook/research-plan-gen", "pubmed")

    assert isinstance(data, datasets.DatasetDict)
    dataset = data["test"] if is_eval else data["train"]

    # --- Setup Tinker clients ---
    service_client = create_service_client(base_url=config.base_url)

    effective_batch_size = config.batch_size
    effective_group_size = config.group_size

    if config.use_server_cap_limits:
        try:
            capabilities = await service_client.get_server_capabilities_async()
            effective_batch_size, effective_group_size = _apply_server_cap_limits(
                batch_size=config.batch_size,
                group_size=config.group_size,
                capabilities=capabilities,
            )
        except Exception:
            logger.exception(
                "Failed to query server capabilities; using configured batch/group sizes"
            )

    n_train_batches = len(dataset) // effective_batch_size
    if n_train_batches <= 0:
        logger.warning(
            "No train batches available: dataset=%s batch_size=%s",
            len(dataset), effective_batch_size,
        )
        ml_logger.close()
        return

    # --- Load or create checkpoint ---
    last_checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path)
    resume_info: dict[str, object] | bool = False

    if (config.eval_epoch == 0 and config.run_eval) or (
        not config.run_eval and last_checkpoint is None
    ):
        resume_info = False
    elif config.eval_epoch > 0 and config.run_eval:
        checkpoints = checkpoint_utils.load_checkpoints_file(config.log_path)
        checkpoints_with_key = [c for c in checkpoints if "state_path" in c]
        if checkpoints_with_key and config.eval_epoch <= len(checkpoints_with_key):
            resume_info = checkpoints_with_key[config.eval_epoch - 1]
            logger.info(
                f"Found {len(checkpoints_with_key)} valid checkpoints in {config.log_path}"
            )
        elif checkpoints_with_key:
            logger.warning(
                "Requested eval_epoch=%s exceeds available checkpoints=%s; using latest",
                config.eval_epoch, len(checkpoints_with_key),
            )
            resume_info = checkpoints_with_key[-1]
    else:
        resume_info = last_checkpoint

    if resume_info:
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(
            str(resume_info["state_path"])
        )
        resume_batch = int(resume_info["batch"])
        if is_eval:
            actual_batch = resume_batch + 1
            start_batch = 0
        elif (resume_batch + 1) // n_train_batches > 0:
            logger.info(f"Training for epoch: {(resume_batch + 1) // n_train_batches}")
            actual_batch = resume_batch + 1
            start_batch = 0
        else:
            start_batch = resume_batch + 1
            actual_batch = 0
        if is_eval:
            logger.info(f"Evaluating for Checkpoint {resume_info['batch']}")
        else:
            logger.info(f"Resuming from batch {resume_info['batch']}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_name,
            rank=config.lora_rank,
            train_mlp=config.train_mlp,
            train_attn=config.train_attn,
            train_unembed=config.train_unembed,
        )
        start_batch = 0
        actual_batch = 0

    # Grader uses base model (no LoRA) so its scores are stable across training
    grader_client = await service_client.create_sampling_client_async(
        base_model=config.grader_model_name
    )

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )
    refinement_sampling_params = tinker.types.SamplingParams(
        max_tokens=config.refinement_max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )
    grader_sampling_params = tinker.types.SamplingParams(
        max_tokens=config.grader_max_tokens,
        temperature=config.grader_temperature,
    )

    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    if is_eval:
        logger.info("Eval mode: preparing fixed sampler")
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(
            name=f"Eval_Checkpoint{actual_batch - 1}"
        )

    logger.info(f"Training for {n_train_batches} batches")
    last_real_batch = -1

    # ================================================================
    # MAIN LOOP — one iteration per batch of research goals
    # ================================================================
    for batch_idx in range(start_batch, n_train_batches):
        t_start = time.time()

        if is_eval:
            real_batch = actual_batch
        else:
            real_batch = actual_batch + (batch_idx - start_batch)
        last_real_batch = real_batch

        # Select batch rows from dataset
        batch_start = batch_idx * effective_batch_size
        batch_end = min((batch_idx + 1) * effective_batch_size, len(dataset))
        batch_rows = dataset.select(range(batch_start, batch_end))

        if not is_eval:
            # Snapshot current LoRA weights for sampling (policy is frozen during generation)
            sampling_client = await training_client.save_weights_and_get_sampling_client_async()

        logger.info(
            f"{'EVAL' if is_eval else 'TRAIN'} Batch {real_batch}: "
            f"{len(batch_rows)} goals x {effective_group_size} samples"
        )

        # ==============================================================
        # PHASE 1: GENERATE INITIAL PLANS
        # For each goal, sample `group_size` plans using the initial
        # generation prompt. All goals are launched concurrently.
        # ==============================================================
        policy_tasks = []
        for goal_idx, goal in enumerate(batch_rows["Goal"]):
            prompt_text = build_research_plan_prompt(scenario=goal, examples=None)
            model_input = renderer.build_generation_prompt(
                [{"role": "user", "content": prompt_text}]
            )
            policy_tasks.append(
                asyncio.create_task(
                    _sample_policy_with_timeout(
                        sampling_client=sampling_client,
                        model_input=model_input,
                        goal_idx=goal_idx,
                        prompt_tokens=model_input.to_ints(),
                        group_size=effective_group_size,
                        sampling_params=sampling_params,
                        timeout_sec=config.policy_timeout_sec,
                    ),
                    name=f"policy_goal_{goal_idx}",
                )
            )

        # ==============================================================
        # PHASE 2: COLLECT INITIAL PLANS & LAUNCH GRADERS
        # As each goal's plans arrive, repair truncated </solution> tags,
        # then launch detailed grader for each plan. Store raw (pre-repair)
        # text separately for compliance checking.
        # ==============================================================
        logger.info(f"Batch {real_batch}: Collecting initial plans, launching graders...")

        batch_groups_data: list[dict | None] = [None] * len(batch_rows["Goal"])
        initial_grader_tasks = []

        for p_task in asyncio.as_completed(policy_tasks):
            policy_result = await p_task
            if policy_result is None:
                continue

            goal_idx, prompt_tokens, result = policy_result
            goal = batch_rows["Goal"][goal_idx]
            rubric = batch_rows["Rubric"][goal_idx]
            ref_sol = batch_rows["Reference solution"][goal_idx]

            group_samples_info = []
            group_grader_results: list[types.SampleResponse | None] = []

            for sample_idx, group_result in enumerate(result.sequences):
                raw_plan = renderers.get_text_content(
                    renderer.parse_response(group_result.tokens)[0]
                )
                proposed_plan = close_solution_tag(raw_plan)

                grader_prompt_text = build_detailed_grader_prompt(
                    scenario=goal,
                    rubric_items=rubric,
                    proposed_plan=proposed_plan,
                    reference_solution=ref_sol,
                    fast_grader_mode=True,
                )
                grader_input = renderer.build_generation_prompt(
                    [{"role": "user", "content": grader_prompt_text}]
                )

                initial_grader_tasks.append(
                    asyncio.create_task(
                        _sample_grader_with_timeout(
                            grader_client=grader_client,
                            grader_input=grader_input,
                            group_idx=goal_idx,
                            sample_idx=sample_idx,
                            grader_sampling_params=grader_sampling_params,
                            timeout_sec=config.grader_timeout_sec,
                        ),
                        name=f"init_grader_g{goal_idx}_s{sample_idx}",
                    )
                )
                group_samples_info.append({
                    "tokens": group_result.tokens,
                    "logprobs": group_result.logprobs,
                    "text": proposed_plan,      # repaired (for grading)
                    "raw_text": raw_plan,        # original (for compliance check)
                })
                group_grader_results.append(None)

            batch_groups_data[goal_idx] = {
                "goal": goal,
                "rubric": rubric,
                "ref_sol": ref_sol,
                "prompt_tokens": prompt_tokens,
                "samples_info": group_samples_info,
                "grader_results": group_grader_results,
            }

        # Collect grader results as they finish
        for g_task in asyncio.as_completed(initial_grader_tasks):
            group_idx, sample_idx, grader_result = await g_task
            group_data = batch_groups_data[group_idx]
            if group_data is not None:
                group_data["grader_results"][sample_idx] = grader_result

        # ==============================================================
        # PHASE 3: PARSE INITIAL GRADES & EXTRACT FEEDBACK
        # For each plan: compute reward from grader XML, then extract
        # structured feedback (weaknesses, fixes, desiderata scores)
        # that will be inserted into the refinement prompt.
        # Grader failures → zero reward + generic feedback.
        # ==============================================================
        logger.info(f"Batch {real_batch}: Parsing initial grades, extracting feedback...")

        all_initial_grades: list[list[dict] | None] = [None] * len(batch_rows["Goal"])

        for group_idx, group_data in enumerate(batch_groups_data):
            if group_data is None:
                continue

            group_grades = []
            for j, grader_result in enumerate(group_data["grader_results"]):
                plan_text = group_data["samples_info"][j]["text"]
                raw_text = group_data["samples_info"][j]["raw_text"]

                if grader_result is None:
                    group_grades.append({
                        "xml_text": "",
                        "plan_text": plan_text,
                        "feedback_text": "Critical revision needed: strengthen implementation detail, rationale, and risk controls.",
                        "feedback_bullets": 1,
                        "rubric_score": 0.0,
                        "word_count": len(plan_text.strip().split()),
                        "solution_word_count": 0,
                        "is_compliant": sdpo_check_format_compliance(raw_text, config.max_word_count),
                        "format_penalty": 0.2,
                        "length_bonus": 0.0,
                        "final_reward": 0.0,
                        "failure_reason": "grader_timeout_or_error",
                    })
                    continue

                try:
                    parsed_msg, _ = renderer.parse_response(grader_result.sequences[0].tokens)
                    xml_text = renderers.get_text_content(parsed_msg)
                except Exception:
                    logger.exception(
                        "Failed to parse grader output for group_idx=%s sample_idx=%s",
                        group_idx, j,
                    )
                    group_grades.append({
                        "xml_text": "",
                        "plan_text": plan_text,
                        "feedback_text": "Critical revision needed: strengthen implementation detail, rationale, and risk controls.",
                        "feedback_bullets": 1,
                        "rubric_score": 0.0,
                        "word_count": len(plan_text.strip().split()),
                        "solution_word_count": 0,
                        "is_compliant": sdpo_check_format_compliance(raw_text, config.max_word_count),
                        "format_penalty": 0.2,
                        "length_bonus": 0.0,
                        "final_reward": 0.0,
                        "failure_reason": "grader_parse_error",
                    })
                    continue

                reward_info = compute_reward(plan_text, xml_text, config, raw_plan_text=raw_text)

                # Extract structured feedback for the refinement prompt
                if is_valid_grader_xml(xml_text):
                    feedback_text, bullet_count, _ = extract_weaknesses_from_grader(
                        xml_text,
                        max_bullets_per_item=config.feedback_max_bullets_per_item,
                        max_items=config.feedback_max_items,
                        focus_low_confidence_items_only=True,
                        include_sample_review=True,
                        include_item_reasoning_feedback=True,
                        include_item_desiderata_review=False,  # D-score profiles add tokens without helping revision
                    )
                else:
                    feedback_text = "Critical revision needed: strengthen implementation detail, rationale, and risk controls."
                    bullet_count = 1

                group_grades.append({
                    "xml_text": xml_text,
                    "plan_text": plan_text,
                    "feedback_text": feedback_text,
                    "feedback_bullets": bullet_count,
                    **reward_info,
                })

            all_initial_grades[group_idx] = group_grades

        # ==============================================================
        # PHASE 4: GENERATE REFINED PLANS
        # For each initial plan, build a refinement prompt containing:
        #   - the research scenario
        #   - the rubric items (so the model knows what it's graded on)
        #   - the initial draft (so it can revise, not start from scratch)
        #   - the expert critique (extracted weaknesses/fixes/desiderata)
        # Then sample one revised plan per initial plan.
        # ==============================================================
        refined_samples_info: list[list[dict | None] | None] = [None] * len(batch_rows["Goal"])
        refined_prompt_tokens_store: list[list[list[int]] | None] = [None] * len(batch_rows["Goal"])

        if config.use_refinement:
            logger.info(f"Batch {real_batch}: Launching refinement generations...")

            refinement_tasks = []

            for group_idx, group_data in enumerate(batch_groups_data):
                if group_data is None or all_initial_grades[group_idx] is None:
                    continue

                goal = group_data["goal"]
                group_grades = all_initial_grades[group_idx]
                group_refine_prompt_tokens = []
                refined_samples_info[group_idx] = [None] * len(group_grades)
                refined_prompt_tokens_store[group_idx] = []

                for j, grade in enumerate(group_grades):
                    # Pass only the solution content as draft (strip <think> blocks
                    # to save tokens in the refinement prompt). Skip draft entirely
                    # for 0-scoring plans (grader failure or degenerate output).
                    if grade["rubric_score"] > 0.0:
                        draft = extract_solution_text(grade["plan_text"]) or grade["plan_text"]
                    else:
                        draft = None
                    refinement_prompt = build_self_teacher_prompt(
                        scenario=goal,
                        grader_feedback=grade["feedback_text"],
                        initial_draft=draft,
                        rubric_items=group_data["rubric"],
                        policy_output_mode="solution_only",
                        target_word_count=config.target_word_count,
                        max_solution_words=config.max_word_count,
                    )
                    refine_input = renderer.build_generation_prompt(
                        [{"role": "user", "content": refinement_prompt}]
                    )
                    refine_prompt_tokens = refine_input.to_ints()
                    group_refine_prompt_tokens.append(refine_prompt_tokens)

                    refinement_tasks.append(
                        asyncio.create_task(
                            _sample_refinement_with_timeout(
                                sampling_client=sampling_client,
                                refine_input=refine_input,
                                group_idx=group_idx,
                                sample_idx=j,
                                sampling_params=refinement_sampling_params,
                                timeout_sec=config.refinement_timeout_sec,
                            ),
                            name=f"refine_g{group_idx}_s{j}",
                        )
                    )

                refined_prompt_tokens_store[group_idx] = group_refine_prompt_tokens

            # Collect refinement results
            for r_task in asyncio.as_completed(refinement_tasks):
                group_idx, sample_idx, r_result = await r_task
                if r_result is None or refined_samples_info[group_idx] is None:
                    continue
                refined_seq = r_result.sequences[0]
                raw_refined = renderers.get_text_content(
                    renderer.parse_response(refined_seq.tokens)[0]
                )
                refined_plan = close_solution_tag(raw_refined)
                refined_samples_info[group_idx][sample_idx] = {
                    "tokens": refined_seq.tokens,
                    "logprobs": refined_seq.logprobs,
                    "text": refined_plan,       # repaired
                    "raw_text": raw_refined,     # original (for compliance)
                }

        # ==============================================================
        # PHASE 5: GRADE REFINED PLANS
        # Same detailed grader as Phase 2, applied to each refined plan.
        # Grader failures → zero-reward entries (symmetric with Phase 3).
        # ==============================================================
        all_refined_grades: list[list[dict | None] | None] = [None] * len(batch_rows["Goal"])

        if config.use_refinement:
            logger.info(f"Batch {real_batch}: Launching refined plan graders...")

            refined_grader_tasks = []

            for group_idx, group_data in enumerate(batch_groups_data):
                if group_data is None or refined_samples_info[group_idx] is None:
                    continue

                goal = group_data["goal"]
                rubric = group_data["rubric"]
                ref_sol = group_data["ref_sol"]
                all_refined_grades[group_idx] = [None] * len(refined_samples_info[group_idx])

                for j, r_sample in enumerate(refined_samples_info[group_idx]):
                    if r_sample is None:
                        continue

                    grader_prompt_text = build_detailed_grader_prompt(
                        scenario=goal,
                        rubric_items=rubric,
                        proposed_plan=r_sample["text"],
                        reference_solution=ref_sol,
                        fast_grader_mode=True,
                    )
                    grader_input = renderer.build_generation_prompt(
                        [{"role": "user", "content": grader_prompt_text}]
                    )

                    refined_grader_tasks.append(
                        asyncio.create_task(
                            _sample_grader_with_timeout(
                                grader_client=grader_client,
                                grader_input=grader_input,
                                group_idx=group_idx,
                                sample_idx=j,
                                grader_sampling_params=grader_sampling_params,
                                timeout_sec=config.grader_timeout_sec,
                            ),
                            name=f"ref_grader_g{group_idx}_s{j}",
                        )
                    )

            # Collect refined grader results
            for rg_task in asyncio.as_completed(refined_grader_tasks):
                group_idx, sample_idx, rg_result = await rg_task
                if all_refined_grades[group_idx] is None:
                    continue

                r_sample = refined_samples_info[group_idx][sample_idx]
                plan_text = r_sample["text"] if r_sample is not None else ""
                raw_text = r_sample["raw_text"] if r_sample is not None else ""

                def _zero_reward_grade(pt: str, rt: str, reason: str) -> dict:
                    return {
                        "xml_text": "",
                        "plan_text": pt,
                        "rubric_score": 0.0,
                        "word_count": len(pt.strip().split()),
                        "solution_word_count": 0,
                        "is_compliant": sdpo_check_format_compliance(rt, config.max_word_count) if rt else False,
                        "format_penalty": 0.2,
                        "length_bonus": 0.0,
                        "final_reward": 0.0,
                        "failure_reason": reason,
                    }

                if rg_result is None:
                    all_refined_grades[group_idx][sample_idx] = _zero_reward_grade(
                        plan_text, raw_text, "grader_timeout_or_error"
                    )
                    continue
                try:
                    parsed_msg, _ = renderer.parse_response(rg_result.sequences[0].tokens)
                    xml_text = renderers.get_text_content(parsed_msg)
                except Exception:
                    logger.exception(
                        "Failed to parse refined grader output for group_idx=%s sample_idx=%s",
                        group_idx, sample_idx,
                    )
                    all_refined_grades[group_idx][sample_idx] = _zero_reward_grade(
                        plan_text, raw_text, "grader_parse_error"
                    )
                    continue

                reward_info = compute_reward(plan_text, xml_text, config, raw_plan_text=raw_text)
                all_refined_grades[group_idx][sample_idx] = {
                    "xml_text": xml_text,
                    "plan_text": plan_text,
                    **reward_info,
                }

        # ==============================================================
        # PHASE 6: COMPUTE GRPO ADVANTAGES & BUILD TRAINING DATUMS
        #
        # Two SEPARATE GRPO groups per goal, each with independent
        # advantage normalization (advantage = reward - group_mean):
        #   Group A (initial):  trains the model to produce better first drafts
        #   Group B (refined):  trains the model to produce better revisions
        #
        # Both groups' datums go into the SAME optimization step so the
        # model learns from both generation modes simultaneously.
        #
        # Delta rubric (refined - initial) is logged as a diagnostic
        # but NOT used for training — we use absolute rewards for GRPO.
        # ==============================================================
        logger.info(f"Batch {real_batch}: Computing advantages...")

        training_datums = []
        batch_logs_to_save = []
        dropped_samples = 0
        drop_reasons = {}

        # Batch-level diagnostics
        batch_initial_rewards = []
        batch_initial_rubrics = []
        batch_refined_rewards = []
        batch_refined_rubrics = []
        batch_deltas = []
        batch_advantages = []
        batch_word_counts = []
        batch_format_penalties = []
        num_total_samples = 0
        num_valid_samples = 0
        num_initial_datums = 0
        num_refined_datums = 0
        feedback_bullets_all = []

        for group_idx, group_data in enumerate(batch_groups_data):
            if group_data is None or all_initial_grades[group_idx] is None:
                continue

            # ----- GROUP A: INITIAL PLANS -----
            # GRPO advantage = reward_i - mean(rewards in this group)
            initial_rewards = []
            initial_valid = []

            for j, grade in enumerate(all_initial_grades[group_idx]):
                num_total_samples += 1
                plan_text = grade["plan_text"]
                batch_word_counts.append(grade["word_count"])
                batch_format_penalties.append(grade["format_penalty"])
                batch_initial_rubrics.append(grade["rubric_score"])

                sample_info = group_data["samples_info"][j]
                valid, drop_reason = is_valid_sample(plan_text, sample_info.get("logprobs"), config)
                if not valid:
                    dropped_samples += 1
                    drop_reasons[drop_reason] = drop_reasons.get(drop_reason, 0) + 1
                    continue

                initial_rewards.append(grade["final_reward"])
                batch_initial_rewards.append(grade["final_reward"])
                initial_valid.append({
                    "sample_info": sample_info,
                    "reward": grade["final_reward"],
                })
                num_valid_samples += 1

                batch_logs_to_save.append({
                    "batch_idx": real_batch,
                    "group_idx": group_idx,
                    "sample_idx": j,
                    "group_type": "initial",
                    "raw_policy_output": group_data["samples_info"][j]["raw_text"],
                    "normalized_for_grader": plan_text,
                    "grader_output": grade["xml_text"],
                    "rubric_score": grade["rubric_score"],
                    "format_penalty": grade["format_penalty"],
                    "final_reward": grade["final_reward"],
                    "word_count": grade["word_count"],
                    "is_compliant": grade["is_compliant"],
                    "feedback_text": grade["feedback_text"],
                    "feedback_bullets": grade["feedback_bullets"],
                })

            feedback_bullets_all.extend(
                g["feedback_bullets"] for g in all_initial_grades[group_idx]
            )

            # Compute GRPO advantages for initial group
            if initial_rewards and not is_eval:
                mean_r = np.mean(initial_rewards)
                if config.normalize_advantages:
                    std_r = np.std(initial_rewards)
                    advantages = [(r - mean_r) / std_r for r in initial_rewards] if std_r > 1e-8 else [0.0] * len(initial_rewards)
                else:
                    advantages = [(r - mean_r) for r in initial_rewards]
                batch_advantages.extend(advantages)

                # Need ≥2 valid samples and non-degenerate advantages to produce datums
                if not all(a == 0.0 for a in advantages) and len(initial_valid) >= 2:
                    prompt_tokens = [int(t) for t in group_data["prompt_tokens"]]
                    for k, sample in enumerate(initial_valid):
                        datum = create_training_datum(
                            prompt_tokens, sample["sample_info"], advantages[k]
                        )
                        if datum is not None:
                            training_datums.append(datum)
                            num_initial_datums += 1
                        else:
                            dropped_samples += 1

            # ----- GROUP B: REFINED PLANS -----
            # Same GRPO logic but with independent advantage normalization.
            # refinement_group_weight scales the advantages (default 1.0 = equal weight).
            if (
                config.use_refinement
                and all_refined_grades[group_idx] is not None
                and refined_samples_info[group_idx] is not None
                and refined_prompt_tokens_store[group_idx] is not None
            ):
                refined_rewards = []
                refined_valid = []

                for j, grade in enumerate(all_refined_grades[group_idx]):
                    if grade is None or refined_samples_info[group_idx][j] is None:
                        continue

                    num_total_samples += 1
                    plan_text = grade["plan_text"]
                    batch_word_counts.append(grade["word_count"])
                    batch_format_penalties.append(grade["format_penalty"])
                    batch_refined_rubrics.append(grade["rubric_score"])

                    # Delta = refined rubric - initial rubric (diagnostic only, not used for training)
                    orig_rubric = all_initial_grades[group_idx][j]["rubric_score"]
                    delta = grade["rubric_score"] - orig_rubric
                    batch_deltas.append(delta)

                    r_sample = refined_samples_info[group_idx][j]
                    valid, drop_reason = is_valid_sample(plan_text, r_sample.get("logprobs"), config)
                    if not valid:
                        dropped_samples += 1
                        drop_reasons[drop_reason] = drop_reasons.get(drop_reason, 0) + 1
                        continue

                    refined_rewards.append(grade["final_reward"])
                    batch_refined_rewards.append(grade["final_reward"])
                    refined_valid.append({
                        "sample_info": r_sample,
                        "reward": grade["final_reward"],
                        "prompt_token_idx": j,
                    })
                    num_valid_samples += 1

                    batch_logs_to_save.append({
                        "batch_idx": real_batch,
                        "group_idx": group_idx,
                        "sample_idx": j,
                        "group_type": "refined",
                        "raw_policy_output": r_sample["raw_text"],
                        "normalized_for_grader": plan_text,
                        "grader_output": grade["xml_text"],
                        "rubric_score": grade["rubric_score"],
                        "format_penalty": grade["format_penalty"],
                        "final_reward": grade["final_reward"],
                        "word_count": grade["word_count"],
                        "is_compliant": grade["is_compliant"],
                        "delta_rubric": delta,
                    })

                # Compute GRPO advantages for refined group
                if refined_rewards and not is_eval:
                    mean_r = np.mean(refined_rewards)
                    if config.normalize_advantages:
                        std_r = np.std(refined_rewards)
                        advantages = [(r - mean_r) / std_r for r in refined_rewards] if std_r > 1e-8 else [0.0] * len(refined_rewards)
                    else:
                        advantages = [(r - mean_r) for r in refined_rewards]
                    if config.refinement_group_weight != 1.0:
                        advantages = [a * config.refinement_group_weight for a in advantages]
                    batch_advantages.extend(advantages)

                    if not all(a == 0.0 for a in advantages) and len(refined_valid) >= 2:
                        for k, sample in enumerate(refined_valid):
                            pt_idx = sample["prompt_token_idx"]
                            refine_prompt_tokens = [
                                int(t)
                                for t in refined_prompt_tokens_store[group_idx][pt_idx]
                            ]
                            datum = create_training_datum(
                                refine_prompt_tokens,
                                sample["sample_info"],
                                advantages[k],
                            )
                            if datum is not None:
                                training_datums.append(datum)
                                num_refined_datums += 1
                            else:
                                dropped_samples += 1

        # ==============================================================
        # BATCH SUMMARY & LOGGING
        # ==============================================================
        batch_summary = {
            "batch_idx": real_batch,
            "group_initial/rubric_mean": float(np.mean(batch_initial_rubrics)) if batch_initial_rubrics else 0.0,
            "group_initial/reward_mean": float(np.mean(batch_initial_rewards)) if batch_initial_rewards else 0.0,
            "group_refined/rubric_mean": float(np.mean(batch_refined_rubrics)) if batch_refined_rubrics else 0.0,
            "group_refined/reward_mean": float(np.mean(batch_refined_rewards)) if batch_refined_rewards else 0.0,
            "refinement/delta_rubric_mean": float(np.mean(batch_deltas)) if batch_deltas else 0.0,
            "refinement/improvement_rate": float(np.mean([1.0 if d > 0 else 0.0 for d in batch_deltas])) if batch_deltas else 0.0,
            "refinement/regression_rate": float(np.mean([1.0 if d < 0 else 0.0 for d in batch_deltas])) if batch_deltas else 0.0,
            "refinement/feedback_bullets_mean": float(np.mean(feedback_bullets_all)) if feedback_bullets_all else 0.0,
            "advantage_mean": float(np.mean(batch_advantages)) if batch_advantages else 0.0,
            "advantage_std": float(np.std(batch_advantages)) if batch_advantages else 0.0,
            "length_mean": float(np.mean(batch_word_counts)) if batch_word_counts else 0.0,
            "format_penalty_mean": float(np.mean(batch_format_penalties)) if batch_format_penalties else 0.0,
            "format_compliance_rate": float(np.mean([1.0 if p == 0.0 else 0.0 for p in batch_format_penalties])) if batch_format_penalties else 0.0,
            "format_violation_rate": float(np.mean([1.0 if p > 0.0 else 0.0 for p in batch_format_penalties])) if batch_format_penalties else 0.0,
            "samples/generated": num_total_samples,
            "samples/valid": num_valid_samples,
            "datums/initial_count": num_initial_datums,
            "datums/refined_count": num_refined_datums,
            "group_initial/reward_std": float(np.std(batch_initial_rewards)) if batch_initial_rewards else 0.0,
            "group_refined/reward_std": float(np.std(batch_refined_rewards)) if batch_refined_rewards else 0.0,
            "drops/total": dropped_samples,
            "drops/too_short": drop_reasons.get("too_short", 0),
            "drops/bare_tag": drop_reasons.get("bare_tag", 0),
            "drops/no_logprobs": drop_reasons.get("no_logprobs", 0),
        }

        # Write per-sample logs
        if batch_logs_to_save:
            if is_eval:
                log_path = os.path.join(
                    config.log_path, f"evaluation/eval_logs({actual_batch - 1}).jsonl"
                )
            else:
                log_path = os.path.join(config.log_path, "train/training_logs.jsonl")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a") as f:
                for log_item in batch_logs_to_save:
                    f.write(json.dumps(log_item) + "\n")

        # Write batch summary
        if batch_summary:
            if is_eval:
                summary_path = os.path.join(
                    config.log_path, f"evaluation/eval_batch_summary({actual_batch - 1}).jsonl"
                )
            else:
                summary_path = os.path.join(config.log_path, "train/batch_summary.jsonl")
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, "a") as f:
                f.write(json.dumps(batch_summary) + "\n")

        logger.info(
            f"Batch {real_batch}: "
            f"init_rubric={batch_summary['group_initial/rubric_mean']:.3f} "
            f"ref_rubric={batch_summary['group_refined/rubric_mean']:.3f} "
            f"delta={batch_summary['refinement/delta_rubric_mean']:+.3f} "
            f"impr_rate={batch_summary['refinement/improvement_rate']:.1%} "
            f"datums={num_initial_datums}+{num_refined_datums}"
        )

        # --- EVAL MODE: skip optimization ---
        if is_eval:
            logger.info("EVAL MODE: skipping optimization step.")
            continue

        # ==============================================================
        # OPTIMIZATION STEP
        # All training datums (initial + refined) go into one
        # forward/backward pass with PPO-clipped loss, then one Adam step.
        # ==============================================================
        if not training_datums:
            logger.warning("No valid datums this batch. Skipping optimization.")
            continue

        try:
            fwd_bwd_future = await training_client.forward_backward_async(
                training_datums,
                loss_fn="ppo",
                loss_fn_config={
                    "clip_low_threshold": 1 - config.clip_eps,
                    "clip_high_threshold": 1 + config.clip_eps,
                },
            )
            optim_step_future = await training_client.optim_step_async(adam_params)

            t0 = time.time()
            _fwd_bwd_result = await asyncio.wait_for(
                fwd_bwd_future.result_async(),
                timeout=config.train_timeout_sec,
            )
            logger.info("Forward/Backward took %.2fs", time.time() - t0)

            t1 = time.time()
            _optim_result = await asyncio.wait_for(
                optim_step_future.result_async(),
                timeout=config.train_timeout_sec,
            )
            logger.info("Optim step took %.2fs", time.time() - t1)
        except asyncio.TimeoutError:
            logger.exception("Training step timed out")
            continue
        except Exception:
            logger.exception("Training step failed")
            continue

        # Log metrics to wandb / ml_logger
        metrics: dict[str, float] = {
            "progress/batch": real_batch,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (real_batch + 1) / n_train_batches,
            "time/total": time.time() - t_start,
            "reward/initial_mean": batch_summary["group_initial/reward_mean"],
            "reward/refined_mean": batch_summary["group_refined/reward_mean"],
            "refinement/delta_rubric_mean": batch_summary["refinement/delta_rubric_mean"],
            "refinement/improvement_rate": batch_summary["refinement/improvement_rate"],
            "dropped_samples": dropped_samples,
        }
        ml_logger.log_metrics(metrics, step=real_batch)

        # Periodic checkpoint (saved AFTER training, so batch N is fully complete)
        if config.save_every > 0 and batch_idx % config.save_every == 0 and batch_idx > 0:
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{real_batch:06d}_{config.today_date}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": real_batch},
            )

    # --- Save final checkpoint ---
    if not is_eval and last_real_batch >= 0:
        await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name=f"{last_real_batch:06d}_final_{config.today_date}",
            log_path=config.log_path,
            kind="both",
            loop_state={"batch": last_real_batch},
        )
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
