import logging
import time
import re
import textwrap
import asyncio
import math
from collections import deque
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
from co_scientist.shared.api_profiles import create_service_client
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from datasets import load_dataset


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    # ===== Runtime / IO =====
    # Tinker server endpoint; `None` uses local/default server config.
    base_url: str | None = None
    # Optional named API profile; resolves CO_SCIENTIST_API_KEY_<PROFILE>.
    api_profile: str | None = None
    # Root directory for checkpoints, metrics, and per-sample logs.
    log_path: str = "/home/silas/co-scientist-project/runs/2026/2/SDPO/27(rich)"
    # Student policy model used for sampling and training.
    model_name: str = "Qwen/Qwen3-30B-A3B"
    # Judge model used to grade rubric satisfaction.
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"

    # ===== Evaluation mode =====
    # If True, run eval-only pass and skip optimization updates.
    run_eval: bool = False
    # Number of epochs between eval runs (when eval flow is used).
    eval_epoch: int = 1
    # Metric logging verbosity:
    # - "compact": essential monitoring metrics only
    # - "full": include all diagnostic metrics
    metrics_mode: str = "compact"
    # Optional cap on number of batches to execute in this run.
    # Set None for full training runs.
    max_batches: int | None = None

    # ===== Dataset selection (enable exactly one) =====
    # Use ML split from facebook/research-plan-gen.
    ml_data: bool = True
    # Use arXiv split from facebook/research-plan-gen.
    arxiv_data: bool = False
    # Use PubMed split from facebook/research-plan-gen.
    pubmed_data: bool = False

    # ===== Optimization / rollout scale =====
    # Number of research goals per batch.
    batch_size: int = 64
    # Number of sampled candidates per goal (GRPO group size).
    group_size: int = 8
    # Adam learning rate for policy update.
    learning_rate: float = 5e-6
    # PPO/GRPO clip epsilon for importance-ratio clipping.
    clip_eps: float = 0.2

    # ===== Model/context/token budgets =====
    # Max model context length loaded for training.
    max_length: int = 32768
    # LoRA rank for adapter training (higher rank => more capacity/memory).
    lora_rank: int = 64
    # Checkpoint cadence in batches; 0 disables periodic saving.
    save_every: int = 0
    # Max generation tokens for policy outputs (think + solution combined).
    max_tokens: int = 4096
    # Max generation tokens for grader XML output.
    grader_max_tokens: int = 12288
    # Grader prompt contract:
    # - "detailed": full critique + fixes + confidence levels
    # - "compact_confidence_only": confidence levels only per rubric/desiderata
    grader_prompt_mode: str = "detailed"
    # If True, use concise grader prompt contract for faster grading latency.
    fast_grader_mode: bool = False
    # Upper cap on grader max tokens when fast_grader_mode is enabled.
    fast_grader_max_tokens: int = 12288
    # Target upper bound for words inside each rubric-item <reasoning> block in fast mode.
    fast_grader_reasoning_max_words_per_item: int = 60
    # Target upper bound for words in sample-level <desiderata_review><summary>.
    fast_grader_summary_max_words: int = 45
    # Extra clamp for SDPO feedback chars in fast mode to keep teacher prompts short.
    fast_grader_feedback_max_chars: int = 1200
    # Policy sampling temperature.
    temperature: float = 1.0
    # Grader sampling temperature (keep near deterministic).
    grader_temperature: float = 0.0
    # Auto-clamp batch/group size to server capabilities.
    use_server_cap_limits: bool = True
    # Toggle trainable parameter groups.
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True

    # ===== Reward, format, and trainability policy =====
    # Max allowed words inside <solution> for compliance checks.
    max_word_count: int = 750
    # Preferred solution length center for composite reward bonus.
    target_word_count: int = 600
    # Width of Gaussian length bonus around target_word_count.
    scale_length_bonus: float = 120.0
    # Weight of length bonus in composite reward.
    scaling_factor: float = 0.14
    # Minimum solution words for a sample to be trainable after warmup.
    min_words: int = 180
    # If True, use a lower min_words threshold for early stabilization batches.
    min_words_warmup_enabled: bool = True
    # Number of early batches that use min_words_warmup_value.
    min_words_warmup_batches: int = 30
    # Early-stage min_words used while model learns stable format behavior.
    min_words_warmup_value: int = 120
    # If True, drop samples where <solution> is just copied template/placeholder text.
    drop_template_copy_solutions: bool = True
    # Reward composition:
    # - "rubric_only": rubric score only
    # - "composite": rubric + length bonus - format penalty
    # - "paper": rubric - hard format penalty
    reward_mode: str = "composite"
    # Penalty subtracted when sample is format-noncompliant.
    paper_format_penalty: float = 1.0
    # If True, noncompliant outputs are dropped from training datums.
    drop_noncompliant_samples: bool = True
    # If True, perform one retry sample when output is format-noncompliant.
    format_retry_enabled: bool = False
    # Maximum retry attempts for format-noncompliant outputs.
    format_retry_max_attempts: int = 0
    # Retry sampling temperature for format recovery.
    format_retry_temperature: float = 0.7
    # Retry sampling max tokens (0 uses max_tokens).
    format_retry_max_tokens: int = 1536
    # Refresh policy sampling client every N train batches (1 = every batch, on-policy strict).
    sampling_client_refresh_interval_batches: int = 1
    # Max concurrent policy requests in async launch stage.
    max_concurrent_policy_requests: int = 32
    # Max concurrent grader requests in async launch stage.
    max_concurrent_grader_requests: int = 96
    # If True, grader sees extracted <solution> only (not raw full output).
    grade_solution_only: bool = True
    # GRPO advantage normalization within group.
    # False uses paper-style centered but unnormalized advantages.
    normalize_grpo_advantages: bool = False
    # Output contract mode:
    # - "solution_only": require only <solution> block
    # - "think_solution": require <think> then <solution>
    policy_output_mode: str = "solution_only"
    # Soft think-length threshold (words) where think penalty starts.
    max_think_words_soft: int = 60
    # Hard think-length threshold (words) for potential trainability drop.
    max_think_words_hard: int = 140
    # Overflow normalization factor for think penalty growth.
    think_penalty_words_per_step: float = 10.0
    # Base slope of think penalty after soft threshold.
    think_penalty_slope: float = 0.12
    # Nonlinear exponent for think penalty growth.
    think_penalty_power: float = 1.3
    # Upper clamp for think penalty per sample.
    think_penalty_max: float = 1.0
    # Handling mode for think_too_long samples:
    # - "disabled": never drop for think length (penalty-only)
    # - "adaptive": drop but cap drop fraction per batch
    # - "hard": always drop when think exceeds hard threshold
    think_too_long_drop_mode: str = "adaptive"
    # In adaptive mode, maximum fraction of trainability checks dropped for think_too_long.
    think_too_long_max_drop_rate: float = 0.20
    # Soft threshold (words) for all content outside <solution>.
    max_non_solution_words_soft: int = 140
    # Hard threshold (words) for outside-<solution> verbosity trainability drop.
    max_non_solution_words_hard: int = 260
    # Overflow normalization factor for non-solution verbosity penalty growth.
    non_solution_penalty_words_per_step: float = 20.0
    # Base slope of non-solution verbosity penalty after soft threshold.
    non_solution_penalty_slope: float = 0.05
    # Nonlinear exponent for non-solution verbosity penalty growth.
    non_solution_penalty_power: float = 1.2
    # Upper clamp for non-solution verbosity penalty per sample.
    non_solution_penalty_max: float = 1.0
    # Handling mode for non_solution_too_long samples:
    # - "disabled": never drop for outside-<solution> verbosity (penalty-only)
    # - "adaptive": drop but cap drop fraction per batch
    # - "hard": always drop when over hard threshold
    non_solution_too_long_drop_mode: str = "adaptive"
    # In adaptive mode, maximum fraction of trainability checks dropped for non_solution_too_long.
    non_solution_too_long_max_drop_rate: float = 0.10

    # ===== SDPO (2601.20802) controls =====
    # Enable SDPO teacher-student distillation term.
    use_sdpo: bool = True
    # Success cutoff mode:
    # - "adaptive_quantile": threshold from score distribution
    # - "fixed": use sdpo_success_threshold
    sdpo_success_mode: str = "adaptive_quantile"
    # Quantile used when success mode is adaptive (e.g., 0.90 => top 10%).
    sdpo_success_adaptive_quantile: float = 0.90
    # Number of previous batches to pool for adaptive cutoff estimation.
    sdpo_success_history_batches: int = 3
    # Minimum pooled samples required before trusting history-only quantile.
    sdpo_success_min_samples: int = 128
    # If history pool is too small, allow current batch scores as bootstrap.
    sdpo_success_use_current_batch_bootstrap: bool = True
    # Minimum successful-group rate target; cutoff can back off to satisfy this.
    sdpo_success_min_rate: float = 0.05
    # Absolute minimum number of successful groups (if enough candidates exist).
    sdpo_success_min_groups: int = 1
    # Clamp adaptive/fixed success cutoff to [floor, ceiling].
    sdpo_success_floor: float = 0.60
    sdpo_success_ceiling: float = 1.0
    # Fixed success cutoff (used only when sdpo_success_mode == "fixed").
    sdpo_success_threshold: float = 0.70
    # Advantage mixing weight:
    # mixed_adv = lambda * GRPO_adv + (1-lambda) * SDPO_adv
    sdpo_grpo_mix_lambda: float = 0.30
    # Interpolate self-teacher logprobs with frozen initial teacher logprobs.
    sdpo_teacher_reg_alpha: float = 0.01
    # Clip SDPO advantages to [-sdpo_adv_clip, +sdpo_adv_clip].
    sdpo_adv_clip: float = 5.0
    # If True, apply SDPO term only to tokens inside extracted <solution> span.
    sdpo_solution_only_tokens: bool = True
    # Warning threshold when solution masks are frequently empty/missing.
    sdpo_solution_mask_warn_zero_rate: float = 0.20
    # If True, try loose level-based scoring when strict grader XML parsing fails.
    allow_grader_xml_fallback: bool = True
    # Minimum count of <level> tags required for fallback scoring.
    grader_xml_fallback_min_levels: int = 7
    # Max chars of grader feedback passed into self-teacher prompt (truncate if larger).
    sdpo_feedback_max_chars: int = 12000
    # Maximum paired weakness/fix bullets extracted per rubric item for SDPO feedback.
    sdpo_feedback_max_bullets_per_item: int = 6
    # Max number of rubric items to include in teacher feedback (after filtering).
    sdpo_teacher_max_criticized_items: int = 10
    # If True, only rubric items with at least one low desiderata level (<=2) feed teacher critique.
    sdpo_teacher_focus_low_confidence_items_only: bool = True
    # Include sample-level weaknesses/fixes (<sample_review>) in teacher prompt feedback.
    sdpo_teacher_include_sample_review: bool = True
    # Include per-item reasoning weaknesses/fixes from <reasoning> in teacher prompt feedback.
    sdpo_teacher_include_item_reasoning_feedback: bool = True
    # Include per-item desiderata review/suggestion blocks in teacher prompt feedback.
    sdpo_teacher_include_item_desiderata_review: bool = True
    # Include final sample-level desiderata summary in teacher prompt feedback.
    sdpo_teacher_include_global_desiderata_summary: bool = True

    # Date stamp used in final checkpoint naming.
    today_date = time.strftime("%Y-%m-%d", time.localtime())
   

# ============================================================
# For sampling
# ============================================================

def build_research_plan_prompt(
    scenario: str,
    examples: list[dict] | None = None,
    policy_output_mode: str = "solution_only",
) -> str:


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

    # Actual scenario
    prompt += f"""
            Here is the research scenario.
            Scenario: {scenario}
            """

    # Global instructions
    shared_instructions = textwrap.dedent(f"""
        Here is the research scenario.
        Scenario: {scenario}
        
        # Instructions
        First, come up with a detailed research plan to address the scenario based on the following overall solution guidelines:
        - The plan should address the goals of the scenario, and account for all constraints and confounders.
        - Do NOT just say WHAT you will do. Explain HOW you will do it and WHY it is needed. Provide clear explanation and justification for each proposed step. The solution inside <solution></solution> tags should be readable for humans, and not in XML itself.
        - The phrasing should NOT be verbose, and NOT be in past tense, as in "the author's approach" but rather in present tense, as how you would approach the problem.
        - Do not claim to have done any experiments or have results, just provide the plan.
        - Do not add self-proclaimed praises of your solution. For example do NOT say yourself it satisfies some desiderata, we will let the evaluator decide that.
        - Aim for around 550-700 words in <solution> with concrete, implementation-ready detail.
    """).strip()

    if policy_output_mode == "solution_only":
        prompt += shared_instructions + "\n\n" + textwrap.dedent("""
            Return exactly one XML block using ONLY <solution> tags.
            The first output token must be <solution>.
            The final output token must be </solution>.
            Put the complete research plan between those tags.
            Do not output any text before <solution> or after </solution>.
            Do not copy instruction/template text.
        """).strip()
    elif policy_output_mode == "think_solution":
        prompt += shared_instructions + "\n\n" + textwrap.dedent("""
            Return exactly two XML blocks and close all tags:
            1) 
            <think>
            Brief reasoning only, at most 60 words.
            </think>
            2) 
            <solution>
            Complete and self-contained research plan.
            </solution>

            Important:
            - Keep all substantial content inside <solution>.
            - Always include both blocks exactly once.
            - Keep <solution> around 550-700 words.
            - Do not copy instruction/template text into either block.
        """).strip()
    else:
        raise ValueError(
            f"Invalid policy_output_mode={policy_output_mode!r}. Expected 'solution_only' or 'think_solution'."
        )

    return prompt

# ============================================================
# For grading
# ============================================================


def build_grader_prompt(
    scenario: str,
    rubric_items: list[str],
    proposed_plan: str,
    reference_solution: str | None = None,
    grader_prompt_mode: str = "detailed",
    fast_grader_mode: bool = True,
    fast_reasoning_max_words_per_item: int = 90,
    fast_summary_max_words: int = 70,
) -> str:
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

    if grader_prompt_mode == "compact_confidence_only":
        prompt += textwrap.dedent(f"""
            # Proposed Research Plan
            {proposed_plan}

            # Instructions
            First, come up with weaknesses of the proposed plan specific to the scenario. Then, return the following nested XML block for each of the grading items (always close opened XML tags):

            <rubric>
                <item num=1>
                    <criteria>Repeat the rubric item string you are checking here. </criteria>
                    <reasoning>
                    Analyze how well the proposed plan satisfies EACH of the following 7 GENERAL DESIDERATA with respect to the rubric item, using an integer satisfaction level. 
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
                    
                    - Be skeptical, careful, and come up with valid criticisms. Be as strict as possible, while being unbiased and reasonable. 
                    - Note that the plan should not just say it satisfies these desiderata, don’t be fooled by that. Check carefully WHETHER, HOW and WHY the proposed plan meets each desiderata for this rubric item one by one. 
                    - Based on the above analysis, list the satisfaction level for each desiderata. Don't be lazy and assign the same level to all desiderata. Be precise and careful.
                    </reasoning>
                    <desiderata num=1>
                    Repeat for all 7 desiderata:
                        <level>[Satisfaction level for desiderata, put a single integer from 0, 1, 2, 3]</level>
                    </desiderata>
                </item>
                
                ... Similarly, for all rubric items...
            </rubric>
        """).strip()
        return prompt

    if grader_prompt_mode != "detailed":
        raise ValueError(
            f"Invalid grader_prompt_mode={grader_prompt_mode!r}. "
            "Expected 'detailed' or 'compact_confidence_only'."
        )

    min_weakness_bullets = 1 if fast_grader_mode else 2
    min_fix_bullets = 1 if fast_grader_mode else 2
    reasoning_budget = max(40, int(fast_reasoning_max_words_per_item))
    summary_budget = max(30, int(fast_summary_max_words))

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
          3) output exactly 7 <desiderata num=1..7> blocks.
          4) each desiderata block must contain exactly one integer <level> in {{0,1,2,3}}.
          5) after the 7 desiderata blocks, add one per-item <desiderata_review> block:
             <desiderata_review>
               <summary>Briefly explain how the 7 desiderata behave for this rubric item.</summary>
               <suggested_new_desiderata>Optional: suggest better/new desiderata for this case; write "none" if none.</suggested_new_desiderata>
             </desiderata_review>
        - Add one sample-level summary after all items:
          <desiderata_review><summary>Brief overall note (about {summary_budget} words max).</summary></desiderata_review>
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
            <desiderata num=1><level>0|1|2|3</level></desiderata>
            <desiderata num=2><level>0|1|2|3</level></desiderata>
            <desiderata num=3><level>0|1|2|3</level></desiderata>
            <desiderata num=4><level>0|1|2|3</level></desiderata>
            <desiderata num=5><level>0|1|2|3</level></desiderata>
            <desiderata num=6><level>0|1|2|3</level></desiderata>
            <desiderata num=7><level>0|1|2|3</level></desiderata>
            <desiderata_review>
              <summary>[how the 7 desiderata behave for this item]</summary>
              <suggested_new_desiderata>[optional; "none" if no suggestion]</suggested_new_desiderata>
            </desiderata_review>
          </item>
          ...
          <desiderata_review>
            <summary>[one concise overall review]</summary>
          </desiderata_review>
        </rubric>
    """).strip()
    return prompt

def compute_rubric_reward_from_xml(xml_text: str) -> float:
    """
    Compute rubric score with non-linear level mapping.

    For each rubric item:
        - map each level using level_map
        - item_score = mean(mapped levels)
    Total score = mean(item_score over items)

    Returns a float in [0, 1].
    """

    level_map = {
        0: 0.0,
        1: 0.2,
        2: 0.6,
        3: 1.0,
    }

    # Find all rubric items
    item_blocks = re.findall(
        r"<item\s+num=.*?>.*?</item>",
        xml_text,
        flags=re.DOTALL
    )

    if not item_blocks:
        return 0.0

    item_scores = []

    for item_xml in item_blocks:
        # Extract levels inside this item
        levels = re.findall(r"<level>(\d+)</level>", item_xml)
        levels = [int(l) for l in levels if l.isdigit()]

        if not levels:
            continue

        # Pad / truncate to 7 desiderata for safety
        if len(levels) < 7:
            levels = levels + [0] * (7 - len(levels))
        levels = levels[:7]

        # Non-linear mapping
        mapped = [level_map.get(l, 0.0) for l in levels]

        item_score = sum(mapped) / len(mapped)
        item_scores.append(item_score)

    if not item_scores:
        return 0.0

    return sum(item_scores) / len(item_scores)


def is_valid_grader_xml(xml_text: str) -> bool:
    """Basic structural validation for grader XML before reward/distillation use."""
    if re.search(r"<rubric\b", xml_text, flags=re.IGNORECASE) is None:
        return False

    item_blocks = re.findall(
        r"<item\s+num=.*?>.*?</item>",
        xml_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not item_blocks:
        return False

    return any(
        re.search(r"<level>\s*[0-3]\s*</level>", item_xml, flags=re.IGNORECASE) is not None
        for item_xml in item_blocks
    )


def check_format_compliance(text: str, max_words: int) -> bool:
    """
    Checks Section 3.2 constraints:
    1. Content must be within <solution></solution> tags.
    2. Content within tags must not exceed max_words (750).
    """
    content = extract_solution_text(text, require_tags=True)
    if content is None:
        return False

    # Simple whitespace split for word count
    word_count = len(content.split())

    if word_count > max_words:
        return False

    return True


def classify_format_noncompliance(text: str, max_words: int) -> str | None:
    """
    Return a detailed noncompliance reason for diagnostics, or None when compliant.
    """
    content = extract_solution_text(text, require_tags=True)
    if content is not None:
        if len(content.split()) > max_words:
            return "solution_too_long"
        return None

    lower = text.lower()
    has_open = "<solution>" in lower
    has_close = "</solution>" in lower
    if has_open and not has_close:
        return "missing_solution_close_tag"
    if has_close and not has_open:
        return "missing_solution_open_tag"
    if not has_open and not has_close:
        return "missing_solution_tags"
    return "malformed_solution_tags"


def normalize_policy_output_for_grader(plan_text: str) -> str:
    """
    Best-effort XML repair used only for grader input robustness.
    Training compliance checks must still use the raw policy output.
    """
    normalized_plan = plan_text.rstrip()
    lower_plan = normalized_plan.lower()
    if "<solution>" not in lower_plan:
        normalized_plan = normalized_plan + "\n<solution>\n</solution>"
    elif "</solution>" not in lower_plan:
        normalized_plan = normalized_plan + "\n</solution>"
    return normalized_plan


def _strip_think_blocks(text: str) -> str:
    """
    Remove closed <think>...</think> spans so nested/quoted <solution> tags
    inside thinking content are never treated as the final plan.
    """
    return re.sub(
        r"<think\b[^>]*>.*?</think>",
        " ",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )


def _strip_solution_blocks(text: str) -> str:
    """
    Remove closed <solution>...</solution> spans so quoted/embedded <think> tags
    inside solution content are ignored when measuring think length.
    """
    return re.sub(
        r"<solution\b[^>]*>.*?</solution>",
        " ",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )


def extract_solution_text(plan_text: str, require_tags: bool = False) -> str | None:
    """Extract plain content from <solution> tags; optionally require tags."""
    pattern = r"<solution>\s*(.*?)\s*</solution>"
    candidate_text = _strip_think_blocks(plan_text)
    matches = re.findall(pattern, candidate_text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        # Use the last block to be robust to stray earlier XML fragments.
        return matches[-1].strip()
    if require_tags:
        return None
    return plan_text.strip()


def extract_think_text(plan_text: str, require_tags: bool = False) -> str | None:
    """
    Extract plain content from <think> tags; optionally require tags.
    Returns empty string when tags are not required and no <think> block exists.
    """
    pattern = r"<think>\s*(.*?)\s*</think>"
    candidate_text = _strip_solution_blocks(plan_text)
    matches = re.findall(pattern, candidate_text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    if require_tags:
        return None
    return ""


def extract_non_solution_text(plan_text: str) -> str:
    """
    Return all textual content outside <solution>...</solution> blocks.
    XML tags are stripped before word counting.
    """
    candidate_text = _strip_solution_blocks(plan_text)
    no_xml_tags = re.sub(r"</?[^>]+>", " ", candidate_text)
    return re.sub(r"\s+", " ", no_xml_tags).strip()


def is_template_copy_solution(solution_text: str | None) -> bool:
    """
    Detect trivial placeholder/template outputs copied from prompt scaffolding.
    """
    if solution_text is None:
        return False
    normalized = re.sub(r"\s+", " ", solution_text).strip().lower().rstrip(" .")
    if not normalized:
        return False

    exact_placeholders = {
        "a complete and self-contained research plan",
        "a complete and self-contained research plan in at most 750 words",
        "a complete and self-contained research plan in at most 600 words",
        "write your research plan here",
        "your research plan here",
    }
    if normalized in exact_placeholders:
        return True

    return (
        re.fullmatch(
            r"(?:a )?complete and self-contained research plan(?: in at most \d+ words)?",
            normalized,
        )
        is not None
    )


def _span_overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def find_solution_content_span(text: str) -> tuple[int, int] | None:
    """
    Return char-span (start, end) for the last <solution> content block that does not
    overlap any closed <think>...</think> span. Whitespace inside solution tags is trimmed.
    """
    think_spans = [
        (m.start(), m.end())
        for m in re.finditer(
            r"<think\b[^>]*>.*?</think>",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
    ]
    solution_matches = list(
        re.finditer(
            r"<solution\b[^>]*>(.*?)</solution>",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
    )
    if not solution_matches:
        return None

    candidate_spans: list[tuple[int, int]] = []
    for match in solution_matches:
        span = (match.start(1), match.end(1))
        if any(_span_overlaps(span, think_span) for think_span in think_spans):
            continue

        start, end = span
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            candidate_spans.append((start, end))

    if not candidate_spans:
        return None
    return candidate_spans[-1]


def build_solution_token_mask_from_tokens(
    *,
    tokenizer,
    sample_tokens: list[int],
) -> np.ndarray:
    """
    Build a token mask (1.0 for <solution> content, 0.0 elsewhere) for a sampled response.
    If span detection fails, returns all-zero mask.
    """
    if not sample_tokens:
        return np.zeros(0, dtype=np.float32)

    try:
        decoded = tokenizer.decode(
            sample_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except Exception:
        return np.zeros(len(sample_tokens), dtype=np.float32)

    solution_span = find_solution_content_span(decoded)
    if solution_span is None:
        return np.zeros(len(sample_tokens), dtype=np.float32)

    try:
        start_char, end_char = solution_span
        start_tok = len(tokenizer.encode(decoded[:start_char], add_special_tokens=False))
        end_tok = len(tokenizer.encode(decoded[:end_char], add_special_tokens=False))
    except Exception:
        return np.zeros(len(sample_tokens), dtype=np.float32)

    start_tok = max(0, min(start_tok, len(sample_tokens)))
    end_tok = max(start_tok, min(end_tok, len(sample_tokens)))
    mask = np.zeros(len(sample_tokens), dtype=np.float32)
    if end_tok > start_tok:
        mask[start_tok:end_tok] = 1.0
    return mask


def compute_final_reward(
    rubric_score: float,
    is_compliant: bool,
    solution_word_count: int,
    config: Config,
) -> tuple[float, float]:
    """
    Return (final_reward, format_penalty) using configured reward mode.
    """
    format_penalty = 0.0 if is_compliant else float(config.paper_format_penalty)
    length_bonus = np.exp(
        -((solution_word_count - config.target_word_count) / config.scale_length_bonus) ** 2
    )

    if config.reward_mode == "rubric_only":
        final_reward = rubric_score
    elif config.reward_mode == "composite":
        final_reward = rubric_score + config.scaling_factor * length_bonus - format_penalty
    else:
        final_reward = rubric_score - format_penalty
    return float(final_reward), float(format_penalty)


def compute_group_advantages(group_rewards: list[float], normalize: bool) -> list[float]:
    """
    Compute GRPO-style group-relative advantages.
    """
    if not group_rewards:
        return []
    mean_reward = float(np.mean(group_rewards))
    if not normalize:
        return [float(r - mean_reward) for r in group_rewards]

    std_reward = float(np.std(group_rewards))
    if std_reward <= 1e-8:
        return [0.0 for _ in group_rewards]
    return [float((r - mean_reward) / (std_reward + 1e-8)) for r in group_rewards]


def compute_think_penalty(think_word_count: int, config: Config) -> float:
    """
    Soft penalty for long <think> traces in think_solution mode.
    Penalty is zero at/below soft threshold, then grows nonlinearly.
    """
    if config.policy_output_mode != "think_solution":
        return 0.0
    if think_word_count <= config.max_think_words_soft:
        return 0.0

    overflow = float(think_word_count - config.max_think_words_soft)
    scaled_overflow = overflow / max(config.think_penalty_words_per_step, 1e-6)
    penalty = config.think_penalty_slope * (scaled_overflow ** config.think_penalty_power)
    return float(min(max(penalty, 0.0), config.think_penalty_max))


def compute_non_solution_penalty(non_solution_word_count: int, config: Config) -> float:
    """
    Soft penalty for excessive verbosity outside <solution>.
    Helps control out-of-tag text that still consumes generation budget.
    """
    if non_solution_word_count <= config.max_non_solution_words_soft:
        return 0.0

    overflow = float(non_solution_word_count - config.max_non_solution_words_soft)
    scaled_overflow = overflow / max(config.non_solution_penalty_words_per_step, 1e-6)
    penalty = config.non_solution_penalty_slope * (scaled_overflow ** config.non_solution_penalty_power)
    return float(min(max(penalty, 0.0), config.non_solution_penalty_max))


def _should_drop_rate_capped(
    *,
    mode: str,
    max_drop_rate: float,
    current_reason_drops: int,
    num_trainability_checks: int,
) -> bool:
    if mode == "disabled":
        return False
    if mode == "hard":
        return True
    projected_checks = max(1, num_trainability_checks + 1)
    projected_drop_rate = (current_reason_drops + 1) / float(projected_checks)
    return projected_drop_rate <= max_drop_rate


def should_drop_think_too_long(
    *,
    config: Config,
    current_think_too_long_drops: int,
    num_trainability_checks: int,
) -> bool:
    """
    Decide whether to drop a think_too_long sample.
    - disabled: never drop for think length (penalty-only behavior)
    - hard: always drop
    - adaptive: cap think_too_long drop fraction within the batch
    """
    if config.policy_output_mode != "think_solution":
        return False

    return _should_drop_rate_capped(
        mode=config.think_too_long_drop_mode,
        max_drop_rate=config.think_too_long_max_drop_rate,
        current_reason_drops=current_think_too_long_drops,
        num_trainability_checks=num_trainability_checks,
    )


def should_drop_non_solution_too_long(
    *,
    config: Config,
    current_non_solution_too_long_drops: int,
    num_trainability_checks: int,
) -> bool:
    return _should_drop_rate_capped(
        mode=config.non_solution_too_long_drop_mode,
        max_drop_rate=config.non_solution_too_long_max_drop_rate,
        current_reason_drops=current_non_solution_too_long_drops,
        num_trainability_checks=num_trainability_checks,
    )


def compute_rubric_reward_from_loose_levels(
    xml_text: str,
    min_total_levels: int,
) -> tuple[float | None, int]:
    """
    Fallback scorer for malformed grader XML.
    Uses global <level> tags and aggregates them in groups of 7 desiderata.
    """
    raw_levels = re.findall(r"<level>\s*([0-3])\s*</level>", xml_text, flags=re.IGNORECASE)
    levels = [int(level) for level in raw_levels]
    found_levels = len(levels)
    if found_levels < min_total_levels:
        return None, found_levels

    level_map = {
        0: 0.0,
        1: 0.2,
        2: 0.6,
        3: 1.0,
    }

    num_items = found_levels // 7
    if num_items <= 0:
        return None, found_levels

    trimmed = levels[: num_items * 7]
    item_scores: list[float] = []
    for item_idx in range(num_items):
        chunk = trimmed[item_idx * 7 : (item_idx + 1) * 7]
        mapped = [level_map.get(level, 0.0) for level in chunk]
        item_scores.append(float(sum(mapped) / len(mapped)))

    if not item_scores:
        return None, found_levels
    return float(sum(item_scores) / len(item_scores)), found_levels


def is_sample_trainable(
    *,
    plan_text: str,
    solution_word_count: int,
    solution_text: str | None = None,
    think_word_count: int | None = None,
    max_think_words_hard: int | None = None,
    non_solution_word_count: int | None = None,
    max_non_solution_words_hard: int | None = None,
    is_compliant: bool,
    sample_logprobs: list[float] | None,
    drop_noncompliant_samples: bool,
    drop_template_copy_solutions: bool = True,
    min_words: int,
) -> bool:
    return (
        get_trainability_drop_reason(
            plan_text=plan_text,
            solution_word_count=solution_word_count,
            solution_text=solution_text,
            think_word_count=think_word_count,
            max_think_words_hard=max_think_words_hard,
            non_solution_word_count=non_solution_word_count,
            max_non_solution_words_hard=max_non_solution_words_hard,
            is_compliant=is_compliant,
            sample_logprobs=sample_logprobs,
            drop_noncompliant_samples=drop_noncompliant_samples,
            drop_template_copy_solutions=drop_template_copy_solutions,
            min_words=min_words,
        )
        is None
    )


def get_trainability_drop_reason(
    *,
    plan_text: str,
    solution_word_count: int,
    solution_text: str | None = None,
    think_word_count: int | None = None,
    max_think_words_hard: int | None = None,
    non_solution_word_count: int | None = None,
    max_non_solution_words_hard: int | None = None,
    is_compliant: bool,
    sample_logprobs: list[float] | None,
    drop_noncompliant_samples: bool,
    drop_template_copy_solutions: bool = True,
    min_words: int,
) -> str | None:
    if drop_noncompliant_samples and not is_compliant:
        return "format_noncompliant"
    if (
        think_word_count is not None
        and max_think_words_hard is not None
        and max_think_words_hard >= 0
        and think_word_count > max_think_words_hard
    ):
        return "think_too_long"
    if (
        non_solution_word_count is not None
        and max_non_solution_words_hard is not None
        and max_non_solution_words_hard >= 0
        and non_solution_word_count > max_non_solution_words_hard
    ):
        return "non_solution_too_long"
    if drop_template_copy_solutions and is_template_copy_solution(solution_text):
        return "template_copy_solution"
    if solution_word_count < min_words:
        return "too_short_solution"
    if plan_text.strip() in {"<think>", "</think>", "<solution>", "</solution>"}:
        return "degenerate_plan_text"
    if sample_logprobs is None:
        return "missing_logprobs"
    return None


def compute_sdpo_success_cutoff(
    *,
    history_batches_scores: list[list[float]],
    current_batch_scores: list[float],
    mode: str,
    adaptive_quantile: float,
    min_samples: int,
    use_current_batch_bootstrap: bool,
    floor: float,
    ceiling: float,
    fixed_threshold: float,
) -> tuple[float, str]:
    """
    Compute adaptive success cutoff for selecting successful previous rollouts.
    """
    if mode == "fixed":
        return float(fixed_threshold), "fixed"

    pool = [float(s) for batch in history_batches_scores for s in batch]
    source = "history"
    if len(pool) < min_samples and use_current_batch_bootstrap and current_batch_scores:
        pool = pool + [float(s) for s in current_batch_scores]
        source = "history+current_bootstrap" if history_batches_scores else "current_bootstrap"

    if not pool:
        return float(np.clip(1.0, floor, ceiling)), "no_data"

    cutoff = float(np.quantile(np.asarray(pool, dtype=np.float32), adaptive_quantile))
    cutoff = float(np.clip(cutoff, floor, ceiling))
    return cutoff, source


_DESIDERATA_LABELS = {
    1: "handling all rubric criteria",
    2: "specific implementation detail",
    3: "addressing critical flaws and risks",
    4: "strong justification and rationale",
    5: "cost and effort efficiency",
    6: "ethical and safety safeguards",
    7: "consistency with the overall plan",
}

_DESIDERATA_FULL_TEXT = {
    1: (
        "HANDLES ALL CRITERIA: Does the plan satisfy all criteria mentioned in the rubric item? "
        "If the criteria includes wording like 'such as', 'for example', or 'including', exact examples "
        "are not required, but the provided alternatives must be valid and reasonable."
    ),
    2: (
        "DETAILED, SPECIFIC SOLUTION: Does the relevant part of the plan include fully specified implementation "
        "details of HOW to execute it, without vague claims, ambiguity, or unsupported self-claims?"
    ),
    3: (
        "NO OVERLOOKED FLAWS OR WEAKNESSES: Are there important overlooked flaws or weaknesses in the part of "
        "the plan addressing this rubric item that undermine rubric satisfaction?"
    ),
    4: (
        "WELL JUSTIFIED RATIONALE: Is the approach for this rubric item well motivated and justified, including "
        "why it is preferable to simpler alternatives or competing hypotheses?"
    ),
    5: (
        "COST AND EFFORT EFFICIENT: Does the plan satisfy this item efficiently without unnecessary complexity?"
    ),
    6: (
        "NO ETHICAL ISSUES: Does this part of the plan avoid potential negative consequences and ethical problems?"
    ),
    7: (
        "CONSISTENT WITH OVERALL PLAN: Is this part coherent with the rest of the plan, with no internal contradictions?"
    ),
}


def _clean_reasoning_text(reasoning_raw: str) -> str:
    """Remove XML artifacts and normalize whitespace from grader reasoning."""
    no_tags = re.sub(r"<[^>]+>", " ", reasoning_raw)
    cleaned = re.sub(r"\s+", " ", no_tags).strip()
    return cleaned


def _extract_bullets(section_text: str) -> list[str]:
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
    parts = [p.strip() for p in re.split(r"[.;]\s+", fallback) if p.strip()]
    return parts


def _extract_reasoning_sections(reasoning_raw: str) -> tuple[list[str], list[str]]:
    clean_text = reasoning_raw if isinstance(reasoning_raw, str) else ""
    weak_match = re.search(
        r"Weaknesses:\s*(.*?)(?:Actionable\s+fixes:|$)",
        clean_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    fix_match = re.search(
        r"Actionable\s+fixes:\s*(.*)$",
        clean_text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    weakness_lines = _extract_bullets(weak_match.group(1)) if weak_match else []
    fix_lines = _extract_bullets(fix_match.group(1)) if fix_match else []

    if weakness_lines or fix_lines:
        return weakness_lines, fix_lines

    # Fallback: treat full reasoning as weakness text when sections are missing.
    fallback_line = _clean_reasoning_text(clean_text)
    return ([fallback_line] if fallback_line else []), []


def _extract_xml_tag_text(text: str, tag: str) -> str:
    match = re.search(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return ""
    return _clean_reasoning_text(match.group(1))


def _extract_desiderata_review_feedback(item_xml: str) -> list[str]:
    """
    Parse optional <desiderata_review> blocks and convert them to concise lines.
    """
    review_block_match = re.search(
        r"<desiderata_review>\s*(.*?)\s*</desiderata_review>",
        item_xml,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if review_block_match is None:
        return []

    review_body = review_block_match.group(1)
    summary = _extract_xml_tag_text(review_body, "summary")
    suggested_new_desiderata = _extract_xml_tag_text(review_body, "suggested_new_desiderata")
    if not suggested_new_desiderata:
        suggested_new_desiderata = _extract_xml_tag_text(review_body, "new_desiderata")
    lines: list[str] = []
    if summary:
        lines.append(f"[desiderata_overall] {summary}")
    if (
        suggested_new_desiderata
        and suggested_new_desiderata.lower() not in {"none", "n/a", "na", "no"}
    ):
        lines.append(
            f"[desiderata_suggestion] Suggested new desiderata: {suggested_new_desiderata}"
        )
    if lines:
        return lines

    review_entries = re.findall(
        r"<desideratum\s+num\s*=\s*['\"]?([1-7])['\"]?\s*>(.*?)</desideratum>",
        review_body,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not review_entries:
        return []

    lines = []
    for desideratum_num_str, entry_xml in review_entries:
        idx = int(desideratum_num_str)
        label = _DESIDERATA_LABELS.get(idx, f"desiderata {idx}")
        reasonable_raw = _extract_xml_tag_text(entry_xml, "reasonable").lower()
        comment = _extract_xml_tag_text(entry_xml, "comment")
        better_option = _extract_xml_tag_text(entry_xml, "better_option")
        if not better_option:
            better_option = _extract_xml_tag_text(entry_xml, "suggested_revision")

        if not comment and not better_option:
            continue

        status = "reasonable"
        if reasonable_raw in {"no", "not_reasonable", "unreasonable"}:
            status = "not_reasonable"
        elif reasonable_raw in {"partly", "partial", "mixed"}:
            status = "partly_reasonable"

        if better_option and better_option.lower() not in {"none", "n/a", "na"}:
            line = (
                f"[{label}] {status}: {comment} "
                f"Suggested better option: {better_option}"
            ).strip()
        else:
            line = f"[{label}] {status}: {comment}".strip()

        line = re.sub(r"\s+", " ", line)
        if line:
            lines.append(line)

    return lines


def _extract_global_desiderata_review_feedback(xml_text: str) -> str:
    """
    Parse a sample-level <desiderata_review> block (outside per-item XML).
    """
    rubric_match = re.search(
        r"<rubric\b[^>]*>(.*)</rubric>",
        xml_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    search_scope = rubric_match.group(1) if rubric_match is not None else xml_text
    # Ignore per-item reviews; keep only sample-level desiderata review blocks.
    search_scope = re.sub(
        r"<item\b[^>]*>.*?</item>",
        " ",
        search_scope,
        flags=re.DOTALL | re.IGNORECASE,
    )
    review_matches = re.findall(
        r"<desiderata_review>\s*(.*?)\s*</desiderata_review>",
        search_scope,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not review_matches:
        return ""

    for review_body in review_matches:
        summary = _extract_xml_tag_text(review_body, "summary")
        if summary:
            return summary

    # Fallback for malformed summary tags.
    return _clean_reasoning_text(review_matches[-1])


def _extract_sample_review_feedback(xml_text: str) -> list[str]:
    """
    Parse optional sample-level review block:
    <sample_review><weaknesses>...</weaknesses><potential_fixes>...</potential_fixes></sample_review>
    """
    review_match = re.search(
        r"<sample_review>\s*(.*?)\s*</sample_review>",
        xml_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if review_match is None:
        return []

    review_body = review_match.group(1)
    weaknesses = _extract_xml_tag_text(review_body, "weaknesses")
    fixes = _extract_xml_tag_text(review_body, "potential_fixes")
    lines: list[str] = []
    if weaknesses:
        lines.append(f"[sample_overall] Weaknesses: {weaknesses}")
    if fixes:
        lines.append(f"[sample_overall] Potential fixes: {fixes}")
    return lines


def _is_generic_feedback_line(text: str) -> bool:
    lower = text.lower()
    if len(lower) < 24:
        return True

    generic_phrases = (
        "needs more detail",
        "be more specific",
        "needs improvement",
        "improve clarity",
        "unclear",
        "insufficient detail",
        "not detailed enough",
    )
    concrete_hints = (
        "dataset",
        "baseline",
        "metric",
        "ablation",
        "evaluation",
        "control",
        "confound",
        "power",
        "random",
        "risk",
        "failure",
        "implementation",
        "resource",
        "timeline",
        "cost",
        "criterion",
        "token",
        "format",
        "compliance",
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
    """
    Return (weaknesses, potential_fixes) extracted from sample-level <sample_review>.
    """
    review_match = re.search(
        r"<sample_review>\s*(.*?)\s*</sample_review>",
        xml_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if review_match is None:
        return [], []

    review_body = review_match.group(1)
    weaknesses_raw = _extract_xml_tag_text(review_body, "weaknesses")
    fixes_raw = _extract_xml_tag_text(review_body, "potential_fixes")
    return _extract_bullets(weaknesses_raw), _extract_bullets(fixes_raw)


def _extract_item_desiderata_review_parts(item_xml: str) -> tuple[str, str]:
    """
    Return (summary, suggested_new_desiderata) for per-item <desiderata_review>.
    """
    review_match = re.search(
        r"<desiderata_review>\s*(.*?)\s*</desiderata_review>",
        item_xml,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if review_match is None:
        return "", ""

    review_body = review_match.group(1)
    summary = _extract_xml_tag_text(review_body, "summary")
    suggested = _extract_xml_tag_text(review_body, "suggested_new_desiderata")
    if not suggested:
        suggested = _extract_xml_tag_text(review_body, "new_desiderata")
    if suggested.lower() in {"none", "n/a", "na", "no"}:
        suggested = ""
    return summary, suggested


def _build_desiderata_reference_text() -> str:
    entries = [
        f"D{i}: {_DESIDERATA_FULL_TEXT.get(i, _DESIDERATA_LABELS.get(i, f'desiderata {i}'))}"
        for i in range(1, 8)
    ]
    return "; ".join(entries)


def _build_desiderata_profile_text(levels: list[int]) -> str:
    normalized: list[str] = []
    for idx in range(1, 8):
        if idx - 1 < len(levels) and levels[idx - 1] in {0, 1, 2, 3}:
            lvl = levels[idx - 1]
            normalized.append(f"D{idx}={lvl}")
        else:
            normalized.append(f"D{idx}=NA")
    return ", ".join(normalized)


def _infer_desiderata_behavior_from_levels(levels: list[int]) -> str:
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
    include_global_desiderata_summary: bool = True,
) -> tuple[str, int, int]:
    """
    Build a compact teacher-facing critique from grader XML.
    Only selected high-signal parts are used:
    1) weaknesses,
    2) potential fixes,
    3) per-rubric-item 7-desiderata behavior summaries with rubric context.
    """
    default_feedback = (
        "Critical revision needed: strengthen implementation detail, rationale, and risk controls."
    )
    if not xml_text:
        return default_feedback, 1, 0

    item_blocks = re.findall(
        r"<item\b[^>]*>.*?</item>",
        xml_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not item_blocks:
        return default_feedback, 1, 0

    max_lines = max(1, int(max_bullets_per_item))
    global_desiderata_review = (
        _extract_global_desiderata_review_feedback(xml_text)
        if include_global_desiderata_summary
        else ""
    )
    sample_review_weaknesses, sample_review_fixes = (
        _extract_sample_review_sections(xml_text)
        if include_sample_review
        else ([], [])
    )
    critical_weaknesses: list[str] = []
    required_fixes: list[str] = []
    rubric_desiderata_behavior: list[str] = []
    seen: set[str] = set()
    low_level_item_count = 0

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

    processed_items = 0
    max_items_effective = max(1, int(max_items))
    for item_xml in item_blocks:
        # Prefer numeric levels from explicit <level> tags; use last 7 for robustness
        # when malformed outputs duplicate level tags inside <reasoning>.
        all_numeric_levels = [
            int(level)
            for level in re.findall(
                r"<level>\s*([0-3])\s*</level>",
                item_xml,
                flags=re.IGNORECASE,
            )
        ]
        numeric_levels = all_numeric_levels[-7:] if len(all_numeric_levels) >= 7 else all_numeric_levels
        has_low_level = any(level <= 2 for level in numeric_levels)

        # Fallback textual detection only when numeric levels are absent.
        if not numeric_levels:
            has_low_level = (
                re.search(r"<level>\s*[0-2]\s*</level>", item_xml, flags=re.IGNORECASE) is not None
                or re.search(r"\bLevel\s*[0-2]\b", item_xml, flags=re.IGNORECASE) is not None
            )
        if focus_low_confidence_items_only and not has_low_level:
            continue
        if processed_items >= max_items_effective:
            break
        processed_items += 1
        if has_low_level:
            low_level_item_count += 1

        criteria_match = re.search(
            r"<criteria>\s*(.*?)\s*</criteria>",
            item_xml,
            flags=re.DOTALL | re.IGNORECASE,
        )
        criteria = ""
        if criteria_match is not None:
            criteria = re.sub(r"\s+", " ", criteria_match.group(1)).strip()

        reasoning_match = re.search(
            r"<reasoning>\s*(.*?)\s*</reasoning>",
            item_xml,
            flags=re.DOTALL | re.IGNORECASE,
        )
        reasoning_raw = ""
        if reasoning_match is not None:
            reasoning_raw = reasoning_match.group(1)

        item_weaknesses: list[str] = []
        item_fixes: list[str] = []
        if include_item_reasoning_feedback:
            item_weaknesses, item_fixes = _extract_reasoning_sections(reasoning_raw)
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
            review_summary, review_suggestion = _extract_item_desiderata_review_parts(item_xml)
            inferred_behavior = _infer_desiderata_behavior_from_levels(numeric_levels[:7])
            summary_text = review_summary or inferred_behavior
            profile_text = _build_desiderata_profile_text(numeric_levels[:7])
            behavior_line = (
                f'Rubric item: "{criteria}". '
                f"7-desiderata behavior: {summary_text} "
                f"Confidence profile: {profile_text}. "
                f"Desiderata reference: {_build_desiderata_reference_text()}"
            )
            if review_suggestion:
                behavior_line += f" Suggested new desiderata: {review_suggestion}."
            behavior_line = re.sub(r"\s+", " ", behavior_line).strip()
            if behavior_line and behavior_line not in seen:
                rubric_desiderata_behavior.append(behavior_line)
                seen.add(behavior_line)

    feedback_sections: list[str] = []
    if critical_weaknesses:
        weakness_lines = [
            f"- Critical weakness to fix now: {line}. This must be corrected in the next revision."
            for line in critical_weaknesses
        ]
        feedback_sections.append("# Mandatory Weaknesses\n" + "\n".join(weakness_lines))

    if required_fixes:
        fix_lines = [
            f"- Required corrective action: {line}. Implement this concretely in the revised plan."
            for line in required_fixes
        ]
        feedback_sections.append("# Required Fixes\n" + "\n".join(fix_lines))

    if rubric_desiderata_behavior:
        behavior_lines = [
            f"- {line}" for line in rubric_desiderata_behavior
        ]
        feedback_sections.append(
            "# Rubric-Desiderata Behavior by Item\n" + "\n".join(behavior_lines)
        )

    if global_desiderata_review:
        global_line = re.sub(r"\s+", " ", global_desiderata_review).strip()
        if global_line:
            feedback_sections.append(
                "# Overall Desiderata Summary\n"
                + f"- Across all items: {global_line}"
            )

    if not feedback_sections:
        return default_feedback, 1, low_level_item_count

    feedback_text = "\n\n".join(feedback_sections).strip()
    bullet_count = (
        len(critical_weaknesses)
        + len(required_fixes)
        + len(rubric_desiderata_behavior)
        + (1 if global_desiderata_review else 0)
    )
    return feedback_text, max(1, bullet_count), low_level_item_count


def build_self_teacher_prompt(
    scenario: str,
    grader_feedback: str | None,
    successful_previous_rollout: str | None,
    policy_output_mode: str = "solution_only",
    target_word_count: int = 600,
    max_solution_words: int = 750,
    max_think_words: int = 60,
) -> str:
    """
    Self-teacher template adapted from SDPO Table 2:
    - include successful rollout if available
    - otherwise include rich feedback from environment/grader
    """
    target_low = max(120, int(target_word_count - 80))
    target_high = min(int(max_solution_words), int(target_word_count + 100))

    sections: list[str] = [
        "You are revising a research plan after expert critique. "
        "Fix concrete weaknesses and improve feasibility without adding fluff.",
        f"# Research Scenario\n{scenario}",
    ]

    if successful_previous_rollout:
        sections.append(
            "# Successful Previous Rollout (reference quality signal)\n"
            "Use this as a quality reference, but do not copy it verbatim.\n"
            f"{successful_previous_rollout}"
        )

    if grader_feedback:
        sections.append(
            "# Critique to Address\n"
            f"{grader_feedback}"
        )
    else:
        sections.append(
            "# Critique to Address\n"
            "- Strengthen concrete implementation details and rationale.\n"
            "- Add explicit risk checks, evaluation criteria, and failure handling."
        )

    sections.append(
        textwrap.dedent(f"""
            # Revision Constraints
            - Address scenario goals, constraints, and confounders explicitly.
            - Explain HOW and WHY for each major step, not only WHAT to do.
            - Do not claim completed experiments or fabricated results.
            - Do not self-claim desiderata satisfaction.
            - Keep <solution> around {target_low}-{target_high} words (hard cap: {max_solution_words} words).
            - Do not copy instruction/template text.
        """).strip()
    )

    if policy_output_mode == "think_solution":
        sections.append(
            textwrap.dedent(f"""
                # Output Contract (strict)
                Return exactly two XML blocks and close all tags:
                <think>
                Brief reasoning only, at most {max_think_words} words.
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
            textwrap.dedent("""
                # Output Contract (strict)
                Return exactly one XML block using ONLY <solution> tags.
                <solution>
                Complete and self-contained revised research plan.
                </solution>

                Important:
                - First output token must be <solution>, and final output token must be </solution>.
                - Do not output any text before <solution> or after </solution>.
            """).strip()
        )
    else:
        raise ValueError(
            f"Invalid policy_output_mode={policy_output_mode!r}. "
            "Expected 'solution_only' or 'think_solution'."
        )
    return "\n\n".join(sections).strip()


def _summarize_training_datums(
    datums: list[types.Datum], max_items: int = 3
) -> list[dict[str, object]]:
    """Return compact shape/size diagnostics for debugging corrupted PPO batches."""
    summary: list[dict[str, object]] = []
    for idx, datum in enumerate(datums[:max_items]):
        loss_inputs = datum.loss_fn_inputs
        target = loss_inputs.get("target_tokens")
        logprobs = loss_inputs.get("logprobs")
        advantages = loss_inputs.get("advantages")
        summary.append(
            {
                "datum_idx": idx,
                "model_input_len": len(datum.model_input.to_ints()),
                "target_shape": getattr(target, "shape", None),
                "target_size": len(getattr(target, "data", [])),
                "logprobs_shape": getattr(logprobs, "shape", None),
                "logprobs_size": len(getattr(logprobs, "data", [])),
                "advantages_shape": getattr(advantages, "shape", None),
                "advantages_size": len(getattr(advantages, "data", [])),
            }
        )
    return summary


async def _compute_logprobs_async(
    sampling_client: tinker.SamplingClient,
    sequences: list[types.ModelInput],
) -> list[list[float]]:
    return await asyncio.gather(
        *[sampling_client.compute_logprobs_async(sequence) for sequence in sequences]
    )


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


async def _sample_policy(
    sampling_client: tinker.SamplingClient,
    model_input: types.ModelInput,
    goal_idx: int,
    prompt_tokens: list[int],
    group_size: int,
    sampling_params: types.SamplingParams,
) -> tuple[int, list[int], types.SampleResponse] | None:
    try:
        response = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=group_size,
            sampling_params=sampling_params,
        )
        return goal_idx, prompt_tokens, response
    except Exception:
        logger.exception("Policy sampling failed for goal_idx=%s", goal_idx)
    return None


async def _sample_policy_limited(
    *,
    semaphore: asyncio.Semaphore,
    sampling_client: tinker.SamplingClient,
    model_input: types.ModelInput,
    goal_idx: int,
    prompt_tokens: list[int],
    group_size: int,
    sampling_params: types.SamplingParams,
) -> tuple[int, list[int], types.SampleResponse] | None:
    async with semaphore:
        return await _sample_policy(
            sampling_client=sampling_client,
            model_input=model_input,
            goal_idx=goal_idx,
            prompt_tokens=prompt_tokens,
            group_size=group_size,
            sampling_params=sampling_params,
        )


async def _sample_grader(
    grader_client: tinker.SamplingClient,
    grader_input: types.ModelInput,
    group_idx: int,
    sample_idx: int,
    grader_sampling_params: types.SamplingParams,
) -> tuple[int, int, types.SampleResponse | None]:
    try:
        response = await grader_client.sample_async(
            prompt=grader_input,
            num_samples=1,
            sampling_params=grader_sampling_params,
        )
        return group_idx, sample_idx, response
    except Exception:
        logger.exception("Grader sampling failed for group_idx=%s sample_idx=%s", group_idx, sample_idx)
    return group_idx, sample_idx, None


async def _sample_grader_limited(
    *,
    semaphore: asyncio.Semaphore,
    grader_client: tinker.SamplingClient,
    grader_input: types.ModelInput,
    group_idx: int,
    sample_idx: int,
    grader_sampling_params: types.SamplingParams,
) -> tuple[int, int, types.SampleResponse | None]:
    async with semaphore:
        return await _sample_grader(
            grader_client=grader_client,
            grader_input=grader_input,
            group_idx=group_idx,
            sample_idx=sample_idx,
            grader_sampling_params=grader_sampling_params,
        )


async def _retry_format_sample_once(
    *,
    sampling_client: tinker.SamplingClient,
    grader_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    tokenizer,
    prompt_tokens: list[int],
    scenario: str,
    rubric_items: list[str],
    reference_solution: str | None,
    sampling_params: types.SamplingParams,
    grader_sampling_params: types.SamplingParams,
    grade_solution_only: bool,
    fast_grader_mode: bool,
    grader_prompt_mode: str,
    fast_grader_reasoning_max_words_per_item: int,
    fast_grader_summary_max_words: int,
) -> tuple[dict[str, object], types.SampleResponse | None] | None:
    """
    Retry once for format recovery and return (sample_info, grader_result) on success.
    """
    try:
        response = await sampling_client.sample_async(
            prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
            num_samples=1,
            sampling_params=sampling_params,
        )
    except Exception:
        logger.exception("Format retry policy sampling failed")
        return None

    if response is None or not response.sequences:
        return None

    sequence = response.sequences[0]
    try:
        proposed_plan_raw = renderers.get_text_content(
            renderer.parse_response(sequence.tokens)[0]
        )
    except Exception:
        proposed_plan_raw = tokenizer.decode(
            sequence.tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    proposed_plan_for_grader = normalize_policy_output_for_grader(proposed_plan_raw)
    if grade_solution_only:
        strict_solution = extract_solution_text(
            proposed_plan_for_grader,
            require_tags=True,
        )
        proposed_plan_for_grader = (
            strict_solution
            if (strict_solution is not None and strict_solution.strip())
            else "FORMAT VIOLATION: Missing or malformed <solution>...</solution> block."
        )

    grader_prompt_text = build_grader_prompt(
        scenario=scenario,
        rubric_items=rubric_items,
        proposed_plan=proposed_plan_for_grader,
        reference_solution=reference_solution,
        grader_prompt_mode=grader_prompt_mode,
        fast_grader_mode=fast_grader_mode,
        fast_reasoning_max_words_per_item=fast_grader_reasoning_max_words_per_item,
        fast_summary_max_words=fast_grader_summary_max_words,
    )
    grader_input = renderer.build_generation_prompt(
        [{"role": "user", "content": grader_prompt_text}]
    )
    try:
        grader_result = await grader_client.sample_async(
            prompt=grader_input,
            num_samples=1,
            sampling_params=grader_sampling_params,
        )
    except Exception:
        logger.exception("Format retry grader sampling failed")
        grader_result = None

    sample_info = {
        "tokens": sequence.tokens,
        "logprobs": sequence.logprobs,
        "text": proposed_plan_raw,
        "text_raw": proposed_plan_raw,
        "text_for_grader": proposed_plan_for_grader,
    }
    return sample_info, grader_result

# ============================================================
# Main
# ============================================================

async def main(config: Config):
    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    is_eval = config.run_eval
    os.makedirs(config.log_path, exist_ok=True)
    if config.reward_mode not in {"rubric_only", "composite", "paper"}:
        raise ValueError(
            f"Invalid reward_mode={config.reward_mode!r}. Expected 'rubric_only', 'composite', or 'paper'."
        )
    if config.policy_output_mode not in {"solution_only", "think_solution"}:
        raise ValueError(
            f"Invalid policy_output_mode={config.policy_output_mode!r}. "
            "Expected 'solution_only' or 'think_solution'."
        )
    if config.sdpo_success_mode not in {"adaptive_quantile", "fixed"}:
        raise ValueError(
            f"Invalid sdpo_success_mode={config.sdpo_success_mode!r}. "
            "Expected 'adaptive_quantile' or 'fixed'."
        )
    if config.metrics_mode not in {"compact", "full"}:
        raise ValueError(
            f"Invalid metrics_mode={config.metrics_mode!r}. "
            "Expected 'compact' or 'full'."
        )
    if config.max_batches is not None and int(config.max_batches) < 1:
        raise ValueError(
            f"Invalid max_batches={config.max_batches!r}. Expected None or >= 1."
        )
    if config.grader_prompt_mode not in {"detailed", "compact_confidence_only"}:
        raise ValueError(
            f"Invalid grader_prompt_mode={config.grader_prompt_mode!r}. "
            "Expected 'detailed' or 'compact_confidence_only'."
        )
    if not (0.0 < config.sdpo_success_adaptive_quantile < 1.0):
        raise ValueError(
            f"Invalid sdpo_success_adaptive_quantile={config.sdpo_success_adaptive_quantile!r}. "
            "Expected a value in (0, 1)."
        )
    if config.sdpo_success_history_batches < 1:
        raise ValueError(
            f"Invalid sdpo_success_history_batches={config.sdpo_success_history_batches!r}. "
            "Expected >= 1."
        )
    if config.sdpo_success_min_samples < 1:
        raise ValueError(
            f"Invalid sdpo_success_min_samples={config.sdpo_success_min_samples!r}. "
            "Expected >= 1."
        )
    if not (0.0 <= config.sdpo_success_min_rate <= 1.0):
        raise ValueError(
            f"Invalid sdpo_success_min_rate={config.sdpo_success_min_rate!r}. "
            "Expected a value in [0, 1]."
        )
    if config.sdpo_success_min_groups < 0:
        raise ValueError(
            f"Invalid sdpo_success_min_groups={config.sdpo_success_min_groups!r}. "
            "Expected >= 0."
        )
    if config.sdpo_success_floor > config.sdpo_success_ceiling:
        raise ValueError(
            f"Invalid sdpo_success_floor/sdpo_success_ceiling: "
            f"{config.sdpo_success_floor!r} > {config.sdpo_success_ceiling!r}."
        )
    if config.min_words < 0:
        raise ValueError(
            f"Invalid min_words={config.min_words!r}. Expected >= 0."
        )
    if config.min_words_warmup_batches < 0:
        raise ValueError(
            f"Invalid min_words_warmup_batches={config.min_words_warmup_batches!r}. Expected >= 0."
        )
    if config.min_words_warmup_value < 0:
        raise ValueError(
            f"Invalid min_words_warmup_value={config.min_words_warmup_value!r}. Expected >= 0."
        )
    if config.format_retry_max_attempts < 0:
        raise ValueError(
            f"Invalid format_retry_max_attempts={config.format_retry_max_attempts!r}. Expected >= 0."
        )
    if not (0.0 <= config.format_retry_temperature <= 2.0):
        raise ValueError(
            f"Invalid format_retry_temperature={config.format_retry_temperature!r}. Expected in [0, 2]."
        )
    if config.format_retry_max_tokens < 0:
        raise ValueError(
            f"Invalid format_retry_max_tokens={config.format_retry_max_tokens!r}. Expected >= 0."
        )
    if config.sampling_client_refresh_interval_batches < 1:
        raise ValueError(
            "Invalid sampling_client_refresh_interval_batches="
            f"{config.sampling_client_refresh_interval_batches!r}. Expected >= 1."
        )
    if config.max_concurrent_policy_requests < 1:
        raise ValueError(
            f"Invalid max_concurrent_policy_requests={config.max_concurrent_policy_requests!r}. Expected >= 1."
        )
    if config.max_concurrent_grader_requests < 1:
        raise ValueError(
            f"Invalid max_concurrent_grader_requests={config.max_concurrent_grader_requests!r}. Expected >= 1."
        )
    if config.fast_grader_max_tokens < 1:
        raise ValueError(
            f"Invalid fast_grader_max_tokens={config.fast_grader_max_tokens!r}. Expected >= 1."
        )
    if config.fast_grader_reasoning_max_words_per_item < 10:
        raise ValueError(
            "Invalid fast_grader_reasoning_max_words_per_item="
            f"{config.fast_grader_reasoning_max_words_per_item!r}. Expected >= 10."
        )
    if config.fast_grader_summary_max_words < 10:
        raise ValueError(
            f"Invalid fast_grader_summary_max_words={config.fast_grader_summary_max_words!r}. Expected >= 10."
        )
    if config.fast_grader_feedback_max_chars < 1:
        raise ValueError(
            f"Invalid fast_grader_feedback_max_chars={config.fast_grader_feedback_max_chars!r}. Expected >= 1."
        )
    if config.sdpo_feedback_max_bullets_per_item < 1:
        raise ValueError(
            "Invalid sdpo_feedback_max_bullets_per_item="
            f"{config.sdpo_feedback_max_bullets_per_item!r}. Expected >= 1."
        )
    if config.sdpo_teacher_max_criticized_items < 1:
        raise ValueError(
            "Invalid sdpo_teacher_max_criticized_items="
            f"{config.sdpo_teacher_max_criticized_items!r}. Expected >= 1."
        )
    if not (0.0 <= config.sdpo_solution_mask_warn_zero_rate <= 1.0):
        raise ValueError(
            "Invalid sdpo_solution_mask_warn_zero_rate="
            f"{config.sdpo_solution_mask_warn_zero_rate!r}. Expected [0, 1]."
        )
    if config.grader_xml_fallback_min_levels < 1:
        raise ValueError(
            f"Invalid grader_xml_fallback_min_levels={config.grader_xml_fallback_min_levels!r}. "
            "Expected >= 1."
        )
    if config.max_think_words_soft < 0:
        raise ValueError(
            f"Invalid max_think_words_soft={config.max_think_words_soft!r}. Expected >= 0."
        )
    if config.max_think_words_hard < config.max_think_words_soft:
        raise ValueError(
            "Invalid think length thresholds: "
            f"max_think_words_hard={config.max_think_words_hard!r} must be >= "
            f"max_think_words_soft={config.max_think_words_soft!r}."
        )
    if config.think_penalty_words_per_step <= 0:
        raise ValueError(
            f"Invalid think_penalty_words_per_step={config.think_penalty_words_per_step!r}. "
            "Expected > 0."
        )
    if config.think_penalty_slope < 0:
        raise ValueError(
            f"Invalid think_penalty_slope={config.think_penalty_slope!r}. Expected >= 0."
        )
    if config.think_penalty_power <= 0:
        raise ValueError(
            f"Invalid think_penalty_power={config.think_penalty_power!r}. Expected > 0."
        )
    if config.think_penalty_max < 0:
        raise ValueError(
            f"Invalid think_penalty_max={config.think_penalty_max!r}. Expected >= 0."
        )
    if config.think_too_long_drop_mode not in {"disabled", "adaptive", "hard"}:
        raise ValueError(
            f"Invalid think_too_long_drop_mode={config.think_too_long_drop_mode!r}. "
            "Expected 'disabled', 'adaptive', or 'hard'."
        )
    if not (0.0 <= config.think_too_long_max_drop_rate <= 1.0):
        raise ValueError(
            "Invalid think_too_long_max_drop_rate="
            f"{config.think_too_long_max_drop_rate!r}. Expected [0, 1]."
        )
    if config.max_non_solution_words_soft < 0:
        raise ValueError(
            "Invalid max_non_solution_words_soft="
            f"{config.max_non_solution_words_soft!r}. Expected >= 0."
        )
    if config.max_non_solution_words_hard < config.max_non_solution_words_soft:
        raise ValueError(
            "Invalid non-solution length thresholds: "
            f"max_non_solution_words_hard={config.max_non_solution_words_hard!r} must be >= "
            f"max_non_solution_words_soft={config.max_non_solution_words_soft!r}."
        )
    if config.non_solution_penalty_words_per_step <= 0:
        raise ValueError(
            "Invalid non_solution_penalty_words_per_step="
            f"{config.non_solution_penalty_words_per_step!r}. Expected > 0."
        )
    if config.non_solution_penalty_slope < 0:
        raise ValueError(
            f"Invalid non_solution_penalty_slope={config.non_solution_penalty_slope!r}. "
            "Expected >= 0."
        )
    if config.non_solution_penalty_power <= 0:
        raise ValueError(
            f"Invalid non_solution_penalty_power={config.non_solution_penalty_power!r}. "
            "Expected > 0."
        )
    if config.non_solution_penalty_max < 0:
        raise ValueError(
            f"Invalid non_solution_penalty_max={config.non_solution_penalty_max!r}. "
            "Expected >= 0."
        )
    if config.non_solution_too_long_drop_mode not in {"disabled", "adaptive", "hard"}:
        raise ValueError(
            "Invalid non_solution_too_long_drop_mode="
            f"{config.non_solution_too_long_drop_mode!r}. "
            "Expected 'disabled', 'adaptive', or 'hard'."
        )
    if not (0.0 <= config.non_solution_too_long_max_drop_rate <= 1.0):
        raise ValueError(
            "Invalid non_solution_too_long_max_drop_rate="
            f"{config.non_solution_too_long_max_drop_rate!r}. Expected [0, 1]."
        )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load dataset, use ML
    logger.info("Loading dataset...")
    if config.ml_data:
        data = load_dataset("facebook/research-plan-gen", "ml")
    elif config.arxiv_data:
        data = load_dataset("facebook/research-plan-gen", "arxiv")
    elif config.pubmed_data:
        data = load_dataset("facebook/research-plan-gen", "pubmed")

    assert isinstance(data, datasets.DatasetDict)
    if is_eval:
        dataset = data["test"]
    else:
        dataset = data["train"]

    # Setup training client
    service_client = create_service_client(
        base_url=config.base_url,
        api_profile=config.api_profile,
    )
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
            logger.exception("Failed to query server capabilities; using configured batch/group sizes")

    n_train_batches = len(dataset) // effective_batch_size
    if n_train_batches <= 0:
        logger.warning("No train batches available: dataset=%s batch_size=%s", len(dataset), effective_batch_size)
        ml_logger.close()
        return

    max_batches_effective = int(config.max_batches) if config.max_batches is not None else None

  
    # Load for checkpoint, if no checkpoints, start from beginning. 
    # IF in EVAL, checkpoint=0 is eval for baseline, checkpoint=-1 is eval for last checkpoint, checkpoint=n is eval for the nth epoch
    last_checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path)
    resume_info: dict[str, object] | bool = False
    if (config.eval_epoch == 0 and config.run_eval) or (not config.run_eval and last_checkpoint is None):
        resume_info = False
    elif config.eval_epoch > 0 and config.run_eval:
        checkpoints = checkpoint_utils.load_checkpoints_file(config.log_path)
        checkpoints_with_key = [c for c in checkpoints if "state_path" in c]
        if checkpoints_with_key and config.eval_epoch <= len(checkpoints_with_key):
            resume_info = checkpoints_with_key[config.eval_epoch - 1]
            logger.info(
            f"Found {len(checkpoints_with_key)} valid checkpoints with key '{"state_path"}' in {config.log_path}")
            logger.info(f"Using checkpoint: {checkpoints_with_key[config.eval_epoch - 1]}")
        elif checkpoints_with_key:
            logger.warning(
                "Requested eval_epoch=%s exceeds available checkpoints=%s; using latest checkpoint",
                config.eval_epoch,
                len(checkpoints_with_key),
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
    # After this point we have a `training_client` ready. If resuming, it's
    # created from saved state; otherwise it's a fresh LoRA training client.
   
    # Create my grader client 
    grader_client = await service_client.create_sampling_client_async(
        base_model=config.grader_model_name
    )

    # Initial teacher for regularized self-teacher (SDPO Appendix A.2 interpolation variant)
    initial_teacher_client = None
    if config.use_sdpo and config.sdpo_teacher_reg_alpha > 0.0:
        initial_teacher_client = await service_client.create_sampling_client_async(
            base_model=config.model_name
        )
 
    # Parameters for sampling from the training client
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature
    )
    retry_max_tokens = (
        config.format_retry_max_tokens
        if config.format_retry_max_tokens > 0
        else config.max_tokens
    )
    format_retry_sampling_params = tinker.types.SamplingParams(
        max_tokens=retry_max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.format_retry_temperature,
    )
    effective_grader_max_tokens = int(config.grader_max_tokens)
    if config.fast_grader_mode:
        effective_grader_max_tokens = min(
            effective_grader_max_tokens,
            int(config.fast_grader_max_tokens),
        )
    grader_sampling_params = tinker.types.SamplingParams(
        max_tokens=effective_grader_max_tokens,
        temperature=config.grader_temperature,
    )
    logger.info(
        "Grader mode: fast=%s, max_tokens=%s (configured=%s)",
        config.fast_grader_mode,
        effective_grader_max_tokens,
        config.grader_max_tokens,
    )
    policy_request_semaphore = asyncio.Semaphore(
        max(1, int(config.max_concurrent_policy_requests))
    )
    grader_request_semaphore = asyncio.Semaphore(
        max(1, int(config.max_concurrent_grader_requests))
    )

    # Parameters for optimization step
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    # --------------------------------------------------
    # Prepare sampler
    # --------------------------------------------------
    sampling_client: tinker.SamplingClient | None = None
    if is_eval:
        # In eval mode, use the current model weights once
        logger.info("Eval mode: preparing fixed sampler")

        # If resuming, training_client already has loaded weights
        # If not resuming, this is the freshly initialized / final model
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(
            name=f"Eval_Checkpoint{actual_batch - 1}"
        )


    loop_end_batch = n_train_batches
    if max_batches_effective is not None:
        loop_end_batch = min(n_train_batches, start_batch + max_batches_effective)
    effective_loop_batches = max(0, loop_end_batch - start_batch)
    if effective_loop_batches <= 0:
        logger.warning(
            "No batches to run after applying max_batches: start_batch=%s n_train_batches=%s max_batches=%s",
            start_batch,
            n_train_batches,
            max_batches_effective,
        )
        ml_logger.close()
        return

    logger.info(
        "Training for %s batches (this run executes %s batch(es): start_batch=%s, end_batch=%s, max_batches=%s)",
        n_train_batches,
        effective_loop_batches,
        start_batch,
        loop_end_batch,
        max_batches_effective,
    )
    last_real_batch = -1
    sdpo_success_score_history: deque[list[float]] = deque(
        maxlen=max(1, int(config.sdpo_success_history_batches))
    )

    #  Main training loop
    for batch_idx in range(start_batch, loop_end_batch):
        # Start time of the batch
        t_start = time.time()
        t_batch_start = time.perf_counter()
        # Per-batch phase timing diagnostics (seconds)
        t_phase_sampler_refresh = 0.0
        t_phase_policy_launch = 0.0
        t_phase_policy_wait_and_grader_launch = 0.0
        t_phase_grader_wait = 0.0
        t_phase_scoring = 0.0
        t_phase_format_retry = 0.0
        t_phase_sdpo_success_selection = 0.0
        t_phase_grpo_datum_build = 0.0
        t_phase_sdpo_datum_build_total = 0.0
        t_phase_sdpo_teacher_logprobs = 0.0
        t_phase_sdpo_ref_logprobs = 0.0
        t_phase_write_training_logs = 0.0
        t_phase_write_batch_summary = 0.0
        t_phase_forward_backward = 0.0
        t_phase_optimizer_step = 0.0
        
        if is_eval:
            real_batch = actual_batch
        else:
            real_batch = actual_batch + (batch_idx - start_batch)
        last_real_batch = real_batch
        effective_min_words = int(config.min_words)
        if (
            not is_eval
            and config.min_words_warmup_enabled
            and real_batch < config.min_words_warmup_batches
        ):
            effective_min_words = min(
                int(config.min_words),
                int(config.min_words_warmup_value),
            )

        # Save checkpoint periodically
        if config.save_every > 0 and batch_idx % config.save_every == 0 and batch_idx > 0:
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{real_batch:06d}_{config.today_date}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": real_batch},
            )

        # Get training batch 
        batch_start = batch_idx * effective_batch_size # Since one batch swipe batch_size data, the start batch should be n*batch_size
        batch_end = min((batch_idx + 1) * effective_batch_size, len(dataset))
        batch_rows = dataset.select(range(batch_start, batch_end)) # batch_rows is updated for different batches

        sampler_refresh_seconds = 0.0
        sampler_refreshed = 0.0
        if not is_eval:
            refresh_interval = max(1, int(config.sampling_client_refresh_interval_batches))
            should_refresh_sampler = (
                sampling_client is None
                or ((batch_idx - start_batch) % refresh_interval == 0)
            )
            if should_refresh_sampler:
                t_sampler_refresh = time.perf_counter()
                sampling_client = await training_client.save_weights_and_get_sampling_client_async()
                sampler_refresh_seconds = time.perf_counter() - t_sampler_refresh
                t_phase_sampler_refresh = sampler_refresh_seconds
                sampler_refreshed = 1.0
                logger.info(
                    "Sampling client refresh took %.2fs (interval=%s)",
                    sampler_refresh_seconds,
                    refresh_interval,
                )
            else:
                logger.info(
                    "Reusing sampling client for batch %s (refresh interval=%s)",
                    real_batch,
                    refresh_interval,
                )
        # else:
        #   eval mode: sampling_client already prepared

        assert sampling_client is not None, "Sampling client must be initialized before batch sampling."

        policy_tasks = []


        if is_eval:
            logger.info(f"EVAL MODE: Checkpoint {real_batch -1}, Launching generation for {len(batch_rows)} goals...")
            logger.info(
                "EVAL MODE: Batch %s, Launching %s samples for %s goals",
                batch_idx,
                effective_group_size,
                effective_batch_size,
            )
        else:
            logger.info(f"TRAINING MODE: Batch {real_batch}: Launching generation for {len(batch_rows)} goals...")
            logger.info(
                "TRAINING MODE: For batch %s, Launching %s samples for %s goals",
                real_batch,
                effective_group_size,
                effective_batch_size,
            )

        # --- PHASE 1: LAUNCH POLICY GENERATIONS (ASYNC) ---
        t_policy_launch_start = time.perf_counter()
        for goal_idx, goal in enumerate(batch_rows["Goal"]): #interate for every research goal in the batch
            #logger.info(f"Batch {real_batch}: launching {config.group_size} samples for goal index {goal_idx}")
            
            prompt_text = build_research_plan_prompt(
                scenario=goal,
                examples=None,
                policy_output_mode=config.policy_output_mode,
            )

            convo = [
                {
                    "role": "user", 
                    "content": prompt_text,
                 },
            ]

            #logger.info(f"Batch {real_batch}: Launching generation for {len(batch_rows)} goals...")

            # Turn the prompt from text to tokens
            model_input = renderer.build_generation_prompt(convo)
            policy_tasks.append(
                asyncio.create_task(
                    _sample_policy_limited(
                        semaphore=policy_request_semaphore,
                        sampling_client=sampling_client,
                        model_input=model_input,
                        goal_idx=goal_idx,
                        prompt_tokens=model_input.to_ints(),
                        group_size=effective_group_size,
                        sampling_params=sampling_params,
                    ),
                    name=f"policy_goal_{goal_idx}",
                )
            )
        t_phase_policy_launch = time.perf_counter() - t_policy_launch_start

        # --- PHASE 2: COLLECT PLANS & LAUNCH GRADERS (ASYNC) ---
        # We process policy results as they arrive and fire off grader requests.

        if is_eval:
            logger.info(f"EVAL MODE: Batch {batch_idx}, collecting plans and launching graders...")
        else:   
            logger.info(f"TRAINING MODE: Batch {real_batch}: collecting plans and launching graders...")
        
        batch_groups_data: list[dict[str, object] | None] = [None] * len(batch_rows["Goal"])
        grader_tasks = []

        t_policy_wait_and_grader_launch_start = time.perf_counter()
        for p_task in asyncio.as_completed(policy_tasks):
            policy_result = await p_task
            if policy_result is None:
                continue

            goal_idx, prompt_tokens, result = policy_result

            # Prepare data for this batch to grade
            goal = batch_rows["Goal"][goal_idx]
            rubric = batch_rows["Rubric"][goal_idx]
            ref_sol = batch_rows["Reference solution"][goal_idx]
            
            group_samples_info: list[dict[str, object]] = []
            grader_results: list[types.SampleResponse | None] = []

            # Iterate through a group_size
            for sample_idx, group_result in enumerate(result.sequences):
                proposed_plan_raw = renderers.get_text_content(
                    renderer.parse_response(group_result.tokens)[0]
                )
                proposed_plan_for_grader = normalize_policy_output_for_grader(proposed_plan_raw)
                if config.grade_solution_only:
                    strict_solution = extract_solution_text(
                        proposed_plan_for_grader,
                        require_tags=True,
                    )
                    proposed_plan_for_grader = (
                        strict_solution
                        if (strict_solution is not None and strict_solution.strip())
                        else "FORMAT VIOLATION: Missing or malformed <solution>...</solution> block."
                    )
                
                # Build Grader Prompt (CPU operation - Fast)
                grader_prompt_text = build_grader_prompt(
                    scenario=goal,
                    rubric_items=rubric,
                    proposed_plan=proposed_plan_for_grader,
                    reference_solution=ref_sol,
                    grader_prompt_mode=config.grader_prompt_mode,
                    fast_grader_mode=config.fast_grader_mode,
                    fast_reasoning_max_words_per_item=config.fast_grader_reasoning_max_words_per_item,
                    fast_summary_max_words=config.fast_grader_summary_max_words,
                )

                # Convert prompt to tokens
                grader_input = renderer.build_generation_prompt([{"role": "user", "content": grader_prompt_text}])

                # CRITICAL OPTIMIZATION: Launch Grader and DO NOT WAIT.
                # Store the future and move to the next item immediately.
                grader_tasks.append(
                    asyncio.create_task(
                        _sample_grader_limited(
                            semaphore=grader_request_semaphore,
                            grader_client=grader_client,
                            grader_input=grader_input,
                            group_idx=goal_idx,
                            sample_idx=sample_idx,
                            grader_sampling_params=grader_sampling_params,
                        ),
                        name=f"grader_goal_{goal_idx}_sample_{sample_idx}",
                    )
                )
                
                group_samples_info.append({
                    "tokens": group_result.tokens,
                    "logprobs": group_result.logprobs,
                    "text": proposed_plan_raw,  # Backward-compatible key for diagnostics
                    "text_raw": proposed_plan_raw,
                    "text_for_grader": proposed_plan_for_grader,
                })
                grader_results.append(None)

            batch_groups_data[goal_idx] = {
                "goal": goal,
                "rubric": rubric,
                "reference_solution": ref_sol,
                "prompt_tokens": prompt_tokens,
                "samples_info": group_samples_info,
                "grader_results": grader_results,
            }
        t_phase_policy_wait_and_grader_launch = (
            time.perf_counter() - t_policy_wait_and_grader_launch_start
        )

        t_grader_wait_start = time.perf_counter()
        for g_task in asyncio.as_completed(grader_tasks):
            group_idx, sample_idx, grader_result = await g_task
            group_data = batch_groups_data[group_idx]
            if group_data is None:
                continue
            group_data["grader_results"][sample_idx] = grader_result
        t_phase_grader_wait = time.perf_counter() - t_grader_wait_start


        # --- PHASE 3: COLLECT GRADES & COMPUTE ADVANTAGES ---
        if is_eval:
            logger.info(f"EVAL MODE: Batch {batch_idx}: Waiting for grading results...")
        else:
            logger.info(f"TRAINING MODE: Batch {real_batch}: Waiting for grading results...")
        
        training_datums = []
        batch_rewards = []
        batch_sample_rewards = []
        batch_valid_rubric_scores = []
        batch_logs_to_save = []
        dropped_samples = 0
        drop_reason_counts: dict[str, int] = {
            "grader_no_response": 0,
            "grader_parse_error": 0,
            "grader_xml_invalid": 0,
            "format_noncompliant": 0,
            "think_too_long": 0,
            "non_solution_too_long": 0,
            "template_copy_solution": 0,
            "too_short_solution": 0,
            "degenerate_plan_text": 0,
            "missing_logprobs": 0,
        }
        format_noncompliance_scored_counts: dict[str, int] = {
            "missing_solution_tags": 0,
            "missing_solution_open_tag": 0,
            "missing_solution_close_tag": 0,
            "malformed_solution_tags": 0,
            "solution_too_long": 0,
        }
        format_noncompliance_drop_counts: dict[str, int] = {
            "missing_solution_tags": 0,
            "missing_solution_open_tag": 0,
            "missing_solution_close_tag": 0,
            "malformed_solution_tags": 0,
            "solution_too_long": 0,
        }
        num_scored_samples = 0
        num_trainability_checks = 0
        num_trainability_pass = 0
        num_noncompliant_scored = 0
        num_think_too_long_scored = 0
        num_think_too_long_kept = 0
        num_non_solution_too_long_scored = 0
        num_non_solution_too_long_kept = 0
        num_template_copy_scored = 0
        num_too_short_scored = 0
        think_penalty_applied = 0
        non_solution_penalty_applied = 0
        num_grader_fallback_scored = 0
        num_grader_fallback_from_parse_error = 0
        num_format_retry_attempts = 0
        num_format_retry_recovered = 0
        
        # Batch-level diagnostics containers
        batch_word_counts = []
        batch_total_word_counts = []
        batch_think_word_counts = []
        batch_non_solution_word_counts = []
        batch_think_penalties = []
        batch_non_solution_penalties = []
        batch_format_penalties = []
        batch_format_violations = []
        batch_rubric_scores = []

        # Diagnostics for advantage distributions
        batch_grpo_advantages = []
        batch_sdpo_advantages = []
        batch_sdpo_solution_token_fractions = []
        batch_sdpo_solution_mask_missing = 0
        batch_sdpo_solution_mask_zero = 0
        batch_sdpo_solution_mask_bad_shape = 0
        batch_sdpo_solution_mask_expected = 0
        batch_sdpo_teacher_requests = 0
        batch_sdpo_teacher_overlap = 0
        batch_sdpo_teacher_prompt_chars = []
        batch_sdpo_teacher_prompt_tokens = []
        batch_sdpo_feedback_chars = []
        batch_sdpo_feedback_bullets = []
        batch_sdpo_feedback_low_level_items = []
        batch_advantages = []
        num_total_samples = 0
        num_valid_samples = 0

        group_training_payloads = []
        batch_sdpo_success_cutoff = 0.0
        batch_sdpo_success_cutoff_source = "disabled"
        batch_sdpo_success_rate_groups = 0.0
        batch_sdpo_success_min_required = 0
        batch_sdpo_success_candidates = 0
        batch_sdpo_success_backoff_applied = 0.0
        groups_seen = 0
        groups_skipped_no_rewards = 0
        groups_skipped_zero_adv = 0
        groups_skipped_too_few_valid = 0
        groups_used_for_training = 0

        # Iterate over batch_size of groups
        t_scoring_start = time.perf_counter()
        for group_idx, group_data in enumerate(batch_groups_data):
            if group_data is None:
                continue
            groups_seen += 1
            
            group_rewards = []
            valid_samples = []

            # Iterate over group_size to get Rewards for a single group
            for j, each_group_result in enumerate(group_data["grader_results"]):

                num_total_samples += 1
                sample_info = group_data["samples_info"][j]
                plan_text_raw = sample_info.get("text_raw", sample_info.get("text", ""))
                plan_text_for_grader = sample_info.get("text_for_grader", plan_text_raw)
                sample_logprobs = sample_info["logprobs"]
                total_word_count = len(plan_text_raw.strip().split())
                solution_text = extract_solution_text(plan_text_raw, require_tags=True)
                solution_word_count = (
                    len(solution_text.split()) if solution_text is not None else 0
                )
                word_count = solution_word_count
                think_text = extract_think_text(plan_text_raw, require_tags=True)
                think_word_count = len(think_text.split()) if think_text is not None else 0
                non_solution_text = extract_non_solution_text(plan_text_raw)
                non_solution_word_count = len(non_solution_text.split())
                is_template_copy = is_template_copy_solution(solution_text)
                format_failure_subtype = classify_format_noncompliance(
                    plan_text_raw, config.max_word_count
                )
                format_retry_attempted = False
                format_retry_recovered = False

                if (
                    not is_eval
                    and config.format_retry_enabled
                    and format_failure_subtype is not None
                    and config.format_retry_max_attempts > 0
                ):
                    for _ in range(config.format_retry_max_attempts):
                        num_format_retry_attempts += 1
                        format_retry_attempted = True
                        t_format_retry_start = time.perf_counter()
                        retry_result = await _retry_format_sample_once(
                            sampling_client=sampling_client,
                            grader_client=grader_client,
                            renderer=renderer,
                            tokenizer=tokenizer,
                            prompt_tokens=[int(t) for t in group_data["prompt_tokens"]],
                            scenario=group_data["goal"],
                            rubric_items=group_data["rubric"],
                            reference_solution=group_data["reference_solution"],
                            sampling_params=format_retry_sampling_params,
                            grader_sampling_params=grader_sampling_params,
                            grade_solution_only=config.grade_solution_only,
                            fast_grader_mode=config.fast_grader_mode,
                            grader_prompt_mode=config.grader_prompt_mode,
                            fast_grader_reasoning_max_words_per_item=config.fast_grader_reasoning_max_words_per_item,
                            fast_grader_summary_max_words=config.fast_grader_summary_max_words,
                        )
                        t_phase_format_retry += time.perf_counter() - t_format_retry_start
                        if retry_result is None:
                            continue

                        retry_sample_info, retry_grader_result = retry_result
                        sample_info = retry_sample_info
                        each_group_result = retry_grader_result
                        plan_text_raw = sample_info.get("text_raw", sample_info.get("text", ""))
                        plan_text_for_grader = sample_info.get("text_for_grader", plan_text_raw)
                        sample_logprobs = sample_info["logprobs"]
                        total_word_count = len(plan_text_raw.strip().split())
                        solution_text = extract_solution_text(plan_text_raw, require_tags=True)
                        solution_word_count = (
                            len(solution_text.split()) if solution_text is not None else 0
                        )
                        word_count = solution_word_count
                        think_text = extract_think_text(plan_text_raw, require_tags=True)
                        think_word_count = len(think_text.split()) if think_text is not None else 0
                        non_solution_text = extract_non_solution_text(plan_text_raw)
                        non_solution_word_count = len(non_solution_text.split())
                        is_template_copy = is_template_copy_solution(solution_text)
                        format_failure_subtype = classify_format_noncompliance(
                            plan_text_raw, config.max_word_count
                        )
                        if format_failure_subtype is None:
                            format_retry_recovered = True
                            num_format_retry_recovered += 1
                            break

                if each_group_result is None:
                    dropped_samples += 1
                    drop_reason_counts["grader_no_response"] += 1
                    batch_logs_to_save.append(
                        {
                            "batch_idx": real_batch,
                            "group_idx": group_idx,
                            "sample_idx": j,
                            "status": "dropped",
                            "drop_reason": "grader_no_response",
                            "policy_output": plan_text_raw,
                            "policy_output_raw": plan_text_raw,
                            "policy_output_repaired": plan_text_for_grader,
                            "grader_output": None,
                            "rubric_score": None,
                            "grader_score_source": None,
                            "grader_fallback_levels_found": 0,
                            "format_penalty": None,
                            "think_penalty": None,
                            "non_solution_penalty": None,
                            "final_reward": None,
                            "word_count": word_count,
                            "word_count_total": total_word_count,
                            "word_count_solution": solution_word_count,
                            "think_word_count": think_word_count,
                            "non_solution_word_count": non_solution_word_count,
                            "format_failure_subtype": format_failure_subtype,
                            "format_retry_attempted": format_retry_attempted,
                            "format_retry_recovered": format_retry_recovered,
                            "is_template_copy_solution": is_template_copy,
                            "is_compliant": None,
                            "is_trainable": False,
                        }
                    )
                    continue
                    
                # Parse grader output; if strict parsing fails, try raw decode fallback.
                xml_text: str | None = None
                parse_failed = False
                try:
                    parsed_message, _ = renderer.parse_response(each_group_result.sequences[0].tokens)
                    xml_text = renderers.get_text_content(parsed_message)
                except Exception:
                    parse_failed = True
                    logger.exception(
                        "Failed to parse grader output for group_idx=%s sample_idx=%s",
                        group_idx,
                        j,
                    )
                    try:
                        xml_text = tokenizer.decode(
                            each_group_result.sequences[0].tokens,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    except Exception:
                        xml_text = None

                rubric_score: float | None = None
                grader_score_source = "xml_valid"
                grader_fallback_levels_found = 0
                if xml_text is not None and is_valid_grader_xml(xml_text):
                    rubric_score = compute_rubric_reward_from_xml(xml_text)
                elif xml_text is not None and config.allow_grader_xml_fallback:
                    fallback_score, grader_fallback_levels_found = compute_rubric_reward_from_loose_levels(
                        xml_text,
                        min_total_levels=config.grader_xml_fallback_min_levels,
                    )
                    if fallback_score is not None:
                        rubric_score = float(fallback_score)
                        grader_score_source = "xml_fallback_levels"
                        num_grader_fallback_scored += 1
                        if parse_failed:
                            num_grader_fallback_from_parse_error += 1

                if rubric_score is None:
                    logger.debug(
                        "Dropping sample with invalid grader output for group_idx=%s sample_idx=%s "
                        "(parse_failed=%s, fallback_levels=%s)",
                        group_idx,
                        j,
                        parse_failed,
                        grader_fallback_levels_found,
                    )
                    dropped_samples += 1
                    drop_reason = "grader_parse_error" if parse_failed else "grader_xml_invalid"
                    drop_reason_counts[drop_reason] += 1
                    batch_logs_to_save.append(
                        {
                            "batch_idx": real_batch,
                            "group_idx": group_idx,
                            "sample_idx": j,
                            "status": "dropped",
                            "drop_reason": drop_reason,
                            "policy_output": plan_text_raw,
                            "policy_output_raw": plan_text_raw,
                            "policy_output_repaired": plan_text_for_grader,
                            "grader_output": xml_text,
                            "rubric_score": None,
                            "grader_score_source": None,
                            "grader_fallback_levels_found": grader_fallback_levels_found,
                            "format_penalty": None,
                            "think_penalty": None,
                            "non_solution_penalty": None,
                            "final_reward": None,
                            "word_count": word_count,
                            "word_count_total": total_word_count,
                            "word_count_solution": solution_word_count,
                            "think_word_count": think_word_count,
                            "non_solution_word_count": non_solution_word_count,
                            "format_failure_subtype": format_failure_subtype,
                            "format_retry_attempted": format_retry_attempted,
                            "format_retry_recovered": format_retry_recovered,
                            "is_template_copy_solution": is_template_copy,
                            "is_compliant": None,
                            "is_trainable": False,
                        }
                    )
                    continue

                num_scored_samples += 1

                # Format Penalty 
                # "We penalize the model if the content within <solution> tags exceeds 750 words or if tags are missing"
                # "reward = (satisfied/total) - 1{format penalty}"
                is_compliant = check_format_compliance(plan_text_raw, config.max_word_count)
                final_reward, format_penalty = compute_final_reward(
                    rubric_score=rubric_score,
                    is_compliant=is_compliant,
                    solution_word_count=word_count,
                    config=config,
                )
                think_penalty = compute_think_penalty(think_word_count, config)
                if think_penalty > 0.0:
                    final_reward = final_reward - think_penalty
                    think_penalty_applied += 1
                non_solution_penalty = compute_non_solution_penalty(
                    non_solution_word_count,
                    config,
                )
                if non_solution_penalty > 0.0:
                    final_reward = final_reward - non_solution_penalty
                    non_solution_penalty_applied += 1
                batch_think_penalties.append(think_penalty)
                batch_non_solution_penalties.append(non_solution_penalty)
                if not is_compliant:
                    num_noncompliant_scored += 1
                    if format_failure_subtype in format_noncompliance_scored_counts:
                        format_noncompliance_scored_counts[format_failure_subtype] += 1
                if (
                    config.policy_output_mode == "think_solution"
                    and think_word_count > config.max_think_words_hard
                ):
                    num_think_too_long_scored += 1
                if non_solution_word_count > config.max_non_solution_words_hard:
                    num_non_solution_too_long_scored += 1
                if is_template_copy:
                    num_template_copy_scored += 1
                if solution_word_count < effective_min_words:
                    num_too_short_scored += 1


                # Record raw sample statistics
                batch_word_counts.append(word_count)
                batch_total_word_counts.append(total_word_count)
                batch_think_word_counts.append(think_word_count)
                batch_non_solution_word_counts.append(non_solution_word_count)
                batch_format_penalties.append(format_penalty)
                batch_format_violations.append(0.0 if is_compliant else 1.0)
                batch_rubric_scores.append(rubric_score)

                trainability_drop_reason = get_trainability_drop_reason(
                    plan_text=plan_text_raw,
                    solution_word_count=word_count,
                    solution_text=solution_text,
                    think_word_count=think_word_count,
                    max_think_words_hard=(
                        config.max_think_words_hard
                        if config.policy_output_mode == "think_solution"
                        else None
                    ),
                    non_solution_word_count=non_solution_word_count,
                    max_non_solution_words_hard=config.max_non_solution_words_hard,
                    is_compliant=is_compliant,
                    sample_logprobs=sample_logprobs,
                    drop_noncompliant_samples=config.drop_noncompliant_samples,
                    drop_template_copy_solutions=config.drop_template_copy_solutions,
                    min_words=effective_min_words,
                )
                if trainability_drop_reason == "think_too_long":
                    if not should_drop_think_too_long(
                        config=config,
                        current_think_too_long_drops=drop_reason_counts["think_too_long"],
                        num_trainability_checks=num_trainability_checks,
                    ):
                        trainability_drop_reason = None
                        num_think_too_long_kept += 1
                elif trainability_drop_reason == "non_solution_too_long":
                    if not should_drop_non_solution_too_long(
                        config=config,
                        current_non_solution_too_long_drops=drop_reason_counts["non_solution_too_long"],
                        num_trainability_checks=num_trainability_checks,
                    ):
                        trainability_drop_reason = None
                        num_non_solution_too_long_kept += 1

                is_trainable = trainability_drop_reason is None
                num_trainability_checks += 1
                if is_trainable:
                    num_trainability_pass += 1
                else:
                    dropped_samples += 1
                    if trainability_drop_reason is not None:
                        drop_reason_counts[trainability_drop_reason] += 1
                        if (
                            trainability_drop_reason == "format_noncompliant"
                            and format_failure_subtype in format_noncompliance_drop_counts
                        ):
                            format_noncompliance_drop_counts[format_failure_subtype] += 1

                # Log generation info for analysis (both kept and dropped)
                batch_logs_to_save.append(
                    {
                        "batch_idx": real_batch,
                        "group_idx": group_idx,
                        "sample_idx": j,
                        "status": "valid" if is_trainable else "dropped",
                        "drop_reason": trainability_drop_reason,
                        "policy_output": plan_text_raw,
                        "policy_output_raw": plan_text_raw,
                        "policy_output_repaired": plan_text_for_grader,
                        "grader_output": xml_text,
                        "rubric_score": rubric_score,
                        "grader_score_source": grader_score_source,
                        "grader_fallback_levels_found": grader_fallback_levels_found,
                        "format_penalty": format_penalty,
                        "think_penalty": think_penalty,
                        "non_solution_penalty": non_solution_penalty,
                        "final_reward": final_reward,
                        "word_count": word_count,
                        "word_count_total": total_word_count,
                        "word_count_solution": solution_word_count,
                        "think_word_count": think_word_count,
                        "non_solution_word_count": non_solution_word_count,
                        "format_failure_subtype": format_failure_subtype,
                        "format_retry_attempted": format_retry_attempted,
                        "format_retry_recovered": format_retry_recovered,
                        "is_template_copy_solution": is_template_copy,
                        "is_compliant": is_compliant,
                        "is_trainable": is_trainable,
                    }
                )
                if not is_trainable:
                    continue

                # Only being used in a group level
                group_rewards.append(final_reward)
                # Being used in a batch level, for diagnostics
                batch_sample_rewards.append(final_reward)

                # Count for valid samples
                valid_samples.append(
                    {
                        "sample_info": sample_info,
                        "reward": final_reward,
                        "rubric_score": rubric_score,
                        "grader_output": xml_text,
                    }
                )
                batch_valid_rubric_scores.append(rubric_score)

                num_valid_samples += 1

            # GRPO Advantage Calculation
            if not group_rewards:
                groups_skipped_no_rewards += 1
                continue

            mean_reward = float(np.mean(group_rewards))
            advantages = compute_group_advantages(
                group_rewards=group_rewards,
                normalize=config.normalize_grpo_advantages,
            )
            batch_rewards.append(mean_reward)

            # Collect GRPO advantages for diagnostics
            batch_grpo_advantages.extend(advantages)


            # Skip if no learning signal (all rewards identical)
            if np.allclose(advantages, 0.0):
                groups_skipped_zero_adv += 1
                continue

            # Skip if not enough valid samples
            if len(valid_samples) < 2:
                logger.debug("Skipping group: not enough valid samples")
                groups_skipped_too_few_valid += 1
                continue
            if not is_eval:
                groups_used_for_training += 1
                best_valid_sample = max(
                    valid_samples,
                    key=lambda x: (x["rubric_score"], x["reward"]),
                )
                best_valid_solution_text = extract_solution_text(
                    best_valid_sample["sample_info"].get(
                        "text_raw",
                        best_valid_sample["sample_info"].get("text", ""),
                    ),
                    require_tags=True,
                )

                group_training_payloads.append(
                    {
                        "group_idx": group_idx,
                        "goal": group_data["goal"],
                        "prompt_tokens": [int(t) for t in group_data["prompt_tokens"]],
                        "valid_samples": valid_samples,
                        "grpo_advantages": advantages,
                        "best_valid_rubric_score": float(best_valid_sample["rubric_score"]),
                        "best_valid_solution_text": best_valid_solution_text,
                        "successful_previous_rollout": None,
                    }
                )

        if not is_eval and config.use_sdpo and group_training_payloads:
            t_sdpo_success_start = time.perf_counter()
            current_batch_scores = [
                float(payload["best_valid_rubric_score"])
                for payload in group_training_payloads
            ]
            (
                batch_sdpo_success_cutoff,
                batch_sdpo_success_cutoff_source,
            ) = compute_sdpo_success_cutoff(
                history_batches_scores=list(sdpo_success_score_history),
                current_batch_scores=current_batch_scores,
                mode=config.sdpo_success_mode,
                adaptive_quantile=config.sdpo_success_adaptive_quantile,
                min_samples=config.sdpo_success_min_samples,
                use_current_batch_bootstrap=config.sdpo_success_use_current_batch_bootstrap,
                floor=config.sdpo_success_floor,
                ceiling=config.sdpo_success_ceiling,
                fixed_threshold=config.sdpo_success_threshold,
            )

            total_groups = len(group_training_payloads)
            required_by_rate = int(math.ceil(config.sdpo_success_min_rate * total_groups))
            batch_sdpo_success_min_required = max(
                0,
                min(total_groups, max(int(config.sdpo_success_min_groups), required_by_rate)),
            )

            candidate_rollouts: list[tuple[int, float, str]] = []
            for payload_idx, payload in enumerate(group_training_payloads):
                best_score = float(payload["best_valid_rubric_score"])
                best_solution_text = payload.get("best_valid_solution_text")
                if isinstance(best_solution_text, str) and best_solution_text.strip():
                    candidate_rollouts.append((payload_idx, best_score, best_solution_text))
            batch_sdpo_success_candidates = len(candidate_rollouts)

            success_payload_idxs = {
                payload_idx
                for payload_idx, score, _solution in candidate_rollouts
                if score >= batch_sdpo_success_cutoff
            }

            target_success_count = min(batch_sdpo_success_min_required, len(candidate_rollouts))
            if target_success_count > 0 and len(success_payload_idxs) < target_success_count:
                sorted_scores = sorted(
                    [score for _payload_idx, score, _solution in candidate_rollouts],
                    reverse=True,
                )
                relaxed_cutoff = sorted_scores[target_success_count - 1]
                if relaxed_cutoff < batch_sdpo_success_cutoff:
                    batch_sdpo_success_cutoff = float(relaxed_cutoff)
                    batch_sdpo_success_cutoff_source = (
                        f"{batch_sdpo_success_cutoff_source}+min_success_backoff"
                    )
                    batch_sdpo_success_backoff_applied = 1.0

                success_payload_idxs = {
                    payload_idx
                    for payload_idx, score, _solution in candidate_rollouts
                    if score >= batch_sdpo_success_cutoff
                }

            success_groups = 0
            for payload_idx, payload in enumerate(group_training_payloads):
                best_solution_text = payload.get("best_valid_solution_text")
                if not isinstance(best_solution_text, str):
                    best_solution_text = None

                is_success = (
                    payload_idx in success_payload_idxs
                    and bool(best_solution_text and best_solution_text.strip())
                )
                payload["successful_previous_rollout"] = (
                    best_solution_text if is_success else None
                )
                if is_success:
                    success_groups += 1

            batch_sdpo_success_rate_groups = success_groups / float(len(group_training_payloads))
            sdpo_success_score_history.append(current_batch_scores)
            t_phase_sdpo_success_selection = time.perf_counter() - t_sdpo_success_start
        t_phase_scoring = time.perf_counter() - t_scoring_start

        if not is_eval and not config.use_sdpo:
            # Fallback: standard GRPO-style scalar sequence advantages.
            t_grpo_datum_start = time.perf_counter()
            for payload in group_training_payloads:
                prompt_tokens = payload["prompt_tokens"]
                ob_len = len(prompt_tokens) - 1
                for k, sample in enumerate(payload["valid_samples"]):
                    sample_info = sample["sample_info"]
                    grpo_advantage = payload["grpo_advantages"][k]

                    generated_tokens = [int(t) for t in sample_info["tokens"]]
                    full_seq = prompt_tokens + generated_tokens
                    input_tokens = full_seq[:-1]
                    target_tokens = full_seq[1:]

                    student_logprobs = [float(lp) for lp in sample_info["logprobs"]]
                    all_logprobs = [0.0] * ob_len + student_logprobs
                    all_advantages = [0.0] * ob_len + [grpo_advantage] * len(student_logprobs)
                    batch_advantages.extend([grpo_advantage] * len(student_logprobs))

                    training_datums.append(
                        types.Datum(
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
                    )
            t_phase_grpo_datum_build = time.perf_counter() - t_grpo_datum_start

        if not is_eval and config.use_sdpo and group_training_payloads:
            # SDPO: compute self-teacher logprobs for original sampled trajectories.
            t_sdpo_datum_start = time.perf_counter()
            teacher_requests = []
            for payload in group_training_payloads:
                for k, sample in enumerate(payload["valid_samples"]):
                    sample_info = sample["sample_info"]
                    sample_tokens = [int(t) for t in sample_info["tokens"]]
                    student_logprobs = [float(lp) for lp in sample_info["logprobs"]]
                    sdpo_solution_mask = None
                    if config.sdpo_solution_only_tokens:
                        solution_mask_np = build_solution_token_mask_from_tokens(
                            tokenizer=tokenizer,
                            sample_tokens=sample_tokens,
                        )
                        sdpo_solution_mask = solution_mask_np.tolist()

                    (
                        feedback_text,
                        feedback_bullet_count,
                        feedback_low_level_item_count,
                    ) = extract_weaknesses_from_grader(
                        sample["grader_output"],
                        max_bullets_per_item=config.sdpo_feedback_max_bullets_per_item,
                        max_items=config.sdpo_teacher_max_criticized_items,
                        focus_low_confidence_items_only=config.sdpo_teacher_focus_low_confidence_items_only,
                        include_sample_review=config.sdpo_teacher_include_sample_review,
                        include_item_reasoning_feedback=config.sdpo_teacher_include_item_reasoning_feedback,
                        include_item_desiderata_review=config.sdpo_teacher_include_item_desiderata_review,
                        include_global_desiderata_summary=config.sdpo_teacher_include_global_desiderata_summary,
                    )
                    feedback_char_cap = int(config.sdpo_feedback_max_chars)
                    if config.fast_grader_mode:
                        feedback_char_cap = min(
                            feedback_char_cap,
                            int(config.fast_grader_feedback_max_chars),
                        )
                    if feedback_char_cap > 0 and len(feedback_text) > feedback_char_cap:
                        feedback_text = (
                            feedback_text[:feedback_char_cap].rstrip()
                            + "\n...[truncated]"
                        )
                    batch_sdpo_feedback_chars.append(len(feedback_text))
                    batch_sdpo_feedback_bullets.append(float(feedback_bullet_count))
                    batch_sdpo_feedback_low_level_items.append(
                        float(feedback_low_level_item_count)
                    )
                    teacher_prompt_text = build_self_teacher_prompt(
                        scenario=payload["goal"],
                        grader_feedback=feedback_text,
                        successful_previous_rollout=payload["successful_previous_rollout"],
                        policy_output_mode=config.policy_output_mode,
                        target_word_count=config.target_word_count,
                        max_solution_words=config.max_word_count,
                        max_think_words=config.max_think_words_soft,
                    )
                    teacher_messages = [{"role": "user", "content": teacher_prompt_text}]
                    teacher_prompt = renderer.build_generation_prompt(
                        messages=teacher_messages,
                        role="assistant",
                    )
                    # Important token-boundary invariant for SDPO:
                    # [System/User prompt] -> [assistant start/header tokens] -> [sampled response tokens].
                    # build_generation_prompt provides the renderer-specific assistant header, so
                    # concatenating sample_tokens below matches the model's expected dialogue format.
                    teacher_prompt_tokens = [int(t) for t in teacher_prompt.to_ints()]
                    batch_sdpo_teacher_prompt_chars.append(len(teacher_prompt_text))
                    batch_sdpo_teacher_prompt_tokens.append(len(teacher_prompt_tokens))
                    teacher_full_sequence = types.ModelInput.from_ints(
                        tokens=teacher_prompt_tokens + sample_tokens
                    )

                    teacher_requests.append(
                        {
                            "prompt_tokens": payload["prompt_tokens"],
                            "sample_tokens": sample_tokens,
                            "student_logprobs": student_logprobs,
                            "teacher_prompt_len": len(teacher_prompt_tokens),
                            "grpo_advantage": float(payload["grpo_advantages"][k]),
                            "teacher_full_sequence": teacher_full_sequence,
                            "sdpo_solution_mask": sdpo_solution_mask,
                        }
                    )

            batch_sdpo_teacher_requests = len(teacher_requests)
            if teacher_requests:
                teacher_sequences = [req["teacher_full_sequence"] for req in teacher_requests]
                t_sdpo_teacher_logprob_start = time.perf_counter()
                teacher_logprobs_all = await _compute_logprobs_async(sampling_client, teacher_sequences)
                t_phase_sdpo_teacher_logprobs += time.perf_counter() - t_sdpo_teacher_logprob_start
                if len(teacher_logprobs_all) != len(teacher_requests):
                    logger.error(
                        "Mismatch in SDPO teacher logprobs: got %s logprob lists for %s requests. "
                        "Skipping mismatched requests.",
                        len(teacher_logprobs_all),
                        len(teacher_requests),
                    )

                ref_logprobs_all = None
                if initial_teacher_client is not None and config.sdpo_teacher_reg_alpha > 0.0:
                    t_sdpo_ref_logprob_start = time.perf_counter()
                    ref_logprobs_all = await _compute_logprobs_async(
                        initial_teacher_client,
                        teacher_sequences,
                    )
                    t_phase_sdpo_ref_logprobs += time.perf_counter() - t_sdpo_ref_logprob_start

                for idx, req in enumerate(teacher_requests):
                    if idx >= len(teacher_logprobs_all):
                        logger.warning(
                            "Missing teacher logprobs for request idx=%s (available=%s). Skipping.",
                            idx,
                            len(teacher_logprobs_all),
                        )
                        continue

                    prompt_tokens = req["prompt_tokens"]
                    sample_tokens = req["sample_tokens"]
                    student_logprobs = np.asarray(req["student_logprobs"], dtype=np.float32)
                    if student_logprobs.size == 0:
                        continue

                    t_start_idx = req["teacher_prompt_len"]
                    teacher_logprobs = teacher_logprobs_all[idx][
                        t_start_idx : t_start_idx + len(sample_tokens)
                    ]
                    teacher_logprobs = np.asarray(teacher_logprobs, dtype=np.float32)

                    if ref_logprobs_all is not None and idx < len(ref_logprobs_all):
                        ref_lp = ref_logprobs_all[idx][t_start_idx : t_start_idx + len(sample_tokens)]
                        ref_lp = np.asarray(ref_lp, dtype=np.float32)
                        if len(ref_lp) == len(teacher_logprobs):
                            alpha = float(config.sdpo_teacher_reg_alpha)
                            teacher_logprobs = (1.0 - alpha) * teacher_logprobs + alpha * ref_lp
                    elif ref_logprobs_all is not None:
                        logger.warning(
                            "Missing reference teacher logprobs for idx=%s (available=%s).",
                            idx,
                            len(ref_logprobs_all),
                        )

                    n = min(len(sample_tokens), len(student_logprobs), len(teacher_logprobs))
                    if n == 0:
                        logger.debug(
                            "Skipping SDPO sample with zero overlap lengths: "
                            "sample_tokens=%s student_logprobs=%s teacher_logprobs=%s",
                            len(sample_tokens),
                            len(student_logprobs),
                            len(teacher_logprobs),
                        )
                        continue

                    batch_sdpo_teacher_overlap += 1
                    sample_tokens = sample_tokens[:n]
                    student_logprobs = student_logprobs[:n]
                    teacher_logprobs = teacher_logprobs[:n]

                    sdpo_advantages = teacher_logprobs - student_logprobs
                    if config.sdpo_solution_only_tokens:
                        batch_sdpo_solution_mask_expected += 1
                        solution_mask_raw = req.get("sdpo_solution_mask")
                        if solution_mask_raw is None:
                            batch_sdpo_solution_mask_missing += 1
                            batch_sdpo_solution_mask_zero += 1
                            batch_sdpo_solution_token_fractions.append(0.0)
                            solution_mask = np.zeros(n, dtype=np.float32)
                        else:
                            solution_mask = np.asarray(solution_mask_raw, dtype=np.float32)
                            if solution_mask.size != n:
                                batch_sdpo_solution_mask_bad_shape += 1
                            if solution_mask.size < n:
                                batch_sdpo_solution_mask_missing += 1
                                padded = np.zeros(n, dtype=np.float32)
                                padded[: solution_mask.size] = solution_mask
                                solution_mask = padded
                            else:
                                solution_mask = solution_mask[:n]

                            solution_token_fraction = float(np.mean(solution_mask > 0.0))
                            batch_sdpo_solution_token_fractions.append(solution_token_fraction)
                            if solution_mask.sum() <= 0.0:
                                batch_sdpo_solution_mask_zero += 1

                        sdpo_advantages = sdpo_advantages * solution_mask

                    if config.sdpo_adv_clip > 0:
                        sdpo_advantages = np.clip(
                            sdpo_advantages, -config.sdpo_adv_clip, config.sdpo_adv_clip
                        )
                    batch_sdpo_advantages.extend(sdpo_advantages.tolist())

                    lambda_grpo = float(config.sdpo_grpo_mix_lambda)
                    mixed_advantages = lambda_grpo * req["grpo_advantage"] + (
                        1.0 - lambda_grpo
                    ) * sdpo_advantages
                    batch_advantages.extend(mixed_advantages.tolist())

                    full_seq = prompt_tokens + sample_tokens
                    input_tokens = full_seq[:-1]
                    target_tokens = full_seq[1:]
                    ob_len = len(prompt_tokens) - 1

                    all_logprobs = [0.0] * ob_len + student_logprobs.tolist()
                    all_advantages = [0.0] * ob_len + mixed_advantages.tolist()

                    training_datums.append(
                        types.Datum(
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
                    )
            t_phase_sdpo_datum_build_total = time.perf_counter() - t_sdpo_datum_start

            if config.sdpo_solution_only_tokens:
                if batch_sdpo_solution_mask_expected > 0:
                    mask_zero_rate = (
                        batch_sdpo_solution_mask_zero
                        / float(batch_sdpo_solution_mask_expected)
                    )
                    if mask_zero_rate >= config.sdpo_solution_mask_warn_zero_rate:
                        logger.warning(
                            "High SDPO solution-mask zero rate this batch: %.3f (%s/%s). "
                            "Teacher requests=%s, aligned=%s.",
                            mask_zero_rate,
                            batch_sdpo_solution_mask_zero,
                            batch_sdpo_solution_mask_expected,
                            batch_sdpo_teacher_requests,
                            batch_sdpo_teacher_overlap,
                        )
                elif batch_sdpo_teacher_requests > 0:
                    logger.warning(
                        "SDPO solution-mask diagnostics unavailable: no aligned teacher/student token spans "
                        "(teacher requests=%s, aligned=%s).",
                        batch_sdpo_teacher_requests,
                        batch_sdpo_teacher_overlap,
                    )

        pre_scoring_drops = (
            drop_reason_counts["grader_no_response"]
            + drop_reason_counts["grader_parse_error"]
            + drop_reason_counts["grader_xml_invalid"]
        )
        trainability_drops = (
            drop_reason_counts["format_noncompliant"]
            + drop_reason_counts["think_too_long"]
            + drop_reason_counts["non_solution_too_long"]
            + drop_reason_counts["template_copy_solution"]
            + drop_reason_counts["too_short_solution"]
            + drop_reason_counts["degenerate_plan_text"]
            + drop_reason_counts["missing_logprobs"]
        )
        t_phase_sdpo_datum_build_nonlogprob = max(
            0.0,
            t_phase_sdpo_datum_build_total
            - t_phase_sdpo_teacher_logprobs
            - t_phase_sdpo_ref_logprobs,
        )
        t_pre_optimization_total = time.perf_counter() - t_batch_start

        batch_summary = {
                "batch_idx": real_batch,
                "reward/mode": config.reward_mode,
                "run/max_batches_effective": (
                    float(max_batches_effective)
                    if max_batches_effective is not None
                    else float(n_train_batches)
                ),
                "run/effective_loop_batches": float(effective_loop_batches),
                "runtime/sampler_refreshed": sampler_refreshed,
                "runtime/sampler_refresh_seconds": sampler_refresh_seconds,
                "runtime/policy_max_concurrency": float(config.max_concurrent_policy_requests),
                "runtime/grader_max_concurrency": float(config.max_concurrent_grader_requests),
                "grader/fast_mode": 1.0 if config.fast_grader_mode else 0.0,
                "grader/prompt_mode_compact": (
                    1.0 if config.grader_prompt_mode == "compact_confidence_only" else 0.0
                ),
                "grader/max_tokens_effective": float(effective_grader_max_tokens),
                "time/sampler_refresh": float(t_phase_sampler_refresh),
                "time/policy_launch": float(t_phase_policy_launch),
                "time/policy_wait_and_grader_launch": float(t_phase_policy_wait_and_grader_launch),
                "time/grader_wait": float(t_phase_grader_wait),
                "time/scoring_and_filtering": float(t_phase_scoring),
                "time/format_retry": float(t_phase_format_retry),
                "time/sdpo_success_selection": float(t_phase_sdpo_success_selection),
                "time/grpo_datum_build": float(t_phase_grpo_datum_build),
                "time/sdpo_datum_build_total": float(t_phase_sdpo_datum_build_total),
                "time/sdpo_teacher_logprobs": float(t_phase_sdpo_teacher_logprobs),
                "time/sdpo_ref_logprobs": float(t_phase_sdpo_ref_logprobs),
                "time/sdpo_datum_build_nonlogprob": float(
                    t_phase_sdpo_datum_build_nonlogprob
                ),
                "time/pre_optimization_total": float(t_pre_optimization_total),

                # reward
                "rubric/sample_mean_all": float(np.mean(batch_rubric_scores)) if batch_rubric_scores else 0.0,
                "rubric/sample_std_all": float(np.std(batch_rubric_scores)) if batch_rubric_scores else 0.0,
                "rubric/sample_mean_valid": float(np.mean(batch_valid_rubric_scores)) if batch_valid_rubric_scores else 0.0,
                "rubric/sample_std_valid": float(np.std(batch_valid_rubric_scores)) if batch_valid_rubric_scores else 0.0,
                "reward/sample_mean_valid": float(np.mean(batch_sample_rewards)) if batch_sample_rewards else 0.0,
                "reward/sample_std_valid": float(np.std(batch_sample_rewards)) if batch_sample_rewards else 0.0,
                "reward/group_mean_valid": float(np.mean(batch_rewards)) if batch_rewards else 0.0,
                "reward/group_std_valid": float(np.std(batch_rewards)) if batch_rewards else 0.0,

                # advantage (optimizer view, without padding with zeros)
                "advantage_mean": float(np.mean(batch_advantages)) if batch_advantages else 0.0,
                "advantage_std": float(np.std(batch_advantages)) if batch_advantages else 0.0,
                "advantage/grpo_mean": float(np.mean(batch_grpo_advantages)) if batch_grpo_advantages else 0.0,
                "advantage/grpo_std": float(np.std(batch_grpo_advantages)) if batch_grpo_advantages else 0.0,
                "advantage/sdpo_mean": float(np.mean(batch_sdpo_advantages)) if batch_sdpo_advantages else 0.0,
                "advantage/sdpo_std": float(np.std(batch_sdpo_advantages)) if batch_sdpo_advantages else 0.0,
                "sdpo/enabled": 1.0 if config.use_sdpo else 0.0,
                "sdpo/success_cutoff": batch_sdpo_success_cutoff if config.use_sdpo else 0.0,
                "sdpo/success_rate_groups": batch_sdpo_success_rate_groups if config.use_sdpo else 0.0,
                "sdpo/success_cutoff_source": batch_sdpo_success_cutoff_source if config.use_sdpo else "disabled",
                "sdpo/success_min_required": batch_sdpo_success_min_required if config.use_sdpo else 0,
                "sdpo/success_candidates": batch_sdpo_success_candidates if config.use_sdpo else 0,
                "sdpo/success_backoff_applied": batch_sdpo_success_backoff_applied if config.use_sdpo else 0.0,
                "sdpo/solution_only_tokens": 1.0 if config.sdpo_solution_only_tokens else 0.0,
                "sdpo/teacher_requests": batch_sdpo_teacher_requests,
                "sdpo/teacher_overlap": batch_sdpo_teacher_overlap,
                "sdpo/solution_token_fraction_mean": (
                    float(np.mean(batch_sdpo_solution_token_fractions))
                    if batch_sdpo_solution_token_fractions
                    else 0.0
                ),
                "sdpo/solution_mask_expected": batch_sdpo_solution_mask_expected,
                "sdpo/solution_mask_missing": batch_sdpo_solution_mask_missing,
                "sdpo/solution_mask_zero": batch_sdpo_solution_mask_zero,
                "sdpo/solution_mask_bad_shape": batch_sdpo_solution_mask_bad_shape,
                "sdpo/solution_mask_missing_rate": (
                    batch_sdpo_solution_mask_missing / float(batch_sdpo_solution_mask_expected)
                    if batch_sdpo_solution_mask_expected
                    else 0.0
                ),
                "sdpo/solution_mask_zero_rate": (
                    batch_sdpo_solution_mask_zero / float(batch_sdpo_solution_mask_expected)
                    if batch_sdpo_solution_mask_expected
                    else 0.0
                ),
                "sdpo/feedback_chars_mean": (
                    float(np.mean(batch_sdpo_feedback_chars))
                    if batch_sdpo_feedback_chars
                    else 0.0
                ),
                "sdpo/feedback_bullets_mean": (
                    float(np.mean(batch_sdpo_feedback_bullets))
                    if batch_sdpo_feedback_bullets
                    else 0.0
                ),
                "sdpo/feedback_low_level_items_mean": (
                    float(np.mean(batch_sdpo_feedback_low_level_items))
                    if batch_sdpo_feedback_low_level_items
                    else 0.0
                ),
                "sdpo/teacher_prompt_chars_mean": (
                    float(np.mean(batch_sdpo_teacher_prompt_chars))
                    if batch_sdpo_teacher_prompt_chars
                    else 0.0
                ),
                "sdpo/teacher_prompt_tokens_mean": (
                    float(np.mean(batch_sdpo_teacher_prompt_tokens))
                    if batch_sdpo_teacher_prompt_tokens
                    else 0.0
                ),

                # length
                "length_mean": float(np.mean(batch_word_counts)) if batch_word_counts else 0.0,
                "length_p90": float(np.percentile(batch_word_counts, 90)) if batch_word_counts else 0.0,
                "length/total_mean": float(np.mean(batch_total_word_counts)) if batch_total_word_counts else 0.0,
                "length/total_p90": float(np.percentile(batch_total_word_counts, 90)) if batch_total_word_counts else 0.0,
                "length/think_mean": float(np.mean(batch_think_word_counts)) if batch_think_word_counts else 0.0,
                "length/think_p90": float(np.percentile(batch_think_word_counts, 90)) if batch_think_word_counts else 0.0,
                "length/non_solution_mean": (
                    float(np.mean(batch_non_solution_word_counts))
                    if batch_non_solution_word_counts
                    else 0.0
                ),
                "length/non_solution_p90": (
                    float(np.percentile(batch_non_solution_word_counts, 90))
                    if batch_non_solution_word_counts
                    else 0.0
                ),

                # format
                "format_rate": float(np.mean(batch_format_violations)) if batch_format_violations else 0.0,
                "format/penalty_mean": float(np.mean(batch_format_penalties)) if batch_format_penalties else 0.0,
                "think/penalty_mean": float(np.mean(batch_think_penalties)) if batch_think_penalties else 0.0,
                "non_solution/penalty_mean": (
                    float(np.mean(batch_non_solution_penalties))
                    if batch_non_solution_penalties
                    else 0.0
                ),
                "think/penalty_applied_rate_scored": (
                    think_penalty_applied / float(num_scored_samples)
                    if num_scored_samples
                    else 0.0
                ),
                "non_solution/penalty_applied_rate_scored": (
                    non_solution_penalty_applied / float(num_scored_samples)
                    if num_scored_samples
                    else 0.0
                ),
                "think/too_long_drop_mode": config.think_too_long_drop_mode,
                "think/too_long_max_drop_rate": config.think_too_long_max_drop_rate,
                "non_solution/too_long_drop_mode": config.non_solution_too_long_drop_mode,
                "non_solution/too_long_max_drop_rate": config.non_solution_too_long_max_drop_rate,
                "min_words/config": config.min_words,
                "min_words/effective": effective_min_words,

                # samples
                "samples/generated": num_total_samples,
                "samples/scored": num_scored_samples,
                "samples/scored_xml_fallback": num_grader_fallback_scored,
                "samples/scored_xml_fallback_from_parse_error": num_grader_fallback_from_parse_error,
                "samples/format_retry_attempts": num_format_retry_attempts,
                "samples/format_retry_recovered": num_format_retry_recovered,
                "samples/format_retry_recovery_rate": (
                    num_format_retry_recovered / float(num_format_retry_attempts)
                    if num_format_retry_attempts
                    else 0.0
                ),
                "samples/trainability_checked": num_trainability_checks,
                "samples/trainability_pass": num_trainability_pass,
                "samples/trainability_pass_rate": (
                    num_trainability_pass / float(num_trainability_checks)
                    if num_trainability_checks
                    else 0.0
                ),
                "samples/noncompliant_scored": num_noncompliant_scored,
                "samples/noncompliant_missing_solution_tags_scored": format_noncompliance_scored_counts["missing_solution_tags"],
                "samples/noncompliant_missing_solution_open_tag_scored": format_noncompliance_scored_counts["missing_solution_open_tag"],
                "samples/noncompliant_missing_solution_close_tag_scored": format_noncompliance_scored_counts["missing_solution_close_tag"],
                "samples/noncompliant_malformed_solution_tags_scored": format_noncompliance_scored_counts["malformed_solution_tags"],
                "samples/noncompliant_solution_too_long_scored": format_noncompliance_scored_counts["solution_too_long"],
                "samples/think_too_long_scored": num_think_too_long_scored,
                "samples/think_too_long_kept": num_think_too_long_kept,
                "samples/non_solution_too_long_scored": num_non_solution_too_long_scored,
                "samples/non_solution_too_long_kept": num_non_solution_too_long_kept,
                "samples/template_copy_scored": num_template_copy_scored,
                "samples/too_short_scored": num_too_short_scored,
                "samples/valid": num_valid_samples,
                "groups/seen": groups_seen,
                "groups/used_for_training": groups_used_for_training,
                "groups/skipped_no_rewards": groups_skipped_no_rewards,
                "groups/skipped_zero_adv": groups_skipped_zero_adv,
                "groups/skipped_too_few_valid": groups_skipped_too_few_valid,
                "drops/pre_scoring_total": pre_scoring_drops,
                "drops/trainability_total": trainability_drops,
                "drops/grader_no_response": drop_reason_counts["grader_no_response"],
                "drops/grader_parse_error": drop_reason_counts["grader_parse_error"],
                "drops/grader_xml_invalid": drop_reason_counts["grader_xml_invalid"],
                "drops/format_noncompliant": drop_reason_counts["format_noncompliant"],
                "drops/format_missing_solution_tags": format_noncompliance_drop_counts["missing_solution_tags"],
                "drops/format_missing_solution_open_tag": format_noncompliance_drop_counts["missing_solution_open_tag"],
                "drops/format_missing_solution_close_tag": format_noncompliance_drop_counts["missing_solution_close_tag"],
                "drops/format_malformed_solution_tags": format_noncompliance_drop_counts["malformed_solution_tags"],
                "drops/format_solution_too_long": format_noncompliance_drop_counts["solution_too_long"],
                "drops/think_too_long": drop_reason_counts["think_too_long"],
                "drops/non_solution_too_long": drop_reason_counts["non_solution_too_long"],
                "drops/template_copy_solution": drop_reason_counts["template_copy_solution"],
                "drops/too_short_solution": drop_reason_counts["too_short_solution"],
                "drops/degenerate_plan_text": drop_reason_counts["degenerate_plan_text"],
                "drops/missing_logprobs": drop_reason_counts["missing_logprobs"],
            }
        
        # Write batch logs to file
        if batch_logs_to_save:
            t_write_training_logs_start = time.perf_counter()
            if is_eval:
                generations_log_path = os.path.join(config.log_path, f"evaluation/eval_logs({actual_batch - 1}).jsonl")
            else:    
                generations_log_path = os.path.join(config.log_path, "train/training_logs.jsonl")
            os.makedirs(os.path.dirname(generations_log_path), exist_ok=True)
            with open(generations_log_path, "a") as f:
                for log_item in batch_logs_to_save:
                    f.write(json.dumps(log_item) + "\n")
            t_phase_write_training_logs = time.perf_counter() - t_write_training_logs_start

        if batch_summary:
            t_write_batch_summary_start = time.perf_counter()
            if is_eval:
                batch_summary_path = os.path.join(config.log_path, f"evaluation/eval_batch_summary({actual_batch - 1}).jsonl")
            else:    
                batch_summary_path = os.path.join(config.log_path, "train/batch_summary.jsonl")
            os.makedirs(os.path.dirname(batch_summary_path), exist_ok=True)
            with open(batch_summary_path, "a") as f:
                f.write(json.dumps(batch_summary) + "\n")
            t_phase_write_batch_summary = time.perf_counter() - t_write_batch_summary_start
        
        # --- EVAL MODE: skip optimization ---
        if is_eval:
            logger.info("EVAL MODE: skipping optimization step.")
            continue

        # --- PHASE 4: OPTIMIZATION STEP ---
        if not training_datums:
            logger.warning("No valid datums this batch. Skipping optimization.")
            continue

        datum_diagnostics = _summarize_training_datums(training_datums)
        try:
            fwd_bwd_future = await training_client.forward_backward_async(
                training_datums,
                loss_fn="ppo",
                loss_fn_config={
                    "clip_low_threshold": 1-config.clip_eps, 
                    "clip_high_threshold": 1+config.clip_eps,
                    #"kl_coeff": 0.0, # Explicitly disable KL penalty as per Paper Appendix A.2
                }
            )
            optim_step_future = await training_client.optim_step_async(adam_params)

            t0 = time.perf_counter()
            _fwd_bwd_result = await fwd_bwd_future.result_async()
            t_phase_forward_backward = time.perf_counter() - t0
            logger.info(f"Forward/Backward took {t_phase_forward_backward:.2f}s")

            t1 = time.perf_counter()
            _optim_result = await optim_step_future.result_async()
            t_phase_optimizer_step = time.perf_counter() - t1
            logger.info(f"Optim step took {t_phase_optimizer_step:.2f}s")
        except Exception as exc:
            logger.exception(
                "Training step failed (%s) for %s datums. Datum diagnostics: %s",
                type(exc).__name__,
                len(training_datums),
                datum_diagnostics,
            )
            continue

        # Log metrics
        metrics: dict[str, float] = {
            "progress/batch": real_batch,
            "optim/lr": config.learning_rate,
            "run/max_batches_effective": (
                float(max_batches_effective)
                if max_batches_effective is not None
                else float(n_train_batches)
            ),
            "run/effective_loop_batches": float(effective_loop_batches),
            "progress/done_frac": (
                (batch_idx - start_batch + 1) / float(max(1, effective_loop_batches))
            ),
            "time/total": time.time() - t_start,
            "time/sampler_refresh": float(t_phase_sampler_refresh),
            "time/policy_launch": float(t_phase_policy_launch),
            "time/policy_wait_and_grader_launch": float(t_phase_policy_wait_and_grader_launch),
            "time/grader_wait": float(t_phase_grader_wait),
            "time/scoring_and_filtering": float(t_phase_scoring),
            "time/format_retry": float(t_phase_format_retry),
            "time/sdpo_success_selection": float(t_phase_sdpo_success_selection),
            "time/grpo_datum_build": float(t_phase_grpo_datum_build),
            "time/sdpo_datum_build_total": float(t_phase_sdpo_datum_build_total),
            "time/sdpo_teacher_logprobs": float(t_phase_sdpo_teacher_logprobs),
            "time/sdpo_ref_logprobs": float(t_phase_sdpo_ref_logprobs),
            "time/sdpo_datum_build_nonlogprob": float(t_phase_sdpo_datum_build_nonlogprob),
            "time/pre_optimization_total": float(t_pre_optimization_total),
            "time/forward_backward": float(t_phase_forward_backward),
            "time/optimizer_step": float(t_phase_optimizer_step),
            "time/write_training_logs": float(t_phase_write_training_logs),
            "time/write_batch_summary": float(t_phase_write_batch_summary),
            "runtime/sampler_refreshed": sampler_refreshed,
            "runtime/sampler_refresh_seconds": sampler_refresh_seconds,
            "runtime/policy_max_concurrency": float(config.max_concurrent_policy_requests),
            "runtime/grader_max_concurrency": float(config.max_concurrent_grader_requests),
            "runtime/policy_tasks": float(len(policy_tasks)),
            "runtime/grader_tasks": float(len(grader_tasks)),
            "grader/fast_mode": 1.0 if config.fast_grader_mode else 0.0,
            "grader/prompt_mode_compact": (
                1.0 if config.grader_prompt_mode == "compact_confidence_only" else 0.0
            ),
            "grader/max_tokens_effective": float(effective_grader_max_tokens),
            "reward/total": sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0,
            "objective/total": sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0,
            "rubric/total_all": float(np.mean(batch_rubric_scores)) if batch_rubric_scores else 0.0,
            "rubric/total_valid": float(np.mean(batch_valid_rubric_scores)) if batch_valid_rubric_scores else 0.0,
            "format_rate": float(np.mean(batch_format_violations)) if batch_format_violations else 0.0,
            "length/solution_mean": float(np.mean(batch_word_counts)) if batch_word_counts else 0.0,
            "length/total_mean": float(np.mean(batch_total_word_counts)) if batch_total_word_counts else 0.0,
            "length/think_mean": float(np.mean(batch_think_word_counts)) if batch_think_word_counts else 0.0,
            "reward/is_rubric_only": 1.0 if config.reward_mode == "rubric_only" else 0.0,
            "reward/is_paper": 1.0 if config.reward_mode == "paper" else 0.0,
            "dropped_samples": dropped_samples,
            "samples/generated": float(num_total_samples),
            "samples/scored": float(num_scored_samples),
            "samples/scored_xml_fallback": float(num_grader_fallback_scored),
            "samples/scored_xml_fallback_from_parse_error": float(
                num_grader_fallback_from_parse_error
            ),
            "samples/format_retry_attempts": float(num_format_retry_attempts),
            "samples/format_retry_recovered": float(num_format_retry_recovered),
            "samples/format_retry_recovery_rate": (
                num_format_retry_recovered / float(num_format_retry_attempts)
                if num_format_retry_attempts
                else 0.0
            ),
            "samples/valid": float(num_valid_samples),
            "samples/trainability_pass_rate": (
                num_trainability_pass / float(num_trainability_checks)
                if num_trainability_checks
                else 0.0
            ),
            "min_words/effective": float(effective_min_words),
            "groups/seen": float(groups_seen),
            "groups/used_for_training": float(groups_used_for_training),
            "groups/skipped_no_rewards": float(groups_skipped_no_rewards),
            "groups/skipped_zero_adv": float(groups_skipped_zero_adv),
            "groups/skipped_too_few_valid": float(groups_skipped_too_few_valid),
            "drops/grader_xml_invalid": float(drop_reason_counts["grader_xml_invalid"]),
            "drops/too_short_solution": float(drop_reason_counts["too_short_solution"]),
            "drops/format_noncompliant": float(drop_reason_counts["format_noncompliant"]),
            "drops/format_missing_solution_tags": float(
                format_noncompliance_drop_counts["missing_solution_tags"]
            ),
            "drops/format_missing_solution_open_tag": float(
                format_noncompliance_drop_counts["missing_solution_open_tag"]
            ),
            "drops/format_missing_solution_close_tag": float(
                format_noncompliance_drop_counts["missing_solution_close_tag"]
            ),
            "drops/format_solution_too_long": float(
                format_noncompliance_drop_counts["solution_too_long"]
            ),
            "drops/think_too_long": float(drop_reason_counts["think_too_long"]),
            "drops/non_solution_too_long": float(drop_reason_counts["non_solution_too_long"]),
            "drops/template_copy_solution": float(drop_reason_counts["template_copy_solution"]),
            "samples/think_too_long_kept": float(num_think_too_long_kept),
            "samples/non_solution_too_long_kept": float(num_non_solution_too_long_kept),
            "samples/template_copy_scored": float(num_template_copy_scored),
            "think/penalty_mean": float(np.mean(batch_think_penalties)) if batch_think_penalties else 0.0,
            "non_solution/penalty_mean": (
                float(np.mean(batch_non_solution_penalties))
                if batch_non_solution_penalties
                else 0.0
            ),
            "think/too_long_max_drop_rate": float(config.think_too_long_max_drop_rate),
            "non_solution/too_long_max_drop_rate": float(config.non_solution_too_long_max_drop_rate),
            "think/drop_mode_disabled": 1.0 if config.think_too_long_drop_mode == "disabled" else 0.0,
            "think/drop_mode_adaptive": 1.0 if config.think_too_long_drop_mode == "adaptive" else 0.0,
            "think/drop_mode_hard": 1.0 if config.think_too_long_drop_mode == "hard" else 0.0,
            "non_solution/drop_mode_disabled": (
                1.0 if config.non_solution_too_long_drop_mode == "disabled" else 0.0
            ),
            "non_solution/drop_mode_adaptive": (
                1.0 if config.non_solution_too_long_drop_mode == "adaptive" else 0.0
            ),
            "non_solution/drop_mode_hard": (
                1.0 if config.non_solution_too_long_drop_mode == "hard" else 0.0
            ),
            "sdpo/solution_token_fraction_mean": (
                float(np.mean(batch_sdpo_solution_token_fractions))
                if batch_sdpo_solution_token_fractions
                else 0.0
            ),
            "sdpo/teacher_requests": float(batch_sdpo_teacher_requests),
            "sdpo/teacher_overlap": float(batch_sdpo_teacher_overlap),
            "sdpo/solution_mask_expected": float(batch_sdpo_solution_mask_expected),
            "sdpo/solution_mask_missing": float(batch_sdpo_solution_mask_missing),
            "sdpo/solution_mask_zero": float(batch_sdpo_solution_mask_zero),
            "sdpo/solution_mask_bad_shape": float(batch_sdpo_solution_mask_bad_shape),
            "sdpo/solution_mask_missing_rate": (
                batch_sdpo_solution_mask_missing / float(batch_sdpo_solution_mask_expected)
                if batch_sdpo_solution_mask_expected
                else 0.0
            ),
            "sdpo/solution_mask_zero_rate": (
                batch_sdpo_solution_mask_zero / float(batch_sdpo_solution_mask_expected)
                if batch_sdpo_solution_mask_expected
                else 0.0
            ),
            "sdpo/feedback_chars_mean": (
                float(np.mean(batch_sdpo_feedback_chars))
                if batch_sdpo_feedback_chars
                else 0.0
            ),
            "sdpo/feedback_bullets_mean": (
                float(np.mean(batch_sdpo_feedback_bullets))
                if batch_sdpo_feedback_bullets
                else 0.0
            ),
            "sdpo/feedback_low_level_items_mean": (
                float(np.mean(batch_sdpo_feedback_low_level_items))
                if batch_sdpo_feedback_low_level_items
                else 0.0
            ),
            "sdpo/teacher_prompt_chars_mean": (
                float(np.mean(batch_sdpo_teacher_prompt_chars))
                if batch_sdpo_teacher_prompt_chars
                else 0.0
            ),
            "sdpo/teacher_prompt_tokens_mean": (
                float(np.mean(batch_sdpo_teacher_prompt_tokens))
                if batch_sdpo_teacher_prompt_tokens
                else 0.0
            ),
        }
        if config.use_sdpo:
            metrics["sdpo/success_cutoff"] = batch_sdpo_success_cutoff
            metrics["sdpo/success_rate_groups"] = batch_sdpo_success_rate_groups
            metrics["sdpo/success_min_required"] = float(batch_sdpo_success_min_required)
            metrics["sdpo/success_candidates"] = float(batch_sdpo_success_candidates)
            metrics["sdpo/success_backoff_applied"] = float(
                batch_sdpo_success_backoff_applied
            )

        if config.metrics_mode == "compact":
            compact_keys = [
                "progress/batch",
                "progress/done_frac",
                "run/max_batches_effective",
                "run/effective_loop_batches",
                "time/total",
                "time/sampler_refresh",
                "time/policy_launch",
                "time/policy_wait_and_grader_launch",
                "time/grader_wait",
                "time/scoring_and_filtering",
                "time/format_retry",
                "time/sdpo_success_selection",
                "time/grpo_datum_build",
                "time/sdpo_datum_build_total",
                "time/sdpo_teacher_logprobs",
                "time/sdpo_ref_logprobs",
                "time/sdpo_datum_build_nonlogprob",
                "time/pre_optimization_total",
                "time/forward_backward",
                "time/optimizer_step",
                "time/write_training_logs",
                "time/write_batch_summary",
                "runtime/sampler_refresh_seconds",
                "runtime/sampler_refreshed",
                "runtime/policy_tasks",
                "runtime/grader_tasks",
                "optim/lr",
                "grader/max_tokens_effective",
                "reward/total",
                "rubric/total_valid",
                "format_rate",
                "length/solution_mean",
                "length/total_mean",
                "length/think_mean",
                "samples/generated",
                "samples/scored",
                "samples/valid",
                "samples/trainability_pass_rate",
                "samples/format_retry_attempts",
                "samples/format_retry_recovered",
                "samples/format_retry_recovery_rate",
                "min_words/effective",
                "dropped_samples",
                "drops/grader_xml_invalid",
                "drops/format_noncompliant",
                "drops/format_missing_solution_tags",
                "drops/format_missing_solution_close_tag",
                "drops/template_copy_solution",
                "drops/too_short_solution",
                "drops/think_too_long",
                "drops/non_solution_too_long",
                "groups/used_for_training",
                "groups/skipped_no_rewards",
                "groups/skipped_zero_adv",
                "sdpo/success_cutoff",
                "sdpo/success_rate_groups",
                "sdpo/feedback_chars_mean",
                "sdpo/feedback_bullets_mean",
                "sdpo/teacher_prompt_tokens_mean",
            ]
            metrics = {k: metrics[k] for k in compact_keys if k in metrics}

        # Load sample outputs into metrics for easy viewing
        # if batch_logs_to_save:
        #     first_sample = batch_logs_to_save[0]
        #     metrics["sample/policy_output"] = first_sample["policy_output"]
        #     metrics["sample/grader_output"] = first_sample["grader_output"]
        #     metrics["sample/rubric_score"] = first_sample["rubric_score"]
        #     metrics["sample/format_penalty"] = first_sample["format_penalty"]
        #     metrics["sample/final_reward"] = first_sample["final_reward"]

        ml_logger.log_metrics(metrics, step=real_batch)

    # Save final checkpoint, one per epoch
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
