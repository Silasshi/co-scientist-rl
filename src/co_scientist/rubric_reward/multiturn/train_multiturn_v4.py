"""
Multi-turn conversational trainer v4 (V4): wave-based, ternary PRM, robust parsing.

Architecture:
  - Researcher model (Gemini Flash on OpenRouter) drives conversation naturally
  - Evaluator (Gemini Flash on OpenRouter, separate call) scores each turn {-1, 0, +1}
  - Collaborator (Qwen3-30B-A3B Thinker + LoRA on Tinker) generates responses
  - Canonical grader (30B base on Tinker) scores final plan
  - OPD teacher (same LoRA policy) provides token-level distillation signal

Pipeline per batch (wave-based):
  For each wave (turn position):
    1. All active conversations: collaborator generates one turn
    2. Researcher responds naturally
    3. Evaluator scores turn {-1, 0, +1} + improvement hint
    4. OPD teacher logprobs on turns with valid hints
    5. Advantages: ternary score + OPD (no whitening)
    6. PPO training step → update weights for next wave
  Post-waves:
    7. Grade final plans (canonical rubric grader)
    8. GRPO advantages across K conversations per goal
    9. PPO training step on plan datums

Usage:
  source tools/use_api_profile.sh NEW
  source .env
  python train_multiturn_v4.py batch_size=8 group_size=4
"""

import asyncio
import json
import logging
import os
import re
import sys
import textwrap
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
import torch
import tinker
from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.openrouter_client import OpenRouterClient
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from datasets import load_dataset

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


# ============================================================
# Config
# ============================================================

@chz.chz
class Config:
    base_url: str | None = None
    api_profile: str | None = "NEW"
    log_path: str = "/home/silas/co-scientist-project/runs/2026/3/multiturn_v4/4"

    model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"
    student_model: str = "google/gemini-2.0-flash-001"
    prm_model: str = "google/gemini-2.0-flash-001"
    lora_rank: int = 64

    ml_data: bool = True
    arxiv_data: bool = False
    pubmed_data: bool = False

    max_plan_revisions: int = 2
    max_policy_turns: int = 15
    min_plan_turns: int = 1
    stuck_discussion_threshold: int = 8
    group_size: int = 4
    batch_size: int = 8

    max_tokens: int = 2048
    max_length: int = 32768
    grader_max_tokens: int = 8192
    prm_max_tokens: int = 2048
    temperature: float = 1.0
    grader_temperature: float = 0.0
    student_temperature: float = 0.7
    prm_temperature: float = 0.3
    student_max_tokens: int = 2048

    learning_rate: float = 1e-5
    clip_eps: float = 0.2
    save_every: int = 5
    max_concurrent_tinker: int = 4
    enable_per_turn_training: bool = True

    w_rl: float = 1.0
    w_opd: float = 1.0
    opd_advantage_clip: float = 5.0
    min_hint_length: int = 10

    max_word_count: int = 750
    min_words: int = 30
    format_penalty_base: float = 0.3
    length_penalty_base: float = 0.2
    length_penalty_per_word: float = 0.0005


# ============================================================
# Prompt Constants
# ============================================================

COLLABORATOR_SYSTEM_PROMPT = textwrap.dedent("""
    You are a research collaborator helping a researcher develop a research plan.
    Provide specific, substantive guidance. Build on the researcher's ideas.
    Avoid generic advice — every suggestion should include concrete methods,
    parameters, or evaluation approaches relevant to the specific problem.

    When the researcher explicitly asks you to write a research plan, wrap it
    in <solution></solution> tags. The plan must be self-contained, under
    750 words, and explain HOW and WHY for each step. Do not include
    placeholder text or claim to have results.

    Do NOT generate a <solution> block unless the researcher has explicitly
    asked you to write a research plan. For discussion turns, respond
    naturally without any XML tags.
""").strip()

RESEARCHER_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""
    You are a {domain} researcher discussing a research problem with
    a knowledgeable collaborator. Your goal is to develop a research plan.

    ## Research Goal
    {research_goal}

    ## Instructions
    - Discuss ideas, methodology, and challenges with the collaborator before
      requesting a plan. Take as many turns as you need.
    - Be critical: if the collaborator gives vague advice, push for specifics.
      If you see methodological gaps, point them out. Do not accept generic
      responses without challenge.
    - When ready, ask the collaborator to write a complete research plan.
    - Review the plan critically. If it has specific weaknesses, request
      revisions with concrete feedback (at most {max_plan_revisions} revisions).
    - When satisfied with the plan, end your message with [PLAN_FINAL].
    - Stay in character as a researcher. Do not mention scoring, rubrics,
      or evaluation criteria.
""").strip()

RESEARCHER_FIRST_MESSAGE_TEMPLATE = textwrap.dedent("""
    I'm working on a research problem and would like to discuss it
    before developing a full plan.

    {research_goal}

    What are the key challenges here, and what approaches would you
    suggest exploring?
""").strip()

FORCED_PLAN_REQUEST = textwrap.dedent("""
    I think we've had a productive discussion and covered a lot of ground.
    Now I'd like you to bring everything together into a formal research
    plan. Please write a complete, self-contained plan inside
    <solution></solution> tags that:
    - Synthesizes the key ideas and approaches we discussed
    - Specifies concrete methodology with clear steps
    - Explains HOW each step would be implemented and WHY
    - Includes evaluation approach with specific metrics
    - Stays under 750 words
""").strip()

# --- Evaluator Prompts (ternary {-1, 0, +1} with few-shot + anchored questions) ---

PRM_DISCUSSION_PROMPT = textwrap.dedent("""
    You are an evaluator assessing a research collaborator's discussion response.
    You will see the full conversation between a researcher and collaborator.
    Evaluate the collaborator's MOST RECENT response only.

    ### Before Scoring, Answer These Questions
    1. Did the collaborator provide ANY specific method, algorithm, or
       approach? If no -> score <= 0.
    2. Did the collaborator explain WHY their suggestion fits THIS specific
       problem? If no -> score <= 0.
    3. Did the collaborator give actionable details (parameters, steps,
       metrics)? If no -> score <= 0.
    4. Could this advice apply to ANY research problem without modification?
       If yes -> score <= 0.
    5. Does the response contain filler phrases like "that's a great question"
       or "there are many approaches"? If yes -> score -1.

    ### Scoring Scale
    +1 HELPFUL: Provides specific, useful guidance with concrete methods,
       reasoning, or actionable details that advance the research.
     0 NEUTRAL: Neither helpful nor harmful. Restates known information,
       asks questions without insight, or gives generic advice.
    -1 UNHELPFUL: Vague, off-topic, technically inaccurate, or fails to
       address the researcher's question.

    ### Examples

    Example 1 — Score: +1
    Collaborator: "For handling class imbalance, I recommend focal loss with
    gamma=2.0, combined with stratified sampling. Focal loss downweights easy
    examples, which is ideal for your 1:50 imbalance ratio."
    -> Specific method, parameter, justified for the problem.

    Example 2 — Score: -1
    Collaborator: "That's a great question! There are many approaches to
    handle this. You might want to look into various techniques."
    -> No specific advice. Generic filler.

    Example 3 — Score: 0
    Collaborator: "So you want to improve the model's performance on rare
    classes. That's indeed an important challenge in machine learning."
    -> Restates the problem. No guidance.

    Example 4 — Score: -1
    Collaborator: "You should use PCA to reduce dimensionality before
    applying your transformer model."
    -> PCA destroys sequential structure. Technically misleading.

    ### Output Format
    You MUST output BOTH a score AND a hint. Use EXACTLY this XML format:

    <score>INTEGER: -1, 0, or +1</score>
    <hint>YOUR HINT HERE</hint>

    Hint rules:
    - If score is +1: summarize the key useful insight from the response
      in 1-2 sentences (e.g., "The collaborator correctly identified focal
      loss as suitable for the imbalance ratio and provided specific parameters.")
    - If score is 0 or -1: write 1-3 sentences describing what the collaborator
      SHOULD have said instead. Be concrete and actionable.
    - NEVER write "none". Always provide a substantive hint.
""").strip()

PRM_PLAN_PROMPT = textwrap.dedent("""
    You are an evaluator assessing a research plan written by a collaborator.
    You will see the full conversation between a researcher and collaborator.
    Evaluate the collaborator's MOST RECENT plan response only.

    ### Before Scoring, Answer These Questions
    1. Does the plan specify concrete methods with enough detail to
       implement? If no -> score <= 0.
    2. Does the plan explain WHY each method was chosen for THIS problem?
       If no -> score <= 0.
    3. Does the plan include specific evaluation metrics and baselines?
       If no -> score <= 0.
    4. Could this plan apply to any research problem without modification?
       If yes -> score <= 0.
    5. Does the plan contain vague phrases like "use standard techniques"
       or "apply appropriate methods"? If yes -> score -1.

    ### Scoring Scale
    +1 GOOD PLAN: Comprehensive, specific, well-justified. Methods are
       concrete with clear rationale. Evaluation criteria are measurable.
     0 ADEQUATE PLAN: Has the right structure but too generic. Some methods
       vaguely described. Missing justifications for key choices.
    -1 WEAK PLAN: Significant gaps, vague methodology, technically unsound,
       or doesn't address the research goal adequately.

    ### Examples

    Example 1 — Score: +1
    Plan specifies: "Use RoBERTa-base fine-tuned with lr=2e-5, batch=32.
    Evaluate with F1, BLEU, BERTScore on 200-example stratified test set.
    Compare against GPT-3.5 zero-shot baseline."
    -> Specific model, parameters, metrics, baselines.

    Example 2 — Score: 0
    Plan says: "Apply deep learning techniques to the dataset. Use transfer
    learning for better performance. Evaluate on standard benchmarks."
    -> Right ideas but no specifics.

    Example 3 — Score: -1
    Plan says: "Explore various approaches and see what works best."
    -> No concrete methodology. Could apply to anything.

    ### Output Format
    You MUST output BOTH a score AND a hint. Use EXACTLY this XML format:

    <score>INTEGER: -1, 0, or +1</score>
    <hint>YOUR HINT HERE</hint>

    Hint rules:
    - If score is +1: summarize the plan's key strengths in 1-2 sentences
      (e.g., "The plan correctly specifies RoBERTa with concrete hyperparameters
      and includes three complementary evaluation metrics.")
    - If score is 0 or -1: write 1-3 sentences describing what the plan is
      missing or should improve. Be concrete and actionable.
    - NEVER write "none". Always provide a substantive hint.
""").strip()


# ============================================================
# Parsing Utilities
# ============================================================

def get_text_content(message) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(p["text"] for p in content if p["type"] == "text")


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text).strip()
    text = re.sub(
        r"^(?:Okay|Ok|Alright|Let me|Let's|Hmm|So),?\s+"
        r"(?:the user|let me|let's|I need to|I should|I'll|this is|they)[\s\S]*?\n\n",
        "", text, count=1, flags=re.IGNORECASE,
    ).strip()
    return text


def extract_solution(text: str) -> str:
    match = re.search(r"<solution>\s*(.*?)\s*</solution>", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def close_solution_tag(text: str) -> str:
    if "<solution>" in text.lower() and "</solution>" not in text.lower():
        text = text + "\n</solution>"
    return text


def detect_plan_turn(text: str) -> bool:
    return bool(re.search(r"<solution>", text, re.IGNORECASE))


def detect_domain(ml_data: bool, arxiv_data: bool, pubmed_data: bool) -> str:
    if ml_data:
        return "machine learning"
    elif pubmed_data:
        return "biomedical science"
    else:
        return "scientific research"


def check_format_compliance(text: str, max_words: int) -> bool:
    pattern = r"<solution>\s*(.*?)\s*</solution>"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return False
    return len(match.group(1).split()) <= max_words


def compute_rubric_reward_from_xml(xml_text: str) -> float:
    level_map = {0: 0.0, 1: 0.2, 2: 0.6, 3: 1.0}
    item_blocks = re.findall(r"<item\s+num=.*?>.*?</item>", xml_text, flags=re.DOTALL)
    if not item_blocks:
        return 0.0
    item_scores = []
    for item_xml in item_blocks:
        levels = [int(l) for l in re.findall(r"<level>(\d+)</level>", item_xml) if l.isdigit()]
        if not levels:
            continue
        levels = (levels + [0] * 7)[:7]
        item_scores.append(sum(level_map.get(l, 0.0) for l in levels) / 7)
    return sum(item_scores) / len(item_scores) if item_scores else 0.0


_PRM_PARSE_FALLBACK_COUNT = 0


def parse_prm_response(text: str) -> tuple[int, str | None]:
    """Parse evaluator response with robust fallback cascade."""
    global _PRM_PARSE_FALLBACK_COUNT
    score = 0
    for i, pattern in enumerate([
        r"<score>\s*([+-]?\d+)\s*</score>",
        r"([+-]?\d+)\s*</score>",
        r"<score>\s*([+-]?\d+)",
        r"(?:score|Score|SCORE)[:\s]+([+-]?\d+)",
    ]):
        m = re.search(pattern, text)
        if m:
            try:
                score = max(-1, min(1, int(m.group(1))))
            except ValueError:
                continue
            if i > 0:
                _PRM_PARSE_FALLBACK_COUNT += 1
            break

    hint = None
    for hp in [r"<hint>(.*?)</hint>", r"(.*?)</hint>", r"<hint>(.*?)$"]:
        m = re.search(hp, text, re.DOTALL)
        if m:
            h = m.group(1).strip()
            # Accept any hint >= 10 chars (evaluator is instructed to never write "none")
            if len(h) >= 10:
                hint = h
            break

    return score, hint


def build_prm_prompt(conversation: list[dict], is_plan_turn: bool) -> str:
    base = PRM_PLAN_PROMPT if is_plan_turn else PRM_DISCUSSION_PROMPT
    conv_text = "\n\n".join(f"**{msg['role'].upper()}**: {msg['content']}" for msg in conversation)
    return f"{base}\n\n## Full Conversation\n{conv_text}"


def build_grader_prompt(scenario, rubric_items, proposed_plan, reference_solution=None):
    rubric_block = "\n".join(f"Item {i+1}: {item}" for i, item in enumerate(rubric_items))
    desiderata_text = textwrap.dedent("""
    **DESIDERATA**
    1. HANDLES ALL CRITERIA: Does the plan satisfy all criteria mentioned in the rubric item? An exception is if the criteria says "such as", "for example", or "including", the response does not have to include the same examples listed to meet the criteria, but whatever is provided must be valid and reasonable.
    2. DETAILED, SPECIFIC SOLUTION: Does the part of the plan relevant to satisfying this rubric item include fully specified details on HOW to implement it? There should be no self-proclaimed claims of handling something without doing so. There should be no vague terms, ambiguity, or lack of clarity. It should be described in simple to understand language.
    3. NO OVERLOOKED FLAWS OR WEAKNESSES: Are there any important overlooked flaws or weaknesses in the part of the plan addressing this rubric item that invalidate its satisfaction of the rubric item?
    4. WELL JUSTIFIED RATIONALE: Is the part of the plan relevant to this grading item well-motivated and justified?
    5. COST AND EFFORT EFFICIENT: Does the plan handle this item efficiently without unnecessary complexity?
    6. NO ETHICAL ISSUES: Does this part of the plan have any potential for negative consequences, or is it ethically problematic?
    7. CONSISTENT WITH OVERALL PLAN: Is this part of the plan consistent with the rest of the plan?
    """).strip()
    prompt = f"Evaluate if the Proposed Research Plan satisfies the Research Scenario.\n\n# Research Scenario\n{scenario}\n\n# Rubric\n{rubric_block}"
    if reference_solution is not None:
        prompt += f"\n\n# Reference Solution\n{reference_solution}\n\n\u2022 The reference demonstrates one possible approach. The plan may differ if choices are valid and justified."
    prompt += f"\n\n# Proposed Research Plan\n{proposed_plan}\n\n# Instructions\nFirst, identify weaknesses. Then return XML for each rubric item:\n\n<rubric>\n    <item num=1>\n        <criteria>Rubric item text</criteria>\n        <reasoning>\n        Analyze against these 7 desiderata:\n        {desiderata_text}\n\n        For EACH desideratum, assign: Level 0 (not satisfied), 1 (weakly), 2 (partially), 3 (fully satisfied).\n        Be skeptical and strict.\n        </reasoning>\n        <desiderata num=1>\n            <level>[0-3]</level>\n        </desiderata>\n    </item>\n    ... for all rubric items ...\n</rubric>"
    return prompt


# ============================================================
# Async Helpers
# ============================================================

async def sample_policy_async(sampling_client, model_input, sampling_params, semaphore, timeout=300.0):
    async with semaphore:
        future = sampling_client.sample(prompt=model_input, num_samples=1, sampling_params=sampling_params)
        return await asyncio.wait_for(asyncio.to_thread(future.result), timeout=timeout)


async def grade_plan_async(grader_client, model_input, sampling_params, semaphore, timeout=600.0):
    async with semaphore:
        future = grader_client.sample(prompt=model_input, num_samples=1, sampling_params=sampling_params)
        return await asyncio.wait_for(asyncio.to_thread(future.result), timeout=timeout)


async def compute_teacher_lps_async(sampling_client, full_sequence, semaphore, timeout=300.0):
    async with semaphore:
        future = sampling_client.compute_logprobs(full_sequence)
        return await asyncio.wait_for(asyncio.to_thread(future.result), timeout=timeout)


# ============================================================
# Conversation State
# ============================================================

@dataclass
class ConversationState:
    goal_idx: int
    sample_idx: int
    goal_text: str
    domain: str
    conversation: list[dict] = field(default_factory=list)
    turn_data: list[dict] = field(default_factory=list)
    pending_student_msg: str = ""
    policy_turn_count: int = 0
    discussion_turn_count: int = 0
    plan_revision_count: int = 0
    plan_turn_count: int = 0
    active: bool = True
    final_plan_text: str = ""
    final_plan_turn_idx: int = -1
    completion_reason: str = ""
    forced_plan_used: bool = False
    stuck_injection_used: bool = False
    _layer4_used: bool = False


def _mark_final_plan(turn_data):
    for i in range(len(turn_data) - 1, -1, -1):
        if turn_data[i]["is_plan_turn"]:
            turn_data[i]["is_final_plan"] = True
            return i
    return -1


# ============================================================
# Advance One Turn
# ============================================================

async def advance_one_turn(state, sampling_client, renderer, openrouter_client, config, tinker_semaphore):
    if not state.active:
        return
    try:
        state.conversation.append({"role": "user", "content": state.pending_student_msg})
        messages = [{"role": "system", "content": COLLABORATOR_SYSTEM_PROMPT}] + state.conversation
        model_input = renderer.build_generation_prompt(messages, role="assistant")
        prompt_tokens = [int(t) for t in model_input.to_ints()]

        if len(prompt_tokens) > config.max_length - config.max_tokens:
            state.active = False
            state.completion_reason = "context_length"
            _mark_final_plan(state.turn_data)
            return

        sp = tinker.types.SamplingParams(max_tokens=config.max_tokens, stop=renderer.get_stop_sequences(), temperature=config.temperature)
        response = await sample_policy_async(sampling_client, model_input, sp, tinker_semaphore)
        seq = response.sequences[0]
        raw_text = get_text_content(renderer.parse_response(seq.tokens)[0])
        visible_text = close_solution_tag(strip_thinking(raw_text))
        is_plan = detect_plan_turn(visible_text)

        state.turn_data.append({
            "turn_idx": state.policy_turn_count,
            "prompt_tokens": prompt_tokens,
            "response_tokens": [int(t) for t in seq.tokens],
            "response_logprobs": [float(lp) for lp in seq.logprobs],
            "raw_text": raw_text, "visible_text": visible_text,
            "is_plan_turn": is_plan, "is_final_plan": False,
            "prm_score": None, "prm_hint": None,
        })
        state.conversation.append({"role": "assistant", "content": visible_text})
        state.policy_turn_count += 1

        if is_plan:
            state.final_plan_text = visible_text
            state.plan_revision_count += 1
            state.plan_turn_count += 1
        else:
            state.discussion_turn_count += 1

        if state.policy_turn_count >= config.max_policy_turns:
            if state.plan_turn_count == 0 and not state._layer4_used:
                state._layer4_used = True
                state.forced_plan_used = True
                state.pending_student_msg = FORCED_PLAN_REQUEST
                return
            state.active = False
            state.completion_reason = "max_turns"
            _mark_final_plan(state.turn_data)
            return

        if is_plan and state.plan_revision_count >= config.max_plan_revisions:
            state.active = False
            state.completion_reason = "max_revisions"
            _mark_final_plan(state.turn_data)
            return

        student_system = RESEARCHER_SYSTEM_PROMPT_TEMPLATE.format(
            domain=state.domain, research_goal=state.goal_text, max_plan_revisions=config.max_plan_revisions)
        student_messages = [{"role": "system", "content": student_system}] + state.conversation

        try:
            student_response = await openrouter_client.chat(
                model=config.student_model, messages=student_messages,
                temperature=config.student_temperature, max_tokens=config.student_max_tokens)
        except Exception as e:
            logger.warning("Researcher failed g=%d s=%d: %s", state.goal_idx, state.sample_idx, e)
            state.active = False
            state.completion_reason = "student_error"
            _mark_final_plan(state.turn_data)
            return

        if not student_response or not student_response.strip():
            try:
                student_response = await openrouter_client.chat(
                    model=config.student_model,
                    messages=student_messages + [{"role": "user", "content": "Please continue the discussion."}],
                    temperature=config.student_temperature, max_tokens=config.student_max_tokens)
            except Exception:
                pass

        if not student_response or not student_response.strip():
            if state.plan_turn_count > 0:
                state.conversation.append({"role": "user", "content": "[PLAN_FINAL]"})
                state.completion_reason = "student_accept"
            else:
                state.completion_reason = "student_empty"
            state.active = False
            _mark_final_plan(state.turn_data)
            return

        if "[PLAN_FINAL]" in student_response:
            if state.plan_turn_count < config.min_plan_turns:
                state.pending_student_msg = FORCED_PLAN_REQUEST
                state.forced_plan_used = True
                return
            state.conversation.append({"role": "user", "content": student_response})
            state.active = False
            state.completion_reason = "plan_final"
            _mark_final_plan(state.turn_data)
            return

        if state.discussion_turn_count >= config.stuck_discussion_threshold and state.plan_turn_count == 0:
            state.pending_student_msg = FORCED_PLAN_REQUEST
            state.forced_plan_used = True
            state.stuck_injection_used = True
            return

        state.pending_student_msg = student_response

    except Exception as e:
        logger.error("advance_one_turn failed g=%d s=%d: %s", state.goal_idx, state.sample_idx, e, exc_info=True)
        state.active = False
        state.completion_reason = "error"
        _mark_final_plan(state.turn_data)


# ============================================================
# Thinking Token Masking + Datum Construction
# ============================================================

def find_thinking_ranges(response_tokens, tokenizer):
    think_open = tokenizer.encode("<think>", add_special_tokens=False)
    think_close = tokenizer.encode("</think>", add_special_tokens=False)
    ranges, i = [], 0
    while i <= len(response_tokens) - len(think_open):
        if response_tokens[i:i + len(think_open)] == think_open:
            start, j = i, i + len(think_open)
            while j <= len(response_tokens) - len(think_close):
                if response_tokens[j:j + len(think_close)] == think_close:
                    ranges.append((start, j + len(think_close)))
                    i = j + len(think_close)
                    break
                j += 1
            else:
                ranges.append((start, len(response_tokens)))
                break
        else:
            i += 1
    return ranges


def mask_thinking_advantages(advantages, response_tokens, tokenizer):
    ranges = find_thinking_ranges(response_tokens, tokenizer)
    if not ranges:
        return advantages
    masked = list(advantages)
    for s, e in ranges:
        for k in range(s, min(e, len(masked))):
            masked[k] = 0.0
    return masked


def create_training_datum(prompt_tokens, response_tokens, response_logprobs, per_token_advantages, tokenizer):
    if not response_tokens or not response_logprobs:
        return None
    masked = mask_thinking_advantages(per_token_advantages, response_tokens, tokenizer)
    full_seq = prompt_tokens + response_tokens
    ob_len = len(prompt_tokens) - 1
    inp, tgt = full_seq[:-1], full_seq[1:]
    lps = [0.0] * ob_len + list(response_logprobs)
    advs = [0.0] * ob_len + masked
    n = min(len(inp), len(tgt), len(lps), len(advs))
    if n == 0:
        return None
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=inp[:n]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(tgt[:n], dtype=torch.long)),
            "logprobs": TensorData.from_torch(torch.tensor(lps[:n], dtype=torch.float)),
            "advantages": TensorData.from_torch(torch.tensor(advs[:n], dtype=torch.float)),
        },
    )


# ============================================================
# Main Training Loop
# ============================================================

async def main(config: Config):
    global _PRM_PARSE_FALLBACK_COUNT
    ml_logger = ml_log.setup_logging(config.log_path)
    logger.info("Config: %s", config)

    if config.ml_data:
        data = load_dataset("facebook/research-plan-gen", "ml")
    elif config.arxiv_data:
        data = load_dataset("facebook/research-plan-gen", "arxiv")
    elif config.pubmed_data:
        data = load_dataset("facebook/research-plan-gen", "pubmed")
    else:
        raise ValueError("No dataset selected")

    dataset = data["train"]
    domain = detect_domain(config.ml_data, config.arxiv_data, config.pubmed_data)
    n_train_batches = len(dataset) // config.batch_size
    logger.info("Dataset: %d samples, %d batches of %d", len(dataset), n_train_batches, config.batch_size)

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    service_client = create_service_client(base_url=config.base_url, api_profile=config.api_profile)

    os.makedirs(config.log_path, exist_ok=True)
    batch_summary_path = os.path.join(config.log_path, "train/batch_summary.jsonl")
    training_logs_path = os.path.join(config.log_path, "train/training_logs.jsonl")
    conv_logs_path = os.path.join(config.log_path, "train/conversations.jsonl")
    os.makedirs(os.path.dirname(batch_summary_path), exist_ok=True)

    last_checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path)
    if last_checkpoint:
        logger.info("Resuming from checkpoint batch=%d", last_checkpoint["batch"])
        training_client = service_client.create_training_client_from_state_with_optimizer(last_checkpoint["state_path"])
        actual_batch = last_checkpoint["batch"] + 1
        start_batch = actual_batch % n_train_batches
    else:
        training_client = service_client.create_lora_training_client(base_model=config.model_name, rank=config.lora_rank)
        actual_batch = 0
        start_batch = 0

    sampling_result = training_client.save_weights_for_sampler(name=f"init_{actual_batch:06d}").result()
    sampling_client = service_client.create_sampling_client(model_path=sampling_result.path)
    grader_client = service_client.create_sampling_client(base_model=config.grader_model_name)
    openrouter_client = OpenRouterClient()
    adam_params = types.AdamParams(learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    tinker_semaphore = asyncio.Semaphore(config.max_concurrent_tinker)
    grader_params = tinker.types.SamplingParams(max_tokens=config.grader_max_tokens, temperature=config.grader_temperature)

    logger.info("Setup complete. Starting training from batch %d", start_batch)

    for batch_idx in range(start_batch, n_train_batches):
        real_batch = actual_batch + (batch_idx - start_batch)
        batch_start = (batch_idx * config.batch_size) % len(dataset)
        batch_end = min(batch_start + config.batch_size, len(dataset))
        batch_rows = dataset.select(range(batch_start, batch_end))

        logger.info("=== Batch %d (real=%d) === goals %d-%d", batch_idx, real_batch, batch_start, batch_end)
        batch_t0 = time.time()
        openrouter_client.reset_token_counter()
        _PRM_PARSE_FALLBACK_COUNT = 0

        # Initialize states
        states = [ConversationState(
            goal_idx=gi, sample_idx=si, goal_text=batch_rows["Goal"][gi], domain=domain,
            pending_student_msg=RESEARCHER_FIRST_MESSAGE_TEMPLATE.format(research_goal=batch_rows["Goal"][gi]),
        ) for gi in range(len(batch_rows["Goal"])) for si in range(config.group_size)]

        all_wave_datums, all_prm_scores, all_hints = [], [], []
        wave_active_counts, opd_kl_all, opd_adv_all = [], [], []

        # ====== Wave Loop ======
        wave_t0 = time.time()
        for wave_idx in range(config.max_policy_turns + 2):
            active = [s for s in states if s.active]
            if not active:
                break
            wave_active_counts.append(len(active))

            # A: Advance turns
            await asyncio.gather(*[advance_one_turn(s, sampling_client, renderer, openrouter_client, config, tinker_semaphore) for s in active], return_exceptions=True)

            # B: PRM score
            prm_tasks = {}
            for s in active:
                if not s.turn_data:
                    continue
                turn = s.turn_data[-1]
                if turn["turn_idx"] != s.policy_turn_count - 1 or turn.get("is_final_plan"):
                    continue
                prm_prompt = build_prm_prompt(s.conversation[:], turn["is_plan_turn"])
                prm_tasks[(s.goal_idx, s.sample_idx, turn["turn_idx"])] = asyncio.create_task(
                    openrouter_client.chat(model=config.prm_model, messages=[{"role": "user", "content": prm_prompt}],
                                           temperature=config.prm_temperature, max_tokens=config.prm_max_tokens))

            for key, task in prm_tasks.items():
                try:
                    resp = await task
                    score, hint = parse_prm_response(resp)
                except Exception:
                    score, hint = 0, None
                g, si, t = key
                conv = next(st for st in states if st.goal_idx == g and st.sample_idx == si)
                conv.turn_data[t]["prm_score"] = score
                conv.turn_data[t]["prm_hint"] = hint
                all_prm_scores.append(score)
                all_hints.append((score, hint is not None))

            # C: OPD
            opd_tasks = {}
            for key in prm_tasks:
                g, si, t = key
                conv = next(st for st in states if st.goal_idx == g and st.sample_idx == si)
                turn = conv.turn_data[t]
                hint = turn["prm_hint"]
                if not hint or len(hint) < config.min_hint_length:
                    continue
                smi = t * 2
                if smi >= len(conv.conversation):
                    continue
                ec = []
                for i, msg in enumerate(conv.conversation[:smi + 1]):
                    if i == smi:
                        ec.append({"role": msg["role"], "content": msg["content"] + f"\n\n[user's hint / instruction]\n{hint.strip()}"})
                    else:
                        ec.append(msg)
                em = [{"role": "system", "content": COLLABORATOR_SYSTEM_PROMPT}] + ec
                ei = renderer.build_generation_prompt(em, role="assistant")
                ept = [int(tk) for tk in ei.to_ints()]
                fs = types.ModelInput.from_ints(tokens=ept + turn["response_tokens"])
                opd_tasks[key] = (asyncio.create_task(compute_teacher_lps_async(sampling_client, fs, tinker_semaphore)), len(ept), len(turn["response_tokens"]))

            teacher_lps_store = {}
            for key, (task, pl, rl) in opd_tasks.items():
                try:
                    al = await task
                    tlps = list(al[pl:pl + rl])
                    if len(tlps) < rl:
                        tlps.extend([0.0] * (rl - len(tlps)))
                    teacher_lps_store[key] = tlps
                    g, si, t = key
                    conv = next(st for st in states if st.goal_idx == g and st.sample_idx == si)
                    old = conv.turn_data[t]["response_logprobs"]
                    n = min(len(old), len(tlps))
                    if n > 0:
                        opd_kl_all.append(np.mean(np.abs(np.array(old[:n]) - np.array(tlps[:n]))))
                except Exception:
                    pass

            # D: Advantages + datums
            wave_datums = []
            if config.enable_per_turn_training:
                for key in prm_tasks:
                    g, si, t = key
                    conv = next(st for st in states if st.goal_idx == g and st.sample_idx == si)
                    turn = conv.turn_data[t]
                    if turn.get("is_final_plan"):
                        continue
                    ps = turn.get("prm_score", 0) or 0
                    norm = float(ps)
                    rlps = turn["response_logprobs"]
                    tlps = teacher_lps_store.get(key)
                    pt = []
                    for k in range(len(rlps)):
                        ot = 0.0
                        if tlps and k < len(tlps):
                            ro = tlps[k] - rlps[k]
                            ot = config.w_opd * np.clip(ro, -config.opd_advantage_clip, config.opd_advantage_clip)
                            opd_adv_all.append(abs(ot))
                        pt.append(config.w_rl * norm + ot)
                    d = create_training_datum(turn["prompt_tokens"], turn["response_tokens"], rlps, pt, tokenizer)
                    if d:
                        wave_datums.append(d)

            # E: Train + update
            if wave_datums:
                training_client.forward_backward(wave_datums, loss_fn="ppo",
                    loss_fn_config={"clip_low_threshold": 1 - config.clip_eps, "clip_high_threshold": 1 + config.clip_eps}).result()
                training_client.optim_step(adam_params).result()
                sr = training_client.save_weights_for_sampler(name=f"b{real_batch:06d}_w{wave_idx:02d}").result()
                sampling_client = service_client.create_sampling_client(model_path=sr.path)

            all_wave_datums.extend(wave_datums)
            logger.info("Wave %d: %d active, %d datums, %d PRM, %d OPD", wave_idx, len(active), len(wave_datums), len(prm_tasks), len(teacher_lps_store))

        waves_time = time.time() - wave_t0

        # ====== Grade Final Plans ======
        grade_t0 = time.time()
        grading_tasks = {}
        for s in states:
            if not s.final_plan_text:
                continue
            sol = extract_solution(s.final_plan_text)
            if len(sol.split()) < config.min_words:
                continue
            gp = build_grader_prompt(batch_rows["Goal"][s.goal_idx], batch_rows["Rubric"][s.goal_idx], sol, batch_rows["Reference solution"][s.goal_idx])
            gi = renderer.build_generation_prompt([{"role": "user", "content": gp}], role="assistant")
            grading_tasks[(s.goal_idx, s.sample_idx)] = asyncio.create_task(grade_plan_async(grader_client, gi, grader_params, tinker_semaphore))

        grading_results = {}
        for key, task in grading_tasks.items():
            try:
                resp = await task
                xml = get_text_content(renderer.parse_response(resp.sequences[0].tokens)[0])
                rubric = compute_rubric_reward_from_xml(xml)
                s = next(st for st in states if st.goal_idx == key[0] and st.sample_idx == key[1])
                ht = bool(re.search(r"<solution>", s.final_plan_text, re.IGNORECASE))
                wc = len(extract_solution(s.final_plan_text).split())
                ex = max(0, wc - config.max_word_count)
                fp = 0.0 if ht else config.format_penalty_base
                lp = (config.length_penalty_base + config.length_penalty_per_word * ex) if ex > 0 else 0.0
                grading_results[key] = {"rubric_score": rubric, "format_penalty": fp, "length_penalty": lp, "final_reward": rubric - fp - lp, "word_count": wc}
            except Exception as e:
                logger.warning("Grading failed %s: %s", key, e)
        grade_time = time.time() - grade_t0

        # ====== GRPO + Train Plan Datums ======
        plan_datums = []
        for goal_idx in range(len(batch_rows["Goal"])):
            rw, ps = [], []
            for s in states:
                if s.goal_idx == goal_idx and (s.goal_idx, s.sample_idx) in grading_results:
                    rw.append(grading_results[(s.goal_idx, s.sample_idx)]["final_reward"])
                    ps.append(s)
            if len(rw) < 2:
                continue
            mr = np.mean(rw)
            for i, s in enumerate(ps):
                ga = rw[i] - mr
                for turn in s.turn_data:
                    if turn.get("is_final_plan"):
                        d = create_training_datum(turn["prompt_tokens"], turn["response_tokens"], turn["response_logprobs"],
                                                   [config.w_rl * ga] * len(turn["response_logprobs"]), tokenizer)
                        if d:
                            plan_datums.append(d)

        plan_t0 = time.time()
        if plan_datums:
            training_client.forward_backward(plan_datums, loss_fn="ppo",
                loss_fn_config={"clip_low_threshold": 1 - config.clip_eps, "clip_high_threshold": 1 + config.clip_eps}).result()
            training_client.optim_step(adam_params).result()
            sr = training_client.save_weights_for_sampler(name=f"b{real_batch:06d}_final").result()
            sampling_client = service_client.create_sampling_client(model_path=sr.path)
        plan_time = time.time() - plan_t0

        # ====== Logging ======
        bt = time.time() - batch_t0
        rs = [r["rubric_score"] for r in grading_results.values()]
        fr = [r["final_reward"] for r in grading_results.values()]
        wcs = [r["word_count"] for r in grading_results.values()]
        pd = dict(Counter(all_prm_scores))
        pc = max(Counter(all_prm_scores).values()) > 0.8 * len(all_prm_scores) if all_prm_scores else False
        hbs = defaultdict(int)
        for sc, hh in all_hints:
            if hh:
                hbs[sc] += 1
        aa = []
        for d in all_wave_datums:
            at = d.loss_fn_inputs["advantages"].to_torch()
            aa.extend(at[at != 0].tolist())
        oc = np.mean(opd_adv_all) / max(np.mean(np.abs(aa)), 1e-8) if opd_adv_all and aa else 0.0
        comp = sum(1 for s in states if s.completion_reason == "plan_final")
        forced = sum(1 for s in states if s.forced_plan_used)
        stuck = sum(1 for s in states if s.stuck_injection_used)

        bs = {
            "batch_idx": real_batch,
            "rubric/mean": float(np.mean(rs)) if rs else 0.0, "rubric/std": float(np.std(rs)) if rs else 0.0,
            "reward/mean": float(np.mean(fr)) if fr else 0.0, "reward/std": float(np.std(fr)) if fr else 0.0,
            "prm_score/mean": float(np.mean(all_prm_scores)) if all_prm_scores else 0.0,
            "prm_score/std": float(np.std(all_prm_scores)) if all_prm_scores else 0.0,
            "prm_score/dist": pd, "prm_score/collapse_alert": pc,
            "prm/parse_fallback_count": _PRM_PARSE_FALLBACK_COUNT,
            "hint/total": sum(1 for _, h in all_hints if h), "hint/by_score": dict(hbs),
            "opd/sets_computed": len(opd_kl_all),
            "opd/mean_kl": float(np.mean(opd_kl_all)) if opd_kl_all else 0.0,
            "opd/advantage_contribution": oc,
            "conv/total": len(states), "conv/completed_rate": comp / max(len(states), 1),
            "conv/turns_mean": float(np.mean([len(s.turn_data) for s in states])),
            "conv/discussion_turns_mean": float(np.mean([s.discussion_turn_count for s in states])),
            "conv/plan_revisions_mean": float(np.mean([s.plan_revision_count for s in states])),
            "conv/stuck_injections": stuck,
            "advantage/mean": float(np.mean(aa)) if aa else 0.0, "advantage/std": float(np.std(aa)) if aa else 0.0,
            "datums/wave_total": len(all_wave_datums), "datums/plan_total": len(plan_datums),
            "forced_plan/count": forced,
            "format/compliance_rate": sum(1 for r in grading_results.values() if r["format_penalty"] == 0.0) / max(len(grading_results), 1),
            "length/mean": float(np.mean(wcs)) if wcs else 0.0, "length/p90": float(np.percentile(wcs, 90)) if wcs else 0.0,
            "wave/count": len(wave_active_counts), "wave/active_convos": wave_active_counts,
            "time/total": bt, "time/waves": waves_time, "time/grading": grade_time, "time/plan_train": plan_time,
            "api/openrouter_tokens": openrouter_client.total_tokens,
        }
        with open(batch_summary_path, "a") as f:
            f.write(json.dumps(bs) + "\n")

        for s in states:
            gr = grading_results.get((s.goal_idx, s.sample_idx), {})
            with open(training_logs_path, "a") as f:
                f.write(json.dumps({
                    "batch_idx": real_batch, "goal_idx": s.goal_idx, "sample_idx": s.sample_idx,
                    "goal": s.goal_text[:200], "num_turns": len(s.turn_data),
                    "num_discussion": s.discussion_turn_count, "num_revisions": s.plan_revision_count,
                    "completed": s.completion_reason == "plan_final", "completion_reason": s.completion_reason,
                    "rubric_score": gr.get("rubric_score"), "final_reward": gr.get("final_reward"),
                    "format_penalty": gr.get("format_penalty"), "length_penalty": gr.get("length_penalty"),
                    "word_count": gr.get("word_count"),
                    "prm_scores": [t["prm_score"] for t in s.turn_data if t["prm_score"] is not None],
                    "hint_count": sum(1 for t in s.turn_data if t["prm_hint"] is not None),
                    "forced_plan": s.forced_plan_used, "stuck_injection": s.stuck_injection_used,
                    "final_plan": extract_solution(s.final_plan_text)[:500] if s.final_plan_text else "",
                }) + "\n")

            # Full conversation log for debugging and verification
            researcher_system = RESEARCHER_SYSTEM_PROMPT_TEMPLATE.format(
                domain=domain, research_goal=s.goal_text, max_plan_revisions=config.max_plan_revisions)
            with open(conv_logs_path, "a") as f:
                f.write(json.dumps({
                    "batch_idx": real_batch, "goal_idx": s.goal_idx, "sample_idx": s.sample_idx,
                    "collaborator_system_prompt": COLLABORATOR_SYSTEM_PROMPT,
                    "researcher_system_prompt": researcher_system,
                    "conversation": s.conversation,
                    "turn_details": [
                        {"turn": t["turn_idx"], "is_plan": t["is_plan_turn"],
                         "prm_score": t["prm_score"],
                         "prm_hint": t["prm_hint"],
                         "visible_text": t["visible_text"][:500]}
                        for t in s.turn_data
                    ],
                    "completion_reason": s.completion_reason,
                }) + "\n")

        logger.info("Batch %d done in %.1fs | rubric=%.3f reward=%.3f prm=%.2f comp=%.0f%% datums=%d+%d fb=%d",
                     real_batch, bt, bs["rubric/mean"], bs["reward/mean"], bs["prm_score/mean"],
                     bs["conv/completed_rate"] * 100, len(all_wave_datums), len(plan_datums), _PRM_PARSE_FALLBACK_COUNT)

        if config.save_every > 0 and real_batch > 0 and real_batch % config.save_every == 0:
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client, name=f"{real_batch:06d}",
                log_path=config.log_path, kind="state", loop_state={"batch": real_batch})
            logger.info("Checkpoint saved at batch %d", real_batch)

    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client, name=f"{real_batch:06d}_final",
        log_path=config.log_path, kind="both", loop_state={"batch": real_batch})
    logger.info("Training complete. Final checkpoint saved.")
    await openrouter_client.close()


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
