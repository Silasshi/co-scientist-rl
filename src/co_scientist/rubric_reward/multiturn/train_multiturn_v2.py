"""
Multi-turn conversational GRPO trainer v2: PRM + OPD (OpenClaw-RL pipeline).

Adds to v1:
  - PRM: per-turn reward (was this tutor response helpful?)
  - OPD: token-level teacher guidance (hint from grader → teacher logprobs)

Usage:
  source tools/use_api_profile.sh OLD
  python train_multiturn_v2.py config.batch_size=8 config.group_size=2
"""

import json
import logging
import os
import re
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
import torch
import tinker
from co_scientist.shared.api_profiles import create_service_client
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
    # API
    base_url: str | None = None
    api_profile: str | None = None
    log_path: str = "/home/silas/co-scientist-project/runs/2026/3/multiturn_v2/1"
    model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 64

    # Dataset / curriculum
    ml_data: bool = True
    curriculum_path: str = "/home/silas/co-scientist-project/data/curriculum/ml_train_curriculum.jsonl"

    # Multi-turn
    num_tutor_turns: int = 4
    group_size: int = 4

    # Sampling
    max_tokens: int = 2048
    grader_max_tokens: int = 8192
    temperature: float = 1.0
    grader_temperature: float = 0.0

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-5
    clip_eps: float = 0.2
    save_every: int = 10

    # Reward (same as bestversion)
    max_word_count: int = 750
    target_word_count: int = 600
    scale_length_bonus: float = 120.0
    scaling_factor: float = 0.08
    min_words: int = 30

    # PRM (per-turn evaluation)
    prm_enabled: bool = True
    prm_temperature: float = 0.6
    prm_max_tokens: int = 1024

    # OPD (teacher logprob distillation)
    opd_enabled: bool = True
    opd_kl_coef: float = 1.0
    opd_w_rl: float = 0.5
    opd_w_opd: float = 0.5
    opd_hint_min_length: int = 20


# ============================================================
# Helpers (shared with v1)
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


TUTOR_SYSTEM_PROMPT = textwrap.dedent("""
    You are a knowledgeable research tutor helping a graduate student develop a research plan.
    Your role is to:
    - Help explore research directions when the student presents initial ideas
    - Discuss methodology, potential pitfalls, and alternative approaches
    - Provide substantive, specific guidance — avoid generic advice
    - Build on the student's insights rather than ignoring them

    IMPORTANT formatting rules:
    - Do NOT start your response with internal reasoning like "Okay, the user is asking..." or "Let me think..."
    - Respond directly and professionally to the student.
    - When asked for a research plan, wrap it in <think>...</think> then <solution>...</solution> tags. The solution must not exceed 750 words.
    - For discussion turns (not plan generation), just respond naturally without XML tags.
""").strip()


# ============================================================
# Grader (same as v1 / best_ver.py)
# ============================================================

def build_grader_prompt(scenario, rubric_items, proposed_plan, reference_solution=None):
    rubric_block = "\n".join([f"Item {i+1}: {item}" for i, item in enumerate(rubric_items)])
    desiderata_text = textwrap.dedent("""
    **DESIDERATA**
    1. HANDLES ALL CRITERIA: Does the plan satisfy all criteria mentioned in the rubric item?
    2. DETAILED, SPECIFIC SOLUTION: Does the part of the plan include fully specified details on HOW?
    3. NO OVERLOOKED FLAWS OR WEAKNESSES: Are there important overlooked flaws?
    4. WELL JUSTIFIED RATIONALE: Is the approach well-motivated and justified?
    5. COST AND EFFORT EFFICIENT: Is it efficient without unnecessary complexity?
    6. NO ETHICAL ISSUES: Any potential for negative consequences?
    7. CONSISTENT WITH OVERALL PLAN: Consistent with the rest of the plan?
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

            • It is just meant to demonstrate one possible approach.
            • The Research Plan you have to grade might have different design choices. This is okay, if valid.
        """).strip()

    prompt += textwrap.dedent(f"""
        # Proposed Research Plan
        {proposed_plan}

        # Instructions
        First, come up with weaknesses of the proposed plan. Then, return the following nested XML block for each grading item:

        <rubric>
            <item num=1>
                <criteria>Repeat the rubric item string here.</criteria>
                <reasoning>
                Analyze the plan against all 7 GENERAL DESIDERATA:
                {desiderata_text}

                For EACH desideratum, assign: Level 0 (NOT SATISFIED), 1 (WEAKLY), 2 (PARTIALLY), 3 (FULLY SATISFIED).
                Be skeptical and strict. Check WHETHER, HOW, and WHY.
                </reasoning>
                <desiderata num=1><level>[0-3]</level></desiderata>
                ... (7 total)
            </item>
            ... (all rubric items)
        </rubric>
    """).strip()
    return prompt


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
        mapped = [level_map.get(l, 0.0) for l in levels]
        item_scores.append(sum(mapped) / len(mapped))
    return sum(item_scores) / len(item_scores) if item_scores else 0.0


def check_format_compliance(text: str, max_words: int) -> bool:
    match = re.search(r"<solution>\s*(.*?)\s*</solution>", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return False
    return len(match.group(1).split()) <= max_words


def compute_reward(rubric_score, plan_text, config):
    solution = extract_solution(plan_text)
    word_count = len(solution.split())
    is_compliant = check_format_compliance(plan_text, config.max_word_count)
    excess = max(0, word_count - config.max_word_count)
    format_penalty = 0.0 if is_compliant else 0.2 + 0.0005 * excess
    length_bonus = np.exp(-((word_count - config.target_word_count) / config.scale_length_bonus) ** 2)
    return rubric_score + config.scaling_factor * length_bonus - format_penalty


# ============================================================
# PRM: Per-turn evaluation (NEW in v2)
# ============================================================

def build_prm_prompt(tutor_response: str, next_student_message: str) -> str:
    """Build PRM evaluation prompt adapted from OpenClaw for research tutor domain."""
    return textwrap.dedent(f"""
        You are evaluating a research tutor's response in a multi-turn discussion with a student.

        Score the tutor's response based on quality and helpfulness:
        \\boxed{{1}} = Substantive, specific, builds on student's ideas, moves discussion forward with concrete directions
        \\boxed{{-1}} = Vague, off-topic, generic, doesn't address student's question, or provides incorrect guidance
        \\boxed{{0}} = Adequate but unremarkable, neither clearly helpful nor harmful

        ## Tutor's response
        {tutor_response}

        ## Student's follow-up (evidence of quality)
        {next_student_message}

        Think step-by-step about the tutor response quality, then give your final score inside \\boxed{{}}.
    """).strip()


def parse_prm_score(text: str) -> float:
    """Extract \\boxed{N} score from PRM output. Returns 0.0 on failure."""
    matches = re.findall(r"\\boxed\{([+-]?[01])\}", text)
    if matches:
        return float(matches[-1])
    return 0.0


# ============================================================
# OPD: Hint extraction + teacher logprobs (NEW in v2)
# ============================================================

def extract_hint_from_grader(grader_xml: str, min_length: int = 20) -> str | None:
    """Extract concise actionable hint from grader XML reasoning sections."""
    items = re.findall(r"<item\s+num=.*?>.*?</item>", grader_xml, re.DOTALL)
    weaknesses = []
    for item_xml in items:
        levels = [int(l) for l in re.findall(r"<level>(\d+)</level>", item_xml) if l.isdigit()]
        if not levels or min(levels) > 2:
            continue
        reasoning = re.search(r"<reasoning>(.*?)</reasoning>", item_xml, re.DOTALL)
        if reasoning:
            text = reasoning.group(1).strip()
            # Extract sentences mentioning weaknesses
            sentences = re.split(r"[.!?]\s+", text)
            weak_sentences = [s for s in sentences if any(
                w in s.lower() for w in ["lack", "miss", "vague", "not", "weak", "should", "fail", "absent", "insufficient"]
            )]
            if weak_sentences:
                weaknesses.append(weak_sentences[0].strip()[:200])
    if not weaknesses:
        return None
    hint = "Key areas to improve: " + ". ".join(weaknesses[:3]) + "."
    if len(hint) < min_length:
        return None
    return hint


def compute_teacher_logprobs_for_turn(
    base_client, renderer, conversation, hint, turn_idx, turn_data,
):
    """Compute base model logprobs on tutor response with hint-augmented prompt."""
    # Build augmented conversation: hint appended to student message before this turn
    student_msg_idx = turn_idx * 2  # Student messages are at even indices
    augmented_convo = list(conversation[:student_msg_idx + 1])  # Up to and including student msg
    # Append hint to last student message
    augmented_convo[-1] = {
        "role": augmented_convo[-1]["role"],
        "content": augmented_convo[-1]["content"] + f"\n\n[Grader feedback: {hint}]",
    }

    messages = [{"role": "system", "content": TUTOR_SYSTEM_PROMPT}] + augmented_convo
    teacher_input = renderer.build_generation_prompt(messages, role="assistant")
    teacher_prompt_tokens = [int(t) for t in teacher_input.to_ints()]

    response_tokens = turn_data["response_tokens"]
    full_sequence = types.ModelInput.from_ints(tokens=teacher_prompt_tokens + response_tokens)

    try:
        logprobs_result = base_client.compute_logprobs(full_sequence).result()
        # Slice to response portion only
        start = len(teacher_prompt_tokens)
        end = start + len(response_tokens)
        teacher_lps = logprobs_result[start:end]
        return [float(lp) for lp in teacher_lps]
    except Exception as e:
        logger.warning(f"Teacher logprob computation failed: {e}")
        return None


# ============================================================
# Curriculum loader (same as v1)
# ============================================================

def load_curriculum(path: str) -> dict[int, dict]:
    curriculum = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            curriculum[entry["goal_id"]] = entry
    return curriculum


# ============================================================
# Conversation generation (same as v1)
# ============================================================

def generate_conversation(goal_idx, sample_idx, curriculum_entry, sampling_client, renderer, config):
    conversation = []
    turn_data = []
    student_messages = curriculum_entry["student_messages"]
    sampling_params = types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )
    for turn_idx in range(min(config.num_tutor_turns, len(student_messages))):
        student_msg = student_messages[turn_idx]
        conversation.append({"role": "user", "content": student_msg})

        messages = [{"role": "system", "content": TUTOR_SYSTEM_PROMPT}] + conversation
        model_input = renderer.build_generation_prompt(messages, role="assistant")
        prompt_tokens = [int(t) for t in model_input.to_ints()]

        future = sampling_client.sample(prompt=model_input, num_samples=1, sampling_params=sampling_params)
        response = future.result()
        seq = response.sequences[0]
        raw_text = get_text_content(renderer.parse_response(seq.tokens)[0])
        tutor_text = strip_thinking(raw_text)

        if "<solution>" in tutor_text and "</solution>" not in tutor_text:
            tutor_text = tutor_text.rstrip() + "\n</solution>"

        turn_data.append({
            "turn_idx": turn_idx,
            "prompt_tokens": prompt_tokens,
            "response_tokens": [int(t) for t in seq.tokens],
            "response_logprobs": [float(lp) for lp in seq.logprobs],
            "text": tutor_text,
        })
        conversation.append({"role": "assistant", "content": tutor_text})

    final_plan = conversation[-1]["content"] if conversation else ""
    return goal_idx, sample_idx, turn_data, final_plan, conversation


# ============================================================
# Main training loop
# ============================================================

def main(config: Config):
    os.makedirs(os.path.join(config.log_path, "train"), exist_ok=True)
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path, wandb_project=None, wandb_name=None,
        config=config, do_configure_logging_module=True,
    )

    logger.info(f"Loading curriculum from {config.curriculum_path}...")
    curriculum = load_curriculum(config.curriculum_path)
    logger.info(f"Loaded {len(curriculum)} curriculum entries")

    logger.info("Loading dataset (train split)...")
    data = load_dataset("facebook/research-plan-gen", "ml")
    dataset = data["train"]
    n_train_batches = len(dataset) // config.batch_size
    logger.info(f"Train set: {len(dataset)} goals → {n_train_batches} batches of {config.batch_size}")

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    service_client = create_service_client(base_url=config.base_url, api_profile=config.api_profile)
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name, rank=config.lora_rank,
    )
    logger.info(f"Created fresh LoRA training client (rank={config.lora_rank})")

    sampling_result = training_client.save_weights_for_sampler(name="init").result()
    sampling_client = service_client.create_sampling_client(model_path=sampling_result.path)
    grader_client = service_client.create_sampling_client(base_model=config.grader_model_name)

    # Base model client for OPD teacher logprobs (no LoRA)
    base_client = grader_client  # Same base model, reuse

    adam_params = types.AdamParams(learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    training_logs_path = os.path.join(config.log_path, "train/training_logs.jsonl")
    batch_summary_path = os.path.join(config.log_path, "train/batch_summary.jsonl")

    t_global_start = time.time()

    logger.info("=" * 60)
    logger.info(
        f"Starting multi-turn v2 (PRM+OPD) | {n_train_batches} batches | "
        f"group_size={config.group_size} | PRM={config.prm_enabled} | OPD={config.opd_enabled}"
    )
    logger.info("=" * 60)

    for batch_idx in range(n_train_batches):
        t_batch_start = time.time()
        batch_start = batch_idx * config.batch_size
        batch_rows = dataset[batch_start:batch_start + config.batch_size]

        # ============================================================
        # PHASE 1: Generate conversations (same as v1)
        # ============================================================
        logger.info(f"[Batch {batch_idx}] Generating {config.batch_size}×{config.group_size} conversations...")
        t0 = time.time()
        conversation_results = []
        gen_errors = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            for i in range(config.batch_size):
                goal_id = batch_start + i
                if goal_id not in curriculum:
                    continue
                for s in range(config.group_size):
                    f = executor.submit(
                        generate_conversation, goal_idx=i, sample_idx=s,
                        curriculum_entry=curriculum[goal_id],
                        sampling_client=sampling_client, renderer=renderer, config=config,
                    )
                    futures[f] = (i, s)
            for f in as_completed(futures):
                try:
                    conversation_results.append(f.result())
                except Exception as e:
                    gi, si = futures[f]
                    logger.error(f"Conversation failed goal={gi} sample={si}: {e}")
                    gen_errors += 1

        # Retry on mass failure
        expected = sum(1 for i in range(config.batch_size) if batch_start + i in curriculum) * config.group_size
        if gen_errors > expected * 0.5 and expected > 0:
            logger.warning(f"[Batch {batch_idx}] >50% failures — refreshing sampling client...")
            try:
                result = training_client.save_weights_for_sampler(name=f"refresh_{batch_idx}").result()
                sampling_client = service_client.create_sampling_client(model_path=result.path)
                conversation_results = []
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = {}
                    for i in range(config.batch_size):
                        goal_id = batch_start + i
                        if goal_id not in curriculum:
                            continue
                        for s in range(config.group_size):
                            f = executor.submit(
                                generate_conversation, goal_idx=i, sample_idx=s,
                                curriculum_entry=curriculum[goal_id],
                                sampling_client=sampling_client, renderer=renderer, config=config,
                            )
                            futures[f] = (i, s)
                    for f in as_completed(futures):
                        try:
                            conversation_results.append(f.result())
                        except Exception as e:
                            pass
            except Exception:
                logger.exception(f"[Batch {batch_idx}] Refresh failed")
                continue

        t_gen = time.time() - t0

        conversations = {}
        for r in conversation_results:
            goal_idx, sample_idx, turn_data, final_plan, convo = r
            conversations.setdefault(goal_idx, {})[sample_idx] = (turn_data, final_plan, convo)

        # ============================================================
        # PHASE 2: Grade final plans (same as v1)
        # ============================================================
        n_plans = sum(len(s) for s in conversations.values())
        logger.info(f"[Batch {batch_idx}] Grading {n_plans} plans...")
        t0 = time.time()

        grading_futures = []
        for goal_idx, samples in conversations.items():
            for sample_idx, (turn_data, final_plan, convo) in samples.items():
                grader_prompt = build_grader_prompt(
                    scenario=batch_rows["Goal"][goal_idx],
                    rubric_items=batch_rows["Rubric"][goal_idx],
                    proposed_plan=extract_solution(final_plan),
                    reference_solution=batch_rows["Reference solution"][goal_idx],
                )
                grader_input = renderer.build_generation_prompt(
                    [{"role": "user", "content": grader_prompt}], role="assistant",
                )
                grader_params = types.SamplingParams(
                    max_tokens=config.grader_max_tokens,
                    stop=renderer.get_stop_sequences(), temperature=config.grader_temperature,
                )
                f = grader_client.sample(prompt=grader_input, num_samples=1, sampling_params=grader_params)
                grading_futures.append((goal_idx, sample_idx, final_plan, f))

        grading_results = {}
        grader_xmls = {}
        for goal_idx, sample_idx, final_plan, f in grading_futures:
            try:
                response = f.result()
                xml_text = get_text_content(renderer.parse_response(response.sequences[0].tokens)[0])
                rubric_score = compute_rubric_reward_from_xml(xml_text)
                reward = compute_reward(rubric_score, final_plan, config)
                grading_results[(goal_idx, sample_idx)] = (rubric_score, reward)
                grader_xmls[(goal_idx, sample_idx)] = xml_text
            except Exception as e:
                logger.error(f"Grading failed goal={goal_idx} sample={sample_idx}: {e}")

        t_grade = time.time() - t0

        # ============================================================
        # PHASE 3: PRM per-turn evaluation (NEW in v2)
        # ============================================================
        prm_scores = {}  # {(goal_idx, sample_idx, turn_idx): float}
        batch_prm_positive = 0
        batch_prm_total = 0

        if config.prm_enabled:
            logger.info(f"[Batch {batch_idx}] Running PRM evaluation...")
            t0_prm = time.time()

            prm_futures = []
            for goal_idx, samples in conversations.items():
                for sample_idx, (turn_data, final_plan, convo) in samples.items():
                    if (goal_idx, sample_idx) not in grading_results:
                        continue
                    # PRM for turns 0, 1, 2 (not the final revision turn — rubric handles that)
                    for turn_idx in range(min(config.num_tutor_turns - 1, len(turn_data) - 1)):
                        tutor_text = turn_data[turn_idx]["text"]
                        next_student_msg = convo[(turn_idx + 1) * 2]["content"] if (turn_idx + 1) * 2 < len(convo) else ""
                        if not next_student_msg:
                            continue
                        prm_prompt = build_prm_prompt(tutor_text, next_student_msg)
                        prm_input = renderer.build_generation_prompt(
                            [{"role": "user", "content": prm_prompt}], role="assistant",
                        )
                        prm_params = types.SamplingParams(
                            max_tokens=config.prm_max_tokens,
                            stop=renderer.get_stop_sequences(),
                            temperature=config.prm_temperature,
                        )
                        f = grader_client.sample(prompt=prm_input, num_samples=1, sampling_params=prm_params)
                        prm_futures.append((goal_idx, sample_idx, turn_idx, f))

            for goal_idx, sample_idx, turn_idx, f in prm_futures:
                try:
                    response = f.result()
                    text = get_text_content(renderer.parse_response(response.sequences[0].tokens)[0])
                    score = parse_prm_score(text)
                    prm_scores[(goal_idx, sample_idx, turn_idx)] = score
                    batch_prm_total += 1
                    if score > 0:
                        batch_prm_positive += 1
                except Exception as e:
                    prm_scores[(goal_idx, sample_idx, turn_idx)] = 0.0

            t_prm = time.time() - t0_prm
            prm_rate = batch_prm_positive / batch_prm_total if batch_prm_total > 0 else 0
            logger.info(f"[Batch {batch_idx}] PRM: {batch_prm_total} evals, {prm_rate:.0%} positive, took {t_prm:.1f}s")
        else:
            t_prm = 0.0

        # ============================================================
        # PHASE 4: OPD hint extraction + teacher logprobs (NEW in v2)
        # ============================================================
        teacher_logprobs_store = {}  # {(goal_idx, sample_idx, turn_idx): list[float]}
        batch_opd_accepted = 0
        batch_opd_total = 0

        if config.opd_enabled:
            logger.info(f"[Batch {batch_idx}] Running OPD hint extraction + teacher logprobs...")
            t0_opd = time.time()

            for goal_idx, samples in conversations.items():
                for sample_idx, (turn_data, final_plan, convo) in samples.items():
                    key = (goal_idx, sample_idx)
                    if key not in grader_xmls:
                        continue

                    hint = extract_hint_from_grader(grader_xmls[key], config.opd_hint_min_length)
                    batch_opd_total += 1

                    if hint is None:
                        continue
                    batch_opd_accepted += 1

                    # Compute teacher logprobs for plan turns (last 2 turns)
                    plan_turn_indices = [t for t in range(len(turn_data)) if t >= config.num_tutor_turns - 2]
                    for turn_idx in plan_turn_indices:
                        teacher_lps = compute_teacher_logprobs_for_turn(
                            base_client, renderer, convo, hint, turn_idx, turn_data[turn_idx],
                        )
                        if teacher_lps is not None:
                            teacher_logprobs_store[(goal_idx, sample_idx, turn_idx)] = teacher_lps

            t_opd = time.time() - t0_opd
            accept_rate = batch_opd_accepted / batch_opd_total if batch_opd_total > 0 else 0
            logger.info(
                f"[Batch {batch_idx}] OPD: {batch_opd_accepted}/{batch_opd_total} hints accepted ({accept_rate:.0%}), "
                f"{len(teacher_logprobs_store)} teacher logprob sets, took {t_opd:.1f}s"
            )
        else:
            t_opd = 0.0

        # ============================================================
        # PHASE 5: Combined advantages + datum building
        # ============================================================
        training_datums = []
        batch_advantages = []
        batch_rubric_all = []
        batch_rewards_all = []
        batch_word_counts = []
        batch_format_penalties = []
        batch_logs = []
        dropped = 0

        for goal_idx in range(config.batch_size):
            group_keys = []
            group_rewards = []
            for s in range(config.group_size):
                key = (goal_idx, s)
                if key in grading_results:
                    rubric_score, reward = grading_results[key]
                    group_keys.append(key)
                    group_rewards.append(reward)
                    batch_rubric_all.append(rubric_score)
                    batch_rewards_all.append(reward)

                    turn_data, final_plan, convo = conversations[goal_idx][s]
                    solution = extract_solution(final_plan)
                    batch_word_counts.append(len(solution.split()))
                    is_compliant = check_format_compliance(final_plan, config.max_word_count)
                    batch_format_penalties.append(0.0 if is_compliant else 1.0)

            if len(group_rewards) < 2:
                continue

            mean_reward = np.mean(group_rewards)
            rl_advantages = [r - mean_reward for r in group_rewards]

            if all(a == 0.0 for a in rl_advantages):
                continue

            # Build datums per turn per sample
            for k, key in enumerate(group_keys):
                goal_idx_k, sample_idx_k = key
                rl_adv = rl_advantages[k]
                turn_data = conversations[goal_idx_k][sample_idx_k][0]

                for turn in turn_data:
                    turn_idx = turn["turn_idx"]
                    prompt_tokens = turn["prompt_tokens"]
                    response_tokens = turn["response_tokens"]
                    response_logprobs = turn["response_logprobs"]

                    # Determine per-token advantages
                    n_resp = len(response_logprobs)
                    opd_key = (goal_idx_k, sample_idx_k, turn_idx)
                    teacher_lps = teacher_logprobs_store.get(opd_key)

                    if teacher_lps is not None and config.opd_enabled:
                        # Per-token combined advantage (OPD + RL)
                        n = min(n_resp, len(teacher_lps))
                        resp_advantages = []
                        for i in range(n):
                            rl_component = config.opd_w_rl * rl_adv
                            opd_component = config.opd_w_opd * (
                                -config.opd_kl_coef * (response_logprobs[i] - teacher_lps[i])
                            )
                            resp_advantages.append(rl_component + opd_component)
                        # Pad remaining tokens with RL-only
                        for i in range(n, n_resp):
                            resp_advantages.append(rl_adv)
                    else:
                        # Discussion turns or no OPD: use PRM or RL scalar
                        prm_score = prm_scores.get((goal_idx_k, sample_idx_k, turn_idx), 0.0)
                        if config.prm_enabled and turn_idx < config.num_tutor_turns - 1:
                            # Discussion/early plan turns: PRM reward
                            turn_advantage = prm_score
                        else:
                            # Final turn or PRM disabled: RL advantage
                            turn_advantage = rl_adv
                        resp_advantages = [turn_advantage] * n_resp

                    batch_advantages.extend(resp_advantages)

                    full_seq = prompt_tokens + response_tokens
                    ob_len = len(prompt_tokens) - 1
                    input_tokens = full_seq[:-1]
                    target_tokens = full_seq[1:]

                    all_logprobs = [0.0] * ob_len + response_logprobs
                    all_advantages = [0.0] * ob_len + resp_advantages

                    n = min(len(input_tokens), len(target_tokens), len(all_logprobs), len(all_advantages))
                    if n == 0:
                        continue

                    datum = types.Datum(
                        model_input=types.ModelInput.from_ints(tokens=input_tokens[:n]),
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens[:n], dtype=torch.long)),
                            "logprobs": TensorData.from_torch(torch.tensor(all_logprobs[:n], dtype=torch.float)),
                            "advantages": TensorData.from_torch(torch.tensor(all_advantages[:n], dtype=torch.float)),
                        },
                    )
                    training_datums.append(datum)

        # ============================================================
        # PHASE 6: Optimization (same as v1)
        # ============================================================
        if not training_datums:
            logger.warning(f"[Batch {batch_idx}] No valid datums. Skipping.")
            continue

        try:
            logger.info(f"[Batch {batch_idx}] Training with {len(training_datums)} datums...")
            fwd_bwd_future = training_client.forward_backward(
                training_datums, loss_fn="ppo",
                loss_fn_config={
                    "clip_low_threshold": 1 - config.clip_eps,
                    "clip_high_threshold": 1 + config.clip_eps,
                },
            )
            optim_future = training_client.optim_step(adam_params)
            _fb = fwd_bwd_future.result()
            _opt = optim_future.result()
        except Exception as e:
            logger.exception(f"[Batch {batch_idx}] Training step failed")
            continue

        # Weight update
        try:
            result = training_client.save_weights_for_sampler(name=f"batch_{batch_idx}").result()
            sampling_client = service_client.create_sampling_client(model_path=result.path)
        except Exception as e:
            logger.warning(f"[Batch {batch_idx}] Weight snapshot failed: {e}")

        # ============================================================
        # Logging
        # ============================================================
        t_batch = time.time() - t_batch_start

        batch_summary = {
            "batch_idx": batch_idx,
            "rubric/mean": float(np.mean(batch_rubric_all)) if batch_rubric_all else 0.0,
            "rubric/std": float(np.std(batch_rubric_all)) if batch_rubric_all else 0.0,
            "reward/mean": float(np.mean(batch_rewards_all)) if batch_rewards_all else 0.0,
            "advantage/mean": float(np.mean(batch_advantages)) if batch_advantages else 0.0,
            "advantage/std": float(np.std(batch_advantages)) if batch_advantages else 0.0,
            "format/compliance_rate": 1.0 - (float(np.mean(batch_format_penalties)) if batch_format_penalties else 0.0),
            "length/mean": float(np.mean(batch_word_counts)) if batch_word_counts else 0.0,
            "samples/total": len(batch_rubric_all),
            "datums": len(training_datums),
            "prm/positive_rate": batch_prm_positive / batch_prm_total if batch_prm_total > 0 else 0.0,
            "prm/total": batch_prm_total,
            "opd/hint_accept_rate": batch_opd_accepted / batch_opd_total if batch_opd_total > 0 else 0.0,
            "opd/teacher_logprob_sets": len(teacher_logprobs_store),
            "time/batch_sec": t_batch,
            "time/gen_sec": t_gen,
            "time/grade_sec": t_grade,
            "time/prm_sec": t_prm,
            "time/opd_sec": t_opd,
        }

        logger.info(
            f"[Batch {batch_idx}] rubric={batch_summary['rubric/mean']:.4f} "
            f"reward={batch_summary['reward/mean']:.4f} datums={len(training_datums)} "
            f"PRM={batch_prm_positive}/{batch_prm_total} "
            f"OPD={batch_opd_accepted}/{batch_opd_total} hints "
            f"time={t_batch:.0f}s"
        )

        with open(batch_summary_path, "a") as f:
            f.write(json.dumps(batch_summary) + "\n")

        ml_logger.log_metrics({
            "progress/batch": batch_idx,
            "rubric/mean": batch_summary["rubric/mean"],
            "reward/mean": batch_summary["reward/mean"],
            "prm/positive_rate": batch_summary["prm/positive_rate"],
            "opd/hint_accept_rate": batch_summary["opd/hint_accept_rate"],
        }, step=batch_idx)

        if config.save_every > 0 and (batch_idx + 1) % config.save_every == 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"batch_{batch_idx:06d}",
                log_path=config.log_path, kind="both",
                loop_state={"batch": batch_idx},
            )
            logger.info(f"[Batch {batch_idx}] Checkpoint saved")

    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=f"batch_{batch_idx:06d}_final",
        log_path=config.log_path, kind="both",
        loop_state={"batch": batch_idx},
    )
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.entrypoint(main)
