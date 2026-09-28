"""
Multi-turn conversational GRPO trainer v3: 235B PRM + Unified OPD.

Key changes from v2:
  - Qwen3-235B-A22B as PRM (much more discriminative)
  - -3 to +3 confidence scale (fine-grained GRPO signal)
  - PRM analysis text = OPD hint (100% activation, no quality gate)
  - OPD on all 4 turns (not just plan turns)
  - group_size=8 (proper GRPO contrast)
  - Semaphore=4 (fix dmel_tokens)

Usage:
  source tools/use_api_profile.sh OLD
  python train_multiturn_v3.py config.batch_size=8 config.group_size=8
"""

import json
import logging
import os
import re
import sys
import textwrap
import threading
import time
from collections import Counter
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
    base_url: str | None = None
    api_profile: str | None = None
    log_path: str = "/home/silas/co-scientist-project/runs/2026/3/multiturn_v3/1"

    # Three model roles
    model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"
    prm_model_name: str = "Qwen/Qwen3-32B"
    lora_rank: int = 64

    curriculum_path: str = "/home/silas/co-scientist-project/data/curriculum/ml_train_curriculum.jsonl"

    num_tutor_turns: int = 4
    group_size: int = 8
    batch_size: int = 8

    max_tokens: int = 2048
    grader_max_tokens: int = 8192
    prm_max_tokens: int = 2048
    temperature: float = 1.0
    grader_temperature: float = 0.0
    prm_temperature: float = 0.3

    learning_rate: float = 1e-5
    clip_eps: float = 0.2
    save_every: int = 5
    max_concurrent_requests: int = 4

    opd_kl_coef: float = 1.0
    opd_w_rl: float = 0.5
    opd_w_opd: float = 0.5

    max_word_count: int = 750
    target_word_count: int = 600
    scale_length_bonus: float = 120.0
    scaling_factor: float = 0.08
    min_words: int = 30
    plan_turn_rubric_weight: float = 0.5


# ============================================================
# Helpers
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
    - Do NOT start your response with internal reasoning like "Okay, the user is asking..."
    - Respond directly and professionally to the student.
    - When asked for a research plan, wrap it in <think>...</think> then <solution>...</solution> tags. The solution must not exceed 750 words.
    - For discussion turns (not plan generation), just respond naturally without XML tags.
""").strip()


# ============================================================
# Rubric grader (30B, same as bestversion)
# ============================================================

def build_grader_prompt(scenario, rubric_items, proposed_plan, reference_solution=None):
    rubric_block = "\n".join([f"Item {i+1}: {item}" for i, item in enumerate(rubric_items)])
    desiderata_text = textwrap.dedent("""
    **DESIDERATA**
    1. HANDLES ALL CRITERIA  2. DETAILED, SPECIFIC SOLUTION  3. NO OVERLOOKED FLAWS
    4. WELL JUSTIFIED RATIONALE  5. COST AND EFFORT EFFICIENT  6. NO ETHICAL ISSUES
    7. CONSISTENT WITH OVERALL PLAN
    """).strip()

    prompt = f"Evaluate if the Proposed Research Plan satisfies the Research Scenario.\n\n# Research Scenario\n{scenario}\n\n# Rubric\n{rubric_block}\n"
    if reference_solution is not None:
        prompt += f"\n# Reference Solution\n{reference_solution}\n"
    prompt += f"\n# Proposed Research Plan\n{proposed_plan}\n\n"
    prompt += textwrap.dedent(f"""
        # Instructions
        Evaluate each rubric item against all 7 DESIDERATA ({desiderata_text}).
        For each desideratum: Level 0 (NOT SATISFIED), 1 (WEAKLY), 2 (PARTIALLY), 3 (FULLY).

        <rubric>
          <item num=1>
            <criteria>[rubric item]</criteria>
            <reasoning>[analysis]</reasoning>
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


def check_format_compliance(text, max_words):
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
# PRM: 235B evaluation with -3 to +3 scale
# ============================================================

PRM_DISCUSSION_PROMPT = textwrap.dedent("""
    You are an expert research methodology evaluator assessing a tutor's response
    in a multi-turn research discussion with a graduate student.

    ## Scoring Scale (-3 to +3)

    +3 EXCEPTIONAL: Deep domain insight, novel connections between concepts,
       highly specific and actionable research directions that the student
       couldn't easily find elsewhere.

    +2 GOOD: Substantive response that builds on the student's specific insight,
       provides concrete methodology suggestions with clear rationale.

    +1 ADEQUATE: Relevant to the topic but could be more specific. Gives
       reasonable directions without deep analysis.

     0 NEUTRAL: Generic response. Neither advances nor hinders the discussion.
       Could apply to any research topic.

    -1 WEAK: Vague or surface-level. Doesn't engage with the student's specific
       insight. Restates obvious points.

    -2 POOR: Misses the student's point, gives irrelevant advice, or provides
       generic template responses disconnected from the specific research context.

    -3 HARMFUL: Contains factually incorrect methodological guidance, contradicts
       the student's valid reasoning without justification, or would lead the
       research in a clearly wrong direction.

    ## Evaluation Criteria
    1. SPECIFICITY: Does it reference the student's exact insight and build on it?
    2. ACTIONABILITY: Does it suggest concrete next steps, methods, or frameworks?
    3. DEPTH: Does it go beyond surface-level advice into domain-specific details?
    4. VALIDITY: Are the suggested approaches methodologically sound?
    5. AWARENESS: Does it identify relevant pitfalls, limitations, or alternatives?
""").strip()

PRM_PLAN_PROMPT = textwrap.dedent("""
    You are an expert research methodology evaluator assessing a research plan
    generated by a tutor in a multi-turn discussion with a graduate student.

    ## Scoring Scale (-3 to +3)

    +3 EXCEPTIONAL: Comprehensive plan with novel methodology, concrete implementation
       details, well-justified rationale, and thorough risk analysis.

    +2 GOOD: Solid plan addressing the research goal with specific methods,
       clear structure, and reasonable justification.

    +1 ADEQUATE: Covers the main aspects but lacks depth in methodology or
       justification. Some vagueness in implementation details.

     0 NEUTRAL: Generic plan that could apply to many research goals.
       Missing specific methodology for this particular problem.

    -1 WEAK: Incomplete plan with significant gaps. Vague methods or
       unjustified choices.

    -2 POOR: Misses key aspects of the research goal. Methods are inappropriate
       or poorly reasoned.

    -3 HARMFUL: Plan contains methodological errors, would waste resources,
       or fundamentally misunderstands the research objective.

    ## Evaluation Criteria
    1. COMPREHENSIVENESS: Does the plan address all aspects of the research goal?
    2. METHODOLOGY: Are research methods concrete, specific, and implementable?
    3. JUSTIFICATION: Is each step motivated with clear rationale?
    4. RISK AWARENESS: Are limitations, confounders, and failure modes addressed?
    5. COHERENCE: Is the plan internally consistent and well-structured?
    6. FEASIBILITY: Is the plan realistic given typical research constraints?
""").strip()


def build_prm_prompt(tutor_response: str, next_student_message: str, is_plan_turn: bool) -> str:
    base = PRM_PLAN_PROMPT if is_plan_turn else PRM_DISCUSSION_PROMPT
    return (
        f"{base}\n\n"
        f"## Tutor's Response\n{tutor_response}\n\n"
        f"## Student's Follow-up\n{next_student_message}\n\n"
        f"## Instructions\n"
        f"Provide a detailed analysis (3-5 sentences) evaluating the response against the criteria above. "
        f"Be specific about what the tutor did well or poorly. Reference specific parts of the response. "
        f"Explain concretely what should be improved and how.\n\n"
        f"FORMAT (strict):\n"
        f"<analysis>\nYour detailed analysis here.\n</analysis>\n"
        f"<score>[integer from -3 to +3]</score>"
    )


def parse_prm_response(text: str) -> tuple[str, float]:
    """Returns (analysis_text, normalized_score in [-1, 1])."""
    analysis = ""
    score = 0.0

    analysis_match = re.search(r"<analysis>(.*?)</analysis>", text, re.DOTALL)
    if analysis_match:
        analysis = analysis_match.group(1).strip()

    score_match = re.search(r"<score>\s*([+-]?\d)\s*</score>", text)
    if score_match:
        raw = int(score_match.group(1))
        raw = max(-3, min(3, raw))
        score = raw / 3.0

    return analysis, score


# ============================================================
# OPD: teacher logprobs
# ============================================================

def compute_teacher_logprobs_for_turn(
    base_client, renderer, conversation, hint, turn_idx, turn_data, semaphore,
):
    student_msg_idx = turn_idx * 2
    augmented_convo = list(conversation[:student_msg_idx + 1])
    augmented_convo[-1] = {
        "role": augmented_convo[-1]["role"],
        "content": augmented_convo[-1]["content"] + f"\n\n[Expert feedback: {hint}]",
    }

    messages = [{"role": "system", "content": TUTOR_SYSTEM_PROMPT}] + augmented_convo
    teacher_input = renderer.build_generation_prompt(messages, role="assistant")
    teacher_prompt_tokens = [int(t) for t in teacher_input.to_ints()]

    response_tokens = turn_data["response_tokens"]
    full_sequence = types.ModelInput.from_ints(tokens=teacher_prompt_tokens + response_tokens)

    try:
        with semaphore:
            logprobs_result = base_client.compute_logprobs(full_sequence).result()
        start = len(teacher_prompt_tokens)
        teacher_lps = logprobs_result[start:start + len(response_tokens)]
        return [float(lp) for lp in teacher_lps]
    except Exception as e:
        logger.warning(f"Teacher logprob failed: {e}")
        return None


# ============================================================
# Curriculum + conversation generation
# ============================================================

def load_curriculum(path):
    curriculum = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            curriculum[entry["goal_id"]] = entry
    return curriculum


def generate_conversation(goal_idx, sample_idx, curriculum_entry, sampling_client, renderer, config, semaphore):
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

        with semaphore:
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

    data = load_dataset("facebook/research-plan-gen", "ml")
    dataset = data["train"]
    n_train_batches = len(dataset) // config.batch_size
    logger.info(f"Train: {len(dataset)} goals → {n_train_batches} batches of {config.batch_size}")

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    service_client = create_service_client(base_url=config.base_url, api_profile=config.api_profile)

    training_client = service_client.create_lora_training_client(
        base_model=config.model_name, rank=config.lora_rank,
    )
    logger.info(f"Created LoRA training client (rank={config.lora_rank})")

    sampling_result = training_client.save_weights_for_sampler(name="init").result()
    sampling_client = service_client.create_sampling_client(model_path=sampling_result.path)
    grader_client = service_client.create_sampling_client(base_model=config.grader_model_name)
    prm_client = service_client.create_sampling_client(base_model=config.prm_model_name)
    base_client = grader_client  # OPD teacher = base 30B (same as grader)

    logger.info(f"PRM model: {config.prm_model_name}")
    logger.info(f"Grader model: {config.grader_model_name}")

    adam_params = types.AdamParams(learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    tinker_semaphore = threading.Semaphore(config.max_concurrent_requests)

    training_logs_path = os.path.join(config.log_path, "train/training_logs.jsonl")
    batch_summary_path = os.path.join(config.log_path, "train/batch_summary.jsonl")
    t_global_start = time.time()

    logger.info("=" * 60)
    logger.info(
        f"v3 training | {n_train_batches} batches | group={config.group_size} | "
        f"PRM={config.prm_model_name.split('/')[-1]} | OPD on all turns"
    )
    logger.info("=" * 60)

    for batch_idx in range(n_train_batches):
        t_batch_start = time.time()
        batch_start = batch_idx * config.batch_size
        batch_rows = dataset[batch_start:batch_start + config.batch_size]

        # ==== PHASE 1: Generate conversations ====
        logger.info(f"[B{batch_idx}] Generating {config.batch_size}×{config.group_size} conversations...")
        t0 = time.time()
        conversation_results = []
        gen_errors = 0
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {}
            for i in range(config.batch_size):
                goal_id = batch_start + i
                if goal_id not in curriculum:
                    continue
                for s in range(config.group_size):
                    f = executor.submit(
                        generate_conversation, goal_idx=i, sample_idx=s,
                        curriculum_entry=curriculum[goal_id],
                        sampling_client=sampling_client, renderer=renderer,
                        config=config, semaphore=tinker_semaphore,
                    )
                    futures[f] = (i, s)
            for f in as_completed(futures):
                try:
                    conversation_results.append(f.result())
                except Exception as e:
                    gi, si = futures[f]
                    logger.error(f"Gen failed goal={gi} sample={si}: {e}")
                    gen_errors += 1

        # Retry on mass failure
        expected = sum(1 for i in range(config.batch_size) if batch_start + i in curriculum) * config.group_size
        if gen_errors > expected * 0.5 and expected > 0:
            logger.warning(f"[B{batch_idx}] >50% gen failures — refreshing client...")
            try:
                result = training_client.save_weights_for_sampler(name=f"refresh_{batch_idx}").result()
                sampling_client = service_client.create_sampling_client(model_path=result.path)
                conversation_results = []
                with ThreadPoolExecutor(max_workers=16) as executor:
                    futures = {}
                    for i in range(config.batch_size):
                        goal_id = batch_start + i
                        if goal_id not in curriculum:
                            continue
                        for s in range(config.group_size):
                            f = executor.submit(
                                generate_conversation, goal_idx=i, sample_idx=s,
                                curriculum_entry=curriculum[goal_id],
                                sampling_client=sampling_client, renderer=renderer,
                                config=config, semaphore=tinker_semaphore,
                            )
                            futures[f] = (i, s)
                    for f in as_completed(futures):
                        try:
                            conversation_results.append(f.result())
                        except Exception:
                            pass
            except Exception:
                logger.exception(f"[B{batch_idx}] Refresh failed")
                continue

        t_gen = time.time() - t0
        conversations = {}
        for r in conversation_results:
            goal_idx, sample_idx, turn_data, final_plan, convo = r
            conversations.setdefault(goal_idx, {})[sample_idx] = (turn_data, final_plan, convo)

        # ==== PHASE 2: Grade final plans (30B) ====
        n_plans = sum(len(s) for s in conversations.values())
        logger.info(f"[B{batch_idx}] Grading {n_plans} plans (30B)...")
        t0 = time.time()

        grade_futures = []
        for goal_idx, samples in conversations.items():
            for sample_idx, (td, fp, convo) in samples.items():
                gp = build_grader_prompt(
                    scenario=batch_rows["Goal"][goal_idx],
                    rubric_items=batch_rows["Rubric"][goal_idx],
                    proposed_plan=extract_solution(fp),
                    reference_solution=batch_rows["Reference solution"][goal_idx],
                )
                gi = renderer.build_generation_prompt([{"role": "user", "content": gp}], role="assistant")
                gparams = types.SamplingParams(
                    max_tokens=config.grader_max_tokens,
                    stop=renderer.get_stop_sequences(), temperature=config.grader_temperature,
                )
                with tinker_semaphore:
                    gf = grader_client.sample(prompt=gi, num_samples=1, sampling_params=gparams)
                grade_futures.append((goal_idx, sample_idx, fp, gf))

        grading_results = {}
        for goal_idx, sample_idx, fp, gf in grade_futures:
            try:
                resp = gf.result()
                xml = get_text_content(renderer.parse_response(resp.sequences[0].tokens)[0])
                rs = compute_rubric_reward_from_xml(xml)
                rw = compute_reward(rs, fp, config)
                grading_results[(goal_idx, sample_idx)] = (rs, rw)
            except Exception as e:
                logger.error(f"Grade failed g={goal_idx} s={sample_idx}: {e}")
        t_grade = time.time() - t0

        # ==== PHASE 3: PRM evaluation on ALL turns (235B) ====
        logger.info(f"[B{batch_idx}] PRM evaluation (235B, all turns)...")
        t0 = time.time()

        prm_data = {}  # {(goal, sample, turn): (analysis, norm_score)}
        score_counter = Counter()
        prm_futures = []

        prm_params = types.SamplingParams(
            max_tokens=config.prm_max_tokens,
            stop=renderer.get_stop_sequences(),
            temperature=config.prm_temperature,
        )

        for goal_idx, samples in conversations.items():
            for sample_idx, (turn_data, fp, convo) in samples.items():
                if (goal_idx, sample_idx) not in grading_results:
                    continue
                for turn_idx in range(len(turn_data)):
                    tutor_text = turn_data[turn_idx]["text"]
                    # Next student message (or final plan summary for last turn)
                    next_msg_idx = (turn_idx + 1) * 2
                    if next_msg_idx < len(convo):
                        next_msg = convo[next_msg_idx]["content"]
                    else:
                        next_msg = "(This was the final response in the conversation.)"

                    is_plan = turn_idx >= config.num_tutor_turns - 2
                    prompt = build_prm_prompt(tutor_text, next_msg, is_plan)
                    pi = renderer.build_generation_prompt([{"role": "user", "content": prompt}], role="assistant")

                    with tinker_semaphore:
                        pf = prm_client.sample(prompt=pi, num_samples=1, sampling_params=prm_params)
                    prm_futures.append((goal_idx, sample_idx, turn_idx, pf))

        for goal_idx, sample_idx, turn_idx, pf in prm_futures:
            try:
                resp = pf.result()
                text = get_text_content(renderer.parse_response(resp.sequences[0].tokens)[0])
                analysis, norm_score = parse_prm_response(text)
                prm_data[(goal_idx, sample_idx, turn_idx)] = (analysis, norm_score)
                raw_score = round(norm_score * 3)
                score_counter[raw_score] += 1
            except Exception as e:
                logger.warning(f"PRM failed g={goal_idx} s={sample_idx} t={turn_idx}: {e}")
                prm_data[(goal_idx, sample_idx, turn_idx)] = ("", 0.0)

        t_prm = time.time() - t0
        logger.info(
            f"[B{batch_idx}] PRM done: {len(prm_data)} evals, "
            f"dist={dict(sorted(score_counter.items()))}, took {t_prm:.1f}s"
        )

        # ==== PHASE 4: OPD teacher logprobs (30B base, all turns) ====
        logger.info(f"[B{batch_idx}] OPD teacher logprobs (all turns)...")
        t0 = time.time()

        teacher_lps_store = {}
        opd_kl_magnitudes = []

        with ThreadPoolExecutor(max_workers=8) as executor:
            opd_futures = {}
            for (goal_idx, sample_idx, turn_idx), (analysis, _) in prm_data.items():
                if not analysis:
                    continue
                td, fp, convo = conversations[goal_idx][sample_idx]
                f = executor.submit(
                    compute_teacher_logprobs_for_turn,
                    base_client, renderer, convo, analysis, turn_idx,
                    td[turn_idx], tinker_semaphore,
                )
                opd_futures[f] = (goal_idx, sample_idx, turn_idx)

            for f in as_completed(opd_futures):
                key = opd_futures[f]
                try:
                    teacher_lps = f.result()
                    if teacher_lps is not None:
                        teacher_lps_store[key] = teacher_lps
                        # Track KL magnitude
                        goal_idx, sample_idx, turn_idx = key
                        student_lps = conversations[goal_idx][sample_idx][0][turn_idx]["response_logprobs"]
                        n = min(len(student_lps), len(teacher_lps))
                        if n > 0:
                            kl = np.mean(np.abs(np.array(student_lps[:n]) - np.array(teacher_lps[:n])))
                            opd_kl_magnitudes.append(kl)
                except Exception as e:
                    logger.warning(f"OPD failed {key}: {e}")

        t_opd = time.time() - t0
        mean_kl = float(np.mean(opd_kl_magnitudes)) if opd_kl_magnitudes else 0.0
        logger.info(
            f"[B{batch_idx}] OPD done: {len(teacher_lps_store)} sets, "
            f"mean |KL|={mean_kl:.4f}, took {t_opd:.1f}s"
        )

        # ==== PHASE 5: Combined advantages + datums ====
        training_datums = []
        batch_advantages = []
        batch_rubric_all = []
        batch_rewards_all = []
        batch_word_counts = []
        batch_format_penalties = []
        per_turn_prm_scores = {t: [] for t in range(config.num_tutor_turns)}

        for goal_idx in range(config.batch_size):
            # Collect per-turn rewards for GRPO grouping
            group_turn_rewards = {t: [] for t in range(config.num_tutor_turns)}
            group_keys = []

            for s in range(config.group_size):
                key = (goal_idx, s)
                if key not in grading_results:
                    continue
                rubric_score, reward = grading_results[key]
                group_keys.append(key)
                batch_rubric_all.append(rubric_score)
                batch_rewards_all.append(reward)

                td, fp, convo = conversations[goal_idx][s]
                solution = extract_solution(fp)
                batch_word_counts.append(len(solution.split()))
                batch_format_penalties.append(0.0 if check_format_compliance(fp, config.max_word_count) else 1.0)

                for turn_idx in range(len(td)):
                    analysis, prm_score = prm_data.get((goal_idx, s, turn_idx), ("", 0.0))
                    per_turn_prm_scores[turn_idx].append(prm_score)

                    if turn_idx == config.num_tutor_turns - 1:
                        turn_reward = (1 - config.plan_turn_rubric_weight) * prm_score + config.plan_turn_rubric_weight * reward
                    else:
                        turn_reward = prm_score

                    group_turn_rewards[turn_idx].append(turn_reward)

            if len(group_keys) < 2:
                continue

            # GRPO per turn position
            for turn_idx in range(config.num_tutor_turns):
                rewards_at_turn = group_turn_rewards[turn_idx]
                if len(rewards_at_turn) != len(group_keys):
                    continue
                mean_r = np.mean(rewards_at_turn)
                turn_advantages = [r - mean_r for r in rewards_at_turn]

                for k, key in enumerate(group_keys):
                    g, s = key
                    td = conversations[g][s][0]
                    if turn_idx >= len(td):
                        continue
                    turn = td[turn_idx]
                    rl_adv = turn_advantages[k]
                    response_tokens = turn["response_tokens"]
                    response_logprobs = turn["response_logprobs"]
                    prompt_tokens = turn["prompt_tokens"]
                    n_resp = len(response_logprobs)

                    # Per-token advantages (OPD + RL)
                    teacher_lps = teacher_lps_store.get((g, s, turn_idx))
                    if teacher_lps is not None:
                        n = min(n_resp, len(teacher_lps))
                        resp_advantages = []
                        for i in range(n):
                            opd_comp = config.opd_w_opd * (
                                -config.opd_kl_coef * (response_logprobs[i] - teacher_lps[i])
                            )
                            resp_advantages.append(config.opd_w_rl * rl_adv + opd_comp)
                        for i in range(n, n_resp):
                            resp_advantages.append(rl_adv)
                    else:
                        resp_advantages = [rl_adv] * n_resp

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

        # ==== PHASE 6: Optimization ====
        if not training_datums:
            logger.warning(f"[B{batch_idx}] No datums. Skipping.")
            continue

        try:
            logger.info(f"[B{batch_idx}] Training with {len(training_datums)} datums...")
            fb = training_client.forward_backward(
                training_datums, loss_fn="ppo",
                loss_fn_config={
                    "clip_low_threshold": 1 - config.clip_eps,
                    "clip_high_threshold": 1 + config.clip_eps,
                },
            )
            opt = training_client.optim_step(adam_params)
            fb.result()
            opt.result()
        except Exception as e:
            logger.exception(f"[B{batch_idx}] Training failed")
            continue

        try:
            result = training_client.save_weights_for_sampler(name=f"batch_{batch_idx}").result()
            sampling_client = service_client.create_sampling_client(model_path=result.path)
        except Exception as e:
            logger.warning(f"[B{batch_idx}] Weight update failed: {e}")

        # ==== Logging ====
        t_batch = time.time() - t_batch_start

        batch_summary = {
            "batch_idx": batch_idx,
            "rubric/mean": float(np.mean(batch_rubric_all)) if batch_rubric_all else 0.0,
            "rubric/std": float(np.std(batch_rubric_all)) if batch_rubric_all else 0.0,
            "reward/mean": float(np.mean(batch_rewards_all)) if batch_rewards_all else 0.0,
            "advantage/mean": float(np.mean(batch_advantages)) if batch_advantages else 0.0,
            "advantage/std": float(np.std(batch_advantages)) if batch_advantages else 0.0,
            "format/compliance": 1.0 - (float(np.mean(batch_format_penalties)) if batch_format_penalties else 0.0),
            "length/mean": float(np.mean(batch_word_counts)) if batch_word_counts else 0.0,
            "datums": len(training_datums),
            "prm/score_dist": dict(sorted(score_counter.items())),
            "prm/mean": float(np.mean([s for _, s in prm_data.values()])) if prm_data else 0.0,
            "prm/per_turn_mean": {t: float(np.mean(v)) if v else 0.0 for t, v in per_turn_prm_scores.items()},
            "opd/sets": len(teacher_lps_store),
            "opd/mean_kl": mean_kl,
            "time/total": t_batch, "time/gen": t_gen, "time/grade": t_grade,
            "time/prm": t_prm, "time/opd": t_opd,
        }

        logger.info(
            f"[B{batch_idx}] rubric={batch_summary['rubric/mean']:.4f} "
            f"PRM_mean={batch_summary['prm/mean']:.2f} "
            f"OPD_kl={mean_kl:.3f} datums={len(training_datums)} "
            f"adv_std={batch_summary['advantage/std']:.4f} "
            f"time={t_batch:.0f}s"
        )

        with open(batch_summary_path, "a") as f:
            f.write(json.dumps(batch_summary) + "\n")

        ml_logger.log_metrics({
            "progress/batch": batch_idx,
            "rubric/mean": batch_summary["rubric/mean"],
            "prm/mean": batch_summary["prm/mean"],
            "opd/mean_kl": mean_kl,
            "advantage/std": batch_summary["advantage/std"],
        }, step=batch_idx)

        if config.save_every > 0 and (batch_idx + 1) % config.save_every == 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"batch_{batch_idx:06d}", log_path=config.log_path,
                kind="both", loop_state={"batch": batch_idx},
            )
            logger.info(f"[B{batch_idx}] Checkpoint saved")

    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=f"batch_{batch_idx:06d}_final", log_path=config.log_path,
        kind="both", loop_state={"batch": batch_idx},
    )
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.entrypoint(main)
