"""
Multi-turn conversational GRPO trainer for research plan generation.

Trains a model to act as a research tutor through multi-turn conversations.
Uses pre-generated curriculum (student messages) and trains with terminal
rubric reward broadcast to all turns.

Usage:
  source tools/use_api_profile.sh NEW
  python train_multiturn.py
  python train_multiturn.py config.batch_size=32 config.log_path=runs/2026/3/multiturn/2
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
    log_path: str = "/home/silas/co-scientist-project/runs/2026/3/multiturn/1"
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

    # Resume
    resume_from_batch: int = -1  # -1 = auto-detect from checkpoints


# ============================================================
# Helpers (from PoC / best_ver patterns)
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
# Grader (identical to best_ver.py / eval_only.py)
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
# Curriculum loader
# ============================================================

def load_curriculum(path: str) -> dict[int, dict]:
    """Load curriculum JSONL, keyed by goal_id."""
    curriculum = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            curriculum[entry["goal_id"]] = entry
    return curriculum


# ============================================================
# Core: single conversation generation (sync, uses Tinker Futures)
# ============================================================

def generate_conversation(
    goal_idx: int,
    sample_idx: int,
    curriculum_entry: dict,
    sampling_client,
    renderer,
    config: Config,
):
    """Generate one multi-turn conversation. Sync — each turn blocks on the previous."""
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

        # Sync Future — launch, then .result() blocks until done
        future = sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=sampling_params,
        )
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
# Core: grade a plan (sync)
# ============================================================

def grade_plan(plan_text, goal, rubric_items, reference_solution, grader_client, renderer, config):
    grader_prompt = build_grader_prompt(
        scenario=goal,
        rubric_items=rubric_items,
        proposed_plan=extract_solution(plan_text),
        reference_solution=reference_solution,
    )
    grader_input = renderer.build_generation_prompt(
        [{"role": "user", "content": grader_prompt}], role="assistant",
    )
    grader_params = types.SamplingParams(
        max_tokens=config.grader_max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.grader_temperature,
    )
    future = grader_client.sample(
        prompt=grader_input, num_samples=1, sampling_params=grader_params,
    )
    response = future.result()
    xml_text = get_text_content(renderer.parse_response(response.sequences[0].tokens)[0])
    rubric_score = compute_rubric_reward_from_xml(xml_text)
    reward = compute_reward(rubric_score, plan_text, config)
    return rubric_score, reward, xml_text


# ============================================================
# Main training loop
# ============================================================

def main(config: Config):
    # --- Setup logging ---
    os.makedirs(os.path.join(config.log_path, "train"), exist_ok=True)
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None, wandb_name=None,
        config=config, do_configure_logging_module=True,
    )

    # --- Load curriculum ---
    logger.info(f"Loading curriculum from {config.curriculum_path}...")
    curriculum = load_curriculum(config.curriculum_path)
    logger.info(f"Loaded {len(curriculum)} curriculum entries")

    # --- Load dataset (for goal ordering + rubric) ---
    logger.info("Loading dataset (train split)...")
    data = load_dataset("facebook/research-plan-gen", "ml")
    dataset = data["train"]
    n_train_batches = len(dataset) // config.batch_size
    logger.info(f"Train set: {len(dataset)} goals → {n_train_batches} batches of {config.batch_size}")

    # --- Tokenizer / renderer ---
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Renderer: {renderer_name}")

    # --- Service client ---
    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )

    # --- Create fresh training client ---
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name, rank=config.lora_rank,
    )
    logger.info(f"Created fresh LoRA training client (rank={config.lora_rank})")

    # --- Sampling + grader clients ---
    sampling_result = training_client.save_weights_for_sampler(name="init").result()
    sampling_client = service_client.create_sampling_client(model_path=sampling_result.path)
    grader_client = service_client.create_sampling_client(base_model=config.grader_model_name)

    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    # --- Log paths ---
    training_logs_path = os.path.join(config.log_path, "train/training_logs.jsonl")
    batch_summary_path = os.path.join(config.log_path, "train/batch_summary.jsonl")

    t_global_start = time.time()
    start_batch = 0

    logger.info("=" * 60)
    logger.info(f"Starting multi-turn training | {n_train_batches} batches | group_size={config.group_size} | turns={config.num_tutor_turns}")
    logger.info("=" * 60)

    for batch_idx in range(start_batch, n_train_batches):
        t_batch_start = time.time()
        batch_start = batch_idx * config.batch_size
        batch_end = batch_start + config.batch_size
        batch_rows = dataset[batch_start:batch_end]

        # ============================================================
        # PHASE 1: Generate conversations (parallel via ThreadPool)
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
                        generate_conversation,
                        goal_idx=i, sample_idx=s,
                        curriculum_entry=curriculum[goal_id],
                        sampling_client=sampling_client,
                        renderer=renderer, config=config,
                    )
                    futures[f] = (i, s)
            for f in as_completed(futures):
                try:
                    conversation_results.append(f.result())
                except Exception as e:
                    gi, si = futures[f]
                    logger.error(f"Conversation failed goal={gi} sample={si}: {e}")
                    gen_errors += 1
        t_gen = time.time() - t0
        logger.info(f"[Batch {batch_idx}] Generation took {t_gen:.1f}s ({gen_errors} errors)")

        # If most conversations failed, refresh sampling client and retry
        expected = sum(1 for i in range(config.batch_size) if batch_start + i in curriculum) * config.group_size
        if gen_errors > expected * 0.5 and expected > 0:
            logger.warning(f"[Batch {batch_idx}] >50% generation failures — refreshing sampling client and retrying...")
            try:
                result = training_client.save_weights_for_sampler(
                    name=f"refresh_batch_{batch_idx}"
                ).result()
                sampling_client = service_client.create_sampling_client(model_path=result.path)
                logger.info(f"[Batch {batch_idx}] Sampling client refreshed, retrying generation...")

                conversation_results = []
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = {}
                    for i in range(config.batch_size):
                        goal_id = batch_start + i
                        if goal_id not in curriculum:
                            continue
                        for s in range(config.group_size):
                            f = executor.submit(
                                generate_conversation,
                                goal_idx=i, sample_idx=s,
                                curriculum_entry=curriculum[goal_id],
                                sampling_client=sampling_client,
                                renderer=renderer, config=config,
                            )
                            futures[f] = (i, s)
                    for f in as_completed(futures):
                        try:
                            conversation_results.append(f.result())
                        except Exception as e:
                            gi, si = futures[f]
                            logger.error(f"Retry failed goal={gi} sample={si}: {e}")
                t_gen = time.time() - t0
                logger.info(f"[Batch {batch_idx}] Retry generation took {t_gen:.1f}s")
            except Exception as e:
                logger.exception(f"[Batch {batch_idx}] Sampling client refresh failed")
                continue

        # Organize by goal
        conversations = {}  # {goal_idx: {sample_idx: (turn_data, final_plan, convo)}}
        for r in conversation_results:
            if isinstance(r, Exception):
                logger.error(f"Conversation failed: {r}")
                continue
            goal_idx, sample_idx, turn_data, final_plan, convo = r
            if goal_idx not in conversations:
                conversations[goal_idx] = {}
            conversations[goal_idx][sample_idx] = (turn_data, final_plan, convo)

        # ============================================================
        # PHASE 2: Grade final plans (parallel via Tinker Futures)
        # ============================================================
        n_plans = sum(len(s) for s in conversations.values())
        logger.info(f"[Batch {batch_idx}] Grading {n_plans} plans...")
        t0 = time.time()

        # Launch all grading futures at once (Tinker handles parallelism)
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
                    stop=renderer.get_stop_sequences(),
                    temperature=config.grader_temperature,
                )
                f = grader_client.sample(
                    prompt=grader_input, num_samples=1, sampling_params=grader_params,
                )
                grading_futures.append((goal_idx, sample_idx, final_plan, f))

        # Collect grading results
        grading_results = []
        for goal_idx, sample_idx, final_plan, f in grading_futures:
            try:
                response = f.result()
                xml_text = get_text_content(renderer.parse_response(response.sequences[0].tokens)[0])
                rubric_score = compute_rubric_reward_from_xml(xml_text)
                reward = compute_reward(rubric_score, final_plan, config)
                grading_results.append(((goal_idx, sample_idx), (rubric_score, reward, xml_text)))
            except Exception as e:
                logger.error(f"Grading failed goal={goal_idx} sample={sample_idx}: {e}")
                grading_results.append(((goal_idx, sample_idx), e))

        t_grade = time.time() - t0
        logger.info(f"[Batch {batch_idx}] Grading took {t_grade:.1f}s")

        # Store rewards
        rewards = {}  # {(goal_idx, sample_idx): reward}
        rubric_scores = {}
        batch_rubric_all = []
        batch_rewards_all = []
        batch_word_counts = []
        batch_format_penalties = []
        batch_logs = []
        dropped = 0

        for (goal_idx, sample_idx), result in grading_results:
            if isinstance(result, Exception):
                dropped += 1
                continue
            rubric_score, reward, grader_xml = result
            turn_data, final_plan, convo = conversations[goal_idx][sample_idx]

            solution = extract_solution(final_plan)
            word_count = len(solution.split())

            if word_count < config.min_words:
                dropped += 1
                continue

            rewards[(goal_idx, sample_idx)] = reward
            rubric_scores[(goal_idx, sample_idx)] = rubric_score
            batch_rubric_all.append(rubric_score)
            batch_rewards_all.append(reward)
            batch_word_counts.append(word_count)

            is_compliant = check_format_compliance(final_plan, config.max_word_count)
            batch_format_penalties.append(0.0 if is_compliant else 1.0)

            batch_logs.append({
                "batch_idx": batch_idx,
                "goal_idx": goal_idx,
                "sample_idx": sample_idx,
                "rubric_score": rubric_score,
                "reward": reward,
                "word_count": word_count,
                "format_compliant": is_compliant,
                "conversation": [{"role": m["role"], "content": m["content"][:500]} for m in convo],
            })

        # ============================================================
        # PHASE 3: GRPO Advantages
        # ============================================================
        training_datums = []
        batch_advantages = []

        for goal_idx in range(config.batch_size):
            # Get rewards for all samples of this goal
            group_rewards = []
            group_keys = []
            for s in range(config.group_size):
                key = (goal_idx, s)
                if key in rewards:
                    group_rewards.append(rewards[key])
                    group_keys.append(key)

            if len(group_rewards) < 2:
                continue

            mean_reward = np.mean(group_rewards)
            advantages = [r - mean_reward for r in group_rewards]

            if all(a == 0.0 for a in advantages):
                continue

            batch_advantages.extend(advantages)

            # Build datums for each turn of each valid sample
            for k, key in enumerate(group_keys):
                advantage = advantages[k]
                turn_data = conversations[goal_idx][key[1]][0]  # turn_data list

                for turn in turn_data:
                    prompt_tokens = turn["prompt_tokens"]
                    response_tokens = turn["response_tokens"]
                    response_logprobs = turn["response_logprobs"]

                    full_seq = prompt_tokens + response_tokens
                    ob_len = len(prompt_tokens) - 1
                    input_tokens = full_seq[:-1]
                    target_tokens = full_seq[1:]

                    all_logprobs = [0.0] * ob_len + response_logprobs
                    all_advantages = [0.0] * ob_len + [advantage] * len(response_logprobs)

                    # Truncate if misaligned
                    n = min(len(input_tokens), len(target_tokens), len(all_logprobs), len(all_advantages))
                    if n == 0:
                        continue

                    datum = types.Datum(
                        model_input=types.ModelInput.from_ints(tokens=input_tokens[:n]),
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(
                                torch.tensor(target_tokens[:n], dtype=torch.long)
                            ),
                            "logprobs": TensorData.from_torch(
                                torch.tensor(all_logprobs[:n], dtype=torch.float)
                            ),
                            "advantages": TensorData.from_torch(
                                torch.tensor(all_advantages[:n], dtype=torch.float)
                            ),
                        },
                    )
                    training_datums.append(datum)

        # ============================================================
        # PHASE 4: Optimization
        # ============================================================
        if not training_datums:
            logger.warning(f"[Batch {batch_idx}] No valid datums. Skipping optimization.")
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

        # ============================================================
        # PHASE 5: Update weights (every batch to stay on-policy)
        # ============================================================
        try:
            result = training_client.save_weights_for_sampler(
                name=f"batch_{batch_idx}"
            ).result()
            sampling_client = service_client.create_sampling_client(
                model_path=result.path
            )
            logger.info(f"[Batch {batch_idx}] Weights updated")
        except Exception as e:
            logger.warning(f"[Batch {batch_idx}] Weight snapshot failed: {e}. Will retry next batch.")

        # ============================================================
        # Logging
        # ============================================================
        t_batch = time.time() - t_batch_start

        batch_summary = {
            "batch_idx": batch_idx,
            "rubric/mean": float(np.mean(batch_rubric_all)) if batch_rubric_all else 0.0,
            "rubric/std": float(np.std(batch_rubric_all)) if batch_rubric_all else 0.0,
            "reward/mean": float(np.mean(batch_rewards_all)) if batch_rewards_all else 0.0,
            "reward/std": float(np.std(batch_rewards_all)) if batch_rewards_all else 0.0,
            "advantage/mean": float(np.mean(batch_advantages)) if batch_advantages else 0.0,
            "advantage/std": float(np.std(batch_advantages)) if batch_advantages else 0.0,
            "format/compliance_rate": 1.0 - (float(np.mean(batch_format_penalties)) if batch_format_penalties else 0.0),
            "length/mean": float(np.mean(batch_word_counts)) if batch_word_counts else 0.0,
            "samples/total": len(batch_rubric_all),
            "samples/dropped": dropped,
            "datums": len(training_datums),
            "time/batch_sec": t_batch,
            "time/gen_sec": t_gen,
            "time/grade_sec": t_grade,
        }

        logger.info(
            f"[Batch {batch_idx}] rubric={batch_summary['rubric/mean']:.4f}±{batch_summary['rubric/std']:.4f} "
            f"reward={batch_summary['reward/mean']:.4f} datums={len(training_datums)} "
            f"compliance={batch_summary['format/compliance_rate']:.1%} "
            f"time={t_batch:.0f}s (gen={t_gen:.0f}s grade={t_grade:.0f}s)"
        )

        with open(batch_summary_path, "a") as f:
            f.write(json.dumps(batch_summary) + "\n")

        if batch_logs:
            with open(training_logs_path, "a") as f:
                for log_item in batch_logs:
                    f.write(json.dumps(log_item, ensure_ascii=False) + "\n")

        ml_logger.log_metrics({
            "progress/batch": batch_idx,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
            "time/total": time.time() - t_global_start,
            "rubric/mean": batch_summary["rubric/mean"],
            "reward/mean": batch_summary["reward/mean"],
        }, step=batch_idx)

        # Checkpoint
        if config.save_every > 0 and (batch_idx + 1) % config.save_every == 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"batch_{batch_idx:06d}",
                log_path=config.log_path,
                kind="both",
                loop_state={"batch": batch_idx},
            )
            logger.info(f"[Batch {batch_idx}] Checkpoint saved")

    # Final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=f"batch_{batch_idx:06d}_final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": batch_idx},
    )
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.entrypoint(main)
