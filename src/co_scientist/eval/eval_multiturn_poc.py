"""
Multi-turn eval PoC: Does conversation improve research plans?

Compares two modes on the same test goals:
  1. Multi-turn: Sonnet (student) + bestversion model (tutor) conversation → plan
  2. Single-shot: Standard bestversion prompt → plan

Logs full conversations to JSONL for inspection.

Usage:
  # Set API keys first
  export ANTHROPIC_API_KEY="sk-ant-..."
  source tools/use_api_profile.sh <profile>

  # Run with defaults (20 goals, bestversion checkpoint)
  python eval_multiturn_poc.py

  # Custom settings
  python eval_multiturn_poc.py num_goals=10 checkpoint_batch=214
"""

import asyncio
import json
import logging
import os
import re
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import subprocess
import chz
import datasets
import tinker
from co_scientist.shared.api_profiles import create_service_client
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from datasets import load_dataset

# Import from eval_only.py (shared utilities)
from co_scientist.shared.eval_core import resolve_checkpoint


def get_text_content(message) -> str:
    """Extract text content from a parsed message."""
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(p["text"] for p in content if p["type"] == "text")


def strip_thinking(text: str) -> str:
    """Strip <think> blocks and inline thinking preambles from model output."""
    # Strip explicit <think>...</think> blocks
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text).strip()
    # Strip inline thinking preambles (Qwen3 chat mode leaks these without tags)
    text = re.sub(
        r"^(?:Okay|Ok|Alright|Let me|Let's|Hmm|So),?\s+"
        r"(?:the user|let me|let's|I need to|I should|I'll|this is|they)[\s\S]*?\n\n",
        "", text, count=1, flags=re.IGNORECASE
    ).strip()
    return text

logger = logging.getLogger(__name__)


# ============================================================
# Config
# ============================================================

@chz.chz
class Config:
    # Checkpoint
    checkpoint_run_path: str = "/home/silas/co-scientist-project/runs/2026/2/withA1,A2/2(ml)"
    checkpoint_batch: int = -1

    # Output
    output_dir: str = "/home/silas/co-scientist-project/runs/2026/3/multiturn/poc"

    # Model / API
    base_url: str | None = None
    api_profile: str | None = None
    model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 64

    # Anthropic (student model)
    anthropic_model: str = "claude-sonnet-4-20250514"

    # Eval settings
    num_goals: int = 20
    max_tokens: int = 2048
    grader_max_tokens: int = 8192
    temperature: float = 1.0
    grader_temperature: float = 0.0

    # Reward
    max_word_count: int = 750
    target_word_count: int = 600
    scale_length_bonus: float = 120.0
    scaling_factor: float = 0.08
    min_words: int = 30

    # Multi-turn
    num_tutor_turns: int = 4
    num_insights: int = 3


# ============================================================
# Prompt builders (reused from eval_only / best_ver)
# ============================================================

def build_research_plan_prompt(scenario: str) -> str:
    return textwrap.dedent(f"""
        I will provide you a research scenario. You have to provide me a concise yet thoughtful research plan with all details needed to execute it.

        Here is the research scenario.
        Scenario: {scenario}

        # Instructions
        First, come up with a detailed research plan to address the scenario based on the following overall solutionguidelines:
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


def build_grader_prompt(scenario, rubric_items, proposed_plan, reference_solution=None):
    rubric_block = "\n".join(
        [f"Item {i+1}: {item}" for i, item in enumerate(rubric_items)]
    )
    desiderata_text = textwrap.dedent("""
    **DESIDERATA**
    1. HANDLES ALL CRITERIA: Does the plan satisfy all criteria mentioned in the rubric item?
    2. DETAILED, SPECIFIC SOLUTION: Does the part of the plan relevant to satisfying this rubric item include fully specified details on HOW to implement it?
    3. NO OVERLOOKED FLAWS OR WEAKNESSES: Are there any important overlooked flaws or weaknesses?
    4. WELL JUSTIFIED RATIONALE: Is the part of the plan relevant to this grading item well-motivated and justified?
    5. COST AND EFFORT EFFICIENT: Does the plan handle this item efficiently without unnecessary complexity?
    6. NO ETHICAL ISSUES: Does this part of the plan have any potential for negative consequences?
    7. CONSISTENT WITH OVERALL PLAN: Is this part of the plan consistent with the rest of the plan?
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

            • It is just meant to demonstrate one possible approach that satisfies the scenario.
            • The Research Plan you have to grade might have different design choices. This is okay, if the choices are valid, and supported with correct rationale.
        """).strip()

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
                - Note that the plan should not just say it satisfies these desiderata, don't be fooled by that. Check carefully WHETHER, HOW and WHY the proposed plan meets each desiderata for this rubric item one by one.
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


def compute_rubric_reward_from_xml(xml_text: str) -> float:
    level_map = {0: 0.0, 1: 0.2, 2: 0.6, 3: 1.0}
    item_blocks = re.findall(r"<item\s+num=.*?>.*?</item>", xml_text, flags=re.DOTALL)
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
    return sum(item_scores) / len(item_scores) if item_scores else 0.0


def check_format_compliance(text: str, max_words: int) -> bool:
    match = re.search(r"<solution>\s*(.*?)\s*</solution>", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return False
    return len(match.group(1).split()) <= max_words


def extract_solution(text: str) -> str:
    match = re.search(r"<solution>\s*(.*?)\s*</solution>", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def compute_reward(rubric_score, plan_text, config):
    word_count = len(plan_text.strip().split())
    is_compliant = check_format_compliance(plan_text, config.max_word_count)
    excess = max(0, word_count - config.max_word_count)
    format_penalty = 0.0 if is_compliant else 0.2 + 0.0005 * excess
    length_bonus = np.exp(-((word_count - config.target_word_count) / config.scale_length_bonus) ** 2)
    return rubric_score + config.scaling_factor * length_bonus - format_penalty


# ============================================================
# Sonnet student simulation (via openclaw agent gateway)
# ============================================================

def _call_sonnet(message: str, session_id: str = "poc-sonnet", timeout: int = 60) -> str:
    """Call Sonnet via openclaw agent gateway (handles OAuth auth)."""
    result = subprocess.run(
        [
            "openclaw", "agent",
            "--message", message,
            "--session-id", session_id,
            "--json",
            "--timeout", str(timeout),
        ],
        capture_output=True, text=True, timeout=timeout + 10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"openclaw agent failed: {result.stderr[:500]}")
    data = json.loads(result.stdout)
    return data["result"]["payloads"][0]["text"]


def decompose_into_insights(goal: str, num_insights: int, goal_idx: int) -> list[str]:
    prompt = textwrap.dedent(f"""
        Given this research scenario, identify {num_insights} key insights or fragments
        that a researcher might develop before formulating a full plan.
        Each insight should be a partial observation, hypothesis, or constraint —
        NOT a complete solution, just a starting point for discussion.

        Research scenario:
        {goal}

        Return ONLY a JSON array of strings, e.g.: ["insight 1", "insight 2", "insight 3"]
    """).strip()
    text = _call_sonnet(prompt, session_id=f"poc-insights-{goal_idx}")
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())[:num_insights]
        except json.JSONDecodeError:
            pass
    lines = [l.strip().lstrip("-•0123456789.) ") for l in text.split("\n") if l.strip()]
    return lines[:num_insights]


def generate_student_message(goal: str, insights: list[str],
                             conversation: list[dict], turn_idx: int,
                             num_tutor_turns: int, goal_idx: int) -> str:
    if turn_idx == 0:
        return (
            f"I've been thinking about a research problem and have some initial ideas. "
            f"Here's one insight I've been considering:\n\n"
            f"\"{insights[0]}\"\n\n"
            f"What research directions could this lead to? What approaches would you suggest?"
        )

    if turn_idx == num_tutor_turns - 2:
        return (
            f"Those are great points. Let me share the full research goal I'm working on:\n\n"
            f"{goal}\n\n"
            f"Based on our discussion, can you synthesize a complete research plan? "
            f"Please put your plan inside <solution></solution> tags and keep it under 750 words. "
            f"Make it self-contained and detailed."
        )

    if turn_idx == num_tutor_turns - 1:
        convo_text = "\n".join(
            f"{'Student' if m['role']=='user' else 'Tutor'}: {m['content'][:500]}"
            for m in conversation[-4:]
        )
        prompt = textwrap.dedent(f"""
            You are a graduate student who just received a research plan from your tutor.
            Read the conversation below and ask ONE specific follow-up question to strengthen
            the plan. Focus on methodology, potential pitfalls, or missing justification.
            Keep your message to 2-3 sentences. End by asking the tutor to provide the
            revised plan inside <solution></solution> tags, under 750 words.

            Recent conversation:
            {convo_text}

            Your follow-up question (end with the <solution> tag request):
        """).strip()
        return _call_sonnet(prompt, session_id=f"poc-student-{goal_idx}-t{turn_idx}")

    # Middle turns: present additional insights
    insight_idx = min(turn_idx, len(insights) - 1)
    convo_text = "\n".join(
        f"{'Student' if m['role']=='user' else 'Tutor'}: {m['content'][:300]}"
        for m in conversation[-2:]
    )
    prompt = textwrap.dedent(f"""
        You are a curious graduate student discussing a research problem with a tutor.
        You have this additional insight you want to explore:

        "{insights[insight_idx]}"

        Based on the recent conversation:
        {convo_text}

        Ask a natural follow-up question that connects this insight to what the tutor said.
        Keep it to 2-3 sentences. Be curious and specific.

        Your message:
    """).strip()
    return _call_sonnet(prompt, session_id=f"poc-student-{goal_idx}-t{turn_idx}")


# ============================================================
# Tutor system prompt
# ============================================================

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
# Main async logic
# ============================================================

async def run_single_conversation(
    goal_idx: int,
    goal: str,
    rubric_items: list[str],
    reference_solution: str,
    config: Config,
    sampling_client,
    grader_client,
    renderer,
):
    """Run one multi-turn conversation and one single-shot generation for a goal."""
    t_start = time.time()
    logger.info(f"[Goal {goal_idx}] Starting...")

    # --- Decompose into insights ---
    insights = decompose_into_insights(goal, config.num_insights, goal_idx)
    logger.info(f"[Goal {goal_idx}] Insights: {[i[:60] + '...' for i in insights]}")

    # --- Multi-turn conversation ---
    conversation = []  # list of {"role": "user"|"assistant", "content": str}
    turn_log = []  # full turn details for logging

    for turn_idx in range(config.num_tutor_turns):
        # Generate student message
        student_msg = generate_student_message(
            goal, insights, conversation, turn_idx,
            config.num_tutor_turns, goal_idx,
        )
        conversation.append({"role": "user", "content": student_msg})
        turn_log.append({"turn": turn_idx * 2, "role": "student", "content": student_msg})
        logger.info(f"[Goal {goal_idx}] Turn {turn_idx} student: {student_msg[:80]}...")

        # Generate tutor response via Tinker
        messages_for_model = [{"role": "system", "content": TUTOR_SYSTEM_PROMPT}] + conversation
        model_input = renderer.build_generation_prompt(messages_for_model, role="assistant")

        sampling_params = types.SamplingParams(
            max_tokens=config.max_tokens,
            stop=renderer.get_stop_sequences(),
            temperature=config.temperature,
        )
        response = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        raw_tutor_text = get_text_content(
            renderer.parse_response(response.sequences[0].tokens)[0]
        )

        # Strip thinking content (Qwen3 leaks internal reasoning in chat mode)
        tutor_text = strip_thinking(raw_tutor_text)

        # Close unclosed solution tags
        if "<solution>" in tutor_text and "</solution>" not in tutor_text:
            tutor_text = tutor_text.rstrip() + "\n</solution>"

        conversation.append({"role": "assistant", "content": tutor_text})
        turn_log.append({
            "turn": turn_idx * 2 + 1, "role": "tutor",
            "content": tutor_text, "raw_content": raw_tutor_text,
        })
        logger.info(f"[Goal {goal_idx}] Turn {turn_idx} tutor: {tutor_text[:80]}...")

    # Extract final plan from last tutor response
    final_plan_text = conversation[-1]["content"]

    # --- Grade multi-turn plan ---
    grader_prompt = build_grader_prompt(
        scenario=goal,
        rubric_items=rubric_items,
        proposed_plan=extract_solution(final_plan_text),
        reference_solution=reference_solution,
    )
    grader_input = renderer.build_generation_prompt(
        [{"role": "user", "content": grader_prompt}], role="assistant"
    )
    grader_params = types.SamplingParams(
        max_tokens=config.grader_max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.grader_temperature,
    )
    grader_response = await grader_client.sample_async(
        prompt=grader_input, num_samples=1, sampling_params=grader_params,
    )
    mt_grader_xml = get_text_content(
        renderer.parse_response(grader_response.sequences[0].tokens)[0]
    )
    mt_rubric = compute_rubric_reward_from_xml(mt_grader_xml)
    mt_solution = extract_solution(final_plan_text)
    mt_reward = compute_reward(mt_rubric, final_plan_text, config)
    mt_word_count = len(mt_solution.split())
    mt_compliant = check_format_compliance(final_plan_text, config.max_word_count)

    logger.info(f"[Goal {goal_idx}] Multi-turn rubric={mt_rubric:.4f} reward={mt_reward:.4f} words={mt_word_count}")

    # --- Single-shot generation ---
    ss_prompt = build_research_plan_prompt(goal)
    ss_input = renderer.build_generation_prompt(
        [{"role": "user", "content": ss_prompt}], role="assistant"
    )
    ss_response = await sampling_client.sample_async(
        prompt=ss_input, num_samples=1, sampling_params=sampling_params,
    )
    ss_plan_text = get_text_content(
        renderer.parse_response(ss_response.sequences[0].tokens)[0]
    )
    if "<solution>" in ss_plan_text and "</solution>" not in ss_plan_text:
        ss_plan_text = ss_plan_text.rstrip() + "\n</solution>"

    # Grade single-shot plan
    ss_grader_prompt = build_grader_prompt(
        scenario=goal,
        rubric_items=rubric_items,
        proposed_plan=extract_solution(ss_plan_text),
        reference_solution=reference_solution,
    )
    ss_grader_input = renderer.build_generation_prompt(
        [{"role": "user", "content": ss_grader_prompt}], role="assistant"
    )
    ss_grader_response = await grader_client.sample_async(
        prompt=ss_grader_input, num_samples=1, sampling_params=grader_params,
    )
    ss_grader_xml = get_text_content(
        renderer.parse_response(ss_grader_response.sequences[0].tokens)[0]
    )
    ss_rubric = compute_rubric_reward_from_xml(ss_grader_xml)
    ss_reward = compute_reward(ss_rubric, ss_plan_text, config)
    ss_word_count = len(ss_plan_text.strip().split())
    ss_compliant = check_format_compliance(ss_plan_text, config.max_word_count)

    logger.info(f"[Goal {goal_idx}] Single-shot rubric={ss_rubric:.4f} reward={ss_reward:.4f} words={ss_word_count}")

    delta = mt_rubric - ss_rubric
    logger.info(f"[Goal {goal_idx}] Delta rubric={delta:+.4f} | time={time.time()-t_start:.1f}s")

    return {
        "goal_id": goal_idx,
        "goal": goal,
        "rubric_items": rubric_items,
        "insights": insights,
        "conversation": turn_log,
        "final_plan": final_plan_text,
        "multiturn_rubric_score": mt_rubric,
        "multiturn_reward": mt_reward,
        "multiturn_word_count": mt_word_count,
        "multiturn_format_compliant": mt_compliant,
        "singleshot_plan": ss_plan_text,
        "singleshot_rubric_score": ss_rubric,
        "singleshot_reward": ss_reward,
        "singleshot_word_count": ss_word_count,
        "singleshot_format_compliant": ss_compliant,
        "delta_rubric": delta,
        "grader_xml_multiturn": mt_grader_xml,
        "grader_xml_singleshot": ss_grader_xml,
    }


async def run_all(config: Config):
    # --- Setup ---
    os.makedirs(config.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(config.output_dir, "run.log")),
        ],
    )

    # --- Dataset ---
    logger.info("Loading dataset (test split)...")
    data = load_dataset("facebook/research-plan-gen", "ml")
    dataset = data["test"]
    goals = dataset[:config.num_goals]
    logger.info(f"Running {config.num_goals} goals from test set ({len(dataset)} total)")

    # --- Tokenizer / renderer ---
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Renderer: {renderer_name}")

    # --- Tinker clients ---
    service_client = create_service_client(
        base_url=config.base_url,
        api_profile=config.api_profile,
    )

    resume_info, batch_label = resolve_checkpoint(
        config.checkpoint_run_path, config.checkpoint_batch
    )
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )

    logger.info(f"Saving weights for eval sampler (checkpoint=b{batch_label})...")
    sampling_result = training_client.save_weights_for_sampler(
        name=f"multiturn_poc_b{batch_label}"
    ).result()
    sampling_client = service_client.create_sampling_client(
        model_path=sampling_result.path
    )
    grader_client = service_client.create_sampling_client(
        base_model=config.grader_model_name
    )

    # --- Sonnet via openclaw agent gateway ---
    logger.info("Using openclaw agent gateway for Sonnet student simulation")

    # --- Run conversations ---
    logger.info("=" * 60)
    logger.info("Starting multi-turn PoC evaluation")
    logger.info("=" * 60)

    # Run goals with limited concurrency to avoid overwhelming APIs
    semaphore = asyncio.Semaphore(4)
    results = []

    async def run_with_semaphore(goal_idx):
        async with semaphore:
            return await run_single_conversation(
                goal_idx=goal_idx,
                goal=goals["Goal"][goal_idx],
                rubric_items=goals["Rubric"][goal_idx],
                reference_solution=goals["Reference solution"][goal_idx],
                config=config,
                sampling_client=sampling_client,
                grader_client=grader_client,
                renderer=renderer,
            )

    tasks = [run_with_semaphore(i) for i in range(config.num_goals)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # --- Filter errors ---
    valid_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error(f"[Goal {i}] Failed: {r}")
        else:
            valid_results.append(r)

    # --- Write conversation logs ---
    logs_path = os.path.join(config.output_dir, "conversations.jsonl")
    with open(logs_path, "w") as f:
        for r in valid_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Conversations saved to {logs_path}")

    # --- Compute summary ---
    if not valid_results:
        logger.error("No valid results!")
        return

    mt_scores = [r["multiturn_rubric_score"] for r in valid_results]
    ss_scores = [r["singleshot_rubric_score"] for r in valid_results]
    deltas = [r["delta_rubric"] for r in valid_results]
    improved = sum(1 for d in deltas if d > 0.01)
    degraded = sum(1 for d in deltas if d < -0.01)
    same = len(deltas) - improved - degraded

    summary = {
        "num_goals": len(valid_results),
        "num_failed": len(results) - len(valid_results),
        "checkpoint": f"b{batch_label}",
        "multiturn": {
            "rubric_mean": float(np.mean(mt_scores)),
            "rubric_std": float(np.std(mt_scores)),
            "rubric_median": float(np.median(mt_scores)),
            "format_compliance": sum(r["multiturn_format_compliant"] for r in valid_results) / len(valid_results),
            "word_count_mean": float(np.mean([r["multiturn_word_count"] for r in valid_results])),
        },
        "singleshot": {
            "rubric_mean": float(np.mean(ss_scores)),
            "rubric_std": float(np.std(ss_scores)),
            "rubric_median": float(np.median(ss_scores)),
            "format_compliance": sum(r["singleshot_format_compliant"] for r in valid_results) / len(valid_results),
            "word_count_mean": float(np.mean([r["singleshot_word_count"] for r in valid_results])),
        },
        "comparison": {
            "delta_mean": float(np.mean(deltas)),
            "delta_std": float(np.std(deltas)),
            "improved": improved,
            "degraded": degraded,
            "same": same,
            "improved_pct": improved / len(deltas) * 100,
        },
    }

    summary_path = os.path.join(config.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # --- Print results ---
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Goals evaluated: {summary['num_goals']}")
    logger.info(f"  Multi-turn rubric:  {summary['multiturn']['rubric_mean']:.4f} ± {summary['multiturn']['rubric_std']:.4f}")
    logger.info(f"  Single-shot rubric: {summary['singleshot']['rubric_mean']:.4f} ± {summary['singleshot']['rubric_std']:.4f}")
    logger.info(f"  Delta:              {summary['comparison']['delta_mean']:+.4f} ± {summary['comparison']['delta_std']:.4f}")
    logger.info(f"  Improved: {improved}/{len(deltas)} ({improved/len(deltas)*100:.0f}%)")
    logger.info(f"  Degraded: {degraded}/{len(deltas)} ({degraded/len(deltas)*100:.0f}%)")
    logger.info(f"  Same:     {same}/{len(deltas)} ({same/len(deltas)*100:.0f}%)")
    logger.info(f"  Multi-turn format compliance: {summary['multiturn']['format_compliance']:.1%}")
    logger.info(f"  Single-shot format compliance: {summary['singleshot']['format_compliance']:.1%}")
    logger.info("=" * 60)
    logger.info(f"Full logs: {logs_path}")
    logger.info(f"Summary:   {summary_path}")


# ============================================================
# Entry point
# ============================================================

def main(config: Config):
    asyncio.run(run_all(config))


if __name__ == "__main__":
    chz.entrypoint(main)
