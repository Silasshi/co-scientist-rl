"""
Think-Solution Trainer: Split per-token advantages for think and solution stages.

Grades the think block with a 5-criterion rubric and the solution block with the
existing 7-desiderata rubric. Applies different GRPO advantages to think tokens
vs solution tokens, enabling direct supervision of the reasoning process.
"""

import asyncio
import logging
import time
import re
import textwrap
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
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from datasets import load_dataset


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    api_profile: str | None = None
    log_path: str = "/home/silas/co-scientist-project/runs/2026/4/think_solution/1"
    model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"

    run_eval: bool = False
    eval_epoch: int = -1

    ml_data: bool = True
    arxiv_data: bool = False
    pubmed_data: bool = False

    batch_size: int = 64
    group_size: int = 8
    learning_rate: float = 1e-5
    clip_eps: float = 0.2

    max_length: int = 32768
    lora_rank: int = 64
    save_every: int = 20
    max_tokens: int = 4096
    grader_max_tokens: int = 8192
    temperature: float = 1.0
    grader_temperature: float = 0.0

    # Solution format constraints (same as best_ver_async)
    max_word_count: int = 850
    target_word_count: int = 700
    scale_length_bonus: float = 140.0
    scaling_factor: float = 0.08
    min_words: int = 30

    # Think-solution specific
    think_weight: float = 0.3           # alpha for fallback combined reward
    max_think_words: int = 400          # think block budget
    think_grader_max_tokens: int = 4096
    min_think_words: int = 10           # below this, think_reward = 0

    today_date = time.strftime("%Y-%m-%d", time.localtime())


# ============================================================
# Text extraction
# ============================================================

def extract_solution_text(plan_text: str) -> str | None:
    pattern = r"<solution>\s*(.*?)\s*</solution>"
    matches = re.findall(pattern, plan_text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


# ============================================================
# Prompt: research plan (think + solution mode)
# ============================================================

def build_research_plan_prompt(
    scenario: str,
    max_think_words: int = 400,
    examples: list[dict] | None = None,
) -> str:
    prompt = textwrap.dedent(f"""
        I will provide you a research scenario. You have to provide me a concise yet thoughtful research plan with all details needed to execute it.
    """).strip()

    if examples:
        prompt += "\n\nFirst, I will show you some examples of research scenarios and how the researchers approached it."
        for i, ex in enumerate(examples):
            prompt += textwrap.dedent(f"""
                **Example {i+1}:**
                Scenario: {ex["scenario"]}

                Researcher's Plan:
                {ex["solution"]}
            """).strip()

    prompt += textwrap.dedent(f"""

        Here is the research scenario.
        Scenario: {scenario}

        # Instructions
        First, come up with a detailed research plan to address the scenario based on the following overall solution guidelines:
        - The plan should address the goals of the scenario, and account for all constraints and confounders.
        - Do NOT just say WHAT you will do. Explain HOW you will do it and WHY it is needed. Provide clear explanation and justification for each proposed step.
        - The phrasing should NOT be verbose, and NOT be in past tense, as in "the author's approach" but rather in present tense, as how you would approach the problem.
        - Do not claim to have done any experiments or have results, just provide the plan.
        - Do not add self-proclaimed praises of your solution.

        Return exactly two XML blocks and close all tags:

        1)
        <think>
        Your detailed reasoning process before writing the plan. Up to {max_think_words} words.
        - Systematically analyze what each evaluation criterion requires
        - Consider multiple approaches and compare their trade-offs
        - Use concrete technical details, not vague generalities
        - Identify potential weaknesses or failure modes in your approach
        - Organize your reasoning before jumping to the final plan
        </think>

        2)
        <solution>
        Complete and self-contained research plan, 550-700 words.
        The solution should be readable for humans, not in XML.
        Include all details — do not refer back to the think block.
        </solution>

        Important: The <think> block should contain genuine, substantive reasoning
        that improves your plan. Both blocks must appear exactly once.
    """).strip()

    return prompt


# ============================================================
# Grader: solution
# ============================================================

def build_solution_grader_prompt(
    scenario: str,
    rubric_items: list[str],
    proposed_plan: str,
    reference_solution: str | None = None,
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


# ============================================================
# Grader: think stage (NEW)
# ============================================================

def build_think_grader_prompt(
    scenario: str,
    rubric_items: list[str],
    think_text: str,
) -> str:
    rubric_block = "\n".join(
        [f"Item {i+1}: {item}" for i, item in enumerate(rubric_items)]
    )

    prompt = textwrap.dedent(f"""
        Evaluate the quality of the REASONING PROCESS below. This is a researcher's
        thinking block generated before writing a research plan for the given scenario.
        You are evaluating the THINKING QUALITY, not a final plan.

        # Research Scenario
        {scenario}

        # Evaluation Rubric Items (what the final plan will be judged on)
        {rubric_block}

        # Researcher's Thinking Process
        {think_text}

        # Instructions
        Evaluate the thinking process against these 5 criteria. For each criterion,
        assign a level from 0 to 3:

        Level 0 — NOT SATISFIED: The thinking does not address this criterion at all.
        Level 1 — WEAKLY SATISFIED: The thinking touches on this but is vague or superficial.
        Level 2 — PARTIALLY SATISFIED: The thinking addresses this reasonably but with gaps.
        Level 3 — FULLY SATISFIED: The thinking clearly and substantively addresses this.

        **CRITERIA**

        1. CRITERION ANALYSIS: Does the thinking systematically analyze what each rubric
           item requires? Does it identify the key challenges and requirements from each
           evaluation criterion, or does it skip rubric items?

        2. METHOD EXPLORATION: Does the thinking consider multiple approaches or methods?
           Does it compare trade-offs between alternatives? Or does it immediately jump
           to a single approach without considering others?

        3. SPECIFICITY: Does the reasoning reference concrete techniques, methods,
           datasets, metrics, or tools? Or is it vague ("use appropriate methods",
           "consider various factors")?

        4. RISK ASSESSMENT: Does the thinking identify potential weaknesses, failure
           modes, confounders, or limitations? Does it anticipate what could go wrong?

        5. PLANNING STRUCTURE: Is the reasoning organized logically? Does it build from
           analysis to synthesis? Or is it a disorganized stream of loosely connected
           thoughts?

        Return exactly this XML structure:
        <think_eval>
            <criterion num=1>
                <reasoning>Brief explanation.</reasoning>
                <level>[0, 1, 2, or 3]</level>
            </criterion>
            <criterion num=2>
                <reasoning>Brief explanation.</reasoning>
                <level>[0, 1, 2, or 3]</level>
            </criterion>
            <criterion num=3>
                <reasoning>Brief explanation.</reasoning>
                <level>[0, 1, 2, or 3]</level>
            </criterion>
            <criterion num=4>
                <reasoning>Brief explanation.</reasoning>
                <level>[0, 1, 2, or 3]</level>
            </criterion>
            <criterion num=5>
                <reasoning>Brief explanation.</reasoning>
                <level>[0, 1, 2, or 3]</level>
            </criterion>
        </think_eval>
    """).strip()
    return prompt


# ============================================================
# Reward computation
# ============================================================

LEVEL_MAP = {0: 0.0, 1: 0.2, 2: 0.6, 3: 1.0}


def compute_solution_reward_from_xml(xml_text: str) -> float:
    """Compute solution rubric score — same as best_ver_async compute_rubric_reward_from_xml."""

    item_blocks = re.findall(
        r"<item\s+num=.*?>.*?</item>", xml_text, flags=re.DOTALL
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
        mapped = [LEVEL_MAP.get(l, 0.0) for l in levels]
        item_scores.append(sum(mapped) / len(mapped))

    if not item_scores:
        return 0.0
    return sum(item_scores) / len(item_scores)


def compute_think_reward_from_xml(xml_text: str) -> float:
    """Parse think grader XML and compute think reward from 5 criteria."""

    criterion_blocks = re.findall(
        r"<criterion\s+num=.*?>.*?</criterion>", xml_text, flags=re.DOTALL
    )
    if not criterion_blocks:
        return 0.0

    scores = []
    for block in criterion_blocks:
        levels = re.findall(r"<level>(\d+)</level>", block)
        if levels:
            level = int(levels[0])
            scores.append(LEVEL_MAP.get(level, 0.0))

    if not scores:
        return 0.0

    # Pad to 5 if fewer parsed
    while len(scores) < 5:
        scores.append(0.0)
    scores = scores[:5]

    return sum(scores) / len(scores)


def check_format_compliance(text: str, max_words: int) -> bool:
    """Check solution tags present and word count within limit."""
    pattern = r"<solution>\s*(.*?)\s*</solution>"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return False
    word_count = len(match.group(1).split())
    return word_count <= max_words


# ============================================================
# Token boundary detection
# ============================================================

_BOUNDARY_CACHE: dict[int, tuple[list[int], list[int]]] = {}

def _get_boundary_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    key = id(tokenizer)
    if key not in _BOUNDARY_CACHE:
        _BOUNDARY_CACHE[key] = (
            tokenizer.encode("</think>", add_special_tokens=False),
            tokenizer.encode("<solution>", add_special_tokens=False),
        )
    return _BOUNDARY_CACHE[key]


def find_think_solution_boundary(
    response_tokens: list[int],
    tokenizer,
) -> tuple[int, int] | None:
    """
    Find where think tokens end and solution tokens begin.

    Returns (think_end_idx, solution_start_idx) into response_tokens, or None.
    Think tokens: [0, think_end), solution tokens: [solution_start, len).
    Boundary tokens between them get 0 advantage.
    """
    think_close_ids, solution_open_ids = _get_boundary_token_ids(tokenizer)

    # Find </think> position
    think_close_end = None
    for i in range(len(response_tokens) - len(think_close_ids) + 1):
        if response_tokens[i:i + len(think_close_ids)] == think_close_ids:
            think_close_end = i + len(think_close_ids)
            break

    # Find <solution> position (search after </think>)
    search_start = think_close_end if think_close_end is not None else 0
    solution_open_start = None
    for i in range(search_start, len(response_tokens) - len(solution_open_ids) + 1):
        if response_tokens[i:i + len(solution_open_ids)] == solution_open_ids:
            solution_open_start = i
            break

    if think_close_end is None and solution_open_start is None:
        return None

    think_end = think_close_end if think_close_end is not None else 0
    sol_start = solution_open_start if solution_open_start is not None else think_end

    return (think_end, sol_start)


# ============================================================
# Async sampling helpers
# ============================================================

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


async def _sample_grader(
    grader_client: tinker.SamplingClient,
    grader_input: types.ModelInput,
    group_idx: int,
    sample_idx: int,
    grader_tag: str,
    grader_sampling_params: types.SamplingParams,
) -> tuple[int, int, str, types.SampleResponse | None]:
    """Grade one sample. Returns (group_idx, sample_idx, grader_tag, response)."""
    try:
        response = await grader_client.sample_async(
            prompt=grader_input,
            num_samples=1,
            sampling_params=grader_sampling_params,
        )
        return group_idx, sample_idx, grader_tag, response
    except Exception:
        logger.exception(
            "Grader sampling failed: group=%s sample=%s tag=%s",
            group_idx, sample_idx, grader_tag,
        )
    return group_idx, sample_idx, grader_tag, None


# ============================================================
# Main (async)
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
    n_train_batches = len(dataset) // config.batch_size

    service_client = create_service_client(
        base_url=config.base_url,
        api_profile=config.api_profile,
    )

    run_metadata = {"run_name": Path(config.log_path).name, "method": "think_solution"}
    last_checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path)
    resume_info = False

    if (config.eval_epoch == 0 and config.run_eval) or (not config.run_eval and last_checkpoint is None):
        resume_info = False
    elif config.eval_epoch > 0 and config.run_eval:
        checkpoints = checkpoint_utils.load_checkpoints_file(config.log_path)
        checkpoints_with_key = [c for c in checkpoints if "state_path" in c]
        if checkpoints_with_key:
            resume_info = checkpoints_with_key[config.eval_epoch - 1]
    else:
        resume_info = last_checkpoint

    if resume_info:
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(
            str(resume_info["state_path"]), user_metadata=run_metadata,
        )
        if (resume_info["batch"] + 1) // n_train_batches > 0:
            actual_batch = resume_info["batch"] + 1
            start_batch = 0
        else:
            start_batch = resume_info["batch"] + 1
            actual_batch = start_batch
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_name, rank=config.lora_rank,
            user_metadata=run_metadata,
        )
        start_batch = 0
        actual_batch = 0

    grader_client = await service_client.create_sampling_client_async(
        base_model=config.grader_model_name
    )

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )
    solution_grader_params = tinker.types.SamplingParams(
        max_tokens=config.grader_max_tokens,
        temperature=config.grader_temperature,
    )
    think_grader_params = tinker.types.SamplingParams(
        max_tokens=config.think_grader_max_tokens,
        temperature=config.grader_temperature,
    )
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    if is_eval:
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(
            name=f"Eval_Checkpoint{actual_batch - 1}"
        )

    logger.info(f"Training for {n_train_batches} batches")
    last_real_batch = -1

    # ================================================================
    # MAIN LOOP
    # ================================================================
    for batch_idx in range(start_batch, n_train_batches):
        t_start = time.time()

        if is_eval:
            real_batch = actual_batch
        else:
            real_batch = actual_batch + (batch_idx - start_batch)
        last_real_batch = real_batch

        # Periodic checkpoint
        if config.save_every > 0 and batch_idx % config.save_every == 0 and batch_idx > 0:
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{real_batch:06d}_{config.today_date}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": real_batch},
            )

        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(dataset))
        batch_rows = dataset.select(range(batch_start, batch_end))

        if not is_eval:
            sampling_client = await training_client.save_weights_and_get_sampling_client_async(
                name=f"{real_batch:06d}"
            )

        logger.info(
            f"{'EVAL' if is_eval else 'TRAIN'} Batch {real_batch}: "
            f"{len(batch_rows)} goals x {config.group_size} samples"
        )

        # ==============================================================
        # PHASE 1: LAUNCH ALL POLICY GENERATIONS
        # ==============================================================
        policy_tasks = []
        for goal_idx, goal in enumerate(batch_rows["Goal"]):
            prompt_text = build_research_plan_prompt(
                scenario=goal, max_think_words=config.max_think_words,
            )
            model_input = renderer.build_generation_prompt(
                [{"role": "user", "content": prompt_text}]
            )
            policy_tasks.append(
                asyncio.create_task(
                    _sample_policy(
                        sampling_client=sampling_client,
                        model_input=model_input,
                        goal_idx=goal_idx,
                        prompt_tokens=model_input.to_ints(),
                        group_size=config.group_size,
                        sampling_params=sampling_params,
                    ),
                    name=f"policy_goal_{goal_idx}",
                )
            )

        # ==============================================================
        # PHASE 2: COLLECT PLANS & LAUNCH BOTH GRADERS PER SAMPLE
        # ==============================================================
        logger.info(f"Batch {real_batch}: Collecting plans, launching graders...")

        batch_groups_data: list[dict | None] = [None] * len(batch_rows["Goal"])
        grader_tasks = []

        for p_task in asyncio.as_completed(policy_tasks):
            policy_result = await p_task
            if policy_result is None:
                continue

            goal_idx, prompt_tokens, result = policy_result
            goal = batch_rows["Goal"][goal_idx]
            rubric = batch_rows["Rubric"][goal_idx]
            ref_sol = batch_rows["Reference solution"][goal_idx]

            group_samples_info = []

            for sample_idx, group_result in enumerate(result.sequences):
                parsed_msg, _ = renderer.parse_response(group_result.tokens)
                content = parsed_msg["content"]

                # Extract think text from Qwen3 native ThinkingPart
                think_text = ""
                if isinstance(content, list):
                    thinking_parts = [
                        p["thinking"] for p in content if p.get("type") == "thinking"
                    ]
                    if thinking_parts:
                        think_text = thinking_parts[0].strip()

                # Extract solution from text content (get_text_content strips thinking)
                proposed_plan = renderers.get_text_content(parsed_msg)

                # Handle truncation
                if "<solution>" in proposed_plan and "</solution>" not in proposed_plan:
                    proposed_plan = proposed_plan.rstrip() + "\n</solution>"

                # Extract solution text from the text portion
                solution_text = extract_solution_text(proposed_plan)
                # If no <solution> tags, use the full text content as solution
                if not solution_text:
                    solution_text = proposed_plan.strip()

                # Launch SOLUTION grader (grades solution only, not think)
                sol_grader_prompt = build_solution_grader_prompt(
                    scenario=goal,
                    rubric_items=rubric,
                    proposed_plan=solution_text,
                    reference_solution=ref_sol,
                )
                sol_grader_input = renderer.build_generation_prompt(
                    [{"role": "user", "content": sol_grader_prompt}]
                )
                grader_tasks.append(
                    asyncio.create_task(
                        _sample_grader(
                            grader_client=grader_client,
                            grader_input=sol_grader_input,
                            group_idx=goal_idx,
                            sample_idx=sample_idx,
                            grader_tag="solution",
                            grader_sampling_params=solution_grader_params,
                        ),
                        name=f"sol_grader_{goal_idx}_{sample_idx}",
                    )
                )

                # Launch THINK grader (grades think only)
                think_word_count = len(think_text.split()) if think_text else 0
                if think_word_count >= config.min_think_words and think_text:
                    think_grader_prompt = build_think_grader_prompt(
                        scenario=goal,
                        rubric_items=rubric,
                        think_text=think_text,
                    )
                    think_grader_input = renderer.build_generation_prompt(
                        [{"role": "user", "content": think_grader_prompt}]
                    )
                    grader_tasks.append(
                        asyncio.create_task(
                            _sample_grader(
                                grader_client=grader_client,
                                grader_input=think_grader_input,
                                group_idx=goal_idx,
                                sample_idx=sample_idx,
                                grader_tag="think",
                                grader_sampling_params=think_grader_params,
                            ),
                            name=f"think_grader_{goal_idx}_{sample_idx}",
                        )
                    )

                group_samples_info.append({
                    "tokens": group_result.tokens,
                    "logprobs": group_result.logprobs,
                    "text": proposed_plan,
                    "think_text": think_text,
                    "solution_text": solution_text,
                    "think_word_count": think_word_count,
                })

            batch_groups_data[goal_idx] = {
                "prompt_tokens": prompt_tokens,
                "samples_info": group_samples_info,
                "rubric_len": len(rubric),
            }

        # ==============================================================
        # PHASE 3: COLLECT GRADES & COMPUTE REWARDS
        # ==============================================================
        logger.info(f"Batch {real_batch}: Collecting grader results...")

        # Per-sample accumulators: (group_idx, sample_idx) -> scores
        sample_scores: dict[tuple[int, int], dict] = {}
        dropped_samples = 0
        num_total_samples = 0
        think_grader_failures = 0
        solution_grader_failures = 0

        for g_task in asyncio.as_completed(grader_tasks):
            group_idx, sample_idx, grader_tag, grader_response = await g_task

            group_data = batch_groups_data[group_idx]
            if group_data is None or grader_response is None:
                if grader_tag == "think":
                    think_grader_failures += 1
                else:
                    solution_grader_failures += 1
                continue

            parsed_message, _ = renderer.parse_response(grader_response.sequences[0].tokens)
            xml_text = renderers.get_text_content(parsed_message)

            key = (group_idx, sample_idx)
            if key not in sample_scores:
                sample_scores[key] = {"think_score": 0.0, "solution_score": 0.0, "solution_xml": ""}

            if grader_tag == "solution":
                sample_scores[key]["solution_score"] = compute_solution_reward_from_xml(xml_text)
                sample_scores[key]["solution_xml"] = xml_text
            elif grader_tag == "think":
                sample_scores[key]["think_score"] = compute_think_reward_from_xml(xml_text)
                sample_scores[key]["think_xml"] = xml_text

        # Now compute rewards and build group results
        group_results: dict[int, list] = {}
        batch_think_scores = []
        batch_solution_scores = []
        batch_think_word_counts = []
        batch_solution_word_counts = []
        batch_format_penalties = []
        batch_sample_rewards = []
        batch_logs_to_save = []
        num_valid_samples = 0
        boundary_detected_count = 0

        for (group_idx, sample_idx), scores in sample_scores.items():
            num_total_samples += 1
            group_data = batch_groups_data[group_idx]
            if group_data is None:
                dropped_samples += 1
                continue

            sample_info = group_data["samples_info"][sample_idx]
            plan_text = sample_info["text"]
            solution_text = sample_info["solution_text"]
            think_text = sample_info["think_text"]
            think_word_count = sample_info["think_word_count"]

            # Solution reward (same formula as best_ver_async)
            solution_rubric = scores["solution_score"]
            solution_word_count = len(solution_text.split()) if solution_text else len(plan_text.strip().split())

            is_compliant = check_format_compliance(plan_text, config.max_word_count)
            excess = max(0, solution_word_count - config.max_word_count)
            format_penalty = 0.0 if is_compliant else 0.2 + 0.0005 * excess
            length_bonus = np.exp(-((solution_word_count - config.target_word_count) / config.scale_length_bonus) ** 2)
            solution_reward = solution_rubric + config.scaling_factor * length_bonus - format_penalty

            think_score = scores["think_score"]

            # Drop degenerate samples
            total_word_count = len(plan_text.strip().split())
            if total_word_count < config.min_words:
                dropped_samples += 1
                continue
            if plan_text.strip() in {"<think>", "</think>", "<solution>", "</solution>"}:
                dropped_samples += 1
                continue

            # Detect token boundary
            boundary = find_think_solution_boundary(
                sample_info["tokens"], tokenizer
            )
            if boundary is not None:
                boundary_detected_count += 1

            num_valid_samples += 1

            # Accumulate
            batch_think_scores.append(think_score)
            batch_solution_scores.append(solution_rubric)
            batch_think_word_counts.append(think_word_count)
            batch_solution_word_counts.append(solution_word_count)
            batch_format_penalties.append(format_penalty)
            batch_sample_rewards.append(solution_reward)

            if group_idx not in group_results:
                group_results[group_idx] = []
            group_results[group_idx].append({
                "sample_idx": sample_idx,
                "think_reward": think_score,
                "solution_reward": solution_reward,
                "sample_info": sample_info,
                "boundary": boundary,
            })

            batch_logs_to_save.append({
                "batch_idx": real_batch,
                "group_idx": group_idx,
                "sample_idx": sample_idx,
                "policy_output": plan_text,
                "think_text": think_text,
                "solution_text": solution_text or "",
                "think_grader_output": scores.get("think_xml", ""),
                "solution_grader_output": scores.get("solution_xml", ""),
                "think_score": think_score,
                "solution_rubric_score": solution_rubric,
                "format_penalty": format_penalty,
                "solution_reward": solution_reward,
                "think_word_count": think_word_count,
                "solution_word_count": solution_word_count,
                "is_compliant": is_compliant,
                "boundary_detected": boundary is not None,
            })

        # ==============================================================
        # PHASE 3b: COMPUTE SPLIT GRPO ADVANTAGES & BUILD DATUMS
        # ==============================================================
        training_datums = []
        batch_think_advantages = []
        batch_solution_advantages = []

        for group_idx, samples in group_results.items():
            group_data = batch_groups_data[group_idx]
            if group_data is None or len(samples) < 2:
                continue

            # Independent GRPO baselines per stage
            think_rewards = [s["think_reward"] for s in samples]
            solution_rewards = [s["solution_reward"] for s in samples]
            think_mean = np.mean(think_rewards)
            solution_mean = np.mean(solution_rewards)
            think_advs = [float(r - think_mean) for r in think_rewards]
            solution_advs = [float(r - solution_mean) for r in solution_rewards]

            # Skip if no signal in either stage
            if all(a == 0.0 for a in think_advs) and all(a == 0.0 for a in solution_advs):
                continue

            batch_think_advantages.extend(think_advs)
            batch_solution_advantages.extend(solution_advs)

            # Pre-compute fallback combined mean (used when boundary detection fails)
            combined_rewards = [
                config.think_weight * s["think_reward"] + (1 - config.think_weight) * s["solution_reward"]
                for s in samples
            ]
            combined_mean = float(np.mean(combined_rewards))

            if not is_eval:
                for k, sample in enumerate(samples):
                    sample_info = sample["sample_info"]
                    boundary = sample["boundary"]
                    t_adv = think_advs[k]
                    s_adv = solution_advs[k]

                    prompt_tokens = [int(t) for t in group_data["prompt_tokens"]]
                    generated_tokens = [int(t) for t in sample_info["tokens"]]
                    full_seq = prompt_tokens + generated_tokens

                    ob_len = len(prompt_tokens) - 1
                    input_tokens = full_seq[:-1]
                    target_tokens = full_seq[1:]

                    all_logprobs = [0.0] * ob_len + list(sample_info["logprobs"])

                    # Build per-token advantages
                    if boundary is not None:
                        think_end, sol_start = boundary
                        per_token_advs = []
                        for t_idx in range(len(generated_tokens)):
                            if t_idx < think_end:
                                per_token_advs.append(t_adv)
                            elif t_idx >= sol_start:
                                per_token_advs.append(s_adv)
                            else:
                                per_token_advs.append(0.0)
                    else:
                        fallback_adv = float(combined_rewards[k] - combined_mean)
                        per_token_advs = [fallback_adv] * len(generated_tokens)

                    all_advantages = [0.0] * ob_len + per_token_advs

                    datum = types.Datum(
                        model_input=types.ModelInput.from_ints(tokens=input_tokens),
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
                            "logprobs": TensorData.from_torch(torch.tensor(all_logprobs, dtype=torch.float)),
                            "advantages": TensorData.from_torch(torch.tensor(all_advantages, dtype=torch.float)),
                        },
                    )
                    training_datums.append(datum)

        # Build batch summary
        bd_rate = boundary_detected_count / num_valid_samples if num_valid_samples > 0 else 0.0
        batch_summary = {
            "batch_idx": real_batch,
            "think/score_mean": float(np.mean(batch_think_scores)) if batch_think_scores else 0.0,
            "think/score_std": float(np.std(batch_think_scores)) if batch_think_scores else 0.0,
            "think/word_count_mean": float(np.mean(batch_think_word_counts)) if batch_think_word_counts else 0.0,
            "think/advantage_std": float(np.std(batch_think_advantages)) if batch_think_advantages else 0.0,
            "solution/rubric_mean": float(np.mean(batch_solution_scores)) if batch_solution_scores else 0.0,
            "solution/rubric_std": float(np.std(batch_solution_scores)) if batch_solution_scores else 0.0,
            "solution/reward_mean": float(np.mean(batch_sample_rewards)) if batch_sample_rewards else 0.0,
            "solution/word_count_mean": float(np.mean(batch_solution_word_counts)) if batch_solution_word_counts else 0.0,
            "solution/advantage_std": float(np.std(batch_solution_advantages)) if batch_solution_advantages else 0.0,
            "format_penalty_mean": float(np.mean(batch_format_penalties)) if batch_format_penalties else 0.0,
            "boundary_detection_rate": bd_rate,
            "samples/generated": num_total_samples,
            "samples/valid": num_valid_samples,
            "samples/dropped": dropped_samples,
            "samples/think_grader_failed": think_grader_failures,
            "samples/solution_grader_failed": solution_grader_failures,
        }

        # Write logs
        if batch_logs_to_save:
            if is_eval:
                log_path = os.path.join(config.log_path, f"evaluation/eval_logs({actual_batch - 1}).jsonl")
            else:
                log_path = os.path.join(config.log_path, "train/training_logs.jsonl")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a") as f:
                for item in batch_logs_to_save:
                    f.write(json.dumps(item) + "\n")

        if is_eval:
            summary_path = os.path.join(config.log_path, f"evaluation/eval_batch_summary({actual_batch - 1}).jsonl")
        else:
            summary_path = os.path.join(config.log_path, "train/batch_summary.jsonl")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "a") as f:
            f.write(json.dumps(batch_summary) + "\n")

        logger.info(
            f"Batch {real_batch}: think={batch_summary['think/score_mean']:.4f}  "
            f"solution={batch_summary['solution/rubric_mean']:.4f}  "
            f"reward={batch_summary['solution/reward_mean']:.4f}  "
            f"boundary={bd_rate:.1%}  "
            f"valid={num_valid_samples}/{num_total_samples}  "
            f"time={time.time() - t_start:.1f}s"
        )

        if is_eval:
            continue

        # ==============================================================
        # PHASE 4: OPTIMIZATION STEP
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
            _fwd_bwd_result = await fwd_bwd_future.result_async()
            logger.info(f"Forward/Backward took {time.time() - t0:.2f}s")

            t1 = time.time()
            _optim_result = await optim_step_future.result_async()
            logger.info(f"Optim step took {time.time() - t1:.2f}s")
        except Exception:
            logger.exception("Training step failed")
            continue

        metrics: dict[str, float] = {
            "progress/batch": real_batch,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (real_batch + 1) / n_train_batches,
            "time/total": time.time() - t_start,
            "think/score_mean": batch_summary["think/score_mean"],
            "solution/rubric_mean": batch_summary["solution/rubric_mean"],
            "boundary_detection_rate": bd_rate,
        }
        ml_logger.log_metrics(metrics, step=real_batch)

    # Save final checkpoint
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
