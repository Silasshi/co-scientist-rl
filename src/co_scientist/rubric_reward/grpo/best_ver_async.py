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
    log_path: str = "/home/silas/co-scientist-project/runs/2026/4/best_ver_async/4"
    model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"

    # Evaluation parameter
    run_eval: bool = False
    eval_epoch: str = -1

    #Dataset
    ml_data: bool = True
    arxiv_data: bool = False
    pubmed_data: bool = False

    batch_size: int = 64     # Number of research goals
    group_size: int = 8      # G=8 indicated in the paper
    learning_rate: float = 1e-5
    clip_eps: float = 0.2       # Clipping epsilon for PPO/GRPO

    max_length: int = 32768
    lora_rank: int = 64
    save_every: int = 20  # 0 = disabled
    max_tokens: int = 3072
    grader_max_tokens: int = 8192
    temperature: float = 1.0
    grader_temperature: float = 0.0

    # Length and format constraints
    max_word_count: int = 850
    target_word_count: int = 700  # Ideal word count for length bonus
    scale_length_bonus: float = 140.0  # Scale for length bonus Gaussian
    scaling_factor: float = 0.08  # Scaling factor for length bonus
    min_words: int = 30  # Minimum words required to consider a plan valid

    today_date = time.strftime("%Y-%m-%d", time.localtime())


# ============================================================
# For sampling
# ============================================================

def build_research_plan_prompt(
    scenario: str,
    examples: list[dict] | None = None,
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
    prompt += textwrap.dedent(f"""
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

    return prompt

# ============================================================
# For grading
# ============================================================


def build_grader_prompt(
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


def check_format_compliance(text: str, max_words: int) -> bool:
    """
    Checks Section 3.2 constraints:
    1. Content must be within <solution></solution> tags.
    2. Content within tags must not exceed max_words (750).
    """
    pattern = r"<solution>\s*(.*?)\s*</solution>"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)

    if not match:
        return False  # Tags missing

    content = match.group(1)
    # Simple whitespace split for word count
    word_count = len(content.split())

    if word_count > max_words:
        return False

    return True


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
    """Sample `group_size` plans for one goal. Returns (goal_idx, prompt_tokens, response)."""
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
    grader_sampling_params: types.SamplingParams,
) -> tuple[int, int, types.SampleResponse | None]:
    """Grade one plan. Returns (group_idx, sample_idx, response_or_None)."""
    try:
        response = await grader_client.sample_async(
            prompt=grader_input,
            num_samples=1,
            sampling_params=grader_sampling_params,
        )
        return group_idx, sample_idx, response
    except Exception:
        logger.exception(
            "Grader sampling failed for group_idx=%s sample_idx=%s",
            group_idx, sample_idx,
        )
    return group_idx, sample_idx, None


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

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load dataset
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

    # Setup clients (async)
    service_client = create_service_client(
        base_url=config.base_url,
        api_profile=config.api_profile,
    )

    # Load or create checkpoint
    run_metadata = {"run_name": Path(config.log_path).name, "method": "best_ver_async"}
    last_checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path)
    resume_info = False

    if (config.eval_epoch == 0 and config.run_eval) or (not config.run_eval and last_checkpoint is None):
        resume_info = False
    elif config.eval_epoch > 0 and config.run_eval:
        checkpoints = checkpoint_utils.load_checkpoints_file(config.log_path)
        checkpoints_with_key = [c for c in checkpoints if "state_path" in c]
        if checkpoints_with_key:
            resume_info = checkpoints_with_key[config.eval_epoch - 1]
            logger.info(
                f"Found {len(checkpoints_with_key)} valid checkpoints in {config.log_path}"
            )
            logger.info(f"Using checkpoint: {resume_info}")
    else:
        resume_info = last_checkpoint

    if resume_info:
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(
            str(resume_info["state_path"]), user_metadata=run_metadata,
        )
        if (resume_info["batch"] + 1) // n_train_batches > 0:
            if not is_eval:
                logger.info(f"Training for epoch: {(resume_info['batch'] + 1) // n_train_batches}")
            actual_batch = resume_info["batch"] + 1
            start_batch = 0
        else:
            start_batch = resume_info["batch"] + 1
            actual_batch = start_batch
        if is_eval:
            logger.info(f"Evaluating for Checkpoint {resume_info['batch']}")
        else:
            logger.info(f"Resuming from batch {resume_info['batch']}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_name, rank=config.lora_rank,
            user_metadata=run_metadata,
        )
        start_batch = 0
        actual_batch = 0

    # Create grader client (async)
    grader_client = await service_client.create_sampling_client_async(
        base_model=config.grader_model_name
    )

    # Sampling parameters
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
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

    # Prepare sampler for eval mode
    if is_eval:
        logger.info("Eval mode: preparing fixed sampler")
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

        # Get batch
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(dataset))
        batch_rows = dataset.select(range(batch_start, batch_end))

        if not is_eval:
            # Snapshot current LoRA weights for sampling
            sampling_client = await training_client.save_weights_and_get_sampling_client_async(
                name=f"{real_batch:06d}"
            )

        logger.info(
            f"{'EVAL' if is_eval else 'TRAIN'} Batch {real_batch}: "
            f"{len(batch_rows)} goals x {config.group_size} samples"
        )

        # ==============================================================
        # PHASE 1: LAUNCH ALL POLICY GENERATIONS (fully concurrent)
        # ==============================================================
        policy_tasks = []
        for goal_idx, goal in enumerate(batch_rows["Goal"]):
            prompt_text = build_research_plan_prompt(scenario=goal, examples=None)
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
        # PHASE 2: COLLECT PLANS AS THEY ARRIVE & LAUNCH GRADERS
        # Uses asyncio.as_completed so graders start as soon as plans land.
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
                proposed_plan = renderers.get_text_content(
                    renderer.parse_response(group_result.tokens)[0]
                )

                # HARD-CODE SOLUTION CLOSURE (required under RL + truncation)
                if "<solution>" in proposed_plan and "</solution>" not in proposed_plan:
                    proposed_plan = proposed_plan.rstrip() + "\n</solution>"

                # Build and launch grader (non-blocking)
                grader_prompt_text = build_grader_prompt(
                    scenario=goal,
                    rubric_items=rubric,
                    proposed_plan=proposed_plan,
                    reference_solution=ref_sol,
                )
                grader_input = renderer.build_generation_prompt(
                    [{"role": "user", "content": grader_prompt_text}]
                )

                grader_tasks.append(
                    asyncio.create_task(
                        _sample_grader(
                            grader_client=grader_client,
                            grader_input=grader_input,
                            group_idx=goal_idx,
                            sample_idx=sample_idx,
                            grader_sampling_params=grader_sampling_params,
                        ),
                        name=f"grader_{goal_idx}_{sample_idx}",
                    )
                )

                group_samples_info.append({
                    "tokens": group_result.tokens,
                    "logprobs": group_result.logprobs,
                    "text": proposed_plan,
                })

            batch_groups_data[goal_idx] = {
                "prompt_tokens": prompt_tokens,
                "samples_info": group_samples_info,
                "rubric_len": len(rubric),
            }

        # ==============================================================
        # PHASE 3: COLLECT GRADES AS THEY ARRIVE & COMPUTE REWARDS
        # ==============================================================
        logger.info(f"Batch {real_batch}: Collecting grader results...")

        # Per-group reward accumulator: group_idx -> list of (sample_idx, reward, sample_info, log_entry)
        group_results: dict[int, list] = {}

        batch_word_counts = []
        batch_format_penalties = []
        batch_rubric_scores = []
        batch_sample_rewards = []
        batch_logs_to_save = []
        dropped_samples = 0
        num_total_samples = 0
        num_valid_samples = 0

        for g_task in asyncio.as_completed(grader_tasks):
            group_idx, sample_idx, grader_response = await g_task
            num_total_samples += 1

            group_data = batch_groups_data[group_idx]
            if group_data is None or grader_response is None:
                dropped_samples += 1
                continue

            # Parse grader output
            parsed_message, _ = renderer.parse_response(grader_response.sequences[0].tokens)
            xml_text = renderers.get_text_content(parsed_message)

            # Compute score
            rubric_score = compute_rubric_reward_from_xml(xml_text)

            plan_text = group_data["samples_info"][sample_idx]["text"]
            word_count = len(plan_text.strip().split())

            # Format penalty (identical to best_ver.py)
            is_compliant = check_format_compliance(plan_text, config.max_word_count)
            excess = max(0, word_count - config.max_word_count)
            format_penalty = 0.0 if is_compliant else 0.2 + 0.0005 * excess

            # Gaussian length bonus centered at target_word_count
            length_bonus = np.exp(-((word_count - config.target_word_count) / config.scale_length_bonus) ** 2)
            final_reward = rubric_score + config.scaling_factor * length_bonus - format_penalty

            # Record raw sample statistics
            batch_word_counts.append(word_count)
            batch_format_penalties.append(format_penalty)
            batch_rubric_scores.append(rubric_score)

            # Drop degenerate samples
            if word_count < config.min_words:
                logger.debug(f"Dropping degenerate sample (too short): {word_count} words")
                dropped_samples += 1
                continue

            if plan_text.strip() in {"<think>", "</think>", "<solution>", "</solution>"}:
                logger.debug("Dropping degenerate sample (only tags)")
                dropped_samples += 1
                continue

            batch_sample_rewards.append(final_reward)
            num_valid_samples += 1

            # Accumulate per-group
            if group_idx not in group_results:
                group_results[group_idx] = []
            group_results[group_idx].append({
                "sample_idx": sample_idx,
                "reward": final_reward,
                "sample_info": group_data["samples_info"][sample_idx],
            })

            # Log entry
            batch_logs_to_save.append({
                "batch_idx": real_batch,
                "group_idx": group_idx,
                "sample_idx": sample_idx,
                "policy_output": plan_text,
                "grader_output": xml_text,
                "rubric_score": rubric_score,
                "format_penalty": format_penalty,
                "final_reward": final_reward,
                "word_count": word_count,
                "is_compliant": is_compliant,
            })

        # ==============================================================
        # PHASE 3b: COMPUTE GRPO ADVANTAGES & BUILD TRAINING DATUMS
        # ==============================================================
        training_datums = []
        batch_rewards = []
        batch_advantages = []

        for group_idx, samples in group_results.items():
            group_data = batch_groups_data[group_idx]
            if group_data is None:
                continue

            group_reward_list = [s["reward"] for s in samples]
            if not group_reward_list:
                continue

            mean_reward = np.mean(group_reward_list)
            advantages = [(r - mean_reward) for r in group_reward_list]
            batch_rewards.append(mean_reward)
            batch_advantages.extend(advantages)

            # Skip if no learning signal
            if all(a == 0.0 for a in advantages):
                continue
            if len(samples) < 2:
                continue

            if not is_eval:
                for k, sample in enumerate(samples):
                    sample_info = sample["sample_info"]
                    advantage = advantages[k]

                    prompt_tokens = [int(t) for t in group_data["prompt_tokens"]]
                    generated_tokens = [int(t) for t in sample_info["tokens"]]
                    full_seq = prompt_tokens + generated_tokens

                    ob_len = len(prompt_tokens) - 1
                    input_tokens = full_seq[:-1]
                    target_tokens = full_seq[1:]

                    all_logprobs = [0.0] * ob_len + sample_info["logprobs"]
                    all_advantages = [0.0] * ob_len + [advantage] * len(sample_info["logprobs"])

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
        batch_summary = {
            "batch_idx": real_batch,
            "rubric/sample_mean_all": float(np.mean(batch_rubric_scores)) if batch_rubric_scores else 0.0,
            "rubric/sample_std_all": float(np.std(batch_rubric_scores)) if batch_rubric_scores else 0.0,
            "reward/sample_mean_valid": float(np.mean(batch_sample_rewards)) if batch_sample_rewards else 0.0,
            "reward/sample_std_valid": float(np.std(batch_sample_rewards)) if batch_sample_rewards else 0.0,
            "reward/group_mean_valid": float(np.mean(batch_rewards)) if batch_rewards else 0.0,
            "reward/group_std_valid": float(np.std(batch_rewards)) if batch_rewards else 0.0,
            "advantage_mean": float(np.mean(batch_advantages)) if batch_advantages else 0.0,
            "advantage_std": float(np.std(batch_advantages)) if batch_advantages else 0.0,
            "length_mean": float(np.mean(batch_word_counts)) if batch_word_counts else 0.0,
            "length_p90": float(np.percentile(batch_word_counts, 90)) if batch_word_counts else 0.0,
            "format_rate": float(np.mean(batch_format_penalties)) if batch_format_penalties else 0.0,
            "samples/generated": num_total_samples,
            "samples/valid": num_valid_samples,
        }

        # Write logs
        if batch_logs_to_save:
            if is_eval:
                generations_log_path = os.path.join(config.log_path, f"evaluation/eval_logs({actual_batch - 1}).jsonl")
            else:
                generations_log_path = os.path.join(config.log_path, "train/training_logs.jsonl")
            os.makedirs(os.path.dirname(generations_log_path), exist_ok=True)
            with open(generations_log_path, "a") as f:
                for log_item in batch_logs_to_save:
                    f.write(json.dumps(log_item) + "\n")

        if batch_summary:
            if is_eval:
                batch_summary_path = os.path.join(config.log_path, f"evaluation/eval_batch_summary({actual_batch - 1}).jsonl")
            else:
                batch_summary_path = os.path.join(config.log_path, "train/batch_summary.jsonl")
            os.makedirs(os.path.dirname(batch_summary_path), exist_ok=True)
            with open(batch_summary_path, "a") as f:
                f.write(json.dumps(batch_summary) + "\n")

        logger.info(
            f"Batch {real_batch}: rubric={batch_summary['rubric/sample_mean_all']:.4f}  "
            f"reward={batch_summary['reward/sample_mean_valid']:.4f}  "
            f"valid={num_valid_samples}/{num_total_samples}  "
            f"time={time.time() - t_start:.1f}s"
        )

        # --- EVAL MODE: skip optimization ---
        if is_eval:
            continue

        # ==============================================================
        # PHASE 4: OPTIMIZATION STEP (async)
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

        # Log metrics
        metrics: dict[str, float] = {
            "progress/batch": real_batch,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (real_batch + 1) / n_train_batches,
            "time/total": time.time() - t_start,
            "reward/total": sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0,
            "dropped_samples": dropped_samples,
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
