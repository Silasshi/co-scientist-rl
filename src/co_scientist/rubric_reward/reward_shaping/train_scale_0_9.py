import logging
import time
import re
import textwrap
import numpy as np
from concurrent.futures import Future, TimeoutError
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
    base_url: str | None = None
    api_profile: str | None = None
    log_path: str = "/home/silas/co-scientist-project/runs/2026/2/0-9_scale/15(All)"
    model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"

    # Evaluation parameter
    run_eval: bool = False
    eval_epoch: int = 1

    #Dataset
    ml_data: bool = True
    arxiv_data: bool = False
    pubmed_data: bool = False

    batch_size: int = 64     # Number of research goals
    group_size: int = 8      # G=8 indicated in the paper
    learning_rate: float = 1e-5
    clip_eps: float = 0.2       # Clipping epsilon for PPO/GRPO

    max_length: int = 32768
    lora_rank: int = 64     # Only 32 for openai/gpt-oss-20b due to memory constraints  
    save_every: int = 0  # 0 = disabled
    max_tokens: int = 2048
    grader_max_tokens: int = 8192
    temperature: float = 1.0
    grader_temperature: float = 0.0

    # Length and format constraints
    max_word_count: int = 750
    target_word_count: int = 600  # Ideal word count for length bonus
    scale_length_bonus: float = 120.0  # Scale for length bonus Gaussian
    scaling_factor: float = 0.08  # Scaling factor for length bonus
    min_words: int = 30  # Minimum words required to consider a plan valid

    today_date = time.strftime("%Y-%m-%d", time.localtime())

    # 哪些 desiderata 使用 0–9 scale（1-based index）
    dense_desiderata: list[int] = 1,2,3,4,5,6,7  # None = 全部 0–3, [2,3] = 对 2，3 desiderata 启用 dense scale


    # Hyperparameters used in the paper
    #base_url: str | None = None
    #log_path: str = "/tmp/tinker-examples/rl_co_scientist"
    #model_name: str = "Qwen/Qwen3-30B-A3B"
    #grader_model_name: str = "Qwen/Qwen3-30B-A3B"

    #batch_size: int = 64     # Number of research goals
    #group_size: int = 8      # G=8 indicated in the paper
    #learning_rate: float = 1e-6
    #clip_eps: float = 0.2       # Clipping epsilon for PPO/GRPO

    #max_length: int = 32768
    #lora_rank: int = 64
    #save_every: int = 20  # 0 = disabled
    #max_tokens: int = 2048
   

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
        - The phrasing should NOT be verbose, and NOT be in past tense, as in "the author’s approach" but rather in present tense, as how you would approach the problem.
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

                For EACH desideratum, assign a satisfaction score from 0 to 9 (integer only).

                Use the following scoring guideline:

                0 — Completely absent.
                The plan does not mention this desideratum at all.
                There is no relevant discussion, reasoning, or attempt to address it.

                1 — Minimally addressed.
                The plan briefly touches on this desideratum, but only in a token or symbolic way.
                The content lacks actionable detail, clarity, or meaningful explanation.

                2 — Very weak.
                The plan attempts to address this desideratum, but the treatment is superficial, unclear, or poorly reasoned.
                Major components required to satisfy the desideratum are missing.

                3 — Weak but identifiable effort.
                The plan includes some relevant elements, but they are underdeveloped, incomplete, or partially incorrect.
                Significant logical gaps or practical flaws remain.

                4 — Below average.
                The plan addresses the desideratum in a structured way, but key steps, justifications, or constraints are inadequately handled.
                The plan would require substantial improvement to be reliable.

                5 — Partially satisfied.
                The plan satisfies this desideratum to a moderate extent.
                Core elements are present, but there are clear weaknesses, missing justifications, or insufficient specificity.

                6 — Reasonably strong.
                The plan addresses the desideratum well, and is generally coherent and implementable.
                However, some important details, evidence, or edge cases are not fully developed.

                7 — Strong but imperfect.
                The plan clearly satisfies the desideratum with convincing reasoning and mostly complete implementation details.
                Only minor weaknesses or limited omissions are present.

                8 — Very strong.
                The plan convincingly and thoroughly satisfies the desideratum.
                Implementation steps, reasoning, and consistency are clear, detailed, and well-structured, with only very small limitations.

                9 — Excellent.
                The plan fully satisfies the desideratum with exceptional clarity, completeness, and rigor.
                The reasoning is precise, implementation steps are concrete and feasible, potential weaknesses are anticipated, and no meaningful flaws are apparent.

                Important:
                - Use the full 0-9 range.
                - Be skeptical, careful, and come up with valid criticisms. Be as strict as possible, while being unbiased and reasonable. 
                - Note that the plan should not just say it satisfies these desiderata, don't be fooled by that. Check carefully WHETHER, HOW and WHY the proposed plan meets each desiderata for this rubric item one by one. 
                - Based on the above analysis, list the satisfaction level for each desiderata. Don't be lazy and assign the same level to all desiderata. Be precise and careful.
                </reasoning>
                <desiderata num=1>
                Repeat for all 7 desiderata:
                    <level>[Satisfaction level for desiderata, put a single integer from 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]</level>
                </desiderata>
            </item>
            
            ... Similarly, for all rubric items...
        </rubric>
    """).strip()
    return prompt


def extract_item_desiderata_levels(xml_text):
    """
    Extract desiderata levels per rubric item.

    Returns:
        np.ndarray of shape (num_items, 7)
    """
    item_blocks = re.findall(
        r"<item\s+num=.*?>.*?</item>",
        xml_text,
        flags=re.DOTALL
    )

    all_items = []

    for item_xml in item_blocks:
        levels = re.findall(r"<level>(\d+)</level>", item_xml)
        levels = [int(l) for l in levels if l.isdigit()]

        if len(levels) < 7:
            levels = levels + [0] * (7 - len(levels))
        levels = levels[:7]

        all_items.append(levels)

    if not all_items:
        return None

    return np.array(all_items)  # shape (num_items, 7)


def compute_mixed_reward(xml_text: str, config: Config) -> float:

    item_levels = extract_item_desiderata_levels(xml_text)
    if item_levels is None:
        return 0.0

    # grader 输出 0–9
    levels = item_levels.astype(float)

    dense_idx = []
    if config.dense_desiderata is not None:
        dense_idx = [d - 1 for d in config.dense_desiderata]

    mapped = np.zeros_like(levels)

    for d in range(levels.shape[1]):

        if d in dense_idx:
            # dense desiderata: 保持 0–9 精度
            mapped[:, d] = levels[:, d] / 9.0

        else:
            # 非 dense：压缩回 4-level 结构
            coarse = np.zeros_like(levels[:, d])

            coarse[(levels[:, d] >= 3) & (levels[:, d] <= 4)] = 1
            coarse[(levels[:, d] >= 5) & (levels[:, d] <= 7)] = 2
            coarse[(levels[:, d] >= 8)] = 3

            mapped[:, d] = coarse / 3.0

    # mean over rubric items
    sample_scores = mapped.mean(axis=0)

    # mean over desiderata
    final_score = sample_scores.mean()

    return float(final_score)



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
# Main
# ============================================================

def main(config: Config):
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

    n_train_batches = len(dataset) // config.batch_size

    # Setup training client
    service_client = create_service_client(
        base_url=config.base_url,
        api_profile=config.api_profile,
    )

  
    # Load for checkpoint, if no checkpoints, start from beginning. 
    # IF in EVAL, checkpoint=0 is eval for baseline, checkpoint=-1 is eval for last checkpoint, checkpoint=n is eval for the nth epoch
    last_checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path)
    if (config.eval_epoch == 0 and config.run_eval) or (not config.run_eval and last_checkpoint is None):
        resume_info = False
    elif config.eval_epoch > 0 and config.run_eval:
        checkpoints = checkpoint_utils.load_checkpoints_file(config.log_path)
        checkpoints_with_key = [c for c in checkpoints if "state_path" in c]
        if checkpoints_with_key:
            resume_info = checkpoints_with_key[config.eval_epoch - 1]
            logger.info(
            f"Found {len(checkpoints_with_key)} valid checkpoints with key '{"state_path"}' in {config.log_path}")
            logger.info(f"Using checkpoint: {checkpoints_with_key[config.eval_epoch - 1]}")
    else:
        resume_info = last_checkpoint
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
        if resume_info["batch"] + 1 // n_train_batches > 0:
            if not is_eval: 
                logger.info(f"Training for epoch: {resume_info['batch'] + 1 // n_train_batches}")
            actual_batch = resume_info["batch"] + 1
            start_batch = 0
        else:
            start_batch = resume_info["batch"] + 1
        if is_eval:
            logger.info(f"Evaluating for Checkpoint {resume_info['batch']}")
        else:
            logger.info(f"Resuming from batch {resume_info['batch']}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0
        actual_batch = 0
    # After this point we have a `training_client` ready. If resuming, it's
    # created from saved state; otherwise it's a fresh LoRA training client.
   
    # Create my grader client 
    grader_client = service_client.create_sampling_client(
        base_model=config.grader_model_name
    )
    # Parameters for sampling from the training client
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature
    )

    # Parameters for optimization step
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    # --------------------------------------------------
    # Prepare sampler
    # --------------------------------------------------
    if is_eval:
        # In eval mode, use the current model weights once
        logger.info("Eval mode: preparing fixed sampler")

        # If resuming, training_client already has loaded weights
        # If not resuming, this is the freshly initialized / final model

        sampling_result = training_client.save_weights_for_sampler(
            name=f"Eval_Checkpoint{actual_batch - 1}"
        ).result()
        sampling_path = sampling_result.path

        sampling_client = service_client.create_sampling_client(
            model_path=sampling_path
        )


    logger.info(f"Training for {n_train_batches} batches")

    #  Main training loop
    for batch_idx in range(start_batch, n_train_batches):
        # Start time of the batch
        t_start = time.time()
        
        if is_eval:
            real_batch = actual_batch
        else:
            real_batch = actual_batch + (batch_idx - start_batch)

        # Save checkpoint periodically
        if config.save_every > 0 and batch_idx % config.save_every == 0 and batch_idx > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{real_batch:06d}_{config.today_date}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": real_batch},
            )

        # Get training batch 
        batch_start = batch_idx * config.batch_size # Since one batch swipe batch_size data, the start batch should be n*batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(dataset))
        batch_rows = dataset.select(range(batch_start, batch_end)) # batch_rows is updated for different batches

        t0 = time.time()
        
        if not is_eval:
            # Train mode: save sampler checkpoint every batch
            sampling_result = training_client.save_weights_for_sampler(
                name=f"{real_batch:06d}"
            ).result()
            sampling_path = sampling_result.path

            sampling_client = service_client.create_sampling_client(
                model_path=sampling_path
            )
        # else:
        #   eval mode: sampling_client already prepared

        policy_futures = []
        policy_prompts_tokens = []


        if is_eval:
            logger.info(f"EVAL MODE: Checkpoint {real_batch -1}, Launching generation for {len(batch_rows)} goals...")
            logger.info(f"EVAL MODE: Batch {batch_idx}, Launching {config.group_size} samples for {config.batch_size} goals")
        else:
            logger.info(f"TRAINING MODE: Batch {real_batch}: Launching generation for {len(batch_rows)} goals...")
            logger.info(f"TRAINING MODE: For batch {real_batch}, Launching {config.group_size} samples for {config.batch_size} goals")

        # --- PHASE 1: LAUNCH POLICY GENERATIONS (ASYNC) ---
        for goal_idx, goal in enumerate(batch_rows["Goal"]): #interate for every research goal in the batch
            #logger.info(f"Batch {real_batch}: launching {config.group_size} samples for goal index {goal_idx}")
            
            prompt_text = build_research_plan_prompt(scenario=goal, examples=None)  # or pass few-shot examples

            convo = [
                {
                    "role": "user", 
                    "content": prompt_text,
                 },
            ]

            #logger.info(f"Batch {real_batch}: Launching generation for {len(batch_rows)} goals...")

            # Turn the prompt from text to tokens
            model_input = renderer.build_generation_prompt(convo)
            # Store prompt tokens for each goal's input in the batch
            policy_prompts_tokens.append(model_input.to_ints())
            
            # Generate batch_size sampling calls for each goal
            policy_futures.append(
                sampling_client.sample(
                    prompt=model_input,
                    num_samples=config.group_size,
                    sampling_params=sampling_params,
                )
            )

        # --- PHASE 2: COLLECT PLANS & LAUNCH GRADERS (ASYNC) ---
        # We process policy results as they arrive and fire off grader requests.

        if is_eval:
            logger.info(f"EVAL MODE: Batch {batch_idx}, collecting plans and launching graders...")
        else:   
            logger.info(f"TRAINING MODE: Batch {real_batch}: collecting plans and launching graders...")
        
        batch_groups_data = [] 
        all_grader_futures = []

        # Iterate through a batch_size
        for i, p_future in enumerate(policy_futures):           
            # Result is the research plan for one research goal
            result = p_future.result()

            # Prepare data for this batch to grade
            goal = batch_rows["Goal"][i]
            rubric = batch_rows["Rubric"][i]
            ref_sol = batch_rows["Reference solution"][i]
            prompt_tokens = policy_prompts_tokens[i]
            
            group_grader_futures = []
            group_samples_info = []

            # Iterate through a group_size
            for group_result in result.sequences:
                proposed_plan = renderers.get_text_content(renderer.parse_response(group_result.tokens)[0])


                # HARD-CODE SOLUTION CLOSURE (required under RL + truncation)
                if "<solution>" in proposed_plan and "</solution>" not in proposed_plan:
                    proposed_plan = proposed_plan.rstrip() + "\n</solution>"
                
                # Build Grader Prompt (CPU operation - Fast)
                grader_prompt_text = build_grader_prompt(
                    scenario=goal,
                    rubric_items=rubric,
                    proposed_plan=proposed_plan,
                    reference_solution=ref_sol 
                )

                # Convert prompt to tokens
                grader_input = renderer.build_generation_prompt([{"role": "user", "content": grader_prompt_text}])

                # CRITICAL OPTIMIZATION: Launch Grader and DO NOT WAIT.
                # Store the future and move to the next item immediately.
                g_future = grader_client.sample(
                    grader_input,
                    num_samples=1,
                    sampling_params=tinker.types.SamplingParams(max_tokens=config.grader_max_tokens, temperature=config.grader_temperature)
                )
                
                group_grader_futures.append(g_future)
                group_samples_info.append({
                    "tokens": group_result.tokens,
                    "logprobs": group_result.logprobs,
                    "text": proposed_plan, # Store text for format check
                })

            batch_groups_data.append({
                "prompt_tokens": prompt_tokens,
                "samples_info": group_samples_info,
                "rubric_len": len(rubric)
            })
            all_grader_futures.append(group_grader_futures)


        # --- PHASE 3: COLLECT GRADES & COMPUTE ADVANTAGES ---
        if is_eval:
            logger.info(f"EVAL MODE: Batch {batch_idx}: Waiting for grading results...")
        else:
            logger.info(f"TRAINING MODE: Batch {real_batch}: Waiting for grading results...")
        
        training_datums = []
        batch_rewards = []
        batch_sample_rewards = []
        batch_logs_to_save = []
        dropped_samples = 0
        
        # Batch-level diagnostics containers 
        batch_word_counts = []
        batch_format_penalties = []
        batch_rubric_scores = []

        batch_advantages = []   # collect all advantages across groups
        num_total_samples = 0
        num_valid_samples = 0


        # Iterate over batch_size of groups
        for group_idx, group_futures in enumerate(all_grader_futures):
            
            group_data = batch_groups_data[group_idx]
            group_rewards = []
            valid_samples = []

            # Iterate over group_size to get Rewards for a single group
            for j, each_group_future in enumerate(group_futures):

                num_total_samples += 1

                # Now we wait. Since they all started moments ago, 
                # they should all finish roughly at the same time.
                each_group_result = each_group_future.result()
                    
                # Parse Grader Output
                parsed_message, _ = renderer.parse_response(each_group_result.sequences[0].tokens)
                xml_text = renderers.get_text_content(parsed_message)

                # Compute Score
                rubric_score = compute_mixed_reward(xml_text, config)

                plan_text = group_data["samples_info"][j]["text"]
                word_count = len(plan_text.strip().split())

                # Format Penalty 
                # "We penalize the model if the content within <solution> tags exceeds 750 words or if tags are missing"
                # "reward = (satisfied/total) - 1{format penalty}"
                is_compliant = check_format_compliance(plan_text, config.max_word_count)

                # Compute format penalty, should be 1.0 if not compliant else 0.0
                # Using a linear penalty based on word count excess
                excess = max(0, word_count - config.max_word_count)
                format_penalty = 0.0 if is_compliant else 0.2 + 0.0005 * excess
                
                # gaussian length bonus centered at 600 words
            
                length_bonus = np.exp(-((word_count - config.target_word_count) / config.scale_length_bonus) ** 2)  # in (0,1]
                final_reward = rubric_score + config.scaling_factor * length_bonus - format_penalty

                # Record raw sample statistics 
                batch_word_counts.append(word_count)
                batch_format_penalties.append(format_penalty)
                batch_rubric_scores.append(rubric_score)


                # Case 1: too short
                if word_count < config.min_words:
                    logger.debug(f"Dropping degenerate sample (too short): {word_count} words")
                    dropped_samples += 1
                    continue

                # Case 2: only structural tags
                if plan_text.strip() in {"<think>", "</think>", "<solution>", "</solution>"}:
                    logger.debug("Dropping degenerate sample (only tags)")
                    dropped_samples += 1
                    continue
 
                # Only being used in a group level
                group_rewards.append(final_reward)
                # Being used in a batch level, for diagnostics
                batch_sample_rewards.append(final_reward)

                # Count for valid samples
                valid_samples.append({
                    "sample_info": group_data["samples_info"][j],
                    "reward": final_reward
                })

                num_valid_samples += 1

                # Log generation info for analysis
                batch_logs_to_save.append({
                    "batch_idx": real_batch,
                    "group_idx": group_idx,
                    "sample_idx": j,
                    
                    #Core outputs
                    "policy_output": plan_text,
                    "grader_output": xml_text,

                    # Reward components
                    "rubric_score": rubric_score,
                    "format_penalty": format_penalty,   
                    "final_reward": final_reward,

                    # Stats
                    "word_count": word_count,
                    "is_compliant": is_compliant,
                })

            # GRPO Advantage Calculation
            if not group_rewards: continue
            
            mean_reward = np.mean(group_rewards)
            advantages = [(r - mean_reward) for r in group_rewards]
            batch_rewards.append(mean_reward)

            # Collect advantages for diagnostics 
            batch_advantages.extend(advantages)


            # Skip if no learning signal (all rewards identical)
            if all(a == 0.0 for a in advantages): continue

            # Skip if not enough valid samples
            if len(valid_samples) < 2:
                logger.debug("Skipping group: not enough valid samples")
                continue
            if not is_eval:
                # Iterate over group_size to Create Training Datums
                for k, sample in enumerate(valid_samples):

                    sample_info = sample["sample_info"]
                    advantage = advantages[k]

                    # Reconstruct full sequence
                    # Ensure tokens are simple integers
                    prompt_tokens = [int(t) for t in group_data["prompt_tokens"]]
                    grader_tokens = [int(t) for t in sample_info["tokens"]]
                    full_seq = prompt_tokens + grader_tokens
                    
                    ob_len = len(prompt_tokens) -1 # -1 because we usually don't mask the very last prompt token in some configs, but standard is len(prompt)
                    
                    input_tokens = full_seq[:-1]
                    target_tokens = full_seq[1:]
                    
                    # Pad logprobs/advantages for the prompt portion
                    # Note: sample_info["logprobs"] corresponds only to generated tokens
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

        batch_summary = {
                "batch_idx": real_batch,

                # reward
                "rubric/sample_mean_all": float(np.mean(batch_rubric_scores)) if batch_rubric_scores else 0.0,
                "rubric/sample_std_all": float(np.std(batch_rubric_scores)) if batch_rubric_scores else 0.0,
                "reward/sample_mean_valid": float(np.mean(batch_sample_rewards)) if batch_sample_rewards else 0.0,
                "reward/sample_std_valid": float(np.std(batch_sample_rewards)) if batch_sample_rewards else 0.0,
                "reward/group_mean_valid": float(np.mean(batch_rewards)) if batch_rewards else 0.0,
                "reward/group_std_valid": float(np.std(batch_rewards)) if batch_rewards else 0.0,

                # advantage (optimizer view, without padding with zeros)
                "advantage_mean": float(np.mean(batch_advantages)) if batch_advantages else 0.0,
                "advantage_std": float(np.std(batch_advantages)) if batch_advantages else 0.0,

                # length
                "length_mean": float(np.mean(batch_word_counts)) if batch_word_counts else 0.0,
                "length_p90": float(np.percentile(batch_word_counts, 90)) if batch_word_counts else 0.0,

                # format
                "format_rate": float(np.mean(batch_format_penalties)) if batch_format_penalties else 0.0,

                # samples
                "samples/generated": num_total_samples,
                "samples/valid": num_valid_samples,
            }
        
        # Write batch logs to file
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
        
        # --- EVAL MODE: skip optimization ---
        if is_eval:
            logger.info("EVAL MODE: skipping optimization step.")
            continue

        # --- PHASE 4: OPTIMIZATION STEP ---
        if not training_datums:
            logger.warning("No valid datums this batch. Skipping optimization.")
            continue

        try:
            fwd_bwd_future = training_client.forward_backward(
                training_datums,
                loss_fn="ppo",
                loss_fn_config={
                    "clip_low_threshold": 1-config.clip_eps, 
                    "clip_high_threshold": 1+config.clip_eps,
                    #"kl_coeff": 0.0, # Explicitly disable KL penalty as per Paper Appendix A.2
                }
            )
            optim_step_future = training_client.optim_step(adam_params)

            t0 = time.time()
            _fwd_bwd_result = fwd_bwd_future.result()
            logger.info(f"Forward/Backward took {time.time()-t0:.2f}s")

            t1 = time.time()
            _optim_result = optim_step_future.result()
            logger.info(f"Optim step took {time.time()-t1:.2f}s")
        except Exception as e:
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
    if not is_eval:
        checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{real_batch:06d}_final_{config.today_date}",
                log_path=config.log_path,
                kind="both",
                loop_state={"batch": real_batch},
            )    
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
