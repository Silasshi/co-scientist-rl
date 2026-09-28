"""
Standalone evaluation script for co-scientist research plan models.

Supports loading any saved checkpoint by batch number (not just epoch boundaries).
Writes results to eval_output_path/eval_b{checkpoint_batch}/.

Usage examples:
  # Evaluate the last checkpoint in a run
  python eval_only.py checkpoint_run_path=/path/to/run

  # Evaluate a specific batch checkpoint (e.g. batch 135)
  python eval_only.py checkpoint_run_path=/path/to/run checkpoint_batch=135

  # Evaluate the base model (no checkpoint)
  python eval_only.py checkpoint_run_path=none checkpoint_batch=0

  # Write results to a custom location
  python eval_only.py checkpoint_run_path=/path/to/run eval_output_path=/path/to/out
"""

import logging
import time
import re
import textwrap
import numpy as np
from concurrent.futures import TimeoutError
import json
import os
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2]
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
    # ---- Checkpoint selection ----
    # Path to the run directory that contains the checkpoint to evaluate.
    # Set to "none" to evaluate the base model with no fine-tuning.
    checkpoint_run_path: str = "/home/silas/co-scientist-project/runs/2026/3/rubric_dropout/13"

    # Which batch checkpoint to load.
    #   -1  → last available checkpoint (default)
    #    0  → no checkpoint (base model)
    #    N  → the checkpoint saved at batch N (exact match on loop_state["batch"])
    checkpoint_batch: int = -1

    # ---- Output ----
    # Directory where eval results are written.
    # Defaults to <checkpoint_run_path>/eval/ when left as empty string.
    eval_output_path: str = ""

    # ---- Model / API ----
    base_url: str | None = None
    api_profile: str | None = None
    model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"

    # ---- Dataset ----
    ml_data: bool = True
    arxiv_data: bool = False
    pubmed_data: bool = False

    # ---- Sampling ----
    batch_size: int = 64
    group_size: int = 8
    max_tokens: int = 2048
    grader_max_tokens: int = 8192
    temperature: float = 1.0
    grader_temperature: float = 0.0

    # ---- Reward / format ----
    max_word_count: int = 750
    target_word_count: int = 600
    scale_length_bonus: float = 120.0
    scaling_factor: float = 0.08
    min_words: int = 30

    # ---- LoRA rank (needed to reconstruct the training client) ----
    lora_rank: int = 64


# ============================================================
# Prompt builders  (identical to best_ver.py)
# ============================================================

def build_research_plan_prompt(scenario: str, examples: list[dict] | None = None) -> str:
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


# ============================================================
# Reward helpers  (identical to best_ver.py)
# ============================================================

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


# ============================================================
# Checkpoint loading
# ============================================================

def resolve_checkpoint(checkpoint_run_path: str, checkpoint_batch: int):
    """
    Returns (resume_info | None, resolved_batch_label: str).

    resume_info is a checkpoint dict with at least "state_path" and "batch",
    or None if checkpoint_batch == 0 (base model).
    """
    if checkpoint_run_path.lower() == "none" or checkpoint_batch == 0:
        logger.info("No checkpoint requested — evaluating base model.")
        return None, "base"

    checkpoints = checkpoint_utils.load_checkpoints_file(checkpoint_run_path)
    checkpoints_with_state = [c for c in checkpoints if "state_path" in c]

    if not checkpoints_with_state:
        raise RuntimeError(
            f"No checkpoints with 'state_path' found in {checkpoint_run_path}. "
            "Available checkpoint keys: " + str([list(c.keys()) for c in checkpoints])
        )

    if checkpoint_batch == -1:
        # Use the last available checkpoint
        chosen = checkpoints_with_state[-1]
        logger.info(f"Using last checkpoint: batch {chosen['batch']}")
        return chosen, str(chosen["batch"])

    # Exact match on batch number
    matches = [c for c in checkpoints_with_state if c.get("batch") == checkpoint_batch]
    if not matches:
        available = sorted(c.get("batch", "?") for c in checkpoints_with_state)
        raise ValueError(
            f"No checkpoint found for batch={checkpoint_batch} in {checkpoint_run_path}.\n"
            f"Available batch numbers: {available}"
        )
    chosen = matches[0]
    logger.info(f"Using checkpoint at batch {chosen['batch']}: {chosen['state_path']}")
    return chosen, str(chosen["batch"])


# ============================================================
# Main
# ============================================================

def main(config: Config):
    # --- Resolve output path ---
    if config.eval_output_path:
        output_dir = config.eval_output_path
    elif config.checkpoint_run_path.lower() != "none":
        output_dir = os.path.join(config.checkpoint_run_path, "eval")
    else:
        output_dir = os.path.join(os.getcwd(), "eval_base_model")
    os.makedirs(output_dir, exist_ok=True)

    ml_logger = ml_log.setup_logging(
        log_dir=output_dir,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    # --- Load dataset (test split) ---
    logger.info("Loading dataset (test split)...")
    if config.ml_data:
        data = load_dataset("facebook/research-plan-gen", "ml")
    elif config.arxiv_data:
        data = load_dataset("facebook/research-plan-gen", "arxiv")
    elif config.pubmed_data:
        data = load_dataset("facebook/research-plan-gen", "pubmed")
    assert isinstance(data, datasets.DatasetDict)
    dataset = data["test"]

    n_eval_batches = len(dataset) // config.batch_size
    logger.info(f"Test set: {len(dataset)} examples → {n_eval_batches} batches of {config.batch_size}")

    # --- Tokenizer / renderer ---
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # --- Service client ---
    service_client = create_service_client(
        base_url=config.base_url,
        api_profile=config.api_profile,
    )

    # --- Checkpoint ---
    resume_info, batch_label = resolve_checkpoint(
        config.checkpoint_run_path, config.checkpoint_batch
    )

    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
        loaded_batch = resume_info["batch"]
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        loaded_batch = -1

    # --- Prepare fixed sampler (weights frozen for entire eval) ---
    logger.info(f"Saving weights for eval sampler (checkpoint=b{batch_label})...")
    sampling_result = training_client.save_weights_for_sampler(
        name=f"eval_b{batch_label}"
    ).result()
    sampling_client = service_client.create_sampling_client(
        model_path=sampling_result.path
    )

    grader_client = service_client.create_sampling_client(
        base_model=config.grader_model_name
    )

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )

    # --- Per-eval-run output files ---
    logs_path = os.path.join(output_dir, f"eval_b{batch_label}_logs.jsonl")
    summary_path = os.path.join(output_dir, f"eval_b{batch_label}_summary.jsonl")

    logger.info(
        f"Starting eval | checkpoint=b{batch_label} | "
        f"{n_eval_batches} batches | output → {output_dir}"
    )

    all_rubric_scores: list[float] = []
    all_rewards: list[float] = []
    all_format_penalties: list[float] = []

    # ============================================================
    # Eval loop
    # ============================================================
    for batch_idx in range(n_eval_batches):
        t_start = time.time()
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(dataset))
        batch_rows = dataset.select(range(batch_start, batch_end))

        logger.info(
            f"Eval batch {batch_idx + 1}/{n_eval_batches} "
            f"(goals {batch_start}–{batch_end - 1})"
        )

        # --- Phase 1: launch policy generations ---
        policy_futures = []
        policy_prompts_tokens = []

        for goal in batch_rows["Goal"]:
            prompt_text = build_research_plan_prompt(scenario=goal)
            convo = [{"role": "user", "content": prompt_text}]
            model_input = renderer.build_generation_prompt(convo)
            policy_prompts_tokens.append(model_input.to_ints())
            policy_futures.append(
                sampling_client.sample(
                    prompt=model_input,
                    num_samples=config.group_size,
                    sampling_params=sampling_params,
                )
            )

        # --- Phase 2: collect plans & launch graders ---
        batch_groups_data = []
        all_grader_futures = []

        for i, p_future in enumerate(policy_futures):
            result = p_future.result()
            goal = batch_rows["Goal"][i]
            rubric = batch_rows["Rubric"][i]
            ref_sol = batch_rows["Reference solution"][i]

            group_grader_futures = []
            group_samples_info = []

            for group_result in result.sequences:
                proposed_plan = renderers.get_text_content(
                    renderer.parse_response(group_result.tokens)[0]
                )
                if "<solution>" in proposed_plan and "</solution>" not in proposed_plan:
                    proposed_plan = proposed_plan.rstrip() + "\n</solution>"

                grader_prompt_text = build_grader_prompt(
                    scenario=goal,
                    rubric_items=rubric,
                    proposed_plan=proposed_plan,
                    reference_solution=ref_sol,
                )
                grader_input = renderer.build_generation_prompt(
                    [{"role": "user", "content": grader_prompt_text}]
                )
                g_future = grader_client.sample(
                    grader_input,
                    num_samples=1,
                    sampling_params=tinker.types.SamplingParams(
                        max_tokens=config.grader_max_tokens,
                        temperature=config.grader_temperature,
                    ),
                )
                group_grader_futures.append(g_future)
                group_samples_info.append({"text": proposed_plan})

            batch_groups_data.append({
                "goal": goal,
                "samples_info": group_samples_info,
            })
            all_grader_futures.append(group_grader_futures)

        # --- Phase 3: collect grades & compute metrics ---
        batch_rubric_scores: list[float] = []
        batch_rewards: list[float] = []
        batch_format_penalties: list[float] = []
        batch_logs: list[dict] = []
        dropped = 0
        num_total = 0
        num_valid = 0

        for group_idx, group_futures in enumerate(all_grader_futures):
            group_data = batch_groups_data[group_idx]

            for j, g_future in enumerate(group_futures):
                num_total += 1
                g_result = g_future.result()
                parsed_msg, _ = renderer.parse_response(g_result.sequences[0].tokens)
                xml_text = renderers.get_text_content(parsed_msg)

                rubric_score = compute_rubric_reward_from_xml(xml_text)
                plan_text = group_data["samples_info"][j]["text"]

                # Extract solution text for word count / format penalty
                sol_match = re.search(
                    r"<solution>\s*(.*?)\s*</solution>", plan_text,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                solution_text = sol_match.group(1) if sol_match else plan_text.strip()
                word_count = len(solution_text.split())

                is_compliant = check_format_compliance(plan_text, config.max_word_count)
                excess = max(0, word_count - config.max_word_count)
                format_penalty = 0.0 if is_compliant else 0.2 + 0.0005 * excess
                length_bonus = np.exp(
                    -((word_count - config.target_word_count) / config.scale_length_bonus) ** 2
                )
                final_reward = rubric_score + config.scaling_factor * length_bonus - format_penalty

                batch_rubric_scores.append(rubric_score)
                batch_format_penalties.append(format_penalty)

                if word_count < config.min_words:
                    dropped += 1
                    continue
                if plan_text.strip() in {"<think>", "</think>", "<solution>", "</solution>"}:
                    dropped += 1
                    continue

                batch_rewards.append(final_reward)
                num_valid += 1

                batch_logs.append({
                    "eval_batch_idx": batch_idx,
                    "checkpoint_batch": batch_label,
                    "group_idx": group_idx,
                    "sample_idx": j,
                    "goal": group_data["goal"],
                    "policy_output": plan_text,
                    "grader_output": xml_text,
                    "rubric_score": rubric_score,
                    "format_penalty": format_penalty,
                    "final_reward": final_reward,
                    "word_count": word_count,
                    "is_compliant": is_compliant,
                })

        # Accumulate across batches
        all_rubric_scores.extend(batch_rubric_scores)
        all_rewards.extend(batch_rewards)
        all_format_penalties.extend(batch_format_penalties)

        # Per-batch summary
        batch_summary = {
            "eval_batch_idx": batch_idx,
            "checkpoint_batch": batch_label,
            "rubric/mean": float(np.mean(batch_rubric_scores)) if batch_rubric_scores else 0.0,
            "rubric/std": float(np.std(batch_rubric_scores)) if batch_rubric_scores else 0.0,
            "reward/mean": float(np.mean(batch_rewards)) if batch_rewards else 0.0,
            "reward/std": float(np.std(batch_rewards)) if batch_rewards else 0.0,
            "format_penalty/mean": float(np.mean(batch_format_penalties)) if batch_format_penalties else 0.0,
            "samples/total": num_total,
            "samples/valid": num_valid,
            "samples/dropped": dropped,
            "time_s": round(time.time() - t_start, 1),
        }
        logger.info(
            f"  rubric={batch_summary['rubric/mean']:.4f}  "
            f"reward={batch_summary['reward/mean']:.4f}  "
            f"format_pen={batch_summary['format_penalty/mean']:.4f}  "
            f"valid={num_valid}/{num_total}"
        )

        # Write batch logs
        with open(logs_path, "a") as f:
            for item in batch_logs:
                f.write(json.dumps(item) + "\n")

        with open(summary_path, "a") as f:
            f.write(json.dumps(batch_summary) + "\n")

    # --- Final aggregate summary ---
    aggregate = {
        "checkpoint_batch": batch_label,
        "loaded_batch": loaded_batch,
        "n_eval_batches": n_eval_batches,
        "n_samples_total": len(all_rubric_scores),
        "n_samples_valid": len(all_rewards),
        "rubric/mean": float(np.mean(all_rubric_scores)) if all_rubric_scores else 0.0,
        "rubric/std": float(np.std(all_rubric_scores)) if all_rubric_scores else 0.0,
        "reward/mean": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "reward/std": float(np.std(all_rewards)) if all_rewards else 0.0,
        "format_penalty/mean": float(np.mean(all_format_penalties)) if all_format_penalties else 0.0,
    }

    aggregate_path = os.path.join(output_dir, f"eval_b{batch_label}_aggregate.json")
    with open(aggregate_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"EVAL COMPLETE  checkpoint=b{batch_label}")
    logger.info(f"  rubric mean : {aggregate['rubric/mean']:.4f}")
    logger.info(f"  reward mean : {aggregate['reward/mean']:.4f}")
    logger.info(f"  format pen  : {aggregate['format_penalty/mean']:.4f}")
    logger.info(f"  valid/total : {aggregate['n_samples_valid']}/{aggregate['n_samples_total']}")
    logger.info(f"  results     : {output_dir}")
    logger.info("=" * 60)

    ml_logger.close()


if __name__ == "__main__":
    chz.nested_entrypoint(main)
