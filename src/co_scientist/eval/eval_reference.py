"""
Evaluate the reference (expert) solutions from the dataset using the same grader.

This establishes the TRUE ceiling: what score does the grader give to expert-written
plans? If reference plans score ~0.75, then a model scoring 0.69 is nearly optimal.
If they score 0.95, there's massive headroom.

No model checkpoint is needed — we grade the dataset's "Reference solution" directly.

Usage:
  python eval_reference.py
  python eval_reference.py eval_output_path=/path/to/output
  python eval_reference.py batch_size=32
"""

import logging
import time
import re
import textwrap
import numpy as np
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
from co_scientist.shared.api_profiles import create_service_client
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from datasets import load_dataset

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    eval_output_path: str = "/home/silas/co-scientist-project/runs/2026/2/withA1,A2/2(ml)/reference_plan_score"

    # ---- Model / API ----
    base_url: str | None = None
    api_profile: str | None = None
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"

    # ---- Dataset ----
    ml_data: bool = True
    arxiv_data: bool = False
    pubmed_data: bool = False
    use_train_split: bool = False  # grade train split references too

    # ---- Grading ----
    batch_size: int = 64
    grader_max_tokens: int = 8192
    grader_temperature: float = 0.0


# ============================================================
# Grader prompt (identical to eval_only.py / best_ver.py)
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


def extract_desiderata_levels(xml_text: str) -> list[list[int]]:
    """Return list of [7 levels] per rubric item, for per-desideratum analysis."""
    item_blocks = re.findall(r"<item\s+num=.*?>.*?</item>", xml_text, flags=re.DOTALL)
    result = []
    for item_xml in item_blocks:
        levels = re.findall(r"<level>(\d+)</level>", item_xml)
        levels = [int(l) for l in levels if l.isdigit()]
        if len(levels) < 7:
            levels = levels + [0] * (7 - len(levels))
        result.append(levels[:7])
    return result


# ============================================================
# Main
# ============================================================

def main(config: Config):
    # --- Output path ---
    output_dir = config.eval_output_path or os.path.join(os.getcwd(), "eval_reference")
    os.makedirs(output_dir, exist_ok=True)

    ml_logger = ml_log.setup_logging(
        log_dir=output_dir,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    # --- Load dataset ---
    split = "train" if config.use_train_split else "test"
    logger.info(f"Loading dataset ({split} split)...")
    if config.ml_data:
        data = load_dataset("facebook/research-plan-gen", "ml")
    elif config.arxiv_data:
        data = load_dataset("facebook/research-plan-gen", "arxiv")
    elif config.pubmed_data:
        data = load_dataset("facebook/research-plan-gen", "pubmed")
    assert isinstance(data, datasets.DatasetDict)
    dataset = data[split]

    n_batches = len(dataset) // config.batch_size
    remainder = len(dataset) % config.batch_size
    if remainder > 0:
        n_batches += 1  # include partial last batch
    logger.info(f"{split} set: {len(dataset)} examples → {n_batches} batches of {config.batch_size}")

    # --- Tokenizer / renderer (for grader only) ---
    tokenizer = get_tokenizer(config.grader_model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.grader_model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # --- Grader client ---
    service_client = create_service_client(
        base_url=config.base_url,
        api_profile=config.api_profile,
    )
    grader_client = service_client.create_sampling_client(
        base_model=config.grader_model_name
    )

    # --- Output files ---
    logs_path = os.path.join(output_dir, f"eval_reference_{split}_logs.jsonl")
    summary_path = os.path.join(output_dir, f"eval_reference_{split}_summary.jsonl")

    logger.info(f"Starting reference eval | {n_batches} batches | output → {output_dir}")

    all_rubric_scores: list[float] = []
    all_word_counts: list[int] = []

    # ============================================================
    # Eval loop
    # ============================================================
    for batch_idx in range(n_batches):
        t_start = time.time()
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(dataset))
        batch_rows = dataset.select(range(batch_start, batch_end))

        logger.info(
            f"Eval batch {batch_idx + 1}/{n_batches} "
            f"(examples {batch_start}–{batch_end - 1})"
        )

        # --- Launch graders for all reference solutions in this batch ---
        grader_futures = []
        ref_plans = []
        word_counts = []

        for i in range(len(batch_rows)):
            goal = batch_rows["Goal"][i]
            rubric = batch_rows["Rubric"][i]
            ref_sol = batch_rows["Reference solution"][i]

            ref_plans.append(ref_sol)
            word_counts.append(len(ref_sol.strip().split()))

            # Grade the reference solution itself — pass ref_sol as BOTH
            # the proposed plan AND the reference (so the grader sees the
            # reference context, matching how model outputs are graded)
            grader_prompt_text = build_grader_prompt(
                scenario=goal,
                rubric_items=rubric,
                proposed_plan=ref_sol,
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
            grader_futures.append(g_future)

        # --- Collect grades ---
        batch_rubric_scores: list[float] = []
        batch_logs: list[dict] = []

        for i, g_future in enumerate(grader_futures):
            g_result = g_future.result()
            parsed_msg, _ = renderer.parse_response(g_result.sequences[0].tokens)
            xml_text = renderers.get_text_content(parsed_msg)

            rubric_score = compute_rubric_reward_from_xml(xml_text)
            desd_levels = extract_desiderata_levels(xml_text)

            # Per-desideratum mean across items
            if desd_levels:
                desd_means = np.array(desd_levels).mean(axis=0).tolist()
            else:
                desd_means = [0.0] * 7

            batch_rubric_scores.append(rubric_score)

            batch_logs.append({
                "eval_batch_idx": batch_idx,
                "example_idx": batch_start + i,
                "goal": batch_rows["Goal"][i],
                "reference_solution": ref_plans[i],
                "grader_output": xml_text,
                "rubric_score": rubric_score,
                "word_count": word_counts[i],
                "desiderata_mean_levels": desd_means,
            })

        all_rubric_scores.extend(batch_rubric_scores)
        all_word_counts.extend(word_counts[: len(batch_rubric_scores)])

        # --- Batch summary ---
        batch_summary = {
            "eval_batch_idx": batch_idx,
            "rubric/mean": float(np.mean(batch_rubric_scores)) if batch_rubric_scores else 0.0,
            "rubric/std": float(np.std(batch_rubric_scores)) if batch_rubric_scores else 0.0,
            "rubric/min": float(np.min(batch_rubric_scores)) if batch_rubric_scores else 0.0,
            "rubric/max": float(np.max(batch_rubric_scores)) if batch_rubric_scores else 0.0,
            "word_count/mean": float(np.mean(word_counts[: len(batch_rubric_scores)])),
            "n_examples": len(batch_rubric_scores),
            "time_s": round(time.time() - t_start, 1),
        }
        logger.info(
            f"  rubric={batch_summary['rubric/mean']:.4f} ± {batch_summary['rubric/std']:.4f}  "
            f"min={batch_summary['rubric/min']:.4f}  max={batch_summary['rubric/max']:.4f}  "
            f"words={batch_summary['word_count/mean']:.0f}  n={batch_summary['n_examples']}"
        )

        with open(logs_path, "a") as f:
            for item in batch_logs:
                f.write(json.dumps(item) + "\n")
        with open(summary_path, "a") as f:
            f.write(json.dumps(batch_summary) + "\n")

    # --- Final aggregate ---
    aggregate = {
        "split": split,
        "n_examples": len(all_rubric_scores),
        "rubric/mean": float(np.mean(all_rubric_scores)) if all_rubric_scores else 0.0,
        "rubric/std": float(np.std(all_rubric_scores)) if all_rubric_scores else 0.0,
        "rubric/min": float(np.min(all_rubric_scores)) if all_rubric_scores else 0.0,
        "rubric/max": float(np.max(all_rubric_scores)) if all_rubric_scores else 0.0,
        "rubric/median": float(np.median(all_rubric_scores)) if all_rubric_scores else 0.0,
        "word_count/mean": float(np.mean(all_word_counts)) if all_word_counts else 0.0,
    }

    aggregate_path = os.path.join(output_dir, f"eval_reference_{split}_aggregate.json")
    with open(aggregate_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"REFERENCE EVAL COMPLETE ({split} split)")
    logger.info(f"  rubric mean   : {aggregate['rubric/mean']:.4f}")
    logger.info(f"  rubric std    : {aggregate['rubric/std']:.4f}")
    logger.info(f"  rubric median : {aggregate['rubric/median']:.4f}")
    logger.info(f"  rubric range  : [{aggregate['rubric/min']:.4f}, {aggregate['rubric/max']:.4f}]")
    logger.info(f"  word count    : {aggregate['word_count/mean']:.0f}")
    logger.info(f"  n examples    : {aggregate['n_examples']}")
    logger.info(f"  results       : {output_dir}")
    logger.info("=" * 60)

    ml_logger.close()


if __name__ == "__main__":
    chz.nested_entrypoint(main)
