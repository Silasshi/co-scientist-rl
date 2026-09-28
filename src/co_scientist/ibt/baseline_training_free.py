"""
Baseline A: Training-Free Iterative Prompting.

Same iterative loop as IBT (K turns per goal, grader produces hint, hint feeds
into next turn), but NO weight updates.  The model stays at base 4B weights
throughout.  Only improvement comes from accumulated hints in the prompt.

This tests: does RL training help, or is in-context feedback enough?

Usage:
  python baseline_training_free.py num_turns=5 num_goals=25 api_profile=NEW
"""

import json
import logging
import os
import sys
import textwrap
import time
from pathlib import Path

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

# Reuse prompts and scoring from IBT
from co_scientist.ibt.train_ibt import (
    build_grader_with_hint_prompt,
    build_plan_prompt,
    compute_rubric_score,
    compute_reward,
    extract_hint,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    api_profile: str | None = "NEW"
    log_path: str = "/home/silas/co-scientist-project/runs/2026/4/ibt/1"

    policy_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    grader_model: str = "Qwen/Qwen3-30B-A3B"

    ml_data: bool = True
    arxiv_data: bool = False
    pubmed_data: bool = False

    num_goals: int = 25
    start_goal: int = 0
    num_turns: int = 5

    # True = accumulate all hints; False = only latest hint (same as IBT)
    # Set both to False for regenerate baseline (no hints at all)
    accumulate_hints: bool = True
    use_hints: bool = True  # False = regenerate baseline (no hints in prompt)
    max_hints: int = 0      # 0 = unlimited; >0 = keep only last N hints (accumulation ablation)

    max_tokens: int = 2048
    temperature: float = 1.0
    grader_max_tokens: int = 8192
    grader_temperature: float = 0.0

    max_word_count: int = 750
    target_word_count: int = 600
    scale_length_bonus: float = 140.0
    scaling_factor: float = 0.08
    min_words: int = 30


# ============================================================
# Prompt with accumulated hints
# ============================================================

def build_plan_prompt_accumulated(scenario: str, hints: list[str]) -> str:
    """Build prompt with all accumulated hints from previous turns."""
    prompt = textwrap.dedent(f"""\
        I will provide you a research scenario. You have to provide me a concise yet thoughtful research plan with all details needed to execute it.

        Here is the research scenario.
        Scenario: {scenario}
    """).strip()

    if hints:
        prompt += "\n\n# Reviewer Feedback from Previous Iterations\n"
        prompt += "A reviewer has provided the following feedback on previous versions of the plan. "
        prompt += "Please address ALL of these points in your improved plan:\n\n"
        for i, h in enumerate(hints):
            prompt += f"## Feedback Round {i + 1}\n{h}\n\n"

    prompt += textwrap.dedent("""
        # Instructions
        First, come up with a detailed research plan to address the scenario based on the following overall solution guidelines:
        - The plan should address the goals of the scenario, and account for all constraints and confounders.
        - Do NOT just say WHAT you will do. Explain HOW you will do it and WHY it is needed. Provide clear explanation and justification for each proposed step. The solution inside <solution></solution> tags should be readable for humans, and not in XML itself.
        - The phrasing should NOT be verbose, and NOT be in past tense, as in "the author's approach" but rather in present tense, as how you would approach the problem.
        - Do not claim to have done any experiments or have results, just provide the plan.
        - Do not add self-proclaimed praises of your solution.

        Then, return the following nested XML block as your output (always close opened XML tags):
        <think>
        Here is your thinking process. You can use this to reason about the problem before giving the final solution. But only the content within <solution></solution> tags will be judged so make sure to include (potentially repeat) all details in it.
        </think>
        <solution>
        Here is your final research plan. Make sure it is complete and self-contained. And it should not exceed 750 words.
        ... Your detailed research plan goes here ...
        </solution>
    """)

    return prompt


# ============================================================
# Main
# ============================================================

def main(config: Config):
    if not config.use_hints:
        variant = "baseline_regenerate"
    elif config.accumulate_hints:
        suffix = f"_top{config.max_hints}" if config.max_hints > 0 else ""
        variant = f"baseline_tf_accumulated{suffix}"
    else:
        variant = "baseline_tf_latest"
    run_dir = os.path.join(config.log_path, variant)
    os.makedirs(run_dir, exist_ok=True)
    train_dir = os.path.join(run_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    ml_log.setup_logging(
        log_dir=run_dir, wandb_project=None, wandb_name=None,
        config=config, do_configure_logging_module=True,
    )

    if not config.use_hints:
        hint_mode = "no hints (regenerate)"
    elif config.accumulate_hints:
        hint_mode = "accumulated"
    else:
        hint_mode = "latest only"
    logger.info(f"Baseline: Training-Free ({hint_mode})")
    logger.info(f"Policy: {config.policy_model} (NO training)")
    logger.info(f"Grader: {config.grader_model}")
    logger.info(f"Goals: {config.num_goals}, Turns/goal: {config.num_turns}")

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({
            "baseline": "training_free",
            "policy_model": config.policy_model,
            "grader_model": config.grader_model,
            "num_goals": config.num_goals,
            "num_turns": config.num_turns,
            "accumulate_hints": config.accumulate_hints,
            "training": False,
        }, f, indent=2)

    # ── Tokenizer & renderer ──
    tokenizer = get_tokenizer(config.policy_model)
    renderer_name = model_info.get_recommended_renderer_name(config.policy_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    grader_tokenizer = get_tokenizer(config.grader_model)
    grader_renderer_name = model_info.get_recommended_renderer_name(config.grader_model)
    grader_renderer = renderers.get_renderer(grader_renderer_name, grader_tokenizer)

    # ── Dataset ──
    if config.ml_data:
        data = load_dataset("facebook/research-plan-gen", "ml")
    elif config.arxiv_data:
        data = load_dataset("facebook/research-plan-gen", "arxiv")
    elif config.pubmed_data:
        data = load_dataset("facebook/research-plan-gen", "pubmed")
    assert isinstance(data, datasets.DatasetDict)
    dataset = data["train"]
    num_goals = min(config.num_goals, len(dataset))

    # ── Clients (no training client — just sampling) ──
    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )

    sampling_client = service_client.create_sampling_client(
        base_model=config.policy_model
    )

    grader_client = service_client.create_sampling_client(
        base_model=config.grader_model
    )

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )

    grader_sampling_params = tinker.types.SamplingParams(
        max_tokens=config.grader_max_tokens,
        stop=grader_renderer.get_stop_sequences(),
        temperature=config.grader_temperature,
    )

    # ── Log files ──
    f_summary = open(os.path.join(train_dir, "batch_summary.jsonl"), "a")
    f_logs = open(os.path.join(train_dir, "training_logs.jsonl"), "a")

    global_step = config.start_goal * config.num_turns

    if config.start_goal > 0:
        logger.info(f"Resuming from goal {config.start_goal}")

    for goal_idx in range(config.start_goal, num_goals):
        goal_row = dataset[goal_idx]
        goal_text = goal_row["Goal"]
        rubric_items = goal_row["Rubric"]
        ref_solution = goal_row["Reference solution"]

        logger.info(f"\n{'='*60}")
        logger.info(f"GOAL {goal_idx}/{num_goals}: {goal_text[:100]}...")

        accumulated_hints = []
        goal_turn_scores = []

        for turn_idx in range(config.num_turns):
            turn_t0 = time.time()
            logger.info(f"  Turn {turn_idx}/{config.num_turns} (goal {goal_idx})")

            # ── Generate plan (base model, no LoRA updates) ──
            if not config.use_hints:
                prompt_text = build_plan_prompt(scenario=goal_text, hint=None)
            elif config.accumulate_hints:
                hints_to_use = accumulated_hints[-config.max_hints:] if config.max_hints > 0 else accumulated_hints
                prompt_text = build_plan_prompt_accumulated(
                    scenario=goal_text, hints=hints_to_use
                )
            else:
                latest_hint = accumulated_hints[-1] if accumulated_hints else None
                prompt_text = build_plan_prompt(
                    scenario=goal_text, hint=latest_hint
                )
            convo = [{"role": "user", "content": prompt_text}]
            model_input = renderer.build_generation_prompt(convo)

            result = sampling_client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
            ).result()

            plan_text = renderers.get_text_content(
                renderer.parse_response(result.sequences[0].tokens)[0]
            )
            if "<solution>" in plan_text and "</solution>" not in plan_text:
                plan_text = plan_text.rstrip() + "\n</solution>"

            # ── Grade ──
            grader_prompt = build_grader_with_hint_prompt(
                scenario=goal_text,
                rubric_items=rubric_items,
                proposed_plan=plan_text,
                reference_solution=ref_solution,
            )
            grader_input = grader_renderer.build_generation_prompt(
                [{"role": "user", "content": grader_prompt}]
            )
            grader_result = grader_client.sample(
                grader_input, num_samples=1,
                sampling_params=grader_sampling_params,
            ).result()

            grader_text = renderers.get_text_content(
                grader_renderer.parse_response(grader_result.sequences[0].tokens)[0]
            )

            rubric_score = compute_rubric_score(grader_text)
            reward, word_count = compute_reward(plan_text, rubric_score, config)
            plan_hint = extract_hint(grader_text)

            # Accumulate hint for next turn
            if plan_hint:
                accumulated_hints.append(plan_hint)

            goal_turn_scores.append(rubric_score)

            logger.info(
                f"    rubric={rubric_score:.3f}, reward={reward:.3f}, "
                f"words={word_count}, hints_accumulated={len(accumulated_hints)}"
            )

            # ── Log (NO training step) ──
            turn_summary = {
                "global_step": global_step,
                "goal_idx": goal_idx,
                "turn_idx": turn_idx,
                "mode": "training_free",
                "rubric/mean": rubric_score,
                "rubric/best": rubric_score,
                "reward/mean": reward,
                "n_hints": len(accumulated_hints),
                "prompt_length": len(prompt_text),
                "time": time.time() - turn_t0,
            }
            f_summary.write(json.dumps(turn_summary) + "\n")

            sample_log = {
                "global_step": global_step,
                "goal_idx": goal_idx,
                "turn_idx": turn_idx,
                "sample_idx": 0,
                "rubric_score": rubric_score,
                "reward": reward,
                "word_count": word_count,
                "hint_length": len(plan_hint),
                "policy_output": plan_text[:500],
                "hint": plan_hint[:500],
            }
            f_logs.write(json.dumps(sample_log) + "\n")

            global_step += 1

        # ── Goal summary ──
        if goal_turn_scores:
            logger.info(
                f"  Goal {goal_idx} done: rubric trajectory = "
                f"{[f'{s:.3f}' for s in goal_turn_scores]}"
            )
            goal_summary = {
                "type": "goal_summary",
                "goal_idx": goal_idx,
                "rubric_trajectory": goal_turn_scores,
                "rubric_first": goal_turn_scores[0],
                "rubric_last": goal_turn_scores[-1],
                "rubric_best": max(goal_turn_scores),
                "improvement": goal_turn_scores[-1] - goal_turn_scores[0],
                "total_steps": len(goal_turn_scores),
            }
            f_summary.write(json.dumps(goal_summary) + "\n")
            f_summary.flush()

    f_summary.close()
    f_logs.close()
    logger.info(f"Done. {global_step} steps across {num_goals} goals.")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
