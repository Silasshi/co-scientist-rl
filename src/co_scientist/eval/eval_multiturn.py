"""
Multi-turn evaluation script for co-scientist research plan models.

Generates multi-turn conversations (student + tutor) and grades final plans
with the canonical rubric grader. No training, no PRM, no OPD.

Usage:
  # Evaluate a checkpoint
  python eval_multiturn.py checkpoint_run_path=/path/to/run

  # Evaluate base model
  python eval_multiturn.py checkpoint_run_path=none checkpoint_batch=0

  # Custom output
  python eval_multiturn.py checkpoint_run_path=/path/to/run eval_output_path=/out
"""

import asyncio
import json
import logging
import os
import re
import sys
import textwrap
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
import tinker
from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.eval_core import resolve_checkpoint
from co_scientist.shared.openrouter_client import OpenRouterClient
from co_scientist.rubric_reward.multiturn.train_multiturn_v4 import (
    COLLABORATOR_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT_TEMPLATE,
    RESEARCHER_FIRST_MESSAGE_TEMPLATE,
    FORCED_PLAN_REQUEST,
    get_text_content,
    strip_thinking,
    extract_solution,
    close_solution_tag,
    detect_plan_turn,
    detect_domain,
    build_grader_prompt,
    compute_rubric_reward_from_xml,
    sample_policy_async,
    grade_plan_async,
    ConversationResult,
    _mark_final_plan,
)
from tinker import types
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from datasets import load_dataset

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    checkpoint_run_path: str = "none"
    checkpoint_batch: int = -1
    eval_output_path: str = ""

    base_url: str | None = None
    api_profile: str | None = "NEW"

    model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"
    student_model: str = "google/gemini-2.0-flash-001"
    lora_rank: int = 64

    ml_data: bool = True
    arxiv_data: bool = False
    pubmed_data: bool = False

    # Multi-turn
    min_discussion_turns: int = 1
    max_discussion_turns: int = 5
    max_plan_revisions: int = 2
    max_policy_turns: int = 7
    min_plan_turns: int = 1
    group_size: int = 4
    batch_size: int = 8
    num_eval_batches: int = 10

    # Generation
    max_tokens: int = 2048
    max_length: int = 32768
    grader_max_tokens: int = 8192
    temperature: float = 1.0
    grader_temperature: float = 0.0
    student_temperature: float = 0.7
    student_max_tokens: int = 2048
    max_concurrent_tinker: int = 4


async def generate_eval_conversation(
    goal_idx, sample_idx, goal_text, domain, sampling_client, renderer,
    openrouter_client, config, tinker_semaphore,
) -> ConversationResult | None:
    """Generate one conversation for evaluation (same as training but no scoring)."""
    try:
        student_system = RESEARCHER_SYSTEM_PROMPT_TEMPLATE.format(
            domain=domain, research_goal=goal_text,
            min_discussion_turns=config.min_discussion_turns,
            max_discussion_turns=config.max_discussion_turns,
            max_plan_revisions=config.max_plan_revisions,
        )
        first_msg = RESEARCHER_FIRST_MESSAGE_TEMPLATE.format(research_goal=goal_text)

        conversation = []
        turn_data = []
        policy_turn_count = 0
        plan_revision_count = 0
        plan_turn_count = 0
        discussion_turn_count = 0
        student_msg = first_msg
        final_plan_text = ""
        forced_plan_used = False
        layer4_used = False

        sampling_params = tinker.types.SamplingParams(
            max_tokens=config.max_tokens,
            stop=renderer.get_stop_sequences(),
            temperature=config.temperature,
        )

        while policy_turn_count < config.max_policy_turns or (not layer4_used and plan_turn_count == 0):
            conversation.append({"role": "user", "content": student_msg})
            messages = [{"role": "system", "content": COLLABORATOR_SYSTEM_PROMPT}] + conversation
            model_input = renderer.build_generation_prompt(messages, role="assistant")
            prompt_tokens = [int(t) for t in model_input.to_ints()]

            if len(prompt_tokens) > config.max_length - config.max_tokens:
                break

            response = await sample_policy_async(
                sampling_client, model_input, sampling_params, tinker_semaphore,
            )
            seq = response.sequences[0]
            raw_text = get_text_content(renderer.parse_response(seq.tokens)[0])
            visible_text = strip_thinking(raw_text)
            visible_text = close_solution_tag(visible_text)
            is_plan = detect_plan_turn(visible_text)

            turn_data.append({
                "turn_idx": policy_turn_count,
                "prompt_tokens": prompt_tokens,
                "response_tokens": [int(t) for t in seq.tokens],
                "response_logprobs": [float(lp) for lp in seq.logprobs],
                "visible_text": visible_text,
                "is_plan_turn": is_plan,
                "is_final_plan": False,
                "prm_score": None,
                "prm_hint": None,
            })

            conversation.append({"role": "assistant", "content": visible_text})
            policy_turn_count += 1

            if is_plan:
                final_plan_text = visible_text
                plan_revision_count += 1
                plan_turn_count += 1
            else:
                discussion_turn_count += 1

            if policy_turn_count >= config.max_policy_turns:
                if plan_turn_count == 0 and not layer4_used:
                    layer4_used = True
                    forced_plan_used = True
                    student_msg = FORCED_PLAN_REQUEST
                    continue
                else:
                    break

            if is_plan and plan_revision_count >= config.max_plan_revisions:
                break

            student_messages = [{"role": "system", "content": student_system}] + conversation
            try:
                student_response = await openrouter_client.chat(
                    model=config.student_model,
                    messages=student_messages,
                    temperature=config.student_temperature,
                    max_tokens=config.student_max_tokens,
                )
            except Exception:
                break

            if not student_response or not student_response.strip():
                try:
                    student_response = await openrouter_client.chat(
                        model=config.student_model,
                        messages=student_messages + [
                            {"role": "user", "content": "Please continue the discussion."}
                        ],
                        temperature=config.student_temperature,
                        max_tokens=config.student_max_tokens,
                    )
                except Exception:
                    pass

            if not student_response or not student_response.strip():
                if plan_turn_count > 0:
                    conversation.append({"role": "user", "content": "[PLAN_FINAL]"})
                break

            if "[PLAN_FINAL]" in student_response:
                if plan_turn_count < config.min_plan_turns:
                    student_msg = FORCED_PLAN_REQUEST
                    forced_plan_used = True
                    continue
                conversation.append({"role": "user", "content": student_response})
                break

            student_msg = student_response

        if not final_plan_text and turn_data:
            final_plan_text = turn_data[-1]["visible_text"]

        _mark_final_plan(turn_data)

        completed = conversation and "[PLAN_FINAL]" in conversation[-1].get("content", "")
        reason = "plan_final" if completed else (
            "max_revisions" if plan_revision_count >= config.max_plan_revisions else "max_turns"
        )

        return ConversationResult(
            goal_idx=goal_idx, sample_idx=sample_idx,
            turn_data=turn_data, conversation=conversation,
            final_plan_text=final_plan_text, completed=completed,
            completion_reason=reason,
            num_discussion_turns=discussion_turn_count,
            num_plan_revisions=plan_revision_count,
            forced_plan_used=forced_plan_used,
        )
    except Exception as e:
        logger.error("Eval conversation failed g=%d s=%d: %s", goal_idx, sample_idx, e, exc_info=True)
        return None


async def main(config: Config):
    # --- Output path ---
    if config.eval_output_path:
        output_dir = config.eval_output_path
    elif config.checkpoint_run_path.lower() != "none":
        output_dir = os.path.join(config.checkpoint_run_path, "eval_multiturn")
    else:
        output_dir = os.path.join(os.getcwd(), "eval_multiturn_base")
    os.makedirs(output_dir, exist_ok=True)

    ml_logger = ml_log.setup_logging(output_dir)
    logger.info("Multi-turn eval config: %s", config)

    # --- Dataset ---
    if config.ml_data:
        data = load_dataset("facebook/research-plan-gen", "ml")
    elif config.arxiv_data:
        data = load_dataset("facebook/research-plan-gen", "arxiv")
    else:
        data = load_dataset("facebook/research-plan-gen", "pubmed")
    dataset = data["train"]
    domain = detect_domain(config.ml_data, config.arxiv_data, config.pubmed_data)

    # --- Setup ---
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    service_client = create_service_client(base_url=config.base_url, api_profile=config.api_profile)

    # --- Load checkpoint ---
    resume_info, batch_label = resolve_checkpoint(config.checkpoint_run_path, config.checkpoint_batch)
    if resume_info:
        sampling_client = service_client.create_sampling_client(model_path=resume_info["sampler_path"])
    else:
        sampling_client = service_client.create_sampling_client(base_model=config.model_name)

    grader_client = service_client.create_sampling_client(base_model=config.grader_model_name)
    openrouter_client = OpenRouterClient()
    tinker_semaphore = asyncio.Semaphore(config.max_concurrent_tinker)
    grader_params = tinker.types.SamplingParams(
        max_tokens=config.grader_max_tokens, temperature=config.grader_temperature,
    )

    logger.info("Evaluating checkpoint=%s, %d batches of %d goals × %d samples",
                batch_label, config.num_eval_batches, config.batch_size, config.group_size)

    # --- Eval loop ---
    all_rubrics = []
    all_results = []
    results_path = os.path.join(output_dir, f"eval_b{batch_label}.jsonl")

    for batch_idx in range(config.num_eval_batches):
        batch_start = (batch_idx * config.batch_size) % len(dataset)
        batch_end = min(batch_start + config.batch_size, len(dataset))
        batch_rows = dataset.select(range(batch_start, batch_end))

        logger.info("Eval batch %d: goals %d-%d", batch_idx, batch_start, batch_end)
        batch_t0 = time.time()

        # Phase 1: Generate conversations
        conv_tasks = []
        for goal_idx in range(len(batch_rows["Goal"])):
            for sample_idx in range(config.group_size):
                task = asyncio.create_task(
                    generate_eval_conversation(
                        goal_idx, sample_idx, batch_rows["Goal"][goal_idx], domain,
                        sampling_client, renderer, openrouter_client, config, tinker_semaphore,
                    )
                )
                conv_tasks.append(task)

        conv_results = await asyncio.gather(*conv_tasks)
        valid = [r for r in conv_results if r is not None]

        # Phase 2: Grade final plans
        grading_tasks = {}
        for conv in valid:
            solution = extract_solution(conv.final_plan_text)
            if len(solution.split()) < 30:
                continue
            grader_prompt = build_grader_prompt(
                scenario=batch_rows["Goal"][conv.goal_idx],
                rubric_items=batch_rows["Rubric"][conv.goal_idx],
                proposed_plan=solution,
                reference_solution=batch_rows["Reference solution"][conv.goal_idx],
            )
            grader_input = renderer.build_generation_prompt(
                [{"role": "user", "content": grader_prompt}], role="assistant",
            )
            task = asyncio.create_task(
                grade_plan_async(grader_client, grader_input, grader_params, tinker_semaphore),
            )
            grading_tasks[id(conv)] = (task, conv)

        batch_rubrics = []
        for conv_id, (task, conv) in grading_tasks.items():
            try:
                resp = await task
                xml_text = get_text_content(renderer.parse_response(resp.sequences[0].tokens)[0])
                rubric_score = compute_rubric_reward_from_xml(xml_text)
                batch_rubrics.append(rubric_score)
                all_rubrics.append(rubric_score)

                result = {
                    "batch_idx": batch_idx,
                    "goal_idx": conv.goal_idx,
                    "sample_idx": conv.sample_idx,
                    "rubric_score": rubric_score,
                    "num_turns": len(conv.turn_data),
                    "num_discussion": conv.num_discussion_turns,
                    "num_revisions": conv.num_plan_revisions,
                    "completed": conv.completed,
                    "completion_reason": conv.completion_reason,
                    "forced_plan": conv.forced_plan_used,
                    "word_count": len(extract_solution(conv.final_plan_text).split()),
                    "plan_preview": extract_solution(conv.final_plan_text)[:300],
                }
                all_results.append(result)
                with open(results_path, "a") as f:
                    f.write(json.dumps(result) + "\n")

            except Exception as e:
                logger.warning("Grading failed: %s", e)

        batch_time = time.time() - batch_t0
        completed = sum(1 for r in valid if r.completed)
        logger.info(
            "Eval batch %d: %.1fs | n=%d rubric=%.3f comp=%d/%d",
            batch_idx, batch_time, len(batch_rubrics),
            np.mean(batch_rubrics) if batch_rubrics else 0,
            completed, len(valid),
        )

    # --- Summary ---
    if all_rubrics:
        print(f"\n{'='*50}")
        print(f"Multi-Turn Eval Summary (checkpoint={batch_label})")
        print(f"{'='*50}")
        print(f"Total plans graded: {len(all_rubrics)}")
        print(f"Mean rubric: {np.mean(all_rubrics):.4f}")
        print(f"Std rubric: {np.std(all_rubrics):.4f}")
        print(f"Median rubric: {np.median(all_rubrics):.4f}")
        print(f"Min: {np.min(all_rubrics):.4f}, Max: {np.max(all_rubrics):.4f}")

        completed_count = sum(1 for r in all_results if r["completed"])
        print(f"Completion rate: {completed_count}/{len(all_results)} ({completed_count/len(all_results)*100:.0f}%)")

        reasons = Counter(r["completion_reason"] for r in all_results)
        print(f"Completion reasons: {dict(reasons)}")

        turns = [r["num_turns"] for r in all_results]
        print(f"Avg turns: {np.mean(turns):.1f}")

        summary = {
            "checkpoint": batch_label,
            "n_graded": len(all_rubrics),
            "rubric_mean": float(np.mean(all_rubrics)),
            "rubric_std": float(np.std(all_rubrics)),
            "rubric_median": float(np.median(all_rubrics)),
            "completion_rate": completed_count / len(all_results),
            "completion_reasons": dict(reasons),
            "avg_turns": float(np.mean(turns)),
        }
        with open(os.path.join(output_dir, f"summary_b{batch_label}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    await openrouter_client.close()


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
