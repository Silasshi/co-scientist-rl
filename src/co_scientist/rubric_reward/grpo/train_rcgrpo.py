"""
RC-GRPO (Rubric-Conditioned GRPO) trainer.

Modified version of best_ver.py that adds rubric prediction before plan generation.
A frozen rubric predictor predicts rubric items for each goal, then injects them
into the generation prompt. The grader still uses REAL rubric items (unchanged).

The only structural changes from best_ver.py:
  - Phase 0 (NEW): predict rubric items for all goals in the batch
  - Phase 1: uses build_conditioned_research_plan_prompt() instead of build_research_plan_prompt()
  - Logging: predicted_rubrics saved alongside other batch logs

Usage:
  python src/co_scientist/trainers/grpo/train_rcgrpo.py

  # With specific rubric predictor checkpoint
  python src/co_scientist/trainers/grpo/train_rcgrpo.py \\
      rubric_predictor_checkpoint_path=runs/2026/4/rubric_predictor/1 \\
      rubric_predictor_checkpoint_batch=200
"""

import logging
import time
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
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from datasets import load_dataset

from co_scientist.rubric_reward.grpo.train_rubric_predictor import build_predictor_prompt
from co_scientist.shared.eval_core import (
    build_grader_prompt,
    compute_rubric_reward_from_xml,
    check_format_compliance,
    resolve_checkpoint,
)


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    api_profile: str | None = None
    model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 64
    learning_rate: float = 1e-5
    clip_eps: float = 0.2
    batch_size: int = 64
    group_size: int = 8
    max_tokens: int = 2048
    grader_max_tokens: int = 8192
    temperature: float = 1.0
    grader_temperature: float = 0.0
    max_word_count: int = 750
    target_word_count: int = 600
    scale_length_bonus: float = 120.0
    scaling_factor: float = 0.08
    min_words: int = 30
    save_every: int = 0
    ml_data: bool = True
    run_eval: bool = False
    eval_epoch: int = -1
    max_length: int = 32768

    # RC-GRPO additions
    rubric_predictor_model: str = "Qwen/Qwen3-8B-Base"
    rubric_predictor_checkpoint_path: str = "runs/2026/4/rubric_predictor/1"
    rubric_predictor_checkpoint_batch: int = -1
    rubric_predictor_lora_rank: int = 32
    rubric_predictor_max_tokens: int = 1024
    rubric_predictor_temperature: float = 0.3

    log_path: str = "runs/2026/4/rcgrpo/1"
    today_date = time.strftime("%Y-%m-%d", time.localtime())


# ============================================================
# Prompt builders
# ============================================================

def build_conditioned_research_plan_prompt(
    scenario: str,
    predicted_rubrics: str,
    examples: list[dict] | None = None,
) -> str:
    """Same as build_research_plan_prompt but with predicted rubric criteria injected."""

    prompt = textwrap.dedent(f"""
        I will provide you a research scenario. You have to provide me a concise yet thoughtful research plan with all details needed to execute it.

        The following evaluation criteria are likely important for this research goal. Address them in your plan:
        {predicted_rubrics}
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
# Rubric predictor setup
# ============================================================

def setup_rubric_predictor(config: Config, service_client):

    resume_info, batch_label = resolve_checkpoint(
        config.rubric_predictor_checkpoint_path,
        config.rubric_predictor_checkpoint_batch,
    )

    if resume_info:
        predictor_training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
        logger.info(f"Loaded rubric predictor checkpoint: batch {resume_info['batch']}")
    else:
        predictor_training_client = service_client.create_lora_training_client(
            base_model=config.rubric_predictor_model,
            rank=config.rubric_predictor_lora_rank,
        )
        logger.info("Using base rubric predictor (no checkpoint)")

    # Save weights and create frozen sampling client
    predictor_result = predictor_training_client.save_weights_for_sampler(
        name=f"rubric_predictor_b{batch_label}"
    ).result()

    predictor_client = service_client.create_sampling_client(
        model_path=predictor_result.path
    )
    logger.info(f"Rubric predictor sampling client ready (checkpoint b{batch_label})")

    return predictor_client


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

    # Also need renderer for the rubric predictor (may be different model)
    predictor_tokenizer = get_tokenizer(config.rubric_predictor_model)
    predictor_renderer_name = model_info.get_recommended_renderer_name(config.rubric_predictor_model)
    predictor_renderer = renderers.get_renderer(predictor_renderer_name, predictor_tokenizer)
    logger.info(f"Using predictor renderer: {predictor_renderer_name}")

    # Load dataset
    logger.info("Loading dataset...")
    if config.ml_data:
        data = load_dataset("facebook/research-plan-gen", "ml")

    assert isinstance(data, datasets.DatasetDict)
    if is_eval:
        dataset = data["test"]
    else:
        dataset = data["train"]

    n_train_batches = len(dataset) // config.batch_size

    # Setup service client
    service_client = create_service_client(
        base_url=config.base_url,
        api_profile=config.api_profile,
    )

    # --- Setup rubric predictor (frozen) ---
    predictor_client = setup_rubric_predictor(config, service_client)
    predictor_sampling_params = tinker.types.SamplingParams(
        max_tokens=config.rubric_predictor_max_tokens,
        stop=predictor_renderer.get_stop_sequences(),
        temperature=config.rubric_predictor_temperature,
    )

    # --- Setup policy model (same as best_ver.py) ---
    last_checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path)
    if (config.eval_epoch == 0 and config.run_eval) or (not config.run_eval and last_checkpoint is None):
        resume_info = False
    elif config.eval_epoch > 0 and config.run_eval:
        checkpoints = checkpoint_utils.load_checkpoints_file(config.log_path)
        checkpoints_with_key = [c for c in checkpoints if "state_path" in c]
        if checkpoints_with_key:
            resume_info = checkpoints_with_key[config.eval_epoch - 1]
            logger.info(
                f"Found {len(checkpoints_with_key)} valid checkpoints in {config.log_path}")
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

    # Create grader client
    grader_client = service_client.create_sampling_client(
        base_model=config.grader_model_name
    )

    # Sampling parameters for the policy model
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )

    # Optimization parameters
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    # Prepare sampler for eval mode
    if is_eval:
        logger.info("Eval mode: preparing fixed sampler")
        sampling_result = training_client.save_weights_for_sampler(
            name=f"Eval_Checkpoint{actual_batch - 1}"
        ).result()
        sampling_path = sampling_result.path
        sampling_client = service_client.create_sampling_client(
            model_path=sampling_path
        )

    logger.info(f"Training for {n_train_batches} batches")

    # Main training loop
    for batch_idx in range(start_batch, n_train_batches):
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
        batch_start_idx = batch_idx * config.batch_size
        batch_end_idx = min((batch_idx + 1) * config.batch_size, len(dataset))
        batch_rows = dataset.select(range(batch_start_idx, batch_end_idx))

        t0 = time.time()

        if not is_eval:
            # Save current policy weights for sampling
            sampling_result = training_client.save_weights_for_sampler(
                name=f"{real_batch:06d}"
            ).result()
            sampling_path = sampling_result.path
            sampling_client = service_client.create_sampling_client(
                model_path=sampling_path
            )

        # --- PHASE 0 (NEW): PREDICT RUBRICS FOR ALL GOALS ---
        logger.info(f"Batch {real_batch}: Phase 0 - Predicting rubrics for {len(batch_rows)} goals...")

        predictor_futures = []
        for goal in batch_rows["Goal"]:
            predictor_prompt_text = build_predictor_prompt(goal_text=goal)
            predictor_convo = [{"role": "user", "content": predictor_prompt_text}]
            predictor_input = predictor_renderer.build_generation_prompt(predictor_convo)

            predictor_futures.append(
                predictor_client.sample(
                    prompt=predictor_input,
                    num_samples=1,
                    sampling_params=predictor_sampling_params,
                )
            )

        # Collect predicted rubrics
        predicted_rubrics_list = []
        for pf in predictor_futures:
            p_result = pf.result()
            predicted_text = renderers.get_text_content(
                predictor_renderer.parse_response(p_result.sequences[0].tokens)[0]
            )
            predicted_rubrics_list.append(predicted_text)

        logger.info(f"Batch {real_batch}: Phase 0 complete. Rubrics predicted ({time.time()-t0:.1f}s)")

        # --- PHASE 1: LAUNCH POLICY GENERATIONS WITH PREDICTED RUBRICS ---
        policy_futures = []
        policy_prompts_tokens = []

        if is_eval:
            logger.info(f"EVAL MODE: Checkpoint {real_batch - 1}, Launching generation for {len(batch_rows)} goals...")
        else:
            logger.info(f"TRAINING MODE: Batch {real_batch}: Launching generation for {len(batch_rows)} goals...")

        for goal_idx, goal in enumerate(batch_rows["Goal"]):
            # Use conditioned prompt with predicted rubrics
            prompt_text = build_conditioned_research_plan_prompt(
                scenario=goal,
                predicted_rubrics=predicted_rubrics_list[goal_idx],
                examples=None,
            )

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

        # --- PHASE 2: COLLECT PLANS & LAUNCH GRADERS (ASYNC) ---
        if is_eval:
            logger.info(f"EVAL MODE: Batch {batch_idx}, collecting plans and launching graders...")
        else:
            logger.info(f"TRAINING MODE: Batch {real_batch}: collecting plans and launching graders...")

        batch_groups_data = []
        all_grader_futures = []

        for i, p_future in enumerate(policy_futures):
            result = p_future.result()
            goal = batch_rows["Goal"][i]
            rubric = batch_rows["Rubric"][i]
            ref_sol = batch_rows["Reference solution"][i]
            prompt_tokens = policy_prompts_tokens[i]

            group_grader_futures = []
            group_samples_info = []

            for group_result in result.sequences:
                proposed_plan = renderers.get_text_content(
                    renderer.parse_response(group_result.tokens)[0]
                )

                # Hard-code solution closure
                if "<solution>" in proposed_plan and "</solution>" not in proposed_plan:
                    proposed_plan = proposed_plan.rstrip() + "\n</solution>"

                # Build grader prompt with REAL rubric items (not predicted)
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
                group_samples_info.append({
                    "tokens": group_result.tokens,
                    "logprobs": group_result.logprobs,
                    "text": proposed_plan,
                })

            batch_groups_data.append({
                "prompt_tokens": prompt_tokens,
                "samples_info": group_samples_info,
                "rubric_len": len(rubric),
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

        batch_word_counts = []
        batch_format_penalties = []
        batch_rubric_scores = []
        batch_advantages = []
        num_total_samples = 0
        num_valid_samples = 0

        for group_idx, group_futures in enumerate(all_grader_futures):
            group_data = batch_groups_data[group_idx]
            group_rewards = []
            valid_samples = []

            for j, each_group_future in enumerate(group_futures):
                num_total_samples += 1

                each_group_result = each_group_future.result()
                parsed_message, _ = renderer.parse_response(each_group_result.sequences[0].tokens)
                xml_text = renderers.get_text_content(parsed_message)

                rubric_score = compute_rubric_reward_from_xml(xml_text)
                plan_text = group_data["samples_info"][j]["text"]
                word_count = len(plan_text.strip().split())

                is_compliant = check_format_compliance(plan_text, config.max_word_count)
                excess = max(0, word_count - config.max_word_count)
                format_penalty = 0.0 if is_compliant else 0.2 + 0.0005 * excess
                length_bonus = np.exp(
                    -((word_count - config.target_word_count) / config.scale_length_bonus) ** 2
                )
                final_reward = rubric_score + config.scaling_factor * length_bonus - format_penalty

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

                group_rewards.append(final_reward)
                batch_sample_rewards.append(final_reward)

                valid_samples.append({
                    "sample_info": group_data["samples_info"][j],
                    "reward": final_reward,
                })

                num_valid_samples += 1

                batch_logs_to_save.append({
                    "batch_idx": real_batch,
                    "group_idx": group_idx,
                    "sample_idx": j,

                    # Core outputs
                    "policy_output": plan_text,
                    "grader_output": xml_text,
                    "predicted_rubrics": predicted_rubrics_list[group_idx],

                    # Reward components
                    "rubric_score": rubric_score,
                    "format_penalty": format_penalty,
                    "final_reward": final_reward,

                    # Stats
                    "word_count": word_count,
                    "is_compliant": is_compliant,
                })

            # GRPO Advantage Calculation
            if not group_rewards:
                continue

            mean_reward = np.mean(group_rewards)
            advantages = [(r - mean_reward) for r in group_rewards]
            batch_rewards.append(mean_reward)
            batch_advantages.extend(advantages)

            # Skip if no learning signal
            if all(a == 0.0 for a in advantages):
                continue
            if len(valid_samples) < 2:
                logger.debug("Skipping group: not enough valid samples")
                continue

            if not is_eval:
                # Create training datums
                for k, sample in enumerate(valid_samples):
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

        # --- Batch summary ---
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

        # Write batch logs
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
                    "clip_low_threshold": 1 - config.clip_eps,
                    "clip_high_threshold": 1 + config.clip_eps,
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
        ml_logger.log_metrics(metrics, step=real_batch)

    # Save final checkpoint
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
