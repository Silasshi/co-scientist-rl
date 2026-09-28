"""
Blended Generation Trainer (async GRPO)

Extends doublegeneration with prompt relabeling: Y2 samples (generated from
the enriched refinement prompt P2) are relabeled as if produced from the
initial prompt P1. All 8 samples per goal (4 Y1 + 4 Y2) form one GRPO
advantage group, teaching the model to produce Y2-quality outputs directly
from P1.

When enable_blending=False, behaves identically to train_double_generation.py.

Key differences from doublegeneration when blending is enabled:
  - Y1 and Y2 rewards both from detailed grader (scale-consistent, strong signal)
  - Y2 generated in solution_only mode (high quality), with <think></think> prepended
    to training datums for format consistency with P1's think+solution output
  - Y2 logprobs recomputed under P1 (or zeroed in reinforce mode)
  - Single combined advantage group instead of two independent groups
"""
import asyncio
import logging
import time
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
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from datasets import load_dataset

# Refinement pipeline (detailed grader, feedback extraction, Stage 2 prompt)
from co_scientist.rubric_reward.refinement.train_double_generation import (
    build_research_plan_prompt,
    build_detailed_grader_prompt,
    build_self_teacher_prompt,
    extract_weaknesses_from_grader,
    extract_solution_text,
    sdpo_compute_rubric_reward_from_xml,
    sdpo_check_format_compliance,
    create_service_client,
    close_solution_tag,
    is_valid_sample,
    create_training_datum,
    compute_reward,
    is_valid_grader_xml,
    _sample_policy_with_timeout,
    _sample_grader_with_timeout,
    _sample_refinement_with_timeout,
    _get_capability_int,
    _apply_server_cap_limits,
)



logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


# ============================================================
# 1. CONFIG
# ============================================================

@chz.chz
class Config:
    # --- Connection ---
    base_url: str | None = None
    log_path: str = "/home/silas/co-scientist-project/runs/2026/3/blended/18"
    model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"

    # --- Evaluation ---
    run_eval: bool = False
    eval_epoch: int = -1

    # --- Dataset ---
    ml_data: bool = True
    arxiv_data: bool = False
    pubmed_data: bool = False

    # --- Batch & sampling ---
    batch_size: int = 64
    group_size: int = 4      # Samples per goal per stage (4 Y1 + 4 Y2 = 8 total)
    learning_rate: float = 1e-5
    clip_eps: float = 0.2

    # --- Model generation ---
    max_length: int = 32768
    lora_rank: int = 64
    save_every: int = 5
    max_tokens: int = 2048
    refinement_max_tokens: int = 4096
    grader_max_tokens: int = 12288
    temperature: float = 1.0
    grader_temperature: float = 0.0

    # --- Timeouts ---
    policy_timeout_sec: float = None
    grader_timeout_sec: float = None
    refinement_timeout_sec: float = None
    train_timeout_sec: float = None

    # --- Server capability limits ---
    use_server_cap_limits: bool = True

    # --- LoRA training targets ---
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True

    # --- Reward shaping ---
    max_word_count: int = 750
    target_word_count: int = 600
    scale_length_bonus: float = 120.0
    scaling_factor: float = 0.08
    min_words: int = 30

    # --- Refinement settings ---
    use_refinement: bool = True
    refinement_group_weight: float = 1.0   # Used when enable_blending=False
    normalize_advantages: bool = True
    feedback_max_bullets_per_item: int = 3
    feedback_max_items: int = 10

    # --- Blended generation settings ---
    enable_blending: bool = True
    blend_logprob_mode: str = "recompute"      # "recompute" or "reinforce" (zero logprobs)
    blend_y2_advantage_weight: float = 1.0     # Scale Y2 advantages (for ablation)
    blend_prepend_think: bool = True           # Prepend <think></think> to solution-only Y2 for format match

    today_date = time.strftime("%Y-%m-%d", time.localtime())


# ============================================================
# 2. NEW HELPERS FOR BLENDING
# ============================================================

def create_blended_datum(
    p1_prompt_tokens: list[int],
    y2_generated_tokens: list[int],
    recomputed_logprobs: list[float],
    advantage: float,
) -> types.Datum | None:
    """Create a GRPO datum for a Y2 sample relabeled with P1 prompt and recomputed logprobs."""
    full_seq = p1_prompt_tokens + y2_generated_tokens
    ob_len = len(p1_prompt_tokens) - 1

    input_tokens = full_seq[:-1]
    target_tokens = full_seq[1:]

    all_logprobs = [0.0] * ob_len + list(recomputed_logprobs)
    all_advantages = [0.0] * ob_len + [advantage] * len(recomputed_logprobs)

    if not (
        len(input_tokens)
        == len(target_tokens)
        == len(all_logprobs)
        == len(all_advantages)
    ):
        logger.warning(
            "Skipping malformed blended datum: input=%s target=%s logprobs=%s advantages=%s",
            len(input_tokens), len(target_tokens), len(all_logprobs), len(all_advantages),
        )
        return None

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(
                torch.tensor(target_tokens, dtype=torch.long)
            ),
            "logprobs": TensorData.from_torch(
                torch.tensor(all_logprobs, dtype=torch.float)
            ),
            "advantages": TensorData.from_torch(
                torch.tensor(all_advantages, dtype=torch.float)
            ),
        },
    )


async def _recompute_logprobs_for_blending(
    sampling_client: tinker.SamplingClient,
    p1_tokens: list[int],
    y2_samples: list[dict | None],
    prepend_tokens: list[int] | None = None,
) -> list[list[float] | None]:
    """Recompute Y2 logprobs conditioned on P1 instead of P2.

    For each non-None Y2 sample, scores [P1_tokens + prepend + Y2_tokens] and
    extracts logprobs from the P1 boundary onward (covering prepend + Y2).
    """
    tasks = []
    valid_indices = []
    p1_len = len(p1_tokens)
    prepend = prepend_tokens or []

    for j, sample in enumerate(y2_samples):
        if sample is None:
            continue
        y2_tokens = [int(t) for t in sample["tokens"]]
        full_seq = types.ModelInput.from_ints(tokens=p1_tokens + prepend + y2_tokens)
        tasks.append(sampling_client.compute_logprobs_async(full_seq))
        valid_indices.append((j, len(prepend) + len(y2_tokens)))

    if not tasks:
        return [None] * len(y2_samples)

    all_logprobs = await asyncio.gather(*tasks)

    result: list[list[float] | None] = [None] * len(y2_samples)
    for idx, (j, total_len) in enumerate(valid_indices):
        result[j] = all_logprobs[idx][p1_len : p1_len + total_len]

    return result


# ============================================================
# 3. MAIN TRAINING LOOP
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

    # --- Setup tokenizer, renderer, dataset ---
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

    # --- Setup Tinker clients ---
    service_client = create_service_client(base_url=config.base_url)

    effective_batch_size = config.batch_size
    effective_group_size = config.group_size

    if config.use_server_cap_limits:
        try:
            capabilities = await service_client.get_server_capabilities_async()
            effective_batch_size, effective_group_size = _apply_server_cap_limits(
                batch_size=config.batch_size,
                group_size=config.group_size,
                capabilities=capabilities,
            )
        except Exception:
            logger.exception(
                "Failed to query server capabilities; using configured batch/group sizes"
            )

    n_train_batches = len(dataset) // effective_batch_size
    if n_train_batches <= 0:
        logger.warning(
            "No train batches available: dataset=%s batch_size=%s",
            len(dataset), effective_batch_size,
        )
        ml_logger.close()
        return

    # --- Load or create checkpoint ---
    last_checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path)
    resume_info: dict[str, object] | bool = False

    if (config.eval_epoch == 0 and config.run_eval) or (
        not config.run_eval and last_checkpoint is None
    ):
        resume_info = False
    elif config.eval_epoch > 0 and config.run_eval:
        checkpoints = checkpoint_utils.load_checkpoints_file(config.log_path)
        checkpoints_with_key = [c for c in checkpoints if "state_path" in c]
        if checkpoints_with_key and config.eval_epoch <= len(checkpoints_with_key):
            resume_info = checkpoints_with_key[config.eval_epoch - 1]
            logger.info(
                f"Found {len(checkpoints_with_key)} valid checkpoints in {config.log_path}"
            )
        elif checkpoints_with_key:
            logger.warning(
                "Requested eval_epoch=%s exceeds available checkpoints=%s; using latest",
                config.eval_epoch, len(checkpoints_with_key),
            )
            resume_info = checkpoints_with_key[-1]
    else:
        resume_info = last_checkpoint

    if resume_info:
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(
            str(resume_info["state_path"])
        )
        resume_batch = int(resume_info["batch"])
        if is_eval:
            actual_batch = resume_batch + 1
            start_batch = 0
        elif (resume_batch + 1) // n_train_batches > 0:
            logger.info(f"Training for epoch: {(resume_batch + 1) // n_train_batches}")
            actual_batch = resume_batch + 1
            start_batch = 0
        else:
            start_batch = resume_batch + 1
            actual_batch = 0
        if is_eval:
            logger.info(f"Evaluating for Checkpoint {resume_info['batch']}")
        else:
            logger.info(f"Resuming from batch {resume_info['batch']}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_name,
            rank=config.lora_rank,
            train_mlp=config.train_mlp,
            train_attn=config.train_attn,
            train_unembed=config.train_unembed,
        )
        start_batch = 0
        actual_batch = 0

    # Grader uses base model (no LoRA) so scores are stable across training
    grader_client = await service_client.create_sampling_client_async(
        base_model=config.grader_model_name
    )

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )
    refinement_sampling_params = tinker.types.SamplingParams(
        max_tokens=config.refinement_max_tokens,
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

    if is_eval:
        logger.info("Eval mode: preparing fixed sampler")
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(
            name=f"Eval_Checkpoint{actual_batch - 1}"
        )

    # Precompute think wrapper tokens for format-matching Y2 (solution_only) to P1 (think+solution)
    think_wrapper_tokens: list[int] = []
    if config.enable_blending and config.blend_prepend_think:
        think_wrapper_tokens = tokenizer.encode("<think>\n</think>\n", add_special_tokens=False)
        logger.info(f"Think wrapper: {len(think_wrapper_tokens)} tokens prepended to Y2 datums")

    logger.info(
        f"Training for {n_train_batches} batches "
        f"(blending={'ON' if config.enable_blending else 'OFF'}, "
        f"logprob_mode={config.blend_logprob_mode})"
    )
    last_real_batch = -1

    # ================================================================
    # MAIN LOOP — one iteration per batch of research goals
    # ================================================================
    for batch_idx in range(start_batch, n_train_batches):
        t_start = time.time()

        if is_eval:
            real_batch = actual_batch
        else:
            real_batch = actual_batch + (batch_idx - start_batch)
        last_real_batch = real_batch

        batch_start = batch_idx * effective_batch_size
        batch_end = min((batch_idx + 1) * effective_batch_size, len(dataset))
        batch_rows = dataset.select(range(batch_start, batch_end))
        n_goals = len(batch_rows["Goal"])

        if not is_eval:
            sampling_client = await training_client.save_weights_and_get_sampling_client_async()

        logger.info(
            f"{'EVAL' if is_eval else 'TRAIN'} Batch {real_batch}: "
            f"{n_goals} goals x {effective_group_size} samples"
        )

        # ==============================================================
        # PHASE 1: GENERATE INITIAL PLANS (Y1) — UNCHANGED
        # ==============================================================
        policy_tasks = []
        for goal_idx, goal in enumerate(batch_rows["Goal"]):
            prompt_text = build_research_plan_prompt(scenario=goal, examples=None)
            model_input = renderer.build_generation_prompt(
                [{"role": "user", "content": prompt_text}]
            )
            policy_tasks.append(
                asyncio.create_task(
                    _sample_policy_with_timeout(
                        sampling_client=sampling_client,
                        model_input=model_input,
                        goal_idx=goal_idx,
                        prompt_tokens=model_input.to_ints(),
                        group_size=effective_group_size,
                        sampling_params=sampling_params,
                        timeout_sec=config.policy_timeout_sec,
                    ),
                    name=f"policy_goal_{goal_idx}",
                )
            )

        # ==============================================================
        # PHASE 2: COLLECT Y1, LAUNCH DETAILED GRADERS
        # Detailed grader used for: feedback extraction + Y1 rewards
        # ==============================================================
        logger.info(f"Batch {real_batch}: Collecting initial plans, launching graders...")

        batch_groups_data: list[dict | None] = [None] * n_goals
        initial_grader_tasks = []

        for p_task in asyncio.as_completed(policy_tasks):
            policy_result = await p_task
            if policy_result is None:
                continue

            goal_idx, prompt_tokens, result = policy_result
            goal = batch_rows["Goal"][goal_idx]
            rubric = batch_rows["Rubric"][goal_idx]
            ref_sol = batch_rows["Reference solution"][goal_idx]

            group_samples_info = []
            group_grader_results: list[types.SampleResponse | None] = []

            for sample_idx, group_result in enumerate(result.sequences):
                raw_plan = renderers.get_text_content(
                    renderer.parse_response(group_result.tokens)[0]
                )
                proposed_plan = close_solution_tag(raw_plan)

                # Detailed grader (always — for feedback extraction)
                grader_prompt_text = build_detailed_grader_prompt(
                    scenario=goal,
                    rubric_items=rubric,
                    proposed_plan=proposed_plan,
                    reference_solution=ref_sol,
                    fast_grader_mode=True,
                )
                grader_input = renderer.build_generation_prompt(
                    [{"role": "user", "content": grader_prompt_text}]
                )
                initial_grader_tasks.append(
                    asyncio.create_task(
                        _sample_grader_with_timeout(
                            grader_client=grader_client,
                            grader_input=grader_input,
                            group_idx=goal_idx,
                            sample_idx=sample_idx,
                            grader_sampling_params=grader_sampling_params,
                            timeout_sec=config.grader_timeout_sec,
                        ),
                        name=f"init_grader_g{goal_idx}_s{sample_idx}",
                    )
                )

                group_samples_info.append({
                    "tokens": group_result.tokens,
                    "logprobs": group_result.logprobs,
                    "text": proposed_plan,
                    "raw_text": raw_plan,
                })
                group_grader_results.append(None)

            batch_groups_data[goal_idx] = {
                "goal": goal,
                "rubric": rubric,
                "ref_sol": ref_sol,
                "prompt_tokens": prompt_tokens,
                "samples_info": group_samples_info,
                "grader_results": group_grader_results,
            }

        # Collect detailed grader results
        for g_task in asyncio.as_completed(initial_grader_tasks):
            group_idx, sample_idx, grader_result = await g_task
            group_data = batch_groups_data[group_idx]
            if group_data is not None:
                group_data["grader_results"][sample_idx] = grader_result

        # ==============================================================
        # PHASE 3: PARSE DETAILED GRADES, EXTRACT FEEDBACK
        # ==============================================================
        logger.info(f"Batch {real_batch}: Parsing grades, extracting feedback...")

        all_initial_grades: list[list[dict] | None] = [None] * n_goals

        for group_idx, group_data in enumerate(batch_groups_data):
            if group_data is None:
                continue

            group_grades = []
            for j, grader_result in enumerate(group_data["grader_results"]):
                plan_text = group_data["samples_info"][j]["text"]
                raw_text = group_data["samples_info"][j]["raw_text"]

                if grader_result is None:
                    group_grades.append({
                        "xml_text": "",
                        "plan_text": plan_text,
                        "feedback_text": "Critical revision needed: strengthen implementation detail, rationale, and risk controls.",
                        "feedback_bullets": 1,
                        "rubric_score": 0.0,
                        "word_count": len(plan_text.strip().split()),
                        "solution_word_count": 0,
                        "is_compliant": sdpo_check_format_compliance(raw_text, config.max_word_count),
                        "format_penalty": 0.2,
                        "length_bonus": 0.0,
                        "final_reward": 0.0,
                        "failure_reason": "grader_timeout_or_error",
                    })
                    continue

                try:
                    parsed_msg, _ = renderer.parse_response(grader_result.sequences[0].tokens)
                    xml_text = renderers.get_text_content(parsed_msg)
                except Exception:
                    logger.exception(
                        "Failed to parse grader output for group_idx=%s sample_idx=%s",
                        group_idx, j,
                    )
                    group_grades.append({
                        "xml_text": "",
                        "plan_text": plan_text,
                        "feedback_text": "Critical revision needed: strengthen implementation detail, rationale, and risk controls.",
                        "feedback_bullets": 1,
                        "rubric_score": 0.0,
                        "word_count": len(plan_text.strip().split()),
                        "solution_word_count": 0,
                        "is_compliant": sdpo_check_format_compliance(raw_text, config.max_word_count),
                        "format_penalty": 0.2,
                        "length_bonus": 0.0,
                        "final_reward": 0.0,
                        "failure_reason": "grader_parse_error",
                    })
                    continue

                reward_info = compute_reward(plan_text, xml_text, config, raw_plan_text=raw_text)

                if is_valid_grader_xml(xml_text):
                    feedback_text, bullet_count, _ = extract_weaknesses_from_grader(
                        xml_text,
                        max_bullets_per_item=config.feedback_max_bullets_per_item,
                        max_items=config.feedback_max_items,
                        focus_low_confidence_items_only=True,
                        include_sample_review=True,
                        include_item_reasoning_feedback=True,
                        include_item_desiderata_review=False,
                    )
                else:
                    feedback_text = "Critical revision needed: strengthen implementation detail, rationale, and risk controls."
                    bullet_count = 1

                group_grades.append({
                    "xml_text": xml_text,
                    "plan_text": plan_text,
                    "feedback_text": feedback_text,
                    "feedback_bullets": bullet_count,
                    **reward_info,
                })

            all_initial_grades[group_idx] = group_grades

        # ==============================================================
        # PHASE 4: GENERATE REFINED PLANS (Y2) — MODIFIED
        # When blending: use think_solution format (matching P1 output)
        # ==============================================================
        refined_samples_info: list[list[dict | None] | None] = [None] * n_goals
        refined_prompt_tokens_store: list[list[list[int]] | None] = [None] * n_goals
        all_refined_grades: list[list[dict | None] | None] = [None] * n_goals
        y2_grader_tasks = []

        if config.use_refinement:
            logger.info(f"Batch {real_batch}: Launching refinement generations...")

            y2_output_mode = "solution_only"
            refinement_tasks = []

            for group_idx, group_data in enumerate(batch_groups_data):
                if group_data is None or all_initial_grades[group_idx] is None:
                    continue

                goal = group_data["goal"]
                group_grades = all_initial_grades[group_idx]
                group_refine_prompt_tokens = []
                refined_samples_info[group_idx] = [None] * len(group_grades)
                all_refined_grades[group_idx] = [None] * len(group_grades)
                refined_prompt_tokens_store[group_idx] = []

                for j, grade in enumerate(group_grades):
                    if grade["rubric_score"] > 0.0:
                        draft = extract_solution_text(grade["plan_text"]) or grade["plan_text"]
                    else:
                        draft = None
                    refinement_prompt = build_self_teacher_prompt(
                        scenario=goal,
                        grader_feedback=grade["feedback_text"],
                        initial_draft=draft,
                        rubric_items=group_data["rubric"],
                        policy_output_mode=y2_output_mode,
                        target_word_count=config.target_word_count,
                        max_solution_words=config.max_word_count,
                    )
                    refine_input = renderer.build_generation_prompt(
                        [{"role": "user", "content": refinement_prompt}]
                    )
                    refine_prompt_tokens = refine_input.to_ints()
                    group_refine_prompt_tokens.append(refine_prompt_tokens)

                    refinement_tasks.append(
                        asyncio.create_task(
                            _sample_refinement_with_timeout(
                                sampling_client=sampling_client,
                                refine_input=refine_input,
                                group_idx=group_idx,
                                sample_idx=j,
                                sampling_params=refinement_sampling_params,
                                timeout_sec=config.refinement_timeout_sec,
                            ),
                            name=f"refine_g{group_idx}_s{j}",
                        )
                    )

                refined_prompt_tokens_store[group_idx] = group_refine_prompt_tokens

            # Collect refinement results + immediately launch Y2 graders
            # (overlaps grading with remaining Y2 generation)
            for r_task in asyncio.as_completed(refinement_tasks):
                group_idx, sample_idx, r_result = await r_task
                if r_result is None or refined_samples_info[group_idx] is None:
                    continue
                refined_seq = r_result.sequences[0]
                raw_refined = renderers.get_text_content(
                    renderer.parse_response(refined_seq.tokens)[0]
                )
                refined_plan = close_solution_tag(raw_refined)
                refined_samples_info[group_idx][sample_idx] = {
                    "tokens": refined_seq.tokens,
                    "logprobs": refined_seq.logprobs,
                    "text": refined_plan,
                    "raw_text": raw_refined,
                }

                # Launch Y2 grader immediately
                group_data = batch_groups_data[group_idx]
                if group_data is not None:
                    grader_prompt_text = build_detailed_grader_prompt(
                        scenario=group_data["goal"],
                        rubric_items=group_data["rubric"],
                        proposed_plan=refined_plan,
                        reference_solution=group_data["ref_sol"],
                        fast_grader_mode=True,
                    )
                    grader_input = renderer.build_generation_prompt(
                        [{"role": "user", "content": grader_prompt_text}]
                    )
                    y2_grader_tasks.append(
                        asyncio.create_task(
                            _sample_grader_with_timeout(
                                grader_client=grader_client,
                                grader_input=grader_input,
                                group_idx=group_idx,
                                sample_idx=sample_idx,
                                grader_sampling_params=grader_sampling_params,
                                timeout_sec=config.grader_timeout_sec,
                            ),
                            name=f"ref_grader_g{group_idx}_s{sample_idx}",
                        )
                    )

        # ==============================================================
        # PHASE 4.5: RECOMPUTE Y2 LOGPROBS UNDER P1 — NEW
        # Only when blending with recompute mode.
        # ==============================================================
        recomputed_logprobs_store: list[list[list[float] | None] | None] = [None] * n_goals
        recompute_time = 0.0
        batch_logprob_deltas: list[float] = []

        if config.enable_blending and config.blend_logprob_mode == "recompute" and config.use_refinement:
            logger.info(f"Batch {real_batch}: Recomputing Y2 logprobs under P1...")
            t_recompute_start = time.time()

            recompute_tasks = []
            recompute_goal_indices = []

            for group_idx, group_data in enumerate(batch_groups_data):
                if group_data is None or refined_samples_info[group_idx] is None:
                    continue
                p1_tokens = [int(t) for t in group_data["prompt_tokens"]]
                recompute_tasks.append(
                    _recompute_logprobs_for_blending(
                        sampling_client, p1_tokens, refined_samples_info[group_idx],
                        prepend_tokens=think_wrapper_tokens or None,
                    )
                )
                recompute_goal_indices.append(group_idx)

            if recompute_tasks:
                recomputed_results = await asyncio.gather(*recompute_tasks)
                for gi, result in zip(recompute_goal_indices, recomputed_results):
                    recomputed_logprobs_store[gi] = result

                    # Diagnostic: logprob delta between P1 and P2 conditioning (Y2 tokens only)
                    prepend_len = len(think_wrapper_tokens)
                    if refined_samples_info[gi] is not None:
                        for j, r_sample in enumerate(refined_samples_info[gi]):
                            if r_sample is None or result[j] is None:
                                continue
                            p2_lp = r_sample.get("logprobs")
                            p1_lp = result[j][prepend_len:]  # skip think wrapper logprobs
                            if p2_lp is not None and len(p2_lp) == len(p1_lp):
                                delta = float(np.mean(np.abs(
                                    np.array(p1_lp, dtype=np.float32) - np.array(p2_lp, dtype=np.float32)
                                )))
                                batch_logprob_deltas.append(delta)

            recompute_time = time.time() - t_recompute_start
            logger.info(f"Batch {real_batch}: Logprob recomputation took {recompute_time:.1f}s")

        # ==============================================================
        # PHASE 5: COLLECT Y2 GRADES
        # (graders were launched during Phase 4 as Y2 samples arrived)
        # ==============================================================
        if config.use_refinement and y2_grader_tasks:
            logger.info(
                f"Batch {real_batch}: Collecting {len(y2_grader_tasks)} Y2 grader results..."
            )

            # Collect Y2 grader results
            for rg_task in asyncio.as_completed(y2_grader_tasks):
                group_idx, sample_idx, rg_result = await rg_task
                if all_refined_grades[group_idx] is None:
                    continue

                r_sample = refined_samples_info[group_idx][sample_idx]
                plan_text = r_sample["text"] if r_sample is not None else ""
                raw_text = r_sample["raw_text"] if r_sample is not None else ""

                if rg_result is None:
                    all_refined_grades[group_idx][sample_idx] = {
                        "xml_text": "", "plan_text": plan_text,
                        "rubric_score": 0.0, "word_count": len(plan_text.strip().split()),
                        "solution_word_count": 0,
                        "is_compliant": sdpo_check_format_compliance(raw_text, config.max_word_count) if raw_text else False,
                        "format_penalty": 0.2, "length_bonus": 0.0, "final_reward": 0.0,
                        "failure_reason": "grader_timeout_or_error",
                    }
                    continue

                try:
                    parsed_msg, _ = renderer.parse_response(rg_result.sequences[0].tokens)
                    xml_text = renderers.get_text_content(parsed_msg)
                except Exception:
                    logger.exception(
                        "Failed to parse Y2 grader output for g%s s%s", group_idx, sample_idx,
                    )
                    all_refined_grades[group_idx][sample_idx] = {
                        "xml_text": "", "plan_text": plan_text,
                        "rubric_score": 0.0, "word_count": len(plan_text.strip().split()),
                        "solution_word_count": 0,
                        "is_compliant": sdpo_check_format_compliance(raw_text, config.max_word_count) if raw_text else False,
                        "format_penalty": 0.2, "length_bonus": 0.0, "final_reward": 0.0,
                        "failure_reason": "grader_parse_error",
                    }
                    continue

                reward_info = compute_reward(plan_text, xml_text, config, raw_plan_text=raw_text)

                all_refined_grades[group_idx][sample_idx] = {
                    "xml_text": xml_text,
                    "plan_text": plan_text,
                    **reward_info,
                }

        # ==============================================================
        # PHASE 6: COMPUTE GRPO ADVANTAGES & BUILD TRAINING DATUMS
        # When blending: single combined group (4 Y1 + 4 Y2), all use P1
        # When not blending: two independent groups (original behavior)
        # ==============================================================
        logger.info(f"Batch {real_batch}: Computing advantages...")

        training_datums = []
        batch_logs_to_save = []
        dropped_samples = 0
        drop_reasons = {}

        # Batch-level diagnostics
        batch_initial_rewards = []
        batch_initial_rubrics = []
        batch_refined_rewards = []
        batch_refined_rubrics = []
        batch_deltas = []
        batch_advantages = []
        batch_word_counts = []
        batch_format_penalties = []
        num_total_samples = 0
        num_valid_samples = 0
        num_initial_datums = 0
        num_refined_datums = 0
        feedback_bullets_all = []

        # Blend-specific diagnostics
        batch_blend_y1_rewards = []
        batch_blend_y2_rewards = []
        batch_blend_y1_pos_adv = []
        batch_blend_y2_pos_adv = []

        for group_idx, group_data in enumerate(batch_groups_data):
            if group_data is None or all_initial_grades[group_idx] is None:
                continue

            # Always track detailed-grader metrics for Y1 (for feedback diagnostics)
            for j, grade in enumerate(all_initial_grades[group_idx]):
                num_total_samples += 1
                batch_word_counts.append(grade["word_count"])
                batch_format_penalties.append(grade["format_penalty"])
                batch_initial_rubrics.append(grade["rubric_score"])

            feedback_bullets_all.extend(
                g["feedback_bullets"] for g in all_initial_grades[group_idx]
            )

            # ===== BLENDED PATH =====
            if config.enable_blending:
                # Collect valid Y1 samples with detailed-grader rewards
                y1_rewards = []
                y1_valid = []

                for j, grade in enumerate(all_initial_grades[group_idx]):
                    sample_info = group_data["samples_info"][j]
                    plan_text = grade["plan_text"]

                    valid, drop_reason = is_valid_sample(plan_text, sample_info.get("logprobs"), config)
                    if not valid:
                        dropped_samples += 1
                        drop_reasons[drop_reason] = drop_reasons.get(drop_reason, 0) + 1
                        continue

                    y1_rewards.append(grade["final_reward"])
                    batch_initial_rewards.append(grade["final_reward"])
                    batch_blend_y1_rewards.append(grade["final_reward"])
                    y1_valid.append({"sample_info": sample_info, "idx": j})
                    num_valid_samples += 1

                    batch_logs_to_save.append({
                        "batch_idx": real_batch,
                        "group_idx": group_idx,
                        "sample_idx": j,
                        "group_type": "initial",
                        "raw_policy_output": group_data["samples_info"][j]["raw_text"],
                        "normalized_for_grader": plan_text,
                        "grader_output": grade["xml_text"],
                        "rubric_score": grade["rubric_score"],
                        "format_penalty": grade["format_penalty"],
                        "final_reward": grade["final_reward"],
                        "word_count": grade["word_count"],
                        "is_compliant": grade["is_compliant"],
                        "feedback_text": grade["feedback_text"],
                        "feedback_bullets": grade["feedback_bullets"],
                    })

                # Collect valid Y2 samples with detailed-grader rewards
                y2_rewards = []
                y2_valid = []

                if (
                    config.use_refinement
                    and all_refined_grades[group_idx] is not None
                    and refined_samples_info[group_idx] is not None
                ):
                    for j, grade in enumerate(all_refined_grades[group_idx]):
                        if grade is None or refined_samples_info[group_idx][j] is None:
                            continue

                        num_total_samples += 1
                        plan_text = grade["plan_text"]
                        batch_word_counts.append(grade["word_count"])
                        batch_format_penalties.append(grade["format_penalty"])
                        batch_refined_rubrics.append(grade["rubric_score"])

                        orig_rubric = all_initial_grades[group_idx][j]["rubric_score"]
                        delta = grade["rubric_score"] - orig_rubric
                        batch_deltas.append(delta)

                        r_sample = refined_samples_info[group_idx][j]
                        valid, drop_reason = is_valid_sample(plan_text, r_sample.get("logprobs"), config)
                        if not valid:
                            dropped_samples += 1
                            drop_reasons[drop_reason] = drop_reasons.get(drop_reason, 0) + 1
                            continue

                        y2_rewards.append(grade["final_reward"])
                        batch_refined_rewards.append(grade["final_reward"])
                        batch_blend_y2_rewards.append(grade["final_reward"])
                        y2_valid.append({"sample_info": r_sample, "idx": j})
                        num_valid_samples += 1

                        batch_logs_to_save.append({
                            "batch_idx": real_batch,
                            "group_idx": group_idx,
                            "sample_idx": j,
                            "group_type": "refined_blended",
                            "raw_policy_output": r_sample["raw_text"],
                            "normalized_for_grader": plan_text,
                            "grader_output": grade["xml_text"],
                            "rubric_score": grade["rubric_score"],
                            "format_penalty": grade["format_penalty"],
                            "final_reward": grade["final_reward"],
                            "word_count": grade["word_count"],
                            "is_compliant": grade["is_compliant"],
                        })

                # Combined GRPO advantage group
                all_rewards = y1_rewards + y2_rewards
                if len(all_rewards) >= 2 and not is_eval:
                    mean_r = np.mean(all_rewards)
                    if config.normalize_advantages:
                        std_r = np.std(all_rewards)
                        if std_r > 1e-8:
                            advantages = [(r - mean_r) / std_r for r in all_rewards]
                        else:
                            advantages = [0.0] * len(all_rewards)
                    else:
                        advantages = [(r - mean_r) for r in all_rewards]

                    # Apply Y2 advantage weight
                    n_y1 = len(y1_rewards)
                    if config.blend_y2_advantage_weight != 1.0:
                        for k in range(n_y1, len(advantages)):
                            advantages[k] *= config.blend_y2_advantage_weight

                    # Track positive advantage rates
                    for k in range(n_y1):
                        batch_blend_y1_pos_adv.append(1.0 if advantages[k] > 0 else 0.0)
                    for k in range(n_y1, len(advantages)):
                        batch_blend_y2_pos_adv.append(1.0 if advantages[k] > 0 else 0.0)

                    batch_advantages.extend(advantages)

                    if not all(a == 0.0 for a in advantages):
                        prompt_tokens = [int(t) for t in group_data["prompt_tokens"]]

                        # Y1 datums — standard (P1 prompt, original logprobs)
                        for k, entry in enumerate(y1_valid):
                            datum = create_training_datum(
                                prompt_tokens, entry["sample_info"], advantages[k]
                            )
                            if datum is not None:
                                training_datums.append(datum)
                                num_initial_datums += 1
                            else:
                                dropped_samples += 1

                        # Y2 datums — blended (P1 prompt, think wrapper + Y2 tokens, recomputed/zero logprobs)
                        for k, entry in enumerate(y2_valid):
                            adv_idx = n_y1 + k
                            j = entry["idx"]
                            raw_y2_tokens = [int(t) for t in entry["sample_info"]["tokens"]]
                            y2_tokens = think_wrapper_tokens + raw_y2_tokens

                            if config.blend_logprob_mode == "recompute":
                                recomputed = (
                                    recomputed_logprobs_store[group_idx][j]
                                    if recomputed_logprobs_store[group_idx] is not None
                                    else None
                                )
                                if recomputed is None:
                                    logger.warning(
                                        "Missing recomputed logprobs for g%s s%s, falling back to reinforce",
                                        group_idx, j,
                                    )
                                    recomputed = [0.0] * len(y2_tokens)
                            else:  # reinforce
                                recomputed = [0.0] * len(y2_tokens)

                            datum = create_blended_datum(
                                prompt_tokens, y2_tokens, recomputed, advantages[adv_idx]
                            )
                            if datum is not None:
                                training_datums.append(datum)
                                num_refined_datums += 1
                            else:
                                dropped_samples += 1

            # ===== NON-BLENDED PATH (original doublegeneration behavior) =====
            else:
                # ----- GROUP A: INITIAL PLANS -----
                initial_rewards = []
                initial_valid = []

                for j, grade in enumerate(all_initial_grades[group_idx]):
                    sample_info = group_data["samples_info"][j]
                    plan_text = grade["plan_text"]

                    valid, drop_reason = is_valid_sample(plan_text, sample_info.get("logprobs"), config)
                    if not valid:
                        dropped_samples += 1
                        drop_reasons[drop_reason] = drop_reasons.get(drop_reason, 0) + 1
                        continue

                    initial_rewards.append(grade["final_reward"])
                    batch_initial_rewards.append(grade["final_reward"])
                    initial_valid.append({"sample_info": sample_info, "reward": grade["final_reward"]})
                    num_valid_samples += 1

                    batch_logs_to_save.append({
                        "batch_idx": real_batch,
                        "group_idx": group_idx,
                        "sample_idx": j,
                        "group_type": "initial",
                        "raw_policy_output": group_data["samples_info"][j]["raw_text"],
                        "normalized_for_grader": plan_text,
                        "grader_output": grade["xml_text"],
                        "rubric_score": grade["rubric_score"],
                        "format_penalty": grade["format_penalty"],
                        "final_reward": grade["final_reward"],
                        "word_count": grade["word_count"],
                        "is_compliant": grade["is_compliant"],
                        "feedback_text": grade["feedback_text"],
                        "feedback_bullets": grade["feedback_bullets"],
                    })

                # Compute GRPO advantages for initial group
                if initial_rewards and not is_eval:
                    mean_r = np.mean(initial_rewards)
                    if config.normalize_advantages:
                        std_r = np.std(initial_rewards)
                        advantages = [(r - mean_r) / std_r for r in initial_rewards] if std_r > 1e-8 else [0.0] * len(initial_rewards)
                    else:
                        advantages = [(r - mean_r) for r in initial_rewards]
                    batch_advantages.extend(advantages)

                    if not all(a == 0.0 for a in advantages) and len(initial_valid) >= 2:
                        prompt_tokens = [int(t) for t in group_data["prompt_tokens"]]
                        for k, sample in enumerate(initial_valid):
                            datum = create_training_datum(
                                prompt_tokens, sample["sample_info"], advantages[k]
                            )
                            if datum is not None:
                                training_datums.append(datum)
                                num_initial_datums += 1
                            else:
                                dropped_samples += 1

                # ----- GROUP B: REFINED PLANS -----
                if (
                    config.use_refinement
                    and all_refined_grades[group_idx] is not None
                    and refined_samples_info[group_idx] is not None
                    and refined_prompt_tokens_store[group_idx] is not None
                ):
                    refined_rewards = []
                    refined_valid = []

                    for j, grade in enumerate(all_refined_grades[group_idx]):
                        if grade is None or refined_samples_info[group_idx][j] is None:
                            continue

                        num_total_samples += 1
                        plan_text = grade["plan_text"]
                        batch_word_counts.append(grade["word_count"])
                        batch_format_penalties.append(grade["format_penalty"])
                        batch_refined_rubrics.append(grade["rubric_score"])

                        orig_rubric = all_initial_grades[group_idx][j]["rubric_score"]
                        delta = grade["rubric_score"] - orig_rubric
                        batch_deltas.append(delta)

                        r_sample = refined_samples_info[group_idx][j]
                        valid, drop_reason = is_valid_sample(plan_text, r_sample.get("logprobs"), config)
                        if not valid:
                            dropped_samples += 1
                            drop_reasons[drop_reason] = drop_reasons.get(drop_reason, 0) + 1
                            continue

                        refined_rewards.append(grade["final_reward"])
                        batch_refined_rewards.append(grade["final_reward"])
                        refined_valid.append({
                            "sample_info": r_sample,
                            "reward": grade["final_reward"],
                            "prompt_token_idx": j,
                        })
                        num_valid_samples += 1

                        batch_logs_to_save.append({
                            "batch_idx": real_batch,
                            "group_idx": group_idx,
                            "sample_idx": j,
                            "group_type": "refined",
                            "raw_policy_output": r_sample["raw_text"],
                            "normalized_for_grader": plan_text,
                            "grader_output": grade["xml_text"],
                            "rubric_score": grade["rubric_score"],
                            "format_penalty": grade["format_penalty"],
                            "final_reward": grade["final_reward"],
                            "word_count": grade["word_count"],
                            "is_compliant": grade["is_compliant"],
                            "delta_rubric": delta,
                        })

                    # Compute GRPO advantages for refined group
                    if refined_rewards and not is_eval:
                        mean_r = np.mean(refined_rewards)
                        if config.normalize_advantages:
                            std_r = np.std(refined_rewards)
                            advantages = [(r - mean_r) / std_r for r in refined_rewards] if std_r > 1e-8 else [0.0] * len(refined_rewards)
                        else:
                            advantages = [(r - mean_r) for r in refined_rewards]
                        if config.refinement_group_weight != 1.0:
                            advantages = [a * config.refinement_group_weight for a in advantages]
                        batch_advantages.extend(advantages)

                        if not all(a == 0.0 for a in advantages) and len(refined_valid) >= 2:
                            for k, sample in enumerate(refined_valid):
                                pt_idx = sample["prompt_token_idx"]
                                refine_prompt_tokens = [
                                    int(t)
                                    for t in refined_prompt_tokens_store[group_idx][pt_idx]
                                ]
                                datum = create_training_datum(
                                    refine_prompt_tokens,
                                    sample["sample_info"],
                                    advantages[k],
                                )
                                if datum is not None:
                                    training_datums.append(datum)
                                    num_refined_datums += 1
                                else:
                                    dropped_samples += 1

        # ==============================================================
        # BATCH SUMMARY & LOGGING
        # ==============================================================
        batch_summary = {
            "batch_idx": real_batch,
            "group_initial/rubric_mean": float(np.mean(batch_initial_rubrics)) if batch_initial_rubrics else 0.0,
            "group_initial/reward_mean": float(np.mean(batch_initial_rewards)) if batch_initial_rewards else 0.0,
            "group_refined/rubric_mean": float(np.mean(batch_refined_rubrics)) if batch_refined_rubrics else 0.0,
            "group_refined/reward_mean": float(np.mean(batch_refined_rewards)) if batch_refined_rewards else 0.0,
            "refinement/delta_rubric_mean": float(np.mean(batch_deltas)) if batch_deltas else 0.0,
            "refinement/improvement_rate": float(np.mean([1.0 if d > 0 else 0.0 for d in batch_deltas])) if batch_deltas else 0.0,
            "refinement/regression_rate": float(np.mean([1.0 if d < 0 else 0.0 for d in batch_deltas])) if batch_deltas else 0.0,
            "refinement/feedback_bullets_mean": float(np.mean(feedback_bullets_all)) if feedback_bullets_all else 0.0,
            "advantage_mean": float(np.mean(batch_advantages)) if batch_advantages else 0.0,
            "advantage_std": float(np.std(batch_advantages)) if batch_advantages else 0.0,
            "length_mean": float(np.mean(batch_word_counts)) if batch_word_counts else 0.0,
            "format_penalty_mean": float(np.mean(batch_format_penalties)) if batch_format_penalties else 0.0,
            "format_compliance_rate": float(np.mean([1.0 if p == 0.0 else 0.0 for p in batch_format_penalties])) if batch_format_penalties else 0.0,
            "format_violation_rate": float(np.mean([1.0 if p > 0.0 else 0.0 for p in batch_format_penalties])) if batch_format_penalties else 0.0,
            "samples/generated": num_total_samples,
            "samples/valid": num_valid_samples,
            "datums/initial_count": num_initial_datums,
            "datums/refined_count": num_refined_datums,
            "group_initial/reward_std": float(np.std(batch_initial_rewards)) if batch_initial_rewards else 0.0,
            "group_refined/reward_std": float(np.std(batch_refined_rewards)) if batch_refined_rewards else 0.0,
            "drops/total": dropped_samples,
            "drops/too_short": drop_reasons.get("too_short", 0),
            "drops/bare_tag": drop_reasons.get("bare_tag", 0),
            "drops/no_logprobs": drop_reasons.get("no_logprobs", 0),
        }

        # Blend-specific metrics
        if config.enable_blending:
            batch_summary.update({
                "blend/y1_reward_mean": float(np.mean(batch_blend_y1_rewards)) if batch_blend_y1_rewards else 0.0,
                "blend/y2_reward_mean": float(np.mean(batch_blend_y2_rewards)) if batch_blend_y2_rewards else 0.0,
                "blend/y2_minus_y1_mean": (
                    float(np.mean(batch_blend_y2_rewards)) - float(np.mean(batch_blend_y1_rewards))
                    if batch_blend_y1_rewards and batch_blend_y2_rewards else 0.0
                ),
                "blend/y1_positive_adv_rate": float(np.mean(batch_blend_y1_pos_adv)) if batch_blend_y1_pos_adv else 0.0,
                "blend/y2_positive_adv_rate": float(np.mean(batch_blend_y2_pos_adv)) if batch_blend_y2_pos_adv else 0.0,
                "blend/combined_reward_mean": (
                    float(np.mean(batch_blend_y1_rewards + batch_blend_y2_rewards))
                    if batch_blend_y1_rewards or batch_blend_y2_rewards else 0.0
                ),
                "blend/logprob_delta_mean": float(np.mean(batch_logprob_deltas)) if batch_logprob_deltas else 0.0,
                "blend/recompute_time_sec": recompute_time,
            })

        # Write per-sample logs
        if batch_logs_to_save:
            if is_eval:
                log_path = os.path.join(
                    config.log_path, f"evaluation/eval_logs({actual_batch - 1}).jsonl"
                )
            else:
                log_path = os.path.join(config.log_path, "train/training_logs.jsonl")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a") as f:
                for log_item in batch_logs_to_save:
                    f.write(json.dumps(log_item) + "\n")

        # Write batch summary
        if batch_summary:
            if is_eval:
                summary_path = os.path.join(
                    config.log_path, f"evaluation/eval_batch_summary({actual_batch - 1}).jsonl"
                )
            else:
                summary_path = os.path.join(config.log_path, "train/batch_summary.jsonl")
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, "a") as f:
                f.write(json.dumps(batch_summary) + "\n")

        if config.enable_blending:
            logger.info(
                f"Batch {real_batch}: "
                f"y1_reward={batch_summary.get('blend/y1_reward_mean', 0):.3f} "
                f"y2_reward={batch_summary.get('blend/y2_reward_mean', 0):.3f} "
                f"gap={batch_summary.get('blend/y2_minus_y1_mean', 0):+.3f} "
                f"y2_pos_adv={batch_summary.get('blend/y2_positive_adv_rate', 0):.1%} "
                f"datums={num_initial_datums}+{num_refined_datums}"
            )
        else:
            logger.info(
                f"Batch {real_batch}: "
                f"init_rubric={batch_summary['group_initial/rubric_mean']:.3f} "
                f"ref_rubric={batch_summary['group_refined/rubric_mean']:.3f} "
                f"delta={batch_summary['refinement/delta_rubric_mean']:+.3f} "
                f"impr_rate={batch_summary['refinement/improvement_rate']:.1%} "
                f"datums={num_initial_datums}+{num_refined_datums}"
            )

        # --- EVAL MODE: skip optimization ---
        if is_eval:
            logger.info("EVAL MODE: skipping optimization step.")
            continue

        # ==============================================================
        # OPTIMIZATION STEP — UNCHANGED
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
            _fwd_bwd_result = await asyncio.wait_for(
                fwd_bwd_future.result_async(),
                timeout=config.train_timeout_sec,
            )
            logger.info("Forward/Backward took %.2fs", time.time() - t0)

            t1 = time.time()
            _optim_result = await asyncio.wait_for(
                optim_step_future.result_async(),
                timeout=config.train_timeout_sec,
            )
            logger.info("Optim step took %.2fs", time.time() - t1)
        except asyncio.TimeoutError:
            logger.exception("Training step timed out")
            continue
        except Exception:
            logger.exception("Training step failed")
            continue

        # Log metrics
        metrics: dict[str, float] = {
            "progress/batch": real_batch,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (real_batch + 1) / n_train_batches,
            "time/total": time.time() - t_start,
            "reward/initial_mean": batch_summary["group_initial/reward_mean"],
            "reward/refined_mean": batch_summary["group_refined/reward_mean"],
            "refinement/delta_rubric_mean": batch_summary["refinement/delta_rubric_mean"],
            "refinement/improvement_rate": batch_summary["refinement/improvement_rate"],
            "dropped_samples": dropped_samples,
        }
        if config.enable_blending:
            metrics["blend/y1_reward_mean"] = batch_summary.get("blend/y1_reward_mean", 0.0)
            metrics["blend/y2_reward_mean"] = batch_summary.get("blend/y2_reward_mean", 0.0)
            metrics["blend/y2_minus_y1_mean"] = batch_summary.get("blend/y2_minus_y1_mean", 0.0)
            metrics["blend/y2_positive_adv_rate"] = batch_summary.get("blend/y2_positive_adv_rate", 0.0)
            metrics["blend/recompute_time_sec"] = batch_summary.get("blend/recompute_time_sec", 0.0)
        ml_logger.log_metrics(metrics, step=real_batch)

        # Periodic checkpoint
        if config.save_every > 0 and batch_idx % config.save_every == 0 and batch_idx > 0:
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{real_batch:06d}_{config.today_date}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": real_batch},
            )

    # --- Save final checkpoint ---
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
