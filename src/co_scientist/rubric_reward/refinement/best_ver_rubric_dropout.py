"""
Criteria-Aware Generation with Signal-Driven Rubric Dropout.

Extends best_ver.py's single-stage GRPO with:
  1. Stochastic rubric visibility in the policy prompt (probability p per goal)
  2. Criteria-inference instructions in <think> when rubric is hidden
  3. Signal-driven p = gap / reference_gap: p tracks what fraction of the
     initial reward gap remains, with no manual decay rates or thresholds

When rubric is visible, the model learns what good criterion coverage looks like.
When rubric is hidden, the model learns to infer criteria autonomously.
As training closes the gap, p drops proportionally.

At eval time, rubric is NEVER shown — matching the real task.

Usage:
  python best_ver_rubric_dropout.py log_path=/path/to/run
  python best_ver_rubric_dropout.py rubric_prob_min=0.05
"""

import logging
import time
import random
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

# Import shared utilities from eval_only.py
from co_scientist.shared.eval_core import (
    build_grader_prompt,
    compute_rubric_reward_from_xml,
    check_format_compliance,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    api_profile: str | None = None
    log_path: str = "/home/silas/co-scientist-project/runs/2026/3/rubric_dropout/13"
    model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"

    # Evaluation parameter
    run_eval: bool = False
    eval_epoch: int = -1

    # Dataset
    ml_data: bool = True
    arxiv_data: bool = False
    pubmed_data: bool = False

    batch_size: int = 64     # Number of research goals
    group_size: int = 8      # G=8 indicated in the paper
    learning_rate: float = 1e-5
    clip_eps: float = 0.2       # Clipping epsilon for PPO/GRPO

    max_length: int = 32768
    lora_rank: int = 64
    save_every: int = 5  # 0 = disabled
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

    # ---- Rubric dropout ----
    rubric_prob_min: float = 0.05             # floor for p (>0 prevents dead state)
    rubric_prob_max: float = 0.80             # ceiling for p (guarantees hidden samples)
    rubric_target_reward: float = 0.86        # target hidden reward (reference score)

    today_date = time.strftime("%Y-%m-%d", time.localtime())


# ============================================================
# Policy prompt builder (modified for rubric dropout)
# ============================================================

def build_research_plan_prompt(
    scenario: str,
    rubric_items: list[str] | None = None,
    examples: list[dict] | None = None,
) -> str:
    """
    Build the policy prompt with optional rubric visibility.

    When rubric_items is provided: adds evaluation criteria section and
    criterion-aware think instructions.
    When rubric_items is None: adds criteria-inference instructions in <think>.

    The prompt structure matches best_ver.py / eval_only.py as closely as
    possible to avoid whitespace divergence when fine-tuning from existing
    checkpoints.
    """

    prompt = (
        "I will provide you a research scenario. "
        "You have to provide me a concise yet thoughtful research plan "
        "with all details needed to execute it."
    )

    # Few-shot examples (optional)
    if examples:
        prompt += (
            "\n\nFirst, I will show you some examples of research scenarios "
            "and how the researchers approached it."
        )
        for i, ex in enumerate(examples):
            prompt += (
                f"\n\n**Example {i+1}:**\n"
                f"Scenario: {ex['scenario']}\n\n"
                f"Researcher's Plan:\n{ex['solution']}"
            )

    # --- Build conditional sections ---

    # Think block content depends on rubric visibility
    if rubric_items is not None:
        think_content = (
            "Consider each evaluation criterion. Reason about how your plan "
            "will address it, then synthesize into a coherent plan. Only the "
            "content within <solution></solution> tags will be judged so make "
            "sure to include all details in it."
        )
    else:
        think_content = (
            "Carefully analyze the research scenario. Identify key technical "
            "requirements, constraints, and specific criteria an expert "
            "evaluator would likely check. Reason about how your plan "
            "addresses each. Only the content within <solution></solution> "
            "tags will be judged so make sure to include all details in it."
        )

    # --- Assemble main body (explicit strings, no textwrap.dedent) ---
    prompt += (
        f"\nHere is the research scenario.\n"
        f"Scenario: {scenario}\n"
    )

    # Criteria section (only when rubric is visible)
    if rubric_items is not None:
        criteria_lines = "\n".join(f"- {item}" for item in rubric_items)
        prompt += (
            f"\n# Evaluation Criteria\n"
            f"Your plan will be evaluated on the following specific criteria:\n"
            f"{criteria_lines}\n\n"
            f"Consider these criteria as you develop your plan.\n"
        )

    prompt += (
        "\n# Instructions\n"
        "First, come up with a detailed research plan to address the scenario "
        "based on the following overall solution guidelines:\n"
        "- The plan should address the goals of the scenario, and account for "
        "all constraints and confounders.\n"
        "- Do NOT just say WHAT you will do. Explain HOW you will do it and "
        "WHY it is needed. Provide clear explanation and justification for each "
        "proposed step. The solution inside <solution></solution> tags should be "
        "readable for humans, and not in XML itself.\n"
        "- The phrasing should NOT be verbose, and NOT be in past tense, as in "
        "\"the author's approach\" but rather in present tense, as how you would "
        "approach the problem.\n"
        "- Do not claim to have done any experiments or have results, just "
        "provide the plan.\n"
        "- Do not add self-proclaimed praises of your solution. For example do "
        "NOT say yourself it satisfies some desiderata, we will let the "
        "evaluator decide that.\n"
        "\nThen, return the following nested XML block as your output "
        "(always close opened XML tags):\n"
        "<think>\n"
        f"{think_content}\n"
        "</think>\n"
        "<solution>\n"
        "Here is your final research plan. Make sure it is complete and "
        "self-contained. And it should not exceed 750 words.\n"
        "... Your detailed research plan goes here ...\n"
        "</solution>"
    )

    return prompt


# ============================================================
# Signal-Driven Rubric Scheduler
# ============================================================

class RubricScheduler:
    """
    Signal-driven rubric visibility scheduler based on absolute hidden reward.

    p tracks how far hidden (no-rubric) performance is from a target reward.
    As hidden reward improves, p decreases — the model needs less rubric exposure.

    Formula:
      p = p_min + (p_max - p_min) * (target - hidden_ema) / (target - baseline_hidden)

    Lifecycle:
      1. Bootstrap (until hidden EMA has data): p = p_max
      2. First hidden measurement: record baseline_hidden, keep p = p_max
      3. Signal-driven: p scales linearly from p_max (no improvement) to p_min (target reached)
    """

    def __init__(self, p_min: float = 0.0, p_max: float = 0.80, target: float = 0.86, ema_alpha: float = 0.1):
        self.p = p_max  # bootstrap: start at ceiling
        self.p_min = p_min
        self.p_max = p_max
        self.target = target
        self.ema_alpha = ema_alpha
        self.reward_visible_ema: float | None = None
        self.reward_hidden_ema: float | None = None
        self.baseline_hidden: float | None = None  # recorded on first hidden measurement

    def update(self, reward_visible_samples: list[float], reward_hidden_samples: list[float]) -> float:
        """Update p after each batch. Returns the new p value."""

        # Update EMAs
        if reward_visible_samples:
            r_vis = float(np.mean(reward_visible_samples))
            self.reward_visible_ema = (
                r_vis if self.reward_visible_ema is None
                else self.ema_alpha * r_vis + (1 - self.ema_alpha) * self.reward_visible_ema
            )

        if reward_hidden_samples:
            r_hid = float(np.mean(reward_hidden_samples))
            self.reward_hidden_ema = (
                r_hid if self.reward_hidden_ema is None
                else self.ema_alpha * r_hid + (1 - self.ema_alpha) * self.reward_hidden_ema
            )

        # Need hidden EMA before computing signal
        if self.reward_hidden_ema is None:
            return self.p  # stay at p_max

        # First hidden measurement: record baseline, keep p = p_max
        if self.baseline_hidden is None:
            self.baseline_hidden = self.reward_hidden_ema
            return self.p

        # Signal-driven: how far is hidden reward from target?
        denom = self.target - self.baseline_hidden
        if denom <= 0:
            # Baseline already at or above target → go to floor
            self.p = self.p_min
        else:
            remaining = max(0.0, self.target - self.reward_hidden_ema) / denom
            self.p = self.p_min + (self.p_max - self.p_min) * min(1.0, remaining)
        return self.p

    def should_show_rubric(self) -> bool:
        """Per-goal coin flip based on current p."""
        return random.random() < self.p

    @property
    def gap(self) -> float | None:
        """Current distance from target (for logging)."""
        if self.reward_hidden_ema is not None:
            return self.target - self.reward_hidden_ema
        return None

    def state_dict(self) -> dict:
        return {
            "p": self.p,
            "reward_visible_ema": self.reward_visible_ema,
            "reward_hidden_ema": self.reward_hidden_ema,
            "baseline_hidden": self.baseline_hidden,
        }

    def load_state_dict(self, state: dict):
        self.p = state["p"]
        self.reward_visible_ema = state["reward_visible_ema"]
        self.reward_hidden_ema = state["reward_hidden_ema"]
        self.baseline_hidden = state["baseline_hidden"]


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

    # Load dataset
    logger.info("Loading dataset...")
    if config.ml_data:
        data = load_dataset("facebook/research-plan-gen", "ml")
    elif config.arxiv_data:
        data = load_dataset("facebook/research-plan-gen", "arxiv")
    elif config.pubmed_data:
        data = load_dataset("facebook/research-plan-gen", "pubmed")
    else:
        raise ValueError("No dataset selected — set one of ml_data, arxiv_data, or pubmed_data to True")

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

    # Load checkpoint (same logic as best_ver.py)
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
                f"Found {len(checkpoints_with_key)} valid checkpoints with key 'state_path' in {config.log_path}")
            logger.info(f"Using checkpoint: {checkpoints_with_key[config.eval_epoch - 1]}")
        else:
            logger.warning(f"eval_epoch={config.eval_epoch} but no checkpoints with state_path found")
    else:
        resume_info = last_checkpoint
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
        if (resume_info["batch"] + 1) // n_train_batches > 0:
            if not is_eval:
                logger.info(f"Training for epoch: {(resume_info['batch'] + 1) // n_train_batches}")
            actual_batch = resume_info["batch"] + 1
            start_batch = 0
        else:
            actual_batch = resume_info["batch"] + 1
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

    # Sampling parameters
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature
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

    # Initialize rubric scheduler (only used in training mode)
    scheduler = RubricScheduler(
        p_min=config.rubric_prob_min,
        p_max=config.rubric_prob_max,
        target=config.rubric_target_reward,
    )
    if resume_info and "scheduler" in resume_info:
        scheduler.load_state_dict(resume_info["scheduler"])
        logger.info(
            f"RubricScheduler restored: p={scheduler.p:.4f}, "
            f"hidden_ema={scheduler.reward_hidden_ema}, baseline={scheduler.baseline_hidden}"
        )
    elif resume_info:
        # Fallback: reconstruct from last batch_summary entry
        summary_path = os.path.join(config.log_path, "train/batch_summary.jsonl")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                last_line = None
                for line in f:
                    last_line = line
            if last_line:
                last = json.loads(last_line)
                hidden_ema = last.get("rubric/reward_hidden_ema")
                baseline = last.get("rubric/baseline_hidden")
                p = last.get("rubric/p")
                vis_ema = last.get("rubric/reward_visible_ema")
                if hidden_ema is not None and hidden_ema > 0 and baseline is not None and baseline > 0:
                    scheduler.load_state_dict({
                        "p": p,
                        "reward_visible_ema": vis_ema if vis_ema and vis_ema > 0 else None,
                        "reward_hidden_ema": hidden_ema,
                        "baseline_hidden": baseline,
                    })
                    logger.info(
                        f"RubricScheduler reconstructed from batch_summary: p={scheduler.p:.4f}, "
                        f"hidden_ema={scheduler.reward_hidden_ema}, baseline={scheduler.baseline_hidden}"
                    )
                else:
                    logger.info(
                        f"RubricScheduler initialized fresh (no scheduler state in checkpoint or summary): "
                        f"p={scheduler.p:.2f}"
                    )
    else:
        logger.info(
            f"RubricScheduler initialized: p={scheduler.p:.2f}, "
            f"p_min={scheduler.p_min}, p_max={scheduler.p_max}, target={scheduler.target}, "
            f"signal-driven (p tracks hidden reward toward target)"
        )

    logger.info(f"Running for {n_train_batches} batches")

    real_batch = actual_batch  # default in case loop doesn't execute

    # ============================================================
    # Main loop
    # ============================================================
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
                loop_state={"batch": real_batch, "scheduler": scheduler.state_dict()},
            )

        # Get batch
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(dataset))
        batch_rows = dataset.select(range(batch_start, batch_end))

        if not is_eval:
            # Train mode: save sampler checkpoint every batch
            sampling_result = training_client.save_weights_for_sampler(
                name=f"{real_batch:06d}"
            ).result()
            sampling_path = sampling_result.path
            sampling_client = service_client.create_sampling_client(
                model_path=sampling_path
            )

        policy_futures = []
        policy_prompts_tokens = []

        if is_eval:
            logger.info(f"EVAL MODE: Checkpoint {real_batch - 1}, Launching generation for {len(batch_rows)} goals...")
        else:
            logger.info(f"TRAINING MODE: Batch {real_batch}: Launching generation for {len(batch_rows)} goals...")

        # --- PHASE 1: LAUNCH POLICY GENERATIONS (ASYNC) ---
        # Track rubric visibility per goal for reward split
        goal_rubric_visible: list[bool] = []
        n_visible = 0
        n_hidden = 0

        for goal_idx, goal in enumerate(batch_rows["Goal"]):
            rubric = batch_rows["Rubric"][goal_idx]

            if is_eval:
                # Eval mode: rubric always hidden (real task)
                show = False
            else:
                show = scheduler.should_show_rubric()

            goal_rubric_visible.append(show)
            if show:
                n_visible += 1
            else:
                n_hidden += 1

            prompt_text = build_research_plan_prompt(
                scenario=goal,
                rubric_items=rubric if show else None,
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

        if not is_eval:
            logger.info(
                f"  Rubric visible: {n_visible}/{len(batch_rows)} goals "
                f"(p={scheduler.p:.3f})"
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
                proposed_plan = renderers.get_text_content(renderer.parse_response(group_result.tokens)[0])

                # Hard-code solution closure
                if "<solution>" in proposed_plan and "</solution>" not in proposed_plan:
                    proposed_plan = proposed_plan.rstrip() + "\n</solution>"

                # Grader always gets rubric items + reference (same as best_ver.py)
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

        # Batch-level diagnostics
        batch_word_counts = []
        batch_format_penalties = []
        batch_rubric_scores = []

        batch_advantages = []
        num_total_samples = 0
        num_valid_samples = 0

        # Track rewards by rubric visibility for scheduler
        reward_visible_batch: list[float] = []
        reward_hidden_batch: list[float] = []

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
                    dropped_samples += 1
                    continue

                if plan_text.strip() in {"<think>", "</think>", "<solution>", "</solution>"}:
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
                    "rubric_visible": goal_rubric_visible[group_idx],
                    "policy_output": plan_text,
                    "grader_output": xml_text,
                    "rubric_score": rubric_score,
                    "format_penalty": format_penalty,
                    "final_reward": final_reward,
                    "word_count": word_count,
                    "is_compliant": is_compliant,
                })

            # Track rewards by visibility for scheduler
            if group_rewards:
                if goal_rubric_visible[group_idx]:
                    reward_visible_batch.extend(group_rewards)
                else:
                    reward_hidden_batch.extend(group_rewards)

            # GRPO advantage calculation
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
                continue

            if not is_eval:
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
                    training_datums.append(datum)

        # Update rubric scheduler (training only)
        if not is_eval:
            scheduler.update(reward_visible_batch, reward_hidden_batch)

        # --- Batch summary ---
        batch_summary = {
            "batch_idx": real_batch,

            # Reward
            "rubric/sample_mean_all": float(np.mean(batch_rubric_scores)) if batch_rubric_scores else 0.0,
            "rubric/sample_std_all": float(np.std(batch_rubric_scores)) if batch_rubric_scores else 0.0,
            "reward/sample_mean_valid": float(np.mean(batch_sample_rewards)) if batch_sample_rewards else 0.0,
            "reward/sample_std_valid": float(np.std(batch_sample_rewards)) if batch_sample_rewards else 0.0,
            "reward/group_mean_valid": float(np.mean(batch_rewards)) if batch_rewards else 0.0,
            "reward/group_std_valid": float(np.std(batch_rewards)) if batch_rewards else 0.0,

            # Advantage
            "advantage_mean": float(np.mean(batch_advantages)) if batch_advantages else 0.0,
            "advantage_std": float(np.std(batch_advantages)) if batch_advantages else 0.0,

            # Length
            "length_mean": float(np.mean(batch_word_counts)) if batch_word_counts else 0.0,
            "length_p90": float(np.percentile(batch_word_counts, 90)) if batch_word_counts else 0.0,

            # Format
            "format_rate": float(np.mean(batch_format_penalties)) if batch_format_penalties else 0.0,

            # Samples
            "samples/generated": num_total_samples,
            "samples/valid": num_valid_samples,

            # Rubric dropout stats
            "rubric/p": scheduler.p,
            "rubric/visible_count": n_visible,
            "rubric/hidden_count": n_hidden,
            "rubric/reward_visible_ema": scheduler.reward_visible_ema if scheduler.reward_visible_ema is not None else -1.0,
            "rubric/reward_hidden_ema": scheduler.reward_hidden_ema if scheduler.reward_hidden_ema is not None else -1.0,
            "rubric/gap_to_target": scheduler.gap if scheduler.gap is not None else -1.0,
            "rubric/baseline_hidden": scheduler.baseline_hidden if scheduler.baseline_hidden is not None else -1.0,
            "rubric/target": scheduler.target,
            "rubric/reward_visible_batch": float(np.mean(reward_visible_batch)) if reward_visible_batch else -1.0,
            "rubric/reward_hidden_batch": float(np.mean(reward_hidden_batch)) if reward_hidden_batch else -1.0,
        }

        logger.info(
            f"  rubric={batch_summary['rubric/sample_mean_all']:.4f} "
            f"reward={batch_summary['reward/sample_mean_valid']:.4f} "
            f"p={scheduler.p:.3f} "
            f"hidden_ema={scheduler.reward_hidden_ema if scheduler.reward_hidden_ema is not None else 'N/A'} "
            f"gap_to_target={scheduler.gap if scheduler.gap is not None else 'N/A'} "
            f"baseline={scheduler.baseline_hidden if scheduler.baseline_hidden is not None else 'N/A'} "
            f"vis={n_visible} hid={n_hidden}"
        )

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
            logger.info(f"Forward/Backward took {time.time() - t0:.2f}s")

            t1 = time.time()
            _optim_result = optim_step_future.result()
            logger.info(f"Optim step took {time.time() - t1:.2f}s")
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
            "rubric/p": scheduler.p,
            "rubric/gap_to_target": scheduler.gap if scheduler.gap is not None else 0.0,
            "rubric/baseline_hidden": scheduler.baseline_hidden if scheduler.baseline_hidden is not None else 0.0,
            "rubric/target": scheduler.target,
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
