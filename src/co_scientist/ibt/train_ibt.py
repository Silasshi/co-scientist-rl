"""
Iterative Brainstorming Training (IBT).

Two modes for the same iterative loop:
  - single_chain : 1 sample per turn, running-baseline advantage
  - mini_grpo    : G samples per turn, group-relative advantage

Pipeline per goal (K turns):
  Turn 0: prompt = research_goal
  Turn k: prompt = research_goal + hint_{k-1}

  Each turn:
    1. Policy generates plan(s)
    2. Grader scores (rubric) + generates improvement hint
    3. Compute advantage  (mode-dependent)
    4. PPO update
    5. Feed hint into next turn

After K turns on one goal, move to next goal (weights carry over).

Usage:
  # Single-chain
  python train_ibt.py mode=single_chain num_turns=5 num_goals=20 api_profile=NEW

  # Mini-GRPO
  python train_ibt.py mode=mini_grpo group_size=4 num_turns=5 num_goals=20 api_profile=NEW
"""

import json
import logging
import os
import re
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

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


# ============================================================
# Config
# ============================================================

@chz.chz
class Config:
    base_url: str | None = None
    api_profile: str | None = "NEW"
    log_path: str = "/home/silas/co-scientist-project/runs/2026/4/ibt/1"

    # Model
    policy_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    grader_model: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 32

    # Load shared init checkpoint (empty string = fresh LoRA)
    init_checkpoint: str = ""

    # Mode: "single_chain" or "mini_grpo"
    mode: str = "single_chain"

    # Disable hint feedback into next turn's prompt (for SC-no-hint ablation)
    use_hint_in_prompt: bool = True

    # Dataset
    ml_data: bool = True
    arxiv_data: bool = False
    pubmed_data: bool = False

    # IBT loop parameters
    num_goals: int = 20         # How many goals to train on sequentially
    start_goal: int = 0         # Resume from this goal index
    num_turns: int = 5          # K turns per goal
    group_size: int = 4         # Only used in mini_grpo mode

    # Training
    learning_rate: float = 1e-5
    clip_eps: float = 0.2

    # Generation
    max_tokens: int = 2048
    temperature: float = 1.0
    grader_max_tokens: int = 8192
    grader_temperature: float = 0.0

    # Reward
    max_word_count: int = 750
    target_word_count: int = 600
    scale_length_bonus: float = 140.0
    scaling_factor: float = 0.08
    min_words: int = 30

    # Running baseline (single_chain mode)
    baseline_decay: float = 0.9     # EMA decay for running baseline

    # OPD (Online Policy Distillation)
    use_opd: bool = False
    w_rl: float = 1.0              # Weight for RL advantage
    w_opd: float = 1.0             # Weight for OPD signal
    opd_clip: float = 5.0          # Clip range for per-token OPD advantage

    today_date: str = time.strftime("%Y-%m-%d", time.localtime())


# ============================================================
# Prompts
# ============================================================

def build_plan_prompt(scenario: str, hint: str | None = None) -> str:
    """Build the policy model's input prompt, optionally including a hint."""
    prompt = textwrap.dedent(f"""\
        I will provide you a research scenario. You have to provide me a concise yet thoughtful research plan with all details needed to execute it.

        Here is the research scenario.
        Scenario: {scenario}
    """).strip()

    if hint:
        prompt += textwrap.dedent(f"""

        A reviewer has provided the following feedback on a previous version of the plan. Please use this feedback to improve your plan:
        {hint}
        """).strip()

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


def build_grader_with_hint_prompt(
    scenario: str,
    rubric_items: list[str],
    proposed_plan: str,
    reference_solution: str | None = None,
) -> str:
    """
    Canonical grader prompt (identical to best_ver.py build_grader_prompt)
    with an appended <improvement_hint> request.

    The rubric section is kept verbatim so scores are comparable across
    experiments.  The hint section is appended AFTER the rubric instructions.
    """
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

        After the rubric evaluation above, provide specific improvement suggestions in the following block:

        <improvement_hint>
        Identify the TOP 3 most important weaknesses in the plan and provide specific, actionable suggestions for improvement. For each weakness:
        1. State the specific weakness
        2. Explain exactly what should be changed or added
        3. Give a concrete example of what the improved version should look like

        Be specific — do not give generic advice like "be more detailed". Instead, point to exact parts of the plan and suggest concrete changes.
        </improvement_hint>
    """).strip()

    return prompt


# ============================================================
# Scoring & Parsing
# ============================================================

def compute_rubric_score(xml_text: str) -> float:
    """Compute rubric score from grader XML output. Returns float in [0, 1]."""
    level_map = {0: 0.0, 1: 0.2, 2: 0.6, 3: 1.0}

    item_blocks = re.findall(
        r"<item\s+num=.*?>.*?</item>",
        xml_text,
        flags=re.DOTALL,
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
        mapped = [level_map.get(l, 0.0) for l in levels]
        item_scores.append(sum(mapped) / len(mapped))

    if not item_scores:
        return 0.0
    return sum(item_scores) / len(item_scores)


def extract_hint(grader_output: str) -> str:
    """Extract the improvement hint from grader output."""
    match = re.search(
        r"<improvement_hint>\s*(.*?)\s*</improvement_hint>",
        grader_output,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # Fallback: use text after </rubric> as hint
    match2 = re.search(r"</rubric>\s*(.*)", grader_output, flags=re.DOTALL)
    if match2 and len(match2.group(1).strip()) > 20:
        return match2.group(1).strip()[:2000]

    return ""


def extract_solution_text(plan_text: str) -> str:
    """Extract text inside <solution> tags for word count / format check."""
    match = re.search(
        r"<solution>\s*(.*?)\s*</solution>",
        plan_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else plan_text.strip()


def compute_reward(plan_text: str, rubric_score: float, config: Config) -> tuple[float, int]:
    """Compute reward and word count. Single regex extraction."""
    solution_text = extract_solution_text(plan_text)
    word_count = len(solution_text.split())

    is_compliant = word_count <= config.max_word_count
    excess = max(0, word_count - config.max_word_count)
    format_penalty = 0.0 if is_compliant else 0.2 + 0.0005 * excess

    length_bonus = np.exp(
        -((word_count - config.target_word_count) / config.scale_length_bonus) ** 2
    )
    reward = rubric_score + config.scaling_factor * length_bonus - format_penalty
    return reward, word_count


# ============================================================
# OPD: Teacher logprobs
# ============================================================

def build_hint_augmented_prompt(scenario: str, prev_hint: str | None, current_hint: str) -> str:
    """Build the teacher's prompt for OPD: same as policy prompt but with current hint appended.

    The teacher sees both the previous hint (same as policy) AND the current grader
    feedback, so its logprobs reflect "what a model knowing the feedback would say."
    """
    prompt = build_plan_prompt(scenario=scenario, hint=prev_hint)
    prompt += textwrap.dedent(f"""

        [Additional reviewer feedback on this specific plan — use this to guide your response:]
        {current_hint}
    """).strip()
    return prompt


def compute_per_token_advantages(
    rl_advantage: float,
    policy_logprobs: list[float],
    teacher_logprobs: list[float],
    config: Config,
) -> list[float]:
    """Combine RL scalar advantage with per-token OPD signal.

    per_token[i] = w_rl * rl_advantage + w_opd * clip(teacher_lp[i] - policy_lp[i])
    """
    n = min(len(policy_logprobs), len(teacher_logprobs))
    per_token = []
    for i in range(n):
        opd_signal = teacher_logprobs[i] - policy_logprobs[i]
        opd_clipped = np.clip(opd_signal, -config.opd_clip, config.opd_clip)
        per_token.append(config.w_rl * rl_advantage + config.w_opd * opd_clipped)
    # If policy has more tokens than teacher logprobs, fill with RL-only
    for _ in range(len(policy_logprobs) - n):
        per_token.append(config.w_rl * rl_advantage)
    return per_token


# ============================================================
# Training datum creation
# ============================================================

def create_training_datum(
    prompt_tokens: list[int],
    generated_tokens: list[int],
    logprobs: list[float],
    advantages: float | list[float],
) -> types.Datum:
    """Create a PPO training datum.

    advantages: scalar (broadcast to all tokens) or per-token list (when OPD is used).
    """
    full_seq = prompt_tokens + generated_tokens
    ob_len = len(prompt_tokens) - 1

    input_tokens = full_seq[:-1]
    target_tokens = full_seq[1:]

    all_logprobs = [0.0] * ob_len + logprobs

    if isinstance(advantages, list):
        all_advantages = [0.0] * ob_len + advantages
    else:
        all_advantages = [0.0] * ob_len + [advantages] * len(logprobs)

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


# ============================================================
# Main training loop
# ============================================================

def main(config: Config):
    assert config.mode in ("single_chain", "mini_grpo"), \
        f"Unknown mode: {config.mode}. Use 'single_chain' or 'mini_grpo'."

    # ── Setup ──
    variant = config.mode + ("_opd" if config.use_opd else "") + ("_nohint" if not config.use_hint_in_prompt else "")
    run_dir = os.path.join(config.log_path, variant)
    os.makedirs(run_dir, exist_ok=True)
    train_dir = os.path.join(run_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    ml_logger = ml_log.setup_logging(
        log_dir=run_dir,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    logger.info(f"IBT mode: {config.mode}, OPD: {config.use_opd}")
    logger.info(f"Policy model: {config.policy_model}")
    logger.info(f"Grader model: {config.grader_model}")
    logger.info(f"Goals: {config.num_goals}, Turns/goal: {config.num_turns}")
    if config.mode == "mini_grpo":
        logger.info(f"Group size: {config.group_size}")
    if config.use_opd:
        logger.info(f"OPD: w_rl={config.w_rl}, w_opd={config.w_opd}, clip={config.opd_clip}")

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({
            "mode": config.mode,
            "policy_model": config.policy_model,
            "grader_model": config.grader_model,
            "num_goals": config.num_goals,
            "num_turns": config.num_turns,
            "group_size": config.group_size if config.mode == "mini_grpo" else 1,
            "learning_rate": config.learning_rate,
            "lora_rank": config.lora_rank,
            "baseline_decay": config.baseline_decay if config.mode == "single_chain" else None,
            "use_opd": config.use_opd,
            "w_rl": config.w_rl,
            "w_opd": config.w_opd,
            "opd_clip": config.opd_clip,
        }, f, indent=2)

    # ── Tokenizer & renderer (separate for policy vs grader) ──
    tokenizer = get_tokenizer(config.policy_model)
    renderer_name = model_info.get_recommended_renderer_name(config.policy_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Policy renderer: {renderer_name}")

    grader_tokenizer = get_tokenizer(config.grader_model)
    grader_renderer_name = model_info.get_recommended_renderer_name(config.grader_model)
    grader_renderer = renderers.get_renderer(grader_renderer_name, grader_tokenizer)
    logger.info(f"Grader renderer: {grader_renderer_name}")

    # ── Dataset ──
    logger.info("Loading dataset...")
    if config.ml_data:
        data = load_dataset("facebook/research-plan-gen", "ml")
    elif config.arxiv_data:
        data = load_dataset("facebook/research-plan-gen", "arxiv")
    elif config.pubmed_data:
        data = load_dataset("facebook/research-plan-gen", "pubmed")
    assert isinstance(data, datasets.DatasetDict)
    dataset = data["train"]

    # Limit to num_goals
    num_goals = min(config.num_goals, len(dataset))
    logger.info(f"Using {num_goals} goals from training set")

    # ── Clients ──
    service_client = create_service_client(
        base_url=config.base_url,
        api_profile=config.api_profile,
    )

    if config.init_checkpoint:
        logger.info(f"Loading shared init checkpoint: {config.init_checkpoint}")
        ckpt = checkpoint_utils.get_last_checkpoint(config.init_checkpoint)
        assert ckpt is not None, f"No checkpoint found at {config.init_checkpoint}"
        training_client = service_client.create_training_client_from_state_with_optimizer(
            ckpt["state_path"]
        )
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.policy_model, rank=config.lora_rank
        )

    # Grader client also serves as OPD teacher (same 30B model)
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

    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    # ── Running baseline (single_chain mode) ──
    running_baseline = None  # Initialized from first actual reward

    # ── Log files (kept open for entire training) ──
    summary_path = os.path.join(train_dir, "batch_summary.jsonl")
    logs_path = os.path.join(train_dir, "training_logs.jsonl")
    f_summary = open(summary_path, "a")
    f_logs = open(logs_path, "a")

    # ── Global step counter ──
    global_step = config.start_goal * config.num_turns
    t_start = time.time()

    if config.start_goal > 0:
        logger.info(f"Resuming from goal {config.start_goal}")

    # ============================================================
    # Main loop: iterate over goals sequentially
    # ============================================================
    for goal_idx in range(config.start_goal, num_goals):
        goal_row = dataset[goal_idx]
        goal_text = goal_row["Goal"]
        rubric_items = goal_row["Rubric"]
        ref_solution = goal_row["Reference solution"]

        logger.info(f"\n{'='*60}")
        logger.info(f"GOAL {goal_idx}/{num_goals}: {goal_text[:100]}...")
        logger.info(f"{'='*60}")

        hint = None  # No hint for first turn
        goal_turn_scores = []

        # ── Iterate K turns on this goal ──
        for turn_idx in range(config.num_turns):
            turn_t0 = time.time()
            logger.info(f"  Turn {turn_idx}/{config.num_turns} (goal {goal_idx})")

            # ── 1. Save weights for sampling ──
            sr = training_client.save_weights_for_sampler(
                name=f"g{goal_idx:04d}_t{turn_idx:02d}"
            ).result()
            sampling_client = service_client.create_sampling_client(
                model_path=sr.path
            )

            # ── 2. Build prompt ──
            prompt_hint = hint if config.use_hint_in_prompt else None
            prompt_text = build_plan_prompt(scenario=goal_text, hint=prompt_hint)
            convo = [{"role": "user", "content": prompt_text}]
            model_input = renderer.build_generation_prompt(convo)
            prompt_tokens = [int(t) for t in model_input.to_ints()]

            # ── 3. Generate plan(s) ──
            n_samples = config.group_size if config.mode == "mini_grpo" else 1

            sample_result = sampling_client.sample(
                prompt=model_input,
                num_samples=n_samples,
                sampling_params=sampling_params,
            ).result()

            # ── 4. Parse plans & launch grading ──
            plans = []
            grader_futures = []

            for seq in sample_result.sequences:
                plan_text = renderers.get_text_content(
                    renderer.parse_response(seq.tokens)[0]
                )
                # Force-close solution tag if truncated
                if "<solution>" in plan_text and "</solution>" not in plan_text:
                    plan_text = plan_text.rstrip() + "\n</solution>"

                plans.append({
                    "text": plan_text,
                    "tokens": seq.tokens,
                    "logprobs": seq.logprobs,
                })

                # Launch grader
                grader_prompt = build_grader_with_hint_prompt(
                    scenario=goal_text,
                    rubric_items=rubric_items,
                    proposed_plan=plan_text,
                    reference_solution=ref_solution,
                )
                grader_input = grader_renderer.build_generation_prompt(
                    [{"role": "user", "content": grader_prompt}]
                )
                grader_futures.append(
                    grader_client.sample(
                        grader_input,
                        num_samples=1,
                        sampling_params=grader_sampling_params,
                    )
                )

            # ── 5. Collect grades ──
            valid_plans = []

            for k, gf in enumerate(grader_futures):
                grader_result = gf.result()
                grader_text = renderers.get_text_content(
                    grader_renderer.parse_response(grader_result.sequences[0].tokens)[0]
                )

                rubric_score = compute_rubric_score(grader_text)
                plan_text = plans[k]["text"]
                reward, word_count = compute_reward(plan_text, rubric_score, config)
                plan_hint = extract_hint(grader_text)

                # Skip degenerate samples
                if word_count < config.min_words:
                    logger.warning(f"    Dropping sample {k}: too short ({word_count} words)")
                    continue

                valid_plans.append({
                    **plans[k],
                    "reward": reward,
                    "rubric_score": rubric_score,
                    "hint": plan_hint,
                    "word_count": word_count,
                })

            if not valid_plans:
                logger.warning(f"    No valid plans at turn {turn_idx}, skipping update")
                continue

            # ── 6. Compute advantages ──
            rewards = [vp["reward"] for vp in valid_plans]
            rubric_scores = [vp["rubric_score"] for vp in valid_plans]

            if config.mode == "single_chain":
                r = rewards[0]
                if running_baseline is None:
                    running_baseline = r
                    advantage = 0.0
                else:
                    advantage = r - running_baseline
                running_baseline = (
                    config.baseline_decay * running_baseline
                    + (1 - config.baseline_decay) * r
                )
                advantages = [advantage]
                logger.info(
                    f"    Single-chain: reward={r:.3f}, baseline={running_baseline:.3f}, "
                    f"advantage={advantage:.3f}"
                )
            else:
                mean_reward = np.mean(rewards)
                advantages = [r - mean_reward for r in rewards]

                if all(a == 0.0 for a in advantages):
                    logger.info(f"    All advantages zero, skipping update")
                    best_idx = int(np.argmax(rubric_scores))
                    best_hint = valid_plans[best_idx]["hint"]
                    if best_hint:
                        hint = best_hint
                    goal_turn_scores.append(float(np.mean(rubric_scores)))
                    continue

                logger.info(
                    f"    Mini-GRPO: mean_reward={mean_reward:.3f}, "
                    f"advantages={[f'{a:.3f}' for a in advantages]}"
                )

            # ── 6b. OPD: compute teacher logprobs and per-token advantages ──
            best_idx = int(np.argmax(rubric_scores))
            opd_stats = []
            if config.use_opd:
                current_hint = valid_plans[best_idx]["hint"]

                if current_hint:
                    teacher_prompt = build_hint_augmented_prompt(
                        scenario=goal_text,
                        prev_hint=hint,
                        current_hint=current_hint,
                    )
                    teacher_prompt_tokens = [int(t) for t in
                        grader_renderer.build_generation_prompt(
                            [{"role": "user", "content": teacher_prompt}]
                        ).to_ints()]
                    teacher_prompt_len = len(teacher_prompt_tokens)

                    # Launch all teacher logprob calls in parallel
                    teacher_futures = {}
                    for k, vp in enumerate(valid_plans[:len(advantages)]):
                        full_seq = teacher_prompt_tokens + [int(t) for t in vp["tokens"]]
                        future = grader_client.compute_logprobs(
                            types.ModelInput.from_ints(tokens=full_seq)
                        )
                        teacher_futures[k] = future

                    # Collect results
                    for k, future in teacher_futures.items():
                        vp = valid_plans[k]
                        try:
                            all_lps = future.result()
                            teacher_lps = list(all_lps[teacher_prompt_len: teacher_prompt_len + len(vp["tokens"])])
                            if len(teacher_lps) < len(vp["tokens"]):
                                teacher_lps.extend([0.0] * (len(vp["tokens"]) - len(teacher_lps)))

                            vp["per_token_advantages"] = compute_per_token_advantages(
                                rl_advantage=advantages[k],
                                policy_logprobs=vp["logprobs"],
                                teacher_logprobs=teacher_lps,
                                config=config,
                            )
                            n = min(len(vp["logprobs"]), len(teacher_lps))
                            opd_diffs = [teacher_lps[i] - vp["logprobs"][i] for i in range(n)]
                            opd_stats.append(float(np.mean(np.abs(opd_diffs))))
                        except Exception as e:
                            logger.warning(f"    OPD teacher logprobs failed for sample {k}: {e}")
                            vp["per_token_advantages"] = None
                else:
                    logger.info("    OPD: no hint available, using RL-only advantages")

            # ── 7. Create training datums ──
            training_datums = []
            for k, vp in enumerate(valid_plans):
                if k >= len(advantages):
                    break
                per_token = vp.get("per_token_advantages")
                adv = per_token if per_token is not None else advantages[k]
                datum = create_training_datum(
                    prompt_tokens=prompt_tokens,
                    generated_tokens=[int(t) for t in vp["tokens"]],
                    logprobs=vp["logprobs"],
                    advantages=adv,
                )
                training_datums.append(datum)

            # ── 8. PPO update (skip if all advantages are zero) ──
            if training_datums and not all(a == 0.0 for a in advantages[:len(training_datums)]):
                try:
                    fwd_bwd = training_client.forward_backward(
                        training_datums,
                        loss_fn="ppo",
                        loss_fn_config={
                            "clip_low_threshold": 1 - config.clip_eps,
                            "clip_high_threshold": 1 + config.clip_eps,
                        },
                    )
                    optim = training_client.optim_step(adam_params)
                    fwd_bwd.result()
                    optim.result()
                    logger.info(f"    PPO update done ({len(training_datums)} datums)")
                except Exception as e:
                    logger.exception(f"    Training step failed: {e}")

            # ── 9. Pick best plan's hint for next turn ──
            # best_idx already computed in step 6b
            best_hint = valid_plans[best_idx]["hint"]
            if best_hint:
                hint = best_hint
            # If no hint extracted, keep previous hint

            # ── 10. Log ──
            turn_mean_rubric = float(np.mean(rubric_scores))
            turn_mean_reward = float(np.mean(rewards))
            goal_turn_scores.append(turn_mean_rubric)

            turn_summary = {
                "global_step": global_step,
                "goal_idx": goal_idx,
                "turn_idx": turn_idx,
                "mode": config.mode,
                "rubric/mean": turn_mean_rubric,
                "rubric/best": float(max(rubric_scores)),
                "reward/mean": turn_mean_reward,
                "reward/best": float(max(rewards)),
                "n_samples": len(valid_plans),
                "n_datums": len(training_datums),
                "time": time.time() - turn_t0,
            }
            if config.mode == "single_chain":
                turn_summary["baseline"] = running_baseline
                turn_summary["advantage"] = advantages[0]
            else:
                turn_summary["advantage/mean"] = float(np.mean(advantages))
                turn_summary["advantage/std"] = float(np.std(advantages))
            if config.use_opd and opd_stats:
                turn_summary["opd/mean_kl"] = float(np.mean(opd_stats))
                turn_summary["opd/n_computed"] = len(opd_stats)

            f_summary.write(json.dumps(turn_summary) + "\n")

            # Per-sample logs
            for k, vp in enumerate(valid_plans):
                sample_log = {
                    "global_step": global_step,
                    "goal_idx": goal_idx,
                    "turn_idx": turn_idx,
                    "sample_idx": k,
                    "rubric_score": vp["rubric_score"],
                    "reward": vp["reward"],
                    "word_count": vp["word_count"],
                    "hint_length": len(vp["hint"]),
                    "policy_output": vp["text"][:500],
                    "hint": vp["hint"][:500],
                }
                f_logs.write(json.dumps(sample_log) + "\n")

            global_step += 1

            ml_logger.log_metrics({
                "progress/goal": goal_idx,
                "progress/turn": turn_idx,
                "progress/global_step": global_step,
                "rubric/turn_mean": turn_mean_rubric,
                "reward/turn_mean": turn_mean_reward,
                "time/total": time.time() - t_start,
            }, step=global_step)

        # ── End of goal: log goal summary ──
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

    # ── Close log files ──
    f_summary.close()
    f_logs.close()

    # ── Save final checkpoint ──
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=f"final_{num_goals}goals_{config.today_date}",
        log_path=run_dir,
        kind="state",
        loop_state={"goal": num_goals - 1, "global_step": global_step},
    )
    logger.info(f"Training complete. {global_step} total steps across {num_goals} goals.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    chz.nested_entrypoint(main)
