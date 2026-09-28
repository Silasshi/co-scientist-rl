"""Buffer-conditioned Test-Time Training trainer (train_buffer_ttt).

Implements a per-goal optimization loop following the TTT-Discover paper
(Yuksekgonul et al. 2026, arXiv:2601.16175), adapted for research plan
generation with an LLM-based multi-signal reward.

Pipeline (one iteration):
  1. Select K context plans from the buffer (top-K_exploit + K_explore random)
  2. Policy generates M fresh plans conditioned on (goal + context examples)
  3. Grade each plan:
       - Hard gates (Goal-Contrast, Claim Verification) from seven_signal_reward
       - 8 gradient signals × N=2 repeats via ten_signal_reward separate_call
  4. Add graded plans to the buffer
  5. Policy update via entropic objective J_β with adaptive β (KL budget ln 2)

Key differences from train_7signal_vector.py (the GRPO trainer):
  - No vector-advantage normalization — uses scalar entropic weights
  - Buffer-conditioned generation (not independent rollouts)
  - Entropic (max-seeking) objective, not mean-advantage GRPO
  - Single hard-coded target goal (TTT-Discover paper's research problem)

Usage:
    source tools/use_api_profile.sh new
    python src/co_scientist/ttt_discover/train_buffer_ttt.py
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.seven_signal_reward import (
    SIGNAL_1_1_SCORING_PROMPT,
    SIGNAL_1_2_EXTRACTION_PROMPT,
    extract_json,
    programmatic_claim_verification,
)
from co_scientist.shared.ten_signal_reward import (
    SIGNALS as GRADIENT_SIGNALS,
    SIGNAL_WEIGHTS as GRADIENT_WEIGHTS,
    build_single_signal_prompt,
    parse_scores,
    median_over_repeats,
    aggregate_reward as gradient_aggregate,
    normalize_score,
    aggregate_critique_across_repeats,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)


# =============================================================================
# Target goal (TTT-Discover paper's research problem, self-referential)
# =============================================================================

_HERE = Path(__file__).resolve()
_GOAL_PATH = _HERE.parents[3] / "projects" / "ttt_discover" / "analysis" / "sanity_check" / "research_goal.txt"
TARGET_GOAL = _GOAL_PATH.read_text() if _GOAL_PATH.exists() else ""
TARGET_TARGET = (
    "A test-time training method that enables a pretrained LLM to discover new "
    "state-of-the-art solutions on single scientific problems, using a "
    "reward-weighted entropic objective and buffer-based state reuse to "
    "concentrate gradient updates on max-reward samples rather than mean "
    "performance"
)

# Alt goals for Goal-Contrast Margin (must be in the same subdomain but with
# fundamentally different constraint structure)
ALT_GOALS: list[tuple[str, str]] = [
    (
        "Design a reinforcement learning from human feedback (RLHF) method that "
        "reduces reward hacking in code generation, using program execution "
        "outcomes as a verifiable reward signal.",
        "An RLHF training procedure that uses program execution outcomes as a "
        "verifiable reward signal to prevent reward hacking in LLM code "
        "generation, with demonstrated improvements on HumanEval and MBPP.",
    ),
    (
        "Develop an evolutionary search method over a frozen LLM for "
        "mathematical discovery, using hand-crafted mutation operators and a "
        "population-based fitness tracker.",
        "A frozen-LLM evolutionary search system with domain-specific "
        "mutation/crossover operators that beats AlphaEvolve on at least one "
        "mathematical benchmark without any model fine-tuning.",
    ),
]


# =============================================================================
# Config
# =============================================================================


@chz.chz
class Config:
    base_url: str | None = None
    api_profile: str | None = "new"

    log_path: str = "/home/silas/co-scientist-project/projects/ttt_discover/runs/_archive/2026_04_buffer_ttt/v1"
    model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"

    # TTT loop
    n_iterations: int = 50
    samples_per_iter: int = 8      # M: plans generated per iteration
    grader_repeats: int = 2        # N: grader calls per signal per plan (for median)

    # Buffer context selection (UCB-like)
    K_exploit: int = 3             # top-K plans by reward
    K_explore: int = 2             # random plans for diversity
    cold_start_iters: int = 1      # first N iterations: no context (zero-shot)

    # Entropic objective
    kl_budget: float = 0.693       # ln 2 — per TTT-Discover paper
    beta_max: float = 20.0         # clip β search upper bound

    # Optimization
    learning_rate: float = 4e-5
    clip_eps: float = 0.2          # PPO clip — used as the surrogate loss here
    lora_rank: int = 32

    # Sampling
    max_length: int = 32768
    max_tokens: int = 2048         # policy generation max tokens
    grader_max_tokens: int = 4096  # per-signal grader call max tokens (v8.1 rubrics need 4096)
    grader_hard_gate_max_tokens: int = 8192  # hard-gate grader calls are longer
    temperature: float = 1.0
    grader_temperature: float = 0.0

    # Format constraints
    max_word_count: int = 750
    min_words: int = 30

    # Logging / checkpoint
    save_every: int = 5
    today_date: str = time.strftime("%Y-%m-%d", time.localtime())
    seed: int = 0


# =============================================================================
# Buffer data structure
# =============================================================================


@dataclass
class BufferEntry:
    iteration: int
    plan_text: str
    signal_vector: dict[str, int]       # 8 gradient signals: {S1..S8 → 1-5}
    hard_gate_passed: bool
    hard_gate_failed_name: str | None
    aggregate_reward: float              # 0 if hard gate failed
    goal_contrast_diag: dict
    claim_verification_diag: dict
    per_repeat_scores: list[dict[str, int]]  # raw per-repeat grader scores
    word_count: int


# =============================================================================
# Prompt building
# =============================================================================


def build_research_plan_prompt(
    goal: str,
    context: list[BufferEntry] | None = None,
    retrieved_papers: str | None = None,
) -> str:
    """Build the policy's generation prompt with optional buffer context.

    Structure mirrors `build_whole_plan_revision_prompt` so fresh and revised
    plans are drawn from comparable conditional distributions. Any score Δ
    between iter-0 fresh and iter-1 revise then reflects critique-driven
    content change, not scaffolding change (2026-04-17 prompt-consistency fix).
    """
    prompt = f"# Research Goal\n{goal}"

    if retrieved_papers:
        prompt += (
            "\n\n# Relevant Published Papers\n"
            "Use these real papers to ground your proposal with accurate citations.\n\n"
            f"{retrieved_papers}"
        )

    if context:
        prompt += (
            f"\n\n# Past Attempts\n"
            f"Below are {len(context)} past attempts at this goal with their "
            f"evaluation scores (1-5 per dimension)."
        )
        for i, entry in enumerate(context, 1):
            sig_str = " ".join(f"{k.split('_', 1)[0]}={v}" for k, v in entry.signal_vector.items())
            prompt += (
                f"\n\n## Past Attempt {i}  (aggregate reward: {entry.aggregate_reward:.3f})\n"
                f"Signal scores: {sig_str}\n\n"
                f"{entry.plan_text.strip()[:8000]}"
            )
        prompt += "\n\n---"

    prompt += (
        "\n\n# Output Format\n"
        "Write a research plan for the goal above.\n\n"
        "<think>\n"
        "...your reasoning...\n"
        "</think>\n"
        "<solution>\n"
        "...your research plan...\n"
        "</solution>"
    )

    return prompt


# =============================================================================
# Context selection (UCB-like top-K exploit + K_explore random)
# =============================================================================


def select_context(
    buffer: list[BufferEntry],
    K_exploit: int,
    K_explore: int,
    rng: random.Random,
) -> list[BufferEntry]:
    """Select context plans from the buffer.

    Only hard-gate-passing plans are eligible. Returns an empty list if the
    buffer has no valid entries (cold-start case).
    """
    valid = [e for e in buffer if e.hard_gate_passed]
    if not valid:
        return []
    sorted_buf = sorted(valid, key=lambda e: e.aggregate_reward, reverse=True)
    exploit = sorted_buf[:K_exploit]
    remaining = sorted_buf[K_exploit:]
    if len(remaining) <= K_explore:
        explore = list(remaining)
    else:
        explore = rng.sample(remaining, K_explore)
    return exploit + explore


# =============================================================================
# Reward computation: hard gates + gradient signals
# =============================================================================


@dataclass
class TenSignalReward:
    hard_gate_passed: bool
    hard_gate_failed_name: str | None
    signal_vector: dict[str, int | None]
    aggregate_reward: float
    goal_contrast_diag: dict
    claim_verification_diag: dict
    per_repeat_scores: list[dict[str, int]]
    # CR-v7 (2026-04-17): per-signal prose critique emitted by grader when
    # `emit_critique=True`. Empty dict when critiques were not requested
    # (e.g., pre-CR-v7 runs). Key = signal_id, value = 1-3 sentence prose.
    per_signal_critiques: dict[str, str] = field(default_factory=dict)


def launch_hard_gate_futures(
    grader_client, renderer, plan: str, max_tokens: int, temperature: float
) -> dict:
    """Launch the 4 hard-gate grader calls (3 for goal-contrast + 1 for claim verif)."""
    def _launch(prompt: str) -> "tinker.Future":
        convo = [{"role": "user", "content": prompt}]
        model_input = renderer.build_generation_prompt(convo)
        return grader_client.sample(
            model_input,
            num_samples=1,
            sampling_params=tinker.types.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                stop=renderer.get_stop_sequences(),
            ),
        )

    return {
        "gc_target": _launch(
            SIGNAL_1_1_SCORING_PROMPT.format(goal=TARGET_GOAL, target=TARGET_TARGET, plan=plan)
        ),
        "gc_alt1": _launch(
            SIGNAL_1_1_SCORING_PROMPT.format(goal=ALT_GOALS[0][0], target=ALT_GOALS[0][1], plan=plan)
        ),
        "gc_alt2": _launch(
            SIGNAL_1_1_SCORING_PROMPT.format(goal=ALT_GOALS[1][0], target=ALT_GOALS[1][1], plan=plan)
        ),
        "cv": _launch(
            SIGNAL_1_2_EXTRACTION_PROMPT.format(goal=TARGET_GOAL, plan=plan)
        ),
    }


def launch_gradient_futures(
    grader_client, renderer, plan: str, goal: str,
    n_repeats: int, max_tokens: int, temperature: float,
    emit_critique: bool = False,
    grader_client_alt=None, renderer_alt=None,
) -> list[dict[str, "tinker.Future"]]:
    """Launch n_repeats × N_signals grader calls in separate-call mode.

    Returns a list of length n_repeats; each element is a dict
    {signal_id: future}.

    When `emit_critique=True`, the grader prompt asks for a <critique>
    XML block alongside <reasoning> and <score>; CR-v7 per-signal
    context REINFORCE uses this field.

    When `grader_client_alt` / `renderer_alt` are provided, signals with
    `grader_model_override` set are routed to the alt client (v9:
    SA_arithmetic uses Qwen3-235B for stronger math reasoning).
    """
    def _launch(prompt: str, client, rend) -> "tinker.Future":
        convo = [{"role": "user", "content": prompt}]
        model_input = rend.build_generation_prompt(convo)
        return client.sample(
            model_input,
            num_samples=1,
            sampling_params=tinker.types.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                stop=rend.get_stop_sequences(),
            ),
        )

    all_repeats = []
    for _ in range(n_repeats):
        repeat_futures: dict[str, "tinker.Future"] = {}
        for spec in GRADIENT_SIGNALS:
            prompt = build_single_signal_prompt(goal, plan, spec, emit_critique=emit_critique)
            if spec.grader_model_override and grader_client_alt is not None:
                repeat_futures[spec.id] = _launch(prompt, grader_client_alt, renderer_alt)
            else:
                repeat_futures[spec.id] = _launch(prompt, grader_client, renderer)
        all_repeats.append(repeat_futures)
    return all_repeats


def collect_hard_gates(hg_futures: dict, tokenizer, timeout: float = 900.0) -> tuple[bool, str | None, float, dict, dict]:
    """Collect hard-gate futures and decide pass/fail.

    Returns (passed, failed_name, goal_contrast_margin, gc_diag, cv_diag).
    """
    def _decode(future: "tinker.Future") -> str:
        result = future.result(timeout=timeout)
        return tokenizer.decode(result.sequences[0].tokens)

    def _parse_score(text: str) -> float:
        parsed = extract_json(text)
        if parsed is None:
            return 0.0
        raw = parsed.get("score", 0.0)
        try:
            return max(0.0, min(1.0, float(raw)))
        except (TypeError, ValueError):
            return 0.0

    # --- Goal-Contrast Margin ---
    gc_target_score = _parse_score(_decode(hg_futures["gc_target"]))
    gc_alt1_score = _parse_score(_decode(hg_futures["gc_alt1"]))
    gc_alt2_score = _parse_score(_decode(hg_futures["gc_alt2"]))
    margin = gc_target_score - 0.5 * (gc_alt1_score + gc_alt2_score)
    gc_diag = {
        "target": gc_target_score,
        "alt1": gc_alt1_score,
        "alt2": gc_alt2_score,
        "margin": margin,
    }
    # Require meaningful margin (not just barely positive).
    # With a 4B grader, noisy scores can produce margin ~0.01 by chance.
    if margin < 0.10:
        return False, "goal_contrast_margin", margin, gc_diag, {}

    # --- Claim Verification ---
    cv_text = _decode(hg_futures["cv"])
    cv_parsed = extract_json(cv_text)
    cv_score, cv_diag = programmatic_claim_verification(cv_parsed)
    if cv_score < 0.15:
        return False, "claim_verification", margin, gc_diag, cv_diag

    return True, None, margin, gc_diag, cv_diag


def collect_gradient_signals(
    all_repeats: list[dict], tokenizer, timeout: float = 900.0,
) -> tuple[dict[str, int | None], list[dict[str, int]], dict[str, str]]:
    """Collect all repeat × signal futures, parse, and take median per signal.

    Returns (median_scores, per_repeat_scores, per_signal_critiques).

    per_signal_critiques is {signal_id: critique_str} picked from the
    repeat whose score equals the median (CR-v7 aggregator). Empty
    string per signal when grader did not emit the <critique> block.
    """
    per_repeat_scores: list[dict[str, int]] = []
    repeat_as_parsed: list[dict[str, dict]] = []

    for repeat_futures in all_repeats:
        repeat_parsed: dict[str, dict] = {}
        for sig_id, fut in repeat_futures.items():
            try:
                result = fut.result(timeout=timeout)
                raw = tokenizer.decode(result.sequences[0].tokens)
                scores = parse_scores(raw)
                if sig_id in scores:
                    repeat_parsed[sig_id] = scores[sig_id]
            except Exception as e:
                logger.warning(f"Grader call failed for {sig_id}: {type(e).__name__}: {e}")
        repeat_as_parsed.append(repeat_parsed)
        per_repeat_scores.append({
            sid: (info or {}).get("score")
            for sid, info in repeat_parsed.items()
        })

    median_scores = median_over_repeats(repeat_as_parsed)
    per_signal_critiques = aggregate_critique_across_repeats(
        repeat_as_parsed, median_scores,
    )
    return median_scores, per_repeat_scores, per_signal_critiques


@dataclass
class PlanFutures:
    """Futures for one plan: 4 hard-gate calls + n_repeats × 8 gradient calls."""
    hg_futures: dict              # {gc_target, gc_alt1, gc_alt2, cv}
    grad_futures: list[dict]      # list of {signal_id: future}, len = n_repeats


def launch_plan_reward(
    plan: str, goal: str, grader_client, renderer,
    n_repeats: int, grader_max_tokens: int, hg_max_tokens: int, temperature: float,
    skip_hard_gates: bool = False, emit_critique: bool = False,
    grader_client_alt=None, renderer_alt=None,
) -> PlanFutures:
    """Launch all grader futures for a single plan. Does NOT wait."""
    if skip_hard_gates:
        hg = {}  # empty — no hard gate calls
    else:
        hg = launch_hard_gate_futures(grader_client, renderer, plan, hg_max_tokens, temperature)
    grad = launch_gradient_futures(
        grader_client, renderer, plan, goal, n_repeats, grader_max_tokens, temperature,
        emit_critique=emit_critique,
        grader_client_alt=grader_client_alt, renderer_alt=renderer_alt,
    )
    return PlanFutures(hg_futures=hg, grad_futures=grad)


def collect_plan_reward(futures: PlanFutures, tokenizer, skip_hard_gates: bool = False) -> TenSignalReward:
    """Collect all futures for one plan and build the reward record."""
    if skip_hard_gates:
        passed, failed_name, gc_diag, cv_diag = True, None, {}, {}
    else:
        passed, failed_name, _, gc_diag, cv_diag = collect_hard_gates(futures.hg_futures, tokenizer)
    median_scores, per_repeat_scores, per_signal_critiques = collect_gradient_signals(
        futures.grad_futures, tokenizer,
    )

    if not passed:
        return TenSignalReward(
            hard_gate_passed=False,
            hard_gate_failed_name=failed_name,
            signal_vector={s.id: None for s in GRADIENT_SIGNALS},
            aggregate_reward=0.0,
            goal_contrast_diag=gc_diag,
            claim_verification_diag=cv_diag,
            per_repeat_scores=per_repeat_scores,
            per_signal_critiques=per_signal_critiques,
        )

    aggregate = gradient_aggregate(median_scores)
    return TenSignalReward(
        hard_gate_passed=True,
        hard_gate_failed_name=None,
        signal_vector=median_scores,
        aggregate_reward=aggregate,
        goal_contrast_diag=gc_diag,
        claim_verification_diag=cv_diag,
        per_repeat_scores=per_repeat_scores,
        per_signal_critiques=per_signal_critiques,
    )


# =============================================================================
# Entropic objective: adaptive β + weights
# =============================================================================


def _entropy(p: np.ndarray) -> float:
    return float(-np.sum(p * np.log(p + 1e-12)))


def _softmax(rewards: np.ndarray, beta: float) -> np.ndarray:
    x = beta * rewards
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def adaptive_beta(rewards: np.ndarray, kl_budget: float, beta_max: float) -> float:
    """Binary search for β so that KL(softmax(βR) || uniform) ≈ kl_budget.

    KL(p || uniform) = log(n) - H(p). We want H(p) ≈ log(n) - kl_budget.
    """
    n = len(rewards)
    if n <= 1:
        return 0.0
    if float(rewards.std()) < 1e-9:
        return 0.0  # no spread → no β can concentrate
    target_entropy = math.log(n) - kl_budget
    lo, hi = 0.0, beta_max
    for _ in range(50):
        mid = (lo + hi) / 2
        p = _softmax(rewards, mid)
        h = _entropy(p)
        if h > target_entropy:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-4:
            break
    return (lo + hi) / 2


def entropic_weights(rewards: np.ndarray, kl_budget: float, beta_max: float) -> tuple[np.ndarray, float]:
    """Compute softmax weights with adaptive β, scaled so the mean weight is 1.

    This makes the weights directly usable as "advantages" in a PPO-style loss.
    """
    beta = adaptive_beta(rewards, kl_budget, beta_max)
    p = _softmax(rewards, beta)
    weights = p * len(rewards)  # scale so average = 1
    return weights, beta


# =============================================================================
# Format check
# =============================================================================


def check_format_compliance(text: str, max_words: int) -> bool:
    has_open = "<solution>" in text
    has_close = "</solution>" in text
    word_count = len(text.strip().split())
    return has_open and has_close and word_count <= max_words


# =============================================================================
# Main training loop
# =============================================================================


def main(config: Config):
    # --- Setup logging ---
    os.makedirs(config.log_path, exist_ok=True)
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    logger.info(f"TARGET GOAL ({len(TARGET_GOAL)} chars):")
    logger.info(TARGET_GOAL[:400] + "...")
    logger.info(f"Training plan: n_iter={config.n_iterations}, M={config.samples_per_iter}, N_repeat={config.grader_repeats}")
    logger.info(f"Buffer context: K_exploit={config.K_exploit}, K_explore={config.K_explore}")
    logger.info(f"Entropic β: adaptive with KL budget {config.kl_budget:.3f}")

    rng = random.Random(config.seed)

    # --- Clients ---
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Renderer: {renderer_name}")

    service_client = create_service_client(
        base_url=config.base_url,
        api_profile=config.api_profile,
    )

    # Resume from checkpoint if present
    last_checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path)
    if last_checkpoint is None:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_iter = 0
        logger.info(f"Fresh start: creating LoRA client, rank={config.lora_rank}")
    else:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            last_checkpoint["state_path"]
        )
        start_iter = last_checkpoint["batch"] + 1
        logger.info(f"Resuming from iteration {start_iter}")

    grader_client = service_client.create_sampling_client(
        base_model=config.grader_model_name
    )

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )

    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    # --- Buffer ---
    buffer: list[BufferEntry] = []
    buffer_path = os.path.join(config.log_path, "buffer.jsonl")
    # (Resume buffer from disk if exists)
    if os.path.exists(buffer_path):
        with open(buffer_path) as f:
            for line in f:
                d = json.loads(line)
                buffer.append(BufferEntry(**d))
        logger.info(f"Resumed buffer with {len(buffer)} entries")

    train_log_path = os.path.join(config.log_path, "train", "training_logs.jsonl")
    os.makedirs(os.path.dirname(train_log_path), exist_ok=True)
    iter_summary_path = os.path.join(config.log_path, "train", "iter_summary.jsonl")

    # =========================================================================
    # Iteration loop
    # =========================================================================
    for iter_idx in range(start_iter, config.n_iterations):
        t_start = time.time()

        # --- Checkpoint ---
        if config.save_every > 0 and iter_idx > 0 and iter_idx % config.save_every == 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{iter_idx:04d}_{config.today_date}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": iter_idx},
            )

        # --- Save weights for sampler ---
        sampling_result = training_client.save_weights_for_sampler(
            name=f"iter_{iter_idx:04d}"
        ).result()
        sampling_client = service_client.create_sampling_client(model_path=sampling_result.path)

        # --- Select context from buffer (cold-start for first iterations) ---
        if iter_idx < config.cold_start_iters:
            context = []
        else:
            context = select_context(buffer, config.K_exploit, config.K_explore, rng)
        logger.info(
            f"Iter {iter_idx}: buffer_size={len(buffer)}, context_size={len(context)}"
        )

        # --- Build policy prompt with buffer context ---
        prompt_text = build_research_plan_prompt(goal=TARGET_GOAL, context=context)
        model_input = renderer.build_generation_prompt(
            [{"role": "user", "content": prompt_text}]
        )
        prompt_tokens_ints = model_input.to_ints()

        # --- Phase 1: policy generation (M samples in parallel) ---
        policy_future = sampling_client.sample(
            prompt=model_input,
            num_samples=config.samples_per_iter,
            sampling_params=sampling_params,
        )

        gen_result = policy_future.result()
        generated_samples = []
        for seq in gen_result.sequences:
            text = renderer.parse_response(seq.tokens)[0]["content"]
            if "<solution>" in text and "</solution>" not in text:
                text = text.rstrip() + "\n</solution>"
            generated_samples.append({"tokens": seq.tokens, "logprobs": seq.logprobs, "text": text})
        logger.info(f"Iter {iter_idx}: generated {len(generated_samples)} plans")

        # --- Phase 2a: launch all grader futures (parallel across plans) ---
        # For each valid plan, launch (4 hard-gate + n_repeats * 8 gradient) futures.
        # All M * (4 + n_repeats*8) futures are dispatched before any collection.
        plan_futures: list[PlanFutures | None] = []
        n_launched = 0
        for s_idx, sample in enumerate(generated_samples):
            plan_text = sample["text"]
            word_count = len(plan_text.strip().split())
            if word_count < config.min_words:
                logger.warning(f"Iter {iter_idx} sample {s_idx}: dropped (word_count={word_count})")
                plan_futures.append(None)
                continue
            plan_futures.append(
                launch_plan_reward(
                    plan=plan_text,
                    goal=TARGET_GOAL,
                    grader_client=grader_client,
                    renderer=renderer,
                    n_repeats=config.grader_repeats,
                    grader_max_tokens=config.grader_max_tokens,
                    hg_max_tokens=config.grader_hard_gate_max_tokens,
                    temperature=config.grader_temperature,
                )
            )
            n_launched += 1

        total_grader_calls = n_launched * (4 + config.grader_repeats * len(GRADIENT_SIGNALS))
        logger.info(
            f"Iter {iter_idx}: launched {total_grader_calls} grader futures "
            f"across {n_launched} plans. Collecting..."
        )

        # --- Phase 2b: collect all grader futures ---
        plan_rewards: list[TenSignalReward | None] = []
        t_collect = time.time()
        for s_idx, futures in enumerate(plan_futures):
            if futures is None:
                plan_rewards.append(None)
                continue
            try:
                reward = collect_plan_reward(futures, tokenizer)
                plan_rewards.append(reward)
            except Exception as e:
                err_str = str(e) or type(e).__name__
                logger.error(f"Iter {iter_idx} sample {s_idx}: reward collection failed: {err_str}")
                plan_rewards.append(None)
        logger.info(f"Iter {iter_idx}: collection complete in {time.time() - t_collect:.0f}s")

        # --- Phase 3: add to buffer ---
        new_entries = 0
        for s_idx, (sample, reward) in enumerate(zip(generated_samples, plan_rewards)):
            if reward is None:
                continue
            entry = BufferEntry(
                iteration=iter_idx,
                plan_text=sample["text"],
                signal_vector={k: v for k, v in reward.signal_vector.items() if v is not None},
                hard_gate_passed=reward.hard_gate_passed,
                hard_gate_failed_name=reward.hard_gate_failed_name,
                aggregate_reward=reward.aggregate_reward,
                goal_contrast_diag=reward.goal_contrast_diag,
                claim_verification_diag=reward.claim_verification_diag,
                per_repeat_scores=reward.per_repeat_scores,
                word_count=len(sample["text"].strip().split()),
            )
            buffer.append(entry)
            new_entries += 1
            # Persist to disk
            with open(buffer_path, "a") as f:
                f.write(json.dumps(entry.__dict__, ensure_ascii=False) + "\n")

        # --- Phase 4: entropic weights + build training datums ---
        valid_samples = [
            (s, r) for s, r in zip(generated_samples, plan_rewards) if r is not None
        ]
        if len(valid_samples) < 2:
            logger.warning(f"Iter {iter_idx}: {len(valid_samples)} valid samples (<2), skipping update")
            continue

        rewards_arr = np.array([r.aggregate_reward for _, r in valid_samples], dtype=np.float64)
        weights, beta = entropic_weights(rewards_arr, config.kl_budget, config.beta_max)
        logger.info(
            f"Iter {iter_idx}: rewards={rewards_arr.tolist()}, β={beta:.3f}, "
            f"weights_min={weights.min():.3f} max={weights.max():.3f}"
        )

        training_datums = []
        for (sample, reward), weight in zip(valid_samples, weights):
            prompt_tokens = [int(t) for t in prompt_tokens_ints]
            generated_tokens = [int(t) for t in sample["tokens"]]
            full_seq = prompt_tokens + generated_tokens
            ob_len = len(prompt_tokens) - 1

            input_tokens = full_seq[:-1]
            target_tokens = full_seq[1:]
            all_logprobs = [0.0] * ob_len + sample["logprobs"]
            # Use the entropic weight as the per-token advantage (centered at 1 on average)
            all_advantages = [0.0] * ob_len + [float(weight)] * len(sample["logprobs"])

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

        # --- Phase 5: optimization step ---
        try:
            # importance_sampling = weighted log-likelihood (REINFORCE).
            # Entropic weights w_β are passed as advantages.
            fwd_bwd_future = training_client.forward_backward(
                training_datums,
                loss_fn="importance_sampling",
            )
            optim_step_future = training_client.optim_step(adam_params)
            _ = fwd_bwd_future.result()
            _ = optim_step_future.result()
        except Exception:
            logger.exception(f"Iter {iter_idx}: training step failed")
            continue

        # --- Phase 6: log summary ---
        valid_rewards = [r for _, r in valid_samples]
        passed_rewards = [r.aggregate_reward for r in valid_rewards if r.hard_gate_passed]
        n_hard_gate_failed = sum(1 for r in valid_rewards if not r.hard_gate_passed)
        per_signal_means = {}
        for spec in GRADIENT_SIGNALS:
            scores = [r.signal_vector.get(spec.id) for r in valid_rewards if r.hard_gate_passed]
            scores = [s for s in scores if s is not None]
            if scores:
                per_signal_means[f"signal/{spec.id}/mean"] = float(np.mean(scores))
                per_signal_means[f"signal/{spec.id}/std"] = float(np.std(scores))

        iter_summary = {
            "iter": iter_idx,
            "time/total": time.time() - t_start,
            "samples/generated": len(generated_samples),
            "samples/valid": len(valid_samples),
            "samples/hard_gate_failed": n_hard_gate_failed,
            "reward/mean": float(rewards_arr.mean()),
            "reward/std": float(rewards_arr.std()),
            "reward/max": float(rewards_arr.max()),
            "reward/min": float(rewards_arr.min()),
            "reward/buffer_max": max((e.aggregate_reward for e in buffer), default=0.0),
            "beta": beta,
            "weights/min": float(weights.min()),
            "weights/max": float(weights.max()),
            "buffer/size": len(buffer),
            "buffer/new_entries": new_entries,
            "context/size": len(context),
            **per_signal_means,
        }

        with open(iter_summary_path, "a") as f:
            f.write(json.dumps(iter_summary) + "\n")

        # Per-sample logs
        with open(train_log_path, "a") as f:
            for s_idx, (sample, reward) in enumerate(zip(generated_samples, plan_rewards)):
                if reward is None:
                    f.write(json.dumps({
                        "iter": iter_idx, "sample_idx": s_idx,
                        "dropped": True, "reason": "too_short",
                        "word_count": len(sample["text"].strip().split()),
                    }) + "\n")
                    continue
                f.write(json.dumps({
                    "iter": iter_idx,
                    "sample_idx": s_idx,
                    "text": sample["text"],
                    "word_count": len(sample["text"].strip().split()),
                    "aggregate_reward": reward.aggregate_reward,
                    "signal_vector": reward.signal_vector,
                    "hard_gate_passed": reward.hard_gate_passed,
                    "hard_gate_failed_name": reward.hard_gate_failed_name,
                }, ensure_ascii=False) + "\n")

        ml_logger.log_metrics(iter_summary, step=iter_idx)
        logger.info(
            f"Iter {iter_idx}: reward mean={iter_summary['reward/mean']:.3f} "
            f"max={iter_summary['reward/max']:.3f} buffer_max={iter_summary['reward/buffer_max']:.3f} "
            f"(β={beta:.2f}) time={iter_summary['time/total']:.1f}s"
        )

    # --- Final checkpoint ---
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=f"{iter_idx:04d}_final_{config.today_date}",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": iter_idx},
    )
    ml_logger.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    chz.entrypoint(main)
