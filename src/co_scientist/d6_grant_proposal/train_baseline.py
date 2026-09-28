"""D6 frozen baseline — frozen Qwen3-30B-A3B + oracle, no training.

Generates n_eval_plans proposals using the frozen base model with the student
prompt (goal + oracle abstraction). Scores each plan with the 12-signal grant
reward and writes buffer.jsonl + metrics.jsonl in the same schema as train_opd.py,
so the same audit.py analysis tools apply.

This is the baseline arm of the paper: what you get from the base model + oracle
without any gradient updates. All OPD/GRPO runs must beat the baseline on audit
to justify the training cost.

Usage:
    PYTHONPATH=src python -m co_scientist.d6_grant_proposal.train_baseline \\
        goal_domain=ai goal_name=02_foundational_rl \\
        log_path=projects/d6_grant_proposal/runs/2026_04_30_pilot_foundrl_baseline
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

import chz
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.grant_signal_reward import (
    SIGNALS, SIGNAL_VARIANTS, aggregate_reward, build_single_call_prompt, parse_scores,
)
from co_scientist.d6_grant_proposal.prompts import build_student_prompt, extract_solution

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _active_signals(goal_domain: str) -> list:
    non_stem = {"social_science"}
    if goal_domain in non_stem:
        return [
            SIGNAL_VARIANTS["G12a_analytical_framework"]
            if s.id == "G12_formalism" else s
            for s in SIGNALS
        ]
    return list(SIGNALS)


@chz.chz
class Config:
    api_profile: str | None = "new"
    base_url: str | None = None

    goal_domain: str = "ai"
    goal_name: str = "02_foundational_rl"

    log_path: str = "projects/d6_grant_proposal/runs/2026_04_30_pilot_foundrl_baseline"
    dataset_base: str = "projects/d6_grant_proposal/dataset"
    data_base: str = "projects/d6_grant_proposal/data"

    model_name: str = "Qwen/Qwen3-30B-A3B"

    n_eval_plans: int = 8
    max_tokens: int = 8192
    temperature: float = 1.0
    top_p: float = 0.95

    # Grader settings (also Qwen3-30B-A3B, frozen)
    grader_max_tokens: int = 8192
    grader_temperature: float = 0.0

    seed: int = 42


def main(config: Config):
    log_dir = _resolve(config.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    goal_dir = _resolve(f"{config.dataset_base}/{config.goal_domain}/{config.goal_name}")
    goal = (goal_dir / "research_goal.md").read_text().strip()
    oracle_path = _resolve(
        f"{config.data_base}/{config.goal_domain}/{config.goal_name}/oracle/slim.md"
    )
    if not oracle_path.exists():
        raise FileNotFoundError(
            f"Slim oracle not found at {oracle_path}. "
            "Run extract_oracle.py and extract_oracle_slim.py first."
        )
    oracle = oracle_path.read_text().strip()

    logger.info("D6 frozen baseline")
    logger.info("  goal_domain: %s, goal_name: %s", config.goal_domain, config.goal_name)
    logger.info("  n_eval_plans: %d", config.n_eval_plans)
    logger.info("  log_dir: %s", log_dir)

    (log_dir / "config.json").write_text(json.dumps(
        {k: getattr(config, k) for k in dir(config)
         if not k.startswith("_") and not callable(getattr(config, k))},
        default=str, indent=2
    ))

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )
    sampling_client = service_client.create_sampling_client(base_model=config.model_name)
    grader_client = service_client.create_sampling_client(base_model=config.model_name)

    active_signals = _active_signals(config.goal_domain)

    student_text = build_student_prompt(goal, oracle)
    student_convo = [{"role": "user", "content": student_text}]
    student_input = renderer.build_generation_prompt(student_convo)

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=renderer.get_stop_sequences(),
    )
    grader_params = tinker.types.SamplingParams(
        max_tokens=config.grader_max_tokens,
        temperature=config.grader_temperature,
        stop=renderer.get_stop_sequences(),
    )

    t0 = time.time()
    result = sampling_client.sample(
        prompt=student_input,
        num_samples=config.n_eval_plans,
        sampling_params=sampling_params,
    ).result()
    logger.info("Generated %d plans in %.1fs", len(result.sequences), time.time() - t0)

    records = []
    rewards = []
    for k, seq in enumerate(result.sequences):
        try:
            parsed = renderer.parse_response(seq.tokens)
            raw = parsed[0].get("content", "") if parsed else tokenizer.decode(seq.tokens)
        except Exception:
            raw = tokenizer.decode(seq.tokens)
        plan_text = extract_solution(raw)

        # Grade with Qwen3
        grader_prompt_str = build_single_call_prompt(goal, plan_text[:15000], active_signals)
        grader_convo = [{"role": "user", "content": grader_prompt_str}]
        grader_input = renderer.build_generation_prompt(grader_convo)
        try:
            grader_result = grader_client.sample(
                grader_input, num_samples=1, sampling_params=grader_params,
            ).result()
            grader_raw = grader_result.sequences[0]
            grader_parsed = renderer.parse_response(grader_raw.tokens)
            grader_text = grader_parsed[0].get("content", "") if grader_parsed else tokenizer.decode(grader_raw.tokens)
            scores_parsed = parse_scores(grader_text)
            signal_scores = {sid: d["score"] for sid, d in scores_parsed.items()}
            reward = aggregate_reward(signal_scores)
        except Exception as e:
            logger.warning("Plan %d grading failed: %s", k, e)
            signal_scores = {}
            reward = 0.0

        rewards.append(reward)
        records.append({
            "iter": 0,
            "plan_idx": k,
            "plan_id": f"baseline_plan_{k:03d}",
            "n_gen_tokens": len(seq.tokens),
            "plan_text": plan_text,
            "signal_scores": signal_scores,
            "aggregate_reward": reward,
            "stop_reason": seq.stop_reason,
            "sampled_by": "baseline_frozen",
        })
        logger.info("Plan %d: reward=%.4f", k, reward)

    buffer_path = log_dir / "buffer.jsonl"
    with open(buffer_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    std_reward = float(np.std(rewards)) if rewards else 0.0
    metrics_path = log_dir / "metrics.jsonl"
    with open(metrics_path, "w") as f:
        f.write(json.dumps({
            "iter": 0,
            "n_plans": len(records),
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "min_reward": float(min(rewards)) if rewards else 0.0,
            "max_reward": float(max(rewards)) if rewards else 0.0,
            "wall_sec": time.time() - t0,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }) + "\n")

    logger.info("=== D6 FROZEN BASELINE RESULT ===")
    logger.info("Mean reward: %.4f ± %.4f (n=%d, D4 weighted mean 0-1)", mean_reward, std_reward, len(rewards))
    logger.info("Decision threshold: OPD/GRPO must exceed baseline+0.02 on audit to claim improvement")
    logger.info("============================")
    logger.info("Wrote %d plans to %s", len(records), buffer_path)


if __name__ == "__main__":
    config = chz.entrypoint(Config)
    main(config)
