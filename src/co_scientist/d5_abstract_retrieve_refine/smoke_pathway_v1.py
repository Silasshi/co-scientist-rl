"""D5 Phase 0c: Pathway validation smoke test.

Hypothesis: frozen Qwen3-30B-A3B with oracle abstraction + research goal
produces higher-quality plans than with goal alone.

Generates 8 plans per condition (A=with abstraction, B=baseline) via Tinker
sampling on the frozen base model, then writes buffer.jsonl.

Scoring (ten_signal_reward) and Opus audit are separate steps — see
run_smoke_scoring.py and opus_audit_subagent.md.

Run:
    python -m co_scientist.d5_abstract_retrieve_refine.smoke_pathway_v1 \
        config.api_profile=new \
        config.log_path=projects/d5_abstract_retrieve_refine/runs/2026_04_smoke_pathway_v1
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

import chz
import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from co_scientist.shared.api_profiles import create_service_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@chz.chz
class Config:
    api_profile: str | None = "new"
    base_url: str | None = None

    # Data paths (D5-owned dataset, copied from D3 sanity_check on 2026-04-25)
    goal_path: str = (
        "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"
    )
    reference_plan_path: str = (
        "projects/d5_abstract_retrieve_refine/dataset/reference_solution.txt"
    )
    oracle_abstraction_path: str = (
        "projects/d5_abstract_retrieve_refine/data/oracles/smoke/v1.md"
    )

    # Output path
    log_path: str = (
        "projects/d5_abstract_retrieve_refine/runs/2026_04_smoke_pathway_v1"
    )

    # Model + sampling (revised 2026-04-25: temp=1.0 + top_p to fix duplicate-collapse bug)
    policy_model: str = "Qwen/Qwen3-30B-A3B"
    num_samples: int = 8  # per condition
    max_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 0.95
    seed: int | None = None  # None = random per-call; set int only for reproducibility

    # Length constraints (D3 convention)
    max_word_count: int = 750
    target_word_count: int = 600
    min_words: int = 30


# -----------------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------------

def _plan_format_footer(target_word: int, max_word: int) -> str:
    return (
        f"\n\nWrite your research plan inside <solution>...</solution> tags."
        f" Target {target_word} words, max {max_word} words. Use structured sections"
        f" (Problem Statement, Background, Hypothesis, Methodology, Evaluation, Limitations)."
    )


def build_prompt_baseline(goal: str, target_word: int, max_word: int) -> str:
    return (
        "I will provide you a research scenario. You have to provide me a concise"
        " yet thoughtful research plan with all details needed to execute it."
        f"\n\nScenario: {goal}"
        + _plan_format_footer(target_word, max_word)
    )


def build_prompt_with_abstraction(
    goal: str, abstraction: str, target_word: int, max_word: int
) -> str:
    return (
        "I will provide you a research scenario and a set of methodological patterns"
        " extracted from a high-quality research plan on this same scenario. Use the"
        " patterns to guide the structure and reasoning of your plan, but do not copy"
        " the phrasing literally."
        f"\n\nScenario: {goal}"
        f"\n\n# Methodological patterns to apply\n\n{abstraction}"
        + _plan_format_footer(target_word, max_word)
    )


# -----------------------------------------------------------------------------
# Main smoke loop
# -----------------------------------------------------------------------------

def main(config: Config):
    # Resolve paths relative to repo root
    repo_root = Path(__file__).resolve().parents[3]
    goal_path = (repo_root / config.goal_path).resolve()
    ref_path = (repo_root / config.reference_plan_path).resolve()
    abs_path = (repo_root / config.oracle_abstraction_path).resolve()
    log_dir = (repo_root / config.log_path).resolve()

    assert goal_path.exists(), f"Missing goal: {goal_path}"
    assert ref_path.exists(), f"Missing reference plan: {ref_path}"
    assert abs_path.exists(), f"Missing oracle abstraction: {abs_path}"
    log_dir.mkdir(parents=True, exist_ok=True)

    goal = goal_path.read_text().strip()
    oracle_abstraction = abs_path.read_text().strip()

    logger.info("D5 smoke_pathway_v1 starting")
    logger.info("  goal_path = %s (%d chars)", goal_path, len(goal))
    logger.info("  oracle_abs = %s (%d chars)", abs_path, len(oracle_abstraction))
    logger.info("  log_dir = %s", log_dir)
    logger.info("  model = %s, N = %d per condition, T = %.2f",
                config.policy_model, config.num_samples, config.temperature)

    # Save run config
    config_dict = {
        "api_profile": config.api_profile,
        "policy_model": config.policy_model,
        "num_samples": config.num_samples,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "seed": config.seed,
        "goal_path": str(goal_path.relative_to(repo_root)),
        "reference_plan_path": str(ref_path.relative_to(repo_root)),
        "oracle_abstraction_path": str(abs_path.relative_to(repo_root)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(log_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Tokenizer + renderer
    tokenizer = get_tokenizer(config.policy_model)
    renderer_name = model_info.get_recommended_renderer_name(config.policy_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Tinker service + frozen-base sampling client
    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile
    )
    sampling_client = service_client.create_sampling_client(
        base_model=config.policy_model
    )
    sp_kwargs = dict(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=renderer.get_stop_sequences(),
    )
    if config.seed is not None:
        sp_kwargs["seed"] = config.seed
    sampling_params = tinker.types.SamplingParams(**sp_kwargs)

    # Two conditions
    prompt_B = build_prompt_baseline(goal, config.target_word_count, config.max_word_count)
    prompt_A = build_prompt_with_abstraction(
        goal, oracle_abstraction, config.target_word_count, config.max_word_count
    )

    conditions = [
        ("B_baseline", prompt_B),
        ("A_with_abstraction", prompt_A),
    ]

    buffer_path = log_dir / "buffer.jsonl"
    if buffer_path.exists():
        logger.warning("Overwriting existing buffer.jsonl at %s", buffer_path)
        buffer_path.unlink()

    # Launch all futures up front (parallel across conditions and samples)
    logger.info("Launching %d futures (%d conditions × %d samples)",
                len(conditions) * config.num_samples, len(conditions), config.num_samples)

    futures_by_cond = {}
    for cond_name, prompt_text in conditions:
        convo = [{"role": "user", "content": prompt_text}]
        model_input = renderer.build_generation_prompt(convo)
        future = sampling_client.sample(
            prompt=model_input,
            num_samples=config.num_samples,
            sampling_params=sampling_params,
        )
        futures_by_cond[cond_name] = (future, prompt_text, model_input)

    # Collect results
    with open(buffer_path, "w") as fbuf:
        for cond_name, (future, prompt_text, _model_input) in futures_by_cond.items():
            logger.info("Collecting %s ...", cond_name)
            result = future.result()
            for sample_idx, seq in enumerate(result.sequences):
                plan_text = tokenizer.decode(seq.tokens)
                entry = {
                    "condition": cond_name,
                    "sample_idx": sample_idx,
                    "plan_text": plan_text,
                    "stop_reason": seq.stop_reason,
                    "prompt_text": prompt_text,
                    "prompt_char_len": len(prompt_text),
                    "plan_char_len": len(plan_text),
                    "model": config.policy_model,
                    "temperature": config.temperature,
                    "seed": config.seed,
                    "sample_seed_effective": (config.seed + sample_idx) if config.seed is not None else None,
                }
                fbuf.write(json.dumps(entry) + "\n")
            logger.info("  %s done: %d samples", cond_name, len(result.sequences))

    logger.info("Generation complete. Buffer at %s", buffer_path)
    logger.info("Next steps:")
    logger.info("  1. Run scoring: python -m co_scientist.d5_abstract_retrieve_refine.smoke_pathway_v1_score ...")
    logger.info("  2. Spawn Opus audit subagent (see projects/d5_abstract_retrieve_refine/scripts/opus_audit_subagent.md)")


if __name__ == "__main__":
    config = chz.entrypoint(Config)
    main(config)
