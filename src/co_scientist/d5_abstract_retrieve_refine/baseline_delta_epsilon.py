"""D5 Phase 0.6 baseline: frozen Qwen + reference plan in context (no training).

Two conditions:
  - δ: Qwen3-30B-A3B + reference plan as few-shot
  - ε: Qwen3-235B-A22B-Instruct-2507 + reference plan as few-shot

Generates 8 plans per condition; writes buffer for downstream Opus audit.

Run:
  PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.baseline_delta_epsilon \
      log_path=projects/d5_abstract_retrieve_refine/runs/2026_04_baseline_delta_epsilon \
      condition=delta   # or 'epsilon'
"""
import json
import logging
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

    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"
    reference_plan_path: str = "projects/d5_abstract_retrieve_refine/dataset/reference_solution.txt"

    log_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_baseline_delta_epsilon"

    # Which baseline to run: "delta" (30B) or "epsilon" (235B)
    condition: str = "delta"

    delta_model: str = "Qwen/Qwen3-30B-A3B"
    epsilon_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"

    num_samples: int = 8
    max_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 0.95


def build_prompt_with_reference(goal: str, reference_plan: str) -> str:
    """Reference plan as few-shot in-context."""
    return (
        "I will provide you a research scenario and a reference example of a high-quality "
        "research plan on a related (but different) problem. Use the example as a structural "
        "and methodological guide for your own plan, but do NOT copy its specific hypotheses "
        "or methods — adapt to the new scenario.\n\n"
        f"# Reference example (a complete research plan)\n\n{reference_plan}\n\n"
        f"---\n\n"
        f"# Now write a research plan for THIS scenario:\n\nScenario: {goal}"
        f"\n\nWrite your research plan inside <solution>...</solution> tags. "
        f"Target 600 words, max 750 words. Use structured sections "
        f"(Problem Statement, Background, Hypothesis, Methodology, Evaluation, Limitations)."
    )


def main(config: Config):
    repo_root = Path(__file__).resolve().parents[3]
    goal_path = (repo_root / config.goal_path).resolve()
    ref_path = (repo_root / config.reference_plan_path).resolve()
    log_dir = (repo_root / config.log_path).resolve()
    assert config.condition in ("delta", "epsilon"), f"Bad condition: {config.condition}"

    log_dir.mkdir(parents=True, exist_ok=True)
    goal = goal_path.read_text().strip()
    reference_plan = ref_path.read_text().strip()

    if config.condition == "delta":
        model = config.delta_model
    else:
        model = config.epsilon_model

    logger.info("D5 Phase 0.6 baseline: %s (model=%s)", config.condition, model)

    tokenizer = get_tokenizer(model)
    renderer_name = model_info.get_recommended_renderer_name(model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    service_client = create_service_client(api_profile=config.api_profile, base_url=config.base_url)
    sampling_client = service_client.create_sampling_client(base_model=model)

    sp_kwargs = dict(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=renderer.get_stop_sequences(),
    )
    sampling_params = tinker.types.SamplingParams(**sp_kwargs)

    prompt_text = build_prompt_with_reference(goal, reference_plan)
    convo = [{"role": "user", "content": prompt_text}]
    model_input = renderer.build_generation_prompt(convo)

    config_dict = {
        "condition": config.condition,
        "model": model,
        "num_samples": config.num_samples,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "goal_path": str(goal_path.relative_to(repo_root)),
        "reference_plan_path": str(ref_path.relative_to(repo_root)),
        "prompt_char_len": len(prompt_text),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(log_dir / f"config_{config.condition}.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info("  prompt char_len = %d, num_samples = %d", len(prompt_text), config.num_samples)
    logger.info("Sampling ...")
    future = sampling_client.sample(
        prompt=model_input, num_samples=config.num_samples, sampling_params=sampling_params
    )
    result = future.result()

    buffer_path = log_dir / "buffer.jsonl"
    with open(buffer_path, "a") as fbuf:
        for sample_idx, seq in enumerate(result.sequences):
            plan_text = tokenizer.decode(seq.tokens)
            entry = {
                "condition": config.condition,
                "sample_idx": sample_idx,
                "plan_text": plan_text,
                "stop_reason": seq.stop_reason,
                "model": model,
                "prompt_char_len": len(prompt_text),
                "plan_char_len": len(plan_text),
            }
            fbuf.write(json.dumps(entry) + "\n")

    logger.info("Done. Wrote %d plans for %s to %s", len(result.sequences), config.condition, buffer_path)


if __name__ == "__main__":
    config = chz.entrypoint(Config)
    main(config)
