"""D5 Phase 2E baseline_frozen_v2 — single parameterized frozen-model script.

Replaces baseline_delta_epsilon.py + smoke_pathway runs. Generates 8 plans per
condition without any training. Used for ξ / σ / δ / ε baselines:

  --variant xi      : Qwen3-30B-A3B,         goal only
  --variant sigma   : Qwen3-30B-A3B,         goal + oracle_v2
  --variant delta   : Qwen3-30B-A3B,         goal + reference_plan
  --variant epsilon : Qwen3-235B-A22B-2507,  goal + reference_plan

Outputs `buffer.jsonl` in the run dir; each entry tagged with variant + sample_idx.
Subsequent audit / pairwise runs read this file.

Usage:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.baseline_frozen_v2 \
        log_path=projects/d5_abstract_retrieve_refine/runs/2026_05_NN_xi_v1 \
        variant=xi
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import chz
import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.shared.api_profiles import create_service_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    api_profile: str | None = "new"
    base_url: str | None = None

    log_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_05_xx_frozen_v2"
    variant: str = "xi"  # xi | sigma | delta | epsilon | mu_inference

    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"
    oracle_path: str = "projects/d5_abstract_retrieve_refine/data/oracles/oracle_v2_2026_04_26_build/slim.md"
    reference_plan_path: str = "projects/d5_abstract_retrieve_refine/dataset/reference_solution.txt"

    model_30b: str = "Qwen/Qwen3-30B-A3B"
    model_235b: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"

    # mu_inference: load saved LoRA sampler (e.g. iter-4 from runs/2026_04_27_mu_v4/checkpoints.jsonl)
    # Empty for non-mu_inference variants (use base_model instead).
    sampler_path: str = ""

    num_samples: int = 8
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.95

    # footer_version: "v1" = 600/750 word target (legacy); "v8" = 900/1100 target
    # (length-matched to reference_solution.txt = 962 words). σ_v8 baseline runs
    # use "v8". Default "v1" preserves backward compat with existing run dirs.
    footer_version: str = "v1"


_PLAN_FOOTER_V1 = (
    "\n\nWrite your research plan inside <solution>...</solution> tags."
    " Target 600 words, max 750 words. Use structured sections (Problem"
    " Statement, Background, Hypothesis, Methodology, Evaluation, Limitations)."
)

_PLAN_FOOTER_V8 = (
    "\n\nWrite your research plan inside <solution>...</solution> tags."
    " Target 900 words, max 1100 words. Use structured sections (Problem"
    " Statement, Background, Hypothesis, Methodology, Evaluation, Limitations)."
)


def _get_footer(footer_version: str) -> str:
    if footer_version == "v8":
        return _PLAN_FOOTER_V8
    if footer_version == "v1":
        return _PLAN_FOOTER_V1
    raise ValueError(f"Unknown footer_version={footer_version!r} (expected 'v1' or 'v8')")


_PLAN_FOOTER = _PLAN_FOOTER_V1  # backward-compat default for any external imports


def build_prompt_xi(goal: str, footer_version: str = "v1") -> str:
    return (
        "I will provide you a research scenario. You have to provide me a"
        " concise yet thoughtful research plan with all details needed to"
        " execute it."
        f"\n\nScenario: {goal}"
        + _get_footer(footer_version)
    )


def build_prompt_sigma(goal: str, oracle: str, footer_version: str = "v1") -> str:
    return (
        "I will provide you a research scenario and a set of methodological"
        " patterns extracted from a high-quality research plan on this same"
        " scenario. Use the patterns to guide the structure and reasoning of"
        " your plan, but do not copy the phrasing literally."
        f"\n\nScenario: {goal}"
        f"\n\n# Methodological patterns to apply\n\n{oracle}"
        + _get_footer(footer_version)
    )


def build_prompt_delta_epsilon(goal: str, reference_plan: str, footer_version: str = "v1") -> str:
    return (
        "I will provide you a research scenario and a reference example of a"
        " high-quality research plan on a related (but different) problem. Use"
        " the example as a structural and methodological guide for your own plan,"
        " but do NOT copy its specific hypotheses or methods — adapt to the new"
        " scenario."
        f"\n\n# Reference example (a complete research plan)\n\n{reference_plan}"
        f"\n\n---\n\n# Now write a research plan for THIS scenario:"
        f"\n\nScenario: {goal}"
        + _get_footer(footer_version)
    )


def main(config: Config):
    log_dir = (REPO_ROOT / config.log_path).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    goal = (REPO_ROOT / config.goal_path).resolve().read_text().strip()

    variant = config.variant.lower()
    fv = config.footer_version
    if variant == "xi":
        model = config.model_30b
        prompt_text = build_prompt_xi(goal, footer_version=fv)
    elif variant == "sigma":
        model = config.model_30b
        oracle = (REPO_ROOT / config.oracle_path).resolve().read_text().strip()
        prompt_text = build_prompt_sigma(goal, oracle, footer_version=fv)
    elif variant == "delta":
        model = config.model_30b
        ref = (REPO_ROOT / config.reference_plan_path).resolve().read_text().strip()
        prompt_text = build_prompt_delta_epsilon(goal, ref, footer_version=fv)
    elif variant == "epsilon":
        model = config.model_235b
        ref = (REPO_ROOT / config.reference_plan_path).resolve().read_text().strip()
        prompt_text = build_prompt_delta_epsilon(goal, ref, footer_version=fv)
    elif variant == "mu_inference":
        # Load a saved μ-trained LoRA sampler and run sigma-style inference.
        # Same prompt as σ (goal + slim oracle) since μ-v4 student-side EVAL uses this prompt.
        model = config.model_30b
        oracle = (REPO_ROOT / config.oracle_path).resolve().read_text().strip()
        prompt_text = build_prompt_sigma(goal, oracle, footer_version=fv)
        if not config.sampler_path:
            raise ValueError("variant=mu_inference requires sampler_path "
                             "(e.g. tinker://<run-uuid>:train:0/sampler_weights/iter_NNNN)")
    else:
        raise ValueError(f"Unknown variant: {variant!r} "
                         f"(expected xi|sigma|delta|epsilon|mu_inference)")

    logger.info("Variant %s | model=%s | prompt=%d chars", variant, model, len(prompt_text))

    tokenizer = get_tokenizer(model)
    renderer_name = model_info.get_recommended_renderer_name(model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    service_client = create_service_client(api_profile=config.api_profile, base_url=config.base_url)
    if variant == "mu_inference":
        sampling_client = service_client.create_sampling_client(model_path=config.sampler_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=model)

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=renderer.get_stop_sequences(),
    )

    convo = [{"role": "user", "content": prompt_text}]
    model_input = renderer.build_generation_prompt(convo)

    config_dict = {
        "variant": variant,
        "footer_version": fv,
        "model": model,
        "num_samples": config.num_samples,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "goal_path": config.goal_path,
        "oracle_path": config.oracle_path if variant in ("sigma", "mu_inference") else None,
        "reference_plan_path": config.reference_plan_path if variant in ("delta", "epsilon") else None,
        "sampler_path": config.sampler_path if variant == "mu_inference" else None,
        "prompt_char_len": len(prompt_text),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (log_dir / f"config_{variant}.json").write_text(json.dumps(config_dict, indent=2))

    logger.info("Sampling %d plans...", config.num_samples)
    result = sampling_client.sample(
        prompt=model_input, num_samples=config.num_samples, sampling_params=sampling_params,
    ).result()

    buffer_path = log_dir / "buffer.jsonl"
    with open(buffer_path, "a") as f:
        for sample_idx, seq in enumerate(result.sequences):
            plan_text = tokenizer.decode(seq.tokens)
            f.write(json.dumps({
                "variant": variant,
                "sample_idx": sample_idx,
                "plan_text": plan_text,
                "stop_reason": seq.stop_reason,
                "model": model,
                "prompt_char_len": len(prompt_text),
                "plan_char_len": len(plan_text),
            }) + "\n")

    logger.info("Wrote %d plans for %s -> %s", len(result.sequences), variant, buffer_path)


if __name__ == "__main__":
    main(chz.entrypoint(Config))
