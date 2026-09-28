"""D5 Phase 3 Step 2.5 — σ_v4 + μ-v4-replan critical controls.

Both controls share: slim oracle (Phase 2 σ's substrate) + plan_v4 footer
(mandatory T1/T2/T3 instructions extracted from kappa_prompts_v1's
_KAPPA_PLAN_FOOTER_V4, simplified to single-shot — no "three distillation
lists" framing). Single-shot inference, no distillation rounds, no LoRA training.

Variants:
  sigma_v4       : Qwen3-30B base + slim oracle + plan_v4 footer
  mu_v4_replan   : Qwen3-30B + μ-v4 iter-4 LoRA sampler + slim oracle + plan_v4 footer

Critical Phase 3 control questions:
  - σ_v4 vs τ_v4 (=26.50): is τ_v4's gain attributable to plan_v4 prompt
    alone, or does the 3-round distillation pipeline add net value?
  - μ-v4-replan vs μ-v4 (=28.00): is plan_v4 a universal upgrade that
    SDPO weights also benefit from?

Forks baseline_frozen_v2.py pattern. Does NOT modify baseline_frozen_v2.py
itself (cross-session namespace lock — that file is Session B's territory).

Usage:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.train_sigma_v4_v1 \
        log_path=projects/d5_abstract_retrieve_refine/runs/2026_04_28_sigma_v4 \
        variant=sigma_v4

    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.train_sigma_v4_v1 \
        log_path=projects/d5_abstract_retrieve_refine/runs/2026_04_28_mu_v4_replan \
        variant=mu_v4_replan \
        sampler_path=tinker://6b9d996d-8079-556d-82ac-560247551960:train:0/sampler_weights/iter_0004
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

    log_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_sigma_v4"
    variant: str = "sigma_v4"  # sigma_v4 | mu_v4_replan

    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"
    oracle_path: str = "projects/d5_abstract_retrieve_refine/data/oracles/oracle_v2_2026_04_26_build/slim.md"

    model: str = "Qwen/Qwen3-30B-A3B"

    # Required for variant=mu_v4_replan; ignored for sigma_v4.
    sampler_path: str = ""

    num_samples: int = 8
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.95


# Plan v4 footer simplified for single-shot use (drops "three distillation lists" framing
# from kappa_prompts_v1._KAPPA_PLAN_FOOTER_V4; keeps the mandatory T1/T2/T3 instructions
# verbatim so substance is identical to the τ_v4 plan-prompt change).
_SIGMA_V4_FOOTER = (
    "\n\nWrite your research plan inside <solution>...</solution> tags."
    " Target 600 words, max 750 words. Use structured sections (Problem"
    " Statement, Background, Hypothesis, Methodology, Evaluation, Limitations)."
    "\n\n**MANDATORY plan-quality elements**:"
    "\n- **Disentanglement ablation** (Methodology or Evaluation section):"
    " name 1-2 specific ablations that separate the LLM-prior contribution"
    " from the method's contribution (e.g., 'frozen-LLM same-prompt baseline',"
    " 'method without per-problem updates', 'random-init LoRA'). Cite which"
    " plan component each ablation isolates."
    "\n- **Compute accounting in operational units** (Evaluation or"
    " Limitations section): state compute as GPU-hours per problem (or"
    " wall-clock minutes per problem), NOT as FLOP-only or memory-only."
    " Give a concrete number range (e.g., '2-4 H100-hours per problem')."
    "\n- **Quantified frozen-LLM baseline gap** (Background or Evaluation):"
    " state the prior-art frozen-LLM number on at least one named benchmark"
    " (e.g., 'frozen GPT-4 scores 50.4% on MATH-500') AND your method's"
    " expected delta with rationale."
)


def build_sigma_v4_prompt(goal: str, oracle: str) -> str:
    """σ-style single-shot prompt + v4 mandatory plan-quality instructions."""
    return (
        "I will provide you a research scenario and a set of methodological"
        " patterns extracted from a high-quality research plan on this same"
        " scenario. Use the patterns to guide the structure and reasoning of"
        " your plan, but do not copy the phrasing literally."
        f"\n\nScenario: {goal}"
        f"\n\n# Methodological patterns to apply\n\n{oracle}"
        + _SIGMA_V4_FOOTER
    )


def main(config: Config):
    log_dir = (REPO_ROOT / config.log_path).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    goal = (REPO_ROOT / config.goal_path).resolve().read_text().strip()
    oracle = (REPO_ROOT / config.oracle_path).resolve().read_text().strip()
    prompt_text = build_sigma_v4_prompt(goal, oracle)

    variant = config.variant.lower()
    if variant not in ("sigma_v4", "mu_v4_replan"):
        raise ValueError(f"Unknown variant: {variant!r} (expected sigma_v4 | mu_v4_replan)")
    if variant == "mu_v4_replan" and not config.sampler_path:
        raise ValueError("variant=mu_v4_replan requires sampler_path "
                         "(e.g. tinker://...sampler_weights/iter_0004)")

    logger.info("Variant %s | model=%s | prompt=%d chars (~%d tokens)",
                variant, config.model, len(prompt_text), len(prompt_text)//4)

    tokenizer = get_tokenizer(config.model)
    renderer_name = model_info.get_recommended_renderer_name(config.model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    service_client = create_service_client(api_profile=config.api_profile, base_url=config.base_url)
    if variant == "mu_v4_replan":
        sampling_client = service_client.create_sampling_client(model_path=config.sampler_path)
        logger.info("Loaded LoRA sampler: %s", config.sampler_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=config.model)

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=renderer.get_stop_sequences(),
    )

    config_dump = {
        "variant": variant,
        "model": config.model,
        "sampler_path": config.sampler_path if variant == "mu_v4_replan" else None,
        "oracle_path": str(config.oracle_path),
        "goal_path": str(config.goal_path),
        "num_samples": config.num_samples,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "prompt_char_len": len(prompt_text),
        "plan_prompt": "sigma_v4_footer (plan_v4 mandatory T1/T2/T3 instructions, single-shot)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (log_dir / f"config_{variant}.json").write_text(json.dumps(config_dump, indent=2))

    convo = [{"role": "user", "content": prompt_text}]
    model_input = renderer.build_generation_prompt(convo)

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
                "model": config.model,
                "sampler_path": config.sampler_path if variant == "mu_v4_replan" else None,
                "prompt_char_len": len(prompt_text),
                "plan_char_len": len(plan_text),
            }) + "\n")

    logger.info("Wrote %d plans for %s -> %s", len(result.sequences), variant, buffer_path)


if __name__ == "__main__":
    main(chz.entrypoint(Config))
