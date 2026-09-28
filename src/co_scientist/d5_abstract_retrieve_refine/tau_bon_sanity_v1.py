"""D5 Phase 3 Step 2a — τ_v4 BoN sanity check (1 instance × K=8 plan samples).

Per ML-scientist subagent killer Q1 (2026-04-29): before committing $16 to a full
8-instance × K=4 BoN ablation, measure within-instance plan-sampling variance on
ONE instance with K=8 reseeded plan samples. If σ < 1.5 across K=8 plans, BoN-4
selection signal is interpretable; if σ > 2.0, K=4 may not separate signal from
sampling noise and we need K=8 or a different audit strategy.

Reuses existing instance-0 distillations from `runs/2026_04_28_tau_v4/buffer.jsonl`
(skips Stage 1 distillation re-run — same 3 A_i, same plan_v4 prompt, only
plan-generation step varies). Outputs 8 plans tagged sample_idx 0-7, all from
instance_idx=0.

Usage:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.tau_bon_sanity_v1
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
from co_scientist.d5_abstract_retrieve_refine.kappa_prompts_v1 import (
    build_kappa_plan_prompt,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    api_profile: str | None = "new"
    base_url: str | None = None

    log_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_29_tau_v4_bon_sanity"
    source_tau_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_tau_v4"
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"

    model: str = "Qwen/Qwen3-30B-A3B"
    instance_idx: int = 0  # which τ_v4 instance to reuse distillations from
    num_samples: int = 8  # K=8 plan candidates from same prompt

    plan_max_tokens: int = 4096
    plan_prompt_version: str = "v4"

    temperature: float = 1.0
    top_p: float = 0.95


def main(config: Config):
    log_dir = (REPO_ROOT / config.log_path).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    goal = (REPO_ROOT / config.goal_path).resolve().read_text().strip()
    src_buffer = (REPO_ROOT / config.source_tau_run / "buffer.jsonl").resolve()
    if not src_buffer.exists():
        raise FileNotFoundError(f"Source τ_v4 buffer missing: {src_buffer}")

    src_row = None
    with open(src_buffer) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("sample_idx") == config.instance_idx:
                src_row = d
                break
    if src_row is None:
        raise ValueError(f"sample_idx={config.instance_idx} not found in {src_buffer}")

    A_list = src_row["distillations"]
    if len(A_list) != 3:
        raise ValueError(f"Expected 3 distillations, got {len(A_list)}")
    logger.info("Reusing instance %d distillations: lens=%s",
                config.instance_idx, [len(a) for a in A_list])

    plan_prompt = build_kappa_plan_prompt(goal, A_list, version=config.plan_prompt_version)
    logger.info("Plan prompt (%s): %d chars (~%d tokens)",
                config.plan_prompt_version, len(plan_prompt), len(plan_prompt)//4)

    tokenizer = get_tokenizer(config.model)
    renderer_name = model_info.get_recommended_renderer_name(config.model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    service_client = create_service_client(api_profile=config.api_profile, base_url=config.base_url)
    sampling_client = service_client.create_sampling_client(base_model=config.model)

    plan_params = tinker.types.SamplingParams(
        max_tokens=config.plan_max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=renderer.get_stop_sequences(),
    )

    config_dump = {
        "variant": "tau_v4_bon_sanity",
        "model": config.model,
        "source_tau_run": config.source_tau_run,
        "instance_idx": config.instance_idx,
        "num_samples": config.num_samples,
        "plan_prompt_version": config.plan_prompt_version,
        "plan_max_tokens": config.plan_max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (log_dir / "config_bon_sanity.json").write_text(json.dumps(config_dump, indent=2))

    convo = [{"role": "user", "content": plan_prompt}]
    model_input = renderer.build_generation_prompt(convo)
    result = sampling_client.sample(
        prompt=model_input,
        num_samples=config.num_samples,
        sampling_params=plan_params,
    ).result()

    buffer_path = log_dir / "buffer.jsonl"
    with open(buffer_path, "w") as f:
        for sample_idx, seq in enumerate(result.sequences):
            raw_plan_text = tokenizer.decode(seq.tokens)
            f.write(json.dumps({
                "variant": "tau_v4_bon_sanity",
                "sample_idx": sample_idx,
                "instance_idx": config.instance_idx,
                "plan_text": raw_plan_text,
                "stop_reason": seq.stop_reason,
                "model": config.model,
                "plan_char_len": len(raw_plan_text),
            }) + "\n")
            logger.info("  sample_%d: plan=%d chars, stop=%s",
                        sample_idx, len(raw_plan_text), seq.stop_reason)

    logger.info("BoN sanity run complete: %d plans → %s", config.num_samples, buffer_path)


if __name__ == "__main__":
    main(chz.entrypoint(Config))
