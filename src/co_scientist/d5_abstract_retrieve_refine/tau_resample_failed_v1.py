"""Resample τ baseline plan generations that hit Qwen3 thinking-mode early-EOS.

Known failure mode: Qwen3-30B-A3B in thinking mode occasionally samples `<|im_end|>`
right after `<think>` open, producing a 17-char output `<think><|im_end|>`. This is
a sampling artifact (~25% in our first τ run, plan_2 + plan_4 both hit it),
not a prompt issue.

Standard ML practice: retry the failed plans with a different sampling seed,
keeping the original distillations intact. Replaces the corrupted entry in
buffer.jsonl atomically.

Usage:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.tau_resample_failed_v1 \
        log_path=projects/d5_abstract_retrieve_refine/runs/2026_04_28_tau_v1 \
        min_plan_chars=200 max_retries=3
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

    log_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_tau_v1"
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"
    model: str = "Qwen/Qwen3-30B-A3B"

    min_plan_chars: int = 200       # plans shorter than this are resampled
    max_retries: int = 3
    plan_max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.95


def main(config: Config):
    log_dir = (REPO_ROOT / config.log_path).resolve()
    buffer_path = log_dir / "buffer.jsonl"
    if not buffer_path.exists():
        raise FileNotFoundError(f"buffer.jsonl not found: {buffer_path}")

    rows = [json.loads(line) for line in buffer_path.read_text().splitlines() if line.strip()]
    failed = [r for r in rows if len(r.get("plan_text", "")) < config.min_plan_chars]
    healthy = [r for r in rows if len(r.get("plan_text", "")) >= config.min_plan_chars]

    if not failed:
        logger.info("No failed plans (all >= %d chars). Nothing to do.", config.min_plan_chars)
        return

    logger.info("Found %d failed plans (out of %d): sample_idx %s",
                len(failed), len(rows), [r["sample_idx"] for r in failed])

    goal = (REPO_ROOT / config.goal_path).resolve().read_text().strip()
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

    fixed_count = 0
    for row in failed:
        plan_idx = row["sample_idx"]
        A_list = row["distillations"]
        plan_prompt = build_kappa_plan_prompt(goal, A_list)
        convo = [{"role": "user", "content": plan_prompt}]
        model_input = renderer.build_generation_prompt(convo)

        for attempt in range(1, config.max_retries + 1):
            result = sampling_client.sample(
                prompt=model_input, num_samples=1, sampling_params=plan_params,
            ).result()
            seq = result.sequences[0]
            new_plan = tokenizer.decode(seq.tokens)
            logger.info("plan_%d retry %d: %d chars, stop=%s",
                        plan_idx, attempt, len(new_plan), seq.stop_reason)
            if len(new_plan) >= config.min_plan_chars:
                row["plan_text"] = new_plan
                row["stop_reason"] = seq.stop_reason
                row["plan_char_len"] = len(new_plan)
                row["resample_attempt"] = attempt
                row["resample_ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
                fixed_count += 1
                break
        else:
            logger.error("plan_%d FAILED after %d retries; leaving original",
                         plan_idx, config.max_retries)

    # Reorder rows by sample_idx then rewrite atomically
    all_rows = sorted(healthy + failed, key=lambda r: r["sample_idx"])
    tmp_path = buffer_path.with_suffix(".jsonl.tmp")
    tmp_path.write_text("".join(json.dumps(r) + "\n" for r in all_rows))
    tmp_path.replace(buffer_path)
    logger.info("Resampled %d / %d failed plans; rewrote %s",
                fixed_count, len(failed), buffer_path)


if __name__ == "__main__":
    main(chz.entrypoint(Config))
