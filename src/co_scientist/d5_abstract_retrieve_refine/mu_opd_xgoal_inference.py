"""D5 μ-OPD inference on cross-goal forward-citation papers (Phase 4a redo).

Loads μ-OPD iter 4 sampler weights, samples 8 plans under student-context
(goal + slim_oracle, no critique) for each of 3 forward-citation goals
(meta_ttl, tool_verification_ttrl, tt_control). No training.

Per F13 / Phase 4a redo plan: tests if canonical OPD's training transfers
to cross-goal where (c) HER (μ-v4) showed F7 NULL at n=24.

Usage:
    source shared/tools/use_api_profile.sh new
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.mu_opd_xgoal_inference \
        goal_label=meta_ttl \
        out_path=projects/d5_abstract_retrieve_refine/runs/2026_04_29_xgoal_meta_ttl_mu_opd
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from co_scientist.shared.api_profiles import create_service_client
from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v2 import (
    build_student_prompt,
    extract_solution,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)

REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    api_profile: str | None = "new"
    base_url: str | None = None

    # μ-OPD iter 4 production sampler
    sampler_path: str = (
        "tinker://99ee87da-3a77-5289-8feb-efc601fb99b5:train:0/sampler_weights/iter_0004"
    )

    # Cross-goal target. One of: meta_ttl, tool_verification_ttrl, tt_control
    goal_label: str = "meta_ttl"

    # Resolves automatically from goal_label if empty
    goal_path: str = ""
    oracle_path: str = (
        "projects/d5_abstract_retrieve_refine/data/oracles/oracle_v2_2026_04_26_build/slim.md"
    )

    model_name: str = "Qwen/Qwen3-30B-A3B"
    n_plans: int = 8
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.95

    out_path: str = ""  # auto: runs/2026_04_29_xgoal_<goal_label>_mu_opd


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def main(config: Config) -> None:
    goal_path = config.goal_path or (
        f"projects/d5_abstract_retrieve_refine/data/cross_goal/{config.goal_label}/research_goal.txt"
    )
    out_path = config.out_path or (
        f"projects/d5_abstract_retrieve_refine/runs/2026_04_29_xgoal_{config.goal_label}_mu_opd"
    )
    out_dir = _resolve(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    goal = _resolve(goal_path).read_text().strip()
    oracle = _resolve(config.oracle_path).read_text().strip()
    logger.info("Loaded goal=%d chars (%s), oracle=%d chars",
                 len(goal), config.goal_label, len(oracle))

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    student_text = build_student_prompt(goal, oracle)
    student_input = renderer.build_generation_prompt(
        [{"role": "user", "content": student_text}]
    )
    logger.info("Student prompt: %d tokens", len(student_input.to_ints()))

    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )
    sampling_client = service_client.create_sampling_client(
        model_path=config.sampler_path,
    )
    logger.info("Sampling client ready @ %s", config.sampler_path)

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=renderer.get_stop_sequences(),
    )

    fut = sampling_client.sample(
        prompt=student_input, num_samples=config.n_plans, sampling_params=sampling_params,
    )
    result = fut.result()
    logger.info("Sampled %d plans", len(result.sequences))

    def _decode(seq) -> str:
        parsed = renderer.parse_response(seq.tokens)
        content = parsed[0].get("content", "") if parsed else ""
        return extract_solution(content) if content else tokenizer.decode(seq.tokens)

    # buffer.jsonl format compatible with existing audit_v3_isolated.py and pairwise scripts
    buffer_path = out_dir / "buffer.jsonl"
    eval_path = out_dir / "eval_rollouts.jsonl"
    with open(buffer_path, "w") as bf, open(eval_path, "w") as ef:
        for k, seq in enumerate(result.sequences):
            text = _decode(seq)
            buf_rec = {
                "variant": "mu_opd_inference",
                "sample_idx": k,
                "plan_text": text,
                "stop_reason": seq.stop_reason,
                "model": config.model_name,
                "goal_label": config.goal_label,
                "sampler_path": config.sampler_path,
                "prompt_char_len": len(student_text),
                "plan_char_len": len(text),
            }
            bf.write(json.dumps(buf_rec) + "\n")
            ev_rec = {
                "iter": 0, "plan_id": f"mu_opd_xgoal_{config.goal_label}_eval_{k}",
                "text": text, "goal_label": config.goal_label,
            }
            ef.write(json.dumps(ev_rec) + "\n")
    logger.info("Wrote %s", buffer_path)
    logger.info("Wrote %s", eval_path)

    config_path = out_dir / "config_mu_opd_inference.json"
    config_path.write_text(json.dumps({
        "variant": "mu_opd_inference",
        "model": config.model_name,
        "num_samples": config.n_plans,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "goal_path": str(goal_path), "oracle_path": str(config.oracle_path),
        "sampler_path": config.sampler_path,
        "goal_label": config.goal_label,
        "prompt_char_len": len(student_text),
    }, indent=2))
    logger.info("Done. Run dir: %s", out_dir)


if __name__ == "__main__":
    config = chz.entrypoint(Config)
    main(config)
