"""D5 Phase 0c: Score buffer.jsonl via ten_signal_reward.

For each plan in buffer.jsonl:
  - Strip <think>...</think> preamble and extract <solution>...</solution>
  - Call Qwen3-30B-A3B grader with build_single_call_prompt(goal, plan)
  - Parse scores via parse_scores, aggregate via aggregate_reward
  - Write metrics.jsonl

Run:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.smoke_pathway_v1_score \
        config.api_profile=new
"""
import json
import logging
import re
import time
from pathlib import Path

import chz
import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.ten_signal_reward import (
    aggregate_reward,
    build_single_call_prompt,
    parse_scores,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@chz.chz
class Config:
    api_profile: str | None = "new"
    base_url: str | None = None

    log_path: str = (
        "projects/d5_abstract_retrieve_refine/runs/2026_04_smoke_pathway_v1"
    )
    goal_path: str = (
        "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"
    )

    grader_model: str = "Qwen/Qwen3-30B-A3B"
    grader_max_tokens: int = 8192
    grader_temperature: float = 0.0


_THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_SOLUTION_PATTERN = re.compile(r"<solution>(.*?)</solution>", re.DOTALL)


def extract_solution(plan_text: str) -> str:
    """Strip <think>...</think> preamble, extract <solution>...</solution> body."""
    stripped = _THINK_PATTERN.sub("", plan_text, count=1)
    m = _SOLUTION_PATTERN.search(stripped)
    if m:
        return m.group(1).strip()
    # Fallback: entire stripped text
    return stripped.strip()


def strip_think(text: str) -> str:
    """Remove <think>...</think> block from grader output before XML parsing."""
    return _THINK_PATTERN.sub("", text, count=1).strip()


def main(config: Config):
    repo_root = Path(__file__).resolve().parents[3]
    log_dir = (repo_root / config.log_path).resolve()
    goal_path = (repo_root / config.goal_path).resolve()
    buffer_path = log_dir / "buffer.jsonl"
    metrics_path = log_dir / "metrics.jsonl"

    assert goal_path.exists(), f"Missing goal: {goal_path}"
    assert buffer_path.exists(), f"Missing buffer: {buffer_path}"

    goal = goal_path.read_text().strip()

    # Load buffer
    with open(buffer_path) as f:
        entries = [json.loads(l) for l in f]
    logger.info("Loaded %d plans from %s", len(entries), buffer_path)

    # Setup grader
    tokenizer = get_tokenizer(config.grader_model)
    renderer_name = model_info.get_recommended_renderer_name(config.grader_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile
    )
    grader_client = service_client.create_sampling_client(base_model=config.grader_model)
    grader_params = tinker.types.SamplingParams(
        max_tokens=config.grader_max_tokens,
        temperature=config.grader_temperature,
        stop=renderer.get_stop_sequences(),
    )

    # Launch all grader futures in parallel
    logger.info("Launching %d grader futures ...", len(entries))
    futures = []
    solutions = []
    for e in entries:
        plan = extract_solution(e["plan_text"])
        solutions.append(plan)
        prompt = build_single_call_prompt(goal=goal, plan=plan)
        convo = [{"role": "user", "content": prompt}]
        grader_input = renderer.build_generation_prompt(convo)
        fut = grader_client.sample(
            prompt=grader_input,
            num_samples=1,
            sampling_params=grader_params,
        )
        futures.append(fut)

    # Collect and parse
    per_plan_metrics = []
    for idx, (e, fut, plan) in enumerate(zip(entries, futures, solutions)):
        result = fut.result()
        raw = tokenizer.decode(result.sequences[0].tokens)
        clean = strip_think(raw)
        scores = parse_scores(clean)
        # scores: dict[signal_key, {"score": int|None, "critique": str}]
        signal_scores = {k: v.get("score") for k, v in scores.items()}
        agg = aggregate_reward(signal_scores)

        metric = {
            "condition": e["condition"],
            "sample_idx": e["sample_idx"],
            "solution_char_len": len(plan),
            "signal_scores": signal_scores,
            "aggregate_reward": agg,
        }
        per_plan_metrics.append(metric)
        logger.info(
            "[%2d] %s sample_idx=%d agg=%.3f signals=%s",
            idx, e["condition"], e["sample_idx"], agg,
            {k: v for k, v in signal_scores.items() if v is not None},
        )

    # Write per-plan metrics
    with open(metrics_path, "w") as f:
        for m in per_plan_metrics:
            f.write(json.dumps(m) + "\n")
    logger.info("Wrote per-plan metrics to %s", metrics_path)

    # Summary per condition
    summary = {}
    for cond in {"B_baseline", "A_with_abstraction"}:
        subset = [m for m in per_plan_metrics if m["condition"] == cond]
        aggs = [m["aggregate_reward"] for m in subset if m["aggregate_reward"] is not None]
        if not aggs:
            continue
        # Per-signal means
        signal_keys = set()
        for m in subset:
            for k, v in m["signal_scores"].items():
                if v is not None:
                    signal_keys.add(k)
        per_sig = {}
        for sig_key in sorted(signal_keys):
            vals = [m["signal_scores"].get(sig_key) for m in subset]
            vals = [v for v in vals if v is not None]
            if vals:
                per_sig[sig_key] = {"mean": sum(vals)/len(vals), "n": len(vals)}

        summary[cond] = {
            "n": len(subset),
            "aggregate_mean": sum(aggs)/len(aggs),
            "aggregate_min": min(aggs),
            "aggregate_max": max(aggs),
            "per_signal_mean": per_sig,
        }

    summary_path = log_dir / "scoring_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for cond, s in summary.items():
        logger.info("%s: n=%d, agg_mean=%.4f (min=%.3f, max=%.3f)",
                    cond, s["n"], s["aggregate_mean"], s["aggregate_min"], s["aggregate_max"])
        for sig, sv in s["per_signal_mean"].items():
            logger.info("   %s: mean=%.3f (n=%d)", sig, sv["mean"], sv["n"])
    if "A_with_abstraction" in summary and "B_baseline" in summary:
        delta = summary["A_with_abstraction"]["aggregate_mean"] - summary["B_baseline"]["aggregate_mean"]
        logger.info("Δ(A − B) aggregate = %+.4f", delta)

    logger.info("Full summary at %s", summary_path)


if __name__ == "__main__":
    config = chz.entrypoint(Config)
    main(config)
