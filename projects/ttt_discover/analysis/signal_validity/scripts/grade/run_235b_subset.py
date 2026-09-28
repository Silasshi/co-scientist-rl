#!/usr/bin/env python3
"""Grade a 20-reference subset with 235B grader using same v4 rubrics.

Purpose: determine whether the 30B grader's ~0.54 ceiling is a grader limitation
or a rubric limitation. If 235B gives mean >= 0.7, grader is the bottleneck.
"""

import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import tinker
import tinker.types as types
from tinker_cookbook import model_info, renderers
from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.ten_signal_reward import (
    SIGNALS, aggregate_reward, build_single_signal_prompt, parse_scores,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)
logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
BASE = Path(__file__).resolve().parents[2]
REFS_PATH = BASE / "data" / "refs" / "references.jsonl"
RESULTS_PATH = BASE / "archive" / "grading_235b_subset.jsonl"
GRADER_MODEL = "openai/gpt-oss-120b"
N_SAMPLE = 20
SIGNAL_IDS = [s.id for s in SIGNALS]


def main():
    # Load refs, take first 20
    with open(REFS_PATH) as f:
        all_refs = [json.loads(l) for l in f]
    # Stratified: ~5 per source, or first 20 with diversity
    refs = all_refs[:N_SAMPLE]
    logger.info(f"Grading {len(refs)} references with 235B")

    service_client = create_service_client()
    client = service_client.create_sampling_client(base_model=GRADER_MODEL)
    tokenizer = client.get_tokenizer()
    renderer_name = model_info.get_recommended_renderer_name(GRADER_MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    stop_seqs = renderer.get_stop_sequences()

    with open(RESULTS_PATH, "w") as fout:
        for i, ref in enumerate(refs):
            t0 = time.time()
            # Launch all 9 signal futures in parallel
            futures = {}
            for sig in SIGNALS:
                prompt = build_single_signal_prompt(ref["goal"], ref["reference_solution"], sig)
                mi = renderer.build_generation_prompt([{"role": "user", "content": prompt}])
                futures[sig.id] = client.sample(
                    mi, num_samples=1,
                    sampling_params=types.SamplingParams(
                        max_tokens=2048, temperature=0.0, stop=stop_seqs,
                    ),
                )

            scores = {}
            for sig_id, future in futures.items():
                try:
                    resp = future.result(timeout=180)
                    text = renderers.get_text_content(renderer.parse_response(resp.sequences[0].tokens)[0])
                    parsed = parse_scores(text)
                    scores[sig_id] = parsed.get(sig_id, {}).get("score")
                except Exception as e:
                    logger.warning(f"Failed {ref['source_id']} / {sig_id}: {e}")
                    scores[sig_id] = None

            agg = aggregate_reward(scores)
            out = {
                "source_id": ref["source_id"],
                "source": ref["source"],
                "signals": scores,
                "aggregate": agg,
            }
            fout.write(json.dumps(out) + "\n")
            fout.flush()

            elapsed = time.time() - t0
            logger.info(f"  [{i+1}/{len(refs)}] {ref['source_id']}: agg={agg:.3f}, "
                        f"signals={scores}, elapsed={elapsed:.0f}s")

    # Summary
    with open(RESULTS_PATH) as f:
        results = [json.loads(l) for l in f]
    aggs = [r["aggregate"] for r in results]
    print(f"\n235B grader summary (n={len(aggs)}):")
    print(f"  Mean: {sum(aggs)/len(aggs):.3f}")
    print(f"  Max: {max(aggs):.3f}")
    print(f"  >= 0.7: {sum(1 for a in aggs if a >= 0.7)}/{len(aggs)}")
    print(f"  >= 0.8: {sum(1 for a in aggs if a >= 0.8)}/{len(aggs)}")


if __name__ == "__main__":
    main()
