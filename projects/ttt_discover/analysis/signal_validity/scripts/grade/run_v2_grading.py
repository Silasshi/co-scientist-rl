#!/usr/bin/env python3
"""Grade references_v2 with updated 6-criteria-aligned signals.

Target: mean aggregate >= 0.8 on reference plans.
"""

import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev

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
REFS_PATH = BASE / "data" / "refs" / "references_v2.jsonl"
RESULTS_PATH = BASE / "archive" / "grading_results_v2.jsonl"
GRADER_MODEL = "Qwen/Qwen3-30B-A3B"
WAVE_SIZE = 4
SIGNAL_IDS = [s.id for s in SIGNALS]


def main():
    with open(REFS_PATH) as f:
        refs = [json.loads(l) for l in f]
    logger.info(f"Grading {len(refs)} v2 references with {GRADER_MODEL}")

    done = set()
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            for line in f:
                done.add(json.loads(line)['source_id'])
        logger.info(f"Resuming: {len(done)} already graded")

    svc = create_service_client()
    client = svc.create_sampling_client(base_model=GRADER_MODEL)
    tok = client.get_tokenizer()
    renderer_name = model_info.get_recommended_renderer_name(GRADER_MODEL)
    renderer = renderers.get_renderer(renderer_name, tok)
    stop = renderer.get_stop_sequences()

    pending = [r for r in refs if r['source_id'] not in done]
    logger.info(f"Processing {len(pending)} remaining")

    t_start = time.time()
    with open(RESULTS_PATH, "a") as fout:
        for wave_start in range(0, len(pending), WAVE_SIZE):
            wave = pending[wave_start:wave_start + WAVE_SIZE]
            wave_futures = []
            for ref in wave:
                fs = {}
                for sig in SIGNALS:
                    prompt = build_single_signal_prompt(ref['goal'], ref['reference_solution'], sig)
                    mi = renderer.build_generation_prompt([{"role": "user", "content": prompt}])
                    fs[sig.id] = client.sample(
                        mi, num_samples=1,
                        sampling_params=types.SamplingParams(max_tokens=2048, temperature=0.0, stop=stop),
                    )
                wave_futures.append((ref, fs))

            for ref, fs in wave_futures:
                scores = {}
                for sid, future in fs.items():
                    try:
                        resp = future.result(timeout=180)
                        text = renderers.get_text_content(renderer.parse_response(resp.sequences[0].tokens)[0])
                        parsed = parse_scores(text)
                        scores[sid] = parsed.get(sid, {}).get('score')
                    except Exception as e:
                        logger.warning(f"  Failed {ref['source_id']} / {sid}: {e}")
                        scores[sid] = None
                agg = aggregate_reward(scores)
                out = {
                    "source_id": ref['source_id'],
                    "source": ref['source'],
                    "subdomain": ref['subdomain'],
                    "signals": scores,
                    "aggregate": agg,
                }
                fout.write(json.dumps(out) + "\n")
                fout.flush()

                elapsed = time.time() - t_start
                n_done = wave_start + len(wave)
                eta = elapsed / n_done * (len(pending) - n_done) if n_done > 0 else 0
                logger.info(f"  [{wave_start + 1 + wave.index(ref)}/{len(pending)}] {ref['source_id']}: agg={agg:.3f} | ETA={eta:.0f}s")

    # Summary
    with open(RESULTS_PATH) as f:
        results = [json.loads(l) for l in f]
    aggs = [r['aggregate'] for r in results]
    print(f"\n=== v2 Results (n={len(aggs)}) ===")
    print(f"  Mean: {mean(aggs):.3f}")
    print(f"  Median: {median(aggs):.3f}")
    print(f"  Max: {max(aggs):.3f}")
    print(f"  >= 0.7: {sum(1 for a in aggs if a >= 0.7)}/{len(aggs)} ({sum(1 for a in aggs if a >= 0.7)/len(aggs):.0%})")
    print(f"  >= 0.8: {sum(1 for a in aggs if a >= 0.8)}/{len(aggs)} ({sum(1 for a in aggs if a >= 0.8)/len(aggs):.0%})")


if __name__ == "__main__":
    main()
