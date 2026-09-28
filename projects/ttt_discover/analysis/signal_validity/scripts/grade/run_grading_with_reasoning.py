#!/usr/bin/env python3
"""Grade 60 refs with N=5 repeats, saving per-repeat scores AND reasoning text.

This is Test 2 (noise reduction via repeats) + data preservation.

Output: grading_results_v5.jsonl with structure:
  {
    "source_id": ...,
    "signals_median": {...},          # median across N repeats
    "signals_per_repeat": [{...}]×N,  # all raw scores per repeat
    "reasoning_per_signal": {sid: [text×N]},  # grader reasoning for debugging
    "aggregate": float,
  }
"""

import json
import logging
import statistics
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
REFS_PATH = BASE / "data" / "refs" / "references_v2.jsonl"
RESULTS_PATH = BASE / "archive" / "grading_results_v5.jsonl"
GRADER_MODEL = "Qwen/Qwen3-30B-A3B"
N_REPEATS = 5               # Test 2: increase from default 2 to 5
WAVE_SIZE = 2               # 2 refs × 9 signals × 5 repeats = 90 concurrent
SIGNAL_IDS = [s.id for s in SIGNALS]


def main():
    with open(REFS_PATH) as f:
        refs = [json.loads(l) for l in f]
    logger.info(f"Grading {len(refs)} refs with N={N_REPEATS} repeats each")

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
            # Launch N_REPEATS × 9 signals = 45 futures per ref × WAVE_SIZE refs
            wave_futures = []
            for ref in wave:
                per_ref = {sid: [] for sid in SIGNAL_IDS}
                for rep in range(N_REPEATS):
                    for sig in SIGNALS:
                        prompt = build_single_signal_prompt(
                            ref['goal'], ref['reference_solution'], sig
                        )
                        mi = renderer.build_generation_prompt(
                            [{"role": "user", "content": prompt}]
                        )
                        future = client.sample(
                            mi, num_samples=1,
                            sampling_params=types.SamplingParams(
                                max_tokens=2048,
                                # Vary temperature slightly across repeats for noise reduction
                                temperature=0.3 if rep > 0 else 0.0,
                                stop=stop,
                            ),
                        )
                        per_ref[sig.id].append(future)
                wave_futures.append((ref, per_ref))

            # Collect
            for ref, per_ref in wave_futures:
                per_repeat_scores = [{sid: None for sid in SIGNAL_IDS} for _ in range(N_REPEATS)]
                per_signal_reasoning = {sid: [] for sid in SIGNAL_IDS}

                for sid, futures in per_ref.items():
                    for rep, future in enumerate(futures):
                        try:
                            resp = future.result(timeout=180)
                            text = renderers.get_text_content(
                                renderer.parse_response(resp.sequences[0].tokens)[0]
                            )
                            parsed = parse_scores(text)
                            info = parsed.get(sid, {})
                            per_repeat_scores[rep][sid] = info.get('score')
                            per_signal_reasoning[sid].append(info.get('reasoning', ''))
                        except Exception as e:
                            logger.warning(f"  {ref['source_id']} {sid} rep {rep}: {e}")
                            per_signal_reasoning[sid].append(None)

                # Compute median across repeats per signal
                signals_median = {}
                for sid in SIGNAL_IDS:
                    vals = [per_repeat_scores[r][sid] for r in range(N_REPEATS)
                            if per_repeat_scores[r][sid] is not None]
                    signals_median[sid] = int(statistics.median_low(vals)) if vals else None

                agg = aggregate_reward(signals_median)

                out = {
                    "source_id": ref['source_id'],
                    "source": ref['source'],
                    "subdomain": ref.get('subdomain', ''),
                    "signals_median": signals_median,
                    "signals_per_repeat": per_repeat_scores,
                    "reasoning_per_signal": per_signal_reasoning,
                    "aggregate": agg,
                    "n_repeats": N_REPEATS,
                }
                fout.write(json.dumps(out) + "\n")
                fout.flush()

                elapsed = time.time() - t_start
                n_done = wave_start + 1 + wave.index(ref)
                eta = elapsed / n_done * (len(pending) - n_done) if n_done > 0 else 0
                logger.info(f"  [{n_done}/{len(pending)}] {ref['source_id']}: agg={agg:.3f} | ETA={eta:.0f}s")

    # Summary
    with open(RESULTS_PATH) as f:
        results = [json.loads(l) for l in f]
    aggs = [r['aggregate'] for r in results]
    print(f"\n=== v5 (N=5 repeats) Results (n={len(aggs)}) ===")
    print(f"  Mean: {statistics.mean(aggs):.3f}")
    print(f"  Median: {statistics.median(aggs):.3f}")
    print(f"  Max: {max(aggs):.3f}")
    print(f"  >= 0.7: {sum(1 for a in aggs if a >= 0.7)}/{len(aggs)} ({sum(1 for a in aggs if a >= 0.7)/len(aggs):.0%})")
    print(f"  >= 0.8: {sum(1 for a in aggs if a >= 0.8)}/{len(aggs)} ({sum(1 for a in aggs if a >= 0.8)/len(aggs):.0%})")

    # Per-signal noise analysis
    print(f"\n=== Per-signal noise (per-repeat stddev) ===")
    from collections import defaultdict
    sig_stdevs = defaultdict(list)
    for r in results:
        for sid in SIGNAL_IDS:
            vals = [r['signals_per_repeat'][rep][sid] for rep in range(N_REPEATS)
                    if r['signals_per_repeat'][rep][sid] is not None]
            if len(vals) >= 2:
                sig_stdevs[sid].append(statistics.stdev(vals))
    for sid in SIGNAL_IDS:
        if sig_stdevs[sid]:
            print(f"  {sid:<22} mean_stddev={statistics.mean(sig_stdevs[sid]):.2f}")


if __name__ == "__main__":
    main()
