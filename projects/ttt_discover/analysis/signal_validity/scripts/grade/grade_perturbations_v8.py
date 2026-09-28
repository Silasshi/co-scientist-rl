#!/usr/bin/env python3
"""Re-grade all perturbations with v8 rubrics, N=5.

Output: grading_perturbations_v8.jsonl
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
PERTURBS_PATH = BASE / "data" / "perturbations" / "perturbations.jsonl"
RESULTS_PATH = BASE / "data" / "perturbations" / "grading_perturbations_v8.jsonl"
GRADER_MODEL = "Qwen/Qwen3-30B-A3B"
N_REPEATS = 5
WAVE_SIZE = 2
SIGNAL_IDS = [s.id for s in SIGNALS]


def main():
    with open(PERTURBS_PATH) as f:
        perturbs = [json.loads(l) for l in f]
    logger.info(f"Grading {len(perturbs)} perturbations with v8 rubrics, N={N_REPEATS}")

    done = set()
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            for line in f:
                done.add(json.loads(line)['perturbation_id'])
    pending = [p for p in perturbs if p['perturbation_id'] not in done]
    logger.info(f"Pending: {len(pending)}")

    svc = create_service_client()
    client = svc.create_sampling_client(base_model=GRADER_MODEL)
    tok = client.get_tokenizer()
    renderer_name = model_info.get_recommended_renderer_name(GRADER_MODEL)
    renderer = renderers.get_renderer(renderer_name, tok)
    stop = renderer.get_stop_sequences()

    t_start = time.time()
    with open(RESULTS_PATH, "a") as fout:
        for wave_start in range(0, len(pending), WAVE_SIZE):
            wave = pending[wave_start:wave_start + WAVE_SIZE]
            wave_futures = []
            for p in wave:
                per_p = {sid: [] for sid in SIGNAL_IDS}
                for rep in range(N_REPEATS):
                    for sig in SIGNALS:
                        prompt = build_single_signal_prompt(
                            p['goal'], p['perturbed_plan'], sig
                        )
                        mi = renderer.build_generation_prompt(
                            [{"role": "user", "content": prompt}]
                        )
                        future = client.sample(
                            mi, num_samples=1,
                            sampling_params=types.SamplingParams(
                                max_tokens=4096,
                                temperature=0.3 if rep > 0 else 0.0,
                                stop=stop,
                            ),
                        )
                        per_p[sig.id].append(future)
                wave_futures.append((p, per_p))

            for p, per_p in wave_futures:
                per_repeat_scores = [{sid: None for sid in SIGNAL_IDS} for _ in range(N_REPEATS)]
                reasoning = {sid: [] for sid in SIGNAL_IDS}
                for sid, futures in per_p.items():
                    for rep, future in enumerate(futures):
                        try:
                            resp = future.result(timeout=180)
                            text = renderers.get_text_content(
                                renderer.parse_response(resp.sequences[0].tokens)[0]
                            )
                            parsed = parse_scores(text)
                            info = parsed.get(sid, {})
                            per_repeat_scores[rep][sid] = info.get('score')
                            reasoning[sid].append(info.get('reasoning', ''))
                        except Exception as e:
                            logger.warning(f"  {p['perturbation_id']} {sid} rep {rep}: {e}")
                            reasoning[sid].append(None)

                signals_median = {}
                for sid in SIGNAL_IDS:
                    vals = [per_repeat_scores[r][sid] for r in range(N_REPEATS)
                            if per_repeat_scores[r][sid] is not None]
                    signals_median[sid] = int(statistics.median_low(vals)) if vals else None
                agg = aggregate_reward(signals_median)

                out = {
                    "perturbation_id": p['perturbation_id'],
                    "base_id": p['base_id'],
                    "perturbation_type": p['perturbation_type'],
                    "target_signal": p['target_signal'],
                    "signals_median": signals_median,
                    "signals_per_repeat": per_repeat_scores,
                    "reasoning_per_signal": reasoning,
                    "aggregate": agg,
                }
                fout.write(json.dumps(out) + "\n")
                fout.flush()

                elapsed = time.time() - t_start
                n_done = wave_start + 1 + wave.index(p)
                eta = elapsed / n_done * (len(pending) - n_done) if n_done > 0 else 0
                logger.info(f"  [{n_done}/{len(pending)}] {p['perturbation_id']}: agg={agg:.3f} target={p['target_signal']} | ETA={eta:.0f}s")

    logger.info("Done grading perturbations v8.")


if __name__ == "__main__":
    main()
