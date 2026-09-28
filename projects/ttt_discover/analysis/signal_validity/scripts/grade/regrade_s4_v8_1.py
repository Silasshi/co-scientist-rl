#!/usr/bin/env python3
"""Re-grade ONLY S4_significance on all refs + perturbations with v8.1 rubric.

Uses Tinker (Qwen3-30B-A3B, not OpenRouter). N=5 repeats per plan, so this is
(60 + 162) * 5 = 1110 API calls — 1/9 of a full re-grade.

Output: s4_regraded_v8_1.jsonl — merged into v8 results by analyze_v8_1.py.
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
    SIGNALS, build_single_signal_prompt, parse_scores,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)
logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
BASE = Path(__file__).resolve().parents[2]
REFS_PATH = BASE / "data" / "refs" / "references_v2.jsonl"
PERTURBS_PATH = BASE / "data" / "perturbations" / "perturbations.jsonl"
RESULTS_PATH = BASE / "data" / "perturbations" / "s4_regraded_v8_1.jsonl"
GRADER_MODEL = "Qwen/Qwen3-30B-A3B"
N_REPEATS = 5
WAVE_SIZE = 4  # can be larger since we only run 1 signal per plan

S4_SPEC = [s for s in SIGNALS if s.id == "S4_significance"][0]


def load_items():
    """Yield (id, plan_text, goal, item_type)."""
    items = []
    with open(REFS_PATH) as f:
        for line in f:
            r = json.loads(line)
            items.append({
                "id": r['source_id'],
                "type": "ref",
                "plan": r['reference_solution'],
                "goal": r['goal'],
            })
    with open(PERTURBS_PATH) as f:
        for line in f:
            p = json.loads(line)
            items.append({
                "id": p['perturbation_id'],
                "type": "perturbation",
                "plan": p['perturbed_plan'],
                "goal": p['goal'],
            })
    return items


def main():
    items = load_items()
    logger.info(f"Re-grading S4 on {len(items)} items (refs + perturbations), N={N_REPEATS}")

    done = set()
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            for line in f:
                done.add(json.loads(line)['id'])
        logger.info(f"Resuming: {len(done)} already graded")
    pending = [it for it in items if it['id'] not in done]
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
            for it in wave:
                futures = []
                for rep in range(N_REPEATS):
                    prompt = build_single_signal_prompt(it['goal'], it['plan'], S4_SPEC)
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
                    futures.append(future)
                wave_futures.append((it, futures))

            for it, futures in wave_futures:
                per_repeat = []
                reasoning = []
                for rep, future in enumerate(futures):
                    try:
                        resp = future.result(timeout=180)
                        text = renderers.get_text_content(
                            renderer.parse_response(resp.sequences[0].tokens)[0]
                        )
                        parsed = parse_scores(text)
                        info = parsed.get("S4_significance", {})
                        per_repeat.append(info.get('score'))
                        reasoning.append(info.get('reasoning', ''))
                    except Exception as e:
                        logger.warning(f"  {it['id']} rep {rep}: {e}")
                        per_repeat.append(None)
                        reasoning.append(None)

                vals = [v for v in per_repeat if v is not None]
                s4_median = int(statistics.median_low(vals)) if vals else None

                out = {
                    "id": it['id'],
                    "type": it['type'],
                    "S4_per_repeat": per_repeat,
                    "S4_median": s4_median,
                    "S4_reasoning": reasoning,
                }
                fout.write(json.dumps(out) + "\n")
                fout.flush()

                elapsed = time.time() - t_start
                n_done = wave_start + 1 + wave.index(it)
                eta = elapsed / n_done * (len(pending) - n_done) if n_done > 0 else 0
                logger.info(f"  [{n_done}/{len(pending)}] {it['id']}: S4={s4_median} | ETA={eta:.0f}s")

    logger.info("Done re-grading S4.")


if __name__ == "__main__":
    main()
