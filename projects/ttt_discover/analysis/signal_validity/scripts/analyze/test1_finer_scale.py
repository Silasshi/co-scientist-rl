#!/usr/bin/env python3
"""Test 1: Opus 4.1 with 0-100 scale + forced spread. Determine if low correlation
is due to Opus compression or genuine signal-quality mismatch.
"""

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from co_scientist.shared.openrouter_client import OpenRouterClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)
logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
BASE = Path(__file__).resolve().parents[2]
REFS_PATH = BASE / "data" / "refs" / "references_v2.jsonl"
OUT_PATH = BASE / "data" / "human_ratings" / "human_ratings_v2.jsonl"
MODEL = "anthropic/claude-opus-4.1"


REVIEW_PROMPT = """You are an expert research plan reviewer. Rate this plan on a 0-100 scale — use the FULL RANGE.

CALIBRATION:
- 90-100: Exceptional. Top 5% quality. Concrete, rigorous, compelling across all dimensions.
- 80-89: Strong. A well-written published paper at a top venue.
- 70-79: Solid. A publishable methodology with minor weaknesses.
- 60-69: Adequate. Has gaps on some criteria but core ideas work.
- 50-59: Weak. Notable gaps that would likely require revision.
- <50: Poor.

IMPORTANT: Plans in this set are all genuinely of published-paper quality, but they VARY in how fully they satisfy each criterion. Do NOT cluster everything at 80. Find the differentiators:
- A plan with a more specific problem statement scores higher than one with a generic problem
- A plan with 3 named baselines + insufficiency reasons scores higher than one with "prior work" mentioned vaguely
- A plan with exact hyperparameters + architecture scores higher than one with typical defaults
- A plan with 3 specific failure modes + fallbacks scores higher than one with "limitations might exist"

Be precise: if two plans are clearly similar quality, give them the same score. If one is noticeably better on any criterion, reflect it in the score.

Research goal:
{goal}

Research plan:
{plan}

Rate the plan on each criterion (0-100) and overall (0-100):

{{
  "problem": <int 0-100>,
  "motivation": <int 0-100>,
  "core_idea": <int 0-100>,
  "methodology": <int 0-100>,
  "evaluation": <int 0-100>,
  "risk_awareness": <int 0-100>,
  "overall": <int 0-100>,
  "key_strength": "<one sentence>",
  "key_weakness": "<one sentence>"
}}

Output ONLY the JSON, no preamble.
"""


async def review_one(client: OpenRouterClient, ref: dict) -> dict | None:
    msg = [{"role": "user", "content": REVIEW_PROMPT.format(
        goal=ref["goal"],
        plan=ref["reference_solution"],
    )}]
    resp = await client.chat(MODEL, msg, temperature=0.0, max_tokens=1000)
    m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', resp, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        data["source_id"] = ref["source_id"]
        data["source"] = ref.get("source", "")
        return data
    except json.JSONDecodeError:
        return None


async def main():
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("Set OPENROUTER_API_KEY")

    with open(REFS_PATH) as f:
        refs = [json.loads(l) for l in f]

    done = set()
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            for line in f:
                done.add(json.loads(line)["source_id"])
    pending = [r for r in refs if r["source_id"] not in done]
    logger.info(f"Pending: {len(pending)}/{len(refs)}")

    async with OpenRouterClient() as client:
        with open(OUT_PATH, "a") as fout:
            for i, ref in enumerate(pending):
                try:
                    rating = await review_one(client, ref)
                    if rating:
                        fout.write(json.dumps(rating) + "\n")
                        fout.flush()
                        logger.info(f"  [{i+1}/{len(pending)}] {ref['source_id']}: overall={rating['overall']}")
                    else:
                        logger.warning(f"  Failed parse: {ref['source_id']}")
                except Exception as e:
                    logger.error(f"  Error {ref['source_id']}: {e}")
        logger.info(f"Tokens: {client.total_tokens}")

    # Compute correlation
    import statistics
    from scipy.stats import pearsonr, spearmanr
    ratings = {}
    with open(OUT_PATH) as f:
        for line in f:
            r = json.loads(line)
            ratings[r["source_id"]] = r
    autos = {}
    with open(BASE / "archive" / "grading_results_v2.jsonl") as f:
        for line in f:
            r = json.loads(line)
            autos[r["source_id"]] = r["aggregate"]

    pairs = [(ratings[s]["overall"] / 100.0, autos[s]) for s in ratings if s in autos]
    xs, ys = zip(*pairs)
    pr = pearsonr(xs, ys)
    sr = spearmanr(xs, ys)
    opus_vals = [r["overall"] for r in ratings.values()]
    print(f"\n=== Opus 0-100 vs Auto (n={len(pairs)}) ===")
    print(f"  Pearson r: {pr.statistic:.3f}  (R² = {pr.statistic**2:.3f})")
    print(f"  Spearman ρ: {sr.statistic:.3f}")
    print(f"  Opus distribution: mean={statistics.mean(opus_vals):.1f}, "
          f"min={min(opus_vals)}, max={max(opus_vals)}, std={statistics.stdev(opus_vals):.1f}")


if __name__ == "__main__":
    asyncio.run(main())
