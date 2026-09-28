#!/usr/bin/env python3
"""Use Claude Opus 4.1 as a rigorous reviewer to rate all 60 reference plans on the 6 criteria.

Produces:
- human_ratings.jsonl: per-plan scores (1-10) on each criterion + overall
- Correlation analysis with auto grading scores

This is intended as a proxy for human review (Opus is careful and consistent).
Spot-check 5 plans manually after to verify Opus calibration.
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
OUT_PATH = BASE / "data" / "human_ratings" / "human_ratings.jsonl"
MODEL = "anthropic/claude-opus-4.1"


REVIEW_PROMPT = """You are an expert research plan reviewer evaluating a plan against the 6-criteria framework.

Research goal:
{goal}

Research plan:
{plan}

Rate the plan on EACH of the 6 criteria on a 1-10 scale, where:
- 10 = exceptional (would be a top-tier publication contribution on this criterion)
- 8 = strong (typical good published paper quality)
- 6 = adequate (minor weaknesses)
- 4 = weak (significant gaps)
- 2 = poor (major failing)
- 1 = absent (criterion not addressed)

The 6 criteria:
1. **Problem**: Is the problem clearly stated with importance and specific gap?
2. **Motivation**: Does it name specific prior methods (2-5) and explain their specific insufficiency?
3. **Core Idea**: Is there a one-sentence core hypothesis + mechanism intuition + plausibility?
4. **Methodology**: Is the methodology detailed enough to judge feasibility (not over-specified)?
5. **Evaluation**: Are metrics, success criteria, and baselines specific and discriminative?
6. **Risk Awareness**: Does it name specific failure modes and acknowledge scope boundaries (not generic disclaimers)?

Be calibrated and strict. A well-written published paper methodology with all 6 criteria fully addressed should typically score 7-8 per criterion. Score 9-10 only for exceptional clarity or rigor. Score 5-6 if a criterion is partially addressed but notably weak.

Output in EXACTLY this JSON format (no preamble, no markdown):
{{
  "problem": <int 1-10>,
  "motivation": <int 1-10>,
  "core_idea": <int 1-10>,
  "methodology": <int 1-10>,
  "evaluation": <int 1-10>,
  "risk_awareness": <int 1-10>,
  "overall": <int 1-10>,
  "brief_reasoning": "<2-3 sentences summarizing strengths/weaknesses>"
}}
"""


async def review_one(client: OpenRouterClient, ref: dict) -> dict | None:
    msg = [{"role": "user", "content": REVIEW_PROMPT.format(
        goal=ref["goal"],
        plan=ref["reference_solution"],
    )}]
    resp = await client.chat(MODEL, msg, temperature=0.0, max_tokens=800)
    # Extract JSON
    m = re.search(r'\{[^{}]*\}', resp, re.DOTALL)
    if not m:
        logger.warning(f"No JSON in response for {ref['source_id']}: {resp[:200]}")
        return None
    try:
        data = json.loads(m.group(0))
        data["source_id"] = ref["source_id"]
        data["source"] = ref.get("source", "")
        return data
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed for {ref['source_id']}: {e}")
        return None


async def main():
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("Set OPENROUTER_API_KEY")

    with open(REFS_PATH) as f:
        refs = [json.loads(l) for l in f]

    # Resume
    done = set()
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            for line in f:
                done.add(json.loads(line)["source_id"])
    logger.info(f"Total {len(refs)}, already rated {len(done)}")

    pending = [r for r in refs if r["source_id"] not in done]
    async with OpenRouterClient() as client:
        with open(OUT_PATH, "a") as fout:
            for i, ref in enumerate(pending):
                try:
                    rating = await review_one(client, ref)
                    if rating:
                        fout.write(json.dumps(rating) + "\n")
                        fout.flush()
                        logger.info(
                            f"  [{i+1}/{len(pending)}] {ref['source_id']}: "
                            f"overall={rating['overall']}, "
                            f"min={min(rating[k] for k in ['problem','motivation','core_idea','methodology','evaluation','risk_awareness'])}"
                        )
                except Exception as e:
                    logger.error(f"  Failed {ref['source_id']}: {e}")
        logger.info(f"Tokens: {client.total_tokens}")

    # Compute correlation with auto scores
    grading_path = BASE / "archive" / "grading_results_v2.jsonl"
    if grading_path.exists():
        with open(OUT_PATH) as f:
            ratings = {r["source_id"]: json.loads(l) if (l := r) else None for r in [json.loads(ln) for ln in f]}
            # fix: read properly
        ratings = {}
        with open(OUT_PATH) as f:
            for line in f:
                r = json.loads(line)
                ratings[r["source_id"]] = r

        autos = {}
        with open(grading_path) as f:
            for line in f:
                r = json.loads(line)
                autos[r["source_id"]] = r["aggregate"]

        pairs = []
        for sid, rating in ratings.items():
            if sid in autos:
                # Scale human 1-10 to 0-1 to compare with auto 0-1
                pairs.append((rating["overall"] / 10.0, autos[sid]))

        if pairs:
            import statistics
            from scipy.stats import pearsonr, spearmanr
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            pr = pearsonr(xs, ys)
            sr = spearmanr(xs, ys)
            print(f"\n=== Human (Opus) vs Auto Correlation (n={len(pairs)}) ===")
            print(f"  Pearson r: {pr.statistic:.3f} (R² = {pr.statistic**2:.3f})")
            print(f"  Spearman ρ: {sr.statistic:.3f}")
            print(f"  Human mean: {statistics.mean(xs):.3f}")
            print(f"  Auto mean:  {statistics.mean(ys):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
