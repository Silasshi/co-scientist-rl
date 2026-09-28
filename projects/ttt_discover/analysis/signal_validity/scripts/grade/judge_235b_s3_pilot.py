#!/usr/bin/env python3
"""Phase 0d: Use Qwen3-235B as external judge to validate 30B's S3 grading
and locus attributions on the 24-plan pilot.

Two outputs per plan:
  1. 235B's independent S3 score (ground-truth proxy)
  2. 235B's judgment on each of 30B's emitted loci:
     - For each 30B locus, does 235B agree this is a valid weakness quote?

This gives:
  - score agreement rate: how often do 30B and 235B agree on S3 score (±1)?
  - locus attribution validity: of loci 30B emitted, what fraction does 235B
    rate as CORRECT (real weakness) vs INCORRECT (fabricated / fine)?

Expected outcome (per Phase 0 diagnostic):
  - Score disagreement will be large — 30B hallucinates high scores on plans
    lacking named prior methods; 235B should correctly score them low
  - Locus attribution will fail for most non-verbatim loci

24 plans × 1 grading call = 24 calls. ~8-10 min.
"""
import json
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import tinker.types as types
from tinker_cookbook import model_info, renderers
from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.ten_signal_reward import SIGNALS, build_single_signal_prompt, parse_scores

sys.path.insert(0, str(Path(__file__).parent))
from grade_locus_s3_pilot import extract_plan_body

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]
SAMPLE_PATH = BASE / "data" / "locus_pilot" / "sample_s3_pilot.jsonl"
LOCUS_GRADING_PATH = BASE / "data" / "locus_pilot" / "grading_locus_s3.jsonl"
OUT_PATH = BASE / "data" / "locus_pilot" / "judge_235b_s3.jsonl"
GOAL_PATH = PROJECT_ROOT / "projects/ttt_discover/analysis/sanity_check/research_goal.txt"

JUDGE_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
MAX_TOKENS = 8192
S3_SPEC = next(s for s in SIGNALS if s.id == "S3_positioning")


def build_judge_prompt(goal: str, plan_body: str, score_30b: int | None,
                        loci_30b: list[dict]) -> str:
    """Ask 235B to (a) independently score S3 and (b) judge each 30B locus."""
    canonical = build_single_signal_prompt(goal, plan_body, S3_SPEC)

    loci_summary = json.dumps(loci_30b, indent=2) if loci_30b else "[] (no loci emitted)"

    suffix = f"""

---

# Additional Validation Task

Another grader (30B model) previously assigned S3 score = {score_30b} and
emitted the following "weakness loci":

```json
{loci_summary}
```

As a separate, independent verification step, output a <validation> block
containing a JSON object with these keys:

  - "judge_score": YOUR independent S3 score (1-5), based on the plan and
    rubric — ignore the 30B score above
  - "judge_score_agrees_within_1": true if your score is within ±1 of the
    30B score, else false
  - "loci_judgments": for EACH 30B locus entry (in order, same length as
    input), an object:
      - "quote_is_verbatim_in_plan": true if the 30B quote actually appears
        in the plan text (you must verify by Ctrl-F), else false
      - "attribution_correct": one of "CORRECT" (locus IS a real S3
        weakness per rubric), "INCORRECT_PARAPHRASED" (content is real but
        quote differs from plan), "INCORRECT_FABRICATED" (quote doesn't
        exist in plan AND reflects content not actually present),
        "INCORRECT_NOT_WEAKNESS" (quote is verbatim but isn't actually a
        positioning weakness — the grader misidentified)
      - "brief_why": 1 sentence

Format:
<validation>
{{
  "judge_score": INT,
  "judge_score_agrees_within_1": BOOL,
  "loci_judgments": [
    {{"quote_is_verbatim_in_plan": BOOL, "attribution_correct": STRING, "brief_why": STRING}},
    ...
  ]
}}
</validation>
"""
    return canonical + suffix


VAL_RE = re.compile(r"<validation>\s*(.*?)\s*</validation>", re.DOTALL | re.IGNORECASE)


def parse_validation(text: str) -> dict | None:
    m = VAL_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def main():
    goal = GOAL_PATH.read_text().strip()

    with SAMPLE_PATH.open() as f:
        samples = {s["sample_id"]: s for s in [json.loads(l) for l in f]}
    with LOCUS_GRADING_PATH.open() as f:
        locus_rows = [json.loads(l) for l in f]
    logger.info(f"Loaded {len(locus_rows)} 30B locus grading rows")

    done = set()
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            done = {json.loads(l)["sample_id"] for l in f}
        logger.info(f"Resuming — {len(done)} already judged")
    pending = [r for r in locus_rows if r["sample_id"] not in done]
    logger.info(f"Pending: {len(pending)}")

    svc = create_service_client()
    client = svc.create_sampling_client(base_model=JUDGE_MODEL)
    tok = client.get_tokenizer()
    renderer_name = model_info.get_recommended_renderer_name(JUDGE_MODEL)
    renderer = renderers.get_renderer(renderer_name, tok)
    stop = renderer.get_stop_sequences()

    with OUT_PATH.open("a") as fout:
        for lrow in pending:
            sid = lrow["sample_id"]
            s = samples[sid]
            plan_body, _ = extract_plan_body(s["plan_text"])
            loci_30b = lrow.get("locus_raw") or []
            # Strip the why field from 30B loci to avoid biasing 235B
            loci_input = [
                {
                    "section_hint": e.get("section_hint", ""),
                    "quote": e.get("quote", ""),
                }
                for e in loci_30b if isinstance(e, dict)
            ]

            prompt = build_judge_prompt(goal, plan_body, lrow["score_locus"], loci_input)
            mi = renderer.build_generation_prompt(
                [{"role": "user", "content": prompt}]
            )
            future = client.sample(
                mi, num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=MAX_TOKENS, temperature=0.0, stop=stop,
                ),
            )
            resp = future.result(timeout=480)
            text = renderers.get_text_content(
                renderer.parse_response(resp.sequences[0].tokens)[0]
            )

            # Also parse 235B's own <evaluation> block to get its independent S3 score
            eval_parsed = parse_scores(text).get("S3_positioning", {})
            judge_score_canonical = eval_parsed.get("score")
            validation = parse_validation(text)

            out = {
                "sample_id": sid,
                "orig_s3": lrow["s3_score_orig"],
                "score_30b_locus": lrow["score_locus"],
                "score_235b_canonical": judge_score_canonical,
                "validation": validation,
                "n_loci_30b": len(loci_input),
                "raw_head": text[:2500],
                "raw_tail": text[-1500:] if len(text) > 2500 else "",
            }
            fout.write(json.dumps(out) + "\n")
            fout.flush()
            logger.info(
                f"  {sid}: orig={lrow['s3_score_orig']} "
                f"30B={lrow['score_locus']} 235B={judge_score_canonical} "
                f"val_ok={validation is not None}"
            )

    # Summary
    with OUT_PATH.open() as f:
        rows = [json.loads(l) for l in f]
    print("\n" + "=" * 75)
    print(f"{'sample_id':<16} {'orig':<5} {'30B':<5} {'235B':<5} {'|diff|':<7} {'loci_judg':<10}")
    print("-" * 75)
    diffs = []
    attribution_counts = {"CORRECT": 0, "INCORRECT_PARAPHRASED": 0,
                          "INCORRECT_FABRICATED": 0, "INCORRECT_NOT_WEAKNESS": 0,
                          "other": 0}
    for r in rows:
        s30, s235 = r["score_30b_locus"], r["score_235b_canonical"]
        if s30 is not None and s235 is not None:
            d = abs(s30 - s235)
            diffs.append(d)
        else:
            d = "?"
        val = r["validation"] or {}
        judgments = val.get("loci_judgments", []) if isinstance(val, dict) else []
        for j in judgments:
            ac = j.get("attribution_correct", "other")
            attribution_counts[ac] = attribution_counts.get(ac, 0) + 1
        judg_str = ",".join(
            j.get("attribution_correct", "?")[:3] if isinstance(j, dict) else "?"
            for j in judgments
        ) or "-"
        print(f"{r['sample_id']:<16} {r['orig_s3']:<5} {s30!s:<5} {s235!s:<5} "
              f"{d!s:<7} {judg_str[:10]}")

    if diffs:
        mean_diff = sum(diffs) / len(diffs)
        agree_within_1 = sum(1 for d in diffs if d <= 1) / len(diffs)
        print(f"\n30B vs 235B score: mean |diff|={mean_diff:.2f}, agree-within-1={100*agree_within_1:.0f}%")
    total_loci = sum(attribution_counts.values())
    if total_loci:
        print(f"\nLocus attribution breakdown (N={total_loci}):")
        for k, v in attribution_counts.items():
            print(f"  {k}: {v} ({100*v/total_loci:.0f}%)")


if __name__ == "__main__":
    main()
