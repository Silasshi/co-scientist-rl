#!/usr/bin/env python3
"""One-off diagnostic: run canonical S3 prompt (no locus directive) on the
same 3 plans that the locus-extended grader produced suspicious scores for.

Comparison logic:
  - If canonical score == locus score AND both are high despite orig=1/2
    → 30B is intrinsically hallucinating on this goal's S3 (locus NOT the cause)
  - If canonical score is low (matches orig) but locus score is high
    → locus directive is inducing hallucination (prompt-fix may help)
  - If canonical score is ALSO high
    → original buffer scores were from corrupted max_tokens=2048 cuts, not
      real weaknesses; our pilot sample is unreliable

Runs 3 samples × 1 repeat = 3 API calls.
"""
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import tinker.types as types
from tinker_cookbook import model_info, renderers
from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.ten_signal_reward import SIGNALS, build_single_signal_prompt, parse_scores

# Import extract_plan_body from the pilot grader
sys.path.insert(0, str(Path(__file__).parent))
from grade_locus_s3_pilot import extract_plan_body

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]
SAMPLE_PATH = BASE / "data" / "locus_pilot" / "sample_s3_pilot.jsonl"
LOCUS_GRADING_PATH = BASE / "data" / "locus_pilot" / "grading_locus_s3.jsonl"
GOAL_PATH = PROJECT_ROOT / "projects/ttt_discover/analysis/sanity_check/research_goal.txt"
OUT_PATH = BASE / "data" / "locus_pilot" / "diag_canonical_s3.jsonl"

GRADER_MODEL = "Qwen/Qwen3-30B-A3B"
MAX_TOKENS = 8192
S3_SPEC = next(s for s in SIGNALS if s.id == "S3_positioning")


def main():
    goal = GOAL_PATH.read_text().strip()

    # Read sample + locus grading to know which plans to test
    with SAMPLE_PATH.open() as f:
        samples = {s["sample_id"]: s for s in [json.loads(l) for l in f]}
    with LOCUS_GRADING_PATH.open() as f:
        locus_rows = [json.loads(l) for l in f]

    svc = create_service_client()
    client = svc.create_sampling_client(base_model=GRADER_MODEL)
    tok = client.get_tokenizer()
    renderer_name = model_info.get_recommended_renderer_name(GRADER_MODEL)
    renderer = renderers.get_renderer(renderer_name, tok)
    stop = renderer.get_stop_sequences()

    with OUT_PATH.open("w") as fout:
        for lrow in locus_rows:
            sid = lrow["sample_id"]
            s = samples[sid]
            plan_body, _ = extract_plan_body(s["plan_text"])

            prompt = build_single_signal_prompt(goal, plan_body, S3_SPEC)
            mi = renderer.build_generation_prompt(
                [{"role": "user", "content": prompt}]
            )
            future = client.sample(
                mi, num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=MAX_TOKENS, temperature=0.0, stop=stop,
                ),
            )
            resp = future.result(timeout=240)
            text = renderers.get_text_content(
                renderer.parse_response(resp.sequences[0].tokens)[0]
            )
            parsed = parse_scores(text)
            info = parsed.get("S3_positioning", {})
            score = info.get("score")
            reasoning = info.get("reasoning", "")

            out = {
                "sample_id": sid,
                "orig_s3": lrow["s3_score_orig"],
                "locus_score": lrow["score_locus"],
                "canonical_score": score,
                "canonical_reasoning_head": reasoning[:600],
                "canonical_raw_head": text[:2000],
            }
            fout.write(json.dumps(out) + "\n")
            fout.flush()
            logger.info(
                f"  {sid}: orig={lrow['s3_score_orig']} "
                f"locus={lrow['score_locus']} canonical={score}"
            )

    # Summary
    with OUT_PATH.open() as f:
        rows = [json.loads(l) for l in f]
    print("\n" + "=" * 70)
    print(f"{'sample_id':<18} {'orig':<6} {'locus':<8} {'canonical':<10} {'interpretation'}")
    print("-" * 70)
    for r in rows:
        o, lo, c = r["orig_s3"], r["locus_score"], r["canonical_score"]
        if c is None:
            interp = "canonical parse fail"
        elif c == lo and c >= 4:
            interp = "both high → 30B hallucinates intrinsically"
        elif c < lo and lo >= 4:
            interp = "locus directive induces hallucination"
        elif c >= 4 and lo >= 4:
            interp = "both high → orig score was corrupted (max_tokens=2048)"
        else:
            interp = "mixed"
        print(f"{r['sample_id']:<18} {o:<6} {lo:<8} {str(c):<10} {interp}")


if __name__ == "__main__":
    main()
