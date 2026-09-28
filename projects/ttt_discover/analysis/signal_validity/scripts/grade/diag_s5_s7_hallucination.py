#!/usr/bin/env python3
"""Phase 0 diagnostic branch: test whether the S3 hallucination finding
generalizes to S5_feasibility and S7_specificity.

S3 is a semantically-loaded signal (judges "named prior methods + insufficiency")
which is susceptible to grader priors. S5 and S7 are more lexically grounded:
  - S7_specificity counts vague markers vs specific commitments (lexical)
  - S5_feasibility judges core algorithm + dependencies + key params (plan-
    structural)

If S5/S7 reasoning stays grounded in actual plan content while S3 hallucinates,
Combo 1 is viable with per-signal gating (S3 excluded from locus mode).
If S5/S7 also hallucinate, the 30B grader cannot be trusted for revision
targeting on this goal at all.

Tests 3 plans × 2 signals = 6 API calls.
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
GOAL_PATH = PROJECT_ROOT / "projects/ttt_discover/analysis/sanity_check/research_goal.txt"
OUT_PATH = BASE / "data" / "locus_pilot" / "diag_remaining_signals.jsonl"

GRADER_MODEL = "Qwen/Qwen3-30B-A3B"
MAX_TOKENS = 8192
SIGNALS_TO_TEST = [
    "S1_depth",         # GLOBAL, semantic — could share S3's goal-prior issue
    "S2_rigor",         # LOCAL (Evaluation), structural counting
    "S6_risk_awareness", # GLOBAL, semantic
    "S8_scope",         # MULTI-LOCAL, counts lexical triggers
    "S9_focus",         # LOCAL (Core Idea), counts techniques
]


def check_reasoning_grounded(reasoning: str, plan_body: str) -> dict:
    """Heuristic: pull quoted strings from reasoning, check verbatim in plan.

    Returns dict with:
      - n_quotes: # of quoted strings (of >15 chars) in reasoning
      - n_grounded: # that appear verbatim in plan_body
      - sample_ungrounded: first ungrounded quote (if any)
    """
    # Extract quoted strings (prefer double-quoted)
    quotes = re.findall(r'"([^"]{15,400})"', reasoning)
    # Deduplicate
    quotes = list(dict.fromkeys(quotes))
    n_grounded = 0
    sample_ungrounded = None
    for q in quotes:
        # Whitespace-tolerant match
        q_norm = re.sub(r"\s+", " ", q).strip()
        plan_norm = re.sub(r"\s+", " ", plan_body).strip()
        if q_norm in plan_norm or q in plan_body:
            n_grounded += 1
        elif sample_ungrounded is None:
            sample_ungrounded = q[:200]
    return {
        "n_quotes": len(quotes),
        "n_grounded": n_grounded,
        "sample_ungrounded": sample_ungrounded,
    }


def main():
    goal = GOAL_PATH.read_text().strip()

    with SAMPLE_PATH.open() as f:
        samples = {s["sample_id"]: s for s in [json.loads(l) for l in f]}
    with LOCUS_GRADING_PATH.open() as f:
        locus_rows = [json.loads(l) for l in f]  # only the 3 already-tested plans

    svc = create_service_client()
    client = svc.create_sampling_client(base_model=GRADER_MODEL)
    tok = client.get_tokenizer()
    renderer_name = model_info.get_recommended_renderer_name(GRADER_MODEL)
    renderer = renderers.get_renderer(renderer_name, tok)
    stop = renderer.get_stop_sequences()

    signal_specs = {sig.id: sig for sig in SIGNALS if sig.id in SIGNALS_TO_TEST}

    with OUT_PATH.open("w") as fout:
        for lrow in locus_rows:
            sid = lrow["sample_id"]
            s = samples[sid]
            plan_body, _ = extract_plan_body(s["plan_text"])

            for sig_id in SIGNALS_TO_TEST:
                spec = signal_specs[sig_id]
                prompt = build_single_signal_prompt(goal, plan_body, spec)
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
                info = parsed.get(sig_id, {})
                score = info.get("score")
                reasoning = info.get("reasoning", "")

                grounded = check_reasoning_grounded(reasoning, plan_body)

                out = {
                    "sample_id": sid,
                    "signal_id": sig_id,
                    "score": score,
                    "reasoning": reasoning,
                    "grounded_check": grounded,
                    "raw_head": text[:2500],
                    "plan_body_preview": plan_body[:200],
                }
                fout.write(json.dumps(out) + "\n")
                fout.flush()
                logger.info(
                    f"  {sid} {sig_id}: score={score} "
                    f"quotes_grounded={grounded['n_grounded']}/{grounded['n_quotes']}"
                )

    # Summary
    with OUT_PATH.open() as f:
        rows = [json.loads(l) for l in f]
    print("\n" + "=" * 75)
    print(f"{'sample_id':<18} {'signal':<18} {'score':<6} {'grounded':<12} {'ungrounded_sample'}")
    print("-" * 75)
    for r in rows:
        g = r["grounded_check"]
        gstr = f"{g['n_grounded']}/{g['n_quotes']}"
        ung = (g['sample_ungrounded'] or "")[:30]
        print(f"{r['sample_id']:<18} {r['signal_id']:<18} {str(r['score']):<6} {gstr:<12} {ung!r}")


if __name__ == "__main__":
    main()
