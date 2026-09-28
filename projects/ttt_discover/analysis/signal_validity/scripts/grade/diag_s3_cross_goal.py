#!/usr/bin/env python3
"""Phase 0 Check 2: test if S3 hallucination is TTT-Discover-specific or
baked into the S3 rubric across any goal.

Takes 5 reference plans from 5 DIFFERENT topics (not just test_time_search).
Grades each with canonical S3 prompt. Checks:
  (a) Does grader name methods that actually appear in the plan?
  (b) Does the specific "AlphaEvolve/OpenEvolve/ThetaEvolve" hallucination
      pattern leak into other-goal grading?
  (c) If not, the TTT-Discover hallucination is goal-memorized (S3 broken
      only on that goal). If yes, S3 rubric is broken in general.

5 API calls.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]
REFS_PATH = BASE / "data" / "refs" / "references_v2.jsonl"
OUT_PATH = BASE / "data" / "locus_pilot" / "diag_s3_cross_goal.jsonl"

GRADER_MODEL = "Qwen/Qwen3-30B-A3B"
MAX_TOKENS = 8192
S3_SPEC = next(s for s in SIGNALS if s.id == "S3_positioning")

# Pick 5 refs from 5 different topics (1 from test_time_search but not TTT-Discover
# — to see if hallucination is TTT-specific or test_time_search-wide)
TARGET_PICKS = [
    ("02_harness_synthesis", None),       # any from this topic
    ("05_self_critique_no_ground_truth", None),
    ("07_rl_methods", None),
    ("09_efficient_training", None),
    ("01_test_time_search", "RISE"),      # same topic as TTT-Discover, different paper
]


def load_picks():
    with REFS_PATH.open() as f:
        refs = [json.loads(l) for l in f]
    picks = []
    for topic, source_id_filter in TARGET_PICKS:
        topic_refs = [r for r in refs if r["subdomain"] == topic]
        if source_id_filter:
            topic_refs = [r for r in topic_refs if source_id_filter.lower() in r["source_id"].lower()]
        if topic_refs:
            picks.append(topic_refs[0])
        else:
            logger.warning(f"No match for {topic} / {source_id_filter}")
    return picks


# Hallucination markers specific to TTT-Discover's grader pattern
TTT_HALLUC_MARKERS = [
    "AlphaEvolve", "OpenEvolve", "ThetaEvolve",
    "Prior work using evolutionary search",
    "cannot internalize new ideas",
]


def extract_named_methods(reasoning: str) -> list[str]:
    """Pull out the things the grader claims as "named prior methods"."""
    # Look for quoted proper nouns in STEP 1 section
    # Patterns: 'named methods: X, Y, Z', 'NAMED prior methods: X', or quoted strings
    methods = []
    # Heuristic: grader typically lists methods after "named:" or in quotes
    step1_match = re.search(r"STEP 1[^S]*?(?=STEP 2|$)", reasoning, re.DOTALL)
    if step1_match:
        step1 = step1_match.group(0)
        # Extract quoted proper nouns
        quoted = re.findall(r'"([A-Z][A-Za-z0-9\-]+)"', step1)
        methods.extend(quoted)
    return methods


def main():
    picks = load_picks()
    logger.info(f"Loaded {len(picks)} reference plans from different topics")

    svc = create_service_client()
    client = svc.create_sampling_client(base_model=GRADER_MODEL)
    tok = client.get_tokenizer()
    renderer_name = model_info.get_recommended_renderer_name(GRADER_MODEL)
    renderer = renderers.get_renderer(renderer_name, tok)
    stop = renderer.get_stop_sequences()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as fout:
        for ref in picks:
            goal = ref["goal"]
            plan = ref["reference_solution"]
            prompt = build_single_signal_prompt(goal, plan, S3_SPEC)
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

            # Diagnose hallucination
            ttt_leak = [m for m in TTT_HALLUC_MARKERS if m in reasoning]
            in_plan = [m for m in TTT_HALLUC_MARKERS if m in plan]
            ttt_leak_truly_halluc = list(set(ttt_leak) - set(in_plan))

            # Extract claimed named methods from reasoning
            claimed_methods = extract_named_methods(reasoning)
            # Check which are actually in plan
            methods_in_plan = [m for m in claimed_methods if m in plan]
            methods_hallucinated = [m for m in claimed_methods if m not in plan]

            out = {
                "source_id": ref["source_id"],
                "subdomain": ref["subdomain"],
                "score": score,
                "ttt_leak_truly_halluc": ttt_leak_truly_halluc,
                "claimed_methods": claimed_methods,
                "methods_in_plan": methods_in_plan,
                "methods_hallucinated": methods_hallucinated,
                "reasoning_head": reasoning[:1500],
                "plan_preview": plan[:400],
            }
            fout.write(json.dumps(out) + "\n")
            fout.flush()
            logger.info(
                f"  [{ref['subdomain']}/{ref['source_id']}] score={score} "
                f"ttt_leak={ttt_leak_truly_halluc or 'none'} "
                f"hallucinated_methods={methods_hallucinated}"
            )

    # Summary
    with OUT_PATH.open() as f:
        rows = [json.loads(l) for l in f]
    print("\n" + "=" * 85)
    print(f"{'topic':<32} {'source_id':<20} {'score':<6} {'TTT-leak?':<10} {'halluc_methods'}")
    print("-" * 85)
    for r in rows:
        print(
            f"{r['subdomain']:<32} {r['source_id']:<20} {str(r['score']):<6} "
            f"{'YES' if r['ttt_leak_truly_halluc'] else 'no':<10} "
            f"{r['methods_hallucinated']}"
        )


if __name__ == "__main__":
    main()
