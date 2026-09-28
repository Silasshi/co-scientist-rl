#!/usr/bin/env python3
"""Phase 0 — locus accuracy pilot for S3_positioning.

Grades 28 sampled plans (from sample_s3_pilot.jsonl) with a locus-extended
S3 rubric: canonical S3 grading PLUS a request for `<locus>` JSON identifying
where in the plan the weakness appears.

Design decisions (see /home/silas/.claude/plans/compressed-swimming-zephyr.md):
- N=1 repeat per plan (we're testing locus validity, not score stability)
- Temperature=0.0 for determinism
- max_tokens=6144 (vs current prod 4096 — locus directive adds ~200-500 tokens)
- Grader: Qwen3-30B-A3B (same as prod policy/grader)
- Do NOT modify ten_signal_reward.py yet (Phase 1 work). This script builds
  the locus-extended prompt inline.
- Extract plan body (strip <think>, unwrap <solution>) before grading so the
  locus's verbatim-substring check is meaningful.

Output: `analysis/signal_validity/data/locus_pilot/grading_locus_s3.jsonl`

Auto-validation metrics logged per entry:
  - parse_status: can we parse <evaluation> + <locus>?
  - verbatim_fail_count: how many quotes are not substrings of plan_body?
  - n_loci_valid / n_loci_requested
"""
import json
import logging
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import tinker
import tinker.types as types
from tinker_cookbook import model_info, renderers
from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.ten_signal_reward import SIGNALS, _format_signal_block, SHARED_PREAMBLE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]
SAMPLE_PATH = BASE / "data" / "locus_pilot" / "sample_s3_pilot.jsonl"
OUT_PATH = BASE / "data" / "locus_pilot" / "grading_locus_s3.jsonl"
GOAL_PATH = PROJECT_ROOT / "projects/ttt_discover/analysis/sanity_check/research_goal.txt"

GRADER_MODEL = "Qwen/Qwen3-30B-A3B"
SIGNAL_ID = "S3_positioning"
MAX_TOKENS = 8192  # Qwen3-30B reasoning can be long; give headroom for <think>+<evaluation>+<locus>
WAVE_SIZE = 4  # plans per wave — keeps futures manageable


S3_SPEC = next(s for s in SIGNALS if s.id == SIGNAL_ID)


# =============================================================================
# Plan body extraction
# =============================================================================

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.DOTALL | re.IGNORECASE)


def extract_plan_body(raw_text: str) -> tuple[str, dict]:
    """Strip <think>...</think> and unwrap <solution>...</solution>.

    Returns (cleaned_body, trace) where trace is a dict describing what
    was stripped (for debugging locus mismatches).
    """
    trace = {
        "raw_len": len(raw_text),
        "had_think": False,
        "had_solution_wrap": False,
        "think_chars_stripped": 0,
    }
    body = raw_text

    # Strip <think>...</think> blocks (may have multiple; be permissive)
    if "<think>" in body.lower():
        trace["had_think"] = True
        orig_len = len(body)
        body = THINK_RE.sub("", body).strip()
        trace["think_chars_stripped"] = orig_len - len(body)

    # Unwrap <solution>...</solution> if present
    sol_match = SOLUTION_RE.search(body)
    if sol_match:
        trace["had_solution_wrap"] = True
        body = sol_match.group(1).strip()

    trace["final_len"] = len(body)
    return body, trace


# =============================================================================
# Locus-extended S3 prompt
# =============================================================================

LOCUS_DIRECTIVE = """

---

# Additional Output: Locus Attribution

After the <evaluation> block, output a separate <locus> block containing a
JSON array with up to 3 entries identifying WHERE in the Research Plan the
Positioning weakness appears. Each entry has exactly these three keys:

  - "section_hint": name of the plan section the span belongs to
     (e.g., "Problem", "Motivation", "Related Work", "Core Idea",
     "Methodology", "Evaluation", "Risks"). Use "" if unclear.
  - "quote": a VERBATIM substring copied from the Research Plan text above
     (≤ 250 characters).
  - "why": 1-2 sentences explaining why this span hurts the Positioning
     signal per the rubric above. Cite the specific rubric language.

**VERBATIM RULE (STRICT)**: The "quote" MUST be a literal copy-paste from the
"# Research Plan" text above. DO NOT paraphrase, DO NOT merge non-adjacent
sentences, DO NOT fix typos, DO NOT insert words. If you cannot find a clean
contiguous substring that captures the weakness, use the single BEST
available substring and describe the missing content in "why". Before
emitting, mentally Ctrl-F the quote against the plan text — if not an exact
substring, rewrite it so it is.

Choose spans where the weakness is CONCENTRATED — the 1-3 places a reviser
would most naturally edit. If the plan has NO Positioning weakness (score 5),
output an empty JSON array.

If the weakness is ABSENCE of content (e.g., "no named prior methods anywhere"),
quote the SHORTEST sentence/phrase where a reviser would most naturally ADD
the missing content — but still only VERBATIM from the plan.

**CONCISENESS**: Keep your <reasoning> concise (≤ 400 words). The locus and
score are the critical outputs. Avoid long exposition.

Format (output exactly this structure after the evaluation block):
<locus>
[
  {"section_hint": "Motivation", "quote": "...", "why": "..."}
]
</locus>
"""


def build_locus_prompt(goal: str, plan_body: str) -> str:
    """Canonical S3 prompt + locus directive appended."""
    # Truncate like prod
    if len(plan_body) > 15000:
        plan_body = plan_body[:15000] + "\n\n[... truncated for context limit ...]"

    signal_block = _format_signal_block(S3_SPEC)

    return f"""{SHARED_PREAMBLE}

You will evaluate the following research plan on ONE dimension: **{S3_SPEC.name}**.

# Research Goal
{goal}

# Research Plan
{plan_body}

# Evaluation Dimension

{signal_block}

---

# Output Format

<evaluation>
    <dim id="{S3_SPEC.id}">
        <reasoning>Your analysis (do required reasoning steps if any).</reasoning>
        <score>INTEGER 1-5</score>
    </dim>
</evaluation>
{LOCUS_DIRECTIVE}

Begin your evaluation now."""


# =============================================================================
# Response parsing + validation
# =============================================================================

LOCUS_BLOCK_RE = re.compile(r"<locus>\s*(.*?)\s*</locus>", re.DOTALL | re.IGNORECASE)
EVAL_DIM_RE = re.compile(
    r'<dim\s+id="([^"]+)">(.*?)</dim>', re.DOTALL
)
SCORE_RE = re.compile(r"<score>\s*(\d+)\s*</score>", re.DOTALL)
REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)


def parse_grader_response(text: str, plan_body: str) -> dict:
    """Parse score + locus; validate each quote is verbatim in plan_body."""
    out = {
        "parse_status": "ok",
        "score": None,
        "reasoning": None,
        "locus_raw": None,
        "locus_validated": [],
        "verbatim_fail_count": 0,
        "n_loci_parsed": 0,
        "n_loci_valid": 0,
    }

    # --- Score + reasoning ---
    dim_match = EVAL_DIM_RE.search(text)
    if not dim_match:
        out["parse_status"] = "no_evaluation_block"
        return out
    dim_body = dim_match.group(2)
    score_match = SCORE_RE.search(dim_body)
    reasoning_match = REASONING_RE.search(dim_body)
    if score_match:
        s = int(score_match.group(1))
        out["score"] = s if 1 <= s <= 5 else None
    if reasoning_match:
        out["reasoning"] = reasoning_match.group(1).strip()

    # --- Locus JSON ---
    locus_match = LOCUS_BLOCK_RE.search(text)
    if not locus_match:
        out["parse_status"] = "no_locus_block"
        return out

    locus_str = locus_match.group(1).strip()
    try:
        locus_list = json.loads(locus_str)
    except json.JSONDecodeError as e:
        out["parse_status"] = f"malformed_locus_json: {e}"
        out["locus_raw"] = locus_str[:500]
        return out

    if not isinstance(locus_list, list):
        out["parse_status"] = "locus_not_list"
        out["locus_raw"] = locus_str[:500]
        return out

    out["locus_raw"] = locus_list
    out["n_loci_parsed"] = len(locus_list)

    # --- Verbatim check ---
    for entry in locus_list:
        if not isinstance(entry, dict):
            out["verbatim_fail_count"] += 1
            continue
        quote = entry.get("quote", "")
        if not isinstance(quote, str) or not quote.strip():
            out["verbatim_fail_count"] += 1
            continue
        # Normalize whitespace for match — tolerant of grader collapsing whitespace
        quote_norm = re.sub(r"\s+", " ", quote).strip()
        plan_norm = re.sub(r"\s+", " ", plan_body).strip()
        # Strip trailing ellipsis variants (grader often appends "..." when truncating)
        quote_stripped = re.sub(r"[.\u2026]+\s*$", "", quote_norm).strip()
        if (quote_norm in plan_norm
            or quote in plan_body
            or (len(quote_stripped) >= 40 and quote_stripped in plan_norm)):
            out["locus_validated"].append(entry)
        else:
            out["verbatim_fail_count"] += 1

    out["n_loci_valid"] = len(out["locus_validated"])
    return out


# =============================================================================
# Main driver
# =============================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit to first N pending samples (for dry runs).")
    args = ap.parse_args()

    goal = GOAL_PATH.read_text().strip()
    logger.info(f"Loaded target goal ({len(goal)} chars)")

    with SAMPLE_PATH.open() as f:
        samples = [json.loads(l) for l in f]
    logger.info(f"Loaded {len(samples)} pilot plans from {SAMPLE_PATH.name}")

    # Resume: skip already-graded samples
    done = set()
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            done = {json.loads(l)["sample_id"] for l in f}
        logger.info(f"Resuming — {len(done)} already graded")
    pending = [s for s in samples if s["sample_id"] not in done]
    if args.limit is not None:
        pending = pending[:args.limit]
        logger.info(f"LIMIT applied — pending: {len(pending)} (dry run)")
    else:
        logger.info(f"Pending: {len(pending)}")
    if not pending:
        logger.info("Nothing to do.")
        return

    svc = create_service_client()
    client = svc.create_sampling_client(base_model=GRADER_MODEL)
    tok = client.get_tokenizer()
    renderer_name = model_info.get_recommended_renderer_name(GRADER_MODEL)
    renderer = renderers.get_renderer(renderer_name, tok)
    stop = renderer.get_stop_sequences()

    t_start = time.time()
    with OUT_PATH.open("a") as fout:
        for wave_start in range(0, len(pending), WAVE_SIZE):
            wave = pending[wave_start:wave_start + WAVE_SIZE]
            wave_state = []
            for s in wave:
                plan_body, extract_trace = extract_plan_body(s["plan_text"])
                prompt = build_locus_prompt(goal, plan_body)
                mi = renderer.build_generation_prompt(
                    [{"role": "user", "content": prompt}]
                )
                future = client.sample(
                    mi, num_samples=1,
                    sampling_params=types.SamplingParams(
                        max_tokens=MAX_TOKENS,
                        temperature=0.0,
                        stop=stop,
                    ),
                )
                wave_state.append((s, plan_body, extract_trace, future))

            for s, plan_body, extract_trace, future in wave_state:
                try:
                    resp = future.result(timeout=240)
                    raw_text = renderers.get_text_content(
                        renderer.parse_response(resp.sequences[0].tokens)[0]
                    )
                except Exception as e:
                    logger.warning(f"  {s['sample_id']}: sampling failed — {e}")
                    raw_text = ""

                parsed = parse_grader_response(raw_text, plan_body)

                out = {
                    "sample_id": s["sample_id"],
                    "source_run": s["source_run"],
                    "s3_score_orig": s["s3_score"],
                    "score_locus": parsed["score"],
                    "reasoning": parsed["reasoning"],
                    "locus_raw": parsed["locus_raw"],
                    "locus_validated": parsed["locus_validated"],
                    "parse_status": parsed["parse_status"],
                    "n_loci_parsed": parsed["n_loci_parsed"],
                    "n_loci_valid": parsed["n_loci_valid"],
                    "verbatim_fail_count": parsed["verbatim_fail_count"],
                    "extract_trace": extract_trace,
                    "plan_body_preview": plan_body[:200],
                    "raw_response_len": len(raw_text),
                    "raw_response_head": raw_text[:3000],
                    "raw_response_tail": raw_text[-1500:] if len(raw_text) > 3000 else "",
                }
                fout.write(json.dumps(out) + "\n")
                fout.flush()

                n_done = len(done) + wave_start + 1 + wave.index(s)
                elapsed = time.time() - t_start
                logger.info(
                    f"  [{n_done}/{len(samples)}] {s['sample_id']} "
                    f"(orig_s3={s['s3_score']}): "
                    f"score={parsed['score']} loci={parsed['n_loci_valid']}/{parsed['n_loci_parsed']} "
                    f"status={parsed['parse_status']} "
                    f"elapsed={elapsed:.0f}s"
                )

    logger.info(f"Done. Output: {OUT_PATH}")

    # Summary
    with OUT_PATH.open() as f:
        rows = [json.loads(l) for l in f]
    n = len(rows)
    n_parse_ok = sum(1 for r in rows if r["parse_status"] == "ok")
    n_verbatim_fail_any = sum(1 for r in rows if r["verbatim_fail_count"] > 0)
    n_loci_total = sum(r["n_loci_parsed"] for r in rows)
    n_loci_valid = sum(r["n_loci_valid"] for r in rows)
    logger.info("=" * 60)
    logger.info(f"Summary ({n} plans):")
    logger.info(f"  Parse OK (evaluation + locus both parseable): {n_parse_ok}/{n} ({100*n_parse_ok/n:.0f}%)")
    logger.info(f"  At least one verbatim fail: {n_verbatim_fail_any}/{n}")
    logger.info(f"  Total loci emitted: {n_loci_total}, verbatim-valid: {n_loci_valid} ({100*n_loci_valid/max(1,n_loci_total):.0f}%)")


if __name__ == "__main__":
    main()
