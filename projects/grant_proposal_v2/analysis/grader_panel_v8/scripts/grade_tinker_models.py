"""Grade 15 test plans with 8 tinker models using v8 rubric + CoT scaffolding.

Each (model, plan, signal) emits one jsonl line with: plan_id, signal_id,
model, score, reasoning, raw_output, parse_success.

Usage:
    python grade_tinker_models.py [--model MODEL_KEY] [--limit N]
    # Without --model, grades all models. --model can be specified multiple times.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJ / "src"))

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.grant_rubric_v8 import SIGNALS as V8_SIGNALS
from co_scientist.shared.grant_signal_reward import (
    build_single_signal_prompt,
    parse_scores,
)

PANEL_DIR = Path(__file__).resolve().parent.parent
PLANS_DIR = PANEL_DIR / "test_plans"
GRADES_DIR = PANEL_DIR / "grades"
GRADES_DIR.mkdir(parents=True, exist_ok=True)

GOAL_TEXT = (
    PROJ / "projects/grant_proposal/dataset/goals/01_foundopt/research_goal.md"
).read_text().strip()

MODELS = [
    # (key, tinker_id, context_notes)
    ("qwen3_4b", "Qwen/Qwen3-4B-Instruct-2507"),
    ("qwen3_30b", "Qwen/Qwen3-30B-A3B"),
    ("qwen3_235b", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
    ("llama_3_1_8b", "meta-llama/Llama-3.1-8B-Instruct"),
    ("gpt_oss_20b", "openai/gpt-oss-20b"),
    ("gpt_oss_120b", "openai/gpt-oss-120b"),
    ("deepseek_v3_1", "deepseek-ai/DeepSeek-V3.1"),
    ("kimi_k2_thinking", "moonshotai/Kimi-K2-Thinking"),
]

MAX_TOKENS = 4096  # enough headroom for reasoning models
MAX_RETRIES = 2


def load_plans() -> list[tuple[str, str]]:
    """Return list of (plan_id, plan_text)."""
    sources = [
        json.loads(l) for l in open(PLANS_DIR / "sources.jsonl")
    ]
    plans = []
    for s in sources:
        pid = s["plan_id"]
        text = (PLANS_DIR / f"{pid}.txt").read_text()
        plans.append((pid, text))
    return plans


def strip_thinking(raw: str) -> str:
    """Strip reasoning/thinking blocks from model output."""
    for tag in ("</think>", "</reasoning>", "</Thought>"):
        m = re.search(re.escape(tag) + r"\s*(.*)", raw, re.DOTALL)
        if m:
            return m.group(1).strip()
    return raw


def extract_raw(parsed_resp, tokenizer, tokens) -> str:
    """Extract plain text from renderer.parse_response output.

    Different renderers return different structures. Fall back to decoding
    the tokens directly if the structured parse fails.
    """
    try:
        if isinstance(parsed_resp, tuple):
            parsed_resp = parsed_resp[0]
        if isinstance(parsed_resp, list) and parsed_resp:
            parts = []
            for p in parsed_resp:
                if isinstance(p, dict):
                    parts.append(str(p.get("content", p.get("thinking", ""))))
                else:
                    parts.append(str(p))
            out = "\n".join(x for x in parts if x)
            if out:
                return out
        if isinstance(parsed_resp, dict):
            return str(parsed_resp.get("content", parsed_resp))
    except Exception:
        pass
    # Fallback: decode tokens directly
    try:
        return tokenizer.decode(tokens)
    except Exception:
        return ""


def parse_one_signal(raw: str, signal_id: str, score_max: int) -> tuple[int | None, str]:
    """Best-effort parse of a single signal's score + reasoning/critique.

    Returns (score, reasoning_or_critique). Score is None if unparseable.
    """
    # Try full parse_scores first (correct XML structure)
    try:
        parsed = parse_scores(raw)
        if signal_id in parsed:
            info = parsed[signal_id]
            s = info.get("score")
            if s is not None and 1 <= s <= score_max:
                reasoning = info.get("reasoning", "") or info.get("critique", "")
                return s, reasoning[:500]
    except Exception:
        pass
    # Fallback: find the last <score>N</score>
    matches = re.findall(r"<score>\s*(\d+)\s*</score>", raw)
    if matches:
        try:
            s = int(matches[-1])
            if 1 <= s <= score_max:
                return s, "(fallback regex parse)"
        except Exception:
            pass
    return None, ""


def grade_model(
    model_key: str,
    model_id: str,
    plans: list[tuple[str, str]],
    service_client: tinker.ServiceClient,
    limit: int | None = None,
) -> None:
    """Grade all plans × all v8 signals with one model. Append to jsonl."""
    out_path = GRADES_DIR / f"{model_key}.jsonl"

    try:
        rn = model_info.get_recommended_renderer_name(model_id)
    except ValueError:
        print(f"[{model_key}] ERROR: unknown model {model_id}")
        return
    tokenizer = get_tokenizer(model_id)
    renderer = renderers.get_renderer(rn, tokenizer)
    sampling_client = service_client.create_sampling_client(base_model=model_id)
    sp = tinker.SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    # Skip already-graded (plan_id, signal_id) pairs for resumability
    done: set[tuple[str, str]] = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                r = json.loads(line)
                done.add((r["plan_id"], r["signal_id"]))
            except Exception:
                pass

    total = len(plans) * len(V8_SIGNALS)
    if limit:
        total = min(total, limit)
    processed = 0
    t0 = time.time()

    with open(out_path, "a") as f:
        for pid, plan_text in plans:
            for signal in V8_SIGNALS:
                if (pid, signal.id) in done:
                    continue
                if limit and processed >= limit:
                    return

                prompt = build_single_signal_prompt(
                    goal=GOAL_TEXT,
                    plan=plan_text,
                    signal=signal,
                    emit_locus=False,
                    emit_critique=True,
                    include_cot=True,
                )
                mi = renderer.build_generation_prompt(
                    [{"role": "user", "content": prompt}]
                )

                score = None
                reasoning = ""
                raw = ""
                parse_success = False
                last_err = None

                for attempt in range(MAX_RETRIES):
                    try:
                        result = sampling_client.sample(
                            prompt=mi, num_samples=1, sampling_params=sp
                        ).result()
                        parsed_resp = renderer.parse_response(
                            result.sequences[0].tokens
                        )
                        raw = strip_thinking(
                            extract_raw(
                                parsed_resp, tokenizer, result.sequences[0].tokens
                            )
                        )
                        score, reasoning = parse_one_signal(
                            raw, signal.id, signal.score_max
                        )
                        parse_success = score is not None
                        break
                    except Exception as e:
                        last_err = str(e)[:100]
                        time.sleep(1)

                record = {
                    "plan_id": pid,
                    "signal_id": signal.id,
                    "model": model_key,
                    "model_id": model_id,
                    "score": score,
                    "reasoning": reasoning,
                    "raw_preview": raw[:400],
                    "parse_success": parse_success,
                    "error": last_err,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

                processed += 1
                elapsed = time.time() - t0
                if processed % 5 == 0 or not parse_success:
                    status = "✓" if parse_success else "✗"
                    print(
                        f"[{model_key}] {status} {pid} {signal.id:<30s} "
                        f"score={score} ({processed}/{total}, {elapsed:.0f}s)"
                    )

    print(f"[{model_key}] DONE: {processed} calls in {time.time()-t0:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", action="append", default=None,
        help="Model key(s) to grade (default: all). Can be specified multiple times.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max calls per model (for debugging)",
    )
    args = parser.parse_args()

    plans = load_plans()
    print(f"Loaded {len(plans)} plans")
    print(f"Grading with {len(V8_SIGNALS)} v8 signals")
    print()

    service_client = create_service_client(api_profile="new")

    selected = args.model or [k for k, _ in MODELS]
    model_map = dict(MODELS)
    for key in selected:
        if key not in model_map:
            print(f"Skipping unknown model key: {key} (valid: {list(model_map)})")
            continue
        print(f"=== {key} ({model_map[key]}) ===")
        grade_model(
            key, model_map[key], plans, service_client, limit=args.limit,
        )
        print()


if __name__ == "__main__":
    main()
