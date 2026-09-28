"""Exp A — Long-context retrieval probe (frozen Qwen3-30B-A3B + full oracle).

Falsifier for F16 H16-1: can Qwen3-30B reliably retrieve a specific formula
verbatim from the 10K-word oracle slim when explicitly instructed?

5 queries × 3 samples (temp=0.0):
- Q1-Q4: verbatim-quote requests targeting Math 5 (J_RS), Math 7 (GRPO loss),
  Methodology 9 (RS-GRPO advantage), Empirical 4 (AIME pass@1 trajectory)
- Q5: control — write J_RS WITHOUT consulting the oracle (still in context,
  instruction is to ignore it). Establishes prior knowledge baseline.

Decision rule (pre-registered, EXPERIMENT_PLAN_oracle_transfer_ABC.md L57-60):
- 0/4 Q1-Q4 verbatim-correct → H16-1 STRONGLY corroborated
- 1-2/4 → ambiguous (retrieval partial bottleneck)
- 3-4/4 → H16-1 FALSIFIED (bottleneck is downstream H16-2/H16-3)

Auto-scoring: anchor-string presence check per query (informative hint only).
Final verdict requires manual review of responses/*.json.

Usage:
    source shared/tools/use_api_profile.sh new
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.exp_A_oracle_retrieval
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from co_scientist.shared.api_profiles import create_service_client
from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v1 import _to_str

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)

REPO_ROOT = Path(__file__).resolve().parents[3]


# 5 queries — (label, query_text, target_oracle_item, anchor_strings_for_auto_check)
# Anchor strings: 2-4 highly distinctive substrings that should appear in a verbatim
# correct response. Lowercase + whitespace-collapsed comparison.
QUERIES = [
    (
        "Q1_math5_jrs",
        "Quote verbatim the formula labeled 'Math 5' from the oracle, including its "
        "source attribution. Output the formula in LaTeX-equivalent form, then the "
        "source line on a new line.",
        ("Math", 5),
        ["math 5", "exp(", "log e", "jiang"],
    ),
    (
        "Q2_math7_grpo",
        "Quote verbatim the formula labeled 'Math 7' from the oracle (GRPO loss with "
        "clipped policy ratio). Output the full RHS, then the source line on a new line.",
        ("Math", 7),
        ["math 7", "min(", "clip(", "grpo"],
    ),
    (
        "Q3_method9_rsgrpo",
        "Quote verbatim the formula labeled 'Methodology 9' from the oracle (RS-GRPO "
        "drop-in risk-sensitive advantage). Output the advantage formula and the source.",
        ("Methodology", 9),
        ["methodology 9", "rs-grpo", "advantage", "exp("],
    ),
    (
        "Q4_empirical4_aime",
        "Quote the AIME 2024 pass@1 trajectory numbers from Empirical 4 in the oracle "
        "(initial → final percentages and step count). Output the numbers and the source.",
        ("Empirical", 4),
        ["empirical 4", "15.6", "77.9", "aime"],
    ),
    (
        "Q5_control_no_oracle",
        "Without consulting the oracle, write the J_RS formula from risk-sensitive RL "
        "(the exponential-utility objective). Output only what you can recall from your "
        "general training, NOT from the oracle text above. State explicitly that you are "
        "not using the oracle.",
        None,  # control — no specific oracle target
        ["exp(", "log", "beta", "1/"],
    ),
]


@chz.chz
class Config:
    api_profile: str | None = "new"
    base_url: str | None = None

    model_name: str = "Qwen/Qwen3-30B-A3B"
    oracle_path: str = (
        "projects/d5_abstract_retrieve_refine/data/oracles/oracle_v2_2026_04_26_build/slim.md"
    )

    n_samples_per_query: int = 3
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0

    log_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_29_exp_A_oracle_retrieval"


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _build_prompt(oracle: str, query_text: str) -> str:
    return (
        "I will provide you with an oracle of methodological patterns extracted from "
        "high-quality research papers. Read it carefully, then answer the question precisely.\n\n"
        f"# Oracle\n\n{oracle}\n\n"
        f"# Question\n\n{query_text}\n\n"
        "Answer (be concise; quote verbatim where requested):"
    )


def _extract_oracle_item(oracle: str, type_str: str, num: int) -> str:
    """Extract body content for ### {type_str} {num}: ... up to next ### or EOF."""
    pattern = f"### {type_str} {num}:"
    start = oracle.find(pattern)
    if start == -1:
        return ""
    after = oracle[start:]
    next_match = re.search(r"\n### ", after[len(pattern):])
    if next_match is None:
        return after.strip()
    return after[: len(pattern) + next_match.start()].strip()


def _anchor_check(response: str, anchors: list[str]) -> tuple[int, list[bool]]:
    """Lowercase + whitespace-collapse the response, return (n_matched, per_anchor_bool)."""
    norm = re.sub(r"\s+", " ", response.lower())
    hits = [a.lower() in norm for a in anchors]
    return sum(hits), hits


def main(config: Config) -> None:
    out_dir = _resolve(config.log_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    responses_dir = out_dir / "responses"
    responses_dir.mkdir(exist_ok=True)

    oracle = _resolve(config.oracle_path).read_text().strip()
    logger.info("Loaded oracle: %d chars", len(oracle))

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )
    sampling_client = service_client.create_sampling_client(base_model=config.model_name)
    logger.info("Frozen sampling client ready @ base_model=%s", config.model_name)

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=renderer.get_stop_sequences(),
    )

    def _decode(seq) -> str:
        parsed = renderer.parse_response(seq.tokens)
        content = parsed[0].get("content", "") if parsed else ""
        text = _to_str(content) if content else tokenizer.decode(seq.tokens)
        return text

    all_responses = []
    for label, query_text, target_loc, anchors in QUERIES:
        target_body = (
            _extract_oracle_item(oracle, target_loc[0], target_loc[1])
            if target_loc is not None
            else ""
        )
        prompt_text = _build_prompt(oracle, query_text)
        prompt_input = renderer.build_generation_prompt(
            [{"role": "user", "content": prompt_text}]
        )
        n_tokens = len(prompt_input.to_ints())
        logger.info("[%s] prompt=%d tokens, anchors=%s", label, n_tokens, anchors)
        fut = sampling_client.sample(
            prompt=prompt_input,
            num_samples=config.n_samples_per_query,
            sampling_params=sampling_params,
        )
        result = fut.result()
        for k, seq in enumerate(result.sequences):
            text = _decode(seq)
            n_hits, hits = _anchor_check(text, anchors)
            rec = {
                "query_label": label,
                "query_text": query_text,
                "target_loc": list(target_loc) if target_loc else None,
                "target_body": target_body,
                "anchor_strings": anchors,
                "anchor_n_matched": n_hits,
                "anchor_per_string_hit": hits,
                "sample_idx": k,
                "model": config.model_name,
                "temperature": config.temperature,
                "stop_reason": str(seq.stop_reason),
                "prompt_n_tokens": n_tokens,
                "response_text": text,
                "response_n_chars": len(text),
            }
            out_path = responses_dir / f"{label}_sample_{k}.json"
            out_path.write_text(json.dumps(rec, indent=2))
            all_responses.append(rec)
            logger.info(
                "  sample %d: %d chars, anchor hits %d/%d",
                k,
                len(text),
                n_hits,
                len(anchors),
            )

    # ----- Auto-verdict (anchor presence; manual review still required for verbatim) -----
    auto_verdict = {}
    for label, _, target_loc, anchors in QUERIES:
        samples = [r for r in all_responses if r["query_label"] == label]
        any_full_match = any(s["anchor_n_matched"] == len(anchors) for s in samples)
        max_hits = max(s["anchor_n_matched"] for s in samples) if samples else 0
        auto_verdict[label] = {
            "n_samples": len(samples),
            "anchor_total": len(anchors),
            "any_sample_all_anchors": any_full_match,
            "max_anchor_hits": max_hits,
            "per_sample_hits": [s["anchor_n_matched"] for s in samples],
        }

    # ----- Verdict markdown -----
    verdict_md = ["# Exp A — Oracle retrieval probe verdicts\n"]
    verdict_md.append(
        f"**Setup**: frozen Qwen3-30B-A3B + full oracle slim ({len(oracle)} chars), "
        f"5 queries × {config.n_samples_per_query} samples, temp={config.temperature}.\n"
    )
    verdict_md.append(
        "**Auto-scoring caveat**: anchor-string presence is a *necessary* condition for "
        "verbatim-match, not sufficient. A response can hit all anchors and still get "
        "the formula wrong. Always cross-check with target_body field of each response JSON.\n"
    )

    verdict_md.append("\n## Per-query auto-anchor check\n")
    verdict_md.append("| Query | Target | Anchors | Best sample anchor hits | All anchors any sample? |")
    verdict_md.append("|---|---|---:|---:|:---:|")
    for label, _, target_loc, anchors in QUERIES:
        v = auto_verdict[label]
        target_label = (
            f"{target_loc[0]} {target_loc[1]}" if target_loc is not None else "(control)"
        )
        verdict_md.append(
            f"| {label} | {target_label} | {len(anchors)} | "
            f"{v['max_anchor_hits']}/{len(anchors)} | "
            f"{'YES' if v['any_sample_all_anchors'] else 'no'} |"
        )

    # H16-1 first-pass verdict from auto-scoring (Q1-Q4 only)
    q1q4 = [auto_verdict[lbl] for lbl, _, tloc, _ in QUERIES if tloc is not None]
    n_q_with_full_anchor = sum(1 for v in q1q4 if v["any_sample_all_anchors"])
    if n_q_with_full_anchor == 0:
        first_pass = "H16-1 PRELIM-CORROBORATED (0/4 hit all anchors in any sample)"
    elif n_q_with_full_anchor >= 3:
        first_pass = f"H16-1 PRELIM-FALSIFIED ({n_q_with_full_anchor}/4 hit all anchors)"
    else:
        first_pass = f"AMBIGUOUS ({n_q_with_full_anchor}/4 hit all anchors) — manual review decisive"
    verdict_md.append(f"\n**Auto first-pass verdict (Q1-Q4)**: {first_pass}\n")
    verdict_md.append(
        "(Anchor presence is a weak proxy. Verify by opening each `responses/Q*_sample_*.json` "
        "and comparing `response_text` to `target_body` for character-level match.)\n"
    )

    verdict_md.append("\n## Pre-registered decision rule (apply after manual verbatim review)")
    verdict_md.append("- **0/4 Q1-Q4 verbatim-correct** → H16-1 STRONGLY corroborated (retrieval bottleneck)")
    verdict_md.append("- **1-2/4** → ambiguous (retrieval is partial bottleneck)")
    verdict_md.append("- **3-4/4** → H16-1 FALSIFIED; bottleneck is downstream (H16-2 / H16-3)")
    verdict_md.append(
        "\n**Verbatim-correct** = formula RHS character-by-character match modulo whitespace and "
        "equivalent unicode/LaTeX rendering, AND correct source attribution (paper / ref number).\n"
    )

    verdict_md.append("\n## Q5 control note")
    verdict_md.append(
        "Q5 asks model to write J_RS *without* consulting the oracle (still in context, "
        "instruction-driven control). If Q1 (Math 5) succeeds but Q5 fails, the oracle is "
        "the source of formula recall; if both succeed, model already knows J_RS from "
        "general training and Q1 doesn't isolate retrieval. Compare Q1 vs Q5 manually.\n"
    )

    verdict_path = out_dir / "verdict.md"
    verdict_path.write_text("\n".join(verdict_md) + "\n")
    logger.info("Wrote verdict markdown: %s", verdict_path)

    # Save config
    config_path = out_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": config.model_name,
                "oracle_chars": len(oracle),
                "n_samples_per_query": config.n_samples_per_query,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "auto_verdict": auto_verdict,
            },
            indent=2,
        )
    )
    logger.info("Wrote %s", config_path)


if __name__ == "__main__":
    main(chz.entrypoint(Config))
