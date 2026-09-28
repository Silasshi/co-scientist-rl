"""D4-v5 signal generalization test across diverse grant proposals.

Grades proposals from multiple sources (EPSRC outlines, NIH R01/R21) with
all 10 D4-v5 signals using per-signal grading with N repeats + median.

Usage:
    source shared/tools/use_api_profile.sh new
    python projects/grant_proposal/analysis/generalization_test/grade_d4v5.py \
        --proposals-dir projects/grant_proposal/analysis/sanity_check/outlines \
        --goal "Develop a competitive EPSRC AI research hub proposal." \
        --corpus epsrc \
        --repeats 2

    python projects/grant_proposal/analysis/generalization_test/grade_d4v5.py \
        --proposals-dir projects/grant_proposal/data/proposals/nih_samples/_extracted_text \
        --goal "Develop a competitive NIH research proposal." \
        --corpus nih \
        --repeats 2 \
        --extract-research-section
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.grant_signal_reward import (
    SIGNALS,
    SIGNAL_WEIGHTS,
    build_single_signal_prompt,
    parse_scores,
    median_over_repeats,
    aggregate_reward,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)

GRADER_MODEL = "Qwen/Qwen3-30B-A3B"
MAX_PLAN_CHARS = 15000
MIN_TEXT_LEN = 500

_SECTION_RE = re.compile(
    r"(?:^|\n)\s*(?:Specific\s+Aims|SPECIFIC\s+AIMS)\s*\n",
    re.IGNORECASE,
)


def extract_research_section(text: str) -> str:
    match = _SECTION_RE.search(text)
    if match:
        return text[match.start() :]
    for fallback in ["\nResearch Strategy", "\nResearch Plan", "\nSignificance"]:
        pos = text.find(fallback)
        if pos != -1:
            return text[pos:]
    return text


def truncate(text: str) -> str:
    if len(text) > MAX_PLAN_CHARS:
        return text[:MAX_PLAN_CHARS] + "\n\n[... truncated for context limit ...]"
    return text


def load_proposals(
    proposals_dir: Path,
    do_extract: bool = False,
    extra_files: list[Path] | None = None,
) -> list[tuple[str, str]]:
    results = []
    all_files = sorted(proposals_dir.glob("*.txt")) + sorted(proposals_dir.glob("*.md"))
    if extra_files:
        all_files.extend(extra_files)
    for path in all_files:
        text = path.read_text()
        if len(text) < MIN_TEXT_LEN:
            logger.warning(f"Skipping {path.name}: only {len(text)} chars")
            continue
        if do_extract:
            text = extract_research_section(text)
        text = truncate(text)
        results.append((path.stem, text))
    return results


def grade_proposal(
    goal: str,
    plan: str,
    grader_client,
    renderer,
    tokenizer,
    n_repeats: int = 2,
) -> tuple[dict[str, int | None], list[dict[str, dict]]]:
    all_parsed = []
    for _ in range(n_repeats):
        repeat_results = {}
        for signal in SIGNALS:
            prompt = build_single_signal_prompt(goal, plan, signal, emit_critique=True)
            model_input = renderer.build_generation_prompt(
                [{"role": "user", "content": prompt}]
            )
            future = grader_client.sample(
                model_input,
                num_samples=1,
                sampling_params=tinker.types.SamplingParams(
                    max_tokens=4096,
                    temperature=0.3,
                    stop=renderer.get_stop_sequences(),
                ),
            )
            result = future.result(timeout=300)
            response_text = tokenizer.decode(result.sequences[0].tokens)
            parsed = parse_scores(response_text)
            if signal.id in parsed:
                repeat_results[signal.id] = parsed[signal.id]
            else:
                repeat_results[signal.id] = {"score": None, "reasoning": "", "critique": ""}
        all_parsed.append(repeat_results)

    medians = median_over_repeats(all_parsed)
    return medians, all_parsed


def main():
    parser = argparse.ArgumentParser(description="D4-v5 generalization test")
    parser.add_argument("--proposals-dir", required=True, type=Path)
    parser.add_argument("--goal", default="Develop a competitive research proposal.")
    parser.add_argument("--corpus", default="mixed", help="Label for this corpus (epsrc/nih/mixed)")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--extract-research-section", action="store_true")
    parser.add_argument("--extra-files", nargs="*", type=Path, default=[])
    args = parser.parse_args()

    proposals = load_proposals(args.proposals_dir, args.extract_research_section, args.extra_files)
    if not proposals:
        print(f"No proposals found in {args.proposals_dir}")
        return

    logger.info(f"Corpus: {args.corpus} | Proposals: {len(proposals)} | Repeats: {args.repeats}")

    service_client = create_service_client()
    grader_client = service_client.create_sampling_client(base_model=GRADER_MODEL)
    tokenizer = get_tokenizer(GRADER_MODEL)
    renderer_name = model_info.get_recommended_renderer_name(GRADER_MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / f"results_{args.corpus}.jsonl"

    signal_ids = [s.id for s in SIGNALS]
    all_results = []

    for name, text in proposals:
        logger.info(f"Grading: {name} ({len(text)} chars)...")
        t0 = time.time()
        medians, raw_parsed = grade_proposal(
            args.goal, text, grader_client, renderer, tokenizer, args.repeats
        )
        dt = time.time() - t0
        agg = aggregate_reward(medians)

        critiques = {}
        for sid in signal_ids:
            for repeat in raw_parsed:
                if sid in repeat and repeat[sid].get("critique"):
                    critiques[sid] = repeat[sid]["critique"]
                    break
            else:
                critiques[sid] = ""

        entry = {
            "name": name,
            "corpus": args.corpus,
            "word_count": len(text.split()),
            "scores": medians,
            "aggregate": round(agg, 4),
            "critiques": critiques,
            "time_s": round(dt, 1),
        }
        all_results.append(entry)

        score_str = " ".join(f"{sid.split('_')[0]}={medians[sid]}" for sid in signal_ids)
        logger.info(f"  {score_str}  agg={agg:.3f}  ({dt:.0f}s)")

    with open(out_path, "w") as f:
        for entry in all_results:
            f.write(json.dumps(entry) + "\n")
    logger.info(f"Results saved to {out_path}")

    # Summary table
    header_ids = [sid.split("_", 1)[0] for sid in signal_ids]
    header = f"{'Proposal':45s} | " + " | ".join(f"{h:>3s}" for h in header_ids) + " |  Agg"
    print(f"\n{'=' * len(header)}")
    print(f"D4-v5 Generalization: {args.corpus} ({len(all_results)} proposals)")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    for entry in all_results:
        scores = entry["scores"]
        vals = [scores.get(sid) for sid in signal_ids]
        row = f"{entry['name'][:45]:45s} | " + " | ".join(
            f"{(str(v) if v is not None else '?'):>3s}" for v in vals
        ) + f" | {entry['aggregate']:.2f}"
        print(row)

    print("-" * len(header))
    # Per-signal averages
    avgs = {}
    for sid in signal_ids:
        vals = [e["scores"].get(sid) for e in all_results if e["scores"].get(sid) is not None]
        avgs[sid] = sum(vals) / len(vals) if vals else 0
    avg_row = f"{'AVERAGE':45s} | " + " | ".join(
        f"{avgs[sid]:3.1f}" for sid in signal_ids
    ) + f" | {sum(e['aggregate'] for e in all_results) / len(all_results):.2f}"
    print(avg_row)

    agg_vals = [e["aggregate"] for e in all_results]
    print(f"\nAggregate range: {min(agg_vals):.3f} — {max(agg_vals):.3f}")
    print(f"Aggregate mean:  {sum(agg_vals)/len(agg_vals):.3f}")


if __name__ == "__main__":
    main()
