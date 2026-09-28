"""Build slim oracle variants from full oracle_v2 + Round-0 relevance scores.

Phase 2 fix for context-window: oracle_v2.md is ~115K tokens but Qwen3-30B-A3B
base context is 40K. Slim variants filter the 580-item full oracle down to fit
in-prompt comfortably:

  oracle_v2_slim.md   — score=3 papers only,    ≤1 item/category/paper  (~20K tokens)
  oracle_v2_medium.md — score≥2 papers,         ≤1 item/category/paper  (~30K tokens)

Both reuse Round-3 cumulative output, so each item already reflects R0-R3
cross-paper refinement and Opus's preferred deduplication. We just clip per
category × paper.

Item-pick policy when paper has >1 item in a category:
  - prefer items whose `action` is "refine" or "new" (Opus's later-pass preferred form)
  - else first item from that paper for that category

Outputs:
  data/oracles/oracle_v2_2026_04_26_build/slim.md
  data/oracles/oracle_v2_2026_04_26_build/medium.md

Usage:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.build_slim_oracle_v2
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import chz

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]

CATEGORIES = ["Insights", "Methodology", "Theory", "Math", "Empirical", "Failure_modes"]


@chz.chz
class Config:
    """Paths are relative to data_dir unless absolute. data_dir is relative to REPO_ROOT unless absolute."""
    data_dir: str = "projects/d5_abstract_retrieve_refine/data"
    round0_path: str = "oracle_v2_build/round_0_summary.json"
    round3_path: str = "oracle_v2_build/round_3_summary.json"
    bib_path: str = "ttt_discover_bibliography_v2.jsonl"
    out_slim: str = "ttt_discover_oracle_v2_slim.md"
    out_medium: str = "ttt_discover_oracle_v2_medium.md"
    full_oracle_path: str = "ttt_discover_oracle_v2.md"
    label: str = "TTT-Discover Goal"


def _resolve(base: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else base / p


def _short_authors(authors: list[str]) -> str:
    if not authors:
        return "?"
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    return f"{authors[0]} et al."


def _full_cite(b: dict) -> str:
    a = _short_authors(b.get("authors", []))
    yr = b.get("year") or "?"
    title = b.get("title", "")
    arxiv = b.get("arxiv_id")
    cite = f'{a} {yr}. "{title}"'
    if arxiv:
        cite += f" arxiv:{arxiv}"
    return cite


def _short_cite(b: dict) -> str:
    a = _short_authors(b.get("authors", []))
    yr = b.get("year") or "?"
    return f"({a} {yr})"


def _pick_item(items: list[dict]) -> dict | None:
    """Pick the single best item for one (paper, category) pair."""
    if not items:
        return None
    # Prefer Round-3 'refine' or 'new', then anything
    for action in ("refine", "new", "keep"):
        for it in items:
            if it.get("action") == action:
                return it
    return items[0]


def _render_item(num: int, cat: str, item: dict, ref_num: int, b: dict, first_seen: set) -> str:
    """Render one ### item block in oracle markdown form."""
    summary = item.get("summary", "").strip() or "(no summary)"
    content = item.get("content", "").strip() or "(no content)"
    applic = item.get("applicability", "").strip() or "(no applicability)"

    cite = _full_cite(b) if ref_num not in first_seen else _short_cite(b)
    first_seen.add(ref_num)

    return (
        f"### {cat} {num}: {summary}\n"
        f"- **Source**: {cite} (Ref [{ref_num}])\n"
        f"- **Content**: {content}\n"
        f"- **Applicability to TTT-Discover goal**: {applic}\n"
    )


def build_oracle(min_score: int, out_path: Path, label: str,
                 round0_path: Path, round3_path: Path, bib_path: Path,
                 oracle_label: str = "TTT-Discover Goal") -> int:
    """Filter Round-3 items to entries from papers with score ≥ min_score, ≤1 per
    (paper, category). Return total chars.
    """
    relevance = json.loads(round0_path.read_text())  # {ref_str: {relevance_score, ...}}
    extractions = json.loads(round3_path.read_text())  # {ref_str: {category: [items]}}

    # Load bib + sort like build_oracle_v2 does
    bib = []
    with open(bib_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("is_paper", True):
                bib.append(d)
    bib.sort(key=lambda r: r.get("cite_key", ""))
    ref_to_bib = {i + 1: b for i, b in enumerate(bib)}

    # Filter ref_nums by relevance score
    eligible = {
        int(rn): info for rn, info in relevance.items()
        if info.get("relevance_score", 0) >= min_score
    }
    logger.info("[%s] eligible papers (score >= %d): %d", label, min_score, len(eligible))

    # Build category -> [(item, ref_num, paper)] preserving ref_num order
    cat_items: dict[str, list[tuple[dict, int, dict]]] = {c: [] for c in CATEGORIES}
    n_paper_contributing = 0
    for rn in sorted(eligible):
        ext = extractions.get(str(rn))
        if not ext:
            continue
        b = ref_to_bib.get(rn, {})
        contributed = False
        for cat in CATEGORIES:
            items = ext.get(cat) or []
            picked = _pick_item(items)
            if picked is not None:
                cat_items[cat].append((picked, rn, b))
                contributed = True
        if contributed:
            n_paper_contributing += 1

    # Render
    header = (
        f"# D5 Oracle Abstraction v2 — {oracle_label} ({label})\n\n"
        f"*Slim variant filtered from full oracle (≤1 item/category/paper, score ≥ {min_score}).*\n\n"
        f"## Metadata\n"
        f"- Filter rule: relevance_score ≥ {min_score}, ≤1 item / (paper, category)\n"
        f"- Papers contributing: {n_paper_contributing} / {len(eligible)} eligible\n"
        f"- Total items: {sum(len(v) for v in cat_items.values())}\n\n"
    )

    body_parts = [header]
    first_seen: set[int] = set()
    for cat in CATEGORIES:
        items = cat_items[cat]
        if not items:
            body_parts.append(f"## {cat}\n\n(no items at this filter level)\n\n")
            continue
        body_parts.append(f"## {cat}\n\n")
        for i, (item, rn, b) in enumerate(items, start=1):
            body_parts.append(_render_item(i, cat, item, rn, b, first_seen) + "\n")

    md = "".join(body_parts)
    out_path.write_text(md)
    logger.info("[%s] Wrote %s (%d chars, ~%dK tokens)", label, out_path, len(md), len(md) // 4 // 1000)
    return len(md)


def main(config: Config) -> None:
    data_dir = _resolve(REPO_ROOT, config.data_dir)
    round0 = _resolve(data_dir, config.round0_path)
    round3 = _resolve(data_dir, config.round3_path)
    bib = _resolve(data_dir, config.bib_path)
    out_slim = _resolve(data_dir, config.out_slim)
    out_medium = _resolve(data_dir, config.out_medium)
    full_oracle = _resolve(data_dir, config.full_oracle_path)

    if not round0.exists() or not round3.exists():
        logger.error("Need round_0_summary.json and round_3_summary.json to exist (got %s, %s)", round0, round3)
        sys.exit(1)
    if not bib.exists():
        logger.error("Need bibliography_v2.jsonl (got %s)", bib)
        sys.exit(1)

    n_slim = build_oracle(min_score=3, out_path=out_slim, label="slim score≥3",
                          round0_path=round0, round3_path=round3, bib_path=bib,
                          oracle_label=config.label)
    n_med = build_oracle(min_score=2, out_path=out_medium, label="medium score≥2",
                         round0_path=round0, round3_path=round3, bib_path=bib,
                         oracle_label=config.label)

    logger.info("=" * 60)
    logger.info("Oracle size triplet:")
    if full_oracle.exists():
        full = full_oracle.stat().st_size
        logger.info("  full   (%s):      %7d chars (~%dK tokens)", full_oracle.name, full, full // 4 // 1000)
    else:
        logger.info("  full   (%s): NOT FOUND (skipping)", full_oracle.name)
    logger.info("  medium (%s): %7d chars (~%dK tokens)", out_medium.name, n_med, n_med // 4 // 1000)
    logger.info("  slim   (%s):   %7d chars (~%dK tokens)", out_slim.name, n_slim, n_slim // 4 // 1000)


if __name__ == "__main__":
    main(chz.entrypoint(Config))
