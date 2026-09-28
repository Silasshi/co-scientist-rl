"""Build D5 oracle abstraction v2 via multi-round Opus extraction.

Phase 2B of D5 Realignment. Reads bibliography (from fetch_bibliography_v1.py),
runs Round 0 relevance filter + Round 1-3 extraction with cumulative memory,
writes the final oracle to `data/oracles/oracle_v2_2026_04_26_build/final.md`.

Per-paper Opus calls dispatched via file-bus pattern: orchestrator writes
JSON requests, subagent (spawned by main agent) processes them and writes
JSON responses, orchestrator collects.

Workflow:
    1. submit_round0(): write all relevance-filter requests
    2. main agent dispatches subagents to process them
    3. collect_round0(): parse responses, filter to relevance ≥ 1
    4. submit_round1(): write extraction requests for surviving papers
    5. dispatch
    6. collect_round1()
    7. submit_round2(): write refine requests with cumulative state
    8. dispatch
    9. collect_round2()
    10. submit_round3(): same
    11. collect_round3()
    12. assemble(): write final oracle_v2.md

Each step persists state so the orchestrator can be resumed.

Usage (each phase invoked separately so user can confirm + main agent can dispatch):
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.build_oracle_v2 phase=submit_round0
    # main agent dispatches subagents to process oracle_requests/round_0/*.json
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.build_oracle_v2 phase=collect_round0
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.build_oracle_v2 phase=submit_round1
    # ... etc
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.build_oracle_v2 phase=assemble
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import chz

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.d5_abstract_retrieve_refine.oracle_prompts_v2 import (
    ORACLE_CATEGORIES,
    ROUND0_RELEVANCE_PROMPT,
    ROUND1_EXTRACT_PROMPT,
    ROUND_REFINE_PROMPT,
    render_category_descriptions,
    render_accumulated_state,
    render_prior_extraction,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    phase: str = "submit_round0"  # submit_round{0,1,2,3} | collect_round{0,1,2,3} | assemble
    bibliography_path: str = "projects/d5_abstract_retrieve_refine/data/bibliography/resolved_v2.jsonl"
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"
    out_dir: str = "projects/d5_abstract_retrieve_refine/data/oracles/oracle_v2_2026_04_26_build"
    final_oracle_path: str = "projects/d5_abstract_retrieve_refine/data/oracles/oracle_v2_2026_04_26_build/final.md"
    # Round 1/2/3 content: title+abstract+TLDR for all; full text for top-K (relevance=3)
    fulltext_top_k: int = 5


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _load_bibliography(path: Path) -> list[dict]:
    """Load bibliography_v2 schema. Filters non-paper entries. Assigns ref_num + paper_id
    for compatibility with the orchestrator's existing ref_num/paper_id keys.

    bibliography_v2 schema: cite_key, title, authors, year, venue, abstract, tldr,
    citation_count, s2_paper_id, full_text_path, is_paper, doi, arxiv_id, resolution_source.
    """
    refs = []
    raw = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            raw.append(json.loads(line))
    # Filter is_paper=True only; deterministic order by cite_key
    raw_papers = [r for r in raw if r.get("is_paper", True)]
    raw_papers.sort(key=lambda r: r.get("cite_key", ""))
    for i, r in enumerate(raw_papers, start=1):
        r["ref_num"] = i
        # paper_id used for top-K; fall back to cite_key if no S2 id
        r["paper_id"] = r.get("s2_paper_id") or r.get("cite_key")
        refs.append(r)
    logger.info("Loaded %d papers (filtered out %d non-paper entries)", len(refs), len(raw) - len(refs))
    return refs


def _round_dir(out_dir: Path, round_idx: int, kind: str) -> Path:
    """kind in {'requests', 'responses'}"""
    d = out_dir / f"round_{round_idx}" / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def _short_authors(authors: list[str]) -> str:
    if not authors: return "?"
    if len(authors) == 1: return authors[0]
    if len(authors) == 2: return f"{authors[0]} and {authors[1]}"
    return f"{authors[0]} et al."


FULLTEXT_CHAR_LIMIT = 20000


def _paper_content_for_round(ref: dict, round_idx: int, top5_ids: set[str]) -> str:
    """Decide what content to include for this paper at this round.
    Priority: full_text (truncated to FULLTEXT_CHAR_LIMIT chars) > abstract+TLDR fallback.
    """
    parts = []
    full_path_str = ref.get("full_text_path")
    if full_path_str:
        full_path = (REPO_ROOT / full_path_str).resolve() if not Path(full_path_str).is_absolute() else Path(full_path_str)
        if full_path.exists():
            text = full_path.read_text(errors="replace")
            if len(text) > FULLTEXT_CHAR_LIMIT:
                text = text[:FULLTEXT_CHAR_LIMIT] + f"\n\n[... truncated at {FULLTEXT_CHAR_LIMIT} chars]"
            parts.append(f"Full Text:\n{text}")
        else:
            logger.warning("full_text_path does not exist for %s: %s", ref.get("cite_key"), full_path)
    if not parts:
        if ref.get("abstract"):
            parts.append(f"Abstract: {ref['abstract']}")
        if ref.get("tldr"):
            parts.append(f"TLDR: {ref['tldr']}")
    if not parts:
        parts.append("(no full text / abstract / TLDR available)")
    s2_id = ref.get("s2_paper_id") or ref.get("paper_id")
    if s2_id in top5_ids:
        parts.append("(top-5 by citation count — Opus, please go deep on this paper)")
    return "\n\n".join(parts)


# =============================================================================
# Phase: submit_round0 — write relevance-filter requests
# =============================================================================

def submit_round0(config: Config):
    bib_path = _resolve(config.bibliography_path)
    goal = _resolve(config.goal_path).read_text().strip()
    refs = _load_bibliography(bib_path)
    out_dir = _resolve(config.out_dir)
    req_dir = _round_dir(out_dir, 0, "requests")
    logger.info("Round 0: writing %d relevance-filter requests to %s", len(refs), req_dir)

    for ref in refs:
        prompt = ROUND0_RELEVANCE_PROMPT.format(
            goal=goal,
            title=ref.get("title", ""),
            authors_short=_short_authors(ref.get("authors", [])),
            year=ref.get("year") or "?",
            venue=ref.get("venue") or "(unknown)",
            abstract=ref.get("abstract") or "(no abstract)",
            tldr=ref.get("tldr") or "(no TLDR)",
        )
        payload = {
            "round": 0,
            "kind": "relevance_filter",
            "ref_num": ref.get("ref_num"),
            "paper_id": ref.get("paper_id"),
            "title": ref.get("title"),
            "prompt": prompt,
        }
        req_path = req_dir / f"ref_{ref.get('ref_num', 0):03d}.json"
        tmp = req_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(req_path)
    logger.info("Submitted %d round-0 requests. Main agent should now spawn subagents to process them.", len(refs))


def collect_round0(config: Config) -> dict[int, dict]:
    """Read round-0 responses, return {ref_num: {relevance_score, rationale, expected_categories}}."""
    out_dir = _resolve(config.out_dir)
    resp_dir = _round_dir(out_dir, 0, "responses")
    bib_path = _resolve(config.bibliography_path)
    refs = _load_bibliography(bib_path)

    relevance: dict[int, dict] = {}
    n_missing = 0
    for ref in refs:
        ref_num = ref.get("ref_num")
        rp = resp_dir / f"ref_{ref_num:03d}.json"
        if not rp.exists():
            n_missing += 1
            continue
        try:
            r = json.loads(rp.read_text())
            relevance[ref_num] = {
                "relevance_score": r.get("relevance_score", 0),
                "rationale": r.get("rationale", ""),
                "expected_categories": r.get("expected_categories", []),
                "title": ref.get("title"),
            }
        except json.JSONDecodeError as e:
            logger.warning("Round 0 ref %d response parse error: %s", ref_num, e)
            n_missing += 1

    logger.info("Round 0 collected: %d/%d responses (%d missing)",
                 len(relevance), len(refs), n_missing)
    # Distribution
    score_dist = {}
    for k in range(4):
        score_dist[k] = sum(1 for v in relevance.values() if v["relevance_score"] == k)
    logger.info("Score distribution: 0=%d, 1=%d, 2=%d, 3=%d",
                 score_dist[0], score_dist[1], score_dist[2], score_dist[3])

    # Save
    summary_path = out_dir / "round_0_summary.json"
    summary_path.write_text(json.dumps(relevance, indent=2))
    logger.info("Saved %s", summary_path)
    return relevance


# =============================================================================
# Phase: submit_round1 — extraction requests for surviving papers (relevance >= 1)
# =============================================================================

def submit_round1(config: Config):
    out_dir = _resolve(config.out_dir)
    relevance = json.loads((out_dir / "round_0_summary.json").read_text())
    relevance = {int(k): v for k, v in relevance.items()}
    refs = _load_bibliography(_resolve(config.bibliography_path))
    refs_by_num = {r.get("ref_num"): r for r in refs}

    # Top-K by citation count
    top_k_by_cit = sorted(refs, key=lambda r: -r.get("citation_count", 0))[:config.fulltext_top_k]
    top_k_ids = {r.get("paper_id") for r in top_k_by_cit}

    goal = _resolve(config.goal_path).read_text().strip()
    req_dir = _round_dir(out_dir, 1, "requests")
    cat_desc = render_category_descriptions()

    n = 0
    for ref_num, info in relevance.items():
        if info["relevance_score"] < 1:
            continue
        ref = refs_by_num.get(ref_num)
        if not ref: continue
        content = _paper_content_for_round(ref, 1, top_k_ids)
        prompt = ROUND1_EXTRACT_PROMPT.format(
            goal=goal,
            title=ref.get("title", ""),
            authors_short=_short_authors(ref.get("authors", [])),
            year=ref.get("year") or "?",
            venue=ref.get("venue") or "(unknown)",
            content=content,
            expected_categories_csv=", ".join(info.get("expected_categories", [])) or "(judge as you read)",
            category_descriptions=cat_desc,
        )
        payload = {
            "round": 1,
            "kind": "extract",
            "ref_num": ref_num,
            "paper_id": ref.get("paper_id"),
            "title": ref.get("title"),
            "prompt": prompt,
        }
        req_path = req_dir / f"ref_{ref_num:03d}.json"
        tmp = req_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(req_path)
        n += 1
    logger.info("Submitted %d round-1 extraction requests (filtered to relevance >=1)", n)


def collect_round_extract(config: Config, round_idx: int) -> dict[int, dict]:
    """Read round-N extraction responses. Used for round 1, 2, 3."""
    out_dir = _resolve(config.out_dir)
    resp_dir = _round_dir(out_dir, round_idx, "responses")
    extractions: dict[int, dict] = {}
    n_missing = 0
    for rp in sorted(resp_dir.glob("ref_*.json")):
        ref_num = int(rp.stem.replace("ref_", ""))
        try:
            r = json.loads(rp.read_text())
            extractions[ref_num] = {
                "Insights": r.get("Insights", []),
                "Methodology": r.get("Methodology", []),
                "Theory": r.get("Theory", []),
                "Math": r.get("Math", []),
                "Empirical": r.get("Empirical", []),
                "Failure_modes": r.get("Failure_modes", []),
            }
        except json.JSONDecodeError as e:
            logger.warning("Round %d ref %d parse error: %s", round_idx, ref_num, e)
            n_missing += 1
    logger.info("Round %d collected: %d extractions (%d malformed)", round_idx, len(extractions), n_missing)
    summary_path = out_dir / f"round_{round_idx}_summary.json"
    summary_path.write_text(json.dumps(extractions, indent=2))
    return extractions


# =============================================================================
# Phase: submit_round2 / submit_round3 — refine with cumulative state
# =============================================================================

def submit_round_refine(config: Config, round_idx: int):
    """round_idx must be 2 or 3. Uses round_idx-1 extractions as accumulated state +
    prior_extraction."""
    assert round_idx in (2, 3)
    out_dir = _resolve(config.out_dir)
    bib_path = _resolve(config.bibliography_path)
    refs = _load_bibliography(bib_path)
    refs_by_num = {r.get("ref_num"): r for r in refs}

    prior = json.loads((out_dir / f"round_{round_idx-1}_summary.json").read_text())
    prior = {int(k): v for k, v in prior.items()}

    # Build accumulated state by category, with source paper attribution
    accumulated = {cat: [] for cat in ORACLE_CATEGORIES}
    for ref_num, ext in prior.items():
        ref = refs_by_num.get(ref_num)
        if not ref: continue
        cite = f"[{ref_num}] {_short_authors(ref.get('authors', []))} {ref.get('year', '?')}"
        for cat in ORACLE_CATEGORIES:
            for it in ext.get(cat, []):
                accumulated[cat].append({**it, "source_paper": cite})

    accumulated_state_text = render_accumulated_state(accumulated)

    top_k_by_cit = sorted(refs, key=lambda r: -r.get("citation_count", 0))[:config.fulltext_top_k]
    top_k_ids = {r.get("paper_id") for r in top_k_by_cit}
    goal = _resolve(config.goal_path).read_text().strip()
    req_dir = _round_dir(out_dir, round_idx, "requests")
    cat_desc = render_category_descriptions()

    n = 0
    for ref_num, ext in prior.items():
        ref = refs_by_num.get(ref_num)
        if not ref: continue
        content = _paper_content_for_round(ref, round_idx, top_k_ids)
        prior_text = render_prior_extraction(ext)
        prompt = ROUND_REFINE_PROMPT.format(
            goal=goal,
            title=ref.get("title", ""),
            authors_short=_short_authors(ref.get("authors", [])),
            content=content,
            accumulated_state=accumulated_state_text,
            prior_extraction=prior_text,
            category_descriptions=cat_desc,
        )
        payload = {
            "round": round_idx,
            "kind": "refine",
            "ref_num": ref_num,
            "paper_id": ref.get("paper_id"),
            "title": ref.get("title"),
            "prompt": prompt,
        }
        req_path = req_dir / f"ref_{ref_num:03d}.json"
        tmp = req_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(req_path)
        n += 1
    logger.info("Submitted %d round-%d refine requests", n, round_idx)


# =============================================================================
# Phase: assemble — write final oracle_v2.md
# =============================================================================

def assemble(config: Config):
    out_dir = _resolve(config.out_dir)
    final_path = _resolve(config.final_oracle_path)
    bib_path = _resolve(config.bibliography_path)
    refs = _load_bibliography(bib_path)
    refs_by_num = {r.get("ref_num"): r for r in refs}

    # Use last available round (3 ideally)
    round_summary_path = None
    last_round = None
    for r in [3, 2, 1]:
        p = out_dir / f"round_{r}_summary.json"
        if p.exists():
            round_summary_path = p
            last_round = r
            break
    if round_summary_path is None:
        logger.error("No round-N summary found; run rounds 1-3 first")
        sys.exit(1)
    extractions = json.loads(round_summary_path.read_text())
    extractions = {int(k): v for k, v in extractions.items()}
    logger.info("Assembling from round-%d data: %d papers contributing", last_round, len(extractions))

    # Collect items per category with attribution
    final = {cat: [] for cat in ORACLE_CATEGORIES}
    seen_first_cite: dict[str, str] = {}  # paper_id → bib full citation (for first occurrence)

    def _full_cite(ref) -> str:
        """Author1 et al. YEAR. 'Title' arxiv:XXXX"""
        authors = ref.get("authors", [])
        year = ref.get("year", "?")
        title = ref.get("title", "")
        arxiv = ref.get("arxiv_id")
        cite = f"{_short_authors(authors)} {year}. \"{title}\""
        if arxiv:
            cite += f" arxiv:{arxiv}"
        return cite

    for ref_num, ext in extractions.items():
        ref = refs_by_num.get(ref_num)
        if not ref: continue
        for cat in ORACLE_CATEGORIES:
            for it in ext.get(cat, []):
                final[cat].append({"ref_num": ref_num, "ref": ref, **it})

    # Write markdown
    lines = [
        f"# D5 Oracle Abstraction v2 — TTT-Discover Goal",
        f"",
        f"*Generated by Opus 4.7 multi-round extraction across the TTT-Discover bibliography on {time.strftime('%Y-%m-%d')}.*",
        f"",
        f"## Metadata",
        f"- Bibliography source: TTT-Discover (arxiv 2601.16175) backward citations parsed from PDF",
        f"- Total references parsed: {len(refs)}",
        f"- Papers contributing to oracle (relevance ≥ 1): {len(extractions)}",
        f"- Rounds executed: 0 (relevance) + 1 (extract) + 2-{last_round} (refine)",
        f"- Generated by: Opus 4.7 simulating ceiling output of D5 multi-round abstraction pipeline",
        f"",
    ]

    for cat in ORACLE_CATEGORIES:
        items = final[cat]
        if not items:
            continue
        lines.append(f"## {cat}")
        lines.append("")
        for i, it in enumerate(items, 1):
            ref = it["ref"]
            paper_id = ref.get("paper_id", "")
            if paper_id not in seen_first_cite:
                cite = _full_cite(ref)
                seen_first_cite[paper_id] = cite
                source_str = f"**Source**: {cite} (Ref [{it['ref_num']}])"
            else:
                short = f"({_short_authors(ref.get('authors', []))} {ref.get('year', '?')})"
                source_str = f"**Source**: {short} (Ref [{it['ref_num']}])"
            lines.append(f"### {cat} {i}: {it.get('summary', '?')}")
            lines.append(f"- {source_str}")
            lines.append(f"- **Content**: {it.get('content', '')}")
            applicability = it.get('applicability') or '(not specified)'
            lines.append(f"- **Applicability to TTT-Discover goal**: {applicability}")
            lines.append("")

    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text("\n".join(lines))
    logger.info("Wrote oracle to %s (%d chars)", final_path, sum(len(l) for l in lines))


# =============================================================================
# Main dispatcher
# =============================================================================

def main(config: Config):
    phase = config.phase
    if phase == "submit_round0":
        submit_round0(config)
    elif phase == "collect_round0":
        collect_round0(config)
    elif phase == "submit_round1":
        submit_round1(config)
    elif phase == "collect_round1":
        collect_round_extract(config, 1)
    elif phase == "submit_round2":
        submit_round_refine(config, 2)
    elif phase == "collect_round2":
        collect_round_extract(config, 2)
    elif phase == "submit_round3":
        submit_round_refine(config, 3)
    elif phase == "collect_round3":
        collect_round_extract(config, 3)
    elif phase == "assemble":
        assemble(config)
    else:
        logger.error("Unknown phase: %s", phase)
        sys.exit(1)


if __name__ == "__main__":
    cfg = chz.entrypoint(Config)
    main(cfg)
