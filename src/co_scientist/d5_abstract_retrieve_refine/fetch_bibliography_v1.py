"""Fetch TTT-Discover bibliography by parsing the PDF's References section
+ resolving each ref via S2 (by arxiv ID if present, else title search).

Phase 2A of D5 Realignment. S2 backward-references endpoint returned empty
for TTT-Discover (paper too new — Feb 2026, not yet ingested by S2). Fallback:
parse the local paper.pdf, extract references, look up each one via S2.

Output: `projects/d5_abstract_retrieve_refine/data/bibliography/resolved_v1.jsonl`
Each line = {paper_id, title, authors, year, abstract, tldr, citation_count,
arxiv_id, doi, venue, ref_num} (ref_num = numeric index in original paper).

Usage:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.fetch_bibliography_v1
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

import chz
import pypdf

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.shared.paper_retrieval import S2BackwardRetriever, S2Reference

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    pdf_path: str = "shared/papers/by_topic/01_test_time_search/TTT-Discover/paper.pdf"
    out_path: str = "projects/d5_abstract_retrieve_refine/data/bibliography/resolved_v1.jsonl"
    raw_refs_path: str = "projects/d5_abstract_retrieve_refine/data/bibliography/raw_refs_v1.jsonl"
    unresolved_path: str = ""  # if empty, derived as out_path with "ttt_discover_unresolved_refs.jsonl" sibling


# Pattern: [N] Authors. Title. Venue/Journal, year.
# Multiple lines per ref due to PDF wrapping. Refs separated by [N+1].
REF_NUM_RE = re.compile(r"\n\[(\d+)\]\s+", re.MULTILINE)
ARXIV_ID_RE = re.compile(r"arXiv[:\s]?(\d{4}\.\d{4,5})", re.IGNORECASE)
URL_HTTP_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)


def extract_refs_section(pdf_path: Path) -> str:
    pdf = pypdf.PdfReader(str(pdf_path))
    text = ""
    for p in pdf.pages:
        try:
            text += p.extract_text() + "\n"
        except Exception:
            pass
    # Find "References\n" (with newline)
    m = re.search(r"\n\s*References\s*\n", text)
    if not m:
        # fallback: rfind
        idx = text.lower().rfind("references")
        return text[idx:] if idx > 0 else ""
    return text[m.end():]


def split_refs(refs_text: str) -> list[tuple[int, str]]:
    """Split into list of (ref_num, raw_text). Stops at non-numeric headers like Appendix."""
    refs: list[tuple[int, str]] = []
    cuts = list(REF_NUM_RE.finditer(refs_text))
    if not cuts:
        return refs
    for i, cut in enumerate(cuts):
        ref_num = int(cut.group(1))
        start = cut.end()
        end = cuts[i + 1].start() if i + 1 < len(cuts) else len(refs_text)
        body = refs_text[start:end].strip()
        # Stop if body looks like an Appendix break (heuristic: contains "Appendix" or page break)
        if "Appendix" in body[:50]:
            break
        refs.append((ref_num, body))
    return refs


def parse_ref_text(body: str) -> dict:
    """Heuristic parse of one reference body to {authors, title, arxiv_id, year}."""
    body = re.sub(r"\s+", " ", body).strip()
    arxiv_match = ARXIV_ID_RE.search(body)
    arxiv_id = arxiv_match.group(1) if arxiv_match else None
    # Try year — last 4-digit number that looks like 19xx-20xx
    year_matches = re.findall(r"\b(19[6-9]\d|20[0-2]\d)\b", body)
    year = int(year_matches[-1]) if year_matches else None
    # Title heuristic: between first ". " and second ". " (after authors)
    parts = body.split(". ", 2)
    title = ""
    if len(parts) >= 2:
        title = parts[1].strip().rstrip(".")
        # Clean — drop trailing journal info
    return {
        "raw": body,
        "title": title,
        "year": year,
        "arxiv_id": arxiv_id,
    }


def main(config: Config):
    pdf_path = Path(config.pdf_path)
    if not pdf_path.is_absolute():
        pdf_path = REPO_ROOT / pdf_path
    out_path = Path(config.out_path)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    raw_refs_path = Path(config.raw_refs_path)
    if not raw_refs_path.is_absolute():
        raw_refs_path = REPO_ROOT / raw_refs_path
    if config.unresolved_path:
        unresolved_path = Path(config.unresolved_path)
        if not unresolved_path.is_absolute():
            unresolved_path = REPO_ROOT / unresolved_path
    else:
        unresolved_path = out_path.with_name("ttt_discover_unresolved_refs.jsonl")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1/3: Extract References section from %s", pdf_path)
    refs_text = extract_refs_section(pdf_path)
    if not refs_text:
        logger.error("No References section found in PDF.")
        sys.exit(1)
    logger.info("References section: %d chars", len(refs_text))

    raw_refs = split_refs(refs_text)
    logger.info("Step 2/3: Parsed %d numbered references", len(raw_refs))

    parsed_refs = [(num, parse_ref_text(body)) for num, body in raw_refs]
    # Persist raw for diagnosis
    with open(raw_refs_path, "w") as f:
        for num, p in parsed_refs:
            f.write(json.dumps({"ref_num": num, **p}) + "\n")
    logger.info("Persisted raw refs to %s", raw_refs_path)

    # Stats
    with_arxiv = sum(1 for _, p in parsed_refs if p["arxiv_id"])
    logger.info(
        "Distribution: %d/%d refs have arxiv_id; %d have title",
        with_arxiv, len(parsed_refs),
        sum(1 for _, p in parsed_refs if p["title"]),
    )

    logger.info("Step 3/3: Resolve via S2 (batch endpoint for arxiv IDs, then title search for rest)...")
    retriever = S2BackwardRetriever()
    resolved: list[dict] = []
    failed: list[dict] = []
    try:
        # Phase 3a: BATCH lookup for arxiv-ID-bearing refs
        with_arxiv = [(num, p) for num, p in parsed_refs if p.get("arxiv_id")]
        without_arxiv = [(num, p) for num, p in parsed_refs if not p.get("arxiv_id")]
        logger.info("Phase 3a: batch lookup for %d arxiv-IDs", len(with_arxiv))
        batch_ids = [f"ARXIV:{p['arxiv_id']}" for _, p in with_arxiv]
        batch_results = retriever.fetch_batch(batch_ids)
        for (num, p), batch_id in zip(with_arxiv, batch_ids):
            ref = batch_results.get(batch_id)
            if ref is None:
                logger.info("  [%d] arxiv:%s batch missed; will retry by title", num, p["arxiv_id"])
                # retry via title in 3b
                without_arxiv.append((num, p))
            else:
                resolved.append({"ref_num": num, **ref.to_dict()})
        logger.info("Phase 3a done: %d/%d arxiv refs resolved via batch", len(resolved), len(with_arxiv))

        # Phase 3b: per-paper title search for the rest
        logger.info("Phase 3b: title search for %d refs (slower, rate-limited)", len(without_arxiv))
        for i, (num, p) in enumerate(without_arxiv):
            ref = None
            if p["title"]:
                ref = retriever.search_by_title(p["title"], year_hint=p["year"])
            if ref is None:
                failed.append({"ref_num": num, **p})
                logger.warning("  [%d] could not resolve: %s...", num, p["raw"][:80])
                continue
            resolved.append({"ref_num": num, **ref.to_dict()})
            if (i + 1) % 5 == 0:
                logger.info("  Resolved [%d/%d title-search]; latest = %s (cit=%d)",
                             i + 1, len(without_arxiv), ref.title[:50], ref.citation_count)
    finally:
        retriever.close()

    logger.info("Resolved %d/%d refs (%d failed)", len(resolved), len(parsed_refs), len(failed))
    with open(out_path, "w") as f:
        for r in resolved:
            f.write(json.dumps(r) + "\n")
    if failed:
        with open(unresolved_path, "w") as f:
            for r in failed:
                f.write(json.dumps(r) + "\n")
        logger.warning("Unresolved refs persisted to %s", unresolved_path)

    # Top-5 by citation count
    top5 = sorted(resolved, key=lambda r: -r.get("citation_count", 0))[:5]
    logger.info("Top-5 by citationCount:")
    for r in top5:
        logger.info(
            "  %d cit | [%d] %s — %s (%s)",
            r.get("citation_count", 0), r["ref_num"],
            (r.get("title") or "")[:60],
            r.get("authors", ["?"])[0] if r.get("authors") else "?",
            r.get("year"),
        )
    logger.info("Total S2 API calls: %d", retriever.total_api_calls)
    logger.info("Output: %s (%d entries)", out_path, len(resolved))


if __name__ == "__main__":
    main(chz.entrypoint(Config))
