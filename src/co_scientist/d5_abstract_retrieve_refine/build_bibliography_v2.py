"""Phase 2A-bis-3 + 4 + 5 + 7: build clean bibliography_v2 with full text.

Reads `data/bibliography/cited_bibtex_v1.jsonl` (90 entries with structured BibTeX metadata),
resolves each via S2 batch (arxiv/doi) + OpenAlex (title), verifies match, downloads
arxiv PDF for arxiv-bearing papers, and outputs:

- `data/bibliography/resolved_v2.jsonl` (resolved entries with full_text_path field)
- `data/bibliography/full_text/{arxiv_id}.md` (per-paper extracted text from arxiv PDF)
- `data/bibliography/REDO_SUMMARY_v2.md` (final accounting)
- `data/_archive_phase2a_v1/` (move old artifacts)

Each output bibliography entry:
  {
    "cite_key": "guo2025deepseek",
    "title": "DeepSeek-R1 incentivizes reasoning...",
    "authors": [...],
    "year": 2025,
    "venue": "Nature",
    "arxiv_id": "2501.12948",
    "doi": "...",
    "abstract": "...",
    "tldr": "...",
    "citation_count": NNN,
    "s2_paper_id": "...",
    "is_paper": true,
    "full_text_path": "data/bibliography/full_text/2501.12948.md",  // null if not fetched
    "resolution_source": "s2_arxiv" | "s2_doi" | "openalex_title" | "non_paper"
  }
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

import chz
import httpx
import pypdf

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.shared.paper_retrieval import S2BackwardRetriever, PaperRetriever, S2Reference

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    input_path: str = "projects/d5_abstract_retrieve_refine/data/bibliography/cited_bibtex_v1.jsonl"
    out_bib: str = "projects/d5_abstract_retrieve_refine/data/bibliography/resolved_v2.jsonl"
    out_summary: str = "projects/d5_abstract_retrieve_refine/data/bibliography/REDO_SUMMARY_v2.md"
    pdf_dir: str = "projects/d5_abstract_retrieve_refine/data/bibliography/full_text"
    unresolved_path: str = ""  # if empty, derived as out_bib sibling

# Cite keys clearly NOT papers (web refs, tools, press releases)
NON_PAPER_KEYS = {
    "atcoderinc": "AtCoder competition platform website",
    "tttatcoder_ahc039_submission_59660035": "AtCoder submission page (not a paper)",
    "sakana2025ahc058": "Sakana AI press release (blog/news)",
    "sutton2019bitter": "Sutton's blog 'The Bitter Lesson' (not peer-reviewed paper)",
    "tml2025tinker": "Tinker tool product page (not a paper)",
}


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[\-—]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _word_overlap_score(a: str, b: str) -> int:
    """Word-triplet overlap count between two texts (normalized)."""
    a_norm, b_norm = _normalize(a), _normalize(b)
    words = a_norm.split()
    if len(words) < 3:
        return sum(1 for i in range(len(words) - 1) if " ".join(words[i:i+2]) in b_norm) * 2
    return sum(1 for i in range(len(words) - 2) if " ".join(words[i:i+3]) in b_norm)


def _last_name(name: str) -> str:
    """Get last name from BibTeX-style 'LastName, FirstName' or 'FirstName LastName'."""
    n = _normalize(name)
    if "," in n:
        return n.split(",")[0].strip()
    parts = n.split()
    return parts[-1] if parts else ""


def _verify_match(bibtex: dict, resolved: dict) -> tuple[bool, str]:
    """Returns (passed, reason). Compares bibtex truth against resolved S2/OpenAlex result.

    Designed to be PERMISSIVE — only flag wrong matches, not slight title diffs.
    Match passes if EITHER (a) title-overlap is sufficient OR (b) author last-names match.
    Both checks fail → wrong match.
    """
    bx_title = bibtex.get("title", "") or ""
    rs_title = resolved.get("title", "") or ""
    bx_year = bibtex.get("year")
    rs_year = resolved.get("year")
    bx_authors = bibtex.get("authors") or []
    rs_authors = resolved.get("authors") or []

    # Title overlap: compute against the SHORTER title's word-grams
    title_score = max(
        _word_overlap_score(bx_title, rs_title),
        _word_overlap_score(rs_title, bx_title),
    )
    short_title_words = min(
        len(_normalize(bx_title).split()), len(_normalize(rs_title).split())
    )
    # Looser threshold based on how short the title is
    if short_title_words <= 3:
        title_thresh = 1
    elif short_title_words <= 6:
        title_thresh = 2
    else:
        title_thresh = 3
    title_ok = title_score >= title_thresh

    # Year check (loose: ±2 years)
    year_ok = True
    if bx_year and rs_year and abs(bx_year - rs_year) > 2:
        year_ok = False

    # Author last-name match
    author_ok = True
    bx_last = _last_name(bx_authors[0]) if bx_authors else ""
    rs_last = _last_name(rs_authors[0]) if rs_authors else ""
    if bx_last and rs_last:
        if not (bx_last in rs_last or rs_last in bx_last or bx_last == rs_last):
            author_ok = False

    # Combine: pass if (title OR author) AND year is ok
    # This is permissive: title alone or author alone is enough
    if not year_ok:
        return False, f"year_mismatch (bibtex={bx_year}, resolved={rs_year})"
    if not (title_ok or author_ok):
        return False, (
            f"BOTH title_overlap={title_score}<{title_thresh} "
            f"AND author_mismatch ({bx_last} vs {rs_last}) "
            f"(bibtex='{bx_title[:40]}', got='{rs_title[:40]}')"
        )
    return True, "ok"


def download_arxiv_pdf(arxiv_id: str, dest_path: Path) -> bool:
    """Download arxiv PDF + extract text via pypdf, save markdown text to dest_path."""
    if dest_path.exists():
        return True
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            pdf_bytes = resp.content
    except Exception as e:
        logger.warning("arxiv PDF download failed for %s: %s", arxiv_id, e)
        return False
    pdf_path = dest_path.with_suffix(".pdf.tmp")
    pdf_path.write_bytes(pdf_bytes)
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        pages_text = []
        for p in reader.pages:
            try:
                pages_text.append(p.extract_text() or "")
            except Exception:
                pass
        full = "\n\n".join(pages_text)
        full = re.sub(r"[ \t]+\n", "\n", full)
        full = re.sub(r"\n{3,}", "\n\n", full)
        # Trim to first 30K chars (oracle round 1-3 doesn't need everything)
        text = full[:30000]
        dest_path.write_text(text)
        pdf_path.unlink(missing_ok=True)
        return True
    except Exception as e:
        logger.warning("pypdf extract failed for %s: %s", arxiv_id, e)
        pdf_path.unlink(missing_ok=True)
        return False


def main(config: Config) -> None:
    input_path = Path(config.input_path)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    out_bib = Path(config.out_bib)
    if not out_bib.is_absolute():
        out_bib = REPO_ROOT / out_bib
    out_summary = Path(config.out_summary)
    if not out_summary.is_absolute():
        out_summary = REPO_ROOT / out_summary
    pdf_dir = Path(config.pdf_dir)
    if not pdf_dir.is_absolute():
        pdf_dir = REPO_ROOT / pdf_dir
    if config.unresolved_path:
        unresolved_path = Path(config.unresolved_path)
        if not unresolved_path.is_absolute():
            unresolved_path = REPO_ROOT / unresolved_path
    else:
        unresolved_path = out_bib.with_name("ttt_discover_unresolved_v2.jsonl")

    if not input_path.exists():
        logger.error("Input %s not found — run parse_bibtex_v1 first.", input_path)
        sys.exit(1)
    cited_bibtex = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            cited_bibtex.append(json.loads(line))
    logger.info("Loaded %d cited BibTeX entries", len(cited_bibtex))

    pdf_dir.mkdir(parents=True, exist_ok=True)
    out_bib.parent.mkdir(parents=True, exist_ok=True)

    s2 = S2BackwardRetriever()
    oa = PaperRetriever()

    final_entries: list[dict] = []
    failed: list[dict] = []
    counts = {"non_paper": 0, "s2_arxiv": 0, "s2_doi": 0, "openalex_title": 0, "failed": 0,
               "with_full_text": 0}

    try:
        # ============================================================
        # Phase A: separate non-papers + arxiv-bearing for batch
        # ============================================================
        non_paper_entries = []
        arxiv_entries: list[dict] = []
        doi_entries: list[dict] = []
        title_only_entries: list[dict] = []
        for bx in cited_bibtex:
            if bx["cite_key"] in NON_PAPER_KEYS:
                non_paper_entries.append(bx)
            elif bx.get("arxiv_id"):
                arxiv_entries.append(bx)
            elif bx.get("doi"):
                doi_entries.append(bx)
            else:
                title_only_entries.append(bx)
        logger.info(
            "Cited bibtex split: %d non-paper, %d arxiv, %d doi, %d title-only",
            len(non_paper_entries), len(arxiv_entries), len(doi_entries), len(title_only_entries),
        )

        # ============================================================
        # Non-paper handling
        # ============================================================
        for bx in non_paper_entries:
            final_entries.append({
                "cite_key": bx["cite_key"],
                "title": bx.get("title", ""),
                "authors": bx.get("authors", []),
                "year": bx.get("year"),
                "venue": bx.get("venue", ""),
                "arxiv_id": None,
                "doi": None,
                "abstract": "",
                "tldr": "",
                "citation_count": 0,
                "s2_paper_id": None,
                "is_paper": False,
                "non_paper_reason": NON_PAPER_KEYS[bx["cite_key"]],
                "full_text_path": None,
                "resolution_source": "non_paper",
            })
            counts["non_paper"] += 1

        # ============================================================
        # Phase B: S2 batch for arxiv IDs
        # ============================================================
        if arxiv_entries:
            arxiv_id_list = [f"ARXIV:{e['arxiv_id']}" for e in arxiv_entries]
            logger.info("S2 batch lookup for %d arxiv IDs...", len(arxiv_id_list))
            results = s2.fetch_batch(arxiv_id_list)
            for bx, batch_id in zip(arxiv_entries, arxiv_id_list):
                ref: S2Reference | None = results.get(batch_id)
                if ref is None:
                    failed.append({"cite_key": bx["cite_key"], "reason": "s2_arxiv_not_found", "tried": batch_id})
                    continue
                resolved_dict = ref.to_dict()
                ok, reason = _verify_match(bx, resolved_dict)
                if not ok:
                    failed.append({"cite_key": bx["cite_key"], "reason": f"s2_arxiv_verify_fail: {reason}"})
                    continue
                final_entries.append({
                    "cite_key": bx["cite_key"],
                    "title": ref.title or bx.get("title", ""),
                    "authors": ref.authors or bx.get("authors", []),
                    "year": ref.year or bx.get("year"),
                    "venue": ref.venue or bx.get("venue", ""),
                    "arxiv_id": ref.arxiv_id or bx.get("arxiv_id"),
                    "doi": ref.doi or bx.get("doi"),
                    "abstract": ref.abstract,
                    "tldr": ref.tldr,
                    "citation_count": ref.citation_count,
                    "s2_paper_id": ref.paper_id,
                    "is_paper": True,
                    "full_text_path": None,  # filled in Phase D
                    "resolution_source": "s2_arxiv",
                })
                counts["s2_arxiv"] += 1

        # ============================================================
        # Phase C: S2 batch for DOI IDs
        # ============================================================
        if doi_entries:
            doi_id_list = [f"DOI:{e['doi']}" for e in doi_entries]
            logger.info("S2 batch lookup for %d DOIs...", len(doi_id_list))
            results = s2.fetch_batch(doi_id_list)
            for bx, batch_id in zip(doi_entries, doi_id_list):
                ref: S2Reference | None = results.get(batch_id)
                if ref is None:
                    # Fallback to title search
                    title_only_entries.append(bx)
                    continue
                resolved_dict = ref.to_dict()
                ok, reason = _verify_match(bx, resolved_dict)
                if not ok:
                    title_only_entries.append(bx)
                    continue
                final_entries.append({
                    "cite_key": bx["cite_key"],
                    "title": ref.title or bx.get("title", ""),
                    "authors": ref.authors or bx.get("authors", []),
                    "year": ref.year or bx.get("year"),
                    "venue": ref.venue or bx.get("venue", ""),
                    "arxiv_id": ref.arxiv_id,
                    "doi": ref.doi or bx.get("doi"),
                    "abstract": ref.abstract,
                    "tldr": ref.tldr,
                    "citation_count": ref.citation_count,
                    "s2_paper_id": ref.paper_id,
                    "is_paper": True,
                    "full_text_path": None,
                    "resolution_source": "s2_doi",
                })
                counts["s2_doi"] += 1

        # ============================================================
        # Phase D: OpenAlex title search for the rest
        # ============================================================
        if title_only_entries:
            logger.info("OpenAlex title search for %d entries...", len(title_only_entries))
            for bx in title_only_entries:
                title = bx.get("title", "")
                if not title:
                    failed.append({"cite_key": bx["cite_key"], "reason": "no_title_in_bibtex"})
                    continue
                results = oa.search_papers(title[:300], limit=1)
                if not results:
                    failed.append({"cite_key": bx["cite_key"], "reason": "openalex_no_match", "title": title[:60]})
                    continue
                p = results[0]
                resolved_dict = {
                    "title": p.title,
                    "authors": p.authors,
                    "year": p.year,
                }
                ok, reason = _verify_match(bx, resolved_dict)
                if not ok:
                    failed.append({"cite_key": bx["cite_key"], "reason": f"openalex_verify_fail: {reason}"})
                    continue
                final_entries.append({
                    "cite_key": bx["cite_key"],
                    "title": p.title or bx.get("title", ""),
                    "authors": p.authors or bx.get("authors", []),
                    "year": p.year or bx.get("year"),
                    "venue": p.venue or bx.get("venue", ""),
                    "arxiv_id": None,
                    "doi": p.doi or bx.get("doi"),
                    "abstract": p.abstract,
                    "tldr": "",
                    "citation_count": p.citation_count,
                    "s2_paper_id": None,
                    "is_paper": True,
                    "full_text_path": None,
                    "resolution_source": "openalex_title",
                })
                counts["openalex_title"] += 1

        # ============================================================
        # Phase E: Download arxiv PDFs for resolved arxiv-bearing papers
        # ============================================================
        arxiv_resolved = [e for e in final_entries if e.get("arxiv_id") and e["is_paper"]]
        logger.info("Phase E: downloading %d arxiv PDFs (parallel-ish, no rate limit on arxiv)...", len(arxiv_resolved))
        for i, e in enumerate(arxiv_resolved):
            arxiv_id = e["arxiv_id"]
            dest = pdf_dir / f"{arxiv_id}.md"
            ok = download_arxiv_pdf(arxiv_id, dest)
            if ok:
                e["full_text_path"] = str(dest.relative_to(REPO_ROOT))
                counts["with_full_text"] += 1
            if (i + 1) % 5 == 0:
                logger.info("  Downloaded %d/%d", i + 1, len(arxiv_resolved))

    finally:
        s2.close()
        oa.close()

    counts["failed"] = len(failed)
    final_entries.sort(key=lambda x: x["cite_key"])
    with open(out_bib, "w") as f:
        for e in final_entries:
            f.write(json.dumps(e) + "\n")
    if failed:
        with open(unresolved_path, "w") as f:
            for e in failed:
                f.write(json.dumps(e) + "\n")
        logger.warning("Wrote %d unresolved to %s", len(failed), unresolved_path)

    # ============================================================
    # Summary report
    # ============================================================
    md = ["# Bibliography Redo Summary (Phase 2A-bis)\n"]
    md.append(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
    md.append(f"## Accounting\n\n")
    md.append(f"- Total cite keys (from main.tex): {len(cited_bibtex)}\n")
    md.append(f"- Resolved via S2 arxiv batch: **{counts['s2_arxiv']}**\n")
    md.append(f"- Resolved via S2 DOI batch:   **{counts['s2_doi']}**\n")
    md.append(f"- Resolved via OpenAlex title: **{counts['openalex_title']}**\n")
    md.append(f"- Marked non-paper:            **{counts['non_paper']}** (web/tool/press)\n")
    md.append(f"- Failed to resolve:           **{counts['failed']}**\n")
    accounted = sum(counts[k] for k in ["s2_arxiv", "s2_doi", "openalex_title", "non_paper", "failed"])
    md.append(f"- Total accounted:             **{accounted}**\n\n")
    md.append(f"## Full-text status\n\n")
    md.append(f"- Papers with full text downloaded: **{counts['with_full_text']}** (arxiv-bearing)\n\n")

    md.append(f"## Failed to resolve ({len(failed)} entries)\n\n")
    for f in failed:
        md.append(f"- `{f['cite_key']}` — {f['reason']}\n")

    md.append(f"\n## Non-paper entries\n\n")
    for cite_key, reason in NON_PAPER_KEYS.items():
        md.append(f"- `{cite_key}` — {reason}\n")

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text("".join(md))
    logger.info("Wrote summary: %s", out_summary)
    logger.info(
        "FINAL: %d entries in bibliography_v2 (%d with full text), %d failed",
        len(final_entries), counts["with_full_text"], len(failed),
    )


if __name__ == "__main__":
    main(chz.entrypoint(Config))
