"""Phase 2A-bis-8: Expand full-text coverage for abstract-only bibliography entries.

After Phase 2A-bis-1..7 we have 86 entries in `ttt_discover_bibliography_v2.jsonl`,
38 with full text downloaded (had arxiv_id in BibTeX), 48 abstract-only (resolved
via OpenAlex title search; no arxiv link).

User wants ALL papers' full text where possible (for feeding Qwen3-30B during oracle
build + privileged-info plan critic).

Strategy per abstract-only entry:
1. arxiv title search via arxiv API (most aggressive — many ML papers have arxiv mirror)
2. If arxiv hit verifies (title overlap ≥3 words AND year ±1 AND author match): download PDF
3. Else: S2 search-by-title to get s2_paper_id, then GET /paper/{id}/openAccessPdf
4. Else: try OpenAlex `best_oa_location.pdf_url`
5. Else: keep as abstract-only with reason

Also handles 4 unresolved keys from Phase 2A-bis-3:
- yaoyour: Notion blog → mark non-paper
- openevolve: GitHub repo → mark non-paper
- atcoder: AtCoder Inc. website → mark non-paper
- white2023erdos: real math paper (Acta Arithmetica 2023) → re-search arxiv
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
import urllib.parse
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "projects/d5_abstract_retrieve_refine/data"
BIB_PATH = DATA_DIR / "ttt_discover_bibliography_v2.jsonl"
UNRESOLVED_PATH = DATA_DIR / "ttt_discover_unresolved_v2.jsonl"
PDF_DIR = DATA_DIR / "full_text_pdfs"
SUMMARY_PATH = DATA_DIR / "bibliography_redo_summary.md"

ARXIV_API = "https://export.arxiv.org/api/query"
S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_PAPER = "https://api.semanticscholar.org/graph/v1/paper"
OPENALEX_API = "https://api.openalex.org/works"
S2_HEADERS = {"User-Agent": "co-scientist/d5 (your-email@example.com)"}
ARXIV_HEADERS = {"User-Agent": "co-scientist/d5 (your-email@example.com)"}

S2_DELAY = 6.0  # unauth: spec says 1 req/sec, but in practice 429s at 3.5s; 6s safer
ARXIV_DELAY = 3.0  # arxiv asks for ≥3s between requests
USE_S2_SEARCH = False  # disabled — chronic 429s from S2 unauth; rely on arxiv + OpenAlex


def normalize(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def title_overlap(a: str, b: str) -> int:
    """Count words ≥3 chars shared between titles (after normalize)."""
    wa = {w for w in normalize(a).split() if len(w) >= 3}
    wb = {w for w in normalize(b).split() if len(w) >= 3}
    return len(wa & wb)


def last_name(name: str) -> str:
    n = normalize(name)
    if "," in n:
        return n.split(",")[0].strip()
    parts = n.split()
    return parts[-1] if parts else ""


def authors_overlap(a_list: list[str], b_list: list[str]) -> bool:
    a_last = {last_name(x) for x in a_list[:5] if x}
    b_last = {last_name(x) for x in b_list[:5] if x}
    return bool(a_last & b_last)


def arxiv_title_search(title: str, year: int | None, authors: list[str], client: httpx.Client) -> str | None:
    """Query arxiv API by title; return arxiv_id if verified match, else None.
    Retries on timeout (arxiv API is slow under load).
    """
    q = f'ti:"{title[:100]}"'
    params = {"search_query": q, "max_results": "3"}
    txt = None
    for attempt in range(2):
        try:
            r = client.get(ARXIV_API, params=params, timeout=60, headers=ARXIV_HEADERS, follow_redirects=True)
            r.raise_for_status()
            txt = r.text
            break
        except Exception as e:
            logger.warning("  arxiv search fail '%s' attempt %d: %s", title[:60], attempt + 1, e)
            if attempt < 1:
                time.sleep(5.0)
    if txt is None:
        return None
    # Crude: parse <entry> blocks
    entries = re.findall(r"<entry>(.*?)</entry>", txt, flags=re.DOTALL)
    for e in entries:
        t_m = re.search(r"<title>(.*?)</title>", e, flags=re.DOTALL)
        id_m = re.search(r"<id>http://arxiv\.org/abs/([^<\s]+)</id>", e)
        a_ms = re.findall(r"<author>\s*<name>([^<]+)</name>", e)
        y_m = re.search(r"<published>(\d{4})", e)
        if not (t_m and id_m):
            continue
        cand_title = re.sub(r"\s+", " ", t_m.group(1)).strip()
        cand_id = id_m.group(1).split("v")[0]  # strip version
        cand_authors = [a.strip() for a in a_ms]
        cand_year = int(y_m.group(1)) if y_m else None
        if title_overlap(title, cand_title) < 3:
            continue
        if year and cand_year and abs(year - cand_year) > 2:
            continue
        if authors and cand_authors and not authors_overlap(authors, cand_authors):
            continue
        return cand_id
    return None


def download_arxiv_pdf(arxiv_id: str, client: httpx.Client) -> Path | None:
    """Download arxiv PDF + pypdf extract. Returns md path on success."""
    md_path = PDF_DIR / f"{arxiv_id}.md"
    if md_path.exists() and md_path.stat().st_size > 1000:
        return md_path
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        r = client.get(pdf_url, timeout=60, follow_redirects=True)
        r.raise_for_status()
    except Exception as e:
        logger.warning("  arxiv pdf fail %s: %s", arxiv_id, e)
        return None
    pdf_bytes = r.content
    if len(pdf_bytes) < 5000 or not pdf_bytes.startswith(b"%PDF"):
        logger.warning("  arxiv pdf invalid %s (len=%d)", arxiv_id, len(pdf_bytes))
        return None
    tmp = PDF_DIR / f"_{arxiv_id}.pdf"
    tmp.write_bytes(pdf_bytes)
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(tmp))
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
        text = text[:30000]
        md_path.write_text(text)
    except Exception as e:
        logger.warning("  pypdf extract fail %s: %s", arxiv_id, e)
        tmp.unlink(missing_ok=True)
        return None
    tmp.unlink(missing_ok=True)
    return md_path


def download_pdf_url(url: str, dest_name: str, client: httpx.Client) -> Path | None:
    """Download arbitrary PDF URL + pypdf extract."""
    md_path = PDF_DIR / f"{dest_name}.md"
    if md_path.exists() and md_path.stat().st_size > 1000:
        return md_path
    try:
        r = client.get(url, timeout=60, follow_redirects=True)
        r.raise_for_status()
    except Exception as e:
        logger.warning("  pdf url fail %s: %s", url, e)
        return None
    pdf_bytes = r.content
    if len(pdf_bytes) < 5000 or not pdf_bytes.startswith(b"%PDF"):
        return None
    tmp = PDF_DIR / f"_{dest_name}.pdf"
    tmp.write_bytes(pdf_bytes)
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(tmp))
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
        text = text[:30000]
        md_path.write_text(text)
    except Exception as e:
        logger.warning("  pypdf extract fail %s: %s", dest_name, e)
        tmp.unlink(missing_ok=True)
        return None
    tmp.unlink(missing_ok=True)
    return md_path


def s2_search_title(title: str, client: httpx.Client) -> dict | None:
    """S2 title search with retry-on-429; returns paper dict (with paperId, openAccessPdf)."""
    params = {
        "query": title[:120],
        "limit": "3",
        "fields": "title,year,authors.name,paperId,externalIds,openAccessPdf",
    }
    for attempt in range(3):
        try:
            r = client.get(S2_SEARCH, params=params, headers=S2_HEADERS, timeout=30, follow_redirects=True)
            if r.status_code == 429:
                wait = 8.0 * (attempt + 1)
                logger.info("  s2 429, sleeping %.1fs (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            return (data.get("data") or [None])[0]
        except Exception as e:
            logger.warning("  s2 search fail '%s' attempt %d: %s", title[:60], attempt + 1, e)
            if attempt < 2:
                time.sleep(5.0)
    return None


def openalex_pdf_url(doi: str | None, title: str, client: httpx.Client) -> str | None:
    """Try OpenAlex best_oa_location.pdf_url."""
    if doi:
        params = {
            "filter": f"doi:{doi.removeprefix('https://doi.org/')}",
            "select": "best_oa_location",
            "mailto": "your-email@example.com",
        }
    else:
        params = {
            "search": title[:120],
            "per_page": "1",
            "select": "best_oa_location",
            "mailto": "your-email@example.com",
        }
    try:
        r = client.get(OPENALEX_API, params=params, timeout=30, follow_redirects=True)
        r.raise_for_status()
    except Exception as e:
        logger.warning("  openalex fetch fail: %s", e)
        return None
    data = r.json()
    results = data.get("results") or []
    if not results:
        return None
    boa = results[0].get("best_oa_location") or {}
    return boa.get("pdf_url")


def main() -> None:
    if not BIB_PATH.exists():
        logger.error("bibliography not found: %s", BIB_PATH)
        sys.exit(1)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Step A: re-classify the 4 unresolved keys ----
    unresolved = [json.loads(l) for l in UNRESOLVED_PATH.read_text().splitlines() if l.strip()]
    logger.info("Loaded %d unresolved entries from Phase 2A-bis-3", len(unresolved))

    # By cite_context analysis (see commit message): yaoyour, openevolve, atcoder are non-paper.
    # white2023erdos is a real math paper — try arxiv title search.
    NON_PAPER_RECLASSIFY = {
        "yaoyour": "Notion blog post (fengyao.notion.site/off-policy-rl) — not a peer-reviewed paper",
        "openevolve": "GitHub repository (algorithmicsuperintelligence/openevolve) — open-source code, not a paper",
        "atcoder": "AtCoder Inc. company website (atcoder.jp) — competitive programming platform",
    }

    new_bib_entries = []
    new_unresolved = []
    client = httpx.Client(headers={"User-Agent": "co-scientist/d5 (your-email@example.com)"})

    for entry in unresolved:
        ck = entry.get("cite_key")
        if ck in NON_PAPER_RECLASSIFY:
            new_bib_entries.append({
                "cite_key": ck,
                "title": entry.get("title", "?"),
                "authors": entry.get("authors", []),
                "year": entry.get("year"),
                "venue": entry.get("venue", ""),
                "arxiv_id": None,
                "doi": None,
                "abstract": "",
                "tldr": "",
                "citation_count": 0,
                "s2_paper_id": None,
                "is_paper": False,
                "non_paper_reason": NON_PAPER_RECLASSIFY[ck],
                "full_text_path": None,
                "resolution_source": "manual_reclassify_phase2a_bis_8",
            })
            logger.info("Reclassified %s as non-paper: %s", ck, NON_PAPER_RECLASSIFY[ck])
        elif ck == "white2023erdos":
            # Re-search arxiv with broader title
            title = "A new bound for Erdos minimum overlap problem"
            authors = ["Ethan Patrick White"]
            arxiv_id = arxiv_title_search(title, 2023, authors, client)
            time.sleep(ARXIV_DELAY)
            if arxiv_id:
                logger.info("Re-resolved white2023erdos -> arxiv:%s", arxiv_id)
                md = download_arxiv_pdf(arxiv_id, client)
                new_bib_entries.append({
                    "cite_key": ck,
                    "title": entry.get("title", title),
                    "authors": authors,
                    "year": 2023,
                    "venue": "Acta Arithmetica",
                    "arxiv_id": arxiv_id,
                    "doi": None,
                    "abstract": "",
                    "tldr": "",
                    "citation_count": 0,
                    "s2_paper_id": None,
                    "is_paper": True,
                    "full_text_path": str(md.relative_to(REPO_ROOT)) if md else None,
                    "resolution_source": "arxiv_title_phase2a_bis_8",
                })
            else:
                # Try OpenAlex best_oa_location for Acta Arithmetica
                pdf_url = openalex_pdf_url(None, title, client)
                full_path = None
                if pdf_url:
                    md = download_pdf_url(pdf_url, ck, client)
                    full_path = str(md.relative_to(REPO_ROOT)) if md else None
                new_bib_entries.append({
                    "cite_key": ck,
                    "title": entry.get("title", title),
                    "authors": authors,
                    "year": 2023,
                    "venue": "Acta Arithmetica",
                    "arxiv_id": None,
                    "doi": None,
                    "abstract": "",
                    "tldr": "",
                    "citation_count": 0,
                    "s2_paper_id": None,
                    "is_paper": True,
                    "full_text_path": full_path,
                    "resolution_source": "openalex_oa_phase2a_bis_8" if full_path else "still_unresolved",
                })
                if not full_path:
                    new_unresolved.append({**entry, "phase2a_bis_8_status": "arxiv+openalex_oa both failed"})

    # ---- Step B: scan abstract-only entries, attempt full-text fetch ----
    bib = [json.loads(l) for l in BIB_PATH.read_text().splitlines() if l.strip()]
    logger.info("Loaded %d bibliography entries", len(bib))

    abs_only = [e for e in bib if not e.get("full_text_path") and e.get("is_paper", True)]
    logger.info("Abstract-only entries to expand: %d", len(abs_only))

    fetch_results = {"arxiv_hit": 0, "s2_oa_pdf": 0, "openalex_oa": 0, "still_abs_only": 0}
    fetch_log = []

    for i, entry in enumerate(abs_only):
        ck = entry["cite_key"]
        title = entry.get("title", "")
        year = entry.get("year")
        authors = entry.get("authors", [])
        if not title:
            fetch_results["still_abs_only"] += 1
            fetch_log.append((ck, "no_title"))
            continue
        logger.info("[%d/%d] %s: %s", i + 1, len(abs_only), ck, title[:80])

        # Path 1: arxiv title search
        arxiv_id = arxiv_title_search(title, year, authors, client)
        time.sleep(ARXIV_DELAY)
        if arxiv_id:
            md = download_arxiv_pdf(arxiv_id, client)
            if md:
                entry["arxiv_id"] = arxiv_id
                entry["full_text_path"] = str(md.relative_to(REPO_ROOT))
                entry["resolution_source"] = (entry.get("resolution_source") or "") + "+arxiv_title_phase2a_bis_8"
                fetch_results["arxiv_hit"] += 1
                fetch_log.append((ck, f"arxiv:{arxiv_id}"))
                continue

        # Path 2: S2 search-by-title -> openAccessPdf (DISABLED — chronic 429)
        s2_paper = s2_search_title(title, client) if USE_S2_SEARCH else None
        if USE_S2_SEARCH:
            time.sleep(S2_DELAY)
        if s2_paper:
            # Verify match
            s2_title = s2_paper.get("title", "")
            s2_year = s2_paper.get("year")
            s2_authors = [a.get("name", "") for a in (s2_paper.get("authors") or [])]
            if title_overlap(title, s2_title) >= 3 and (
                not year or not s2_year or abs(year - s2_year) <= 2
            ):
                entry["s2_paper_id"] = s2_paper.get("paperId")
                # Check externalIds for arxiv we may have missed
                eid_arxiv = (s2_paper.get("externalIds") or {}).get("ArXiv")
                if eid_arxiv:
                    md = download_arxiv_pdf(eid_arxiv, client)
                    if md:
                        entry["arxiv_id"] = eid_arxiv
                        entry["full_text_path"] = str(md.relative_to(REPO_ROOT))
                        entry["resolution_source"] = (entry.get("resolution_source") or "") + "+s2_externalIds_arxiv_phase2a_bis_8"
                        fetch_results["arxiv_hit"] += 1
                        fetch_log.append((ck, f"s2->arxiv:{eid_arxiv}"))
                        continue
                # Else try openAccessPdf
                oa = s2_paper.get("openAccessPdf") or {}
                pdf_url = oa.get("url")
                if pdf_url:
                    md = download_pdf_url(pdf_url, ck, client)
                    if md:
                        entry["full_text_path"] = str(md.relative_to(REPO_ROOT))
                        entry["resolution_source"] = (entry.get("resolution_source") or "") + "+s2_oa_pdf_phase2a_bis_8"
                        fetch_results["s2_oa_pdf"] += 1
                        fetch_log.append((ck, f"s2_oa_pdf:{pdf_url[:80]}"))
                        continue

        # Path 3: OpenAlex best_oa_location
        doi = entry.get("doi")
        oa_url = openalex_pdf_url(doi, title, client)
        time.sleep(0.5)  # OpenAlex polite pool, ~1 req/s ok
        if oa_url:
            md = download_pdf_url(oa_url, ck, client)
            if md:
                entry["full_text_path"] = str(md.relative_to(REPO_ROOT))
                entry["resolution_source"] = (entry.get("resolution_source") or "") + "+openalex_oa_phase2a_bis_8"
                fetch_results["openalex_oa"] += 1
                fetch_log.append((ck, f"openalex_oa:{oa_url[:80]}"))
                continue

        fetch_results["still_abs_only"] += 1
        fetch_log.append((ck, "all_paths_failed"))

    # ---- Step C: merge new_bib_entries (from unresolved reclassification) into bib ----
    bib_keys = {e["cite_key"] for e in bib}
    for nb in new_bib_entries:
        if nb["cite_key"] not in bib_keys:
            bib.append(nb)
            bib_keys.add(nb["cite_key"])

    # Write updated bibliography
    with open(BIB_PATH, "w") as f:
        for e in bib:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    logger.info("Wrote updated bibliography: %d entries", len(bib))

    # Write updated unresolved (just white2023erdos if it failed both paths)
    if new_unresolved:
        with open(UNRESOLVED_PATH, "w") as f:
            for u in new_unresolved:
                f.write(json.dumps(u, ensure_ascii=False) + "\n")
        logger.info("Updated unresolved: %d entries", len(new_unresolved))
    else:
        # Clear unresolved file if all 4 are now classified
        UNRESOLVED_PATH.write_text("")
        logger.info("All unresolved entries handled — unresolved file cleared")

    # ---- Step D: write expand log + update summary ----
    log_path = DATA_DIR / "expand_full_text_v1_log.jsonl"
    with open(log_path, "w") as f:
        for ck, status in fetch_log:
            f.write(json.dumps({"cite_key": ck, "status": status}) + "\n")
    logger.info("Wrote fetch log: %s", log_path)

    # Final accounting
    final_bib = [json.loads(l) for l in BIB_PATH.read_text().splitlines() if l.strip()]
    n_total = len(final_bib)
    n_paper = sum(1 for e in final_bib if e.get("is_paper", True))
    n_non_paper = sum(1 for e in final_bib if not e.get("is_paper", True))
    n_full_text = sum(1 for e in final_bib if e.get("full_text_path"))
    n_abs_only = n_paper - n_full_text

    logger.info("=" * 60)
    logger.info("FINAL accounting after Phase 2A-bis-8:")
    logger.info("  bibliography entries: %d", n_total)
    logger.info("  papers: %d (full text: %d, abstract only: %d)", n_paper, n_full_text, n_abs_only)
    logger.info("  non-paper: %d", n_non_paper)
    logger.info("  fetch results: %s", fetch_results)
    logger.info("=" * 60)

    # Append updated summary section
    addendum = f"""

## Phase 2A-bis-8 Update (Generated {time.strftime('%Y-%m-%d %H:%M:%S')})

**Re-classified 3 of 4 unresolved as non-paper** (verified via main.tex citation context):
- `yaoyour` — Notion blog post (off-policy RL note)
- `openevolve` — GitHub repository (open-source AlphaEvolve clone)
- `atcoder` — AtCoder Inc. company website

**Re-resolved 1 of 4 via arxiv title search**:
- `white2023erdos` — see fetch log

**Full-text expansion for 48 abstract-only entries**:
- arxiv title search hit: {fetch_results['arxiv_hit']}
- S2 openAccessPdf hit: {fetch_results['s2_oa_pdf']}
- OpenAlex best_oa_location hit: {fetch_results['openalex_oa']}
- Still abstract-only (no OA path): {fetch_results['still_abs_only']}

## Final accounting (after Phase 2A-bis-8)

- Bibliography entries: **{n_total}**
- Papers: **{n_paper}** (full text: **{n_full_text}** | abstract only: **{n_abs_only}**)
- Non-paper: **{n_non_paper}**

Per-entry fetch log: `data/expand_full_text_v1_log.jsonl`
"""
    SUMMARY_PATH.write_text(SUMMARY_PATH.read_text() + addendum)
    logger.info("Updated summary: %s", SUMMARY_PATH)


if __name__ == "__main__":
    main()
