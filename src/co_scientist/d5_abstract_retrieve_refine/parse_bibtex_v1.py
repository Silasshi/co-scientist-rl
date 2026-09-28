"""Phase 2A-bis-2: Parse main.bib + cross-reference cite keys to get clean metadata.

Inputs:
- /mnt/d/AI/Co-scientist/reference/arXiv-2601.16175v2/main.bib (BibTeX, 416 entries)
- data/bibliography/cite_keys_v1.json (90 cite keys actually used)

Output:
- data/bibliography/cited_bibtex_v1.jsonl (one entry per cite key with structured metadata)

Each output entry:
  {
    "cite_key": "guo2025deepseek",
    "entry_type": "article",
    "title": "Deepseek-r1 incentivizes reasoning in llms through reinforcement learning",
    "authors": ["Daya Guo", "Dejian Yang", ...],
    "year": 2025,
    "venue": "Nature",
    "arxiv_id": "2501.12948",  // parsed from journal field "arXiv preprint arXiv:XXXX.XXXXX" or url
    "doi": "...",  // parsed from doi field if present
    "url": "...",
    "_raw_bibtex": "<original bibtex entry text>"
  }
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


@chz.chz
class Config:
    bibtex_path: str = "/mnt/d/AI/Co-scientist/reference/arXiv-2601.16175v2/main.bib"
    keys_path: str = "projects/d5_abstract_retrieve_refine/data/bibliography/cite_keys_v1.json"
    out_path: str = "projects/d5_abstract_retrieve_refine/data/bibliography/cited_bibtex_v1.jsonl"
    missing_keys_path: str = ""  # if empty, derived as out_path sibling

# Match: @article{key, ... } / @inproceedings{key, ... } / @misc{key, ... } / etc.
# Track brace depth to find end of entry.

ARXIV_ID_RE = re.compile(r"arxiv[:\s]*(\d{4}\.\d{4,5})", re.IGNORECASE)
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\}\"]+)", re.IGNORECASE)


def parse_bibtex_entries(text: str) -> dict[str, dict]:
    """Parse all @entry{...} from BibTeX text, return {cite_key: parsed_dict}."""
    entries = {}
    i = 0
    n = len(text)
    while i < n:
        # Find next @
        at = text.find("@", i)
        if at == -1:
            break
        # Find entry type (@article, @inproceedings, @misc, etc.)
        type_m = re.match(r"@(\w+)\s*\{", text[at:])
        if not type_m:
            i = at + 1
            continue
        entry_type = type_m.group(1).lower()
        if entry_type in ("comment", "string", "preamble"):
            # Skip BibTeX directives
            i = at + len(type_m.group(0))
            continue
        # Find matching closing brace (track depth)
        body_start = at + len(type_m.group(0))
        depth = 1
        j = body_start
        while j < n and depth > 0:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            j += 1
        body = text[body_start : j - 1]  # exclude the final }

        # First token before first comma = cite_key
        comma_idx = body.find(",")
        if comma_idx == -1:
            i = j
            continue
        cite_key = body[:comma_idx].strip()
        fields_text = body[comma_idx + 1 :]

        # Parse fields: name = {value} or name = "value"
        parsed = {"cite_key": cite_key, "entry_type": entry_type}
        # Use simple regex for field = {...} extraction
        for fm in re.finditer(r"(\w+)\s*=\s*([{\"])", fields_text):
            name = fm.group(1).lower()
            opener = fm.group(2)
            closer = "}" if opener == "{" else '"'
            # Track depth for { ... }
            start = fm.end()
            d = 1
            k = start
            while k < len(fields_text) and d > 0:
                ch = fields_text[k]
                if opener == "{":
                    if ch == "{":
                        d += 1
                    elif ch == "}":
                        d -= 1
                elif ch == closer:
                    d = 0
                k += 1
            value = fields_text[start : k - 1]
            # Strip whitespace + LaTeX accents simple
            value = re.sub(r"\s+", " ", value).strip()
            parsed[name] = value

        entries[cite_key] = parsed
        i = j
    return entries


def parse_authors_field(authors_str: str) -> list[str]:
    """BibTeX 'A and B and C' → list. Strip LaTeX accent commands."""
    if not authors_str:
        return []
    # Handle "Aky{\"u}rek" -> "Akyürek" (basic)
    s = re.sub(r"\{\\([\"'`^~])([A-Za-z])\}", r"\2", authors_str)
    s = re.sub(r"\\([\"'`^~])\{?([A-Za-z])\}?", r"\2", s)
    s = re.sub(r"[\{\}]", "", s)
    parts = re.split(r"\s+and\s+", s)
    return [p.strip() for p in parts if p.strip()]


def extract_arxiv_id(entry: dict) -> str | None:
    """Look for arxiv id in journal/url/eprint fields."""
    for field in ("journal", "url", "eprint", "note", "howpublished"):
        v = entry.get(field, "")
        m = ARXIV_ID_RE.search(v)
        if m:
            return m.group(1)
    return None


def extract_doi(entry: dict) -> str | None:
    if "doi" in entry and entry["doi"].strip():
        return entry["doi"].strip()
    for field in ("url", "note", "journal"):
        v = entry.get(field, "")
        m = DOI_RE.search(v)
        if m:
            return m.group(1)
    return None


def main(config: Config) -> None:
    bibtex = Path(config.bibtex_path)
    if not bibtex.is_absolute():
        bibtex = REPO_ROOT / bibtex
    keys_path = Path(config.keys_path)
    if not keys_path.is_absolute():
        keys_path = REPO_ROOT / keys_path
    out_path = Path(config.out_path)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    if config.missing_keys_path:
        miss_path = Path(config.missing_keys_path)
        if not miss_path.is_absolute():
            miss_path = REPO_ROOT / miss_path
    else:
        miss_path = out_path.with_name("ttt_discover_missing_in_bibtex.json")

    if not bibtex.exists():
        logger.error("main.bib not found: %s", bibtex)
        sys.exit(1)
    if not keys_path.exists():
        logger.error("cite keys file not found: %s — run extract_cite_keys_v1 first", keys_path)
        sys.exit(1)

    bib_text = bibtex.read_text()
    keys_data = json.loads(keys_path.read_text())
    cite_keys = keys_data["keys"]

    logger.info("Parsing %d-byte main.bib...", len(bib_text))
    entries = parse_bibtex_entries(bib_text)
    logger.info("Parsed %d total BibTeX entries from main.bib", len(entries))

    logger.info("Looking up %d cite keys...", len(cite_keys))
    out_entries = []
    missing_keys = []
    for k in cite_keys:
        if k not in entries:
            missing_keys.append(k)
            continue
        e = entries[k]
        arxiv_id = extract_arxiv_id(e)
        doi = extract_doi(e)
        out_entries.append({
            "cite_key": k,
            "entry_type": e.get("entry_type"),
            "title": e.get("title", "").strip(),
            "authors": parse_authors_field(e.get("author", "")),
            "year": int(e["year"]) if e.get("year", "").isdigit() else None,
            "venue": (e.get("journal") or e.get("booktitle") or e.get("howpublished") or "").strip(),
            "arxiv_id": arxiv_id,
            "doi": doi,
            "url": e.get("url", "").strip(),
            "publisher": e.get("publisher", "").strip(),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for o in out_entries:
            f.write(json.dumps(o) + "\n")

    n_with_arxiv = sum(1 for o in out_entries if o["arxiv_id"])
    n_with_doi = sum(1 for o in out_entries if o["doi"])
    logger.info(
        "Resolved %d/%d cite keys to BibTeX entries (%d missing). With arxiv_id: %d. With doi: %d.",
        len(out_entries), len(cite_keys), len(missing_keys), n_with_arxiv, n_with_doi,
    )
    logger.info("Output: %s", out_path)
    if missing_keys:
        logger.warning("Cite keys NOT found in main.bib:")
        for k in missing_keys:
            logger.warning("  %s", k)
        miss_path.write_text(json.dumps({"missing_cite_keys": missing_keys}, indent=2))
        logger.info("Missing keys saved to %s", miss_path)


if __name__ == "__main__":
    main(chz.entrypoint(Config))
