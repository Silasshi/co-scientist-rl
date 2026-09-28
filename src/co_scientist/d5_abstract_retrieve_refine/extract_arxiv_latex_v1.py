"""Phase 2A-bis-9: Re-extract all arxiv-bearing bibliography papers from LaTeX source.

Replaces pypdf-extracted full_text_pdfs/{arxiv_id}.md (PDF artifacts: hyphenation,
mangled equations, special chars) with LaTeX-source-extracted full_text_latex/{arxiv_id}.md
(clean prose + equations preserved, like source_paper_ttt_discover_v2.md).

For each arxiv_id in bibliography_v2.jsonl with full_text_path:
1. Download e-print tarball from arxiv.org/e-print/{arxiv_id}
2. Extract — supports .tar.gz / .tar / single .tex / .gz wrap
3. Find main .tex (largest with \\documentclass + \\begin{document})
4. Recursively inline \\input{} and \\include{} references
5. Run latex_to_markdown() (reused from extract_source_paper_v2)
6. Save to data/bibliography/full_text_latex/{arxiv_id}.md (truncated to 30K chars)
7. Update bibliography_v2 entry's `full_text_path` to point to new file

Some arxiv papers are NOT LaTeX (older scanned papers, slide decks). Falls back to
keeping pypdf-extracted version with a warning.
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import re
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import chz
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.d5_abstract_retrieve_refine.extract_source_paper_v2 import latex_to_markdown

ARXIV_EPRINT = "https://arxiv.org/e-print/{id}"
ARXIV_DELAY = 2.0  # arxiv asks for ≥3s between e-print requests in bulk; be polite
CHAR_LIMIT = 30000


@chz.chz
class Config:
    bib_path: str = "projects/d5_abstract_retrieve_refine/data/ttt_discover_bibliography_v2.jsonl"
    latex_dir: str = "projects/d5_abstract_retrieve_refine/data/full_text_latex"
    summary_path: str = "projects/d5_abstract_retrieve_refine/data/bibliography_redo_summary.md"
    log_path: str = ""  # if empty, derived as latex_dir parent / extract_arxiv_latex_v1_log.jsonl

INPUT_RE = re.compile(r"\\(?:input|include|subfile)\s*\{([^}]+)\}")

# Capture: \newcommand{\name}{body}  /  \def\name{body}  /  \DeclareRobustCommand{\name}{body}
# Only zero-argument macros (\newcommand{\foo}[N]{...} with N>0 is skipped)
NEWCMD_RE = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand|DeclareRobustCommand)\*?"
    r"\s*\{?\\([A-Za-z]+)\}?\s*(?!\[)"  # name, NOT followed by [args]
    r"\s*\{((?:[^{}]|\{[^{}]*\})*)\}",
)
DEF_RE = re.compile(r"\\def\s*\\([A-Za-z]+)\s*\{((?:[^{}]|\{[^{}]*\})*)\}")


def resolve_macros(text: str, max_iter: int = 4) -> str:
    """Inline simple zero-arg LaTeX macros (newcommand/def) defined in the document.

    Returns text with macro calls replaced by their bodies. Iterates to handle
    nested macros (a macro body may reference another macro). Macros with arguments
    are NOT resolved (we leave them as-is; latex_to_markdown will strip what it can).
    """
    macros: dict[str, str] = {}
    for m in NEWCMD_RE.finditer(text):
        name, body = m.group(1), m.group(2)
        macros[name] = body
    for m in DEF_RE.finditer(text):
        name, body = m.group(1), m.group(2)
        macros.setdefault(name, body)

    if not macros:
        return text

    # Strip the macro definitions from the body so they don't leak through
    text = NEWCMD_RE.sub("", text)
    text = DEF_RE.sub("", text)

    for _ in range(max_iter):
        changed = False
        for name, body in macros.items():
            # \name{} or \name\xspace or \name followed by non-letter
            pat = re.compile(rf"\\{re.escape(name)}(\{{}}|\\xspace\b|(?=[^A-Za-z]))")
            # Use lambda to make replacement literal (body may contain \backreferences)
            new_text, n = pat.subn(lambda m, b=body: b, text)
            if n > 0:
                changed = True
                text = new_text
        if not changed:
            break
    return text


def find_main_tex(work_dir: Path) -> Path | None:
    """Find the entry-point .tex file in extracted dir.
    Heuristic: file containing both \\documentclass and \\begin{document}.
    Tie-break: largest such file.
    """
    candidates = []
    for tex in work_dir.rglob("*.tex"):
        try:
            text = tex.read_text(errors="replace")
        except Exception:
            continue
        if "\\documentclass" in text and "\\begin{document}" in text:
            candidates.append((tex, tex.stat().st_size))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p[1], reverse=True)
    return candidates[0][0]


def inline_inputs(tex_path: Path, root: Path, depth: int = 0, max_depth: int = 6) -> str:
    """Recursively inline \\input{X} and \\include{X} references."""
    if depth > max_depth:
        logger.warning("    inline depth exceeded at %s", tex_path)
        return ""
    try:
        text = tex_path.read_text(errors="replace")
    except Exception as e:
        logger.warning("    cannot read %s: %s", tex_path, e)
        return ""

    def replace(m: re.Match) -> str:
        name = m.group(1).strip()
        for ext in ("", ".tex"):
            cand = (tex_path.parent / f"{name}{ext}")
            if cand.exists() and cand.is_file():
                return inline_inputs(cand, root, depth + 1, max_depth)
            cand2 = root / f"{name}{ext}"
            if cand2.exists() and cand2.is_file():
                return inline_inputs(cand2, root, depth + 1, max_depth)
        return ""  # Missing input — drop quietly

    return INPUT_RE.sub(replace, text)


def extract_tarball(arxiv_id: str, raw: bytes, work_dir: Path) -> bool:
    """Extract arxiv e-print bytes (tar.gz / tar / single .gz / single .tex)."""
    work_dir.mkdir(parents=True, exist_ok=True)

    # Try as gzipped tar
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
            tf.extractall(work_dir)
        return True
    except (tarfile.ReadError, OSError):
        pass

    # Try as plain tar
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tf:
            tf.extractall(work_dir)
        return True
    except (tarfile.ReadError, OSError):
        pass

    # Try as single gzipped .tex
    try:
        decompressed = gzip.decompress(raw)
        if b"\\documentclass" in decompressed[:5000]:
            (work_dir / f"{arxiv_id}.tex").write_bytes(decompressed)
            return True
    except (OSError, gzip.BadGzipFile):
        pass

    # Try as plain .tex
    if b"\\documentclass" in raw[:5000]:
        (work_dir / f"{arxiv_id}.tex").write_bytes(raw)
        return True

    return False


def fetch_and_convert(arxiv_id: str, client: httpx.Client, latex_dir: Path) -> Path | None:
    """Download e-print, extract, find main, inline, convert. Returns md path or None."""
    latex_dir.mkdir(parents=True, exist_ok=True)
    out_path = latex_dir / f"{arxiv_id}.md"
    if out_path.exists() and out_path.stat().st_size > 1000:
        logger.info("  [cached] %s -> %s", arxiv_id, out_path.relative_to(REPO_ROOT))
        return out_path

    url = ARXIV_EPRINT.format(id=arxiv_id)
    try:
        r = client.get(url, timeout=120, follow_redirects=True)
        r.raise_for_status()
    except Exception as e:
        logger.warning("  [%s] e-print fetch fail: %s", arxiv_id, e)
        return None

    raw = r.content
    if len(raw) < 1000:
        logger.warning("  [%s] tarball too small (%d bytes); likely no LaTeX source available", arxiv_id, len(raw))
        return None

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        if not extract_tarball(arxiv_id, raw, work_dir):
            logger.warning("  [%s] extraction failed (not tar/gz/tex)", arxiv_id)
            return None

        main_tex = find_main_tex(work_dir)
        if not main_tex:
            logger.warning("  [%s] no main .tex with \\documentclass + \\begin{document}", arxiv_id)
            return None

        try:
            inlined = inline_inputs(main_tex, work_dir)
        except Exception as e:
            logger.warning("  [%s] inline_inputs fail: %s", arxiv_id, e)
            return None

        try:
            inlined = resolve_macros(inlined)
        except Exception as e:
            logger.warning("  [%s] resolve_macros fail: %s", arxiv_id, e)

        try:
            md = latex_to_markdown(inlined)
        except Exception as e:
            logger.warning("  [%s] latex_to_markdown fail: %s", arxiv_id, e)
            return None

        if len(md) < 500:
            logger.warning("  [%s] converted markdown too short (%d chars)", arxiv_id, len(md))
            return None

        if len(md) > CHAR_LIMIT:
            md = md[:CHAR_LIMIT] + f"\n\n[... truncated at {CHAR_LIMIT} chars]"

        header = (
            f"# arxiv:{arxiv_id} (LaTeX source extraction)\n\n"
            f"*Extracted from arxiv e-print tarball via extract_arxiv_latex_v1.py "
            f"(replaces pypdf version). Inlined {len(inlined)} chars LaTeX → "
            f"{len(md)} chars markdown.*\n\n---\n\n"
        )
        out_path.write_text(header + md)
        logger.info("  [%s] OK -> %s (%d chars)", arxiv_id, out_path.relative_to(REPO_ROOT), len(md))
        return out_path


def main(config: Config) -> None:
    bib_path = Path(config.bib_path)
    if not bib_path.is_absolute():
        bib_path = REPO_ROOT / bib_path
    latex_dir = Path(config.latex_dir)
    if not latex_dir.is_absolute():
        latex_dir = REPO_ROOT / latex_dir
    summary_path = Path(config.summary_path)
    if not summary_path.is_absolute():
        summary_path = REPO_ROOT / summary_path
    if config.log_path:
        log_path = Path(config.log_path)
        if not log_path.is_absolute():
            log_path = REPO_ROOT / log_path
    else:
        log_path = latex_dir.parent / "extract_arxiv_latex_v1_log.jsonl"

    latex_dir.mkdir(parents=True, exist_ok=True)

    bib = [json.loads(l) for l in bib_path.read_text().splitlines() if l.strip()]
    arxiv_entries = [e for e in bib if e.get("arxiv_id")]
    logger.info("Found %d entries with arxiv_id (out of %d total)", len(arxiv_entries), len(bib))

    client = httpx.Client(headers={"User-Agent": "co-scientist/d5 (your-email@example.com)"})

    results = {"latex_ok": 0, "latex_fail_keep_pdf": 0, "latex_fail_no_pdf": 0}
    fail_log = []
    for i, entry in enumerate(arxiv_entries):
        ck = entry["cite_key"]
        aid = entry["arxiv_id"]
        logger.info("[%d/%d] %s arxiv:%s", i + 1, len(arxiv_entries), ck, aid)

        out_md = fetch_and_convert(aid, client, latex_dir)
        time.sleep(ARXIV_DELAY)

        if out_md:
            entry["full_text_path"] = str(out_md.relative_to(REPO_ROOT))
            entry["full_text_source"] = "latex_eprint"
            results["latex_ok"] += 1
        else:
            existing = entry.get("full_text_path", "")
            if existing and (REPO_ROOT / existing).exists():
                entry["full_text_source"] = "pypdf_fallback"
                results["latex_fail_keep_pdf"] += 1
                fail_log.append((ck, aid, "kept_pypdf"))
            else:
                results["latex_fail_no_pdf"] += 1
                fail_log.append((ck, aid, "no_full_text"))

    with open(bib_path, "w") as f:
        for e in bib:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    logger.info("Updated bibliography: %d entries", len(bib))

    with open(log_path, "w") as f:
        for ck, aid, status in fail_log:
            f.write(json.dumps({"cite_key": ck, "arxiv_id": aid, "status": status}) + "\n")
    logger.info("Wrote fail log: %s", log_path)

    final_bib = [json.loads(l) for l in bib_path.read_text().splitlines() if l.strip()]
    n_full_text_latex = sum(1 for e in final_bib if e.get("full_text_source") == "latex_eprint")
    n_full_text_pdf = sum(1 for e in final_bib if e.get("full_text_source") == "pypdf_fallback")
    n_no_full = sum(1 for e in final_bib if e.get("is_paper", True) and not e.get("full_text_path"))

    logger.info("=" * 60)
    logger.info("FINAL accounting:")
    logger.info("  LaTeX-extracted full text (high quality): %d", n_full_text_latex)
    logger.info("  pypdf fallback (LaTeX unavailable):       %d", n_full_text_pdf)
    logger.info("  abstract-only:                            %d", n_no_full)
    logger.info("=" * 60)

    addendum = f"""

## extract_arxiv_latex_v1 Update (Generated {time.strftime('%Y-%m-%d %H:%M:%S')})

LaTeX-source re-extraction for all arxiv-bearing entries:
- LaTeX-extracted full text: **{n_full_text_latex}**
- pypdf fallback: **{n_full_text_pdf}**
- abstract-only: **{n_no_full}**

bibliography entries now have `full_text_source` field: `"latex_eprint"` or `"pypdf_fallback"`.
"""
    if summary_path.exists():
        summary_path.write_text(summary_path.read_text() + addendum)
    else:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(addendum)
    logger.info("Updated summary: %s", summary_path)


if __name__ == "__main__":
    main(chz.entrypoint(Config))
