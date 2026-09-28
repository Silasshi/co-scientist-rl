"""One-shot script: extract TTT-Discover paper PDF to markdown for D5 μ baseline.

The output `source_paper_ttt_discover.md` is the privileged-info source for
the plan critic in `train_mu_baseline_v1.py`. Run once, commit the output.

Usage:
    python -m co_scientist.d5_abstract_retrieve_refine.extract_paper_v1
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pypdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
PDF_PATH = REPO_ROOT / "shared/papers/by_topic/01_test_time_search/TTT-Discover/paper.pdf"
OUT_PATH = REPO_ROOT / "projects/d5_abstract_retrieve_refine/data/source_paper/v1.md"

# Common arxiv-paper section headings — case-insensitive, must start a line
# after stripping leading numbering/bullets. Order matters only for fallback;
# regex matches anywhere line begins with these.
SECTION_NAMES = [
    "abstract", "introduction", "related work", "background", "preliminaries",
    "method", "methods", "methodology", "approach", "model", "framework",
    "experiments", "experimental setup", "experimental results", "evaluation",
    "results", "discussion", "analysis", "ablation", "ablations",
    "limitations", "conclusion", "conclusions", "future work",
    "acknowledgments", "acknowledgements", "references", "appendix",
]
# Match: optional numbering ("1", "1.", "1.1", "A", "A.1") + section name + EOL
SECTION_RE = re.compile(
    r"^\s*(?:[A-Z]?\.?\d+(?:\.\d+)?\.?\s+)?(" + "|".join(SECTION_NAMES) + r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_text(pdf_path: Path) -> str:
    reader = pypdf.PdfReader(str(pdf_path))
    pages_text: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.warning("Page %d extraction error: %s", i, e)
            text = ""
        pages_text.append(text)
    full = "\n\n".join(pages_text)
    return full


def insert_section_headings(text: str) -> str:
    """Convert recognized section-heading lines to '## Heading' markdown."""
    def repl(m: re.Match) -> str:
        name = m.group(1).strip().title()
        return f"\n## {name}\n"
    return SECTION_RE.sub(repl, text)


def normalize_whitespace(text: str) -> str:
    # Collapse repeated blank lines, strip trailing spaces.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main() -> None:
    assert PDF_PATH.exists(), f"PDF not found: {PDF_PATH}"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Reading %s (%.2f MB)", PDF_PATH, PDF_PATH.stat().st_size / 1e6)

    raw = extract_text(PDF_PATH)
    logger.info("Raw text: %d chars across %d pages", len(raw), raw.count("\f") + 1)

    headed = insert_section_headings(raw)
    cleaned = normalize_whitespace(headed)
    n_sections = cleaned.count("\n## ")
    logger.info("Inserted %d section headings", n_sections)

    header = (
        "# TTT-Discover (Yuksekgonul et al., arxiv 2601.16175)\n\n"
        "*Extracted from `shared/papers/by_topic/01_test_time_search/TTT-Discover/paper.pdf` "
        "by `extract_paper_v1.py` for D5 μ baseline plan critic privileged info. "
        "DO NOT show this file to the audit subagent.*\n\n"
    )
    OUT_PATH.write_text(header + cleaned)
    logger.info("Wrote %d chars to %s", len(cleaned) + len(header), OUT_PATH)


if __name__ == "__main__":
    main()
