"""Phase 2A-bis-1: Parse main.tex to extract every \\cite{} key actually used.

Reads `/mnt/d/AI/Co-scientist/reference/arXiv-2601.16175v2/main.tex` and outputs
`projects/d5_abstract_retrieve_refine/data/bibliography/cite_keys_v1.json` —
deduplicated list of BibTeX keys cited in the paper.

Handles all variants: \\cite, \\citep, \\citet, \\citealp, \\citeauthor, \\citeyear,
\\citeyearpar — with optional [opt] argument and comma-separated key lists.
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
    latex_path: str = "/mnt/d/AI/Co-scientist/reference/arXiv-2601.16175v2/main.tex"
    out_path: str = "projects/d5_abstract_retrieve_refine/data/bibliography/cite_keys_v1.json"

# Match \cite, \citep, \citet, \citealp, \citeauthor, \citeyear, \citeyearpar etc.
# with optional [arg] and {key1,key2,...}
CITE_RE = re.compile(
    r"\\cite(?:p|t|alp|author|year|yearpar|alt|num)?\*?(?:\[[^\]]*\])?(?:\[[^\]]*\])?\s*\{([^}]+)\}",
    re.IGNORECASE,
)


def main(config: Config) -> None:
    latex_tex = Path(config.latex_path)
    if not latex_tex.is_absolute():
        latex_tex = REPO_ROOT / latex_tex
    out_path = Path(config.out_path)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    if not latex_tex.exists():
        logger.error("LaTeX source not found: %s", latex_tex)
        sys.exit(1)
    text = latex_tex.read_text()
    logger.info("Read main.tex: %d chars", len(text))

    keys_seen: list[str] = []
    seen_set: set[str] = set()
    n_cite_calls = 0
    for m in CITE_RE.finditer(text):
        n_cite_calls += 1
        keys_str = m.group(1)
        for k in keys_str.split(","):
            k = k.strip()
            if not k:
                continue
            if k in seen_set:
                continue
            seen_set.add(k)
            keys_seen.append(k)

    logger.info(
        "Found %d \\cite-style calls, %d unique keys",
        n_cite_calls, len(keys_seen),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n_cite_calls": n_cite_calls,
        "n_unique_keys": len(keys_seen),
        "keys": keys_seen,
    }, indent=2))
    logger.info("Wrote %s", out_path)
    print("\nFirst 30 keys:")
    for k in keys_seen[:30]:
        print(f"  {k}")
    if len(keys_seen) > 30:
        print(f"  ... and {len(keys_seen) - 30} more")


if __name__ == "__main__":
    main(chz.entrypoint(Config))
