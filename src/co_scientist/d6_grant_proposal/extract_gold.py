"""D6 gold file builder v1 — extract equations and citations from reference_proposal.md.

One-time per-goal script. Writes data/{domain}/{goal}/gold/gold.json
for use by train_opd_grounded.py.

Output format:
  {
    "equations": [{"id": "eq_001", "text": "...", "normalized": "..."}],
    "citations":  [{"id": "cite_001", "aliases": ["Author YYYY", "[1]", ...]}]
  }

Domain rules:
  social_science → equations = [] (G12a mode, no equation matching needed)
  ai / natural_science → equations extracted via regex

Usage:
    PYTHONPATH=src python -m co_scientist.d6_grant_proposal.extract_gold \\
        goal_domain=ai goal_name=02_foundational_rl
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import chz

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.d6_grant_proposal.grounding import (
    _DISPLAY_MATH_RE,
    _INLINE_MATH_RE,
    _LATEX_ENV_RE,
    _CITE_PATTERNS,
    _normalize_eq,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[3]


# =============================================================================
# Extraction helpers
# =============================================================================

def _extract_equations(text: str) -> list[dict]:
    """Extract all equations from markdown text."""
    items: list[tuple[str]] = []
    seen_normalized: set[str] = set()

    def _add(raw: str):
        raw = raw.strip()
        if not raw:
            return
        norm = _normalize_eq(raw)
        if norm in seen_normalized or len(norm) < 3:
            return
        seen_normalized.add(norm)
        items.append({"text": raw, "normalized": norm})

    for m in _DISPLAY_MATH_RE.finditer(text):
        _add(m.group(1) or m.group(2) or "")
    for m in _INLINE_MATH_RE.finditer(text):
        _add(m.group(1))
    for m in _LATEX_ENV_RE.finditer(text):
        _add(m.group(2))

    result = []
    for i, item in enumerate(items):
        result.append({"id": f"eq_{i:03d}", **item})
    return result


def _extract_citations(text: str) -> list[dict]:
    """Extract all citations from markdown text. Build alias list per unique key."""
    alias_to_key: dict[str, str] = {}
    key_to_aliases: dict[str, list[str]] = {}

    for pattern in _CITE_PATTERNS:
        for m in pattern.finditer(text):
            alias = m.group(0).strip()
            key = m.group(1).strip()
            if key not in key_to_aliases:
                key_to_aliases[key] = []
            if alias not in key_to_aliases[key]:
                key_to_aliases[key].append(alias)
            alias_to_key[alias] = key

    result = []
    for i, (key, aliases) in enumerate(key_to_aliases.items()):
        result.append({"id": f"cite_{i:03d}", "key": key, "aliases": aliases})
    return result


# =============================================================================
# Config + main
# =============================================================================

@chz.chz
class Config:
    goal_domain: str = "ai"           # "ai" | "natural_science" | "social_science"
    goal_name: str = "02_foundational_rl"
    dataset_base: str = "projects/d6_grant_proposal/dataset"
    data_base: str = "projects/d6_grant_proposal/data"


def main(config: Config):
    goal_dir = (REPO_ROOT / config.dataset_base / config.goal_domain / config.goal_name).resolve()
    ref_path = goal_dir / "reference_proposal.md"

    if not ref_path.exists():
        raise FileNotFoundError(f"reference_proposal.md not found at {ref_path}")

    text = ref_path.read_text().strip()
    logger.info("Reference proposal: %d chars, %s/%s", len(text), config.goal_domain, config.goal_name)

    # Social science: skip equation extraction (G12a analytical framework domain)
    if config.goal_domain == "social_science":
        equations: list[dict] = []
        logger.info("Social science domain — skipping equation extraction (G12a mode)")
    else:
        equations = _extract_equations(text)
        logger.info("Extracted %d equations", len(equations))

    citations = _extract_citations(text)
    logger.info("Extracted %d citations", len(citations))

    out_dir = (REPO_ROOT / config.data_base / config.goal_domain / config.goal_name / "gold").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gold.json"

    gold = {"equations": equations, "citations": citations}
    out_path.write_text(json.dumps(gold, indent=2))
    logger.info("Wrote gold file to %s", out_path)
    logger.info("  equations: %d, citations: %d", len(equations), len(citations))


if __name__ == "__main__":
    chz.entrypoint(main)
