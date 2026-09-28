"""D6 slim oracle builder v1 — prune oracle.jsonl to fit within 4096-token policy budget.

Reads data/{goal_domain}/{goal_name}/oracle/oracle.jsonl and formats items into a
readable markdown block, truncating if needed.

Output: data/{goal_domain}/{goal_name}/oracle/slim.md

Usage:
    PYTHONPATH=src python -m co_scientist.d6_grant_proposal.extract_oracle_slim \
        goal_domain=ai goal_name=02_foundational_rl
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import chz

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[3]

MAX_CHARS = 12000  # ~3000 tokens for slim oracle; leaves budget for goal + critique + plan


def _format_items(items: list[dict]) -> str:
    """Format oracle items into a readable markdown block grouped by category."""
    from collections import defaultdict
    by_category: dict[str, list[str]] = defaultdict(list)
    for item in items:
        by_category[item["category"]].append(item["text"])

    category_order = ["Insights", "Methodology", "Evidence", "Structure", "Deliverables", "Risk"]
    lines = []
    for cat in category_order:
        if cat not in by_category:
            continue
        lines.append(f"## {cat}")
        for i, text in enumerate(by_category[cat], 1):
            lines.append(f"{i}. {text}")
        lines.append("")
    return "\n".join(lines).strip()


@chz.chz
class Config:
    goal_domain: str = "ai"           # "ai" | "natural_science" | "social_science"
    goal_name: str = "02_foundational_rl"
    data_base: str = "projects/d6_grant_proposal/data"
    max_chars: int = MAX_CHARS


def main(config: Config):
    oracle_dir = (REPO_ROOT / config.data_base / config.goal_domain / config.goal_name / "oracle").resolve()
    oracle_path = oracle_dir / "oracle.jsonl"
    slim_path = oracle_dir / "slim.md"

    if not oracle_path.exists():
        raise FileNotFoundError(
            f"Oracle not found at {oracle_path}. Run extract_oracle.py first."
        )

    items = []
    with open(oracle_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    logger.info("Loaded %d oracle items", len(items))

    full_text = _format_items(items)
    logger.info("Full oracle: %d chars", len(full_text))

    if len(full_text) <= config.max_chars:
        slim_text = full_text
    else:
        # Truncate: drop items from the end until within budget
        truncated_items = list(items)
        while len(_format_items(truncated_items)) > config.max_chars and len(truncated_items) > 1:
            truncated_items.pop()
        slim_text = _format_items(truncated_items)
        logger.info(
            "Truncated to %d items (%d chars)", len(truncated_items), len(slim_text)
        )

    slim_path.write_text(slim_text)
    logger.info("Wrote slim oracle to %s (%d chars)", slim_path, len(slim_text))


if __name__ == "__main__":
    chz.entrypoint(main)
