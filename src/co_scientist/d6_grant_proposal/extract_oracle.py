"""D6 oracle builder v1 — extract typed oracle items from reference_proposal.md.

For each goal, reads reference_proposal.md and calls Opus (via file-bus subagent)
to extract structured oracle items in 6 categories:
  - Insights: high-level conceptual takeaways
  - Methodology: concrete algorithmic or procedural steps
  - Evidence: named citations, quantitative claims, named systems
  - Structure: how the proposal organizes its argument
  - Deliverables: specific named outputs
  - Risk: failure modes or scope boundaries stated

Output: data/{goal_domain}/{goal_name}/oracle/oracle.jsonl

Usage:
    PYTHONPATH=src python -m co_scientist.d6_grant_proposal.extract_oracle \
        goal_domain=ai goal_name=02_foundational_rl
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

from co_scientist.d6_grant_proposal.prompt_loader import load_prompt
from co_scientist.shared.api_profiles import create_service_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[3]

_EXTRACT_PROMPT = load_prompt("oracle/extract_items.md")


def _parse_items(raw: str) -> list[dict]:
    """Parse <item>...</item> blocks from raw Opus output."""
    import re
    items = []
    for m in re.finditer(r"<item>(.*?)</item>", raw, re.DOTALL):
        block = m.group(1).strip()
        cat_m = re.search(r"<category>(.*?)</category>", block, re.DOTALL)
        txt_m = re.search(r"<text>(.*?)</text>", block, re.DOTALL)
        if cat_m and txt_m:
            items.append({
                "category": cat_m.group(1).strip(),
                "text": txt_m.group(1).strip(),
            })
    return items


@chz.chz
class Config:
    goal_domain: str = "ai"           # "ai" | "natural_science" | "social_science"
    goal_name: str = "02_foundational_rl"
    api_profile: str | None = "new"
    base_url: str | None = None
    oracle_model: str = "claude-opus-4-7"
    max_tokens: int = 4096
    dataset_base: str = "projects/d6_grant_proposal/dataset"
    output_base: str = "projects/d6_grant_proposal/data"


def main(config: Config):
    goal_dir = (REPO_ROOT / config.dataset_base / config.goal_domain / config.goal_name).resolve()
    goal_text = (goal_dir / "research_goal.md").read_text().strip()
    ref_proposal = (goal_dir / "reference_proposal.md").read_text().strip()

    out_dir = (REPO_ROOT / config.output_base / config.goal_domain / config.goal_name / "oracle").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "oracle.jsonl"

    if out_path.exists():
        logger.info("Oracle already exists at %s — delete to rebuild", out_path)
        return

    logger.info("Building oracle for %s/%s", config.goal_domain, config.goal_name)
    logger.info("Goal: %d chars, Reference: %d chars", len(goal_text), len(ref_proposal))

    prompt = _EXTRACT_PROMPT.format(goal=goal_text, reference_proposal=ref_proposal)

    client = create_service_client(api_profile=config.api_profile, base_url=config.base_url)

    logger.info("Calling Opus to extract oracle items...")
    # Use anthropic client (Opus is accessed via subagent file-bus normally, but for
    # a one-shot build script, direct API call is fine — this is not a training loop)
    from anthropic import Anthropic
    anthropic_client = Anthropic()
    response = anthropic_client.messages.create(
        model=config.oracle_model,
        max_tokens=config.max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text
    logger.info("Raw response: %d chars", len(raw))

    items = _parse_items(raw)
    logger.info("Extracted %d oracle items", len(items))

    if len(items) < 5:
        logger.warning("Too few items (%d) — check raw output:\n%s", len(items), raw[:2000])

    with open(out_path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    logger.info("Wrote oracle to %s", out_path)


if __name__ == "__main__":
    chz.entrypoint(main)
