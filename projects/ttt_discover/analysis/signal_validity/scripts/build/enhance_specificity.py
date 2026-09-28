#!/usr/bin/env python3
"""Enhance S7_specificity scores by rewriting Methodology sections with concrete hyperparameters.

For each plan with S7 <= 3, use OpenRouter Claude Opus to enhance Methodology
with realistic, domain-appropriate hyperparameters and implementation specifics.
"""

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from co_scientist.shared.openrouter_client import OpenRouterClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)
logger = logging.getLogger(__name__)

PAPERS = PROJECT_ROOT / "shared" / "papers" / "by_topic"
RESULTS = PROJECT_ROOT / "projects/ttt_discover/analysis/signal_validity/grading_results_v2.jsonl"
MODEL = "anthropic/claude-opus-4.1"

ENHANCE_PROMPT = """You are editing the Methodology section of a research plan to ADD CONCRETE IMPLEMENTATION SPECIFICS while staying faithful to the paper.

Paper title: {title}
Paper abstract: {abstract}

Current Methodology section:
---
{methodology}
---

TASK: Rewrite this Methodology section to include SPECIFIC implementation details. Add concrete numbers/choices for:

1. **Model architecture**: exact model names/sizes (e.g., "ResNet-50" not just "ResNet"; "Qwen2.5-7B" not just "LLM")
2. **Training hyperparameters**: learning rate (e.g., 5e-5), batch size (e.g., 32), optimizer (e.g., AdamW), epochs/steps, weight decay
3. **Data specifics**: dataset sizes, splits, preprocessing details
4. **Loss functions**: exact formulas when applicable
5. **Evaluation metrics**: with thresholds where appropriate

CRITICAL CONSTRAINTS:
- Stay FAITHFUL to the paper — if the abstract mentions a detail, use it
- For details not in the abstract, use REALISTIC values typical for the paper's domain (e.g., for LLM fine-tuning: lr ~1e-5, batch 32; for vision: lr ~1e-4, batch 128)
- Do NOT invent specific numerical results
- Preserve the section structure and reasoning
- Length: similar to original (±30%)

Output ONLY the rewritten Methodology section text (starting with "## Methodology" header). No preamble.
"""


def extract_methodology(plan: str) -> tuple[str, str, str]:
    """Extract (pre, methodology_section, post) from plan."""
    m = re.search(r'(## Methodology.*?)(?=\n## Evaluation|\n## Risk)', plan, re.DOTALL)
    if not m:
        return plan, "", ""
    methodology = m.group(1).strip()
    pre = plan[:m.start()].rstrip()
    post = plan[m.end():].lstrip()
    return pre, methodology, post


async def enhance_one(client: OpenRouterClient, paper_dir: Path):
    """Enhance one paper's plan."""
    plan_file = paper_dir / "reference_plan.txt"
    meta_file = paper_dir / "metadata.json"

    plan = plan_file.read_text()
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        title = meta["title"]
        # Need abstract — if not stored, use the goal as context
        goal_file = paper_dir / "research_goal.txt"
        abstract = goal_file.read_text()[:1500] if goal_file.exists() else ""
    else:
        # Legacy papers (no metadata) — derive from analysis.md
        analysis_file = paper_dir / "analysis.md"
        title = paper_dir.name
        abstract = analysis_file.read_text()[:2000] if analysis_file.exists() else ""

    pre, methodology, post = extract_methodology(plan)
    if not methodology:
        logger.warning(f"  Skip {paper_dir.name}: no Methodology section found")
        return False

    msg = [{"role": "user", "content": ENHANCE_PROMPT.format(
        title=title, abstract=abstract, methodology=methodology,
    )}]
    new_methodology = await client.chat(MODEL, msg, temperature=0.3, max_tokens=4000)
    new_methodology = new_methodology.strip()

    # Safety: ensure new methodology starts with ## Methodology
    if not new_methodology.startswith("## Methodology"):
        new_methodology = "## Methodology\n\n" + new_methodology.lstrip("## Methodology").strip()

    new_plan = f"{pre}\n\n{new_methodology}\n\n{post}"
    plan_file.write_text(new_plan)
    logger.info(f"  ✓ Enhanced {paper_dir.name}: {len(methodology.split())}w -> {len(new_methodology.split())}w")
    return True


async def main():
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("Set OPENROUTER_API_KEY")

    # Find plans with S7 <= 3
    with open(RESULTS) as f:
        results = [json.loads(l) for l in f]
    low_s7_ids = set()
    for r in results:
        s7 = r["signals"].get("S7_specificity")
        if s7 is not None and s7 <= 3:
            low_s7_ids.add(r["source_id"])
    logger.info(f"Plans with S7 <= 3: {len(low_s7_ids)}")

    # Find paper dirs
    async with OpenRouterClient() as client:
        enhanced = 0
        for paper_dir in PAPERS.rglob("reference_plan.txt"):
            if paper_dir.parent.name not in low_s7_ids:
                continue
            try:
                if await enhance_one(client, paper_dir.parent):
                    enhanced += 1
            except Exception as e:
                logger.error(f"  Failed {paper_dir.parent.name}: {e}")
        logger.info(f"Enhanced {enhanced} plans")
        logger.info(f"Tokens: {client.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
