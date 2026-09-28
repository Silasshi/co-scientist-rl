#!/usr/bin/env python3
"""Rebuild reference plans from FULL paper PDFs (not just abstracts).

Pipeline:
1. For each paper with metadata.json (has arxiv_id), download PDF from arxiv
2. Extract text from PDF
3. Feed full paper text to Claude Opus with 6-criteria template
4. Replace reference_plan.txt with new, more specific version

This addresses S7_specificity by giving the writer model access to actual
implementation details (hyperparameters, architecture specs, etc.) that are
in the full paper but not in the abstract.
"""

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from co_scientist.shared.openrouter_client import OpenRouterClient
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)
logger = logging.getLogger(__name__)

PAPERS = PROJECT_ROOT / "shared" / "papers" / "by_topic"
MODEL = "anthropic/claude-opus-4.1"
MAX_PAPER_CHARS = 60000  # ~15K tokens, safely within Opus 200K context


PLAN_PROMPT = """You are writing a research plan that describes what THIS paper actually did — treat the paper as the reference solution.

Research goal (already written):
{goal}

Full paper text (including implementation details, hyperparameters, and experimental setup):
---
{paper_text}
---

Write a research plan (2000-3500 words) with EXACTLY these 6 sections (use these exact headers):

## Problem
[Sharp problem statement, importance, specific gap, what changes if successful. 200-400 words.]

## Motivation
[Name 2-5 SPECIFIC prior methods by name (use names from the paper). For each, explain its specific insufficiency for this problem. 300-500 words.]

## Core Idea
[One-sentence core hypothesis. Then mechanism intuition (WHY this might work). Then a plausibility argument. 200-400 words.]

## Methodology
[The actual algorithm/pipeline/procedure with reasoning for each major design choice. MUST INCLUDE CONCRETE SPECIFICS from the paper:
- Exact model names and sizes (e.g., "Qwen3-8B" not just "LLM")
- Specific hyperparameters: learning rate, batch size, optimizer, number of steps/epochs, weight decay, etc.
- Loss function formulas
- Dataset specifics: sizes, splits, preprocessing
- Hardware/compute details
Make clear what's learnable vs frozen. 600-1200 words.]

## Evaluation
[Specific metrics, success/failure criteria, named baselines chosen to be fair. Include actual benchmark names and where possible actual numbers reported in the paper. What specific results would distinguish the hypothesis being right vs wrong. 400-600 words.]

## Risk & Boundary
[2-3 SPECIFIC failure modes. Which assumptions are most fragile. What the plan does NOT claim (scope boundary). Sketch of fallback. 300-500 words.]

CRITICAL:
- USE THE FULL PAPER TEXT to extract real specifics — this is the key difference from a plan written only from the abstract
- For hyperparameters, architecture details, dataset sizes: USE THE EXACT VALUES from the paper
- Stay faithful to what the paper actually did — don't invent
- Length: 2000-3500 words total
- Use markdown with the 6 exact section headers above

Output ONLY the plan text, no preamble.
"""


async def download_arxiv_pdf(arxiv_id: str, out_path: Path) -> bool:
    """Download paper PDF from arxiv."""
    if out_path.exists():
        return True
    # Strip version suffix if present
    base_id = re.sub(r'v\d+$', '', arxiv_id)
    url = f"https://arxiv.org/pdf/{base_id}"
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as c:
            r = await c.get(url)
            r.raise_for_status()
        out_path.write_bytes(r.content)
        return True
    except Exception as e:
        logger.warning(f"  Download failed for {arxiv_id}: {e}")
        return False


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from PDF, truncated to MAX_PAPER_CHARS."""
    try:
        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() for page in reader.pages)
    except Exception as e:
        logger.warning(f"  PDF read failed: {e}")
        return ""
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    if len(text) > MAX_PAPER_CHARS:
        text = text[:MAX_PAPER_CHARS] + "\n\n[... rest of paper truncated ...]"
    return text


async def rebuild_one(client: OpenRouterClient, paper_dir: Path) -> bool:
    """Rebuild one paper's plan from full PDF."""
    meta_file = paper_dir / "metadata.json"
    goal_file = paper_dir / "research_goal.txt"
    plan_file = paper_dir / "reference_plan.txt"

    if not meta_file.exists() or not goal_file.exists():
        # Legacy papers without metadata — skip for now
        return False

    meta = json.loads(meta_file.read_text())
    arxiv_id = meta.get("arxiv_id")
    if not arxiv_id:
        return False

    pdf_path = paper_dir / "paper.pdf"
    if not await download_arxiv_pdf(arxiv_id, pdf_path):
        return False

    paper_text = extract_pdf_text(pdf_path)
    if len(paper_text) < 2000:
        logger.warning(f"  {paper_dir.name}: paper text too short ({len(paper_text)} chars)")
        return False

    goal = goal_file.read_text().strip()
    msg = [{"role": "user", "content": PLAN_PROMPT.format(
        goal=goal, paper_text=paper_text,
    )}]
    new_plan = await client.chat(MODEL, msg, temperature=0.3, max_tokens=10000)
    new_plan = new_plan.strip()

    # Validate 6 sections present
    required = ['## Problem', '## Motivation', '## Core Idea', '## Methodology', '## Evaluation', '## Risk']
    missing = [s for s in required if s not in new_plan]
    if missing:
        logger.warning(f"  {paper_dir.name}: missing sections {missing}")
        return False

    wc = len(new_plan.split())
    if wc < 1500 or wc > 6000:
        logger.warning(f"  {paper_dir.name}: word count {wc} out of range")
        return False

    plan_file.write_text(new_plan)
    logger.info(f"  ✓ Rebuilt {paper_dir.name}: {wc} words from {len(paper_text)//1000}K paper chars")
    return True


async def main():
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("Set OPENROUTER_API_KEY")

    papers_with_meta = [
        p.parent for p in PAPERS.rglob("metadata.json")
    ]
    logger.info(f"Found {len(papers_with_meta)} papers with metadata")

    async with OpenRouterClient() as client:
        rebuilt = 0
        for paper_dir in papers_with_meta:
            try:
                if await rebuild_one(client, paper_dir):
                    rebuilt += 1
            except Exception as e:
                logger.error(f"  Failed {paper_dir.name}: {e}")
        logger.info(f"=== Rebuilt {rebuilt}/{len(papers_with_meta)} plans ===")
        logger.info(f"Tokens: {client.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
