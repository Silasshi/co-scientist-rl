#!/usr/bin/env python3
"""Build 6-criteria reference plan dataset using OpenRouter API.

Pipeline:
1. Fetch recent papers from arxiv by category
2. For each paper, use OpenRouter (strong model) to generate:
   - research_goal.txt (400-800 words)
   - reference_plan.txt (1500-3500 words, 6 sections)
3. Save to shared/papers/by_topic/{category}/{shortname}/

Usage:
    source .env && python projects/ttt_discover/analysis/signal_validity/build_reference_dataset.py
"""

import asyncio
import json
import logging
import os
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from co_scientist.shared.openrouter_client import OpenRouterClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)
logger = logging.getLogger(__name__)

PAPERS_DIR = PROJECT_ROOT / "shared" / "papers" / "by_topic"
TEMPLATE_PATH = PROJECT_ROOT / "shared" / "docs" / "RESEARCH_PLAN_TEMPLATE.md"

# Strong model for writing high-quality reference plans
WRITER_MODEL = "anthropic/claude-opus-4.1"  # strong reasoning, good structure

# Arxiv categories and search queries
# Each tuple: (folder_name, arxiv_query, target_count)
TOPICS = [
    # Existing topics (expand to target ~25 each)
    ("01_test_time_search", "(abs:\"test-time training\" OR abs:\"test-time adaptation\" OR abs:\"self-refine\" OR abs:\"self-improvement\") AND cat:cs.LG", 15),
    ("04_self_evolution", "(abs:\"self-evolution\" OR abs:\"self-improve\" OR abs:\"iterative refinement\") AND cat:cs.CL", 20),
    ("05_self_critique_no_ground_truth", "(abs:\"self-critique\" OR abs:\"self-verification\" OR abs:\"LLM verifier\") AND cat:cs.CL", 20),
    ("06_scientific_ai", "(abs:\"AI scientist\" OR abs:\"autonomous research\" OR abs:\"research agent\" OR abs:\"hypothesis generation\") AND cat:cs.AI", 20),
    ("07_rl_methods", "(abs:\"RLHF\" OR abs:\"GRPO\" OR abs:\"DPO\" OR abs:\"process reward model\") AND cat:cs.LG", 20),
    ("08_agent_systems", "(abs:\"LLM agent\" OR abs:\"agentic\" OR abs:\"tool use\") AND cat:cs.AI", 20),
    # New topics for diversity
    ("10_reasoning", "(abs:\"chain-of-thought\" OR abs:\"reasoning\" OR abs:\"step-by-step\") AND cat:cs.CL", 20),
    ("11_benchmarks", "(abs:\"benchmark\" OR abs:\"evaluation\" OR abs:\"LLM evaluation\") AND cat:cs.CL", 15),
    ("12_interpretability", "(abs:\"interpretability\" OR abs:\"mechanistic\" OR abs:\"probing\") AND cat:cs.LG", 15),
    ("13_alignment", "(abs:\"safety\" OR abs:\"alignment\" OR abs:\"red-teaming\") AND cat:cs.CL", 15),
]


# ── Arxiv fetching ─────────────────────────────────────────────────────────

async def search_arxiv(query: str, max_results: int = 30, start: int = 0) -> list[dict]:
    """Search arxiv, return list of papers (title, abstract, authors, id, date)."""
    encoded = urllib.parse.quote(query)
    url = (
        f"https://export.arxiv.org/api/query?"
        f"search_query={encoded}&start={start}&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )
    # Arxiv asks for 3s between queries — give 15s to avoid tight rate limits
    await asyncio.sleep(15.0)
    for attempt in range(5):
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                r = await client.get(url)
                r.raise_for_status()
            break
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 503):
                wait = 10 * (2 ** attempt)
                logger.warning(f"Arxiv {e.response.status_code}, waiting {wait}s")
                await asyncio.sleep(wait)
            else:
                raise
    else:
        raise RuntimeError("Arxiv query failed after retries")

    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(r.text)
    papers = []
    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        abstract = entry.find('atom:summary', ns).text.strip()
        arxiv_id = entry.find('atom:id', ns).text.strip().split('/')[-1]
        published = entry.find('atom:published', ns).text.strip()
        authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
        papers.append({
            'title': title,
            'abstract': abstract,
            'arxiv_id': arxiv_id,
            'published': published,
            'authors': authors,
        })
    return papers


def shortname_from_title(title: str) -> str:
    """Generate a short folder name from title."""
    # Take first 3-4 meaningful words
    words = re.findall(r'[A-Za-z][A-Za-z0-9-]*', title)
    meaningful = [w for w in words if len(w) >= 3 and w.lower() not in
                  {'for', 'the', 'and', 'with', 'via', 'using', 'from', 'towards', 'novel', 'new'}][:3]
    name = ''.join(w.capitalize() for w in meaningful)
    # Clean up
    name = re.sub(r'[^A-Za-z0-9]', '', name)
    return name[:30] if name else 'Paper'


# ── Plan generation ───────────────────────────────────────────────────────

GOAL_PROMPT = """You are writing a research goal statement in the style of a research plan dataset.

Here is a paper's title and abstract:

Title: {title}
Authors: {authors}
Abstract: {abstract}

Based on this paper, write a research goal statement (400-800 words) that:

1. Describes the research SCENARIO/CHALLENGE as if posing it to someone before the work started (e.g., "You are tasked with..." or "The challenge is...")
2. Clearly states the SPECIFIC PROBLEM to solve
3. Explains why this problem is IMPORTANT (what becomes possible)
4. Describes the GAP versus existing approaches (what prior methods fail to achieve and why)
5. Should read as a concrete research brief, not a paper summary

Do NOT:
- Summarize the paper's contributions (we'll do that separately in the plan)
- Mention the paper's specific solution
- Use phrases like "this paper proposes" or "the authors"

The goal describes the PROBLEM SETTING, not the solution.

Output ONLY the goal text, no preamble.
"""

PLAN_PROMPT = """You are writing a research plan in a strict 6-criteria format. This plan should describe what THIS paper actually did — treat the paper as the reference solution.

Paper:
Title: {title}
Authors: {authors}
Abstract: {abstract}

Research goal (already written):
{goal}

Write a research plan (1500-3500 words) with EXACTLY these 6 sections (use these exact headers):

## Problem
[Criterion 1: sharp problem statement, importance, specific gap, what changes if successful. 150-300 words.]

## Motivation
[Criterion 2: Name 2-5 SPECIFIC prior methods by name. For each, explain its specific insufficiency for this problem. Do NOT list generic "prior work". 200-400 words.]

## Core Idea
[Criterion 3: One-sentence core hypothesis. Then mechanism intuition (WHY this might work). Then a plausibility argument (preliminary evidence, mathematical grounding, or concrete example). 200-400 words.]

## Methodology
[Criterion 4: The actual algorithm/pipeline/procedure with reasoning for each major design choice. Include key parameters at methodology level (not exhaustive appendix). Make clear what's learnable vs frozen. 400-800 words.]

## Evaluation
[Criterion 5: Specific metrics, success/failure criteria, named baselines chosen to be fair (not strawmen). What specific results would distinguish the hypothesis being right vs wrong. 300-500 words.]

## Risk & Boundary
[Criterion 6: 2-3 SPECIFIC failure modes (not generic "might not work"). Which assumptions are most fragile. What the plan does NOT claim (scope boundary). Sketch of fallback direction. 200-400 words.]

CRITICAL:
- Stay faithful to what the paper actually did (use abstract + your knowledge, don't invent)
- No padding with "5 random seeds, bootstrap resamples, FDR correction" unless the paper specifically uses them
- No generic "limitations exist" — name specific failure modes
- Length: 1500-3500 words total (aim for 2000-2500)
- Use markdown with the 6 exact section headers above

Output ONLY the plan text, no preamble.
"""


async def generate_goal_and_plan(client: OpenRouterClient, paper: dict) -> tuple[str, str]:
    """Generate research_goal and reference_plan for one paper."""
    goal_msg = [{"role": "user", "content": GOAL_PROMPT.format(
        title=paper['title'],
        authors=', '.join(paper['authors'][:5]),
        abstract=paper['abstract'],
    )}]
    goal = await client.chat(WRITER_MODEL, goal_msg, temperature=0.3, max_tokens=2000)

    plan_msg = [{"role": "user", "content": PLAN_PROMPT.format(
        title=paper['title'],
        authors=', '.join(paper['authors'][:5]),
        abstract=paper['abstract'],
        goal=goal.strip(),
    )}]
    plan = await client.chat(WRITER_MODEL, plan_msg, temperature=0.3, max_tokens=8000)

    return goal.strip(), plan.strip()


def validate_plan(plan: str) -> list[str]:
    """Return list of problems with the plan (empty if OK)."""
    problems = []
    required_sections = ['## Problem', '## Motivation', '## Core Idea', '## Methodology', '## Evaluation', '## Risk & Boundary']
    for sec in required_sections:
        if sec not in plan:
            problems.append(f'missing: {sec}')
    wc = len(plan.split())
    if wc < 1000:
        problems.append(f'too short: {wc} words')
    elif wc > 5000:
        problems.append(f'too long: {wc} words')
    return problems


async def process_topic(client: OpenRouterClient, folder: str, query: str, target: int):
    """Collect papers for one topic, generate plans, save to disk."""
    logger.info(f"=== {folder}: target {target} papers ===")

    # Check existing papers to skip
    topic_dir = PAPERS_DIR / folder
    topic_dir.mkdir(parents=True, exist_ok=True)
    existing = {d.name for d in topic_dir.iterdir() if d.is_dir()}
    logger.info(f"  Existing: {len(existing)} ({sorted(existing)})")

    papers = await search_arxiv(query, max_results=target * 3)
    logger.info(f"  Arxiv returned {len(papers)} candidates")

    added = 0
    for paper in papers:
        if added >= target:
            break
        shortname = shortname_from_title(paper['title'])
        if shortname in existing:
            continue
        paper_dir = topic_dir / shortname
        if paper_dir.exists() and (paper_dir / 'reference_plan.txt').exists():
            continue

        logger.info(f"  Processing: {shortname} ({paper['title'][:60]})")
        try:
            goal, plan = await generate_goal_and_plan(client, paper)
            problems = validate_plan(plan)
            if problems:
                logger.warning(f"    Skipped {shortname}: {problems}")
                continue

            paper_dir.mkdir(parents=True, exist_ok=True)
            (paper_dir / 'research_goal.txt').write_text(goal)
            (paper_dir / 'reference_plan.txt').write_text(plan)
            (paper_dir / 'metadata.json').write_text(json.dumps({
                'title': paper['title'],
                'authors': paper['authors'],
                'arxiv_id': paper['arxiv_id'],
                'published': paper['published'],
            }, indent=2))
            logger.info(f"    ✓ Saved {shortname} (goal: {len(goal.split())}w, plan: {len(plan.split())}w)")
            added += 1
        except Exception as e:
            logger.error(f"    Failed {shortname}: {e}")

    logger.info(f"  Added {added} papers to {folder}")
    return added


async def main():
    if not os.environ.get('OPENROUTER_API_KEY'):
        raise RuntimeError("Set OPENROUTER_API_KEY (source .env)")

    async with OpenRouterClient() as client:
        total = 0
        for folder, query, target in TOPICS:
            try:
                n = await process_topic(client, folder, query, target)
                total += n
            except Exception as e:
                logger.error(f"Topic {folder} failed: {e}")
        logger.info(f"=== Total added: {total} papers ===")
        logger.info(f"Token usage: {client.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
