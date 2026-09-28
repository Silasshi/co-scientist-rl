#!/usr/bin/env python3
"""Expand reference dataset using Semantic Scholar API, filtering for high-impact papers.

Focus: papers from top venues (NeurIPS, ICLR, ICML, Nature, Science) with sufficient citations.
Goal: add ~140 high-value papers to reach ~200 total.

Semantic Scholar API: https://api.semanticscholar.org/graph/v1
- Higher rate limits than arxiv (1 req/sec unauthenticated, 10 req/sec with API key)
- Venue filtering
- Citation counts
- Returns arxiv IDs for PDF download
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
MAX_PAPER_CHARS = 60000
S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
MIN_CITATIONS = 20  # filter for "noticed" papers

# High-impact venues we want
TOP_VENUES = {
    'NeurIPS', 'Neural Information Processing Systems',
    'ICLR', 'International Conference on Learning Representations',
    'ICML', 'International Conference on Machine Learning',
    'Nature', 'Science',
    'Nature Machine Intelligence', 'Nature Methods',
    'EMNLP', 'ACL', 'NAACL',
    'CVPR', 'ICCV', 'ECCV',
}

# Search queries + target topic folder
QUERIES = [
    # ML methodology (target venue: NeurIPS/ICLR/ICML)
    ("LLM reasoning chain-of-thought", "10_reasoning", 15),
    ("test-time compute inference scaling", "01_test_time_search", 10),
    ("LoRA parameter-efficient fine-tuning", "09_efficient_training", 10),
    ("mixture of experts efficient", "09_efficient_training", 8),
    ("reinforcement learning human feedback RLHF", "07_rl_methods", 15),
    ("preference optimization DPO direct", "07_rl_methods", 10),
    ("process reward model PRM verifier", "07_rl_methods", 10),
    ("LLM agent tool use function calling", "08_agent_systems", 15),
    ("multi-agent LLM collaboration", "08_agent_systems", 10),
    ("LLM evaluation benchmark", "11_benchmarks", 12),
    ("alignment safety LLM", "13_alignment", 12),
    ("mechanistic interpretability probing", "12_interpretability", 12),
    ("autonomous research agent AI scientist", "06_scientific_ai", 10),
    ("self-improvement iterative refinement", "04_self_evolution", 12),
    ("self-critique self-correction LLM", "05_self_critique_no_ground_truth", 10),
]


# ── Semantic Scholar fetching ─────────────────────────────────────────────

async def search_s2(query: str, limit: int = 30) -> list[dict]:
    """Search Semantic Scholar, return list of papers with venue + citation info."""
    await asyncio.sleep(1.1)  # respect 1 req/sec unauthenticated rate limit
    fields = "title,abstract,authors,venue,citationCount,year,externalIds"
    url = f"{S2_API}?query={query}&limit={limit}&fields={fields}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(url)
            if r.status_code == 429:
                logger.warning(f"S2 rate limit, waiting 30s")
                await asyncio.sleep(30)
                r = await client.get(url)
            r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning(f"S2 query '{query}' failed: {e}")
        return []

    papers = []
    for p in data.get("data", []):
        venue = (p.get("venue") or "").strip()
        citations = p.get("citationCount") or 0
        arxiv_id = (p.get("externalIds") or {}).get("ArXiv")
        # Filter: top venue OR high citations OR has arxiv ID with citations
        is_top = any(v.lower() in venue.lower() for v in TOP_VENUES) if venue else False
        if not (is_top or citations >= MIN_CITATIONS):
            continue
        if not arxiv_id:
            continue  # need arxiv for PDF
        if not p.get("abstract"):
            continue
        papers.append({
            'title': p['title'],
            'abstract': p['abstract'],
            'arxiv_id': arxiv_id,
            'venue': venue,
            'citations': citations,
            'year': p.get('year'),
            'authors': [a['name'] for a in p.get('authors', [])],
        })
    logger.info(f"  '{query}' → {len(papers)} high-value papers (from {len(data.get('data',[]))} raw)")
    return papers


async def download_arxiv_pdf(arxiv_id: str, out_path: Path) -> bool:
    if out_path.exists():
        return True
    base_id = re.sub(r'v\d+$', '', arxiv_id)
    url = f"https://arxiv.org/pdf/{base_id}"
    try:
        await asyncio.sleep(5.0)  # arxiv PDF rate limit
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as c:
            r = await c.get(url)
            r.raise_for_status()
        out_path.write_bytes(r.content)
        return True
    except Exception as e:
        logger.warning(f"  PDF download failed {arxiv_id}: {e}")
        return False


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
        text = "\n".join(p.extract_text() for p in reader.pages)
    except Exception as e:
        return ""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text[:MAX_PAPER_CHARS]


def shortname(title: str) -> str:
    words = re.findall(r'[A-Za-z][A-Za-z0-9-]*', title)
    meaningful = [w for w in words if len(w) >= 3 and w.lower() not in
                  {'for', 'the', 'and', 'with', 'via', 'using', 'from', 'towards', 'novel', 'new', 'our'}][:3]
    name = ''.join(w.capitalize() for w in meaningful)
    return re.sub(r'[^A-Za-z0-9]', '', name)[:30] or 'Paper'


GOAL_PROMPT = """Write a research goal statement (400-800 words) as a research scenario for this paper.

Title: {title}
Venue: {venue} ({year}), {citations} citations
Abstract: {abstract}

Describe the problem setting, importance, and gap without mentioning the paper's solution. Use "You are tasked with..." style. Output only the goal text.
"""


PLAN_PROMPT = """Write a research plan (1800-3500 words) describing what THIS paper did, using the 6-criteria format.

Paper: {title} ({venue} {year}, {citations} citations)
Research goal:
{goal}

Full paper text:
---
{paper_text}
---

Use EXACTLY these 6 section headers:
## Problem
## Motivation
## Core Idea
## Methodology
## Evaluation
## Risk & Boundary

For Methodology, USE CONCRETE SPECIFICS from the paper:
- Exact model names/sizes
- Hyperparameters (lr, batch, optimizer, steps)
- Dataset specifics
- Loss formulas

For Motivation, name 2-5 SPECIFIC prior methods with their insufficiency.
For Risk, name 2-3 SPECIFIC failure modes.

Output ONLY the plan text.
"""


async def process_paper(client: OpenRouterClient, paper: dict, topic_dir: Path) -> bool:
    name = shortname(paper['title'])
    paper_dir = topic_dir / name
    if paper_dir.exists() and (paper_dir / 'reference_plan.txt').exists():
        return False

    pdf_path = paper_dir / 'paper.pdf' if paper_dir.exists() else topic_dir / f"{name}.pdf"
    paper_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = paper_dir / 'paper.pdf'

    if not await download_arxiv_pdf(paper['arxiv_id'], pdf_path):
        return False

    paper_text = extract_pdf_text(pdf_path)
    if len(paper_text) < 2000:
        logger.warning(f"  {name}: paper text too short")
        return False

    # Generate goal
    goal_msg = [{"role": "user", "content": GOAL_PROMPT.format(
        title=paper['title'],
        venue=paper['venue'] or 'arXiv',
        year=paper['year'] or '',
        citations=paper['citations'],
        abstract=paper['abstract'],
    )}]
    goal = (await client.chat(MODEL, goal_msg, temperature=0.3, max_tokens=1500)).strip()

    # Generate plan
    plan_msg = [{"role": "user", "content": PLAN_PROMPT.format(
        title=paper['title'],
        venue=paper['venue'] or 'arXiv',
        year=paper['year'] or '',
        citations=paper['citations'],
        goal=goal,
        paper_text=paper_text,
    )}]
    plan = (await client.chat(MODEL, plan_msg, temperature=0.3, max_tokens=8000)).strip()

    # Validate
    required = ['## Problem', '## Motivation', '## Core Idea', '## Methodology', '## Evaluation', '## Risk']
    if not all(s in plan for s in required):
        logger.warning(f"  {name}: missing sections")
        return False
    if len(plan.split()) < 1500:
        logger.warning(f"  {name}: plan too short")
        return False

    # Save
    (paper_dir / 'research_goal.txt').write_text(goal)
    (paper_dir / 'reference_plan.txt').write_text(plan)
    (paper_dir / 'metadata.json').write_text(json.dumps({
        'title': paper['title'],
        'arxiv_id': paper['arxiv_id'],
        'venue': paper['venue'],
        'citations': paper['citations'],
        'year': paper['year'],
        'authors': paper['authors'],
    }, indent=2))
    logger.info(f"  ✓ {name}: [{paper['venue'] or 'arxiv'}, {paper['citations']}c, {paper['year']}] plan={len(plan.split())}w")
    return True


async def main():
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("Set OPENROUTER_API_KEY")

    async with OpenRouterClient() as client:
        total_new = 0
        for query, topic, target in QUERIES:
            topic_dir = PAPERS / topic
            topic_dir.mkdir(parents=True, exist_ok=True)
            existing = {d.name for d in topic_dir.iterdir() if d.is_dir()}
            logger.info(f"\n=== '{query}' → {topic} (existing: {len(existing)}, target new: {target})")

            papers = await search_s2(query, limit=max(target * 4, 30))
            added = 0
            for paper in papers:
                if added >= target:
                    break
                name = shortname(paper['title'])
                if name in existing:
                    continue
                try:
                    if await process_paper(client, paper, topic_dir):
                        added += 1
                        existing.add(name)
                        total_new += 1
                except Exception as e:
                    logger.error(f"  Failed {paper['title'][:50]}: {e}")
            logger.info(f"  +{added} added to {topic}")
        logger.info(f"\n=== Total new: {total_new} ===")
        logger.info(f"Tokens: {client.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
