#!/usr/bin/env python3
"""Add perturbations for S1_depth and S2_rigor (not included in first batch)."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from co_scientist.shared.openrouter_client import OpenRouterClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)
logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
BASE = Path(__file__).resolve().parents[2]
PERTURB_DIR = BASE / "perturbations"
PERTURB_DIR.mkdir(exist_ok=True)
REFS_PATH = BASE / "data" / "refs" / "references_v2.jsonl"
OUT_PATH = BASE / "data" / "perturbations" / "perturbations.jsonl"
MODEL = "anthropic/claude-opus-4.1"


NEW_PERTURBS = {
    "P_S1_asserted": (
        "S1_depth",
        "Rewrite the Core Idea and Methodology sections to REPLACE every derived "
        "justification with bald assertions. For each key design choice, remove "
        "the 'why' and keep only the 'what'. Replace phrases like 'we use X "
        "because Y implies Z' with 'we use X'. Remove all 'this follows from', "
        "'we motivated by', 'the mechanism is', etc. The plan should still list "
        "the techniques but not explain why any of them work. Keep other sections "
        "unchanged."
    ),
    "P_S2_strawman": (
        "S2_rigor",
        "Rewrite the Evaluation section to REPLACE all named fair baselines with "
        "obvious strawmen (random baseline, untrained baseline, hand-coded "
        "heuristic from 2015). REMOVE specific success metrics/thresholds and "
        "replace with vague 'we expect improvement'. The Evaluation should read "
        "as if the authors picked weak baselines and can't articulate what "
        "success looks like. Keep other sections unchanged."
    ),
}


PROMPT = """You are creating a PERTURBATION of a research plan for signal-validation testing.

Original plan:
---
{plan}
---

Perturbation instruction: {instruction}

Rules:
- Keep the 6 section headers (## Problem, ## Motivation, ## Core Idea, ## Methodology, ## Evaluation, ## Risk & Boundary).
- Only modify the section(s) the instruction targets.
- Preserve overall length roughly (±20%).

Output ONLY the perturbed plan text, no preamble.
"""


async def perturb_one(client, ref, ptype, instruction):
    msg = [{"role": "user", "content": PROMPT.format(
        plan=ref["reference_solution"], instruction=instruction,
    )}]
    result = (await client.chat(MODEL, msg, temperature=0.2, max_tokens=8000)).strip()
    required = ['## Problem', '## Motivation', '## Core Idea', '## Methodology', '## Evaluation']
    if sum(1 for s in required if s in result) < 4:
        return None
    return result


async def main():
    with open(REFS_PATH) as f:
        refs = [json.loads(l) for l in f]

    done = set()
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            for line in f:
                done.add(json.loads(line)['perturbation_id'])

    SAMPLE_N = 20
    sampled = refs[:SAMPLE_N]

    tasks = []
    for ref in sampled:
        for ptype, (target, instr) in NEW_PERTURBS.items():
            pid = f"{ref['source_id']}__{ptype}"
            if pid in done:
                continue
            tasks.append((ref, ptype, target, instr, pid))
    logger.info(f"Running {len(tasks)} new tasks (S1+S2 perturbations)")

    CONCURRENCY = 5
    async with OpenRouterClient() as client:
        fout = open(OUT_PATH, "a")
        lock = asyncio.Lock()

        async def process(ref, ptype, target, instr, pid):
            try:
                text = await perturb_one(client, ref, ptype, instr)
                if not text:
                    return
                (PERTURB_DIR / f"{pid}.txt").write_text(text)
                entry = {
                    "perturbation_id": pid, "base_id": ref['source_id'],
                    "perturbation_type": ptype, "target_signal": target,
                    "goal": ref['goal'], "perturbed_plan": text,
                    "source": "perturbation",
                }
                async with lock:
                    fout.write(json.dumps(entry) + "\n")
                    fout.flush()
                logger.info(f"  ✓ {pid}")
            except Exception as e:
                logger.error(f"  ✗ {pid}: {e}")

        sem = asyncio.Semaphore(CONCURRENCY)
        async def bounded(t):
            async with sem:
                await process(*t)
        await asyncio.gather(*[bounded(t) for t in tasks])
        fout.close()


if __name__ == "__main__":
    asyncio.run(main())
