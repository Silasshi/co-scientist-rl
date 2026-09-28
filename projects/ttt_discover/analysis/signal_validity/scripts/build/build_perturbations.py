#!/usr/bin/env python3
"""Generate targeted perturbations of reference plans to test signal discrimination.

KEPT STRICTLY SEPARATE from reference plans:
- References: shared/papers/by_topic/{topic}/{name}/reference_plan.txt
- Perturbations: projects/ttt_discover/analysis/signal_validity/perturbations/{name}_{type}.txt
- Grading: grading_perturbations.jsonl (separate from grading_results_v5.jsonl)

For each reference plan, create targeted perturbations that should LOWER specific signals:
  - P_S3_no_positioning: delete all prior method names and citations
  - P_S4_weak_problem: replace problem section with generic research-area statement
  - P_S5_vague_method: replace methodology with "we use appropriate methods"
  - P_S6_no_risk: delete Risk & Boundary section
  - P_S7_no_specifics: replace specific numbers/names with vague terms
  - P_S8_overclaim: add "universal across all domains" to Problem section
  - P_S9_stacked: add 3 unrelated techniques to Core Idea
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

HERE = Path(__file__).parent
BASE = Path(__file__).resolve().parents[2]
PERTURB_DIR = BASE / "perturbations"
PERTURB_DIR.mkdir(exist_ok=True)
REFS_PATH = BASE / "data" / "refs" / "references_v2.jsonl"
OUT_PATH = BASE / "data" / "perturbations" / "perturbations.jsonl"
MODEL = "anthropic/claude-opus-4.1"


PERTURBATIONS = {
    "P_S3_no_positioning": (
        "S3_positioning",
        "Rewrite the Motivation section to REMOVE all named prior methods and "
        "citations. Replace specific method names (like 'GRPO', 'AlphaEvolve', "
        "'ESM-2') with vague phrases like 'prior work', 'existing methods', "
        "'some approaches'. Keep the rest of the plan unchanged."
    ),
    "P_S4_weak_problem": (
        "S4_significance",
        "Rewrite the Problem section to be GENERIC and VAGUE. Remove specific "
        "details about what changes, who cares, or the concrete gap. Make it "
        "sound like a generic research-area statement ('We study X. X is "
        "important. More work is needed.'). Keep other sections unchanged."
    ),
    "P_S5_vague_method": (
        "S5_feasibility",
        "Rewrite the Methodology section to REMOVE concrete details. Replace "
        "specific algorithm descriptions with 'we use appropriate methods', "
        "remove specific parameters, make it impossible to judge if the "
        "approach is implementable. Keep other sections unchanged."
    ),
    "P_S6_no_risk": (
        "S6_risk_awareness",
        "DELETE the entire Risk & Boundary section. Replace it with a single "
        "sentence: 'We expect this approach to work well.' Keep all other "
        "sections unchanged."
    ),
    "P_S7_no_specifics": (
        "S7_specificity",
        "Rewrite the Methodology section to REPLACE all specific numbers, model "
        "names, hyperparameters, and dataset names with vague terms. E.g., 'lr=5e-5' "
        "→ 'appropriate learning rate'; 'Qwen3-30B' → 'a language model'; "
        "'WikiText2' → 'a standard benchmark'. Keep other sections unchanged."
    ),
    "P_S8_overclaim": (
        "S8_scope",
        "Rewrite the Problem section to add UNIVERSAL claims: 'This approach "
        "works across all scientific domains', 'applicable to any research "
        "task', 'universal framework for scientific discovery'. Keep Evaluation "
        "section narrow (unchanged). Create an overclaim mismatch."
    ),
    "P_S9_stacked": (
        "S9_focus",
        "Rewrite the Core Idea section to STACK 3-4 UNRELATED techniques. Add "
        "techniques like Bayesian optimization, meta-learning, evolutionary "
        "algorithms, curiosity bonuses — without explaining why they are "
        "necessary or how they unify. Keep other sections unchanged."
    ),
}


PERTURB_PROMPT = """You are creating a PERTURBATION of a research plan for signal-validation testing.

Original plan:
---
{plan}
---

Perturbation instruction: {instruction}

Produce the perturbed plan. Rules:
- Keep the 6 section headers (## Problem, ## Motivation, ## Core Idea, ## Methodology, ## Evaluation, ## Risk & Boundary) — even if instructed to delete the Risk section, replace it with the specified placeholder sentence.
- Only modify the section(s) the instruction targets. Leave other sections EXACTLY as-is.
- Preserve overall length roughly (±20%).

Output ONLY the perturbed plan text, no preamble.
"""


async def perturb_one(client: OpenRouterClient, ref: dict, ptype: str, instruction: str) -> str | None:
    msg = [{"role": "user", "content": PERTURB_PROMPT.format(
        plan=ref["reference_solution"],
        instruction=instruction,
    )}]
    result = await client.chat(MODEL, msg, temperature=0.2, max_tokens=8000)
    result = result.strip()
    # Validate: still has most section headers
    required = ['## Problem', '## Motivation', '## Core Idea', '## Methodology', '## Evaluation']
    if sum(1 for s in required if s in result) < 4:
        return None
    return result


async def main():
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("Set OPENROUTER_API_KEY")

    with open(REFS_PATH) as f:
        refs = [json.loads(l) for l in f]

    # Resume
    done = set()
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            for line in f:
                entry = json.loads(l) if (l := line) else None
                if entry:
                    done.add(entry["perturbation_id"])
        done = set()
        with open(OUT_PATH) as f:
            for line in f:
                entry = json.loads(line)
                done.add(entry["perturbation_id"])

    # For each ref, generate all perturbation types
    # To limit cost, sample a subset first — 20 refs × 7 perturbations = 140 perturbations
    SAMPLE_N = 20
    sampled = refs[:SAMPLE_N]  # take first 20 (diverse topics)

    # Build task list
    tasks = []
    for ref in sampled:
        for ptype, (target_signal, instruction) in PERTURBATIONS.items():
            pid = f"{ref['source_id']}__{ptype}"
            if pid in done:
                continue
            tasks.append((ref, ptype, target_signal, instruction, pid))

    CONCURRENCY = 5
    logger.info(f"Processing {len(tasks)} tasks with concurrency={CONCURRENCY}")

    async with OpenRouterClient() as client:
        fout = open(OUT_PATH, "a")
        write_lock = asyncio.Lock()

        async def process_task(ref, ptype, target_signal, instruction, pid):
            try:
                text = await perturb_one(client, ref, ptype, instruction)
                if not text:
                    logger.warning(f"  ✗ {pid}: validation failed")
                    return
                ptext_path = PERTURB_DIR / f"{pid}.txt"
                ptext_path.write_text(text)
                entry = {
                    "perturbation_id": pid,
                    "base_id": ref["source_id"],
                    "perturbation_type": ptype,
                    "target_signal": target_signal,
                    "goal": ref["goal"],
                    "perturbed_plan": text,
                    "source": "perturbation",
                }
                async with write_lock:
                    fout.write(json.dumps(entry) + "\n")
                    fout.flush()
                logger.info(f"  ✓ {pid} ({len(text.split())}w)")
            except Exception as e:
                logger.error(f"  ✗ {pid}: {e}")

        # Run tasks with bounded concurrency
        semaphore = asyncio.Semaphore(CONCURRENCY)
        async def bounded(t):
            async with semaphore:
                await process_task(*t)
        await asyncio.gather(*[bounded(t) for t in tasks])

        fout.close()
        logger.info(f"Tokens: {client.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
