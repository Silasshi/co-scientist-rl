#!/usr/bin/env python3
"""Test diversity of fresh plan generation with approach seeding + negative conditioning.

Compares:
  A) Baseline: current prompt (no diversity mechanism)
  B) Diverse: approach seeding + negative conditioning

Generates 8 plans per condition, measures pairwise diversity.
No training, no grading — pure generation comparison.
"""

import json
import logging
import random
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import tinker
import tinker.types as types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from co_scientist.shared.api_profiles import create_service_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GOAL_PATH = PROJECT_ROOT / "projects/ttt_discover/analysis/sanity_check/research_goal.txt"
OUTPUT_DIR = Path(__file__).parent
MODEL = "Qwen/Qwen3-30B-A3B"
N_PLANS_PER_CONDITION = 8

# ── Approach Seeds ──────────────────────────────────────────────────────

APPROACH_SEEDS = [
    "evolutionary and population-based methods (e.g., genetic programming, MAP-Elites, quality-diversity)",
    "information-theoretic objectives (e.g., entropy maximization, mutual information, curiosity-driven exploration)",
    "Bayesian and probabilistic methods (e.g., Bayesian optimization, Thompson sampling, posterior inference)",
    "meta-learning and learning-to-learn (e.g., MAML, hypernetworks, learned optimizers)",
    "game-theoretic and adversarial methods (e.g., self-play, minimax, adversarial training)",
    "hierarchical planning and decomposition (e.g., subgoal discovery, options framework, divide-and-conquer)",
    "neuro-symbolic and program synthesis approaches (e.g., DSL-guided search, neural program induction)",
    "reward shaping and intrinsic motivation (e.g., novelty search, empowerment, surprise minimization)",
]


# ── Prompt Builders ─────────────────────────────────────────────────────

def build_baseline_prompt(goal: str) -> str:
    """Current standard prompt (same as build_research_plan_prompt with no context)."""
    return textwrap.dedent(f"""
        I will provide you a research scenario. You have to provide me a concise yet thoughtful research plan with all details needed to execute it.

        Here is the research scenario.
        Scenario: {goal}

        # Instructions
        First, come up with a detailed research plan to address the scenario based on the following overall solution guidelines:
        - The plan should address the goals of the scenario and account for all constraints and confounders.
        - Do NOT just say WHAT you will do. Explain HOW you will do it and WHY it is needed.
        - The phrasing should be in present tense, describing how you would approach the problem.
        - Do not claim to have done any experiments or have results, just provide the plan.
        - Do not add self-proclaimed praises of your solution.

        Then, return the following nested XML block as your output:
        <think>
        ...your reasoning process...
        </think>
        <solution>
        ...your detailed research plan here. Prioritize depth and quality.
        </solution>
    """).strip()


def build_diverse_prompt(
    goal: str,
    approach_seed: str,
    existing_summaries: list[str] | None = None,
) -> str:
    """Prompt with approach seeding + negative conditioning."""
    neg_cond = ""
    if existing_summaries:
        summaries_text = "\n".join(f"  - {s}" for s in existing_summaries)
        neg_cond = textwrap.dedent(f"""

        # Approaches Already Explored
        The following approaches have already been proposed. Your plan must take a FUNDAMENTALLY DIFFERENT direction — do not reuse their core methodology or framework.

{summaries_text}

        """).strip() + "\n\n"

    return textwrap.dedent(f"""
        I will provide you a research scenario. You have to provide me a concise yet thoughtful research plan with all details needed to execute it.

        Here is the research scenario.
        Scenario: {goal}

        # Methodological Focus
        For this plan, focus specifically on {approach_seed}. Build your entire approach around this paradigm. Do not mix in unrelated methods — develop one coherent methodology in depth.

        {neg_cond}# Instructions
        First, come up with a detailed research plan to address the scenario based on the following overall solution guidelines:
        - The plan should address the goals of the scenario and account for all constraints and confounders.
        - Do NOT just say WHAT you will do. Explain HOW you will do it and WHY it is needed.
        - The phrasing should be in present tense, describing how you would approach the problem.
        - Do not claim to have done any experiments or have results, just provide the plan.
        - Do not add self-proclaimed praises of your solution.

        Then, return the following nested XML block as your output:
        <think>
        ...your reasoning process...
        </think>
        <solution>
        ...your detailed research plan here. Prioritize depth and quality.
        </solution>
    """).strip()


def extract_solution(text: str) -> str:
    """Extract plan text from <solution> tags."""
    import re
    m = re.search(r"<solution>(.*?)(?:</solution>|$)", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def one_line_summary(plan: str, max_len: int = 120) -> str:
    """Extract a one-line summary of the plan's core approach."""
    # Take the first substantive line
    for line in plan.split("\n"):
        line = line.strip()
        if len(line) > 30 and not line.startswith("#") and not line.startswith("**"):
            return line[:max_len]
    return plan[:max_len]


# ── Diversity Metrics ───────────────────────────────────────────────────

def ngram_set(text: str, n: int = 3) -> set:
    words = text.lower().split()
    return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}


def pairwise_jaccard(plans: list[str], n: int = 3) -> float:
    """Average pairwise Jaccard similarity of n-gram sets."""
    sets = [ngram_set(p, n) for p in plans]
    similarities = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if union:
                similarities.append(len(sets[i] & sets[j]) / len(union))
    return sum(similarities) / len(similarities) if similarities else 0.0


def unique_methods_count(plans: list[str]) -> dict:
    """Count unique method keywords across plans."""
    METHOD_KEYWORDS = [
        "evolutionary", "genetic", "MAP-Elites", "quality-diversity",
        "Bayesian", "Thompson", "posterior", "GP regression",
        "meta-learning", "MAML", "hypernetwork", "learned optimizer",
        "information-theoretic", "entropy", "mutual information", "curiosity",
        "adversarial", "self-play", "minimax", "GAN",
        "hierarchical", "subgoal", "options framework", "decomposition",
        "neuro-symbolic", "program synthesis", "DSL",
        "reward shaping", "intrinsic motivation", "novelty search",
        "reinforcement learning", "PPO", "GRPO", "DPO",
        "gradient descent", "LoRA", "fine-tuning",
        "ensemble", "mixture", "multi-agent",
        "curriculum", "progressive", "staged",
    ]
    counts = Counter()
    for plan in plans:
        plan_lower = plan.lower()
        for kw in METHOD_KEYWORDS:
            if kw.lower() in plan_lower:
                counts[kw] += 1
    return dict(counts)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    goal = GOAL_PATH.read_text().strip()
    rng = random.Random(42)

    # Setup model
    service_client = create_service_client()
    sampling_client = service_client.create_sampling_client(base_model=MODEL)
    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    def generate_plan(prompt: str) -> str:
        convo = [{"role": "user", "content": prompt}]
        model_input = renderer.build_generation_prompt(convo)
        future = sampling_client.sample(
            model_input,
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=2048, temperature=1.0,
                stop=renderer.get_stop_sequences(),
            ),
        )
        resp = future.result(timeout=120)
        text = renderers.get_text_content(renderer.parse_response(resp.sequences[0].tokens)[0])
        return extract_solution(text)

    # ── Condition A: Baseline (no diversity) ────────────────────────────
    logger.info("=== Condition A: Baseline (8 plans) ===")
    baseline_prompt = build_baseline_prompt(goal)
    baseline_plans = []
    t0 = time.time()
    for i in range(N_PLANS_PER_CONDITION):
        plan = generate_plan(baseline_prompt)
        baseline_plans.append(plan)
        logger.info(f"  A{i}: {len(plan)} chars, {len(plan.split())} words")
    baseline_time = time.time() - t0

    # ── Condition B: Diverse (approach seeding + negative conditioning) ─
    logger.info("=== Condition B: Diverse (8 plans) ===")
    seeds = rng.sample(APPROACH_SEEDS, min(N_PLANS_PER_CONDITION, len(APPROACH_SEEDS)))
    diverse_plans = []
    t0 = time.time()
    for i, seed in enumerate(seeds):
        # Negative conditioning: summarize previous plans
        summaries = [one_line_summary(p) for p in diverse_plans[-3:]] if diverse_plans else None
        prompt = build_diverse_prompt(goal, seed, summaries)
        plan = generate_plan(prompt)
        diverse_plans.append(plan)
        logger.info(f"  B{i} ({seed[:30]}...): {len(plan)} chars, {len(plan.split())} words")
    diverse_time = time.time() - t0

    # ── Analysis ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DIVERSITY COMPARISON")
    print("=" * 70)

    # Pairwise similarity (lower = more diverse)
    base_sim = pairwise_jaccard(baseline_plans)
    div_sim = pairwise_jaccard(diverse_plans)
    print(f"\nPairwise 3-gram Jaccard similarity (lower = more diverse):")
    print(f"  Baseline: {base_sim:.4f}")
    print(f"  Diverse:  {div_sim:.4f}")
    print(f"  Change:   {div_sim - base_sim:+.4f} ({(div_sim/base_sim - 1)*100:+.1f}%)")

    # Unique methods
    base_methods = unique_methods_count(baseline_plans)
    div_methods = unique_methods_count(diverse_plans)
    print(f"\nUnique method keywords mentioned:")
    print(f"  Baseline: {len(base_methods)} unique keywords")
    print(f"  Diverse:  {len(div_methods)} unique keywords")

    # Methods unique to each condition
    base_only = set(base_methods) - set(div_methods)
    div_only = set(div_methods) - set(base_methods)
    if div_only:
        print(f"  New in diverse: {', '.join(sorted(div_only))}")

    # Word count comparison
    base_wc = [len(p.split()) for p in baseline_plans]
    div_wc = [len(p.split()) for p in diverse_plans]
    print(f"\nWord count: baseline={sum(base_wc)/len(base_wc):.0f} avg, diverse={sum(div_wc)/len(div_wc):.0f} avg")
    print(f"Time: baseline={baseline_time:.0f}s, diverse={diverse_time:.0f}s")

    # Per-plan summaries
    print(f"\n--- Baseline plan summaries ---")
    for i, p in enumerate(baseline_plans):
        print(f"  A{i}: {one_line_summary(p)}")
    print(f"\n--- Diverse plan summaries ---")
    for i, (p, s) in enumerate(zip(diverse_plans, seeds)):
        print(f"  B{i} [{s.split('(')[0].strip()[:25]}]: {one_line_summary(p)}")

    # Save results
    results = {
        "baseline": {"plans": baseline_plans, "similarity": base_sim, "methods": base_methods, "time_s": baseline_time},
        "diverse": {"plans": diverse_plans, "seeds": seeds, "similarity": div_sim, "methods": div_methods, "time_s": diverse_time},
    }
    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
