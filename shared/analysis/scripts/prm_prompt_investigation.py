#!/usr/bin/env python3
"""
PRM Prompt Investigation: Test different prompt variants to see if Gemini Flash
can produce negative scores (-3 to -1) and use the full scoring range.

Approach:
  1. Extract real conversations from V3 training logs
  2. Test the current PRM prompt (baseline)
  3. Test prompt variants with different strategies:
     - Few-shot examples of negative scoring
     - Forced calibration ("assign at least one negative score")
     - Simplified scale (-1/0/+1)
     - Rubric-anchored scoring
  4. Log all results for comparison

Usage:
  source .env
  python analysis/prm_prompt_investigation.py
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from collections import Counter

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT / "src"))

from co_scientist.trainers._shared.openrouter_client import OpenRouterClient
from co_scientist.trainers.multiturn.train_multiturn_v4 import (
    PRM_DISCUSSION_PROMPT, PRM_PLAN_PROMPT, parse_prm_response, build_prm_prompt,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("analysis/prm_investigation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Prompt Variants
# ============================================================

# Variant A: Current prompt (baseline)
VARIANT_A_NAME = "baseline"
# Uses PRM_DISCUSSION_PROMPT / PRM_PLAN_PROMPT directly

# Variant B: Few-shot examples of negative scoring
VARIANT_B_DISCUSSION = """
You are evaluating a research tutor's discussion response.
You will see the full conversation between a student and tutor.
Rate the tutor's MOST RECENT response on a scale of -3 to +3.

+3 EXCEPTIONAL: Breakthrough insight. Highly specific, actionable.
+2 VERY HELPFUL: Substantive, technically deep, specific methods.
+1 GOOD: Useful but somewhat generic. Partially addresses question.
 0 NEUTRAL: Restates what student said. No new insight.
-1 WEAK: Vague or off-topic. Generic platitudes.
-2 POOR: Technical inaccuracies. Ignores student's questions.
-3 HARMFUL: Fundamentally wrong advice. Filler text.

## Example Evaluations

Example 1 — Score: +2
Tutor: "For handling class imbalance, I recommend focal loss with gamma=2.0, combined with stratified sampling. Focal loss downweights easy examples, which is ideal for your 1:50 imbalance ratio."
Reasoning: Specific method with parameters, justified for the problem.

Example 2 — Score: -1
Tutor: "That's a great question! There are many approaches to handle this. You might want to look into various techniques that have been proposed in the literature."
Reasoning: No specific advice. Generic filler that doesn't help.

Example 3 — Score: -2
Tutor: "You should use PCA to reduce dimensionality before applying your transformer model. This will significantly improve performance."
Reasoning: PCA destroys sequential structure needed by transformers. Technically misleading.

Example 4 — Score: 0
Tutor: "So you want to improve the model's performance on rare classes. That's indeed an important challenge in machine learning."
Reasoning: Just restates the student's problem. No actionable guidance.

Be critical. Use negative scores when the response is unhelpful.

Output EXACTLY:
<score>INTEGER from -3 to +3</score>
<hint>1-3 sentences: specific improvement. "none" if excellent.</hint>
""".strip()

# Variant C: Simplified ternary (-1/0/+1)
VARIANT_C_DISCUSSION = """
You are evaluating a research tutor's discussion response.
Rate the tutor's MOST RECENT response as one of:

+1 HELPFUL: The response provides specific, useful guidance that advances the research discussion. It addresses the student's question with concrete suggestions.

 0 NEUTRAL: The response neither helps nor hurts. It may restate what was already said, ask clarifying questions without adding insight, or provide generic advice.

-1 UNHELPFUL: The response is vague, off-topic, technically inaccurate, or fails to address the student's question. It wastes the student's time.

Most responses should be +1 or 0. Use -1 when the response actively fails to help.

Output EXACTLY:
<score>INTEGER: -1, 0, or +1</score>
<hint>1-3 sentences: specific improvement. "none" if helpful.</hint>
""".strip()

# Variant D: Forced distribution
VARIANT_D_DISCUSSION = """
You are evaluating a research tutor's discussion response.
Rate the tutor's MOST RECENT response on a scale of -3 to +3.

IMPORTANT CALIBRATION: In a typical batch of tutor responses:
- ~20% should score negative (-1 to -3): vague, off-topic, or wrong
- ~30% should score neutral (0): restates known info, no new insight
- ~50% should score positive (+1 to +3): provides useful guidance

If you find yourself giving only positive scores, you are not being critical enough. Many tutor responses contain generic advice that deserves a 0 or negative score.

+3: Breakthrough insight with specific, actionable details
+2: Substantive with specific methods and reasoning
+1: Useful but somewhat generic
 0: Restates known info, no new insight
-1: Vague, off-topic, generic platitudes
-2: Technical inaccuracies, ignores the question
-3: Fundamentally wrong, filler text

Output EXACTLY:
<score>INTEGER from -3 to +3</score>
<hint>1-3 sentences: specific improvement. "none" if excellent.</hint>
""".strip()

# Variant E: Comparative anchoring
VARIANT_E_DISCUSSION = """
You are evaluating a research tutor's discussion response.

Before scoring, ask yourself these questions:
1. Did the tutor provide ANY specific method, algorithm, or approach name? If no → score ≤ 0.
2. Did the tutor explain WHY their suggestion fits this specific problem? If no → score ≤ +1.
3. Did the tutor give actionable details (parameters, steps, metrics)? If no → score ≤ +1.
4. Could this advice apply to ANY research problem without modification? If yes → score ≤ 0.
5. Does the response contain filler phrases like "that's a great question" or "there are many approaches"? If yes → score -1.

Score scale: -3 (harmful) to +3 (exceptional).

Output EXACTLY:
<score>INTEGER from -3 to +3</score>
<hint>1-3 sentences: specific improvement. "none" if excellent.</hint>
""".strip()

VARIANTS = {
    "A_baseline": None,  # uses build_prm_prompt() directly
    "B_fewshot": VARIANT_B_DISCUSSION,
    "C_ternary": VARIANT_C_DISCUSSION,
    "D_forced_dist": VARIANT_D_DISCUSSION,
    "E_anchored": VARIANT_E_DISCUSSION,
}


# ============================================================
# Load test conversations
# ============================================================

def load_test_conversations(logs_path: str, n: int = 10):
    """Load diverse conversations from training logs."""
    convos = []
    with open(logs_path) as f:
        for line in f:
            convos.append(json.loads(line))

    # Select diverse: mix of high/low rubric, different turn counts
    convos_with_rubric = [c for c in convos if c.get("rubric_score") is not None]
    convos_with_rubric.sort(key=lambda c: c["rubric_score"])

    selected = []
    # Take from low, mid, high rubric
    n_each = n // 3
    selected.extend(convos_with_rubric[:n_each])  # low rubric
    mid = len(convos_with_rubric) // 2
    selected.extend(convos_with_rubric[mid:mid + n_each])  # mid rubric
    selected.extend(convos_with_rubric[-n_each:])  # high rubric

    return selected[:n]


def build_test_conversation(goal_text: str, plan_text: str, n_turns: int):
    """Build a synthetic conversation for testing."""
    conversation = [
        {"role": "user", "content": f"I'm working on: {goal_text}\nWhat are the key challenges?"},
        {"role": "assistant", "content": "That's a great question! There are many interesting approaches to explore here. Let me think about the key challenges and get back to you with some ideas."},
    ]
    if n_turns > 1:
        conversation.extend([
            {"role": "user", "content": "Can you be more specific about what methods to use?"},
            {"role": "assistant", "content": "Sure! I'd recommend looking into various machine learning techniques that could be applicable. The literature has several promising directions you could explore."},
        ])
    return conversation


# ============================================================
# Run investigation
# ============================================================

async def test_variant(
    client: OpenRouterClient,
    variant_name: str,
    prompt_template: str | None,
    conversations: list[list[dict]],
    is_plan: bool = False,
    model: str = "google/gemini-2.0-flash-001",
    temperature: float = 0.3,
) -> list[dict]:
    """Test one prompt variant on multiple conversations."""
    results = []

    for i, conv in enumerate(conversations):
        if prompt_template is None:
            # Variant A: use default build_prm_prompt
            prompt = build_prm_prompt(conv, is_plan_turn=is_plan)
        else:
            conv_text = "\n\n".join(
                f"**{msg['role'].upper()}**: {msg['content']}" for msg in conv
            )
            prompt = f"{prompt_template}\n\n## Full Conversation\n{conv_text}"

        try:
            response = await client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1024,
            )
            score, hint = parse_prm_response(response)
            results.append({
                "variant": variant_name,
                "conv_idx": i,
                "score": score,
                "hint": hint,
                "raw_response": response[:500] if response else "",
                "is_plan": is_plan,
            })
        except Exception as e:
            logger.warning("Variant %s conv %d failed: %s", variant_name, i, e)
            results.append({
                "variant": variant_name,
                "conv_idx": i,
                "score": None,
                "hint": None,
                "raw_response": str(e),
                "is_plan": is_plan,
            })

    return results


async def main():
    client = OpenRouterClient()

    # Load real conversations from V3 logs
    logs_path = "runs/2026/3/multiturn_v4/3/train/training_logs.jsonl"
    logger.info("Loading test conversations from %s", logs_path)

    raw_convos = []
    with open(logs_path) as f:
        for line in f:
            raw_convos.append(json.loads(line))

    # Build test conversations: mix of real goals + synthetic tutor responses
    # We need conversations where the tutor gives BAD responses to test negative scoring
    goals = list(set(c["goal"] for c in raw_convos[:32]))[:5]

    # Test set 1: deliberately bad tutor responses (should get negative scores)
    bad_conversations = []
    for goal in goals:
        bad_conversations.append([
            {"role": "user", "content": f"I'm working on: {goal}\nWhat are the key challenges?"},
            {"role": "assistant", "content": "That's a really interesting research problem! There are definitely some challenges here. I think you should explore various approaches and see what works best. The literature has a lot of relevant work that could be helpful. Let me know if you have more specific questions!"},
        ])

    # Test set 2: mediocre tutor responses (should get 0 or +1)
    mediocre_conversations = []
    for goal in goals:
        mediocre_conversations.append([
            {"role": "user", "content": f"I'm working on: {goal}\nWhat methods should I consider?"},
            {"role": "assistant", "content": "You could consider using neural networks for this task. Deep learning approaches have shown good results in similar problems. You might want to try different architectures and see which one performs best on your dataset. Transfer learning could also be worth exploring."},
        ])

    # Test set 3: good tutor responses (should get +2 or +3)
    good_conversations = []
    for goal in goals:
        good_conversations.append([
            {"role": "user", "content": f"I'm working on: {goal}\nWhat specific methodology would you recommend?"},
            {"role": "assistant", "content": f"For this problem, I'd recommend a three-phase approach: First, establish a strong baseline using a pre-trained transformer (e.g., RoBERTa-base) fine-tuned on your specific task with a learning rate of 2e-5 and batch size 32. Second, implement contrastive learning with hard negative mining to improve the model's discriminative ability — specifically, use in-batch negatives plus BM25-retrieved hard negatives, with temperature tau=0.07. Third, evaluate using both automatic metrics (F1, BLEU, BERTScore) and human evaluation on a stratified sample of 200 examples. This gives you reproducible, comparable results with clear ablation points."},
        ])

    all_conversations = bad_conversations + mediocre_conversations + good_conversations
    labels = (["bad"] * len(bad_conversations) +
              ["mediocre"] * len(mediocre_conversations) +
              ["good"] * len(good_conversations))

    # Run all variants
    all_results = []
    for variant_name, prompt_template in VARIANTS.items():
        logger.info("Testing variant: %s", variant_name)
        t0 = time.time()
        results = await test_variant(
            client, variant_name, prompt_template, all_conversations,
        )
        elapsed = time.time() - t0

        # Add labels
        for r, label in zip(results, labels):
            r["quality_label"] = label

        all_results.extend(results)

        scores = [r["score"] for r in results if r["score"] is not None]
        dist = Counter(scores)
        neg = sum(1 for s in scores if s < 0)
        logger.info(
            "  %s: n=%d scores=%s neg=%d mean=%.2f (%.1fs)",
            variant_name, len(scores), dict(sorted(dist.items())),
            neg, sum(scores)/max(len(scores),1), elapsed,
        )

    # Save results
    results_path = OUT_DIR / "prm_investigation_results.jsonl"
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    # Summary
    summary_path = OUT_DIR / "prm_investigation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("PRM Prompt Investigation Summary\n")
        f.write("=" * 60 + "\n\n")

        for variant_name in VARIANTS:
            variant_results = [r for r in all_results if r["variant"] == variant_name]
            scores = [r["score"] for r in variant_results if r["score"] is not None]
            dist = Counter(scores)

            f.write(f"Variant: {variant_name}\n")
            f.write(f"  Distribution: {dict(sorted(dist.items()))}\n")
            f.write(f"  Mean: {sum(scores)/max(len(scores),1):.2f}\n")
            f.write(f"  Negative scores: {sum(1 for s in scores if s < 0)}/{len(scores)}\n")

            # By quality label
            for label in ["bad", "mediocre", "good"]:
                label_scores = [r["score"] for r in variant_results
                               if r.get("quality_label") == label and r["score"] is not None]
                if label_scores:
                    f.write(f"  {label:>10}: mean={sum(label_scores)/len(label_scores):.2f} scores={label_scores}\n")
            f.write("\n")

        # Best variant
        f.write("\nBest variant for negative score production:\n")
        for variant_name in VARIANTS:
            variant_results = [r for r in all_results if r["variant"] == variant_name]
            scores = [r["score"] for r in variant_results if r["score"] is not None]
            neg = sum(1 for s in scores if s < 0)
            bad_scores = [r["score"] for r in variant_results
                         if r.get("quality_label") == "bad" and r["score"] is not None]
            bad_mean = sum(bad_scores)/max(len(bad_scores),1) if bad_scores else 0
            f.write(f"  {variant_name}: {neg} negative scores, bad_response_mean={bad_mean:.2f}\n")

    logger.info("Results saved to %s", OUT_DIR)
    logger.info("Summary saved to %s", summary_path)

    # Print summary to stdout
    with open(summary_path) as f:
        print(f.read())

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
