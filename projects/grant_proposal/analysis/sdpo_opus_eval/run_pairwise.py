"""Pairwise preference comparison: Qwen 30B judge on C2/C3/C4 best plans."""
import json, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import tinker
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook import model_info, renderers
from co_scientist.shared.api_profiles import create_service_client

MODEL = "Qwen/Qwen3-30B-A3B"
GOAL = (Path(__file__).resolve().parents[2] / "dataset/goals/01_foundopt/research_goal.md").read_text().strip()
EVAL_DIR = Path(__file__).resolve().parent

plans = {
    "C2": (EVAL_DIR / "C2_B4_scores_only_best_plan.txt").read_text(),
    "C3": (EVAL_DIR / "C3_sdpo_best_plan.txt").read_text(),
    "C4": (EVAL_DIR / "C4_aggregate_best_plan.txt").read_text(),
}

PAIRWISE_PROMPT = """\
You are an expert grant proposal reviewer (NIH study section level). You will see two research proposals (Plan A and Plan B) addressing the same research goal.

Choose which proposal is BETTER overall. Consider:
1. Problem depth and genuine understanding (not surface-level restating)
2. Methodological substance (adapted methods, not textbook descriptions)
3. Feasibility and specificity (realistic scope, concrete milestones)
4. Scholarly grounding (real citations, properly integrated)

Focus on SUBSTANCE over surface formatting. A longer proposal is not necessarily better.

# Research Goal
{goal}

# Plan A
{plan_a}

# Plan B
{plan_b}

Which plan is better overall? Respond with ONLY a JSON object:
{{"winner": "A" or "B", "reason": "one sentence justification"}}"""

pairs = [("C2", "C3"), ("C2", "C4"), ("C3", "C4")]
results = []

service_client = create_service_client(api_profile="new")
sampling_client = service_client.create_sampling_client(base_model=MODEL)
tokenizer = get_tokenizer(MODEL)
renderer_name = model_info.get_recommended_renderer_name(MODEL)
renderer = renderers.get_renderer(renderer_name, tokenizer)
sampling_params = tinker.SamplingParams(temperature=0.0, max_tokens=4096)

for tag_a, tag_b in pairs:
    for order in ["AB", "BA"]:
        if order == "AB":
            a_tag, b_tag = tag_a, tag_b
        else:
            a_tag, b_tag = tag_b, tag_a

        prompt = PAIRWISE_PROMPT.format(
            goal=GOAL, plan_a=plans[a_tag], plan_b=plans[b_tag]
        )

        print(f"Comparing {a_tag} (A) vs {b_tag} (B)...", end=" ", flush=True)
        model_input = renderer.build_generation_prompt(
            [{"role": "user", "content": prompt}]
        )
        result = sampling_client.sample(prompt=model_input, num_samples=1, sampling_params=sampling_params).result()
        parsed_resp = renderer.parse_response(result.sequences[0].tokens)
        raw = str(parsed_resp[0].get("content", "")) if isinstance(parsed_resp[0], dict) else str(parsed_resp)
        # Strip Qwen3 <think> tags
        think_match = re.search(r'</think>\s*(.*)', raw, re.DOTALL)
        if think_match:
            raw = think_match.group(1).strip()

        json_match = re.search(r'\{"winner":\s*"([AB])".*?\}', raw)
        if not json_match:
            json_match = re.search(r'"winner":\s*"([AB])"', raw)
        if json_match:
            winner_letter = json_match.group(1)
            reason_match = re.search(r'"reason":\s*"([^"]*)"', raw)
            reason = reason_match.group(1) if reason_match else ""
            parsed = {"winner": winner_letter, "reason": reason}
        else:
            # Fallback: look for plain A or B after </think>
            ab_match = re.search(r'\b([AB])\b', raw[:50])
            parsed = {"winner": ab_match.group(1) if ab_match else "?", "reason": raw[:100]}

        # Map winner back to original tag
        winner_pos = parsed.get("winner", "?")
        winner_tag = a_tag if winner_pos == "A" else (b_tag if winner_pos == "B" else "?")

        result = {
            "pair": f"{tag_a}-{tag_b}",
            "order": order,
            "plan_a": a_tag,
            "plan_b": b_tag,
            "winner_position": winner_pos,
            "winner_tag": winner_tag,
            "reason": parsed.get("reason", ""),
            "judge": "Qwen3-30B",
        }
        results.append(result)
        print(f"winner={winner_tag} ({winner_pos}): {result['reason'][:80]}")
        time.sleep(1)

# Summary
print("\n" + "=" * 70)
print("Qwen 30B Pairwise Preference Summary")
print("=" * 70)
wins = {t: 0 for t in plans}
for r in results:
    if r["winner_tag"] in wins:
        wins[r["winner_tag"]] += 1
for tag, w in sorted(wins.items(), key=lambda x: -x[1]):
    print(f"  {tag}: {w} wins / {len(results)} comparisons")

# Check position bias
print("\nPosition bias check:")
for tag_a, tag_b in pairs:
    ab = [r for r in results if r["pair"] == f"{tag_a}-{tag_b}" and r["order"] == "AB"][0]
    ba = [r for r in results if r["pair"] == f"{tag_a}-{tag_b}" and r["order"] == "BA"][0]
    consistent = ab["winner_tag"] == ba["winner_tag"]
    print(f"  {tag_a} vs {tag_b}: AB→{ab['winner_tag']}, BA→{ba['winner_tag']} {'✓ consistent' if consistent else '✗ POSITION BIAS'}")

# Save
out_path = EVAL_DIR / "pairwise_qwen30b.jsonl"
with open(out_path, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
print(f"\nSaved to {out_path}")
