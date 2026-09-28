"""Validate v9 signals (S2a_formalism, SA_arithmetic) on Opus-audited plans.

Grades 6 plans (5 best + 1 reference) with the two new signals.
Compares against known Opus depth scores to check rank correlation.

Go/no-go: Spearman ρ(S2a, Opus Math) > 0.3.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.ten_signal_reward import (
    SIGNALS,
    build_single_signal_prompt,
    parse_scores,
)

PLANS_DIR = ROOT / "projects" / "ttt_discover" / "analysis" / "external_judge_eval"
REFERENCE_PLAN = ROOT / "projects" / "ttt_discover" / "analysis" / "sanity_check" / "perturbations" / "00_reference.txt"
GOAL_PATH = ROOT / "projects" / "ttt_discover" / "analysis" / "sanity_check" / "research_goal.txt"

PLAN_FILES = {
    "MAIN_v6": PLANS_DIR / "MAIN_v6_minimal_best_plan.txt",
    "B4_v6":   PLANS_DIR / "B4_v6_minimal_best_plan.txt",
    "B1_v6":   PLANS_DIR / "B1_v6_minimal_best_plan.txt",
    "MAIN_v7": PLANS_DIR / "MAIN_v7_per_signal_best_plan.txt",
    "B4_v7":   PLANS_DIR / "B4_v7_per_signal_best_plan.txt",
    "Reference": REFERENCE_PLAN,
}

# Known Opus 4.7 scores (from opus_judge_report.md + opus_judge_report_v7.md)
# Format: {plan_name: {math: int, novelty: int, realism: int, rigor: int, total: int}}
OPUS_SCORES = {
    "MAIN_v7": {"math": 1, "novelty": 2, "realism": 2, "rigor": 2, "total": 7},
    "B4_v7":   {"math": 2, "novelty": 2, "realism": 2, "rigor": 2, "total": 8},
    "Reference": {"math": 5, "novelty": 5, "realism": 5, "rigor": 5, "total": 20},
}

V9_SIGNAL_IDS = ["S2a_formalism", "SA_arithmetic"]


def grade_plan(plan_text: str, goal: str, signal_spec, client, renderer, tokenizer,
               n_repeats: int = 2, max_tokens: int = 4096):
    """Grade a plan on a single signal, return median score."""
    prompt = build_single_signal_prompt(goal, plan_text, signal_spec)
    scores = []
    for _ in range(n_repeats):
        convo = [{"role": "user", "content": prompt}]
        model_input = renderer.build_generation_prompt(convo)
        result = client.sample(
            model_input,
            num_samples=1,
            sampling_params=tinker.types.SamplingParams(
                max_tokens=max_tokens, temperature=0.0,
                stop=renderer.get_stop_sequences(),
            ),
        )
        raw = tokenizer.decode(result.result().sequences[0].tokens)
        parsed = parse_scores(raw)
        if signal_spec.id in parsed and parsed[signal_spec.id]["score"] is not None:
            scores.append(parsed[signal_spec.id]["score"])
    if scores:
        import statistics
        return statistics.median_low(scores)
    return None


def main():
    goal = GOAL_PATH.read_text().strip()

    # Create clients
    service_client = create_service_client(api_profile="new")

    model_30b = "Qwen/Qwen3-30B-A3B"
    model_235b = "Qwen/Qwen3-235B-A22B-Instruct-2507"

    tokenizer_30b = get_tokenizer(model_30b)
    renderer_name_30b = model_info.get_recommended_renderer_name(model_30b)
    renderer_30b = renderers.get_renderer(renderer_name_30b, tokenizer_30b)
    client_30b = service_client.create_sampling_client(base_model=model_30b)

    tokenizer_235b = get_tokenizer(model_235b)
    renderer_235b = renderers.get_renderer("qwen3", tokenizer_235b)
    client_235b = service_client.create_sampling_client(base_model=model_235b)

    # Map signal ID → (client, renderer, tokenizer)
    v9_specs = [s for s in SIGNALS if s.id in V9_SIGNAL_IDS]
    grader_map = {}
    for spec in v9_specs:
        if spec.grader_model_override and "235B" in spec.grader_model_override:
            grader_map[spec.id] = (client_235b, renderer_235b, tokenizer_235b)
        else:
            grader_map[spec.id] = (client_30b, renderer_30b, tokenizer_30b)

    # Grade all plans
    results = {}
    for plan_name, plan_path in PLAN_FILES.items():
        if not plan_path.exists():
            print(f"SKIP {plan_name}: {plan_path} not found")
            continue
        plan_text = plan_path.read_text().strip()
        print(f"\nGrading {plan_name} ({len(plan_text)} chars)...")
        results[plan_name] = {}
        for spec in v9_specs:
            client, rend, tok = grader_map[spec.id]
            t0 = time.time()
            score = grade_plan(plan_text, goal, spec, client, rend, tok)
            dt = time.time() - t0
            results[plan_name][spec.id] = score
            print(f"  {spec.id}: {score}/5  ({dt:.1f}s)")

    # Print results table
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS: v9 signals vs Opus depth scores")
    print("=" * 60)
    print(f"\n{'Plan':<15} {'S2a_form':>10} {'SA_arith':>10} {'Opus_Math':>10} {'Opus_Real':>10} {'Opus_Total':>11}")
    print("-" * 66)
    for name in PLAN_FILES:
        if name not in results:
            continue
        s2a = results[name].get("S2a_formalism", "?")
        sa = results[name].get("SA_arithmetic", "?")
        opus = OPUS_SCORES.get(name, {})
        om = opus.get("math", "?")
        orealism = opus.get("realism", "?")
        ototal = opus.get("total", "?")
        print(f"{name:<15} {s2a:>10} {sa:>10} {om:>10} {orealism:>10} {ototal:>11}")

    # Compute rank correlation where Opus scores are available
    from scipy import stats
    plans_with_opus = [n for n in results if n in OPUS_SCORES]
    if len(plans_with_opus) >= 3:
        s2a_scores = [results[n]["S2a_formalism"] for n in plans_with_opus]
        opus_math = [OPUS_SCORES[n]["math"] for n in plans_with_opus]
        if all(s is not None for s in s2a_scores):
            rho, p = stats.spearmanr(s2a_scores, opus_math)
            print(f"\nSpearman ρ(S2a_formalism, Opus Math): {rho:.3f}  (p={p:.3f})")
            print(f"  GO/NO-GO: {'PASS ✓' if rho > 0.3 else 'FAIL ✗'} (threshold: ρ > 0.3)")

        sa_scores = [results[n]["SA_arithmetic"] for n in plans_with_opus]
        opus_realism = [OPUS_SCORES[n]["realism"] for n in plans_with_opus]
        if all(s is not None for s in sa_scores):
            rho_sa, p_sa = stats.spearmanr(sa_scores, opus_realism)
            print(f"\nSpearman ρ(SA_arithmetic, Opus Realism): {rho_sa:.3f}  (p={p_sa:.3f})")
    else:
        print("\nNot enough Opus-scored plans for rank correlation.")

    # Save raw results
    out_path = ROOT / "projects" / "ttt_discover" / "analysis" / "v9_validation_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "opus_scores": OPUS_SCORES}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
