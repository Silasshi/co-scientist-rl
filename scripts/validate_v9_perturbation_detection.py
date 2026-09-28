"""Perturbation detection test for v9 signals (S2a_formalism, SA_arithmetic).

Grades reference + two perturbations (P_S2a, P_SA) with 5 repeats each.
Detection = fraction of repeats where perturbed score < reference score.
Target: >=70% detection for both signals.
"""
import sys
import time
import statistics
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

PERT_DIR = ROOT / "projects" / "ttt_discover" / "analysis" / "sanity_check" / "perturbations"
GOAL_PATH = ROOT / "projects" / "ttt_discover" / "analysis" / "sanity_check" / "research_goal.txt"

FILES = {
    "reference": PERT_DIR / "00_reference.txt",
    "P_S2a":     PERT_DIR / "P_S2a_no_formulas.txt",
    "P_SA":      PERT_DIR / "P_SA_arithmetic_errors.txt",
}

N_REPEATS = 5
TARGET_SIGNALS = {
    "P_S2a": "S2a_formalism",
    "P_SA":  "SA_arithmetic",
}


def grade_n_repeats(plan_text, goal, signal_spec, client, renderer, tokenizer, n=5):
    scores = []
    for i in range(n):
        prompt = build_single_signal_prompt(goal, plan_text, signal_spec)
        convo = [{"role": "user", "content": prompt}]
        model_input = renderer.build_generation_prompt(convo)
        result = client.sample(
            model_input, num_samples=1,
            sampling_params=tinker.types.SamplingParams(
                max_tokens=4096, temperature=0.0,
                stop=renderer.get_stop_sequences(),
            ),
        )
        raw = tokenizer.decode(result.result().sequences[0].tokens)
        parsed = parse_scores(raw)
        s = parsed.get(signal_spec.id, {}).get("score")
        scores.append(s)
        print(f"    repeat {i+1}: {s}/5", flush=True)
    return scores


def main():
    goal = GOAL_PATH.read_text().strip()
    service_client = create_service_client(api_profile="new")

    model_30b = "Qwen/Qwen3-30B-A3B"
    model_235b = "Qwen/Qwen3-235B-A22B-Instruct-2507"

    tokenizer_30b = get_tokenizer(model_30b)
    renderer_30b = renderers.get_renderer(
        model_info.get_recommended_renderer_name(model_30b), tokenizer_30b)
    client_30b = service_client.create_sampling_client(base_model=model_30b)

    tokenizer_235b = get_tokenizer(model_235b)
    renderer_235b = renderers.get_renderer("qwen3", tokenizer_235b)
    client_235b = service_client.create_sampling_client(base_model=model_235b)

    specs = {s.id: s for s in SIGNALS}

    grader_map = {
        "S2a_formalism": (client_30b, renderer_30b, tokenizer_30b),
        "SA_arithmetic": (client_235b, renderer_235b, tokenizer_235b),
    }

    # Grade reference on both signals
    ref_text = FILES["reference"].read_text().strip()
    ref_scores = {}
    for sig_id in TARGET_SIGNALS.values():
        client, rend, tok = grader_map[sig_id]
        print(f"\nReference × {sig_id} ({N_REPEATS} repeats):", flush=True)
        ref_scores[sig_id] = grade_n_repeats(ref_text, goal, specs[sig_id], client, rend, tok, N_REPEATS)

    # Grade perturbations on target signal
    pert_scores = {}
    for pert_name, sig_id in TARGET_SIGNALS.items():
        pert_text = FILES[pert_name].read_text().strip()
        client, rend, tok = grader_map[sig_id]
        print(f"\n{pert_name} × {sig_id} ({N_REPEATS} repeats):", flush=True)
        pert_scores[pert_name] = grade_n_repeats(pert_text, goal, specs[sig_id], client, rend, tok, N_REPEATS)

    # Compute detection rates
    print("\n" + "=" * 50)
    print("PERTURBATION DETECTION RESULTS")
    print("=" * 50)

    for pert_name, sig_id in TARGET_SIGNALS.items():
        ref = ref_scores[sig_id]
        pert = pert_scores[pert_name]
        ref_valid = [s for s in ref if s is not None]
        pert_valid = [s for s in pert if s is not None]

        ref_median = statistics.median_low(ref_valid) if ref_valid else None
        pert_median = statistics.median_low(pert_valid) if pert_valid else None

        detected = sum(1 for p, r in zip(pert, ref) if p is not None and r is not None and p < r)
        total_pairs = sum(1 for p, r in zip(pert, ref) if p is not None and r is not None)
        rate = detected / total_pairs if total_pairs > 0 else 0

        print(f"\n{pert_name} → {sig_id}:")
        print(f"  Reference scores: {ref} (median {ref_median})")
        print(f"  Perturbed scores: {pert} (median {pert_median})")
        print(f"  Detection: {detected}/{total_pairs} = {rate:.0%}")
        print(f"  {'PASS ✓' if rate >= 0.70 else 'FAIL ✗'} (threshold: ≥70%)")


if __name__ == "__main__":
    main()
