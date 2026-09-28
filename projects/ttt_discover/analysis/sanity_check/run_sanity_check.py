"""Sanity check runner for the 8-signal reward implementation.

Runs both sanity checks in one pass:
  1. Single-call vs separate-call comparison (halo effect measurement)
  2. Per-signal validity check (does damaging signal X only lower signal X?)

For each plan, grades with both methods, N=3 repeats each. Saves all results
and runs analysis at the end.
"""

import json
import logging
import sys
import time
from concurrent.futures import Future
from pathlib import Path

HERE = Path(__file__).parent
SRC_ROOT = HERE.parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import tinker
from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.ten_signal_reward import (
    SIGNALS,
    build_single_call_prompt,
    build_single_signal_prompt,
    parse_scores,
    median_over_repeats,
)
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)
logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen3-30B-A3B"
N_REPEATS = 2  # 2 repeats: enough for reliability check without blowing up cost
WAVE_SIZE = 32  # max parallel requests in flight at a time
PER_REQUEST_TIMEOUT = 1500  # 25 minutes per request
RESULTS_PATH = HERE / "results" / "sanity_check_results.jsonl"
ANALYSIS_PATH = HERE / "results" / "sanity_check_analysis.json"

# Plans to evaluate — unified naming aligned with signal IDs
PLAN_FILES = [
    ("00_reference", "reference", None),
    # Hard gate perturbations (not in the 8 gradient signals we grade here)
    ("P_HG1_goal_contrast", "single_perturbation", "HG1_goal_contrast"),
    ("P_HG2_claim_verif", "single_perturbation", "HG2_claim_verif"),
    # Gradient signal perturbations
    ("P_S1_depth", "single_perturbation", "S1_depth"),
    ("P_S2_rigor", "single_perturbation", "S2_rigor"),
    ("P_S3_positioning", "single_perturbation", "S3_positioning"),
    ("P_S4_significance", "single_perturbation", "S4_significance"),
    ("P_S5_stability", "single_perturbation", "S5_stability"),
    ("P_S6_failure_interp", "single_perturbation", "S6_failure_interp"),
    ("P_S7_specificity", "single_perturbation", "S7_specificity"),
    ("P_S8_scope", "single_perturbation", "S8_scope"),
    ("P_S9_focus", "single_perturbation", "S9_focus"),
    # Mixed perturbations
    ("M01_first5_bad", "mixed_perturbation", "first5"),
    ("M02_last5_bad", "mixed_perturbation", "last5"),
    ("M03_alternating", "mixed_perturbation", "alternating"),
]


def load_plans() -> list[dict]:
    """Load all plan files."""
    goal = (HERE / "research_goal.txt").read_text()
    plans = []
    for filename, ptype, target in PLAN_FILES:
        plan_text = (HERE / "perturbations" / f"{filename}.txt").read_text()
        plans.append({
            "plan_id": filename,
            "plan_type": ptype,
            "target_damaged_signal": target,
            "goal": goal,
            "plan": plan_text,
        })
    logger.info(f"Loaded {len(plans)} plans")
    return plans


def run_grading(plans: list[dict]):
    """Launch all grading calls in parallel and collect results."""
    logger.info(f"Setting up grader (model={MODEL})")
    service_client = create_service_client()
    grader_client = service_client.create_sampling_client(base_model=MODEL)
    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Renderer: {renderer_name}")

    def make_sample_fn(prompt_text: str, max_tokens: int):
        """Return a closure that, when called, submits the request and returns a Future."""
        def _submit():
            convo = [{"role": "user", "content": prompt_text}]
            model_input = renderer.build_generation_prompt(convo)
            return grader_client.sample(
                model_input,
                num_samples=1,
                sampling_params=tinker.types.SamplingParams(
                    max_tokens=max_tokens,
                    temperature=0.0,
                ),
            )
        return _submit

    # Build the request queue (deferred submission so we can wave-batch)
    # Each entry: (plan_idx, method, repeat, signal_id_or_None, submit_fn)
    #
    # NOTE: we only run separate_call mode in this sanity check pass.
    # single_call was compared vs separate_call in the previous sanity check
    # (projects/ttt_discover/analysis/sanity_check/SANITY_CHECK_REPORT.md) —
    # separate_call won on halo effect (PC1 0.394 vs 0.523) and is the chosen
    # method going forward. Re-running single_call here would only add cost
    # without giving new information.
    requests = []
    for p_idx, plan in enumerate(plans):
        for rep in range(N_REPEATS):
            for spec in SIGNALS:
                prompt = build_single_signal_prompt(plan["goal"], plan["plan"], spec)
                requests.append((p_idx, "separate_call", rep, spec.id, make_sample_fn(prompt, 2048)))

    total_requests = len(requests)
    logger.info(f"Built {total_requests} requests. Running in waves of {WAVE_SIZE}...")

    # Collect results into nested dict (only separate_call in this pass)
    raw_results = {p_idx: {"separate_call": {}} for p_idx in range(len(plans))}

    completed = 0
    n_errors = 0
    t_start = time.time()

    for wave_idx in range(0, total_requests, WAVE_SIZE):
        wave = requests[wave_idx : wave_idx + WAVE_SIZE]
        wave_num = wave_idx // WAVE_SIZE + 1
        n_waves = (total_requests + WAVE_SIZE - 1) // WAVE_SIZE

        wave_start = time.time()
        logger.info(f"Launching wave {wave_num}/{n_waves} ({len(wave)} requests)")

        # Submit all requests in this wave (get futures)
        submitted = []
        for p_idx, method, rep, signal_id, submit_fn in wave:
            try:
                fut = submit_fn()
                submitted.append((p_idx, method, rep, signal_id, fut))
            except Exception as e:
                logger.error(f"Submit failed p{p_idx} {method} rep{rep} sig={signal_id}: {e}")
                n_errors += 1
                completed += 1

        # Collect all results in this wave
        for p_idx, method, rep, signal_id, fut in submitted:
            try:
                result = fut.result(timeout=PER_REQUEST_TIMEOUT)
                raw_output = renderers.get_text_content(
                    renderer.parse_response(result.sequences[0].tokens)[0]
                )
                scores = parse_scores(raw_output)

                if rep not in raw_results[p_idx][method]:
                    raw_results[p_idx][method][rep] = {}

                if method == "single_call":
                    raw_results[p_idx][method][rep] = scores
                else:
                    if signal_id in scores:
                        raw_results[p_idx][method][rep][signal_id] = scores[signal_id]
            except Exception as e:
                err_str = str(e) or type(e).__name__
                logger.error(f"Collect failed p{p_idx} {method} rep{rep} sig={signal_id}: {err_str}")
                n_errors += 1
            completed += 1

        wave_elapsed = time.time() - wave_start
        total_elapsed = time.time() - t_start
        eta = total_elapsed / completed * (total_requests - completed) if completed > 0 else 0
        logger.info(
            f"Wave {wave_num}/{n_waves} complete in {wave_elapsed:.0f}s. "
            f"Progress: {completed}/{total_requests} "
            f"(errors: {n_errors}, total elapsed: {total_elapsed:.0f}s, ETA: {eta:.0f}s)"
        )

    logger.info(f"All requests complete in {time.time() - t_start:.0f}s")

    # Aggregate: median over repeats for each (plan, method)
    final_results = []
    methods_used = ("separate_call",)
    for p_idx, plan in enumerate(plans):
        for method in methods_used:
            repeats = raw_results[p_idx][method]
            # Convert to list format for median_over_repeats
            repeat_list = [repeats.get(r, {}) for r in range(N_REPEATS)]
            median_scores = median_over_repeats(repeat_list)

            # Also compute per-repeat scores for reliability analysis
            per_repeat = {}
            for r in range(N_REPEATS):
                per_repeat[f"rep{r}"] = {
                    spec.id: (repeats.get(r, {}).get(spec.id) or {}).get("score")
                    for spec in SIGNALS
                }

            final_results.append({
                "plan_id": plan["plan_id"],
                "plan_type": plan["plan_type"],
                "target_damaged_signal": plan["target_damaged_signal"],
                "method": method,
                "median_scores": median_scores,
                "per_repeat_scores": per_repeat,
            })

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        for r in final_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(final_results)} rows to {RESULTS_PATH}")
    return final_results


def run_analysis(results: list[dict] | None = None):
    """Analyze sanity check results.

    Checks:
      (1) Does reference score near the top on all signals?
      (2) For each single-perturbation, does the targeted signal drop significantly?
      (3) For each single-perturbation, do non-targeted signals stay similar?
      (4) How do single-call and separate-call methods compare?
      (5) What is the reliability across repeats?
    """
    import numpy as np

    if results is None:
        results = []
        with open(RESULTS_PATH) as f:
            for line in f:
                results.append(json.loads(line))
        logger.info(f"Loaded {len(results)} results from {RESULTS_PATH}")

    dim_ids = [s.id for s in SIGNALS]

    # Index: results_by_plan[plan_id][method] = {median_scores, per_repeat_scores}
    results_by_plan = {}
    for r in results:
        results_by_plan.setdefault(r["plan_id"], {})[r["method"]] = r

    analysis = {
        "method_comparison": {},
        "per_signal_validity": {},
        "halo_effect": {},
        "reliability": {},
        "summary": {},
    }

    METHODS = ("separate_call",)

    # ---- 1. Reference scores ----
    ref_separate = results_by_plan.get("00_reference", {}).get("separate_call", {}).get("median_scores", {})
    analysis["reference_scores"] = {
        "separate_call": ref_separate,
    }

    # ---- 2. Per-signal validity: for each P_S* plan, check if targeted signal dropped ----
    # Only the 8 gradient signals are graded; hard-gate perturbations (P_HG*)
    # are excluded because they target programmatic checks, not the LLM-scored signals.
    per_signal_validity = {}
    single_perturbation_ids = {
        "P_S1_depth", "P_S2_rigor", "P_S3_positioning", "P_S4_significance",
        "P_S5_stability", "P_S6_failure_interp", "P_S7_specificity", "P_S8_scope",
        "P_S9_focus",
    }
    for plan_id, plan_results in results_by_plan.items():
        if plan_id not in single_perturbation_ids:
            continue
        target = None
        for filename, ptype, t in PLAN_FILES:
            if filename == plan_id:
                target = t
                break
        if target not in dim_ids:
            continue

        for method in METHODS:
            perturbed = plan_results.get(method, {}).get("median_scores", {})
            reference = results_by_plan.get("00_reference", {}).get(method, {}).get("median_scores", {})

            targeted_drop = (reference.get(target) or 0) - (perturbed.get(target) or 0)
            non_targeted_drops = []
            for dim in dim_ids:
                if dim == target:
                    continue
                drop = (reference.get(dim) or 0) - (perturbed.get(dim) or 0)
                non_targeted_drops.append(drop)
            avg_non_targeted_drop = float(np.mean(non_targeted_drops)) if non_targeted_drops else 0
            max_non_targeted_drop = float(max(non_targeted_drops)) if non_targeted_drops else 0

            per_signal_validity.setdefault(plan_id, {})[method] = {
                "target": target,
                "ref_target_score": reference.get(target),
                "perturbed_target_score": perturbed.get(target),
                "targeted_drop": targeted_drop,
                "avg_non_targeted_drop": round(avg_non_targeted_drop, 3),
                "max_non_targeted_drop": max_non_targeted_drop,
                "specificity_ratio": round(
                    targeted_drop / (avg_non_targeted_drop + 0.01), 3
                ),  # high = well-localized damage
            }
    analysis["per_signal_validity"] = per_signal_validity

    # ---- 3. Halo effect: inter-signal correlation across plans ----
    for method in METHODS:
        matrix = []
        for plan_id in sorted(results_by_plan.keys()):
            scores = results_by_plan[plan_id].get(method, {}).get("median_scores", {})
            row = [scores.get(dim) if scores.get(dim) is not None else np.nan for dim in dim_ids]
            matrix.append(row)
        X = np.array(matrix, dtype=float)

        # Correlation matrix
        corr = np.full((len(dim_ids), len(dim_ids)), np.nan)
        for i in range(len(dim_ids)):
            for j in range(len(dim_ids)):
                mask = ~np.isnan(X[:, i]) & ~np.isnan(X[:, j])
                if mask.sum() < 3 or np.std(X[mask, i]) < 1e-6 or np.std(X[mask, j]) < 1e-6:
                    continue
                corr[i, j] = float(np.corrcoef(X[mask, i], X[mask, j])[0, 1])

        # Mean inter-correlation per dimension
        mean_corr = {}
        for i, dim in enumerate(dim_ids):
            others = [corr[i, j] for j in range(len(dim_ids)) if i != j and not np.isnan(corr[i, j])]
            mean_corr[dim] = round(float(np.mean(others)), 3) if others else None

        # PCA
        complete = ~np.any(np.isnan(X), axis=1)
        pca_info = {}
        if complete.sum() >= 5:
            Xc = X[complete] - X[complete].mean(axis=0)
            if Xc.shape[0] > 1:
                cov = np.cov(Xc.T)
                eig = np.linalg.eigvalsh(cov)[::-1]
                total = eig.sum()
                if total > 0:
                    pca_info["top_eigenvalues"] = [round(float(e), 3) for e in eig[:5]]
                    pca_info["pc1_variance_ratio"] = round(float(eig[0] / total), 3)

        analysis["halo_effect"][method] = {
            "mean_inter_correlation": mean_corr,
            "overall_mean_inter_corr": round(
                float(np.mean([v for v in mean_corr.values() if v is not None])), 3
            ),
            "pca": pca_info,
        }

    # ---- 4. Reliability: std across repeats for each (plan, method, signal) ----
    reliability = {}
    for method in METHODS:
        plan_stds = []
        for plan_id, plan_results in results_by_plan.items():
            per_repeat = plan_results.get(method, {}).get("per_repeat_scores", {})
            for dim in dim_ids:
                vals = [per_repeat.get(f"rep{r}", {}).get(dim) for r in range(N_REPEATS)]
                vals = [v for v in vals if v is not None]
                if len(vals) >= 2:
                    std = float(np.std(vals))
                    plan_stds.append(std)
        if plan_stds:
            reliability[method] = {
                "mean_repeat_std": round(float(np.mean(plan_stds)), 3),
                "max_repeat_std": round(float(max(plan_stds)), 3),
                "frac_zero_std": round(float(np.mean([s == 0 for s in plan_stds])), 3),
            }
    analysis["reliability"] = reliability

    # ---- 5. Method comparison summary ----
    method_cmp = {}
    for method in METHODS:
        validities = per_signal_validity
        ratios = []
        for plan_id, method_data in validities.items():
            m = method_data.get(method, {})
            if m.get("specificity_ratio") is not None:
                ratios.append(m["specificity_ratio"])
        method_cmp[method] = {
            "avg_specificity_ratio": round(float(np.mean(ratios)), 3) if ratios else None,
            "halo_mean_corr": analysis["halo_effect"][method]["overall_mean_inter_corr"],
            "halo_pc1_ratio": analysis["halo_effect"][method]["pca"].get("pc1_variance_ratio"),
            "mean_repeat_std": reliability.get(method, {}).get("mean_repeat_std"),
        }
    analysis["method_comparison"] = method_cmp

    # ---- Save analysis ----
    with open(ANALYSIS_PATH, "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    logger.info(f"Analysis saved to {ANALYSIS_PATH}")

    # ---- Print summary ----
    print("\n" + "=" * 72)
    print("SANITY CHECK RESULTS")
    print("=" * 72)

    print("\n--- Reference scores (should be high on all signals) ---")
    print(f"{'Signal':<25} {'separate_call':>14}")
    for dim in dim_ids:
        s2 = ref_separate.get(dim, "-")
        print(f"{dim:<25} {str(s2):>14}")

    print("\n--- Per-signal validity (damaged signal should drop, others stay) ---")
    print(f"{'Plan':<25} {'Target':<20} {'TargDrop':>9} {'OtherDrop':>10} {'Ratio':>7}")
    for plan_id in sorted(per_signal_validity.keys()):
        for method in METHODS:
            v = per_signal_validity[plan_id].get(method, {})
            if not v:
                continue
            print(
                f"{plan_id:<25} {v['target']:<20} "
                f"{v['targeted_drop']:>9} {v['avg_non_targeted_drop']:>10.2f} "
                f"{v['specificity_ratio']:>7.2f}"
            )

    print("\n--- Halo effect ---")
    for method in METHODS:
        halo = analysis["halo_effect"][method]
        print(f"  {method}:")
        print(f"    Mean inter-signal correlation: {halo['overall_mean_inter_corr']}")
        print(f"    PC1 variance ratio: {halo['pca'].get('pc1_variance_ratio')}")

    print(f"\n--- Reliability (std across N={N_REPEATS} repeats) ---")
    for method, r in reliability.items():
        print(f"  {method}: mean_std={r['mean_repeat_std']}, "
              f"frac_zero_std={r['frac_zero_std']}")

    print("\n--- Method summary ---")
    header = f"{'Metric':<30}" + "".join(f"{m:>16}" for m in METHODS)
    print(header)
    for metric in ("avg_specificity_ratio", "halo_mean_corr", "halo_pc1_ratio", "mean_repeat_std"):
        row = f"{metric:<30}"
        for method in METHODS:
            v = method_cmp.get(method, {}).get(metric, "-")
            row += f"{str(v):>16}"
        print(row)

    print("\n" + "=" * 72)
    return analysis


if __name__ == "__main__":
    if "--analyze-only" in sys.argv:
        run_analysis()
    else:
        plans = load_plans()
        run_grading(plans)
        run_analysis()
