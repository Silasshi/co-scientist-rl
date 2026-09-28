"""
Empirical validation of the three-layer research plan rubric dimensions.

Evaluates 20 scoring dimensions (10 universal core + 10 computation/ML)
on ~40 research plans spanning the quality spectrum.

Goal: identify which dimensions are
  (a) LLM-gradable with sufficient variance (not degenerate)
  (b) mutually orthogonal (low inter-dimension correlation)
  (c) externally valid (correlate with known rubric scores)

Usage:
    python validate_dimensions.py                     # run grading
    python validate_dimensions.py --analyze-only      # skip grading, analyze existing results
"""

import json
import logging
import re
import sys
import time
from concurrent.futures import as_completed, Future
from dataclasses import dataclass, field
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import tinker
from co_scientist.shared.api_profiles import create_service_client
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)

OUT_DIR = Path(__file__).parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
PILOT_PATH = PROJECT_ROOT / "shared" / "analysis" / "results" / "refinement_pilot.jsonl"
RESULTS_PATH = OUT_DIR / "dimension_scores.jsonl"
ANALYSIS_PATH = OUT_DIR / "dimension_analysis.json"

MODEL = "Qwen/Qwen3-30B-A3B"

# ============================================================
# Dimension definitions
# ============================================================

CORE_DIMENSIONS = [
    {
        "id": "C1_importance",
        "name": "Problem Importance",
        "prompt": (
            "Is this research goal genuinely important? Does it address a real bottleneck, "
            "fill a clear knowledge gap, or have the potential to change understanding, methods, "
            "or capabilities in the field? Would success materially advance the field?"
        ),
    },
    {
        "id": "C2_clarity",
        "name": "Research Question Clarity",
        "prompt": (
            "Does the plan clearly state what question it aims to answer? Is the core claim "
            "or hypothesis explicit? Are the boundaries of the contribution clear? Can you "
            "distinguish 'what is being done' from 'why it is being done'?"
        ),
    },
    {
        "id": "C3_positioning",
        "name": "Literature Positioning & Motivation",
        "prompt": (
            "Does the plan identify a specific gap, failure mode, or unresolved tension in "
            "existing work? Does it accurately locate its novelty relative to prior work? "
            "Does it explain why existing solutions are insufficient — not just that 'no one "
            "has done this before'?"
        ),
    },
    {
        "id": "C4_plausibility",
        "name": "Plausibility",
        "prompt": (
            "Is there a credible reason to believe this direction could work? Does the idea "
            "have a clear mechanistic explanation? Is the hypothesis consistent with existing "
            "knowledge? Is there any preliminary argument or evidence supporting the approach, "
            "rather than just wishful thinking?"
        ),
    },
    {
        "id": "C5_feasibility",
        "name": "Feasibility & Execution Path",
        "prompt": (
            "Is this plan actually executable? Are the steps clear and concrete? Are resource "
            "requirements realistic? Are key dependencies and prerequisites identified? Is "
            "there a reasonable timeline with milestones?"
        ),
    },
    {
        "id": "C6_discriminative",
        "name": "Discriminative Power of Results",
        "prompt": (
            "After executing this plan, will the results actually change our beliefs? Can "
            "positive results clearly support the hypothesis? Can negative results clearly "
            "refute it? Does the design distinguish between competing explanations, or would "
            "any outcome be ambiguous?"
        ),
    },
    {
        "id": "C7_alignment",
        "name": "Evaluation-Goal Alignment",
        "prompt": (
            "Are the proposed metrics, measurements, and evaluation criteria actually aligned "
            "with the core research question? Do they measure what matters, or just what is "
            "convenient? Are proxies justified? Does the evaluation support the claims the "
            "plan intends to make?"
        ),
    },
    {
        "id": "C8_risk",
        "name": "Risk Identification & Contingency",
        "prompt": (
            "Does the plan identify its main failure risks? Are there stopping rules? Are "
            "there alternative paths if the primary approach fails? Does the plan avoid "
            "assuming everything will go as expected?"
        ),
    },
    {
        "id": "C9_leverage",
        "name": "Research Leverage",
        "prompt": (
            "Even if this plan fails, would it still produce valuable information? Would a "
            "negative result rule out a class of wrong directions? Would the plan produce "
            "reusable tools, datasets, methods, or insights regardless of the main outcome? "
            "Is the information gain high even in the failure case?"
        ),
    },
    {
        "id": "C10_cost_benefit",
        "name": "Cost-Benefit Ratio",
        "prompt": (
            "Is the investment of time, compute, samples, funding, or equipment proportional "
            "to the potential value? Is there a cheaper way to get the same core insight? "
            "Does the plan avoid committing heavy resources before validating key assumptions?"
        ),
    },
]

ML_DIMENSIONS = [
    {
        "id": "M1_failure_targeting",
        "name": "Failure Mode Targeting",
        "prompt": (
            "Does this idea specifically target a concrete, identified problem — rather than "
            "just 'adding a module to see if it helps'? Is the failure mode or limitation "
            "being addressed clearly diagnosed?"
        ),
    },
    {
        "id": "M2_experiment_supports_claim",
        "name": "Experiment Design Supports Claim",
        "prompt": (
            "Can the proposed experiments actually demonstrate WHY the method works (or doesn't), "
            "not just that the score is higher? Does the experimental design go beyond 'my "
            "number is bigger than your number'?"
        ),
    },
    {
        "id": "M3_baselines",
        "name": "Baseline Adequacy",
        "prompt": (
            "Does the plan compare against strong, relevant baselines — not cherry-picked "
            "weak ones? Are the baselines appropriate for the specific claim being made? "
            "Would a reviewer consider the comparisons fair?"
        ),
    },
    {
        "id": "M4_ablations",
        "name": "Ablation Design",
        "prompt": (
            "Does the plan include ablations that isolate which components are responsible "
            "for the improvement? Can you tell what is actually doing the work vs. what is "
            "incidental?"
        ),
    },
    {
        "id": "M5_eval_validity",
        "name": "Evaluation Validity",
        "prompt": (
            "Are the benchmarks, tasks, and metrics truly aligned with the real-world goal? "
            "Is there a risk of benchmark saturation, data leakage, or metric gaming? Do the "
            "metrics capture real capability vs. superficial patterns?"
        ),
    },
    {
        "id": "M6_robustness",
        "name": "Robustness & Variance Design",
        "prompt": (
            "Does the plan account for random seed variance, hyperparameter sensitivity, and "
            "stability across different settings? Or does it rely on a single lucky run?"
        ),
    },
    {
        "id": "M7_compute_realism",
        "name": "Compute Realism",
        "prompt": (
            "Is the compute budget realistic and proportional? Is the improvement from better "
            "methodology, or just from throwing more compute at the problem? Would the result "
            "still hold at a fair compute budget?"
        ),
    },
    {
        "id": "M8_negative_interpretability",
        "name": "Negative Result Interpretability",
        "prompt": (
            "If the method doesn't improve results, can the plan distinguish whether the idea "
            "is wrong, the implementation is buggy, the benchmark is inappropriate, or the "
            "training protocol is suboptimal? Is the failure diagnostic clear?"
        ),
    },
    {
        "id": "M9_generalization",
        "name": "Generalization & Transfer",
        "prompt": (
            "Would the results generalize beyond the specific benchmark or setting tested? "
            "Does the plan evaluate on multiple settings, or only on a single narrow setup?"
        ),
    },
    {
        "id": "M10_reproducibility",
        "name": "Reproducibility",
        "prompt": (
            "Does the plan provide enough detail for someone else to reproduce the experiments? "
            "Are there hidden engineering details that could make reproduction difficult?"
        ),
    },
]

ALL_DIMENSIONS = CORE_DIMENSIONS + ML_DIMENSIONS


def build_dimension_grading_prompt(goal: str, plan: str) -> str:
    """Build a prompt that asks the grader to score all 20 dimensions."""

    dim_block = ""
    for i, dim in enumerate(ALL_DIMENSIONS, 1):
        dim_block += f"""
    <dimension id="{dim['id']}" num="{i}">
        <name>{dim['name']}</name>
        <question>{dim['prompt']}</question>
    </dimension>"""

    prompt = f"""You are a strict, expert-level research plan evaluator. You will evaluate a research plan on {len(ALL_DIMENSIONS)} independent dimensions.

# Research Goal
{goal}

# Research Plan
{plan}

# Evaluation Dimensions
Below are {len(ALL_DIMENSIONS)} evaluation dimensions. For EACH dimension, you must:
1. Think carefully about how the plan performs on that SPECIFIC dimension.
2. Provide a brief reasoning (2-3 sentences).
3. Assign an integer score from 1 to 5 using this scale:
   - 1: Very weak. The plan barely addresses this dimension or fails completely.
   - 2: Weak. The plan touches on this but with major gaps or vagueness.
   - 3: Moderate. The plan addresses this reasonably but with notable room for improvement.
   - 4: Strong. The plan handles this dimension well with only minor gaps.
   - 5: Excellent. The plan excels on this dimension with clear, concrete, convincing content.

CRITICAL RULES:
- Evaluate each dimension INDEPENDENTLY. A plan can score high on one dimension and low on another.
- Do NOT give the same score to all dimensions. Different dimensions capture different qualities.
- Be strict and skeptical. A score of 5 should be rare. Most decent plans should cluster around 2-4.
- Base your score on what is ACTUALLY in the plan, not on what it claims about itself.
- A vague statement like "we will use appropriate baselines" scores LOW unless specific baselines are named.

{dim_block}

# Output Format

Return your evaluation in the following XML structure. You MUST evaluate ALL {len(ALL_DIMENSIONS)} dimensions.

<evaluation>
    <dim id="C1_importance">
        <reasoning>Your 2-3 sentence analysis here.</reasoning>
        <score>INTEGER 1-5</score>
    </dim>
    <dim id="C2_clarity">
        <reasoning>Your 2-3 sentence analysis here.</reasoning>
        <score>INTEGER 1-5</score>
    </dim>
    ... (continue for ALL {len(ALL_DIMENSIONS)} dimensions) ...
</evaluation>

Begin your evaluation now. Think step by step before scoring each dimension."""

    return prompt


# ============================================================
# Grading
# ============================================================

def parse_dimension_scores(xml_text: str) -> dict[str, dict]:
    """Parse XML output into per-dimension scores and reasoning."""
    results = {}
    dim_blocks = re.findall(
        r'<dim\s+id="([^"]+)">(.*?)</dim>',
        xml_text,
        flags=re.DOTALL,
    )
    for dim_id, body in dim_blocks:
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", body, re.DOTALL)
        score_match = re.search(r"<score>\s*(\d+)\s*</score>", body, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        score = int(score_match.group(1)) if score_match else None
        if score is not None and 1 <= score <= 5:
            results[dim_id] = {"score": score, "reasoning": reasoning}
        elif score is not None:
            logger.warning(f"Score out of range for {dim_id}: {score}")
            results[dim_id] = {"score": None, "reasoning": reasoning}
    return results


def load_plans() -> list[dict]:
    """Load plans from refinement_pilot.jsonl (original + refined)."""
    entries = []
    with open(PILOT_PATH) as f:
        for line in f:
            raw = json.loads(line)
            # Original plan
            entries.append({
                "plan_id": f"goal{raw['goal_idx']:02d}_original",
                "goal_idx": raw["goal_idx"],
                "variant": "original",
                "goal": raw["goal"],
                "plan": raw["original_plan"],
                "known_rubric_score": raw["original_rubric_score"],
                "word_count": raw["original_word_count"],
            })
            # Refined plan
            entries.append({
                "plan_id": f"goal{raw['goal_idx']:02d}_refined",
                "goal_idx": raw["goal_idx"],
                "variant": "refined",
                "goal": raw["goal"],
                "plan": raw["refined_plan"],
                "known_rubric_score": raw["refined_rubric_score"],
                "word_count": raw["refined_word_count"],
            })
    logger.info(f"Loaded {len(entries)} plans from refinement_pilot.jsonl")
    return entries


def run_grading(plans: list[dict], max_parallel: int = 8):
    """Grade all plans on all dimensions using Qwen3-30B."""
    logger.info(f"Setting up grader client (model={MODEL})...")
    service_client = create_service_client()
    grader_client = service_client.create_sampling_client(base_model=MODEL)

    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Renderer: {renderer_name}")

    sampling_params = tinker.types.SamplingParams(
        max_tokens=8192,
        temperature=0.0,
    )

    results = []
    # Launch all grading calls
    futures: list[tuple[int, Future]] = []

    for idx, plan_info in enumerate(plans):
        prompt_text = build_dimension_grading_prompt(
            goal=plan_info["goal"],
            plan=plan_info["plan"],
        )
        convo = [{"role": "user", "content": prompt_text}]
        model_input = renderer.build_generation_prompt(convo)

        future = grader_client.sample(
            model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        futures.append((idx, future))
        logger.info(f"Launched grading for plan {idx + 1}/{len(plans)}: {plan_info['plan_id']}")

    # Collect results
    for idx, future in futures:
        plan_info = plans[idx]
        try:
            result = future.result(timeout=900)
            raw_output = renderers.get_text_content(
                renderer.parse_response(result.sequences[0].tokens)[0]
            )
            dim_scores = parse_dimension_scores(raw_output)

            record = {
                **{k: v for k, v in plan_info.items() if k != "plan"},
                "plan_length": len(plan_info["plan"]),
                "dimensions": {
                    dim_id: info["score"]
                    for dim_id, info in dim_scores.items()
                },
                "dimension_reasoning": {
                    dim_id: info["reasoning"]
                    for dim_id, info in dim_scores.items()
                },
                "n_dimensions_parsed": len(dim_scores),
                "raw_output_length": len(raw_output),
            }
            results.append(record)
            parsed_count = len(dim_scores)
            logger.info(
                f"Completed {idx + 1}/{len(plans)}: {plan_info['plan_id']} "
                f"| {parsed_count}/{len(ALL_DIMENSIONS)} dims parsed "
                f"| known_rubric={plan_info['known_rubric_score']:.3f}"
            )
        except Exception as e:
            logger.error(f"Failed for {plan_info['plan_id']}: {e}")
            results.append({
                **{k: v for k, v in plan_info.items() if k != "plan"},
                "plan_length": len(plan_info["plan"]),
                "dimensions": {},
                "dimension_reasoning": {},
                "n_dimensions_parsed": 0,
                "error": str(e),
            })

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(results)} results to {RESULTS_PATH}")
    return results


# ============================================================
# Analysis
# ============================================================

def run_analysis(results: list[dict] | None = None):
    """Analyze dimension scores: variance, correlation, external validity."""
    import numpy as np

    if results is None:
        results = []
        with open(RESULTS_PATH) as f:
            for line in f:
                results.append(json.loads(line))
        logger.info(f"Loaded {len(results)} results from {RESULTS_PATH}")

    # Filter to successfully graded plans
    valid = [r for r in results if r["n_dimensions_parsed"] >= 15]
    logger.info(f"Using {len(valid)}/{len(results)} plans with >= 15 dimensions parsed")

    if len(valid) < 10:
        logger.error("Too few valid results for meaningful analysis")
        return

    dim_ids = [d["id"] for d in ALL_DIMENSIONS]

    # Build score matrix: plans × dimensions
    score_matrix = []
    known_scores = []
    plan_ids = []
    for r in valid:
        row = [r["dimensions"].get(d, np.nan) for d in dim_ids]
        score_matrix.append(row)
        known_scores.append(r["known_rubric_score"])
        plan_ids.append(r["plan_id"])

    X = np.array(score_matrix, dtype=float)  # (n_plans, n_dims)
    y = np.array(known_scores)               # (n_plans,)

    # ---- Per-dimension statistics ----
    dim_stats = {}
    for j, dim_id in enumerate(dim_ids):
        col = X[:, j]
        valid_mask = ~np.isnan(col)
        vals = col[valid_mask]
        if len(vals) < 5:
            dim_stats[dim_id] = {"status": "insufficient_data", "n_valid": int(len(vals))}
            continue
        dim_stats[dim_id] = {
            "n_valid": int(len(vals)),
            "mean": round(float(np.mean(vals)), 3),
            "std": round(float(np.std(vals)), 3),
            "min": int(np.min(vals)),
            "max": int(np.max(vals)),
            "median": round(float(np.median(vals)), 1),
            # Value distribution
            "dist": {str(k): int(np.sum(vals == k)) for k in range(1, 6)},
        }

    # ---- Identify degenerate dimensions (std < 0.5) ----
    degenerate = [
        d for d, s in dim_stats.items()
        if s.get("std", 999) < 0.5 and s.get("status") != "insufficient_data"
    ]

    # ---- Correlation matrix ----
    # Use pairwise complete observations
    n_dims = len(dim_ids)
    corr_matrix = np.full((n_dims, n_dims), np.nan)
    for i in range(n_dims):
        for j in range(n_dims):
            mask = ~np.isnan(X[:, i]) & ~np.isnan(X[:, j])
            if mask.sum() < 5:
                continue
            xi, xj = X[mask, i], X[mask, j]
            if np.std(xi) < 1e-6 or np.std(xj) < 1e-6:
                continue
            corr_matrix[i, j] = float(np.corrcoef(xi, xj)[0, 1])

    # Find high-correlation pairs (|r| > 0.7)
    high_corr_pairs = []
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            r = corr_matrix[i, j]
            if not np.isnan(r) and abs(r) > 0.7:
                high_corr_pairs.append({
                    "dim_a": dim_ids[i],
                    "dim_b": dim_ids[j],
                    "correlation": round(float(r), 3),
                })
    high_corr_pairs.sort(key=lambda x: -abs(x["correlation"]))

    # ---- External validity: correlation with known rubric scores ----
    external_validity = {}
    for j, dim_id in enumerate(dim_ids):
        mask = ~np.isnan(X[:, j])
        if mask.sum() < 5:
            continue
        xj = X[mask, j]
        yj = y[mask]
        if np.std(xj) < 1e-6:
            continue
        r = float(np.corrcoef(xj, yj)[0, 1])
        external_validity[dim_id] = round(r, 3)

    # ---- Mean dimension score correlation with known rubric ----
    mean_dim_scores = np.nanmean(X, axis=1)
    valid_mask = ~np.isnan(mean_dim_scores)
    if valid_mask.sum() >= 5:
        overall_r = float(np.corrcoef(mean_dim_scores[valid_mask], y[valid_mask])[0, 1])
    else:
        overall_r = None

    # ---- PCA-like: how many effective dimensions? ----
    # Use complete cases only
    complete_mask = ~np.any(np.isnan(X), axis=1)
    if complete_mask.sum() >= 10:
        X_complete = X[complete_mask]
        X_centered = X_complete - X_complete.mean(axis=0)
        cov = np.cov(X_centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)[::-1]
        total_var = eigenvalues.sum()
        cumvar = np.cumsum(eigenvalues) / total_var if total_var > 0 else eigenvalues
        n_for_90 = int(np.searchsorted(cumvar, 0.9)) + 1
        n_for_80 = int(np.searchsorted(cumvar, 0.8)) + 1
        pca_info = {
            "n_complete_cases": int(complete_mask.sum()),
            "eigenvalues": [round(float(e), 3) for e in eigenvalues[:10]],
            "cumulative_variance_ratio": [round(float(c), 3) for c in cumvar[:10]],
            "dims_for_80pct_variance": n_for_80,
            "dims_for_90pct_variance": n_for_90,
        }
    else:
        pca_info = {"error": "Too few complete cases for PCA"}

    # ---- Compile analysis ----
    analysis = {
        "summary": {
            "n_plans_total": len(results),
            "n_plans_valid": len(valid),
            "n_dimensions": len(dim_ids),
            "n_degenerate_dims": len(degenerate),
            "n_high_corr_pairs": len(high_corr_pairs),
            "overall_mean_dim_vs_rubric_r": round(overall_r, 3) if overall_r else None,
        },
        "per_dimension": dim_stats,
        "degenerate_dimensions": degenerate,
        "high_correlation_pairs": high_corr_pairs,
        "external_validity": external_validity,
        "pca": pca_info,
        "correlation_matrix": {
            "dim_ids": dim_ids,
            "values": [
                [round(float(corr_matrix[i, j]), 3) if not np.isnan(corr_matrix[i, j]) else None
                 for j in range(n_dims)]
                for i in range(n_dims)
            ],
        },
    }

    with open(ANALYSIS_PATH, "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    logger.info(f"Analysis saved to {ANALYSIS_PATH}")

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("DIMENSION VALIDATION RESULTS")
    print("=" * 70)

    print(f"\nPlans: {len(valid)} valid / {len(results)} total")
    print(f"Dimensions: {len(dim_ids)}")

    print("\n--- Per-Dimension Statistics ---")
    print(f"{'Dimension':<30} {'Mean':>5} {'Std':>5} {'Min':>4} {'Max':>4} {'ExtR':>6}")
    print("-" * 60)
    for dim_id in dim_ids:
        s = dim_stats.get(dim_id, {})
        ext_r = external_validity.get(dim_id, None)
        if s.get("status") == "insufficient_data":
            print(f"{dim_id:<30}  [insufficient data]")
        else:
            ext_str = f"{ext_r:>6.3f}" if ext_r is not None else "   N/A"
            print(
                f"{dim_id:<30} "
                f"{s.get('mean', 0):>5.2f} {s.get('std', 0):>5.2f} "
                f"{s.get('min', 0):>4} {s.get('max', 0):>4} "
                f"{ext_str}"
            )

    if degenerate:
        print(f"\n--- Degenerate Dimensions (std < 0.5) ---")
        for d in degenerate:
            print(f"  {d}: std={dim_stats[d]['std']}")

    if high_corr_pairs:
        print(f"\n--- High Correlation Pairs (|r| > 0.7) ---")
        for p in high_corr_pairs[:15]:
            print(f"  {p['dim_a']} <-> {p['dim_b']}: r={p['correlation']}")

    print(f"\n--- External Validity (corr with known rubric score) ---")
    sorted_ext = sorted(external_validity.items(), key=lambda x: -abs(x[1]))
    for dim_id, r in sorted_ext:
        marker = " ***" if abs(r) > 0.5 else " **" if abs(r) > 0.3 else ""
        print(f"  {dim_id:<30} r={r:>6.3f}{marker}")
    if overall_r is not None:
        print(f"\n  Mean of all dimensions vs rubric: r={overall_r:.3f}")

    print(f"\n--- Effective Dimensionality (PCA) ---")
    if "error" not in pca_info:
        print(f"  Complete cases: {pca_info['n_complete_cases']}")
        print(f"  Dims for 80% variance: {pca_info['dims_for_80pct_variance']}")
        print(f"  Dims for 90% variance: {pca_info['dims_for_90pct_variance']}")
        print(f"  Top eigenvalues: {pca_info['eigenvalues'][:5]}")
    else:
        print(f"  {pca_info['error']}")

    print("\n" + "=" * 70)
    return analysis


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    analyze_only = "--analyze-only" in sys.argv

    if analyze_only:
        run_analysis()
    else:
        plans = load_plans()
        results = run_grading(plans)
        run_analysis(results)
