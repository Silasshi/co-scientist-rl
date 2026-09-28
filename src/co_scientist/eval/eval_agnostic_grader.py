"""
Validate whether a methodology-agnostic grader can discriminate plan quality.

Takes existing plans from BoN eval logs (already have rubric scores + self-eval scores),
re-grades them with a NEW methodology-agnostic prompt, and compares correlations.

The hypothesis: if the agnostic grader has meaningful variance AND low correlation
with the rubric grader, it can serve as the "quality" signal in a contrastive reward.

Usage:
  python eval_agnostic_grader.py
  python eval_agnostic_grader.py n_samples=200
  python eval_agnostic_grader.py api_profile=myprofile
"""

import logging
import time
import re
import textwrap
import numpy as np
import json
import os
from pathlib import Path
import sys
import random

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
import tinker
from co_scientist.shared.api_profiles import create_service_client
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


# ============================================================
# Methodology-Agnostic Grader Prompt
# ============================================================

def build_agnostic_grader_prompt(scenario: str, proposed_plan: str) -> str:
    """
    Grade a research plan on methodology-INDEPENDENT quality dimensions.

    Key differences from the standard grader:
    - NO rubric items (no methodology-specific criteria)
    - NO reference solution (no expected approach to match)
    - Explicit quality dimensions that apply regardless of methodology
    - Strict scoring with concrete level definitions
    - Evaluates whether the plan is good ON ITS OWN TERMS
    """
    prompt = textwrap.dedent(f"""
        You are a strict research plan evaluator. You must assess the quality of a proposed research plan WITHOUT knowing the specific grading rubric or the expected approach. Evaluate the plan purely on its own merits.

        # Research Scenario
        {scenario}

        # Proposed Research Plan
        {proposed_plan}

        # Evaluation Dimensions

        Evaluate the plan on these 6 methodology-independent quality dimensions. For EACH dimension, assign a score from 0-3 using the STRICT definitions below.

        ## Dimension 1: PROBLEM UNDERSTANDING
        Does the plan demonstrate genuine understanding of the research problem, its context, and its challenges?
        - Level 0: Misunderstands the problem or ignores key aspects.
        - Level 1: Surface-level understanding; restates the scenario without insight.
        - Level 2: Good understanding but misses some nuances or constraints.
        - Level 3: Deep understanding; identifies core challenges and situates them correctly.

        ## Dimension 2: METHODOLOGICAL RIGOR
        Whatever approach is proposed, is it designed with appropriate rigor? Does it have proper experimental controls, baselines, evaluation metrics, and statistical considerations?
        - Level 0: No meaningful methodology; hand-waves the approach.
        - Level 1: Has a methodology but lacks controls, metrics, or baselines.
        - Level 2: Reasonably rigorous but with notable gaps (e.g., missing baselines, unclear metrics).
        - Level 3: Well-designed methodology with appropriate controls, metrics, baselines, and statistical rigor.

        ## Dimension 3: SPECIFICITY AND DEPTH
        Does the plan provide concrete, actionable details rather than vague or generic statements? Could a researcher follow this plan?
        - Level 0: Entirely vague; no actionable details.
        - Level 1: Some specifics but mostly generic statements ("we will use a neural network").
        - Level 2: Reasonable detail on key components but some parts remain underspecified.
        - Level 3: Concrete throughout; specifies architectures, datasets, hyperparameters, procedures, or equivalent level of detail.

        ## Dimension 4: FEASIBILITY AND PRACTICALITY
        Is the proposed plan realistic given typical research constraints (compute, data, time)?
        - Level 0: Plan is unrealistic or requires impossible resources.
        - Level 1: Ambitious to the point of being impractical for most settings.
        - Level 2: Feasible but may require significant resources or have practical challenges.
        - Level 3: Clearly feasible; resource requirements are reasonable and well-considered.

        ## Dimension 5: ANTICIPATION OF CHALLENGES
        Does the plan identify potential failure modes, limitations, or challenges and propose mitigations?
        - Level 0: No awareness of potential issues.
        - Level 1: Mentions challenges superficially without mitigation.
        - Level 2: Identifies key challenges and proposes some mitigations.
        - Level 3: Thorough identification of challenges with concrete mitigation strategies.

        ## Dimension 6: INTERNAL COHERENCE
        Do all parts of the plan work together? Is there a clear logical flow from problem to methodology to expected outcomes?
        - Level 0: Plan is disjointed or self-contradictory.
        - Level 1: Some logical flow but parts feel disconnected.
        - Level 2: Generally coherent with minor inconsistencies.
        - Level 3: Tightly integrated; every component supports the overall approach.

        # Scoring Instructions
        - Be STRICT. Most plans should NOT score 3 on every dimension.
        - A mediocre plan with generic content should score mostly 1s and 2s.
        - Reserve level 3 for genuinely excellent performance on that dimension.
        - Do NOT give the plan credit for "mentioning" something — evaluate whether it DOES it well.
        - Evaluate the plan's methodology on its own terms. A plan proposing approach X should be judged on how well it designs approach X, not on whether X is the "right" approach.

        # Output Format
        Return your evaluation as XML. For EACH dimension, provide brief reasoning (2-3 sentences) then the score.

        <evaluation>
            <dimension num=1 name="problem_understanding">
                <reasoning>[Your assessment]</reasoning>
                <score>[0-3]</score>
            </dimension>
            <dimension num=2 name="methodological_rigor">
                <reasoning>[Your assessment]</reasoning>
                <score>[0-3]</score>
            </dimension>
            <dimension num=3 name="specificity_depth">
                <reasoning>[Your assessment]</reasoning>
                <score>[0-3]</score>
            </dimension>
            <dimension num=4 name="feasibility">
                <reasoning>[Your assessment]</reasoning>
                <score>[0-3]</score>
            </dimension>
            <dimension num=5 name="anticipation_challenges">
                <reasoning>[Your assessment]</reasoning>
                <score>[0-3]</score>
            </dimension>
            <dimension num=6 name="coherence">
                <reasoning>[Your assessment]</reasoning>
                <score>[0-3]</score>
            </dimension>
        </evaluation>
    """).strip()

    return prompt


def parse_agnostic_scores(xml_text: str) -> dict | None:
    """Parse the agnostic grader XML output into per-dimension scores."""
    level_map = {0: 0.0, 1: 0.2, 2: 0.6, 3: 1.0}
    dim_names = [
        "problem_understanding", "methodological_rigor", "specificity_depth",
        "feasibility", "anticipation_challenges", "coherence",
    ]

    dimensions = re.findall(
        r'<dimension num="?(\d+)"?[^>]*>(.*?)</dimension>', xml_text, re.DOTALL
    )

    if not dimensions:
        return None

    raw_scores = {}
    mapped_scores = {}
    for num_str, content in dimensions:
        score_match = re.search(r'<score>\s*(\d+)\s*</score>', content)
        if score_match:
            raw = int(score_match.group(1))
            raw = max(0, min(3, raw))
            dim_idx = int(num_str) - 1
            if 0 <= dim_idx < len(dim_names):
                name = dim_names[dim_idx]
                raw_scores[name] = raw
                mapped_scores[name] = level_map[raw]

    if len(raw_scores) < 4:  # require at least 4 of 6 dimensions
        return None

    return {
        "raw_scores": raw_scores,
        "mapped_scores": mapped_scores,
        "mean_raw": np.mean(list(raw_scores.values())),
        "mean_mapped": np.mean(list(mapped_scores.values())),
        "n_dimensions": len(raw_scores),
    }


# ============================================================
# Config
# ============================================================

@chz.chz
class Config:
    # ---- Data source ----
    bon_logs_path: str = "/home/silas/co-scientist-project/runs/2026/2/withA1,A2/2(ml)/eval_bon/eval_b214_bon_logs.jsonl"

    # ---- Sampling ----
    n_samples: int = 150  # how many plans to re-grade
    seed: int = 42

    # ---- Stratified sampling ----
    # Take equal numbers from low/mid/high rubric score bins
    n_strata: int = 3  # low, mid, high rubric score

    # ---- Model / API ----
    base_url: str | None = None
    api_profile: str | None = None
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_max_tokens: int = 4096
    grader_temperature: float = 0.0

    # ---- Output ----
    output_dir: str = "/home/silas/co-scientist-project/analysis/agnostic_grader_validation"


# ============================================================
# Main
# ============================================================

def main(config: Config):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    os.makedirs(config.output_dir, exist_ok=True)

    # --- Load existing BoN eval logs ---
    logger.info(f"Loading BoN eval logs from {config.bon_logs_path}")
    all_logs = []
    with open(config.bon_logs_path) as f:
        for line in f:
            all_logs.append(json.loads(line))
    logger.info(f"Loaded {len(all_logs)} entries")

    # Filter out degenerate/non-compliant
    valid_logs = [
        l for l in all_logs
        if l.get("grader_rubric_score") is not None
        and l.get("self_eval_score") is not None
        and not l.get("is_degenerate", False)
    ]
    logger.info(f"Valid entries: {len(valid_logs)}")

    # --- Stratified sampling ---
    random.seed(config.seed)
    rubric_scores = [l["grader_rubric_score"] for l in valid_logs]
    sorted_indices = np.argsort(rubric_scores)

    per_stratum = config.n_samples // config.n_strata
    strata_size = len(sorted_indices) // config.n_strata
    selected_indices = []

    for s in range(config.n_strata):
        stratum_start = s * strata_size
        stratum_end = (s + 1) * strata_size if s < config.n_strata - 1 else len(sorted_indices)
        stratum_pool = sorted_indices[stratum_start:stratum_end].tolist()
        selected = random.sample(stratum_pool, min(per_stratum, len(stratum_pool)))
        selected_indices.extend(selected)
        lo_score = rubric_scores[stratum_pool[0]]
        hi_score = rubric_scores[stratum_pool[-1]]
        logger.info(f"  Stratum {s+1}: rubric [{lo_score:.3f}, {hi_score:.3f}], selected {len(selected)} samples")

    sample_logs = [valid_logs[i] for i in selected_indices]
    logger.info(f"Total selected: {len(sample_logs)} plans to re-grade")

    # --- Setup Tinker client ---
    logger.info("Creating grader client...")
    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )
    tokenizer = get_tokenizer(config.grader_model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.grader_model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    grader_client = service_client.create_sampling_client(
        base_model=config.grader_model_name
    )
    grader_params = tinker.types.SamplingParams(
        max_tokens=config.grader_max_tokens,
        temperature=config.grader_temperature,
    )

    # --- Grade all plans with agnostic grader ---
    logger.info("Launching agnostic grader on all plans...")
    futures = []
    for entry in sample_logs:
        prompt = build_agnostic_grader_prompt(
            scenario=entry["goal"],
            proposed_plan=entry["policy_output"],
        )
        model_input = renderer.build_generation_prompt(
            [{"role": "user", "content": prompt}]
        )
        futures.append(
            grader_client.sample(model_input, num_samples=1, sampling_params=grader_params)
        )

    # --- Collect results ---
    logger.info("Collecting agnostic grader results...")
    results = []
    n_parsed = 0
    n_failed = 0

    for idx, (entry, future) in enumerate(zip(sample_logs, futures)):
        try:
            result = future.result()
            parsed, _ = renderer.parse_response(result.sequences[0].tokens)
            xml_text = renderers.get_text_content(parsed)
            scores = parse_agnostic_scores(xml_text)

            results.append({
                "goal_idx": entry["goal_idx"],
                "sample_idx": entry["sample_idx"],
                "rubric_score": entry["grader_rubric_score"],
                "self_eval_score": entry["self_eval_score"],
                "agnostic_scores": scores,
                "agnostic_mean_mapped": scores["mean_mapped"] if scores else None,
                "agnostic_mean_raw": scores["mean_raw"] if scores else None,
                "agnostic_raw_output": xml_text,
            })

            if scores:
                n_parsed += 1
            else:
                n_failed += 1
                logger.warning(f"  Sample {idx}: failed to parse agnostic grader output")

        except Exception as e:
            n_failed += 1
            logger.warning(f"  Sample {idx}: error — {e}")
            results.append({
                "goal_idx": entry["goal_idx"],
                "sample_idx": entry["sample_idx"],
                "rubric_score": entry["grader_rubric_score"],
                "self_eval_score": entry["self_eval_score"],
                "agnostic_scores": None,
                "agnostic_mean_mapped": None,
                "agnostic_mean_raw": None,
                "agnostic_raw_output": None,
            })

        if (idx + 1) % 25 == 0:
            logger.info(f"  Progress: {idx+1}/{len(sample_logs)} (parsed={n_parsed}, failed={n_failed})")

    logger.info(f"Done. Parsed: {n_parsed}, Failed: {n_failed}")

    # --- Save raw results ---
    raw_path = os.path.join(config.output_dir, "agnostic_grader_results.jsonl")
    with open(raw_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    logger.info(f"Raw results saved to {raw_path}")

    # --- Analysis ---
    valid_results = [r for r in results if r["agnostic_mean_mapped"] is not None]
    logger.info(f"\n{'='*60}")
    logger.info(f"ANALYSIS ({len(valid_results)} valid samples)")
    logger.info(f"{'='*60}")

    rubric = [r["rubric_score"] for r in valid_results]
    self_eval = [r["self_eval_score"] for r in valid_results]
    agnostic = [r["agnostic_mean_mapped"] for r in valid_results]

    def pearson(a, b):
        n = len(a)
        if n < 2:
            return 0.0
        ma, mb = np.mean(a), np.mean(b)
        cov = np.mean([(ai-ma)*(bi-mb) for ai, bi in zip(a, b)])
        sa, sb = np.std(a), np.std(b)
        return cov / (sa * sb) if sa * sb > 0 else 0.0

    def spearman(a, b):
        n = len(a)
        if n < 2:
            return 0.0
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    logger.info(f"\n--- Score Distributions ---")
    logger.info(f"  Rubric:    mean={np.mean(rubric):.4f}, std={np.std(rubric):.4f}, min={min(rubric):.3f}, max={max(rubric):.3f}")
    logger.info(f"  Self-eval: mean={np.mean(self_eval):.4f}, std={np.std(self_eval):.4f}, min={min(self_eval):.3f}, max={max(self_eval):.3f}")
    logger.info(f"  Agnostic:  mean={np.mean(agnostic):.4f}, std={np.std(agnostic):.4f}, min={min(agnostic):.3f}, max={max(agnostic):.3f}")

    logger.info(f"\n--- Correlations (Pearson) ---")
    logger.info(f"  Rubric vs Self-eval:  {pearson(rubric, self_eval):.4f}")
    logger.info(f"  Rubric vs Agnostic:   {pearson(rubric, agnostic):.4f}")
    logger.info(f"  Self-eval vs Agnostic:{pearson(self_eval, agnostic):.4f}")

    logger.info(f"\n--- Correlations (Spearman) ---")
    logger.info(f"  Rubric vs Self-eval:  {spearman(rubric, self_eval):.4f}")
    logger.info(f"  Rubric vs Agnostic:   {spearman(rubric, agnostic):.4f}")
    logger.info(f"  Self-eval vs Agnostic:{spearman(self_eval, agnostic):.4f}")

    # Per-dimension analysis
    dim_names = [
        "problem_understanding", "methodological_rigor", "specificity_depth",
        "feasibility", "anticipation_challenges", "coherence",
    ]
    logger.info(f"\n--- Per-Dimension Scores (raw 0-3) ---")
    for dim in dim_names:
        dim_vals = [
            r["agnostic_scores"]["raw_scores"].get(dim)
            for r in valid_results
            if r["agnostic_scores"] and dim in r["agnostic_scores"]["raw_scores"]
        ]
        dim_vals = [v for v in dim_vals if v is not None]
        if dim_vals:
            logger.info(f"  {dim:30s}: mean={np.mean(dim_vals):.3f}, std={np.std(dim_vals):.3f}, dist={dict(zip(*np.unique(dim_vals, return_counts=True)))}")

    # Per-dimension correlation with rubric
    logger.info(f"\n--- Per-Dimension Correlation with Rubric (Pearson) ---")
    level_map = {0: 0.0, 1: 0.2, 2: 0.6, 3: 1.0}
    for dim in dim_names:
        dim_mapped = []
        dim_rubric = []
        for r in valid_results:
            if r["agnostic_scores"] and dim in r["agnostic_scores"]["raw_scores"]:
                dim_mapped.append(level_map[r["agnostic_scores"]["raw_scores"][dim]])
                dim_rubric.append(r["rubric_score"])
        if len(dim_mapped) > 2:
            logger.info(f"  {dim:30s}: {pearson(dim_rubric, dim_mapped):.4f}")

    # Quadrant analysis (agnostic vs rubric)
    med_rubric = np.median(rubric)
    med_agnostic = np.median(agnostic)
    q_hh = sum(1 for r, a in zip(rubric, agnostic) if r >= med_rubric and a >= med_agnostic)
    q_hl = sum(1 for r, a in zip(rubric, agnostic) if r >= med_rubric and a < med_agnostic)
    q_lh = sum(1 for r, a in zip(rubric, agnostic) if r < med_rubric and a >= med_agnostic)
    q_ll = sum(1 for r, a in zip(rubric, agnostic) if r < med_rubric and a < med_agnostic)
    n = len(rubric)
    logger.info(f"\n--- Quadrant Analysis (median split) ---")
    logger.info(f"  High rubric + High agnostic: {q_hh:4d} ({100*q_hh/n:.1f}%) — conventional good")
    logger.info(f"  High rubric + Low agnostic:  {q_hl:4d} ({100*q_hl/n:.1f}%) — rubric-hacking?")
    logger.info(f"  Low rubric + High agnostic:  {q_lh:4d} ({100*q_lh/n:.1f}%) — CREATIVE (alt method, good quality)")
    logger.info(f"  Low rubric + Low agnostic:   {q_ll:4d} ({100*q_ll/n:.1f}%) — genuinely bad")

    # --- Save analysis summary ---
    summary = {
        "n_total": len(sample_logs),
        "n_valid": len(valid_results),
        "n_failed": n_failed,
        "distributions": {
            "rubric": {"mean": float(np.mean(rubric)), "std": float(np.std(rubric))},
            "self_eval": {"mean": float(np.mean(self_eval)), "std": float(np.std(self_eval))},
            "agnostic": {"mean": float(np.mean(agnostic)), "std": float(np.std(agnostic))},
        },
        "correlations_pearson": {
            "rubric_vs_self_eval": pearson(rubric, self_eval),
            "rubric_vs_agnostic": pearson(rubric, agnostic),
            "self_eval_vs_agnostic": pearson(self_eval, agnostic),
        },
        "correlations_spearman": {
            "rubric_vs_self_eval": spearman(rubric, self_eval),
            "rubric_vs_agnostic": spearman(rubric, agnostic),
            "self_eval_vs_agnostic": spearman(self_eval, agnostic),
        },
        "quadrants": {
            "high_rubric_high_agnostic": q_hh,
            "high_rubric_low_agnostic": q_hl,
            "low_rubric_high_agnostic": q_lh,
            "low_rubric_low_agnostic": q_ll,
        },
    }
    summary_path = os.path.join(config.output_dir, "analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nAnalysis summary saved to {summary_path}")

    # --- Key verdict ---
    ag_std = float(np.std(agnostic))
    se_std = float(np.std(self_eval))
    ag_rubric_corr = pearson(rubric, agnostic)
    se_rubric_corr = pearson(rubric, self_eval)

    logger.info(f"\n{'='*60}")
    logger.info(f"VERDICT")
    logger.info(f"{'='*60}")
    logger.info(f"  Agnostic grader std:  {ag_std:.4f} (self-eval std: {se_std:.4f})")
    logger.info(f"  Agnostic-rubric corr: {ag_rubric_corr:.4f} (self-eval-rubric corr: {se_rubric_corr:.4f})")

    if ag_std > se_std * 1.3:
        logger.info(f"  ✓ Agnostic grader is MORE discriminative than self-eval ({ag_std/se_std:.1f}x)")
    else:
        logger.info(f"  ✗ Agnostic grader is NOT more discriminative than self-eval")

    if abs(ag_rubric_corr) < 0.3:
        logger.info(f"  ✓ Low correlation with rubric — measures something DIFFERENT (good for contrastive)")
    elif abs(ag_rubric_corr) < 0.6:
        logger.info(f"  ~ Moderate correlation with rubric — partially independent")
    else:
        logger.info(f"  ✗ High correlation with rubric — too similar to be useful for contrastive")

    if ag_std > se_std * 1.3 and abs(ag_rubric_corr) < 0.5:
        logger.info(f"\n  → PROMISING: Agnostic grader could serve as quality signal in contrastive reward")
    else:
        logger.info(f"\n  → UNCERTAIN: May need further prompt refinement")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
