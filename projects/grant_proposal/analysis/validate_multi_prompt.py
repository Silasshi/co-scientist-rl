"""Pre-validation: do multi-prompt rubric variants improve correlation with Opus?

Grades 9 plans (with known Opus scores) on G12, G11, G6 using:
  - Standard rubric (current)
  - Strict variant rubric (proposed)
  - min(standard, strict)

Then computes Spearman correlation of each with Opus total score.

Usage:
  source shared/tools/use_api_profile.sh new
  python projects/grant_proposal/analysis/validate_multi_prompt.py
"""

import asyncio
import json
import os
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from scipy import stats
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.grant_signal_reward import (
    SignalSpec, build_single_signal_prompt, parse_scores,
    SIGNALS, SIGNAL_WEIGHTS, normalize_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "projects/grant_proposal/analysis/d4v7_foundopt_ablation/data"
GOAL_PATH = ROOT / "projects/grant_proposal/dataset/goals/01_foundopt_v2/research_goal.md"

GRADER_MODEL = "Qwen/Qwen3-30B-A3B"

# ============================================================================
# Plans with Opus labels
# ============================================================================

PLANS = [
    {"file": "plan_MAIN_best.txt",    "qwen": 0.855, "opus": 11, "label": "MAIN_best"},
    {"file": "plan_MAIN_median.txt",  "qwen": 0.685, "opus": 10, "label": "MAIN_median"},
    {"file": "plan_MAIN_worst.txt",   "qwen": 0.350, "opus": 5,  "label": "MAIN_worst"},
    {"file": "plan_B4_best.txt",      "qwen": 0.820, "opus": 12, "label": "B4_best"},
    {"file": "plan_B4_median.txt",    "qwen": 0.715, "opus": 9,  "label": "B4_median"},
    {"file": "plan_B4_worst.txt",     "qwen": 0.340, "opus": 5,  "label": "B4_worst"},
    {"file": "plan_A_fresh_best.txt", "qwen": 0.625, "opus": 7,  "label": "A_fresh_best"},
    {"file": "plan_A_fresh_median.txt","qwen": 0.525, "opus": 7,  "label": "A_fresh_median"},
    {"file": "plan_A_fresh_worst.txt","qwen": 0.400, "opus": 4,  "label": "A_fresh_worst"},
]

# ============================================================================
# Standard signals (from grant_signal_reward.py)
# ============================================================================

STANDARD_SIGNALS = {s.id: s for s in SIGNALS}

# ============================================================================
# Strict variant signals
# ============================================================================

G12_STRICT = SignalSpec(
    id="G12_formalism_strict",
    name="Mathematical Formalism (Strict: Applied-Only)",
    question=(
        "Does the proposal contain mathematical formulas that are ACTUALLY USED "
        "in a subsequent algorithmic or analytical step? A formula is APPLIED if "
        "the proposal later differentiates it, bounds it, solves it, composes it "
        "with another formula, or uses it to derive a quantity needed by the "
        "proposed algorithm. A formula that is stated but never referenced again "
        "is DECORATIVE."
    ),
    cot_scaffolding=(
        "Do EXPLICIT COUNTING:\n"
        "\n"
        "STEP 1 — List every equation in the proposal (same as standard).\n"
        "\n"
        "STEP 2 — For EACH equation, search the REST of the proposal for "
        "any sentence that REFERENCES or USES this equation. Classify:\n"
        "  * APPLIED: A later sentence uses this formula's output, extends it, "
        "bounds it, or plugs it into the proposed algorithm. The formula is "
        "load-bearing — removing it would break the argument.\n"
        "  * DECORATIVE: The formula is stated for apparent rigor but no later "
        "text depends on it. Removing it would not change the proposal's logic. "
        "Examples: a standard textbook SDE stated in the introduction but never "
        "analyzed; a loss function written symbolically but never differentiated "
        "or bounded.\n"
        "\n"
        "STEP 3 — Count N_applied.\n"
        "\n"
        "STEP 4 — Apply this TABLE:\n"
        "  N_applied = 0 → score 1\n"
        "  N_applied = 1 → score 2\n"
        "  N_applied = 2 → score 3\n"
        "  N_applied = 3 → score 4\n"
        "  N_applied ≥ 4 AND formulas build on each other → score 5\n"
        "\n"
        "STRICT RULE: A formula followed by 'we use this framework' without "
        "further analytical steps is DECORATIVE, not APPLIED."
    ),
    scoring_rubric=(
        "1: Zero applied formulas (all decorative or absent)\n"
        "2: One formula actually used in a subsequent step\n"
        "3: Two formulas used in subsequent steps\n"
        "4: Three formulas used, with some building on each other\n"
        "5: Four+ formulas forming a connected derivation chain"
    ),
)

G11_STRICT = SignalSpec(
    id="G11_evidence_rigor_strict",
    name="Evidence Rigor (Strict: Domain-Matched Baselines)",
    question=(
        "Are the proposed baselines actually APPROPRIATE for the specific problem "
        "described? A baseline is DOMAIN-MATCHED if it was designed for or is "
        "commonly applied to the EXACT problem type in the proposal. A baseline "
        "is MISMATCHED if it addresses a different problem type, scale, or domain, "
        "even if it is a well-known modern method."
    ),
    cot_scaffolding=(
        "STEP 1 — Identify the proposal's SPECIFIC problem type (e.g., "
        "'non-convex optimization for neural networks', 'fairness-constrained "
        "classification', 'high-dimensional Bayesian optimization').\n"
        "\n"
        "STEP 2 — List every baseline or comparison method named.\n"
        "For each, classify:\n"
        "  * DOMAIN-MATCHED: This method is specifically designed for or "
        "routinely benchmarked on the proposal's exact problem type. State "
        "the evidence (e.g., 'AdamW is the standard optimizer for neural "
        "network training, which matches the proposal's focus').\n"
        "  * MISMATCHED: This method addresses a different problem. State "
        "why (e.g., 'PPO is an RL algorithm, but the proposal is about "
        "supervised optimization').\n"
        "  * UNCLEAR: Not enough information to determine match.\n"
        "\n"
        "STEP 3 — Check success criterion (same as standard):\n"
        "  * OPERATIONALIZED: named metric with threshold\n"
        "  * VAGUE: 'expect improvement', 'should outperform'\n"
        "\n"
        "STEP 4 — Apply this TABLE:\n"
        "  N_matched = 0 → score 1\n"
        "  N_matched = 1 AND criterion VAGUE → score 2\n"
        "  N_matched = 1 AND criterion OPERATIONALIZED → score 3\n"
        "  N_matched ≥ 2 AND criterion OPERATIONALIZED → score 4\n"
        "  N_matched ≥ 3 AND criterion OPERATIONALIZED + ablations → score 5\n"
    ),
    scoring_rubric=(
        "1: No domain-matched baselines\n"
        "2: One matched baseline but vague criterion\n"
        "3: One matched baseline with operationalized criterion\n"
        "4: Two+ matched baselines + operationalized criterion\n"
        "5: Three+ matched baselines + operationalized criterion + ablations"
    ),
)

G6_STRICT = SignalSpec(
    id="G6_reasoning_depth_strict",
    name="Reasoning Depth (Strict: Alternative-Aware)",
    question=(
        "For each major design choice, does the author explain WHY it was made "
        "AND name what goes wrong with the obvious alternative? Deep justification "
        "means: the proposal identifies an alternative approach, states its "
        "SPECIFIC limitation for THIS problem, and explains how the chosen method "
        "avoids that limitation."
    ),
    cot_scaffolding=(
        "STEP 1 — State the core problem and key approach (same as standard).\n"
        "\n"
        "STEP 2 — Identify the TOP 3-5 LOAD-BEARING DESIGN CHOICES "
        "(same criteria as standard: named innovations, methodological "
        "adaptations, novel strategies).\n"
        "\n"
        "STEP 3 — For EACH choice, classify as:\n"
        "  * ALTERNATIVE-AWARE: The proposal (a) names an alternative approach "
        "AND (b) states a SPECIFIC limitation of that alternative for THIS "
        "problem AND (c) explains how the chosen method addresses it. All three "
        "parts required.\n"
        "  * JUSTIFIED-ONLY: The proposal explains WHY but does NOT name an "
        "alternative or its limitation. E.g., 'we use X because it provides Y' "
        "without saying 'unlike Z, which fails because...'.\n"
        "  * ASSERTED: No justification at all.\n"
        "\n"
        "STEP 4 — Count N_alternative_aware, N_justified_only, N_asserted.\n"
        "\n"
        "STEP 5 — Apply this TABLE:\n"
        "  N_alternative_aware = 0 → score 1\n"
        "  N_alternative_aware = 1 → score 2\n"
        "  N_alternative_aware = 2 → score 3\n"
        "  N_alternative_aware = 3 → score 4\n"
        "  N_alternative_aware ≥ 4 → score 5\n"
        "\n"
        "STRICT RULE: 'unlike traditional methods' is NOT alternative-aware. "
        "The alternative must be NAMED and its failure SPECIFIC to this problem."
    ),
    scoring_rubric=(
        "1: No design choices with alternative-aware justification\n"
        "2: One choice with named alternative + specific limitation\n"
        "3: Two choices with named alternatives\n"
        "4: Three choices with named alternatives\n"
        "5: Four+ choices, each with named alternative and specific limitation"
    ),
)

G4_STRICT = SignalSpec(
    id="G4_focus_strict",
    name="Research Focus (Strict: Composability Check)",
    question=(
        "Does the proposal explain how its named techniques COMPOSE into a "
        "coherent pipeline or framework? Composability means: each technique's "
        "output feeds into the next, OR they address different aspects of the "
        "SAME problem with an explicit integration point. Listing independent "
        "techniques that each address a different sub-problem without explaining "
        "how they connect is NOT focused."
    ),
    cot_scaffolding=(
        "STEP 1 — List the distinct methodological techniques (same as standard).\n"
        "\n"
        "STEP 2 — For EACH PAIR of adjacent techniques in the pipeline, classify:\n"
        "  * COMPOSED: The proposal explicitly states how technique A's output "
        "feeds into technique B, OR how A and B address different facets of the "
        "same sub-problem with a named integration point.\n"
        "  * INDEPENDENT: The techniques are listed but no connection is stated.\n"
        "\n"
        "STEP 3 — Count N_composed and N_independent pairs.\n"
        "\n"
        "STEP 4 — Apply this TABLE:\n"
        "  N_total ≤ 2 techniques → score based on justification only (4-5)\n"
        "  N_total ≥ 3 AND all pairs COMPOSED → score 5\n"
        "  N_total ≥ 3 AND ≥ half pairs COMPOSED → score 4\n"
        "  N_total ≥ 3 AND < half pairs COMPOSED → score 3\n"
        "  N_total ≥ 5 AND < half pairs COMPOSED → score 2\n"
        "  N_total ≥ 7 AND most pairs INDEPENDENT → score 1\n"
    ),
    scoring_rubric=(
        "1: Many independent techniques with no composition\n"
        "2: Multiple techniques, mostly independent\n"
        "3: Some composition but gaps in pipeline\n"
        "4: Mostly composed, clear pipeline\n"
        "5: Tight focus OR all techniques clearly compose"
    ),
)

G13_STRICT = SignalSpec(
    id="G13_risk_awareness_strict",
    name="Risk Awareness (Strict: Method-Specific Risks)",
    question=(
        "Does the proposal identify failure modes that are SPECIFIC to the "
        "proposed method, not generic research risks? A method-specific risk "
        "names (a) a concrete assumption of the proposed approach that could "
        "fail AND (b) the consequence for the results. Generic risks like "
        "'data may be insufficient' or 'implementation challenges may arise' "
        "do NOT count unless tied to a specific methodological choice."
    ),
    cot_scaffolding=(
        "STEP 1 — List every risk or failure mode mentioned in the proposal.\n"
        "\n"
        "STEP 2 — For EACH risk, classify as:\n"
        "  * METHOD-SPECIFIC: Names a concrete assumption of the proposed "
        "method (e.g., 'if the convexity assumption in our SDE framework "
        "breaks down, convergence guarantees fail') AND states the consequence.\n"
        "  * GENERIC: Could apply to any research project ('data collection "
        "challenges', 'computational cost', 'implementation complexity').\n"
        "\n"
        "STEP 3 — Count N_method_specific.\n"
        "\n"
        "STEP 4 — Apply this TABLE:\n"
        "  N_method_specific = 0 → score 1\n"
        "  N_method_specific = 1 → score 2\n"
        "  N_method_specific = 2 → score 3\n"
        "  N_method_specific = 2 AND fallback strategy → score 4\n"
        "  N_method_specific ≥ 3 AND fallback strategies → score 5\n"
    ),
    scoring_rubric=(
        "1: No method-specific risks identified\n"
        "2: One method-specific risk\n"
        "3: Two method-specific risks\n"
        "4: Two method-specific risks with fallback strategies\n"
        "5: Three+ method-specific risks with fallback strategies"
    ),
)

STRICT_SIGNALS = {
    "G12_formalism": G12_STRICT,
    "G11_evidence_rigor": G11_STRICT,
    "G6_reasoning_depth": G6_STRICT,
    "G4_focus": G4_STRICT,
    "G13_risk_awareness": G13_STRICT,
}

TARGET_SIGNAL_IDS = ["G12_formalism", "G11_evidence_rigor", "G6_reasoning_depth", "G4_focus", "G13_risk_awareness"]

# ============================================================================
# Grading
# ============================================================================

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer


async def grade_plan(grader_client, renderer, goal: str, plan: str, signal: SignalSpec) -> dict:
    prompt_text = build_single_signal_prompt(goal, plan, signal, emit_critique=True)
    model_input = renderer.build_generation_prompt(
        [{"role": "user", "content": prompt_text}]
    )
    sampling_params = tinker.types.SamplingParams(
        max_tokens=8192,
        stop=renderer.get_stop_sequences(),
        temperature=0.0,
    )
    result = await asyncio.to_thread(
        lambda: grader_client.sample(prompt=model_input, num_samples=1, sampling_params=sampling_params).result()
    )
    raw_text = ""
    for seq in result.sequences:
        parsed = renderer.parse_response(seq.tokens)
        if parsed:
            content = parsed[0]
            if isinstance(content, dict):
                raw_text = str(content.get("content", ""))
            elif isinstance(content, list):
                raw_text = "".join(str(c.get("content", "") if isinstance(c, dict) else c) for c in content)
            else:
                raw_text = str(content)

    # Strip <think>...</think> blocks (Qwen3 reasoning mode)
    import re as _re
    raw_text = _re.sub(r"<think>.*?</think>", "", raw_text, flags=_re.DOTALL).strip()

    scores = parse_scores(raw_text)
    return scores


async def run_validation():
    goal = GOAL_PATH.read_text().strip()

    service_client = create_service_client(api_profile="new")
    grader_client = service_client.create_sampling_client(base_model=GRADER_MODEL)

    tokenizer = get_tokenizer(GRADER_MODEL)
    renderer_name = model_info.get_recommended_renderer_name(GRADER_MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    logger.info(f"Grader: {GRADER_MODEL}, Renderer: {renderer_name}")
    logger.info(f"Plans: {len(PLANS)}, Signals: {TARGET_SIGNAL_IDS}")
    logger.info(f"Total grading calls: {len(PLANS) * len(TARGET_SIGNAL_IDS) * 2}")

    results = []

    for plan_info in PLANS:
        plan_text = (DATA_DIR / plan_info["file"]).read_text().strip()
        label = plan_info["label"]
        opus = plan_info["opus"]
        logger.info(f"Grading {label} (Opus={opus})...")

        row = {"label": label, "opus": opus, "qwen_agg": plan_info["qwen"]}

        for sid in TARGET_SIGNAL_IDS:
            std_signal = STANDARD_SIGNALS[sid]
            strict_signal = STRICT_SIGNALS[sid]

            # Grade with standard rubric
            std_result = await grade_plan(grader_client, renderer, goal, plan_text, std_signal)
            std_score = std_result.get(sid, {}).get("score")

            # Grade with strict rubric
            strict_result = await grade_plan(grader_client, renderer, goal, plan_text, strict_signal)
            strict_score = None
            for k, v in strict_result.items():
                if v.get("score") is not None:
                    strict_score = v["score"]
                    break

            min_score = min(s for s in [std_score, strict_score] if s is not None) if any(
                s is not None for s in [std_score, strict_score]
            ) else None

            row[f"{sid}_std"] = std_score
            row[f"{sid}_strict"] = strict_score
            row[f"{sid}_min"] = min_score

            logger.info(
                f"  {sid}: std={std_score}, strict={strict_score}, min={min_score}"
            )

        # Compute aggregates
        for suffix in ["std", "strict", "min"]:
            scores = [row.get(f"{sid}_{suffix}") for sid in TARGET_SIGNAL_IDS]
            if all(s is not None for s in scores):
                row[f"depth_agg_{suffix}"] = sum(normalize_score(s, 5) for s in scores) / 3
            else:
                row[f"depth_agg_{suffix}"] = None

        results.append(row)
        logger.info(f"  → depth_agg: std={row.get('depth_agg_std', '?'):.3f}, "
                     f"strict={row.get('depth_agg_strict', '?'):.3f}, "
                     f"min={row.get('depth_agg_min', '?'):.3f}")

    # ======================================================================
    # Analysis
    # ======================================================================
    print("\n" + "=" * 80)
    print("MULTI-PROMPT GRADING VALIDATION RESULTS")
    print("=" * 80)

    # Print table
    header = f"{'Plan':20s} {'Opus':>5s}"
    for sid_short in ["G12", "G11", "G6"]:
        header += f" | {sid_short+'_std':>7s} {sid_short+'_str':>7s} {sid_short+'_min':>7s}"
    header += f" | {'agg_std':>7s} {'agg_str':>7s} {'agg_min':>7s}"
    print(header)
    print("-" * len(header))

    for r in sorted(results, key=lambda x: x["opus"], reverse=True):
        line = f"{r['label']:20s} {r['opus']:>5d}"
        for sid in TARGET_SIGNAL_IDS:
            sid_short = sid.split("_")[0] + "_" + sid.split("_")[1][:3]
            for sfx in ["std", "strict", "min"]:
                v = r.get(f"{sid}_{sfx}")
                line += f" {v:>7d}" if v is not None else f" {'?':>7s}"
        for sfx in ["std", "strict", "min"]:
            v = r.get(f"depth_agg_{sfx}")
            line += f" {v:>7.3f}" if v is not None else f" {'?':>7s}"
        print(line)

    # Correlations
    print("\n" + "=" * 80)
    print("SPEARMAN CORRELATIONS WITH OPUS TOTAL SCORE")
    print("=" * 80)

    opus_scores = [r["opus"] for r in results]

    for metric_name, metric_key in [
        ("Qwen aggregate reward", "qwen_agg"),
        ("G12 standard", "G12_formalism_std"),
        ("G12 strict", "G12_formalism_strict"),
        ("G12 min(std,strict)", "G12_formalism_min"),
        ("G11 standard", "G11_evidence_rigor_std"),
        ("G11 strict", "G11_evidence_rigor_strict"),
        ("G11 min(std,strict)", "G11_evidence_rigor_min"),
        ("G6 standard", "G6_reasoning_depth_std"),
        ("G6 strict", "G6_reasoning_depth_strict"),
        ("G6 min(std,strict)", "G6_reasoning_depth_min"),
        ("Depth agg standard", "depth_agg_std"),
        ("Depth agg strict", "depth_agg_strict"),
        ("Depth agg min(std,strict)", "depth_agg_min"),
    ]:
        values = [r.get(metric_key) for r in results]
        if all(v is not None for v in values):
            rho, p = stats.spearmanr(opus_scores, values)
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            print(f"  {metric_name:30s}  ρ={rho:+.3f}  p={p:.3f} {sig}")
        else:
            print(f"  {metric_name:30s}  MISSING DATA")

    # Key question: does min(std, strict) correlate better than std alone?
    print("\n" + "=" * 80)
    print("KEY QUESTION: Does min(standard, strict) improve Opus correlation?")
    print("=" * 80)

    for sid_label, sid in [("G12", "G12_formalism"), ("G11", "G11_evidence_rigor"), ("G6", "G6_reasoning_depth")]:
        std_vals = [r.get(f"{sid}_std") for r in results]
        min_vals = [r.get(f"{sid}_min") for r in results]
        if all(v is not None for v in std_vals) and all(v is not None for v in min_vals):
            rho_std, _ = stats.spearmanr(opus_scores, std_vals)
            rho_min, _ = stats.spearmanr(opus_scores, min_vals)
            delta = rho_min - rho_std
            verdict = "BETTER ✓" if delta > 0.05 else "WORSE ✗" if delta < -0.05 else "SIMILAR ~"
            print(f"  {sid_label}: ρ_std={rho_std:+.3f} → ρ_min={rho_min:+.3f}  Δ={delta:+.3f}  {verdict}")

    agg_std = [r.get("depth_agg_std") for r in results]
    agg_min = [r.get("depth_agg_min") for r in results]
    if all(v is not None for v in agg_std) and all(v is not None for v in agg_min):
        rho_std, _ = stats.spearmanr(opus_scores, agg_std)
        rho_min, _ = stats.spearmanr(opus_scores, agg_min)
        delta = rho_min - rho_std
        verdict = "BETTER ✓" if delta > 0.05 else "WORSE ✗" if delta < -0.05 else "SIMILAR ~"
        print(f"  AGGREGATE: ρ_std={rho_std:+.3f} → ρ_min={rho_min:+.3f}  Δ={delta:+.3f}  {verdict}")

    # Save results
    out_path = DATA_DIR.parent / "multi_prompt_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(run_validation())
