"""Multi-signal reward for D4 grant proposal evaluation.

D4-v7: 12 signals for evaluating full grant proposals. Hybrid of D4
proposal-specific signals + D3 depth signals (focus, reasoning depth,
evidence rigor, mathematical formalism, risk awareness). Saturated
structural signals (G1/G2/G5/G10) kept at low weight as sanity checks;
depth signals carry the majority of gradient weight.

Signal set D4-v7 (2026-04-20): 12 signals, hybrid D4 proposal + D3 depth.
  Structural (low weight, sanity): G1, G2, G5, G10
  Evidence: G3 (citations), G11 (baselines + success criteria, from D3 S2)
  Depth: G4 (focus, D3 S9), G6 (reasoning depth, D3 S1),
         G12 (math formalism, D3 S2a), G13 (risk awareness, D3 S6)
  Proposal: G8 (deliverables), G9 (scope feasibility)
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# Signal definitions
# =============================================================================

@dataclass
class SignalSpec:
    id: str
    name: str
    question: str
    cot_scaffolding: str | None = None  # Optional pre-scoring reasoning steps
    scoring_rubric: str = ""  # Optional rubric expansion
    # CR-v6 locus attribution: when set, build_single_signal_prompt appends a
    # <locus> output block asking the grader to quote (verbatim) the spans
    # where this signal is weak, for revision targeting. None = legacy path
    # (no locus emission). S9_focus stays None (whole Core Idea is local).
    # Phase 0 validation: projects/ttt_discover/analysis/signal_validity/
    # reports/v8_1/locus_accuracy_S3.md
    locus_directive: str | None = None
    grader_model_override: str | None = None  # v9: route to a different grader model
    score_max: int = 5  # Maximum score value (default 5 for backward compat)


SIGNALS: list[SignalSpec] = [
    # =====================================================================
    # Problem definition (G1-G2)
    # =====================================================================
    SignalSpec(
        id="G1_problem_specificity",
        name="Problem Specificity",
        question=(
            "Does the outline name a SPECIFIC technical problem, limitation, "
            "or knowledge gap? A specific problem names what is broken, "
            "missing, or unknown. Generic framing ('X is important', "
            "'challenges remain') is NOT a specific problem."
        ),
        cot_scaffolding=(
            "STEP 1 — Find the problem statement in the outline. Quote it.\n"
            "\n"
            "STEP 2 — Classify:\n"
            "  * SPECIFIC: names a concrete technical limitation, knowledge "
            "gap, or failure mode. Examples:\n"
            "    - 'neural network weights do not converge to stationary points'\n"
            "    - 'current AI lacks uncertainty quantification'\n"
            "    - 'no framework exists for X in domain Y'\n"
            "    - 'an image of a chair can be misclassified as a toaster'\n"
            "  * GENERIC: vague importance or motivation. Examples:\n"
            "    - 'AI is revolutionising society'\n"
            "    - 'this is a growing area of research'\n"
            "    - 'challenges remain in this field'\n"
            "    - 'more work is needed'\n"
            "\n"
            "STEP 3 — If SPECIFIC, does the problem have MULTIPLE facets "
            "(sub-problems) named? Count N_facets.\n"
            "\n"
            "STEP 4 — Apply this TABLE:\n"
            "  GENERIC, no specific problem → score 1\n"
            "  SPECIFIC but only 1 facet → score 3\n"
            "  SPECIFIC with 2 facets → score 4\n"
            "  SPECIFIC with ≥3 distinct facets → score 5\n"
            "  Mixed (starts generic but becomes specific) → score 2\n"
        ),
        scoring_rubric=(
            "1: Only generic framing, no specific problem named\n"
            "2: Starts generic, eventually names something specific\n"
            "3: One clear specific problem\n"
            "4: Two specific problem facets\n"
            "5: Three or more specific problem facets"
        ),
        locus_directive=(
            "Quote sentences with GENERIC problem framing ('X is important', "
            "'challenges remain', 'growing interest in Y')."
        ),
    ),

    SignalSpec(
        id="G2_specific_aims",
        name="Specific Aims Clarity",
        question=(
            "Does the proposal state MEASURABLE specific aims with expected "
            "outcomes? A measurable aim states what will be PRODUCED or "
            "DETERMINED (e.g., 'determine criteria for X', 'develop a "
            "framework that Y', 'identify factors associated with Z'). "
            "Vague aspirations ('explore', 'investigate', 'contribute to "
            "understanding') are NOT measurable aims."
        ),
        cot_scaffolding=(
            "STEP 1 — List every specific aim or objective stated in the "
            "proposal:\n"
            "  * MEASURABLE: states what will be produced, determined, or "
            "delivered. Examples:\n"
            "    - 'determine what criteria should govern return of results'\n"
            "    - 'develop and evaluate a new assay for X'\n"
            "    - 'identify biomarkers that predict response to Y'\n"
            "    - 'create a database of Z for public use'\n"
            "  * VAGUE: 'explore the relationship', 'investigate various "
            "aspects', 'contribute to understanding', 'advance the field'\n"
            "\n"
            "STEP 2 — For each MEASURABLE aim, check: does it specify an "
            "EXPECTED OUTCOME or deliverable? (paper, dataset, framework, "
            "guideline, tool, recommendation)\n"
            "  Count N_with_outcome.\n"
            "\n"
            "STEP 3 — Count N_measurable_aims total.\n"
            "\n"
            "STEP 4 — Apply this TABLE:\n"
            "  N_measurable = 0 → score 1\n"
            "  N_measurable = 1, no outcomes → score 2\n"
            "  N_measurable = 1, with outcome → score 3\n"
            "  N_measurable = 2, at least 1 with outcome → score 4\n"
            "  N_measurable ≥ 3, or 2 both with outcomes → score 5\n"
        ),
        scoring_rubric=(
            "1: No measurable aims (only vague aspirations)\n"
            "2: One measurable aim but no expected outcome specified\n"
            "3: One measurable aim with expected outcome\n"
            "4: Two measurable aims, at least one with outcome\n"
            "5: Three+ measurable aims, or two aims both with outcomes"
        ),
        locus_directive=(
            "Quote sentences containing VAGUE aims ('explore', 'investigate', "
            "'contribute to understanding') used in place of measurable aims "
            "with expected outcomes."
        ),
    ),

    # =====================================================================
    # Technical grounding (G3-G4)
    # =====================================================================
    SignalSpec(
        id="G3_technical_evidence",
        name="Technical Evidence",
        question=(
            "Does the outline cite SPECIFIC technical evidence to support its "
            "claims? Evidence includes: named papers or citations, quantitative "
            "findings, concrete failure cases, specific illustrative scenarios "
            "with technical detail, or named real-world systems/organizations/"
            "initiatives. Generic claims ('research has shown') without naming "
            "the research are NOT evidence."
        ),
        cot_scaffolding=(
            "STEP 1 — List every piece of SPECIFIC technical evidence:\n"
            "  * Named papers: 'Zhang et al. (ICML 2022)', 'Belkin et al. "
            "(PNAS 2020)'\n"
            "  * Quantitative findings: 'error rates exceed 30%', '10x "
            "slower than real-time'\n"
            "  * Concrete failure cases: 'an image of a chair misclassified "
            "as a toaster'\n"
            "  * Specific illustrative scenarios: 'self-driving cars "
            "simultaneously choosing the same route could overload roads', "
            "'spurious correlations from historic data amplifying health "
            "inequalities'\n"
            "  * Named real-world systems/organizations: 'Met Office weather "
            "forecasts', 'Edinburgh's Bayes Centre', 'Urban Observatory "
            "test-bed', 'National AI Strategy'\n"
            "\n"
            "DO NOT count: 'research has shown', 'studies indicate', "
            "'it is well known', 'recent advances', 'current AI cannot'\n"
            "\n"
            "STEP 2 — Count N_evidence.\n"
            "\n"
            "STEP 3 — Apply this TABLE:\n"
            "  N_evidence = 0 → score 1\n"
            "  N_evidence = 1 → score 3\n"
            "  N_evidence = 2 → score 4\n"
            "  N_evidence ≥ 3 → score 5\n"
        ),
        scoring_rubric=(
            "1: No specific evidence cited\n"
            "3: One piece of evidence (notable for a short outline)\n"
            "4: Two pieces of evidence\n"
            "5: Three or more pieces of evidence"
        ),
        locus_directive=(
            "Quote sentences making unsupported claims ('research has shown', "
            "'studies indicate', 'it is well known') without naming specific "
            "sources or data."
        ),
    ),

    SignalSpec(
        id="G4_focus",
        name="Research Focus",
        question=(
            "Does the proposal STAY focused on a small number of core "
            "techniques, or does it STACK unrelated methods without "
            "justifying each? A focused proposal has 1-3 core techniques "
            "that clearly compose. A stacked proposal lists many techniques "
            "without explaining why each is necessary for THIS specific "
            "problem."
        ),
        cot_scaffolding=(
            "STEP 1 — List the distinct METHODOLOGICAL techniques named "
            "in the proposal as part of the central approach. A 'technique' "
            "is a named algorithmic, analytical, or methodological component "
            "FUNDAMENTAL to the proposed work.\n"
            "\n"
            "DO NOT count as techniques:\n"
            "  - Standard infrastructure (computing, data storage)\n"
            "  - Evaluation metrics or benchmarks\n"
            "  - Dissemination activities (papers, workshops)\n"
            "  - Components of a single pipeline where stages are "
            "obviously interdependent\n"
            "\n"
            "Cap the list at 8 items maximum.\n"
            "\n"
            "STEP 2 — For EACH technique, classify as:\n"
            "  * JUSTIFIED: proposal has a sentence explaining WHY this "
            "technique is needed for the stated problem (a causal or "
            "mechanistic reason).\n"
            "  * UNJUSTIFIED: technique is named but no explanation of "
            "why it is needed for THIS proposal's problem.\n"
            "\n"
            "STEP 3 — Count (N_total, N_unjustified).\n"
            "\n"
            "STEP 4 — Apply this TABLE:\n"
            "  N_total ≤ 3 AND N_unjustified = 0 → score 5\n"
            "  N_total ≤ 3 AND N_unjustified ≥ 1 → score 4\n"
            "  N_total ∈ [4, 6] AND N_unjustified ≤ 1 → score 4\n"
            "  N_total ∈ [4, 6] AND N_unjustified ∈ [2, 3] → score 3\n"
            "  N_total ∈ [4, 6] AND N_unjustified ≥ 4 → score 2\n"
            "  N_total ≥ 7 AND N_unjustified ≤ 2 → score 3\n"
            "  N_total ≥ 7 AND N_unjustified ≥ 3 → score 2\n"
            "  N_total ≥ 7 AND N_unjustified ≥ 5 → score 1\n"
        ),
        scoring_rubric=(
            "Focus is a function of BOTH technique count and justification "
            "ratio. Few techniques, justified = 5. Many techniques, "
            "unjustified = 1.\n"
            "1: ≥7 techniques with ≥5 unjustified (severe stacking)\n"
            "2: Moderate stacking or many unjustified\n"
            "3: Borderline (2-3 unjustified of moderate total)\n"
            "4: Focused or mostly justified\n"
            "5: Tight focus (≤3 techniques) with every one justified"
        ),
        locus_directive=(
            "Quote each technique that is UNJUSTIFIED — named without "
            "a mechanistic sentence explaining why it is needed for this "
            "proposal's specific problem."
        ),
    ),

    # =====================================================================
    # Novelty (G5-G6)
    # =====================================================================
    SignalSpec(
        id="G5_gap_identification",
        name="Gap Identification",
        question=(
            "Does the outline explain WHY current approaches are "
            "INSUFFICIENT for the stated problem? Gap identification means "
            "stating a specific limitation of existing work — not just "
            "listing what exists, but arguing why it falls short."
        ),
        cot_scaffolding=(
            "STEP 1 — List every statement about existing work limitations:\n"
            "  * INSUFFICIENCY: states what existing methods CANNOT do or "
            "where they FAIL. Examples:\n"
            "    - 'current methods take shortcuts in modelling and inference'\n"
            "    - 'optimisation state-of-the-art is not fit for purpose'\n"
            "    - 'advancements have been piecemeal and ad hoc'\n"
            "  * NEUTRAL: mentions existing work without critique. Examples:\n"
            "    - 'prior work has studied X'\n"
            "    - 'several methods have been proposed'\n"
            "\n"
            "STEP 2 — Count N_insufficiency (distinct limitation statements).\n"
            "\n"
            "STEP 3 — Apply this TABLE:\n"
            "  N_insufficiency = 0 → score 1\n"
            "  N_insufficiency = 1 → score 3\n"
            "  N_insufficiency = 2 → score 4\n"
            "  N_insufficiency ≥ 3 → score 5\n"
        ),
        scoring_rubric=(
            "1: No gap identified — existing work not critiqued\n"
            "3: One specific insufficiency stated\n"
            "4: Two insufficiencies stated\n"
            "5: Three or more insufficiencies forming a coherent argument"
        ),
        locus_directive=(
            "Quote sentences that mention existing work WITHOUT stating a "
            "limitation ('prior work has studied X', 'methods exist for Y')."
        ),
    ),

    SignalSpec(
        id="G6_reasoning_depth",
        name="Reasoning Depth",
        question=(
            "For each major design choice in the proposal, does the author "
            "explain WHY it was made? Depth means explicit justification: "
            "problem → insight → each choice justified by reference to that "
            "insight. Shallow proposals list techniques with citations but "
            "don't explain the logical connection to the stated problem."
        ),
        cot_scaffolding=(
            "STEP 1 — State the core problem (from the proposal's problem "
            "statement) and the KEY INSIGHT or approach in one sentence each.\n"
            "\n"
            "STEP 2 — Identify the TOP 3-5 LOAD-BEARING DESIGN CHOICES — "
            "the decisions that DISTINGUISH this proposal from a standard "
            "approach. Examples:\n"
            "  - a named algorithmic innovation\n"
            "  - a specific methodological adaptation\n"
            "  - a novel data collection or analysis strategy\n"
            "  - a specific theoretical framework choice\n"
            "\n"
            "DO NOT count: standard infrastructure, named datasets unless "
            "novel, standard hyperparameters, evaluation benchmarks.\n"
            "\n"
            "STEP 3 — For EACH choice, classify as:\n"
            "  * JUSTIFIED: proposal contains a sentence stating WHY this "
            "choice addresses the problem (causal or mechanistic).\n"
            "  * ASSERTED: proposal names the choice but gives no 'why'.\n"
            "\n"
            "STEP 4 — Count N_asserted.\n"
            "\n"
            "STEP 5 — Apply this TABLE:\n"
            "  N_asserted = 0 → score 5\n"
            "  N_asserted = 1 → score 4\n"
            "  N_asserted = 2 → score 3\n"
            "  N_asserted = 3 → score 2\n"
            "  N_asserted ≥ 4 → score 1\n"
            "\n"
            "STRICT RULE: 'inspired by X' or 'following Y' is NOT "
            "justification. But a citation PLUS a mechanistic sentence "
            "('we use X because it provides property Y that addresses Z') "
            "IS justified."
        ),
        scoring_rubric=(
            "1: ≥4 design choices asserted (list of techniques, no argument)\n"
            "2: 3 asserted\n"
            "3: 2 asserted\n"
            "4: 1 asserted (mostly justified)\n"
            "5: 0 asserted — every design choice has an explicit why-sentence"
        ),
        locus_directive=(
            "Quote each design choice that is ASSERTED — named without "
            "a mechanistic 'why' sentence explaining how it addresses "
            "the proposal's stated problem."
        ),
    ),

    # =====================================================================
    # Feasibility & Deliverables (G8-G11)
    # =====================================================================
    SignalSpec(
        id="G8_deliverable_clarity",
        name="Deliverable Clarity",
        question=(
            "Does the proposal name SPECIFIC deliverables or products that "
            "will result from the work? Deliverables are concrete outputs: "
            "published papers, datasets, guidelines, software tools, white "
            "papers, policy recommendations, databases. Generic outcome "
            "claims ('will advance the field', 'will contribute to knowledge') "
            "are NOT deliverables."
        ),
        cot_scaffolding=(
            "STEP 1 — List every SPECIFIC deliverable named in the proposal:\n"
            "  * SPECIFIC: 'publish 3 papers in area X', 'release a public "
            "database', 'produce a white paper with recommendations', "
            "'develop open-source software for Y', 'create guidelines for Z', "
            "'submit policy brief to agency W'\n"
            "  * GENERIC (do NOT count): 'advance the field', 'contribute to "
            "knowledge', 'have significant impact', 'produce findings'\n"
            "\n"
            "STEP 2 — Count N_deliverables.\n"
            "\n"
            "STEP 3 — Apply this TABLE:\n"
            "  N_deliverables = 0 → score 1\n"
            "  N_deliverables = 1 → score 2\n"
            "  N_deliverables = 2 → score 3\n"
            "  N_deliverables = 3 → score 4\n"
            "  N_deliverables ≥ 4 → score 5\n"
        ),
        scoring_rubric=(
            "1: No specific deliverables named\n"
            "2: One deliverable\n"
            "3: Two deliverables\n"
            "4: Three deliverables\n"
            "5: Four or more deliverables"
        ),
        locus_directive=(
            "Quote sentences with GENERIC outcome claims ('will advance the "
            "field', 'will contribute to knowledge') without naming specific "
            "deliverables or products."
        ),
    ),

    SignalSpec(
        id="G9_scope_feasibility",
        name="Scope Feasibility",
        question=(
            "Is the proposed scope of work feasible given typical grant "
            "constraints? Scope red flags include: claiming to analyze an "
            "unrealistically large dataset comprehensively, covering all "
            "countries/all cases, proposing too many aims for the format, "
            "or promising comprehensive coverage of a vast domain without "
            "acknowledging limitations."
        ),
        cot_scaffolding=(
            "STEP 1 — List all scope commitments in the proposal (number of "
            "countries, databases, analyses, aims, sub-studies, etc.).\n"
            "\n"
            "STEP 2 — Identify scope RED FLAGS:\n"
            "  * OVERLY BROAD claims: 'comprehensive analysis of 4,230 laws', "
            "'all 193 UN member states', 'every possible scenario'\n"
            "  * MISMATCH between scope and resources: e.g., 2-year project "
            "claiming to comprehensively analyze thousands of items\n"
            "  * UNACKNOWLEDGED limitations: broad scope claimed without "
            "explicitly stating what is excluded or why the scope is feasible\n"
            "  * TOO MANY sub-studies for the format (e.g., 5+ distinct "
            "analyses in a 2-year exploratory grant)\n"
            "\n"
            "STEP 3 — Count N_red_flags.\n"
            "\n"
            "STEP 4 — Apply this TABLE:\n"
            "  N_red_flags = 0 → score 5\n"
            "  N_red_flags = 1 → score 4\n"
            "  N_red_flags = 2 → score 3\n"
            "  N_red_flags ≥ 3 → score 1\n"
        ),
        scoring_rubric=(
            "5: No scope red flags — all commitments appear feasible\n"
            "4: One minor scope concern\n"
            "3: Two scope red flags\n"
            "1: Three or more scope red flags — project appears infeasible"
        ),
        locus_directive=(
            "Quote sentences containing OVERLY BROAD scope claims "
            "('comprehensive analysis of X thousand', 'all countries', "
            "'every possible') or scope-resource mismatches."
        ),
    ),

    SignalSpec(
        id="G10_approach_coverage",
        name="Approach Coverage",
        question=(
            "Does the proposed approach/methodology cover ALL stated aims? "
            "For each aim or objective, there should be a corresponding "
            "description of HOW it will be accomplished. An aim with no "
            "corresponding method is an UNCOVERED aim."
        ),
        cot_scaffolding=(
            "STEP 1 — List all stated aims/objectives from the proposal.\n"
            "  Count N_aims.\n"
            "\n"
            "STEP 2 — For each aim, check: is there a corresponding "
            "methodological description explaining HOW it will be "
            "accomplished? Mark COVERED or UNCOVERED.\n"
            "  * COVERED: aim has a corresponding paragraph or section "
            "describing specific procedures, data sources, or analytical "
            "steps\n"
            "  * UNCOVERED: aim is stated but no method is described, "
            "or method is only vaguely gestured at ('will use appropriate "
            "methods')\n"
            "\n"
            "STEP 3 — Calculate coverage = N_covered / N_aims.\n"
            "\n"
            "STEP 4 — Apply this TABLE:\n"
            "  coverage = 100% → score 5\n"
            "  coverage ≥ 75% → score 4\n"
            "  coverage ≥ 50% → score 3\n"
            "  coverage ≥ 25% → score 2\n"
            "  coverage < 25% → score 1\n"
        ),
        scoring_rubric=(
            "1: Less than 25% of aims have methodology described\n"
            "2: 25-49% coverage\n"
            "3: 50-74% coverage\n"
            "4: 75-99% coverage\n"
            "5: 100% — every aim has a corresponding method"
        ),
        locus_directive=(
            "Quote sentences stating aims or objectives for which NO "
            "corresponding methodology is described in the Approach section."
        ),
    ),

    SignalSpec(
        id="G11_evidence_rigor",
        name="Evidence Rigor",
        question=(
            "Can the proposed evaluation support the claims? For EMPIRICAL "
            "work: are baselines FAIR (appropriate, modern, matched in scale "
            "— not strawman) and is there an OPERATIONALIZED success criterion "
            "(specific metric with threshold)? For THEORETICAL work: are "
            "assumptions stated and is a proof sketch/strategy given?"
        ),
        cot_scaffolding=(
            "STEP 1 — Identify if the proposed evaluation is EMPIRICAL or "
            "THEORETICAL (or both).\n"
            "\n"
            "STEP 2 (EMPIRICAL) — List every baseline or comparison method. "
            "For each, classify as:\n"
            "  * FAIR: a modern, appropriate-scale method from the same "
            "problem space.\n"
            "  * STRAWMAN: obviously weak (random baseline, method 5+ years "
            "old in a fast-moving field, or from a different problem regime).\n"
            "Count N_fair and N_strawman.\n"
            "\n"
            "STEP 2 (THEORETICAL) — Mark each as present (1) or absent (0):\n"
            "  * Explicit assumptions stated\n"
            "  * Proof strategy / sketch given (not just 'we will prove')\n"
            "  * Tight bounds discussed OR key lemma stated\n"
            "Count T = sum (0-3).\n"
            "\n"
            "STEP 3 — Check the SUCCESS CRITERION:\n"
            "  * OPERATIONALIZED: named metric with benchmark or threshold.\n"
            "  * VAGUE: 'we expect improvement', 'should outperform'.\n"
            "\n"
            "STEP 4 — Apply this TABLE:\n"
            "  EMPIRICAL:\n"
            "    N_strawman ≥ 2 → score 1\n"
            "    N_strawman ≥ 1 AND criterion VAGUE → score 1\n"
            "    N_fair ≥ 2 AND criterion VAGUE → score 2\n"
            "    N_fair ≥ 2 AND criterion OPERATIONALIZED → score 4\n"
            "    N_fair ≥ 3 AND criterion OPERATIONALIZED + ablations → score 5\n"
            "    Otherwise → score 3\n"
            "\n"
            "  THEORETICAL:\n"
            "    T ≤ 1 → score 1-2\n"
            "    T = 2 → score 3\n"
            "    T = 3 + operationalized criterion → score 4-5\n"
        ),
        scoring_rubric=(
            "1: Strawman baselines or vague criterion with no fair comparisons\n"
            "2: Fair baselines but vague success criterion\n"
            "3: One fair baseline with operationalized criterion\n"
            "4: ≥2 fair baselines + operationalized criterion\n"
            "5: ≥3 fair baselines + operationalized criterion + ablations"
        ),
        locus_directive=(
            "Quote each STRAWMAN baseline (weak or obsolete comparison) OR "
            "each VAGUE success-criterion sentence ('expect', 'should', "
            "'aim to' without a measurable threshold)."
        ),
    ),

    # =====================================================================
    # Depth signals from D3 v9 (G12-G13)
    # =====================================================================
    SignalSpec(
        id="G12_formalism",
        name="Mathematical Formalism",
        question=(
            "Does the proposal contain NON-TRIVIAL mathematical content? "
            "Non-trivial means: a named equation written in symbolic form that "
            "is NOT tautological (L = -R is tautological), NOT just hyperparameter "
            "values (lr=3e-5), and NOT standard definitions. Count DISTINCT "
            "equations/formulas that contribute to the proposal's core methodology."
        ),
        cot_scaffolding=(
            "Do EXPLICIT COUNTING (no interpretation):\n"
            "\n"
            "STEP 1 — Scan the ENTIRE proposal for EQUATIONS written in "
            "symbolic form.\n"
            "An EQUATION contains:\n"
            "  - mathematical symbols (=, +, -, *, /, ^, log, exp, sum, "
            "integral, E[], nabla, O())\n"
            "  - AND at least one NAMED variable or function beyond simple "
            "constants\n"
            "\n"
            "List each equation found. Write down the actual formula.\n"
            "\n"
            "STEP 2 — For EACH equation, classify as:\n"
            "  * TAUTOLOGICAL: L = -R, y = f(x) without defining f, or any "
            "formula that restates the problem without analytical content.\n"
            "  * HYPERPARAMETER-ONLY: lr=3e-5, batch=32, rank=8, etc.\n"
            "  * NON-TRIVIAL: A formula that adds analytical content. "
            "Examples:\n"
            "    - A convergence rate: O(1/sqrt(T))\n"
            "    - An objective: J_beta = E[log E[exp(beta * R)]]\n"
            "    - A bound: Q(s) + c * P(s) * sqrt(1+T)/(1+n(s))\n"
            "    - A regularizer: KL(pi || pi_ref) <= delta\n"
            "    - A complexity bound: O(d * m) vs O(n^3)\n"
            "\n"
            "STEP 3 — Count N_nontrivial.\n"
            "\n"
            "STEP 4 — Apply this TABLE:\n"
            "  N_nontrivial = 0 → score 1\n"
            "  N_nontrivial = 1 AND standard textbook → score 2\n"
            "  N_nontrivial = 1 AND adapted/novel → score 3\n"
            "  N_nontrivial = 2 AND at least 1 non-textbook → score 4\n"
            "  N_nontrivial ≥ 3 AND coherent derivation chain → score 5\n"
            "\n"
            "STRICT RULE: Hyperparameter values are NEVER equations. "
            "Section headers with math words are NOT formulas."
        ),
        scoring_rubric=(
            "1: Zero non-trivial formulas\n"
            "2: One standard textbook formula\n"
            "3: One adapted/novel formula\n"
            "4: Two formulas with at least one non-textbook\n"
            "5: Three+ formulas forming a coherent derivation chain"
        ),
        locus_directive=(
            "Quote sentences where a mathematical claim is made in prose "
            "without an accompanying symbolic equation (e.g., 'we prove "
            "convergence' without writing the rate)."
        ),
    ),

    SignalSpec(
        id="G13_risk_awareness",
        name="Risk Awareness",
        question=(
            "Does the proposal demonstrate mature awareness of its own "
            "boundaries, risks, and failure modes? A good proposal names "
            "2-3 SPECIFIC failure modes, identifies which assumptions are "
            "most fragile, states scope boundaries, and sketches fallback "
            "strategies. This is NOT about enumerating 10 generic risks — "
            "it's about showing the author understands the risk structure."
        ),
        cot_scaffolding=(
            "Before scoring:\n"
            "1. Does the proposal identify 2-3 specific ways the research "
            "could fail? (specific = specific assumption breaking, specific "
            "method limitation, specific domain mismatch)\n"
            "2. Does the proposal state which assumptions are most fragile "
            "(the 'key bet')?\n"
            "3. Does the proposal acknowledge scope boundaries (what it "
            "does NOT claim to solve)?\n"
            "4. Does the proposal sketch fallback strategies or contingency "
            "plans for identified risks?\n"
            "5. Is there a coherent risk structure, or just a laundry list?\n"
            "6. Assign a score."
        ),
        scoring_rubric=(
            "1: No risk awareness. Proposal assumes success with no "
            "discussion of failure modes or limitations.\n"
            "2: Generic disclaimers ('there might be limitations', "
            "'further research is needed') without specificity.\n"
            "3: Partial. Names 1 failure mode OR mentions scope boundary, "
            "but risk structure is incomplete.\n"
            "4: Mature. Names 2-3 specific failure modes, identifies "
            "fragile assumptions, states scope boundary.\n"
            "5: Exceptional. Coherent risk model: specific failure modes, "
            "load-bearing assumptions identified, scope boundary explicit, "
            "AND fallback direction sketched."
        ),
        locus_directive=(
            "Quote GENERIC risk disclaimers ('there might be limitations', "
            "'further research is needed', 'various challenges exist'). "
            "If the proposal lacks specific failure modes, quote the last "
            "sentence of the proposal."
        ),
    ),

]

# Domain-specific signal variants (not in main SIGNALS list; activated via signal_overrides)
SIGNAL_VARIANTS: dict[str, SignalSpec] = {
    "G12a_analytical_framework": SignalSpec(
        id="G12a_analytical_framework",
        name="Analytical Framework",
        question=(
            "Does the proposal employ a rigorous analytical or theoretical "
            "framework appropriate to its domain? A framework is a NAMED "
            "perspective, theory, or systematic methodology that structures "
            "the approach — not just a method name. Examples: critical race "
            "theory, implementation science, comparative legal analysis, "
            "design-based implementation research, program theory of change, "
            "institutional analysis (Ostrom), causal process tracing, "
            "intersectionality framework. The framework must go beyond "
            "naming — it must STRUCTURE the proposed approach."
        ),
        cot_scaffolding=(
            "STEP 1 — Scan the proposal for NAMED analytical or theoretical "
            "frameworks. A framework is:\n"
            "  * A named theory from the relevant discipline\n"
            "  * A systematic analytical methodology with stated assumptions\n"
            "  * An established conceptual model that organizes the inquiry\n"
            "\n"
            "DO NOT count as frameworks:\n"
            "  - Generic method labels ('qualitative', 'mixed methods', "
            "'case study') without theoretical grounding\n"
            "  - Data collection techniques ('interviews', 'surveys') alone\n"
            "  - Statistical methods without a conceptual model linking "
            "variables\n"
            "\n"
            "STEP 2 — For EACH named framework, classify as:\n"
            "  * APPLIED: the framework visibly structures how the proposal "
            "organizes its approach (e.g., theory of change maps inputs to "
            "outcomes, comparative framework specifies dimensions of "
            "comparison).\n"
            "  * NAMED-ONLY: framework is cited but does not visibly shape "
            "the methodology.\n"
            "\n"
            "STEP 3 — For each APPLIED framework, check: does the proposal "
            "justify WHY this framework fits this problem (vs alternatives)?\n"
            "  Mark JUSTIFIED or UNJUSTIFIED.\n"
            "\n"
            "STEP 4 — Apply this TABLE:\n"
            "  No framework found → score 1\n"
            "  Framework NAMED-ONLY → score 2\n"
            "  One framework APPLIED but UNJUSTIFIED → score 3\n"
            "  One framework APPLIED and JUSTIFIED → score 4\n"
            "  Framework APPLIED, JUSTIFIED, and explicitly connected to "
            "methodology (structures the entire approach) → score 5\n"
        ),
        scoring_rubric=(
            "1: No named analytical or theoretical framework\n"
            "2: Framework named but not applied (lip service)\n"
            "3: One framework applied to structure analysis\n"
            "4: Framework applied with justification for why it fits\n"
            "5: Framework structures entire approach, justified against "
            "alternatives"
        ),
        locus_directive=(
            "Quote sentences describing methodology WITHOUT naming the "
            "underlying analytical framework (e.g., 'we will analyze' "
            "without stating which theory or model guides the analysis)."
        ),
    ),
}


# =============================================================================
# Signal weights (for scalar aggregation in the entropic objective)
# =============================================================================
# Set after sanity check + deep review. Higher weight for signals with:
#   (a) strong literature grounding (NIH/NSF/ERC)
#   (b) low halo / high independence observed in sanity check
#   (c) unique coverage (captures something no other signal does)

SIGNAL_WEIGHTS: dict[str, float] = {
    # D4-v7 weights (2026-04-20): 12 signals. Saturated structural signals
    # (G1/G2/G5/G10) at 0.04; depth signals from D3 at 0.12; G3 raised
    # to 0.10 for ML domain where real citations are possible.
    "G1_problem_specificity":  0.04,
    "G2_specific_aims":        0.04,
    "G3_technical_evidence":   0.10,
    "G4_focus":                0.12,
    "G5_gap_identification":   0.04,
    "G6_reasoning_depth":      0.12,
    "G8_deliverable_clarity":  0.08,
    "G9_scope_feasibility":    0.08,
    "G10_approach_coverage":   0.04,
    "G11_evidence_rigor":      0.12,
    "G12_formalism":           0.12,
    "G13_risk_awareness":      0.10,
}

assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 1e-6, \
    f"Weights must sum to 1.0, got {sum(SIGNAL_WEIGHTS.values())}"

SCORE_MAX: dict[str, int] = {s.id: s.score_max for s in SIGNALS}


def normalize_score(score: int | None, score_max: int = 5) -> float:
    """Map 1-score_max integer score to [0, 1]."""
    if score is None:
        return 0.0
    return min(max((score - 1) / (score_max - 1), 0.0), 1.0)


def aggregate_reward(signal_scores: dict[str, int | None]) -> float:
    """Scalar aggregate across gradient signals using SIGNAL_WEIGHTS.

    Missing signals contribute 0. Does NOT apply hard gates (those come from
    seven_signal_reward.py Goal-Contrast and Claim Verification).
    """
    total = 0.0
    for sid, weight in SIGNAL_WEIGHTS.items():
        sm = SCORE_MAX.get(sid, 5)
        total += weight * normalize_score(signal_scores.get(sid), score_max=sm)
    return total


# =============================================================================
# Prompt builders
# =============================================================================

def _format_signal_block(spec: SignalSpec, include_cot: bool = True) -> str:
    block = f"## Signal: {spec.name} (id={spec.id})\n\n**Question**: {spec.question}\n"
    if include_cot and spec.cot_scaffolding:
        block += f"\n**Required reasoning before scoring**:\n{spec.cot_scaffolding}\n"
    if spec.scoring_rubric:
        block += f"\n**Scoring rubric**:\n{spec.scoring_rubric}\n"
    return block


def _shared_preamble(score_max: int = 5) -> str:
    return f"""You are an expert-level research plan evaluator. Your job is to assess a research plan on a specific, well-defined evaluation dimension.

RULES:
- Evaluate based on what is ACTUALLY in the plan, not what it claims about itself.
- Follow the rubric strictly. Each score level has specific criteria.
- Be skeptical of surface-level plausibility — check whether claims are supported by concrete content.
- Do not demand rigor-theater (long lists of statistical procedures, exhaustive failure modes) when the plan is concise and focused. Judge the substance of what's there.
- Concise methodology that delegates boilerplate to citations is acceptable if the core is clearly described.
- If a signal has "Required reasoning" scaffolding, you MUST do the reasoning BEFORE assigning a score.
- Integer scores only: 1 through {score_max}."""


def build_single_call_prompt(goal: str, plan: str, signals: list | None = None) -> str:
    """Build a single prompt that asks the grader to score ALL signals at once."""
    active_signals = signals or SIGNALS
    score_max = active_signals[0].score_max if active_signals else 5
    signal_blocks = "\n\n---\n\n".join(_format_signal_block(s) for s in active_signals)

    output_template = "\n".join(
        f"    <dim id=\"{s.id}\">\n"
        f"        <reasoning>Your analysis (do required reasoning steps if any).</reasoning>\n"
        f"        <score>INTEGER 1-{s.score_max}</score>\n"
        f"    </dim>"
        for s in active_signals
    )

    return f"""{_shared_preamble(score_max)}

You will evaluate the following research plan on {len(active_signals)} independent dimensions.

# Research Goal
{goal}

# Research Plan
{plan}

# Evaluation Dimensions

Below are the {len(SIGNALS)} dimensions to evaluate. For EACH dimension, apply the scoring rubric strictly.

{signal_blocks}

---

# Output Format

Evaluate ALL {len(SIGNALS)} dimensions in the following XML structure:

<evaluation>
{output_template}
</evaluation>

Begin your evaluation now."""


MAX_PLAN_CHARS = 15000  # ~3750 tokens — keeps grading prompts within 32K context


def _truncate_plan(plan: str) -> str:
    """Truncate plan text to fit within context window."""
    if len(plan) > MAX_PLAN_CHARS:
        return plan[:MAX_PLAN_CHARS] + "\n\n[... truncated for context limit ...]"
    return plan


def _build_locus_output_block(signal: SignalSpec) -> str:
    """Build the <locus> output instruction appended to per-signal prompts.

    Modeled on the validated Phase 0 prompt in
    `projects/ttt_discover/analysis/signal_validity/scripts/grade/grade_locus_s3_pilot.py`
    (Phase 0 literal-verbatim rates: 7/8 signals ≥90%; S3 is goal-specific
    outlier at 16% on TTT-Discover and is skipped by default in CR-v6).

    Parameterized by `signal.name` (appears in the directive for the grader)
    and `signal.locus_directive` (what spans to quote — signal-specific).
    Returns empty string if the signal has no locus_directive.
    """
    if not signal.locus_directive:
        return ""
    return f"""

---

# Additional Output: Locus Attribution

After the <evaluation> block, output a separate <locus> block containing a
JSON array with up to 3 entries identifying WHERE in the Research Plan the
**{signal.name}** weakness appears. {signal.locus_directive}

Each entry has exactly these three keys:

  - "section_hint": name of the plan section the span belongs to
     (e.g., "Problem", "Motivation", "Related Work", "Core Idea",
     "Methodology", "Evaluation", "Risks"). Use "" if unclear.
  - "quote": a VERBATIM substring copied from the Research Plan text above
     (≤ 250 characters).
  - "why": 1-2 sentences explaining why this span hurts the {signal.name}
     signal per the rubric above. Cite the specific rubric language.

**VERBATIM RULE (STRICT)**: The "quote" MUST be a literal copy-paste from the
"# Research Plan" text above. DO NOT paraphrase, DO NOT merge non-adjacent
sentences, DO NOT fix typos, DO NOT insert words. If you cannot find a clean
contiguous substring that captures the weakness, use the single BEST
available substring and describe the missing content in "why". Before
emitting, mentally Ctrl-F the quote against the plan text — if not an exact
substring, rewrite it so it is.

Choose spans where the weakness is CONCENTRATED — the 1-3 places a reviser
would most naturally edit. If the plan has NO {signal.name} weakness
(score 5), output an empty JSON array.

If the weakness is ABSENCE of content (e.g., "no named prior methods
anywhere"), quote the SHORTEST sentence/phrase where a reviser would most
naturally ADD the missing content — but still only VERBATIM from the plan.

**CONCISENESS**: Keep your <reasoning> concise (≤ 400 words). The locus and
score are the critical outputs. Avoid long exposition.

Format (output exactly this structure after the evaluation block):
<locus>
[
  {{"section_hint": "Motivation", "quote": "...", "why": "..."}}
]
</locus>
"""


def build_single_signal_prompt(
    goal: str, plan: str, signal: SignalSpec,
    *, emit_locus: bool = False, emit_critique: bool = False,
    include_cot: bool = True,
) -> str:
    """Build a prompt that asks the grader to score ONE signal.

    Args:
        emit_locus: When True AND `signal.locus_directive` is set, append a
            <locus> output block requesting verbatim weak-span quotes. Default
            False: standard grading pass (Pass 1) does not request loci. Only
            the CR-v6 bottleneck re-grading pass (Pass 2) sets emit_locus=True.
        emit_critique: When True, require a <critique> block with 1-3
            sentences describing what content is missing, weak, or
            unconvincing for this signal. Actionable and specific, but NOT
            verbatim quotes (separate from locus, which IS verbatim). Used
            by CR-v7 per-signal context REINFORCE to generate a per-signal
            advice string that is stored in the buffer and used at both
            revision-sampling time (all 8 critiques concatenated) and
            loss-computation time (single critique per context_i).
    """
    signal_block = _format_signal_block(signal, include_cot=include_cot)
    plan = _truncate_plan(plan)
    locus_block = _build_locus_output_block(signal) if emit_locus else ""
    critique_output_block = ""
    if emit_critique:
        critique_output_block = (
            f"        <critique>In 1-3 sentences, explain WHAT content is "
            f"missing, weak, or unconvincing for {signal.name} per the rubric "
            f"above. Be actionable and specific; describe the deficiency "
            f"conceptually. Do NOT quote plan text verbatim.</critique>\n"
        )

    return f"""{_shared_preamble(signal.score_max)}

You will evaluate the following research plan on ONE dimension: **{signal.name}**.

# Research Goal
{goal}

# Research Plan
{plan}

# Evaluation Dimension

{signal_block}

---

# Output Format

<evaluation>
    <dim id="{signal.id}">
        <reasoning>Your analysis (do required reasoning steps if any).</reasoning>
        <score>INTEGER 1-{signal.score_max}</score>
{critique_output_block}    </dim>
</evaluation>{locus_block}

Begin your evaluation now."""


# =============================================================================
# Parsing
# =============================================================================

def parse_scores(xml_text: str) -> dict[str, dict]:
    """Parse evaluation XML into {signal_id: {score, reasoning, critique}}.

    Robust to malformed XML from weaker graders (e.g., 4B models).
    Logs a warning if fewer than 50% of signals are successfully parsed.

    The `critique` field is the optional prose emitted when
    `build_single_signal_prompt(..., emit_critique=True)` was used; empty
    string when not present (CR-v7 per-signal context REINFORCE).
    """
    results = {}
    dim_blocks = re.findall(
        r'<dim\s+id="([^"]+)">(.*?)</dim>',
        xml_text,
        flags=re.DOTALL,
    )
    for dim_id, body in dim_blocks:
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", body, re.DOTALL)
        score_match = re.search(r"<score>\s*(\d+)\s*</score>", body, re.DOTALL)
        critique_match = re.search(r"<critique>(.*?)</critique>", body, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        score = int(score_match.group(1)) if score_match else None
        critique = critique_match.group(1).strip() if critique_match else ""
        if score is not None and score >= 1:
            results[dim_id] = {"score": score, "reasoning": reasoning, "critique": critique}
        else:
            results[dim_id] = {"score": None, "reasoning": reasoning, "critique": critique}

    n_parsed = sum(1 for v in results.values() if v["score"] is not None)
    n_dims_found = len(dim_blocks)
    # In separate_call mode only 1 dim is expected; warn only when parsing
    # fails relative to how many dims the grader actually returned.
    if n_dims_found == 0 or (n_dims_found > 1 and n_parsed < n_dims_found * 0.5):
        logger.warning(
            f"parse_scores: {n_parsed}/{n_dims_found} dims parsed successfully. "
            f"Grader output may be malformed (first 200 chars: {xml_text[:200]!r})"
        )
    return results


def aggregate_critique_across_repeats(
    parsed_per_repeat: list[dict[str, dict]],
    median_scores: dict[str, int | None],
) -> dict[str, str]:
    """Pick one critique per signal from `grader_repeats` parsed outputs.

    Strategy: for each signal, prefer the critique from the repeat whose
    score equals the median used downstream. Ties broken by first non-empty
    critique. If no repeat has the median score or all critiques are empty,
    return the first non-empty critique encountered; fall back to "".

    Used by CR-v7 to land a single critique string per signal in buffer
    entries, consistent with the score actually used in the aggregate.

    Args:
        parsed_per_repeat: one `parse_scores` output per grader repeat.
            Each element maps signal_id -> {score, reasoning, critique}.
        median_scores: the median score per signal_id (as used for
            aggregate_reward). Values may be None when all repeats failed.

    Returns:
        dict signal_id -> critique string (possibly empty).
    """
    out: dict[str, str] = {}
    # Collect every signal_id mentioned in any repeat
    all_sids: set[str] = set()
    for parsed in parsed_per_repeat:
        all_sids.update(parsed.keys())

    for sid in all_sids:
        target_score = median_scores.get(sid)
        # Candidate critiques from repeats matching the median score
        median_matches = []
        non_empty = []
        for parsed in parsed_per_repeat:
            entry = parsed.get(sid)
            if not entry:
                continue
            crit = entry.get("critique", "") or ""
            crit = crit.strip()
            if crit:
                non_empty.append(crit)
                if target_score is not None and entry.get("score") == target_score:
                    median_matches.append(crit)
        if median_matches:
            out[sid] = median_matches[0]
        elif non_empty:
            out[sid] = non_empty[0]
        else:
            out[sid] = ""
    return out


# =============================================================================
# Locus parsing (CR-v6)
# =============================================================================

@dataclass
class LocusEntry:
    """A single locus entry parsed from a grader's <locus> block.

    section_hint: Section name hint (e.g., "Methodology"); "" if grader unsure
    quote: Verbatim substring of the plan body (≤ 250 chars by prompt
        convention; hard invariant: `quote in plan_body` is True)
    why: 1-2 sentence explanation of why the span hurts the signal
    """
    section_hint: str
    quote: str
    why: str


_LOCUS_BLOCK_RE = re.compile(r"<locus>\s*(.*?)\s*</locus>", re.DOTALL | re.IGNORECASE)


def _find_best_verbatim_match(
    approximate_quote: str, plan_body: str, min_ratio: float = 0.50,
) -> str | None:
    """Find the plan substring that best matches an approximate grader quote.

    Qwen3-30B grader frequently paraphrases: condenses markdown lists into
    single sentences, strips **bold** formatting, truncates with "...". The
    SEMANTIC target is usually correct but the literal text doesn't match.

    Strategy: slide a LINE-BASED window over the plan (natural boundaries),
    normalize both sides (strip markdown + collapse whitespace), and score
    with SequenceMatcher. Return the raw (un-normalized) plan span.

    Returns the best-matching plan substring, or None if no match exceeds
    `min_ratio`. The returned string is guaranteed `in plan_body`.

    Discovered in Phase 1.5 smoke test: 0/10 plans had literal-verbatim
    locus quotes despite all 10 having semantically correct loci.
    """
    from difflib import SequenceMatcher

    def _norm(s: str) -> str:
        s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)       # strip bold
        s = re.sub(r"^\s*[-*+]\s+", " ", s, flags=re.MULTILINE)  # strip list bullets
        return " ".join(s.split()).lower()

    q_norm = _norm(approximate_quote)
    if len(q_norm) < 10:
        return None

    lines = plan_body.split("\n")
    best_ratio = 0.0
    best_span: str | None = None

    # Try all contiguous line windows up to 15 lines (covers most sections)
    max_window = min(15, len(lines))
    for start in range(len(lines)):
        for end in range(start + 1, min(start + max_window + 1, len(lines) + 1)):
            candidate_raw = "\n".join(lines[start:end])
            c_norm = _norm(candidate_raw)
            if not c_norm:
                continue
            # Quick length filter: avoid comparing tiny/huge candidates
            len_ratio = len(c_norm) / max(len(q_norm), 1)
            if len_ratio < 0.2 or len_ratio > 6.0:
                continue
            r = SequenceMatcher(None, q_norm, c_norm).ratio()
            if r > best_ratio:
                best_ratio = r
                best_span = candidate_raw.strip()

    if best_ratio < min_ratio or best_span is None:
        return None

    # Verify the span is a verbatim substring of plan_body
    if best_span in plan_body:
        return best_span

    # strip() may have broken the exact-substring property for edge lines;
    # try the raw (un-stripped) version
    for start in range(len(lines)):
        for end in range(start + 1, min(start + max_window + 1, len(lines) + 1)):
            raw = "\n".join(lines[start:end])
            if raw.strip() == best_span and raw in plan_body:
                return raw

    return None


def parse_locus(response: str, plan_body: str) -> list[LocusEntry] | None:
    """Parse `<locus>...</locus>` from grader response with fuzzy verbatim matching.

    The grader frequently paraphrases plan text (condenses markdown lists,
    strips bold formatting, truncates). When the literal quote doesn't match,
    we use `_find_best_verbatim_match` to find the closest actual plan
    substring. The returned LocusEntry.quote is ALWAYS a verbatim substring
    of plan_body (guaranteed by the fuzzy-match → exact-recovery pipeline).

    Returns:
        list[LocusEntry]: parsed entries with corrected verbatim quotes.
            Empty list = grader emitted `<locus>[]</locus>` (score 5).
        None: no `<locus>` block, malformed JSON, or no entry could be
            matched (even fuzzily) to plan_body text. Callers fall back
            to legacy revision path.
    """
    match = _LOCUS_BLOCK_RE.search(response)
    if not match:
        return None

    payload = match.group(1).strip()
    try:
        raw = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(raw, list):
        return None

    entries: list[LocusEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        quote = item.get("quote")
        section_hint = item.get("section_hint", "")
        why = item.get("why", "")
        if not isinstance(quote, str) or not quote:
            return None
        if not isinstance(section_hint, str) or not isinstance(why, str):
            return None
        # Try exact match first (cheapest)
        if quote in plan_body:
            matched_quote = quote
        else:
            # Fuzzy match: find the best actual plan substring
            matched_quote = _find_best_verbatim_match(quote, plan_body)
            if matched_quote is None:
                return None  # Can't locate this locus at all
        entries.append(LocusEntry(
            section_hint=section_hint,
            quote=matched_quote,
            why=why,
        ))

    return entries


# =============================================================================
# Aggregation
# =============================================================================

def median_over_repeats(
    repeat_scores: list[dict[str, dict]],
    signals: list[SignalSpec] | None = None,
) -> dict[str, int | None]:
    """Take the median score for each signal across multiple grading repeats."""
    active = signals or SIGNALS
    if not repeat_scores:
        return {s.id: None for s in active}

    out = {}
    for spec in active:
        vals = []
        for r in repeat_scores:
            info = r.get(spec.id) or {}
            score = info.get("score")
            if score is not None:
                vals.append(score)
        if len(vals) == 0:
            out[spec.id] = None
        else:
            out[spec.id] = int(statistics.median_low(vals))
    return out


def confidence_over_repeats(
    repeat_scores: list[dict[str, dict]],
    signals: list[SignalSpec] | None = None,
) -> dict[str, float | None]:
    """Compute per-signal grader confidence from multi-sample variance.

    confidence = 1 - (range / (score_max - 1)), where range = max - min
    across samples. All samples agree → confidence=1.0. Max disagreement
    (1 and score_max) → confidence=0.0.

    Literature: "Cycles of Thought" (2024), "Confidence Improves
    Self-Consistency" (2025) — sample variance > verbalized confidence.

    Returns None for a signal if < 2 samples (can't measure variance).
    """
    active = signals or SIGNALS
    out: dict[str, float | None] = {}
    if not repeat_scores or len(repeat_scores) < 2:
        return {s.id: None for s in active}

    for spec in active:
        vals = []
        for r in repeat_scores:
            info = r.get(spec.id) or {}
            score = info.get("score")
            if score is not None:
                vals.append(score)
        if len(vals) < 2:
            out[spec.id] = None
        else:
            rng = max(vals) - min(vals)
            denom = max(1, spec.score_max - 1)
            out[spec.id] = max(0.0, min(1.0, 1.0 - rng / denom))
    return out


# =============================================================================
# Sanity prints
# =============================================================================

if __name__ == "__main__":
    print(f"D4-v7 Signals: {len(SIGNALS)}")
    for s in SIGNALS:
        w = SIGNAL_WEIGHTS.get(s.id, 0)
        print(f"  {s.id:30s} w={w:.2f}  {s.name}")
    print(f"\nWeights sum: {sum(SIGNAL_WEIGHTS.values()):.4f}")
