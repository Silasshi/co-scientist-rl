"""Grant Rubric v8: 10 signals with stricter thresholds for Qwen self-grading viability.

Designed 2026-04-22 after systematic review of D4-v7 signal behavior across 58 runs.

Key changes from D4-v7 (12 signals):
- REMOVED G3 (Technical Evidence): rewards hallucinated citations (C3 SDPO: Qwen 5/5, Opus grounding 2/10)
- REMOVED G5 (Gap Identification): 99.8% saturation in D4-v7, zero discrimination
- DOWNWEIGHTED G12 (Formalism): 0.12 -> 0.04 (Qwen can't verify "non-trivial" math)
- RAISED thresholds for counting-based signals: G1/G2/G8/G10 require more items for top score
- UPWEIGHTED discriminating signals: G4, G6 (0.12 -> 0.15), G9 (0.08 -> 0.12), G13 (0.10 -> 0.12)

Core insight: counting rubrics with low thresholds enable Goodhart. Either raise thresholds
(make top score require genuinely strong evidence) or downweight/remove signals whose
pattern-match count fails to track quality. The signals retained with higher weight are
those showing real discrimination in D4-v7 data (std > 1.0 across 1655 plans).

Reference: projects/grant_proposal_v2/knowledge/current/SIGNAL_DESIGN_NOTES.md
"""

from __future__ import annotations

from co_scientist.shared.grant_signal_reward import SignalSpec


SIGNALS: list[SignalSpec] = [
    # =====================================================================
    # Problem definition (G1, G2) — kept with stricter thresholds
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
            "gap, or failure mode.\n"
            "  * GENERIC: vague importance or motivation ('X is important', "
            "'challenges remain', 'more work is needed').\n"
            "\n"
            "STEP 3 — If SPECIFIC, count N_facets. A 'facet' must be an "
            "INDEPENDENT technical sub-problem, not a reframing or "
            "consequence of the same problem.\n"
            "\n"
            "STRICT RULE: Two facets that share the same root cause count "
            "as ONE facet. Re-check each quoted facet for independence.\n"
            "\n"
            "STEP 4 — Apply this TABLE (v8 stricter thresholds):\n"
            "  GENERIC, no specific problem → score 1\n"
            "  SPECIFIC but only 1 facet → score 2\n"
            "  SPECIFIC with 2 independent facets → score 3\n"
            "  SPECIFIC with 3 independent facets → score 4\n"
            "  SPECIFIC with ≥4 independent facets → score 5\n"
            "  Mixed (starts generic, later specific) → score 2\n"
        ),
        scoring_rubric=(
            "v8 stricter thresholds (was: 3 facets=5):\n"
            "1: Only generic framing\n"
            "2: One specific problem, or mixed\n"
            "3: Two independent specific facets\n"
            "4: Three independent specific facets\n"
            "5: Four or more independent specific facets (rare)\n"
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
            "DETERMINED. Vague aspirations ('explore', 'investigate', "
            "'contribute to understanding') are NOT measurable aims."
        ),
        cot_scaffolding=(
            "STEP 1 — List every specific aim or objective stated in the "
            "proposal:\n"
            "  * MEASURABLE: states what will be produced, determined, or "
            "delivered.\n"
            "  * VAGUE: 'explore', 'investigate', 'contribute to "
            "understanding', 'advance the field'.\n"
            "\n"
            "STEP 2 — For each MEASURABLE aim, check: does it specify an "
            "EXPECTED OUTCOME or deliverable? Count N_with_outcome.\n"
            "\n"
            "STEP 3 — Check independence: aims must target DIFFERENT "
            "sub-problems, not rewordings of the same goal.\n"
            "\n"
            "STEP 4 — Count N_measurable_aims that are also INDEPENDENT.\n"
            "\n"
            "STEP 5 — Apply this TABLE (v8 stricter):\n"
            "  N_measurable = 0 → score 1\n"
            "  N_measurable = 1, no outcomes → score 2\n"
            "  N_measurable = 1, with outcome → score 2\n"
            "  N_measurable = 2, at least 1 with outcome → score 3\n"
            "  N_measurable = 3, all with outcomes → score 4\n"
            "  N_measurable ≥ 4, all with outcomes → score 5\n"
        ),
        scoring_rubric=(
            "v8 stricter thresholds (was: 3 aims=5):\n"
            "1: No measurable aims (only vague aspirations)\n"
            "2: One measurable aim, or multiple without outcomes\n"
            "3: Two independent measurable aims, ≥1 with outcome\n"
            "4: Three independent aims, all with outcomes\n"
            "5: Four or more independent aims, all with outcomes\n"
        ),
        locus_directive=(
            "Quote sentences containing VAGUE aims ('explore', 'investigate', "
            "'contribute to understanding') used in place of measurable aims."
        ),
    ),

    # =====================================================================
    # REMOVED: G3 Technical Evidence
    # Reason: Rewards hallucinated citations. C3 SDPO best plan scored G3=5/5
    # from Qwen but Opus grounding=2/10 due to fabricated references.
    # Future: re-add as RAG-verified G3 if citation verification infrastructure built.
    # =====================================================================

    # =====================================================================
    # Research design (G4, G6) — KEPT with raised weight (best discriminators)
    # =====================================================================
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
            "STEP 1 — List the distinct METHODOLOGICAL techniques named in "
            "the proposal as part of the central approach.\n"
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
            "technique is needed for THE STATED PROBLEM (a causal or "
            "mechanistic reason that references the problem).\n"
            "  * UNJUSTIFIED: technique is named but no explanation of "
            "why it is needed, OR justification is generic ('well-known "
            "method', 'state-of-the-art') without linking to THIS problem.\n"
            "\n"
            "STRICT RULE: A generic 'because it is effective' is NOT "
            "justification. The sentence must link to a specific problem "
            "property. Examples of justification:\n"
            "  - GOOD: 'We use SDEs because the problem's heavy-tailed "
            "gradients violate the smoothness assumption required by ODE "
            "methods.'\n"
            "  - BAD: 'We use SDEs because they are a powerful modeling "
            "framework.'\n"
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
            "Focus is a function of BOTH technique count and justification ratio. "
            "Few techniques, justified = 5. Many techniques, unjustified = 1.\n"
            "1: ≥7 techniques with ≥5 unjustified (severe stacking)\n"
            "2: Moderate stacking or many unjustified\n"
            "3: Borderline (2-3 unjustified of moderate total)\n"
            "4: Focused or mostly justified\n"
            "5: Tight focus (≤3 techniques) with every one justified to problem\n"
        ),
        locus_directive=(
            "Quote each technique that is UNJUSTIFIED — named without a "
            "mechanistic sentence explaining why it is needed for THIS "
            "proposal's specific problem."
        ),
    ),

    # =====================================================================
    # REMOVED: G5 Gap Identification (99.8% saturated at 5/5, zero gradient)
    # Gap identification is partially covered by G4 (justified to problem) and
    # G13 (risk awareness / scope boundary).
    # =====================================================================

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
            "approach.\n"
            "\n"
            "DO NOT count: standard infrastructure, named datasets unless "
            "novel, standard hyperparameters, evaluation benchmarks.\n"
            "\n"
            "STEP 3 — For EACH choice, classify as:\n"
            "  * JUSTIFIED: proposal contains a sentence stating WHY this "
            "choice addresses the stated problem (causal or mechanistic "
            "LINK to problem).\n"
            "  * ASSERTED: proposal names the choice but gives no 'why' "
            "OR gives a generic why ('state-of-the-art', 'well-known') "
            "without linking to the problem.\n"
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
            "justification. 'because it is powerful' or 'because it has "
            "been shown to work' is NOT justification. A citation PLUS a "
            "mechanistic sentence linking to THIS problem IS justified."
        ),
        scoring_rubric=(
            "1: ≥4 design choices asserted (list of techniques, no argument)\n"
            "2: 3 asserted\n"
            "3: 2 asserted\n"
            "4: 1 asserted (mostly justified)\n"
            "5: 0 asserted — every design choice has an explicit why-sentence "
            "linked to the stated problem\n"
        ),
        locus_directive=(
            "Quote each design choice that is ASSERTED — named without a "
            "mechanistic 'why' sentence explaining how it addresses the "
            "proposal's stated problem."
        ),
    ),

    # =====================================================================
    # Output / proposal structure (G8, G9, G10) — G8/G10 stricter; G9 upweighted
    # =====================================================================
    SignalSpec(
        id="G8_deliverable_clarity",
        name="Deliverable Clarity",
        question=(
            "Does the proposal name SPECIFIC deliverables with substantive "
            "content? Generic outcome claims ('will advance the field', "
            "'will contribute to knowledge') are NOT deliverables. Template "
            "deliverables ('publish papers', 'release data') without "
            "specific content are also NOT specific."
        ),
        cot_scaffolding=(
            "STEP 1 — List every SPECIFIC deliverable named in the proposal:\n"
            "  * SPECIFIC: has substantive content attached, e.g.:\n"
            "    - 'publish a paper on [specific topic X]'\n"
            "    - 'release dataset [name] with [N] samples covering [domain]'\n"
            "    - 'produce a guideline for [specific audience] on [topic]'\n"
            "    - 'develop open-source software [name] that does [function]'\n"
            "  * GENERIC (do NOT count):\n"
            "    - 'publish papers' (no topic)\n"
            "    - 'release data' (no description)\n"
            "    - 'advance the field', 'contribute to knowledge'\n"
            "\n"
            "STRICT RULE: A deliverable that is just a TYPE ('we will publish') "
            "with no substance is GENERIC. Must have either a topic, a dataset "
            "name, a specific audience, or a concrete function.\n"
            "\n"
            "STEP 2 — Count N_specific_deliverables.\n"
            "\n"
            "STEP 3 — Apply this TABLE (v8 stricter: was 4+=5, now 5+=5):\n"
            "  N_specific = 0 → score 1\n"
            "  N_specific = 1 → score 2\n"
            "  N_specific = 2 → score 3\n"
            "  N_specific = 3 → score 3\n"
            "  N_specific = 4 → score 4\n"
            "  N_specific ≥ 5 → score 5\n"
        ),
        scoring_rubric=(
            "v8 stricter thresholds + stricter 'specific' definition:\n"
            "1: No specific deliverables named\n"
            "2: One specific deliverable with substantive content\n"
            "3: Two or three specific deliverables\n"
            "4: Four specific deliverables\n"
            "5: Five or more specific deliverables with substantive content\n"
        ),
        locus_directive=(
            "Quote sentences with GENERIC deliverable templates ('publish papers', "
            "'release data') or GENERIC outcome claims ('will advance the field')."
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
            "STEP 1 — List all scope commitments in the proposal (number "
            "of countries, databases, analyses, aims, sub-studies, etc.).\n"
            "\n"
            "STEP 2 — Identify scope RED FLAGS:\n"
            "  * OVERLY BROAD claims: 'comprehensive analysis of 4,230 laws', "
            "'all 193 UN member states', 'every possible scenario'\n"
            "  * MISMATCH between scope and resources: 2-year project claiming "
            "to comprehensively analyze thousands of items\n"
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
            "1: Three or more scope red flags — project appears infeasible\n"
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
            "Does the proposed approach/methodology cover ALL stated aims "
            "WITH SPECIFIC methods (not vague hand-waves)? An aim with no "
            "corresponding method, or method that is only vaguely described "
            "('use appropriate techniques'), is an UNCOVERED aim."
        ),
        cot_scaffolding=(
            "STEP 1 — List all stated aims/objectives from the proposal. "
            "Count N_aims.\n"
            "\n"
            "STEP 2 — For each aim, check the corresponding methodology:\n"
            "  * SPECIFIC: methodology names concrete procedures, data "
            "sources, algorithms, OR analytical frameworks.\n"
            "  * VAGUE: method is hand-waved ('we will use appropriate "
            "methods', 'standard techniques', 'rigorous approach') without "
            "naming specifics.\n"
            "  * MISSING: aim has no corresponding method section.\n"
            "\n"
            "STRICT RULE (v8): VAGUE methods do NOT count as covered.\n"
            "\n"
            "STEP 3 — Mark each aim COVERED (specific method) or "
            "UNCOVERED (vague or missing).\n"
            "\n"
            "STEP 4 — Calculate coverage = N_covered / N_aims.\n"
            "\n"
            "STEP 5 — Apply this TABLE:\n"
            "  coverage = 100% → score 5\n"
            "  coverage ≥ 75% → score 4\n"
            "  coverage ≥ 50% → score 3\n"
            "  coverage ≥ 25% → score 2\n"
            "  coverage < 25% → score 1\n"
        ),
        scoring_rubric=(
            "v8: 'covered' requires SPECIFIC methodology (no vague hand-waves).\n"
            "1: Less than 25% of aims have specific methodology\n"
            "2: 25-49% coverage\n"
            "3: 50-74% coverage\n"
            "4: 75-99% coverage\n"
            "5: 100% — every aim has a specific, named method\n"
        ),
        locus_directive=(
            "Quote sentences stating aims or objectives for which NO specific "
            "methodology is described, or where methodology is only vaguely "
            "gestured at ('appropriate methods', 'standard techniques')."
        ),
    ),

    # =====================================================================
    # Rigor signals (G11, G12, G13) — G11 kept; G12 downweighted; G13 upweighted
    # =====================================================================
    SignalSpec(
        id="G11_evidence_rigor",
        name="Evidence Rigor",
        question=(
            "Can the proposed evaluation support the claims? For EMPIRICAL "
            "work: are baselines FAIR (appropriate, modern, matched in "
            "scale — not strawman) and is there an OPERATIONALIZED success "
            "criterion (specific metric with threshold)? For THEORETICAL "
            "work: are assumptions stated and is a proof sketch/strategy "
            "given?"
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
            "old in a fast-moving field, or from a different problem "
            "regime).\n"
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
            "    N_fair ≥ 3 AND criterion OPERATIONALIZED + ablations → "
            "score 5\n"
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
            "5: ≥3 fair baselines + operationalized criterion + ablations\n"
        ),
        locus_directive=(
            "Quote each STRAWMAN baseline OR each VAGUE success-criterion "
            "sentence ('expect', 'should', 'aim to' without a measurable "
            "threshold)."
        ),
    ),

    SignalSpec(
        id="G12_formalism",
        name="Mathematical Formalism",
        question=(
            "Does the proposal contain NON-TRIVIAL mathematical content? "
            "Non-trivial means: a named equation written in symbolic form "
            "that is NOT tautological, NOT just hyperparameter values, and "
            "NOT standard definitions."
        ),
        cot_scaffolding=(
            "STEP 1 — Scan the ENTIRE proposal for EQUATIONS written in "
            "symbolic form. An EQUATION contains:\n"
            "  - mathematical symbols (=, +, -, *, /, ^, log, exp, sum, "
            "integral, E[], nabla, O())\n"
            "  - AND at least one NAMED variable or function beyond "
            "simple constants\n"
            "\n"
            "List each equation found.\n"
            "\n"
            "STEP 2 — For EACH equation, classify as:\n"
            "  * TAUTOLOGICAL: L = -R, y = f(x) without defining f, or any "
            "formula that restates the problem without analytical content.\n"
            "  * HYPERPARAMETER-ONLY: lr=3e-5, batch=32, etc.\n"
            "  * NON-TRIVIAL: A formula that adds analytical content.\n"
            "\n"
            "STEP 3 — Count N_nontrivial.\n"
            "\n"
            "STEP 4 — Apply this TABLE:\n"
            "  N_nontrivial = 0 → score 1\n"
            "  N_nontrivial = 1 → score 2\n"
            "  N_nontrivial = 2 → score 3\n"
            "  N_nontrivial = 3 → score 4\n"
            "  N_nontrivial ≥ 4 → score 5\n"
            "\n"
            "STRICT RULE: Hyperparameter values are NEVER equations. "
            "Section headers with math words are NOT formulas."
        ),
        scoring_rubric=(
            "v8 downweighted (0.12 -> 0.04) because Qwen can't verify non-triviality reliably.\n"
            "1: Zero non-trivial formulas\n"
            "2: One non-trivial formula\n"
            "3: Two non-trivial formulas\n"
            "4: Three non-trivial formulas\n"
            "5: Four or more non-trivial formulas\n"
        ),
        locus_directive=(
            "Quote sentences where a mathematical claim is made in prose "
            "without an accompanying symbolic equation."
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
            "strategies."
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
            "6. Assign a score.\n"
        ),
        scoring_rubric=(
            "1: No risk awareness. Proposal assumes success with no discussion "
            "of failure modes or limitations.\n"
            "2: Generic disclaimers ('there might be limitations', 'further "
            "research is needed') without specificity.\n"
            "3: Partial. Names 1 failure mode OR mentions scope boundary, but "
            "risk structure is incomplete.\n"
            "4: Mature. Names 2-3 specific failure modes, identifies fragile "
            "assumptions, states scope boundary.\n"
            "5: Exceptional. Coherent risk model: specific failure modes, "
            "load-bearing assumptions identified, scope boundary explicit, "
            "AND fallback direction sketched.\n"
        ),
        locus_directive=(
            "Quote GENERIC risk disclaimers ('there might be limitations', "
            "'further research is needed', 'various challenges exist'). If "
            "the proposal lacks specific failure modes, quote the last "
            "sentence of the proposal."
        ),
    ),
]


# =============================================================================
# Weights (v8): 10 signals, normalized to 1.0
# =============================================================================
# Design principles:
# - Discriminating signals (D4-v7 data: G4/G6/G9/G13 std > 0.5) upweighted
# - Saturated/gameable signals (G1/G2/G8/G10/G12) downweighted
# - G12 formalism kept but downweighted to 0.04 (Qwen can't verify reliably)
# - G9 scope feasibility upweighted (hardest to Goodhart - red flag counting)
# =============================================================================
SIGNAL_WEIGHTS: dict[str, float] = {
    "G1_problem_specificity":  0.04,
    "G2_specific_aims":        0.04,
    "G4_focus":                0.16,  # was 0.12 - most discriminating
    "G6_reasoning_depth":      0.16,  # was 0.12 - most discriminating
    "G8_deliverable_clarity":  0.08,
    "G9_scope_feasibility":    0.14,  # was 0.08 - inverse signal, hardest to game
    "G10_approach_coverage":   0.06,  # was 0.04 - slight bump for v8 stricter version
    "G11_evidence_rigor":      0.14,  # was 0.12
    "G12_formalism":           0.04,  # was 0.12 - unverifiable
    "G13_risk_awareness":      0.14,  # was 0.10 - qualitative, hardest to game
}

assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 1e-6, \
    f"Weights must sum to 1.0, got {sum(SIGNAL_WEIGHTS.values())}"

SCORE_MAX: dict[str, int] = {s.id: s.score_max for s in SIGNALS}
