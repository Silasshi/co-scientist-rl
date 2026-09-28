"""Grant Rubric v10: v8 + selective v9 backports with length caps + relaxed Opus-ρ-preserving G11-13.

Designed 2026-04-23. Supersedes v9 after panel test showed v9's strict 3-tuple
requirements crashed Opus-ρ on G11/G12/G13 (0.91-0.93 → 0.23-0.63, G13 var=0.21).

Relationship to v8/v9:
- G1/G2/G6/G8/G9: verbatim v8
- G4: v9 composition-logic gate + count cap (≤3 composition statements)
- G10: v9 3-tuple (procedure+dataset+metric) + per-component ≤15 word cap
- G11: v9 citation-format gate ONLY (drops v9's usage+scale requirements that
  divided Opus/GPT-OSS). Primary SDPO hallucination-hack defense — catches
  bare 'arXiv:xxxx' fabrications without author+year wrapper.
- G12: keep v8 N_nontrivial count + v9 vars-defined requirement;
  used-downstream as optional bonus (counted as 1.5)
- G13: 2-tuple (trigger + fallback) instead of v9 3-tuple. Detection method
  as optional bonus. v9's 3-tuple was too strict — Opus gave 11/15 plans
  score 1.

Weights unchanged from v8. Addresses:
1. Length bias on G11/G12/G13 (partially — R² expected 0.2-0.4 vs v8 0.5-0.9)
2. Dead channels G4/G10 (fully — v9 mechanism, proven +0.75-0.79 Opus-ρ)
3. SDPO hallucination hacking (G11 citation format; v7 C3 failure mode)
4. Post-composition length creep (count caps on G4/G10)

Reference:
- projects/grant_proposal_v2/analysis/v7_vs_v8_comparison.md
- projects/grant_proposal_v2/analysis/grader_panel_v9/analysis/v8_vs_v9_comparison.md
- .claude/plans/quirky-herding-emerson.md (approved 2026-04-23)
"""

from __future__ import annotations

from co_scientist.shared.grant_signal_reward import SignalSpec


SIGNALS: list[SignalSpec] = [
    # =====================================================================
    # Problem definition (G1, G2) — UNCHANGED from v8
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
    # Research design (G4 REWRITTEN in v10, G6 UNCHANGED)
    # =====================================================================
    SignalSpec(
        id="G4_focus",
        name="Research Focus",
        question=(
            "Does the proposal STAY focused on a small number of core "
            "techniques AND state the COMPOSITION LOGIC — how those "
            "techniques combine into a pipeline, and why the ordering is "
            "needed? A focused proposal has 1-3 core techniques that clearly "
            "compose; the best proposals explicitly state which technique "
            "feeds into which. A stacked proposal lists many techniques "
            "without composition logic or without justifying each."
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
            "property.\n"
            "\n"
            "STEP 3 — Count (N_total, N_unjustified).\n"
            "\n"
            "STEP 4 (v10 rewrite) — check COMPOSITION LOGIC with count cap:\n"
            "  A COMPOSITION STATEMENT identifies a specific input/output "
            "flow between techniques, in ≤1 sentence. Examples:\n"
            "    - 'The SDE analysis (technique A) yields curvature bounds "
            "that seed the Bayesian optimizer (technique B).'\n"
            "    - 'Output of NSGA-II (technique C) is post-filtered by the "
            "SHAP interpretability check (technique D).'\n"
            "  Generic statements ('methods complement each other', 'used "
            "together', 'integrated pipeline') are NOT composition logic.\n"
            "\n"
            "STRICT RULE (v10): Count at MOST 3 distinct composition "
            "statements. Extra statements beyond 3 are ignored — prevents "
            "length-gaming by padding the composition section. Mark "
            "'COMPOSITION STATED' if N_composition ≥ 2 AND the statements "
            "cover distinct technique pairs (not the same pair restated).\n"
            "\n"
            "STRICT RULE: If only 1 core technique, COMPOSITION STATED is "
            "vacuously true.\n"
            "\n"
            "STEP 5 — Apply this TABLE (v10 = v9 + count cap):\n"
            "  N_total ≤ 3 AND N_unjustified = 0 AND composition stated → 5\n"
            "  N_total ≤ 3 AND N_unjustified = 0 AND composition absent → 4\n"
            "  N_total ≤ 3 AND N_unjustified ≥ 1 → 3\n"
            "  N_total ∈ [4, 6] AND N_unjustified ≤ 1 AND composition stated → 4\n"
            "  N_total ∈ [4, 6] AND N_unjustified ≤ 1 AND composition absent → 3\n"
            "  N_total ∈ [4, 6] AND N_unjustified ∈ [2, 3] → 2\n"
            "  N_total ∈ [4, 6] AND N_unjustified ≥ 4 → 2\n"
            "  N_total ≥ 7 AND N_unjustified ≤ 2 AND composition stated → 3\n"
            "  N_total ≥ 7 AND N_unjustified ≤ 2 AND composition absent → 2\n"
            "  N_total ≥ 7 AND N_unjustified ≥ 3 → 1\n"
        ),
        scoring_rubric=(
            "v10: adds composition-logic gate with count cap ≤3. Middle tier "
            "(score 3) now requires either unjustified techniques OR absence "
            "of composition. Top scores (4-5) require composition logic.\n"
            "1: ≥7 techniques with many unjustified\n"
            "2: Moderate stacking or many unjustified, no composition\n"
            "3: Borderline — either some unjustified or composition absent\n"
            "4: Focused + composition stated, or justified stacking\n"
            "5: Tight focus (≤3 justified techniques) + composition stated\n"
        ),
        locus_directive=(
            "Quote each UNJUSTIFIED technique. If composition is absent, "
            "quote the sentence introducing the technique list. If more "
            "than 3 composition statements are present, quote the "
            "4th/5th/etc. statements (ignored for scoring) as evidence "
            "of padding."
        ),
    ),

    # =====================================================================
    # REMOVED: G5 Gap Identification (99.8% saturated in v7/v8)
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
    # Output / proposal structure (G8 UNCHANGED, G9 UNCHANGED, G10 REWRITTEN)
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
            "with FULLY SPECIFIED methods? In v10, a method is FULLY "
            "SPECIFIED only if the proposal names ALL THREE of: (a) a "
            "concrete procedure or algorithm, (b) a concrete input data "
            "source or dataset, and (c) a concrete measurable output or "
            "metric. Each of the three components must be expressible in "
            "≤15 words — verbose components are treated as padding and "
            "do NOT count."
        ),
        cot_scaffolding=(
            "STEP 1 — List all stated aims/objectives from the proposal. "
            "Count N_aims.\n"
            "\n"
            "STEP 2 — For each aim, check the corresponding methodology "
            "for the FULL 3-tuple:\n"
            "  (a) PROCEDURE: a named algorithm, technique, or procedure "
            "(e.g., 'multinomial logistic regression', 'NSGA-II', "
            "'adaptive SGD with momentum β_t = ...'). Generic category "
            "names ('statistical analysis', 'ML methods', 'regression') "
            "are NOT procedures — must be specific.\n"
            "  (b) DATA SOURCE: a named dataset or data source (e.g., "
            "'MIMIC-III clinical records', 'UCI Adult income dataset', "
            "'synthetic data generated via procedure X'). 'Real-world "
            "data' or 'benchmark datasets' without names do NOT count.\n"
            "  (c) MEASURABLE OUTPUT: a named metric or target quantity "
            "with units or reference scale (e.g., 'convergence rate in "
            "gradient-norm', 'demographic parity violation ≤ 0.05', "
            "'Spearman ρ vs human ranking'). 'Performance improvement' "
            "or 'accuracy' without a concrete metric/threshold does NOT "
            "count.\n"
            "\n"
            "STEP 3 — v10 LENGTH CAP: for each component (a)/(b)/(c), "
            "check that it is stated in ≤15 words of plan text. If any "
            "component requires >15 words to state, padding is present; "
            "treat that component as NOT counting toward the 3-tuple.\n"
            "\n"
            "Examples of compliant (≤15 words per component):\n"
            "  * (a) 'NSGA-II with weighted-sum scalarization' (5 words)\n"
            "  * (b) 'UCI Adult dataset (48K rows)' (5 words)\n"
            "  * (c) 'demographic parity violation ≤ 0.05' (5 words)\n"
            "\n"
            "Example of padding (rejected):\n"
            "  * (a) 'a sophisticated multi-objective optimizer incorporating "
            "Pareto-efficiency constraints and adaptive penalty functions "
            "tuned across iterations' (18 words — too verbose; treat as "
            "vague, NOT a specific procedure).\n"
            "\n"
            "STEP 4 — Mark each aim:\n"
            "  * COVERED: all three (a), (b), (c) present AND each within "
            "15-word cap\n"
            "  * UNCOVERED: any of (a)/(b)/(c) missing, verbose, or aim "
            "lacks corresponding method section\n"
            "\n"
            "STRICT RULE (v10): VAGUE methods OR missing any of the "
            "three components OR >15 words for any component → UNCOVERED. "
            "Partial specification does not count.\n"
            "\n"
            "STEP 5 — Calculate coverage = N_covered / N_aims.\n"
            "\n"
            "STEP 6 — Apply this TABLE:\n"
            "  coverage = 100% → score 5\n"
            "  coverage ≥ 75% → score 4\n"
            "  coverage ≥ 50% → score 3\n"
            "  coverage ≥ 25% → score 2\n"
            "  coverage < 25% → score 1\n"
        ),
        scoring_rubric=(
            "v10: 'covered' requires FULL 3-tuple (procedure + data source + "
            "measurable output), EACH expressible in ≤15 words (length cap to "
            "prevent padding). v8 required only 'specific methodology' (any "
            "one of the three). v9 added 3-tuple. v10 adds per-component "
            "length cap.\n"
            "1: Less than 25% of aims have full 3-tuple methodology\n"
            "2: 25-49% coverage\n"
            "3: 50-74% coverage\n"
            "4: 75-99% coverage\n"
            "5: 100% — every aim has all three (procedure + dataset + metric)\n"
        ),
        locus_directive=(
            "Quote aim sentences for which methodology is missing any of: "
            "procedure name, dataset/data-source name, or measurable "
            "output/metric. Also quote method sentences that are verbose "
            "(>15 words for a single component) or that use generic "
            "category labels ('statistical analysis', 'ML methods') "
            "instead of a named procedure."
        ),
    ),

    # =====================================================================
    # Rigor signals (G11, G12, G13) — RELAXED from v9 for v10
    # =====================================================================
    SignalSpec(
        id="G11_evidence_rigor",
        name="Evidence Rigor",
        question=(
            "Can the proposed evaluation support the claims? For EMPIRICAL "
            "work in v10, each FAIR baseline must have a VALID CITATION: "
            "(Author, Year, Venue) or (Author et al., Year). Bare 'arXiv:xxxx' "
            "IDs without an author+year wrapper do NOT count as valid "
            "citations — this is v10's primary defense against SDPO "
            "hallucination hacking, where policies learned to fabricate "
            "arXiv IDs to appear rigorous. For THEORETICAL work: "
            "assumptions stated + proof sketch/strategy given + bounds "
            "or key lemma discussed."
        ),
        cot_scaffolding=(
            "STEP 1 — Identify if the proposed evaluation is EMPIRICAL, "
            "THEORETICAL, or both.\n"
            "\n"
            "STEP 2 (EMPIRICAL) — For each candidate baseline named in "
            "the proposal, check the CITATION FORMAT:\n"
            "  * VALID CITATION: Either '(Author, Year, Venue)' with a "
            "recognizable venue name (e.g., 'NeurIPS 2021', 'ICML 2020', "
            "'arXiv 2023' as venue), or '(Author et al., Year)' with both "
            "components present. The venue or year must be plausible.\n"
            "  * INVALID: Bare 'arXiv:1234.5678' without author/year "
            "wrapper. Bare author name without year. Bare acronym ('BERT') "
            "without author/year/venue. Citations with fabricated venue "
            "names (e.g., 'Smith et al., NeurIPS 2030' — future date).\n"
            "\n"
            "Classify each baseline as:\n"
            "  * FAIR: valid citation format AND the baseline is a modern, "
            "appropriate-scale method from the same problem space.\n"
            "  * STRAWMAN: obviously weak (random baseline, 5+ years old in "
            "a fast-moving field, or from a different regime).\n"
            "  * UNCITED: named but missing valid citation format.\n"
            "\n"
            "Count N_fair (citation-valid AND not strawman), N_strawman, "
            "N_uncited.\n"
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
            "STEP 4 — Apply this TABLE (v10 uses citation-verified count):\n"
            "  EMPIRICAL:\n"
            "    N_strawman ≥ 2 → score 1\n"
            "    N_strawman ≥ 1 AND criterion VAGUE → score 1\n"
            "    N_fair = 0 AND N_uncited ≥ 1 → score 2 (uncited baselines)\n"
            "    N_fair ≥ 2 AND criterion VAGUE → score 2\n"
            "    N_fair ≥ 2 AND criterion OPERATIONALIZED → score 4\n"
            "    N_fair ≥ 3 AND criterion OPERATIONALIZED + ablations → score 5\n"
            "    Otherwise → score 3\n"
            "\n"
            "  THEORETICAL:\n"
            "    T ≤ 1 → score 1-2\n"
            "    T = 2 → score 3\n"
            "    T = 3 + operationalized criterion → score 4-5\n"
            "\n"
            "STRICT RULE (v10): bare 'arXiv:xxxx' identifiers without a "
            "surrounding (Author, Year) wrapper count as UNCITED. This is "
            "the primary v10 defense against SDPO-style fabricated "
            "citation hacks. Do NOT also require a 'scale-match rationale' "
            "or a 'quoted usage sentence elsewhere in plan' — those v9 "
            "requirements proved too strict (Opus/GPT-OSS diverged)."
        ),
        scoring_rubric=(
            "v10 requires citation format verification (author+year+venue) "
            "per fair baseline. Relaxes v9's usage+scale requirements. "
            "Bare arXiv IDs without author/year are counted as uncited. "
            "This directly catches SDPO hallucination hacking.\n"
            "1: Strawman baselines or vague criterion with no valid citations\n"
            "2: Only uncited baselines (named without valid citation)\n"
            "3: One fair baseline with operationalized criterion\n"
            "4: ≥2 fair baselines + operationalized criterion\n"
            "5: ≥3 fair baselines + operationalized criterion + ablations\n"
        ),
        locus_directive=(
            "Quote each UNCITED baseline (named without valid (Author, "
            "Year, Venue) or (Author et al., Year) wrapper). Quote each "
            "STRAWMAN baseline (obviously weak, outdated, or from a "
            "different regime). Quote each VAGUE success-criterion "
            "sentence ('expect', 'should', 'aim to' without a measurable "
            "threshold)."
        ),
    ),

    SignalSpec(
        id="G12_formalism",
        name="Mathematical Formalism",
        question=(
            "Does the proposal contain non-trivial mathematical content "
            "with VARIABLES DEFINED? In v10, each non-trivial equation "
            "must have all non-standard variables defined inline or via "
            "notation section. Equations that are also referenced "
            "downstream (by label 'Eq. 3' or restated form) count as "
            "1.5 — partial bonus for used-downstream, but not required."
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
            "STEP 2 — For EACH equation, check the v10 BASE GATE:\n"
            "  (a) VARIABLES DEFINED: every non-standard symbol in the "
            "equation is either defined in the sentence introducing the "
            "equation, in a prior notation section, or is a standard "
            "symbol in the obvious convention (e.g., ∇L for gradient of "
            "loss). If any symbol is unexplained, fail (a).\n"
            "\n"
            "STEP 3 — Classify each equation for trivial / non-trivial:\n"
            "  * TAUTOLOGICAL: L = -R, y = f(x) without defining f, or any "
            "formula that restates the problem without analytical content.\n"
            "  * HYPERPARAMETER-ONLY: lr=3e-5, batch=32, etc.\n"
            "  * NON-TRIVIAL: a formula that adds analytical content.\n"
            "\n"
            "STEP 4 — An equation counts as 1.0 if NON-TRIVIAL AND (a) "
            "satisfied. BONUS: if the equation is ALSO referenced "
            "downstream — either by its label ('by (3)', 'using Eq. 2'), "
            "or by restating its symbolic form in a later sentence that "
            "derives/analyzes/applies it — it counts as 1.5 instead of "
            "1.0. Sum over all equations: N_equation_credit.\n"
            "\n"
            "STEP 5 — Apply this TABLE:\n"
            "  N_equation_credit < 1.0 → score 1\n"
            "  N_equation_credit ∈ [1.0, 1.9] → score 2\n"
            "  N_equation_credit ∈ [2.0, 2.9] → score 3\n"
            "  N_equation_credit ∈ [3.0, 3.9] → score 4\n"
            "  N_equation_credit ≥ 4.0 → score 5\n"
            "\n"
            "STRICT RULE (v10): Hyperparameter values are NEVER equations. "
            "Section headers with math words are NOT formulas. Equations "
            "with undefined variables count as 0 — the policy cannot "
            "game G12 by adding Greek letters without defining them. "
            "Used-downstream is a BONUS (1.5×), not a hard requirement, "
            "because Opus/GPT-OSS diverged on strict v9 'referenced by "
            "label' test."
        ),
        scoring_rubric=(
            "v10 relaxes v9's strict 'used-downstream' hard requirement "
            "to a 1.5× bonus, preserving the variables-defined gate. "
            "Reaching score 5 requires either 4+ defined equations (all "
            "with downstream use = 4 × 1.5 = 6 credits) or 5+ defined "
            "equations without downstream use.\n"
            "1: Zero equations with defined variables, or all tautological\n"
            "2: One equation (or partial credit)\n"
            "3: Two equations (or ~2.5 credits with bonus)\n"
            "4: Three equations (or ~3-3.9 credits)\n"
            "5: Four+ equations with downstream use, or 5+ without\n"
        ),
        locus_directive=(
            "Quote each equation that is either (a) missing variable "
            "definitions, (b) tautological / hyperparameter-only. For "
            "score purposes, also note (in reasoning, not quotes) "
            "equations that are defined but NOT referenced downstream — "
            "they still count for base credit, just not for the bonus."
        ),
    ),

    SignalSpec(
        id="G13_risk_awareness",
        name="Risk Awareness",
        question=(
            "Does the proposal demonstrate mature awareness of its own "
            "boundaries, risks, and failure modes? In v10, each identified "
            "risk must be a 2-TUPLE at minimum: (specific trigger "
            "condition) + (specific fallback action). Detection method is "
            "a BONUS (1.5×), not required. Generic disclaimers or "
            "laundry-list risks without triggers or fallbacks do NOT count. "
            "v9's 3-tuple (trigger+detection+fallback) was too strict — "
            "Opus gave 11/15 plans score 1 with variance 0.21, destroying "
            "discrimination."
        ),
        cot_scaffolding=(
            "STEP 1 — Find the risk/limitation/contingency discussion in "
            "the proposal. Quote each distinct risk item.\n"
            "\n"
            "STEP 2 — For EACH risk item, check the v10 2-TUPLE:\n"
            "  (a) TRIGGER CONDITION: a specific, observable condition "
            "under which this risk materializes. Examples:\n"
            "    - 'if the gradient-norm fails to decrease for 5 "
            "consecutive iterations'\n"
            "    - 'if fairness constraint causes feasible-region volume "
            "to drop below 1% of the unconstrained region'\n"
            "    - 'if ≥30% of training samples yield identical output'\n"
            "  Generic or non-measurable triggers ('if the method "
            "fails', 'if results are poor', 'if unexpected issues "
            "arise') do NOT count as (a).\n"
            "\n"
            "  (b) FALLBACK ACTION: a specific alternative direction "
            "or mitigation that would be executed if the trigger fires. "
            "Examples:\n"
            "    - 'switch to count-based rubric (see G11/G12/G13 v10 "
            "redesign)'\n"
            "    - 'fall back to n-gram length cap of 1500 words'\n"
            "    - 'restart with higher kl_budget'\n"
            "  'Adjust approach' or 'revise the methodology' without "
            "naming WHAT would change does NOT count as (b).\n"
            "\n"
            "BONUS (c) DETECTION METHOD: a specific procedure by which the "
            "trigger would be observed. Examples:\n"
            "    - 'monitor sliding-window gradient norm; alert if "
            "below threshold'\n"
            "    - 'pair-wise Spearman ρ between graders on a held-out "
            "validation set every 5 iters'\n"
            "  'We will monitor' or 'check regularly' without naming "
            "WHAT is monitored does NOT count as (c).\n"
            "\n"
            "Classify each risk:\n"
            "  * COMPLETE 2-TUPLE: both (a) and (b) present and specific\n"
            "  * COMPLETE 3-TUPLE (BONUS): (a), (b), AND (c) all present\n"
            "  * PARTIAL: only one of (a) or (b) present, or both "
            "present but generic\n"
            "  * GENERIC: neither specific (a trigger nor a fallback)\n"
            "\n"
            "Count: N_complete_2tuple (base) and N_complete_3tuple (bonus).\n"
            "\n"
            "Total risk credit = N_complete_2tuple + 0.5 × N_complete_3tuple.\n"
            "(A 3-tuple that is ALSO a 2-tuple gets 1 + 0.5 = 1.5 credit, "
            "matching the G12 bonus structure.)\n"
            "\n"
            "STEP 3 — Apply this TABLE:\n"
            "  total_credit < 1.0 → score 1\n"
            "  total_credit ∈ [1.0, 1.9] → score 2\n"
            "  total_credit ∈ [2.0, 2.9] → score 3\n"
            "  total_credit ∈ [3.0, 3.9] → score 4\n"
            "  total_credit ≥ 4.0 → score 5\n"
            "\n"
            "STRICT RULE (v10): listing many generic risks does NOT earn "
            "a higher score. Only the 2-tuple+ triples count. A single "
            "fully-specified 2-tuple risk beats ten generic disclaimers."
        ),
        scoring_rubric=(
            "v10: 2-tuple base (trigger + fallback), detection as 1.5× "
            "bonus. Relaxes v9's 3-tuple-required approach that left "
            "Opus unable to discriminate (var=0.21). Generic disclaimers, "
            "laundry lists, and partial tuples score as 1.\n"
            "1: No complete 2-tuples (only generic/partial)\n"
            "2: One complete 2-tuple (or ~1.5 credit with detection bonus)\n"
            "3: Two complete 2-tuples (or ~2.5 credit with bonuses)\n"
            "4: Three complete 2-tuples (or ~3-3.9 credit)\n"
            "5: Four+ complete 2-tuples, or fewer with most earning "
            "detection-method bonus\n"
        ),
        locus_directive=(
            "Quote each risk item that is PARTIAL or GENERIC — missing "
            "a specific trigger condition or a specific fallback action. "
            "If the proposal has zero complete 2-tuples, quote the "
            "longest generic disclaimer sentence."
        ),
    ),
]


# =============================================================================
# Weights (v10): unchanged from v8. Sums to 1.0.
# =============================================================================
# One-variable-per-diff rationale: v10 changes prompts only, not weights.
# This isolates the prompt-design effect from weight redistribution effects.
# Future v11+ may rebalance weights if dead/saturated channels re-emerge.
# =============================================================================
SIGNAL_WEIGHTS: dict[str, float] = {
    "G1_problem_specificity":  0.04,
    "G2_specific_aims":        0.04,
    "G4_focus":                0.16,
    "G6_reasoning_depth":      0.16,
    "G8_deliverable_clarity":  0.08,
    "G9_scope_feasibility":    0.14,
    "G10_approach_coverage":   0.06,
    "G11_evidence_rigor":      0.14,
    "G12_formalism":           0.04,
    "G13_risk_awareness":      0.14,
}

assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 1e-6, \
    f"Weights must sum to 1.0, got {sum(SIGNAL_WEIGHTS.values())}"

SCORE_MAX: dict[str, int] = {s.id: s.score_max for s in SIGNALS}
