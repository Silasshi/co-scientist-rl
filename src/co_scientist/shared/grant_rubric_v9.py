"""Grant Rubric v9: v8 + prompt rewrite on G4/G10 (dead channels) and G11/G12/G13 (length-hacked).

Designed 2026-04-22 after v8 runs were killed at iter 13-14 with documented
failure modes (see projects/grant_proposal_v2/DECISIONS.md 2026-04-22 entries).

Key changes from v8:
- G4_focus: added COMPOSITION LOGIC requirement. v8 clustered most plans at score 3
  (variance < 0.25 across 3 v8 runs). v9 splits middle into "isolated techniques" (score 3)
  vs "composed pipeline" (score 4-5), restoring gradient.
- G10_approach_coverage: tightened SPECIFIC to require (procedure + dataset/data source +
  measurable output) TRIPLE. v8 saturated at 5 (any named method = specific). v9 breaks ceiling.
- G11_evidence_rigor: each fair baseline must be (paper citation with year) +
  (quoted methodology snippet) + (scale-match rationale). v8 R²(log_wc → G11) = 0.46;
  v9 target < 0.25.
- G12_formalism: each non-trivial equation must be (all vars defined) + (referenced by
  label in downstream methodology step). v8 R² = 0.59; v9 target < 0.25.
- G13_risk_awareness: each risk must be complete 3-tuple (trigger condition + detection
  method + fallback action). v8 R² = 0.53; v9 target < 0.25.

Weights unchanged from v8. SCORE_MAX unchanged (5 per signal). G3/G5 still removed.

Reference: projects/grant_proposal_v2/analysis/v7_vs_v8_comparison.md
           projects/grant_proposal_v2/analysis/odin_length_residual.json
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
    # Research design (G4, G6) — G4 REWRITTEN in v9; G6 UNCHANGED
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
            "property. Examples of justification:\n"
            "  - GOOD: 'We use SDEs because the problem's heavy-tailed "
            "gradients violate the smoothness assumption required by ODE "
            "methods.'\n"
            "  - BAD: 'We use SDEs because they are a powerful modeling "
            "framework.'\n"
            "\n"
            "STEP 3 — Count (N_total, N_unjustified).\n"
            "\n"
            "STEP 4 — NEW IN v9 — check COMPOSITION LOGIC:\n"
            "  * COMPOSITION STATED: proposal describes, in ≤3 sentences, "
            "how at least TWO of the listed techniques feed into each "
            "other (which produces input for which) AND why the ordering "
            "matters for the stated problem. Generic statements "
            "('methods complement each other', 'used together') are NOT "
            "composition logic — must identify specific input/output flow.\n"
            "  * COMPOSITION ABSENT: plan lists techniques without stating "
            "flow or interdependency.\n"
            "\n"
            "STRICT RULE: If only 1 core technique, composition is "
            "vacuously satisfied — mark COMPOSITION STATED.\n"
            "\n"
            "STEP 5 — Apply this TABLE (v9: splits middle tier via "
            "composition gate to break the score-3 clump):\n"
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
            "v9: adds composition-logic gate. Middle tier (score 3) now "
            "requires either unjustified techniques OR absence of composition. "
            "Top scores (4-5) require composition logic stated.\n"
            "1: ≥7 techniques with many unjustified\n"
            "2: Moderate stacking or many unjustified, no composition\n"
            "3: Borderline — either some unjustified or composition absent\n"
            "4: Focused + composition stated, or justified stacking\n"
            "5: Tight focus (≤3 justified techniques) + composition stated\n"
        ),
        locus_directive=(
            "Quote each technique that is UNJUSTIFIED — named without a "
            "mechanistic sentence explaining why it is needed for THIS "
            "proposal's specific problem. If composition is absent, also "
            "quote the sentence introducing the technique list (since no "
            "composition statement accompanies it)."
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
            "with FULLY SPECIFIED methods? In v9, a method is FULLY SPECIFIED "
            "only if the proposal names ALL THREE of: (a) a concrete "
            "procedure or algorithm, (b) a concrete input data source or "
            "dataset, and (c) a concrete measurable output or metric. "
            "Missing any of the three → UNCOVERED."
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
            "STEP 3 — Mark each aim:\n"
            "  * COVERED: all three (a), (b), (c) present for this aim\n"
            "  * UNCOVERED: any of (a)/(b)/(c) missing, or aim lacks "
            "corresponding method section\n"
            "\n"
            "STRICT RULE (v9): VAGUE methods OR missing any of the three "
            "components → UNCOVERED. Partial specification does not "
            "count.\n"
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
            "v9: 'covered' requires FULL 3-tuple (procedure + data source + "
            "measurable output). v8 required only 'specific methodology' "
            "(any one of the three). v9 breaks v8 ceiling by demanding all "
            "three.\n"
            "1: Less than 25% of aims have full 3-tuple methodology\n"
            "2: 25-49% coverage\n"
            "3: 50-74% coverage\n"
            "4: 75-99% coverage\n"
            "5: 100% — every aim has all three (procedure + dataset + metric)\n"
        ),
        locus_directive=(
            "Quote aim sentences for which methodology is missing any of: "
            "procedure name, dataset/data-source name, or measurable "
            "output/metric. Also quote method sentences that use generic "
            "category labels ('statistical analysis', 'ML methods') "
            "instead of a named procedure."
        ),
    ),

    # =====================================================================
    # Rigor signals (G11, G12, G13) — ALL REWRITTEN in v9 to resist length hacking
    # =====================================================================
    SignalSpec(
        id="G11_evidence_rigor",
        name="Evidence Rigor",
        question=(
            "Can the proposed evaluation support the claims? For EMPIRICAL "
            "work, v9 requires each FAIR baseline to be a COMPLETE 3-TUPLE: "
            "(paper citation with year) + (quoted plan sentence showing "
            "where the baseline will be used in methodology) + (scale-match "
            "rationale tying baseline scale to this proposal's scale). "
            "Listed-but-unused baselines do not count. For THEORETICAL "
            "work: assumptions stated + proof sketch/strategy given + "
            "bounds or key lemma discussed."
        ),
        cot_scaffolding=(
            "STEP 1 — Identify if the proposed evaluation is EMPIRICAL, "
            "THEORETICAL, or both.\n"
            "\n"
            "STEP 2 (EMPIRICAL) — For each candidate baseline named in "
            "the proposal, check the 3-TUPLE:\n"
            "  (a) CITATION: named paper with year (e.g., 'NSGA-II (Deb "
            "et al. 2002)', 'BERT fine-tuning (Devlin 2018)'). Generic "
            "acronyms without author/year do not satisfy (a).\n"
            "  (b) USAGE: a quoted sentence elsewhere in the plan "
            "(usually in the methodology or experiments section) that "
            "describes HOW this baseline will be used. Listed in a "
            "'baselines' bullet list alone does NOT count as usage — "
            "must reference the baseline in an active methodology "
            "sentence.\n"
            "  (c) SCALE MATCH: one sentence stating why the baseline is "
            "comparable in scale or regime (e.g., 'matched in parameter "
            "count', 'trained on similar data volume', 'evaluated on "
            "same benchmark suite'). If the baseline is drawn from a "
            "fundamentally different regime with no reconciliation "
            "sentence, do NOT count.\n"
            "\n"
            "Classify each baseline as:\n"
            "  * FAIR (all three components present)\n"
            "  * STRAWMAN (from a different regime, obviously weak, or "
            "outdated for a fast-moving field)\n"
            "  * LIST-ONLY (named but missing any of the three components)\n"
            "\n"
            "Count N_fair_3tuple (baselines with all three components AND "
            "not strawman), N_strawman, N_list_only.\n"
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
            "STEP 4 — Apply this TABLE (v9 uses 3-tuple-verified count):\n"
            "  EMPIRICAL:\n"
            "    N_strawman ≥ 2 → score 1\n"
            "    N_strawman ≥ 1 AND criterion VAGUE → score 1\n"
            "    N_fair_3tuple = 0 → score 2 (only list-only baselines present)\n"
            "    N_fair_3tuple = 1 AND criterion OPERATIONALIZED → score 3\n"
            "    N_fair_3tuple ≥ 2 AND criterion OPERATIONALIZED → score 4\n"
            "    N_fair_3tuple ≥ 3 AND criterion OPERATIONALIZED + ablations → score 5\n"
            "    Otherwise → score 3\n"
            "\n"
            "  THEORETICAL:\n"
            "    T ≤ 1 → score 1-2\n"
            "    T = 2 → score 3\n"
            "    T = 3 + operationalized criterion → score 4-5\n"
            "\n"
            "STRICT RULE (v9): A baseline listed in bullet form without a "
            "usage sentence counts as LIST-ONLY, not FAIR. A baseline "
            "without a scale-match rationale counts as LIST-ONLY, not "
            "FAIR. Length of the baselines section alone does not elevate "
            "score."
        ),
        scoring_rubric=(
            "v9 requires 3-tuple verification per fair baseline (citation + "
            "usage quote + scale match). Padding a list of baselines does not "
            "elevate score.\n"
            "1: Strawman baselines or vague criterion with no verified-fair comparisons\n"
            "2: Only list-only baselines, no verified-fair 3-tuple\n"
            "3: One verified-fair 3-tuple with operationalized criterion\n"
            "4: ≥2 verified-fair 3-tuples with operationalized criterion\n"
            "5: ≥3 verified-fair 3-tuples + operationalized criterion + ablations\n"
        ),
        locus_directive=(
            "Quote each candidate baseline that is LIST-ONLY — missing "
            "citation, missing usage sentence elsewhere in the plan, or "
            "missing scale-match rationale. Quote each STRAWMAN baseline "
            "(obviously weak, outdated, or from a different regime). Quote "
            "each VAGUE success-criterion sentence ('expect', 'should', "
            "'aim to' without a measurable threshold)."
        ),
    ),

    SignalSpec(
        id="G12_formalism",
        name="Mathematical Formalism",
        question=(
            "Does the proposal contain USED non-trivial mathematical content? "
            "In v9, each non-trivial equation must be (a) have all variables "
            "defined either inline or in a notation section, AND (b) be "
            "referenced by its label (e.g., 'Eq. 3', 'see (2)') OR by its "
            "exact symbolic form in a downstream methodology, analysis, or "
            "proof step. Decorative equations (stated but never referenced "
            "again) do NOT count."
        ),
        cot_scaffolding=(
            "STEP 1 — Scan the ENTIRE proposal for EQUATIONS written in "
            "symbolic form. An EQUATION contains:\n"
            "  - mathematical symbols (=, +, -, *, /, ^, log, exp, sum, "
            "integral, E[], nabla, O())\n"
            "  - AND at least one NAMED variable or function beyond "
            "simple constants\n"
            "\n"
            "List each equation found, together with the shortest label "
            "or unique form you can use to reference it later.\n"
            "\n"
            "STEP 2 — For EACH equation, check the v9 2-REQUIREMENT GATE:\n"
            "  (a) VARIABLES DEFINED: every non-standard symbol in the "
            "equation is either defined in the sentence introducing the "
            "equation, in a prior notation section, or is a standard "
            "symbol in the obvious convention (e.g., ∇L for gradient of "
            "loss). If any symbol is unexplained, fail (a).\n"
            "  (b) USED DOWNSTREAM: after the equation is introduced, it "
            "is referenced again — either by its label ('by (3)', "
            "'using Eq. 2'), or by restating its symbolic form in a "
            "later sentence that DERIVES, ANALYZES, or APPLIES it. An "
            "equation stated once and never revisited fails (b).\n"
            "\n"
            "STEP 3 — Independently classify each equation for trivial / "
            "non-trivial:\n"
            "  * TAUTOLOGICAL: L = -R, y = f(x) without defining f, or any "
            "formula that restates the problem without analytical content.\n"
            "  * HYPERPARAMETER-ONLY: lr=3e-5, batch=32, etc.\n"
            "  * NON-TRIVIAL: a formula that adds analytical content.\n"
            "\n"
            "STEP 4 — An equation counts as N_used_equation only if it is "
            "NON-TRIVIAL AND satisfies (a) AND satisfies (b).\n"
            "\n"
            "STEP 5 — Apply this TABLE:\n"
            "  N_used_equation = 0 → score 1\n"
            "  N_used_equation = 1 → score 2\n"
            "  N_used_equation = 2 → score 3\n"
            "  N_used_equation = 3 → score 4\n"
            "  N_used_equation ≥ 4 → score 5\n"
            "\n"
            "STRICT RULE (v9): Hyperparameter values are NEVER equations. "
            "Section headers with math words are NOT formulas. Decorative "
            "equations (symbolic restatement of a prose claim, never used "
            "again) do NOT count toward the score even if non-trivial."
        ),
        scoring_rubric=(
            "v9 requires each equation be USED (referenced downstream) AND "
            "variables defined. Stacking decorative formulas does not elevate "
            "score.\n"
            "1: Zero equations that are both non-trivial and used\n"
            "2: One used non-trivial equation\n"
            "3: Two used non-trivial equations\n"
            "4: Three used non-trivial equations\n"
            "5: Four or more used non-trivial equations\n"
        ),
        locus_directive=(
            "Quote each equation that is either (a) missing variable "
            "definitions, (b) never referenced again after being stated, "
            "or (c) tautological / hyperparameter-only."
        ),
    ),

    SignalSpec(
        id="G13_risk_awareness",
        name="Risk Awareness",
        question=(
            "Does the proposal demonstrate mature awareness of its own "
            "boundaries, risks, and failure modes? In v9, each identified "
            "risk must be a COMPLETE 3-TUPLE: (specific trigger condition) "
            "+ (specific detection method) + (specific fallback action). "
            "Generic disclaimers or laundry-list risks without triggers, "
            "detection, or fallbacks do NOT count."
        ),
        cot_scaffolding=(
            "STEP 1 — Find the risk/limitation/contingency discussion in "
            "the proposal. Quote each distinct risk item.\n"
            "\n"
            "STEP 2 — For EACH risk item, check the v9 3-TUPLE:\n"
            "  (a) TRIGGER CONDITION: a specific, observable condition "
            "under which this risk materializes. Examples of specific "
            "triggers:\n"
            "    - 'if the gradient-norm fails to decrease for 5 "
            "consecutive iterations'\n"
            "    - 'if fairness constraint causes feasible region volume "
            "to drop below 1% of the unconstrained region'\n"
            "    - 'if the 235B grader disagrees with the 120B grader by "
            "more than 1 point on ≥30% of plans'\n"
            "  Generic or non-measurable triggers ('if the method "
            "fails', 'if results are poor', 'if unexpected issues "
            "arise') do NOT count as (a).\n"
            "\n"
            "  (b) DETECTION METHOD: a specific procedure by which the "
            "trigger would be observed. Examples:\n"
            "    - 'monitor sliding-window gradient norm; alert if "
            "below threshold'\n"
            "    - 'pair-wise Spearman ρ between graders on a held-out "
            "validation set every 5 iters'\n"
            "    - 'held-out Opus eval at iter 5/10/25'\n"
            "  'We will monitor' or 'check regularly' without naming "
            "WHAT is monitored does NOT count as (b).\n"
            "\n"
            "  (c) FALLBACK ACTION: a specific alternative direction "
            "or mitigation that would be executed if the trigger fires. "
            "Examples:\n"
            "    - 'switch to count-based rubric (see G11/G12/G13 v9 "
            "redesign)'\n"
            "    - 'fall back to n-gram length cap of 1500 words'\n"
            "    - 'restart with higher kl_budget'\n"
            "  'Adjust approach' or 'revise the methodology' without "
            "naming WHAT would change does NOT count as (c).\n"
            "\n"
            "Classify each risk:\n"
            "  * COMPLETE: all three (a), (b), (c) present and specific.\n"
            "  * PARTIAL: one or two components present.\n"
            "  * GENERIC: none present (a 'risk' with no trigger, "
            "detection, or fallback — just a worry).\n"
            "\n"
            "Count N_complete_triple.\n"
            "\n"
            "STEP 3 — Apply this TABLE:\n"
            "  N_complete_triple = 0 → score 1\n"
            "  N_complete_triple = 1 → score 2\n"
            "  N_complete_triple = 2 → score 3\n"
            "  N_complete_triple = 3 → score 4\n"
            "  N_complete_triple ≥ 4 → score 5\n"
            "\n"
            "STRICT RULE (v9): Listing many generic risks does NOT earn a "
            "higher score. Only the 3-tuple triples count. A single "
            "fully-specified risk (score 2) beats ten generic disclaimers "
            "(score 1)."
        ),
        scoring_rubric=(
            "v9: count-based on complete 3-tuples (trigger + detection + "
            "fallback). Generic disclaimers, laundry lists, and partial "
            "triples score as 1 (no triples) or 2 (one triple) — length "
            "of the risk section alone does not elevate score.\n"
            "1: No complete 3-tuples (only generic disclaimers or partial)\n"
            "2: One complete 3-tuple\n"
            "3: Two complete 3-tuples\n"
            "4: Three complete 3-tuples\n"
            "5: Four or more complete 3-tuples\n"
        ),
        locus_directive=(
            "Quote each risk item that is PARTIAL or GENERIC — missing a "
            "specific trigger condition, a specific detection method, or a "
            "specific fallback action. If the proposal has zero complete "
            "3-tuples, quote the longest generic disclaimer sentence."
        ),
    ),
]


# =============================================================================
# Weights (v9): unchanged from v8. Sums to 1.0.
# =============================================================================
# Decision rationale: v9 isolates the prompt rewrite as a single variable.
# If G4/G10 come alive under v9 prompts and G11/G12/G13 lose length correlation,
# weights can be re-balanced in a later revision (v10). Until then, use v8
# weights for direct before/after comparison.
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
