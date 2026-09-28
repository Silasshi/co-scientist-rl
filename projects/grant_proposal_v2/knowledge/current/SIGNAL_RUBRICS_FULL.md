# D4-v7 Signal Rubrics — Complete Text

*Extracted from `src/co_scientist/shared/grant_signal_reward.py` on 2026-04-22.*
*These are the exact prompts fed to Qwen3-30B grader for each signal.*

---

## G1_problem_specificity — Problem Specificity

**Weight**: 0.04 | **Scale**: 1-5

**Question**:
> Does the outline name a SPECIFIC technical problem, limitation, or knowledge gap? A specific problem names what is broken, missing, or unknown. Generic framing ('X is important', 'challenges remain') is NOT a specific problem.

**Required reasoning before scoring (CoT scaffolding)**:

    STEP 1 — Find the problem statement in the outline. Quote it.
    
    STEP 2 — Classify:
      * SPECIFIC: names a concrete technical limitation, knowledge gap, or failure mode. Examples:
        - 'neural network weights do not converge to stationary points'
        - 'current AI lacks uncertainty quantification'
        - 'no framework exists for X in domain Y'
        - 'an image of a chair can be misclassified as a toaster'
      * GENERIC: vague importance or motivation. Examples:
        - 'AI is revolutionising society'
        - 'this is a growing area of research'
        - 'challenges remain in this field'
        - 'more work is needed'
    
    STEP 3 — If SPECIFIC, does the problem have MULTIPLE facets (sub-problems) named? Count N_facets.
    
    STEP 4 — Apply this TABLE:
      GENERIC, no specific problem → score 1
      SPECIFIC but only 1 facet → score 3
      SPECIFIC with 2 facets → score 4
      SPECIFIC with ≥3 distinct facets → score 5
      Mixed (starts generic but becomes specific) → score 2

**Scoring rubric**:

    1: Only generic framing, no specific problem named
    2: Starts generic, eventually names something specific
    3: One clear specific problem
    4: Two specific problem facets
    5: Three or more specific problem facets

**Locus directive**: Quote sentences with GENERIC problem framing ('X is important', 'challenges remain', 'growing interest in Y').

---

## G2_specific_aims — Specific Aims Clarity

**Weight**: 0.04 | **Scale**: 1-5

**Question**:
> Does the proposal state MEASURABLE specific aims with expected outcomes? A measurable aim states what will be PRODUCED or DETERMINED (e.g., 'determine criteria for X', 'develop a framework that Y', 'identify factors associated with Z'). Vague aspirations ('explore', 'investigate', 'contribute to understanding') are NOT measurable aims.

**Required reasoning before scoring (CoT scaffolding)**:

    STEP 1 — List every specific aim or objective stated in the proposal:
      * MEASURABLE: states what will be produced, determined, or delivered. Examples:
        - 'determine what criteria should govern return of results'
        - 'develop and evaluate a new assay for X'
        - 'identify biomarkers that predict response to Y'
        - 'create a database of Z for public use'
      * VAGUE: 'explore the relationship', 'investigate various aspects', 'contribute to understanding', 'advance the field'
    
    STEP 2 — For each MEASURABLE aim, check: does it specify an EXPECTED OUTCOME or deliverable? (paper, dataset, framework, guideline, tool, recommendation)
      Count N_with_outcome.
    
    STEP 3 — Count N_measurable_aims total.
    
    STEP 4 — Apply this TABLE:
      N_measurable = 0 → score 1
      N_measurable = 1, no outcomes → score 2
      N_measurable = 1, with outcome → score 3
      N_measurable = 2, at least 1 with outcome → score 4
      N_measurable ≥ 3, or 2 both with outcomes → score 5

**Scoring rubric**:

    1: No measurable aims (only vague aspirations)
    2: One measurable aim but no expected outcome specified
    3: One measurable aim with expected outcome
    4: Two measurable aims, at least one with outcome
    5: Three+ measurable aims, or two aims both with outcomes

**Locus directive**: Quote sentences containing VAGUE aims ('explore', 'investigate', 'contribute to understanding') used in place of measurable aims with expected outcomes.

---

## G3_technical_evidence — Technical Evidence

**Weight**: 0.10 | **Scale**: 1-5

**Question**:
> Does the outline cite SPECIFIC technical evidence to support its claims? Evidence includes: named papers or citations, quantitative findings, concrete failure cases, specific illustrative scenarios with technical detail, or named real-world systems/organizations/initiatives. Generic claims ('research has shown') without naming the research are NOT evidence.

**Required reasoning before scoring (CoT scaffolding)**:

    STEP 1 — List every piece of SPECIFIC technical evidence:
      * Named papers: 'Zhang et al. (ICML 2022)', 'Belkin et al. (PNAS 2020)'
      * Quantitative findings: 'error rates exceed 30%', '10x slower than real-time'
      * Concrete failure cases: 'an image of a chair misclassified as a toaster'
      * Specific illustrative scenarios: 'self-driving cars simultaneously choosing the same route could overload roads', 'spurious correlations from historic data amplifying health inequalities'
      * Named real-world systems/organizations: 'Met Office weather forecasts', 'Edinburgh's Bayes Centre', 'Urban Observatory test-bed', 'National AI Strategy'
    
    DO NOT count: 'research has shown', 'studies indicate', 'it is well known', 'recent advances', 'current AI cannot'
    
    STEP 2 — Count N_evidence.
    
    STEP 3 — Apply this TABLE:
      N_evidence = 0 → score 1
      N_evidence = 1 → score 3
      N_evidence = 2 → score 4
      N_evidence ≥ 3 → score 5

**Scoring rubric**:

    1: No specific evidence cited
    3: One piece of evidence (notable for a short outline)
    4: Two pieces of evidence
    5: Three or more pieces of evidence

**Locus directive**: Quote sentences making unsupported claims ('research has shown', 'studies indicate', 'it is well known') without naming specific sources or data.

---

## G4_focus — Research Focus

**Weight**: 0.12 | **Scale**: 1-5

**Question**:
> Does the proposal STAY focused on a small number of core techniques, or does it STACK unrelated methods without justifying each? A focused proposal has 1-3 core techniques that clearly compose. A stacked proposal lists many techniques without explaining why each is necessary for THIS specific problem.

**Required reasoning before scoring (CoT scaffolding)**:

    STEP 1 — List the distinct METHODOLOGICAL techniques named in the proposal as part of the central approach. A 'technique' is a named algorithmic, analytical, or methodological component FUNDAMENTAL to the proposed work.
    
    DO NOT count as techniques:
      - Standard infrastructure (computing, data storage)
      - Evaluation metrics or benchmarks
      - Dissemination activities (papers, workshops)
      - Components of a single pipeline where stages are obviously interdependent
    
    Cap the list at 8 items maximum.
    
    STEP 2 — For EACH technique, classify as:
      * JUSTIFIED: proposal has a sentence explaining WHY this technique is needed for the stated problem (a causal or mechanistic reason).
      * UNJUSTIFIED: technique is named but no explanation of why it is needed for THIS proposal's problem.
    
    STEP 3 — Count (N_total, N_unjustified).
    
    STEP 4 — Apply this TABLE:
      N_total ≤ 3 AND N_unjustified = 0 → score 5
      N_total ≤ 3 AND N_unjustified ≥ 1 → score 4
      N_total ∈ [4, 6] AND N_unjustified ≤ 1 → score 4
      N_total ∈ [4, 6] AND N_unjustified ∈ [2, 3] → score 3
      N_total ∈ [4, 6] AND N_unjustified ≥ 4 → score 2
      N_total ≥ 7 AND N_unjustified ≤ 2 → score 3
      N_total ≥ 7 AND N_unjustified ≥ 3 → score 2
      N_total ≥ 7 AND N_unjustified ≥ 5 → score 1

**Scoring rubric**:

    Focus is a function of BOTH technique count and justification ratio. Few techniques, justified = 5. Many techniques, unjustified = 1.
    1: ≥7 techniques with ≥5 unjustified (severe stacking)
    2: Moderate stacking or many unjustified
    3: Borderline (2-3 unjustified of moderate total)
    4: Focused or mostly justified
    5: Tight focus (≤3 techniques) with every one justified

**Locus directive**: Quote each technique that is UNJUSTIFIED — named without a mechanistic sentence explaining why it is needed for this proposal's specific problem.

---

## G5_gap_identification — Gap Identification

**Weight**: 0.04 | **Scale**: 1-5

**Question**:
> Does the outline explain WHY current approaches are INSUFFICIENT for the stated problem? Gap identification means stating a specific limitation of existing work — not just listing what exists, but arguing why it falls short.

**Required reasoning before scoring (CoT scaffolding)**:

    STEP 1 — List every statement about existing work limitations:
      * INSUFFICIENCY: states what existing methods CANNOT do or where they FAIL. Examples:
        - 'current methods take shortcuts in modelling and inference'
        - 'optimisation state-of-the-art is not fit for purpose'
        - 'advancements have been piecemeal and ad hoc'
      * NEUTRAL: mentions existing work without critique. Examples:
        - 'prior work has studied X'
        - 'several methods have been proposed'
    
    STEP 2 — Count N_insufficiency (distinct limitation statements).
    
    STEP 3 — Apply this TABLE:
      N_insufficiency = 0 → score 1
      N_insufficiency = 1 → score 3
      N_insufficiency = 2 → score 4
      N_insufficiency ≥ 3 → score 5

**Scoring rubric**:

    1: No gap identified — existing work not critiqued
    3: One specific insufficiency stated
    4: Two insufficiencies stated
    5: Three or more insufficiencies forming a coherent argument

**Locus directive**: Quote sentences that mention existing work WITHOUT stating a limitation ('prior work has studied X', 'methods exist for Y').

---

## G6_reasoning_depth — Reasoning Depth

**Weight**: 0.12 | **Scale**: 1-5

**Question**:
> For each major design choice in the proposal, does the author explain WHY it was made? Depth means explicit justification: problem → insight → each choice justified by reference to that insight. Shallow proposals list techniques with citations but don't explain the logical connection to the stated problem.

**Required reasoning before scoring (CoT scaffolding)**:

    STEP 1 — State the core problem (from the proposal's problem statement) and the KEY INSIGHT or approach in one sentence each.
    
    STEP 2 — Identify the TOP 3-5 LOAD-BEARING DESIGN CHOICES — the decisions that DISTINGUISH this proposal from a standard approach. Examples:
      - a named algorithmic innovation
      - a specific methodological adaptation
      - a novel data collection or analysis strategy
      - a specific theoretical framework choice
    
    DO NOT count: standard infrastructure, named datasets unless novel, standard hyperparameters, evaluation benchmarks.
    
    STEP 3 — For EACH choice, classify as:
      * JUSTIFIED: proposal contains a sentence stating WHY this choice addresses the problem (causal or mechanistic).
      * ASSERTED: proposal names the choice but gives no 'why'.
    
    STEP 4 — Count N_asserted.
    
    STEP 5 — Apply this TABLE:
      N_asserted = 0 → score 5
      N_asserted = 1 → score 4
      N_asserted = 2 → score 3
      N_asserted = 3 → score 2
      N_asserted ≥ 4 → score 1
    
    STRICT RULE: 'inspired by X' or 'following Y' is NOT justification. But a citation PLUS a mechanistic sentence ('we use X because it provides property Y that addresses Z') IS justified.

**Scoring rubric**:

    1: ≥4 design choices asserted (list of techniques, no argument)
    2: 3 asserted
    3: 2 asserted
    4: 1 asserted (mostly justified)
    5: 0 asserted — every design choice has an explicit why-sentence

**Locus directive**: Quote each design choice that is ASSERTED — named without a mechanistic 'why' sentence explaining how it addresses the proposal's stated problem.

---

## G8_deliverable_clarity — Deliverable Clarity

**Weight**: 0.08 | **Scale**: 1-5

**Question**:
> Does the proposal name SPECIFIC deliverables or products that will result from the work? Deliverables are concrete outputs: published papers, datasets, guidelines, software tools, white papers, policy recommendations, databases. Generic outcome claims ('will advance the field', 'will contribute to knowledge') are NOT deliverables.

**Required reasoning before scoring (CoT scaffolding)**:

    STEP 1 — List every SPECIFIC deliverable named in the proposal:
      * SPECIFIC: 'publish 3 papers in area X', 'release a public database', 'produce a white paper with recommendations', 'develop open-source software for Y', 'create guidelines for Z', 'submit policy brief to agency W'
      * GENERIC (do NOT count): 'advance the field', 'contribute to knowledge', 'have significant impact', 'produce findings'
    
    STEP 2 — Count N_deliverables.
    
    STEP 3 — Apply this TABLE:
      N_deliverables = 0 → score 1
      N_deliverables = 1 → score 2
      N_deliverables = 2 → score 3
      N_deliverables = 3 → score 4
      N_deliverables ≥ 4 → score 5

**Scoring rubric**:

    1: No specific deliverables named
    2: One deliverable
    3: Two deliverables
    4: Three deliverables
    5: Four or more deliverables

**Locus directive**: Quote sentences with GENERIC outcome claims ('will advance the field', 'will contribute to knowledge') without naming specific deliverables or products.

---

## G9_scope_feasibility — Scope Feasibility

**Weight**: 0.08 | **Scale**: 1-5

**Question**:
> Is the proposed scope of work feasible given typical grant constraints? Scope red flags include: claiming to analyze an unrealistically large dataset comprehensively, covering all countries/all cases, proposing too many aims for the format, or promising comprehensive coverage of a vast domain without acknowledging limitations.

**Required reasoning before scoring (CoT scaffolding)**:

    STEP 1 — List all scope commitments in the proposal (number of countries, databases, analyses, aims, sub-studies, etc.).
    
    STEP 2 — Identify scope RED FLAGS:
      * OVERLY BROAD claims: 'comprehensive analysis of 4,230 laws', 'all 193 UN member states', 'every possible scenario'
      * MISMATCH between scope and resources: e.g., 2-year project claiming to comprehensively analyze thousands of items
      * UNACKNOWLEDGED limitations: broad scope claimed without explicitly stating what is excluded or why the scope is feasible
      * TOO MANY sub-studies for the format (e.g., 5+ distinct analyses in a 2-year exploratory grant)
    
    STEP 3 — Count N_red_flags.
    
    STEP 4 — Apply this TABLE:
      N_red_flags = 0 → score 5
      N_red_flags = 1 → score 4
      N_red_flags = 2 → score 3
      N_red_flags ≥ 3 → score 1

**Scoring rubric**:

    5: No scope red flags — all commitments appear feasible
    4: One minor scope concern
    3: Two scope red flags
    1: Three or more scope red flags — project appears infeasible

**Locus directive**: Quote sentences containing OVERLY BROAD scope claims ('comprehensive analysis of X thousand', 'all countries', 'every possible') or scope-resource mismatches.

---

## G10_approach_coverage — Approach Coverage

**Weight**: 0.04 | **Scale**: 1-5

**Question**:
> Does the proposed approach/methodology cover ALL stated aims? For each aim or objective, there should be a corresponding description of HOW it will be accomplished. An aim with no corresponding method is an UNCOVERED aim.

**Required reasoning before scoring (CoT scaffolding)**:

    STEP 1 — List all stated aims/objectives from the proposal.
      Count N_aims.
    
    STEP 2 — For each aim, check: is there a corresponding methodological description explaining HOW it will be accomplished? Mark COVERED or UNCOVERED.
      * COVERED: aim has a corresponding paragraph or section describing specific procedures, data sources, or analytical steps
      * UNCOVERED: aim is stated but no method is described, or method is only vaguely gestured at ('will use appropriate methods')
    
    STEP 3 — Calculate coverage = N_covered / N_aims.
    
    STEP 4 — Apply this TABLE:
      coverage = 100% → score 5
      coverage ≥ 75% → score 4
      coverage ≥ 50% → score 3
      coverage ≥ 25% → score 2
      coverage < 25% → score 1

**Scoring rubric**:

    1: Less than 25% of aims have methodology described
    2: 25-49% coverage
    3: 50-74% coverage
    4: 75-99% coverage
    5: 100% — every aim has a corresponding method

**Locus directive**: Quote sentences stating aims or objectives for which NO corresponding methodology is described in the Approach section.

---

## G11_evidence_rigor — Evidence Rigor

**Weight**: 0.12 | **Scale**: 1-5

**Question**:
> Can the proposed evaluation support the claims? For EMPIRICAL work: are baselines FAIR (appropriate, modern, matched in scale — not strawman) and is there an OPERATIONALIZED success criterion (specific metric with threshold)? For THEORETICAL work: are assumptions stated and is a proof sketch/strategy given?

**Required reasoning before scoring (CoT scaffolding)**:

    STEP 1 — Identify if the proposed evaluation is EMPIRICAL or THEORETICAL (or both).
    
    STEP 2 (EMPIRICAL) — List every baseline or comparison method. For each, classify as:
      * FAIR: a modern, appropriate-scale method from the same problem space.
      * STRAWMAN: obviously weak (random baseline, method 5+ years old in a fast-moving field, or from a different problem regime).
    Count N_fair and N_strawman.
    
    STEP 2 (THEORETICAL) — Mark each as present (1) or absent (0):
      * Explicit assumptions stated
      * Proof strategy / sketch given (not just 'we will prove')
      * Tight bounds discussed OR key lemma stated
    Count T = sum (0-3).
    
    STEP 3 — Check the SUCCESS CRITERION:
      * OPERATIONALIZED: named metric with benchmark or threshold.
      * VAGUE: 'we expect improvement', 'should outperform'.
    
    STEP 4 — Apply this TABLE:
      EMPIRICAL:
        N_strawman ≥ 2 → score 1
        N_strawman ≥ 1 AND criterion VAGUE → score 1
        N_fair ≥ 2 AND criterion VAGUE → score 2
        N_fair ≥ 2 AND criterion OPERATIONALIZED → score 4
        N_fair ≥ 3 AND criterion OPERATIONALIZED + ablations → score 5
        Otherwise → score 3
    
      THEORETICAL:
        T ≤ 1 → score 1-2
        T = 2 → score 3
        T = 3 + operationalized criterion → score 4-5

**Scoring rubric**:

    1: Strawman baselines or vague criterion with no fair comparisons
    2: Fair baselines but vague success criterion
    3: One fair baseline with operationalized criterion
    4: ≥2 fair baselines + operationalized criterion
    5: ≥3 fair baselines + operationalized criterion + ablations

**Locus directive**: Quote each STRAWMAN baseline (weak or obsolete comparison) OR each VAGUE success-criterion sentence ('expect', 'should', 'aim to' without a measurable threshold).

---

## G12_formalism — Mathematical Formalism

**Weight**: 0.12 | **Scale**: 1-5

**Question**:
> Does the proposal contain NON-TRIVIAL mathematical content? Non-trivial means: a named equation written in symbolic form that is NOT tautological (L = -R is tautological), NOT just hyperparameter values (lr=3e-5), and NOT standard definitions. Count DISTINCT equations/formulas that contribute to the proposal's core methodology.

**Required reasoning before scoring (CoT scaffolding)**:

    Do EXPLICIT COUNTING (no interpretation):
    
    STEP 1 — Scan the ENTIRE proposal for EQUATIONS written in symbolic form.
    An EQUATION contains:
      - mathematical symbols (=, +, -, *, /, ^, log, exp, sum, integral, E[], nabla, O())
      - AND at least one NAMED variable or function beyond simple constants
    
    List each equation found. Write down the actual formula.
    
    STEP 2 — For EACH equation, classify as:
      * TAUTOLOGICAL: L = -R, y = f(x) without defining f, or any formula that restates the problem without analytical content.
      * HYPERPARAMETER-ONLY: lr=3e-5, batch=32, rank=8, etc.
      * NON-TRIVIAL: A formula that adds analytical content. Examples:
        - A convergence rate: O(1/sqrt(T))
        - An objective: J_beta = E[log E[exp(beta * R)]]
        - A bound: Q(s) + c * P(s) * sqrt(1+T)/(1+n(s))
        - A regularizer: KL(pi || pi_ref) <= delta
        - A complexity bound: O(d * m) vs O(n^3)
    
    STEP 3 — Count N_nontrivial.
    
    STEP 4 — Apply this TABLE:
      N_nontrivial = 0 → score 1
      N_nontrivial = 1 AND standard textbook → score 2
      N_nontrivial = 1 AND adapted/novel → score 3
      N_nontrivial = 2 AND at least 1 non-textbook → score 4
      N_nontrivial ≥ 3 AND coherent derivation chain → score 5
    
    STRICT RULE: Hyperparameter values are NEVER equations. Section headers with math words are NOT formulas.

**Scoring rubric**:

    1: Zero non-trivial formulas
    2: One standard textbook formula
    3: One adapted/novel formula
    4: Two formulas with at least one non-textbook
    5: Three+ formulas forming a coherent derivation chain

**Locus directive**: Quote sentences where a mathematical claim is made in prose without an accompanying symbolic equation (e.g., 'we prove convergence' without writing the rate).

---

## G13_risk_awareness — Risk Awareness

**Weight**: 0.10 | **Scale**: 1-5

**Question**:
> Does the proposal demonstrate mature awareness of its own boundaries, risks, and failure modes? A good proposal names 2-3 SPECIFIC failure modes, identifies which assumptions are most fragile, states scope boundaries, and sketches fallback strategies. This is NOT about enumerating 10 generic risks — it's about showing the author understands the risk structure.

**Required reasoning before scoring (CoT scaffolding)**:

    Before scoring:
    1. Does the proposal identify 2-3 specific ways the research could fail? (specific = specific assumption breaking, specific method limitation, specific domain mismatch)
    2. Does the proposal state which assumptions are most fragile (the 'key bet')?
    3. Does the proposal acknowledge scope boundaries (what it does NOT claim to solve)?
    4. Does the proposal sketch fallback strategies or contingency plans for identified risks?
    5. Is there a coherent risk structure, or just a laundry list?
    6. Assign a score.

**Scoring rubric**:

    1: No risk awareness. Proposal assumes success with no discussion of failure modes or limitations.
    2: Generic disclaimers ('there might be limitations', 'further research is needed') without specificity.
    3: Partial. Names 1 failure mode OR mentions scope boundary, but risk structure is incomplete.
    4: Mature. Names 2-3 specific failure modes, identifies fragile assumptions, states scope boundary.
    5: Exceptional. Coherent risk model: specific failure modes, load-bearing assumptions identified, scope boundary explicit, AND fallback direction sketched.

**Locus directive**: Quote GENERIC risk disclaimers ('there might be limitations', 'further research is needed', 'various challenges exist'). If the proposal lacks specific failure modes, quote the last sentence of the proposal.

---
