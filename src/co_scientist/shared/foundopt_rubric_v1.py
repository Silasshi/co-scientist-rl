"""FoundOpt-specific rubric v1: 8 signals on 1-10 scale for the Better Reward experiment.

Fused from D3 v9 validated counting rubrics + one new signal (R1_theoretical_novelty).
Designed for Qwen3-235B grader. Reference proposal should score ~7-8/10 per signal.
"""

from __future__ import annotations

from co_scientist.shared.grant_signal_reward import SignalSpec

SIGNALS: list[SignalSpec] = [
    SignalSpec(
        id="R1_theoretical_novelty",
        name="Theoretical Novelty",
        question=(
            "Does the proposal introduce a genuinely NEW theoretical concept, "
            "framework, or proof strategy — or does it merely apply existing "
            "techniques to a new setting? Novelty markers include: a named new "
            "concept with a formal definition, a new proof strategy not seen in "
            "cited prior work, or a new problem formulation that reframes the "
            "question in a fundamentally different way."
        ),
        cot_scaffolding=(
            "Do EXPLICIT CLASSIFICATION:\n"
            "\n"
            "STEP 1 — List every CLAIMED CONTRIBUTION in the proposal "
            "(typically in Specific Aims or Core Idea). Cap at 5.\n"
            "\n"
            "STEP 2 — For EACH contribution, classify as:\n"
            "  * NEW CONCEPT: a named abstraction/definition not present in "
            "cited work (e.g., 'functional stationarity', 'bilevel fairness "
            "formulation'). Must have: (a) a name, (b) a formal or operational "
            "definition, (c) explicit statement of how it differs from prior.\n"
            "  * NEW PROOF STRATEGY: a proof approach stated to be different from "
            "how prior results were proved (e.g., 'shifting convergence target "
            "from parameter space to function space').\n"
            "  * NEW PROBLEM FORMULATION: the problem itself is formulated "
            "differently from prior work (not just applied to a new domain).\n"
            "  * APPLICATION: existing technique applied to new setting without "
            "the above novelty markers.\n"
            "  * INCREMENTAL: minor extension of existing method (e.g., adding "
            "a regularizer, changing a loss term, scaling up).\n"
            "\n"
            "STEP 3 — Count:\n"
            "  N_novel = (NEW CONCEPT + NEW PROOF STRATEGY + NEW PROBLEM FORMULATION)\n"
            "  N_application = APPLICATION count\n"
            "  N_incremental = INCREMENTAL count\n"
            "\n"
            "STEP 4 — For each NOVEL contribution, check DEVELOPMENT DEPTH:\n"
            "  * DEVELOPED: has formal definition + motivation + at least one "
            "consequence stated (e.g., 'this enables proving X')\n"
            "  * SKETCHED: named and motivated but no formal definition yet\n"
            "  * CLAIMED: just asserted as novel without development\n"
            "  Count N_developed, N_sketched, N_claimed.\n"
            "\n"
            "STEP 5 — Apply this TABLE:\n"
            "  N_novel = 0 → score 1-3 (pure application/incremental)\n"
            "    N_application ≥ 2 with new setting → 3\n"
            "    N_application = 1 → 2\n"
            "    All incremental → 1\n"
            "  N_novel = 1, CLAIMED → 4\n"
            "  N_novel = 1, SKETCHED → 5\n"
            "  N_novel = 1, DEVELOPED → 6\n"
            "  N_novel = 2, at least 1 SKETCHED → 7\n"
            "  N_novel = 2, at least 1 DEVELOPED → 8\n"
            "  N_novel ≥ 3, at least 2 DEVELOPED → 9\n"
            "  N_novel ≥ 3, all DEVELOPED + implications stated → 10\n"
        ),
        scoring_rubric=(
            "1: All contributions are incremental extensions\n"
            "2: One application of existing technique to new setting\n"
            "3: Multiple applications to new settings, reasonable novelty\n"
            "4: One novel concept/strategy but only claimed, not developed\n"
            "5: One novel concept, sketched with motivation\n"
            "6: One novel concept, formally developed with consequences\n"
            "7: Two novel contributions, at least one sketched\n"
            "8: Two novel contributions, at least one fully developed\n"
            "9: Three+ novel contributions, most developed\n"
            "10: Three+ fully developed novel contributions with stated implications"
        ),
        score_max=10,
    ),

    SignalSpec(
        id="R2_formalism",
        name="Mathematical Formalism",
        question=(
            "Does the proposal contain FORMAL MATHEMATICAL CONTENT that goes "
            "beyond prose description? This includes: symbolic equations, "
            "complexity/convergence RATES (e.g., O(1/√T), O(n³)), formal "
            "problem specifications (named objective + constraints), and "
            "precise quantitative bounds. Count ALL formal mathematical "
            "expressions — both display equations AND inline formal "
            "expressions embedded in prose."
        ),
        cot_scaffolding=(
            "Do EXPLICIT COUNTING (no interpretation):\n"
            "\n"
            "STEP 1 — Scan the ENTIRE proposal for FORMAL MATHEMATICAL "
            "EXPRESSIONS. These include:\n"
            "  (a) Display equations (standalone formulas)\n"
            "  (b) Inline complexity/rate expressions: O(1/√T), O(n³), "
            "O(d·m), Ω(n²), etc.\n"
            "  (c) Formal problem specifications: 'minimize f(θ) subject to "
            "g(θ) ≤ 0' or equivalent in prose with named objective/constraint\n"
            "  (d) Quantitative bounds with mathematical structure: "
            "'convergence rate ≤ O(1/√T) for width ≥ poly(n)'\n"
            "  (e) Named mathematical relationships: 'KL(π || π_ref) ≤ δ', "
            "'coverage = N_covered / N_aims'\n"
            "\n"
            "DO NOT count:\n"
            "  - Pure numbers without mathematical structure (e.g., '3 papers', "
            "'2-year project')\n"
            "  - Hyperparameter values alone (lr=3e-5)\n"
            "  - Percentage thresholds without formal structure ('≥2% accuracy')\n"
            "\n"
            "List each expression found.\n"
            "\n"
            "STEP 2 — For EACH expression, classify as:\n"
            "  * NOVEL: new to this proposal (new objective, new rate, new bound)\n"
            "  * ADAPTED: known formula modified for this setting\n"
            "  * STANDARD: well-known formula cited or reproduced\n"
            "  * RATE/BOUND: complexity or convergence rate expression\n"
            "\n"
            "STEP 3 — Count:\n"
            "  N_total = all formal expressions\n"
            "  N_novel_or_adapted = NOVEL + ADAPTED\n"
            "\n"
            "STEP 4 — Check LOGICAL CHAINS: are expressions connected "
            "(one informs/derives the next)? Count N_chains.\n"
            "\n"
            "STEP 5 — Apply this TABLE:\n"
            "  N_total = 0 → score 1\n"
            "  N_total = 1 → score 2\n"
            "  N_total = 2, all standard/rate → score 3\n"
            "  N_total = 2, at least 1 novel/adapted → score 4\n"
            "  N_total ∈ [3, 4], at least 1 novel/adapted → score 5\n"
            "  N_total ∈ [3, 4], at least 2 novel/adapted → score 6\n"
            "  N_total ≥ 5, at least 2 novel/adapted → score 7\n"
            "  N_total ≥ 5, at least 2 novel/adapted + chain → score 8\n"
            "  N_total ≥ 7, ≥3 novel/adapted + chain → score 9\n"
            "  N_total ≥ 9, ≥4 novel/adapted + multiple chains → score 10\n"
        ),
        scoring_rubric=(
            "1: Zero formal mathematical expressions\n"
            "2: One expression (any type)\n"
            "3: Two standard/rate expressions\n"
            "4: Two expressions with at least one novel/adapted\n"
            "5: Three-four expressions with at least one novel/adapted\n"
            "6: Three-four expressions with at least two novel/adapted\n"
            "7: Five+ expressions with at least two novel/adapted\n"
            "8: Five+ expressions with two novel/adapted + logical chain\n"
            "9: Seven+ expressions with three novel/adapted + chain\n"
            "10: Nine+ expressions with four novel/adapted + multiple chains"
        ),
        score_max=10,
    ),

    SignalSpec(
        id="R3_reasoning_depth",
        name="Reasoning Depth",
        question=(
            "How DEEP is the justification for each major design choice? "
            "Depth is NOT just having a 'because' sentence — it requires "
            "multi-layer reasoning: mechanism + limitation acknowledgment + "
            "compositional connection to other choices. A shallow proposal "
            "says 'we use X because Y'. A deep proposal says 'we use X "
            "because Y, but X has limitation Z, so we also need W to "
            "compensate, and together X+W enable property P'."
        ),
        cot_scaffolding=(
            "Do EXPLICIT DEPTH CLASSIFICATION:\n"
            "\n"
            "STEP 1 — Identify the TOP 4-6 LOAD-BEARING DESIGN CHOICES "
            "(algorithmic innovations, framework selections, methodological "
            "adaptations). Cap at 6.\n"
            "\n"
            "STEP 2 — For EACH choice, classify DEPTH LEVEL:\n"
            "\n"
            "  * DEEP (Level 3): ALL of the following:\n"
            "    (a) Mechanistic reason WHY this choice addresses the "
            "specific problem (not just 'because it works well')\n"
            "    (b) Limitation acknowledged (what this choice CANNOT do "
            "or where it might fail)\n"
            "    (c) Compositional connection: how this choice interacts "
            "with or enables another design choice in the proposal\n"
            "    Example: 'We use NTK because it provides tractable "
            "linearisation in function space [mechanism]. However, NTK "
            "only describes lazy training [limitation]. To address this, "
            "we also develop mean-field analysis [composition].'\n"
            "\n"
            "  * MODERATE (Level 2): Has (a) mechanistic reason PLUS one "
            "of (b) or (c), but not all three.\n"
            "    Example: 'We use bilevel optimization to separate fairness "
            "from generalisation [mechanism], which avoids the feasibility "
            "collapse of direct constrained formulation [limitation].'\n"
            "\n"
            "  * SHALLOW (Level 1): Has ONLY a reason sentence ('we use X "
            "because Y') without limitation or composition. The reason may "
            "be mechanistic but stands alone.\n"
            "    Example: 'We use random Fourier features for scalability.'\n"
            "\n"
            "  * ASSERTED (Level 0): Choice named with no reason at all.\n"
            "\n"
            "STRICT RULES:\n"
            "  - 'inspired by X' / 'following Y' / 'as in [cite]' without "
            "mechanism = ASSERTED, not SHALLOW\n"
            "  - A citation PLUS mechanism = at least SHALLOW\n"
            "  - Limitation must be SPECIFIC to this choice (not generic "
            "'there might be challenges')\n"
            "\n"
            "STEP 3 — Count (N_deep, N_moderate, N_shallow, N_asserted).\n"
            "\n"
            "STEP 4 — Compute depth_score = (3×N_deep + 2×N_moderate + "
            "1×N_shallow + 0×N_asserted) / (3 × N_total).\n"
            "  This gives a ratio in [0, 1] where 1.0 = all DEEP.\n"
            "\n"
            "STEP 5 — Apply this TABLE:\n"
            "  depth_score ≥ 0.90 → score 10\n"
            "  depth_score ≥ 0.80 → score 9\n"
            "  depth_score ≥ 0.70 → score 8\n"
            "  depth_score ≥ 0.60 → score 7\n"
            "  depth_score ≥ 0.50 → score 6\n"
            "  depth_score ≥ 0.40 → score 5\n"
            "  depth_score ≥ 0.33 → score 4 (all SHALLOW = 0.33)\n"
            "  depth_score ≥ 0.25 → score 3\n"
            "  depth_score ≥ 0.15 → score 2\n"
            "  depth_score < 0.15 → score 1\n"
        ),
        scoring_rubric=(
            "1: Nearly all asserted (depth_score < 0.15)\n"
            "2: Mostly asserted with occasional shallow reason\n"
            "3: Mix of shallow and asserted\n"
            "4: All shallow (has reasons but no depth) — typical LLM output\n"
            "5: Some moderate depth (1-2 choices with limitation or composition)\n"
            "6: Half of choices at moderate+ depth\n"
            "7: Most choices at moderate depth\n"
            "8: Most choices at moderate, some at deep\n"
            "9: Most choices at deep depth\n"
            "10: Nearly all deep (mechanism + limitation + composition for each)"
        ),
        score_max=10,
    ),

    SignalSpec(
        id="R4_evidence_rigor",
        name="Evidence Rigor",
        question=(
            "Can the proposed evaluation support the claims? For EMPIRICAL "
            "work: are baselines FAIR (appropriate, modern, matched in scale "
            "— not strawman) and is there an OPERATIONALIZED success criterion "
            "(specific metric with threshold)? For THEORETICAL work: are "
            "assumptions stated and is a proof sketch/strategy given?"
        ),
        cot_scaffolding=(
            "STEP 1 — Identify if the evaluation is EMPIRICAL, THEORETICAL, or BOTH.\n"
            "\n"
            "STEP 2 (EMPIRICAL) — List every baseline or comparison method:\n"
            "  * FAIR: modern, appropriate-scale method from same problem space\n"
            "  * STRAWMAN: obviously weak or outdated\n"
            "  Count N_fair, N_strawman.\n"
            "\n"
            "STEP 2 (THEORETICAL) — Check:\n"
            "  * Explicit assumptions stated (1/0)\n"
            "  * Proof strategy / sketch given, not just 'we will prove' (1/0)\n"
            "  * Tight bounds discussed OR key lemma stated (1/0)\n"
            "  * Comparison to prior bounds stated (1/0)\n"
            "  Count T = sum (0-4).\n"
            "\n"
            "STEP 3 — Check SUCCESS CRITERIA:\n"
            "  * OPERATIONALIZED: named metric + benchmark + threshold\n"
            "  * PARTIAL: named metric but no threshold\n"
            "  * VAGUE: 'we expect improvement'\n"
            "  Count N_operationalized, N_partial.\n"
            "\n"
            "STEP 4 — Check ABLATION DESIGN:\n"
            "  * Named ablations with rationale (count)\n"
            "  * Statistical significance plan (yes/no)\n"
            "\n"
            "STEP 5 — Apply this TABLE:\n"
            "  EMPIRICAL path:\n"
            "    N_strawman ≥ 2 → cap at 2\n"
            "    N_fair=0 → score 1-2\n"
            "    N_fair=1, criteria VAGUE → score 3\n"
            "    N_fair=1, criteria PARTIAL → score 4\n"
            "    N_fair≥2, criteria PARTIAL → score 5\n"
            "    N_fair≥2, criteria OPERATIONALIZED → score 6-7\n"
            "    N_fair≥3, criteria OPERATIONALIZED + ablations → score 8\n"
            "    N_fair≥3, OPERATIONALIZED + ablations + significance → score 9\n"
            "    All above + failure criteria + multiple benchmarks → score 10\n"
            "\n"
            "  THEORETICAL path:\n"
            "    T=0 → score 1\n"
            "    T=1 → score 3\n"
            "    T=2 → score 5\n"
            "    T=3 → score 7\n"
            "    T=4 + operationalized empirical validation → score 9-10\n"
            "\n"
            "  BOTH paths: take max of the two scores, +1 if cross-validated.\n"
        ),
        scoring_rubric=(
            "1: No evaluation plan or only strawman baselines\n"
            "2: Strawman baselines with vague criteria\n"
            "3: One fair baseline, vague criteria\n"
            "4: One fair baseline, partial criteria\n"
            "5: Two+ fair baselines, partial criteria\n"
            "6: Two+ fair baselines, operationalized criteria\n"
            "7: Three+ fair baselines, operationalized criteria\n"
            "8: All above + named ablations with rationale\n"
            "9: All above + statistical significance plan\n"
            "10: Comprehensive: multiple benchmarks, ablations, significance, failure criteria"
        ),
        score_max=10,
    ),

    SignalSpec(
        id="R5_risk_awareness",
        name="Risk Awareness",
        question=(
            "Does the proposal demonstrate mature awareness of its own "
            "boundaries, risks, and failure modes? A good proposal names "
            "SPECIFIC failure modes per work package, identifies which "
            "assumptions are most fragile, states scope boundaries, and "
            "sketches fallback strategies."
        ),
        cot_scaffolding=(
            "STEP 1 — List every SPECIFIC failure mode mentioned:\n"
            "  * SPECIFIC: names a concrete assumption that could break, "
            "a concrete resource that could run out, or a concrete domain "
            "mismatch (e.g., 'NTK analysis may not extend to deep narrow "
            "networks', 'bilevel optimization may be too expensive').\n"
            "  * GENERIC: 'there might be limitations', 'further research needed'\n"
            "  Count N_specific, N_generic.\n"
            "\n"
            "STEP 2 — Check for FRAGILE ASSUMPTION identification:\n"
            "  Does the proposal explicitly state which assumption is the 'key bet'?\n"
            "  (yes/no)\n"
            "\n"
            "STEP 3 — Check for SCOPE BOUNDARY:\n"
            "  Does the proposal state what it does NOT claim to solve?\n"
            "  (yes/no)\n"
            "\n"
            "STEP 4 — Check FALLBACK STRATEGIES:\n"
            "  For each specific risk, is there a mitigation or fallback?\n"
            "  Count N_mitigated (specific risks with a stated fallback).\n"
            "\n"
            "STEP 5 — Apply this TABLE:\n"
            "  N_specific=0, no scope boundary → score 1\n"
            "  Only generic disclaimers → score 2\n"
            "  N_specific=1, no mitigation → score 3\n"
            "  N_specific=1, with mitigation → score 4\n"
            "  N_specific=2, partial mitigation → score 5\n"
            "  N_specific=2, both mitigated → score 6\n"
            "  N_specific≥3, most mitigated, fragile assumption identified → score 7\n"
            "  N_specific≥3, all mitigated, scope boundary stated → score 8\n"
            "  All above + risk structure reveals deep understanding → score 9\n"
            "  Coherent risk model: all risks mitigated, fragile assumptions "
            "ranked, scope explicit, fallback could salvage partial results → score 10\n"
        ),
        scoring_rubric=(
            "1: No risk awareness\n"
            "2: Generic disclaimers only\n"
            "3: One specific failure mode, no mitigation\n"
            "4: One specific failure mode with mitigation\n"
            "5: Two specific risks, partial mitigation\n"
            "6: Two specific risks, both mitigated\n"
            "7: Three+ risks, most mitigated, fragile assumption identified\n"
            "8: Three+ risks, all mitigated, scope boundary stated\n"
            "9: Mature risk structure with ranked assumptions\n"
            "10: Exceptional: coherent risk model, all mitigated, partial-salvage fallbacks"
        ),
        score_max=10,
    ),

    SignalSpec(
        id="R6_research_focus",
        name="Research Focus",
        question=(
            "Within each work package or research strand, does the proposal "
            "stay focused on 1-3 core techniques, or does it STACK unrelated "
            "methods? A multi-WP proposal naturally uses different techniques "
            "across WPs — that is NOT stacking. Stacking is: within ONE "
            "strand, listing many techniques without explaining why each is "
            "needed for THAT strand's specific sub-problem."
        ),
        cot_scaffolding=(
            "Do EXPLICIT COUNTING PER WORK PACKAGE:\n"
            "\n"
            "STEP 1 — Identify the proposal's distinct WORK PACKAGES or "
            "research strands (WP1, WP2, ... or equivalent sections). "
            "Count N_wp.\n"
            "\n"
            "STEP 2 — For EACH work package, list the CORE TECHNIQUES "
            "used within it (algorithmic, analytical, methodological "
            "components). Cap at 5 per WP.\n"
            "\n"
            "DO NOT count as techniques:\n"
            "  - Standard infrastructure\n"
            "  - Evaluation benchmarks\n"
            "  - Techniques from OTHER work packages\n"
            "  - Components that are obviously sub-parts of one method\n"
            "\n"
            "STEP 3 — For EACH technique within its WP, classify:\n"
            "  * JUSTIFIED: has a sentence explaining why needed for THIS "
            "WP's sub-problem\n"
            "  * UNJUSTIFIED: named but necessity not explained\n"
            "\n"
            "STEP 4 — Per WP, count (N_tech, N_unjustified). Then compute "
            "the WORST-CASE WP: max_unjustified = max across WPs of "
            "N_unjustified. Also compute avg_tech = average N_tech per WP.\n"
            "\n"
            "STEP 5 — Apply this TABLE:\n"
            "  avg_tech ≤ 2 AND max_unjustified = 0 → score 10\n"
            "  avg_tech ≤ 2 AND max_unjustified = 1 → score 8\n"
            "  avg_tech ≤ 3 AND max_unjustified = 0 → score 9\n"
            "  avg_tech ≤ 3 AND max_unjustified = 1 → score 7\n"
            "  avg_tech ≤ 3 AND max_unjustified = 2 → score 5\n"
            "  avg_tech ∈ [3, 4] AND max_unjustified ≤ 1 → score 6\n"
            "  avg_tech ∈ [3, 4] AND max_unjustified = 2 → score 4\n"
            "  avg_tech ∈ [3, 4] AND max_unjustified ≥ 3 → score 3\n"
            "  avg_tech ≥ 5 AND max_unjustified ≤ 1 → score 4\n"
            "  avg_tech ≥ 5 AND max_unjustified ≥ 2 → score 2\n"
            "  avg_tech ≥ 5 AND max_unjustified ≥ 4 → score 1\n"
            "\n"
            "NOTE: A proposal with 3 WPs each having 2 justified techniques "
            "(6 total) is FOCUSED (score 9-10). A proposal with 1 WP having "
            "6 techniques is STACKING (score 4-6)."
        ),
        scoring_rubric=(
            "1: Severe stacking within WPs (5+ techniques/WP, 4+ unjustified)\n"
            "2: Heavy stacking (5+ techniques/WP, 2+ unjustified)\n"
            "3: Moderate stacking (3-4 techniques/WP, 3+ unjustified)\n"
            "4: Borderline (high technique count or multiple unjustified)\n"
            "5: Acceptable (≤3 tech/WP but 2 unjustified)\n"
            "6: Mostly focused (3-4 tech/WP, ≤1 unjustified)\n"
            "7: Good focus (≤3 tech/WP, at most 1 unjustified)\n"
            "8: Strong focus (≤2 tech/WP, at most 1 unjustified)\n"
            "9: Excellent (≤3 tech/WP, all justified)\n"
            "10: Perfect (≤2 tech/WP, all justified, clearly compose)"
        ),
        score_max=10,
    ),

    SignalSpec(
        id="R7_specificity",
        name="Implementation Specificity",
        question=(
            "Does the proposal make SPECIFIC commitments, or does it rely on "
            "vague placeholders? Specificity means concrete, named commitments "
            "— models, datasets, hyperparameters, theorems, algorithms. "
            "Vagueness shows up as filler words: 'appropriately tuned', "
            "'standard techniques', 'suitable baselines', 'various methods'."
        ),
        cot_scaffolding=(
            "Do EXPLICIT COUNTING of two lists:\n"
            "\n"
            "STEP 1 — Count VAGUE MARKERS in the methodology/approach sections:\n"
            "  - 'appropriately', 'appropriate' (tuned, sized, chosen)\n"
            "  - 'standard' (techniques, optimizer, practice, baselines)\n"
            "  - 'typical' (setup, values, range)\n"
            "  - 'suitable', 'reasonable' (parameters, baselines, threshold)\n"
            "  - 'we will tune/explore/investigate' (without search space)\n"
            "  - 'various', 'several', 'many' (without enumeration)\n"
            "  Count N_vague.\n"
            "\n"
            "STEP 2 — Count SPECIFIC MARKERS:\n"
            "  - Named models (e.g., 'ResNet-18', 'Qwen3-30B')\n"
            "  - Named datasets (e.g., 'CIFAR-10', 'Adult', 'GLUE/MNLI')\n"
            "  - Specific hyperparameters (e.g., 'width ≥ poly(n)', 'rank r')\n"
            "  - Named algorithms (e.g., 'TuRBO', 'BOHB', 'implicit differentiation')\n"
            "  - Specific thresholds (e.g., '≥2% accuracy improvement')\n"
            "  - Named theorems or mathematical objects\n"
            "  Count N_specific.\n"
            "\n"
            "STEP 3 — Compute specificity_ratio = N_specific / (N_specific + N_vague).\n"
            "  If N_specific + N_vague = 0, default to 5.\n"
            "\n"
            "STEP 4 — Apply this TABLE:\n"
            "  specificity_ratio ≥ 0.95 AND N_specific ≥ 10 → score 10\n"
            "  specificity_ratio ≥ 0.90 AND N_specific ≥ 8 → score 9\n"
            "  specificity_ratio ≥ 0.85 → score 8\n"
            "  specificity_ratio ≥ 0.75 → score 7\n"
            "  specificity_ratio ≥ 0.65 → score 6\n"
            "  specificity_ratio ≥ 0.55 → score 5\n"
            "  specificity_ratio ≥ 0.45 → score 4\n"
            "  specificity_ratio ≥ 0.35 → score 3\n"
            "  specificity_ratio ≥ 0.20 → score 2\n"
            "  specificity_ratio < 0.20 → score 1\n"
        ),
        scoring_rubric=(
            "1: <20% specific (almost all vague placeholders)\n"
            "2: 20-34% specific\n"
            "3: 35-44% specific\n"
            "4: 45-54% specific\n"
            "5: 55-64% specific\n"
            "6: 65-74% specific\n"
            "7: 75-84% specific\n"
            "8: 85-89% specific\n"
            "9: 90-94% specific with 8+ named commitments\n"
            "10: 95%+ specific with 10+ named commitments"
        ),
        score_max=10,
    ),

    SignalSpec(
        id="R8_arithmetic",
        name="Arithmetic Consistency",
        question=(
            "Are the NUMERICAL CLAIMS in the proposal internally consistent? "
            "Check: (1) complexity claims consistent with described algorithms, "
            "(2) convergence rates consistent with stated assumptions, "
            "(3) citations resolve to plausible papers, "
            "(4) named quantities use correct units and magnitudes."
        ),
        cot_scaffolding=(
            "Do EXPLICIT CHECKING (line by line):\n"
            "\n"
            "STEP 1 — List every NUMERICAL CLAIM involving:\n"
            "  - complexity bounds (O(n³), O(d·m), O(1/√T))\n"
            "  - convergence rates\n"
            "  - computational costs (FLOPs, memory, wall-time)\n"
            "  - dataset sizes or dimensionality claims\n"
            "  - threshold values with units\n"
            "  - citation years and venues\n"
            "Cap at 10 claims.\n"
            "\n"
            "STEP 2 — For EACH claim, perform CONSISTENCY CHECK:\n"
            "  * Does the ORDER OF MAGNITUDE make sense?\n"
            "  * Is the claim internally consistent with other claims?\n"
            "  * Do cited years/venues seem plausible?\n"
            "  Mark each as:\n"
            "    - CONSISTENT: plausible and internally coherent\n"
            "    - INCONSISTENT: off by ≥3x, wrong units, or contradicts "
            "another claim in the proposal\n"
            "    - UNVERIFIABLE: cannot check without external lookup\n"
            "\n"
            "STEP 3 — Count N_inconsistent, N_consistent, N_claims_total "
            "(CONSISTENT + INCONSISTENT only).\n"
            "\n"
            "STEP 4 — Apply this TABLE:\n"
            "  N_claims_total ≤ 1 → cap at score 4 (too few to verify)\n"
            "  N_inconsistent ≥ 4 → score 1\n"
            "  N_inconsistent = 3 → score 2\n"
            "  N_inconsistent = 2 → score 3\n"
            "  N_inconsistent = 1 → score 5\n"
            "  N_inconsistent = 0 AND N_claims ∈ [2, 4] → score 6\n"
            "  N_inconsistent = 0 AND N_claims ∈ [5, 6] → score 7\n"
            "  N_inconsistent = 0 AND N_claims ∈ [7, 8] → score 8\n"
            "  N_inconsistent = 0 AND N_claims ≥ 9 → score 9\n"
            "  N_inconsistent = 0 AND N_claims ≥ 9 + all internally cross-referenced → score 10\n"
            "\n"
            "NOTE: UNVERIFIABLE claims do NOT count as inconsistent."
        ),
        scoring_rubric=(
            "1: 4+ arithmetic/consistency errors\n"
            "2: 3 errors\n"
            "3: 2 errors\n"
            "4: Too few claims to verify (≤1 verifiable claim)\n"
            "5: 1 error\n"
            "6: 0 errors, 2-4 verifiable claims\n"
            "7: 0 errors, 5-6 verifiable claims\n"
            "8: 0 errors, 7-8 verifiable claims\n"
            "9: 0 errors, 9+ verifiable claims\n"
            "10: 0 errors, 9+ claims, all internally cross-referenced"
        ),
        score_max=10,
    ),
]

SIGNAL_WEIGHTS: dict[str, float] = {
    "R1_theoretical_novelty": 0.18,
    "R2_formalism": 0.16,
    "R3_reasoning_depth": 0.14,
    "R4_evidence_rigor": 0.12,
    "R5_risk_awareness": 0.10,
    "R6_research_focus": 0.12,
    "R7_specificity": 0.10,
    "R8_arithmetic": 0.08,
}

assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 1e-6, \
    f"Weights must sum to 1.0, got {sum(SIGNAL_WEIGHTS.values())}"

SCORE_MAX: dict[str, int] = {s.id: s.score_max for s in SIGNALS}
