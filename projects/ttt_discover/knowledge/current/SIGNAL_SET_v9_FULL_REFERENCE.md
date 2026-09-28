# Signal Set v9 — Complete Reference Document

**Version**: v9 (2026-04-18)
**Code**: `src/co_scientist/shared/ten_signal_reward.py`
**Purpose**: This document contains the full specification of every signal, including the exact grading prompts, CoT scaffolding, scoring rubrics, and locus directives. Intended for external review.

---

## Overview

The signal set evaluates LLM-generated research plans on 11 dimensions (2 hard gates + 9 gradient signals). Each plan is graded by a separate LLM call per signal ("separate-call" mode) to minimize halo effects. Grading is repeated N=2 times per signal, taking the median.

**Aggregate reward**:
```
R = Σ w_i × (s_i - 1) / 4
```
where s_i ∈ {1,...,5} is the integer score for signal i, and w_i is its weight (Σ w_i = 1).

**Grader preamble** (shared across all signals):
> You are an expert-level research plan evaluator. Your job is to assess a research plan on a specific, well-defined evaluation dimension.
>
> RULES:
> - Evaluate based on what is ACTUALLY in the plan, not what it claims about itself.
> - Follow the rubric strictly. Each score 1-5 has specific criteria.
> - Be skeptical of surface-level plausibility — check whether claims are supported by concrete content.
> - Do not demand rigor-theater (long lists of statistical procedures, exhaustive failure modes) when the plan is concise and focused. Judge the substance of what's there.
> - Concise methodology that delegates boilerplate to citations is acceptable if the core is clearly described.
> - If a signal has "Required reasoning" scaffolding, you MUST do the reasoning BEFORE assigning a score.
> - Integer scores only: 1, 2, 3, 4, or 5.

**Prompt template** (per signal):
```
{SHARED_PREAMBLE}

You will evaluate the following research plan on ONE dimension: **{signal.name}**.

# Research Goal
{goal}

# Research Plan
{plan}

# Evaluation Dimension

## Signal: {signal.name} (id={signal.id})

**Question**: {signal.question}

**Required reasoning before scoring**:
{signal.cot_scaffolding}

**Scoring rubric**:
{signal.scoring_rubric}

---

# Output Format

<evaluation>
    <dim id="{signal.id}">
        <reasoning>Your analysis (do required reasoning steps if any).</reasoning>
        <score>INTEGER 1-5</score>
    </dim>
</evaluation>

Begin your evaluation now.
```

---

## Layer 0: Hard Gates (Binary Pass/Fail)

Hard gates are evaluated before gradient signals. If either gate fails, aggregate reward = 0.

### G1: Goal-Contrast Margin

**What it checks**: Is the plan specific to THIS research goal, or is it a generic plan that could apply to any goal?

**Mechanism**: Grade the plan against BOTH the target goal AND an alternative goal. Compute margin = score(target) - score(alt). If margin < 0.10, the plan is too generic.

**Threshold**: margin ≥ 0.10

### G2: Claim Verification

**What it checks**: Are factual claims in the plan real (not fabricated)?

**Mechanism**: Extract factual claims from the plan and verify against known information.

**Threshold**: fabrication ratio < 15%

---

## Layer 1: Gradient Signals (1-5 Integer Scale)

### Signal Weights (v9)

| # | ID | Name | Weight | Grader | Status |
|---|---|---|---|---|---|
| S1 | S1_depth | Reasoning Depth | 0.07 | Qwen3-30B | Active |
| S2 | S2_rigor | Evidence Rigor | 0.08 | Qwen3-30B | Active |
| S2a | S2a_formalism | Mathematical Formalism | 0.10 | Qwen3-30B | Active (v9 NEW) |
| S3 | S3_positioning | Positioning | 0.10 | Qwen3-30B | Active |
| S4 | S4_significance | Significance | 0.00 | — | **Disabled** |
| S5 | S5_feasibility | Feasibility Evaluability | 0.10 | Qwen3-30B | Active |
| S6 | S6_risk_awareness | Mature Risk Awareness | 0.10 | Qwen3-30B | Active |
| S7 | S7_specificity | Implementation Specificity | 0.10 | Qwen3-30B | Active |
| S8 | S8_scope | Scope-Generalization Alignment | 0.13 | Qwen3-30B | Active |
| S9 | S9_focus | Research Focus | 0.14 | Qwen3-30B | Active |
| SA | SA_arithmetic | Arithmetic Consistency | 0.08 | **Qwen3-235B** | Active (v9 NEW) |

---

## S1_depth — Reasoning Depth

**Weight**: 0.07

**Question**:
> For each major design choice in the plan, does the author explain WHY it was made? Depth means explicit derivation: problem → insight → each choice justified by reference to that insight. Shallow plans list techniques with citations but don't explain the logical connection.

**CoT Scaffolding (Required reasoning before scoring)**:

Do EXPLICIT CLASSIFICATION with STRICT definitions:

STEP 1 — State the core problem (from Problem section) and KEY INSIGHT (from Core Idea / Motivation) in one sentence each.

STEP 2 — Identify the TOP 3-5 LOAD-BEARING DESIGN CHOICES — the decisions that DISTINGUISH this plan from a standard approach. A load-bearing design choice is:
  - a named ALGORITHMIC innovation (e.g., 'tree-guided evolution', 'soft parallel decoding', 'on-policy uniform training')
  - a novel TRAINING procedure (e.g., 'RL with self-correction reward')
  - a novel ARCHITECTURE feature (e.g., 'structure-aware 441-token vocab')
  - a novel DATA or OBJECTIVE choice (e.g., 'hybrid on-policy + SFT loss')

DO NOT count as design choices:
  - Hardware (GPUs, memory, compute)
  - Dataset names (unless the plan is about building a new dataset)
  - Standard optimizer/hyperparameter values (lr, batch size, temperature, seed) unless explicitly argued as novel
  - Evaluation benchmarks
  - Boilerplate tools (git, logging, monitoring)

IMPORTANT: Cap the list at 5 items. If you identify more than 5, pick the 5 MOST central to the plan's contribution.

STEP 3 — For EACH of the 3-5 load-bearing choices, classify as:
  * JUSTIFIED: plan contains a sentence stating WHY this choice addresses the insight/problem (causal or mechanistic).
  * ASSERTED: plan names the choice but gives no 'why' sentence.

STEP 4 — Count (N_choices, N_asserted). Note: N_choices ∈ [3, 5].

STEP 5 — Apply this TABLE (pick row matching N_asserted):
  N_asserted = 0 → score 5
  N_asserted = 1 → score 4
  N_asserted = 2 → score 3
  N_asserted = 3 → score 2
  N_asserted ≥ 4 → score 1

STRICT RULE: 'inspired by', 'following', 'as in [cite]' are NEVER justifications. But a citation PLUS a mechanistic sentence ('we use X because it provides property Y that addresses Z') IS justified.

**Scoring Rubric**:
- 1: ≥4 load-bearing choices asserted (list of techniques, no argument)
- 2: 3 asserted
- 3: 2 asserted
- 4: 1 asserted (mostly justified)
- 5: 0 asserted — every load-bearing choice has explicit why-sentence

**Locus Directive**:
> Quote each load-bearing design choice (named algorithmic innovation, training procedure, architecture feature, or data/objective choice) that you classified as ASSERTED — named without a mechanistic 'why' sentence explaining how it addresses the plan's insight. Prefer the SHORTEST sentence containing the asserted choice.

---

## S2_rigor — Evidence Rigor

**Weight**: 0.08

**Question**:
> Can the proposed evidence support the claim? For EMPIRICAL work: are baselines FAIR (appropriate, modern, matched in scale — not strawman) and is there an OPERATIONALIZED success criterion (specific metric)? For THEORETICAL work: are assumptions stated and is a proof sketch/strategy given (not just 'we prove X')?

**CoT Scaffolding**:

Do EXPLICIT CLASSIFICATION (no interpretation):

STEP 1 — Identify if the work is EMPIRICAL or THEORETICAL.

STEP 2 (EMPIRICAL branch) — List every baseline in the Evaluation section. For each baseline, classify as:
  * FAIR: a modern, appropriate-scale method from the same problem space (e.g., comparing a new LLM method to GPT-4o, Claude, or a strong recent approach).
  * STRAWMAN: obviously weak baseline unlikely to be competitive. Signs: 'random baseline', 'untrained model', 'hand-coded heuristic from 201X', a method 5+ years old in a fast-moving field, a method from a different problem regime, or a method the plan already argues is inadequate.
Count N_fair and N_strawman.

STEP 2 (THEORETICAL branch) — Mark each of the following as present (1) or absent (0):
  * Explicit assumptions stated
  * Proof strategy / sketch given (not just 'we will prove')
  * Tight bounds discussed OR key lemma stated
Count T = sum (0-3).

STEP 3 — Check the SUCCESS CRITERION. Is there a specific measurable threshold or operationalized metric?
  * OPERATIONALIZED: named metric with benchmark ('pass@1 on MATH-500 above 0.5', 'perplexity on WikiText2 below X').
  * VAGUE: 'we expect improvement', 'our method should outperform', 'we aim to demonstrate'. Mark VAGUE if criterion uses words like 'expect', 'should', 'aim to' without a threshold.

STEP 4 — Apply this TABLE (no exceptions):
  EMPIRICAL:
    N_strawman ≥ 2 → score 1 (or 2 if a fair baseline also present)
    N_strawman ≥ 1 AND criterion VAGUE → score 1
    N_strawman ≥ 1 AND N_fair ≤ 1 → score 2
    N_fair ≥ 2 AND criterion VAGUE → score 2
    N_fair ≥ 2 AND criterion OPERATIONALIZED → score 4
    N_fair ≥ 3 AND criterion OPERATIONALIZED AND targeted ablations → score 5
    Otherwise (e.g., N_fair = 1) → score 3

  THEORETICAL:
    T = 0 or 1 → score 1-2
    T = 2 → score 3
    T = 3 + operationalized criterion → score 4-5

STRICT RULE: A strawman baseline is a strawman even if the plan calls it 'standard'. Judge baselines by whether they would be a meaningful comparison, not by what the plan calls them.

**Scoring Rubric**:
- 1: ≥2 strawman baselines, OR 1+ strawman + vague criterion
- 2: 1 strawman with only 1 fair baseline, OR fair baselines but vague criterion
- 3: 1 fair baseline with operationalized criterion, or theoretical T=2
- 4: ≥2 fair baselines + operationalized criterion
- 5: ≥3 fair baselines + operationalized criterion + targeted ablations

**Locus Directive**:
> Quote each STRAWMAN baseline (weak, obsolete, or off-regime method proposed as comparison) OR each VAGUE success-criterion sentence (uses 'expect', 'should', 'aim to' without a measurable threshold). For theoretical work, quote spans where assumptions are missing or where a proof strategy is promised but not sketched.

---

## S2a_formalism — Mathematical Formalism (v9 NEW)

**Weight**: 0.10
**Grader**: Qwen3-30B

**Question**:
> Does the plan contain NON-TRIVIAL mathematical content? Non-trivial means: a named equation written in symbolic form that is NOT tautological (L = -R is tautological), NOT just hyperparameter values (lr=3e-5), and NOT standard definitions. Count DISTINCT equations/formulas that contribute to the plan's core methodology.

**CoT Scaffolding**:

Do EXPLICIT COUNTING (no interpretation):

STEP 1 — Scan the ENTIRE plan for EQUATIONS written in symbolic form.
An EQUATION contains:
  - mathematical symbols (=, +, -, *, /, ^, log, exp, sum, integral, E[])
  - AND at least one NAMED variable or function beyond simple constants

List each equation found. Write down the actual formula.

STEP 2 — For EACH equation, classify as:
  * TAUTOLOGICAL: L = -R (loss is negative reward), y = f(x) without defining f, or any formula that restates the problem without adding analytical content.
  * HYPERPARAMETER-ONLY: Just lr=3e-5, batch=32, rank=8, etc. These are parameter settings, NOT formulas.
  * NON-TRIVIAL: A formula that adds analytical content. Examples:
    - A policy gradient: nabla J = E[R * nabla log pi]
    - An objective: J_beta = E[log E[exp(beta * R)]]
    - A bound: Q(s) + c * P(s) * sqrt(1+T)/(1+n(s))
    - A regularizer with specific form: KL(pi || pi_ref) <= delta
    - A proof sketch inequality or lemma statement

STEP 3 — Count N_nontrivial = number of NON-TRIVIAL formulas.

STEP 4 — Apply this TABLE (no exceptions):
  N_nontrivial = 0 → score 1
  N_nontrivial = 1 AND formula is a standard textbook result (plain REINFORCE, plain cross-entropy, plain KL) → score 2
  N_nontrivial = 1 AND formula is adapted/novel for this plan → score 3
  N_nontrivial = 2 AND at least 1 non-textbook → score 4
  N_nontrivial >= 3 AND at least 1 non-textbook AND formulas form a coherent derivation chain → score 5

STRICT RULE: Hyperparameter values (lr=3e-5, beta=0.1, batch=32) are NEVER equations. Section headers with math words ('Mathematical Formulation') are NOT formulas. LaTeX formatting of a number ($\eta = 10^{-5}$) is NOT a formula.

**Scoring Rubric**:
- 1: Zero non-trivial formulas (only prose, hyperparams, or tautologies)
- 2: One standard textbook formula (plain REINFORCE, standard loss)
- 3: One adapted/novel formula
- 4: Two formulas with at least one non-textbook
- 5: Three+ formulas forming a coherent derivation chain

**Locus Directive**:
> Quote the COMPLETE SENTENCE where a mathematical claim is made in prose form without an accompanying symbolic equation (e.g., 'we use a policy gradient objective' without writing nabla J = ...). If score >= 4, output empty array.

**Validation**: Spearman ρ = 1.000 with Opus 4.7 Math Formalism score (N=3). 100% perturbation detection (5/5 repeats).

---

## S3_positioning — Positioning

**Weight**: 0.10

**Question**:
> Does the plan name SPECIFIC prior methods and explain WHY each is insufficient for this problem? Positioning is about argument, not citation count, but it requires NAMED prior work — not vague references like 'prior work', 'existing methods', 'some approaches'.

**CoT Scaffolding**:

Do EXPLICIT COUNTING (no interpretation):

STEP 1 — List every NAMED prior method/work in the plan. A NAMED method has one of: (a) a proper-noun name (e.g., 'GEPA', 'TextGrad', 'ESM-2', 'AlphaEvolve'), (b) a specific citation (e.g., 'Huang 2024', 'Vaswani et al.'), (c) a concrete descriptor that identifies a specific body of work (e.g., 'DPR retrieval', 'LoRA fine-tuning'). DO NOT count vague references: 'prior work', 'existing methods', 'some approaches', 'recent work', 'alternative approaches', 'other methods'. These are NOT named.

STEP 2 — For EACH named method from STEP 1, ask: does the plan contain an explicit sentence stating what is INSUFFICIENT about that method FOR THIS PROBLEM? Mark each as:
  * WITH_INSUFFICIENCY: plan states a specific limitation (e.g., 'needs thousands of mutations', 'requires labeled data', 'scales poorly to long context').
  * NAMED_ONLY: method is named but no insufficiency sentence.

STEP 3 — Count (N_named, N_with_insufficiency).

STEP 4 — Apply this TABLE (no exceptions):
  N_named = 0 → score 1
  N_named = 1, N_with_insufficiency = 0 → score 2
  N_named = 1, N_with_insufficiency = 1 → score 3
  N_named ≥ 2, N_with_insufficiency ≤ 1 → score 3
  N_named ≥ 2, N_with_insufficiency ≥ 2 → score 4
  N_named ≥ 3, N_with_insufficiency ≥ 3 AND forms explicit chain-of-reasoning ('X does A, Y extends to B, both fail on C, we target C') → score 5

STRICT RULE: 'inspired by X' or 'we use X' is NOT insufficiency. Insufficiency must be a sentence explicitly stating a limitation.

**Scoring Rubric**:
- 1: 0 named prior methods
- 2: 1 named without insufficiency
- 3: 1 named with insufficiency, OR 2+ named but ≤1 have insufficiency
- 4: 2+ named, 2+ have insufficiency
- 5: 3+ named with full chain-of-reasoning about gap

---

## S4_significance — Significance (DISABLED, weight=0.00)

**Status**: Disabled. Qwen3-30B grader cannot isolate single-section content (integrates across sections despite instruction). Detection rate: 0%.

**Question**:
> Does the plan make the STAKES of the problem concrete? Specifically, does it name: (a) WHO benefits, (b) WHAT changes, and (c) WHY NOW?

*(Full rubric preserved in code for future re-enablement. See `ten_signal_reward.py` lines 309-395.)*

---

## S5_feasibility — Feasibility Evaluability

**Weight**: 0.10

**Question**:
> Can a domain expert read this plan and judge whether it is implementable? Specifically: is the CORE ALGORITHM specified, are KEY DEPENDENCIES (datasets, models, tools) named, are CRITICAL parameters given? A good plan makes the methodology SELF-CONTAINED enough that a reader in the field can assess feasibility without needing additional context.
>
> DO NOT penalize plans for omitting: cloud providers, exact hardware specs, cost estimates, batch sizes, parallelization strategies, or other deployment-level details. These belong in grant proposals, not research plans. Published paper methodologies ROUTINELY omit these.

**Scoring Rubric**:
- 1: Not evaluable. Core algorithm is vague ('we tune hyperparameters appropriately', 'we use standard techniques') or key dependencies (datasets, models) are unnamed. A domain expert cannot judge if the approach is implementable.
- 2: Marginal. Core algorithm stated but 2+ key design decisions are hand-waved. Reader can guess but not confidently judge feasibility.
- 3: Adequate. Core algorithm described and most dependencies named, but one critical aspect (algorithm detail, key parameter, or dependency) is missing or vague.
- 4: Feasibility-judgeable. Core algorithm is clearly described, key dependencies (datasets, models, tools) are named, critical parameters appear at methodology level (no need for exhaustive implementation details). A domain expert could confidently judge whether this approach works.
- 5: Exceptionally clear. All critical design choices are specified AND briefly justified. Reader understands not just WHAT the plan does but WHY each choice is made. Methodology is self-contained enough that a skilled implementer could execute it.

**Note**: No CoT scaffolding — qualitative rubric bands (already 83-89% detection).

---

## S6_risk_awareness — Mature Risk Awareness

**Weight**: 0.10

**Question**:
> Does the plan demonstrate mature awareness of its own boundaries, risks, and failure modes? A good plan names 2-3 SPECIFIC failure modes, identifies which assumptions are most fragile, states what the plan does NOT claim (scope boundary), and has a coherent view of how failures would be recognized. This is NOT about enumerating 10 failure modes (dilutive theater) — it's about showing the author understands the risk structure.

**CoT Scaffolding**:
Before scoring:
1. Does the plan identify 2-3 specific ways it could fail? (specific failure = specific assumption breaking, specific resource running out, specific domain mismatch)
2. Does the plan state which assumptions are most fragile (the 'key bet')?
3. Does the plan acknowledge scope boundaries (what it does NOT claim)?
4. Is there a coherent risk structure, or just a laundry list?
5. Assign a score.

**Scoring Rubric**:
- 1: No risk awareness. Plan assumes success. No discussion of when/why the method might fail, no scope boundaries, no acknowledgment of fragile assumptions.
- 2: Generic disclaimers. Plan says 'there might be limitations' or 'further research is needed' without specificity.
- 3: Partial risk awareness. Plan identifies 1 failure mode OR mentions scope boundary, but risk structure is incomplete. Doesn't name fragile assumptions.
- 4: Mature risk awareness. Plan names 2-3 specific failure modes (e.g., 'fails on insufficiently capable base models', 'assumes tree-guided evolution outperforms linear chain'). Identifies fragile assumptions. States scope boundary.
- 5: Exceptional risk awareness. Plan has a coherent model of its own risk structure: names specific failure modes, articulates which assumptions are load-bearing, states scope boundary explicitly, AND sketches a fallback direction.

---

## S7_specificity — Implementation Specificity

**Weight**: 0.10

**Question**:
> Does the plan make SPECIFIC commitments, or does it rely on vague placeholders? Specificity means concrete, named commitments — models, datasets, hyperparameters, metrics, theorems. Vagueness shows up as filler words that look like specificity but don't commit to anything: 'appropriately tuned', 'standard techniques', 'suitable baselines'.

**CoT Scaffolding**:

Do EXPLICIT COUNTING of two lists (no interpretation):

STEP 1 — Count VAGUE MARKERS in the Core Idea + Methodology + Evaluation sections. A VAGUE MARKER is any of:
  - 'appropriately', 'appropriate' (tuned, sized, chosen)
  - 'standard' (techniques, optimizer, practice, procedure, baselines)
  - 'typical' (setup, values, range)
  - 'suitable', 'reasonable' (parameters, baselines, threshold)
  - 'we will tune', 'we will explore', 'we will investigate' (without naming the search space)
  - 'various', 'several', 'many' (without enumeration)
  - 'some' as a quantifier (e.g., 'some regularization')
  - 'further details', 'exact values TBD', 'to be determined'
  - 'state-of-the-art' used as a substitute for a named method

Go line-by-line and COUNT occurrences. Write V = the count.

STEP 2 — Count SPECIFIC COMMITMENTS in the same sections. A SPECIFIC COMMITMENT is any of:
  - a named model (e.g., 'GPT-4o', 'Qwen3-30B', 'LLaMA-7B')
  - a named dataset (e.g., 'MATH-500', 'WikiText2', 'PACS')
  - a numeric hyperparameter (lr=3e-5, batch=32, seeds=5, α=2.0)
  - a named loss/objective (e.g., 'GRPO loss', 'cross-entropy')
  - a named metric (e.g., 'perplexity on WikiText2', 'pass@1')
  - a named theorem/lemma or formally stated proposition

Count occurrences. Write C = the count.

STEP 3 — Apply this TABLE (no exceptions):
  V ≥ 6 → score 1 (pervasive vagueness)
  V ≥ 3 and C ≤ 3 → score 2
  V ≥ 3 and C ≥ 4 → score 3 (vagueness offset by some specificity)
  V ≤ 2 and C ≥ 5 → score 4 (specific with minor vagueness)
  V ≤ 1 and C ≥ 8 → score 5 (exceptional specificity, almost no filler language)
  Otherwise → score 3

STRICT RULE: Do NOT rationalize vague markers as 'acceptable delegation to prior work'. If the plan uses 'standard optimizer' without naming it, that is vague — even if 'standard' means Adam in practice.

**Scoring Rubric**:
- 1: V ≥ 6 (pervasive vagueness, even if some specifics present)
- 2: V ≥ 3 and C ≤ 3
- 3: V ≥ 3 and C ≥ 4, OR anything not in other bands
- 4: V ≤ 2 and C ≥ 5
- 5: V ≤ 1 and C ≥ 8

---

## S8_scope — Scope-Generalization Alignment

**Weight**: 0.13

**Question**:
> Does the plan's evaluation COVER what the plan claims? IMPORTANT: This is NOT about breadth (broad is NOT better than narrow). This is about whether the evaluation would SUPPORT the plan's claims if successful. A narrow claim with narrow matching evaluation is a HIGH SCORE (4-5).

**CoT Scaffolding**:

Do EXPLICIT COUNTING (no interpretation):

STEP 1 — COUNT OVERCLAIM TRIGGER WORDS in the Problem + Motivation + Core Idea sections. A trigger word is any of:
  - 'universal', 'universally'
  - 'general', 'generally', 'general-purpose'
  - 'any task', 'any domain', 'any field', 'any research'
  - 'all domains', 'all fields', 'all scientific'
  - 'across domains', 'across fields', 'across all'
  - 'every discipline', 'every field'
  - 'foundation', 'foundational' (used as a claim, not a citation)
Write OC = the count.

STEP 2 — COUNT EVALUATION BENCHMARKS / DOMAINS in the Evaluation section. Group related benchmarks into DOMAIN BUCKETS. For example, GSM8K + MATH-500 + Minerva are ONE bucket (math). HumanEval + MBPP are ONE bucket (code). Write D = number of distinct domain buckets evaluated.

STEP 3 — Apply this TABLE (no exceptions; pick the single matching row):
  OC ≥ 4 AND D ≤ 2 → score 1 (severe overclaim)
  OC ≥ 2 AND D ≤ 2 → score 2 (moderate overclaim)
  OC ≥ 2 AND D ≥ 3 → score 3 (broad claim partly substantiated)
  OC ≤ 1 AND D = 1 → score 3 (narrow claim, narrow eval)
  OC ≤ 1 AND D ≥ 2 AND plan claims transfer/OOD → must include OOD evaluation: yes → 4-5, no → 2
  OC ≤ 1 AND D ≥ 2 AND plan claims a specific problem only → 4
  OC = 0 AND eval matches each claim exactly → 5

STRICT RULE: 'universal' or 'across all domains' in the Problem section IS an overclaim trigger, even if it's prose framing. Count every occurrence.

**Scoring Rubric**:
- 1: OC ≥ 4 AND D ≤ 2 (severe overclaim)
- 2: OC ≥ 2 AND D ≤ 2 (moderate overclaim)
- 3: OC ≥ 2 AND D ≥ 3 (broad claim partly substantiated), OR narrow claim with single benchmark
- 4: OC ≤ 1 AND D ≥ 2 with specific claim
- 5: OC = 0 AND eval exactly matches claim scope

---

## S9_focus — Research Focus

**Weight**: 0.14

**Question**:
> Does the plan STAY focused on a small number of core techniques, or does it STACK unrelated methods without justifying each? A focused plan has 1-3 core techniques that clearly compose. A stacked plan lists many techniques (e.g., 'Bayesian optimization + meta-learning + evolutionary algorithms + curiosity bonuses') without explaining why each is necessary.

**CoT Scaffolding**:

Do EXPLICIT COUNTING with STRICT definitions:

STEP 1 — Identify the CORE IDEA techniques. Read ONLY the '## Core Idea' section. List the distinct METHODOLOGICAL techniques named in the Core Idea as part of the central approach. A 'technique' is a named algorithmic or training component that is FUNDAMENTAL to what the plan proposes.

DO NOT count as techniques:
  - standard infrastructure (GPUs, distributed training, logging)
  - standard datasets (used for evaluation, not as a method)
  - standard optimizer/learning-rate choices
  - Components of a single multi-stage pipeline where the stages are obviously interdependent (e.g., an encoder + decoder counts as ONE architecture, not two)
  - Evaluation metrics or benchmarks
  - Citations of prior methods (unless the plan USES them as part of its method, not just for positioning)

Cap the list at 8 items maximum.

STEP 2 — For EACH technique, mark as exactly ONE of:
  * JUSTIFIED: plan has a sentence explaining WHY this technique is needed for the core idea's insight (a mechanistic/causal sentence).
  * UNJUSTIFIED: technique is named but no explanation of why it's needed for THIS plan's insight.

STEP 3 — Count (N_total, N_unjustified).

STEP 4 — Apply this TABLE:
  N_total ≤ 3 AND N_unjustified = 0 → score 5 (focused, justified)
  N_total ≤ 3 AND N_unjustified ≥ 1 → score 4
  N_total ∈ [4, 6] AND N_unjustified ≤ 1 → score 4 (ok focus)
  N_total ∈ [4, 6] AND N_unjustified ∈ [2, 3] → score 3
  N_total ∈ [4, 6] AND N_unjustified ≥ 4 → score 2
  N_total ≥ 7 AND N_unjustified ≤ 2 → score 3 (many but mostly justified)
  N_total ≥ 7 AND N_unjustified ≥ 3 → score 2
  N_total ≥ 7 AND N_unjustified ≥ 5 → score 1 (classic stacking)

STRICT RULE: 'inspired by X' is NOT justification. Justification must explain the CAUSAL role of the technique for this plan's insight. But techniques that share a sentence explaining a joint mechanism (e.g., 'A and B together enable C because...') both count as JUSTIFIED if C is explained.

**Scoring Rubric**:
- 1: N_total ≥ 7 AND N_unjustified ≥ 5 (severe stacking)
- 2: moderate stacking or many unjustified
- 3: borderline (2-3 unjustified of moderate total)
- 4: focused or mostly justified
- 5: tight focus (≤3 techniques) with every one justified

---

## SA_arithmetic — Arithmetic Consistency (v9 NEW)

**Weight**: 0.08
**Grader**: Qwen3-235B-A22B-Instruct-2507 (requires stronger math reasoning)

**Question**:
> Are the NUMERICAL CLAIMS in the plan internally consistent? Check: (1) parameter counts consistent with stated model sizes, (2) compute budgets consistent with model forward/backward pass costs, (3) citations resolve to plausible papers (not hallucinated arXiv IDs), (4) named quantities use correct units and magnitudes.

**CoT Scaffolding**:

Do EXPLICIT CHECKING (line by line):

STEP 1 — List every NUMERICAL CLAIM in the plan involving:
  - parameter counts (e.g., '3.2B parameters', '10% of layers')
  - compute budgets (FLOPs, GPU-hours, wall-time, dollar cost)
  - LoRA/adapter sizing (rank, which layers, added param count)
  - speed/throughput claims (tokens/sec, samples/iter)
  - dataset sizes or sample counts
  - threshold values with units (e.g., 'QED >= 0.85', '1,000 GFLOPs')
Cap at 8 claims. Pick the most load-bearing ones.

STEP 2 — For EACH numerical claim, perform a QUICK CONSISTENCY CHECK:
  * Does the number's ORDER OF MAGNITUDE make sense?
    - A 7B model has ~7e9 params; 10% of layers ~ 0.7B params, not 3.2B
    - A 7B forward pass ~ 2*N*T FLOPs per token; 1000 tokens ~ 1.4e13 FLOPs
    - LoRA rank r on a layer with hidden dim d adds ~2*r*d params per layer
    - arXiv IDs have format YYMM.NNNNN; the year+month should be plausible
  * Mark each as:
    - CONSISTENT: magnitude and units are plausible
    - INCONSISTENT: off by >=3x, or units wrong, or internally contradictory
    - UNVERIFIABLE: cannot check without external lookup (acceptable)

STEP 3 — Count N_inconsistent and N_claims (CONSISTENT + INCONSISTENT only).

STEP 4 — Apply this TABLE:
  N_inconsistent >= 3 → score 1 (pervasive arithmetic errors)
  N_inconsistent = 2 → score 2
  N_inconsistent = 1 → score 3
  N_inconsistent = 0 AND N_claims >= 5 → score 5
  N_inconsistent = 0 AND N_claims >= 3 → score 4
  N_claims <= 2 (too few to verify) → cap at score 3

NOTE: UNVERIFIABLE claims do NOT count as inconsistent. Only flag claims where you can compute or estimate the correct answer.

**Scoring Rubric**:
- 1: 3+ arithmetic errors (parameter counts, FLOPs, units inconsistent)
- 2: 2 errors
- 3: 1 error, OR too few numerical claims to verify
- 4: 0 errors with 3+ verifiable claims
- 5: 0 errors with 5+ verifiable claims, all consistent

**Locus Directive**:
> Quote the COMPLETE SENTENCE containing each INCONSISTENT numerical claim (parameter count off by >=3x, FLOP budget wrong order of magnitude, hallucinated citation, unit-inconsistent threshold).

**Validation**: Spearman ρ = 0.866 with Opus 4.7 Implementation Realism score (N=3). 100% perturbation detection (5/5 repeats).

---

## Appendix: Aggregation Formula

```python
def aggregate_reward(signal_scores: dict[str, int]) -> float:
    """Scalar aggregate across gradient signals using SIGNAL_WEIGHTS."""
    total = 0.0
    for sid, weight in SIGNAL_WEIGHTS.items():
        score = signal_scores.get(sid)
        if score is not None:
            total += weight * (score - 1) / 4.0  # normalize 1→0.0, 5→1.0
    return total
```
