# Signal Set v8.1-minimal

**Last updated**: 2026-04-16
**Code**: `src/co_scientist/shared/ten_signal_reward.py`
**Grader model**: Qwen/Qwen3-30B-A3B (policy = grader = same model)
**Grader mode**: separate_call (one grader call per signal per plan)
**Grader repeats**: N=2 (median)
**Grader max_tokens**: 4096 (v8.1 rubrics need longer CoT than v5/v6's 2048)
**Template**: `shared/docs/RESEARCH_PLAN_TEMPLATE.md` (6-criteria framework)

---

## Version History

| Version | Date | Key change | Ref aggregate |
|---------|------|-----------|---------------|
| v1 | 2026-04-10 | 7 signals from Phase A.0 | 0.362 |
| v2 | 2026-04-14 | Substance over rigor theater | 0.423 |
| v5 | 2026-04-14 | Reference plans + 6-criteria | 0.778 |
| v6 | 2026-04-15 | Removed anchor, full-paper refs | 0.769 |
| v7 | 2026-04-15 | Explicit COUNT→SCORE tables (7 signals) | 0.764 |
| v8 | 2026-04-15 | S4 GATE + S1/S9 calibration | 0.808 |
| **v8.1** | **2026-04-15** | **S4 GATE loosened + Occam weights** | **0.857** |

---

## Layer 0: Hard Gates (binary pass/fail)

| Gate | What it checks | Threshold |
|---|---|---|
| **G1: Goal-Contrast Margin** | Is the plan specific to THIS goal (not generic)? | margin ≥ 0.10 |
| **G2: Claim Verification** | Are factual claims real (not fabricated)? | fabrication ratio < 15% |

If either gate fails → aggregate reward = 0.

## Layer 1: Gradient Signals (1-5 integer scale)

### v8.1-minimal Weights

| # | ID | Name | Weight | Detection rate | Status |
|---|---|---|---|---|---|
| 1 | **S1_depth** | Reasoning Depth | **0.03** | 55% | Weak — extreme-case catch-all |
| 2 | **S2_rigor** | Evidence Rigor | **0.13** | 100% | Strong |
| 3 | **S3_positioning** | Positioning | **0.12** | 61% | Moderate |
| 4 | **S4_significance** | Significance | **0.00** | 0% (dropped) | Grading disabled |
| 5 | **S5_feasibility** | Feasibility Evaluability | **0.10** | 83% | Strong |
| 6 | **S6_risk_awareness** | Mature Risk Awareness | **0.13** | 88% | Strong |
| 7 | **S7_specificity** | Implementation Specificity | **0.12** | 93% | Strong |
| 8 | **S8_scope** | Scope-Generalization | **0.18** | 94% | Strong |
| 9 | **S9_focus** | Research Focus | **0.19** | 88% | Strong |

**Total weight**: 1.00 (S4 excluded)
**Detection rate**: % of targeted perturbations where signal drops ≥ 1 point.
**Why S4 = 0**: Qwen3-30B grader cannot reliably isolate the Problem section;
P_S4 perturbation only modifies Problem but stakes persist in Motivation/Core Idea.
S4 is still defined in code but `disabled_signals="S4_significance"` skips grading.

### Scoring

```
aggregate_reward = Σ (weight_i × normalize(score_i))
normalize(score) = (score - 1) / 4    # maps 1→0.0, 5→1.0
```

### Validation (v8.1-minimal, 60 refs + 162 perturbations)

- Ref aggregate: mean=0.857, range 0.693–0.970
- Pert aggregate: mean=0.773, range 0.480–0.993
- **Gap**: 0.084
- **AUC P(ref > pert)**: 0.767
- **Perts below ref P50**: 84%
- **Opus correlation**: r=0.038 (low — Opus ratings saturated at 84–89/100 on refs)
- **Inter-signal max |r|**: 0.33 (S5↔S7, no redundancy)

---

## Full Signal Specifications

Below is the complete prompt text for each signal. These are passed to the grader
as part of `build_single_signal_prompt()` in `ten_signal_reward.py`.

### Shared Preamble (all signals)

```
You are an expert-level research plan evaluator. Your job is to assess a research
plan on a specific, well-defined evaluation dimension.

RULES:
- Evaluate based on what is ACTUALLY in the plan, not what it claims about itself.
- Follow the rubric strictly. Each score 1-5 has specific criteria.
- Be skeptical of surface-level plausibility — check whether claims are supported
  by concrete content.
- Do not demand rigor-theater (long lists of statistical procedures, exhaustive
  failure modes) when the plan is concise and focused. Judge the substance of
  what's there.
- Concise methodology that delegates boilerplate to citations is acceptable if the
  core is clearly described.
- If a signal has "Required reasoning" scaffolding, you MUST do the reasoning
  BEFORE assigning a score.
- Integer scores only: 1, 2, 3, 4, or 5.
```

---

### S1_depth — Reasoning Depth (weight 0.03)

**Question**:
> For each major design choice in the plan, does the author explain WHY it was made? Depth means explicit derivation: problem → insight → each choice justified by reference to that insight. Shallow plans list techniques with citations but don't explain the logical connection.

**CoT Scaffolding (required reasoning before scoring)**:
```
Do EXPLICIT CLASSIFICATION with STRICT definitions:

STEP 1 — State the core problem (from Problem section) and KEY INSIGHT (from
Core Idea / Motivation) in one sentence each.

STEP 2 — Identify the TOP 3-5 LOAD-BEARING DESIGN CHOICES — the decisions that
DISTINGUISH this plan from a standard approach. A load-bearing design choice is:
  - a named ALGORITHMIC innovation (e.g., 'tree-guided evolution', 'soft parallel
    decoding', 'on-policy uniform training')
  - a novel TRAINING procedure (e.g., 'RL with self-correction reward')
  - a novel ARCHITECTURE feature (e.g., 'structure-aware 441-token vocab')
  - a novel DATA or OBJECTIVE choice (e.g., 'hybrid on-policy + SFT loss')

DO NOT count as design choices:
  - Hardware (GPUs, memory, compute)
  - Dataset names (unless the plan is about building a new dataset)
  - Standard optimizer/hyperparameter values (lr, batch size, temperature, seed)
    unless explicitly argued as novel
  - Evaluation benchmarks
  - Boilerplate tools (git, logging, monitoring)

IMPORTANT: Cap the list at 5 items. If you identify more than 5, pick the 5 MOST
central to the plan's contribution.

STEP 3 — For EACH of the 3-5 load-bearing choices, classify as:
  * JUSTIFIED: plan contains a sentence stating WHY this choice addresses the
    insight/problem (causal or mechanistic).
  * ASSERTED: plan names the choice but gives no 'why' sentence.

STEP 4 — Count (N_choices, N_asserted). Note: N_choices ∈ [3, 5].

STEP 5 — Apply this TABLE (pick row matching N_asserted):
  N_asserted = 0 → score 5
  N_asserted = 1 → score 4
  N_asserted = 2 → score 3
  N_asserted = 3 → score 2
  N_asserted ≥ 4 → score 1

STRICT RULE: 'inspired by', 'following', 'as in [cite]' are NEVER justifications.
But a citation PLUS a mechanistic sentence ('we use X because it provides property
Y that addresses Z') IS justified.
```

**Scoring Rubric**:
```
1: ≥4 load-bearing choices asserted (list of techniques, no argument)
2: 3 asserted
3: 2 asserted
4: 1 asserted (mostly justified)
5: 0 asserted — every load-bearing choice has explicit why-sentence
```

---

### S2_rigor — Evidence Rigor (weight 0.13)

**Question**:
> Can the proposed evidence support the claim? For EMPIRICAL work: are baselines FAIR (appropriate, modern, matched in scale — not strawman) and is there an OPERATIONALIZED success criterion (specific metric)? For THEORETICAL work: are assumptions stated and is a proof sketch/strategy given (not just 'we prove X')?

**CoT Scaffolding**:
```
Do EXPLICIT CLASSIFICATION (no interpretation):

STEP 1 — Identify if the work is EMPIRICAL or THEORETICAL.

STEP 2 (EMPIRICAL branch) — List every baseline in the Evaluation section. For
each baseline, classify as:
  * FAIR: a modern, appropriate-scale method from the same problem space.
  * STRAWMAN: obviously weak baseline unlikely to be competitive. Signs: 'random
    baseline', 'untrained model', 'hand-coded heuristic from 201X', a method 5+
    years old in a fast-moving field.
Count N_fair and N_strawman.

STEP 2 (THEORETICAL branch) — Mark each as present (1) or absent (0):
  * Explicit assumptions stated
  * Proof strategy / sketch given (not just 'we will prove')
  * Tight bounds discussed OR key lemma stated
Count T = sum (0-3).

STEP 3 — Check the SUCCESS CRITERION:
  * OPERATIONALIZED: named metric with benchmark ('pass@1 on MATH-500 above 0.5').
  * VAGUE: 'we expect improvement', 'our method should outperform'.

STEP 4 — Apply this TABLE:
  EMPIRICAL:
    N_strawman ≥ 2 → score 1 (or 2 if a fair baseline also present)
    N_strawman ≥ 1 AND criterion VAGUE → score 1
    N_strawman ≥ 1 AND N_fair ≤ 1 → score 2
    N_fair ≥ 2 AND criterion VAGUE → score 2
    N_fair ≥ 2 AND criterion OPERATIONALIZED → score 4
    N_fair ≥ 3 AND criterion OPERATIONALIZED AND targeted ablations → score 5
    Otherwise → score 3

  THEORETICAL:
    T = 0 or 1 → score 1-2
    T = 2 → score 3
    T = 3 + operationalized criterion → score 4-5
```

**Scoring Rubric**:
```
1: ≥2 strawman baselines, OR 1+ strawman + vague criterion
2: 1 strawman + 1 fair, OR fair baselines but vague criterion
3: 1 fair baseline with operationalized criterion, or theoretical T=2
4: ≥2 fair baselines + operationalized criterion
5: ≥3 fair baselines + operationalized criterion + targeted ablations
```

---

### S3_positioning — Positioning (weight 0.12)

**Question**:
> Does the plan name SPECIFIC prior methods and explain WHY each is insufficient for this problem? Positioning is about argument, not citation count, but it requires NAMED prior work — not vague references like 'prior work', 'existing methods', 'some approaches'.

**CoT Scaffolding**:
```
STEP 1 — List every NAMED prior method/work in the plan. A NAMED method has:
  (a) a proper-noun name (e.g., 'GEPA', 'TextGrad', 'AlphaEvolve'),
  (b) a specific citation (e.g., 'Huang 2024'),
  (c) a concrete descriptor ('DPR retrieval', 'LoRA fine-tuning').
DO NOT count: 'prior work', 'existing methods', 'recent work'.

STEP 2 — For EACH named method: does the plan state what is INSUFFICIENT about
it FOR THIS PROBLEM?
  * WITH_INSUFFICIENCY: states a specific limitation.
  * NAMED_ONLY: named but no insufficiency sentence.

STEP 3 — Count (N_named, N_with_insufficiency).

STEP 4 — Apply:
  N_named = 0 → score 1
  N_named = 1, N_with_insufficiency = 0 → score 2
  N_named = 1, N_with_insufficiency = 1 → score 3
  N_named ≥ 2, N_with_insufficiency ≤ 1 → score 3
  N_named ≥ 2, N_with_insufficiency ≥ 2 → score 4
  N_named ≥ 3, N_with_insufficiency ≥ 3 AND chain-of-reasoning → score 5
```

**Scoring Rubric**:
```
1: 0 named prior methods
2: 1 named without insufficiency
3: 1 with insufficiency, OR 2+ named but ≤1 have insufficiency
4: 2+ named, 2+ have insufficiency
5: 3+ named with full chain-of-reasoning about gap
```

---

### S4_significance — Significance (weight 0.00 — DISABLED)

**Status**: Grading skipped via `disabled_signals="S4_significance"`. Weight = 0.

**Why disabled**: Qwen3-30B grader integrates across sections despite "read ONLY Problem" instruction. Single-section perturbation (P_S4) produces 0% detection. The concept (stakes articulation) is valuable but not reliably measurable with current grader.

**Full rubric preserved in code** for future re-enablement if grader improves.

---

### S5_feasibility — Feasibility Evaluability (weight 0.10)

**Question**:
> Can a domain expert read this plan and judge whether it is implementable? Specifically: is the CORE ALGORITHM specified, are KEY DEPENDENCIES (datasets, models, tools) named, are CRITICAL parameters given?
>
> DO NOT penalize plans for omitting: cloud providers, exact hardware specs, cost estimates, batch sizes, parallelization strategies, or other deployment-level details.

**CoT Scaffolding**: None (qualitative rubric, not counting-based)

**Scoring Rubric**:
```
1: Not evaluable. Core algorithm vague, key dependencies unnamed.
2: Marginal. Core algorithm stated but 2+ key decisions hand-waved.
3: Adequate. Core algorithm + most dependencies, one critical gap.
4: Feasibility-judgeable. Core algorithm clear, dependencies named, critical
   parameters present.
5: Exceptionally clear. All critical choices specified AND briefly justified.
```

---

### S6_risk_awareness — Mature Risk Awareness (weight 0.13)

**Question**:
> Does the plan demonstrate mature awareness of its own boundaries, risks, and failure modes? A good plan names 2-3 SPECIFIC failure modes, identifies fragile assumptions, states scope boundaries. NOT about enumerating 10 failure modes (dilutive theater).

**CoT Scaffolding**:
```
1. Does the plan identify 2-3 specific ways it could fail?
2. Does it state which assumptions are most fragile (the 'key bet')?
3. Does it acknowledge scope boundaries (what it does NOT claim)?
4. Coherent risk structure, or just a laundry list?
```

**Scoring Rubric**:
```
1: No risk awareness. Plan assumes success.
2: Generic disclaimers ('further research needed').
3: 1 failure mode OR scope boundary, incomplete structure.
4: 2-3 specific failure modes + fragile assumptions + scope boundary.
5: Coherent risk model + specific modes + load-bearing assumptions + fallback.
```

---

### S7_specificity — Implementation Specificity (weight 0.12)

**Question**:
> Does the plan make SPECIFIC commitments, or does it rely on vague placeholders? Specificity = concrete named commitments. Vagueness = filler words: 'appropriately tuned', 'standard techniques', 'suitable baselines'.

**CoT Scaffolding**:
```
STEP 1 — Count VAGUE MARKERS (V) in Core Idea + Methodology + Evaluation:
  'appropriately', 'standard', 'typical', 'suitable', 'reasonable',
  'we will tune/explore/investigate' (without search space),
  'various/several/many' (without enumeration), 'TBD',
  'state-of-the-art' as substitute for a named method.

STEP 2 — Count SPECIFIC COMMITMENTS (C):
  named model, named dataset, numeric hyperparameter, named loss/objective,
  named metric, named theorem/lemma.

STEP 3 — Apply:
  V ≥ 6 → score 1
  V ≥ 3 and C ≤ 3 → score 2
  V ≥ 3 and C ≥ 4 → score 3
  V ≤ 2 and C ≥ 5 → score 4
  V ≤ 1 and C ≥ 8 → score 5
  Otherwise → score 3
```

---

### S8_scope — Scope-Generalization Alignment (weight 0.18)

**Question**:
> Does the plan's evaluation COVER what the plan claims? This is NOT about breadth. A narrow claim with narrow matching evaluation is a HIGH SCORE (4-5).

**CoT Scaffolding**:
```
STEP 1 — COUNT OVERCLAIM TRIGGER WORDS (OC) in Problem + Motivation + Core Idea:
  'universal', 'general', 'any task/domain/field',
  'all domains/fields/scientific', 'across domains/fields/all',
  'every discipline/field', 'foundation/foundational' (as claim).

STEP 2 — COUNT EVALUATION DOMAIN BUCKETS (D):
  Group benchmarks: GSM8K+MATH=1 bucket (math), HumanEval+MBPP=1 (code), etc.

STEP 3 — Apply:
  OC ≥ 4 AND D ≤ 2 → score 1
  OC ≥ 2 AND D ≤ 2 → score 2
  OC ≥ 2 AND D ≥ 3 → score 3
  OC ≤ 1 AND D = 1 → score 3
  OC ≤ 1 AND D ≥ 2 (specific problem) → score 4
  OC = 0 AND eval exactly matches claim → score 5
```

---

### S9_focus — Research Focus (weight 0.19)

**Question**:
> Does the plan STAY focused on a small number of core techniques, or does it STACK unrelated methods without justifying each? Focused = 1-3 core techniques that compose. Stacked = many techniques listed without explaining necessity.

**CoT Scaffolding**:
```
STEP 1 — Identify CORE IDEA techniques (read Core Idea section only).
List METHODOLOGICAL techniques fundamental to the approach.

DO NOT count: infrastructure, datasets, optimizers, evaluation metrics,
citations for positioning. Cap at 8 items.

STEP 2 — For EACH technique:
  * JUSTIFIED: explains WHY needed for the core insight.
  * UNJUSTIFIED: named but no explanation.

STEP 3 — Count (N_total, N_unjustified).

STEP 4 — Apply:
  N_total ≤ 3 AND N_unjustified = 0 → score 5
  N_total ≤ 3 AND N_unjustified ≥ 1 → score 4
  N_total ∈ [4,6] AND N_unjustified ≤ 1 → score 4
  N_total ∈ [4,6] AND N_unjustified ∈ [2,3] → score 3
  N_total ∈ [4,6] AND N_unjustified ≥ 4 → score 2
  N_total ≥ 7 AND N_unjustified ≤ 2 → score 3
  N_total ≥ 7 AND N_unjustified ≥ 3 → score 2
  N_total ≥ 7 AND N_unjustified ≥ 5 → score 1
```

---

## Design Principles

1. **Explicit counting over qualitative judgment**: 7 of 8 active signals use COUNT → SCORE decision tables, preventing grader rationalization.
2. **Evaluate substance, not form**: Don't reward boilerplate; reward actual evidence/logic.
3. **Anti-Goodhart** (S9): Penalizes technique-stacking, the most common LLM generation failure mode.
4. **Occam-minimal weights**: Dropped S4 (unmeasurable), downweighted S1 (weak detection). Concentrated weight on 7 reliable signals.
5. **No preamble anchoring**: Shared preamble is neutral; no "default to 4" instruction.
6. **Section-specific counting**: S8 counts overclaim words, S7 counts vague markers — lexical evidence that directly reflects quality.

## Known Limitations

1. **S4 (significance) disabled** — Cannot reliably measure "stakes articulation" with current grader. Plan uses a holistic reading approach, can't isolate sections.
2. **S1 detection weak (55%)** — Top-5 load-bearing restriction makes it less sensitive to "remove all justifications" perturbation.
3. **Opus correlation low (r=0.038)** — Opus ratings saturated on refs (84-89/100). Structural signals are more discriminative than holistic human judgment on already-good plans.
4. **S7/S8 partially lexical** — High detection rates partly from perturbation-rubric lexical overlap (perturbation adds words that rubric specifically counts).
