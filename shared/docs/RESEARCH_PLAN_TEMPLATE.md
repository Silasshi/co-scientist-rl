# Research Plan Template (6-Criteria Framework)

**Purpose**: Defines what makes a complete research plan. Used to construct reference plans for signal validation and as the target specification for generated plans.

**Principle**: A good research plan makes the reader confident that (a) the problem is worth solving, (b) the approach is sound, and (c) the results will be informative.

---

## The 6 Criteria

A complete research plan must address these six dimensions. Each item below lists what the plan **must contain** and what it **must avoid**.

### 1. Clear, Sharp Problem Statement

**Must contain**:
- The specific problem being solved (not a vague research area)
- Why this problem is important (what becomes possible or what misunderstanding gets corrected)
- The concrete gap relative to existing work
- What would change in the field/practice if the plan succeeds

**Must avoid**:
- Generic research-area statements ("we study LLM improvements")
- Claims without specifying the contribution axis
- Motivation that could apply to any adjacent problem

---

### 2. Strong Motivation (Positioning Serves the Problem)

**Must contain**:
- Named prior work that is most relevant (2-5 specific methods, not vague "existing work")
- Specific reasons those prior approaches are insufficient for this problem
- The insight that suggests this approach has a chance to succeed

**Must avoid**:
- Long literature surveys that don't serve problem localization
- Vague "existing methods fail" without naming them
- Claiming novelty without grounding in the specific prior art

**Calibration**: 2-3 precise citations with a clear argument for "why this gap exists" is better than 10 citations dropped in passing.

---

### 3. Credible Core Hypothesis / Mechanism Intuition

**Must contain**:
- The core idea stated in one sentence
- Why this idea plausibly solves the problem (mechanism intuition)
- One of: (a) a preliminary argument, (b) mathematical grounding, (c) initial evidence

**Must avoid**:
- A list of techniques without an organizing hypothesis
- Claiming the approach will work without explaining why
- Stacking many techniques without identifying the "key bet"

---

### 4. Sufficient Methodology

**Must contain**:
- The main algorithm / pipeline / procedure with enough detail to judge feasibility
- Key hyperparameters or parameters (at methodology level — not full implementation)
- Reasoning for each major design choice (why this, not that)
- Clear identification of key assumptions and dependencies

**Must avoid**:
- Methodology as a pure operation list ("do A, then B, then C") without reasoning
- Over-detailed implementation specs (belongs in appendix, not plan)
- Vague hand-waving ("we tune hyperparameters appropriately")

**Calibration**: Detailed enough that a domain expert could judge the approach's feasibility and logic; not so detailed that it becomes an implementation manual.

---

### 5. Clear Evaluation Logic

**Must contain**:
- Specific metrics and success criteria (what counts as success)
- What results would distinguish this method from alternatives
- Concrete baseline(s) chosen to be fair (not strawmen)
- How failure would be interpretable (what alternative explanations are ruled out)

**Must avoid**:
- "We will evaluate on benchmark X" without specifying what counts as success
- Metrics that can't discriminate between the hypothesis being right or wrong
- Baselines picked because they're easy to beat

**Calibration**: The plan should make clear: "if we see result R, we have learned L". A plan that can't lose is not a plan.

---

### 6. Mature Risk Awareness (Boundary Conditions, Not Self-Negation)

**Must contain**:
- 2-3 specific failure modes this plan could hit
- Which assumptions are most fragile
- Resource/time constraints that matter
- What the plan does NOT claim (scope boundary)
- Fallback direction if the main bet fails

**Must avoid**:
- Generic "there might be limitations" disclaimers
- Exhaustive enumeration of every conceivable failure (dilutes signal)
- Blind optimism that assumes success

**Calibration**: Demonstrates the author has thought about what could go wrong and has a coherent view of the risk structure.

---

## Template Structure

A research plan following this framework has the following sections:

```markdown
## Problem
[Sharp problem statement, importance, gap, what changes if successful — Criterion 1]

## Motivation
[Named prior methods, why they fall short, why this approach has promise — Criterion 2]

## Core Idea
[Central hypothesis in one sentence, mechanism intuition, plausibility argument — Criterion 3]

## Methodology
[Algorithm/pipeline with design-choice reasoning; key parameters at methodology level — Criterion 4]

## Evaluation
[Metrics, success/failure criteria, fair baselines, what results would show — Criterion 5]

## Risk & Boundary
[Specific failure modes, fragile assumptions, scope boundaries, fallback — Criterion 6]
```

---

## Length Guidance

- **Target length**: 1500-3500 words (method section + evaluation + risk)
- **Too short** (< 800 words): Likely missing at least one criterion
- **Too long** (> 5000 words): Likely padding or over-specifying implementation

---

## Self-Check Questions

After writing a plan, ask:

1. Can I state the problem and its importance in 2 sentences?
2. Did I name specific prior methods (≥2) and their specific shortcoming?
3. Can I state the core idea in one sentence?
4. For each design choice, can I explain WHY it was chosen?
5. What specific result would tell me the approach worked vs failed?
6. What are the 2 most likely ways this plan fails, and how would I recognize each?

If any answer is unclear or missing, the plan is incomplete on that criterion.

---

## Use in This Project

- **Reference plans** in `projects/ttt_discover/analysis/signal_validity/data/refs/references_v2.jsonl` must satisfy all 6 criteria
- **Signal rubrics** in `src/co_scientist/shared/ten_signal_reward.py` evaluate these 6 criteria
- **Training target**: generated plans should approach this template structure
