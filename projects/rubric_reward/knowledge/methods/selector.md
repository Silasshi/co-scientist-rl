# Method: CPR & Self-Selector (Pairwise Plan Ranking)

## Motivation

The gap analysis was the turning point. After 60+ runs trying to improve generation quality, the numbers told a different story: the model already generates excellent plans. Oracle best-of-8 scores 0.85, just 0.01 below reference solutions (0.86). The real problem is that the model cannot tell which of its outputs are good. The selection gap (mean 0.70 to oracle 0.85) accounts for **91% of the total performance gap**, while the capability gap (oracle 0.85 to reference 0.86) is only 9%.

Self-evaluation makes this concrete: the model's self-assessed quality correlates with rubric score at r ~ 0.061 -- essentially random. When the model picks its "best" plan from a set of 8, it performs no better than random selection.

The selector approach targets this 91% gap directly. Rather than training the model to generate better plans (diminishing returns), train it to compare two plans and pick the better one. A selector that can reliably identify the best plan from 8 candidates would jump performance from 0.70 to approximately 0.85, far exceeding anything achievable through generation training alone.

Two iterations of this idea were developed: CPR (minimal, preliminary) and Self-Selector (structured, full-featured).

---

## Pipeline

### CPR: Contrastive Plan Ranking (`src/co_scientist/trainers/grpo/train_cpr.py`)

The first, minimal attempt at pairwise ranking.

```
For each batch of 32 preference pairs:
  1. Build pairwise prompt: goal + Plan A + Plan B + "The better plan is: Plan"
  2. Generate 8 responses per pair (temperature=1.5, max_tokens=32)
  3. Extract choice: parse "A" or "B" from response
  4. Reward: 1.0 if correct, 0.0 if wrong
  5. GRPO update over 8 samples per pair
```

**Key CPR design choices**:
- **Minimal prompt**: "Answer with just A or B." No reasoning, no quality dimensions.
- **Very short generation**: max_tokens=32. Only needs to output a letter.
- **High temperature**: 1.5, because binary choice collapses GRPO diversity at lower temperatures.
- **Binary reward**: Correct = 1.0, wrong = 0.0. No partial credit.

### Self-Selector: Improved CPR (`src/co_scientist/trainers/selector/train_selector.py`)

A substantially improved version addressing CPR's limitations.

```
For each batch of 32 preference pairs:
  1. Build structured prompt: goal + Plan A + Plan B + 7 quality dimensions
  2. Generate 8 responses per pair (temperature=1.0, max_tokens=512)
  3. Model reasons through quality dimensions, then outputs "VERDICT: A" or "VERDICT: B"
  4. Extract verdict (multi-pattern fallback: VERDICT line, "Plan X is better", last A/B)
  5. Reward: 1.0 if correct, 0.0 if wrong
  6. GRPO update over 8 samples per pair
  7. Every 50 batches: validation on held-out pairs (deterministic, temp=0.0)
```

**Key improvements over CPR**:

| Feature | CPR | Self-Selector |
|---------|-----|---------------|
| Prompt | Minimal ("just A or B") | Structured with 7 quality dimensions |
| Reasoning | None (max_tokens=32) | Chain-of-thought (max_tokens=512) |
| Temperature | 1.5 (forced diversity) | 1.0 (natural diversity from reasoning) |
| Validation | None | Every 50 batches on held-out pairs |
| Position bias tracking | No | Yes |

**The 7 quality dimensions** (generic, NOT rubric items):
1. Thoroughness
2. Specificity
3. Soundness
4. Justification
5. Efficiency
6. Ethics
7. Coherence

These dimensions mirror the rubric's 7 desiderata but are described generically. The model never sees the actual rubric items or the reference solution.

### Training Data

Preference pairs are extracted from bestversion training logs using `tools/extract_cpr_pairs.py`.

```
Source: runs/2026/2/withA1,A2/2(ml)/train/training_logs.jsonl
Method:
  1. Reconstruct (goal, plan, rubric_score) triples from training logs
  2. Align with dataset goals by batch index
  3. For each goal: create C(8,2) = 28 pairwise comparisons from the 8 samples
  4. Filter: keep only pairs with score_gap >= 0.05
  5. Split by goal (not by pair) to prevent leakage

Output: ~111K preference pairs
  - Train: data/cpr_pairs_train.jsonl
  - Validation: data/cpr_pairs_val.jsonl
```

### Eval Pipeline (`src/co_scientist/trainers/eval/eval_selector.py`)

The evaluation pipeline tests the full end-to-end selector workflow:

```
For each of ~685 test goals:
  1. Generate N=8 plans using bestversion policy checkpoint
  2. Grade all 8 plans with standard grader -> rubric scores
  3. Run pairwise selector tournament: C(8,2) = 28 pairs
     - With position debiasing: 56 comparisons (both orderings per pair)
     - If orderings disagree: tie (0.5 wins each)
  4. Aggregate wins -> select winner (most wins)

Reports:
  - mean-of-N rubric (baseline: 0.70)
  - oracle best-of-N rubric (ceiling: 0.85)
  - selector-selected rubric (what the selector picks)
  - gap_closed%: (selector - mean) / (oracle - mean)
  - rank correlation (Spearman) between selector ranking and rubric ranking
  - position bias metrics
  - oracle hit rate (how often selector picks the actual best plan)
```

---

## Runs & Results

### CPR

| Run | Batches | Status | Notes |
|-----|---------|--------|-------|
| `4/cpr/1` | 12 | Preliminary | Insufficient training for evaluation |

Only 12 batches completed -- far too few to assess viability. CPR served primarily as a proof-of-concept that informed the self-selector design.

### Self-Selector

**Status**: Implemented but not yet trained.

The selector training script, evaluation script, data extraction pipeline, and position-debiased evaluation are all complete. The ~111K training pairs have been extracted. Training has not yet been launched.

---

## Current Status

The self-selector is the **most promising remaining direction** based on convergent evidence from multiple analyses:

1. **The selection gap is where the opportunity is.** 91% of the total performance gap is selection, not generation. See [Selection Gap](../findings/selection_gap.md).

2. **Self-evaluation is random.** The model's prompted self-assessment (r ~ 0.061) provides almost zero signal. A trained comparator should dramatically outperform prompted evaluation.

3. **Selection is orthogonal to generation.** A trained selector can be combined with any generator (bestversion, a more capable future model, etc.) for additive gains.

4. **Even a mediocre selector would be transformative.** The best-of-N scaling curve shows that reliably identifying the top-2 from 8 candidates would boost performance from 0.70 to ~0.82 -- far exceeding the best generation training result (0.693 eval rubric).

---

## What This Led To

CPR was the preliminary prototype. Its minimal prompt and 32-token output made it fast to iterate but provided no reasoning signal. The self-selector replaced it with structured comparison, chain-of-thought reasoning, and validation -- preserving the pairwise GRPO framework while adding the features needed for a robust ranking model.

The self-selector represents the project's strategic pivot from generation improvement (80+ runs, no gain beyond bestversion) to selection improvement (targeting the 91% gap that generation training cannot reach).

---

## Lessons for Future Work

1. **Pairwise comparison is easier than absolute scoring.** Asking "which plan is better?" is a simpler task than "how good is this plan on a 0-3 scale across 7 criteria?" The model may succeed at the comparative task where it fails at absolute evaluation.

2. **Position debiasing is worth the 2x cost.** LLMs exhibit strong position bias in A/B comparisons. Testing both orderings and treating disagreements as ties is more robust than naive single-ordering evaluation.

3. **Training data should come from the policy itself.** Using bestversion's own training logs ensures the preference pairs cover the actual distribution of plans the selector will encounter at inference time.

### Open Questions

- **DPO vs GRPO**: Would DPO (Direct Preference Optimization) work better than GRPO for the binary ranking task? DPO directly optimizes from preference pairs without reward modeling, which may be a more natural fit.
- **How many epochs**: The 111K pairs provide substantial data. Will a single epoch suffice, or does the selector need multiple passes?
- **Will position debiasing help in training?** Currently planned only for evaluation. Training with position-debiased pairs (both orderings) might produce a more robust selector.
- **Score gap threshold**: The current threshold of 0.05 creates many borderline pairs. Would a higher threshold (0.10, 0.15) produce cleaner signal?
- **Generalization to new models**: If the selector is trained on Qwen3-30B-A3B plans, will it generalize to plans from other generators?

---

## Implementation Files

| File | Purpose |
|------|---------|
| `src/co_scientist/trainers/grpo/train_cpr.py` | CPR training (minimal pairwise ranker) |
| `src/co_scientist/trainers/selector/train_selector.py` | Self-selector training (structured pairwise ranker) |
| `src/co_scientist/trainers/eval/eval_selector.py` | Self-selector evaluation (full tournament pipeline) |
| `src/co_scientist/trainers/eval/eval_cpr.py` | CPR evaluation |
| `tools/extract_cpr_pairs.py` | Preference pair extraction from training logs |

## Run Paths

- CPR: `runs/2026/4/cpr/1/`
- Self-selector: `runs/2026/4/selector/1/` (planned)

## Related

- [Selection Gap](../findings/selection_gap.md) -- the finding that motivated this entire direction
- [Negative Results](../findings/negative_results.md) -- self-calibration failure confirms self-eval is random
- [IBT & Self-Calibration](ibt.md) -- self-calibration's failure to leverage self-evaluation
- [What Works](../findings/what_works.md) -- bestversion as the generator baseline for selector eval
- [Back to Catalog](../EXPERIMENT_CATALOG.md)
