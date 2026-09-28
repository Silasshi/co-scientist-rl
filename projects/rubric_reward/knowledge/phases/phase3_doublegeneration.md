# Phase 3: Doublegeneration, Rubric Dropout & Multi-turn (March 2026)

## Overview

17 runs testing advanced training approaches: two-stage refinement (doublegeneration), stochastic rubric visibility (rubric dropout), multi-turn discussion (V1--V4), and blended generation. All failed to beat the bestversion baseline (0.693 eval rubric).

---

## Doublegeneration (refinement/8)

### Architecture

Two-stage async GRPO with compute-matched sampling (4 initial + 4 refined = 8 total samples/goal):

1. **Stage 1**: Generate initial plan, grade with simple grader
2. **Feedback extraction**: Use detailed grader (XML) to produce structured feedback
3. **Stage 2**: Refine plan using feedback, grade refined plan with simple grader
4. Independent GRPO advantage normalization per stage, combined in one optimizer step

### Key Metrics

| Metric | Value |
|--------|-------|
| Batches | 178 |
| Checkpoints | 37 |
| Avg initial rubric | 0.324 |
| Avg refined rubric | 0.549 |
| Refinement delta | 0.225 |
| Improvement rate | 83--89% |
| Format compliance | 86.81% |
| Total samples generated | 91,648 |
| Valid samples | 87,404 (95.4%) |
| **Eval rubric** | **0.654** |

### Collapse Pattern

The critical failure mode of doublegeneration:

| Epoch | Refined Reward | Refinement Delta | Regression Rate |
|-------|----------------|-------------------|-----------------|
| Early (batches 1--50) | Rising | 0.27 | ~10% |
| Peak (batches 130--139) | 0.66 | -- | -- |
| Late (batches 150--178) | Falls to 0.47 | 0.13 | 28% |

**Root cause**: As Stage 1 improves during training, the distribution of inputs to Stage 2 shifts. Stage 2's learned refinement strategy was trained on weaker initial plans and becomes out-of-distribution when applied to better plans. This causes advantage collapse, noisy gradients, and degradation.

Additionally:
- Feedback bullets trend down (21.7 -> 20.6) as the grader gives less actionable feedback on already-good plans
- Stage 2 actively hurts 28% of samples by the end of training

### Result

**Negative.** Eval 0.654 vs bestversion 0.693 (-0.039). Two-stage refinement is fundamentally unstable when Stage 1 improves.

**Run path**: `runs/2026/3/refinement/8/`

**Implementation**: `src/co_scientist/trainers/refinement/train_double_generation.py` (2061 lines)

---

## Rubric Dropout (rubric_dropout/13)

### Approach

During training, stochastically hide some rubric criteria from the model. The hypothesis: forcing the model to infer hidden criteria would improve generalization to unseen criteria at eval time.

### Key Metrics

| Metric | Value |
|--------|-------|
| Batches | 106 |
| Checkpoints | 19 |
| **Eval rubric** | **0.657** |

### Result

**Negative.** Eval 0.657 vs bestversion 0.693 (-0.036). Splitting the training signal between visible and hidden criteria hurt more than criteria-awareness transfer helped. The model needs to see all criteria to optimize effectively.

**Run path**: `runs/2026/3/rubric_dropout/13/`

---

## Multi-turn V1--V4

Four versions of a multi-turn discussion approach where the model engages in a collaborative research discussion before producing a final plan.

### Version History

| Version | Run | Batches | Architecture | Failure Mode |
|---------|-----|---------|--------------|--------------|
| V1 | run4 | 54 | Binary PRM | No learning. Binary signal too coarse. |
| V2 | run1 | 72 | Merged evaluator | 71% empty responses. Merged student+PRM role failed. |
| V3 | run1 | 3 | Fixed PRM | Parsing bug: all negative scores mapped to 0. |
| V4 | 5 runs | ~0 | Wave-based + ternary PRM | No consistent progress across all 5 runs. |

### V4 Design Details

- Wave-based training with max_turns=15
- Ternary PRM scores: {-1, 0, +1}
- Few-shot examples for format compliance
- Robust parsing with fallbacks
- Despite all fixes, no learning signal emerged

### Multi-turn Analysis Results

Correlation between discussion quality and final rubric score:
- Pearson r = -0.115 to 0.052 across versions
- Multi-turn discussion provides **no measurable benefit** to plan quality

### Result

**All negative.** Multi-turn discussion does not help research plan generation. The model cannot use discussion to improve its output -- it either shortcuts the discussion or the discussion adds noise.

**Design docs**: `paper/method_pipeline*.md`

**Implementation**: `src/co_scientist/trainers/multiturn/train_multiturn_v*.py`

---

## Blended Generation

| Run | Batches | Notes |
|-----|---------|-------|
| run 1 | 4 | Too few batches for conclusions |
| run 18 | 22 | Still too early to evaluate |

**Result**: Inconclusive due to insufficient training.

---

## Phase Summary

| Method | Eval Rubric | vs Bestversion (0.693) | Verdict |
|--------|-------------|------------------------|---------|
| Doublegeneration | 0.654 | -0.039 | Negative (Stage 2 OOD collapse) |
| Rubric dropout | 0.657 | -0.036 | Negative (split signal) |
| Multi-turn V1--V4 | -- | No learning | Negative (discussion unhelpful) |
| Blended generation | -- | -- | Inconclusive |

Every advanced technique tested in Phase 3 either performed worse than bestversion or failed to learn at all. This led to a fundamental reassessment of the problem: the issue is not training method but the selection gap and reward misalignment.

See: [Phase 2](phase2_bestversion.md) | [Phase 4](phase4_advanced.md) | [Back to Catalog](../EXPERIMENT_CATALOG.md)
