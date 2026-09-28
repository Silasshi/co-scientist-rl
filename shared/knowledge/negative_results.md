# Finding: Comprehensive Catalog of Negative Results

## Summary

Over 80 runs across 4 months tested 10+ distinct approaches. None exceeded the bestversion baseline (eval rubric 0.693). This document catalogs each negative result with its specific failure mode.

---

## Negative Results Table

| Method | Runs | Best Eval | vs Bestversion | Failure Mode |
|--------|------|-----------|----------------|--------------|
| SDPO | 20 | -- | Did not outperform | Self-distillation adds no value |
| Doublegeneration | 1 (178 batches) | 0.654 | -0.039 | Stage 2 OOD collapse |
| Rubric dropout | 1 (106 batches) | 0.657 | -0.036 | Split training signal |
| Multi-turn V1 | 1 (54 batches) | -- | No learning | Binary PRM too coarse |
| Multi-turn V2 | 1 (72 batches) | -- | No learning | 71% empty responses |
| Multi-turn V3 | 1 (3 batches) | -- | No learning | PRM parsing bug |
| Multi-turn V4 | 5 runs | -- | No learning | Ternary PRM insufficient |
| Think-solution | 1 (106 batches) | -- | Thinking hurts (-0.090) | Process shortcutting |
| Self-calibration | -- | -- | No improvement | Uniform 1.08x weight |
| Cross-model OPD (30B->4B) | -- | -- | Destroys small model | Distribution mismatch |

---

## Detailed Failure Analysis

### 1. SDPO (Self-Distillation Policy Optimization)

**What it is**: Adds a self-distillation loss to GRPO. The model learns from both the reward signal and from its own best previous outputs.

**Configuration space**: Lambda 0.1--0.45, 20 runs including recovery experiments warmstarted from bestversion batch 214.

**Why it failed**: Self-distillation does not provide additional useful signal beyond GRPO's own advantage estimation. The distillation targets (the model's own best outputs) are already captured by the GRPO advantage. Adding the distillation loss just adds noise to the gradient.

**Best run**: `28_recovery_A_warmstart` (244 batches, 0.673 reward) -- still below bestversion.

---

### 2. Doublegeneration (Two-Stage Refinement)

**What it is**: Stage 1 generates initial plan, Stage 2 refines it using grader feedback. Compute-matched: 4+4 samples = 8 total.

**Why it failed**: Distribution shift. As Stage 1 improves during training, Stage 2's inputs change. Stage 2's refinement strategy was learned on weaker plans and becomes out-of-distribution when applied to better plans. After batch ~130, Stage 2 actively hurts 28% of samples.

**Key numbers**:
- Peak refined reward: 0.66 at batch ~130
- Final refined reward: 0.47 at batch 178
- Refinement delta collapse: 0.27 -> 0.13
- Regression rate: 10% -> 28%

**Eval**: 0.654 (vs bestversion 0.693)

---

### 3. Rubric Dropout

**What it is**: During training, randomly hide some rubric criteria from the model. Hypothesis: this forces the model to learn to infer hidden criteria, improving generalization.

**Why it failed**: The model needs all criteria visible to optimize effectively. Hiding criteria reduces the training signal available per sample. The transfer learning from visible to hidden criteria is too weak to compensate for the lost signal.

**Eval**: 0.657 (vs bestversion 0.693)

---

### 4. Multi-turn Discussion (V1--V4)

**What it is**: The model engages in a multi-turn research discussion (with different roles: Collaborator, Researcher, Evaluator) before producing a final plan.

**Why it failed (all versions)**:
- **V1**: Binary PRM ({0,1}) too coarse -- no useful gradient signal for discussion quality.
- **V2**: Merged evaluator role -- model learned to produce empty responses (71% empty rate), completely shortcutting the discussion.
- **V3**: PRM parsing bug -- all negative scores mapped to 0, destroying the reward signal.
- **V4**: Wave-based + ternary PRM ({-1,0,+1}) -- even with fixes, discussion quality showed no correlation with final plan quality (Pearson r = -0.115 to 0.052).

**Fundamental issue**: Discussion is an unrewarded intermediate process. The model either shortcuts it or treats it as noise. See [Process Shortcutting](process_shortcutting.md).

---

### 5. Think-Solution

**What it is**: Separate `<think>` and `<solution>` blocks with independent grading.

**Why it failed**: Think blocks score -0.090 lower than non-thinking plans. Reasoning consumes token budget without improving the plan. The model cannot leverage explicit reasoning to produce better research plans -- or at least, the current reward function doesn't capture the benefit.

**Root cause**: Process shortcutting -- the think process is not rewarded, so it gets eliminated or becomes detrimental.

---

### 6. Self-Calibration

**What it is**: Have the model self-evaluate its outputs and use the self-evaluation to weight the training signal.

**Why it failed**: Self-evaluation produces a nearly uniform 1.08x weight across all samples. The model cannot meaningfully distinguish its better outputs from worse ones (self-eval correlation with rubric: r ~ 0.061). Since the weights are essentially uniform, self-calibration has no effect on training.

---

### 7. Cross-Model OPD (30B -> 4B)

**What it is**: Online Policy Distillation -- use the trained 30B model to teach a smaller 4B model.

**Why it failed**: Distribution mismatch between the 30B model's outputs and the 4B model's capacity. The 4B model cannot reproduce the 30B model's reasoning patterns. The distillation signal overwhelms the smaller model's own learning, causing degradation rather than improvement.

---

## Patterns Across Negative Results

### 1. Adding Complexity Hurts

Every method that added training complexity (stages, processes, roles, distillation) performed worse than simple single-stage GRPO. The additional complexity introduces new failure modes (OOD, shortcutting, parsing bugs) without providing compensating benefits.

### 2. Process Steps Get Eliminated

Any intermediate step not directly tied to the reward signal is minimized or eliminated by the optimizer. This affected think-solution, multi-turn, IBT, and doublegeneration Stage 2.

### 3. Self-Assessment Is Unreliable

The model cannot reliably assess its own output quality (r ~ 0.061 with rubric). This undermines self-calibration, self-selection, and any method that relies on the model's own quality judgment.

### 4. The Problem Is Well-Specified But Hard

The bestversion ceiling (0.693) is robust across many attack angles. The remaining performance gap is primarily in selection (91%), not generation (9%). Training methods that improve generation hit diminishing returns quickly.

---

## Lessons for Future Work

1. **Do not add unrewarded processes.** They will be shortcutted.
2. **Do not rely on self-evaluation.** The model cannot judge its own quality.
3. **Do not add training stages that depend on other stages.** Distribution shift causes collapse.
4. **Focus on selection, not generation.** The 91% selection gap is where the opportunity is.
5. **Consider inference-time interventions** over training-time complexity.

## Related Findings

- [Process Shortcutting](process_shortcutting.md) -- the universal failure mode
- [Reward Misalignment](reward_misalignment.md) -- the reward function contributes to failures
- [Selection Gap](selection_gap.md) -- where the opportunity actually is
- [What Works](what_works.md) -- what succeeded despite everything
- [Back to Catalog](../EXPERIMENT_CATALOG.md)
