# Method: Doublegeneration (Two-Stage Refinement) & Related Approaches

## Motivation

After bestversion reached an eval rubric ceiling of 0.693, the hypothesis was that a two-stage pipeline -- generate, then refine using grader feedback -- could break through. The reasoning: if the grader tells the model what is weak in its plan, a second generation conditioned on that feedback should fix those weaknesses. The approach was compute-matched to bestversion (4 initial + 4 refined = 8 total samples per goal) so any improvement would be attributable to the refinement mechanism, not additional compute.

Two related methods explored variants of this idea:
- **Blended generation**: Relabel Stage 2 outputs as Stage 1 to teach the model to produce refinement-quality plans directly.
- **Rubric dropout**: Stochastic rubric visibility to teach criteria awareness without changing the pipeline architecture.

---

## Pipeline

### Doublegeneration (`src/co_scientist/trainers/refinement/train_double_generation.py`, 2061 lines)

```
For each batch of 64 goals:

  Stage 1 -- Initial Generation:
    1. Generate 4 plans per goal (async, temp=1.0, max_tokens=2048)
    2. Grade each with DETAILED grader (structured XML with weaknesses/fixes)
    3. Extract feedback: weaknesses, fixes, desiderata profiles
    4. Compute Stage 1 GRPO advantages (within 4 initial samples)

  Stage 2 -- Refinement:
    5. For each initial plan: build refinement prompt (original plan + feedback)
    6. Generate 4 refined plans per initial plan (async)
    7. Grade refined plans with detailed grader
    8. Compute Stage 2 GRPO advantages (within 4 refined samples)

  Combined Optimization:
    9. Merge datums from both stages (weighted equally)
    10. Single forward_backward + optim_step
```

Key design choices:
- **Compute-matched**: 4+4 = 8 total samples per goal, same as bestversion's 8.
- **Independent advantages**: Stage 1 and Stage 2 each have their own GRPO advantage groups. This prevents Stage 2's higher absolute scores from dominating.
- **Detailed grader for feedback only**: The `build_detailed_grader_prompt` (structured XML) was used to extract actionable feedback (weaknesses, fixes). The `build_grader_prompt` (simple) was used for scoring. These two prompts produce incomparable score scales -- the detailed grader starts at ~0.24 rubric while the simple grader starts at ~0.61.
- **Feedback extraction**: Regex parsing of grader XML to pull out per-criterion weaknesses and fixes, with fallback handling for malformed XML.

### Blended Generation (`src/co_scientist/trainers/refinement/train_blended_generation.py`)

Same pipeline as doublegeneration, but Stage 2 outputs are relabeled with the Stage 1 prompt. The idea: if a refined plan is good, teach the model to produce it directly without needing the refinement step. This would avoid the distribution shift problem by collapsing both stages into one.

### Rubric Dropout (`src/co_scientist/trainers/refinement/best_ver_rubric_dropout.py`)

Same single-stage pipeline as bestversion, but with stochastic rubric visibility:

```
For each goal in the batch:
  1. Coin flip with probability p --> decide rubric visibility
  2. If visible: inject rubric items in prompt, think block reviews criteria
     If hidden: think block infers criteria autonomously
  3. Generate 8 plans, grade all (grader always sees rubric)

Adaptive scheduler:
  - EMA of hidden reward (alpha=0.1)
  - p = p_min + (p_max - p_min) * (target - hidden_ema) / (target - baseline_hidden)
  - p starts at 0.80, decreases as hidden reward improves
  - p_min = 0.05, p_max = 0.80, target = 0.86 (reference score)
```

---

## Runs & Results

### Doublegeneration (refinement/8)

| Metric | Value |
|--------|-------|
| Run path | `runs/2026/3/refinement/8/` |
| Batches | 178 (epoch 2) |
| Checkpoints | 37 |
| Avg initial rubric | 0.324 |
| Avg refined rubric | 0.549 |
| Refinement delta | 0.225 |
| Improvement rate | 83--89% |
| Format compliance | 86.81% |
| Total samples | 91,648 generated, 87,404 valid (95.4%) |
| **Eval rubric** | **0.654** (vs bestversion 0.693) |

Earlier runs: `runs/2026/3/refinement/7/` (2 batches, debug run) and `7(2)` (48 batches, short run).

### Rubric Dropout (rubric_dropout/13)

| Metric | Value |
|--------|-------|
| Run path | `runs/2026/3/rubric_dropout/13/` |
| Batches | 106 (1 epoch) |
| Checkpoints | 19 |
| hidden_ema | 0.546 --> 0.729 |
| visible_ema | 0.824 --> 0.955 |
| p (visibility probability) | 0.800 --> 0.363 |
| **Eval rubric** | **0.657** (vs bestversion 0.693) |

### Blended Generation

| Run | Batches | Result |
|-----|---------|--------|
| `runs/2026/3/blended/1` | 4 | Inconclusive (too few batches) |
| `runs/2026/3/blended/18` | 22 | Inconclusive (still too early) |

### Two-Stage Eval (eval_only_double.py, batch 133 checkpoint)

| Metric | Stage 1 | Stage 2 (Refined) | Delta |
|--------|---------|-------------------|-------|
| Rubric mean | 0.661 | 0.654 | **-0.008** |
| Reward mean | 0.692 | 0.665 | -0.027 |
| Improvement rate | -- | 60.6% | -- |

At the best checkpoint, Stage 2 refinement actually **hurt** performance.

---

## What Failed and Why

### The OOD Collapse (Doublegeneration)

The training curve tells the story:

| Phase | Batches | Refined Reward | Refinement Delta | Regression Rate |
|-------|---------|----------------|------------------|-----------------|
| Rising | 1--50 | 0.55 --> rising | 0.27 | ~10% |
| Peak | 130--139 | 0.66 | -- | -- |
| Collapse | 150--178 | Falls to 0.47 | 0.13 | 28% |

**Root cause: distribution shift.** As Stage 1 improves during training, the plans entering Stage 2 change distribution. Stage 2's refinement strategy was learned on weaker initial plans (rubric ~0.24). When Stage 1 produces better plans (rubric ~0.41 by late training), Stage 2's strategy is out-of-distribution on these improved inputs. Specifically:

1. **Feedback degrades**: The detailed grader produces fewer actionable feedback bullets on already-good plans (21.7 --> 20.6 bullets). The feedback becomes less informative as there is less to critique.
2. **Refinement becomes counterproductive**: By late training, Stage 2 actively hurts 28% of samples (up from 10%). The model's refinement edits, learned for weak plans, are inappropriate for strong plans.
3. **Advantage collapse**: With smaller refinement deltas (0.27 --> 0.13), the GRPO advantages within Stage 2 become noisier, producing noisy gradients.

This is a fundamental instability: any two-stage pipeline where Stage 1 improves will cause Stage 2 to go OOD, unless Stage 2 is explicitly retrained or the stages are decoupled.

### Split Signal (Rubric Dropout)

The rubric dropout scheduler worked correctly -- p decreased from 0.80 to 0.36 as hidden reward rose from 0.55 to 0.73. The model did learn to score better without rubric during training. But at eval (where rubric is never shown), it scored 0.657 vs bestversion's 0.693.

**Root cause**: Splitting training between rubric-visible (80% initially) and rubric-hidden samples reduced the total signal available for the eval-relevant task (no-rubric generation). Bestversion trains 100% of samples on the eval-relevant task. The criteria-awareness transfer from visible to hidden was too weak to compensate for the reduced training volume.

### Inconclusive (Blended Generation)

Only 4 + 22 batches were run. The idea of collapsing Stage 2 quality into Stage 1 prompts was sound in principle but never received enough training to evaluate.

---

## What This Led To

1. **Confirmed that adding training stages causes distribution shift.** Any multi-stage pipeline where earlier stages improve will invalidate later stages' learned strategies. This applies to doublegeneration, multi-turn discussion, and any iterative refinement approach.

2. **Confirmed that splitting training signal hurts.** Both doublegeneration (splitting compute between two stages) and rubric dropout (splitting between visible/hidden) underperformed the simplest approach that puts 100% of signal on the eval-relevant task.

3. **Pivoted to the selection gap.** The gap analysis (performed after doublegeneration failed) revealed that 91% of the performance gap is selection (mean 0.70 vs oracle 0.85), not capability (oracle 0.85 vs reference 0.86). This shifted the research direction from training improvements to inference-time selection.

4. **Established the grader prompt mismatch warning.** The discovery that `build_grader_prompt` and `build_detailed_grader_prompt` produce incomparable scales became a critical rule for all future work.

---

## Lessons for Future Work

1. **Do not add dependent training stages.** If Stage N+1 receives inputs from Stage N, and Stage N improves during training, Stage N+1 will go OOD. Decouple stages or freeze earlier stages.

2. **Do not split training signal unless the split has proven benefit.** Bestversion's simplicity (100% of compute on the eval task) is a feature, not a limitation. Any alternative must demonstrate that the secondary objective's benefit exceeds the primary objective's loss from reduced signal.

3. **Refinement works at eval time, not train time.** Two-stage eval (Stage 1 --> feedback --> Stage 2) showed 60.6% improvement rate at a good checkpoint. The issue is that training the refinement stage causes OOD collapse. An inference-time refinement step, using a frozen model, avoids this problem.

4. **Monitor regression rate as a leading indicator.** The regression rate (fraction of samples where Stage 2 hurts vs helps) rising from 10% to 28% was the clearest early signal of the collapse, visible before the reward curve turned downward.

5. **Feedback quality depends on input quality.** As plans improve, the grader has less to critique. Any feedback-driven approach must account for the diminishing returns of feedback on already-good inputs.

---

## Related Documents

- [Phase 3 Details](../phases/phase3_doublegeneration.md) -- Full phase overview
- [Negative Results](../findings/negative_results.md) -- Comprehensive failure catalog
- [Process Shortcutting](../findings/process_shortcutting.md) -- The broader pattern of unrewarded process elimination
- [Selection Gap](../findings/selection_gap.md) -- Where the opportunity shifted after refinement failed
- [Experiment Catalog](../EXPERIMENT_CATALOG.md) -- Master experiment reference

## Source Code

| File | Purpose |
|------|---------|
| `src/co_scientist/trainers/refinement/train_double_generation.py` | Two-stage GRPO with refinement (2061 lines) |
| `src/co_scientist/trainers/refinement/train_blended_generation.py` | Blended generation (prompt relabeling) |
| `src/co_scientist/trainers/refinement/best_ver_rubric_dropout.py` | Rubric dropout training |
| `src/co_scientist/trainers/baselines/eval_only_double.py` | Two-stage eval script |
| `analysis/notebooks/active/double_gen_diagnosis.ipynb` | Doublegeneration diagnostic notebook |
