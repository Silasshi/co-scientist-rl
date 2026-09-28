# Finding: What Works

## Summary

Despite 80+ runs testing 10+ approaches, the simplest method -- single-stage GRPO with proper reward shaping -- remains the strongest. This document catalogs the specific configurations and design choices that produce the best results.

---

## The Winning Configuration: bestversion

### Training Method

Single-stage GRPO (Group Relative Policy Optimization) with no additional stages, processes, or distillation.

**Implementation**: `src/co_scientist/trainers/grpo/best_ver.py`

**Best eval rubric**: 0.693 (2 epochs, run 2(ml))

### Model Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | Qwen/Qwen3-30B-A3B | Best of 3 models tested (vs gpt-oss-20b, Llama-3.1-8B) |
| LoRA rank | 64 | Higher ranks not tested; 64 works well |
| Learning rate | 1e-5 | Standard for LoRA fine-tuning |
| Batch size | 64 | Sufficient diversity per update |
| Group size | 8 | 8 samples per goal for stable advantage estimation |
| Max tokens | 2048 | Generation budget |
| Max words | 750 | Soft limit via format penalty |
| Generation temperature | 1.0 | High diversity for exploration |
| Grading temperature | 0 | Greedy for consistent scoring |
| Epochs | 2 (215 batches) | Returns diminish after epoch 2 |

### Reward Formula

```
reward = rubric_score + 0.08 * length_bonus - format_penalty
```

| Component | Details |
|-----------|---------|
| rubric_score | 0->0.0, 1->0.2, 2->0.6, 3->1.0 (non-linear: big jump from 1->2) |
| length_bonus | Gaussian centered at 600 words, sigma=120 |
| format_penalty | 0.2 + 0.0005*excess_words for words > 750 |

### Training Dynamics

| Metric | Value |
|--------|-------|
| Batches to convergence | ~140 (reward peaks at 0.83) |
| Total batches (2 epochs) | 214 |
| Format compliance | ~99% (from ~90% at start) |
| Format penalty (mean) | 0.004 (negligible after training) |
| Training reward trajectory | 0.65 -> 0.78-0.83 |

---

## Why It Works

### 1. Simplicity Avoids Failure Modes

Every additional component introduces new failure modes:
- Multi-stage training -> OOD collapse (doublegeneration)
- Intermediate processes -> process shortcutting (think-solution, multi-turn)
- Self-distillation -> redundant signal (SDPO)
- Rubric manipulation -> reduced training signal (rubric dropout)

Single-stage GRPO has none of these issues. The gradient flows directly from the reward to the policy.

### 2. Group Size 8 Provides Stable Advantages

With 8 samples per goal, GRPO can reliably estimate which samples are better than average. Smaller group sizes produce noisier advantage estimates. Larger group sizes increase compute cost without proportional benefit.

### 3. Format Compliance Through Penalty

The format penalty (0.2 + 0.0005*excess for words > 750) achieves ~99% format compliance. This is important because:
- The grader expects structured plans with specific sections
- Format violations lead to low rubric scores regardless of content quality
- The penalty is harsh enough to enforce compliance but light enough not to dominate the reward

### 4. Length Guidance Through Gaussian Bonus

The Gaussian bonus centered at 600 words (sigma=120) encourages plans that are neither too short (missing content) nor too long (penalized by format penalty). While the length bonus has weak correlation with rubric score (r=0.075), it provides a useful regularization effect.

### 5. Greedy Grading, Generous Generation

- **Temperature=1.0 for generation**: Ensures diverse samples within each group, giving GRPO a meaningful range of quality to estimate advantages from.
- **Temperature=0 for grading**: Ensures consistent, reproducible scoring. Stochastic grading would add noise to the reward signal.

---

## Generation Quality Is Already High

| N | Oracle Score | Gap from Reference |
|---|-------------|--------------------|
| 1 | 0.70 | 0.16 |
| 2 | 0.77 | 0.09 |
| 4 | 0.82 | 0.04 |
| 8 | 0.85 | 0.01 |

The model CAN produce excellent plans -- best-of-8 oracle (0.85) nearly matches reference solutions (0.86). The challenge is SELECTION, not generation.

## What External Models Show

GPT-5.4 achieves 0.843 rubric score with zero-shot generation (no RL training, no rubric access). This suggests:
- Raw model capability can overcome the selection lottery
- The reward function may be suppressing quality by penalizing length/detail
- RL training on Qwen3-30B-A3B may be constrained by the model's base capability

---

## What Definitely Helps (Proven)

| Technique | Evidence | Impact |
|-----------|----------|--------|
| Single-stage GRPO | Best eval (0.693) of all methods | Baseline |
| LoRA rank 64 | Standard across all successful runs | Sufficient capacity |
| Group size 8 | Stable advantages, good explore/exploit | Core to GRPO |
| Format penalty | 99% compliance | Essential for grader |
| Length Gaussian bonus | Regularization effect | Modest positive |
| Greedy grading | Consistent rewards | Reduces noise |
| High generation temperature | Diverse samples | Exploration |

## What Might Help (Untested or Incomplete)

| Technique | Rationale | Status |
|-----------|-----------|--------|
| Self-selector training | Targets 91% selection gap | Implemented, not trained |
| Relaxing length penalty | Reward misalignment evidence | best_ver_async was inconclusive |
| Larger base model | GPT-5.4 achieves 0.843 without RL | Not feasible (compute) |
| CPR (contrastive ranking) | Better selection model | Only 12 batches |
| Rubric-aware selection | Direct attack on selection gap | Not implemented |

## What Definitely Does Not Help (Proven Negative)

| Technique | Evidence | Why |
|-----------|----------|-----|
| Self-distillation (SDPO) | 20 runs, no improvement | Redundant with GRPO advantages |
| Two-stage refinement | Eval 0.654 (-0.039) | Stage 2 OOD collapse |
| Rubric dropout | Eval 0.657 (-0.036) | Split signal |
| Multi-turn discussion | 4 versions, no learning | Discussion doesn't help |
| Explicit thinking | -0.090 score impact | Process shortcutting |
| Self-calibration | Uniform 1.08x weights | Self-eval is unreliable |
| Cross-model distillation | Destroys small model | Distribution mismatch |

---

## Key Principles

1. **Simple > complex**: Single-stage GRPO beats every multi-stage, multi-process variant
2. **Don't add unrewarded processes**: They will be shortcutted (see [Process Shortcutting](process_shortcutting.md))
3. **Reward shaping matters**: The balance between rubric, length, and format is carefully tuned
4. **Focus on selection, not generation**: 91% of the gap is selection (see [Selection Gap](selection_gap.md))
5. **Inference-time > training-time**: Best-of-N selection with a trained selector is the most promising direction

## Recommendations for New Experiments

Based on what works and what doesn't:

1. **Keep bestversion as the generator.** Do not add training complexity.
2. **Focus on selection.** Train a separate model to pick the best plan from N candidates.
3. **If modifying the reward, test on eval rubric** -- training reward improvement does not predict eval improvement.
4. **Do not add intermediate processes** unless they are directly and effectively rewarded.
5. **Consider inference-time scaling** (best-of-N with selection) over training-time scaling.

## Related Findings

- [Selection Gap](selection_gap.md) -- where the remaining opportunity is
- [Negative Results](negative_results.md) -- what to avoid
- [Reward Misalignment](reward_misalignment.md) -- limitations of current reward
- [Process Shortcutting](process_shortcutting.md) -- why complexity fails
- [Back to Catalog](../EXPERIMENT_CATALOG.md)
