# Finding: Reward Function Misalignment

## Summary

The reward function used for training is misaligned with the actual quality signal from the grader. Specifically, the length penalty suppresses the detail that the grader values, and the overall reward formula does not accurately predict eval-time rubric scores.

## The Reward Formula

```
reward = rubric_score + 0.08 * length_bonus - format_penalty
```

| Component | Formula | Effect |
|-----------|---------|--------|
| rubric_score | 0->0.0, 1->0.2, 2->0.6, 3->1.0 | Maps 4-point scale to [0,1] |
| length_bonus | Gaussian(600, 120) | Peaks at 600 words, falls off beyond |
| format_penalty | 0.2 + 0.0005*excess (if >750 words) | Harsh cliff at 750 words |

## The Problem

### Length Penalty vs Grader Preference

The reward formula penalizes plans longer than 750 words, but the grader actually values detail:

| Observation | Value |
|-------------|-------|
| Within-plan correlation (length vs rubric) | Pearson r = 0.075 (negligible) |
| GPT-5.4 rubric score (no RL training) | **0.843** |
| GPT-5.4 achieves this by | Being more capable, generating more detail |
| bestversion rubric (with RL training) | 0.693 |

GPT-5.4 scores 0.843 on the rubric WITHOUT any RL training, purely by being a more capable model that generates more detailed plans. The RL-trained bestversion scores only 0.693 -- the reward function's length penalty actively suppresses the quality the grader wants.

### Rubric Score Mapping Is Coarse

The mapping 0->0.0, 1->0.2, 2->0.6, 3->1.0 creates uneven gradients:
- 0 to 1: +0.2 (small reward for going from terrible to poor)
- 1 to 2: +0.4 (large reward for going from poor to good)
- 2 to 3: +0.4 (same reward for going from good to excellent)

This means the model gets the same reward signal for improving from poor-to-good as for good-to-excellent, even though the latter is much harder and more valuable.

### Training Reward vs Eval Rubric Disconnect

Training reward includes length bonus and format penalty, but eval measures rubric score only. This means:
- A plan can have high training reward (short, well-formatted, mediocre content) but low eval rubric
- A plan can have low training reward (long, detailed, excellent content) but high eval rubric
- The model is optimized for the wrong objective

## Evidence

### GPT-5.4 Comparison

| Model | Rubric Score | RL Training | Notes |
|-------|-------------|-------------|-------|
| GPT-5.4 | 0.843 | None | Pure capability, no length constraint |
| bestversion (Qwen3-30B) | 0.693 | 2 epochs GRPO | Constrained by reward formula |
| Reference solutions | 0.860 | N/A | Human-written |

GPT-5.4 nearly matches reference solutions without any RL training. This strongly suggests the reward function is the bottleneck, not model capability or training method.

### Length Is Not the Score Driver

Pearson r = 0.075 between length and rubric score within plans means length explains less than 1% of score variance. The length bonus/penalty components of the reward function are optimizing for a dimension that barely matters for quality.

## Implications

1. **Removing or relaxing the length penalty** might allow the model to generate more detailed (and higher-quality) plans.
2. **The reward formula should more closely match the eval metric** (pure rubric score).
3. **Scaling model capability** (e.g., using a larger base model) might be more effective than reward engineering on a smaller model.
4. **The best_ver_async experiment** (850 words, 3072 tokens) showed comparable performance, suggesting that simply raising the length limit isn't sufficient -- the reward penalty during training has already shaped the model's behavior.

## Related Findings

- [Selection Gap](selection_gap.md) -- even with misaligned reward, the model generates excellent plans sometimes
- [Process Shortcutting](process_shortcutting.md) -- the reward function drives shortcutting behavior
- [What Works](what_works.md) -- bestversion works despite the misalignment
- [Back to Catalog](../EXPERIMENT_CATALOG.md)
