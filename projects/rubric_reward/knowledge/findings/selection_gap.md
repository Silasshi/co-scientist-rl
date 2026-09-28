# Finding: The Selection Gap (91% of Total Performance Gap)

## Summary

The performance gap between the trained model's average output and reference solutions is overwhelmingly a **selection** problem, not a **capability** problem. The model already generates near-reference-quality plans -- it just cannot identify which of its outputs are good.

## The Numbers

| Metric | Score | Gap Component |
|--------|-------|---------------|
| Model mean (N=1) | 0.70 | -- |
| Model best-of-2 | 0.77 | -- |
| Model best-of-4 | 0.82 | -- |
| Model best-of-8 (oracle) | 0.85 | -- |
| Reference solutions | 0.86 | -- |
| **Selection gap** (mean to oracle) | 0.157 | **91% of total gap** |
| **Capability gap** (oracle to reference) | 0.015 | **9% of total gap** |

## What This Means

1. **The model CAN produce excellent plans.** Oracle best-of-8 (0.85) nearly matches reference solutions (0.86). The generation capability is already there.

2. **The model CANNOT tell which plans are good.** Self-evaluation correlation with rubric score: r ~ 0.061 (essentially random). The model's self-assessment provides almost no signal about actual plan quality.

3. **Training better generators won't help much.** Even a perfect generator (matching reference quality on every sample) would only close the 9% capability gap. The 91% selection gap remains.

## Best-of-N Scaling

| N | Score | Delta from N=1 |
|---|-------|----------------|
| 1 | 0.70 | -- |
| 2 | 0.77 | +0.07 |
| 4 | 0.82 | +0.12 |
| 8 | 0.85 | +0.15 |

The returns diminish as N increases, but even N=2 with a perfect selector would yield +0.07, which exceeds the improvement from any generation training method tested.

## Why Self-Evaluation Fails

Self-evaluation correlation with rubric: r ~ 0.061. This near-zero correlation is because:

1. **Method-specific criteria**: 80.6% of model-low rubric items are ones where the reference scores high using its specific methodology. The rubric checks for specific approaches the reference used, not general quality.

2. **No rubric access at inference**: The model cannot infer the method-specific criteria that the grader will use. Without seeing the rubric, the model has no basis for preferring one plan over another.

3. **Quality is not self-evident**: Research plan quality depends on alignment with specific rubric desiderata, not on surface-level features the model can easily assess.

## Implications for Next Steps

1. **Self-selector training** is the highest-leverage intervention. A trained selector that can reliably pick top-2 from 8 candidates would boost performance from 0.70 to ~0.82.

2. **Selection is orthogonal to generation**. A selector can be combined with any generator (bestversion, doublegeneration, etc.) for additive gains.

3. **Rubric-aware selection** would be ideal, but even rubric-free pairwise comparison should significantly outperform random selection (r ~ 0.061).

## Evidence Base

- Gap analysis: bestversion eval, 552 goals x 8 samples
- Self-eval: `src/co_scientist/trainers/eval/eval_bon.py`
- Oracle analysis: `analysis/notebooks/` (various diagnostic notebooks)

## Related Findings

- [Reward Misalignment](reward_misalignment.md) -- the reward function itself is part of the problem
- [What Works](what_works.md) -- bestversion is the current generation ceiling
- [Back to Catalog](../EXPERIMENT_CATALOG.md)
