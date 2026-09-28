# Does Multi-Turn Discussion Improve Research Plan Quality?

## Summary

Across 4 pipeline versions and ~1400+ scored conversations, **discussion turns show no consistent benefit to final plan quality** as measured by the 7-desiderata rubric grader.

## Evidence

### Correlation: Discussion Turns vs Rubric Score

| Version | Conversations | Pearson r | Interpretation |
|---------|--------------|-----------|----------------|
| V2 (12 batches) | 369 | -0.115 | Weak negative |
| V3 (6 batches) | 191 | **-0.335** | Moderate negative |
| V4 (26 batches) | 832 | 0.052 | Near zero |

No version shows a meaningful positive correlation.

### V4: Rubric by Discussion Turn Count (largest dataset)

| Discussion Turns | Count | Mean Rubric | Median |
|-----------------|-------|-------------|--------|
| 0 | 510 | 0.529 | 0.563 |
| 1 | 53 | 0.521 | 0.517 |
| 2 | 70 | 0.556 | 0.553 |
| 3 | 60 | 0.587 | 0.626 |
| 4 | 47 | 0.531 | 0.549 |
| 5 | 32 | 0.618 | 0.597 |
| 6+ | 60 | 0.556 | 0.579 |

Maximum benefit: 0 turns (0.529) vs 3 turns (0.587) = **+0.058**. Non-monotonic — 4 turns scores lower than 2 turns.

### Within-Goal Comparison (Same Goal, More vs Fewer Discussion)

| Version | More disc → better | More disc → worse | Tie |
|---------|-------------------|-------------------|-----|
| V2 | 30% | 32% | 38% |
| V3 | **19%** | **81%** | 0% |
| V4 | 18% | 28% | 54% |

In V3, more discussion was **4x more likely to hurt** than help within the same goal.

### V2's Zero-Discussion Peak

V2 conversations with 0 discussion turns achieved rubric **0.692** — essentially matching bestversion's eval score of **0.693**. This was the highest rubric across all discussion levels in V2.

### V4 Transition Batches (Same Batch, Fair Comparison)

| Batch | High disc rubric | Low disc rubric | Delta |
|-------|-----------------|-----------------|-------|
| 2 | 0.569 (n=27) | 0.380 (n=1) | +0.189 |
| 4 | 0.556 (n=25) | 0.426 (n=3) | +0.130 |
| 7 | 0.528 (n=10) | 0.632 (n=8) | -0.104 |
| 8 | 0.619 (n=3) | 0.653 (n=10) | -0.034 |

Mixed: early batches show discussion helping, later batches show it hurting.

### V4 Trained Model vs Bestversion

| Method | Rubric | Discussion |
|--------|--------|------------|
| Bestversion eval (single-turn) | 0.693 | None |
| V4 trained, 0 discussion (training-time) | 0.525 | None |
| V4 untrained, 3-5 discussion (training-time) | 0.562 | 3-5 turns |

Note: bestversion rubric is eval-time (held-out), V4 is training-time (in-distribution). Not directly comparable.

## Process-Shortcutting Behavior

All versions exhibit the model learning to skip discussion:

| Version | Discussion collapse | Speed |
|---------|-------------------|-------|
| V2 | 0.7 turns → 0.0 | Immediate (batch 0 already low) |
| V3 | 5.3 → stable (crashed at batch 6) | N/A |
| V4 | 5.6 → 0.0 | Gradual (10 batches) |

The model rationally optimizes: discussion doesn't improve GRPO reward (rubric), so it learns to skip it. This is consistent across all versions regardless of PRM design.

## Possible Explanations

1. **The rubric doesn't capture discussion benefits.** The 7-desiderata rubric evaluates the plan in isolation. It doesn't measure whether the plan addresses the researcher's specific concerns raised during discussion. A plan could score equally well whether or not discussion occurred.

2. **The base model is already good.** Qwen3-30B-A3B produces high-quality research plans without discussion. Discussion doesn't add information the model doesn't already have.

3. **The simulated researcher doesn't add real value.** Gemini Flash's research questions may be too generic to improve the plan. A real human researcher with domain expertise might ask more targeted questions.

4. **OPD hints compensate.** Even without discussion, OPD provides per-turn guidance. The model may be getting the "benefits of discussion" through OPD without needing actual discussion turns.

## Implications for the Paper

### If presenting as a positive result:
- Need a different evaluation metric that specifically measures how well plans address discussion topics
- Or show that multi-turn produces plans with different (better) characteristics even if rubric is similar
- Or demonstrate benefit with real human researchers instead of simulated ones

### If presenting as a negative/analysis result:
- "We find that models trained with multi-turn RL learn to shortcut the discussion process"
- "Process-shortcutting is robust across different PRM designs and training architectures"
- "Current rubric-based evaluation fails to capture potential benefits of collaborative brainstorming"
- This is a genuine research finding about reward misalignment in multi-turn RL

## Files

- V2 logs: `runs/2026/3/multiturn_v4/2/train/`
- V3 logs: `runs/2026/3/multiturn_v4/3/train/`
- V4 logs: `runs/2026/3/multiturn_v4/4/train/`
- This analysis: `analysis/multiturn_discussion_analysis.md`
