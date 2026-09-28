# Phase 4: Specialized Methods (April 2026)

## Overview

12 runs exploring IBT (iterative brainstorming), think-solution, CPR (contrastive plan ranking), self-selector, and async bestversion variants. Think-solution produced a clear negative result. IBT is under investigation. Self-selector is identified as the most promising direction based on gap analysis.

---

## best_ver_async

Same algorithm as bestversion but with relaxed length constraints to allow longer, more detailed plans.

### Configuration Differences from bestversion

| Parameter | bestversion | best_ver_async |
|-----------|-------------|----------------|
| Max words | 750 | 850 |
| Max tokens | 2048 | 3072 |

### Runs

| Run | Batches | Reward | Notes |
|-----|---------|--------|-------|
| Run 1--3 | Various | Various | Early variants |
| **Run 4** | 106 | 0.778 | Best async run |

### Result

Comparable performance to bestversion. Allowing longer plans (850 vs 750 words) did not meaningfully improve quality. The length penalty in the reward function is not the primary bottleneck.

**Implementation**: `src/co_scientist/trainers/grpo/best_ver_async.py`

---

## IBT (Iterative Brainstorming Training)

### Architecture

Multi-step brainstorming where the model iteratively refines ideas before producing a final plan.

| Parameter | Value |
|-----------|-------|
| Policy model | Qwen3-4B-Instruct-2507 (smaller than bestversion's 30B) |
| Grader model | Qwen3-30B-A3B |
| Goals per run | 25 goals x 5 turns |
| Modes | single_chain, mini_grpo |
| Variants | With and without OPD |

### Runs

| Run | Batches | Notes |
|-----|---------|-------|
| Runs 1--6 | Various | Failed or incomplete |
| **Run 7** | 106 | Only completed run |

### Status

Under investigation. The use of a smaller policy model (4B vs 30B) makes direct comparison to bestversion difficult. It is unclear whether the brainstorming structure itself helps or whether the smaller model is the limiting factor.

### Known Issue: Process Shortcutting

Like multi-turn and think-solution, IBT is susceptible to process shortcutting: the model learns to skip or minimize the brainstorming steps that aren't directly rewarded, converging toward single-step generation.

**Implementation**: `src/co_scientist/trainers/ibt/train_ibt.py`

---

## Think-Solution

### Approach

Separate the generation into explicit `<think>` (reasoning) and `<solution>` (plan) blocks, with independent grading for each component.

### Key Finding

| Component | Score Impact |
|-----------|-------------|
| Think blocks | **-0.090 lower** than non-thinking |
| Solution blocks | Standard performance |

### Analysis

Thinking actively hurts plan quality. The model's reasoning in `<think>` blocks does not translate to better plans in `<solution>` blocks. Instead, the think block consumes token budget that could be used for more detailed plan content.

This is another instance of process shortcutting: the think process is not directly rewarded, so the model either eliminates it or fills it with content that doesn't help.

### Result

**Negative.** Explicit reasoning does not improve research plan generation. The optimal strategy (from the reward function's perspective) is to skip thinking and put all effort into the final plan.

---

## CPR (Contrastive Plan Ranking)

### Approach

Pairwise ranking training: the model learns to distinguish better plans from worse ones by training on contrastive pairs.

### Run

| Run | Batches | Status |
|-----|---------|--------|
| 1 run | 12 | Preliminary |

### Result

Insufficient training data for evaluation. Only 12 batches completed. The approach needs significantly more training to assess viability.

**Implementation**: `src/co_scientist/trainers/grpo/train_cpr.py`

**Eval**: `src/co_scientist/trainers/eval/eval_cpr.py`

---

## Self-Selector

### Motivation

Gap analysis revealed:
- **Selection gap** (mean to oracle best-of-8): 0.157 = **91% of total gap**
- **Capability gap** (oracle to reference): 0.015 = 9% of total gap

The model already produces near-reference-quality plans (oracle best-of-8 = 0.85 vs reference = 0.86). The bottleneck is selecting the best output, not generating it.

### Approach

Train a selection model that uses pairwise comparison without rubric items. The model learns to pick the better plan from a pair, enabling best-of-N selection at inference time.

### Status

Implemented but not yet trained. This is the most promising remaining direction because:

1. It targets the largest component of the performance gap (91%)
2. It is orthogonal to generation training (can combine with any generator)
3. Self-eval correlation with rubric is only r ~ 0.061 currently, so there is massive room for improvement

### Best-of-N Ceiling Analysis

| N | Score |
|---|-------|
| 1 | 0.70 |
| 2 | 0.77 |
| 4 | 0.82 |
| 8 | 0.85 |

Even a mediocre selector that can reliably pick top-2 from 8 candidates would boost performance from 0.70 to ~0.82, far exceeding anything achievable through better generation training.

---

## Phase Summary

| Method | Status | Result |
|--------|--------|--------|
| best_ver_async | Completed | Comparable to bestversion, no improvement |
| IBT | Under investigation | Process shortcutting concern |
| Think-solution | Completed | **Negative** (-0.090 from thinking) |
| CPR | Preliminary | Insufficient data (12 batches) |
| Self-selector | Implemented, not trained | **Most promising direction** |

### Strategic Conclusion

After 80+ runs across 4 phases, the evidence points strongly toward selection as the key lever. Further generation training is unlikely to beat the bestversion ceiling. The self-selector approach directly targets the 91% selection gap and should be the primary focus going forward.

See: [Phase 3](phase3_doublegeneration.md) | [Back to Catalog](../EXPERIMENT_CATALOG.md)
