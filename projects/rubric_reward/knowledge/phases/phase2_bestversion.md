# Phase 2: bestversion & Method Exploration (February 2026)

## Overview

34 runs across three workstreams: reward shaping ablations, SDPO, and the bestversion baseline. This phase established the bestversion ceiling (eval rubric 0.693) that no subsequent method has beaten.

## bestversion (withA1,A2) -- THE MAIN BASELINE

bestversion = single-stage GRPO with two key improvements (A1 + A2) over the Phase 1 baseline.

### Architecture

- Single-stage GRPO, synchronous sampling
- 8 samples per goal (group_size=8)
- LoRA rank 64, lr=1e-5
- Max 750 words, 2048 tokens
- Generation temperature=1.0, grading temperature=0

### Reward Formula

```
reward = rubric_score + 0.08 * length_bonus - format_penalty
```

- rubric_score: 0->0.0, 1->0.2, 2->0.6, 3->1.0
- length_bonus: Gaussian centered at 600 words, sigma=120
- format_penalty: 0.2 + 0.0005 * excess (if words > 750)

### Key Runs

| Run | Batches | Checkpoints | Reward | Eval Rubric | Domain |
|-----|---------|-------------|--------|-------------|--------|
| `2(ml)` | 214 | 215 | 0.784 | **0.693** | ML |
| `3(pubmed)` | 199 | -- | 0.853 | -- | PubMed |
| `3(arxiv)` | 203 | -- | 0.726 | -- | arXiv |

### Training Dynamics (run 2(ml))

- Reward grows from 0.65 to 0.78--0.83 over 215 batches
- Peaks at 0.83 around batch 139
- Format compliance: ~99%
- Format penalty mean: 0.004 (negligible)
- Avg rubric: 0.443, avg reward: 0.527 (at batch 107)

### The 0.693 Ceiling

Eval after 2 epochs yielded rubric = **0.693**. Every subsequent method (SDPO, doublegeneration, rubric dropout, multi-turn, think-solution) attempted and failed to beat this number.

**Run path**: `runs/2026/2/withA1,A2/2(ml)/`

**Implementation**: `src/co_scientist/trainers/grpo/best_ver.py`

---

## Reward Shaping Ablations

Multiple reward formulations tested against bestversion:

| Run | Approach | Batches | Reward | Notes |
|-----|----------|---------|--------|-------|
| `0-9_scale/15(*)` x3 | 0-9 rubric scale | 106 each | 0.76--0.80 | Wider scale, marginal difference |
| `hard_min/14` | Hard minimum constraints | 123 | 0.618 | Constraints too strict, lower reward |
| `std+mean_weighted/13(*)` | Weighted rewards | -- | std: 0.819, mean: 0.787 | std_weighted best of the weighted variants |
| `std_stand+band_bonus/8(ml)` | Band bonus | -- | 0.840 | Highest raw reward but no eval improvement |
| `std_stand+band_bonus/9(ml)` | Band bonus | 213 | 0.821 | Similar to 8(ml) |

**Conclusion**: No reward shaping ablation clearly outperformed the bestversion formula (rubric_score + 0.08*length_bonus - format_penalty). Higher training rewards did not translate to higher eval rubric scores. The standard formula was kept.

---

## SDPO (Self-Distillation Policy Optimization)

20 runs (runs 20--28 plus variants). SDPO adds a self-distillation loss to GRPO, parameterized by a mixing lambda.

### Configuration Space

- Lambda: 0.1 to 0.45
- Multiple architectures and loss formulations
- Recovery experiments: warmstart from bestversion checkpoint at batch 214

### Best Run

| Run | Batches | Reward | Notes |
|-----|---------|--------|-------|
| `28_recovery_A_warmstart` | 244 | 0.673 | Best SDPO result; warmstarted from bestversion batch 214 |

### Conclusion

**SDPO did NOT outperform bestversion.** Even warmstarting from the bestversion checkpoint and continuing with self-distillation couldn't improve over the original. The additional complexity of the self-distillation loss provides no benefit for this task. SDPO was abandoned after 20 runs.

---

## Phase Summary

| Method | Best Result | vs Bestversion (0.693) |
|--------|-------------|------------------------|
| bestversion 2(ml) | 0.693 eval | Baseline |
| SDPO (20 runs) | 0.673 reward | Did not outperform |
| Reward ablations | 0.76--0.84 reward | No eval improvement |

The bestversion method (single-stage GRPO with A1+A2) remains the strongest approach.

## Run Artifacts

- bestversion: `runs/2026/2/withA1,A2/`
- SDPO: `runs/2026/2/SDPO/`
- Reward ablations: `runs/2026/2/` (various subdirectories)

See: [Phase 1](phase1_baselines.md) | [Phase 3](phase3_doublegeneration.md) | [Back to Catalog](../EXPERIMENT_CATALOG.md)
