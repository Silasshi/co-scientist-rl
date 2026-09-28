# Experiment Catalog -- Co-Scientist Project

This is the master reference for all past experiments in the co-scientist project. Read this document first to understand the full experimental history, results, and lessons learned.

---

## Project Overview

Universal research plan generation: an LLM learns to produce high-quality research plans for any field. The project explores three training approaches (D1: rubric reward, D2: IBT, D3: TTT-Discover). See `DIRECTIONS.md` for the full map.

**D1 setup** (this catalog covers D1 and D2 experiments):
- Dataset: facebook/research-plan-gen (ML split): ~6,848 train / 685 test goals
- Training model: Qwen/Qwen3-30B-A3B with LoRA rank 64
- Grader: Qwen/Qwen3-30B-A3B with goal-specific 7-desiderata rubric (10 rubric items per goal)
- Reference solution score: 0.86 mean rubric (upper bound)

---

## Experiment Timeline & Results

### Phase 1: Baseline Exploration (January 2026)

16 runs, exploratory phase. Tested multiple model architectures and reward formulations.

| Run | Model | Batches | Reward | Notes |
|-----|-------|---------|--------|-------|
| `success_dense_score` | Qwen3-30B-A3B | 189 | 0.587 | First major training run, dense scoring |
| `25(gsm8k_qwen3_sparse_score)` | Qwen3-30B-A3B | 23 | 0.961 | Different task (GSM8K), not comparable |
| Various | gpt-oss-20b | -- | -- | Tested, inferior to Qwen3 |
| Various | Llama-3.1-8B | -- | -- | Tested, inferior to Qwen3 |

**Key learning**: Qwen3-30B-A3B emerged as the best model for research plan generation. Dense scoring outperformed sparse scoring on this task.

See: [Phase 1 Details](phases/phase1_baselines.md)

---

### Phase 2: Method Exploration (February 2026)

34 runs testing reward shaping, SDPO, and the bestversion baseline.

#### Reward Shaping Ablations

| Run | Batches | Reward | Description |
|-----|---------|--------|-------------|
| `0-9_scale/15(*)` (3 runs) | 106 each | 0.76--0.80 | Tested 0-9 rubric scale instead of 0-3 |
| `hard_min/14` | 123 | 0.618 | Hard minimum constraints on rubric criteria |
| `std+mean_weighted/13(*)` | -- | std: 0.819, mean: 0.787 | Weighted reward variants; std_weighted best |
| `std_stand+band_bonus/8(ml)` | -- | 0.840 | Band bonus reward shaping |
| `std_stand+band_bonus/9(ml)` | 213 | 0.821 | Band bonus variant |

#### SDPO (Self-Distillation Policy Optimization)

| Run | Batches | Reward | Description |
|-----|---------|--------|-------------|
| 20 runs (runs 20--28 + variants) | Various | Various | Lambda 0.1--0.45, multiple configurations |
| `28_recovery_A_warmstart` | 244 | 0.673 | Best SDPO run; warmstarted from bestversion batch 214 |

**Result: SDPO did NOT outperform bestversion.** Warmstarting from bestversion checkpoint didn't help either. Self-distillation adds complexity without benefit in this setting.

#### bestversion (withA1,A2) -- THE MAIN BASELINE

| Run | Batches | Reward | Eval Rubric | Domain | Notes |
|-----|---------|--------|-------------|--------|-------|
| `2(ml)` | 214 | 0.784 | **0.693** | ML | **Primary baseline. Nothing has beaten this.** |
| `3(pubmed)` | 199 | 0.853 | -- | PubMed | Higher reward on different domain |
| `3(arxiv)` | 203 | 0.726 | -- | arXiv | Cross-domain variant |

bestversion = single-stage GRPO with A1 + A2 improvements. Eval after 2 epochs: **rubric 0.693**. This is the ceiling that all subsequent methods attempted to break.

See: [Phase 2 Details](phases/phase2_bestversion.md)

---

### Phase 3: Advanced Techniques (March 2026)

17 runs testing refinement, rubric dropout, and multi-turn approaches.

#### Doublegeneration (refinement/8)

| Metric | Value |
|--------|-------|
| Batches | 178 |
| Checkpoints | 37 |
| Architecture | Two-stage: generate -> feedback -> refine -> grade |
| Avg initial rubric | 0.324 |
| Avg refined rubric | 0.549 |
| Improvement rate | 83--89% |
| Format compliance | 86.81% |
| Total samples | 91,648 generated, 87,404 valid (95.4%) |
| **Eval** | **0.654** (vs bestversion 0.693) -- **WORSE** |

Pattern: refined reward peaks at batch ~130, then collapses to ~0.47 by batch 178. Refinement delta shrinks from 0.27 (early) to 0.13 (late). Regression rate rises from ~10% to 28%.

**Root cause**: Stage 1 improvement causes distribution shift in Stage 2 input. Stage 2's learned refinement strategy becomes out-of-distribution, leading to advantage collapse, noisy gradients, and degradation.

**Result: Negative.** Two-stage refinement hurts more than it helps after peak.

#### Rubric Dropout (rubric_dropout/13)

| Metric | Value |
|--------|-------|
| Batches | 106 |
| Checkpoints | 19 |
| **Eval** | **0.657** (vs bestversion 0.693) -- **WORSE** |

**Root cause**: Splitting training signal between visible/hidden rubric criteria hurt more than criteria-awareness transfer helped.

**Result: Negative.**

#### Multi-turn V1--V4

| Version | Run | Batches | Result | Issue |
|---------|-----|---------|--------|-------|
| V1 | run4 | 54 | No learning | Binary PRM, signal too coarse |
| V2 | run1 | 72 | Failed | Merged evaluator, 71% empty responses |
| V3 | run1 | 3 | Failed | PRM parsing bug (all negative scores mapped to 0) |
| V4 | 5 runs | ~0 progress | Failed | Wave-based + ternary PRM, still no learning |

Multi-turn discussion correlation with rubric: Pearson r = -0.115 to 0.052 (essentially zero).

**Result: All negative.** Multi-turn discussion shows no consistent benefit over single-turn generation.

#### Blended Generation

| Run | Batches | Result |
|-----|---------|--------|
| run 1 | 4 | Inconclusive |
| run 18 | 22 | Inconclusive |

Early-stage, insufficient data to draw conclusions.

See: [Phase 3 Details](phases/phase3_doublegeneration.md)

---

### Phase 4: Specialized Methods (April 2026)

12 runs exploring IBT, think-solution, CPR, and self-selector.

#### best_ver_async

| Run | Batches | Reward | Notes |
|-----|---------|--------|-------|
| 4 runs total | -- | -- | Same algorithm as bestversion |
| Run 4 (best) | 106 | 0.778 | Longer plans: 850 vs 750 max words, 3072 vs 2048 tokens |

**Result**: Comparable performance to bestversion. Slight length parameter change doesn't meaningfully improve quality.

#### IBT (Iterative Brainstorming Training)

| Run | Batches | Notes |
|-----|---------|-------|
| 6 runs attempted | -- | Only run 7 completed |
| Run 7 | 106 | Policy: Qwen3-4B-Instruct-2507 (smaller model) |

Script: `src/co_scientist/ibt/train_ibt.py`. Modes: single_chain and mini_grpo, with/without OPD.

**Result**: Under investigation.

#### Think-Solution

| Run | Batches | Finding |
|-----|---------|---------|
| 1 run | 106 | Think blocks score -0.090 LOWER than non-thinking |

Separate grading for `<think>` vs `<solution>` blocks. Thinking actually hurts plan quality -- model optimizes for shortest path to reward.

**Result: Negative.**

#### CPR (Contrastive Plan Ranking)

| Run | Batches | Finding |
|-----|---------|---------|
| 1 run | 12 | Preliminary, insufficient data |

Pairwise ranking training approach. Not enough training to evaluate.

#### Self-Selector

Implemented but not yet trained. Targets the selection gap (91% of total performance gap). Uses pairwise comparison without rubric items.

**This is the most promising remaining direction** based on gap analysis.

See: [Phase 4 Details](phases/phase4_advanced.md)

---

## Key Findings Summary

### 1. The Gap is Selection (91%), Not Capability (9%)

| Metric | Score |
|--------|-------|
| Mean (N=1) | 0.70 |
| Best-of-2 | 0.77 |
| Best-of-4 | 0.82 |
| Best-of-8 (oracle) | 0.85 |
| Reference solutions | 0.86 |

The model CAN produce near-reference-quality plans. It just can't tell which of its outputs are good. The selection gap (mean to oracle) accounts for 91% of the total gap to reference. The capability gap (oracle to reference) is only 9%.

### 2. Method-Specific Rubric Items Are the Bottleneck

80.6% of model-low items: the reference scores HIGH on the same rubric criterion. The rubric checks the reference's specific methodology. When the model proposes a different (but potentially valid) approach, it scores low on method-specific criteria. This is a rubric design issue, not a capability issue.

### 3. All Training Improvements Failed to Beat Bestversion (0.693)

Over 10 approaches tested across 80+ runs:

| Method | Eval Rubric | vs Bestversion |
|--------|-------------|----------------|
| bestversion | 0.693 | -- |
| SDPO (best of 20 runs) | -- | Did not outperform |
| Doublegeneration | 0.654 | -0.039 |
| Rubric dropout | 0.657 | -0.036 |
| Multi-turn V1--V4 | -- | No learning |
| Think-solution | -- | Thinking hurts (-0.090) |
| IBT | -- | Under investigation |

### 4. Process-Shortcutting Is Universal

Across all methods, models learn to eliminate processes that are not directly rewarded:
- **Think-solution**: Think blocks score -0.090 lower than non-thinking
- **Multi-turn**: Discussion provides no rubric benefit (r ~ 0)
- **IBT**: Models shortcut brainstorming to get to the rewarded output faster

This is a fundamental property of reward-based optimization, not a bug in any specific method.

### 5. Reward Function Misalignment

The reward function penalizes length (>750 words) but the grader actually values detail. Within-plan correlation between length and score: Pearson r = 0.075 (negligible). GPT-5.4 achieves rubric = 0.843 without any RL training, just by being a more capable base model. The reward function's length penalty actively suppresses the quality the grader wants.

### 6. Self-Evaluation Cannot Replace Rubric

Self-evaluation correlation with rubric score: r ~ 0.061 (essentially random). The model cannot infer method-specific criteria without seeing the rubric. Self-selection without rubric access is no better than random selection.

---

## Metrics Reference

### Reward Formula

```
reward = rubric_score + 0.08 * length_bonus - format_penalty
```

| Component | Formula |
|-----------|---------|
| rubric_score | 0 -> 0.0, 1 -> 0.2, 2 -> 0.6, 3 -> 1.0 |
| length_bonus | Gaussian centered at 600 words, sigma = 120 |
| format_penalty | 0.2 + 0.0005 * excess_words (if words > 750) |
| Reference solution score | 0.86 mean rubric |

### Grader Prompt Warning

Two grader prompts exist with INCOMPARABLE score scales:
- `build_grader_prompt` (simple) -- used in eval scripts, produces higher scores
- `build_detailed_grader_prompt` (structured XML) -- used in doublegeneration for feedback extraction

Always use the simple grader for scoring comparisons. Detailed grader only for feedback.

---

## Model & Infrastructure

| Parameter | Value |
|-----------|-------|
| Training model | Qwen/Qwen3-30B-A3B |
| LoRA rank | 64 |
| Framework | tinker 0.7.0, PyTorch 2.9, Python 3.13 |
| Batch size | 64 |
| Learning rate | 1e-5 |
| Group size | 8 |
| Max tokens | 2048 (standard), 3072 (async variant) |
| Generation temperature | 1.0 |
| Grading temperature | 0 (greedy) |
| Dataset | facebook/research-plan-gen (ML split) |
| Train goals | ~6,848 |
| Test goals | 685 |

---

## Cross-References

- Phase details: [Phase 1](phases/phase1_baselines.md) | [Phase 2](phases/phase2_bestversion.md) | [Phase 3](phases/phase3_doublegeneration.md) | [Phase 4](phases/phase4_advanced.md)
- Findings: [Selection Gap](findings/selection_gap.md) | [Reward Misalignment](findings/reward_misalignment.md) | [Process Shortcutting](findings/process_shortcutting.md) | [Negative Results](findings/negative_results.md) | [What Works](findings/what_works.md)
- Run artifacts: `projects/<direction>/runs/<year_month>/<method>/<run_name>/`
- Source code: `src/co_scientist/{rubric_reward,ibt,ttt_discover,eval,shared}/`
- Analysis notebooks: `shared/analysis/`
