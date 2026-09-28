# Phase 1: Baseline Exploration (January 2026)

## Overview

16 exploratory runs testing different model architectures, reward formulations, and task configurations. The goal was to identify the best model and training setup for research plan generation.

## Models Tested

| Model | Outcome |
|-------|---------|
| Qwen3-30B-A3B | Best overall. Selected as primary model going forward. |
| gpt-oss-20b | Tested, inferior to Qwen3 on this task. |
| Llama-3.1-8B | Tested, inferior to Qwen3 on this task. |

## Key Runs

### success_dense_score

- **Model**: Qwen3-30B-A3B
- **Batches**: 189
- **Reward**: 0.587
- **Significance**: First major training run. Established dense scoring as the preferred reward formulation for research plan generation.

### 25(gsm8k_qwen3_sparse_score)

- **Model**: Qwen3-30B-A3B
- **Batches**: 23
- **Reward**: 0.961
- **Task**: GSM8K (math), NOT research plan generation
- **Significance**: Demonstrated that sparse scoring works well on tasks with clear correctness signals (math), but dense scoring is needed for the more nuanced research plan task.

## Key Decisions Made

1. **Model selection**: Qwen3-30B-A3B chosen as the training model for all subsequent experiments.
2. **Reward formulation**: Dense scoring preferred over sparse scoring for research plan generation.
3. **Task focus**: ML split of facebook/research-plan-gen selected as the primary training dataset.

## Lessons Learned

- Model scale matters: 30B-parameter models significantly outperformed 8B on this task.
- Dense vs sparse scoring: For open-ended generation tasks like research plans, dense rubric-based scoring provides much better training signal than binary correctness.
- The 7-desiderata rubric provides a sufficiently rich reward signal for GRPO training.

## Run Artifacts

- Run directory: `runs/2026/1/`
- 16 runs total in this phase

## Next Phase

Results from Phase 1 led directly to Phase 2, where bestversion (A1+A2 improvements) was developed on top of the Qwen3-30B-A3B + dense scoring foundation.

See: [Phase 2: bestversion](phase2_bestversion.md) | [Back to Catalog](../EXPERIMENT_CATALOG.md)
