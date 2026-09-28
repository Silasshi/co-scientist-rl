# EXPERIMENTS

*Last updated: 2026-04-06*

## Complete Method Catalog

### Core Methods

Source of truth for rubric-reward methods: `src/co_scientist/rubric_reward/`
Source of truth for IBT methods: `src/co_scientist/ibt/`

| Family | Script | Description | Best Result |
|--------|--------|-------------|-------------|
| **bestversion** | `rubric_reward/grpo/best_ver.py` | Single-stage GRPO, 8 samples/goal | **rubric 0.693** |
| bestversion-async | `rubric_reward/grpo/best_ver_async.py` | Async variant, longer plans | reward 0.778 |
| SDPO | `rubric_reward/sdpo/train_sdpo.py` | Self-distillation + GRPO hybrid | < bestversion |
| doublegeneration | `rubric_reward/refinement/train_double_generation.py` | Two-stage generate-refine | rubric 0.654 |
| blended generation | `rubric_reward/refinement/train_blended_generation.py` | Relabel refinements as initial | Inconclusive |
| rubric dropout | `rubric_reward/refinement/best_ver_rubric_dropout.py` | Stochastic rubric visibility | rubric 0.657 |
| multi-turn V1-V4 | `rubric_reward/multiturn/train_multiturn_v*.py` | Multi-turn dialogue GRPO | All < bestversion |
| IBT | `ibt/train_ibt.py` | Iterative brainstorming with hints | Under investigation |
| self-calibration | `ibt/train_self_calibration.py` | Calibration-weighted GRPO | Not yet run |
| CPR | `rubric_reward/grpo/train_cpr.py` | Contrastive pairwise ranking | 12 batches |
| RC-GRPO | `rubric_reward/grpo/train_rcgrpo.py` | Rubric-conditioned GRPO | Not yet run |
| self-selector | `rubric_reward/selector/train_selector.py` | Pairwise plan selection | Not yet trained |
| think-solution | `rubric_reward/grpo/train_think_solution.py` | Separate think/solution grading | Thinking hurts |
| rubric predictor | `rubric_reward/grpo/train_rubric_predictor.py` | Predict rubric from goal | Utility for RC-GRPO |
| weighted (mean) | `rubric_reward/reward_shaping/train_weighted_mean.py` | Mean-weighted reward | reward 0.787 |
| weighted (std) | `rubric_reward/reward_shaping/train_weighted_std.py` | Std-weighted reward | reward 0.819 |
| weighted (dense) | `rubric_reward/reward_shaping/train_weighted_dense_score.py` | Per-desideratum reward | reward 0.587 |
| hard min | `rubric_reward/reward_shaping/train_hard_min.py` | Hard minimum constraint | reward 0.618 |
| scale 0-9 | `rubric_reward/reward_shaping/train_scale_0_9.py` | 0-9 rubric scale | reward 0.760-0.802 |

### Evaluation Scripts

All eval scripts live under `src/co_scientist/eval/`.

| Script | Purpose |
|--------|---------|
| `shared/eval_core.py` | Single-pass eval with simple grader (shared utility) |
| `eval/eval_double.py` | Two-stage eval (generate + refine) |
| `eval/eval_bon.py` | Best-of-N with self-selection |
| `eval/eval_cpr.py` | CPR pairwise ranking eval |
| `eval/eval_rag.py` | RAG with few-shot examples |
| `eval/eval_sota.py` | SOTA LLM eval via OpenRouter |
| `eval/eval_reference.py` | Grade reference solutions |
| `eval/eval_agnostic_grader.py` | Methodology-agnostic grader validation |
| `eval/eval_ibt.py` | IBT single-pass or iterative eval |
| `eval/eval_selector.py` | Selector tournament eval |
| `eval/eval_multiturn.py` | Multi-turn eval |

### Dataset-Specific Baselines

| Script | Dataset |
|--------|---------|
| `baselines/train_baseline_best.py` | Multi-dataset (ML + ArXiv + PubMed) |
| `baselines/train_baseline_arxiv.py` | ArXiv only |
| `baselines/train_baseline_pubmed.py` | PubMed only |
| `baselines/train_baseline_gsm8k.py` | GSM8K (math reasoning) |
| `baselines/train_baseline_sparse_qwen3.py` | Sparse reward variant |

## Run Directory Structure

Each run in `projects/<direction>/runs/<year_month>/<method>/<run_name>/` should contain:

| File | Required | Description |
|------|----------|-------------|
| `config.json` | Yes | Full hyperparameter configuration |
| `code.diff` | Yes | Code state at run time |
| `logs.log` | Yes | Execution logs |
| `metrics.jsonl` | Recommended | Per-batch training metrics |
| `checkpoints.jsonl` | Optional | Checkpoint metadata |
| `train/training_logs.jsonl` | Recommended | Per-sample training details |
| `train/batch_summary.jsonl` | Recommended | Aggregate batch metrics |
| `evaluation/` | Optional | Evaluation results |

## Standard Config Template

```json
{
  "model_name": "Qwen/Qwen3-30B-A3B",
  "lora_rank": 64,
  "batch_size": 64,
  "learning_rate": 1e-5,
  "group_size": 8,
  "max_tokens": 2048,
  "temperature": 1.0,
  "grader_temperature": 0.0,
  "grader_max_tokens": 8192,
  "save_every": 30
}
```

## Reward Formula

```
reward = rubric_score + 0.08 * length_bonus - format_penalty
```
- rubric_score: {0→0.0, 1→0.2, 2→0.6, 3→1.0}
- length_bonus: Gaussian(center=600, σ=120)
- format_penalty: 0.2 + 0.0005×excess for words > 750

## Add New Experiment: Procedure

1. Create script under `src/co_scientist/<direction>/` (e.g., `rubric_reward/`, `ibt/`, `ttt_discover/`)
2. Create run folder: `projects/<direction>/runs/<year_month>/<method>/<run_name>/`
3. Save `config.json` and capture `code.diff` before running
4. Execute training, persist logs/metrics in run folder
5. Run `python3 shared/tools/index_runs.py`
6. Document conclusions in `shared/docs/STATUS.md`
7. Add entry to `shared/knowledge/EXPERIMENT_CATALOG.md`

## Metrics Reference

Common metrics in `metrics.jsonl`:
- `reward/total` — main reward metric
- `progress/batch` — batch number
- `optim/lr` — learning rate
- `progress/done_frac` — epoch fraction
- `time/total` — elapsed time

Common metrics in `train/batch_summary.jsonl`:
- `rubric_mean`, `reward_mean` — per-batch averages
- `format_compliance` — format pass rate
- `valid_rate` — sample validity
- Advantage statistics (mean/std)
