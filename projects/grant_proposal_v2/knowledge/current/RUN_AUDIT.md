# D4 Run Audit

*Source: 58 runs in `projects/grant_proposal/runs/`, 2026-04-20 to 2026-04-22.*
*Structured data: `analysis/run_classification/all_runs_config_table.jsonl`*

## Summary

| Signal Set | Total | Complete | Incomplete | No Data |
|---|---|---|---|---|
| D4-v7 (G1-G13, 1-5 scale) | 38 | 30 | 6 | 2 |
| Better Reward (R1-R8, 1-10 scale) | 9 | 4 | 3 | 2 |
| Legacy (v5/v6) | 7 | 4 | 1 | 2 |

⚠️ **Runs with different signal sets are NOT directly comparable on Opus eval.**

## Configuration Dimensions

### 1. Signal Set
- **D4-v7 standard**: G1-G13, 12 signals, 1-5 scale. Uses `dataset/goals/01_foundopt/` etc.
- **Better Reward**: R1-R8, 8 signals, 1-10 scale. Uses `dataset/goals/01_foundopt_v2/` with `custom_signal_module=co_scientist.shared.foundopt_rubric_v1`.
- **Legacy v5/v6**: Early prototypes, missing goal_dir in some configs. Not usable for comparison.

### 2. Critique Mode
| Mode | Config | Effect |
|---|---|---|
| full_critique | `scores_only=F, strip_critiques=F` | Model sees signal name + score + critique text |
| scores_only | `scores_only=T` | Model sees signal name + score + question only |
| strip_critiques | `strip_critiques=T` | Model sees only aggregate score |

### 3. RL Method
| Method | Config | D4-v7 complete runs |
|---|---|---|
| B4 (no RL) | `skip_rl_update=T` | 20 |
| MAIN (full RL) | `train_on_fresh=T, skip_revision_rl=F` | 7 |
| SDPO | `sdpo=T` | 1 |
| Aggregate RL | `revision_rl_mode=aggregate` | 1 |
| GAPO diversity | `diversity_method=gapo` | 1 |

### 4. Model Size
| Setup | Runs |
|---|---|
| 30B policy + 30B grader | ~48 |
| 235B policy + 235B grader | ~6 (Better Reward only) |

### 5. Other Variations
- `include_cot_scaffolding`: True (default) vs False (1 ablation run)
- `use_retrieval`: True (3 RAG experiments) vs False (default)
- `novelty_weight`: 0.0 (default) vs 0.3 (goals 05, 06)
- `grader_repeats`: 1 or 2

## Comparable Run Pairs (same signal set, same goal, one variable changed)

### Critique Effect (D4-v7, FoundOpt, 30B, B4)
| Run | Critique | Qwen buf_max | Opus |
|---|---|---|---|
| b4_paper_01_foundopt | full | 0.970 | 19 |
| sdpo_C2_B4_scores_only | scores_only | 0.855 | 19 |
| b4_no_scaffold_foundopt | full, no CoT | ? | — |

### RL Effect (D4-v7, FoundOpt, 30B, scores_only)
| Run | RL Method | Qwen | Opus |
|---|---|---|---|
| sdpo_C2 | B4 (none) | 0.855 | 19 |
| sdpo_C4 | Aggregate | 0.940 | 15 |
| sdpo_C3 | SDPO | 0.970 | 13 |

### RL Effect (D4-v7, FoundOpt, 30B, full_critique)
| Run | RL Method | Qwen | Opus |
|---|---|---|---|
| d4v7_ablation_B4 | B4 | ~0.82 | — |
| d4v7_01_foundopt_MAIN | MAIN | ~0.97 | — |
| gapo_foundopt | GAPO | ~1.00 | 17 |
| d4v7_A_fresh_only | Fresh only | ~0.63 | 13 |

### Model Scale (Better Reward, FoundOpt, full_critique, B4)
| Run | Model | Qwen | Opus |
|---|---|---|---|
| better_reward_B4 | 235B | 0.953 | 29 |
| 30b_scores_only_B4 | 30B | 0.789 | 12 |
*⚠️ Different critique mode (full vs scores_only) confounds this comparison.*
