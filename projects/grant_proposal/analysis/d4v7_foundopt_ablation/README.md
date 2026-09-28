# D4-v7 FoundOpt Ablation Study

Date: 2026-04-20
Goal: FoundOpt (AI Optimization Theory)
Signal set: D4-v7 (12 hybrid signals)
Model: Qwen3-30B-A3B

## Runs

| Run | Config | Description |
|---|---|---|
| MAIN | buffer + critique-revise + fresh RL + revise RL | Full pipeline |
| B4_no_training | buffer + critique-revise, `skip_rl_update=true` | In-context only, no RL gradient |
| A_fresh_only | buffer + fresh RL, `n_revise=0` | Fresh exploration only, no critique-revise |
| B1_zero_shot | no buffer, no RL, no critique | Base model zero-shot generation |

## Directory Structure

```
data/
  metrics_MAIN.jsonl          Per-iteration metrics (reward, signals, timing)
  metrics_B4_no_training.jsonl
  metrics_A_fresh_only.jsonl
  best_plan_MAIN.txt          Best plan text from each run's buffer
  best_plan_B4_no_training.txt
  best_plan_A_fresh_only.txt
  ablation_summary.json       Combined summary (best/median/worst per run)
  opus_eval_results.jsonl     Opus 4.7 depth evaluation (when available)

figures/                      Generated plots for paper
notebooks/                    Analysis notebooks
```

## Key Results (preliminary, B1 still running)

| Run | buf_max | G6 depth | G11 rigor | G12 formal | G13 risk |
|---|---|---|---|---|---|
| MAIN | **0.855** | **5** | **4** | **4** | 4 |
| B4 (no RL) | 0.820 | 5 | 4 | 4 | **5** |
| A_fresh (no revise) | 0.625 | 4 | **2** | **1** | 3 |
| B1 (zero-shot) | TBD | | | | |

## Findings

1. **Critique-revise is essential for depth signals.** A_fresh: G11=2, G12=1 after 25 iters. MAIN/B4: G11=4, G12=4. Without per-signal critique, the model cannot learn to write baselines or formulas.

2. **RL adds marginal value over in-context.** MAIN buf_max=0.855 vs B4=0.820 (Δ=+0.035). D3's finding (B4 ≥ MAIN) partially replicated.

3. **Hard gates reject more plans over time.** MAIN had several iters with fresh=0.000 (all plans rejected by goal-contrast margin), suggesting exploration narrowing.
