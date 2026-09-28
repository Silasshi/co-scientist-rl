# D2 IBT — Run Index

## Run catalog (2026_04)

All IBT runs used: Policy = Qwen3-4B-Instruct-2507, Grader = Qwen3-30B-A3B

| Run | Variant | Key change | Result | Notes |
|---|---|---|---|---|
| `2026_04_ibt/1/mini_grpo` | Mini GRPO | Small-scale GRPO test | — | |
| `2026_04_ibt/1/single_chain` | Single chain | Single conversation chain | — | |
| `2026_04_ibt/2/baseline_*` | Baselines | 4 baseline variants (grpo, regenerate, tf_accumulated, tf_latest) | — | |
| `2026_04_ibt/2/mini_grpo*` | Mini GRPO | GRPO variants | — | |
| `2026_04_ibt/2/single_chain*` | Single chain | Chain variants (with/without hint) | — | |
| `2026_04_ibt/3/baseline_tf_*` | Training-free | Accumulated top-k variants | — | |
| `2026_04_ibt/3/tf_acc_*` | Training-free | 10/20 turn variants | — | |
| `2026_04_ibt/5/*` | IBT v5 | Multiple variants | — | |
| `2026_04_ibt/6/*` | IBT v6 | Dense GRPO, self-distill | — | |
| **`2026_04_ibt/7`** | **IBT v7** | **Final IBT run** | **106 batches** | **Only completed run. Under investigation.** |
