# D1 Rubric Reward — Run Index

## Naming convention (for new runs)

```
{method}_{short_description}/
```
- method: lowercase method name (bestversion, sdpo, multiturn_v2, etc.)
- short_description: what's unique about this run (a1a2_ml, warmstart, scale09, etc.)
- Inside each run: config.json, train/, evaluation/, and a brief RUN_NOTES.md

## Run catalog

### Phase 1: Baselines (2026_01)

| Run | Method | Key change | Result | Notes |
|---|---|---|---|---|
| `2026_01/baseline_exploration/21(1)` | GRPO baseline | First baseline run | — | Exploration |
| `2026_01/baseline_exploration/22(1-7)` | GRPO baseline | 7 config variants | — | Settled on Qwen3-30B-A3B |
| `2026_01/baseline_exploration/23(*),29,30` | GRPO baseline | Further exploration | — | |
| `2026_01/dense_score/success_dense_score` | Dense score | Dense reward signal | — | |
| `2026_01/gsm8k/*` | GSM8K | Math task baseline | — | Different task, not plan generation |

### Phase 2: bestversion + SDPO (2026_02)

| Run | Method | Key change | Result | Notes |
|---|---|---|---|---|
| **`2026_02/withA1,A2/2(ml)`** | **bestversion** | **A1+A2 improvements, ML data** | **eval 0.693** | **BEST RESULT. 215 batches.** |
| `2026_02/withA1,A2/3(arxiv)` | bestversion | Same config, arxiv data | < 0.693 | |
| `2026_02/withA1,A2/3(pubmed)` | bestversion | Same config, pubmed data | < 0.693 | |
| `2026_02/SDPO/20-28` | SDPO | 15 SDPO variants | All < 0.693 | SDPO failed to outperform |
| `2026_02/hard_min/14` | Reward shaping | Hard min reward | < 0.693 | |
| `2026_02/0-9_scale/15(*)` | Reward shaping | 0-9 scale | < 0.693 | |
| `2026_02/std+mean_weighted/*` | Reward shaping | Std/mean weighted | < 0.693 | |
| `2026_02/std_stand+band_bonus/*` | Reward shaping | Band bonus | < 0.693 | |

### Phase 3: Advanced techniques (2026_03)

| Run | Method | Key change | Result | Notes |
|---|---|---|---|---|
| `2026_03/refinement/7,7(2),8` | Doublegeneration | Two-stage: generate→feedback→refine | eval 0.654 | Negative: Stage 2 OOD collapse |
| `2026_03/blended/1,18` | Blended gen | Blended generation | < 0.693 | |
| `2026_03/rubric_dropout/13` | Rubric dropout | Drop rubric items | eval 0.657 | Negative: split signal |
| `2026_03/multiturn/run1,run4` | Multi-turn V1 | Multi-turn dialogue | < 0.693 | Process shortcutting |
| `2026_03/multiturn_v2/run1` | Multi-turn V2 | Improved multi-turn | < 0.693 | Process shortcutting |
| `2026_03/multiturn_v3/run1,run2` | Multi-turn V3 | Further improved | < 0.693 | Process shortcutting |
| `2026_03/multiturn_v4/*` | Multi-turn V4 | Final multi-turn | < 0.693 | Process shortcutting |
| `2026_03/eval_bon_base_model` | Eval only | Base model BON eval | — | |

### Phase 4: Late-stage experiments (2026_04)

| Run | Method | Key change | Result | Notes |
|---|---|---|---|---|
| `2026_04/best_ver_async/1-4` | bestversion-async | Async + longer plans | — | |
| `2026_04/cpr/1` | CPR | Pairwise ranking | 12 batches | Preliminary |
| `2026_04/think_solution/1` | Think-solution | Add thinking step | -0.090 | Negative: thinking hurts |
| `2026_04/eval_sota/*` | Eval only | GPT-5.4 eval | 0.843 | External upper bound |
| `2026_04/eval_rag/*` | Eval only | RAG eval | — | |
