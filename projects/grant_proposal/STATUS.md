# Grant Proposal Generation STATUS

*Last updated: 2026-04-22*

> **See also**: `projects/grant_proposal_v2/` — clean investigation workspace with structured audit of all runs and findings.

## Current Phase

Legacy. 58 runs complete, findings documented. Active work moved to D4v2.

## Key Findings (2026-04-21)

1. **Qwen coarse ranking ρ=+0.908 with Opus** — reward IS directionally correct. Previous ρ=-0.16 was within-condition fine ranking only (saturation at high scores).
2. **Template collapse is the primary RL failure** — not reward misspecification. RL concentrates policy on single template.
3. **Standard rubric design is good** — multi-prompt strict variants all performed WORSE than standard on 9-plan validation.
4. **GAPO diversity RL**: NN-distance-based diversity bonus works mechanically (penalizes duplicates, boosts unique plans), but mid-training Opus eval shows 7/20 (needs full run + fair comparison).
5. **B4 (critique-revise, no RL) is the strongest method** across all 6+ goals tested.

## Active Experiments

| Experiment | Status | Goals |
|---|---|---|
| B4 paper runs (6 goals) | **Complete** | 01_foundopt, 05_causal_healthcare, 07_chemo_toxicity, 08_ecosystem_dynamics, 09_sentencing_disparities, 12_climate_displacement |
| **SDPO scores-only revision RL** | **Code complete, awaiting launch** | 01_foundopt |

## Paper Configuration (unified B4)

```
skip_rl_update: True, train_on_fresh: False, grader_repeats: 1
cold_start_iters: 0, n_iterations: 10, n_fresh: 4, n_revise: 4
skip_hard_gates: False, emit_signal_critique: True, seed: 0
```

Per-goal rubric differences:
- AI goals (01, 08): default D4-v7 (12 signals, G12_formalism)
- Healthcare (05): G3↑0.12, G12↓0.10
- Biomedical (07): G12→G12a, G3↑0.14, G4↓0.08
- Criminal justice (09): G12→G12a
- Climate policy (12): G12→G12a, G5↑0.08, G4↓0.06, G13↑0.12

## Target

NeurIPS 2026

## Historical Results

### FoundOpt Ablation (D4-v7, 25 iters)

| Run | buf_max | Opus best /20 |
|---|---|---|
| B4 (no RL) | 1.000 | 16 |
| MAIN (full RL) | 0.970 | 11 |
| A_fresh (no revise) | 0.625 | 7 |
| GAPO diversity RL | 0.975 | 7 (mid-training, iter 7-8) |
| Reference | — | 20 |

### Cross-condition Qwen-Opus Correlation

- **Cross-condition** (Qwen 0.34-0.855): ρ=+0.908 (p=0.001)
- **Within-condition** (Qwen 0.78-0.82): ρ=-0.16 (saturation)
- Multi-prompt strict variants: ALL worse than standard (G12 ρ 0.902→0.552)

### Better Reward Experiment (2026-04-20/21)

8 conditions tested (30B/235B × full_critique/scores_only × MAIN/B4). Template collapse confirmed as primary RL failure mode. See `analysis/d4_better_reward_vs_d3_v9.ipynb`.

## LSE-v1: RL Trainer (abandoned 2026-04-22)

Fresh-gen RL showed +0.027 over frozen (noise level). Direction abandoned.

## SDPO Revision RL (2026-04-22, active)

**Status**: Code complete in `train_cr_v7.py`. SDPO mode + aggregate REINFORCE mode added.

**Goal**: Teach model to revise plans from scores-only feedback (no critique text). Use SDPO (Hübotter et al. 2026) for dense per-token credit assignment via self-teacher distillation.

**Mechanism**: Student sees scores-only → generates revision. Self-teacher re-evaluates same revision with full critique context. Per-token advantage = log(teacher_prob / student_prob). Plugs into existing `importance_sampling` loss.

**Config**: `sdpo=True, scores_only=True, skip_rl_update=False, train_on_fresh=False`

**Experiment conditions**:
- C1: B4-full-critique (upper bound, exists)
- C2: B4-scores-only (lower bound, to launch)
- C3: SDPO-scores-only (core experiment, to launch)
- C4: GRPO-aggregate-scores-only (ablation, to launch)

**Design doc**: `~/.claude/plans/quirky-herding-emerson.md`
**Design doc**: `knowledge/current/LSE_V1_DESIGN.md`

## Dataset

12 goals total: 6 AI/ML + 2 biomedical + 1 ecology + 2 social science + 1 climate policy. All have reference proposals + per-goal weight configs in `dataset/goals/`.
