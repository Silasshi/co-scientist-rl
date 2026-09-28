# Direction 3: TTT-Discover (Per-Goal Optimization with Multi-Signal Reward)

**Status**: Signal set v8.1-minimal validated. Paper experiments in progress.
**Period**: April 2026 onwards
**Philosophy**: Per-goal optimization (overfit to one problem at a time), NOT generalist training
**Model**: Qwen3-30B-A3B (policy = grader = same model)
**Training**: CR-v5 signal-targeted critique-revise + delta-weighted RL on revisions

## Overview

Replace the goal-specific rubric reward (D1) with a UNIVERSAL multi-signal reward that works for any research field. Each signal evaluates a different dimension of plan quality through structured grader prompts (1-5 integer scale, separate grader call per signal to minimize halo).

## Current Signal Design (v8.1-minimal)

**Full documentation with prompts**: `knowledge/current/SIGNAL_SET_v8_1.md`

**Layer 0 — Hard Gates** (programmatic):
- G1. Goal-Contrast Margin — plan must be specific to the goal
- G2. Claim Verification — factual claims cannot be fabricated

**Layer 1 — Gradient Signals** (8 active, 1 disabled):

| Signal | Weight | Detection | Method |
|--------|--------|-----------|--------|
| S9_focus | **0.19** | 88% | Core Idea technique count + justification |
| S8_scope | **0.18** | 94% | Overclaim trigger words vs domain buckets |
| S2_rigor | 0.13 | 100% | FAIR/STRAWMAN baseline classification |
| S6_risk_awareness | 0.13 | 88% | Failure modes + scope boundaries |
| S3_positioning | 0.12 | 61% | Named prior methods + insufficiency |
| S7_specificity | 0.12 | 93% | Vague markers vs specific commitments |
| S5_feasibility | 0.10 | 83% | Core algorithm + dependency naming |
| S1_depth | 0.03 | 55% | Load-bearing choice justification ratio |
| ~~S4_significance~~ | ~~0.00~~ | ~~0%~~ | ~~Disabled — grader can't isolate sections~~ |

**Validation** (60 refs + 162 perturbations, N=5 repeats):
- Ref mean aggregate: 0.857
- AUC P(ref > pert): 0.767
- Average detection rate: 83% (excluding S4/S5)

## Training Method: CR-v5

Per-iteration loop on a single target goal:
1. **Fresh generation** (4 plans with buffer context)
2. **Critique-revise** (4 buffer entries selected via UCB → identify bottleneck signal → paragraph-level edit → best-of-2)
3. **Delta RL** (positive-delta revisions → importance_sampling loss update)

Key: `train_on_fresh=False` — fresh plans populate the buffer but don't contribute RL gradients. Only revision delta drives policy learning.

## Code

| File | Purpose |
|------|---------|
| `src/co_scientist/ttt_discover/train_critique_revise.py` | CR-v5 trainer (current) |
| `src/co_scientist/ttt_discover/train_buffer_ttt.py` | Buffer-TTT base (shared components) |
| `src/co_scientist/shared/ten_signal_reward.py` | Signal definitions, weights, prompts, parsing |
| `src/co_scientist/shared/seven_signal_reward.py` | Hard gates (G1, G2) |

## Project Structure

```
projects/ttt_discover/
├── README.md                  ← this file
├── STATUS.md                  ← live state (read first!)
├── DECISIONS.md               ← append-only decision log
├── CONVENTIONS.md             ← naming + placement rules
├── FOCUSED_PLAN_2026.md       ← active research plan
├── knowledge/
│   ├── current/
│   │   ├── SIGNAL_SET_v8_1.md ← signal system with full prompts
│   │   └── PIPELINE.md        ← CR-v5 pipeline architecture
│   └── archive/               ← historical signal designs, methods
├── analysis/
│   ├── signal_validity/       ← perturbation-based validation (main)
│   │   ├── data/{refs,perturbations,human_ratings}/
│   │   ├── scripts/{build,grade,analyze}/
│   │   ├── reports/{v6,v7,v8,v8_1}/    ← v8_1 is canonical current
│   │   ├── logs/              ← grading-run logs
│   │   └── archive/           ← pre-v6 grading data
│   ├── sanity_check/          ← early signal validation (v1/v2)
│   ├── qualitative_review/    ← hand-curated plan samples (renamed from signal_validity_check)
│   ├── grader_comparison/     ← 30B vs 235B grader comparison
│   ├── dimension_validation/, diversity_test/, multi_goal/
│   └── _archive/
│       └── phase_a0_jan2026/  ← superseded Phase A.0 artifact
├── paper_experiments/
│   ├── EXPERIMENT_PLAN.md     ← MAIN + B1-B4 + A1-A9 matrix
│   ├── launch_all.sh          ← v5 launcher (archived)
│   └── launch_all_v81.sh      ← v8.1 launcher (current)
├── runs/                      ← training run data
│   ├── 2026_04_30b_paper_experiments/      ← v5 signals (kept for comparison)
│   ├── 2026_04_30b_paper_experiments_v81/  ← v8.1 signals (current)
│   ├── 2026_04_4b_paper_experiments/       ← 4B model runs
│   └── _archive/
│       ├── 2026_04_buffer_ttt/             ← pre-CR-v5 method exploration
│       └── 2026_04_7signal/                ← early 7-signal GRPO runs (v1 signals)
├── configs/                   ← (reserved)
├── data/                      ← (reserved)
└── tools/
    └── _archive/              ← phase_a0 sanity scripts (superseded)
```

## Key Results

| Method | buf_max | Signal version |
|--------|---------|----------------|
| Zero-shot (B1) | 0.790 | v8.1 |
| Buffer-TTT entropic | 0.565 | v1 |
| CR-v5 (MAIN, v5 signals) | 0.778 | v5 |
| CR-v5 (MAIN, v8.1, corrupted run) | 0.833 | v8.1 (18% parse failure) |
| Reference plans | 0.857 | v8.1 |
