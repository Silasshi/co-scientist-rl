# D4v2: Grant Proposal Generation — Investigation Phase

Continuation of D4 (`projects/grant_proposal/`). Clean workspace for diagnosing the Qwen-Opus quality inversion discovered in SDPO experiments (2026-04-22).

## Background

D4 explored grant proposal generation with CR-v7 pipeline + D4-v7 signal set across 58 runs. Key outcome: B4 (no RL) achieves 19/40 on Opus, while all RL methods (SDPO, aggregate, GAPO, MAIN) score 13-17. RL consistently Goodharts on Qwen 30B reward.

## This Directory

- `knowledge/current/` — Audit of all D4 runs, established findings, investigation questions
- `analysis/run_classification/` — Structured config data for all 58 runs + Opus eval scores
- `analysis/sdpo_experiment/` — SDPO experiment results + 10-judge pairwise comparison
- `dataset/` → symlink to D4 dataset (12 goals)
- `runs/` — Future experiments go here
- `configs/` — Future configs go here

## Code

Shared with D4: `src/co_scientist/grant_proposal/` (train_cr_v7.py, train_buffer_ttt.py, etc.)

## Target

NeurIPS 2026.
