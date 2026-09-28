# Direction 1: Rubric Reward

**Paper**: "Training AI Scientists Using Rubric Rewards"
**Status**: Best result 0.693 (bestversion). Performance ceiling reached after 80+ runs.
**Period**: January - March 2026

## Overview

Single-scalar rubric grader + GRPO variants. The grader uses 10 goal-specific rubric items to score a plan on 7 desiderata. Training via GRPO maximizes this scalar reward.

## Key results

| Method | Eval Rubric | Status |
|---|---|---|
| bestversion (GRPO) | **0.693** | Best RL result |
| SDPO (20 runs) | < 0.693 | Negative |
| Reward shaping variants | < 0.693 | Negative |
| Doublegeneration | 0.654 | Negative (OOD collapse) |
| Multi-turn V1-V4 | < 0.693 | Negative (process shortcutting) |
| Rubric dropout | 0.657 | Negative |
| Self-selector | Not trained | Implemented |

## Key findings

- Selection gap = 91% of total gap (mean 0.70, oracle best-of-8 0.85, reference 0.86)
- 80.6% of rubric failures check reference solution's specific methodology (goal-specific, not universal)
- Process shortcutting is universal across all multi-step methods

## Code

- `src/co_scientist/rubric_reward/grpo/` — GRPO trainers (bestversion, CPR, RCGRPO, think-solution)
- `src/co_scientist/rubric_reward/sdpo/` — SDPO trainer
- `src/co_scientist/rubric_reward/reward_shaping/` — Reward function variants
- `src/co_scientist/rubric_reward/multiturn/` — Multi-turn V1-V4
- `src/co_scientist/rubric_reward/refinement/` — Doublegeneration, blended, rubric dropout
- `src/co_scientist/rubric_reward/selector/` — Self-selector pairwise ranker

## Runs

Historical runs in `runs/2026/` (organized by month). Key runs:
- `runs/2026/2/withA1,A2/2(ml)/` — bestversion (215 batches, eval 0.693)
