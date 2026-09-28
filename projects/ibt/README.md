# Direction 2: IBT (Iterative Brainstorming Training)

**Status**: Run 7 completed (106 batches). Results under investigation.
**Period**: April 2026
**Policy**: Qwen3-4B-Instruct-2507 (smaller model)
**Grader**: Qwen3-30B-A3B

## Overview

Iterative dialogue-style training where the model generates a plan, receives feedback, and refines. Uses a smaller 4B policy with a 30B grader (unlike rubric_reward which uses 30B for both).

## Code

- `src/co_scientist/ibt/train_ibt.py` — Main IBT trainer
- `src/co_scientist/ibt/train_ibt_v2.py` — IBT v2 variant
- `src/co_scientist/ibt/baseline_grpo.py` — GRPO baseline for comparison
- `src/co_scientist/ibt/baseline_training_free.py` — Training-free baseline
- `src/co_scientist/ibt/train_self_calibration.py` — Self-calibration variant

## Runs

- `runs/2026/4/ibt/7/` — Only completed IBT run (106 batches)
