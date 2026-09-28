# LSE-v1: RL for Revision Capability

## Status: Pivoting from fresh-gen RL to revision RL

## Background

CR-v7 RL was redundant: B4 (no RL) ≥ MAIN (RL) across all metrics. LSE paper (arxiv 2603.18620) inspired initial fork with 4B policy + 30B grader + GRPO advantage on fresh generation. Result: RL +0.027 over frozen baseline (noise level). Fresh generation RL doesn't work.

## Key Insight (2026-04-22)

Current CR success relies on grader scaffolding producing detailed critique text — the model just copy-pastes the feedback. If we remove critique (scores-only revision), the frozen model can't revise well. RL should teach the model to internalize the grading rubric so it can revise effectively with only score feedback.

This aligns with LSE: LSE trains an editing policy (context → better context). We train a revision policy (plan + scores → better plan). Delta reward has causal interpretation in both cases.

## Next Steps (for next session)

1. Modify `train_lse_v1.py`: RL on revision (not fresh gen), scores-only, full-context (no HER)
2. Experiments: RL revision vs frozen revision vs full-critique revision
3. Validate with Opus eval

## Code

- Trainer: `src/co_scientist/grant_proposal/train_lse_v1.py`
- Runs: `projects/grant_proposal/runs/lse_v1/`
