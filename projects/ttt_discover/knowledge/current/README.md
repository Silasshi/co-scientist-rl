# Current Knowledge (Latest Architecture)

This folder contains ONLY the up-to-date documentation. For historical versions, see `../archive/`.

| File | Content |
|---|---|
| **SIGNAL_SET_v8_1.md** | Signal set v8.1-minimal: 8 active signals (S4 disabled), explicit counting rubrics, full prompts, weights, validation results |
| **PIPELINE.md** | Full pipeline description (3 phases, all config flags, design decisions) |

## Quick Reference

- **Model**: Qwen3-30B-A3B (policy = grader = same model)
- **Signals**: 8 active gradient signals (S1-S3, S5-S9) + 2 hard gates (G1, G2). S4 disabled (weight=0, grading skipped). See `SIGNAL_SET_v8_1.md` for full prompts.
- **Weights**: S8 0.18, S9 0.19, S2/S6 0.13, S3/S7 0.12, S5 0.10, S1 0.03, S4 0.00
- **Validation**: 60 refs (mean agg 0.857) + 162 perturbations. AUC P(ref>pert) = 0.767, avg detection 83%
- **Pipeline**: CR-v5 paragraph-level critique-revise + delta RL on revisions
  - 4 fresh plans (buffer context) → grade 8 signals
  - 4 UCB-selected revision targets → identify bottleneck signal → paragraph edit (best-of-2) → grade → delta advantage
  - RL update: importance_sampling loss on positive-delta revisions only
- **Grader**: separate_call mode, N=2 repeats, max_tokens=4096
- **Code**: `src/co_scientist/ttt_discover/train_critique_revise.py`
- **Experiment plan**: `projects/ttt_discover/paper_experiments/EXPERIMENT_PLAN.md`
