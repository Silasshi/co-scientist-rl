# evolving_rubric/data

| Path | Contents |
|---|---|
| [`runs.csv`](runs.csv) | the three runs and how they differ |
| `runs/<name>/rubric_buffer.jsonl` | **the evolved rubric items**: title, description, positive or negative, weight, creation iteration, and the per-iteration binary grades of every rollout on that item |
| `runs/<name>/rubric_anchor_log.jsonl` | drift diagnostic: the reference proposal graded against the active rubric |
| `runs/<name>/metrics.csv`, `iter_summary.csv`, `config.json` | per-iteration training metrics (`r_active/*` once the rubric is coupled into RL) and the exact config |
| [`pilot/RESULTS_SUMMARY.md`](pilot/RESULTS_SUMMARY.md) | offline pilot: 3 contrast pairs → 9 generated items, checked against pre-set success criteria |
| `goals/01_foundopt/weights.json` | signal weights for the goal used |

The runs are **mechanism tests**. They have 3–10 iterations on one grant-proposal goal, and no long training run or held-out evaluation exists.

`smoke_no_rl` was resumed mid-run, so `metrics.csv` repeats some `iter` values.

Excluded:
- The goal text (`research_goal.md`) and distractor goals (`alt_goals.json`), whose provenance is not ours to republish.
- The reference proposal.
- Plan buffers.

To run the trainer, place your own `research_goal.md` (and optionally `alt_goals.json` = `{"alt_goals": [{"goal": ..., "target": ...}]}`) in `goals/<name>/`.
