# ttt/data

Per-goal test-time training runs on one research goal ([`research_goal.txt`](research_goal.txt), our own description of the TTT-Discover problem). Each run has 25 iterations.

| File | Contents |
|---|---|
| [`runs.csv`](runs.csv) | run name → original run dir → what differs |
| `runs/<name>/metrics.csv` | per iteration: `reward/buffer_max` (best plan so far, training grader), fresh / revision reward means, per-signal means, β of the entropic weights |
| `runs/<name>/iter_summary.csv` | per-iteration training summary |
| `runs/<name>/config.json` | exact config of the run |
| [`opus_judge_scores.csv`](opus_judge_scores.csv) | independent judge (Claude Opus) score of each run's best plan: /20 total, with 4 sub-scores out of 5 |

The training-grader score is the 10-signal reward from [`../rewards/ten_signal_reward.py`](../rewards/ten_signal_reward.py), graded by Qwen3-30B-A3B.

**Caveat:** the Opus scores are **one judgment of one plan per run**, recorded by hand in the analysis notebook and decision log. The `source` column says where each was recorded. Treat differences of 1–2 points as noise.

Excluded: plan buffers (plan text), the reference plan.
