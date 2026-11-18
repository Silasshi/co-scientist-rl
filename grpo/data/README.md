# grpo/data

All numeric, no plan or critique text. The source is the cleaned numeric layer of the project's run logs. Run names map to the original run ids in [`runs.csv`](runs.csv).

| Folder | Contents | Granularity |
|---|---|---|
| `train_metrics/<name>.csv` | training reward per batch (`reward/total`, drops, lr); `.batch_summary.csv` adds advantage / length / format stats | 1 row per batch |
| `train_samples/<name>.parquet` | `batch_idx, group_idx, sample_idx, rubric_score, format_penalty, final_reward` | 1 row per **valid** training sample |
| `eval/<name>.<tag>.parquet` | held-out eval per sample (`rubric_score`, `word_count`, `is_compliant`; BoN files add `self_eval_score`, `is_oracle_pick`, `is_self_pick`) | 1 row per **valid** eval sample |
| `eval/<name>.<tag>_summary.csv` | per eval batch: `samples/generated`, `samples/valid`, `rubric/sample_mean_all`, … | 1 row per eval batch |

**Two eval averages.**
- The headline scores (e.g. base 0.560, trained 0.693) are the rubric mean over **all generated samples**, including format-invalid ones (`rubric/sample_mean_all`, weighted by `samples/generated`).
- The per-sample parquet files keep **valid samples only**, so their plain mean is the valid-only score (e.g. base 0.567 on 3,900 of 5,120).

[`../analysis/check_numbers.py`](../analysis/check_numbers.py) computes both.

**Protocol.**
- Eval set: 640 held-out ML goals × 8 samples, generation at temperature 1.0.
- Grader: frozen Qwen3-30B-A3B at temperature 0. It sees the goal-specific rubric and the reference plan.
- The reference plans are graded with themselves in the grader context, which can inflate their score.

Excluded: the dataset itself (`facebook/research-plan-gen`, available from Hugging Face), generated plans, grader outputs, checkpoints.
