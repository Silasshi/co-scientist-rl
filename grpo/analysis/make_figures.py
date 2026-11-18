"""Regenerate grpo/analysis/figures/*.png from grpo/data/."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common.analysis import eval_mean_all, expected_max, save, style

ROOT = Path(__file__).resolve().parents[1]
D, F = ROOT / "data", ROOT / "analysis" / "figures"
style()

# 1. training reward: binary vs graded
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for name, label in [("binary_reward", "binary reward (Goel et al. rule)"),
                    ("graded_reward", "4-level graded reward"),
                    ("grpo_main", "graded + sample filtering (main run)")]:
    m = pd.read_csv(D / f"train_metrics/{name}.csv")
    ax.plot(m["progress/batch"], m["reward/total"].rolling(5, min_periods=1).mean(), label=label)
ax.set(xlabel="training batch (64 goals x 8 samples)", ylabel="training reward (5-batch mean)",
       title="Binary rubric reward gives no learning signal; graded reward trains")
ax.legend(frameon=False, fontsize=8)
save(fig, F / "training_curves.png")

# 2. held-out eval
rows = [("base model", eval_mean_all(D / "eval/grpo_main.base_m1_summary.csv")[0]),
        ("GRPO (ours)", eval_mean_all(D / "eval/grpo_main.eval_b214_summary.csv")[0]),
        ("GPT-5.4 zero-shot", pd.read_parquet(D / "eval/gpt54_zeroshot.sota.parquet").rubric_score.mean()),
        ("reference plans", pd.read_parquet(D / "eval/grpo_main.reference.parquet").rubric_score.mean())]
fig, ax = plt.subplots(figsize=(5.6, 3.0))
bars = ax.barh([r[0] for r in rows][::-1], [r[1] for r in rows][::-1], color=["#6c757d", "#3a9d5d", "#2f6db5", "#6c757d"][::-1])
ax.bar_label(bars, fmt="%.3f", padding=3)
ax.set(xlim=(0, 1), xlabel="held-out rubric score (same grader; 640 goals, references n=685)", title="Held-out evaluation")
save(fig, F / "eval_comparison.png")

# 3. selection gap
b = pd.read_parquet(D / "eval/grpo_main.bon_b214.parquet")
goals = b.groupby(["eval_batch_idx", "goal_idx"]).grader_rubric_score
ks = [1, 2, 3, 4, 5, 6, 7, 8]
curve = [goals.apply(lambda s, k=k: expected_max(s.values, k)).mean() for k in ks]
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.plot(ks, curve, "o-", label="oracle best-of-N (grader picks)")
ax.axhline(b.loc[b.is_self_pick == 1, "grader_rubric_score"].mean(), color="#d1495b", ls="--", label="model picks its own best of 8")
ax.axhline(rows[3][1], color="#6c757d", ls=":", label="reference plans")
ax.set(xlabel="N samples per goal", ylabel="rubric score", title="The policy writes good plans but cannot pick them")
ax.legend(frameon=False, fontsize=8)
save(fig, F / "selection_gap.png")
