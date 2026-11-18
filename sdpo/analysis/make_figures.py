"""Regenerate sdpo/analysis/figures/*.png from sdpo/data/."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common.analysis import eval_mean_all, save, style

ROOT = Path(__file__).resolve().parents[1]
D, F = ROOT / "data", ROOT / "analysis" / "figures"
style()

fig, ax = plt.subplots(figsize=(6.4, 3.4))
for name, label, color in [("grpo_control", "GRPO only (same script, distillation off)", "#3a9d5d"),
                           ("sdpo", "SDPO: GRPO + self-teacher distillation", "#d1495b")]:
    m = pd.read_csv(D / f"train_metrics/{name}.csv")
    ax.plot(m["progress/batch"], m["reward/total"].rolling(5, min_periods=1).mean(), label=label, color=color)
ax.set(xlabel="training batch", ylabel="training reward (5-batch mean)",
       title="SDPO learns faster than GRPO, then collapses after batch ~86")
ax.axvline(106, color="#6c757d", ls=":", lw=1)
ax.text(105, ax.get_ylim()[0] + 0.005, "eval\ncheckpoint", ha="right", fontsize=7, color="#6c757d")
ax.legend(frameon=False, fontsize=8)
save(fig, F / "training_curves.png")

rows = []
for name, tag, label in [("grpo_control", "eval_b106", "GRPO only\n107 b"),
                         ("sdpo", "eval_b106", "SDPO\n107 b"),
                         ("sdpo_warmstart", "eval_b235", "SDPO warm-start\nGRPO b214 + 21 b")]:
    mean, n_gen, n_valid = eval_mean_all(D / f"eval/{name}.{tag}_summary.csv")
    valid = pd.read_parquet(D / f"eval/{name}.{tag}.parquet").rubric_score.mean()
    rows.append((label, mean, valid, n_valid / n_gen))
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
x = range(len(rows))
axes[0].bar([i - 0.2 for i in x], [r[1] for r in rows], 0.4, label="all samples (headline)")
axes[0].bar([i + 0.2 for i in x], [r[2] for r in rows], 0.4, label="valid samples only")
axes[0].axhline(0.6926, color="#6c757d", ls=":", label="GRPO main run (0.693)")
axes[0].set(xticks=list(x), xticklabels=[r[0] for r in rows], ylim=(0.4, 0.75), ylabel="held-out rubric score")
axes[0].legend(frameon=False, fontsize=7)
axes[1].bar(list(x), [100 * r[3] for r in rows], color="#6c757d")
axes[1].set(xticks=list(x), xticklabels=[r[0] for r in rows], ylim=(0, 105), ylabel="% eval samples valid (format)")
for i, r in enumerate(rows):
    axes[1].text(i, 100 * r[3] + 1.5, f"{100 * r[3]:.0f}%", ha="center", fontsize=8)
fig.suptitle("Held-out eval at the final checkpoint (640 goals x 8)")
save(fig, F / "eval_comparison.png")
