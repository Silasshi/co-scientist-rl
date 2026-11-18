"""Regenerate ttt/analysis/figures/*.png from ttt/data/."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common.analysis import save, style

ROOT = Path(__file__).resolve().parents[1]
D, F = ROOT / "data", ROOT / "analysis" / "figures"
style()
run = lambda n: pd.read_csv(D / f"runs/{n}/metrics.csv")  # noqa: E731

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
for n, label, col in [("MAIN_v9", "critique-revise + RL", "#2f6db5"), ("B4_v9", "critique-revise, no RL", "#d1495b"),
                      ("B4_stripped", "revise without per-signal critique", "#6c757d")]:
    m = run(n); axes[0].plot(m["iter"], m["reward/buffer_max"], label=label, color=col)
axes[0].set(title="Best-of-2: critique conditioning\ndoes most of the work", xlabel="iteration", ylabel="best plan so far (training grader)")
axes[0].legend(frameon=False, fontsize=8)
for n, label, col in [("MAIN_fresh_bon1", "RL incl. entropic-weighted fresh plans", "#2f6db5"), ("B4_bon1", "no RL", "#d1495b")]:
    m = run(n); axes[1].plot(m["iter"], m["reward/buffer_max"], label=label, color=col)
axes[1].set(title="Best-of-1: the RL run ends higher\n(one run per condition)", xlabel="iteration")
axes[1].legend(frameon=False, fontsize=8)
save(fig, F / "trajectories.png")

o = pd.read_csv(D / "opus_judge_scores.csv").set_index("name")
names = ["B4_stripped", "B4_bon1", "MAIN_fresh_bon1", "MAIN_v9", "B4_v9", "MAIN_v8.1", "B4_v8.1"]
labels = {"MAIN_v9": "MAIN_v9 = B4_v9", "B4_v9": ""}
x = [run(n)["reward/buffer_max"].iloc[-1] for n in names]
y = [o.loc[n, "opus_total_20"] for n in names]
fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.scatter(x, y, color=["#2f6db5" if "v8.1" not in n else "#d1495b" for n in names])
for xi, yi, n in zip(x, y, names):
    ax.annotate(labels.get(n, n), (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=7)
ax.set(xlabel="training-grader score of the best plan (each run's own signal set)", ylabel="independent judge, Opus /20\n(reference plan = 20)",
       title="Old signals (v8.1, red) scored highest but were judged worse", ylim=(4, 12))
save(fig, F / "grader_vs_judge.png")
