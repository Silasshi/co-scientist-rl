"""Regenerate evolving_rubric/analysis/figures/*.png from evolving_rubric/data/."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common.analysis import save, style
from evolving_rubric.analysis._load import D, pass_table

F = Path(__file__).resolve().parent / "figures"
style()
t = pass_table("smoke_rl_coupled")

# 1. pass-rate heatmap: one row per evolved item, ordered by creation
order = t.drop_duplicates("title").sort_values(["created", "type"])
grid = t.pivot(index="title", columns="iter", values="pass").reindex(order.title)
fig, ax = plt.subplots(figsize=(8.2, 3.8))
im = ax.imshow(grid.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
ax.set_yticks(range(len(grid)), [("(+) " if ty == "positive" else "(-) ") + ti[:52] for ti, ty in zip(order.title, order.type)], fontsize=7)
ax.set_xticks(range(grid.shape[1]), grid.columns)
ax.set(xlabel="iteration", title="Evolved rubric items: when each was added, and the share of rollouts passing it")
fig.colorbar(im, ax=ax, label="pass rate (8 rollouts)")
ax.grid(False)
save(fig, F / "rubric_items.png")

# 2. aggregate rise is mostly composition
m = pd.read_csv(D / "runs/smoke_rl_coupled/metrics.csv")
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.plot(m["iter"], m["r_active/rollout_agg_mean"], "o-", label="logged R_active pass rate (all active items)")
fixed = t[t.created <= 3].groupby("iter")["pass"].mean()
ax.plot(fixed.index, fixed.values, "s--", label="same 4 items held fixed (created by iter 3)")
ax2 = ax.twinx()
ax2.bar(m["iter"], m["r_active/n_items"], alpha=0.15, color="#6c757d")
ax2.set_ylabel("active items", color="#6c757d")
ax2.grid(False)
ax.set(xlabel="iteration", ylabel="pass rate", title="Most of the rise comes from adding easier items")
ax.legend(frameon=False, fontsize=8, loc="upper left")
save(fig, F / "composition_effect.png")
