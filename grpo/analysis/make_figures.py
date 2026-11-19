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
                    ("graded_reward", "continuous rubric score (ours)"),
                    ("grpo_main", "continuous score + sample filtering (main run)")]:
    m = pd.read_csv(D / f"train_metrics/{name}.csv")
    ax.plot(m["progress/batch"], m["reward/total"].rolling(5, min_periods=1).mean(), label=label)
ax.set(xlabel="training batch (64 goals x 8 samples)", ylabel="training reward (5-batch mean)",
       title="Binary rubric reward gives no learning signal; a continuous score trains")
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

# 4. what training changed besides the rubric score: length moves to the shaped target, format errors vanish
m = pd.read_csv(D / "train_metrics/grpo_main.batch_summary.csv")
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.plot(m.batch_idx, m.length_mean.rolling(5, min_periods=1).mean(), label="mean plan length (words)")
ax.axhline(600, color="#6c757d", ls=":", label="centre of the length bonus (600 words)")
ax.set(xlabel="training batch", ylabel="words", title="The policy learns the length shaping and the output format")
ax2 = ax.twinx()
ax2.plot(m.batch_idx, m.format_rate.rolling(5, min_periods=1).mean(), color="#d1495b", label="mean format penalty")
ax2.set_ylabel("format penalty", color="#d1495b"); ax2.grid(False); ax2.spines["right"].set_visible(True)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper right")
save(fig, F / "length_format.png")

# 5. rubric score vs an independent depth audit (Claude Opus, blind, /20)
a = pd.read_csv(D / "rubric_audit.csv")
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for aset, mk, label in [("same_goal", "o", "8 samples, one goal"), ("mixed_goals", "s", "5 samples, different goals")]:
    s = a[a.audit_set == aset]
    ax.scatter(s.rubric_score, s.opus_total, marker=mk, s=46, label=label, alpha=0.85)
ax.set(xlabel="rubric score (training grader)", ylabel="Opus depth audit (/20)", xlim=(-0.03, 1.03), ylim=(4, 20),
       title="High rubric scores do not mean deep plans")
ax.legend(frameon=False, fontsize=8, loc="upper left")
save(fig, F / "rubric_vs_opus.png")
