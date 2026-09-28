#!/usr/bin/env python3
"""Generate analysis figures for the V2 multi-turn training run."""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

RUN_DIR = Path("/home/silas/co-scientist-project/runs/2026/3/multiturn_v4/2/train")
OUT_DIR = Path("/home/silas/co-scientist-project/analysis")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
batch_summaries = []
with open(RUN_DIR / "batch_summary.jsonl") as f:
    for line in f:
        batch_summaries.append(json.loads(line))

training_logs = []
with open(RUN_DIR / "training_logs.jsonl") as f:
    for line in f:
        training_logs.append(json.loads(line))

batch_indices = [b["batch_idx"] for b in batch_summaries]
bi = np.array(batch_indices)

# Group training logs by batch
logs_by_batch = defaultdict(list)
for entry in training_logs:
    logs_by_batch[entry["batch_idx"]].append(entry)


# ===========================================================================
# Figure 1: Core Training Metrics (2x2)
# ===========================================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle("V2 Training — Core Metrics", fontsize=15, fontweight="bold", y=0.98)

# 1a. Rubric & Reward
ax = axes1[0, 0]
rubric_means = [b["rubric/mean"] for b in batch_summaries]
reward_means = [b["reward/mean"] for b in batch_summaries]
ax.plot(bi, rubric_means, "o-", label="Rubric", color="#2196F3", linewidth=2, markersize=6)
ax.plot(bi, reward_means, "s--", label="Reward", color="#4CAF50", linewidth=1.5, markersize=5)
ax.axhline(y=0.523, color="#E91E63", linestyle="--", linewidth=1.5, alpha=0.7, label="V1 avg (0.523)")
ax.axhline(y=0.624, color="#FF9800", linestyle=":", linewidth=1.5, alpha=0.7, label="V1 peak (0.624)")
ax.set_ylabel("Score")
ax.set_title("Rubric & Reward")
ax.legend(fontsize=8, loc="lower right")
ax.set_ylim(0.35, 0.85)
ax.grid(True, alpha=0.3)

# 1b. Student Score Distribution (stacked bar)
ax = axes1[0, 1]
score_levels = [-3, -2, -1, 0, 1, 2, 3]
score_colors = {-3: "#B71C1C", -2: "#E53935", -1: "#EF9A9A",
                0: "#BDBDBD", 1: "#90CAF9", 2: "#42A5F5", 3: "#1565C0"}
bottoms = np.zeros(len(bi))
for s in score_levels:
    counts = np.array([b["student_score/dist"].get(str(s), 0) for b in batch_summaries])
    if counts.sum() > 0:
        ax.bar(bi, counts, bottom=bottoms, color=score_colors[s],
               label=f"Score {s}", edgecolor="white", linewidth=0.5)
        bottoms += counts
ax.set_ylabel("Count")
ax.set_title("Student Score Distribution")
ax.legend(fontsize=7, loc="upper right", ncol=2)

# 1c. Completion Rate
ax = axes1[1, 0]
comp_rates = [b["conv/completed_rate"] for b in batch_summaries]
ax.plot(bi, comp_rates, "o-", color="#7E57C2", linewidth=2, markersize=6)
ax.fill_between(bi, 0, comp_rates, alpha=0.15, color="#7E57C2")
ax.set_ylabel("Rate")
ax.set_title("Completion Rate ([PLAN_FINAL])")
ax.set_ylim(0, max(comp_rates) * 1.5 + 0.05)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.grid(True, alpha=0.3)

# 1d. Datums per Batch
ax = axes1[1, 1]
wave_datums = [b["datums/wave_total"] for b in batch_summaries]
plan_datums = [b["datums/plan_total"] for b in batch_summaries]
ax.bar(bi, wave_datums, color="#26A69A", label="Wave (discussion)", edgecolor="white", linewidth=0.5)
ax.bar(bi, plan_datums, bottom=wave_datums, color="#AB47BC", label="Plan (GRPO)",
       edgecolor="white", linewidth=0.5)
ax.set_ylabel("Datums")
ax.set_title("Training Datums per Batch")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

for ax in axes1.flat:
    ax.set_xlabel("Batch")
    ax.set_xlim(bi[0] - 0.5, bi[-1] + 0.5)

fig1.tight_layout(rect=[0, 0, 1, 0.96])
fig1.savefig(OUT_DIR / "v2_core_metrics.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'v2_core_metrics.png'}")


# ===========================================================================
# Figure 2: Diagnostics & Issues (2x2)
# ===========================================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle("V2 Training — Diagnostics", fontsize=15, fontweight="bold", y=0.98)

# 2a. Completion Reason Distribution
ax = axes2[0, 0]
reason_counts_per_batch = defaultdict(Counter)
for entry in training_logs:
    reason_counts_per_batch[entry["batch_idx"]][entry.get("completion_reason", "unknown")] += 1

reason_order = ["plan_final", "max_revisions", "max_turns", "student_empty", "error"]
reason_colors_map = {
    "plan_final": "#66BB6A", "max_revisions": "#42A5F5", "max_turns": "#FFA726",
    "student_empty": "#EF5350", "error": "#9E9E9E",
}
bottoms = np.zeros(len(bi))
for reason in reason_order:
    counts = np.array([reason_counts_per_batch[bidx].get(reason, 0) for bidx in batch_indices])
    if counts.sum() > 0:
        ax.bar(bi, counts, bottom=bottoms, color=reason_colors_map.get(reason, "#78909C"),
               label=reason, edgecolor="white", linewidth=0.5)
        bottoms += counts
ax.set_ylabel("Conversations")
ax.set_title("Completion Reasons")
ax.legend(fontsize=7, loc="upper right")

# 2b. First-Turn Plan Rate (THE KEY DIAGNOSTIC)
ax = axes2[0, 1]
first_turn_plan_rates = []
for bidx in batch_indices:
    batch_logs = logs_by_batch[bidx]
    ftp = sum(1 for c in batch_logs if c["num_discussion"] == 0 and c["num_revisions"] >= 1)
    first_turn_plan_rates.append(ftp / max(len(batch_logs), 1))

ax.plot(bi, first_turn_plan_rates, "o-", color="#E53935", linewidth=2.5, markersize=7)
ax.fill_between(bi, 0, first_turn_plan_rates, alpha=0.15, color="#E53935")
ax.set_ylabel("Rate")
ax.set_title("Tutor Skips Discussion (plan on turn 0)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
ax.annotate(f"{first_turn_plan_rates[-1]:.0%}", xy=(bi[-1], first_turn_plan_rates[-1]),
            xytext=(bi[-1]-1.5, first_turn_plan_rates[-1]-0.12),
            arrowprops=dict(arrowstyle="->", color="#E53935"), fontsize=10, color="#E53935", fontweight="bold")

# 2c. Student Score Mean + Discussion Turn Count
ax = axes2[1, 0]
ss_means = np.array([b["student_score/mean"] for b in batch_summaries])
ss_stds = np.array([b["student_score/std"] for b in batch_summaries])
disc_means = [b["conv/discussion_turns_mean"] for b in batch_summaries]

ax.plot(bi, ss_means, "o-", color="#1565C0", linewidth=2, markersize=5, label="Score mean")
ax.fill_between(bi, ss_means - ss_stds, ss_means + ss_stds, alpha=0.15, color="#1565C0")
ax2 = ax.twinx()
ax2.plot(bi, disc_means, "s--", color="#FF7043", linewidth=1.5, markersize=5, label="Disc. turns")
ax.set_ylabel("Student Score", color="#1565C0")
ax2.set_ylabel("Discussion Turns", color="#FF7043")
ax.set_title("Student Score vs Discussion Engagement")
ax.tick_params(axis="y", labelcolor="#1565C0")
ax2.tick_params(axis="y", labelcolor="#FF7043")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower left")
ax.grid(True, alpha=0.3)

# 2d. Wave Datums & Hints over time
ax = axes2[1, 1]
hints = [b["hint/total"] for b in batch_summaries]
opd_sets = [b.get("opd/sets_computed", 0) for b in batch_summaries]
ax.bar(bi - 0.15, wave_datums, width=0.3, color="#26A69A", label="Wave datums", alpha=0.8)
ax.bar(bi + 0.15, hints, width=0.3, color="#FF8F00", label="Hints extracted", alpha=0.8)
ax.set_ylabel("Count")
ax.set_title("Wave Datums & OPD Hints")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

for ax in axes2.flat:
    ax.set_xlabel("Batch")
    ax.set_xlim(bi[0] - 0.5, bi[-1] + 0.5)

fig2.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig(OUT_DIR / "v2_diagnostics.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'v2_diagnostics.png'}")


# ===========================================================================
# Figure 3: V1 vs V2 Comparison
# ===========================================================================
# Load V1 data if available
v1_summary_path = Path("/home/silas/co-scientist-project/runs/2026/3/multiturn_v4/run1/train/batch_summary.jsonl")
if v1_summary_path.exists():
    v1_summaries = []
    with open(v1_summary_path) as f:
        for line in f:
            v1_summaries.append(json.loads(line))

    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
    fig3.suptitle("V1 vs V2 Comparison", fontsize=15, fontweight="bold", y=1.02)

    v1_bi = np.array([b["batch_idx"] for b in v1_summaries])
    v2_bi = bi

    # 3a. Rubric comparison
    ax = axes3[0]
    v1_rubric = [b["rubric/mean"] for b in v1_summaries]
    v2_rubric = rubric_means
    ax.plot(v1_bi, v1_rubric, "o-", color="#E91E63", linewidth=2, markersize=5, label="V1 (binary PRM)")
    ax.plot(v2_bi, v2_rubric, "s-", color="#2196F3", linewidth=2, markersize=5, label="V2 (merged student)")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Rubric Mean")
    ax.set_title("Rubric Score")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3b. Reward comparison
    ax = axes3[1]
    v1_reward = [b["reward/mean"] for b in v1_summaries]
    v2_reward = reward_means
    ax.plot(v1_bi, v1_reward, "o-", color="#E91E63", linewidth=2, markersize=5, label="V1")
    ax.plot(v2_bi, v2_reward, "s-", color="#2196F3", linewidth=2, markersize=5, label="V2")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Reward Mean")
    ax.set_title("Final Reward")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3c. Completion rate comparison
    ax = axes3[2]
    v1_comp = [b["conv/completed_rate"] for b in v1_summaries]
    v2_comp = comp_rates
    ax.plot(v1_bi, v1_comp, "o-", color="#E91E63", linewidth=2, markersize=5, label="V1")
    ax.plot(v2_bi, v2_comp, "s-", color="#2196F3", linewidth=2, markersize=5, label="V2")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Completion Rate")
    ax.set_title("Natural Completion Rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(OUT_DIR / "v2_vs_v1_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'v2_vs_v1_comparison.png'}")
else:
    print("V1 data not found, skipping comparison plot")

print("Done!")
