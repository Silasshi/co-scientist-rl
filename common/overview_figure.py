"""Regenerate assets/overview.png (the README teaser) from the per-method data/ folders.

python -m common.overview_figure
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common.analysis import PALETTE, eval_mean_all, save, style

ROOT = Path(__file__).resolve().parents[1]
G, S, T = ROOT / "grpo/data", ROOT / "sdpo/data", ROOT / "ttt/data"
style()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})

# (a) every trained model on the same held-out eval (640 goals x 8, same frozen grader)
rows = [
    ("Base model", eval_mean_all(G / "eval/grpo_main.base_m1_summary.csv")[0], PALETTE[5]),
    ("GRPO, continuous rubric score\n(reproduction)", eval_mean_all(G / "eval/grpo_main.eval_b214_summary.csv")[0], PALETTE[2]),
    ("GRPO control, 107 batches", eval_mean_all(S / "eval/grpo_control.eval_b106_summary.csv")[0], PALETTE[0]),
    ("SDPO, 107 batches", eval_mean_all(S / "eval/sdpo.eval_b106_summary.csv")[0], PALETTE[1]),
    ("SDPO warm-started\nfrom the reproduction", eval_mean_all(S / "eval/sdpo_warmstart.eval_b235_summary.csv")[0], PALETTE[1]),
    ("Reference plans", pd.read_parquet(G / "eval/grpo_main.reference.parquet").rubric_score.mean(), PALETTE[5]),
]
bars = a1.barh([r[0] for r in rows][::-1], [r[1] for r in rows][::-1], color=[r[2] for r in rows][::-1])
a1.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
a1.axvline(rows[1][1], color=PALETTE[2], ls=":", lw=1)
a1.set(xlim=(0, 1), xlabel="held-out rubric score (640 goals × 8, same grader)",
       title="(a) Same held-out eval: SDPO does not beat the reproduction")
a1.tick_params(axis="y", labelsize=9)

# (b) training-grader score vs an independent depth judge (Claude Opus, /20)
audit = pd.read_csv(G / "rubric_audit.csv")
a2.scatter(audit.rubric_score, audit.opus_total, s=40, color=PALETTE[2], alpha=0.85,
           label="GRPO checkpoint, goal rubric (13 plans)")
opus = pd.read_csv(T / "opus_judge_scores.csv").set_index("name")["opus_total_20"]
runs = [n for n in opus.index if n != "reference"]
best = [pd.read_csv(T / f"runs/{n}/metrics.csv")["reward/buffer_max"].iloc[-1] for n in runs]
a2.scatter(best, [opus[n] for n in runs], s=46, marker="s", color=PALETTE[3], alpha=0.9,
           label="TTT best plans, 10-signal grader (8 runs)")
a2.axhline(opus["reference"], color=PALETTE[5], ls="--", lw=1)
a2.text(0.02, opus["reference"] - 0.9, "reference plan (TTT goal): 20 / 20", fontsize=8, color=PALETTE[5])
a2.set(xlim=(-0.02, 1.02), ylim=(0, 21), xlabel="score from the training grader",
       ylabel="independent judge (Opus, /20)", title="(b) High grader scores, shallow plans")
a2.legend(frameon=False, fontsize=8, loc="lower left")

save(fig, ROOT / "assets/overview.png")
