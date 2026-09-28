#!/usr/bin/env python3
"""Generate a methodology pipeline diagram for the presentation."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Colours ──────────────────────────────────────────────────────────────
OXFORD_BLUE  = "#002147"
GOLD         = "#C7A439"
LIGHT_GRAY   = "#F0F0F0"
MID_GRAY     = "#E0E0E0"
DARK_TEXT     = "#1A1A1A"
WHITE        = "#FFFFFF"
SOFT_BLUE    = "#3A6186"
LIGHT_GOLD   = "#F5ECD7"

# ── Figure ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 6.4),
                         gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.38})
for ax in axes:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.axis("off")

fig.patch.set_facecolor(WHITE)


# ── Helper: rounded box ─────────────────────────────────────────────────
def draw_box(ax, cx, cy, w, h, label, fc=OXFORD_BLUE, tc=WHITE,
             fontsize=8.5, boxstyle="round,pad=0.12", lw=0.8,
             ec=None, fontstyle="normal", fontweight="normal",
             sublabel=None, sublabel_fs=6.5, sublabel_tc=None):
    """Draw a rounded-corner box centred at (cx, cy)."""
    if ec is None:
        ec = fc
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                         boxstyle=boxstyle, facecolor=fc,
                         edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(box)
    dy = 0.07 if sublabel else 0
    ax.text(cx, cy + dy, label, ha="center", va="center",
            fontsize=fontsize, color=tc, fontweight=fontweight,
            fontstyle=fontstyle, family="sans-serif", zorder=4,
            linespacing=1.25)
    if sublabel:
        stc = sublabel_tc if sublabel_tc else tc
        ax.text(cx, cy - 0.2, sublabel, ha="center", va="center",
                fontsize=sublabel_fs, color=stc, fontstyle="italic",
                family="sans-serif", zorder=4, alpha=0.85)
    return box


def draw_arrow(ax, x0, y0, x1, y1, color=OXFORD_BLUE, lw=1.2,
               style="->", connectionstyle="arc3,rad=0.0", shrinkA=6, shrinkB=6):
    """Draw a fancy arrow between two points."""
    arrow = FancyArrowPatch((x0, y0), (x1, y1),
                            arrowstyle=style,
                            connectionstyle=connectionstyle,
                            color=color, lw=lw, zorder=2,
                            shrinkA=shrinkA, shrinkB=shrinkB,
                            mutation_scale=12)
    ax.add_patch(arrow)
    return arrow


# ═══════════════════════════════════════════════════════════════════════
#  ROW 1 — Data Creation Pipeline
# ═══════════════════════════════════════════════════════════════════════
ax1 = axes[0]

# background panel
bg1 = FancyBboxPatch((0.15, 0.12), 9.7, 1.76,
                      boxstyle="round,pad=0.15", facecolor=LIGHT_GRAY,
                      edgecolor=MID_GRAY, linewidth=0.6, zorder=0)
ax1.add_patch(bg1)

# Row label
ax1.text(5.0, 1.92, "Data Creation Pipeline", ha="center", va="bottom",
         fontsize=11, fontweight="bold", color=OXFORD_BLUE,
         family="sans-serif")

# ── Document icon (scientific paper) ──
doc_cx, doc_cy = 1.0, 1.0
# Draw a document shape as stacked rectangles
doc_w, doc_h = 0.75, 0.68
# Main rectangle
doc_box = FancyBboxPatch((doc_cx - doc_w/2, doc_cy - doc_h/2), doc_w, doc_h,
                          boxstyle="round,pad=0.06", facecolor=OXFORD_BLUE,
                          edgecolor=OXFORD_BLUE, linewidth=0.8, zorder=3)
ax1.add_patch(doc_box)
# Little "page" lines
for yoff in [0.14, 0.04, -0.06, -0.16]:
    ax1.plot([doc_cx - 0.22, doc_cx + 0.22], [doc_cy + yoff, doc_cy + yoff],
             color=WHITE, lw=0.6, alpha=0.5, zorder=4)
# Corner fold
fold_x = doc_cx + doc_w/2 - 0.03
fold_y = doc_cy + doc_h/2 - 0.03
tri = plt.Polygon([[fold_x - 0.13, fold_y + 0.03],
                    [fold_x + 0.03, fold_y + 0.03],
                    [fold_x + 0.03, fold_y - 0.13]],
                   facecolor=LIGHT_GRAY, edgecolor=OXFORD_BLUE,
                   linewidth=0.5, zorder=5)
ax1.add_patch(tri)
ax1.text(doc_cx, doc_cy - doc_h/2 - 0.16, "Scientific\nPaper", ha="center",
         va="top", fontsize=7.5, color=DARK_TEXT, family="sans-serif",
         fontweight="bold", linespacing=1.1)

# ── Extract Key Insights ──
draw_box(ax1, 3.0, 1.0, 1.55, 0.55, "Extract Key\nInsights",
         fc=SOFT_BLUE, tc=WHITE, fontsize=8.5, fontweight="bold")

# Arrow: paper -> extract
draw_arrow(ax1, 1.42, 1.0, 2.18, 1.0, color=OXFORD_BLUE)

# ── Research Goal ──
draw_box(ax1, 5.3, 1.4, 1.35, 0.42, "Research Goal",
         fc=GOLD, tc=DARK_TEXT, fontsize=8, fontweight="bold")

# ── Grading Rubric ──
draw_box(ax1, 5.3, 0.6, 1.55, 0.42, "Grading Rubric",
         fc=GOLD, tc=DARK_TEXT, fontsize=8, fontweight="bold",
         sublabel="(10 items)", sublabel_fs=6.5, sublabel_tc=DARK_TEXT)

# Arrows: extract -> goal, extract -> rubric
draw_arrow(ax1, 3.82, 1.12, 4.58, 1.38, color=OXFORD_BLUE,
           connectionstyle="arc3,rad=-0.08")
draw_arrow(ax1, 3.82, 0.88, 4.48, 0.65, color=OXFORD_BLUE,
           connectionstyle="arc3,rad=0.08")

# ── Reference Solution ──
draw_box(ax1, 7.6, 1.0, 1.55, 0.50, "Reference\nSolution",
         fc=OXFORD_BLUE, tc=WHITE, fontsize=8.5, fontweight="bold")

# Arrows: goal -> reference, rubric -> reference
draw_arrow(ax1, 6.0, 1.32, 6.78, 1.1, color=OXFORD_BLUE,
           connectionstyle="arc3,rad=0.08")
draw_arrow(ax1, 6.12, 0.68, 6.78, 0.92, color=OXFORD_BLUE,
           connectionstyle="arc3,rad=-0.08")

# ── Count label ──
ax1.text(9.3, 1.0, "6,500+\narXiv\npapers", ha="center", va="center",
         fontsize=8, color=OXFORD_BLUE, family="sans-serif",
         fontweight="bold", fontstyle="italic", linespacing=1.2,
         bbox=dict(boxstyle="round,pad=0.2", facecolor=LIGHT_GOLD,
                   edgecolor=GOLD, linewidth=0.8, alpha=0.9))


# ═══════════════════════════════════════════════════════════════════════
#  ROW 2 — Self-Grading Training Loop
# ═══════════════════════════════════════════════════════════════════════
ax2 = axes[1]

# background panel
bg2 = FancyBboxPatch((0.15, 0.02), 9.7, 1.92,
                      boxstyle="round,pad=0.15", facecolor=LIGHT_GRAY,
                      edgecolor=MID_GRAY, linewidth=0.6, zorder=0)
ax2.add_patch(bg2)

ax2.text(5.0, 1.98, "Self-Grading Training Loop (GRPO)", ha="center", va="bottom",
         fontsize=11, fontweight="bold", color=OXFORD_BLUE, family="sans-serif")

# ── Research Goal (input) ──
draw_box(ax2, 0.75, 1.15, 1.05, 0.42, "Research\nGoal",
         fc=GOLD, tc=DARK_TEXT, fontsize=8, fontweight="bold")

# ── Plan Generator ──
draw_box(ax2, 2.85, 1.15, 1.55, 0.55, "Plan Generator",
         fc=OXFORD_BLUE, tc=WHITE, fontsize=9, fontweight="bold",
         sublabel="Qwen 30B", sublabel_fs=7, sublabel_tc="#B0C4DE")

# Arrow: goal -> generator
draw_arrow(ax2, 1.3, 1.15, 2.03, 1.15, color=OXFORD_BLUE)

# ── 8 Candidate Plans ──
draw_box(ax2, 5.05, 1.15, 1.5, 0.50, "8 Candidate\nPlans",
         fc=SOFT_BLUE, tc=WHITE, fontsize=8.5, fontweight="bold")

# Arrow: generator -> plans
draw_arrow(ax2, 3.67, 1.15, 4.26, 1.15, color=OXFORD_BLUE)

# ── Self-Grader ──
draw_box(ax2, 7.15, 1.15, 1.55, 0.55, "Self-Grader",
         fc=OXFORD_BLUE, tc=WHITE, fontsize=9, fontweight="bold",
         sublabel="Frozen Copy", sublabel_fs=7, sublabel_tc="#B0C4DE")

# Arrow: plans -> grader
draw_arrow(ax2, 5.83, 1.15, 6.33, 1.15, color=OXFORD_BLUE)

# ── Rubric (privileged info) — feeding into grader from above ──
draw_box(ax2, 7.15, 1.85, 1.2, 0.28, "Rubric",
         fc=GOLD, tc=DARK_TEXT, fontsize=7.5, fontweight="bold",
         ec=GOLD)
# small label
ax2.text(7.97, 1.85, " privileged", ha="left", va="center",
         fontsize=6.5, color=GOLD, fontstyle="italic", fontweight="bold",
         family="sans-serif",
         bbox=dict(boxstyle="round,pad=0.06", facecolor=LIGHT_GRAY,
                   edgecolor="none"))

# Arrow: rubric -> grader
draw_arrow(ax2, 7.15, 1.69, 7.15, 1.46, color=GOLD, lw=1.3)

# ── Scores + Advantages ──
draw_box(ax2, 7.15, 0.38, 1.4, 0.38, "Scores +\nAdvantages",
         fc=SOFT_BLUE, tc=WHITE, fontsize=7.5, fontweight="bold")

# Arrow: grader -> scores (downward)
draw_arrow(ax2, 7.15, 0.85, 7.15, 0.60, color=OXFORD_BLUE)

# ── GRPO Update ──
draw_box(ax2, 4.3, 0.38, 1.35, 0.42, "GRPO\nUpdate",
         fc=GOLD, tc=DARK_TEXT, fontsize=8.5, fontweight="bold",
         ec=GOLD)

# Arrow: scores -> GRPO
draw_arrow(ax2, 6.42, 0.38, 5.0, 0.38, color=OXFORD_BLUE)

# Arrow: GRPO -> generator (curved, going up-left)
draw_arrow(ax2, 3.60, 0.50, 2.55, 0.84, color=GOLD, lw=1.6,
           style="-|>", connectionstyle="arc3,rad=0.20",
           shrinkA=8, shrinkB=8)

# "updates" label near the GRPO->Generator arrow
ax2.text(3.55, 0.58, "updates\nweights", ha="center", va="center",
         fontsize=6.0, color=GOLD, fontstyle="italic",
         fontweight="bold", family="sans-serif",
         bbox=dict(boxstyle="round,pad=0.06", facecolor=WHITE,
                   edgecolor="none", alpha=0.85))

# ── Key insight annotation ──
# Draw a highlighted annotation box at the bottom
annot_text = "Generator-Verifier Gap: grader sees rubric, generator does not"
ax2.text(5.0, 0.04, annot_text, ha="center", va="bottom",
         fontsize=7.5, color=OXFORD_BLUE, family="sans-serif",
         fontweight="bold", fontstyle="italic",
         bbox=dict(boxstyle="round,pad=0.18", facecolor=LIGHT_GOLD,
                   edgecolor=GOLD, linewidth=0.8, alpha=0.95))

# ── Dashed box around the no-rubric zone (generator) ──
no_rubric = FancyBboxPatch((1.95, 0.72), 1.80, 0.75,
                            boxstyle="round,pad=0.08",
                            facecolor="none", edgecolor=GOLD,
                            linewidth=1.2, linestyle="--", zorder=2)
ax2.add_patch(no_rubric)
ax2.text(2.85, 0.74, "no rubric access", ha="center", va="top",
         fontsize=5.8, color=GOLD, fontstyle="italic",
         fontweight="bold", family="sans-serif",
         bbox=dict(boxstyle="round,pad=0.04", facecolor=LIGHT_GRAY,
                   edgecolor="none"))


# ═══════════════════════════════════════════════════════════════════════
#  Save
# ═══════════════════════════════════════════════════════════════════════
out = "/mnt/d/Oxford/Individual_presentation/charts/methodology_pipeline.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=WHITE, pad_inches=0.25)
plt.close(fig)
print(f"Saved → {out}")
