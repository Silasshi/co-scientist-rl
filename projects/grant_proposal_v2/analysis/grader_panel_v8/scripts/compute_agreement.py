"""Compute grader agreement metrics: Spearman ρ, MAE, within-1 vs Opus ground truth.

Input: grades/*.jsonl (one per model + opus.jsonl)
Output: analysis/correlation_matrix.jsonl + overall_ranking.md + heatmap.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

PANEL = Path(__file__).resolve().parent.parent
GRADES = PANEL / "grades"
OUT = PANEL / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

# Order matters for the final table
SIGNAL_ORDER = [
    "G1_problem_specificity",
    "G2_specific_aims",
    "G4_focus",
    "G6_reasoning_depth",
    "G8_deliverable_clarity",
    "G9_scope_feasibility",
    "G10_approach_coverage",
    "G11_evidence_rigor",
    "G12_formalism",
    "G13_risk_awareness",
]


def load_grades(path: Path) -> dict:
    """Return {(plan_id, signal_id): score}. Ignores parse failures."""
    if not path.exists():
        return {}
    out = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        s = r.get("score")
        if s is None:
            continue
        out[(r["plan_id"], r["signal_id"])] = int(s)
    return out


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation (no scipy dep)."""
    if len(x) < 3:
        return float("nan")

    def ranks(vals: list[float]) -> list[float]:
        """Fractional ranks with average-tie-breaking."""
        indexed = sorted(enumerate(vals), key=lambda iv: iv[1])
        r = [0.0] * len(vals)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) and indexed[j][1] == indexed[i][1]:
                j += 1
            avg = (i + j - 1) / 2.0 + 1  # 1-indexed ranks
            for k in range(i, j):
                r[indexed[k][0]] = avg
            i = j
        return r

    rx = ranks(list(x))
    ry = ranks(list(y))
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx ** 0.5 * dy ** 0.5)


def main() -> None:
    opus = load_grades(GRADES / "opus.jsonl")
    if not opus:
        print("ERROR: opus.jsonl missing or empty. Run Opus grading first.")
        sys.exit(1)
    print(f"Opus ground truth: {len(opus)} (plan, signal) pairs")

    # List all model files (except opus and its per-group splits)
    model_files = sorted(
        [
            p for p in GRADES.glob("*.jsonl")
            if p.stem != "opus" and not p.stem.startswith("opus_")
        ]
    )
    if not model_files:
        print("ERROR: no tinker model grades found in grades/")
        sys.exit(1)

    # Per-signal aggregation
    rows = []
    per_model_summary = {}
    for mf in model_files:
        model = mf.stem
        grades = load_grades(mf)
        print(f"\n=== {model} ({len(grades)} graded pairs) ===")

        signal_metrics = {}
        all_opus = []
        all_model = []

        for sig_id in SIGNAL_ORDER:
            opus_vals = []
            model_vals = []
            for (pid, sid), s_opus in opus.items():
                if sid != sig_id:
                    continue
                s_model = grades.get((pid, sid))
                if s_model is None:
                    continue
                opus_vals.append(s_opus)
                model_vals.append(s_model)

            if len(opus_vals) < 3:
                rho = float("nan")
                mae = float("nan")
                w1 = float("nan")
            else:
                rho = spearman(opus_vals, model_vals)
                diffs = [abs(o - m) for o, m in zip(opus_vals, model_vals)]
                mae = sum(diffs) / len(diffs)
                w1 = sum(1 for d in diffs if d <= 1) / len(diffs)

            signal_metrics[sig_id] = {
                "n": len(opus_vals),
                "spearman": rho,
                "mae": mae,
                "within1": w1,
                "opus_mean": sum(opus_vals) / len(opus_vals) if opus_vals else float("nan"),
                "model_mean": sum(model_vals) / len(model_vals) if model_vals else float("nan"),
            }
            all_opus.extend(opus_vals)
            all_model.extend(model_vals)

            rho_str = f"{rho:+.2f}" if not np.isnan(rho) else "  nan"
            mae_str = f"{mae:.2f}" if not np.isnan(mae) else " nan"
            w1_str = f"{w1*100:.0f}%" if not np.isnan(w1) else "nan%"
            print(
                f"  {sig_id:<28s} n={len(opus_vals):>2d}  "
                f"ρ={rho_str}  MAE={mae_str}  within1={w1_str}"
            )

        # Aggregate across signals (pooled)
        overall_rho = spearman(all_opus, all_model)
        overall_mae = (
            sum(abs(o - m) for o, m in zip(all_opus, all_model)) / len(all_opus)
            if all_opus else float("nan")
        )
        overall_w1 = (
            sum(1 for o, m in zip(all_opus, all_model) if abs(o - m) <= 1) / len(all_opus)
            if all_opus else float("nan")
        )
        # Mean ρ across signals (different from pooled)
        sig_rhos = [
            v["spearman"] for v in signal_metrics.values()
            if not np.isnan(v["spearman"])
        ]
        mean_sig_rho = sum(sig_rhos) / len(sig_rhos) if sig_rhos else float("nan")

        per_model_summary[model] = {
            "overall_spearman_pooled": overall_rho,
            "mean_spearman_per_signal": mean_sig_rho,
            "overall_mae": overall_mae,
            "overall_within1": overall_w1,
            "n_pairs": len(all_opus),
            "signals": signal_metrics,
        }
        rows.append({"model": model, **per_model_summary[model]})

        print(
            f"  OVERALL: pooled ρ={overall_rho:+.3f}  "
            f"mean-per-signal ρ={mean_sig_rho:+.3f}  "
            f"MAE={overall_mae:.2f}  within1={overall_w1*100:.0f}%  "
            f"(n={len(all_opus)} pairs)"
        )

    # Save JSONL summary
    out_file = OUT / "correlation_matrix.jsonl"
    with open(out_file, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved correlation_matrix.jsonl to {out_file}")

    # Write Markdown ranking
    md_lines = [
        "# Grader Panel v8 Agreement — Results",
        "",
        "Spearman rank correlation with Opus 4.7 ground truth on 15 FoundOpt plans × 10 v8 signals.",
        "",
        "## Overall ranking (mean ρ across signals)",
        "",
        f"| Model | Mean ρ (per-signal) | Pooled ρ | MAE | Within-1 | n |",
        f"|-------|--------------------:|---------:|----:|---------:|---:|",
    ]
    rows_sorted = sorted(
        rows, key=lambda r: (
            -r["mean_spearman_per_signal"]
            if not np.isnan(r["mean_spearman_per_signal"]) else 1e9
        )
    )
    for r in rows_sorted:
        md_lines.append(
            f"| {r['model']} "
            f"| {r['mean_spearman_per_signal']:+.3f} "
            f"| {r['overall_spearman_pooled']:+.3f} "
            f"| {r['overall_mae']:.2f} "
            f"| {r['overall_within1']*100:.0f}% "
            f"| {r['n_pairs']} |"
        )

    md_lines += ["", "## Per-signal correlation (Spearman ρ vs Opus)", ""]
    header = "| Model | " + " | ".join(s.split("_")[0] for s in SIGNAL_ORDER) + " |"
    sep = "|-------|" + "|".join([":---:"] * len(SIGNAL_ORDER)) + "|"
    md_lines += [header, sep]
    for r in rows_sorted:
        sigs = r["signals"]
        cells = []
        for sid in SIGNAL_ORDER:
            v = sigs.get(sid, {}).get("spearman", float("nan"))
            cells.append(f"{v:+.2f}" if not np.isnan(v) else "—")
        md_lines.append(f"| {r['model']} | " + " | ".join(cells) + " |")

    md_lines += ["", "## Per-signal score means (cross-model)", ""]
    md_lines += [
        "| Signal | Opus | "
        + " | ".join(r["model"] for r in rows_sorted) + " |",
        "|--------|------|" + "|".join([":---:"] * len(rows_sorted)) + "|",
    ]
    # For each signal, the Opus mean should be the same across models
    for sid in SIGNAL_ORDER:
        opus_mean = None
        for r in rows_sorted:
            v = r["signals"].get(sid, {})
            if v.get("n"):
                opus_mean = v["opus_mean"]
                break
        if opus_mean is None:
            continue
        cells = [f"{opus_mean:.1f}"]
        for r in rows_sorted:
            v = r["signals"].get(sid, {})
            m = v.get("model_mean", float("nan"))
            cells.append(f"{m:.1f}" if not np.isnan(m) else "—")
        md_lines.append(f"| {sid.split('_')[0]} | " + " | ".join(cells) + " |")

    md_lines += ["", "## Interpretation", ""]
    best = rows_sorted[0] if rows_sorted else None
    if best and not np.isnan(best["mean_spearman_per_signal"]):
        md_lines.append(
            f"- Best tinker grader: **{best['model']}** with mean ρ = "
            f"{best['mean_spearman_per_signal']:+.3f}."
        )
    md_lines += [
        "- Higher ρ → grader ranks plans more similarly to Opus.",
        "- Low MAE → grader calibrates absolute scores closer to Opus.",
        "- within-1 ≥ 0.6 is a typical threshold for inter-rater agreement.",
    ]

    md_path = OUT / "overall_ranking.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Saved overall_ranking.md to {md_path}")

    # Heatmap
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, max(5, 0.6 * len(rows_sorted) + 2)))
        matrix = []
        labels = []
        for r in rows_sorted:
            row = []
            for sid in SIGNAL_ORDER:
                v = r["signals"].get(sid, {}).get("spearman", float("nan"))
                row.append(v)
            matrix.append(row)
            labels.append(r["model"])

        mat = np.array(matrix)
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(SIGNAL_ORDER)))
        ax.set_xticklabels([s.split("_")[0] for s in SIGNAL_ORDER], rotation=45)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title("Spearman ρ vs Opus per (model, v8 signal) — 15 plans")

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                ax.text(
                    j, i, f"{v:+.2f}",
                    ha="center", va="center",
                    color="white" if abs(v) > 0.5 else "black",
                    fontsize=8,
                )
        fig.colorbar(im, ax=ax, label="Spearman ρ")
        fig.tight_layout()
        fig.savefig(OUT / "heatmap.png", dpi=120, bbox_inches="tight")
        print(f"Saved heatmap.png")
    except ImportError:
        print("matplotlib not available; skipping heatmap")


if __name__ == "__main__":
    main()
