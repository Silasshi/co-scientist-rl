"""Compute v8-vs-v9 comparison metrics for the minimal grader panel.

Outputs `analysis/v8_vs_v9_comparison.md` with:
- Per-signal R²(log_wc → score) for v8 vs v9 on GPT-OSS-120B grades
- Per-signal Spearman ρ(Opus, GPT-OSS) for v8 vs v9
- Per-signal variance + ceiling hit rate (for G4/G10 dead-channel check)
- Pass/fail against v9 success criteria
"""
from __future__ import annotations

import json
import math
from pathlib import Path


PANEL_V8 = Path(__file__).resolve().parents[2] / "grader_panel_v8"
PANEL_V9 = Path(__file__).resolve().parent.parent

CHANGED = ["G4_focus", "G10_approach_coverage", "G11_evidence_rigor",
           "G12_formalism", "G13_risk_awareness"]
UNCHANGED = ["G1_problem_specificity", "G2_specific_aims", "G6_reasoning_depth",
             "G8_deliverable_clarity", "G9_scope_feasibility"]


def load_grades(path: Path) -> dict[tuple[str, str], int]:
    """Return {(plan_id, signal_id): score}."""
    out: dict[tuple[str, str], int] = {}
    if not path.exists():
        return out
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


def load_sources() -> dict[str, dict]:
    """Return {plan_id: source_record} with word_count."""
    out = {}
    for l in open(PANEL_V8 / "test_plans/sources.jsonl"):
        r = json.loads(l)
        out[r["plan_id"]] = r
    return out


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")

    def ranks(vals: list[float]) -> list[float]:
        idx = sorted(enumerate(vals), key=lambda iv: iv[1])
        r = [0.0] * len(vals)
        i = 0
        while i < len(idx):
            j = i
            while j < len(idx) and idx[j][1] == idx[i][1]:
                j += 1
            avg = (i + j - 1) / 2.0 + 1
            for k in range(i, j):
                r[idx[k][0]] = avg
            i = j
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    return num / (dx ** 0.5 * dy ** 0.5) if dx > 0 and dy > 0 else float("nan")


def r2_loglin(wcs: list[float], ys: list[float]) -> float:
    if len(wcs) < 3:
        return float("nan")
    xs = [math.log(w) for w in wcs if w > 0]
    ys = [y for w, y in zip(wcs, ys) if w > 0]
    n = len(xs)
    if n < 3:
        return float("nan")
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    b = num / denom
    a = y_mean - b * x_mean
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def variance(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return sum((v - m) ** 2 for v in vals) / (len(vals) - 1)


def main() -> None:
    sources = load_sources()

    v8_gptoss = load_grades(PANEL_V8 / "grades/gpt_oss_120b.jsonl")
    v9_gptoss = load_grades(PANEL_V9 / "grades/gpt_oss_120b.jsonl")
    v8_opus = load_grades(PANEL_V8 / "grades/opus.jsonl")
    v9_opus = load_grades(PANEL_V9 / "grades/opus_v9.jsonl")

    print(f"Loaded: v8 gptoss={len(v8_gptoss)}, v9 gptoss={len(v9_gptoss)}, "
          f"v8 opus={len(v8_opus)}, v9 opus={len(v9_opus)}")

    # Build comparison rows
    rows_out = []
    for sig in CHANGED + UNCHANGED:
        row = {"signal": sig}
        for suffix, grades in [("v8_gptoss", v8_gptoss), ("v9_gptoss", v9_gptoss)]:
            plans = sorted({p for p, s in grades if s == sig})
            wcs = [sources[p]["word_count"] for p in plans if p in sources]
            ys = [grades[(p, sig)] for p in plans if p in sources]
            if len(ys) >= 3:
                row[f"{suffix}_n"] = len(ys)
                row[f"{suffix}_mean"] = sum(ys) / len(ys)
                row[f"{suffix}_var"] = variance(ys)
                row[f"{suffix}_max_frac"] = sum(1 for y in ys if y == 5) / len(ys)
                row[f"{suffix}_r2"] = r2_loglin(wcs, ys)
            else:
                row[f"{suffix}_n"] = len(ys)

        # Opus-GPT-OSS correlation, same grader version
        for label, opus, gptoss in [("v8", v8_opus, v8_gptoss), ("v9", v9_opus, v9_gptoss)]:
            pairs = [
                (opus[(p, sig)], gptoss[(p, sig)])
                for (p, s) in opus if s == sig and (p, s) in gptoss
            ]
            if len(pairs) >= 3:
                xs = [p[0] for p in pairs]
                ys = [p[1] for p in pairs]
                row[f"{label}_opus_gptoss_rho"] = spearman(xs, ys)
                row[f"{label}_opus_gptoss_n"] = len(pairs)
            else:
                row[f"{label}_opus_gptoss_n"] = len(pairs)
        rows_out.append(row)

    # Print table
    print("\n=== Per-signal comparison ===")
    print(f"{'signal':<28s} | {'v8 R²':>7s} {'v9 R²':>7s} | {'v8 var':>7s} {'v9 var':>7s} | {'v8 max%':>7s} {'v9 max%':>7s} | {'v8 ρ':>6s} {'v9 ρ':>6s}")
    for row in rows_out:
        r2_v8 = row.get("v8_gptoss_r2", float("nan"))
        r2_v9 = row.get("v9_gptoss_r2", float("nan"))
        var_v8 = row.get("v8_gptoss_var", float("nan"))
        var_v9 = row.get("v9_gptoss_var", float("nan"))
        mx_v8 = row.get("v8_gptoss_max_frac", float("nan"))
        mx_v9 = row.get("v9_gptoss_max_frac", float("nan"))
        rho_v8 = row.get("v8_opus_gptoss_rho", float("nan"))
        rho_v9 = row.get("v9_opus_gptoss_rho", float("nan"))
        print(f"{row['signal']:<28s} | {r2_v8:>7.3f} {r2_v9:>7.3f} | {var_v8:>7.2f} {var_v9:>7.2f} | {mx_v8:>7.2f} {mx_v9:>7.2f} | {rho_v8:>+6.2f} {rho_v9:>+6.2f}")

    # Evaluate success criteria
    print("\n=== v9 success criteria ===")
    criteria = {
        "G4_focus":             {"var": (">=", 0.8)},
        "G10_approach_coverage": {"mean": ("<",  4.5)},
        "G11_evidence_rigor":   {"r2":  ("<",  0.25)},
        "G12_formalism":        {"r2":  ("<",  0.25)},
        "G13_risk_awareness":   {"r2":  ("<",  0.25)},
    }
    for row in rows_out:
        sig = row["signal"]
        if sig not in criteria:
            continue
        for metric, (op, threshold) in criteria[sig].items():
            v9_key = f"v9_gptoss_{metric}"
            v9_val = row.get(v9_key)
            if v9_val is None:
                verdict = "N/A"
            elif op == "<":
                verdict = "PASS" if v9_val < threshold else "FAIL"
            elif op == ">=":
                verdict = "PASS" if v9_val >= threshold else "FAIL"
            print(f"  {sig} {metric} {op} {threshold}: v9={v9_val:.3f} → {verdict}")

        # Opus-preserve check
        rho = row.get("v9_opus_gptoss_rho")
        if rho is not None:
            verdict = "PASS" if rho >= 0.65 else "FAIL"
            print(f"  {sig} Opus-GPT-OSS ρ ≥ 0.65: v9={rho:+.3f} → {verdict}")

    # Save json artifact
    out = PANEL_V9 / "analysis/comparison.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows_out, indent=2, default=str))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
