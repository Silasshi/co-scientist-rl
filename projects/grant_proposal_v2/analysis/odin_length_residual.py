"""ODIN-style length residual analysis for v7/v8 graders.

For each grader condition, fits regression score ~ length and reports:
- R²(length → aggregate reward)
- R²(length → per-signal score)
- Length residual = observed - predicted (per plan)

Output: analysis/odin_length_residual.json with coefficients + residuals.

Reference: ODIN (Ma et al., ICML 2024) — predict reward from length alone,
subtract to get length-decorrelated reward signal.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[3]

V8_RUNS = {
    "v8_B4": PROJECT / "projects/grant_proposal_v2/runs/2026_04_22_v8_foundopt_B4",
    "v8_C3": PROJECT / "projects/grant_proposal_v2/runs/2026_04_22_v8_foundopt_MAIN_C3",
    "v8_C4": PROJECT / "projects/grant_proposal_v2/runs/2026_04_22_v8_foundopt_MAIN_C4",
}
V7_RUNS = {
    "v7_C2": PROJECT / "projects/grant_proposal/runs/2026_04_22_sdpo_C2_B4_scores_only",
    "v7_C3": PROJECT / "projects/grant_proposal/runs/2026_04_22_sdpo_C3_sdpo_scores_only",
    "v7_C4": PROJECT / "projects/grant_proposal/runs/2026_04_22_sdpo_C4_aggregate_scores_only",
}

# Standard score_max = 5 for all v7/v8 signals (verified by grant_rubric_v8.py)
SCORE_MAX = 5


def load_buffer(run_dir: Path) -> list[dict]:
    """Load hard-gate-passed buffer entries."""
    path = run_dir / "buffer.jsonl"
    out = []
    for line in open(path):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not e.get("hard_gate_passed"):
            continue
        if e.get("word_count") is None or e.get("word_count", 0) < 50:
            continue
        out.append(e)
    return out


def fit_linear(xs: list[float], ys: list[float]) -> dict:
    """OLS y = a + b*x. Return {a, b, r2, n, x_mean, y_mean}."""
    n = len(xs)
    if n < 3:
        return {"a": 0.0, "b": 0.0, "r2": float("nan"), "n": n}
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return {"a": y_mean, "b": 0.0, "r2": 0.0, "n": n, "x_mean": x_mean, "y_mean": y_mean}
    b = num / denom
    a = y_mean - b * x_mean
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"a": a, "b": b, "r2": r2, "n": n, "x_mean": x_mean, "y_mean": y_mean}


def analyze(runs: dict[str, Path], label: str) -> dict:
    """Pool buffer entries across a set of runs and fit length → score models."""
    all_entries = []
    per_run_counts = {}
    for rname, rdir in runs.items():
        entries = load_buffer(rdir)
        per_run_counts[rname] = len(entries)
        for e in entries:
            e["_run"] = rname
            all_entries.append(e)

    if not all_entries:
        return {"label": label, "error": "no entries"}

    wcs = [float(e["word_count"]) for e in all_entries]
    log_wcs = [math.log(w) for w in wcs]
    aggs = [float(e["aggregate_reward"]) for e in all_entries]

    agg_linear = fit_linear(wcs, aggs)
    agg_loglin = fit_linear(log_wcs, aggs)

    # Per-signal
    # Use first entry to discover signal ids
    sig_ids = sorted((all_entries[0].get("signal_vector") or {}).keys())
    per_signal = {}
    for sid in sig_ids:
        ys = []
        xs = []
        lxs = []
        for e in all_entries:
            v = (e.get("signal_vector") or {}).get(sid)
            if v is None:
                continue
            y = (v - 1) / (SCORE_MAX - 1)
            ys.append(y)
            xs.append(float(e["word_count"]))
            lxs.append(math.log(float(e["word_count"])))
        per_signal[sid] = {
            "linear": fit_linear(xs, ys),
            "loglin": fit_linear(lxs, ys),
        }

    # Residuals (log-linear, which usually fits better)
    a, b = agg_loglin["a"], agg_loglin["b"]
    residuals = []
    for e in all_entries:
        predicted = a + b * math.log(float(e["word_count"]))
        res = float(e["aggregate_reward"]) - predicted
        residuals.append({
            "run": e["_run"],
            "iteration": e.get("iteration"),
            "word_count": e["word_count"],
            "observed_reward": float(e["aggregate_reward"]),
            "predicted_from_length": predicted,
            "residual": res,
        })

    return {
        "label": label,
        "n_total": len(all_entries),
        "per_run_counts": per_run_counts,
        "agg_linear": agg_linear,
        "agg_loglin": agg_loglin,
        "per_signal": per_signal,
        "residuals": residuals,
    }


def summarize(result: dict) -> None:
    print(f"\n=== {result['label']} (n={result['n_total']}, per-run: {result['per_run_counts']}) ===")
    al = result["agg_linear"]
    ag = result["agg_loglin"]
    print(f"  aggregate ~ word_count    : R²={al['r2']:.3f}  b={al['b']:.5f}")
    print(f"  aggregate ~ log(word_count): R²={ag['r2']:.3f}  b={ag['b']:.4f}")
    print(f"  per-signal R² (log-linear):")
    for sid, d in sorted(result["per_signal"].items()):
        print(f"    {sid:30s}: R²={d['loglin']['r2']:+.3f}  b={d['loglin']['b']:+.4f}  n={d['loglin']['n']}")


def main() -> None:
    v8 = analyze(V8_RUNS, "v8 (GPT-OSS-120B grader)")
    v7 = analyze(V7_RUNS, "v7 (Qwen-30B grader)")

    summarize(v8)
    summarize(v7)

    out_path = PROJECT / "projects/grant_proposal_v2/analysis/odin_length_residual.json"
    # Trim residuals for json size
    for res in (v8, v7):
        if "residuals" in res:
            res["residuals_sample"] = res["residuals"][:10]
            res["residuals_count"] = len(res["residuals"])
            del res["residuals"]
    out_path.write_text(json.dumps({"v8": v8, "v7": v7}, indent=2))
    print(f"\nSaved to: {out_path}")

    # Comparison summary
    print("\n=== Summary comparison ===")
    print(f"{'':<30s}{'v7':>10s}{'v8':>10s}")
    print(f"{'R²(log_wc → aggregate)':<30s}{v7['agg_loglin']['r2']:>10.3f}{v8['agg_loglin']['r2']:>10.3f}")
    for sid in ["G11_evidence_rigor", "G12_formalism", "G13_risk_awareness",
                "G4_focus", "G6_reasoning_depth", "G9_scope_feasibility",
                "G1_problem_specificity", "G2_specific_aims", "G8_deliverable_clarity", "G10_approach_coverage"]:
        v7_r2 = v7["per_signal"].get(sid, {}).get("loglin", {}).get("r2")
        v8_r2 = v8["per_signal"].get(sid, {}).get("loglin", {}).get("r2")
        v7_s = f"{v7_r2:.3f}" if v7_r2 is not None else "—"
        v8_s = f"{v8_r2:.3f}" if v8_r2 is not None else "—"
        print(f"  R²(log_wc → {sid:22s}){v7_s:>10s}{v8_s:>10s}")


if __name__ == "__main__":
    main()
