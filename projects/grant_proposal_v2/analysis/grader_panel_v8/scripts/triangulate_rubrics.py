"""Triangulate v8-Opus and depth_audit-Opus grades.

Questions answered:
1. How correlated are the two Opus rubrics (plan-level aggregate)?
2. For each tinker model: correlation with v8-Opus AND depth_audit-Opus?
3. Models that correlate with only one → just learned to follow that rubric.
   Models that correlate with both → robust grader signal.
"""
from __future__ import annotations

import json
from pathlib import Path

PANEL = Path(__file__).resolve().parent.parent
GRADES = PANEL / "grades"
OUT = PANEL / "analysis"


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")

    def ranks(vals: list[float]) -> list[float]:
        indexed = sorted(enumerate(vals), key=lambda iv: iv[1])
        r = [0.0] * len(vals)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) and indexed[j][1] == indexed[i][1]:
                j += 1
            avg = (i + j - 1) / 2.0 + 1
            for k in range(i, j):
                r[indexed[k][0]] = avg
            i = j
        return r

    rx, ry = ranks(list(x)), ranks(list(y))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx**0.5 * dy**0.5)


# Load v8-Opus per-plan aggregate (sum of 10 v8 signal scores, normalized)
import sys
sys.path.insert(0, str(PANEL.parent.parent.parent.parent / "src"))
from co_scientist.shared.grant_rubric_v8 import SIGNALS, SIGNAL_WEIGHTS

def load_v8_aggregate() -> dict[str, float]:
    """Return {plan_id: weighted aggregate score}."""
    rows = [json.loads(l) for l in open(GRADES / "opus.jsonl")]
    per_plan: dict[str, dict[str, int]] = {}
    for r in rows:
        per_plan.setdefault(r["plan_id"], {})[r["signal_id"]] = int(r["score"])
    out = {}
    for pid, sigs in per_plan.items():
        # Weighted mean of normalized scores
        total = 0.0
        for s in SIGNALS:
            v = sigs.get(s.id)
            if v is None:
                continue
            norm = (v - 1) / (s.score_max - 1)  # [0, 1]
            total += SIGNAL_WEIGHTS[s.id] * norm
        out[pid] = total
    return out


def load_depth_audit() -> dict[str, float]:
    """Return {plan_id: depth_audit total /40}."""
    rows = [json.loads(l) for l in open(GRADES / "opus_depth_audit.jsonl")]
    return {r["plan_id"]: float(r.get("total", r["depth"] + r["methods"] + r["feasibility"] + r["grounding"])) for r in rows}


def load_tinker_aggregate(model_file: Path) -> dict[str, float]:
    """Return {plan_id: v8 weighted aggregate} from a tinker grader file."""
    rows = [json.loads(l) for l in open(model_file)]
    per_plan: dict[str, dict[str, int]] = {}
    for r in rows:
        s = r.get("score")
        if s is None:
            continue
        per_plan.setdefault(r["plan_id"], {})[r["signal_id"]] = int(s)
    out = {}
    sig_max = {s.id: s.score_max for s in SIGNALS}
    weights = dict(SIGNAL_WEIGHTS)
    for pid, sigs in per_plan.items():
        total = 0.0
        wtot = 0.0
        for sid, w in weights.items():
            v = sigs.get(sid)
            if v is None:
                continue
            sm = sig_max[sid]
            total += w * ((v - 1) / (sm - 1))
            wtot += w
        # Normalize by coverage
        if wtot > 0:
            out[pid] = total * (1.0 / wtot)
    return out


def main() -> None:
    v8 = load_v8_aggregate()
    da = load_depth_audit()
    common = sorted(set(v8) & set(da))
    print(f"Plans with both v8-Opus and depth_audit: {len(common)}")

    # Correlation between the two Opus rubrics
    v8_list = [v8[p] for p in common]
    da_list = [da[p] for p in common]
    rho_opus = spearman(v8_list, da_list)
    print(f"\nSpearman(v8-Opus aggregate, depth_audit-Opus total) = {rho_opus:+.3f}")
    print(f"  (high ρ → two Opus rubrics agree at the plan-level aggregate)")

    # Per-plan comparison table
    print("\n| plan_id | v8 agg | depth/40 | quality |")
    print("|---------|-------:|---------:|---------|")
    sources = {
        json.loads(l)["plan_id"]: json.loads(l)["quality_bucket"]
        for l in open(PANEL / "test_plans/sources.jsonl")
    }
    for p in common:
        print(f"| {p} | {v8[p]:.3f} | {da[p]:.0f} | {sources.get(p, '?')} |")

    # For each tinker model: correlation with both Opus rubrics
    print("\n=== Tinker model agreement with each Opus rubric (plan-level) ===\n")
    print(f"{'Model':<22s} {'ρ vs v8-Opus':>13s} {'ρ vs depth':>13s} {'n':>4s}")
    print("-" * 60)
    tinker_files = sorted(
        [p for p in GRADES.glob("*.jsonl")
         if p.stem not in ("opus", "opus_depth_audit") and not p.stem.startswith("opus_")]
    )
    results = []
    for mf in tinker_files:
        agg = load_tinker_aggregate(mf)
        model = mf.stem
        pids = sorted(set(agg) & set(v8) & set(da))
        if len(pids) < 3:
            results.append({"model": model, "rho_v8": float("nan"), "rho_depth": float("nan"), "n": len(pids)})
            continue
        rho_v8 = spearman([agg[p] for p in pids], [v8[p] for p in pids])
        rho_da = spearman([agg[p] for p in pids], [da[p] for p in pids])
        results.append({"model": model, "rho_v8": rho_v8, "rho_depth": rho_da, "n": len(pids)})
        print(f"{model:<22s} {rho_v8:>+12.3f}  {rho_da:>+12.3f}  {len(pids):>4d}")

    # Save
    out = {
        "rho_opus_rubrics": rho_opus,
        "tinker_agreement": results,
    }
    (OUT / "triangulation.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {OUT / 'triangulation.json'}")


if __name__ == "__main__":
    main()
