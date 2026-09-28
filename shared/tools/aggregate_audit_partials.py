"""Aggregate per-plan audit partials into one iter_NNN.json file.

Used by the v8 reviewer cycle: 8 parallel Opus audit subagents each write
audit_responses/iter_NNN_plan_K.partial.json with the audit_v3 9-dim
per-plan schema. This script combines them into the trainer-readable
iter_NNN.json that cliff-stop logic reads at train_mu_v8_d5sdpo.py:682-712.

Usage:
    python shared/tools/aggregate_audit_partials.py \
        --run-dir projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v8_d5sdpo_Gplus_rag \
        --iter 0
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--iter", required=True, type=int)
    p.add_argument("--n-plans", type=int, default=8)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    audit_dir = run_dir / "audit_responses"
    iter_str = f"{args.iter:03d}"

    partials = sorted(audit_dir.glob(f"iter_{iter_str}_plan_*.partial.json"))
    if len(partials) != args.n_plans:
        print(
            f"[aggregate] iter {args.iter}: expected {args.n_plans} partials, "
            f"got {len(partials)} — not yet ready",
            file=sys.stderr,
        )
        sys.exit(1)

    judgments = {}
    totals = []
    for path in partials:
        rec = json.loads(path.read_text())
        plan_id = rec["plan_id"]
        judgments[plan_id] = rec
        totals.append(rec["total"])

    n_plans = len(judgments)
    aggregated = {
        "iter": args.iter,
        "kind": "audit",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "judgments": judgments,
        "n_plans": n_plans,
        "mean_total_45": sum(totals) / n_plans,
        "min_total_45": min(totals),
        "max_total_45": max(totals),
    }

    out_path = audit_dir / f"iter_{iter_str}.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(aggregated, indent=2))
    tmp.replace(out_path)
    print(
        f"[aggregate] iter {args.iter}: {n_plans} plans → "
        f"mean={aggregated['mean_total_45']:.2f}/45, "
        f"min={aggregated['min_total_45']}, max={aggregated['max_total_45']} "
        f"→ {out_path}"
    )


if __name__ == "__main__":
    main()
