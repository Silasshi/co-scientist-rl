"""D5 audit v2 validation — re-score 48 pairwise plans, check concordance.

Loads 48 unique plans from `runs/2026_04_25_pairwise_v1/matchups_meta.json`
(6 baselines × 8 plans), submits batched v2 audit requests, computes
directional agreement against pairwise tournament outcomes, writes
`audit_v2_concordance.md`.

Acceptance gate (strict, per user 2026-04-25):
- ≥ 6/7 directional agreement on pairwise matchups
- All 3 v1-inverted matchups (μ-δ, smoke_A-δ, α-δ) flip to point correctly
- |Δ(δ - μ)| sign should flip on math + realism dims (sanity)

Usage (Phase 1):
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.validate_audit_v2 \
        out_path=projects/d5_abstract_retrieve_refine/runs/2026_04_25_audit_v2_validation

Then main agent dispatches one v2-prompt subagent per audit_requests/iter_NNN.json
that appears. Re-run with `analyze=true` after responses land to compute
concordance.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz

from co_scientist.shared.opus_audit_subagent import OpusAuditClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    pairwise_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_25_pairwise_v1"
    out_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_25_audit_v2_validation"
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"
    batch_size: int = 8
    timeout_min: int = 30
    analyze: bool = False  # set true after responses arrive


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _load_plan_index(meta_path: Path) -> dict[str, tuple[str, str]]:
    """Return {namespaced_plan_id: (label, plan_text)} de-duplicated.

    Namespaces plan_id with label prefix because μ and β baselines share
    raw plan_ids like `iter_008_eval_0` (no overlap of texts, just same
    naming convention). Output IDs are e.g. `mu::iter_008_eval_0`.
    """
    matchups_meta = json.loads(meta_path.read_text())
    out: dict[str, tuple[str, str]] = {}
    for matchup_id, matchups in matchups_meta.items():
        for m in matchups:
            for side in ("a", "b"):
                raw_pid = m[f"plan_{side}_id"]
                txt = m[f"plan_{side}_text"]
                lbl = m[f"true_{side}_label"]
                pid = f"{lbl}::{raw_pid}"
                out.setdefault(pid, (lbl, txt))
    return out


def _parse_pairwise_outcomes(summary_path: Path, meta_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Return {(la, lb): {wins_la, wins_lb, ties, winner}} from response files.

    Uses true_a_label / true_b_label from the matchup entries (not matchup_id
    string-split) so labels like `smoke_v3_A` map correctly even when matchup_id
    abbreviates them (e.g. `smokeA_vs_delta`).
    """
    matchups_meta = json.loads(meta_path.read_text())
    out: dict[tuple[str, str], dict[str, Any]] = {}
    resp_dir = summary_path.parent / "pairwise_responses"
    for matchup_id, matchups in matchups_meta.items():
        if not matchups: continue
        # Take the unswapped first matchup to recover canonical (la, lb).
        unswapped = next((m for m in matchups if not m.get("position_swapped", False)), matchups[0])
        if unswapped.get("position_swapped", False):
            la = unswapped["true_b_label"]
            lb = unswapped["true_a_label"]
        else:
            la = unswapped["true_a_label"]
            lb = unswapped["true_b_label"]
        rp = resp_dir / f"{matchup_id}.json"
        if not rp.exists():
            continue
        verdicts = json.loads(rp.read_text()).get("verdicts", {})
        wla = wlb = ties = 0
        for m in matchups:
            v = verdicts.get(m["pair_id"])
            if v is None: continue
            w = (v.get("winner") or "").upper().strip()
            if w == "TIE": ties += 1
            elif w == "A":
                if m["true_a_label"] == la: wla += 1
                elif m["true_a_label"] == lb: wlb += 1
            elif w == "B":
                if m["true_b_label"] == la: wla += 1
                elif m["true_b_label"] == lb: wlb += 1
        if wla > wlb: winner = la
        elif wlb > wla: winner = lb
        else: winner = "tie"
        out[(la, lb)] = {"wins_la": wla, "wins_lb": wlb, "ties": ties, "winner": winner}
    return out


def submit_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = _resolve(config.pairwise_run) / "matchups_meta.json"
    plan_index = _load_plan_index(meta_path)
    n_plans = len(plan_index)
    logger.info("Loaded %d unique plans from %s", n_plans, meta_path)

    # Persist plan index for analyze phase
    pidx_path = out_dir / "plan_index.json"
    pidx_path.write_text(json.dumps(
        {pid: {"label": lbl, "text_chars": len(txt)} for pid, (lbl, txt) in plan_index.items()},
        indent=2,
    ))

    goal = _resolve(config.goal_path).read_text().strip()
    audit = OpusAuditClient(log_path=out_dir)

    items = list(plan_index.items())
    n_batches = 0
    for batch_idx, start in enumerate(range(0, len(items), config.batch_size)):
        batch = items[start:start + config.batch_size]
        plans_payload = [{"plan_id": pid, "text": txt} for pid, (_lbl, txt) in batch]
        payload = {
            "kind": "audit",
            "prompt_version": "v2",
            "iter": batch_idx,
            "goal": goal,
            "plans": plans_payload,
        }
        audit.submit(iter_idx=batch_idx, payload=payload)
        n_batches += 1
    logger.info("Submitted %d v2 audit batches → %s", n_batches, out_dir)
    logger.info("Now main agent should dispatch %d subagents (one per batch).", n_batches)
    logger.info("After responses arrive, re-run with `analyze=true` for concordance.")


def analyze_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    pairwise_meta = _resolve(config.pairwise_run) / "matchups_meta.json"
    pairwise_summary = _resolve(config.pairwise_run) / "pairwise_summary.md"

    plan_index_path = out_dir / "plan_index.json"
    plan_index = json.loads(plan_index_path.read_text())  # pid -> {label, text_chars}

    audit = OpusAuditClient(log_path=out_dir)
    summary = audit.collect_all(
        out_path=out_dir / "audit_log_v2.jsonl",
        timeout_sec=10.0, poll_interval_sec=2.0,
    )
    logger.info("Collected %d batches", len(summary))

    # Flatten to plan_id -> {math, novelty, realism, rigor, total}
    per_plan: dict[str, dict[str, Any]] = {}
    for batch in summary:
        for rec in batch.get("per_plan", []):
            per_plan[rec["plan_id"]] = rec

    # Aggregate by label
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pid, rec in per_plan.items():
        if pid not in plan_index: continue
        lbl = plan_index[pid]["label"]
        by_label[lbl].append(rec)

    label_means: dict[str, dict[str, float]] = {}
    for lbl, recs in by_label.items():
        valid = [r for r in recs if r.get("total") is not None]
        if not valid:
            label_means[lbl] = {"n": 0, "mean_total": None}
            continue
        label_means[lbl] = {
            "n": len(valid),
            "mean_total": float(np.mean([r["total"] for r in valid])),
            "mean_math": float(np.mean([r["math"] for r in valid])),
            "mean_novelty": float(np.mean([r["novelty"] for r in valid])),
            "mean_realism": float(np.mean([r["realism"] for r in valid])),
            "mean_rigor": float(np.mean([r["rigor"] for r in valid])),
        }

    pairwise_outcomes = _parse_pairwise_outcomes(pairwise_summary, pairwise_meta)

    # v1 absolute scores from earlier (hardcoded reference, last-3 means)
    v1_scores = {
        "epsilon": 12.13, "mu": 8.88, "beta": 7.75, "delta": 6.75, "alpha": 6.12,
        # smoke_v3_A: from runs/2026_04_smoke_pathway_v3_rerun_div absolute
        "smoke_v3_A": 9.25,
    }

    # Inversions to check
    target_inversions = [("mu", "delta"), ("alpha", "delta"), ("smoke_v3_A", "delta")]

    rows = []
    direct_agree = 0
    n_total = 0
    inversions_flipped: dict[tuple[str, str], bool] = {}
    for (la, lb), outcome in pairwise_outcomes.items():
        n_total += 1
        pw_winner = outcome["winner"]
        ma = label_means.get(la, {}).get("mean_total")
        mb = label_means.get(lb, {}).get("mean_total")
        v2_winner = "tie" if (ma is None or mb is None) else (la if ma > mb else lb if mb > ma else "tie")
        # Tie tolerance: within 0.5 counts as tie on v2 side
        if ma is not None and mb is not None and abs(ma - mb) < 0.5:
            v2_winner = "tie"
        # v1 winner
        v1a = v1_scores.get(la); v1b = v1_scores.get(lb)
        v1_winner = "tie" if (v1a is None or v1b is None or v1a == v1b) else (la if v1a > v1b else lb)
        agree_pw_v2 = (pw_winner == v2_winner)
        if agree_pw_v2: direct_agree += 1
        rows.append({
            "matchup": f"{la}_vs_{lb}",
            "pw_winner": pw_winner, "pw_score": f"{outcome['wins_la']}-{outcome['wins_lb']}",
            "v1_winner": v1_winner, "v1_score": f"{v1a:.2f} vs {v1b:.2f}" if (v1a and v1b) else "?",
            "v2_winner": v2_winner, "v2_score": f"{ma:.2f} vs {mb:.2f}" if (ma is not None and mb is not None) else "?",
            "agree": agree_pw_v2,
        })
        if (la, lb) in target_inversions:
            inversions_flipped[(la, lb)] = (pw_winner == v2_winner == lb)

    # Per-dim discrimination μ vs δ
    per_dim_diff = {}
    if "mu" in label_means and "delta" in label_means:
        for dim in ["math", "novelty", "realism", "rigor"]:
            m_mu = label_means["mu"].get(f"mean_{dim}")
            m_de = label_means["delta"].get(f"mean_{dim}")
            if m_mu is not None and m_de is not None:
                per_dim_diff[dim] = m_de - m_mu

    # Acceptance check
    pass_directional = direct_agree >= 6
    pass_inversions = all(inversions_flipped.values())
    pass_dim = (per_dim_diff.get("math", 0) > 0 and per_dim_diff.get("realism", 0) > 0)
    overall_pass = pass_directional and pass_inversions

    md = ["# D5 Audit Prompt v2 — Validation Concordance Report\n"]
    md.append(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
    md.append("## Per-baseline mean scores (v2)\n\n")
    md.append("| Baseline | n | total /20 | math | novelty | realism | rigor |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for lbl in sorted(label_means.keys(), key=lambda k: -(label_means[k].get("mean_total") or -1)):
        d = label_means[lbl]
        if d["n"] == 0:
            md.append(f"| {lbl} | 0 | — | — | — | — | — |\n")
        else:
            md.append(f"| {lbl} | {d['n']} | {d['mean_total']:.2f} | {d['mean_math']:.2f} | {d['mean_novelty']:.2f} | {d['mean_realism']:.2f} | {d['mean_rigor']:.2f} |\n")

    md.append("\n## v1 vs v2 vs Pairwise — 7-matchup ranking\n\n")
    md.append("| Matchup | Pairwise | v1 (absolute) | v2 (absolute) | v2 = Pairwise? |\n|---|---|---|---|:-:|\n")
    for r in rows:
        sym = "✓" if r["agree"] else "🚨"
        md.append(f"| {r['matchup']} | {r['pw_winner']} ({r['pw_score']}) | {r['v1_winner']} ({r['v1_score']}) | {r['v2_winner']} ({r['v2_score']}) | {sym} |\n")
    md.append(f"\n**Directional agreement**: {direct_agree}/{n_total}\n")

    md.append("\n## Inversion-target check\n\n")
    md.append("| Inverted matchup (v1) | Pairwise winner | v2 winner | Flipped under v2? |\n|---|---|---|:-:|\n")
    for (la, lb), flipped in inversions_flipped.items():
        sym = "✓" if flipped else "🚨"
        v2_w = next((r["v2_winner"] for r in rows if r["matchup"] == f"{la}_vs_{lb}"), "?")
        md.append(f"| {la}_vs_{lb} | {lb} | {v2_w} | {sym} |\n")

    md.append("\n## Per-dim discrimination (δ − μ; positive means δ > μ)\n\n")
    for dim, diff in per_dim_diff.items():
        sym = "✓" if diff > 0 else "🚨"
        md.append(f"- {dim}: {diff:+.2f}  {sym}\n")

    md.append("\n## Acceptance verdict\n\n")
    md.append(f"- Directional agreement ≥ 6/7? {'✓' if pass_directional else '🚨'}  ({direct_agree}/7)\n")
    md.append(f"- All 3 v1-inverted matchups flipped? {'✓' if pass_inversions else '🚨'}\n")
    md.append(f"- Per-dim sanity (math + realism flip)? {'✓' if pass_dim else 'WARN'}\n")
    md.append(f"\n### **{'PASS' if overall_pass else 'FAIL'}**\n")
    if not overall_pass:
        md.append("\nIf FAIL: diagnose by sampling 5 μ + 5 δ v2 outputs, inspect "
                   "scaffold_list completeness on Pattern-N labels. Iterate ONCE "
                   "on anchor wording or penalty cap (-2/dim) before declaring v2 dead.\n")

    out_md = out_dir / "audit_v2_concordance.md"
    out_md.write_text("".join(md))
    logger.info("Wrote concordance report: %s", out_md)
    print(out_md.read_text())


def main(config: Config) -> None:
    if config.analyze:
        analyze_phase(config)
    else:
        submit_phase(config)


if __name__ == "__main__":
    cfg = chz.entrypoint(Config)
    main(cfg)
