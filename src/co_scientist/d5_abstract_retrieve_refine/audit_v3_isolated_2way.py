"""D5 Phase 4a — 2-baseline ISOLATED audit (σ' vs μ' for cross-goal validation).

Fork of audit_v3_isolated.py simplified for the 2-baseline case used in Phase 4a:
8 balanced batches × 2 plans (1 σ', 1 μ', anonymized A/B per batch). Each batch
goes to ONE Opus subagent that scores BOTH plans independently using the 9-dim
rubric (5 universal + 4 subfield).

Both baselines load from `buffer.jsonl` (baseline_frozen_v2 output for both
variant=sigma and variant=mu_inference).

Usage (per goal):
    # Phase 1 — write batch request files
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.audit_v3_isolated_2way \
        out_path=projects/d5_abstract_retrieve_refine/runs/2026_04_28_xgoal_<NAME>_audit \
        goal_path=projects/d5_abstract_retrieve_refine/data/cross_goal/<NAME>/research_goal.txt \
        sigma_run=projects/d5_abstract_retrieve_refine/runs/2026_04_28_xgoal_<NAME>_sigma \
        mu_run=projects/d5_abstract_retrieve_refine/runs/2026_04_28_xgoal_<NAME>_mu

    # External: dispatch 8 Opus subagents per goal (one per batch_NN.json)

    # Phase 2 — aggregate per-baseline + per-dim means
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.audit_v3_isolated_2way \
        out_path=... analyze=true
"""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

import chz

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v2 import extract_solution

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    out_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_xgoal_<name>_audit"
    goal_path: str = "projects/d5_abstract_retrieve_refine/data/cross_goal/<name>/research_goal.txt"

    sigma_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_xgoal_<name>_sigma"
    mu_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_xgoal_<name>_mu"

    seed: int = 7777
    n_batches: int = 8
    analyze: bool = False


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _load_frozen(run_dir: Path, label: str) -> list[dict]:
    """Load buffer.jsonl produced by baseline_frozen_v2 (variant=sigma | mu_inference)."""
    bp = run_dir / "buffer.jsonl"
    plans = []
    if not bp.exists():
        return plans
    with open(bp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            plans.append({
                "true_baseline": label,
                "true_plan_id": f"{label}_{d.get('sample_idx', '?')}",
                "text": extract_solution(d.get("plan_text", "")),
            })
    return plans


def submit_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    goal = _resolve(config.goal_path).read_text().strip()

    # Load 8 plans per baseline → 2 baselines = 16 plans
    by_baseline: dict[str, list[dict]] = {
        "sigma_prime": _load_frozen(_resolve(config.sigma_run), "sigma_prime"),
        "mu_prime":    _load_frozen(_resolve(config.mu_run),    "mu_prime"),
    }

    rng = random.Random(config.seed)
    for plans in by_baseline.values():
        rng.shuffle(plans)

    # Sanity-check plan counts
    for label, plans in by_baseline.items():
        if len(plans) != config.n_batches:
            logger.warning("%s has %d plans, expected %d", label, len(plans), config.n_batches)

    # Build n_batches balanced batches: each batch has exactly 1 σ' + 1 μ' = 2 plans
    batches: list[list[dict]] = [[] for _ in range(config.n_batches)]
    for label, plans in by_baseline.items():
        for i, p in enumerate(plans[:config.n_batches]):
            batches[i].append(p)

    # Within each batch, shuffle order + anonymize as plan_A / plan_B
    decode_map = {}
    for bi, batch in enumerate(batches):
        rng.shuffle(batch)  # randomize A/B assignment per batch (anti-position-bias)
        anonymized = []
        for letter, p in zip(["A", "B"], batch):
            anon_id = f"batch_{bi:02d}_plan_{letter}"
            anonymized.append({"plan_id": anon_id, "text": p["text"]})
            decode_map[anon_id] = {
                "true_baseline": p["true_baseline"],
                "true_plan_id": p["true_plan_id"],
                "batch": bi,
            }
        req = {
            "kind": "audit_v3_isolated_2way",
            "batch_id": bi,
            "goal": goal,
            "plans": anonymized,
            "prompt_version": "v3_isolated_2way",
            "instructions": (
                "These 2 plans (plan_A and plan_B) come from 2 different methods on the "
                "SAME research goal (you don't know which is which). Score each plan "
                "INDEPENDENTLY on its 9-dim rubric. Do NOT batch-score, do NOT let one "
                "plan's quality anchor the other. Read each plan's specific equations, "
                "named benchmarks, mechanisms, failure modes — score on what's actually "
                "there, not on surface impressions. The two plans should rarely receive "
                "identical 9-dim score vectors."
            ),
        }
        req_dir = out_dir / "audit_requests"
        req_dir.mkdir(parents=True, exist_ok=True)
        (req_dir / f"batch_{bi:02d}.json").write_text(json.dumps(req))

    (out_dir / "shuffle_map.json").write_text(json.dumps(decode_map, indent=2))
    logger.info("Wrote %d balanced-batch audit requests (1 σ' + 1 μ' per batch)", config.n_batches)


def analyze_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    decode_map = json.loads((out_dir / "shuffle_map.json").read_text())
    resp_dir = out_dir / "audit_responses"
    if not resp_dir.exists():
        logger.error("No audit_responses dir: %s", resp_dir)
        sys.exit(1)

    all_judg = {}
    for f in sorted(resp_dir.glob("batch_*.json")):
        d = json.loads(f.read_text())
        all_judg.update(d.get("judgments", {}))
    logger.info("Collected %d judgments across batches", len(all_judg))

    per_baseline: dict[str, list[dict]] = {}
    for anon_id, j in all_judg.items():
        meta = decode_map.get(anon_id)
        if not meta:
            logger.warning("Decoding miss: %s", anon_id)
            continue
        bl = meta["true_baseline"]
        scores = j.get("per_dim_scores", {})
        total = j.get("total_raw") or (sum(scores.values()) if scores else None)
        per_baseline.setdefault(bl, []).append({
            "anon_id": anon_id, "true_plan_id": meta["true_plan_id"],
            "scores": scores, "total": total,
            "key_differentiator": j.get("key_differentiator", ""),
        })

    DIMS = ["U1", "U2", "U3", "U4", "U5", "T1", "T2", "T3", "T4"]
    rows = []
    for bl, plans in sorted(per_baseline.items()):
        totals = [p["total"] for p in plans if p["total"] is not None]
        if not totals:
            continue
        n = len(totals)
        mean = sum(totals) / n
        var = sum((t - mean) ** 2 for t in totals) / n
        std = var ** 0.5
        dim_means = {}
        for d in DIMS:
            vals = [p["scores"].get(d) for p in plans if d in p["scores"]]
            dim_means[d] = sum(vals) / len(vals) if vals else None
        rows.append({"baseline": bl, "n": n, "mean": mean, "std": std,
                      "min": min(totals), "max": max(totals),
                      "totals": totals, "dim_means": dim_means})

    # Stable order: sigma_prime, mu_prime
    order = {"sigma_prime": 0, "mu_prime": 1}
    rows.sort(key=lambda r: order.get(r["baseline"], 99))

    md = ["# Phase 4a audit_v3 ISOLATED 2way — σ' vs μ' per goal\n\n"]
    md.append(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
    md.append("Methodology: 8 balanced batches (1 σ' + 1 μ' each) × 8 parallel\n")
    md.append("Opus subagents. Each subagent scores 2 anonymized plans (plan_A, plan_B)\n")
    md.append("with strict per-plan independence instructions.\n\n")

    md.append("## Per-baseline aggregates\n\n")
    md.append("| Baseline | N | Mean /45 | Std | Min | Max | Distinct totals |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        n_distinct = len(set(r["totals"]))
        md.append(
            f"| **{r['baseline']}** | {r['n']} | {r['mean']:.2f} | {r['std']:.2f} |"
            f" {r['min']} | {r['max']} | {n_distinct}/{r['n']} |\n"
        )

    if len(rows) == 2:
        sig = next((r for r in rows if r["baseline"] == "sigma_prime"), None)
        mu = next((r for r in rows if r["baseline"] == "mu_prime"), None)
        if sig and mu:
            delta = mu["mean"] - sig["mean"]
            md.append(f"\n**Δ (μ' − σ') = {delta:+.2f}** /45\n\n")

    md.append("\n## Per-dim means matrix\n\n")
    md.append("| Baseline | U1 | U2 | U3 | U4 | U5 | T1 | T2 | T3 | T4 |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        dm = r["dim_means"]
        md.append(
            f"| **{r['baseline']}** | "
            + " | ".join(f"{dm[d]:.2f}" if dm.get(d) is not None else "—" for d in DIMS)
            + " |\n"
        )

    md.append("\n## All totals per baseline\n\n")
    for r in rows:
        md.append(f"- **{r['baseline']}**: {r['totals']}\n")

    out_path = out_dir / "audit_v3_isolated_2way_summary.md"
    out_path.write_text("".join(md))
    logger.info("Wrote summary: %s", out_path)


def main(config: Config):
    if config.analyze:
        analyze_phase(config)
    else:
        submit_phase(config)


if __name__ == "__main__":
    main(chz.entrypoint(Config))
