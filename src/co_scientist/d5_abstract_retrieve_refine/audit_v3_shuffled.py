"""Phase 2F audit_v3 v2 — shuffled + anonymized to defeat baseline-class templating.

The first audit_v3 pass (audit_v3_runner.py) showed a known LLM-as-judge failure:
auditor gave 8 plans from same baseline IDENTICAL 9-dim scores (e.g., μ all 21,
α all 21, σ all 21). Opus applied a baseline-class template instead of per-plan
scoring. To defeat this:

1. Load all 56 plans (7 baselines × 8 plans)
2. Shuffle into random order
3. Anonymize plan_id → plan_001..plan_056 (no baseline label visible to auditor)
4. Save mapping plan_NNN → (true_baseline, true_plan_id) for post-hoc decoding
5. Submit ONE big audit request with all 56 anonymous plans
6. After response, decode plan_NNN back to baseline + recompute per-baseline aggregates

Usage:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.audit_v3_shuffled \
        out_path=projects/d5_abstract_retrieve_refine/runs/2026_04_26_phase2F_audit_shuffled
        # then dispatch subagent (single session for 56 plans)
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.audit_v3_shuffled \
        out_path=...phase2F_audit_shuffled \
        analyze=true
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
    out_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_26_phase2F_audit_shuffled"
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"

    xi_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_26_xi_v2"
    sigma_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_26_sigma_v2"
    delta_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_26_delta_v2"
    epsilon_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_26_epsilon_v2"
    mu_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_26_mu_v2"
    alpha_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_26_alpha_v2b"
    beta_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_26_beta_v2b"

    mu_iter: int = 8
    alpha_epoch: int = 4
    beta_iter: int = 8

    seed: int = 1234  # reproducibility
    analyze: bool = False


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _load_frozen(run_dir: Path, label: str) -> list[dict]:
    bp = run_dir / "buffer.jsonl"
    plans = []
    if not bp.exists():
        logger.warning("buffer.jsonl missing for %s: %s", label, bp)
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


def _load_eval(run_dir: Path, key_field: str, key_value: int, label: str) -> list[dict]:
    ep = run_dir / "eval_rollouts.jsonl"
    plans = []
    if not ep.exists():
        logger.warning("eval_rollouts.jsonl missing for %s: %s", label, ep)
        return plans
    with open(ep) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get(key_field) == key_value:
                plans.append({
                    "true_baseline": label,
                    "true_plan_id": d.get("plan_id", "?"),
                    "text": extract_solution(d.get("text", "")),
                })
    return plans


def submit_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    goal = _resolve(config.goal_path).read_text().strip()

    all_plans: list[dict] = []
    all_plans.extend(_load_frozen(_resolve(config.xi_run), "xi"))
    all_plans.extend(_load_frozen(_resolve(config.sigma_run), "sigma"))
    all_plans.extend(_load_frozen(_resolve(config.delta_run), "delta"))
    all_plans.extend(_load_frozen(_resolve(config.epsilon_run), "epsilon"))
    all_plans.extend(_load_eval(_resolve(config.mu_run), "iter", config.mu_iter, "mu"))
    all_plans.extend(_load_eval(_resolve(config.alpha_run), "epoch", config.alpha_epoch, "alpha"))
    all_plans.extend(_load_eval(_resolve(config.beta_run), "iter", config.beta_iter, "beta"))

    rng = random.Random(config.seed)
    rng.shuffle(all_plans)
    logger.info("Loaded + shuffled %d plans (7 baselines × 8 plans expected = 56)", len(all_plans))

    # Anonymize: plan_001 through plan_056
    anonymized = []
    decode_map = {}
    for i, p in enumerate(all_plans, start=1):
        anon_id = f"plan_{i:03d}"
        anonymized.append({"plan_id": anon_id, "text": p["text"]})
        decode_map[anon_id] = {"true_baseline": p["true_baseline"], "true_plan_id": p["true_plan_id"]}

    # Persist shuffle map for analyze phase
    (out_dir / "shuffle_map.json").write_text(json.dumps(decode_map, indent=2))

    # Build single big request
    request = {
        "kind": "audit_v3_shuffled",
        "iter": 0,
        "goal": goal,
        "plans": anonymized,
        "prompt_version": "v3",
        "instructions": (
            "These 56 plans come from 7 different methods, but you are NOT told which "
            "plan came from which method. Score each plan INDEPENDENTLY on its 9-dim "
            "rubric. Do NOT batch-template — read each plan's specific content and "
            "score on what's actually there. Two plans from the same source can score "
            "very differently if their content differs."
        ),
    }
    req_dir = out_dir / "audit_requests"
    req_dir.mkdir(parents=True, exist_ok=True)
    (req_dir / "audit_shuffled.json").write_text(json.dumps(request))
    logger.info("Wrote shuffled audit request: %s (56 plans)", req_dir / "audit_shuffled.json")
    logger.info("Now dispatch a subagent to process this. Then run analyze=true.")


def analyze_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    decode_map = json.loads((out_dir / "shuffle_map.json").read_text())
    resp_path = out_dir / "audit_responses" / "audit_shuffled.json"
    if not resp_path.exists():
        logger.error("Response not found: %s", resp_path)
        sys.exit(1)
    resp = json.loads(resp_path.read_text())
    judgments = resp.get("judgments", {})
    logger.info("Got %d judgments", len(judgments))

    # Decode + group by true_baseline
    per_baseline: dict[str, list[dict]] = {}
    for anon_id, j in judgments.items():
        meta = decode_map.get(anon_id)
        if not meta:
            logger.warning("Decoding miss for %s", anon_id)
            continue
        bl = meta["true_baseline"]
        per_baseline.setdefault(bl, []).append({
            "anon_id": anon_id, "true_plan_id": meta["true_plan_id"],
            "scores": j.get("per_dim_scores", {}),
            "total": j.get("total_raw") or sum(j.get("per_dim_scores", {}).values()),
        })

    # Compute aggregates
    DIMS = ["U1", "U2", "U3", "U4", "U5", "T1", "T2", "T3", "T4"]
    rows = []
    for bl, plans in sorted(per_baseline.items()):
        totals = [p["total"] for p in plans]
        n = len(totals)
        mean = sum(totals) / n
        var = sum((t - mean) ** 2 for t in totals) / n
        std = var ** 0.5
        dim_means = {}
        for d in DIMS:
            vals = [p["scores"].get(d) for p in plans if d in p["scores"]]
            dim_means[d] = sum(vals) / len(vals) if vals else None
        rows.append({"baseline": bl, "n": n, "mean": mean, "std": std,
                      "min": min(totals), "max": max(totals), "dim_means": dim_means,
                      "totals": totals})

    rows.sort(key=lambda r: r["mean"], reverse=True)

    md = ["# Phase 2F audit_v3 SHUFFLED — anonymized per-plan scoring\n\n"]
    md.append(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}, seed={config.seed}*\n\n")
    md.append("## Per-baseline aggregates\n\n")
    md.append("| Baseline | N | Mean /45 | Std | Min | Max | All totals |\n")
    md.append("|---|---:|---:|---:|---:|---:|---|\n")
    for r in rows:
        md.append(
            f"| **{r['baseline']}** | {r['n']} | {r['mean']:.2f} | {r['std']:.2f} |"
            f" {r['min']} | {r['max']} | {r['totals']} |\n"
        )

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

    out_path = out_dir / "audit_v3_shuffled_summary.md"
    out_path.write_text("".join(md))
    logger.info("Wrote summary: %s", out_path)


def main(config: Config):
    if config.analyze:
        analyze_phase(config)
    else:
        submit_phase(config)


if __name__ == "__main__":
    main(chz.entrypoint(Config))
