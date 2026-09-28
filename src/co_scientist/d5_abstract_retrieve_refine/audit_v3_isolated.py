"""Phase 2F audit_v3 v3 — ISOLATED per-plan scoring (anti-bias gold standard).

Replaces both v1 (baseline-batched, templated) and v2 (shuffled-batched, still
session-anchored). v3 method:

1. Load all 56 plans (7 baselines × 8)
2. Split into 8 BALANCED batches of 7 plans each — every batch contains 1 plan
   from each baseline. This guarantees no batch is "homogeneous" → no
   baseline-class templating possible
3. Anonymize within each batch (plan_A, plan_B, ..., plan_G inside batch)
4. Dispatch 8 PARALLEL subagents — independent calibration anchors
5. Each subagent: strict instruction to score each plan independently, write
   per-plan key_differentiator field to force differentiation

Result: each plan gets exactly ONE score, but 8 different Opus sessions provide
different perspectives. Compare to v1 (templated, 7 baselines × identical scores)
and v2 (shuffled-but-batched). Most discriminating ranking is the cleanest.

Usage:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.audit_v3_isolated
        # writes 8 batch request files + shuffle_map
        # then dispatch 8 parallel subagents externally
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.audit_v3_isolated \
        analyze=true
"""
from __future__ import annotations

import json
import logging
import random
import string
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


# Default 7-baseline TTT-D specs (preserves Phase 2F audit_v3_isolated behavior).
# Override via CLI: baselines_json='[{"label":"xi_prime","mode":"frozen","run":"..."}, ...]'
_DEFAULT_TTT_D_BASELINES = [
    {"label": "xi", "mode": "frozen", "run": "projects/d5_abstract_retrieve_refine/runs/2026_04_26_xi_v2"},
    {"label": "sigma", "mode": "frozen", "run": "projects/d5_abstract_retrieve_refine/runs/2026_04_26_sigma_v2"},
    {"label": "delta", "mode": "frozen", "run": "projects/d5_abstract_retrieve_refine/runs/2026_04_26_delta_v2"},
    {"label": "epsilon", "mode": "frozen", "run": "projects/d5_abstract_retrieve_refine/runs/2026_04_26_epsilon_v2"},
    {"label": "mu", "mode": "eval", "run": "projects/d5_abstract_retrieve_refine/runs/2026_04_26_mu_v2", "key_field": "iter", "key_value": 8},
    {"label": "alpha", "mode": "eval", "run": "projects/d5_abstract_retrieve_refine/runs/2026_04_26_alpha_v2b", "key_field": "epoch", "key_value": 4},
    {"label": "beta", "mode": "eval", "run": "projects/d5_abstract_retrieve_refine/runs/2026_04_26_beta_v2b", "key_field": "iter", "key_value": 8},
]


@chz.chz
class Config:
    out_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_26_phase2F_audit_isolated"
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"

    # JSON-encoded list of {label, mode, run, [key_field, key_value]} dicts.
    # mode in {"frozen" -> load buffer.jsonl, "eval" -> load eval_rollouts.jsonl filtered by key_field=key_value}
    baselines_json: str = ""  # empty => use _DEFAULT_TTT_D_BASELINES

    # Path to the rubric markdown file. Default = TTT-D universal+TTT-T1-T4 rubric.
    # Cross-domain override: knowledge/current/cross_domain/AUDIT_RUBRIC_<domain>.md.
    # Content is embedded into each batch request payload's `rubric` field for audit traceability.
    rubric_path: str = "projects/d5_abstract_retrieve_refine/knowledge/current/AUDIT_RUBRIC_v3.md"

    seed: int = 7777
    n_batches: int = 8
    analyze: bool = False


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _load_frozen(run_dir: Path, label: str) -> list[dict]:
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


def _load_eval(run_dir: Path, key_field: str, key_value: int, label: str) -> list[dict]:
    ep = run_dir / "eval_rollouts.jsonl"
    plans = []
    if not ep.exists():
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
    rubric_path = _resolve(config.rubric_path)
    if not rubric_path.exists():
        raise FileNotFoundError(f"Rubric file not found: {rubric_path}")
    rubric_content = rubric_path.read_text()
    rubric_name = rubric_path.name

    baselines = json.loads(config.baselines_json) if config.baselines_json else _DEFAULT_TTT_D_BASELINES
    by_baseline: dict[str, list[dict]] = {}
    for spec in baselines:
        label = spec["label"]
        mode = spec["mode"]
        run = _resolve(spec["run"])
        if mode == "frozen":
            by_baseline[label] = _load_frozen(run, label)
        elif mode == "eval":
            by_baseline[label] = _load_eval(run, spec["key_field"], spec["key_value"], label)
        else:
            raise ValueError(f"Unknown mode {mode!r} for label {label!r}; expected 'frozen' or 'eval'")
    n_baselines = len(by_baseline)

    rng = random.Random(config.seed)
    for plans in by_baseline.values():
        rng.shuffle(plans)

    # Build n_batches balanced batches: each batch gets 1 plan from each baseline
    # (assuming n_baselines × n_batches plans available; 7×8=56 for TTT-D default).
    batches: list[list[dict]] = [[] for _ in range(config.n_batches)]
    for label, plans in by_baseline.items():
        for i, p in enumerate(plans):
            batches[i % config.n_batches].append(p)

    # Within each batch, shuffle order + anonymize
    decode_map = {}
    for bi, batch in enumerate(batches):
        rng.shuffle(batch)
        anon_letters = string.ascii_uppercase[:len(batch)]
        anonymized = []
        for letter, p in zip(anon_letters, batch):
            anon_id = f"batch_{bi:02d}_plan_{letter}"
            anonymized.append({"plan_id": anon_id, "text": p["text"]})
            decode_map[anon_id] = {
                "true_baseline": p["true_baseline"],
                "true_plan_id": p["true_plan_id"],
                "batch": bi,
            }
        # Build request payload
        req = {
            "kind": "audit_v3_isolated",
            "batch_id": bi,
            "goal": goal,
            "plans": anonymized,
            "prompt_version": "v3_isolated",
            "rubric_name": rubric_name,
            "rubric": rubric_content,
            "instructions": (
                f"These {len(batch)} plans come from {n_baselines} different methods, ONE plan per method "
                "per slot (you don't know which is which). Each plan must be scored "
                "INDEPENDENTLY against the full rubric provided in the `rubric` field. "
                "Do NOT batch-score. Read each plan's specific equations, named benchmarks, "
                "mechanisms, failure modes — score on what's actually there, not on surface "
                "impressions. Two plans should virtually NEVER receive identical scoring "
                "vectors unless their content is genuinely interchangeable."
            ),
        }
        req_dir = out_dir / "audit_requests"
        req_dir.mkdir(parents=True, exist_ok=True)
        (req_dir / f"batch_{bi:02d}.json").write_text(json.dumps(req))

    (out_dir / "shuffle_map.json").write_text(json.dumps(decode_map, indent=2))
    logger.info("Wrote %d balanced-batch audit requests", config.n_batches)
    for bi in range(config.n_batches):
        logger.info("  batch_%02d: %d plans (1 per baseline of %d)", bi, len(batches[bi]), n_baselines)


def analyze_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    decode_map = json.loads((out_dir / "shuffle_map.json").read_text())
    resp_dir = out_dir / "audit_responses"
    if not resp_dir.exists():
        logger.error("No audit_responses dir: %s", resp_dir)
        sys.exit(1)

    # Collect all judgments across batches
    all_judg = {}
    for f in sorted(resp_dir.glob("batch_*.json")):
        d = json.loads(f.read_text())
        all_judg.update(d.get("judgments", {}))
    logger.info("Collected %d judgments across batches", len(all_judg))

    # Decode → group by true_baseline
    per_baseline: dict[str, list[dict]] = {}
    for anon_id, j in all_judg.items():
        meta = decode_map.get(anon_id)
        if not meta:
            logger.warning("Decoding miss: %s", anon_id)
            continue
        bl = meta["true_baseline"]
        scores = j.get("per_dim_scores", {})
        total = j.get("total_raw") or sum(scores.values()) if scores else None
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

    rows.sort(key=lambda r: r["mean"], reverse=True)

    md = ["# Phase 2F audit_v3 ISOLATED — anti-bias per-plan scoring\n\n"]
    md.append(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
    md.append("Methodology: 8 balanced batches (1 plan/baseline each) × 8 parallel\n")
    md.append("Opus subagents. Each subagent scores 7 anonymized plans with strict\n")
    md.append("per-plan independence instructions.\n\n")

    md.append("## Per-baseline aggregates\n\n")
    md.append("| Baseline | N | Mean /45 | Std | Min | Max | Distinct totals |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        n_distinct = len(set(r["totals"]))
        md.append(
            f"| **{r['baseline']}** | {r['n']} | {r['mean']:.2f} | {r['std']:.2f} |"
            f" {r['min']} | {r['max']} | {n_distinct}/{r['n']} |\n"
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

    md.append("\n## All totals per baseline (validate within-baseline variance)\n\n")
    for r in rows:
        md.append(f"- **{r['baseline']}**: {r['totals']}\n")

    out_path = out_dir / "audit_v3_isolated_summary.md"
    out_path.write_text("".join(md))
    logger.info("Wrote summary: %s", out_path)


def main(config: Config):
    if config.analyze:
        analyze_phase(config)
    else:
        submit_phase(config)


if __name__ == "__main__":
    main(chz.entrypoint(Config))
