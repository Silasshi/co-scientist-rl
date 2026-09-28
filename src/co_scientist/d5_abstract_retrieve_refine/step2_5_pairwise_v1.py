"""D5 Phase 3 Step 2.5 — pairwise corroboration of σ_v4 / τ_v4 / μ-v4-replan controls.

Per cross-validation discipline (M4_audit_isolated_methodology.md): close-cluster
audit comparisons (gap < 3 / 45) MUST be cross-validated by Opus pairwise tournament
before any paper claim. Step 2.5 controls were not pairwise-tested initially (cost-
saving compromise). This script fixes that.

3 matchups × 8 position-randomized pairs each = 24 Opus pairwise calls.

Plans loaded from existing run dirs:
- σ_v4: runs/2026_04_28_sigma_v4/buffer.jsonl
- τ_v4: runs/2026_04_28_tau_v4/buffer.jsonl
- μ-v4-replan: runs/2026_04_28_mu_v4_replan/buffer.jsonl
- μ-v4 (Phase 2): runs/2026_04_27_mu_v4/eval_rollouts.jsonl iter=4

Forks `mu_v4_pairwise_v1.py`. Output dir: runs/2026_04_28_step2_5_pairwise/

Usage:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.step2_5_pairwise_v1
    # main agent dispatches 3 subagents, each fans out 8 pairs
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.step2_5_pairwise_v1 analyze=true
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

from co_scientist.shared.opus_pairwise_subagent import OpusPairwiseClient
from co_scientist.d5_abstract_retrieve_refine.kappa_prompts_v1 import extract_distillation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    out_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_step2_5_pairwise"
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"

    sigma_v4_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_sigma_v4"
    tau_v4_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_tau_v4"
    mu_replan_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_mu_v4_replan"
    mu_v4_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_27_mu_v4"
    mu_v4_iter: int = 4

    n_pairs: int = 8
    seed: int = 51  # distinct from prior pairwise seeds 42-50
    analyze: bool = False


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _load_from_eval_jsonl(path: Path, key_field: str, key_value: int, label: str) -> list[dict[str, str]]:
    plans = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get(key_field) == key_value:
                plans.append({
                    "plan_id": f"{label}::{d['plan_id']}",
                    "text": extract_distillation(d["text"]),
                })
    return plans


def _load_from_buffer(path: Path, label: str, max_n: int = 8) -> list[dict[str, str]]:
    plans = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            plans.append({
                "plan_id": f"{label}::sample_{d.get('sample_idx', len(plans))}",
                "text": extract_distillation(d["plan_text"]),
            })
            if len(plans) >= max_n:
                break
    return plans


def load_all_plans(config: Config) -> dict[str, list[dict[str, str]]]:
    plans: dict[str, list[dict[str, str]]] = {}
    plans["sigma_v4"] = _load_from_buffer(_resolve(config.sigma_v4_run) / "buffer.jsonl",
                                          "sigma_v4", max_n=config.n_pairs)
    plans["tau_v4"] = _load_from_buffer(_resolve(config.tau_v4_run) / "buffer.jsonl",
                                        "tau_v4", max_n=config.n_pairs)
    plans["mu_replan"] = _load_from_buffer(_resolve(config.mu_replan_run) / "buffer.jsonl",
                                           "mu_replan", max_n=config.n_pairs)
    plans["mu_v4"] = _load_from_eval_jsonl(_resolve(config.mu_v4_run) / "eval_rollouts.jsonl",
                                           "iter", config.mu_v4_iter, "mu_v4")

    for label, ps in plans.items():
        logger.info("  %s: %d plans loaded", label, len(ps))
        if len(ps) < config.n_pairs:
            raise RuntimeError(f"baseline {label} has only {len(ps)} plans, need >= {config.n_pairs}")
    return plans


def build_matchups(plans_a: list[dict], plans_b: list[dict],
                   label_a: str, label_b: str, n_pairs: int, rng: random.Random) -> list[dict]:
    matchups = []
    for i in range(n_pairs):
        pa = plans_a[i % len(plans_a)]
        pb = plans_b[i % len(plans_b)]
        swap = rng.random() < 0.5
        if swap:
            pa, pb = pb, pa
            la, lb = label_b, label_a
        else:
            la, lb = label_a, label_b
        matchups.append({
            "pair_id": f"{label_a}_vs_{label_b}_pair_{i:02d}",
            "plan_a_id": pa["plan_id"], "plan_a_text": pa["text"],
            "plan_b_id": pb["plan_id"], "plan_b_text": pb["text"],
            "true_a_label": la, "true_b_label": lb,
            "position_swapped": swap,
        })
    return matchups


def submit_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    goal = _resolve(config.goal_path).read_text().strip()
    plans = load_all_plans(config)

    matchup_specs = [
        ("tau_v4", "sigma_v4", "tauV4_vs_sigmaV4"),     # distillation pipeline net contribution
        ("tau_v4", "mu_replan", "tauV4_vs_muReplan"),    # distillation pipeline vs SDPO weights with same plan_v4
        ("sigma_v4", "mu_v4", "sigmaV4_vs_muV4"),         # does plan_v4 add to baseline?
    ]

    pwc = OpusPairwiseClient(log_path=out_dir, timeout_sec=1800.0)
    meta_path = out_dir / "matchups_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    for la, lb, mid in matchup_specs:
        ms = build_matchups(plans[la], plans[lb], la, lb, config.n_pairs, rng)
        meta[mid] = ms
        payload = {
            "kind": "pairwise",
            "matchup_id": mid,
            "goal": goal,
            "pairs": [
                {k: v for k, v in m.items()
                 if k in ("pair_id", "plan_a_id", "plan_a_text", "plan_b_id", "plan_b_text")}
                for m in ms
            ],
        }
        pwc.submit(mid, payload)
        logger.info("Wrote pairwise request %s (%d pairs)", mid, len(ms))

    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Updated matchups_meta.json with 3 matchups")


def analyze_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    meta = json.loads((out_dir / "matchups_meta.json").read_text())
    matchup_ids = ["tauV4_vs_sigmaV4", "tauV4_vs_muReplan", "sigmaV4_vs_muV4"]

    rows = []
    for mid in matchup_ids:
        if mid not in meta:
            continue
        rp = out_dir / "pairwise_responses" / f"{mid}.json"
        if not rp.exists():
            rows.append({"matchup": mid, "status": "no_response"})
            continue
        verdicts = json.loads(rp.read_text()).get("verdicts", {})
        ms = meta[mid]
        first = ms[0]
        if first.get("position_swapped", False):
            la, lb = first["true_b_label"], first["true_a_label"]
        else:
            la, lb = first["true_a_label"], first["true_b_label"]

        wla = wlb = ties = 0
        a_wins = b_wins = 0
        for m in ms:
            v = verdicts.get(m["pair_id"])
            if v is None:
                continue
            w = (v.get("winner") or "").upper().strip()
            if w == "TIE":
                ties += 1
            elif w == "A":
                a_wins += 1
                if m["true_a_label"] == la: wla += 1
                elif m["true_a_label"] == lb: wlb += 1
            elif w == "B":
                b_wins += 1
                if m["true_b_label"] == la: wla += 1
                elif m["true_b_label"] == lb: wlb += 1
        winner = "tie" if wla == wlb else (la if wla > wlb else lb)
        rows.append({
            "matchup": mid, "la": la, "lb": lb,
            "wins_la": wla, "wins_lb": wlb, "ties": ties,
            "a_wins": a_wins, "b_wins": b_wins,
            "winner": winner,
        })

    md = ["# D5 Step 2.5 pairwise corroboration — close-cluster control verification\n\n"]
    md.append(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*  \n")
    md.append(f"*Seed: {config.seed}, n_pairs/matchup: {config.n_pairs}*  \n")
    md.append(f"*Cross-validation per M4 discipline; close-cluster gap < 3 / 45 requires pairwise.*\n\n")

    md.append("## 3 matchups (close-cluster gap < 3 audit points)\n\n")
    md.append("| Matchup | la wins | lb wins | ties | A-pos win rate | Winner |\n")
    md.append("|---|---:|---:|---:|---:|---|\n")
    for r in rows:
        if "status" in r:
            md.append(f"| {r['matchup']} | (no response) | | | | |\n")
        else:
            apos = r["a_wins"] / max(1, r["a_wins"] + r["b_wins"])
            md.append(f"| **{r['la']}** vs **{r['lb']}** | {r['wins_la']} | {r['wins_lb']} | {r['ties']} | {r['a_wins']}/{r['a_wins'] + r['b_wins']} ({apos:.0%}) | **{r['winner']}** |\n")

    md.append("\n## Audit context (absolute)\n\n")
    md.append("| Baseline | Audit /45 |\n|---|---:|\n")
    md.append("| τ_v4 (medium 3-batch + plan_v4) | 26.50 |\n")
    md.append("| σ_v4 (slim + plan_v4) | 25.75 |\n")
    md.append("| σ Phase 2 (slim + plan_v3) | 25.25 |\n")
    md.append("| μ-v4 Phase 2 prod (μ-v4 LoRA + slim + plan_v3) | 28.00 |\n")
    md.append("| μ-v4-replan (μ-v4 LoRA + slim + plan_v4) | 24.75 |\n\n")

    md.append("## Cross-validation verdict\n\n")
    for r in rows:
        if "status" in r:
            continue
        ratio = max(r["wins_la"], r["wins_lb"])
        if ratio >= 6:
            sig = "STRONG"
        elif ratio >= 5:
            sig = "WEAK"
        else:
            sig = "TIE"
        md.append(f"- **{r['la']} vs {r['lb']}**: {r['winner']} wins {max(r['wins_la'], r['wins_lb'])}-{min(r['wins_la'], r['wins_lb'])} → {sig}\n")

    out_md = out_dir / "step2_5_pairwise_summary.md"
    out_md.write_text("".join(md))
    logger.info("Wrote summary: %s", out_md)
    print(out_md.read_text())


def main(config: Config) -> None:
    if config.analyze:
        analyze_phase(config)
    else:
        submit_phase(config)


if __name__ == "__main__":
    main(chz.entrypoint(Config))
