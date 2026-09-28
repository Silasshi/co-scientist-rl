"""D5 fair pairwise tests — Phase 0.6 sanity take 2.

Original Phase 0.6 pairwise tournament (`pairwise_prefs_v1.py`) compared μ vs δ
and concluded "D5 dead" because δ won 8-0. But the comparison was UNFAIR: δ
sees the full reference plan in its prompt as a few-shot example, while μ at
eval time sees only `goal + oracle abstraction` (no reference). δ is a
semi-oracle condition, not a training-method baseline.

This script runs 5 FAIR matchups where both sides are evaluated under
test-time conditions WITHOUT reference plan in prompt:
  1. μ vs smoke_v3_A — training value at fixed oracle scaffolding
  2. μ vs smoke_v3_B — training+oracle vs frozen + goal only
  3. α vs smoke_v3_B — Opus distillation vs frozen
  4. β vs smoke_v3_B — direct ref SFT vs frozen
  5. smoke_v3_A vs smoke_v3_B — oracle inference value (untrained)

Reuses the existing pairwise_v1 run dir + opus_pairwise_subagent infrastructure.

Usage:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.fair_pairwise_v1 \
        out_path=projects/d5_abstract_retrieve_refine/runs/2026_04_25_pairwise_v1
    # main agent dispatches 5 subagents per matchup file
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.fair_pairwise_v1 \
        out_path=projects/d5_abstract_retrieve_refine/runs/2026_04_25_pairwise_v1 \
        analyze=true
"""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import chz

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.shared.opus_pairwise_subagent import OpusPairwiseClient
from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v1 import extract_solution

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    out_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_25_pairwise_v1"
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"

    mu_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_25_mu_baseline_v1"
    alpha_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_25_alpha_baseline_v1"
    beta_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_25_beta_baseline_v1"
    smoke_v3_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_smoke_pathway_v3_rerun_div"

    n_pairs: int = 8
    seed: int = 45  # different from main tournament (seed=42) and smokeA-vs-delta (seed=43) and delta-vs-eps (seed=44)
    analyze: bool = False


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _load_plans_from_eval_jsonl(path: Path, key_field: str, key_value: int, label: str) -> list[dict[str, str]]:
    """Load 8 plans from eval_rollouts.jsonl filtering on iter or epoch."""
    plans = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            if d.get(key_field) == key_value:
                plans.append({
                    "plan_id": f"{label}::{d['plan_id']}",
                    "text": extract_solution(d["text"]),
                })
    return plans


def _load_plans_from_smoke_buffer(path: Path, condition: str, label: str) -> list[dict[str, str]]:
    """Load plans from smoke runs' buffer.jsonl filtering on condition."""
    plans = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            if d.get("condition") == condition:
                plans.append({
                    "plan_id": f"{label}::{condition}_{d['sample_idx']}",
                    "text": extract_solution(d["plan_text"]),
                })
    return plans


def load_all_plans(config: Config) -> dict[str, list[dict[str, str]]]:
    plans: dict[str, list[dict[str, str]]] = {}

    # μ at iter=8 (last batch with audit data, audit total 9.25/20)
    mu_eval = _resolve(config.mu_run) / "eval_rollouts.jsonl"
    plans["mu"] = _load_plans_from_eval_jsonl(mu_eval, "iter", 8, "mu")

    # α at epoch=4 (last)
    alpha_eval = _resolve(config.alpha_run) / "eval_rollouts.jsonl"
    plans["alpha"] = _load_plans_from_eval_jsonl(alpha_eval, "epoch", 4, "alpha")

    # β at iter=8 (last with audit)
    beta_eval = _resolve(config.beta_run) / "eval_rollouts.jsonl"
    plans["beta"] = _load_plans_from_eval_jsonl(beta_eval, "iter", 8, "beta")

    # smoke_v3_A and B from same buffer
    smoke_buf = _resolve(config.smoke_v3_run) / "buffer.jsonl"
    plans["smoke_v3_A"] = _load_plans_from_smoke_buffer(smoke_buf, "A_with_abstraction", "smoke_v3_A")
    plans["smoke_v3_B"] = _load_plans_from_smoke_buffer(smoke_buf, "B_baseline", "smoke_v3_B")

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
        ("mu", "smoke_v3_A", "mu_vs_smokeA"),
        ("mu", "smoke_v3_B", "mu_vs_smokeB"),
        ("alpha", "smoke_v3_B", "alpha_vs_smokeB"),
        ("beta", "smoke_v3_B", "beta_vs_smokeB"),
        ("smoke_v3_A", "smoke_v3_B", "smokeA_vs_smokeB"),
    ]

    pwc = OpusPairwiseClient(log_path=out_dir, timeout_sec=1800.0)

    # Update matchups_meta.json (extend existing)
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
    logger.info("Updated matchups_meta.json with 5 new fair matchups → %s", meta_path)
    logger.info("Now main agent should dispatch 5 subagents.")
    logger.info("After responses, run with `analyze=true` for decision-matrix.")


def analyze_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    meta_path = out_dir / "matchups_meta.json"
    meta = json.loads(meta_path.read_text())

    fair_ids = ["mu_vs_smokeA", "mu_vs_smokeB", "alpha_vs_smokeB", "beta_vs_smokeB", "smokeA_vs_smokeB"]

    rows = []
    for mid in fair_ids:
        if mid not in meta:
            logger.warning("matchup %s missing from meta", mid)
            continue
        rp = out_dir / "pairwise_responses" / f"{mid}.json"
        if not rp.exists():
            rows.append({"matchup": mid, "status": "no_response"})
            continue
        verdicts = json.loads(rp.read_text()).get("verdicts", {})
        ms = meta[mid]
        # Resolve la, lb from first matchup
        first = ms[0]
        if first.get("position_swapped", False):
            la, lb = first["true_b_label"], first["true_a_label"]
        else:
            la, lb = first["true_a_label"], first["true_b_label"]

        wla = wlb = ties = 0
        a_wins = b_wins = 0  # raw position-A wins
        rationales = []
        for m in ms:
            v = verdicts.get(m["pair_id"])
            if v is None: continue
            w = (v.get("winner") or "").upper().strip()
            r = v.get("rationale", "")
            rationales.append({"pair_id": m["pair_id"], "winner_raw": w,
                                "rationale_chars": len(r), "rationale": r[:200]})
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
            "winner": winner, "rationales": rationales,
        })

    # ---- markdown summary ----
    md = ["# D5 Fair Pairwise Tests — Decision Matrix Report\n"]
    md.append(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
    md.append(f"*Seed: {config.seed}, n_pairs/matchup: {config.n_pairs}*\n")
    md.append(f"*All matchups under test-time-fair conditions: NO reference plan in prompt on either side.*\n\n")

    md.append("## 5 Fair Matchups\n\n")
    md.append("| Matchup | la wins | lb wins | ties | A-pos wins | Winner |\n|---|---:|---:|---:|---:|---|\n")
    for r in rows:
        if "status" in r:
            md.append(f"| {r['matchup']} | (no response) | | | | |\n")
        else:
            md.append(f"| **{r['la']}** vs **{r['lb']}** | {r['wins_la']} | {r['wins_lb']} | {r['ties']} | {r['a_wins']}/{r['a_wins'] + r['b_wins']} | **{r['winner']}** |\n")

    md.append("\n## Decision-Matrix Interpretation\n\n")
    # Extract specific outcomes for matrix
    def get_outcome(mid: str, la: str, lb: str):
        for r in rows:
            if r.get("matchup") == mid:
                wins_la = r.get("wins_la", 0)
                wins_lb = r.get("wins_lb", 0)
                if wins_la >= 6: return f"{la} wins ({wins_la}-{wins_lb})", "la_wins"
                if wins_lb >= 6: return f"{lb} wins ({wins_la}-{wins_lb})", "lb_wins"
                return f"close ({wins_la}-{wins_lb})", "close"
        return "missing", "missing"

    m1, m1_kind = get_outcome("mu_vs_smokeA", "mu", "smoke_v3_A")
    m2, m2_kind = get_outcome("mu_vs_smokeB", "mu", "smoke_v3_B")
    m3, m3_kind = get_outcome("alpha_vs_smokeB", "alpha", "smoke_v3_B")
    m4, m4_kind = get_outcome("beta_vs_smokeB", "beta", "smoke_v3_B")
    m5, m5_kind = get_outcome("smokeA_vs_smokeB", "smoke_v3_A", "smoke_v3_B")

    md.append(f"- **μ vs smoke_v3_A** (training value at fixed oracle): {m1}\n")
    md.append(f"- **μ vs smoke_v3_B** (training+oracle vs goal-only frozen): {m2}\n")
    md.append(f"- **α vs smoke_v3_B** (Opus distillation vs frozen): {m3}\n")
    md.append(f"- **β vs smoke_v3_B** (ref SFT vs frozen): {m4}\n")
    md.append(f"- **smoke_v3_A vs smoke_v3_B** (oracle inference value, no training): {m5}\n\n")

    # Verdict logic
    if m1_kind == "la_wins" and m2_kind == "la_wins":
        verdict = ("**D5 TRAINING WORKS** — SDPO+critic+oracle adds value over both fixed oracle "
                   "scaffolding (μ > smoke_A) AND plain frozen 30B (μ > smoke_B). Continue Phase 0b/1.")
    elif m1_kind == "la_wins" and m2_kind == "close":
        verdict = ("D5 training adds incremental value over oracle scaffolding, but training+oracle "
                   "doesn't decisively beat plain frozen. Investigate further with n=16 eval.")
    elif m1_kind == "close" and m2_kind == "la_wins":
        verdict = ("Oracle abstraction does the lifting; training adds nothing. **PIVOT TO INFERENCE-TIME "
                   "SCAFFOLDING PAPER** (D2 alive again).")
    elif m1_kind == "close" and m2_kind == "close":
        verdict = ("Training neutral, oracle marginal. **D5 design needs rework**. Consider "
                   "redesigning the training mechanism (A1 span-level SDPO, C2 differential critique).")
    elif m1_kind == "lb_wins" or m2_kind == "lb_wins":
        verdict = ("Training actively hurts. **D5 dead in this regime**. Major redesign required.")
    else:
        verdict = "Mixed results; manual interpretation needed."

    md.append(f"### Verdict\n\n{verdict}\n\n")

    md.append("## Auxiliary signals\n\n")
    md.append(f"- α and β vs frozen baseline: useful for paper narrative on training-method comparison\n")
    md.append(f"- smoke_v3_A vs smoke_v3_B: validates pathway claim under pairwise (was withdrawn under v1+pairwise vs δ)\n\n")

    md.append("## Position-bias check (sanity)\n\n")
    md.append("| Matchup | A-pos win rate |\n|---|---:|\n")
    for r in rows:
        if "status" in r: continue
        rate = r["a_wins"] / max(1, r["a_wins"] + r["b_wins"])
        flag = "🚨" if rate > 0.75 or rate < 0.25 else "✓"
        md.append(f"| {r['matchup']} | {r['a_wins']}/{r['a_wins'] + r['b_wins']} ({rate:.0%}) {flag} |\n")

    md.append("\n## Sample rationales (3 from each matchup, first 200 chars)\n\n")
    for r in rows:
        if "status" in r: continue
        md.append(f"### {r['la']} vs {r['lb']}\n\n")
        for rat in r["rationales"][:3]:
            md.append(f"- **{rat['pair_id']}** ({rat['winner_raw']}, {rat['rationale_chars']} chars): {rat['rationale']}\n")
        md.append("\n")

    out_md = out_dir / "fair_pairwise_summary.md"
    out_md.write_text("".join(md))
    logger.info("Wrote summary: %s", out_md)
    print(out_md.read_text())


def main(config: Config) -> None:
    if config.analyze:
        analyze_phase(config)
    else:
        submit_phase(config)


if __name__ == "__main__":
    cfg = chz.entrypoint(Config)
    main(cfg)
