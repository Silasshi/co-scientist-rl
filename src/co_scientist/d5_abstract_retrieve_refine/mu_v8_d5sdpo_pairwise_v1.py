"""D5 μ-v8-d5sdpo-G+ pairwise corroboration of audit_v3 +3.00 over σ_v8.

G+ (mask=True, α=0.1, PPO ε=0.2) iter 4 audit /45 = 22.38 vs σ_v8 = 19.38 (+3.00).
4 paper-grade matchups:
  1. G+ iter 4 vs σ_v8        (PRIMARY — locks v8 lift over length-matched anchor)
  2. G+ iter 4 vs v7-opd-full iter 4  (length-confounded; pairwise is fair)
  3. G+ iter 4 vs μ-v4 iter 4 (Phase 2 winner; methodology-confounded)
  4. G+ iter 4 vs G   iter 3  (within-v8 α=0.1 vs α=0.05 attribution)

Reuses fair_pairwise_v1.py infrastructure: OpusPairwiseClient + matchups_meta.json.

Usage:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.mu_v8_d5sdpo_pairwise_v1
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.mu_v8_d5sdpo_pairwise_v1 analyze=true
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
from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v2 import extract_solution

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    out_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v8_d5sdpo_Gplus_pairwise"
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"

    gplus_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v8_d5sdpo_Gplus"
    gplus_iter: int = 4

    sigma_v8_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v8_d5sdpo_sigma_v8"

    v7_opd_full_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v7_opd_full"
    v7_opd_full_iter: int = 4

    mu_v4_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_27_mu_v4"
    mu_v4_iter: int = 4

    g_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v8_d5sdpo_G"
    g_iter: int = 3

    n_pairs: int = 8
    seed: int = 60  # distinct from prior pairwise seeds 42-50
    analyze: bool = False


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _load_from_eval_jsonl(path: Path, iter_value: int, label: str, extract: bool = False) -> list[dict[str, str]]:
    plans = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("iter") == iter_value:
                text = extract_solution(d["text"]) if extract else d["text"]
                plans.append({
                    "plan_id": f"{label}::{d['plan_id']}",
                    "text": text,
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
                "text": extract_solution(d["plan_text"]),
            })
            if len(plans) >= max_n:
                break
    return plans


def load_all_plans(config: Config) -> dict[str, list[dict[str, str]]]:
    plans: dict[str, list[dict[str, str]]] = {}

    # G+ iter 4 (production checkpoint candidate)
    plans["gplus"] = _load_from_eval_jsonl(
        _resolve(config.gplus_run) / "eval_rollouts.jsonl",
        config.gplus_iter, "gplus", extract=False,
    )

    # σ_v8 baseline (buffer.jsonl, has <solution> wrapper)
    plans["sigma_v8"] = _load_from_buffer(
        _resolve(config.sigma_v8_run) / "buffer.jsonl",
        "sigma_v8", max_n=config.n_pairs,
    )

    # v7-opd-full peak iter 4
    plans["v7_opd_full"] = _load_from_eval_jsonl(
        _resolve(config.v7_opd_full_run) / "eval_rollouts.jsonl",
        config.v7_opd_full_iter, "v7_opd_full", extract=False,
    )

    # μ-v4 iter 4 (Phase 2 winner)
    plans["mu_v4"] = _load_from_eval_jsonl(
        _resolve(config.mu_v4_run) / "eval_rollouts.jsonl",
        config.mu_v4_iter, "mu_v4", extract=False,
    )

    # G iter 3 (peak before cliff at α=0.05)
    plans["g"] = _load_from_eval_jsonl(
        _resolve(config.g_run) / "eval_rollouts.jsonl",
        config.g_iter, "g", extract=False,
    )

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
        ("gplus", "sigma_v8",    "Gplus_vs_sigmaV8"),
        ("gplus", "v7_opd_full", "Gplus_vs_v7opdfull"),
        ("gplus", "mu_v4",       "Gplus_vs_muV4"),
        ("gplus", "g",           "Gplus_vs_G"),
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
    logger.info("Updated matchups_meta.json with 4 matchups → %s", meta_path)
    logger.info("Now main agent should dispatch 4 subagents (one per matchup file).")
    logger.info("After responses arrive, run with `analyze=true`.")


def analyze_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    meta_path = out_dir / "matchups_meta.json"
    meta = json.loads(meta_path.read_text())

    matchup_ids = ["Gplus_vs_sigmaV8", "Gplus_vs_v7opdfull", "Gplus_vs_muV4", "Gplus_vs_G"]

    rows = []
    for mid in matchup_ids:
        if mid not in meta:
            logger.warning("matchup %s missing from meta", mid)
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
        rationales = []
        for m in ms:
            v = verdicts.get(m["pair_id"])
            if v is None:
                continue
            w = (v.get("winner") or "").upper().strip()
            r = v.get("rationale", "")
            rationales.append({"pair_id": m["pair_id"], "winner_raw": w,
                               "rationale_chars": len(r), "rationale": r[:300]})
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

    md = ["# D5 μ-v8-d5sdpo G+ Pairwise Corroboration\n\n"]
    md.append(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*  \n")
    md.append(f"*Seed: {config.seed}, n_pairs/matchup: {config.n_pairs}*  \n")
    md.append(f"*Test-time-fair: NO reference plan in either prompt; both sides see goal+oracle.*\n\n")
    md.append("## Cross-check question\n\n")
    md.append("M8-strict 1-plan/Opus 9-dim audit_v3 reported G+ iter 4 = 22.38 / 45 vs σ_v8 = 19.38 (+3.00). "
              "Does Opus pairwise judge corroborate G+'s lead, and how does G+ compare to v7-opd-full / μ-v4 / G?\n\n")

    md.append("## 4 Pairwise Matchups\n\n")
    md.append("| Matchup | la wins | lb wins | ties | A-pos win rate | Winner |\n")
    md.append("|---|---:|---:|---:|---:|---|\n")
    for r in rows:
        if r.get("status") == "no_response":
            md.append(f"| {r['matchup']} | - | - | - | - | NO RESPONSE |\n")
            continue
        n = r["a_wins"] + r["b_wins"] + r["ties"]
        a_pct = f"{r['a_wins']}/{n}" if n else "-"
        md.append(f"| {r['matchup']} | {r['wins_la']} ({r['la']}) | {r['wins_lb']} ({r['lb']}) | "
                  f"{r['ties']} | {a_pct} | **{r['winner']}** |\n")

    md.append("\n## Decision Matrix\n\n")
    res_by_mid = {r["matchup"]: r for r in rows if r.get("status") != "no_response"}
    g_vs_sigma = res_by_mid.get("Gplus_vs_sigmaV8", {})
    g_vs_v7 = res_by_mid.get("Gplus_vs_v7opdfull", {})
    g_vs_muv4 = res_by_mid.get("Gplus_vs_muV4", {})
    g_vs_g = res_by_mid.get("Gplus_vs_G", {})

    md.append("**Decision rules:**\n")
    md.append("- G+ vs σ_v8: ≥6/8 win → audit lift corroborated, locks v8 production checkpoint\n")
    md.append("- G+ vs v7-opd-full: ≥6/8 win → v8 redesign beats v7-opd-full despite length difference; PAPER HEADLINE\n")
    md.append("- G+ vs μ-v4: ≥6/8 win → v8 surpasses Phase 2 winner; Tier-1 paper claim\n")
    md.append("- G+ vs G: ≥6/8 win → α=0.1 dominates α=0.05 within v8 family\n\n")

    out_md = out_dir / "PAIRWISE_SUMMARY.md"
    out_md.write_text("".join(md))
    logger.info("Wrote analysis → %s", out_md)
    print("".join(md))


def main(config: Config) -> None:
    if config.analyze:
        analyze_phase(config)
    else:
        submit_phase(config)


if __name__ == "__main__":
    config = chz.entrypoint(Config)
    main(config)
