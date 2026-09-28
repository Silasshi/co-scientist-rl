"""D5 D1 — n=24 pairwise tournament on meta_ttl + tool_v_ttrl.

Stresses pairwise tournament noise floor by going from n=8 (12 pairs/goal max with rotation)
to n=24 (24 pairs/goal). Uses the n=24 combined buffers built from existing n=8 + new n=16
inference runs.

Pre-registered binding: 95% CI on Δ /45 (strict) excludes 0 AND pairwise ≥ 14/24 in same
direction → CONFIRMED at n=24. Otherwise → TRUE NULL or noise-limited documented.

Usage:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.d1_n24_pairwise

    # External: dispatch 48 Opus subagents (2 matchups × 24 pairs)

    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.d1_n24_pairwise analyze=true
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
    out_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_d1_n24_pairwise"

    meta_ttl_goal_path: str = "projects/d5_abstract_retrieve_refine/data/cross_goal/meta_ttl/research_goal.txt"
    meta_ttl_sigma_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_xgoal_meta_ttl_sigma_n24_combined"
    meta_ttl_mu_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_xgoal_meta_ttl_mu_n24_combined"

    tool_v_goal_path: str = "projects/d5_abstract_retrieve_refine/data/cross_goal/tool_verification_ttrl/research_goal.txt"
    tool_v_sigma_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_xgoal_tool_verification_ttrl_sigma_n24_combined"
    tool_v_mu_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_xgoal_tool_verification_ttrl_mu_n24_combined"

    n_pairs: int = 24
    seed: int = 80  # distinct from prior pairwise seeds 42-50, 60, 70
    analyze: bool = False


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _load_from_buffer(path: Path, label: str, max_n: int = 24) -> list[dict[str, str]]:
    plans = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            plans.append({
                "plan_id": f"{label}::sample_{d.get('sample_idx', len(plans))}",
                "text": extract_solution(d.get("plan_text", "")),
            })
            if len(plans) >= max_n:
                break
    return plans


def build_matchups(plans_a: list[dict], plans_b: list[dict],
                    label_a: str, label_b: str, n_pairs: int,
                    rng: random.Random) -> list[dict]:
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

    pwc = OpusPairwiseClient(log_path=out_dir, timeout_sec=1800.0)
    meta_path = out_dir / "matchups_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    matchup_specs = [
        ("meta_ttl_n24", config.meta_ttl_goal_path, config.meta_ttl_sigma_run, config.meta_ttl_mu_run),
        ("tool_v_ttrl_n24", config.tool_v_goal_path, config.tool_v_sigma_run, config.tool_v_mu_run),
    ]

    for matchup_id, goal_path, sigma_run, mu_run in matchup_specs:
        goal_text = _resolve(goal_path).read_text().strip()
        sigma_plans = _load_from_buffer(_resolve(sigma_run) / "buffer.jsonl", "sigma_prime", max_n=config.n_pairs)
        mu_plans = _load_from_buffer(_resolve(mu_run) / "buffer.jsonl", "mu_prime", max_n=config.n_pairs)
        if len(sigma_plans) < config.n_pairs or len(mu_plans) < config.n_pairs:
            raise RuntimeError(f"{matchup_id} insufficient: σ' {len(sigma_plans)}, μ' {len(mu_plans)}")
        ms = build_matchups(sigma_plans, mu_plans, "sigma_prime", "mu_prime", config.n_pairs, rng)
        meta[matchup_id] = ms
        pwc.submit(matchup_id, {
            "kind": "pairwise",
            "matchup_id": matchup_id,
            "goal": goal_text,
            "pairs": [{k: v for k, v in m.items()
                       if k in ("pair_id", "plan_a_id", "plan_a_text",
                                 "plan_b_id", "plan_b_text")} for m in ms],
        })
        logger.info("matchup %s: %d pairs", matchup_id, len(ms))

    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Wrote 2 matchups (n=24 each = 48 total subagents)")


def analyze_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    meta_path = out_dir / "matchups_meta.json"
    meta = json.loads(meta_path.read_text())

    matchup_ids = ["meta_ttl_n24", "tool_v_ttrl_n24"]

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
        rationales = []
        for m in ms:
            v = verdicts.get(m["pair_id"])
            if v is None:
                continue
            w = (v.get("winner") or "").upper().strip()
            r = v.get("rationale", "")
            rationales.append({"pair_id": m["pair_id"], "winner_raw": w, "rationale": r[:300]})
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

    md = ["# D5 D1 — n=24 Pairwise Tournament (meta_ttl + tool_v_ttrl)\n\n"]
    md.append(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*  \n")
    md.append(f"*Seed: {config.seed}, n_pairs/matchup: {config.n_pairs}*  \n")
    md.append(f"*Per DECISIONS 2026-04-28 (g). 95% CI on Δ /45 should be ±2.0 at n=24 vs ±3.5 at n=8.*\n\n")

    md.append("| Matchup | σ' wins | μ' wins | ties | A-pos rate | Winner |\n")
    md.append("|---|---:|---:|---:|---:|---|\n")
    for r in rows:
        if "status" in r:
            md.append(f"| {r['matchup']} | (no response) | | | | |\n")
        else:
            apos = r["a_wins"] / max(1, r["a_wins"] + r["b_wins"])
            md.append(
                f"| **{r['matchup']}** | {r['wins_la']} ({r['la']}) | "
                f"{r['wins_lb']} ({r['lb']}) | {r['ties']} | "
                f"{r['a_wins']}/{r['a_wins'] + r['b_wins']} ({apos:.0%}) | **{r['winner']}** |\n"
            )

    md.append("\n## Pre-reg verdict (binding)\n\n")
    md.append("- σ' wins ≥14/24 (>58%) → σ' DIRECTIONAL CONFIRMED at n=24\n")
    md.append("- μ' wins ≥14/24 → μ' DIRECTIONAL CONFIRMED at n=24\n")
    md.append("- 12-12 ± 2 → TRUE NULL at n=24 (F7 verdict: definitive null at this gap range)\n")

    out_path = out_dir / "d1_n24_pairwise_summary.md"
    out_path.write_text("".join(md))
    logger.info("Wrote summary: %s", out_path)


def main(config: Config):
    if config.analyze:
        analyze_phase(config)
    else:
        submit_phase(config)


if __name__ == "__main__":
    main(chz.entrypoint(Config))
