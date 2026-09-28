"""D5 μ-v4 pairwise corroboration of audit_v3 ISOLATED +2.75.

Phase 2F audit_v3 ISOLATED reported μ-v4 (iter 4) = 28.00 / 45 vs σ = 25.25 / 45
(+2.75) — the headline positive result for the D5 paper. Phase 0.6 saw absolute
audit overstate by 8-0 once (μ-v2 vs δ inversion), so this script runs a pairwise
cross-check before the number anchors the paper claim.

Three matchups, each 8 position-randomized pairs, Opus pairwise judge:
  1. μ-v4 (iter 4)  vs σ  (PRIMARY — paper-claim cross-check)
  2. μ-v4 (iter 4)  vs δ  (close cluster ranking neighbor)
  3. μ-v4 (iter 4)  vs α  (close cluster ranking neighbor)

Reuses fair_pairwise_v1.py infrastructure: OpusPairwiseClient + matchups_meta.json
+ analyze_phase decision-matrix logic (verdict text replaced with cross-check).

Usage:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.mu_v4_pairwise_v1
    # main agent dispatches 3 subagents (one per matchup file)
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.mu_v4_pairwise_v1 analyze=true
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
    out_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_27_mu_v4_pairwise"
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"

    mu_v4_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_27_mu_v4"
    mu_v4_iter: int = 4  # production checkpoint

    sigma_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_26_sigma_v2"
    delta_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_26_delta_v2"
    alpha_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_26_alpha_v2b"
    alpha_epoch: int = 4  # last epoch

    n_pairs: int = 8
    seed: int = 50  # distinct from prior pairwise seeds 42/43/44/45
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
                    "text": extract_solution(d["text"]),
                })
    return plans


def _load_from_buffer(path: Path, label: str, max_n: int = 8) -> list[dict[str, str]]:
    """Load up to max_n plans from baseline buffer.jsonl (key plan_text)."""
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

    mu_eval = _resolve(config.mu_v4_run) / "eval_rollouts.jsonl"
    plans["mu_v4"] = _load_from_eval_jsonl(mu_eval, "iter", config.mu_v4_iter, "mu_v4")

    sigma_buf = _resolve(config.sigma_run) / "buffer.jsonl"
    plans["sigma"] = _load_from_buffer(sigma_buf, "sigma", max_n=config.n_pairs)

    delta_buf = _resolve(config.delta_run) / "buffer.jsonl"
    plans["delta"] = _load_from_buffer(delta_buf, "delta", max_n=config.n_pairs)

    alpha_eval = _resolve(config.alpha_run) / "eval_rollouts.jsonl"
    plans["alpha"] = _load_from_eval_jsonl(alpha_eval, "epoch", config.alpha_epoch, "alpha")

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
        ("mu_v4", "sigma", "muV4_vs_sigma"),
        ("mu_v4", "delta", "muV4_vs_delta"),
        ("mu_v4", "alpha", "muV4_vs_alpha"),
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
    logger.info("Updated matchups_meta.json with 3 matchups → %s", meta_path)
    logger.info("Now main agent should dispatch 3 subagents (one per matchup file).")
    logger.info("After responses arrive, run with `analyze=true`.")


def analyze_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    meta_path = out_dir / "matchups_meta.json"
    meta = json.loads(meta_path.read_text())

    matchup_ids = ["muV4_vs_sigma", "muV4_vs_delta", "muV4_vs_alpha"]

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

    md = ["# D5 μ-v4 Pairwise Corroboration of audit_v3 ISOLATED +2.75\n\n"]
    md.append(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*  \n")
    md.append(f"*Seed: {config.seed}, n_pairs/matchup: {config.n_pairs}*  \n")
    md.append(f"*All matchups under test-time-fair conditions: NO reference plan in either prompt; both sides see goal+oracle.*\n\n")
    md.append("## Cross-check question\n\n")
    md.append("Phase 2F audit_v3 ISOLATED ranked μ-v4 iter 4 = 28.00 ≫ σ = 25.25 ≈ δ = 25.12 ≈ α = 25.00. "
              "Does Opus pairwise judge corroborate this top-of-cluster ranking, or does the absolute audit have surface-form bias?\n\n")

    md.append("## 3 Pairwise Matchups\n\n")
    md.append("| Matchup | la wins | lb wins | ties | A-pos win rate | Winner |\n")
    md.append("|---|---:|---:|---:|---:|---|\n")
    for r in rows:
        if "status" in r:
            md.append(f"| {r['matchup']} | (no response) | | | | |\n")
        else:
            apos = r["a_wins"] / max(1, r["a_wins"] + r["b_wins"])
            md.append(
                f"| **{r['la']}** vs **{r['lb']}** | {r['wins_la']} | {r['wins_lb']} | {r['ties']} | "
                f"{r['a_wins']}/{r['a_wins'] + r['b_wins']} ({apos:.0%}) | **{r['winner']}** |\n"
            )

    md.append("\n## Cross-check verdict\n\n")
    primary = next((r for r in rows if r.get("matchup") == "muV4_vs_sigma"), None)
    if primary and "wins_la" in primary:
        wins_mu = primary["wins_la"] if primary["la"] == "mu_v4" else primary["wins_lb"]
        wins_sig = primary["wins_lb"] if primary["la"] == "mu_v4" else primary["wins_la"]
        if wins_mu >= 6:
            verdict = (f"**PASS** — μ-v4 wins {wins_mu}-{wins_sig} vs σ. The audit_v3 ISOLATED "
                       f"+2.75 absolute lift is corroborated by independent pairwise judgment. "
                       f"Paper-claim headline ('first 30B-trained variant to beat σ') is safe.")
        elif wins_mu <= 2:
            verdict = (f"**INVERSION** — σ wins {wins_sig}-{wins_mu}. Same failure mode as Phase 0.6 "
                       f"μ-v2 vs δ. Absolute audit unreliable for this comparison; paper-claim headline "
                       f"must be retracted to 'no statistically clear winner over σ'. Audit redesign needed.")
        else:
            verdict = (f"**FLAG** — μ-v4 wins {wins_mu}-{wins_sig} (close). Absolute may overstate; "
                       f"n=8 too noisy. Run n=16 follow-up before paper claim.")
    else:
        verdict = "Primary matchup missing — re-run submit phase or check subagent dispatch."
    md.append(f"{verdict}\n\n")

    md.append("## Secondary matchups (close-cluster ranking)\n\n")
    for mid in ["muV4_vs_delta", "muV4_vs_alpha"]:
        r = next((x for x in rows if x.get("matchup") == mid), None)
        if r is None or "wins_la" not in r:
            md.append(f"- {mid}: missing\n")
            continue
        wins_mu = r["wins_la"] if r["la"] == "mu_v4" else r["wins_lb"]
        opp_label = r["lb"] if r["la"] == "mu_v4" else r["la"]
        wins_opp = r["wins_lb"] if r["la"] == "mu_v4" else r["wins_la"]
        if wins_mu >= 6:
            sub = f"μ-v4 cleanly above {opp_label} ({wins_mu}-{wins_opp})"
        elif wins_mu <= 2:
            sub = f"⚠️ μ-v4 BELOW {opp_label} ({wins_mu}-{wins_opp}) — inverts close-cluster ranking"
        else:
            sub = f"μ-v4 ≈ {opp_label} ({wins_mu}-{wins_opp}) — close-cluster ordering not separable at n=8"
        md.append(f"- **{mid}**: {sub}\n")
    md.append("\n")

    md.append("## Position-bias sanity\n\n")
    md.append("A-pos win rate per matchup should be in 25-75% (else position bias; rerun with re-seed).\n\n")
    for r in rows:
        if "status" in r:
            continue
        rate = r["a_wins"] / max(1, r["a_wins"] + r["b_wins"])
        flag = "🚨" if (rate > 0.75 or rate < 0.25) else "✓"
        md.append(f"- {r['matchup']}: {r['a_wins']}/{r['a_wins'] + r['b_wins']} ({rate:.0%}) {flag}\n")
    md.append("\n")

    md.append("## Sample rationales (3 per matchup, first 300 chars)\n\n")
    for r in rows:
        if "status" in r:
            continue
        md.append(f"### {r['la']} vs {r['lb']}\n\n")
        for rat in r["rationales"][:3]:
            md.append(f"- **{rat['pair_id']}** (winner={rat['winner_raw']}, {rat['rationale_chars']} chars): "
                      f"{rat['rationale']}\n")
        md.append("\n")

    out_md = out_dir / "mu_v4_pairwise_summary.md"
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
