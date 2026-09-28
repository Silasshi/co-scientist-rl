"""D5 Phase 5 quality audit remediation — pairwise corroboration on close-cluster verdicts.

Per DECISIONS 2026-04-26 (f): pairwise mandatory on close-cluster (new hard rule
`feedback_quality_first.md`). 3 matchups, all close-cluster:

  1. **muV5_v3_vs_muV4_tool_v** (Phase A.1, MANDATORY): Phase 5 NULL hinges on
     audit_v3_isolated_2way Δ +0.62 (within n=8 noise CI). Pairwise corroboration.
  2. **phase4a_meta_ttl** (Phase A.3): μ-v4 σ' (frozen + slim oracle) vs μ-v4 μ'
     (μ-v4 LoRA + slim oracle) on Meta-TTL goal. Phase 4a Δ +2.12 — borderline.
  3. **phase4a_tt_control** (Phase A.3): same on tt_control. Δ -0.88 — borderline.

Reuses `OpusPairwiseClient` + Phase 2 `mu_v4_pairwise_v1.py` template
(commit `0c61337`).

Usage:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.phase5_remediation_pairwise
    # main agent dispatches 3 subagents (one per matchup file)
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.phase5_remediation_pairwise \
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

from co_scientist.shared.opus_pairwise_subagent import OpusPairwiseClient
from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v2 import extract_solution

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    out_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_phase5_remediation_pairwise"

    # A.1: Tool-V matchup (μ-v4 vs μ-v5_v3 iter 0)
    tool_v_goal_path: str = "projects/d5_abstract_retrieve_refine/data/cross_goal/tool_verification_ttrl/research_goal.txt"
    tool_v_mu_v4_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_xgoal_tool_verification_ttrl_mu"
    tool_v_mu_v5_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_mu_v5_v3_iter0_inference"

    # A.3: Phase 4a borderline goals (σ' vs μ')
    meta_ttl_goal_path: str = "projects/d5_abstract_retrieve_refine/data/cross_goal/meta_ttl/research_goal.txt"
    meta_ttl_sigma_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_xgoal_meta_ttl_sigma"
    meta_ttl_mu_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_xgoal_meta_ttl_mu"

    tt_control_goal_path: str = "projects/d5_abstract_retrieve_refine/data/cross_goal/tt_control/research_goal.txt"
    tt_control_sigma_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_xgoal_tt_control_sigma"
    tt_control_mu_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_xgoal_tt_control_mu"

    n_pairs: int = 8
    seed: int = 60  # distinct from prior pairwise seeds 42/43/44/45/50
    analyze: bool = False


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _load_from_buffer(path: Path, label: str, max_n: int = 8) -> list[dict[str, str]]:
    """Load up to max_n plans from a baseline buffer.jsonl."""
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

    # ===== A.1: Tool-V — μ-v4 vs μ-v5_v3 =====
    tool_v_goal = _resolve(config.tool_v_goal_path).read_text().strip()
    mu_v4_plans = _load_from_buffer(_resolve(config.tool_v_mu_v4_run) / "buffer.jsonl",
                                      "mu_v4", max_n=config.n_pairs)
    mu_v5_plans = _load_from_buffer(_resolve(config.tool_v_mu_v5_run) / "buffer.jsonl",
                                      "mu_v5_v3", max_n=config.n_pairs)
    if len(mu_v4_plans) < config.n_pairs or len(mu_v5_plans) < config.n_pairs:
        raise RuntimeError(f"Tool-V plans insufficient: μ-v4 {len(mu_v4_plans)}, μ-v5_v3 {len(mu_v5_plans)}")
    ms_a1 = build_matchups(mu_v4_plans, mu_v5_plans, "mu_v4", "mu_v5_v3",
                            config.n_pairs, rng)
    meta["muV5_v3_vs_muV4_tool_v"] = ms_a1
    pwc.submit("muV5_v3_vs_muV4_tool_v", {
        "kind": "pairwise",
        "matchup_id": "muV5_v3_vs_muV4_tool_v",
        "goal": tool_v_goal,
        "pairs": [{k: v for k, v in m.items()
                   if k in ("pair_id", "plan_a_id", "plan_a_text",
                             "plan_b_id", "plan_b_text")} for m in ms_a1],
    })
    logger.info("A.1 Tool-V: %d pairs (μ-v4 vs μ-v5_v3)", len(ms_a1))

    # ===== A.3: Phase 4a meta_ttl — σ' vs μ' =====
    meta_ttl_goal = _resolve(config.meta_ttl_goal_path).read_text().strip()
    mt_sigma = _load_from_buffer(_resolve(config.meta_ttl_sigma_run) / "buffer.jsonl",
                                   "sigma_prime", max_n=config.n_pairs)
    mt_mu = _load_from_buffer(_resolve(config.meta_ttl_mu_run) / "buffer.jsonl",
                                "mu_prime", max_n=config.n_pairs)
    if len(mt_sigma) < config.n_pairs or len(mt_mu) < config.n_pairs:
        raise RuntimeError(f"meta_ttl plans insufficient: σ' {len(mt_sigma)}, μ' {len(mt_mu)}")
    ms_a3a = build_matchups(mt_sigma, mt_mu, "sigma_prime", "mu_prime",
                              config.n_pairs, rng)
    meta["phase4a_meta_ttl"] = ms_a3a
    pwc.submit("phase4a_meta_ttl", {
        "kind": "pairwise",
        "matchup_id": "phase4a_meta_ttl",
        "goal": meta_ttl_goal,
        "pairs": [{k: v for k, v in m.items()
                   if k in ("pair_id", "plan_a_id", "plan_a_text",
                             "plan_b_id", "plan_b_text")} for m in ms_a3a],
    })
    logger.info("A.3 meta_ttl: %d pairs (σ' vs μ')", len(ms_a3a))

    # ===== A.3: Phase 4a tt_control — σ' vs μ' =====
    tt_control_goal = _resolve(config.tt_control_goal_path).read_text().strip()
    tc_sigma = _load_from_buffer(_resolve(config.tt_control_sigma_run) / "buffer.jsonl",
                                   "sigma_prime", max_n=config.n_pairs)
    tc_mu = _load_from_buffer(_resolve(config.tt_control_mu_run) / "buffer.jsonl",
                                "mu_prime", max_n=config.n_pairs)
    if len(tc_sigma) < config.n_pairs or len(tc_mu) < config.n_pairs:
        raise RuntimeError(f"tt_control plans insufficient: σ' {len(tc_sigma)}, μ' {len(tc_mu)}")
    ms_a3b = build_matchups(tc_sigma, tc_mu, "sigma_prime", "mu_prime",
                              config.n_pairs, rng)
    meta["phase4a_tt_control"] = ms_a3b
    pwc.submit("phase4a_tt_control", {
        "kind": "pairwise",
        "matchup_id": "phase4a_tt_control",
        "goal": tt_control_goal,
        "pairs": [{k: v for k, v in m.items()
                   if k in ("pair_id", "plan_a_id", "plan_a_text",
                             "plan_b_id", "plan_b_text")} for m in ms_a3b],
    })
    logger.info("A.3 tt_control: %d pairs (σ' vs μ')", len(ms_a3b))

    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Updated matchups_meta.json with 3 matchups → %s", meta_path)
    logger.info("Now main agent should dispatch 24 subagents (8 per matchup, 1 per pair = strict task isolation).")
    logger.info("After responses arrive, run with `analyze=true`.")


def analyze_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    meta_path = out_dir / "matchups_meta.json"
    meta = json.loads(meta_path.read_text())

    matchup_ids = ["muV5_v3_vs_muV4_tool_v", "phase4a_meta_ttl", "phase4a_tt_control"]

    rows = []
    for mid in matchup_ids:
        if mid not in meta:
            logger.warning("matchup %s missing", mid)
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
                                "rationale": r[:300]})
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

    md = ["# D5 Phase 5 Remediation — Pairwise Corroboration\n\n"]
    md.append(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*  \n")
    md.append(f"*Seed: {config.seed}, n_pairs/matchup: {config.n_pairs}*  \n")
    md.append(f"*Per DECISIONS 2026-04-26 (f). Strict 1 subagent/pair task isolation.*\n\n")
    md.append("## 3 Matchups (all close-cluster verdicts requiring corroboration)\n\n")
    md.append("| Matchup | Goal | la wins | lb wins | ties | A-pos rate | Winner |\n")
    md.append("|---|---|---:|---:|---:|---:|---|\n")
    for r in rows:
        if "status" in r:
            md.append(f"| {r['matchup']} | (no response) | | | | | |\n")
        else:
            apos = r["a_wins"] / max(1, r["a_wins"] + r["b_wins"])
            md.append(
                f"| **{r['matchup']}** | (see below) | {r['wins_la']} ({r['la']}) | "
                f"{r['wins_lb']} ({r['lb']}) | {r['ties']} | "
                f"{r['a_wins']}/{r['a_wins'] + r['b_wins']} ({apos:.0%}) | **{r['winner']}** |\n"
            )

    md.append("\n## Verdicts vs original audit Δ\n\n")
    expectations = {
        "muV5_v3_vs_muV4_tool_v": ("Phase 5 v3 close-cluster", "+0.62 / 45 (within n=8 noise)"),
        "phase4a_meta_ttl": ("Phase 4a meta_ttl borderline", "+2.12 / 45 (just above +1.0 threshold)"),
        "phase4a_tt_control": ("Phase 4a tt_control borderline", "-0.88 / 45 (just below 0)"),
    }
    for r in rows:
        if "status" in r:
            continue
        mid = r["matchup"]
        title, audit_delta = expectations.get(mid, (mid, ""))
        wins_la, wins_lb = r["wins_la"], r["wins_lb"]
        if wins_la >= 6:
            verd = f"**PASS** — {r['la']} wins {wins_la}-{wins_lb} ≥ 6/8 threshold."
        elif wins_lb >= 6:
            verd = f"**PASS (REVERSE)** — {r['lb']} wins {wins_lb}-{wins_la} ≥ 6/8."
        else:
            verd = f"**INCONCLUSIVE** — {wins_la}-{wins_lb} (split)."
        md.append(f"### {title}\n")
        md.append(f"- Original audit Δ: {audit_delta}\n")
        md.append(f"- Pairwise: {verd}\n\n")

    md.append("## Position-bias sanity\n\n")
    md.append("A-pos win rate per matchup should be 25-75% (else position bias; rerun with re-seed).\n\n")
    for r in rows:
        if "status" in r:
            continue
        rate = r["a_wins"] / max(1, r["a_wins"] + r["b_wins"])
        flag = "🚨" if (rate > 0.75 or rate < 0.25) else "✓"
        md.append(f"- {r['matchup']}: {r['a_wins']}/{r['a_wins'] + r['b_wins']} ({rate:.0%}) {flag}\n")
    md.append("\n## Sample rationales (3/matchup, first 300 chars)\n\n")
    for r in rows:
        if "status" in r:
            continue
        md.append(f"### {r['matchup']}\n\n")
        for rat in r["rationales"][:3]:
            md.append(f"- **{rat['pair_id']}** (winner={rat['winner_raw']}): {rat['rationale']}\n")
        md.append("\n")

    out_md = out_dir / "phase5_remediation_pairwise_summary.md"
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
