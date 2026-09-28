"""D5 D3 — 7-baseline close-cluster pairwise corroboration (M8 broader validation).

Tests whether the M8 audit-pairwise divergence (2-plan/subagent batching inflates
close-cluster Δ) generalizes from cross-goal scenarios to the original 7-baseline
ranking. The Phase 2F audit_v3 ISOLATED reported close cluster σ 25.25 / δ 25.12 /
α 25.00 (effectively zero gaps under 2-plan/subagent). If pairwise tournament gives
the same result (all matchups within ±5/8 ties), then the 7-baseline ranking IS
robust. If pairwise breaks ≥6/8 in either direction, then 2-plan audit DID rank
correctly and M8 effect is dataset-specific.

Three matchups, all on TTT-Discover bibliography goal:
  - σ vs δ (frozen + slim oracle vs. frozen + concept-vec oracle)
  - σ vs α (frozen + slim oracle vs. trained-LoRA-iter-0 + slim oracle)
  - δ vs α

Uses identical pairwise prompt as `phase5_remediation_pairwise.py`. Subagent prompt
file: `runs/2026_04_28_phase5_remediation_pairwise/pairwise_subagent_prompt.md`.

Usage (per matchup):
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.phase6_seven_baseline_pairwise

    # External: dispatch 24 Opus subagents (3 matchups × 8 pairs)

    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.phase6_seven_baseline_pairwise analyze=true
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
    out_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_phase6_seven_baseline_pairwise"

    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"

    # σ: frozen + slim oracle (8 plans in buffer.jsonl)
    sigma_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_26_sigma_v2"
    # δ: frozen + concept-vec oracle (8 plans in buffer.jsonl)
    delta_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_26_delta_v2"
    # α: trained LoRA iter-0 — plans live in eval_rollouts.jsonl filtered by epoch=0
    alpha_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_26_alpha_v2"

    n_pairs: int = 8
    seed: int = 70  # distinct from prior pairwise seeds 42-50, 60
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


def _load_from_eval_rollouts(path: Path, label: str, epoch: int = 0,
                              max_n: int = 8) -> list[dict[str, str]]:
    """Load up to max_n plans from a trainer eval_rollouts.jsonl filtered by epoch.

    α plans are stored in trainer outputs (eval_rollouts.jsonl) rather than
    a frozen buffer.jsonl. Filter by epoch=0 to extract the σ-equivalent
    state (initial weights, before any SDPO updates).
    """
    plans = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("epoch") != epoch:
                continue
            text = d.get("text", "")
            if not text:
                continue
            plans.append({
                "plan_id": f"{label}::epoch{epoch}_{d.get('plan_id', len(plans))}",
                "text": extract_solution(text),
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

    goal_text = _resolve(config.goal_path).read_text().strip()

    sigma_plans = _load_from_buffer(_resolve(config.sigma_run) / "buffer.jsonl",
                                      "sigma", max_n=config.n_pairs)
    delta_plans = _load_from_buffer(_resolve(config.delta_run) / "buffer.jsonl",
                                      "delta", max_n=config.n_pairs)
    alpha_plans = _load_from_eval_rollouts(_resolve(config.alpha_run) / "eval_rollouts.jsonl",
                                             "alpha", epoch=0, max_n=config.n_pairs)

    for label, plans in [("sigma", sigma_plans), ("delta", delta_plans), ("alpha", alpha_plans)]:
        if len(plans) < config.n_pairs:
            raise RuntimeError(f"{label} plans insufficient: {len(plans)} < {config.n_pairs}")

    matchup_specs = [
        ("sigma_vs_delta", sigma_plans, delta_plans, "sigma", "delta"),
        ("sigma_vs_alpha", sigma_plans, alpha_plans, "sigma", "alpha"),
        ("delta_vs_alpha", delta_plans, alpha_plans, "delta", "alpha"),
    ]

    for matchup_id, plans_a, plans_b, label_a, label_b in matchup_specs:
        ms = build_matchups(plans_a, plans_b, label_a, label_b, config.n_pairs, rng)
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
    logger.info("Wrote 3 matchups (sigma/delta/alpha pairwise) to %s", out_dir)
    logger.info("Now main agent should dispatch 24 subagents (3 matchups × 8 pairs).")
    logger.info("After responses arrive, run with `analyze=true`.")


def analyze_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    meta_path = out_dir / "matchups_meta.json"
    meta = json.loads(meta_path.read_text())

    matchup_ids = ["sigma_vs_delta", "sigma_vs_alpha", "delta_vs_alpha"]

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

    md = ["# D5 D3 — 7-Baseline Pairwise Corroboration (M8 broader validation)\n\n"]
    md.append(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*  \n")
    md.append(f"*Seed: {config.seed}, n_pairs/matchup: {config.n_pairs}*  \n")
    md.append(f"*Per DECISIONS 2026-04-28 (g). Strict 1 subagent/pair task isolation.*\n\n")

    md.append("## 3 Matchups (all close-cluster from Phase 2F audit_v3 isolated)\n\n")
    md.append("Phase 2F 2-plan/subagent audit Δ:\n")
    md.append("- σ 25.25 vs δ 25.12: Δ +0.13\n")
    md.append("- σ 25.25 vs α 25.00: Δ +0.25\n")
    md.append("- δ 25.12 vs α 25.00: Δ +0.12\n\n")
    md.append("Pre-registered binding (DECISIONS 2026-04-28 (g)):\n")
    md.append("- All 3 matchups within ±5/8 → 7-baseline close cluster IS true null; M8 cross-goal effect doesn't generalize\n")
    md.append("- ≥2/3 matchups break ≥6/8 in same baseline's favor → 2-plan audit DID rank correctly; M8 effect is cross-goal-specific\n")
    md.append("- Mixed → per-matchup nuance documented\n\n")

    md.append("| Matchup | la wins | lb wins | ties | A-pos rate | Winner |\n")
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

    md.append("\n## Per-matchup verdict (vs Phase 2F audit Δ)\n\n")
    expectations = {
        "sigma_vs_delta": ("σ vs δ close-cluster", "+0.13 / 45 (effective tie)"),
        "sigma_vs_alpha": ("σ vs α close-cluster", "+0.25 / 45 (effective tie)"),
        "delta_vs_alpha": ("δ vs α close-cluster", "+0.12 / 45 (effective tie)"),
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
            verd = f"**INCONCLUSIVE** — {wins_la}-{wins_lb} (within ±5/8 of tie)."
        md.append(f"### {title}\n")
        md.append(f"- Audit Δ: {audit_delta}\n")
        md.append(f"- Pairwise: {wins_la} {r['la']} / {wins_lb} {r['lb']} / {r['ties']} TIE\n")
        md.append(f"- {verd}\n\n")

    md.append("\n## Position-bias check\n\n")
    md.append("A-side win rate per matchup (should be 25-75% — outside this range = position-bias confound):\n\n")
    for r in rows:
        if "status" in r:
            continue
        apos = r["a_wins"] / max(1, r["a_wins"] + r["b_wins"])
        flag = "✓" if 0.25 <= apos <= 0.75 else "⚠️ POSITION BIAS"
        md.append(f"- {r['matchup']}: {apos:.0%} ({flag})\n")

    md.append("\n## Per-pair rationales\n\n")
    for r in rows:
        if "status" in r:
            continue
        md.append(f"### {r['matchup']}\n\n")
        for rp in r.get("rationales", [])[:8]:
            md.append(f"- **{rp['pair_id']}**: {rp['winner_raw']}\n")
            md.append(f"    > {rp['rationale']}\n")

    out_path = out_dir / "phase6_seven_baseline_pairwise_summary.md"
    out_path.write_text("".join(md))
    logger.info("Wrote summary: %s", out_path)


def main(config: Config):
    if config.analyze:
        analyze_phase(config)
    else:
        submit_phase(config)


if __name__ == "__main__":
    main(chz.entrypoint(Config))
