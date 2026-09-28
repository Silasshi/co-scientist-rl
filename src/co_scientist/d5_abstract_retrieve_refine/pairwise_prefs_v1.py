"""D5 Phase 0.6 sanity check — pairwise preference orchestrator.

Tests Opus pairwise-preference ranking vs Opus absolute-score ranking on the
5-baseline matrix: ε > μ > β > δ > α (by absolute audit). Pairwise checks
whether this ranking holds under a different grader formulation, and
whether α < δ ("Opus distillation hurts") is real or a grader artifact.

Tournament: 5 head-to-head pairs × 8 random matchups each = 40 Opus calls.
Each matchup compares one plan from each baseline (position randomized),
Opus picks A or B + brief rationale.

Pairs:
  - μ vs δ — does training help?
  - μ vs β — does architecture beat direct ref SFT (closely)?
  - β vs α — large absolute gap (+1.63), pairwise sanity
  - α vs δ — "Opus distillation hurts" reality check
  - μ vs ε — paper narrative: 30B trained vs 235B prompt

Plans extracted from each baseline's last available eval batch (8 plans).
Run via:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.pairwise_prefs_v1 \
        log_path=projects/d5_abstract_retrieve_refine/runs/2026_04_25_pairwise_v1
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
    log_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_25_pairwise_v1"
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"

    # Source run dirs (extract last eval batch from each)
    mu_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_25_mu_baseline_v1"
    beta_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_25_beta_baseline_v1"
    alpha_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_25_alpha_baseline_v1"
    delta_eps_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_baseline_delta_epsilon"

    # Tournament
    n_pairs_per_matchup: int = 8  # 8 random A/B pairings (one plan from each side)
    seed: int = 42


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _last_eval_iter(eval_path: Path, key_field: str) -> int:
    """Find max value of key_field (iter or epoch) in eval_rollouts.jsonl."""
    best = -1
    with open(eval_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            v = d.get(key_field, -1)
            if v > best: best = v
    return best


def load_baseline_plans(config: Config) -> dict[str, list[dict[str, str]]]:
    """Load 8 plans from each baseline (μ/β/α/δ/ε). Returns {label: [{plan_id, text}, ...]}."""
    plans: dict[str, list[dict[str, str]]] = {}

    # μ — last iter from eval_rollouts (key=iter)
    mu_eval = _resolve(config.mu_run) / "eval_rollouts.jsonl"
    last_iter = _last_eval_iter(mu_eval, "iter")
    plans["mu"] = []
    with open(mu_eval) as f:
        for line in f:
            d = json.loads(line)
            if d["iter"] == last_iter:
                plans["mu"].append({"plan_id": d["plan_id"], "text": extract_solution(d["text"])})
    logger.info("μ: %d plans from iter=%d", len(plans["mu"]), last_iter)

    # β — last iter from eval_rollouts (key=iter)
    beta_eval = _resolve(config.beta_run) / "eval_rollouts.jsonl"
    last_iter = _last_eval_iter(beta_eval, "iter")
    plans["beta"] = []
    with open(beta_eval) as f:
        for line in f:
            d = json.loads(line)
            if d["iter"] == last_iter:
                plans["beta"].append({"plan_id": d["plan_id"], "text": extract_solution(d["text"])})
    logger.info("β: %d plans from iter=%d", len(plans["beta"]), last_iter)

    # α — last epoch from eval_rollouts (key=epoch)
    alpha_eval = _resolve(config.alpha_run) / "eval_rollouts.jsonl"
    last_epoch = _last_eval_iter(alpha_eval, "epoch")
    plans["alpha"] = []
    with open(alpha_eval) as f:
        for line in f:
            d = json.loads(line)
            if d["epoch"] == last_epoch:
                plans["alpha"].append({"plan_id": d["plan_id"], "text": extract_solution(d["text"])})
    logger.info("α: %d plans from epoch=%d", len(plans["alpha"]), last_epoch)

    # δ + ε from buffer.jsonl (condition='delta' or 'epsilon')
    de_buf = _resolve(config.delta_eps_run) / "buffer.jsonl"
    plans["delta"] = []
    plans["epsilon"] = []
    with open(de_buf) as f:
        for line in f:
            d = json.loads(line)
            cond = d.get("condition", "")
            if cond == "delta":
                plans["delta"].append({
                    "plan_id": f"delta_{d['sample_idx']}",
                    "text": extract_solution(d["plan_text"]),
                })
            elif cond == "epsilon":
                plans["epsilon"].append({
                    "plan_id": f"epsilon_{d['sample_idx']}",
                    "text": extract_solution(d["plan_text"]),
                })
    logger.info("δ: %d plans, ε: %d plans", len(plans["delta"]), len(plans["epsilon"]))

    for label, ps in plans.items():
        assert len(ps) >= 8, f"baseline {label} has only {len(ps)} plans, need >= 8"

    return plans


def build_matchups(
    plans_a: list[dict[str, str]],
    plans_b: list[dict[str, str]],
    label_a: str,
    label_b: str,
    n_pairs: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Build n_pairs matchups, each pairing one random plan from each side, position randomized.

    Returns list of:
      {pair_id, plan_a_id, plan_a_text, plan_b_id, plan_b_text,
       true_a_label, true_b_label, position_swapped}
    where (plan_a, plan_b) shown to Opus may have positions swapped.
    """
    matchups = []
    for i in range(n_pairs):
        # Sample with replacement so n_pairs > min(|A|,|B|) is OK (here 8 plans × 8 pairs = perfect 1-1)
        pa = rng.choice(plans_a) if i >= len(plans_a) else plans_a[i % len(plans_a)]
        pb = rng.choice(plans_b) if i >= len(plans_b) else plans_b[i % len(plans_b)]
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


def build_pairwise_request_payload(
    *,
    matchup_id: str,
    goal: str,
    matchups: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pairwise request payload for the daemon."""
    # Strip orchestration-only fields from what the subagent sees
    daemon_pairs = [
        {
            "pair_id": m["pair_id"],
            "plan_a_id": m["plan_a_id"], "plan_a_text": m["plan_a_text"],
            "plan_b_id": m["plan_b_id"], "plan_b_text": m["plan_b_text"],
        }
        for m in matchups
    ]
    return {
        "kind": "pairwise",
        "matchup_id": matchup_id,
        "goal": goal,
        "pairs": daemon_pairs,
    }


def main(config: Config):
    log_dir = _resolve(config.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)
    goal = _resolve(config.goal_path).read_text().strip()
    plans = load_baseline_plans(config)

    # Define the 5 matchups
    matchup_specs = [
        ("mu", "delta"),
        ("mu", "beta"),
        ("beta", "alpha"),
        ("alpha", "delta"),
        ("mu", "epsilon"),
    ]

    pairwise_client = OpusPairwiseClient(log_path=log_dir, timeout_sec=1800.0)

    # Build all matchups + write all 5 request files (subagents will be dispatched by main agent)
    all_matchups: dict[str, list[dict[str, Any]]] = {}
    for la, lb in matchup_specs:
        ms = build_matchups(plans[la], plans[lb], la, lb, config.n_pairs_per_matchup, rng)
        matchup_id = f"{la}_vs_{lb}"
        all_matchups[matchup_id] = ms
        payload = build_pairwise_request_payload(
            matchup_id=matchup_id, goal=goal, matchups=ms,
        )
        pairwise_client.submit(matchup_id, payload)
        logger.info("Wrote pairwise request: %s (%d pairs)", matchup_id, len(ms))

    # Save full matchup metadata (incl. true_labels + position_swapped) for post-hoc analysis
    meta_path = log_dir / "matchups_meta.json"
    meta_path.write_text(json.dumps(all_matchups, indent=2))
    logger.info("Saved %s — pairwise dispatch ready. Now spawn 5 subagents to process.", meta_path)
    logger.info(
        "Once responses land in %s/pairwise_responses/, run: "
        "PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.pairwise_prefs_v1 analyze=true "
        "log_path=%s",
        log_dir, log_dir,
    )


def analyze(config: Config) -> None:
    """Aggregate pairwise responses into win-rate matrix + ranking."""
    log_dir = _resolve(config.log_path)
    meta_path = log_dir / "matchups_meta.json"
    assert meta_path.exists(), f"matchups_meta.json missing in {log_dir}"
    matchups_meta = json.loads(meta_path.read_text())

    summary_path = log_dir / "pairwise_summary.md"
    out_lines = ["# D5 Phase 0.6 Pairwise Tournament — Results\n"]
    out_lines.append(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
    out_lines.append(f"*Seed: {config.seed}, n_pairs per matchup: {config.n_pairs_per_matchup}*\n")
    out_lines.append("\n## Win rates per matchup\n")
    out_lines.append("| Matchup | Wins (left) | Wins (right) | Ties | Left win rate |\n|---|---|---|---|---|\n")

    win_records: dict[tuple[str, str], dict[str, int]] = {}

    for matchup_id, matchups in matchups_meta.items():
        la, _, lb = matchup_id.partition("_vs_")
        resp_path = log_dir / "pairwise_responses" / f"{matchup_id}.json"
        if not resp_path.exists():
            out_lines.append(f"| {matchup_id} | (no response) | | | |\n")
            continue
        resp = json.loads(resp_path.read_text())
        verdicts = resp.get("verdicts", {})
        wins_la = wins_lb = ties = 0
        for m in matchups:
            v = verdicts.get(m["pair_id"])
            if v is None: continue
            winner = (v.get("winner") or "").upper().strip()
            # winner = "A" or "B" or "TIE"; map back through position_swapped to true label
            if winner == "TIE":
                ties += 1
                continue
            elif winner == "A":
                shown_winner = m["true_a_label"]
            elif winner == "B":
                shown_winner = m["true_b_label"]
            else:
                continue  # malformed
            if shown_winner == la: wins_la += 1
            elif shown_winner == lb: wins_lb += 1
        n_total = wins_la + wins_lb + ties
        rate = wins_la / max(1, wins_la + wins_lb)
        out_lines.append(f"| **{la}** vs **{lb}** | {wins_la} | {wins_lb} | {ties} | {rate:.1%} |\n")
        win_records[(la, lb)] = {"left": wins_la, "right": wins_lb, "ties": ties}

    # Bradley-Terry-style ranking via simple win-counting (heuristic)
    out_lines.append("\n## Pairwise vs absolute ranking\n")
    out_lines.append("Absolute ranking by Opus D3-canonical /20:\n")
    out_lines.append("  ε (12.13) > μ (8.88) > β (7.75) > δ (6.75) > α (6.12)\n\n")
    out_lines.append("Pairwise matchup outcomes:\n")
    for (la, lb), rec in win_records.items():
        winner = "tie" if rec["left"] == rec["right"] else (la if rec["left"] > rec["right"] else lb)
        out_lines.append(f"- {la} vs {lb}: **{winner}** wins ({rec['left']}–{rec['right']}, ties={rec['ties']})\n")

    out_lines.append("\n## Inversion check\n")
    out_lines.append("Critical pairs to verify:\n")
    out_lines.append(
        "- α vs δ: absolute says δ > α (6.75 > 6.12). If pairwise reverses, "
        "then 'Opus distillation hurts 30B' is grader artifact.\n"
    )
    out_lines.append(
        "- μ vs δ: absolute says μ > δ (+2.13). If pairwise close to 50/50, "
        "training value is illusory.\n"
    )
    out_lines.append(
        "- μ vs ε: paper narrative — μ should lose; quantify by how much.\n"
    )

    summary_path.write_text("".join(out_lines))
    logger.info("Wrote analysis: %s", summary_path)
    print(summary_path.read_text())


if __name__ == "__main__":
    @chz.chz
    class CLI(Config):
        analyze: bool = False

    cfg = chz.entrypoint(CLI)
    if cfg.analyze:
        analyze(cfg)
    else:
        main(cfg)
