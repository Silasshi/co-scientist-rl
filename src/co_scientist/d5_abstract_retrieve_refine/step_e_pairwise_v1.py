"""D5 Phase 3 Step E — pairwise rerun: τ_v4_clean vs σ_v4.

Per ML-scientist subagent (2026-04-27): the Step 2.5 pairwise verdict (τ_v4 7-1
STRONG over σ_v4) was conducted on the OLD lucky-draw τ_v4 plans (with citation
bug). With τ_v4_clean now at cross-instance mean 24.50 < σ baseline 25.25, the
pairwise generalizability is uncertain. Re-run pairwise on the patched τ_v4
plans to either confirm (≥6/8) or refute (≤5/8) the 7-1 STRONG claim.

Per subagent task isolation rule (one task per subagent), this dispatches via
file-bus: writes 1 pairwise_requests JSON; orchestrator (Claude Code main) then
fans out 8 subagents (one per pair). Each subagent judges 1 pair in isolation,
writes its verdict back to a per-pair response file. Aggregator combines them.

Phases:
- submit (default): writes 8 pair JSONs + matchups_meta.json
- analyze (analyze=true): aggregates per-pair responses → step_e_pairwise_summary.md

Usage:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.step_e_pairwise_v1
    # external: dispatch 8 Opus subagents (one per pair_NN.json)
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.step_e_pairwise_v1 analyze=true
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

from co_scientist.d5_abstract_retrieve_refine.kappa_prompts_v1 import extract_distillation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    out_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_30_step_e_pairwise"
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"

    tau_clean_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_30_tau_v4_clean"
    sigma_v4_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_sigma_v4"

    n_pairs: int = 8
    seed: int = 52  # distinct from prior 42-51
    analyze: bool = False


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


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


def submit_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    req_dir = out_dir / "pair_requests"
    req_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)
    goal = _resolve(config.goal_path).read_text().strip()

    tau_plans = _load_from_buffer(_resolve(config.tau_clean_run) / "buffer.jsonl",
                                  "tau_v4_clean", max_n=config.n_pairs)
    sigma_plans = _load_from_buffer(_resolve(config.sigma_v4_run) / "buffer.jsonl",
                                    "sigma_v4", max_n=config.n_pairs)
    if len(tau_plans) < config.n_pairs or len(sigma_plans) < config.n_pairs:
        raise RuntimeError(f"need {config.n_pairs} plans each, got tau={len(tau_plans)}, sigma={len(sigma_plans)}")

    matchups = []
    for i in range(config.n_pairs):
        pa = tau_plans[i]
        pb = sigma_plans[i]
        swap = rng.random() < 0.5
        if swap:
            pa, pb = pb, pa
            la, lb = "sigma_v4", "tau_v4_clean"
        else:
            la, lb = "tau_v4_clean", "sigma_v4"
        pair_id = f"pair_{i:02d}"
        matchups.append({
            "pair_id": pair_id,
            "true_a_label": la, "true_b_label": lb, "position_swapped": swap,
            "plan_a_id": pa["plan_id"], "plan_b_id": pb["plan_id"],
        })
        req_payload = {
            "kind": "pairwise_single",
            "pair_id": pair_id,
            "goal": goal,
            "plan_a": pa["text"],
            "plan_b": pb["text"],
            "instructions": (
                "You are an Opus 4.7 pairwise research-plan judge. Read the goal, then"
                " plan A and plan B carefully. Judge which is the better research plan"
                " on the dimensions: methodological rigor, originality, reproducibility,"
                " concrete-spec depth (formulas/hparams/baselines), grounding in cited"
                " prior work, disentanglement design, compute realism, reward-hacking"
                " awareness. Be sharp — do not default to ties unless plans are truly"
                " indistinguishable. Output JSON: {\"winner\": \"A\" | \"B\" | \"TIE\","
                " \"rationale\": \"<2-3 sentences citing specific differences>\"}."
            ),
        }
        (req_dir / f"{pair_id}.json").write_text(json.dumps(req_payload, indent=2))

    (out_dir / "matchups_meta.json").write_text(json.dumps(matchups, indent=2))
    logger.info("Wrote %d pair requests to %s", config.n_pairs, req_dir)
    logger.info("A-position labels: tau=%d sigma=%d (should be ~4-4 random)",
                sum(1 for m in matchups if m["true_a_label"] == "tau_v4_clean"),
                sum(1 for m in matchups if m["true_a_label"] == "sigma_v4"))
    logger.info("Now dispatch %d Opus subagents in parallel (one per pair_NN.json)", config.n_pairs)
    logger.info("After responses arrive, run with `analyze=true`")


def analyze_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    matchups = json.loads((out_dir / "matchups_meta.json").read_text())
    resp_dir = out_dir / "pair_responses"
    if not resp_dir.exists():
        logger.error("No pair_responses dir: %s", resp_dir)
        sys.exit(1)

    tau_wins = sigma_wins = ties = unjudged = 0
    a_wins = b_wins = 0
    rows = []
    for m in matchups:
        rp = resp_dir / f"{m['pair_id']}.json"
        if not rp.exists():
            unjudged += 1
            rows.append({**m, "winner_pos": None, "winner_label": None, "rationale": "(missing)"})
            continue
        d = json.loads(rp.read_text())
        w = (d.get("winner") or "").upper().strip()
        if w == "A":
            a_wins += 1
            wl = m["true_a_label"]
        elif w == "B":
            b_wins += 1
            wl = m["true_b_label"]
        elif w == "TIE":
            ties += 1
            wl = "TIE"
        else:
            unjudged += 1
            wl = None
        if wl == "tau_v4_clean":
            tau_wins += 1
        elif wl == "sigma_v4":
            sigma_wins += 1
        rows.append({**m, "winner_pos": w, "winner_label": wl,
                     "rationale": d.get("rationale", "")[:200]})

    # Verdict
    decisive = tau_wins + sigma_wins
    if tau_wins >= 6:
        sig = "STRONG (τ_v4_clean > σ_v4)"
    elif tau_wins == 5:
        sig = "WEAK (τ_v4_clean > σ_v4 marginal)"
    elif tau_wins == 4 and sigma_wins == 4:
        sig = "TIE"
    elif tau_wins == 4:
        sig = "WEAK (mostly tied)"
    elif sigma_wins == 5:
        sig = "WEAK (σ_v4 > τ_v4_clean — REVERSAL from Step 2.5 7-1 STRONG)"
    elif sigma_wins >= 6:
        sig = "STRONG REVERSAL (σ_v4 > τ_v4_clean) — Step 2.5 7-1 was lucky-draw artifact"
    else:
        sig = "UNCLEAR"

    md = ["# D5 Step E pairwise: τ_v4_clean vs σ_v4 (rerun on patched plans)\n\n"]
    md.append(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*  \n")
    md.append(f"*Seed: {config.seed}, n_pairs: {config.n_pairs}*  \n")
    md.append(f"*Comparison to Step 2.5 verdict: τ_v4 (orig, bug+lucky) vs σ_v4 = 7-1 STRONG*\n\n")

    md.append("## Aggregate\n\n")
    md.append(f"- **τ_v4_clean wins**: {tau_wins} / {config.n_pairs}\n")
    md.append(f"- **σ_v4 wins**: {sigma_wins} / {config.n_pairs}\n")
    md.append(f"- **Ties**: {ties}\n")
    md.append(f"- **Unjudged**: {unjudged}\n")
    md.append(f"- **A-position wins**: {a_wins} / {decisive} ({a_wins/max(1,decisive):.0%}) — should be ~50% (no position bias)\n\n")
    md.append(f"## Verdict: **{sig}**\n\n")

    md.append("## Per-pair verdicts\n\n")
    md.append("| pair | A | B | winner_pos | winner_label | rationale |\n")
    md.append("|---|---|---|---|---|---|\n")
    for r in rows:
        md.append(f"| {r['pair_id']} | {r['true_a_label']} | {r['true_b_label']} | {r['winner_pos']} | {r['winner_label']} | {r['rationale']} |\n")

    out_md = out_dir / "step_e_pairwise_summary.md"
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
