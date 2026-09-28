"""D5 v2 pairwise tournament — 3 prompt variants per matchup.

Replaces fair_pairwise_v1 for Phase 2F. Each logical matchup is run THREE
times with different judge prompts:

  PV1 (minimal):       "which plan is better, A / B / TIE? + 1-sentence rationale"
  PV2 (reviewer-lite): PV1 + soundness/novelty/feasibility/empirical-rigor checklist
  PV3 (rubric verdict): full 9-dim audit per side, per-dim diff, then which is better

Each call records the prompt verbatim in pairwise_summary.md so we can audit later.

Plans loaded from each run's buffer.jsonl (frozen ξ/σ/δ/ε) or eval_rollouts.jsonl
(trained μ/α/β with iter or epoch field). 5 essential matchups by default:
  σ vs ξ            (oracle inference value)
  μ vs σ            (training value over fixed oracle)
  μ vs ξ            (D5 end-to-end value vs naive baseline)
  α-v2 vs σ         (Opus distillation vs frozen+oracle)
  β-v2 vs σ         (ref-SFT vs frozen+oracle)

Reference comparisons (informational, not D5-determining):
  σ vs δ            (extracted-pattern vs full-reference-plan in prompt)
  μ vs δ            (training+oracle vs reference-in-prompt)
  δ vs ε            (30B vs 235B at same prompt)

Usage (submit phase, then dispatch subagents externally):
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.pairwise_v2 \
        out_path=projects/d5_abstract_retrieve_refine/runs/2026_05_xx_pairwise_v2

After subagents respond:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.pairwise_v2 \
        out_path=projects/d5_abstract_retrieve_refine/runs/2026_05_xx_pairwise_v2 \
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
from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v2 import extract_solution

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[3]


PV1_TEMPLATE = """\
You are evaluating two research plans on the same scientific problem. Pick the better
one or call it a TIE.

# Research Goal
{goal}

# Plan A
{plan_a}

# Plan B
{plan_b}

Output JSON only:
{{"winner": "A|B|TIE", "rationale": "<1 sentence>"}}
"""

PV2_TEMPLATE = """\
You are evaluating two research plans on the same scientific problem. Pick the better
one or call it a TIE. Consider:
- Soundness: are claims supported by math or empirical evidence?
- Novelty: is the proposed approach distinct from cited prior work?
- Feasibility: are compute, hyperparameters, and evaluation realistic?
- Empirical rigor: are baselines named with prior numbers; are stats/seeds specified?

# Research Goal
{goal}

# Plan A
{plan_a}

# Plan B
{plan_b}

Output JSON only:
{{"winner": "A|B|TIE", "rationale": "<1-2 sentences referencing the criteria>"}}
"""

PV3_TEMPLATE = """\
You are an expert ML/AI conference reviewer comparing two research plans on the same
scientific problem. First score each plan on 9 dimensions (1-5 each), then output a
verdict based on the per-dim differences.

# Dimensions

UNIVERSAL (1-5 each):
- U1 Soundness: claims supported by evidence
- U2 Significance: real problem with real impact
- U3 Originality: novel beyond pretraining-knowledge
- U4 Clarity: well-reasoned and structurally clear
- U5 Reproducibility: compute / hparams / stats specified

SUBFIELD (1-5 each, for test-time-search / search-with-LLMs papers):
- T1 Necessity: shows test-time RL is necessary
- T2 Disentanglement: separates parameter updates from search alone
- T3 Compute accounting: latency / cost honesty
- T4 Reward-hacking awareness: probes for shortcuts/Goodhart

# Research Goal
{goal}

# Plan A
{plan_a}

# Plan B
{plan_b}

# Output

Output JSON only with this schema:
{{
  "scores_a": {{"U1":<1-5>, "U2":<1-5>, ..., "T4":<1-5>}},
  "scores_b": {{"U1":<1-5>, ..., "T4":<1-5>}},
  "winner": "A|B|TIE",
  "rationale": "<2-3 sentences referencing the most decisive per-dim differences>"
}}
"""


PROMPT_VARIANTS = {
    "PV1": ("minimal — A/B/TIE + 1-sentence rationale", PV1_TEMPLATE),
    "PV2": ("reviewer-lite — soundness/novelty/feasibility/empirical-rigor checklist", PV2_TEMPLATE),
    "PV3": ("v3-rubric verdict — 9-dim score per side then verdict", PV3_TEMPLATE),
}


@chz.chz
class Config:
    out_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_05_xx_pairwise_v2"
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"

    # Run dirs (each with buffer.jsonl or eval_rollouts.jsonl)
    xi_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_05_xx_xi_v1"
    sigma_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_05_xx_sigma_v1"
    delta_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_05_xx_delta_v1"
    epsilon_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_05_xx_epsilon_v1"
    mu_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_05_xx_mu_v2"
    alpha_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_05_xx_alpha_v2"
    beta_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_05_xx_beta_v2"

    # For trained baselines, which iter/epoch to read from eval_rollouts.jsonl
    mu_iter: int = 8
    alpha_epoch: int = 4
    beta_iter: int = 8

    n_pairs: int = 8
    seed: int = 100
    analyze: bool = False
    skip_pv: str = ""  # comma-separated prompt-variants to skip (e.g. "PV3" for cheap initial sweep)


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _load_frozen_buffer(run_dir: Path, label: str, variant_field: str = "variant") -> list[dict[str, str]]:
    """Load plans from a buffer.jsonl produced by baseline_frozen_v2."""
    bp = run_dir / "buffer.jsonl"
    plans = []
    if not bp.exists():
        logger.warning("buffer.jsonl missing for %s: %s", label, bp)
        return plans
    with open(bp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            plans.append({
                "plan_id": f"{label}::{d.get(variant_field, '?')}_{d.get('sample_idx', '?')}",
                "text": extract_solution(d.get("plan_text", "")),
            })
    return plans


def _load_eval_rollouts(run_dir: Path, key_field: str, key_value: int, label: str) -> list[dict[str, str]]:
    """Load plans from eval_rollouts.jsonl filtered to a specific iter or epoch."""
    ep = run_dir / "eval_rollouts.jsonl"
    plans = []
    if not ep.exists():
        logger.warning("eval_rollouts.jsonl missing for %s: %s", label, ep)
        return plans
    with open(ep) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get(key_field) == key_value:
                plans.append({
                    "plan_id": f"{label}::{d.get('plan_id', '?')}",
                    "text": extract_solution(d.get("text", "")),
                })
    return plans


def load_all_plans(config: Config) -> dict[str, list[dict]]:
    plans: dict[str, list[dict]] = {}
    plans["xi"] = _load_frozen_buffer(_resolve(config.xi_run), "xi")
    plans["sigma"] = _load_frozen_buffer(_resolve(config.sigma_run), "sigma")
    plans["delta"] = _load_frozen_buffer(_resolve(config.delta_run), "delta")
    plans["epsilon"] = _load_frozen_buffer(_resolve(config.epsilon_run), "epsilon")
    plans["mu"] = _load_eval_rollouts(_resolve(config.mu_run), "iter", config.mu_iter, "mu")
    plans["alpha"] = _load_eval_rollouts(_resolve(config.alpha_run), "epoch", config.alpha_epoch, "alpha")
    plans["beta"] = _load_eval_rollouts(_resolve(config.beta_run), "iter", config.beta_iter, "beta")

    for k, v in plans.items():
        logger.info("  %s: %d plans loaded", k, len(v))
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


# Essential D5 matchups (decision-determining)
ESSENTIAL_MATCHUPS = [
    ("sigma", "xi", "sigma_vs_xi"),
    ("mu", "sigma", "mu_vs_sigma"),
    ("mu", "xi", "mu_vs_xi"),
    ("alpha", "sigma", "alpha_vs_sigma"),
    ("beta", "sigma", "beta_vs_sigma"),
]
# Reference comparisons (informational)
REFERENCE_MATCHUPS = [
    ("sigma", "delta", "sigma_vs_delta"),
    ("mu", "delta", "mu_vs_delta"),
    ("delta", "epsilon", "delta_vs_epsilon"),
]


def submit_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    goal = _resolve(config.goal_path).read_text().strip()
    plans = load_all_plans(config)

    skip = {s.strip() for s in config.skip_pv.split(",") if s.strip()}
    pwc = OpusPairwiseClient(log_path=out_dir, timeout_sec=1800.0)
    meta_path = out_dir / "matchups_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    n_submitted = 0
    for la, lb, mid in ESSENTIAL_MATCHUPS + REFERENCE_MATCHUPS:
        if not plans.get(la) or not plans.get(lb):
            logger.warning("Skipping %s — missing plans", mid)
            continue
        ms = build_matchups(plans[la], plans[lb], la, lb, config.n_pairs, rng)
        meta[mid] = ms
        for pv_key, (pv_desc, pv_template) in PROMPT_VARIANTS.items():
            if pv_key in skip:
                continue
            mid_pv = f"{mid}__{pv_key}"
            payload = {
                "kind": "pairwise",
                "matchup_id": mid_pv,
                "goal": goal,
                "prompt_variant": pv_key,
                "prompt_description": pv_desc,
                "prompt_template": pv_template,
                "pairs": [
                    {k: v for k, v in m.items()
                     if k in ("pair_id", "plan_a_id", "plan_a_text", "plan_b_id", "plan_b_text")}
                    for m in ms
                ],
            }
            pwc.submit(mid_pv, payload)
            n_submitted += 1
            logger.info("Submitted %s (%d pairs)", mid_pv, len(ms))

    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Total %d (matchup × prompt-variant) requests submitted to %s", n_submitted, out_dir)


def analyze_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    meta_path = out_dir / "matchups_meta.json"
    meta = json.loads(meta_path.read_text())

    rows = []
    for la, lb, mid in ESSENTIAL_MATCHUPS + REFERENCE_MATCHUPS:
        if mid not in meta:
            continue
        ms = meta[mid]
        for pv_key, (pv_desc, _) in PROMPT_VARIANTS.items():
            mid_pv = f"{mid}__{pv_key}"
            rp = out_dir / "pairwise_responses" / f"{mid_pv}.json"
            if not rp.exists():
                rows.append({"matchup": mid, "pv": pv_key, "status": "no_response"})
                continue
            verdicts = json.loads(rp.read_text()).get("verdicts", {})
            wla = wlb = ties = 0
            for m in ms:
                v = verdicts.get(m["pair_id"])
                if v is None:
                    continue
                w = (v.get("winner") or "").upper().strip()
                if w == "TIE":
                    ties += 1
                elif w == "A":
                    if m["true_a_label"] == la:
                        wla += 1
                    elif m["true_a_label"] == lb:
                        wlb += 1
                elif w == "B":
                    if m["true_b_label"] == la:
                        wla += 1
                    elif m["true_b_label"] == lb:
                        wlb += 1
            winner = "TIE" if wla == wlb else (la if wla > wlb else lb)
            rows.append({"matchup": mid, "la": la, "lb": lb, "pv": pv_key,
                          "wins_la": wla, "wins_lb": wlb, "ties": ties, "winner": winner})

    md = ["# D5 Pairwise Tournament v2 — 3 prompt variants per matchup\n\n"]
    md.append(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}, seed={config.seed}, n_pairs={config.n_pairs}*\n\n")

    md.append("## Prompt variants used\n\n")
    for pv_key, (pv_desc, pv_text) in PROMPT_VARIANTS.items():
        md.append(f"### {pv_key}: {pv_desc}\n\n```\n{pv_text}\n```\n\n")

    md.append("## Results table\n\n")
    md.append("| Matchup | la | lb | PV | la-wins | lb-wins | ties | Winner |\n")
    md.append("|---|---|---|---|---:|---:|---:|---|\n")
    for r in rows:
        if "status" in r:
            md.append(f"| {r['matchup']} |  |  | {r['pv']} | (no response) |  |  |  |\n")
        else:
            md.append(f"| {r['matchup']} | {r['la']} | {r['lb']} | {r['pv']} |"
                      f" {r['wins_la']} | {r['wins_lb']} | {r['ties']} | **{r['winner']}** |\n")

    out_path = out_dir / "pairwise_summary.md"
    out_path.write_text("".join(md))
    logger.info("Wrote summary: %s", out_path)


def main(config: Config):
    if config.analyze:
        analyze_phase(config)
    else:
        submit_phase(config)


if __name__ == "__main__":
    main(chz.entrypoint(Config))
