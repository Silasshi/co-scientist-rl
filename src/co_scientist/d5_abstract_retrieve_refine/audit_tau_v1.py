"""D5 Phase 3 Step 2b — τ baseline audit (single-baseline, per-plan isolation).

Runs audit_v3 9-dim ISOLATED on the 8 τ plans from `runs/2026_04_28_tau_v1/buffer.jsonl`.

**Architecture choice**: 8 plans → 8 audit subagents, each scores ONE plan on
9 dims. This is stricter isolation than `audit_v3_isolated.py` (which puts 7
plans per batch, one per baseline) — required for τ since there's only 1
baseline. Also aligns with the cross-session feedback (2026-04-27): one task
per subagent; bundling degrades quality.

Pipeline:
  Phase 1 (this script): writes 8 audit_request JSONs to audit_requests/plan_NN.json
  Phase 2 (external): dispatch 8 Opus subagents, each reads its request,
                      calls Opus with DEPTH_AUDIT_PROMPT_V3, parses JSON,
                      writes audit_responses/plan_NN.json
  Phase 3 (analyze=true): aggregate 8 responses → tau_audit_summary.md

Usage:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.audit_tau_v1
    # external dispatch of 8 subagents (one per plan_NN.json)
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.audit_tau_v1 analyze=true
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import chz

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.d5_abstract_retrieve_refine.kappa_prompts_v1 import extract_distillation
from co_scientist.shared.audit_prompt_v3 import DEPTH_AUDIT_PROMPT_V3, parse_audit_v3_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    out_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_tau_v1_audit"
    tau_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_tau_v1"
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"
    analyze: bool = False


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _load_tau_plans(run_dir: Path) -> list[dict]:
    """Load τ plans from buffer.jsonl, strip Qwen3 thinking + extract <solution>."""
    bp = run_dir / "buffer.jsonl"
    plans = []
    if not bp.exists():
        raise FileNotFoundError(f"buffer.jsonl not found: {bp}")
    with open(bp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            plans.append({
                "true_baseline": "tau",
                "true_plan_id": f"tau_{d['sample_idx']}",
                "text": extract_distillation(d.get("plan_text", "")),
            })
    return plans


def submit_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    goal = _resolve(config.goal_path).read_text().strip()
    plans = _load_tau_plans(_resolve(config.tau_run))
    if len(plans) != 8:
        logger.warning("Expected 8 τ plans, got %d", len(plans))

    decode_map = {}
    req_dir = out_dir / "audit_requests"
    req_dir.mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(plans):
        plan_id = f"plan_{i:02d}"
        decode_map[plan_id] = {
            "true_baseline": p["true_baseline"],
            "true_plan_id": p["true_plan_id"],
            "plan_idx": i,
        }
        # Each subagent gets ONE plan + the FULL DEPTH_AUDIT_PROMPT_V3 template
        # already filled with goal + plan text. Subagent dispatches Opus with
        # this prompt verbatim, parses JSON response, writes to audit_responses/.
        rendered_prompt = DEPTH_AUDIT_PROMPT_V3.format(goal=goal, plan=p["text"])
        req = {
            "kind": "audit_v3_isolated_single",
            "plan_id": plan_id,
            "goal": goal,
            "plan_text": p["text"],
            "true_plan_id_hint_DO_NOT_USE_IN_SCORING": p["true_plan_id"],
            "rendered_audit_prompt": rendered_prompt,
            "instructions": (
                "Call Opus 4.7 with the rendered_audit_prompt verbatim. The prompt"
                " is the full v3 9-dim audit rubric pre-filled with goal + plan."
                " Opus returns JSON per the format described inline in the prompt."
                " Parse JSON, validate it has universal_scores + subfield_scores,"
                " write back to audit_responses/{plan_id}.json with the parsed"
                " fields + raw response. DO NOT modify scores, DO NOT batch with"
                " other plans, DO NOT re-rank — single-plan isolated scoring only."
            ),
        }
        (req_dir / f"{plan_id}.json").write_text(json.dumps(req, indent=2))

    (out_dir / "decode_map.json").write_text(json.dumps(decode_map, indent=2))
    logger.info("Wrote %d single-plan audit requests to %s", len(plans), req_dir)
    logger.info("Now dispatch 8 Opus subagents in parallel (one per plan_NN.json)")
    logger.info("After responses arrive, run with `analyze=true`")


def analyze_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    decode_map = json.loads((out_dir / "decode_map.json").read_text())
    resp_dir = out_dir / "audit_responses"
    if not resp_dir.exists():
        logger.error("No audit_responses dir: %s", resp_dir)
        sys.exit(1)

    DIMS_U = ["U1_soundness", "U2_significance", "U3_originality", "U4_clarity", "U5_reproducibility"]
    DIMS_T = ["T1_necessity", "T2_disentanglement", "T3_compute", "T4_reward_hacking"]
    DIMS = DIMS_U + DIMS_T

    rows = []
    for plan_id in sorted(decode_map.keys()):
        rp = resp_dir / f"{plan_id}.json"
        if not rp.exists():
            logger.warning("Missing response: %s", rp)
            continue
        d = json.loads(rp.read_text())
        # Try both shapes: nested in "judgment" or top-level parsed
        scores = (
            d.get("universal_scores", {}) | d.get("subfield_scores", {})
            if isinstance(d.get("universal_scores"), dict) else d
        )
        # Extract per-dim scores (handle both {dim: {"score": int}} and {dim: int})
        per_dim = {}
        for dim in DIMS:
            entry = (
                d.get("universal_scores", {}).get(dim)
                if dim in DIMS_U
                else d.get("subfield_scores", {}).get(dim)
            )
            if isinstance(entry, dict):
                per_dim[dim] = entry.get("score")
            elif isinstance(entry, (int, float)):
                per_dim[dim] = int(entry)
            else:
                per_dim[dim] = None
        total = sum(v for v in per_dim.values() if isinstance(v, int))
        rows.append({
            "plan_id": plan_id,
            "true_plan_id": decode_map[plan_id]["true_plan_id"],
            "scores": per_dim,
            "total": total if all(isinstance(v, int) for v in per_dim.values()) else None,
        })

    valid = [r for r in rows if r["total"] is not None]
    if not valid:
        logger.error("No valid audit responses parsed")
        return

    totals = [r["total"] for r in valid]
    n = len(totals)
    mean = sum(totals) / n
    std = (sum((t - mean) ** 2 for t in totals) / n) ** 0.5

    # per-dim means
    dim_means = {}
    for d in DIMS:
        vals = [r["scores"][d] for r in valid if isinstance(r["scores"].get(d), int)]
        dim_means[d] = sum(vals) / len(vals) if vals else None

    md = ["# τ baseline audit_v3 ISOLATED — single-baseline summary\n\n"]
    md.append(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
    md.append(f"*N = {n} plans, 8 subagents (one per plan, per-plan isolation)*\n\n")
    md.append("## Aggregate\n\n")
    md.append(f"- **Mean total**: **{mean:.2f} / 45**\n")
    md.append(f"- **Std**: {std:.2f}\n")
    md.append(f"- **Min / Max**: {min(totals)} / {max(totals)}\n")
    md.append(f"- **Distinct totals**: {len(set(totals))}/{n}\n\n")

    md.append("## Per-plan totals\n\n")
    md.append("| plan_id | total |\n|---|---:|\n")
    for r in valid:
        md.append(f"| {r['true_plan_id']} | {r['total']} |\n")
    md.append("\n")

    md.append("## Per-dim means\n\n")
    md.append("| Dim | Mean / 5 |\n|---|---:|\n")
    for d in DIMS:
        md.append(f"| {d} | {dim_means[d]:.2f} |\n" if dim_means[d] is not None else f"| {d} | - |\n")
    md.append("\n")

    # Gate 1 verdict
    sigma_baseline = 25.25
    md.append("## Gate 1 verdict (vs σ = 25.25)\n\n")
    md.append(f"- τ mean = {mean:.2f}, gap = {mean - sigma_baseline:+.2f}\n")
    if mean >= 26.25:
        verdict = "**CONFIDENT** — distillation pipeline > Opus slim. Proceed κ training."
    elif 24.25 <= mean <= 26.25:
        verdict = "**NEUTRAL** — pipeline ≈ slim. Proceed κ training, but mandate pairwise verification."
    elif mean < 23.25:
        verdict = "**STOP** — pipeline degraded. DO NOT train κ; debug distillation prompt / batch / shuffle seed."
    else:
        verdict = "**MARGINAL** — between thresholds; manual review."
    md.append(f"- Verdict: {verdict}\n\n")

    out_md = out_dir / "tau_audit_summary.md"
    out_md.write_text("".join(md))
    logger.info("Wrote summary: %s", out_md)
    print(out_md.read_text())


def main(config: Config):
    if config.analyze:
        analyze_phase(config)
    else:
        submit_phase(config)


if __name__ == "__main__":
    main(chz.entrypoint(Config))
