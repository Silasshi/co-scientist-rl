"""D5 audit v3 runner — apply 9-dim hybrid audit to a list of plans.

Phase 2F absolute-audit pass. Reads buffer.jsonl (frozen baselines) or
eval_rollouts.jsonl (trained baselines) from each run dir, dispatches v3
audit subagent payload (file-bus), waits for responses, writes summary.

The daemon should branch on `prompt_version: "v3"` to select
DEPTH_AUDIT_PROMPT_V3 (see shared/audit_prompt_v3.py + opus_audit_subagent.py).

Usage (submit):
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.audit_v3_runner \
        out_path=projects/d5_abstract_retrieve_refine/runs/2026_05_xx_audit_v3

Usage (analyze after responses):
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.audit_v3_runner \
        out_path=projects/d5_abstract_retrieve_refine/runs/2026_05_xx_audit_v3 \
        analyze=true
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

from co_scientist.shared.opus_audit_subagent import OpusAuditClient
from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v2 import (
    build_audit_v3_request_payload,
    extract_solution,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    out_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_05_xx_audit_v3"
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"

    xi_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_05_xx_xi_v1"
    sigma_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_05_xx_sigma_v1"
    delta_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_05_xx_delta_v1"
    epsilon_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_05_xx_epsilon_v1"
    mu_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_05_xx_mu_v2"
    alpha_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_05_xx_alpha_v2"
    beta_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_05_xx_beta_v2"

    mu_iter: int = 8
    alpha_epoch: int = 4
    beta_iter: int = 8

    timeout_sec: float = 1800.0
    analyze: bool = False


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _load_frozen(run_dir: Path, label: str) -> list[dict]:
    bp = run_dir / "buffer.jsonl"
    plans = []
    if not bp.exists():
        return plans
    with open(bp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            plans.append({
                "plan_id": f"{label}::{d.get('variant', '?')}_{d.get('sample_idx', '?')}",
                "text": extract_solution(d.get("plan_text", "")),
            })
    return plans


def _load_eval_rollouts(run_dir: Path, key_field: str, key_value: int, label: str) -> list[dict]:
    ep = run_dir / "eval_rollouts.jsonl"
    plans = []
    if not ep.exists():
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


def load_baselines(config: Config) -> dict[str, list[dict]]:
    return {
        "xi": _load_frozen(_resolve(config.xi_run), "xi"),
        "sigma": _load_frozen(_resolve(config.sigma_run), "sigma"),
        "delta": _load_frozen(_resolve(config.delta_run), "delta"),
        "epsilon": _load_frozen(_resolve(config.epsilon_run), "epsilon"),
        "mu": _load_eval_rollouts(_resolve(config.mu_run), "iter", config.mu_iter, "mu"),
        "alpha": _load_eval_rollouts(_resolve(config.alpha_run), "epoch", config.alpha_epoch, "alpha"),
        "beta": _load_eval_rollouts(_resolve(config.beta_run), "iter", config.beta_iter, "beta"),
    }


def submit_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    goal = _resolve(config.goal_path).read_text().strip()
    baselines = load_baselines(config)

    audit_client = OpusAuditClient(log_path=out_dir)
    n_total_plans = 0
    for label, plans in baselines.items():
        if not plans:
            logger.warning("%s has no plans, skipping", label)
            continue
        # Dispatch one audit request per baseline, with all 8 plans bundled (parser handles per-plan output)
        payload = build_audit_v3_request_payload(iter_idx=0, goal=goal, plans=plans)
        # Tag iter_idx with label-encoded value so file naming is unique
        # OpusAuditClient.submit uses iter_idx in filename; encode label as int hash
        iter_idx = abs(hash(label)) % 100000
        audit_client.submit(iter_idx, payload)
        n_total_plans += len(plans)
        logger.info("Submitted v3 audit for %s (%d plans, iter_idx=%d)", label, len(plans), iter_idx)

    (out_dir / "audit_v3_meta.json").write_text(json.dumps({
        "baselines": {k: len(v) for k, v in baselines.items()},
        "total_plans": n_total_plans,
        "submitted_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))
    logger.info("Submitted %d audit requests, %d total plans", len(baselines), n_total_plans)


def analyze_phase(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    audit_client = OpusAuditClient(log_path=out_dir)
    summaries = audit_client.collect_all(
        out_path=out_dir / "audit_v3_log.jsonl",
        timeout_sec=config.timeout_sec,
        poll_interval_sec=15.0,
    )

    rows = []
    for s in summaries:
        if s.get("mean_total") is None:
            continue
        rows.append(s)

    md = ["# D5 Audit v3 — Absolute Scores per Baseline\n\n"]
    md.append(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
    md.append("| Baseline | n_plans | mean_total /20 | std | min | max |\n")
    md.append("|---|---:|---:|---:|---:|---:|\n")
    for s in rows:
        md.append(
            f"| iter_{s.get('iter', '?')} | {s.get('n', '?')} |"
            f" {s.get('mean_total', '?'):.2f} | {s.get('std_total', '?'):.2f} |"
            f" {s.get('min_total', '?'):.2f} | {s.get('max_total', '?'):.2f} |\n"
        )

    out_path = out_dir / "audit_v3_summary.md"
    out_path.write_text("".join(md))
    logger.info("Wrote summary: %s", out_path)


def main(config: Config):
    if config.analyze:
        analyze_phase(config)
    else:
        submit_phase(config)


if __name__ == "__main__":
    main(chz.entrypoint(Config))
