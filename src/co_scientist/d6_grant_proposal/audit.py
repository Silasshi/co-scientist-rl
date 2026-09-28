"""D6 audit v1 — isolated 12-signal scoring for grant proposals.

Adapted from D5 audit_v3_isolated.py. Uses D4's 12-signal rubric via
shared/grant_signal_reward.py instead of D5's 9-dim TTT-Discover rubric.

Method: identical isolation design from D5 audit_v3_isolated.py:
1. Load all plans from specified baselines
2. Split into N_BATCHES balanced batches (each batch has 1 plan per baseline)
3. Anonymize within each batch (plan_A, plan_B, ...)
4. Dispatch N_BATCHES parallel Opus subagents — independent calibration
5. Each subagent scores each plan independently on all 12 signals

Score: D4 aggregate = weighted mean of normalized (1-5 → 0-1) signal scores.
NOT D5's /45 scale.

Usage:
    # Prepare audit request files
    PYTHONPATH=src python -m co_scientist.d6_grant_proposal.audit \\
        goal_domain=ai goal_name=02_foundational_rl \\
        baselines_json='[{"label":"ref","mode":"file","path":"..."}]'

    # After subagents complete
    PYTHONPATH=src python -m co_scientist.d6_grant_proposal.audit \\
        goal_domain=ai goal_name=02_foundational_rl analyze=true
"""
from __future__ import annotations

import json
import logging
import random
import string
import sys
from pathlib import Path

import chz

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v1 import extract_solution
from co_scientist.d6_grant_proposal.prompt_loader import render_prompt
from co_scientist.d6_grant_proposal.signals import (
    SIGNALS, SIGNAL_VARIANTS, SIGNAL_WEIGHTS, aggregate_reward,
    build_single_call_prompt,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _load_plans_frozen(run_dir: Path, label: str) -> list[dict]:
    """Load plans from a frozen baseline (buffer.jsonl)."""
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
                "true_baseline": label,
                "true_plan_id": f"{label}_{d.get('sample_idx', len(plans))}",
                "text": extract_solution(d.get("plan_text", "")),
            })
    return plans


def _load_plans_eval(run_dir: Path, label: str, key_field: str, key_value) -> list[dict]:
    """Load plans from a training run (eval_rollouts.jsonl) filtered by key."""
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
                    "true_baseline": label,
                    "true_plan_id": d.get("plan_id", f"{label}_{len(plans)}"),
                    "text": extract_solution(d.get("text", "")),
                })
    return plans


def _load_plans_file(file_path: Path, label: str) -> list[dict]:
    """Load plans from a JSONL file with {plan_id, text} entries."""
    plans = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            plans.append({
                "true_baseline": label,
                "true_plan_id": d.get("plan_id", f"{label}_{len(plans)}"),
                "text": extract_solution(d.get("text", d.get("plan_text", ""))),
            })
    return plans


def _build_audit_prompt(goal: str, plans: list[dict], domain: str = "AI/ML") -> str:
    """Build isolation audit prompt for a batch of anonymized plans.

    Uses D4's 12-signal rubric via build_single_call_prompt for each plan.
    The prompt asks the grader to score all 12 signals for each plan
    independently (no cross-plan anchoring).
    """
    # Determine signal set (G12 vs G12a based on domain)
    non_stem_domains = {"social_science", "public_policy", "criminal_justice", "education"}
    active_signals = list(SIGNALS)
    if any(d in domain.lower() for d in non_stem_domains):
        # Replace G12_formalism with G12a_analytical_framework
        active_signals = [
            SIGNAL_VARIANTS["G12a_analytical_framework"]
            if s.id == "G12_formalism" else s
            for s in active_signals
        ]

    plan_blocks = []
    for p in plans:
        anon_id = p["anon_id"]
        text = p["text"][:12000]  # truncate for context
        plan_blocks.append(f"# Plan {anon_id}\n\n{text}")
    plans_text = "\n\n---\n\n".join(plan_blocks)

    signal_names = ", ".join(s.id for s in active_signals)
    dim_lines = "".join(
        f'  <dim id="{s.id}"><reasoning>...</reasoning><score>1-5</score></dim>\n'
        for s in active_signals
    )

    return render_prompt(
        "audit/absolute_score.md",
        goal=goal,
        n_plans=len(plans),
        plans_text=plans_text,
        n_signals=len(active_signals),
        signal_names=signal_names,
        dim_lines=dim_lines,
    )


def _parse_audit_response(raw: str) -> dict[str, dict]:
    """Parse audit response into {anon_id: {scores, aggregate_reward, key_diff}}."""
    import re
    results = {}
    for plan_m in re.finditer(r'<plan id="([^"]+)">(.*?)</plan>', raw, re.DOTALL):
        anon_id = plan_m.group(1)
        block = plan_m.group(2)
        scores = {}
        for dim_m in re.finditer(r'<dim id="([^"]+)">.*?<score>(\d+)</score>', block, re.DOTALL):
            scores[dim_m.group(1)] = int(dim_m.group(2))
        agg_m = re.search(r"<aggregate_reward>([\d.]+)</aggregate_reward>", block)
        agg = float(agg_m.group(1)) if agg_m else aggregate_reward(scores)
        kd_m = re.search(r"<key_differentiator>(.*?)</key_differentiator>", block, re.DOTALL)
        results[anon_id] = {
            "scores": scores,
            "aggregate_reward": agg,
            "key_differentiator": kd_m.group(1).strip() if kd_m else "",
        }
    return results


@chz.chz
class Config:
    goal_domain: str = "ai"           # "ai" | "natural_science" | "social_science"
    goal_name: str = "02_foundational_rl"
    out_path: str = ""  # default: auto-set from goal_domain/goal_name
    dataset_base: str = "projects/d6_grant_proposal/dataset"
    baselines_json: str = ""  # JSON list of {label, mode, run/path, [key_field, key_value]}
    seed: int = 7777
    n_batches: int = 8
    analyze: bool = False


def main(config: Config):
    goal_dir = _resolve(f"{config.dataset_base}/{config.goal_domain}/{config.goal_name}")
    goal = (goal_dir / "research_goal.md").read_text().strip()

    out_path = config.out_path or f"projects/d6_grant_proposal/runs/audit_{config.goal_domain}_{config.goal_name}"
    out_dir = _resolve(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.analyze:
        _analyze(out_dir)
        return

    baselines = json.loads(config.baselines_json) if config.baselines_json else []
    if not baselines:
        logger.warning("No baselines specified. Provide baselines_json to load plans.")
        return

    # Load all plans
    all_plans: list[dict] = []
    for spec in baselines:
        label = spec["label"]
        mode = spec.get("mode", "frozen")
        if mode == "frozen":
            run_dir = _resolve(spec["run"])
            plans = _load_plans_frozen(run_dir, label)
        elif mode == "eval":
            run_dir = _resolve(spec["run"])
            plans = _load_plans_eval(run_dir, label, spec["key_field"], spec["key_value"])
        elif mode == "file":
            plans = _load_plans_file(_resolve(spec["path"]), label)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        logger.info("Loaded %d plans for baseline '%s'", len(plans), label)
        all_plans.extend(plans)

    n_baselines = len(baselines)
    n_per_baseline = min(len([p for p in all_plans if p["true_baseline"] == b["label"]])
                         for b in baselines)
    logger.info("Total: %d plans, %d baselines, %d per baseline",
                 len(all_plans), n_baselines, n_per_baseline)

    # Build balanced batches
    rng = random.Random(config.seed)
    by_baseline = {b["label"]: [] for b in baselines}
    for p in all_plans:
        by_baseline[p["true_baseline"]].append(p)
    for label in by_baseline:
        rng.shuffle(by_baseline[label])

    batches: list[list[dict]] = [[] for _ in range(config.n_batches)]
    for label, plans in by_baseline.items():
        for i, plan in enumerate(plans[: config.n_batches]):
            batches[i].append(plan)

    # Anonymize within each batch
    shuffle_map = []
    batch_requests = []
    for batch_idx, batch in enumerate(batches):
        rng.shuffle(batch)
        letters = list(string.ascii_uppercase[:len(batch)])
        anon_batch = []
        for letter, plan in zip(letters, batch):
            plan["anon_id"] = letter
            anon_batch.append(plan)
            shuffle_map.append({
                "batch": batch_idx,
                "anon_id": letter,
                "true_baseline": plan["true_baseline"],
                "true_plan_id": plan["true_plan_id"],
            })
        prompt = _build_audit_prompt(goal, anon_batch)
        batch_requests.append({"batch_idx": batch_idx, "prompt": prompt, "plans": anon_batch})

    shuffle_map_path = out_dir / "shuffle_map.json"
    shuffle_map_path.write_text(json.dumps(shuffle_map, indent=2))
    logger.info("Wrote shuffle_map (%d entries) to %s", len(shuffle_map), shuffle_map_path)

    req_dir = out_dir / "audit_requests"
    req_dir.mkdir(exist_ok=True)
    for req in batch_requests:
        req_path = req_dir / f"batch_{req['batch_idx']:03d}.json"
        req_path.write_text(json.dumps({
            "batch_idx": req["batch_idx"],
            "goal_domain": config.goal_domain,
            "goal_name": config.goal_name,
            "goal": goal,
            "prompt": req["prompt"],
            "plan_ids": [p["anon_id"] for p in req["plans"]],
        }, indent=2))
    logger.info("Wrote %d batch request files to %s", len(batch_requests), req_dir)
    logger.info("Dispatch %d parallel Opus subagents. After responses, run with analyze=true.", config.n_batches)


def _analyze(out_dir: Path):
    """Aggregate audit responses once subagents have written results."""
    shuffle_map_path = out_dir / "shuffle_map.json"
    if not shuffle_map_path.exists():
        logger.error("No shuffle_map.json found at %s", out_dir)
        return

    shuffle_map = json.loads(shuffle_map_path.read_text())
    anon_to_true = {(e["batch"], e["anon_id"]): e for e in shuffle_map}

    resp_dir = out_dir / "audit_responses"
    if not resp_dir.exists():
        logger.error("No audit_responses/ directory at %s", out_dir)
        return

    # Aggregate per true_baseline
    results: dict[str, list[float]] = {}
    for resp_path in sorted(resp_dir.glob("batch_*.json")):
        batch_idx = int(resp_path.stem.replace("batch_", ""))
        try:
            resp = json.loads(resp_path.read_text())
        except Exception as e:
            logger.warning("Could not read %s: %s", resp_path, e)
            continue
        parsed = _parse_audit_response(resp.get("raw", resp.get("response", "")))
        for anon_id, scores_dict in parsed.items():
            key = (batch_idx, anon_id)
            meta = anon_to_true.get(key)
            if not meta:
                continue
            label = meta["true_baseline"]
            agg = scores_dict.get("aggregate_reward", 0.0)
            results.setdefault(label, []).append(agg)

    logger.info("\n=== D6 Audit Results (%s) ===", out_dir.name)
    logger.info("%-25s  %-6s  %-8s  %-8s", "Baseline", "N", "Mean", "Std")
    logger.info("-" * 55)
    for label, vals in sorted(results.items()):
        import statistics
        mean = statistics.mean(vals) if vals else 0.0
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        logger.info("%-25s  %-6d  %-8.4f  %-8.4f", label, len(vals), mean, std)


if __name__ == "__main__":
    chz.entrypoint(main)
