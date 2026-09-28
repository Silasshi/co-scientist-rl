#!/usr/bin/env python3
"""
Generate summary.md for every run directory and a master RUNS_CATALOG.md.

Usage:
    python3 tools/generate_run_summaries.py

Walks runs/2026/ to find directories containing config.json, metrics.jsonl,
or logs.log. For each, writes a summary.md. Then writes runs/RUNS_CATALOG.md
with all runs organized by method family.
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

RUNS_ROOT = Path(__file__).resolve().parent.parent / "runs" / "2026"
CATALOG_PATH = RUNS_ROOT.parent / "RUNS_CATALOG.md"

# ---------------------------------------------------------------------------
# Method family classification
# ---------------------------------------------------------------------------

METHOD_FAMILIES = {
    "GRPO / Bestversion": [
        "baseline_exploration", "withA1,A2", "best_ver_async",
    ],
    "SDPO": [
        "SDPO",
    ],
    "Reward Shaping": [
        "0-9_scale", "std+mean_weighted", "std_stand+band_bonus",
        "hard_min",
    ],
    "Refinement": [
        "refinement", "blended", "rubric_dropout",
    ],
    "Multi-turn": [
        "multiturn", "multiturn_v2", "multiturn_v3", "multiturn_v4",
    ],
    "IBT": [
        "ibt",
    ],
    "Other": [
        "think_solution", "think_solution_smoke", "cpr",
        "eval_sota", "eval_bon_base_model", "eval_ibt_baseline",
        "eval_rag", "gsm8k", "dense_score",
    ],
}


def classify_family(rel_path: str) -> str:
    """Return the method family for a run based on its relative path."""
    parts = rel_path.split("/")
    # parts[0] = month, parts[1] = method group, ...
    if len(parts) < 2:
        return "Other"
    method_dir = parts[1]
    for family, prefixes in METHOD_FAMILIES.items():
        for pfx in prefixes:
            if method_dir == pfx or method_dir.startswith(pfx):
                return family
    return "Other"


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def read_json(path: Path) -> dict | None:
    """Read a JSON file, return dict or None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def read_metrics(path: Path) -> list[dict]:
    """Read a metrics.jsonl (line-delimited JSON), return list of dicts."""
    results = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass
    return results


def extract_model(config: dict | None) -> str:
    """Extract model name from config."""
    if not config:
        return "unknown"
    for key in ("model_name", "policy_model", "sota_model"):
        val = config.get(key)
        if val:
            return val
    return "unknown"


def extract_dataset(config: dict | None) -> str:
    """Extract dataset info from config flags."""
    if not config:
        return "unknown"
    parts = []
    if config.get("ml_data"):
        parts.append("ml")
    if config.get("arxiv_data"):
        parts.append("arxiv")
    if config.get("pubmed_data"):
        parts.append("pubmed")
    if parts:
        return "+".join(parts)
    # If none of the flags exist at all, it might be a different config schema
    if any(k in config for k in ("ml_data", "arxiv_data", "pubmed_data")):
        return "none"
    return "unknown"


def extract_method(config: dict | None, parent_dir: str) -> str:
    """Extract method name from config or infer from path."""
    if config:
        method = config.get("method") or config.get("mode")
        if method:
            return method
    return parent_dir


def extract_hyperparams(config: dict | None) -> str:
    """Extract key hyperparameters as a compact string."""
    if not config:
        return "N/A"
    parts = []
    if "learning_rate" in config:
        parts.append(f"lr={config['learning_rate']}")
    if "batch_size" in config:
        parts.append(f"bs={config['batch_size']}")
    if "group_size" in config:
        parts.append(f"gs={config['group_size']}")
    if "lora_rank" in config:
        parts.append(f"lora_r={config['lora_rank']}")
    if "num_turns" in config:
        parts.append(f"turns={config['num_turns']}")
    if "num_goals" in config:
        parts.append(f"goals={config['num_goals']}")
    return ", ".join(parts) if parts else "N/A"


def extract_final_reward(metrics: list[dict]) -> str:
    """Extract the final reward from metrics."""
    if not metrics:
        return "N/A"
    last = metrics[-1]
    # Try various reward key names in priority order
    for key in (
        "reward/total",           # bestversion, SDPO, reward-shaping runs
        "reward/turn_mean",       # IBT single_chain / mini_grpo
        "reward/mean",            # multiturn v1/v2/v3/v4
        "reward/refined_mean",    # refinement (doublegeneration) -- report refined
        "reward/initial_mean",    # refinement fallback
        "blend/y2_reward_mean",   # blended
        "rubric/total_valid",     # fallback to rubric if no reward key
        "rubric/turn_mean",       # IBT rubric
        "rubric/mean",            # multiturn rubric
        "solution/rubric_mean",   # think_solution
    ):
        if key in last:
            return f"{last[key]:.4f}"
    return "N/A"


def extract_batch_count(metrics: list[dict]) -> int:
    """Count batches from metrics."""
    return len(metrics)


def has_eval_files(dirpath: Path) -> bool:
    """Check if a directory has evaluation result files."""
    for f in dirpath.iterdir():
        if f.is_file():
            name = f.name.lower()
            if ("eval" in name and name.endswith((".json", ".jsonl"))) or \
               name.startswith("eval_") or \
               "summary" in name or "aggregate" in name:
                return True
    # Also check immediate subdirs named eval/evaluation
    for subdir in ("eval", "evaluation"):
        sub = dirpath / subdir
        if sub.is_dir():
            return True
    return False


def determine_status(batch_count: int, dirpath: Path) -> str:
    """Determine run status based on batch count and contents."""
    if batch_count > 50:
        return "Completed"
    elif batch_count > 0:
        return "Short run"
    else:
        # No metrics -- check for eval files
        if has_eval_files(dirpath):
            return "Evaluation-only"
        # Check for logs.log with content
        logfile = dirpath / "logs.log"
        if logfile.exists() and logfile.stat().st_size > 0:
            return "Evaluation-only"
        return "Failed/Empty"


# ---------------------------------------------------------------------------
# Discovery: find all run directories
# ---------------------------------------------------------------------------

def find_run_dirs(root: Path) -> list[Path]:
    """Find all directories that contain run artifacts."""
    run_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath)
        filenames_set = set(filenames)
        has_config = "config.json" in filenames_set
        has_metrics = "metrics.jsonl" in filenames_set
        has_logs = "logs.log" in filenames_set
        has_summary_json = any(
            f.endswith(".json") and ("summary" in f or "aggregate" in f or "eval_result" in f)
            for f in filenames_set
        )
        if has_config or has_metrics or has_logs:
            run_dirs.append(dirpath)
        elif has_summary_json:
            # Eval result directories without config/metrics/logs
            run_dirs.append(dirpath)
    return sorted(run_dirs)


# ---------------------------------------------------------------------------
# Per-run summary extraction
# ---------------------------------------------------------------------------

def extract_run_info(dirpath: Path, root: Path) -> dict:
    """Extract all info for a single run directory."""
    rel = str(dirpath.relative_to(root))

    config_path = dirpath / "config.json"
    metrics_path = dirpath / "metrics.jsonl"

    config = read_json(config_path) if config_path.exists() else None
    metrics = read_metrics(metrics_path) if metrics_path.exists() else []

    # Infer parent dir name for method inference
    # Use the first meaningful directory name after month
    parts = rel.split("/")
    parent_dir = parts[1] if len(parts) > 1 else "unknown"

    batch_count = extract_batch_count(metrics)
    status = determine_status(batch_count, dirpath)
    model = extract_model(config)
    dataset = extract_dataset(config)
    method = extract_method(config, parent_dir)
    hyperparams = extract_hyperparams(config)
    final_reward = extract_final_reward(metrics)
    family = classify_family(rel)

    return {
        "rel_path": rel,
        "dirpath": dirpath,
        "method": method,
        "model": model,
        "dataset": dataset,
        "batches": batch_count,
        "final_reward": final_reward,
        "status": status,
        "hyperparams": hyperparams,
        "family": family,
        "has_config": config_path.exists(),
        "has_metrics": metrics_path.exists(),
    }


# ---------------------------------------------------------------------------
# Write summary.md for a single run
# ---------------------------------------------------------------------------

def write_summary(info: dict):
    """Write summary.md into the run directory."""
    summary_path = info["dirpath"] / "summary.md"
    content = f"""# Run: {info['rel_path']}

- **Method**: {info['method']}
- **Model**: {info['model']}
- **Dataset**: {info['dataset']}
- **Batches**: {info['batches']}
- **Final Reward**: {info['final_reward']}
- **Status**: {info['status']}
- **Config**: {info['hyperparams']}
"""
    with open(summary_path, "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Write master catalog
# ---------------------------------------------------------------------------

def generate_notes(info: dict) -> str:
    """Generate brief notes for the catalog table."""
    notes = []
    rel = info["rel_path"]
    name = rel.split("/")[-1] if "/" in rel else rel

    if "sanity" in name.lower() or "sanity check" in name.lower():
        notes.append("sanity check")
    if "test" in name.lower():
        notes.append("test run")
    if "smoke" in name.lower():
        notes.append("smoke test")
    if "recovery" in name.lower():
        notes.append("recovery")
    if "warmstart" in name.lower():
        notes.append("warmstart")
    if "baseline" in name.lower() and info["family"] == "IBT":
        notes.append("baseline comparison")
    if "opd" in name.lower():
        notes.append("OPD variant")
    if "nohint" in name.lower():
        notes.append("no-hint ablation")
    if info["model"] and "gpt" in info["model"].lower():
        notes.append(info["model"].split("/")[-1])

    return "; ".join(notes) if notes else ""


def write_catalog(all_runs: list[dict]):
    """Write the master RUNS_CATALOG.md."""
    # Group by family
    families = defaultdict(list)
    for run in all_runs:
        families[run["family"]].append(run)

    total = len(all_runs)
    months = sorted(set(r["rel_path"].split("/")[0] for r in all_runs))
    today = datetime.now().strftime("%Y-%m-%d")

    lines = []
    lines.append("# Runs Catalog\n")
    lines.append(f"*Auto-generated on {today}. {total} runs across {len(months)} months.*\n")
    lines.append("")
    lines.append("## How to Read This Catalog")
    lines.append("- **Run path**: relative to `runs/2026/`")
    lines.append("- **Batches**: number of training batches completed (lines in metrics.jsonl)")
    lines.append("- **Reward**: final reward metric (higher is better)")
    lines.append("- **Status**: Completed (>50 batches), Short (<50), Failed (0), Eval-only")
    lines.append("")

    # Define display order
    family_order = [
        "GRPO / Bestversion",
        "SDPO",
        "Reward Shaping",
        "Refinement",
        "Multi-turn",
        "IBT",
        "Other",
    ]

    family_descriptions = {
        "GRPO / Bestversion": "All runs from baseline_exploration, withA1,A2 (bestversion), and best_ver_async.",
        "SDPO": "All SDPO trainer runs including recovery and sanity checks.",
        "Reward Shaping": "Reward formula ablations: 0-9_scale, std+mean_weighted, std_stand+band_bonus, hard_min.",
        "Refinement": "Two-stage refinement (doublegeneration), blended training, and rubric dropout.",
        "Multi-turn": "Multi-turn pipeline variants V1 through V4.",
        "IBT": "Iterative Brainstorming Training runs across multiple rounds.",
        "Other": "Miscellaneous: think_solution, CPR, eval_*, gsm8k, dense_score.",
    }

    lines.append("## Method Families\n")

    for family in family_order:
        runs = families.get(family, [])
        if not runs:
            continue

        lines.append(f"### {family}")
        lines.append(f"*{family_descriptions.get(family, '')}*\n")
        lines.append(f"**{len(runs)} runs**\n")

        lines.append("| Run Path | Batches | Final Reward | Dataset | Status | Key Notes |")
        lines.append("|----------|---------|-------------|---------|--------|-----------|")

        # Sort runs by path
        for run in sorted(runs, key=lambda r: r["rel_path"]):
            notes = generate_notes(run)
            rp = run["rel_path"]
            lines.append(
                f"| `{rp}` | {run['batches']} | {run['final_reward']} | {run['dataset']} | {run['status']} | {notes} |"
            )

        lines.append("")

    # Summary stats
    lines.append("## Summary Statistics\n")
    completed = sum(1 for r in all_runs if r["status"] == "Completed")
    short = sum(1 for r in all_runs if r["status"] == "Short run")
    failed = sum(1 for r in all_runs if r["status"] == "Failed/Empty")
    eval_only = sum(1 for r in all_runs if r["status"] == "Evaluation-only")

    lines.append(f"| Status | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Completed (>50 batches) | {completed} |")
    lines.append(f"| Short run (1-50 batches) | {short} |")
    lines.append(f"| Evaluation-only | {eval_only} |")
    lines.append(f"| Failed/Empty | {failed} |")
    lines.append(f"| **Total** | **{total}** |")
    lines.append("")

    with open(CATALOG_PATH, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not RUNS_ROOT.exists():
        print(f"Error: {RUNS_ROOT} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {RUNS_ROOT} ...")
    run_dirs = find_run_dirs(RUNS_ROOT)
    print(f"Found {len(run_dirs)} run directories.")

    all_runs = []
    for dirpath in run_dirs:
        info = extract_run_info(dirpath, RUNS_ROOT)
        all_runs.append(info)
        write_summary(info)
        print(f"  summary.md -> {info['rel_path']}  [{info['status']}, {info['batches']} batches]")

    print(f"\nWriting catalog to {CATALOG_PATH} ...")
    write_catalog(all_runs)

    # Print family summary
    families = defaultdict(int)
    for r in all_runs:
        families[r["family"]] += 1
    print("\nFamily breakdown:")
    for fam, count in sorted(families.items(), key=lambda x: -x[1]):
        print(f"  {fam}: {count} runs")

    print(f"\nDone. {len(all_runs)} summaries written + 1 catalog.")


if __name__ == "__main__":
    main()
