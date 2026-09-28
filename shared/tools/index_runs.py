#!/usr/bin/env python3
"""
Index historical run folders into CSV and Markdown summaries.

Scans canonical run roots (default: runs/2026), tolerates inconsistent log layout,
and writes:
- runs/registry/runs_index.csv
- runs/registry/runs_index.md
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    run_id: str
    method: str
    round: str
    timestamp: str
    run_path: str
    config_ref: str
    metrics_source: str
    reward_sample_mean_valid: str
    rubric_sample_mean_valid: str
    valid_rate: str
    format_rate: str
    final_batch_idx: str
    use_sdpo: str
    policy_output_mode: str
    reward_mode: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return rows


def str_or_unknown(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def detect_method(rel_parts: tuple[str, ...], cfg: dict[str, Any]) -> str:
    joined = "/".join(rel_parts).lower()
    if cfg.get("use_sdpo") is True or "/sdpo/" in f"/{joined}/":
        return "sdpo"
    if "0-9_scale" in joined or "0-9" in joined:
        return "scale_0_9"
    if "hard_min" in joined:
        return "hard_min"
    if "std+mean_weighted" in joined:
        return "weighted"
    if "std_stand+band_bonus" in joined:
        return "std_band_bonus"
    if "witha1,a2" in joined:
        return "with_a1_a2"
    return "baseline_legacy"


def detect_round(rel_parts: tuple[str, ...]) -> str:
    leaf = rel_parts[-1] if rel_parts else "unknown"
    return leaf


def latest_metric_row(run_dir: Path) -> tuple[dict[str, Any], str]:
    candidates = [
        (run_dir / "metrics.jsonl", "metrics.jsonl"),
        (run_dir / "train" / "batch_summary.jsonl", "train/batch_summary.jsonl"),
        (run_dir / "batch_summary.jsonl", "batch_summary.jsonl"),
    ]
    for path, label in candidates:
        rows = read_jsonl(path)
        if rows:
            return rows[-1], label
    return {}, "unknown"


def build_run_id(rel_path: Path, method: str) -> str:
    slug = rel_path.as_posix().replace("/", "__")
    slug = slug.replace("(", "_").replace(")", "_").replace(",", "_").replace("+", "plus")
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug.strip("_")
    return f"{method}__{slug}"


def gather_records(repo_root: Path, runs_root: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    run_dirs = sorted({p.parent for p in runs_root.rglob("config.json")})

    for run_dir in run_dirs:
        rel = run_dir.relative_to(runs_root)
        rel_parts = rel.parts

        cfg_path = run_dir / "config.json"
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}

        metric_row, metric_source = latest_metric_row(run_dir)

        all_files = [f for f in run_dir.rglob("*") if f.is_file()]
        mtime = max((f.stat().st_mtime for f in all_files), default=0)
        timestamp = datetime.fromtimestamp(mtime).isoformat(timespec="seconds") if mtime else "unknown"

        method = detect_method(rel_parts, cfg)
        round_name = detect_round(rel_parts)
        run_id = build_run_id(rel, method)

        reward_val = metric_row.get("reward/sample_mean_valid", metric_row.get("reward/total"))
        rubric_val = metric_row.get("rubric/sample_mean_valid", metric_row.get("rubric/sample_mean_all"))
        valid_rate_val = metric_row.get("samples/trainability_pass_rate")
        format_rate_val = metric_row.get("format_rate", metric_row.get("format/penalty_mean"))
        batch_val = metric_row.get("batch_idx", metric_row.get("progress/batch", metric_row.get("step")))

        records.append(
            RunRecord(
                run_id=run_id,
                method=method,
                round=round_name,
                timestamp=timestamp,
                run_path=str(run_dir.relative_to(repo_root)),
                config_ref=str(cfg_path.relative_to(repo_root)),
                metrics_source=metric_source,
                reward_sample_mean_valid=str_or_unknown(reward_val),
                rubric_sample_mean_valid=str_or_unknown(rubric_val),
                valid_rate=str_or_unknown(valid_rate_val),
                format_rate=str_or_unknown(format_rate_val),
                final_batch_idx=str_or_unknown(batch_val),
                use_sdpo=str_or_unknown(cfg.get("use_sdpo")),
                policy_output_mode=str_or_unknown(cfg.get("policy_output_mode")),
                reward_mode=str_or_unknown(cfg.get("reward_mode")),
            )
        )

    records.sort(key=lambda r: (r.timestamp, r.method, r.run_id), reverse=True)
    return records


def write_csv(records: list[RunRecord], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(RunRecord.__annotations__.keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(r.__dict__)


def write_md(records: list[RunRecord], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Runs Index\n\n")
    lines.append(f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`\n\n")
    lines.append(f"Total indexed runs: **{len(records)}**\n\n")

    method_counts: dict[str, int] = {}
    for r in records:
        method_counts[r.method] = method_counts.get(r.method, 0) + 1

    lines.append("## Runs by Method\n")
    for method, count in sorted(method_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- `{method}`: {count}\n")
    lines.append("\n")

    lines.append("## Latest 20 Runs\n\n")
    lines.append("| timestamp | run_id | method | round | reward | rubric | valid_rate | format_rate | config_ref |\n")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---|\n")
    for r in records[:20]:
        lines.append(
            f"| {r.timestamp} | `{r.run_id}` | `{r.method}` | `{r.round}` | {r.reward_sample_mean_valid} | "
            f"{r.rubric_sample_mean_valid} | {r.valid_rate} | {r.format_rate} | `{r.config_ref}` |\n"
        )

    out_md.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index run folders into CSV and Markdown")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runs-root", type=Path, default=None, help="Default: <repo-root>/runs/2026")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    runs_root = args.runs_root.resolve() if args.runs_root else (repo_root / "runs" / "2026")

    records = gather_records(repo_root=repo_root, runs_root=runs_root)

    out_csv = repo_root / "runs" / "registry" / "runs_index.csv"
    out_md = repo_root / "runs" / "registry" / "runs_index.md"
    write_csv(records, out_csv)
    write_md(records, out_md)

    print(f"Indexed {len(records)} runs")
    print(f"CSV: {out_csv}")
    print(f"MD:  {out_md}")


if __name__ == "__main__":
    main()
