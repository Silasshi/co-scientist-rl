import argparse
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def find_row_at_or_before(rows: list[dict], batch_idx: int) -> dict | None:
    eligible = [r for r in rows if int(r.get("batch_idx", -1)) <= batch_idx]
    if not eligible:
        return None
    return max(eligible, key=lambda r: int(r.get("batch_idx", -1)))


def metric(row: dict | None, key: str) -> float | None:
    if row is None:
        return None
    value = row.get(key)
    if value is None:
        return None
    return float(value)


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def print_checkpoint(name: str, row: dict | None) -> None:
    print(f"\n[{name}]")
    if row is None:
        print("  missing")
        return
    print(f"  batch_idx: {int(row['batch_idx'])}")
    for key in [
        "reward/sample_mean_valid",
        "rubric/sample_mean_valid",
        "format_rate",
        "length_mean",
        "length/total_mean",
        "samples/valid",
    ]:
        print(f"  {key}: {fmt(metric(row, key))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SDPO baseline vs fix runs.")
    parser.add_argument("--baseline", required=True, help="Baseline batch_summary.jsonl path")
    parser.add_argument("--fix", required=True, help="Fix run batch_summary.jsonl path")
    args = parser.parse_args()

    baseline_rows = load_rows(Path(args.baseline))
    fix_rows = load_rows(Path(args.fix))

    checkpoints = {
        "batch20": 20,
        "batch50": 50,
        "final": 10**9,
    }

    base_cp = {k: find_row_at_or_before(baseline_rows, v) for k, v in checkpoints.items()}
    fix_cp = {k: find_row_at_or_before(fix_rows, v) for k, v in checkpoints.items()}

    print("=== Baseline ===")
    for name in ["batch20", "batch50", "final"]:
        print_checkpoint(name, base_cp[name])

    print("\n=== Fix ===")
    for name in ["batch20", "batch50", "final"]:
        print_checkpoint(name, fix_cp[name])

    print("\n=== Acceptance Checks (Fix vs Baseline) ===")
    # By batch 20
    fix20 = fix_cp["batch20"]
    base20 = base_cp["batch20"]
    fix20_format = metric(fix20, "format_rate")
    fix20_len = metric(fix20, "length_mean")
    fix20_reward = metric(fix20, "reward/sample_mean_valid")
    base20_reward = metric(base20, "reward/sample_mean_valid")
    c20_a = fix20_format is not None and fix20_format <= 0.20
    c20_b = fix20_len is not None and fix20_len <= 800.0
    c20_c = (
        fix20_reward is not None
        and base20_reward is not None
        and fix20_reward >= (base20_reward - 0.02)
    )
    print(f"batch20 format_rate<=0.20: {c20_a}")
    print(f"batch20 length_mean<=800: {c20_b}")
    print(f"batch20 reward not below baseline by >0.02: {c20_c}")

    # By batch 50
    fix50 = fix_cp["batch50"]
    base50 = base_cp["batch50"]
    fix50_format = metric(fix50, "format_rate")
    fix50_reward = metric(fix50, "reward/sample_mean_valid")
    base50_reward = metric(base50, "reward/sample_mean_valid")
    c50_a = fix50_format is not None and fix50_format <= 0.10
    c50_b = (
        fix50_reward is not None
        and base50_reward is not None
        and fix50_reward >= (base50_reward + 0.02)
    )
    print(f"batch50 format_rate<=0.10: {c50_a}")
    print(f"batch50 reward above baseline by >=0.02: {c50_b}")

    # Final default rule
    base_final_reward = metric(base_cp["final"], "reward/sample_mean_valid")
    fix_final_reward = metric(fix_cp["final"], "reward/sample_mean_valid")
    default_keep_solution_only = (
        base_final_reward is not None
        and fix_final_reward is not None
        and abs(fix_final_reward - base_final_reward) < 0.01
    )
    print(f"final |reward_fix - reward_base| < 0.01: {default_keep_solution_only}")
    if default_keep_solution_only:
        print("Default action: keep solution-only prompt for reliability/debuggability.")


if __name__ == "__main__":
    main()
