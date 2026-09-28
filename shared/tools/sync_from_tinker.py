#!/usr/bin/env python3
"""
Sync all co-scientist project files from tinker-cookbook into this repo.

This is non-destructive to the source tree. It copies files and verifies
that destination contains every source file path.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_SOURCE_PAIRS = [
    (
        Path("/home/silas/tinker-cookbook/tinker_cookbook/recipes/rl_co_scientist"),
        Path("project_files/rl_co_scientist"),
    ),
    (
        Path("/home/silas/tinker-cookbook/tinker_cookbook/recipes/logs_for_co_scientist/2026"),
        Path("project_files/logs_2026"),
    ),
    (
        Path("/home/silas/tinker-cookbook/syh_exercise"),
        Path("project_files/syh_exercise"),
    ),
]


def file_rel_set(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def sync_one(src: Path, dst: Path) -> tuple[int, int, int]:
    if not src.exists():
        raise FileNotFoundError(f"Source path not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)

    src_set = file_rel_set(src)
    dst_set = file_rel_set(dst)
    missing = src_set - dst_set
    extra = dst_set - src_set

    if missing:
        missing_preview = sorted(missing)[:10]
        raise RuntimeError(
            f"Verification failed for {src} -> {dst}. "
            f"Missing {len(missing)} files. Sample: {missing_preview}"
        )

    return (len(src_set), len(dst_set), len(extra))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync all co-scientist files from tinker-cookbook.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Destination project root (default: repo root).",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    print(f"Project root: {project_root}")

    for src, rel_dst in DEFAULT_SOURCE_PAIRS:
        dst = project_root / rel_dst
        src_count, dst_count, extra_count = sync_one(src, dst)
        print(f"[OK] {src} -> {dst}")
        print(f"     src={src_count}, dst={dst_count}, dst_extra={extra_count}")

    print("Sync complete.")


if __name__ == "__main__":
    main()
