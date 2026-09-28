#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chz

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.trainers.grpo.best_ver import Config, main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run best_ver evaluation flow on a target run folder."
    )
    parser.add_argument(
        "--run-path",
        required=True,
        help="Absolute path to the run folder containing checkpoints.jsonl",
    )
    parser.add_argument(
        "--eval-epoch",
        type=int,
        default=-1,
        help="-1 for latest checkpoint, or positive index for specific checkpoint ordering",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    cfg = chz.replace(
        Config(),
        log_path=str(args.run_path),
        run_eval=True,
        eval_epoch=int(args.eval_epoch),
    )
    print(f"Running best_ver eval on {args.run_path} (eval_epoch={args.eval_epoch})")
    main(cfg)


if __name__ == "__main__":
    run()
