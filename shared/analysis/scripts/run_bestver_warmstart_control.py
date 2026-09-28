#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import chz

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.trainers.grpo.best_ver import Config, main

from warmstart_bestver214_common import CONTROL_RUN, prepare_warmstart_runs


def run() -> None:
    prep = prepare_warmstart_runs()
    print("Prepared warm-start runs:", prep)

    cfg = chz.replace(
        Config(),
        log_path=str(CONTROL_RUN),
        run_eval=False,
        save_every=10,
    )
    print(f"Launching best_ver control continuation at: {CONTROL_RUN}")
    main(cfg)


if __name__ == "__main__":
    run()
