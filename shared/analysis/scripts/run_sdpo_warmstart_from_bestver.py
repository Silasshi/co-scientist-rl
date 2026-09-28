#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import chz

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.trainers.sdpo.train_sdpo import Config, main

from warmstart_bestver214_common import SDPO_RUN, prepare_warmstart_runs


def run() -> None:
    prep = prepare_warmstart_runs()
    print("Prepared warm-start runs:", prep)

    cfg = chz.replace(
        Config(),
        log_path=str(SDPO_RUN),
        run_eval=False,
        max_batches=107,
        save_every=20,
        max_tokens=1536,
        format_retry_enabled=False,
        format_retry_max_attempts=1,
        format_retry_temperature=0.3,
        format_retry_max_tokens=1024,
    )
    print(f"Launching SDPO warm-start continuation at: {SDPO_RUN}")
    asyncio.run(main(cfg))


if __name__ == "__main__":
    run()
