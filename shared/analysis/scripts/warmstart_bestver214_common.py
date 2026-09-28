#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_RUN = PROJECT_ROOT / "runs/2026/2/withA1,A2/2(ml)"
SOURCE_BATCH = 214

CONTROL_RUN = PROJECT_ROOT / "runs/2026/2/SDPO/28_recovery_control_bestver214"
SDPO_RUN = PROJECT_ROOT / "runs/2026/2/SDPO/28_recovery_A_warmstart_best214"


def _load_checkpoint_entry(source_run: Path, batch: int) -> dict:
    checkpoint_file = source_run / "checkpoints.jsonl"
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Missing checkpoints file: {checkpoint_file}")

    picked: dict | None = None
    with checkpoint_file.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            if int(row.get("batch", -1)) == int(batch) and "state_path" in row:
                picked = row

    if picked is None:
        raise ValueError(
            f"No checkpoint with batch={batch} and state_path found in {checkpoint_file}"
        )
    return picked


def _prepare_run_dir(
    run_dir: Path,
    ckpt_row: dict,
    *,
    source_run: Path,
    source_batch: int,
) -> str:
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_path = run_dir / "checkpoints.jsonl"
    seeded = False
    if not checkpoints_path.exists() or checkpoints_path.stat().st_size == 0:
        checkpoints_path.write_text(json.dumps(ckpt_row) + "\n", encoding="utf-8")
        seeded = True

    meta = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(source_run),
        "source_batch": int(source_batch),
        "source_checkpoint_name": ckpt_row.get("name"),
        "source_state_path": ckpt_row.get("state_path"),
        "source_sampler_path": ckpt_row.get("sampler_path"),
    }
    (run_dir / "warmstart_source.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    return "seeded" if seeded else "kept_existing_checkpoints"


def prepare_warmstart_runs() -> dict[str, str]:
    ckpt_row = _load_checkpoint_entry(SOURCE_RUN, SOURCE_BATCH)
    control_status = _prepare_run_dir(
        CONTROL_RUN,
        ckpt_row,
        source_run=SOURCE_RUN,
        source_batch=SOURCE_BATCH,
    )
    sdpo_status = _prepare_run_dir(
        SDPO_RUN,
        ckpt_row,
        source_run=SOURCE_RUN,
        source_batch=SOURCE_BATCH,
    )
    return {
        "source_run": str(SOURCE_RUN),
        "source_batch": str(SOURCE_BATCH),
        "control_run": str(CONTROL_RUN),
        "sdpo_run": str(SDPO_RUN),
        "control_status": control_status,
        "sdpo_status": sdpo_status,
    }


if __name__ == "__main__":
    info = prepare_warmstart_runs()
    print(json.dumps(info, indent=2))
