"""Load evolved rubric buffers into a tidy (item, iteration, pass-rate) table."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).resolve().parents[1] / "data"


def items(run: str) -> list[dict]:
    return [json.loads(line) for line in open(D / f"runs/{run}/rubric_buffer.jsonl")]


def pass_table(run: str) -> pd.DataFrame:
    """One row per (item, iteration): fraction of that iteration's rollouts that satisfy the item."""
    rows = []
    for it in items(run):
        for k, grades in enumerate(it["grades"]):
            rows.append({"title": it["title"], "type": it["rubric_type"], "created": it["created_at_iter"],
                         "iter": it["created_at_iter"] + k, "pass": float(np.mean(grades)),
                         "std": float(np.std(grades)), "n_rollouts": len(grades)})
    return pd.DataFrame(rows)
