"""Small helpers shared by the per-method analysis scripts (figures + number checks)."""
from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#2f6db5", "#d1495b", "#3a9d5d", "#e0a526", "#7a5195", "#6c757d"]


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 160, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    })


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def eval_mean_all(summary_csv: Path) -> tuple[float, int, int]:
    """Headline eval score: rubric mean over ALL generated samples (invalid ones included).

    Returns (mean, n_generated, n_valid) from an eval batch-summary CSV.
    """
    s = pd.read_csv(summary_csv)
    w = s["samples/generated"]
    return float((s["rubric/sample_mean_all"] * w).sum() / w.sum()), int(w.sum()), int(s["samples/valid"].sum())


def expected_max(scores: np.ndarray, k: int) -> float:
    """Exact E[max of k draws without replacement] from n observed scores (combinatorial estimator)."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    return float(sum(s[i] * comb(i, k - 1) for i in range(n)) / comb(n, k))


class Checker:
    """Collects (name, computed, expected, tol) checks, prints a table, exits non-zero on any mismatch."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, float, float, float, bool]] = []

    def check(self, name: str, computed: float, expected: float, tol: float = 5e-4) -> float:
        self.rows.append((name, computed, expected, tol, abs(computed - expected) <= tol))
        return computed

    def report(self) -> None:
        w = max(len(r[0]) for r in self.rows)
        for name, c, e, _, ok in self.rows:
            print(f"{'OK ' if ok else 'BAD'} {name:<{w}}  computed={c:.4f}  readme={e:.4f}")
        bad = [r for r in self.rows if not r[4]]
        print(f"{len(self.rows) - len(bad)}/{len(self.rows)} checks pass")
        if bad:
            sys.exit(1)
