"""
Rubric buffer for evolving R_active in GER-CR-v1.

Holds a bounded buffer of binary-graded rubric items generated online during
training (à la DR Tulu RLER Algorithm 1 and OnlineRubrics). Each item carries a
title, a description, a polarity (positive/negative), and a running history of
per-rollout grades. Two operations are supported per training iter:

    1. add(items)                 : merge newly-elicited rubrics into the buffer
    2. record_grades(id, grades)  : append a per-rollout grade vector for an item
    3. filter_and_truncate()      : drop zero-variance items, keep top K_max by std

Scoring uses the DR Tulu weighted-sum formula (Eq. 1):
    score(y) = sum_k (w_k * g_k(y)) / sum_k (w_k where w_k > 0)

Positive rubrics: higher grade = better (satisfied)
Negative rubrics: higher grade = better (successfully avoided the failure pattern).
Negative items get a multiplicative weight boost (default 1.5×) to counter the
RaR finding that soft "pitfall" criteria had 0 effect (Gunjal et al. 2025
Table 2).

This module is self-contained (numpy only) so it can be unit-tested independent
of the trainer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


POSITIVE = "positive"
NEGATIVE = "negative"


def _stable_id(title: str, description: str) -> str:
    """Deterministic short id from the rubric content so duplicates collapse."""
    key = f"{title.strip().lower()}||{description.strip().lower()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


@dataclass
class RubricItem:
    title: str
    description: str
    rubric_type: str  # POSITIVE | NEGATIVE
    weight: float = 1.0
    created_at_iter: int = 0
    # grades[iter_index] = list of per-rollout grades (each 0.0 or 1.0).
    grades: list[list[float]] = field(default_factory=list)
    id: str = ""

    def __post_init__(self) -> None:
        if self.rubric_type not in (POSITIVE, NEGATIVE):
            raise ValueError(
                f"rubric_type must be {POSITIVE!r} or {NEGATIVE!r}, got {self.rubric_type!r}"
            )
        if not self.id:
            self.id = _stable_id(self.title, self.description)

    @property
    def latest_std(self) -> float:
        """Std of the most recent iteration's grades (0.0 if <2 samples)."""
        if not self.grades or len(self.grades[-1]) < 2:
            return 0.0
        return float(np.std(self.grades[-1], ddof=0))

    @property
    def latest_mean(self) -> float:
        if not self.grades or not self.grades[-1]:
            return 0.0
        return float(np.mean(self.grades[-1]))

    @property
    def n_iters_scored(self) -> int:
        return len(self.grades)


class RubricBuffer:
    """
    Bounded evolving rubric buffer.

    Parameters
    ----------
    k_max:
        Max number of rubrics retained after truncation. DR Tulu uses 5 in their
        main run; we default to 12 because our CR pipeline generates more signal
        per iter and our aggregate combines with a 10-item R_persist.
    min_variance:
        Threshold below which a rubric is considered a "dead channel" and
        dropped at filter time (once it has been scored at least `min_scored_iters`
        times so new items get a chance to accumulate signal).
    negative_weight_multiplier:
        Extra weight given to negative rubrics to ensure they bite. Addresses
        RaR's finding that soft pitfalls had no measurable effect.
    min_scored_iters:
        Grace period. Items with fewer scored iters than this are never dropped
        regardless of std.
    """

    def __init__(
        self,
        k_max: int = 12,
        min_variance: float = 0.05,
        negative_weight_multiplier: float = 1.5,
        min_scored_iters: int = 1,
    ) -> None:
        self.k_max = k_max
        self.min_variance = min_variance
        self.negative_weight_multiplier = negative_weight_multiplier
        self.min_scored_iters = min_scored_iters
        self.items: dict[str, RubricItem] = {}

    # ------------------------------------------------------------------ core ops

    def add(self, new_items: Iterable[RubricItem]) -> list[str]:
        """Merge new rubrics. Returns ids of items actually added (skips dups)."""
        added: list[str] = []
        for it in new_items:
            if it.id in self.items:
                continue
            if it.rubric_type == NEGATIVE:
                it.weight = it.weight * self.negative_weight_multiplier
            self.items[it.id] = it
            added.append(it.id)
        return added

    def record_grades(self, rubric_id: str, grades_per_rollout: list[float]) -> None:
        """Append one iteration's per-rollout grades (binary 0/1) for an item."""
        if rubric_id not in self.items:
            return
        clean = [float(x) for x in grades_per_rollout]
        for g in clean:
            if g not in (0.0, 1.0):
                raise ValueError(f"grade must be 0.0 or 1.0, got {g}")
        self.items[rubric_id].grades.append(clean)

    def filter_and_truncate(self) -> tuple[list[str], list[str]]:
        """
        Run DR Tulu-style buffer management: drop low-variance items, keep top-K_max.

        Returns (dropped_ids, retained_ids).
        """
        # Stage 1: drop low-variance items that have passed the grace period.
        dropped: list[str] = []
        keep_after_filter: dict[str, RubricItem] = {}
        for rid, it in self.items.items():
            if it.n_iters_scored >= self.min_scored_iters and it.latest_std < self.min_variance:
                dropped.append(rid)
                continue
            keep_after_filter[rid] = it

        # Stage 2: cap at K_max by latest_std (ties broken by insertion order).
        if len(keep_after_filter) > self.k_max:
            ranked = sorted(
                keep_after_filter.values(),
                key=lambda x: (-x.latest_std, x.created_at_iter, x.id),
            )
            kept = {it.id: it for it in ranked[: self.k_max]}
            for rid in keep_after_filter:
                if rid not in kept:
                    dropped.append(rid)
            keep_after_filter = kept

        self.items = keep_after_filter
        return dropped, list(self.items.keys())

    # -------------------------------------------------------------- aggregation

    def aggregate_reward(self, rollout_grades: dict[str, float]) -> float:
        """
        DR Tulu Eq. 1 aggregation for one rollout's grades.

        rollout_grades: {rubric_id: 0.0 | 1.0}. Missing ids are treated as not-yet-graded
        and excluded from the sum (so the normalizer matches).

        Returns the weighted satisfaction rate in [0, 1]. Returns 0.5 (neutral) if
        buffer is empty, so that early training doesn't see a large step when the
        first rubric appears.
        """
        if not self.items:
            return 0.5
        num = 0.0
        den = 0.0
        for rid, g in rollout_grades.items():
            it = self.items.get(rid)
            if it is None:
                continue
            if it.weight <= 0:
                continue
            num += it.weight * float(g)
            den += it.weight
        if den <= 0:
            return 0.5
        return num / den

    # ------------------------------------------------------------- introspection

    def top_k(self, k: int | None = None) -> list[RubricItem]:
        n = k if k is not None else self.k_max
        return sorted(self.items.values(), key=lambda x: -x.latest_std)[:n]

    def stats(self) -> dict[str, float | int]:
        if not self.items:
            return {"n": 0, "n_positive": 0, "n_negative": 0, "mean_std": 0.0}
        stds = [it.latest_std for it in self.items.values()]
        return {
            "n": len(self.items),
            "n_positive": sum(1 for it in self.items.values() if it.rubric_type == POSITIVE),
            "n_negative": sum(1 for it in self.items.values() if it.rubric_type == NEGATIVE),
            "mean_std": float(np.mean(stds)) if stds else 0.0,
            "min_std": float(np.min(stds)) if stds else 0.0,
            "max_std": float(np.max(stds)) if stds else 0.0,
        }

    # ------------------------------------------------------------- serialization

    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for it in self.items.values():
                f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        k_max: int = 12,
        min_variance: float = 0.05,
        negative_weight_multiplier: float = 1.5,
        min_scored_iters: int = 1,
    ) -> "RubricBuffer":
        buf = cls(
            k_max=k_max,
            min_variance=min_variance,
            negative_weight_multiplier=negative_weight_multiplier,
            min_scored_iters=min_scored_iters,
        )
        path = Path(path)
        if not path.exists():
            return buf
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                it = RubricItem(**data)
                buf.items[it.id] = it
        return buf


# ------------------------------------------------------------------ self-tests

def _run_sanity_checks() -> None:
    """Basic invariants. Run with `python -m co_scientist.shared.rubric_buffer`."""

    buf = RubricBuffer(k_max=3, min_variance=0.05, min_scored_iters=1)
    assert buf.aggregate_reward({}) == 0.5, "empty buffer should be neutral"

    a = RubricItem(title="A", description="positive item a", rubric_type=POSITIVE, created_at_iter=0)
    b = RubricItem(title="B", description="positive item b", rubric_type=POSITIVE, created_at_iter=0)
    c = RubricItem(title="C", description="negative item c", rubric_type=NEGATIVE, created_at_iter=0)
    d_dup = RubricItem(title="A", description="positive item a", rubric_type=POSITIVE)

    added = buf.add([a, b, c, d_dup])
    assert set(added) == {a.id, b.id, c.id}, f"dup should be skipped, got {added}"
    assert buf.items[c.id].weight == pytest_approx(1.5), "negative should get 1.5x weight"

    # iter 1 grades: A varies, B constant (dead), C varies
    buf.record_grades(a.id, [1.0, 0.0, 1.0, 0.0])
    buf.record_grades(b.id, [1.0, 1.0, 1.0, 1.0])
    buf.record_grades(c.id, [0.0, 1.0, 0.0, 1.0])

    dropped, retained = buf.filter_and_truncate()
    assert b.id in dropped, f"constant-grade B should be filtered, dropped={dropped}"
    assert a.id in retained and c.id in retained, f"A and C should survive, retained={retained}"

    # aggregate_reward: one rollout with A=1, C=1 (satisfied)
    # weights: A=1.0, C=1.5 → num=1.0*1 + 1.5*1 = 2.5, den=2.5 → 1.0
    r = buf.aggregate_reward({a.id: 1.0, c.id: 1.0})
    assert r == pytest_approx(1.0)

    r2 = buf.aggregate_reward({a.id: 0.0, c.id: 1.0})
    # num = 0 + 1.5 = 1.5, den = 2.5 → 0.6
    assert r2 == pytest_approx(0.6)

    # truncation: add 3 more items to exceed k_max=3
    buf.add([
        RubricItem(title="D", description="dd", rubric_type=POSITIVE, created_at_iter=1),
        RubricItem(title="E", description="ee", rubric_type=POSITIVE, created_at_iter=1),
    ])
    # B was dropped, so now A, C, D, E → 4 items, k_max=3
    buf.record_grades(buf.items[list(buf.items.keys())[-1]].id, [1.0, 0.0])
    buf.record_grades(buf.items[list(buf.items.keys())[-2]].id, [1.0, 1.0])
    # After another filter: E (std=0.5) and A (iter 1 std=0.5) should beat D (std=0)
    buf.filter_and_truncate()
    assert len(buf.items) <= 3, f"k_max cap violated, n={len(buf.items)}"

    # Round-trip jsonl
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        tmp = tf.name
    buf.to_jsonl(tmp)
    buf2 = RubricBuffer.from_jsonl(tmp, k_max=3)
    assert set(buf2.items.keys()) == set(buf.items.keys()), "round-trip mismatch"
    for rid in buf.items:
        assert buf2.items[rid].weight == buf.items[rid].weight
        assert buf2.items[rid].rubric_type == buf.items[rid].rubric_type

    print("rubric_buffer self-tests passed.")
    print("stats:", buf.stats())


def pytest_approx(v: float, tol: float = 1e-9):
    class _Approx:
        def __eq__(self, other):
            return abs(other - v) < tol
        def __repr__(self):
            return f"~{v}"
    return _Approx()


if __name__ == "__main__":
    _run_sanity_checks()
