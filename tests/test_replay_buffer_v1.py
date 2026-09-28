"""Unit tests for D5 plan-level replay buffer (`replay_buffer_v1.ReplayBuffer`).

Uses a mock tokenizer so the test runs without loading Qwen3-30B-A3B weights.
Real-tokenizer end-to-end smoke is deferred to the actual μ-v6_replay launch
(task-009 — RUN_CONFIRMATION + user-approved training run).

Covered cases:
- Empty buffer → sample() returns [].
- Iter filter respected (allowed_iters).
- Short-plan filter respected (min_n_tokens).
- sample(seed=X) is reproducible across calls with same seed.
- sample(n) returns min(n, len(buffer)) entries.
- Each returned entry has the expected schema.
- Load stats accounting matches.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.d5_abstract_retrieve_refine.replay_buffer_v1 import ReplayBuffer


class MockTokenizer:
    """Encodes text by treating each char as one token id (ord). Cheap, deterministic."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


def _write_buffer(path: Path, entries: list[dict]) -> None:
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _make_entry(iter_: int, plan_idx: int, n_chars: int = 200) -> dict:
    text = "x" * n_chars
    return {
        "iter": iter_,
        "plan_idx": plan_idx,
        "plan_text": text,
        "raw_tokens_text": text,
    }


def test_load_basic():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "buffer.jsonl"
        entries = [_make_entry(i, j) for i in range(5) for j in range(8)]
        _write_buffer(p, entries)
        buf = ReplayBuffer(p, MockTokenizer())
        assert len(buf) == 40
        assert buf.load_stats["n_seen"] == 40
        assert buf.load_stats["n_loaded"] == 40


def test_iter_filter():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "buffer.jsonl"
        entries = [_make_entry(i, j) for i in range(8) for j in range(8)]
        _write_buffer(p, entries)
        buf = ReplayBuffer(p, MockTokenizer(), allowed_iters=[0, 1, 2, 3, 4])
        assert len(buf) == 40  # 5 iters × 8 plans
        for e in buf.entries:
            assert e["iter"] in {0, 1, 2, 3, 4}
        assert buf.load_stats["n_skipped_iter"] == 24


def test_short_plan_filter():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "buffer.jsonl"
        entries = [
            _make_entry(0, 0, n_chars=200),
            _make_entry(0, 1, n_chars=10),  # too short, expect dropped at min_n_tokens=50
            _make_entry(0, 2, n_chars=300),
        ]
        _write_buffer(p, entries)
        buf = ReplayBuffer(p, MockTokenizer(), min_n_tokens=50)
        assert len(buf) == 2
        assert buf.load_stats["n_skipped_short"] == 1


def test_empty_raw_text_skipped():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "buffer.jsonl"
        entries = [
            {"iter": 0, "plan_idx": 0, "plan_text": "x" * 200, "raw_tokens_text": ""},
            _make_entry(0, 1, n_chars=200),
        ]
        _write_buffer(p, entries)
        buf = ReplayBuffer(p, MockTokenizer())
        assert len(buf) == 1
        assert buf.load_stats["n_skipped_empty"] >= 1


def test_sample_reproducibility():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "buffer.jsonl"
        entries = [_make_entry(i, j) for i in range(5) for j in range(8)]
        _write_buffer(p, entries)
        buf = ReplayBuffer(p, MockTokenizer())
        s1 = buf.sample(4, seed=42)
        s2 = buf.sample(4, seed=42)
        assert [(e["iter"], e["plan_idx"]) for e in s1] == [
            (e["iter"], e["plan_idx"]) for e in s2
        ]


def test_sample_different_seeds_differ():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "buffer.jsonl"
        entries = [_make_entry(i, j) for i in range(5) for j in range(8)]
        _write_buffer(p, entries)
        buf = ReplayBuffer(p, MockTokenizer())
        s1 = buf.sample(4, seed=42)
        s2 = buf.sample(4, seed=43)
        assert [(e["iter"], e["plan_idx"]) for e in s1] != [
            (e["iter"], e["plan_idx"]) for e in s2
        ]


def test_sample_n_capped_at_buffer_size():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "buffer.jsonl"
        entries = [_make_entry(0, j) for j in range(3)]
        _write_buffer(p, entries)
        buf = ReplayBuffer(p, MockTokenizer())
        s = buf.sample(10, seed=1)
        assert len(s) == 3


def test_sample_empty_buffer_returns_empty():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "buffer.jsonl"
        _write_buffer(p, [])
        buf = ReplayBuffer(p, MockTokenizer())
        assert buf.sample(4, seed=1) == []


def test_returned_entry_schema():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "buffer.jsonl"
        entries = [_make_entry(0, 0, n_chars=100)]
        _write_buffer(p, entries)
        buf = ReplayBuffer(p, MockTokenizer())
        s = buf.sample(1, seed=1)
        assert len(s) == 1
        e = s[0]
        for key in ("iter", "plan_idx", "plan_text", "raw_tokens_text", "gen_tokens", "n_gen_tokens"):
            assert key in e, f"missing field {key}"
        assert isinstance(e["gen_tokens"], list)
        assert e["n_gen_tokens"] == len(e["gen_tokens"])


def test_missing_buffer_path_raises():
    with pytest.raises(FileNotFoundError):
        ReplayBuffer("/nonexistent/path/buffer.jsonl", MockTokenizer())
