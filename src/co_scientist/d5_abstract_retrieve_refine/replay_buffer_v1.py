"""D5 plan-level replay buffer (D2 track, train_mu_v6_replay).

Loads historical plans from a μ-v* buffer.jsonl and exposes random sampling
for use in mixed-batch SDPO training. Per the D4 plan
(`/home/silas/.claude/plans/snug-dreaming-yao.md`), this is used to anchor
continual SDPO from a μ-v4 iter 2 LoRA-state to a new paper goal (e.g. Tool-V),
mixing 4 fresh new-paper plans + 4 replay μ-v4 plans per iter.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable


class ReplayBuffer:
    """Plan-level replay buffer over a μ-v* buffer.jsonl.

    Each entry returned by sample() is a dict containing:
      - iter: int, plan_idx: int (provenance from source run)
      - plan_text: str (cleaned solution body, post extract_solution)
      - raw_tokens_text: str (full decoded teacher-generated tokens, incl. <think>)
      - gen_tokens: list[int] (re-tokenized raw_tokens_text via passed tokenizer;
        used directly as target tokens for SDPO IS-loss replay step)
      - n_gen_tokens: int

    Sampling is uniform without replacement per call; cross-call independent.
    """

    def __init__(
        self,
        buffer_path: str | Path,
        tokenizer,
        allowed_iters: Iterable[int] | None = None,
        min_n_tokens: int = 50,
    ) -> None:
        self.buffer_path = Path(buffer_path)
        if not self.buffer_path.exists():
            raise FileNotFoundError(f"Replay buffer source not found: {self.buffer_path}")
        self.tokenizer = tokenizer
        allowed = set(allowed_iters) if allowed_iters is not None else None

        entries: list[dict] = []
        n_seen = 0
        n_skipped_iter = 0
        n_skipped_short = 0
        n_skipped_empty = 0
        with open(self.buffer_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_seen += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    n_skipped_empty += 1
                    continue
                if allowed is not None and rec.get("iter") not in allowed:
                    n_skipped_iter += 1
                    continue
                raw_text = rec.get("raw_tokens_text") or ""
                if not raw_text:
                    n_skipped_empty += 1
                    continue
                # add_special_tokens=False: the original seq.tokens did not include BOS.
                try:
                    gen_tokens = self.tokenizer.encode(raw_text, add_special_tokens=False)
                except TypeError:
                    gen_tokens = self.tokenizer.encode(raw_text)
                if len(gen_tokens) < min_n_tokens:
                    n_skipped_short += 1
                    continue
                entries.append({
                    "iter": rec.get("iter"),
                    "plan_idx": rec.get("plan_idx"),
                    "plan_text": rec.get("plan_text", ""),
                    "raw_tokens_text": raw_text,
                    "gen_tokens": gen_tokens,
                    "n_gen_tokens": len(gen_tokens),
                })
        self.entries = entries
        self.load_stats = {
            "n_seen": n_seen,
            "n_loaded": len(entries),
            "n_skipped_iter": n_skipped_iter,
            "n_skipped_short": n_skipped_short,
            "n_skipped_empty": n_skipped_empty,
        }

    def __len__(self) -> int:
        return len(self.entries)

    def sample(self, n: int, seed: int) -> list[dict]:
        """Sample n entries uniformly without replacement, reproducible per seed."""
        if not self.entries:
            return []
        n = min(n, len(self.entries))
        rng = random.Random(seed)
        return rng.sample(self.entries, n)
