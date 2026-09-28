"""Pairwise-preference file-bus client for D5 Phase 0.6 sanity check.

Trainer/orchestrator-side shim: writes pairwise request JSON, polls for
matching response JSON. Daemon (Claude Code subagent) processes by fanning
out one Opus Task per pair → returns winner + rationale.

Protocol:
    request:  <log_path>/pairwise_requests/<matchup_id>.json
    response: <log_path>/pairwise_responses/<matchup_id>.json

Request JSON shape:
{
  "kind": "pairwise",
  "matchup_id": "mu_vs_delta",
  "goal": "<research goal>",
  "pairs": [
    {"pair_id": "...", "plan_a_id": "...", "plan_a_text": "...",
     "plan_b_id": "...", "plan_b_text": "..."}, ...
  ]
}

Response JSON shape:
{
  "matchup_id": "mu_vs_delta",
  "kind": "pairwise",
  "completed_at": "<ISO-8601>",
  "verdicts": {
    "<pair_id>": {"winner": "A" | "B" | "TIE", "rationale": "..."},
    ...
  }
}

Note: the daemon ALWAYS labels winner as A or B (the position shown), NOT
the underlying baseline label. The orchestrator (pairwise_prefs_v1.py) keeps
the position_swapped flag in matchups_meta.json and decodes back.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OpusPairwiseClient:
    """File-bus client for the pairwise-preference Opus subagent."""

    def __init__(
        self,
        log_path: str | Path,
        timeout_sec: float = 1800.0,
        poll_interval_sec: float = 5.0,
    ):
        self.log_path = Path(log_path)
        self.timeout_sec = timeout_sec
        self.poll_interval_sec = poll_interval_sec
        self.req_dir = self.log_path / "pairwise_requests"
        self.resp_dir = self.log_path / "pairwise_responses"
        self.req_dir.mkdir(parents=True, exist_ok=True)
        self.resp_dir.mkdir(parents=True, exist_ok=True)

    def _req_path(self, matchup_id: str) -> Path:
        return self.req_dir / f"{matchup_id}.json"

    def _resp_path(self, matchup_id: str) -> Path:
        return self.resp_dir / f"{matchup_id}.json"

    def submit(self, matchup_id: str, payload: dict[str, Any]) -> Path:
        """Atomically write request JSON. Non-blocking."""
        req_path = self._req_path(matchup_id)
        tmp = req_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(req_path)
        logger.info(
            "OpusPairwiseClient: submitted %s (%d pairs) → %s",
            matchup_id, len(payload.get("pairs", [])), req_path,
        )
        return req_path

    def wait(self, matchup_id: str, timeout_sec: float | None = None) -> dict[str, Any]:
        """Block on response file."""
        resp_path = self._resp_path(matchup_id)
        deadline = time.time() + (timeout_sec if timeout_sec is not None else self.timeout_sec)
        t0 = time.time()
        while not resp_path.exists():
            if time.time() > deadline:
                raise TimeoutError(
                    f"OpusPairwiseClient: timeout waiting for {resp_path}"
                )
            time.sleep(self.poll_interval_sec)
        latency = time.time() - t0
        try:
            resp = json.loads(resp_path.read_text())
        except json.JSONDecodeError as e:
            raise RuntimeError(f"OpusPairwiseClient: parse error {resp_path}: {e}") from e
        logger.info("OpusPairwiseClient: %s ready in %.1fs", matchup_id, latency)
        return resp

    def collect_all(self, matchup_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Read all available pairwise responses (no waiting). Skips missing."""
        out: dict[str, dict[str, Any]] = {}
        for mid in matchup_ids:
            rp = self._resp_path(mid)
            if rp.exists():
                try:
                    out[mid] = json.loads(rp.read_text())
                except json.JSONDecodeError as e:
                    logger.warning("Pairwise %s parse error: %s", mid, e)
        return out
