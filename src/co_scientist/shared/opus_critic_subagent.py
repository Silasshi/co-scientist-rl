"""Plan-critic file-bus client for D5 μ baseline.

Trainer-side shim that ships a (goal, plan, source_paper_md) bundle to the
combined critic+audit Opus daemon (see
projects/d5_abstract_retrieve_refine/scripts/opus_critic_audit_daemon.md),
then blocks polling for the response.

Distinct from `opus_subagent_grader.py`: that one routes a 17-prompt-per-plan
signal+gate batch; this one routes a single critique request per plan.

Protocol:
    request:  <log_path>/critic_requests/iter_NNN.json
    response: <log_path>/critic_responses/iter_NNN.json

Trainer integration:
    critic = OpusCriticClient(log_path=cfg.log_path)
    critic.submit(iter_idx=i, payload=...)
    critique_xml = critic.wait(iter_idx=i, timeout_sec=900)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OpusCriticClient:
    """File-bus client for the plan-critic Opus subagent.

    Synchronous API: submit() then wait(). The daemon is assumed running in a
    separate Claude Code subagent window.
    """

    def __init__(
        self,
        log_path: str | Path,
        timeout_sec: float = 900.0,
        poll_interval_sec: float = 5.0,
    ):
        self.log_path = Path(log_path)
        self.timeout_sec = timeout_sec
        self.poll_interval_sec = poll_interval_sec
        self.req_dir = self.log_path / "critic_requests"
        self.resp_dir = self.log_path / "critic_responses"
        self.req_dir.mkdir(parents=True, exist_ok=True)
        self.resp_dir.mkdir(parents=True, exist_ok=True)

    def _req_path(self, iter_idx: int) -> Path:
        return self.req_dir / f"iter_{iter_idx:03d}.json"

    def _resp_path(self, iter_idx: int) -> Path:
        return self.resp_dir / f"iter_{iter_idx:03d}.json"

    def submit(self, iter_idx: int, payload: dict[str, Any]) -> Path:
        """Atomically write critic request JSON. Returns path."""
        req_path = self._req_path(iter_idx)
        tmp = req_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(req_path)
        n_chars = sum(len(p["text"]) for p in payload.get("plans", []))
        logger.info(
            "OpusCriticClient: submitted iter=%d (%d plans, %d total plan chars) → %s",
            iter_idx, len(payload.get("plans", [])), n_chars, req_path,
        )
        return req_path

    def wait(self, iter_idx: int, timeout_sec: float | None = None) -> dict[str, Any]:
        """Block on response file; return parsed JSON dict."""
        resp_path = self._resp_path(iter_idx)
        deadline = time.time() + (timeout_sec if timeout_sec is not None else self.timeout_sec)
        t0 = time.time()
        while not resp_path.exists():
            if time.time() > deadline:
                raise TimeoutError(
                    f"OpusCriticClient: timeout waiting for {resp_path} after "
                    f"{deadline - t0:.0f}s. Check daemon at {self.log_path}/critic_audit_daemon.log"
                )
            time.sleep(self.poll_interval_sec)
        latency = time.time() - t0
        raw = resp_path.read_text()
        try:
            resp = json.loads(raw)
        except json.JSONDecodeError as e:
            # FALLBACK: regex-extract <critique>...</critique> blocks from raw text
            # to recover from daemon's occasional quote-escape bugs in critique_xml.
            import re as _re
            critique_blocks = _re.findall(r"<critique>.*?</critique>", raw, _re.DOTALL)
            if not critique_blocks:
                raise RuntimeError(
                    f"OpusCriticClient: failed to parse {resp_path} AND no <critique> "
                    f"fallback block found: {e}"
                ) from e
            # Try to also recover iter and pick_id from raw text
            iter_m = _re.search(r'"iter"\s*:\s*(\d+)', raw)
            iter_recovered = int(iter_m.group(1)) if iter_m else iter_idx
            pick_m = _re.search(r'"(iter_\d+_pick_\d+)"\s*:', raw)
            pick_id = pick_m.group(1) if pick_m else f"iter_{iter_idx:03d}_pick_0"
            logger.warning(
                "OpusCriticClient: iter=%d malformed JSON; recovered %d critique block(s) via regex fallback "
                "(pick_id=%s). Re-encoding response file.",
                iter_idx, len(critique_blocks), pick_id,
            )
            # Use first <critique> block (one per plan in our setup)
            resp = {"iter": iter_recovered, "judgments": {pick_id: {"critique_xml": critique_blocks[0]}}}
            # Save the cleaned version atomically so subsequent reads succeed
            tmp = resp_path.with_suffix(".repaired.json")
            tmp.write_text(json.dumps(resp, ensure_ascii=False, indent=2))
            tmp.replace(resp_path)
        logger.info(
            "OpusCriticClient: iter=%d response ready in %.1fs", iter_idx, latency,
        )
        return resp

    def critique_for(
        self, response: dict[str, Any], plan_id: str, fallback: str
    ) -> str:
        """Extract per-plan critique XML from response dict; fallback if missing."""
        judgments = response.get("judgments", {})
        j = judgments.get(plan_id, {})
        critique_xml = j.get("critique_xml") or j.get("critique") or ""
        if not critique_xml:
            logger.warning(
                "OpusCriticClient: no critique_xml for plan_id=%s in response; using fallback",
                plan_id,
            )
            return fallback
        return critique_xml.strip()
