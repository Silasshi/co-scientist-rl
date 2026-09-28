"""D3-canonical depth-audit file-bus client for D5 μ baseline.

Asynchronous fire-and-forget: trainer submits an audit request and continues
training. Audit results land in `<log_path>/audit_responses/iter_NNN.json`
for post-run review (no inline blocking).

Protocol:
    request:  <log_path>/audit_requests/iter_NNN.json
    response: <log_path>/audit_responses/iter_NNN.json

The daemon (combined critic+audit subagent) sees `kind="audit"` and uses
DEPTH_AUDIT_PROMPT (4 dims × 1-5 → /20). NO source paper as privileged info
— audit must remain independent of any paper-leakage signal that goes to the
critic.

Trainer integration:
    audit = OpusAuditClient(log_path=cfg.log_path)
    audit.submit(iter_idx=i, payload=...)
    # ... continue training ...
    # After run: audit.collect_all() reads audit_responses/*.json and
    # writes a flattened audit_log.jsonl for decision-matrix review.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OpusAuditClient:
    """Fire-and-forget file-bus client for the depth-audit Opus subagent."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.req_dir = self.log_path / "audit_requests"
        self.resp_dir = self.log_path / "audit_responses"
        self.req_dir.mkdir(parents=True, exist_ok=True)
        self.resp_dir.mkdir(parents=True, exist_ok=True)

    def _req_path(self, iter_idx: int) -> Path:
        return self.req_dir / f"iter_{iter_idx:03d}.json"

    def _resp_path(self, iter_idx: int) -> Path:
        return self.resp_dir / f"iter_{iter_idx:03d}.json"

    def submit(self, iter_idx: int, payload: dict[str, Any]) -> Path:
        """Atomically write audit request JSON. Non-blocking; trainer continues."""
        req_path = self._req_path(iter_idx)
        tmp = req_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(req_path)
        logger.info(
            "OpusAuditClient: submitted iter=%d (%d plans) → %s (async, no wait)",
            iter_idx, len(payload.get("plans", [])), req_path,
        )
        return req_path

    def collect_all(
        self,
        out_path: str | Path | None = None,
        timeout_sec: float = 1800.0,
        poll_interval_sec: float = 10.0,
    ) -> list[dict[str, Any]]:
        """Read all audit_responses/iter_*.json present, write flat jsonl summary.

        Waits up to `timeout_sec` for responses to all submitted iters.
        Returns a list of per-iter audit summary dicts:
            {"iter": int, "n_plans": int, "totals": [int, ...],
             "mean_total": float, "per_plan": [{...}], "per_dim_means": {...}}
        """
        # Pull the iter list from the request side (since responses arrive late)
        submitted_iters = sorted(
            int(p.stem.replace("iter_", ""))
            for p in self.req_dir.glob("iter_*.json")
        )
        if not submitted_iters:
            logger.info("OpusAuditClient.collect_all: no submitted audits; nothing to collect.")
            return []

        deadline = time.time() + timeout_sec
        seen: dict[int, dict[str, Any]] = {}

        while True:
            for it in submitted_iters:
                if it in seen:
                    continue
                rp = self._resp_path(it)
                if rp.exists():
                    try:
                        seen[it] = json.loads(rp.read_text())
                    except json.JSONDecodeError as e:
                        logger.warning("audit response %s parse error: %s", rp, e)
                        seen[it] = {"iter": it, "_parse_error": str(e)}
            if len(seen) == len(submitted_iters):
                break
            if time.time() > deadline:
                missing = [it for it in submitted_iters if it not in seen]
                logger.warning(
                    "OpusAuditClient.collect_all: timeout; missing iters %s", missing,
                )
                break
            time.sleep(poll_interval_sec)

        summary: list[dict[str, Any]] = []
        for it in submitted_iters:
            entry = seen.get(it)
            if entry is None or "_parse_error" in (entry or {}):
                summary.append({"iter": it, "status": "missing_or_unparsed"})
                continue
            judgments = entry.get("judgments", {})
            per_plan = []
            totals = []
            dim_sums = {"math": 0, "novelty": 0, "realism": 0, "rigor": 0}
            n_valid = 0
            for plan_id, j in judgments.items():
                # v2 prompt nests scores in per_dim_scores; v1 has them top-level.
                pds = j.get("per_dim_scores") or {}
                m = j.get("math") if "math" in j else pds.get("math")
                n = j.get("novelty") if "novelty" in j else pds.get("novelty")
                r = j.get("realism") if "realism" in j else pds.get("realism")
                g = j.get("rigor") if "rigor" in j else pds.get("rigor")
                tot = j.get("total")
                if all(x is not None for x in [m, n, r, g]):
                    if tot is None:
                        tot = m + n + r + g
                    totals.append(tot)
                    dim_sums["math"] += m
                    dim_sums["novelty"] += n
                    dim_sums["realism"] += r
                    dim_sums["rigor"] += g
                    n_valid += 1
                rec = {"plan_id": plan_id, "math": m, "novelty": n,
                       "realism": r, "rigor": g, "total": tot}
                # Surface v2-only fields for archival; lookup in audit_log.jsonl
                if "claim_list" in j:    rec["claim_list"] = j["claim_list"]
                if "scaffold_list" in j: rec["scaffold_list"] = j["scaffold_list"]
                if "per_dim_raw" in j:     rec["per_dim_raw"] = j["per_dim_raw"]
                if "per_dim_penalty" in j: rec["per_dim_penalty"] = j["per_dim_penalty"]
                per_plan.append(rec)
            mean_total = (sum(totals) / len(totals)) if totals else None
            per_dim_means = ({k: v / n_valid for k, v in dim_sums.items()} if n_valid else {})
            summary.append({
                "iter": it,
                "n_plans": len(per_plan),
                "n_valid": n_valid,
                "mean_total": mean_total,
                "totals": totals,
                "per_dim_means": per_dim_means,
                "per_plan": per_plan,
            })

        if out_path:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                for s in summary:
                    f.write(json.dumps(s) + "\n")
            logger.info("OpusAuditClient.collect_all: wrote %d entries to %s",
                         len(summary), out_path)
        return summary
