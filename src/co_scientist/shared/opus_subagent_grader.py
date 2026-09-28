"""Opus subagent file-bus grader shim.

A tinker.SamplingClient-compatible shim that routes all grader .sample()
calls through a JSON file-bus to a Claude Code subagent daemon (see
projects/grant_proposal_v2/scripts/opus_grader_subagent.md).

Per iter the shim:
  1. Buffers every sample() call's prompt, tagged by (plan_id, kind, key).
  2. On first .result() call in the iter (lazy flush), writes one request
     file `<log_path>/grader_requests/iter_NNN.json` carrying the full
     rubric-rendered prompts for all buffered plans.
  3. Blocks polling `<log_path>/grader_responses/iter_NNN.json`.
  4. Decodes per-plan per-signal XML strings back into tokenizer ints so
     the trainer's existing `_decode_grader_output` → `parse_scores` →
     `collect_hard_gates` pipeline runs unchanged.

The daemon on the other side spawns one Opus Task subagent per plan,
fanning out 8 plans in parallel. Full protocol in the subagent.md spec.

Trainer integration (train_ger_cr_v1.py):

    if config.grader_backend == "opus_subagent":
        grader_client = OpusSubagentGrader(
            log_path=config.log_path,
            tokenizer=grader_tokenizer,
            long_critique=config.grader_long_critique,
        )
        grader_renderer = None  # XML passes through tokenizer.decode unchanged

    for iter_idx in range(n_iterations):
        if isinstance(grader_client, OpusSubagentGrader):
            grader_client.begin_iter(iter_idx)
        # Before each launch_plan_reward(plan=text, ...) call, if using
        # OpusSubagentGrader, tag the plan context so sample() routes to
        # the right plan bucket:
        if isinstance(grader_client, OpusSubagentGrader):
            grader_client.set_plan_context(f"fresh_{k}", text)
        launch_plan_reward(plan=text, ...)
        # First .result() in collect triggers flush+wait; no extra trainer call needed.

Non-goals: this shim does NOT implement compute_logprobs,
compute_logprobs_async, or any training-client methods. Strictly for
grading. The SDPO teacher-logprob path (train_ger_cr_v1.py:1410-1412)
still runs on the tinker training/sampling client.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tinker
from tinker import types

logger = logging.getLogger(__name__)


# Prompt markers used to classify incoming sample() prompts.
# SIGNAL_1_1_SCORING_PROMPT (goal-contrast, used for gc_target/gc_alt1/gc_alt2).
_HG_SIG1_1_MARKER = "You are evaluating whether a research plan is SPECIFICALLY tailored"
# SIGNAL_1_2_EXTRACTION_PROMPT (claim verification).
_HG_SIG1_2_MARKER = "Extract VERIFIABLE FACTUAL CLAIMS"
# build_single_signal_prompt output contains `<dim id="{signal.id}">`.
_DIM_ID_RE = re.compile(r'<dim\s+id="([^"]+)">')


@dataclass
class _PendingCall:
    plan_id: str
    kind: str  # "signal" or "hg"
    key: str   # signal_id (opt. with #repeat suffix) or hard-gate name
    prompt_text: str


class _PseudoFuture:
    """Minimal tinker.Future-shape object; .result() lazy-flushes the current iter's batch."""

    __slots__ = ("_grader", "_call", "_tokens", "_lock")

    def __init__(self, grader: "OpusSubagentGrader", call: _PendingCall):
        self._grader = grader
        self._call = call
        self._tokens: list[int] | None = None
        self._lock = threading.Lock()

    def _complete(self, tokens: list[int]) -> None:
        with self._lock:
            self._tokens = tokens

    def result(self, timeout: float | None = None) -> types.SampleResponse:
        with self._lock:
            already_have = self._tokens is not None
        if not already_have:
            self._grader.flush_current_batch(timeout=timeout)
        with self._lock:
            tokens = self._tokens
        if tokens is None:
            raise RuntimeError(
                f"OpusSubagentGrader: future for {self._call.plan_id}/{self._call.key} "
                f"was not populated by flush. Check daemon response."
            )
        seq = types.SampledSequence(
            stop_reason="stop",
            tokens=tokens,
            logprobs=None,
        )
        return types.SampleResponse(sequences=[seq])


class OpusSubagentGrader:
    """File-bus shim routing tinker grader sample() calls to an Opus subagent daemon."""

    def __init__(
        self,
        log_path: str | Path,
        tokenizer,
        long_critique: bool = False,
        timeout_sec: float = 900.0,
        poll_interval_sec: float = 5.0,
    ):
        self.log_path = Path(log_path)
        self.tokenizer = tokenizer
        self.long_critique = long_critique
        self.timeout_sec = timeout_sec
        self.poll_interval_sec = poll_interval_sec

        self._lock = threading.Lock()

        # Iter-scoped state, reset by begin_iter().
        self._current_iter: int | None = None
        self._batch_counter: int = 0  # increments per flush; allows multiple flushes per iter
        self._pending: list[_PseudoFuture] = []

        # Plan tagging: set by set_plan_context() before each launch_plan_reward().
        self._current_plan_id: str | None = None
        self._plan_text_by_id: dict[str, str] = {}
        self._hg_counter_by_plan: dict[str, int] = {}
        self._repeat_counter_by_plan: dict[str, dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Trainer-facing lifecycle
    # ------------------------------------------------------------------

    def begin_iter(self, iter_idx: int) -> None:
        """Reset per-iter state. Call before the iter's launch_plan_reward loop."""
        with self._lock:
            if self._pending:
                logger.warning(
                    "OpusSubagentGrader: begin_iter(%d) invoked with %d unflushed futures from "
                    "a prior iter; these will be dropped.", iter_idx, len(self._pending),
                )
            self._current_iter = iter_idx
            self._batch_counter = 0
            self._pending = []
            self._current_plan_id = None
            self._plan_text_by_id = {}
            self._hg_counter_by_plan = {}
            self._repeat_counter_by_plan = {}

    def set_plan_context(self, plan_id: str, plan_text: str) -> None:
        """Tag subsequent sample() calls with this plan_id.

        Must be called before each launch_plan_reward() in the iter loop,
        so the shim can disambiguate which of the ~17 calls belong to
        which plan.
        """
        with self._lock:
            self._current_plan_id = plan_id
            self._plan_text_by_id[plan_id] = plan_text
            self._hg_counter_by_plan.setdefault(plan_id, 0)
            self._repeat_counter_by_plan.setdefault(plan_id, {})

    # ------------------------------------------------------------------
    # tinker.SamplingClient-compatible API
    # ------------------------------------------------------------------

    def sample(self, model_input, num_samples: int = 1, sampling_params=None) -> _PseudoFuture:
        """Buffer the prompt; return a _PseudoFuture whose .result() will block on flush."""
        if num_samples != 1:
            raise ValueError(f"OpusSubagentGrader: num_samples must be 1, got {num_samples}")
        with self._lock:
            if self._current_plan_id is None:
                raise RuntimeError(
                    "OpusSubagentGrader.sample() called without prior set_plan_context(). "
                    "Trainer must call set_plan_context(plan_id, plan_text) before launch_plan_reward()."
                )
            plan_id = self._current_plan_id

        # Extract prompt text from ModelInput.
        if hasattr(model_input, "to_ints"):
            prompt_tokens = model_input.to_ints()
        else:
            prompt_tokens = list(model_input)
        prompt_text = self.tokenizer.decode(prompt_tokens)

        kind, key = self._classify_prompt(plan_id, prompt_text)
        call = _PendingCall(plan_id=plan_id, kind=kind, key=key, prompt_text=prompt_text)
        fut = _PseudoFuture(self, call)
        with self._lock:
            self._pending.append(fut)
        return fut

    def _classify_prompt(self, plan_id: str, prompt_text: str) -> tuple[str, str]:
        """Identify which (kind, key) bucket this prompt belongs to."""
        m = _DIM_ID_RE.search(prompt_text)
        if m:
            sid = m.group(1)
            with self._lock:
                repeats = self._repeat_counter_by_plan[plan_id]
                ridx = repeats.get(sid, 0)
                repeats[sid] = ridx + 1
            key = sid if ridx == 0 else f"{sid}#{ridx}"
            return ("signal", key)
        if _HG_SIG1_1_MARKER in prompt_text:
            with self._lock:
                ctr = self._hg_counter_by_plan[plan_id]
                self._hg_counter_by_plan[plan_id] = ctr + 1
            if ctr == 0:
                return ("hg", "gc_target")
            if ctr == 1:
                return ("hg", "gc_alt1")
            if ctr == 2:
                return ("hg", "gc_alt2")
            raise RuntimeError(f"Unexpected 4th SIGNAL_1_1 HG call for plan {plan_id}")
        if _HG_SIG1_2_MARKER in prompt_text:
            return ("hg", "cv")
        raise ValueError(
            f"OpusSubagentGrader: cannot classify prompt. First 200 chars: {prompt_text[:200]!r}"
        )

    # ------------------------------------------------------------------
    # Batch flush
    # ------------------------------------------------------------------

    def flush_current_batch(self, timeout: float | None = None) -> None:
        """Write pending prompts as one request file, block on response, populate futures."""
        with self._lock:
            if not self._pending:
                return
            if self._current_iter is None:
                raise RuntimeError("flush_current_batch called before begin_iter")
            iter_idx = self._current_iter
            pending = self._pending
            self._pending = []
            self._batch_counter += 1
            batch_suffix = "" if self._batch_counter == 1 else f"_b{self._batch_counter}"
            batch_id = f"iter_{iter_idx:03d}{batch_suffix}"
            plan_text_by_id = dict(self._plan_text_by_id)

        # Group pending futures by plan_id.
        plan_signal_prompts: dict[str, dict[str, str]] = {}
        plan_hg_prompts: dict[str, dict[str, str]] = {}
        futs_by_plan: dict[str, list[_PseudoFuture]] = {}
        for fut in pending:
            call = fut._call
            if call.kind == "signal":
                plan_signal_prompts.setdefault(call.plan_id, {})[call.key] = call.prompt_text
            else:
                plan_hg_prompts.setdefault(call.plan_id, {})[call.key] = call.prompt_text
            futs_by_plan.setdefault(call.plan_id, []).append(fut)

        # Build request JSON.
        plan_ids = sorted(futs_by_plan.keys())
        request = {
            "iter": iter_idx,
            "batch_id": batch_id,
            "grader_long_critique": self.long_critique,
            "plans": [
                {
                    "plan_id": pid,
                    "text": plan_text_by_id.get(pid, ""),
                    "signal_prompts": plan_signal_prompts.get(pid, {}),
                    "hard_gate_prompts": plan_hg_prompts.get(pid, {}),
                }
                for pid in plan_ids
            ],
        }

        # Write request atomically to grader_requests/.
        req_dir = self.log_path / "grader_requests"
        resp_dir = self.log_path / "grader_responses"
        req_dir.mkdir(parents=True, exist_ok=True)
        resp_dir.mkdir(parents=True, exist_ok=True)
        req_path = req_dir / f"{batch_id}.json"
        tmp_path = req_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(request))
        tmp_path.replace(req_path)
        logger.info(
            "OpusSubagentGrader: wrote batch %s with %d plans, %d futures to %s",
            batch_id, len(plan_ids), len(pending), req_path,
        )

        # Poll for response.
        resp_path = resp_dir / f"{batch_id}.json"
        # Ignore per-call timeout; use our config-set timeout as floor. Tinker's
        # default `collect_hard_gates(timeout=900.0)` would prematurely kill
        # large Opus batches (some plans take 15-20 min via subagent fan-out).
        effective_timeout = max(self.timeout_sec, timeout or 0.0)
        t_start = time.time()
        deadline = t_start + effective_timeout
        while not resp_path.exists():
            if time.time() > deadline:
                raise TimeoutError(
                    f"OpusSubagentGrader: timeout waiting for {resp_path} after "
                    f"{effective_timeout}s. Check daemon at {self.log_path}/opus_grader_daemon.log"
                )
            time.sleep(self.poll_interval_sec)
        batch_latency = time.time() - t_start
        logger.info(
            "OpusSubagentGrader: batch %s response ready in %.1fs", batch_id, batch_latency,
        )

        # Parse response, populate futures.
        try:
            resp = json.loads(resp_path.read_text())
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"OpusSubagentGrader: failed to parse {resp_path}: {e}"
            ) from e
        judgments = resp.get("judgments", {})

        for plan_id, futs in futs_by_plan.items():
            j = judgments.get(plan_id, {})
            sig_xmls = j.get("signal_xmls", {})
            hg_xmls = j.get("hard_gate_xmls", {})
            for fut in futs:
                call = fut._call
                xml = (sig_xmls.get(call.key) if call.kind == "signal"
                       else hg_xmls.get(call.key))
                if not xml:
                    # Daemon failed to produce a verdict for this prompt.
                    # Emit a placeholder the downstream parsers tolerate.
                    if call.kind == "signal":
                        # parse_scores: missing <score>N</score> → score=None
                        sid = call.key.split("#")[0]
                        xml = (
                            f'<evaluation><dim id="{sid}">'
                            f'<reasoning>[GRADER FAILED]</reasoning>'
                            f'<score></score><critique>[GRADER FAILED]</critique>'
                            f'</dim></evaluation>'
                        )
                    else:
                        # collect_hard_gates._parse_score: extract_json with
                        # no score field → 0.0. For cv use empty claims list.
                        if call.key == "cv":
                            xml = '{"score": 0.0, "claims": []}'
                        else:
                            xml = '{"score": 0.0}'
                    logger.warning(
                        "OpusSubagentGrader: missing verdict for %s/%s; using placeholder",
                        plan_id, call.key,
                    )
                tokens = self.tokenizer.encode(xml)
                fut._complete(tokens)
