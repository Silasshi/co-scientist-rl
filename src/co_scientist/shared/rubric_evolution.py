"""
Rubric-evolution orchestration for GER-CR-v1.

Ties together the three components of the evolving-rubric loop:

  1. `rubric_gen_prompt.build_rubric_gen_messages` — builds the chat messages
     for the rubric generator.
  2. A pluggable `GenClient` (protocol) — actually calls a model and returns raw
     text. We keep this pluggable so the same pipeline can run against
     OpenRouter/Opus, OpenRouter/Gemini, a local Qwen-235B via Tinker, or a
     stub for unit tests. The trainer picks which client to wire up.
  3. `rubric_gen_prompt.parse_rubric_response` — parses the JSON back into
     `RubricItem` objects.
  4. `rubric_buffer.RubricBuffer` — receives the new items.

This module is *side-effect-free w.r.t. APIs*: it never imports any specific
API client. Integrate by passing in a callable that does the actual network
call.

Usage sketch (from the trainer):

    buf = RubricBuffer.from_jsonl(run_dir / "rubric_buffer.jsonl")
    for iter_idx in range(n_iterations):
        rollouts = generate_rollouts(...)
        r_active_grades = grade_against_active(rollouts, buf)
        persistent_grades = grade_against_v8(rollouts)
        ...
        gradient_update(...)

        # At end of iter: elicit + update buffer
        added = run_rubric_evolution_step(
            buffer=buf,
            iteration=iter_idx,
            gen_client=my_openrouter_client,
            research_goal=goal_text,
            persistent_rubric_names=[s.name for s in SIGNALS],
            plan_current=rollouts[0].text,          # from π_old
            plan_control=reference_proposal_text,   # or B4 plan, or π_ref
        )
        buf.filter_and_truncate()
        buf.to_jsonl(run_dir / "rubric_buffer.jsonl")
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Callable, Iterable, Protocol

from co_scientist.shared.rubric_buffer import RubricBuffer, RubricItem
from co_scientist.shared.rubric_gen_prompt import (
    build_rubric_gen_messages,
    parse_rubric_response,
)


logger = logging.getLogger(__name__)


class GenClient(Protocol):
    """
    Minimal interface the rubric generator must satisfy.

    The trainer supplies a concrete callable. Examples of valid implementations:

      * an `openrouter_client.OpenRouterClient.chat` wrapper (awaitable, so the
        trainer wraps it with `asyncio.run`),
      * a `anthropic.Anthropic().messages.create` wrapper (if ANTHROPIC_API_KEY
        is set),
      * a file-bus stub that writes the prompt to disk, waits for a manual
        Claude-Code subagent to write the response, and reads it back.
    """

    def __call__(self, messages: list[dict]) -> str:  # returns raw text
        ...


# -------------------------------------------------------------------- main step


def run_rubric_evolution_step(
    *,
    buffer: RubricBuffer,
    iteration: int,
    gen_client: GenClient,
    research_goal: str,
    persistent_rubric_names: list[str],
    plan_current: str,
    plan_control: str,
    plan_current_label: str = "Plan A (current policy)",
    plan_control_label: str = "Plan B (control)",
    max_retries: int = 2,
    log_dir: Path | str | None = None,
) -> dict:
    """
    Do ONE rubric-evolution step: ask generator to produce new items from a
    pair, parse, add to buffer. Does *not* filter / truncate — caller decides
    when to do that (typically after the next iter's grading).

    Returns a dict with keys:
      - `added_ids` (list[str])
      - `n_added` (int)
      - `n_proposed` (int)  — how many items the generator produced (pre-dedup)
      - `raw_response` (str)
      - `messages` (list[dict]) — for reproducibility logging
      - `error` (str | None)

    If `log_dir` is given, writes a `rubric_gen_iter{iteration:03d}.json`
    trace file into it for post-hoc inspection.
    """
    messages = build_rubric_gen_messages(
        research_goal=research_goal,
        persistent_rubric_names=persistent_rubric_names,
        active_rubric_items=buffer.items.values(),
        plan_a=plan_current,
        plan_b=plan_control,
        plan_a_label=plan_current_label,
        plan_b_label=plan_control_label,
    )

    raw_text = ""
    error: str | None = None
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            raw_text = gen_client(messages)
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "rubric_gen call failed (attempt %d/%d): %s",
                attempt + 1,
                max_retries + 1,
                exc,
            )
    else:
        error = f"gen_client failed after {max_retries + 1} attempts: {last_exc}"
        raw_text = ""

    new_items: list[RubricItem] = []
    if raw_text and not error:
        try:
            new_items = parse_rubric_response(raw_text, created_at_iter=iteration)
        except Exception as exc:  # noqa: BLE001
            error = f"parse_rubric_response failed: {exc}"
            logger.warning("rubric_gen parse failed at iter %d: %s", iteration, exc)

    added_ids = buffer.add(new_items) if new_items else []

    trace = {
        "iteration": iteration,
        "messages": messages,
        "raw_response": raw_text,
        "n_proposed": len(new_items),
        "n_added": len(added_ids),
        "added_ids": added_ids,
        "error": error,
    }
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"rubric_gen_iter{iteration:03d}.json"
        log_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2))

    return trace


# ------------------------------------------------------- pair selection helpers


def select_contrast_pair(
    rollouts: list[str],
    rollout_scores: list[float] | None = None,
    control_text: str | None = None,
    mode: str = "best_vs_ref",
    rng: random.Random | None = None,
) -> tuple[str, str, str]:
    """
    Pick one (plan_current, plan_control, description) pair from available rollouts.

    Modes:
      * `best_vs_ref`     : best-scoring current rollout vs `control_text`
      * `best_vs_worst`   : best vs worst of current rollouts (control_text unused)
      * `random_vs_ref`   : random current rollout vs `control_text`

    Returns (plan_current, plan_control, label_description). Raises if inputs
    don't support the chosen mode.
    """
    if not rollouts:
        raise ValueError("no rollouts provided")
    rng = rng or random.Random(0)

    if mode in ("best_vs_ref", "random_vs_ref"):
        if control_text is None:
            raise ValueError(f"mode={mode!r} requires control_text")

    if mode == "best_vs_ref":
        if not rollout_scores or len(rollout_scores) != len(rollouts):
            raise ValueError("best_vs_ref requires rollout_scores aligned with rollouts")
        idx = max(range(len(rollouts)), key=lambda i: rollout_scores[i])
        return rollouts[idx], control_text, f"best rollout (rank 0 of {len(rollouts)}) vs reference"

    if mode == "random_vs_ref":
        idx = rng.randrange(len(rollouts))
        return rollouts[idx], control_text, "random rollout vs reference"

    if mode == "best_vs_worst":
        if not rollout_scores or len(rollout_scores) != len(rollouts):
            raise ValueError("best_vs_worst requires rollout_scores aligned with rollouts")
        if len(rollouts) < 2:
            raise ValueError("best_vs_worst requires ≥2 rollouts")
        best = max(range(len(rollouts)), key=lambda i: rollout_scores[i])
        worst = min(range(len(rollouts)), key=lambda i: rollout_scores[i])
        return rollouts[best], rollouts[worst], "best vs worst current rollout"

    raise ValueError(f"unknown pair-selection mode: {mode!r}")


# ----------------------------------------------------- stub client for testing


def make_stub_gen_client(fixture_json: str) -> GenClient:
    """
    Returns a GenClient that ignores the messages and replies with a fixed JSON
    blob (wrapped in a ```json fence). Useful for unit tests and for running
    train_ger_cr_v1 end-to-end before wiring a real model.
    """

    def _client(messages: list[dict]) -> str:  # noqa: ARG001
        return f"```json\n{fixture_json}\n```"

    return _client


# ------------------------------------------------------------------ self-tests


def _run_sanity_checks() -> None:
    """Exercises the full flow without any network call."""
    fixture = json.dumps(
        {
            "positive_rubrics": [
                {
                    "title": "Named datasets",
                    "description": "Plan names at least two concrete datasets",
                }
            ],
            "negative_rubrics": [
                {
                    "title": "Title padding",
                    "description": "Avoids paragraph-long restatements of the title",
                    "weight": 2,
                }
            ],
        }
    )
    client = make_stub_gen_client(fixture)

    buf = RubricBuffer(k_max=5)
    trace = run_rubric_evolution_step(
        buffer=buf,
        iteration=1,
        gen_client=client,
        research_goal="Develop a foundational optimizer for deep learning.",
        persistent_rubric_names=["G1 Goal-Contrast Margin"],
        plan_current="Plan A body text ...",
        plan_control="Plan B body text ...",
    )
    assert trace["n_proposed"] == 2
    assert trace["n_added"] == 2
    assert trace["error"] is None
    assert len(buf.items) == 2
    titles = {it.title for it in buf.items.values()}
    assert titles == {"Named datasets", "Title padding"}

    # Pair selection
    cur, ctrl, label = select_contrast_pair(
        rollouts=["plan1", "plan2", "plan3"],
        rollout_scores=[0.5, 0.7, 0.3],
        control_text="reference text",
        mode="best_vs_ref",
    )
    assert cur == "plan2"
    assert ctrl == "reference text"
    assert "best rollout" in label

    cur2, ctrl2, _ = select_contrast_pair(
        rollouts=["a", "b", "c"],
        rollout_scores=[0.1, 0.9, 0.5],
        mode="best_vs_worst",
    )
    assert cur2 == "b" and ctrl2 == "a"

    # Error path: stub that raises
    def failing_client(messages):
        raise RuntimeError("boom")

    buf2 = RubricBuffer()
    trace2 = run_rubric_evolution_step(
        buffer=buf2,
        iteration=0,
        gen_client=failing_client,
        research_goal="x",
        persistent_rubric_names=[],
        plan_current="a",
        plan_control="b",
        max_retries=1,
    )
    assert trace2["error"] is not None
    assert len(buf2.items) == 0

    print("rubric_evolution self-tests passed.")


if __name__ == "__main__":
    _run_sanity_checks()
