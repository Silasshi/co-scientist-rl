"""
Binary grader for evolving-rubric (R_active) items in GER-CR-v1.

Each RubricItem is scored independently as yes/no ("does this plan satisfy the
criterion?"). Grades are 0.0 or 1.0, aligned with `RubricBuffer.record_grades`.

Design notes:
  * Reuses the project's existing tinker sampling_client + renderer pipeline so
    we don't introduce a new API dependency (works with Qwen3-30B-A3B,
    GPT-OSS-120B, Qwen3-235B — whichever the trainer wires up).
  * Prompt is deliberately short (binary question, small output): grader
    max_tokens defaults to 512 to keep per-item cost low.
  * Parse failures (missing <answer>, malformed XML) → grade 0.0 (conservative:
    assume criterion not met). This matches how grant_signal_reward handles
    missing Likert scores.
  * Thinking-mode output (Qwen3 <think>) and Harmony reasoning channels
    (GPT-OSS) are stripped via `_decode_grader_output` (imported from
    train_buffer_ttt) so we don't need to re-implement renderer parsing.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import tinker

from co_scientist.shared.rubric_buffer import NEGATIVE, POSITIVE, RubricBuffer, RubricItem

if TYPE_CHECKING:
    # Only imported for type hints; avoids an import cycle at module-load time.
    pass


# --------------------------------------------------------------------- prompt


_ANSWER_RE = re.compile(r"<answer>\s*(yes|no)\s*</answer>", re.IGNORECASE)


def build_binary_rubric_prompt(goal: str, plan: str, item: RubricItem) -> str:
    """Build a binary grader prompt for a single rubric item against a plan.

    The prompt asks the grader to return exactly one `<answer>yes</answer>` or
    `<answer>no</answer>` tag. Additional reasoning is allowed in free text
    before the answer tag (helpful for the grader to think) but is ignored
    by the parser.
    """
    if item.rubric_type == NEGATIVE:
        criterion_framing = (
            "This is a NEGATIVE criterion phrased as 'avoids X'. Answer 'yes' "
            "if the plan SUCCESSFULLY AVOIDS the failure pattern described. "
            "Answer 'no' if the plan exhibits the failure pattern."
        )
    else:
        criterion_framing = (
            "This is a POSITIVE criterion. Answer 'yes' if the plan clearly "
            "demonstrates the property described. Answer 'no' if the property "
            "is absent or only superficially claimed."
        )

    return (
        "You are grading a research grant proposal against a single binary "
        "criterion. Read the goal, the plan, and the criterion. Decide whether "
        "the plan satisfies the criterion.\n\n"
        f"<goal>\n{goal.strip()}\n</goal>\n\n"
        f"<plan>\n{plan.strip()}\n</plan>\n\n"
        f"<criterion>\n"
        f"<title>{item.title.strip()}</title>\n"
        f"<description>{item.description.strip()}</description>\n"
        f"<polarity>{item.rubric_type}</polarity>\n"
        f"</criterion>\n\n"
        f"{criterion_framing}\n\n"
        "Respond with a brief 1-2 sentence justification, then a single final "
        "tag: `<answer>yes</answer>` or `<answer>no</answer>`. Do not output "
        "any other XML tags, scores, or numbers."
    )


def parse_binary_answer(text: str) -> float:
    """Extract yes/no from grader output. Returns 1.0 / 0.0. Failure → 0.0.

    Takes the LAST matching `<answer>...</answer>` in the text — some graders
    restate the question earlier in their reasoning.
    """
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return 0.0
    last = matches[-1].strip().lower()
    return 1.0 if last == "yes" else 0.0


# ---------------------------------------------------------------- async calls


def launch_binary_grade(
    plan: str, goal: str, item: RubricItem,
    grader_client, renderer,
    max_tokens: int = 512, temperature: float = 0.0,
):
    """Fire a single grader call for (plan, item). Returns a tinker future."""
    prompt = build_binary_rubric_prompt(goal, plan, item)
    convo = [{"role": "user", "content": prompt}]
    model_input = renderer.build_generation_prompt(convo)
    return grader_client.sample(
        model_input,
        num_samples=1,
        sampling_params=tinker.types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=renderer.get_stop_sequences(),
        ),
    )


def collect_binary_grade(future, tokenizer, renderer=None, timeout: float = 900.0) -> float:
    """Collect one grader future → 0.0/1.0.

    Uses the same decode path as the Likert grader (handles Qwen3 <think> and
    GPT-OSS Harmony reasoning channels). Any exception → 0.0 (conservative).
    """
    # Import here to avoid circular: train_buffer_ttt imports shared.rubric_buffer
    # indirectly via some future chain. Deferred import keeps module load clean.
    from co_scientist.grant_proposal.train_buffer_ttt import _decode_grader_output
    try:
        result = future.result(timeout=timeout)
        tokens = list(result.sequences[0].tokens)
        text = _decode_grader_output(tokens, tokenizer, renderer)
        return parse_binary_answer(text)
    except Exception:
        return 0.0


def run_binary_grades_for_buffer(
    plans: list[str], buffer: RubricBuffer, goal: str,
    grader_client, renderer, tokenizer,
    max_tokens: int = 512, temperature: float = 0.0,
    timeout: float = 900.0,
    grader_client_alt=None, renderer_alt=None, tokenizer_alt=None,
    use_alt: bool = False,
) -> dict[str, list[float]]:
    """Grade every item in the buffer against every plan.

    Launches |plans| × |buffer.items| futures in parallel, then collects.
    Returns `dict[rubric_id -> list[0.0|1.0]]` aligned with the `plans` list.

    Items not in the buffer when this is called are silently skipped (callers
    may pass a buffer snapshot; items added later won't appear in the result).

    When `use_alt` is True and `grader_client_alt` is provided, ALL binary
    calls go through the alt grader (use-case: route R_active grading through
    Qwen3-235B while R_persist stays on GPT-OSS). `tokenizer_alt` / `renderer_alt`
    must be provided alongside `grader_client_alt` when switching.
    """
    if not plans or not buffer.items:
        return {rid: [] for rid in buffer.items}

    if use_alt and grader_client_alt is not None:
        client, rend, tok = grader_client_alt, renderer_alt, tokenizer_alt
    else:
        client, rend, tok = grader_client, renderer, tokenizer

    # Stage 1: fire all futures (all items × all plans in flight simultaneously).
    # Shape: futures[rubric_id] = list of futures, one per plan, aligned with
    # the input `plans` order.
    futures: dict[str, list] = {}
    for rid, item in buffer.items.items():
        per_plan_futures = []
        for plan in plans:
            fut = launch_binary_grade(
                plan=plan, goal=goal, item=item,
                grader_client=client, renderer=rend,
                max_tokens=max_tokens, temperature=temperature,
            )
            per_plan_futures.append(fut)
        futures[rid] = per_plan_futures

    # Stage 2: collect.
    grades: dict[str, list[float]] = {}
    for rid, fut_list in futures.items():
        per_plan_grades = []
        for fut in fut_list:
            per_plan_grades.append(
                collect_binary_grade(fut, tok, renderer=rend, timeout=timeout)
            )
        grades[rid] = per_plan_grades

    return grades


# ---------------------------------------------------------- R_active helpers


def compute_r_active_aggregate(
    buffer: RubricBuffer, verdicts: dict[str, float]
) -> float:
    """Thin wrapper around `RubricBuffer.aggregate_reward` for a single plan.

    `verdicts` is `{rubric_id: 0.0|1.0}` as returned by the grader. Missing
    ids are treated as not-scored (excluded from the weighted sum).
    """
    return buffer.aggregate_reward(verdicts)


def build_rubric_critique_context(
    plan_text: str, goal: str, buffer: RubricBuffer,
    verdicts: dict[str, float],
    max_plan_chars: int = 12000,
) -> str:
    """Build the revision-context prompt used as `context_i` for the virtual
    R_active signal's per-signal REINFORCE datum (Phase 3 coupling).

    The structure mirrors `build_per_signal_context` in `train_cr_v7.py` but
    the feedback block lists every R_active item's binary verdict plus the
    aggregate satisfaction rate. The model is prompted to address items that
    were failed (NO), with priority to high-weight negative items.

    Parameters
    ----------
    plan_text: parent plan text (will be truncated at `max_plan_chars`).
    goal: research goal.
    buffer: current `RubricBuffer` (snapshot).
    verdicts: `{rubric_id: 0.0|1.0}` — parent's R_active grades.
    """
    if len(plan_text) > max_plan_chars:
        plan_text = plan_text[:max_plan_chars] + "\n\n[... rest truncated ...]"

    # Feedback block
    if not buffer.items:
        feedback_lines = ["- (no active rubric items yet)"]
    else:
        feedback_lines = []
        for rid, item in buffer.items.items():
            v = verdicts.get(rid)
            if v is None:
                verdict_str = "not scored"
            elif v >= 0.5:
                verdict_str = "YES (satisfied)"
            else:
                verdict_str = "NO (failed)"
            tag = "POS" if item.rubric_type == POSITIVE else "NEG"
            feedback_lines.append(
                f"- [{tag} w={item.weight:.1f}] {item.title}: {verdict_str}\n"
                f"    ↳ {item.description}"
            )
    feedback_block = "\n".join(feedback_lines)

    agg = compute_r_active_aggregate(buffer, verdicts) if verdicts else 0.5

    return (
        f"# Research Goal\n{goal.strip()}\n\n"
        f"# Current Plan\n{plan_text.strip()}\n\n"
        f"# Adaptive-rubric verdicts (binary, evolving each iter)\n"
        f"{feedback_block}\n\n"
        f"Aggregate R_active satisfaction: {agg:.2f}/1.00\n\n"
        f"# Task\n"
        f"Rewrite the plan to address the rubric items you FAILED (marked NO). "
        f"Focus especially on high-weight NEG items phrased as 'avoids X' — "
        f"those flag active failure patterns that must be eliminated in the "
        f"rewrite. Do not introduce new fabrications or template padding.\n\n"
        f"<think>\n"
    )


# ------------------------------------------------------------------ self-test


def _run_sanity_checks() -> None:
    """Exercises prompt build + parser without any network call."""
    # Fixtures
    pos_item = RubricItem(
        title="Named datasets",
        description="Plan names at least two concrete datasets with access procedures.",
        rubric_type=POSITIVE,
        created_at_iter=0,
    )
    neg_item = RubricItem(
        title="Avoids title padding",
        description="Avoids paragraph-long restatements of the title without adding content.",
        rubric_type=NEGATIVE,
        created_at_iter=0,
    )

    # Prompt build: positive framing
    pos_prompt = build_binary_rubric_prompt(
        goal="Design a foundational optimizer",
        plan="We will use ImageNet and CIFAR-10 with standard splits...",
        item=pos_item,
    )
    assert "<criterion>" in pos_prompt
    assert "Named datasets" in pos_prompt
    assert "POSITIVE criterion" in pos_prompt
    assert "<polarity>positive</polarity>" in pos_prompt
    assert "<answer>yes</answer>" in pos_prompt  # instruction line mentions format

    # Prompt build: negative framing
    neg_prompt = build_binary_rubric_prompt(
        goal="g", plan="p",
        item=neg_item,
    )
    assert "NEGATIVE criterion" in neg_prompt
    assert "avoids X" in neg_prompt
    assert "<polarity>negative</polarity>" in neg_prompt

    # Parser
    assert parse_binary_answer("blah <answer>yes</answer>") == 1.0
    assert parse_binary_answer("blah <answer>no</answer>") == 0.0
    assert parse_binary_answer("YES <answer>Yes</answer>") == 1.0
    assert parse_binary_answer("  <answer>  no  </answer>  ") == 0.0
    assert parse_binary_answer("no tags here") == 0.0
    assert parse_binary_answer("") == 0.0
    # Multiple answers: take last
    assert parse_binary_answer(
        "<answer>no</answer> wait I changed my mind <answer>yes</answer>"
    ) == 1.0
    # Thinking prefix (already handled by _decode_grader_output upstream, but
    # verify raw parser is robust)
    assert parse_binary_answer(
        "<think>not sure</think> actually <answer>yes</answer>"
    ) == 1.0

    # Stub grader: launch + collect via a fake future
    class _FakeSequence:
        def __init__(self, text: str):
            # Fake tokens — real tokenizer.decode will just see these ids, so
            # we use a fake tokenizer whose decode echoes the stored text.
            self.tokens = [0, 1, 2]
            self._text = text

    class _FakeResult:
        def __init__(self, text: str):
            self.sequences = [_FakeSequence(text)]

    class _FakeFuture:
        def __init__(self, text: str):
            self._text = text

        def result(self, timeout=900.0):
            return _FakeResult(self._text)

    class _FakeTokenizer:
        # Map: the decode() signature matches HF tokenizer; we stash responses
        # keyed off the token list id via a closure in _FakeFuture instead.
        # Simpler: always decode to the same hardcoded string for this smoke test.
        def __init__(self, text: str):
            self._text = text

        def decode(self, tokens) -> str:
            return self._text

    # Direct parser smoke (no renderer)
    # Can't call collect_binary_grade without importing _decode_grader_output,
    # which requires train_buffer_ttt. Skip integration test here — the full
    # path is exercised in the trainer dry-run verification step.

    # Buffer-level wiring: empty buffer returns empty dict (no crash)
    empty_buf = RubricBuffer()
    out = run_binary_grades_for_buffer(
        plans=["plan1"], buffer=empty_buf, goal="g",
        grader_client=None, renderer=None, tokenizer=None,
    )
    assert out == {}

    # Empty plans list returns empty-list per item
    buf = RubricBuffer()
    buf.add([pos_item, neg_item])
    out2 = run_binary_grades_for_buffer(
        plans=[], buffer=buf, goal="g",
        grader_client=None, renderer=None, tokenizer=None,
    )
    assert set(out2.keys()) == {pos_item.id, neg_item.id}
    assert all(v == [] for v in out2.values())

    # compute_r_active_aggregate
    verdicts_all_yes = {pos_item.id: 1.0, neg_item.id: 1.0}
    agg1 = compute_r_active_aggregate(buf, verdicts_all_yes)
    assert agg1 == 1.0, f"all-yes aggregate should be 1.0, got {agg1}"

    verdicts_mixed = {pos_item.id: 1.0, neg_item.id: 0.0}  # neg weight=1.5
    agg2 = compute_r_active_aggregate(buf, verdicts_mixed)
    # num = 1.0*1.0 + 1.5*0.0 = 1.0,  den = 1.0 + 1.5 = 2.5  → 0.4
    assert abs(agg2 - 0.4) < 1e-9, f"mixed aggregate mismatch: {agg2}"

    verdicts_missing = {}
    agg3 = compute_r_active_aggregate(buf, verdicts_missing)
    # empty verdicts → num=den=0 → fallback 0.5
    assert agg3 == 0.5

    # build_rubric_critique_context — structure checks
    ctx = build_rubric_critique_context(
        plan_text="Plan body goes here.", goal="Design an optimizer",
        buffer=buf, verdicts={pos_item.id: 1.0, neg_item.id: 0.0},
    )
    assert "# Research Goal" in ctx
    assert "# Current Plan" in ctx
    assert "Named datasets" in ctx
    assert "YES (satisfied)" in ctx
    assert "NO (failed)" in ctx
    assert "Aggregate R_active satisfaction: 0.40" in ctx
    assert "<think>" in ctx  # Qwen3 thinking-mode bait
    assert "POS w=1.0" in ctx
    assert "NEG w=1.5" in ctx  # negative weight multiplier applied
    assert "# Task" in ctx
    assert "FAILED" in ctx  # instruction line

    # Empty buffer → "no active rubric items" line
    empty_ctx = build_rubric_critique_context(
        plan_text="x", goal="y", buffer=RubricBuffer(), verdicts={},
    )
    assert "no active rubric items" in empty_ctx

    # Truncation
    long_ctx = build_rubric_critique_context(
        plan_text="x" * 30000, goal="y", buffer=RubricBuffer(),
        verdicts={}, max_plan_chars=100,
    )
    assert "rest truncated" in long_ctx
    assert long_ctx.count("x") <= 200  # original 100 chars + maybe some incidental x's

    print("rubric_grader self-tests passed.")


if __name__ == "__main__":
    _run_sanity_checks()
