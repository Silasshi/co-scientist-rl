"""Regression tests for `extract_solution` (mu_prompts_v1).

The M8 quality-audit remediation (2026-04-27, DECISIONS.md (f)) surfaced that
some Qwen3 outputs contain the literal string `<solution>...</solution>` inside
their `<think>` preamble (when the model describes the output format). The
non-greedy `_SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.DOTALL)`
matched that 3-character `...` placeholder before the real plan body, producing
broken extracted plans (e.g., `tt_control mu_prime_6 → "..."`) downstream.

Fix: strip `<think>...</think>` BEFORE applying `_SOLUTION_RE`. See
`mu_prompts_v1._THINK_RE` and `extract_solution` body.

These tests cover: (a) normal extraction, (b) the regression case, (c) edge
case where `<think>` is the only tag, (d) ensure heading-stripping still
works after the think-strip change.
"""
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v1 import extract_solution


def test_normal_solution_extraction():
    """Standard <solution>body</solution> still extracts the body."""
    raw = "<solution>\n# Problem statement\nThis is the plan.\n</solution>"
    out = extract_solution(raw)
    assert out.startswith("# Problem statement")
    assert "This is the plan." in out


def test_regression_think_contains_literal_solution_tags():
    """The exact M8 regression case: <think> preamble describes format
    using literal `<solution>...</solution>`, then real <solution> follows.

    Without the fix, _SOLUTION_RE would match the 3-char `...` placeholder.
    With the fix, _THINK_RE strips the entire <think> block first.
    """
    raw = (
        "<think>\n"
        "Okay, I need to write a plan. The user wants me to use "
        "`<solution>...</solution>` tags around the body.\n"
        "</think>\n"
        "<solution>\n# Real plan title\nReal plan body with substance.\n</solution>"
    )
    out = extract_solution(raw)
    assert out != "..."
    assert out.startswith("# Real plan title")
    assert "Real plan body" in out


def test_only_think_no_solution_tag():
    """When only <think> exists with no real <solution>, fallback returns
    the post-think text (which may be empty)."""
    raw = "<think>preamble reasoning</think>\nSome trailing text."
    out = extract_solution(raw)
    # After think-strip, should return "Some trailing text." (heading-stripped if no heading)
    assert "preamble" not in out
    assert "Some trailing text" in out


def test_heading_stripping_still_works():
    """The heading-stripping logic should still cut "Okay, I need to..."
    style preambles INSIDE <solution> when no markdown heading precedes them
    properly.
    """
    raw = (
        "<solution>\n"
        "Okay, let me think. The user wants a plan.\n"
        "# Real heading\nActual plan body.\n"
        "</solution>"
    )
    out = extract_solution(raw)
    assert out.startswith("# Real heading")
    assert "Okay, let me think" not in out


def test_think_strip_before_solution_search():
    """The <think> strip must happen BEFORE the _SOLUTION_RE search, otherwise
    the literal-text-in-think bug returns. This test fails on the pre-fix code.
    """
    raw = (
        "<think>format is <solution>X</solution></think>\n"
        "<solution># Plan\nBody.\n</solution>"
    )
    out = extract_solution(raw)
    assert "X" not in out
    assert out.startswith("# Plan")


def test_multiple_think_blocks():
    """Some Qwen3 outputs have multiple <think> blocks. All should be stripped."""
    raw = (
        "<think>first</think>\n"
        "<think>second describing <solution>...</solution></think>\n"
        "<solution># Real\nBody.</solution>"
    )
    out = extract_solution(raw)
    assert out.startswith("# Real")
    assert "first" not in out
    assert "second" not in out
