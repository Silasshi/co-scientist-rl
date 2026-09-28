"""D6 signal definitions — re-exports the D4 12-signal rubric from shared.

All signal logic lives in shared/grant_signal_reward.py. This module is a
thin re-export so D6 code imports cleanly from its own namespace.

Usage:
    from co_scientist.d6_grant_proposal.signals import (
        SIGNALS, SIGNAL_VARIANTS, SIGNAL_WEIGHTS, aggregate_reward,
        build_single_signal_prompt, build_single_call_prompt,
    )
"""
from co_scientist.shared.grant_signal_reward import (  # noqa: F401
    SIGNALS,
    SIGNAL_VARIANTS,
    SIGNAL_WEIGHTS,
    SCORE_MAX,
    SignalSpec,
    normalize_score,
    aggregate_reward,
    build_single_signal_prompt,
    build_single_call_prompt,
)

__all__ = [
    "SIGNALS",
    "SIGNAL_VARIANTS",
    "SIGNAL_WEIGHTS",
    "SCORE_MAX",
    "SignalSpec",
    "normalize_score",
    "aggregate_reward",
    "build_single_signal_prompt",
    "build_single_call_prompt",
]
