"""D5 Audit Prompt v3 — 9-dim hybrid (5 universal + 4 TTT-Discover-subfield).

Per AUDIT_RUBRIC_v3.md (locked 2026-04-25). Replaces v1/v2.

Design highlights:
- 5 universal dims (Soundness/Significance/Originality/Clarity/Reproducibility) drawn from
  NeurIPS/ICLR/ICML/NSF reviewer forms
- 4 subfield-specific dims (Necessity/Disentanglement/Compute/Reward-hacking) drawn from
  OpenReview records of MTTT/Voyager/SCoRe/Guided-ReST/etc.
- Option B anchors: T1-T4 anchors quote real reviewer comments verbatim
- Two-pass: claim_list + concern_list enumeration THEN scoring
- No anti-pattern penalty (anchors encode "no substance = low score")
- Equal weighting per dim; total raw /45, normalized /20
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


DEPTH_AUDIT_PROMPT_V3 = """\
You are an expert ML/AI conference reviewer auditing a research plan. Score the plan on 9
dimensions (5 universal + 4 subfield-specific to test-time-search / self-improvement /
search-with-LLMs papers), each 1-5 (integer).

# Research Goal
{goal}

# Research Plan to Evaluate
{plan}

# Pass 1 — Mandatory enumeration (do this BEFORE scoring)

Extract two disjoint lists. Each entry quotes ≤25 words verbatim from the plan.

A) `claim_list` — every VERIFIABLE specific factual claim:
   - Equations with full RHS
   - Hyperparameters with numbers (`LoRA rank 64`, `learning rate 2e-5`)
   - Named real benchmarks with cited prior numbers (`MATH GPT-4 50.4%`)
   - Deterministic operators (`score = α·R + (1-α)·log(1+n)`)

B) `concern_list` — every reviewer-style concern that arises while reading. NOT just
   "Pattern N" labels — also includes:
   - Missing comparisons (no baseline named, or named without prior numbers)
   - Hand-waved mechanisms (passive verbs without operator: "gradient ascent",
     "select high-quality")
   - Vague hyperparameters ("small learning rate", "appropriate batch size")
   - Unattributed novelty claims ("novel" without differentiation from prior work)
   - Generic algorithm names without instantiation ("REINFORCE", "LoRA fine-tuning")

# Pass 2 — Score 9 Dimensions, 1-5 Each

Use Pass 1 enumerations as evidence. Each dim's anchors describe substance density, not
surface presence.

## Universal layer (5 dims)

### U1. Soundness — Are claims supported by evidence (math or empirical)?
- 1: Claims unsupported (no equations, no benchmark numbers, no derivation)
- 2: Some claims supported but key claims hand-waved
- 3: Major claims supported by EITHER inline math (RHS-complete) OR specific benchmark numbers, not both
- 4: Claims supported by both inline math AND specific empirical numbers; minor hand-waving
- 5: All non-trivial claims supported with rigor; assumptions stated; robustness discussed

### U2. Significance — Is this a real problem with real impact?
- 1: Toy problem, no clear practical value
- 2: Real problem but plan addresses only narrow slice
- 3: Real problem, plan addresses central question, but unclear if it advances state of the art
- 4: Real problem, plan claims SOTA-comparable result with credible argument
- 5: Real problem, plan would advance SOTA demonstrably and be picked up by community

### U3. Originality — Novel beyond pretraining-knowledge or trivial recombination?
- 1: Standard toolkit recombination ("REINFORCE + LoRA + buffer") with no insight
- 2: Minor adaptation of known method
- 3: Non-obvious combination with at least one mechanism-level insight
- 4: Substantive modification with clear differentiation from prior work; new operator or new objective
- 5: Genuinely novel algorithm or framing; clear theoretical/empirical hook differentiating from cited prior work

### U4. Clarity — Is the plan well-reasoned and structurally clear?
- 1: Disorganized; key sections missing or vague
- 2: Sections present but mechanisms unspecified
- 3: Mechanisms named but reasoning chain has gaps
- 4: Mechanisms specified with rationale; reasoning chain coherent end-to-end
- 5: Clear non-specialist-readable plan; each step justified; no jargon without explanation

### U5. Reproducibility — Are compute, hyperparameters, code/data, and statistics specified?
- 1: No mention of compute, hparams, statistical methodology
- 2: 1 of {{compute, hparams, eval methodology}} mentioned vaguely
- 3: 2 of 3 mentioned with some specificity
- 4: All 3 stated with concrete values (e.g. compute budget; LoRA rank=N; n=K seeds)
- 5: All stated + statistical methodology (CIs, p-values, multi-seed) + open-model commitment

## Subfield-specific layer (4 dims) — anchors quote real OpenReview reviewer comments

### T1. Necessity of framing — Does the plan demonstrate that test-time RL is *necessary*?

Anchor 1 — quoting MTTT (ICLR 2024 reject) reviewer DvfS:
*"This paper motivates from the TTT perspective, but no TTT experiments are performed."*

This is the canonical failure mode. Score:
- 1: Plan does not address whether test-time RL is needed at all (the MTTT failure mode)
- 2: Plan asserts test-time RL helps but no comparison to in-context-only baseline
- 3: Plan mentions a frozen-LLM baseline but doesn't quantify gap
- 4: Plan specifies a frozen-LLM best-of-N baseline with prior numbers and explains why test-time RL exceeds it
- 5: Plan explicitly characterizes the regime where test-time RL is NECESSARY, with a non-RL baseline named that demonstrably can't reach the target

### T2. Disentanglement — Does the plan separate LLM-prior contribution from method contribution?

Anchor 1 — quoting Voyager (TMLR) reviewer eudD:
*"Voyager does not really learn, ChatGPT does."*

This was THE most-quoted critique of search-with-LLM papers in OpenReview. Score:
- 1: Plan treats LLM+method as one black box (the Voyager failure mode)
- 2: Plan acknowledges base-model dependence but no ablation proposed
- 3: Plan proposes one of {{frozen baseline / different base model / pretraining-only}}
- 4: Plan proposes ≥2 disentanglement ablations with expected results
- 5: Plan proposes full disentanglement matrix (frozen vs adapted × different bases) with hypothesis on which factor dominates

### T3. Compute / cost accounting in operational units

Anchor 2 — quoting Guided-ReST (rejected) reviewer xv2X:
*"No actual latency or compute cost accounting; claims based only on token budgets."*

Reviewers in this subfield demand $/GPU-hours, not just FLOPs. Score:
- 1: No compute statement
- 2: Compute mentioned in non-operational units (e.g. "compute-efficient", FLOP only) — the Guided-ReST failure mode
- 3: Compute stated in one operational unit (e.g. GPU-hours or $)
- 4: Compute stated in multiple operational units with method-specific accounting
- 5: 4 + sensitivity analysis (e.g. "cost scales linearly with X; halving Y doubles cost")

### T4. Reward-hacking / saturation analysis

Anchor 4 — quoting SCoRe (NeurIPS oral) reviewer e4kn (this comment EARNED accept-as-oral):
*"Reward shaping with α>1 could incentivize the model to introduce minor errors in first
steps to enable correction. Authors' analysis ruling this out is exactly the kind of
reward-hacking probe reviewers reward."*

Score (positive — what good looks like):
- 1: No analysis of reward-objective robustness or iteration-limit
- 2: Acknowledges potential reward hacking or saturation but no concrete pathway analysis
- 3: Names ≥1 specific Goodhart pathway OR ≥1 specific saturation mechanism
- 4: Both analyzed: Goodhart pathways identified + saturation behavior characterized; mitigations proposed
- 5: 4 + plan includes adversarial-probe experiment OR stop-criterion based on saturation detection (the SCoRe accept-as-oral pattern)

# Output Format

Respond with ONLY this JSON (no other text, no markdown fence):

{{"claim_list": [
    {{"kind": "equation"|"hparam"|"benchmark"|"operator", "quote": "<≤25 words>"}}
  ],
  "concern_list": [
    {{"kind": "missing_comparison"|"hand_waved"|"vague_hparam"|"unattributed_novelty"|"generic_algo", "quote": "<≤25 words>"}}
  ],
  "universal_scores": {{
    "U1_soundness":       {{"score": <int>, "justification": "<1-2 sentences citing claims/concerns>"}},
    "U2_significance":    {{"score": <int>, "justification": "..."}},
    "U3_originality":     {{"score": <int>, "justification": "..."}},
    "U4_clarity":         {{"score": <int>, "justification": "..."}},
    "U5_reproducibility": {{"score": <int>, "justification": "..."}}
  }},
  "subfield_scores": {{
    "T1_necessity":       {{"score": <int>, "justification": "..."}},
    "T2_disentanglement": {{"score": <int>, "justification": "..."}},
    "T3_compute":         {{"score": <int>, "justification": "..."}},
    "T4_reward_hacking":  {{"score": <int>, "justification": "..."}}
  }},
  "universal_total": <int 5-25>,
  "subfield_total": <int 4-20>,
  "weighted_total_norm20": <float 0-20, computed as (universal_total + subfield_total) / 45 * 20>}}
"""


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL | re.MULTILINE)
_FIRST_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

_UNIVERSAL_KEYS = ["U1_soundness", "U2_significance", "U3_originality", "U4_clarity", "U5_reproducibility"]
_SUBFIELD_KEYS = ["T1_necessity", "T2_disentanglement", "T3_compute", "T4_reward_hacking"]


def parse_audit_v3_response(text: str) -> dict[str, Any]:
    """Parse Opus v3 audit JSON, robust to markdown fences and partial fields."""
    warnings: list[str] = []
    raw = text.strip() if text else ""

    m = _FENCE_RE.search(raw)
    if m:
        raw = m.group(1).strip()
        warnings.append("stripped_markdown_fence")

    parsed: dict[str, Any] | None = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m2 = _FIRST_JSON_OBJ_RE.search(raw)
        if m2:
            try:
                parsed = json.loads(m2.group(0))
                warnings.append("extracted_first_json_object")
            except json.JSONDecodeError as e:
                warnings.append(f"json_decode_error: {e}")
        else:
            warnings.append("no_json_object_found")

    if not isinstance(parsed, dict):
        return _empty_v3_response(warnings, raw[:500])

    # Extract universal_scores + subfield_scores
    universal = parsed.get("universal_scores") or {}
    subfield = parsed.get("subfield_scores") or {}

    universal_int = {}
    for k in _UNIVERSAL_KEYS:
        entry = universal.get(k) or {}
        score = entry.get("score") if isinstance(entry, dict) else None
        if not isinstance(score, int) or score < 1 or score > 5:
            score = None
            warnings.append(f"missing_or_invalid_{k}")
        universal_int[k] = score

    subfield_int = {}
    for k in _SUBFIELD_KEYS:
        entry = subfield.get(k) or {}
        score = entry.get("score") if isinstance(entry, dict) else None
        if not isinstance(score, int) or score < 1 or score > 5:
            score = None
            warnings.append(f"missing_or_invalid_{k}")
        subfield_int[k] = score

    # Aggregate totals
    u_total = sum(s for s in universal_int.values() if s is not None) if all(s is not None for s in universal_int.values()) else None
    s_total = sum(s for s in subfield_int.values() if s is not None) if all(s is not None for s in subfield_int.values()) else None
    weighted = None
    if u_total is not None and s_total is not None:
        weighted = (u_total + s_total) / 45.0 * 20.0

    out = {
        "claim_list": parsed.get("claim_list", []) or [],
        "concern_list": parsed.get("concern_list", []) or [],
        "universal_scores": universal,
        "subfield_scores": subfield,
        "universal_total": u_total,
        "subfield_total": s_total,
        "weighted_total_norm20": weighted,
        # Surface flat per-dim scores for compatibility with audit_log aggregator
        **{f"_{k}": v for k, v in universal_int.items()},
        **{f"_{k}": v for k, v in subfield_int.items()},
    }
    if warnings:
        out["_parse_warnings"] = warnings
    return out


def _empty_v3_response(warnings: list[str], raw_snip: str) -> dict[str, Any]:
    return {
        "claim_list": [], "concern_list": [],
        "universal_scores": {},
        "subfield_scores": {},
        "universal_total": None,
        "subfield_total": None,
        "weighted_total_norm20": None,
        **{f"_{k}": None for k in _UNIVERSAL_KEYS + _SUBFIELD_KEYS},
        "_parse_warnings": warnings,
        "_raw": raw_snip,
    }
