"""D5 Opus depth-audit prompt v2 — substance-aware, surface-form-resistant.

Replaces the v1 prompt at `src/co_scientist/ttt_discover/opus_eval_agent.py:30-77`.
v1 had surface-form bias confirmed by the 2026-04-25 pairwise tournament:
3 of 7 matchups inverted (μ vs δ, smoke_v3_A vs δ, β vs α downgraded).

v2 changes:
1. Two-pass mandatory enumeration: claim_list + scaffold_list BEFORE scoring
2. New anchors keyed to substance density (RHS equations, named hyperparams,
   benchmark numbers) not feature presence
3. Anti-pattern penalty: -1/dim (capped, floored at 1) for hollow scaffolds
4. Output JSON expanded — but `per_dim_scores` keeps v1-compatible
   {math, novelty, realism, rigor} keys so OpusAuditClient.collect_all()
   continues to work unchanged.

Acceptance gate (per user 2026-04-25): on the 7-matchup pairwise validation,
≥ 6/7 directional agreement AND all 3 v1-inverted matchups flip.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


DEPTH_AUDIT_PROMPT_V2 = """\
You are an expert research methodology reviewer auditing a research plan.
Your task is to score the plan on 4 dimensions, each 1-5 (integer), for a /20 total.

Reviewers like you have a known failure mode: rewarding STRUCTURAL SCAFFOLDING
("Pattern 1 / Pattern 2", numbered methodology lists, headed subsections)
even when the scaffolding is empty — no inline equation RHS, no specific
hyperparameter numbers, no concrete operator definitions. You must avoid this
failure mode by following a TWO-PASS procedure: first commit to a list of
verifiable claims AND a list of hollow scaffolds, THEN score using anchors
that explicitly reference your enumeration.

# Research Goal
{goal}

# Research Plan to Evaluate
{plan}

# Pass 1 — Mandatory Claim Extraction (do this BEFORE scoring)

Extract two disjoint lists.

A) `claim_list`: every VERIFIABLE specific factual claim. Include only if:
   - Equations have a full right-hand side
     (`L(θ) = -E[(r/max r) · log π_θ(a|s)]` ✓; `J(θ) is derived` ✗)
   - Hyperparameters have numbers (`LoRA rank 64`, `learning rate 2e-5`,
     `10K compute budget`, `Llama-3-8B`). "Adaptive temperature" ✗.
   - Named real benchmarks/datasets with cited prior numbers
     (`MATH benchmark, GPT-4 baseline 50.4%` ✓; `mathematical theorem proving` ✗)
   - Deterministic operators
     (`score(s) = α·R(s) + (1-α)·log(1+n(s))` ✓; `score combines value and
     uncertainty` ✗)

B) `scaffold_list`: every STRUCTURAL SCAFFOLD that lacks an inline mechanism.
   IMPORTANT: scaffolds are NOT just "Pattern N" labels. ANY vague mechanistic
   gesture WITHOUT a verifiable claim is a scaffold, even if the surrounding
   prose is plain (no numbered headers). Include all of:

   - "Pattern N", "Step N", or numbered methodology bullets that name a
     mechanism without RHS, operator, or numeric setting
     (e.g., `Pattern 5 — Adaptive Hyperparameters: hyperparameters are adjusted`)
   - Equations stated as placeholders (e.g., `J(θ) is derived` with no RHS;
     `we minimize the cross-entropy loss` with no formula)
   - Domains named without benchmark numbers
     (e.g., `evaluate on theorem proving`, `we test on algorithm design`)
   - **Vague prose hyperparameters with no numeric value**
     (e.g., `use a small learning rate`, `appropriate batch size`,
     `adequate compute`, `sufficient training steps`, `moderate temperature`)
   - **Mechanisms in passive voice or generic verbs with no agent/operator/value**
     (e.g., `update parameters via gradient ascent`, `gradients are corrected`,
     `buffer is pruned to top-K` where K unspecified, `select high-quality samples`,
     `apply policy gradient`, `rewards are weighted appropriately`)
   - **Generic algorithm names without instantiation**
     (e.g., `use REINFORCE`, `apply DPO`, `LoRA fine-tuning` without rank/alpha,
     `standard backprop` with no learning rate)

CRITICAL: a plan can have NO "Pattern N" labels yet still be all scaffold.
A plan that says "we use REINFORCE with a small learning rate to optimize the
policy on a buffer of past attempts" has 4 scaffolds (REINFORCE without
hyperparam, "small learning rate" unnamed, "policy" passive, "buffer" without
operator) and 0 verifiable claims — score it accordingly.

Each list entry MUST quote ≤25 words from the plan verbatim, so the
classification is auditable.

# Pass 2 — Score 4 Dimensions, 1-5 Each

Use Pass-1 enumerations as evidence. Anchors below reference SUBSTANCE
DENSITY relative to scaffold count, not surface presence.

## 1. Mathematical Formalism (1-5)
- 1: claim_list contains 0 equations with RHS
- 2: 1 standard formula RHS, no derivation
- 3: ≥1 adapted/novel formula RHS specific to this plan
- 4: full objective + a stated gradient or update rule (both with RHS)
- 5: 4 + a limit/convergence sketch or a one-character deviation from a named
     classical objective with justification
  Reference anchor (5/5): D3-canonical TTT-Discover plan with J_β entropic
  objective, ∇J_β policy gradient, adaptive β via KL budget, MAX-PUCT one-
  character deviation from AlphaZero scores 5/5/5/5.

## 2. Algorithmic Novelty (1-5)
- 1: standard toolkit named without specific operator
- 2: known techniques with minor adaptation; ≥1 deterministic operator in claim_list
- 3: non-obvious combination with ≥2 deterministic operators + clear justification
- 4: substantive modification with at least one novel-looking operator
- 5: novel algorithm design with rigorous justification

## 3. Implementation Realism (1-5)
- 1: 0 specific hyperparameters in claim_list; multiple fatal errors
- 2: 1 specific hyperparameter; otherwise vague
- 3: 2-3 specific hyperparameters; one error or vague on key details
- 4: ≥4 specific hyperparameters covering model size, optimizer, compute
     budget, and an algorithm-specific knob (e.g., LoRA rank). All numerical
     claims internally consistent.
- 5: 4 + every key parameter justified or tied to a published baseline

## 4. Empirical Rigor (1-5)
- 1: 0 named real benchmarks in claim_list
- 2: ≥1 named real benchmark, no prior number
- 3: ≥1 named real benchmark with at least one prior baseline number cited
- 4: ≥2 named real benchmarks with published baseline numbers and comparison axis
- 5: 4 + a testable falsification criterion or open problem with prior AI numbers

# Anti-Pattern Penalty Rule (mandatory; applied AFTER raw scoring)

For each scaffold entry occupying a structural position in a dimension's
evidence, subtract -1 from that dimension's raw score (cap penalty at -1
per dimension; floor each final dim at 1):

- "Pattern N" / "Step N" mechanism in scaffold_list (not claim_list) on
  the math axis: -1 to Mathematical Formalism
- Pattern label whose operator is in scaffold_list (passive / no RHS):
  -1 to Algorithmic Novelty
- Hyperparameter named without a number ("adaptive temperature"): -1 to
  Implementation Realism
- Domain listed without a benchmark number: -1 to Empirical Rigor

The penalty is a sentinel (max -1/dim, total cap -4/20), not a multiplier.
Final dim score = max(1, raw - penalty).

# Output Format

Respond with ONLY this JSON (no other text, no markdown fence):

{{"claim_list": [
    {{"kind": "equation"|"hparam"|"benchmark"|"operator", "quote": "<≤25 words>"}}
  ],
  "scaffold_list": [
    {{"kind": "pattern_label"|"placeholder_eq"|"unnamed_domain"|"passive_mechanism",
      "quote": "<≤25 words>",
      "penalized_dim": "math"|"novelty"|"realism"|"rigor"|null}}
  ],
  "per_dim_raw":     {{"math":<int>, "novelty":<int>, "realism":<int>, "rigor":<int>}},
  "per_dim_penalty": {{"math":<0|1>, "novelty":<0|1>, "realism":<0|1>, "rigor":<0|1>}},
  "per_dim_scores":  {{"math":<int>, "novelty":<int>, "realism":<int>, "rigor":<int>}},
  "total": <int 4-20>}}
"""


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL | re.MULTILINE)
_FIRST_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_audit_v2_response(text: str) -> dict[str, Any]:
    """Parse Opus v2 audit JSON, robust to markdown fences and partial fields.

    Returns a dict with keys: claim_list, scaffold_list, per_dim_raw,
    per_dim_penalty, per_dim_scores, total. Missing fields filled with
    defaults (empty lists, all-1 scores) and a `_parse_warnings` list.
    """
    warnings: list[str] = []
    raw = text.strip() if text else ""

    # Strip markdown fence if present
    m = _FENCE_RE.search(raw)
    if m:
        raw = m.group(1).strip()
        warnings.append("stripped_markdown_fence")

    # Try direct json parse first
    parsed: dict[str, Any] | None = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting first {...} block
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
        # Total parse failure — return null scores
        logger.warning("audit_v2 parse failed; raw=%r", raw[:200])
        return {
            "claim_list": [], "scaffold_list": [],
            "per_dim_raw":     {"math": None, "novelty": None, "realism": None, "rigor": None},
            "per_dim_penalty": {"math": 0, "novelty": 0, "realism": 0, "rigor": 0},
            "per_dim_scores":  {"math": None, "novelty": None, "realism": None, "rigor": None},
            "total": None,
            "_parse_warnings": warnings,
            "_raw": raw[:500],
        }

    # Extract per_dim_scores robustly
    per_dim = parsed.get("per_dim_scores") or {}
    if not isinstance(per_dim, dict):
        per_dim = {}
        warnings.append("per_dim_scores_not_dict")
    for k in ("math", "novelty", "realism", "rigor"):
        v = per_dim.get(k)
        if not isinstance(v, int) or v < 1 or v > 5:
            per_dim[k] = None
            warnings.append(f"missing_or_invalid_{k}")

    total = parsed.get("total")
    if not isinstance(total, int):
        # Compute from per_dim if all present
        try:
            total = sum(per_dim[k] for k in ("math", "novelty", "realism", "rigor"))
        except (TypeError, KeyError):
            total = None
            warnings.append("total_unrecoverable")

    out = {
        "claim_list": parsed.get("claim_list", []) or [],
        "scaffold_list": parsed.get("scaffold_list", []) or [],
        "per_dim_raw": parsed.get("per_dim_raw", {}) or {},
        "per_dim_penalty": parsed.get("per_dim_penalty", {}) or {},
        "per_dim_scores": per_dim,
        "total": total,
    }
    if warnings:
        out["_parse_warnings"] = warnings
    return out
