# signals

**Source:** `src/co_scientist/d6_grant_proposal/signals.py` (~35 lines)
**Stage:** Infrastructure (cross-cutting)

## Purpose

Single-line re-export of D4's 12-signal rubric from `co_scientist.shared.grant_signal_reward`. Keeps the D6 trainers from importing across direction boundaries. The score scale, signal IDs, and weights remain the D4 conventions.

## Key callables / re-exports

| Re-export | Origin | Purpose |
|---|---|---|
| `SIGNAL_DEFINITIONS` | `shared.grant_signal_reward` | List of 12 signal IDs + names + weights |
| `build_grader_prompt(goal, plan_text, domain) -> str` | shared | Per-signal Qwen grader prompt |
| `build_single_call_prompt(...)` | shared | Multi-signal one-shot grader (used by baseline, GRPO) |
| `parse_scores(raw: str) -> dict` | shared | Parse `<evaluation>` XML → `{signal_id: score}` |
| `weighted_aggregate(scores: dict, domain: str) -> float` | shared | Apply per-domain weights, return 0.0–1.0 |

## 12 signals (D4 v8 redesign)

| ID | Name | Weight |
|---|---|---|
| G1 | Problem Specificity | 0.04 |
| G2 | Specific Aims | 0.04 |
| G3 | Technical Evidence | 0.10 |
| G4 | Research Focus | 0.12 |
| G5 | Gap Identification | 0.04 |
| G6 | Reasoning Depth | 0.12 |
| G8 | Deliverable Clarity | 0.08 |
| G9 | Scope Feasibility | 0.08 |
| G10 | Approach Coverage | 0.04 |
| G11 | Evidence Rigor | 0.12 |
| G12 / G12a | Mathematical Formalism / Analytical Framework | 0.12 |
| G13 | Risk Awareness | 0.10 |

`G12` is used in STEM domains (`ai`, `natural_science`); `G12a` (Analytical Framework) replaces it for `social_science`. Auto-detected by domain string.

## Dependencies

- `co_scientist.shared.grant_signal_reward` (the actual implementation)

## Capability

Pure re-export — no logic added in D6.

## Why a re-export instead of direct import?

CONVENTIONS.md "Direction separation": D6 modules import from `co_scientist.d6_grant_proposal` only, with `shared/*` access mediated via this thin shim. Makes future migration easier (e.g., if D6 forks a divergent rubric, only `signals.py` needs to change).
