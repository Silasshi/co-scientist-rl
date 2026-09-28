# audit

**Source:** `src/co_scientist/d6_grant_proposal/audit.py` (~330 lines)
**Stage:** Feedback / Eval — isolated multi-plan audit via parallel Opus subagents

## Purpose

Independent of the training-time grader, this is the canonical evaluation script. Loads N plans (from `buffer.jsonl`, a frozen file, or by checkpoint+iter selectors), anonymizes them (strips run/iter labels so the auditor can't infer arm identity), partitions into `n_batches=8` balanced subagent batches, and dispatches each batch to a separate Opus 4.7 subagent that emits 12-signal scores (1–5 + reasoning) for every plan in the batch. Aggregates per-plan and per-arm.

## Key callables

| Callable | Purpose |
|---|---|
| `class Config` (`@chz.chz`) | `goal_domain`, `goal_name`, `out_path`, `baselines_json`, `n_batches=8`, `analyze: bool` |
| `_load_plans_frozen(run_dir, label) -> list[dict]` | Load all `buffer.jsonl` rows under a label |
| `_load_plans_eval(run_dir, label, key_field, key_value) -> list[dict]` | Filter by `iter` or `entry_type` |
| `_load_plans_file(file_path, label)` | Load from arbitrary file |
| `_build_audit_prompt(goal, plans, domain)` | Render `audit/absolute_score.md` with anonymized batch |
| `_parse_audit_response(raw) -> dict[str, dict]` | Parse Opus's structured output back into per-plan signal scores |
| `_analyze(out_dir)` | Read response files, aggregate per-arm, emit summary table |
| `main(config)` | Two-mode entry: dispatch (default) or `analyze=true` (post-hoc aggregation) |

## Isolation protocol (D5 audit_v3_isolated origin)

- Plans labelled internally only — output to auditor strips arm/iter/seed
- 8 parallel Opus subagents see disjoint plan batches; close-cluster verdicts (`Δ < 5/45` in D5 terms; D6 weighted-mean equivalent ≈ 0.04) require pairwise corroboration per the M8 standing protocol (see project CLAUDE.md "ML research scientist persona")
- Batch size balanced so no single subagent sees > 50% of any one arm — prevents systematic bias

## Domain rule

`audit` auto-selects G12 vs G12a from `goal_domain` (same logic as `signals`):
- `ai`, `natural_science` → G12 Mathematical Formalism
- `social_science` → G12a Analytical Framework

## I/O

- Input: `dataset/{goal}/research_goal.md` + a `baselines_json` describing which arms / iterations to load
- Prompt: `audit/absolute_score.md`
- Output (under `out_path`):
  - `audit_requests/<batch_id>.json` — per-batch payload sent to the subagent
  - `audit_responses/<batch_id>.json` — per-batch raw + parsed
  - `summary.jsonl` (after `analyze=true`) — per-(label, plan, signal) score table

## Dependencies

- `co_scientist.d6_grant_proposal.prompts.build_audit_request_payload`
- `co_scientist.d6_grant_proposal.signals` (signal weight table for aggregation)
- `co_scientist.d5_abstract_retrieve_refine.mu_prompts.extract_solution` (re-exported via prompts)
- Opus 4.7 file-bus subagent client

## Run

Two-step (dispatch + analyze):

```bash
# 1. Dispatch (creates audit_requests/, audit_responses/)
PYTHONPATH=src python -m co_scientist.d6_grant_proposal.audit \
    goal_domain=ai goal_name=02_foundational_rl \
    out_path=projects/d6_grant_proposal/runs/audit_pilot_iter25 \
    baselines_json='[{"label":"baseline","mode":"frozen","run":"runs/2026_04_30_pilot_foundrl_baseline"},{"label":"opd","mode":"eval","run":"runs/2026_04_30_pilot_foundrl_opd","key_field":"iteration","key_value":25},{"label":"grpo","mode":"eval","run":"runs/2026_04_30_pilot_foundrl_grpo","key_field":"iteration","key_value":25}]'

# 2. Analyze (aggregates + emits summary)
PYTHONPATH=src python -m co_scientist.d6_grant_proposal.audit \
    out_path=projects/d6_grant_proposal/runs/audit_pilot_iter25 analyze=true
```

## Capability evaluated

The 12-signal weighted-mean audit score (0.0–1.0) is the **primary metric** for the D6 paper. All cross-arm comparisons (OPD vs GRPO vs baseline) are settled here — the training-time grader is for gradient signal only and should not be reported as a result.

## Note on grader divergence

Training-time grader = Qwen3-30B-A3B (cheap, ρ ≈ 0.40 with Opus per `grader_panel_v8`). Audit grader = Opus 4.7 (expensive, ground truth). Trust the audit, not the buffer score, for paper-level claims.
