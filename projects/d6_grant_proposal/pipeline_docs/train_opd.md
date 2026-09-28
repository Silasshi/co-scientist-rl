# train_opd

**Source:** `src/co_scientist/d6_grant_proposal/train_opd.py`
**Stage:** Refine — main contribution arm (OPD trainer)

## Purpose

The OPD trainer is **the main paper contribution** for D6. Implements D5's option-(c) HER SDPO loop adapted to grant proposals: each iteration samples N student rollouts, sends each to the Opus 4.7 reviewer with the **privileged** `reference_proposal.md` as gold context, receives a structured `<critique>` per plan, then updates the student via teacher–student logprob delta on tokens that match the SDPO `solution_only_mask`.

The 2026-04-30 simplified recipe (shipped for the ICLR workshop paper) has **three components**: solution-only mask, privileged-Opus teacher–student logprob offset, PPO-clip loss. The trust-region α-blend toward the frozen base policy was removed — see `../DECISIONS.md`.

## Key callables

| Callable | Purpose |
|---|---|
| `class Config` (`@chz.chz`) | Goal, paths, model, sampling, SDPO knobs (see below) |
| `_decode_plan(seq, tokenizer, renderer) -> str` | Token-id → text decoder |
| `find_solution_content_span(text) -> (start, end) \| None` | Locate `<solution>...</solution>` for masking |
| `build_solution_token_mask_from_tokens(...)` | Token-level mask matching the solution span |
| `_safe_pos_frac`, `_safe_mean_abs` | Advantage diagnostics |
| `main(config: Config)` | Full training loop: sample → critique → SDPO update → audit → save |

## OPD components (paper recipe)

| Component | Default | Effect |
|---|---|---|
| `solution_only_mask` | `True` | Zero advantage outside the last `<solution>...</solution>` body — prevents gradient on goal/oracle/critique tokens |
| Privileged-Opus offset | (always on) | Per-token advantage = `clamp(teacher_lp - student_lp, ±opd_anchor_clip)`; teacher gets the structured `<critique>` from a reviewer that saw `reference_proposal.md` as privileged ground truth |
| `loss_fn_name` | `"ppo"` | PPO-clipped surrogate (vs `"importance_sampling"` baseline) |
| `ppo_clip_eps` | `0.2` | Standard PPO clipping epsilon |
| `opd_anchor_clip` | `5.0` | Clip per-token advantage magnitude (safety) |

## OPD/SDPO advantage formula

```
A[t] = clamp(sdpo_scale * (lp_teacher[t] - lp_student[t]), ±opd_anchor_clip)
```

where teacher context = `goal + slim_oracle + critique` (current rollout, generated this iter) and student context = `goal + slim_oracle`. Critique conditioning is the only differentiator between teacher and student. The student never sees `reference_proposal.md`; only the Opus reviewer does.

## Reviewer

Opus 4.7 via the file-bus subagent pattern (no OpenRouter). Receives `reference_proposal.md` as privileged context — the mechanism that makes OPD "OPD-style" rather than blind self-distillation.

## I/O

- Input: `dataset/{goal}/research_goal.md`, `dataset/{goal}/reference_proposal.md`, `data/{goal}/oracle/slim.md`
- Per-iter output under `runs/<log_path>/`: `buffer.jsonl`, `metrics.jsonl`, `checkpoints.jsonl`, `critic_requests/`, `critic_responses/`, `audit_requests/`

## Dependencies

- `tinker` + `tinker_cookbook` (training loop, LoRA)
- Opus subagent file-bus client (reviewer)
- `co_scientist.shared.grant_signal_reward` (eval grader for `n_eval_plans`)
- `co_scientist.d6_grant_proposal.prompts` (student/teacher/critic payloads)

## Run

```bash
PYTHONPATH=src python -m co_scientist.d6_grant_proposal.train_opd \
    goal_domain=ai goal_name=02_foundational_rl \
    log_path=projects/d6_grant_proposal/runs/2026_04_30_pilot_foundrl_opd
```

Or via JSON config: `configs/pilot_foundrl_v8.json`.

## Capability trained

Generation policy that produces grant proposals scoring closer to the privileged reference under the 12-signal rubric, *without* the policy ever seeing `reference_proposal.md` directly at inference.

## Decision rule (paper)

OPD ≥ baseline + 0.04 weighted-mean delta over 3 seeds at iter 25 → main claim corroborated. See `/home/silas/.claude/plans/federated-mixing-donut.md` Phase 3.
