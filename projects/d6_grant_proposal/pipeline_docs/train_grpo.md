# train_grpo

**Source:** `src/co_scientist/d6_grant_proposal/train_grpo.py` (~497 lines)
**Stage:** Refine — GRPO ablation arm (no reviewer)

## Purpose

Reward-only baseline. Same Qwen3-30B-A3B policy and same training infrastructure as OPD, but **no Opus reviewer and no SDPO**. Each iteration samples a group of N rollouts, scores each with the Qwen3-30B 12-signal grader (`shared/grant_signal_reward.py`), and applies group-relative policy optimisation: per-rollout advantage = `(reward - group_mean) / (group_std + ε)`. This isolates the value of the Opus reviewer signal — OPD vs GRPO is the head-to-head paper claim.

## Key callables

| Callable | Purpose |
|---|---|
| `class Config` (`@chz.chz`) | Goal, paths, `n_rollouts_per_group=4`, grader settings, advantage clip |
| `_active_signals(goal_domain) -> list` | Domain-aware 12-signal set |
| `_grade_plan(plan_text, ...) -> dict` | Single-call multi-signal grade via Qwen3-30B |
| `main(config: Config)` | Sample group → grade → group-relative advantage → PPO-clipped update |

## GRPO advantage formula

```
A_i = clamp((reward_i - mean(group_rewards)) / (std(group_rewards) + adv_norm_eps), ±reward_clip)
```

Then standard PPO-clipped policy update over `n_grad_steps_per_iter` (default 4).

## Knobs (mirrors OPD where comparable)

| Knob | Default | Note |
|---|---|---|
| `n_rollouts_per_group` | `4` | Group size for normalization |
| `reward_clip` | `5.0` | Advantage magnitude bound |
| `adv_norm_eps` | `1e-8` | Numerical floor in denominator |
| `solution_only_mask` | `True` | ****Same as OPD**** — token mask outside `<solution>` |
| `ppo_clip_eps` | `0.2` | **Same as OPD** |
| `loss_fn_name` | `"ppo"` | **Same as OPD** |
| `grader_max_tokens` | `8192` | Qwen grader budget |
| `grader_temperature` | `0.0` | Deterministic grading |

The training-loop machinery is intentionally as close to OPD as possible — the only difference is the reward source (Qwen 12-signal aggregate vs Opus teacher–student logprob delta).

## I/O

- Input: same as OPD minus `reference_proposal.md` (no reviewer call)
- Output: `runs/<log_path>/{buffer.jsonl, metrics.jsonl, checkpoints.jsonl, config.json}`. No `critic_requests/` (no reviewer).

## Dependencies

- `tinker` + `tinker_cookbook`
- `co_scientist.shared.grant_signal_reward` — `build_single_call_prompt`, `parse_scores`, `weighted_aggregate`
- `co_scientist.d6_grant_proposal.prompts.build_student_prompt`

## Run

```bash
PYTHONPATH=src python -m co_scientist.d6_grant_proposal.train_grpo \
    goal_domain=ai goal_name=02_foundational_rl \
    log_path=projects/d6_grant_proposal/runs/2026_04_30_pilot_foundrl_grpo
```

Or via JSON: `configs/pilot_foundrl_grpo.json`.

## Capability trained

Generation policy optimised for the **Qwen-30B 12-signal aggregate**. Exposes Goodhart on the rubric: any axis the grader rewards but Opus doesn't. The paper thesis is that the ρ ≈ 0.40 Qwen-30B-vs-Opus correlation (per `grader_panel_v8`) caps GRPO's ceiling — Phase 2 of the roadmap quantifies this on the new D6 signal-validation dataset.

## Why this ablation matters

If OPD ≫ GRPO, the Opus reviewer signal is load-bearing — feedback > scalar reward when the reward is imperfect. If OPD ≈ GRPO, the OPD win comes from the training-loop structure (sol-mask, PPO-clip) rather than the reviewer.
