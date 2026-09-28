# train_baseline

**Source:** `src/co_scientist/d6_grant_proposal/train_baseline.py` (~227 lines)
**Stage:** Refine — frozen baseline (no gradient updates)

## Purpose

The frozen-baseline arm of the pilot. Loads a frozen Qwen3-30B-A3B, samples N plans against the slim oracle, scores each via the Qwen3-30B 12-signal grader, and writes results to `buffer.jsonl`. **No training, no LoRA updates** — this establishes the lower-bound score that OPD and GRPO must beat.

## Key callables

| Callable | Purpose |
|---|---|
| `class Config` (`@chz.chz`) | `goal_domain`, `goal_name`, paths, `n_eval_plans=8`, `max_tokens=8192`, sampling params |
| `_active_signals(goal_domain) -> list` | Selects 12 signals (G12 vs G12a) per domain |
| `main(config: Config)` | Sample → grade via single-call grader → write `buffer.jsonl` + `metrics.jsonl` |

## I/O

- Input: `dataset/{goal}/research_goal.md`, `data/{goal}/oracle/slim.md`
- Output: `runs/<log_path>/buffer.jsonl`, `runs/<log_path>/metrics.jsonl`, `runs/<log_path>/config.json`

## Capability evaluated

The frozen-model lower bound. Any trainer claiming "improvement" must beat the baseline on the **same** audit protocol.

## Dependencies

- `tinker` + `tinker_cookbook` for sampling
- `co_scientist.shared.grant_signal_reward` for `build_single_call_prompt` + `parse_scores`
- `co_scientist.d6_grant_proposal.prompts.build_student_prompt`

## Run

```bash
PYTHONPATH=src python -m co_scientist.d6_grant_proposal.train_baseline \
    goal_domain=ai goal_name=02_foundational_rl \
    log_path=projects/d6_grant_proposal/runs/2026_04_30_pilot_foundrl_baseline
```

## Why no training?

Sigma's job is to answer "what does the model produce *before* any of the proposed methods touch it?". Conflating that with adapter loading or partial fine-tuning would make the comparison ambiguous.
