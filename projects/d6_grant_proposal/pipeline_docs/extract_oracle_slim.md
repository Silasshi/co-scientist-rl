# extract_oracle_slim

**Source:** `src/co_scientist/d6_grant_proposal/extract_oracle_slim.py` (~100 lines)
**Stage:** Extract (one-time per goal, after `extract_oracle`)

## Purpose

Take the full `oracle.jsonl` and produce a markdown-formatted, token-budgeted "slim oracle" suitable for inclusion in the STUDENT and TEACHER prompts at training time. The full oracle is too long to fit in every rollout context; the slim version is ≈ 3k tokens and groups items by category with section headings.

## Key callables

| Callable | Purpose |
|---|---|
| `class Config` (`@chz.chz`) | `goal_domain`, `goal_name`, paths, `max_tokens` budget |
| `_format_items(items: list[dict]) -> str` | Group by category, emit markdown sections, truncate to budget |
| `main(config: Config)` | Read `oracle.jsonl` → `_format_items` → write `slim.md` |

## I/O

- Input: `data/{goal_domain}/{goal_name}/oracle/oracle.jsonl`
- Output: `data/{goal_domain}/{goal_name}/oracle/slim.md`

## Token budget

Default ~3k tokens. The trainers (OPD, OPD-grounded, baseline, GRPO) all read this file via `_oracle_path(config)` helpers and inject into `build_student_prompt(oracle_abstraction=...)`.

## Dependencies

Standard library only (`json`, `pathlib`).

## Run

```bash
PYTHONPATH=src python -m co_scientist.d6_grant_proposal.extract_oracle_slim \
    goal_domain=ai goal_name=02_foundational_rl
```

## Capability

Pure data shaping — no model calls.

## When to re-run

When `oracle.jsonl` is regenerated, or when the token budget needs adjusting.
