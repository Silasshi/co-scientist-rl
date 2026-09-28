# extract_oracle

**Source:** `src/co_scientist/d6_grant_proposal/extract_oracle.py` (~119 lines)
**Stage:** Extract (one-time per goal)

## Purpose

Run a single Opus 4.7 call against `reference_proposal.md` to extract typed oracle items in 6 categories — the "what an expert reviewer would point out" signal that gets distilled into the SDPO teacher. Output is consumed downstream by `extract_oracle_slim` and the OPD trainer.

## Key callables

| Callable | Purpose |
|---|---|
| `class Config` (`@chz.chz`) | `goal_domain`, `goal_name`, `dataset_base`, `data_base`, output paths |
| `_parse_items(raw: str) -> list[dict]` | Parse Opus's structured response into typed item records |
| `main(config: Config)` | Orchestrate: load reference → render `oracle/extract_items.md` → call Opus → write `oracle.jsonl` |

## I/O

- Input: `projects/d6_grant_proposal/dataset/{goal_domain}/{goal_name}/reference_proposal.md`
- Prompt: `projects/d6_grant_proposal/prompts/oracle/extract_items.md` (loaded via `prompt_loader`)
- Output: `projects/d6_grant_proposal/data/{goal_domain}/{goal_name}/oracle/oracle.jsonl`
- Each oracle row: `{category, content, source_span?}`

## Item categories

The extract prompt asks Opus for items in 6 typed buckets covering specific aims, evidence claims, methodological choices, deliverables, risks, and writing-style features. Exact category names live in `prompts/oracle/extract_items.md` (single source of truth).

## Dependencies

- `co_scientist.shared.api_profiles.create_service_client` — Anthropic client factory
- `anthropic.Anthropic` — for the Opus call
- `co_scientist.d6_grant_proposal.prompt_loader.load_prompt`

## Run

```bash
PYTHONPATH=src python -m co_scientist.d6_grant_proposal.extract_oracle \
    goal_domain=ai goal_name=02_foundational_rl
```

## Capability

Does not train or evaluate. Produces the privileged signal the OPD teacher conditions on.

## When to re-run

Only when `reference_proposal.md` changes for that goal, OR when `prompts/oracle/extract_items.md` is edited. Outputs are cacheable per `(goal, prompt_version)`.
