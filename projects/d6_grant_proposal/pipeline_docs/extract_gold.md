# extract_gold

**Source:** `src/co_scientist/d6_grant_proposal/extract_gold.py` (~146 lines)
**Stage:** Extract (one-time per goal, OPD-grounded only)

## Purpose

Extract verifiable equations and citations from `reference_proposal.md` to produce a `gold.json` that the OPD-grounded grounded trainer matches against during rollout to compute per-token grounding bonus. Pure regex + LaTeX parsing — no model calls.

## Key callables

| Callable | Purpose |
|---|---|
| `class Config` (`@chz.chz`) | `goal_domain`, `goal_name`, output path |
| `_extract_equations(text: str) -> list[dict]` | Regex-match `$...$`, `$$...$$`, `\begin{equation}...`, `\[...\]`; normalize via `grounding._normalize_eq` |
| `_extract_citations(text: str) -> list[dict]` | Regex-match `(Author et al., YYYY)`, `[N]`, `Author (YYYY)` patterns |
| `main(config: Config)` | Read reference → extract → write `gold.json` |

## I/O

- Input: `dataset/{goal_domain}/{goal_name}/reference_proposal.md`
- Output: `data/{goal_domain}/{goal_name}/gold/gold.json`
- Schema: `{"equations": [{normalized, raw, span}, ...], "citations": [{key, raw, span}, ...]}`

## Domain rule

- `goal_domain ∈ {ai, natural_science}` → equations + citations both extracted
- `goal_domain == social_science` → `equations: []` (G12a Analytical Framework mode; governance / law proposals don't have display math). `train_opd_grounded.py` and `grounding` follow the same rule.

## Dependencies

- Stdlib only (`re`, `json`)
- (Reuses regex patterns conceptually similar to `grounding._extract_eq_candidates`, but applied to gold reference rather than rollout text)

## Run

```bash
PYTHONPATH=src python -m co_scientist.d6_grant_proposal.extract_gold \
    goal_domain=ai goal_name=02_foundational_rl
```

## Capability

Pure feature extraction — no training or scoring. Required by OPD-grounded only; not needed for OPD / GRPO / baseline.

## When to re-run

When `reference_proposal.md` changes, or when equation / citation regexes are tightened.
