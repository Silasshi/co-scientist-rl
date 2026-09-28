# grounding

**Source:** `src/co_scientist/d6_grant_proposal/grounding.py` (~261 lines)
**Stage:** Retrieve (cross-cutting; called by OPD-grounded trainer)

## Purpose

Match policy-rolled-out text against the gold equations + citations from `extract_gold`. For each match, compute a token-span and a per-token bonus that the OPD-grounded trainer adds onto the SDPO advantage. Equations are matched via Jaccard similarity over normalized math tokens (cliff-thresholded to avoid Goodhart); citations are matched via key regex.

## Key callables

| Callable | Purpose |
|---|---|
| `load_gold(gold_path) -> dict` | Read `gold.json` |
| `_normalize_eq(text: str) -> str` | Strip whitespace + `$`, lowercase, normalize symbols |
| `_math_tokens(normalized: str) -> set[str]` | Tokenize for Jaccard |
| `_jaccard(a: set, b: set) -> float` | Standard Jaccard similarity |
| `_extract_eq_candidates(text: str) -> list[(start, end, raw)]` | Regex over rollout text |
| `compute_grounding_bonus_spans(plan_text, gold, threshold=0.85, per_eq_bonus, per_cit_bonus) -> list[(start, end, bonus, kind)]` | Main API: returns scored spans |
| `check_grounding_saturation(spans, ...) -> bool` | Cliff guard — if bonus density exceeds threshold, freeze to prevent runaway |
| `build_grounding_token_mask(spans, tokens, ...) -> np.ndarray` | Convert char spans → per-token bonus array |

## Anti-Goodhart guards

- `cliff_threshold` (default 0.85): if a single equation Jaccard ≥ threshold, stops rewarding further matches in the same sentence — prevents copy-paste attacks
- Per-token bonus is bounded; total grounding contribution is clipped before adding to advantage (handled in trainer)

## Domain rule

`equations` channel disabled for `social_science` (gold has `equations: []`).

## Dependencies

- `numpy`
- Stdlib (`re`, `json`)

## Capability

Auxiliary reward signal for OPD-grounded; does not replace the SDPO logprob-delta advantage — adds to it.

## Used by

- `train_opd_grounded.py`
- (Future) `train_opd_kl_anchor.py` — same grounding bonus, additional KL anchor toward π_oracle SFT
