# Literature retrieval and verifier-grounded rewards

## Retrieval — [`src/co_scientist/shared/paper_retrieval.py`](../../src/co_scientist/shared/paper_retrieval.py)
- **Forward search.** The policy writes search queries ([`extract_search_queries` L586](../../src/co_scientist/shared/paper_retrieval.py#L586)). They run against **OpenAlex**, which needs no API key ([`PaperRetriever` L82](../../src/co_scientist/shared/paper_retrieval.py#L82)). Abstracts are rebuilt from OpenAlex's inverted-index format ([L70](../../src/co_scientist/shared/paper_retrieval.py#L70)).
- **Backward search.** Given an arXiv paper, its full reference list is fetched from **Semantic Scholar** ([`S2BackwardRetriever` L255](../../src/co_scientist/shared/paper_retrieval.py#L255)). The retriever respects S2's rate limit.
- **Citation-match rate** ([L570](../../src/co_scientist/shared/paper_retrieval.py#L570)) measures whether a plan actually cites what it retrieved.

D5 adds an "oracle abstraction" built by parsing the source paper's LaTeX and `.bib`, then extracting its insights over several rounds (see [`experiments/d5_abstract_retrieve_refine/`](../../experiments/d5_abstract_retrieve_refine/): `extract_cite_keys_v1`, `parse_bibtex_v1`, `build_oracle_v2`, `build_slim_oracle_v2`).

## Verifier-grounded reward
**Problem it fixes.** The self-distillation signal rewarded *conformity to the critique* rather than *fidelity to the source* (D5 finding F16, see [self-distillation](02-self-distillation.md)). The fix adds a sparse, programmatically verified bonus on the exact token spans where a plan reproduces a gold equation or citation.

| | D5 research plans | D6 grant proposals |
|---|---|---|
| Code | [`d5_abstract_retrieve_refine/verifier_grounded_reward_v1.py`](../../src/co_scientist/d5_abstract_retrieve_refine/verifier_grounded_reward_v1.py) | [`d6_grant_proposal/grounding.py`](../../src/co_scientist/d6_grant_proposal/grounding.py) |
| Equation match | sympy normalisation, with a fuzzy key-token fallback (only 26.7% of equations parsed) | regex + token-set Jaccard ≥ 0.5 ([L125–L160](../../src/co_scientist/d6_grant_proposal/grounding.py#L125)) |
| Credit assignment | character spans → token spans ([L285](../../src/co_scientist/d5_abstract_retrieve_refine/verifier_grounded_reward_v1.py#L285)) | `compute_grounding_bonus_spans` |
| Goodhart guard | per-equation bonus cut to 0.1 | `check_grounding_saturation` (cliff guard) |

## Result (D5, single goal, Opus audit / 45)
- **The grounded reward alone Goodharted.** The plans maxed out the equation axis (5.00) while citations fell to 2.88 and empirical grounding to 0.
- **Adding a KL anchor toward an SFT'd oracle policy stabilised training but did not raise the peak.**
  - Its peak was 23.38, against 23.50 for the unanchored run.
  - It held 23+ through the last iterations, while the unanchored run fell to 14.75.
  - Its mean over 16 iterations (20.91) was the highest of all cells.

Source: [`D5 DECISIONS.md`, 2026-04-29](../../projects/d5_abstract_retrieve_refine/DECISIONS.md).
