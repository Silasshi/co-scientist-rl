# D1 Rubric vs Opus Depth — Within-Goal Audit (n=8)

**Goal**: PbRL feedback efficiency — identify root cause (ground truth in D1 rubric = "query-policy misalignment": queried segments fall outside visitation distribution of current policy).

**Data**: 8 rollouts from a single D1 training group, rubric scores spanning 0.04–0.98.

**Protocol**: Plans read blind (no rubric score visible) into 4-dim × 1-5 D3 canonical depth audit (math, novelty, realism, rigor → /20). Rubric revealed after all 8 scores finalized.

## Scores

| sample_idx | rubric | math | novelty | realism | rigor | Opus /20 | Short verdict |
|---|---|---|---|---|---|---|---|
| 4 | 0.983 | 3 | 3 | 2 | 2 | **10** | UAQ grad-alignment objective, correct framing |
| 6 | 0.846 | 3 | 3 | 3 | 2 | **11** | PAQS grad-difference, Christiano cited correctly |
| 2 | 0.731 | 2 | 2 | 2 | 2 | 8 | DVQS toy grid-world value-diff |
| 0 | 0.486 | 2 | 2 | 2 | 2 | 8 | PFR generic weighted-mean |
| 3 | 0.200 | 3 | 2 | 2 | 2 | 9 | CQS calibration, has regret theorem |
| 5 | 0.169 | 1 | 2 | 2 | 2 | 7 | No equations at all |
| 7 | 0.071 | 2 | 2 | 2 | 2 | 8 | MI-based, implementation error ("32-hidden-layer PPO") |
| 1 | 0.040 | 3 | 2 | 2 | 2 | 9 | C-PbRL BNN + submodular, real citations |

## Correlation

| Metric | Value | p (n=8) |
|---|---|---|
| Pearson r (rubric vs Opus) | **0.577** | 0.134 |
| Spearman ρ | **0.479** | 0.230 |

Neither statistically significant at n=8. The correlation is driven almost entirely by the two top plans (idx 4, 6) being both rubric-top and Opus-top. Below that, rubric and Opus are essentially uncorrelated.

## Key inversions

**High rubric, low Opus** (rubric rewards without depth):
- **idx 2**: rubric 0.73, Opus 8 — grid-world 10×10 toy setup, single value-diff formula. Gets high rubric because it superficially aligns with "value-aware query selection"
- **idx 0**: rubric 0.49, Opus 8 — generic weighted-mean metric with no derivation, fabricated 40–60% improvement claims

**Low rubric, high Opus** (depth penalized for wrong framing):
- **idx 1** (rubric=0.04, Opus=9): most methodologically substantive — Bayesian reward model with proper BNN loss + submodular diversity score + correct Bayes-by-Backprop / Mirzasoleiman citations. Punished because it framed root cause as "reward miscalibration + query redundancy" rather than "misalignment"
- **idx 3** (rubric=0.20, Opus=9): regret theorem O(√T log T), temperature scaling (Platt 2000), concrete AdamW hyperparams. Punished for "calibration errors" framing

## Why the rubric misses depth

The 10 rubric items collapse into:

1. **Items 1, 2, 4, 5, 6, 7, 9, 10** — all require "query-policy misalignment" framing. A plan with full derivation but alternative root-cause hypothesis scores 0 on these. This is ~80% of rubric weight.
2. **Items 3, 8** — "analyzes uniform/disagreement schemes", "grounded in PbRL context". Trivially satisfied by all 8 plans → no discrimination.
3. **Zero items reward** mathematical formalism, novel algorithms, real published baselines, or reproducible numerical claims.

Consequence: the rubric is a **topic/framing classifier**, not a quality estimator. Rubric score 0.04 vs 0.98 measures "did the plan adopt the specified thesis framing", not "is this plan any good".

## Implication for paper (grader-gap evidence baseline)

D1 training signal validity, as measured here:

- **At the extreme top** (rubric ≥ 0.85): rubric and Opus agree. Rubric-top plans DO also tend to be slightly better methodologically (because engaging the problem framing correlates weakly with taking the problem seriously).
- **Below the top** (rubric 0.04–0.73): rubric and Opus disagree. Plans with substantive math but wrong framing are scored near 0; plans with thin math but correct framing are scored near 0.7.
- **Whole-corpus depth ceiling**: Opus scores span only 7–11 of 20. No plan reaches 12/20 — the entire D1 training corpus is shallow on formalism, novelty, realism, rigor axes. The rubric's 0.04→0.98 spread is not measuring the axis of quality that Opus/human reviewers care about.

**Recommendation**: D1 rubric reward should be framed in the paper as "topic adherence" or "thesis alignment", not "plan quality". A model trained on this reward learns to paraphrase "query-policy misalignment" and propose any metric-with-that-name — regardless of whether the plan is methodologically sound. This is consistent with the observed D1 template collapse (MEMORY: B4 ≥ MAIN on Opus at 30B) and explains why the 0.693 D1 eval ceiling proved impossible to raise with better RL — the reward is orthogonal to the axis of improvement.
