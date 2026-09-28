# D5 Step E pairwise: τ_v4_clean vs σ_v4 (rerun on patched plans)

*Generated 2026-04-27 01:05:48*  
*Seed: 52, n_pairs: 8*  
*Comparison to Step 2.5 verdict: τ_v4 (orig, bug+lucky) vs σ_v4 = 7-1 STRONG*

## Aggregate

- **τ_v4_clean wins**: 4 / 8
- **σ_v4 wins**: 4 / 8
- **Ties**: 0
- **Unjudged**: 0
- **A-position wins**: 4 / 8 (50%) — should be ~50% (no position bias)

## Verdict: **TIE**

## Per-pair verdicts

| pair | A | B | winner_pos | winner_label | rationale |
|---|---|---|---|---|---|
| pair_00 | tau_v4_clean | sigma_v4 | A | tau_v4_clean | Plan A specifies concrete hyperparameters (LoRA r=4, exponential utility beta=8) and a quantitative stopping rule (TV-distance bounds), where Plan B leaves rank, beta, and stopping unspecified, giving |
| pair_01 | sigma_v4 | tau_v4_clean | A | sigma_v4 | Plan A provides materially deeper concrete specs (GRPO with risk-sensitivity beta=4-8, SIFT buffer selection, named 67-problem Georgiev benchmark, Pass@1 and Pass@k targets with specific deltas like 1 |
| pair_02 | tau_v4_clean | sigma_v4 | A | tau_v4_clean | Plan A provides substantively more concrete specs (LoRA rank 64, K=10, beta=8, 100-150 search steps, 5.1B active params) and names specific prior methods (TGD, TTIA, RS-GRPO, EvoTune, RLVR) with a qua |
| pair_03 | sigma_v4 | tau_v4_clean | A | sigma_v4 | Plan A is grounded in a richer set of named prior works with concrete numbers (TTRL 40.2% AIME, AlphaProof, SIFT, Hardt & Sun per-problem adaptation) and introduces a substantive structural prior (pro |
| pair_04 | tau_v4_clean | sigma_v4 | B | sigma_v4 | Plan B has stronger grounding (real author citations: Akyürek 2024, Georgiev 2025, Jiang 2025, Bagatella 2025, Wang 2025, Guo 2025) versus Plan A's vague internal '(Math 9)/(Methodology 10/14/22)' tag |
| pair_05 | sigma_v4 | tau_v4_clean | B | tau_v4_clean | Plan B provides materially deeper concrete-spec depth: an explicit risk-sensitive objective J_RS = (1/β)log E[exp(β·R)] with β=8, LoRA inner-loop hyperparameters (η=0.01, batch=16), a population size  |
| pair_06 | tau_v4_clean | sigma_v4 | B | sigma_v4 | Plan B has stronger concrete-spec depth (β=4 for RS-GRPO, 400k variants, 100K program evals, 10K RL steps, specific per-benchmark targets 73.6% MATH-500 and 77.9% AIME2024) and grounds its targets in  |
| pair_07 | sigma_v4 | tau_v4_clean | B | tau_v4_clean | Plan B provides more concrete spec depth (LoRA rank=64, J_RS β=8, KL-regularized RPG-Style Clip with Schulman 2017 citation) and a stronger disentanglement design with three ablations including a rand |
