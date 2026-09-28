# D5 μ-v4 Pairwise Corroboration of audit_v3 ISOLATED +2.75

*Generated 2026-04-26 07:51:59*  
*Seed: 50, n_pairs/matchup: 8*  
*All matchups under test-time-fair conditions: NO reference plan in either prompt; both sides see goal+oracle.*

## Cross-check question

Phase 2F audit_v3 ISOLATED ranked μ-v4 iter 4 = 28.00 ≫ σ = 25.25 ≈ δ = 25.12 ≈ α = 25.00. Does Opus pairwise judge corroborate this top-of-cluster ranking, or does the absolute audit have surface-form bias?

## 3 Pairwise Matchups

| Matchup | la wins | lb wins | ties | A-pos win rate | Winner |
|---|---:|---:|---:|---:|---|
| **mu_v4** vs **sigma** | 6 | 2 | 0 | 4/8 (50%) | **mu_v4** |
| **mu_v4** vs **delta** | 7 | 1 | 0 | 5/8 (62%) | **mu_v4** |
| **mu_v4** vs **alpha** | 7 | 1 | 0 | 6/8 (75%) | **mu_v4** |

## Cross-check verdict

**PASS** — μ-v4 wins 6-2 vs σ. The audit_v3 ISOLATED +2.75 absolute lift is corroborated by independent pairwise judgment. Paper-claim headline ('first 30B-trained variant to beat σ') is safe.

## Secondary matchups (close-cluster ranking)

- **muV4_vs_delta**: μ-v4 cleanly above delta (7-1)
- **muV4_vs_alpha**: μ-v4 cleanly above alpha (7-1)

## Position-bias sanity

A-pos win rate per matchup should be in 25-75% (else position bias; rerun with re-seed).

- muV4_vs_sigma: 4/8 (50%) ✓
- muV4_vs_delta: 5/8 (62%) ✓
- muV4_vs_alpha: 6/8 (75%) ✓

## Sample rationales (3 per matchup, first 300 chars)

### mu_v4 vs sigma

- **mu_v4_vs_sigma_pair_00** (winner=B, 743 chars): Plan B provides more concrete methodological specifics (β=4 exponential utility, ρ=0.2 latent scope, asymmetric clipping ε_low=0.8/ε_high=1.2) and grounds claims in specific prior numbers (6x ARC improvement, 211% AIME gain, 20% bit-per-byte reduction), whereas Plan A makes similar architectural cho
- **mu_v4_vs_sigma_pair_01** (winner=B, 712 chars): Plan B provides more concrete specifications (K=100 candidates, β=8, ρ=0.2, 10 iterations, LLaMA3.2-1B base model) and names specific benchmarks with prior numbers (AIME-2024 14.4% from Zuo, 67-problem portfolio from Georgiev), whereas Plan A stays at the level of generic component descriptions. Pla
- **mu_v4_vs_sigma_pair_02** (winner=A, 926 chars): Plan A provides more substantive technical detail: explicit hyperparameters (β=8 for RS-GRPO), concrete diversity mechanisms (SIFT data selection, latent-space diversification on first 20% of hidden layers), goal-specific reward shaping (double-ended grading via min over subgoals, dynamic clipping),

### mu_v4 vs delta

- **mu_v4_vs_delta_pair_00** (winner=A, 257 chars): A names concrete tools (TTRL, RS-GRPO beta=4, SIFT, TTIA rho=0.2, MiGrATe clipping eps=0.8/1.2), cites benchmark portfolio (67 problems), and ties each component to a citation. B is generic memory+sigmoid RL without scientific-domain specifics or citations.
- **mu_v4_vs_delta_pair_01** (winner=B, 319 chars): B cites Akyurek, Georgiev, Zuo, Hill, integrates 5 explicit methodological templates (TTRL curriculum, program-space, RS-GRPO beta=8, latent ρ=0.2 updates, SIFT), and specifies AIME-2024 + 67-problem benchmark with quantitative target (>50% pass@1). A is generic memory+curriculum with no citations o
- **mu_v4_vs_delta_pair_02** (winner=B, 270 chars): B (TTRL-Discover) names program-space code generation, RS-GRPO with beta=8, SIFT-based buffer, double-ended grading, and concrete baselines (TTA-RL, MiGrATe, AlphaProof). A is generic LoRA+curriculum scheduling without citations or named scientific-discovery mechanisms.

### mu_v4 vs alpha

- **mu_v4_vs_alpha_pair_00** (winner=B, 209 chars): B (mu_v4) gives concrete hyperparameters (beta=4, rho=0.2, asymmetric clipping bounds) and explicit failure-mode monitoring; A is structurally similar but less specific on RL knobs and program-space mechanism.
- **mu_v4_vs_alpha_pair_01** (winner=A, 192 chars): A (mu_v4) names K=100, beta=8, LLaMA3.2-1B, 10-iter budget, 67-problem portfolio, 5-step pipeline with SIFT and rho=0.2 latents. B is generic, names few specifics, and skimpier on methodology.
- **mu_v4_vs_alpha_pair_02** (winner=A, 200 chars): A (mu_v4) gives beta=8, double-ended grading, dynamic clipping, four named baselines (TTA-RL, MiGrATe, AlphaProof). B mentions Y-shaped trunk and LoRA but is shorter and less rigorous on RL specifics.

