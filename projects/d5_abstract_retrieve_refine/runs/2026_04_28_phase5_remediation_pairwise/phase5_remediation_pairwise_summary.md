# D5 Phase 5 Remediation — Pairwise Corroboration

*Generated 2026-04-26 19:23:09*  
*Seed: 60, n_pairs/matchup: 8*  
*Per DECISIONS 2026-04-26 (f). Strict 1 subagent/pair task isolation.*

## 3 Matchups (all close-cluster verdicts requiring corroboration)

| Matchup | Goal | la wins | lb wins | ties | A-pos rate | Winner |
|---|---|---:|---:|---:|---:|---|
| **muV5_v3_vs_muV4_tool_v** | (see below) | 4 (mu_v4) | 4 (mu_v5_v3) | 0 | 4/8 (50%) | **tie** |
| **phase4a_meta_ttl** | (see below) | 5 (sigma_prime) | 3 (mu_prime) | 0 | 5/8 (62%) | **sigma_prime** |
| **phase4a_tt_control** | (see below) | 5 (sigma_prime) | 3 (mu_prime) | 0 | 5/8 (62%) | **sigma_prime** |

## Verdicts vs original audit Δ

### Phase 5 v3 close-cluster
- Original audit Δ: +0.62 / 45 (within n=8 noise)
- Pairwise: **INCONCLUSIVE** — 4-4 (split).

### Phase 4a meta_ttl borderline
- Original audit Δ: +2.12 / 45 (just above +1.0 threshold)
- Pairwise: **INCONCLUSIVE** — 5-3 (split).

### Phase 4a tt_control borderline
- Original audit Δ: -0.88 / 45 (just below 0)
- Pairwise: **INCONCLUSIVE** — 5-3 (split).

## Position-bias sanity

A-pos win rate per matchup should be 25-75% (else position bias; rerun with re-seed).

- muV5_v3_vs_muV4_tool_v: 4/8 (50%) ✓
- phase4a_meta_ttl: 5/8 (62%) ✓
- phase4a_tt_control: 5/8 (62%) ✓

## Sample rationales (3/matchup, first 300 chars)

### muV5_v3_vs_muV4_tool_v

- **mu_v4_vs_mu_v5_v3_pair_00** (winner=A): Plan A wins on Soundness, Specificity, and goal-alignment. A writes out the full exponential-utility advantage with a complete RHS (A_beta(y_i) = (1/beta)(exp(beta r(y_i))/E_j[exp(beta r(y_j))] - 1)) and pins the external evidence to a concrete Python-program verifier (math library scoring), directl
- **mu_v4_vs_mu_v5_v3_pair_01** (winner=A): Plan A wins on Specificity, Methodological rigor, and Reproducibility. Concretely, A specifies a per-problem budget of 400-800 synthetic variants and gives the RHS-complete risk-sensitive advantage in normalized form (\hat{A}_\beta(y_i) = (1/\beta)(e^{\beta r(y_i)}/E_j[e^{\beta r(y_j)}] - 1)), which
- **mu_v4_vs_mu_v5_v3_pair_02** (winner=B): Plan B dominates on specificity, soundness, and methodological rigor. B provides a written-out risk-sensitive GRPO update equation with a hyperparameter range (beta = 4-8), specifies 400 synthetic variants per problem, names concrete tooling (Sympy, SMT solvers, Lean grammar), and uses K-means clust

### phase4a_meta_ttl

- **sigma_prime_vs_mu_prime_pair_00** (winner=A): Plan B has surface advantages (RHS-complete RS-GRPO advantage equation with beta=4, KL-anchoring equation, 50-100 variants, 200-sample SIFT buffer, 10-15 min A100 compute) but violates two hard goal constraints. First, it proposes a 'LORA+TTRL pipeline' with 'context-conditioned weight updates' and 
- **sigma_prime_vs_mu_prime_pair_01** (winner=B): Plan A directly violates the goal's hard frozen-weights constraint: Methodology 3 states 'The LLM's frozen weights are updated via gradient steps on the task's return' and Methodology 5 calls for resetting 'the LLM's parameters between problems' — both contradict the explicit prompt-space / frozen-a
- **sigma_prime_vs_mu_prime_pair_02** (winner=B): Plan A is more specific (β=4, β_KL=0.01, 500-1000 variants, ρ=0.2, top-50 of 500, named papers Akyurek/Hubert/Jiang/Georgiev/Phan), but it solves the wrong problem: it targets math/physics reasoning (AIME2024, MATH-500, AlphaProof-style proof variants) and names benchmarks (Overcooked, C4) that are 

### phase4a_tt_control

- **sigma_prime_vs_mu_prime_pair_00** (winner=A): Both plans drift from the goal's architectural ask (a differentiable finite-horizon solver inside the forward pass) toward an external TTT/TTRL+RS-GRPO recipe — neither specifies a Bellman/value-iteration unrolling or how the layer 'internally solves' the decision problem. Given that shared weakness
- **sigma_prime_vs_mu_prime_pair_01** (winner=B): Plan B more directly tracks the goal's three-step architectural specification: it lays out context encoding -> a finite-horizon MDP solved by policy gradients over the latent state -> decoding the first-step optimal action as the next token, and concretely names LoRA as the adapter mechanism with th
- **sigma_prime_vs_mu_prime_pair_02** (winner=B): Both plans miss the goal's core architectural requirement (an internal differentiable finite-horizon solver in the forward pass — neither writes a Bellman recursion, value head, or horizon-32 unroll mechanism, both default to external TTT/GRPO loops). On the dimensions they share, Plan B is more spe

