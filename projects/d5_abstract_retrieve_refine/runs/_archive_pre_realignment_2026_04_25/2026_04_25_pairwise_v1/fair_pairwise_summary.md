# D5 Fair Pairwise Tests — Decision Matrix Report
*Generated 2026-04-25 21:48:39*
*Seed: 45, n_pairs/matchup: 8*
*All matchups under test-time-fair conditions: NO reference plan in prompt on either side.*

## 5 Fair Matchups

| Matchup | la wins | lb wins | ties | A-pos wins | Winner |
|---|---:|---:|---:|---:|---|
| **mu** vs **smoke_v3_A** | 2 | 6 | 0 | 6/8 | **smoke_v3_A** |
| **mu** vs **smoke_v3_B** | 8 | 0 | 0 | 4/8 | **mu** |
| **alpha** vs **smoke_v3_B** | 2 | 5 | 1 | 3/7 | **smoke_v3_B** |
| **beta** vs **smoke_v3_B** | 2 | 6 | 0 | 4/8 | **smoke_v3_B** |
| **smoke_v3_A** vs **smoke_v3_B** | 8 | 0 | 0 | 7/8 | **smoke_v3_A** |

## Decision-Matrix Interpretation

- **μ vs smoke_v3_A** (training value at fixed oracle): smoke_v3_A wins (2-6)
- **μ vs smoke_v3_B** (training+oracle vs goal-only frozen): mu wins (8-0)
- **α vs smoke_v3_B** (Opus distillation vs frozen): close (2-5)
- **β vs smoke_v3_B** (ref SFT vs frozen): smoke_v3_B wins (2-6)
- **smoke_v3_A vs smoke_v3_B** (oracle inference value, no training): smoke_v3_A wins (8-0)

### Verdict

Training actively hurts. **D5 dead in this regime**. Major redesign required.

## Auxiliary signals

- α and β vs frozen baseline: useful for paper narrative on training-method comparison
- smoke_v3_A vs smoke_v3_B: validates pathway claim under pairwise (was withdrawn under v1+pairwise vs δ)

## Position-bias check (sanity)

| Matchup | A-pos win rate |
|---|---:|
| mu_vs_smokeA | 6/8 (75%) ✓ |
| mu_vs_smokeB | 4/8 (50%) ✓ |
| alpha_vs_smokeB | 3/7 (43%) ✓ |
| beta_vs_smokeB | 4/8 (50%) ✓ |
| smokeA_vs_smokeB | 7/8 (88%) 🚨 |

## Sample rationales (3 from each matchup, first 200 chars)

### mu vs smoke_v3_A

- **mu_vs_smoke_v3_A_pair_00** (A, 467 chars): Plan A (smoke) includes a dedicated Background section, more concrete evaluation domains (mathematical theorem proving, molecular structure optimization, physical system simulation) with explicit comp
- **mu_vs_smoke_v3_A_pair_01** (B, 344 chars): Plan B (μ) provides an explicit score formula score(s) = value(s) + β · uncertainty(s), names 4 distinct failure modes (a-d) of prior approaches, and gives 4 specific limitations. Plan A (smoke) has m
- **mu_vs_smoke_v3_A_pair_02** (A, 351 chars): Plan A (smoke) presents 5 numbered methodology components each with explicit Pattern attribution and rationale, mentions Pattern 7 evaluation framework, and includes a Background section that grounds 

### mu vs smoke_v3_B

- **mu_vs_smoke_v3_B_pair_00** (B, 647 chars): B (mu) explicitly enumerates Patterns 2-6 with concrete algorithmic content: max-optimized objective with sharpness parameter, accumulated-state buffer with (state, action, outcome) tuples, value+unce
- **mu_vs_smoke_v3_B_pair_01** (A, 649 chars): A (mu) provides a formal training objective J(theta) with sharpness-adjusted weighting, explicit gradient combination of per-action log-likelihoods, KL-constrained per-state hyperparameters, importanc
- **mu_vs_smoke_v3_B_pair_02** (B, 603 chars): B (mu) presents a structured XML-tagged plan covering Patterns 2-6: softmax-with-temperature objective for max-reward, state buffer for horizon extension, value+uncertainty score for warm-start, KL-di

### alpha vs smoke_v3_B

- **alpha_vs_smoke_v3_B_pair_00** (B, 367 chars): B (smoke_v3_B) is slightly more rigorous: it specifies a memory buffer with experience replay, regularization via weight clipping/entropy maximization, and explicitly frames the system as a closed-loo
- **alpha_vs_smoke_v3_B_pair_01** (B, 349 chars): A (alpha) is incomplete—the methodology cuts off mid-sentence at 'Perform a few' with no Evaluation or Limitations sections. B (smoke_v3_B) is complete and substantively richer: MCTS, sparse layer-sub
- **alpha_vs_smoke_v3_B_pair_02** (A, 381 chars): A (smoke_v3_B) names the method (SASGF) and gives a more concrete optimization story—autodiff-based gradient ascent on the reward, 10–50 update steps, explicit per-iteration gradient on parameters. B 

### beta vs smoke_v3_B

- **beta_vs_smoke_v3_B_pair_00** (B, 726 chars): Plan A (beta) is dominated by an unredacted chain-of-thought block and truncates inside step 2 of the Methodology, so Evaluation and Limitations are absent. Math: A only proposes 'gradient ascent' ins
- **beta_vs_smoke_v3_B_pair_01** (B, 775 chars): Both are similar in framing; A (beta) is clean prose for most of the body but truncates inside the Limitations bullet, leaving no closure. Math: A includes the explicit '$-R$ as loss' formulation and 
- **beta_vs_smoke_v3_B_pair_02** (B, 728 chars): Plan A is smoke_v3_B and Plan B is beta in this pair. A truncates inside the 'Compute budget constraints' bullet of the Methodology and never reaches Evaluation or Limitations. B (beta) is fully compl

### smoke_v3_A vs smoke_v3_B

- **smoke_v3_A_vs_smoke_v3_B_pair_00** (A, 848 chars): A (smoke_v3_A) provides 5 explicitly-named methodological components (max-focused softmax objective with temperature, accumulated-state buffer with pruning, hybrid value+uncertainty warm-start scoring
- **smoke_v3_A_vs_smoke_v3_B_pair_01** (A, 620 chars): A (smoke_v3_A) provides explicit J(θ) max-focused objective, KL-divergence-bounded per-state temperature, importance-weighting with clipping, hybrid value+uncertainty selection score, and 3 named dete
- **smoke_v3_A_vs_smoke_v3_B_pair_02** (A, 727 chars): A (smoke_v3_A) gives 5 numbered components (max-likelihood objective, state buffer pruning by visitation, hybrid score for warm-start, per-state KL-bounded sharpness, importance sampling) and 3 named 

