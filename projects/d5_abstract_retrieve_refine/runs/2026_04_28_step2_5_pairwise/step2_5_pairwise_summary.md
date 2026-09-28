# D5 Step 2.5 pairwise corroboration — close-cluster control verification

*Generated 2026-04-26 19:02:02*  
*Seed: 51, n_pairs/matchup: 8*  
*Cross-validation per M4 discipline; close-cluster gap < 3 / 45 requires pairwise.*

## 3 matchups (close-cluster gap < 3 audit points)

| Matchup | la wins | lb wins | ties | A-pos win rate | Winner |
|---|---:|---:|---:|---:|---|
| **tau_v4** vs **sigma_v4** | 7 | 1 | 0 | 5/8 (62%) | **tau_v4** |
| **tau_v4** vs **mu_replan** | 4 | 4 | 0 | 6/8 (75%) | **tie** |
| **sigma_v4** vs **mu_v4** | 1 | 7 | 0 | 3/8 (38%) | **mu_v4** |

## Audit context (absolute)

| Baseline | Audit /45 |
|---|---:|
| τ_v4 (medium 3-batch + plan_v4) | 26.50 |
| σ_v4 (slim + plan_v4) | 25.75 |
| σ Phase 2 (slim + plan_v3) | 25.25 |
| μ-v4 Phase 2 prod (μ-v4 LoRA + slim + plan_v3) | 28.00 |
| μ-v4-replan (μ-v4 LoRA + slim + plan_v4) | 24.75 |

## Cross-validation verdict

- **tau_v4 vs sigma_v4**: tau_v4 wins 7-1 → STRONG
- **tau_v4 vs mu_replan**: tie wins 4-4 → TIE
- **sigma_v4 vs mu_v4**: mu_v4 wins 7-1 → STRONG
