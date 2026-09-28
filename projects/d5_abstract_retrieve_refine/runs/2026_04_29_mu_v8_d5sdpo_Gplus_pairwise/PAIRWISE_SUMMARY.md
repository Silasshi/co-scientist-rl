# D5 μ-v8-d5sdpo G+ Pairwise Corroboration

*Generated 2026-04-28 13:22:04*  
*Seed: 60, n_pairs/matchup: 8*  
*Test-time-fair: NO reference plan in either prompt; both sides see goal+oracle.*

## Cross-check question

M8-strict 1-plan/Opus 9-dim audit_v3 reported G+ iter 4 = 22.38 / 45 vs σ_v8 = 19.38 (+3.00). Does Opus pairwise judge corroborate G+'s lead, and how does G+ compare to v7-opd-full / μ-v4 / G?

## 4 Pairwise Matchups

| Matchup | la wins | lb wins | ties | A-pos win rate | Winner |
|---|---:|---:|---:|---:|---|
| Gplus_vs_sigmaV8 | 8 (gplus) | 0 (sigma_v8) | 0 | 4/8 | **gplus** |
| Gplus_vs_v7opdfull | 3 (gplus) | 5 (v7_opd_full) | 0 | 3/8 | **v7_opd_full** |
| Gplus_vs_muV4 | 8 (gplus) | 0 (mu_v4) | 0 | 6/8 | **gplus** |
| Gplus_vs_G | 7 (gplus) | 1 (g) | 0 | 3/8 | **gplus** |

## Decision Matrix

**Decision rules:**
- G+ vs σ_v8: ≥6/8 win → audit lift corroborated, locks v8 production checkpoint
- G+ vs v7-opd-full: ≥6/8 win → v8 redesign beats v7-opd-full despite length difference; PAPER HEADLINE
- G+ vs μ-v4: ≥6/8 win → v8 surpasses Phase 2 winner; Tier-1 paper claim
- G+ vs G: ≥6/8 win → α=0.1 dominates α=0.05 within v8 family

