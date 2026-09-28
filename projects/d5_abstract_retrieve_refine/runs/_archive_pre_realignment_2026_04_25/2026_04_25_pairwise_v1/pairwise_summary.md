# D5 Phase 0.6 Pairwise Tournament — Results
*Generated 2026-04-25 20:11:04*
*Seed: 42, n_pairs per matchup: 8*

## Win rates per matchup
| Matchup | Wins (left) | Wins (right) | Ties | Left win rate |
|---|---|---|---|---|
| **mu** vs **delta** | 0 | 8 | 0 | 0.0% |
| **mu** vs **beta** | 8 | 0 | 0 | 100.0% |
| **beta** vs **alpha** | 4 | 4 | 0 | 50.0% |
| **alpha** vs **delta** | 0 | 8 | 0 | 0.0% |
| **mu** vs **epsilon** | 0 | 8 | 0 | 0.0% |

## Pairwise vs absolute ranking
Absolute ranking by Opus D3-canonical /20:
  ε (12.13) > μ (8.88) > β (7.75) > δ (6.75) > α (6.12)

Pairwise matchup outcomes:
- mu vs delta: **delta** wins (0–8, ties=0)
- mu vs beta: **mu** wins (8–0, ties=0)
- beta vs alpha: **tie** wins (4–4, ties=0)
- alpha vs delta: **delta** wins (0–8, ties=0)
- mu vs epsilon: **epsilon** wins (0–8, ties=0)

## Inversion check
Critical pairs to verify:
- α vs δ: absolute says δ > α (6.75 > 6.12). If pairwise reverses, then 'Opus distillation hurts 30B' is grader artifact.
- μ vs δ: absolute says μ > δ (+2.13). If pairwise close to 50/50, training value is illusory.
- μ vs ε: paper narrative — μ should lose; quantify by how much.
