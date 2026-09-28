# Phase 2F audit_v3 ISOLATED — anti-bias per-plan scoring

*Generated 2026-04-26 04:04:32*

Methodology: 8 balanced batches (1 plan/baseline each) × 8 parallel
Opus subagents. Each subagent scores 7 anonymized plans with strict
per-plan independence instructions.

## Per-baseline aggregates

| Baseline | N | Mean /45 | Std | Min | Max | Distinct totals |
|---|---:|---:|---:|---:|---:|---:|
| **epsilon** | 8 | 33.62 | 1.93 | 32 | 37 | 5/8 |
| **sigma** | 8 | 25.25 | 2.44 | 22 | 29 | 7/8 |
| **delta** | 8 | 25.12 | 4.31 | 17 | 30 | 6/8 |
| **alpha** | 8 | 25.00 | 2.06 | 21 | 28 | 6/8 |
| **mu** | 8 | 23.88 | 1.83 | 21 | 26 | 5/8 |
| **beta** | 8 | 22.75 | 1.98 | 21 | 26 | 4/8 |
| **xi** | 8 | 15.62 | 2.87 | 12 | 20 | 6/8 |

## Per-dim means matrix

| Baseline | U1 | U2 | U3 | U4 | U5 | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **epsilon** | 4.25 | 4.00 | 3.75 | 4.38 | 4.50 | 3.38 | 2.88 | 4.00 | 2.50 |
| **sigma** | 3.00 | 3.50 | 2.38 | 3.75 | 2.62 | 3.12 | 2.38 | 2.00 | 2.50 |
| **delta** | 3.12 | 3.00 | 2.75 | 3.62 | 3.25 | 2.50 | 2.38 | 2.50 | 2.00 |
| **alpha** | 3.00 | 3.62 | 2.75 | 3.62 | 2.50 | 3.25 | 2.25 | 1.88 | 2.12 |
| **mu** | 3.00 | 3.38 | 2.38 | 3.62 | 2.38 | 3.12 | 2.12 | 1.75 | 2.12 |
| **beta** | 2.75 | 3.25 | 2.38 | 3.62 | 2.12 | 3.00 | 2.12 | 1.50 | 2.00 |
| **xi** | 1.62 | 2.25 | 1.38 | 2.62 | 1.75 | 1.62 | 1.62 | 1.50 | 1.25 |

## All totals per baseline (validate within-baseline variance)

- **epsilon**: [33, 36, 32, 32, 32, 32, 37, 35]
- **sigma**: [23, 22, 23, 28, 27, 26, 29, 24]
- **delta**: [17, 28, 27, 20, 30, 23, 28, 28]
- **alpha**: [25, 25, 23, 27, 25, 26, 28, 21]
- **mu**: [22, 22, 21, 26, 25, 25, 26, 24]
- **beta**: [21, 22, 23, 21, 26, 21, 26, 22]
- **xi**: [13, 17, 12, 18, 18, 12, 20, 15]
