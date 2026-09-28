# F6 — 4-dim D3-canonical audit anchor saturation

## Headline

A 4-dim audit (D1 reasoning depth / D2 empirical rigor / D3 formalism / D4 risk awareness,
each 1-5) with explicit calibration anchors saturates plans of similar quality at the
nearest anchor — leading to **all 8 plans tied at the same total** (e.g. 7.00 across all
8) when plans share the same surface feature profile. This **hides real per-iter
differences** and is unsuitable for tracking training progress.

## Empirical observation

μ-v4 daemon's 4-dim D3-canonical audit, per-iter:

| Iter | Mean | Distinct totals | All 8 totals |
|---:|---:|---:|---|
| 0 | 7.00 | 1/8 | [7,7,7,7,7,7,7,7] |
| 1 | 7.00 | 1/8 | [7,7,7,7,7,7,7,7] |
| 2 | 7.00 | 1/8 | [7,7,7,7,7,7,7,7] |
| 3 | 7.00 | 1/8 | [7,7,7,7,7,7,7,7] |
| 4 | 7.00 | 1/8 | [7,7,7,7,7,7,7,7] |
| 5 | 6.75 | 2/8 | [5,7,7,7,7,7,7,7] (1 plan starts to degrade) |
| 6 | 4.25 | 2/8 | [4,4,4,4,4,4,5,5] (collapse anchor) |

The 9-dim isolated audit on the SAME plans showed clear per-iter differences:

| Iter | 9-dim mean | Distinct totals |
|---:|---:|---:|
| 0 | 24.00 | 6/8 |
| 1 | 24.75 | 6/8 |
| 2 | 24.88 | 7/8 |
| 3 | 27.25 | 7/8 |
| 4 | 28.00 | 8/8 |

## Why anchor saturation happens

The 4-dim D3-canonical anchors are explicit and feature-based:
- "Standard plan with named methods, no equations" → 7
- "+ named benchmarks with prior numbers" → 8-9
- "+ RHS equations" → 10-11
- "+ adversarial probe / careful failure analysis" → 11-12

When 8 plans all share the SAME feature profile (e.g. all have named methods, all have
named benchmarks without prior numbers, all have no RHS equations) they all match the
anchor → all score 7. Per-plan reasoning differs (the daemon's reasoning fields confirmed
substantively distinct content per plan) but the anchor-based scoring collapses them.

The 9-dim rubric has 5 universal + 4 subfield-specific dimensions. With 9 dims × 5
levels = 45 max, a plan's score profile has more "decimal places" — small per-plan
differences in any one dim can shift the total by 1-2 points. Two plans rarely hit
the same 9-dim vector unless content is genuinely interchangeable.

## What this means for the paper

- **Per-iter trajectories MUST use 9-dim isolated audit**, not 4-dim daemon audit
- The 4-dim daemon audit IS useful for fast in-loop checkpoint quality monitoring
  during training (sample 8 plans, get a coarse signal cheap) — but it is not
  paper-grade
- **Both audits use the same Opus 4.7 model** — the difference is purely in
  rubric design, not auditor quality
- **Anchor calibration** (concrete examples for each level) is a double-edged sword:
  it improves cross-rubric reproducibility but can saturate when plans share surface
  features

## Implication for design

For long-form generation evaluation tooling:
- Use a higher-dimensional rubric (≥7 dims, ideally 9+) when within-iter or within-
  baseline variance matters
- Use a smaller rubric only for "is this plan at all coherent" sanity checks during
  training loops
- Calibrate anchors on EXTREME plans (best, worst) and let middle scores fall naturally
  rather than anchoring intermediate categories

## Data pointers

- 4-dim daemon trajectory: `runs/2026_04_27_mu_v4/audit_responses/iter_*.json`
- 9-dim isolated trajectory: `runs/2026_04_27_mu_v4_early_iter_audit/audit_responses/`
- Side-by-side comparison: `paper_materials/figures_data/G1_mu_v4_trajectory.csv`
  (column: 4dim_mean vs 9dim_isolated_mean)
