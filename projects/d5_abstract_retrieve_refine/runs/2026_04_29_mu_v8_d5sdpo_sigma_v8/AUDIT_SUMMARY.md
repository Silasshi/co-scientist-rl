# σ_v8 Baseline Audit Summary

*Run: 2026_04_29_mu_v8_d5sdpo_sigma_v8 — Batch 0 of v8 grid*
*Filed: 2026-04-28*

## Setup (as launched)

- Variant: σ (frozen Qwen3-30B-A3B + slim_oracle_v2, no training)
- **Footer**: v8 (900-word target, max 1100)
- n_plans: 8 (sampled in 76s, all 8/8 contain `<solution>` tag)
- Audit: 8 strict 1-plan/Opus subagents, M8-compliant, dispatched in parallel (no batch anchoring)

## Plan length statistics (8 plans)

| Stat | Value |
|---|---:|
| Mean words (in `<solution>` body) | 811 |
| Range | 752 - 866 |
| `<solution>` tag present | 8/8 |

vs σ legacy (`runs/2026_04_26_sigma_v2/`, v1 footer 600/750): mean 618 words (range 546-675). Footer change verified: model picks up the new target signal (+193 words mean).

## Audit /45 (M8-compliant strict 1-plan/Opus)

| Plan | U/25 | T/20 | /45 | norm /20 |
|---|---:|---:|---:|---:|
| 0 | 11 | 9 | 20 | 8.84 |
| 1 | 11 | 8 | 19 | 8.40 |
| 2 | 12 | 9 | 21 | 9.13 |
| 3 | 11 | 8 | 19 | 8.62 |
| 4 | 11 | 7 | 18 | 8.00 |
| 5 | 11 | 7 | 18 | 7.99 |
| 6 | 12 | 8 | 20 | 8.88 |
| 7 | 11 | 9 | 20 | 8.89 |
| **MEAN** | **11.25** | **8.12** | **19.38** | **8.59** |
| STD | 0.43 | 0.78 | 0.99 | 0.42 |

## Per-dim breakdown (mean across 8 plans)

| Dim | Mean | Range |
|---|---:|---|
| U1 Soundness | 2.00 | 2-2 (flat) |
| U2 Significance | 3.00 | 3-3 (flat) |
| U3 Originality | 2.12 | 2-3 |
| U4 Clarity | 3.00 | 3-3 (flat) |
| **U5 Reproducibility** | **1.12** | 1-2 (floor) |
| T1 Necessity | 2.75 | 2-3 |
| T2 Disentanglement | 2.00 | 2-2 (flat) |
| **T3 Compute** | **1.38** | 1-2 (floor) |
| T4 Reward-hacking | 2.00 | 2-2 (flat) |

## Comparison vs σ legacy (RUN_REGISTRY:155)

| Anchor | Method | /45 |
|---|---|---:|
| σ_v8 (THIS RUN) | strict 1-plan/Opus, 900-word footer | **19.38** |
| σ legacy | audit_v3 batched 8×8, 750-word footer | 25.25 |
| **Δ (σ_v8 − σ legacy)** | | **−5.88** |

**The two anchors are NOT directly comparable**. Two confounds:
1. **Audit methodology**: σ legacy used `audit_v3_isolated.py` with 8 BALANCED batches (each batch contains 1 plan from each of 7 baselines, with anchoring across baselines). σ_v8 used strict 1-plan/Opus dispatch (no anchoring). Per M8 standing rule (RUN_REGISTRY:160 caveat on μ-v4 28.00): batched audit anchoring inflates close-cluster scores ~2-3 points. **σ legacy strict-1-plan estimate: ~22-23**.
2. **Footer length**: σ_v8 plans are +193 words longer on average. The audit /45 is sensitive to length (longer plans tend to score higher when length matches reference). **However, σ_v8 still lost 5.88 points** — the extra length did NOT compensate for the methodology change. Interpretation: the model used the +200 words to add more name-drops (citations, mechanism names) without adding hyperparameters / equations / operational compute. U5 reproducibility crashed to 1.12 floor; T3 compute floored at 1.38.

## Implications for v8 grid

**σ_v8 = 19.38 is the correct anchor for v8 cells (base + A through G+).** Per-cell pairwise vs σ_v8 will measure training lift on **length-matched + audit-methodology-matched** comparator. σ legacy is no longer the right reference.

For paper headline: when reporting v8 cells, cite "v8 cell X = Y/45 (+Z over σ_v8 = 19.38)". The Δ vs σ legacy 25.25 is reportable in a methodology footnote but should not be the headline.

## Floor diagnosis (paper-relevant for v8 design)

The U5 floor (1.12) and T3 floor (1.38) are NOT footer-fixable. They reflect what the **policy itself never produces** — concrete hyperparameters, GPU-hours, statistical methodology — regardless of whether it's given 750 or 1100 words. v8 training (critique-on-current + Opus critic with full source paper) should specifically target U5/T3 by writing critiques like "the plan is missing concrete LoRA rank, learning rate, statistical methodology". This is exactly what F2's per-dim diff iter 0 → iter 4 showed for μ-v4 (U5 +0.7, T4 +0.5, broad-spectrum lift).

Pre-registered v8 success metric: any v8 cell that lifts U5 from 1.12 floor to ≥2.0 is delivering on the architecture's promise (critic-driven content acquisition into student EVAL distribution).

## Files

- Buffer: `runs/2026_04_29_mu_v8_d5sdpo_sigma_v8/buffer.jsonl` (8 plans)
- Audit responses: `runs/2026_04_29_mu_v8_d5sdpo_sigma_v8/audit_responses/plan_*.json` (8 files)
- Audit requests: `runs/2026_04_29_mu_v8_d5sdpo_sigma_v8/audit_requests/plan_*.json` (8 files)
- Launch log: `runs/2026_04_29_mu_v8_d5sdpo_sigma_v8/launch.log`
- Config: `runs/2026_04_29_mu_v8_d5sdpo_sigma_v8/config_sigma.json`

## RUN_REGISTRY update (next session)

Add row to inference-only table:

```
| 2026_04_29_mu_v8_d5sdpo_sigma_v8 | baseline_frozen_v2.py (footer_version=v8) | base Qwen3-30B + slim_oracle, 900/1100 footer | TTT-D | 8 | σ_v8 baseline; mean 19.38/45 strict 1-plan/Opus; mean 811 words; replaces σ legacy 25.25 as anchor for v8 grid |
```

And to anchor table:

```
| **σ_v8** (frozen + slim_oracle, 900/1100 footer) | **19.38** | runs/2026_04_29_mu_v8_d5sdpo_sigma_v8/ | strict 1-plan/Opus; length-matched anchor for v8 grid; Δ -5.88 vs σ legacy due to methodology + length confounds |
```
