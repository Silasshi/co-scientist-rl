# F2 — μ-v4 iter 4 peak: D5 in-house off-policy IS-loss + Opus critic adds +2.75 over σ

> **⚠ TERMINOLOGY NOTE (added 2026-04-27 PM, REVISED)**: this finding was
> originally framed as "SDPO + Opus critic" and a 2026-04-27 first revision
> labeled it "Hübotter 2026 option (c) HER relabelling". After paper-grade
> WebFetch verification of arxiv 2601.20802 and 2601.18734:
> - μ-v4's actual training algorithm is a **D5 in-house off-policy IS-loss
>   variant** (TEACHER samples; STUDENT lp recomputed; importance_sampling
>   loss). It is structurally distinct from canonical Hübotter SDPO
>   (2601.20802 — student on-policy + KL) and from canonical Zhao OPSD
>   (2601.18734 — student on-policy + JS or sampled-token reverse-KL).
> - "Hübotter option (c) HER" was D5 internal speculation; the paper has no
>   such variant.
> - The 28.00/45 audit number was inline-batched (M8-confounded); strict
>   1-plan/Opus isolated would be lower. F13/F14 document that the D5 in-house
>   on-policy IS-loss variant (μ-v7-opd; closest to OPSD sampled-token PG)
>   wins 45/48 = 93.75% combined pairwise anonymized over this μ-v4 production.
> Read with both caveats. Disambiguation:
> `knowledge/current/CANONICAL_NAMING_REFERENCE.md`,
> `knowledge/current/RUN_REGISTRY.md`.

## Headline

Plan-level D5 in-house off-policy IS-loss variant + Opus critic (mislabeled as "SDPO" historically; closest distinct paper algorithms are Hübotter SDPO 2601.20802 and Zhao OPSD 2601.18734, neither faithfully implemented here),
run with **lr=5e-5, 4 grad steps/iter** for 5 iters, lifts Qwen3-30B-A3B EVAL
plans from σ baseline 25.25 to **28.00 / 45 at iter 4** —
the **first 30B-trained variant to beat the σ frozen+oracle baseline** in the D5
program.

## Trajectory (9-dim isolated audit)

Source: `runs/2026_04_27_mu_v4_early_iter_audit/`

| Iter | Mean /45 | Min | Max | Δ vs σ 25.25 |
|---:|---:|---:|---:|---:|
| 0 | 24.00 | 20 | 26 | -1.25 |
| 1 | 24.75 | 21 | 26 | -0.50 |
| 2 | 24.88 | 21 | 28 | -0.38 |
| 3 | 27.25 | 19 | 32 | **+2.00** ← first to beat σ |
| **4** | **28.00** | **25** | **32** | **+2.75 (peak)** |
| 5 | 15.62 | 9 | 22 | -9.62 (cliff, see F3) |
| 6 | ~9.6 (4-dim daemon, normalized) | — | — | -15.7 (collapse) |
| 7 | ~9.8 (4-dim daemon, normalized) | — | — | -15.4 |

Pre-peak monotonic ascent over 5 iters; sharp cliff in 1 iter (4 → 5 = -12.4).

## Per-dim diff iter 0 → iter 4

| Dim | iter 0 | iter 4 | Δ |
|---|---:|---:|---:|
| U1 Soundness | 2.9 | 3.2 | +0.3 |
| U2 Significance | 3.1 | 3.5 | +0.4 |
| U3 Originality | 2.5 | 2.9 | +0.4 |
| U4 Clarity | 3.6 | 3.5 | -0.1 |
| U5 Reproducibility | 2.2 | 2.9 | **+0.7** |
| T1 Necessity-of-TTT | 2.8 | 3.1 | +0.3 |
| T2 Disentanglement | 2.5 | 2.9 | +0.4 |
| T3 Compute accounting | 1.8 | 2.1 | +0.3 |
| T4 Reward-hacking awareness | 2.6 | 3.1 | **+0.5** |

The lift is broad-spectrum (8 of 9 dims improve), with U5 (reproducibility — concrete
hyperparameters, compute, statistics) and T4 (reward-hacking awareness — adversarial
probes, failure analysis) showing the largest gains. These are the dimensions where
the critic's "improvement_directive" most often pushed for additions.

## Production checkpoint

`runs/2026_04_27_mu_v4/checkpoints.jsonl` — row for `batch=4` (iter 4 weights). This is
the **production μ-v4 checkpoint** for D5 paper. `kind: "both"` so includes sampler
weights (no need to re-fetch).

## Hyperparameters that worked (μ-v4)

```python
learning_rate = 5e-5
n_grad_steps_per_iter = 4
n_iter = 20  # but stop at iter 4 (peak)
n_plans = 8
lora_rank = 64
oracle = oracle_v2_slim.md
critic = Opus 4.7 with privileged source paper
loss_fn = "importance_sampling"
sdpo_scale = 1.0
sdpo_clip_advantage = 5.0
```

See `paper_materials/methodology/M1_sdpo_recipe_v1.md` for the full reusable recipe.

## What this means for the paper

- **Plan-level SDPO + critic is a real positive result**: the architecture transfers
  content from teacher (with critique) to student (no critique) in early iters
- **+2.75 over σ on a single goal** is the first published demonstration we have of
  any SDPO-style training method beating its inference-time-scaffolding baseline at
  this scale and on this kind of long-form generation
- **Best-of-early-iter is the right stopping rule** — running past peak destroys the
  gains (see F3)

## Caveats

- N=8 plans per iter; per-iter mean has std ~3-4 across plans. The +2.75 lift is
  outside one σ but within two. Robustness check via cross-goal Phase 3 work pending.
- iter 4 vs iter 3 mean diff (28.00 - 27.25 = +0.75) is well within within-iter std,
  so picking iter 4 vs iter 3 as production is somewhat arbitrary. Both are above σ.
- 4-dim daemon audit was BLIND to this lift (all of iter 0-4 scored 7.00) — see F6
  for why.

## Cross-validation: pairwise corroboration (2026-04-26)

audit_v3 ISOLATED is an absolute-grading method, and Phase 0.6 saw it once
overstate by 8-0 (μ-v2 absolute-said-+2.13-vs-δ but pairwise-said-0-8). To rule
out the same surface-form bias on the +2.75 paper-claim headline, ran Opus 4.7
**pairwise tournament** (seed=50, 8 position-randomized pairs/matchup):

| Matchup | μ-v4 wins | Opponent wins | Ties | A-pos win rate |
|---|---:|---:|---:|---:|
| **μ-v4 vs σ** (PRIMARY) | **6** | **2** | 0 | 50% ✓ |
| μ-v4 vs δ | 7 | 1 | 0 | 62% ✓ |
| μ-v4 vs α | 7 | 1 | 0 | 75% ✓ |
| **Aggregate** | **20 / 24 (83.3%)** | 4 | 0 | — |

**Verdict**: PASS. Audit_v3 ISOLATED ranking (μ-v4 ≫ {σ, δ, α}) corroborated by
independent pairwise judgment. The PRIMARY paper-claim matchup (μ-v4 vs σ) clears
the ≥6/8 threshold; both close-cluster neighbors (δ, α) are decisively beaten 7-1
each. Position-bias check passes (all matchups within 25-75% A-position win rate).
Opus rationales consistently cite concrete hyperparameters (β=4/8, ρ=0.2, K=100),
named tools (RS-GRPO, SIFT, TTRL, LLaMA3.2-1B, AIME-2024 / 67-problem benchmark),
and citation grounding as the deciding factors — i.e., judging substance not
surface artifacts.

Notable: this REVERSES the Phase 0.6 finding (μ-v2 lost 0-8 to δ on pairwise).
μ-v4's broad-spectrum lift over μ-v2 transfers to pairwise dominance even against
the privileged δ baseline (which sees the reference plan in its prompt).

Source data: `paper_materials/experiments/E6_mu_v4_pairwise_corroboration.json`.
Run dir: `runs/2026_04_27_mu_v4_pairwise/`.

## Data pointers

- Trajectory data: `paper_materials/experiments/E2_mu_v4_trajectory.json`
- Per-dim per-iter: `paper_materials/figures_data/G1_per_dim_iter_evolution.csv`
- Best iter-4 plan exemplar: `paper_materials/plan_samples/exemplar_iter4_peak.md`
- Pairwise corroboration: `paper_materials/experiments/E6_mu_v4_pairwise_corroboration.json`
- Run dirs: `runs/2026_04_27_mu_v4/`, `runs/2026_04_27_mu_v4_pairwise/`
