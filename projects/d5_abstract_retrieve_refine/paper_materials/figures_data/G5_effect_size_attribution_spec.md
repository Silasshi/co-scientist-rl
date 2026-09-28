# Figure G5 — Effect-Size Attribution: Where the Audit Lift Comes From

*Authored 2026-04-27. Replaces prior G5_compute_cost_vs_audit_lift_spec.md (cost framing rejected per ML scientist audit — reviewers care about Δ + CI + mechanism, not $/audit-point).*

## Goal

Visualize the **decomposition of audit gains** from ξ floor to ε upper bound, with each path's contribution and 95% CI. Make explicit that **inference-time scaffolding (σ) is the dominant lever**, training (μ-v4) is a bounded marginal addition, and inference-only distillation (τ_v4_clean) lies within σ noise.

## Data points (effect sizes over ξ baseline)

| Path | Audit /45 | Δ vs ξ | 95% CI (est) | Status |
|---|---:|---:|---|---|
| ξ (frozen + goal only) | 15.62 | 0 (anchor) | [13.5, 17.7] | tested |
| σ (frozen + slim oracle) | 25.25 | **+9.63** | [22.9, 27.6] | tested |
| τ_v4_clean (3-round inference distillation) | 24.50 | +8.88 | [22.0, 27.0] | tested (within σ CI) |
| μ-v4 iter 4 (SDPO trained) | 28.00 | **+12.38** | [25.5, 30.5] | tested (production) |
| μ-v4 + BoN-4 (planned) | TBD | TBD | TBD | planned (T1) |
| ε (frozen 235B + reference plan) | 33.62 | +18.00 | [31.2, 36.0] | tested (upper bound) |

CIs are estimated from per-plan std reported in F2/F8 (σ_within ≈ 1.17 to 4.31 across baselines, n=8 → SE ≈ 0.4 to 1.5, 95% CI ≈ ±2.5 typical).

## Three load-bearing comparisons

1. **σ - ξ = +9.63** (inference scaffolding lift, broad-spectrum across dims)
2. **μ-v4 - σ = +2.75** (training marginal contribution over scaffolding)
3. **ε - μ-v4 = +5.62** (remaining gap to 235B+reference upper bound)

**Decomposition narrative**: of the 18.00 total achievable gap (ξ → ε), inference scaffolding alone closes **54%** (9.63/18.00), trained SDPO closes another **15%** (2.75/18.00), and **31% remains unbridged** by 30B methods.

## Figure type + axis spec

**Type**: Horizontal bar chart with error bars (NOT scatter, NOT pareto)

**Justification**:
- Each path is a discrete intervention, not a continuous variable
- Want to compare effect-size magnitudes side-by-side
- Error bars communicate measurement uncertainty crucial for spotlight reviewer
- Horizontal layout fits long path names

**Axes**:
- X: Audit Score / 45 (linear, 13-37 range)
- Y: Path label (categorical, ordered ξ → σ → τ_v4_clean → μ-v4 → ε top-to-bottom OR effect-size ascending)

**Visual encoding**:
- Bars colored by category: ξ (gray, anchor), σ (blue, inference scaffolding), τ_v4_clean (light blue, inference distillation), μ-v4 (green, training), ε (red, upper bound)
- BoN-4 (planned): hatched/dashed bar with TBD annotation
- Error bars: 95% CI as horizontal lines on each bar end
- Vertical reference line at σ=25.25 (gray dashed) — "inference scaffolding ceiling without training"
- Vertical reference line at ε=33.62 (red dashed) — "upper bound (235B+reference)"

## Annotations spec

1. **Δ labels** beside each bar (right side):
   - ξ: "(anchor)"
   - σ: "+9.63"
   - τ_v4_clean: "+8.88 (within σ CI)"
   - μ-v4: "+12.38"
   - ε: "+18.00"

2. **Three load-bearing arcs** between bars (curly braces or arrows):
   - σ-ξ arc: "**inference scaffolding +9.63 (54% of total gap)**"
   - μ-v4-σ arc: "training marginal +2.75 (15%)"
   - ε-μ-v4 arc: "remaining gap +5.62 (31%)"

3. **CI overlap shading**: shade the overlap region between τ_v4_clean and σ CIs to visualize statistical indistinguishability

## Caption draft (3 versions)

### Terse (~40 words)
> **Figure G5: Effect-size attribution at 30B scale.** Inference scaffolding (σ) closes +9.63/45 (54% of gap to 235B+reference upper bound); plan-level SDPO training adds +2.75 (15%); inference distillation lies within σ confidence interval. 31% remains unbridged.

### Moderate (~80 words)
> **Figure G5: Effect-size attribution from ξ floor (15.62/45) to ε upper bound (33.62, 235B+reference).** Inference-time methodological scaffolding (σ, +9.63) dominates the recoverable gap, closing 54%. Plan-level SDPO with Opus critic (μ-v4 iter 4, +12.38 from ξ; +2.75 over σ) adds a bounded marginal lift but caps at multi-round instability (Hübotter 2026 §4). Three-round inference-only distillation (τ_v4_clean, +8.88) is statistically indistinguishable from σ (CI overlap, pairwise 4-4 TIE). 31% of gap remains unbridged by 30B methods.

### Verbose (~140 words)
> **Figure G5: Decomposition of audit gains shows inference scaffolding is the dominant 30B lever.** From a goal-only floor (ξ, 15.62/45) to a 235B+reference plan upper bound (ε, 33.62), the 30B regime recovers +12.38 of the 18.00-point achievable gap. Of this, **inference scaffolding (σ frozen+slim oracle, +9.63) accounts for 54%** of the total gap — broad-spectrum across all 9 audit dimensions, no training. Plan-level SDPO with Opus critic (μ-v4, lr=5e-5, peak iter 4) adds another **+2.75 over σ** (15% of total gap), the first 30B-trained variant to beat σ, but caps at multi-round instability cliff (28.00→15.62 in one iter). Three-round inference-only distillation (τ_v4_clean, +8.88) is statistically indistinguishable from σ (pairwise 4-4 TIE on freshly sampled patched plans). **31% of the 30B gap to ε remains unbridged**, motivating future work on retrieval-then-generate (Phase 3) and stronger critic models (κ-opus).

## Why this figure carries the spotlight argument

The previous cost-framed figure asked "how much $ per audit-point?" — reviewers don't ask this. This figure asks "**where does the lift come from?**" and answers: 54% from no-training scaffolding, 15% from training, 31% unbridged. That decomposition is:

- **Counter-intuitive**: most of the 30B story is inference, not training
- **Mechanistically grounded**: F1 (broad-spectrum scaffolding lift) + F3 (multi-round ceiling) + F4 (token-blindness root cause) all support
- **Forward-pointing**: the 31% unbridged gap motivates κ-opus + retrieve-then-generate as next-step research directions
- **Honest**: τ_v4_clean explicit retraction (4-4 TIE) is in the figure, not hidden

## Sources

**Confirmed**:
- ξ, σ, μ-v4, ε numbers — `F1_sigma_oracle_lift.md` § Numbers; `F2_mu_v4_iter4_peak.md` § Trajectory; `PROJECT_OVERVIEW.md` § Phase 2F baseline table
- τ_v4_clean 24.50 — `F8_phase3_distillation_pathway.md` § Step B (REVISED)
- 4-4 TIE — `F8 § Step E`

**Estimated**:
- 95% CIs derived from per-plan std × 1.96/√n; need raw audit JSONs to compute exact CIs
