# Audit Rubric v3 — Quantile Panel Data Minimum Distance Estimation (Econ)

*Domain-specific rubric for cross-paper audit on Melly & Pons (arxiv 2502.18242) — "Minimum Distance Estimation of Quantile Panel Data Models". U1-U5 verbatim from `AUDIT_RUBRIC_v3.md:28-66` (universal NeurIPS/ICLR/ICML/NSF axes). T1-T4 econometric-tuned per plan §8.11.*

*Status: LOCKED for cross-paper audit. Authored 2026-04-28 PM, BEFORE any σ_quantile_panel/ξ_quantile_panel run. Future edits require explicit retraction notice.*

## User-locked decisions (carried over from `AUDIT_RUBRIC_v3.md`)

- **Total dimensions**: 9 (5 universal + 4 subfield-specific)
- **Anchor format**: Option B — anchors quote source-paper benchmarks where verifiable
- **Weighting**: equal-rank (each dim 1-5; total /45 raw or /20 normalized)
- **Anchor strictness**: strict (5 = excellent, rare)
- **Anti-pattern penalty**: NONE — anchors encode "no substance = low score"

## Two-layer rubric (Universal + Econ-subfield-specific)

### Universal layer (5 dimensions, each 1-5 → /25)

#### U1. Soundness — Are claims supported by evidence (math or empirical)?
- 1: Claims unsupported (no equations, no benchmark numbers, no derivation)
- 2: Some claims supported but key claims hand-waved (e.g. method described but no formula)
- 3: Major claims supported by either inline math (RHS-complete) or named benchmark numbers — but not both
- 4: Claims supported by both inline math AND specific empirical numbers; minor hand-waving acceptable
- 5: All non-trivial claims supported with rigor; assumptions stated; robustness discussed

#### U2. Significance — Is this a real problem with real impact?
- 1: Toy problem, no clear practical value
- 2: Real problem but plan addresses only narrow slice
- 3: Real problem, plan addresses central question, but unclear it advances state of the art
- 4: Real problem, plan claims SOTA-comparable result with credible argument
- 5: Real problem, plan would advance SOTA demonstrably and be picked up by community

#### U3. Originality — Novel beyond pretraining-knowledge or trivial recombination?
- 1: Standard toolkit recombination ("REINFORCE + LoRA + buffer") with no insight
- 2: Minor adaptation of known method; some novelty in application
- 3: Non-obvious combination with at least one mechanism-level insight
- 4: Substantive modification with clear differentiation from prior work; new operator or new objective
- 5: Genuinely novel algorithm or framing; clear theoretical/empirical hook differentiating from cited prior work

#### U4. Clarity — Is the plan well-reasoned and structurally clear?
- 1: Disorganized; key sections missing or vague
- 2: Sections present but mechanisms unspecified
- 3: Mechanisms named but reasoning chain has gaps
- 4: Mechanisms specified with rationale; reasoning chain coherent end-to-end
- 5: Clear non-specialist-readable plan; each step justified; no jargon without explanation

#### U5. Reproducibility — Are compute, hyperparameters, code/data, and statistics specified?
- 1: No mention of compute, hyperparameters, statistical methodology
- 2: 1 of {compute, hparams, eval methodology} mentioned vaguely
- 3: 2 of 3 mentioned with some specificity
- 4: All 3 stated with concrete values (e.g. MC replication counts, m + n grid, τ values, software package)
- 5: All stated + statistical methodology (CIs, bootstrap, finite-sample correction) + open-data + open-software commitment

### Econ-subfield-specific layer (4 dimensions, each 1-5 → /20)

Drawn from Melly-Pons paper structure + standard econometric-method reviewer expectations (Econometrica / JoE / QE).

#### T1. Necessity vs prior MD/IV quantile estimators (CLP 2016 + Galvao-Wang 2015)

The most common rejection pattern for new quantile-panel-data estimators: "this is a marginal improvement over Chetverikov-Larsen-Palmer 2016 and not worth a separate publication." The paper claims order-of-magnitude (up to 20×) MSE reduction over CLP — this MUST be empirically verified across multiple sample sizes and quantile levels with 10,000 Monte Carlo replications.

- 1: Plan does not address comparison vs CLP 2016 or other quantile MD estimators at all
- 2: Plan asserts the new estimator improves but no comparison to even one prior estimator
- 3: Plan mentions ≥1 prior estimator (CLP 2016 OR Galvao-Wang 2015) but doesn't quantify MSE/variance gap
- 4: Plan specifies ≥2 prior estimators with concrete MSE-reduction targets (e.g. "20× lower MSE on n=200, m=25 at τ=0.5")
- 5: Plan explicitly characterizes the regime where the proposed method strictly dominates CLP (e.g. "with non-trivial group heterogeneity and ≥2 individual-level regressors"), AND identifies regimes where CLP suffices (e.g. "with single intercept-only group-level variation")

#### T2. Disentanglement of Stage 1 (within-group) vs Stage 2 (between-group) variance contributions

A two-step estimator's asymptotic variance decomposes into contributions from each stage. The proposed method makes a non-trivial claim that the variance is dominated by Stage 1 (rate √(mn)) when no group heterogeneity exists, but by Stage 2 (rate √m) when group heterogeneity dominates — and provides an adaptive procedure that selects between regimes. This must be both proven asymptotically AND verified in finite samples.

- 1: Plan treats the two-step estimator as one black-box variance contribution
- 2: Plan acknowledges Stage 1 + Stage 2 distinction but no separate variance derivation
- 3: Plan proposes explicit variance decomposition (V = V_stage1/(mn) + V_stage2/m) for one specific case
- 4: Plan provides asymptotic distribution under both Case 1 (group heterogeneity present, mixed rates) and Case 2 (no group heterogeneity, uniform rate)
- 5: Plan additionally provides the intermediate Case 3 (vanishing heterogeneity at exactly the right rate) AND introduces an adaptive procedure that automatically selects the correct asymptotic regime without user input

#### T3. Monte Carlo replication count, (m, n, τ) grid, and finite-sample evidence

Econometric methods reviewers expect explicit Monte Carlo evidence at multiple sample sizes covering both panel dimensions. CLP 2016 used 10,000 replications across (m, n) ∈ {(25, 25), (200, 25), (25, 200), (200, 200)} × τ ∈ {0.1, 0.5, 0.9} × 3 endogeneity conditions = 36 cells. The proposed estimator must match or exceed this rigor.

- 1: No Monte Carlo evidence
- 2: Monte Carlo evidence at ≤2 sample sizes OR ≤1 quantile
- 3: Monte Carlo evidence at 4+ sample sizes × 1 quantile, OR 1 sample size × 3 quantiles, with ≥1000 replications
- 4: Full (m, n) × τ grid (≥4 × ≥3) × ≥10,000 replications, comparing to ≥1 prior estimator on MSE/MAE
- 5: 4 + sensitivity analysis to additional axes (e.g. degree of group heterogeneity, instrument strength, finite-sample bias correction)

#### T4. Endogeneity violation + asymptotic-rate misspecification analysis

Quantile-panel-data estimators face known failure modes: violation of strict exogeneity of group effects, weak instruments, and misspecified asymptotic-rate convergence (assuming m,n→∞ when in fact one is fixed). The strongest defensive framing acknowledges these and tests robustness.

- 1: No analysis of robustness or assumption violations
- 2: Acknowledges potential limitations but no concrete failure-mode analysis
- 3: Names ≥1 specific assumption violation (e.g. weak instrument, group effect endogeneity, finite n) — one of three
- 4: Both ≥2 assumption violations identified + simulation-based robustness analysis (e.g. weak-instrument MC; finite-n MC) + mitigation discussion (e.g. cluster-robust variance)
- 5: 4 + plan includes overidentification test for between-variation exogeneity (Hausman analog) AND adaptive variance estimator robust to unknown convergence-rate regime

## Combined scoring

**Total = Universal /25 + Econ-subfield /20 = /45**

For reporting, **always show both layers separately** so we know which axis a plan is strong/weak on.

## Two-pass procedure (same as base AUDIT_RUBRIC_v3)

### Pass 1: claim enumeration (mandatory)

Opus extracts:
- `claim_list` — every verifiable claim (full-RHS equation, named estimator, MC replication count, asymptotic rate, endogeneity-condition specification). Each with verbatim quote ≤25 words.
- `concern_list` — every reviewer-style concern (missing comparison, hand-waved mechanism, vague hyperparameter, unattributed novelty claim). Each with verbatim quote ≤25 words.

### Pass 2: score each dimension with reference to enumerated claims/concerns

For each of U1-U5 and T1-T4:
- Provide score (1-5)
- Provide 1-2 sentence justification referencing specific claims or concerns
- No anti-pattern penalty floor — anchors themselves penalize missing substance

### Output format

```json
{
  "claim_list": [{"kind": "equation"|"hparam"|"benchmark"|"operator", "quote": "..."}],
  "concern_list": [{"kind": "missing_comparison"|"hand_waved"|"vague_hparam"|"unattributed_novelty", "quote": "..."}],
  "universal_scores": {
    "U1_soundness": {"score": int, "justification": "..."},
    "U2_significance": {...}, "U3_originality": {...}, "U4_clarity": {...}, "U5_reproducibility": {...}
  },
  "subfield_scores": {
    "T1_necessity_vs_prior_MD_estimators": {"score": int, "justification": "..."},
    "T2_stage1_vs_stage2_variance_decomposition": {...},
    "T3_monte_carlo_grid_evidence": {...},
    "T4_endogeneity_misspecification_analysis": {...}
  },
  "universal_total": int (5-25),
  "subfield_total": int (4-20),
  "weighted_total_norm20": float (0-20),
  "raw_total_45": int
}
```

## Anchor calibration evidence (verbatim from Melly & Pons paper, for Opus reference)

- **T1 (Necessity)** anchor 4-5: "Table [?] shows that our minimum distance (MD) estimator exhibits substantially lower variance and mean squared error (MSE) across all sample sizes considered—reducing the MSE by a factor of up to 20."
- **T2 (Disentanglement)** anchor 4-5: "The asymptotic variance has two components: one arising from the first-stage quantile regression (proportional to 1/(mn)) and another from the second-stage GMM regression (proportional to 1/m). The asymptotic distribution depends on which component dominates."
- **T3 (Monte Carlo grid)** anchor 4-5: "Monte Carlo simulations across 4 sample sizes × 3 quantiles × 3 endogeneity conditions × 10,000 replications."
- **T4 (Misspecification)** anchor 4-5: "We introduce an inference procedure that automatically adapts to the potentially unknown convergence rate of the estimator… we suggest an overidentification test, which provides the quantile equivalent of the Hausman test for the exogeneity of the between variation."

## Locked-state evidence

- Authored: 2026-04-28 PM
- Authored before any σ_quantile_panel/ξ_quantile_panel run on Quantile Panel (verifiable via git timestamp)
- T1-T4 anchors derived from source paper structure (per `CROSS_DOMAIN_SOURCE_PAPERS_v1.md:117-126` verified evidence)
- U1-U5 unchanged from `AUDIT_RUBRIC_v3.md` (committed earlier)
