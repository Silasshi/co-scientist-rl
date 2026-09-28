# Audit Rubric v3 — Climate Variability Land-Ocean Fingerprint (Earth/Climate)

*Domain-specific rubric for cross-paper audit on Hébert & Laepple (arxiv 2504.09939) — "Lacking oceanic-driven internal multidecadal climate variability is compensated by forced variability in model simulations". U1-U5 verbatim from AUDIT_RUBRIC_v3.md:28-66 (universal NeurIPS/ICLR/ICML/NSF axes). T1-T4 climate-tuned per plan §6.B + verified evidence from `CROSS_DOMAIN_SOURCE_PAPERS_v1.md:57-69`.*

*Status: LOCKED for cross-paper audit. Authored 2026-04-28 PM, BEFORE any σ_climate/ξ_climate run. Future edits require explicit retraction notice.*

## User-locked decisions (carried over from AUDIT_RUBRIC_v3.md)

- **Total dimensions**: 9 (5 universal + 4 subfield-specific)
- **Anchor format**: Option B — anchors quote source-paper benchmarks where verifiable
- **Weighting**: equal-rank (each dim 1-5; total /45 raw or /20 normalized)
- **Anchor strictness**: strict (5 = excellent, rare)
- **Anti-pattern penalty**: NONE — anchors encode "no substance = low score"

## Two-layer rubric (Universal + Climate-subfield-specific)

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
- 4: All 3 stated with concrete values (e.g. compute budget in $/CPU-hr; CMIP model ensemble size N; n=K reference periods)
- 5: All stated + statistical methodology (CIs, bootstrap, ensemble-spread / spread-skill metrics) + open-data commitment

### Climate-subfield-specific layer (4 dimensions, each 1-5 → /20)

Adapted from Hébert & Laepple paper structure (arxiv 2504.09939) and standard reviewer expectations for analytical climate-model intercomparison studies. Anchors verified against source paper's own analysis.

#### T1. Necessity of analytical method vs simpler diagnostic alternatives

The most common rejection pattern for climate-model evaluation papers: "the proposed analytical decomposition could be replaced by a simpler latitude-band variance comparison or a standard EOF/PCA decomposition." Hébert & Laepple's contribution depends on the *specific* CO₂-congruent variance removal + land-ocean variance ratio formulation being necessary, not just sufficient.

- 1: Plan does not address why the proposed analytical method is necessary vs simpler diagnostics
- 2: Plan asserts the method is useful but no comparison to even one simpler alternative (e.g. raw spatial correlation, EOF first mode)
- 3: Plan mentions ≥1 simpler diagnostic alternative (latitude-band variance, EOF first mode, simple AMV/PDO index) but doesn't quantify gap
- 4: Plan specifies ≥2 alternative diagnostics + explains why the proposed method captures signals they miss
- 5: Plan explicitly characterizes the regime where the analytical method is *necessary* (e.g. specific CO₂-forcing-internal-variability confound that simpler methods conflate; specific multidecadal-to-centennial timescale where forced response is non-trivially separable)

#### T2. Disentanglement — Forced vs internal variability separation

Hébert & Laepple's central claim is that a model's land-ocean variance ratio carries a *fingerprint* distinguishing internal variability from radiative forcing. This requires explicit per-component decomposition. Reviewer-canonical: "have you isolated the forced component and shown the residual is the internal-variability fingerprint, vs simply observing total variance?"

- 1: Plan treats {CO₂-forced response, aerosol forcing, internal variability, model bias} as one black box
- 2: Plan acknowledges forced-vs-internal distinction but no decomposition method proposed
- 3: Plan proposes one decomposition (CO₂-congruent regression / piControl ensemble subtraction / lagged correlation removal)
- 4: Plan proposes ≥2 decomposition methods with cross-validation against each other
- 5: Plan proposes full decomposition matrix (forced × internal × instrumental error × model structural bias) with explicit hypothesis on which factor dominates the residual signal

#### T3. Compute / data accounting in operational units

Climate-model intercomparison studies must state CMIP-ensemble size + observational dataset coverage explicitly. Hébert & Laepple use HadCRUT5 instrumental record (1850-present) + CMIP6 piControl + historical ensembles. Reviewer-canonical: "what is the effective sample size after accounting for autocorrelation? What is the ensemble spread?"

- 1: No statement of data sources or ensemble size
- 2: Data source named (e.g. "HadCRUT5") but no period / ensemble size specified
- 3: Data sources stated with period coverage (e.g. HadCRUT5 1850-2020) AND ensemble size for ≥1 model
- 4: Full per-model accounting: N CMIP models × M ensemble members × T years; observational period + uncertainty propagation; effective sample size after autocorrelation
- 5: 4 + explicit compute statement (CPU-hours for any model rerun; data preprocessing memory footprint) + sensitivity analysis to ensemble-size choice

#### T4. Parametric assumption violations / regional non-stationarity caveats

Climate-fingerprint methods canonically fail when assumptions break: Arctic warming amplification breaks land-ocean ratio stationarity; aerosol-forcing geographic gradients confound latitudinal patterns; volcanic eruptions create transient regimes. Hébert & Laepple's strongest defensive framing should explicitly name where the analytical inversion would NOT work.

- 1: No analysis of method robustness or assumption violations
- 2: Acknowledges potential limitations but no concrete failure-mode analysis
- 3: Names ≥1 specific assumption violation (Arctic amplification breaking land-ocean stationarity / aerosol-forcing gradient / volcanic transient) — one of three
- 4: Both ≥2 assumption violations identified + saturation behavior characterized; mitigations proposed (e.g. excluded periods, regional masking)
- 5: 4 + plan includes an explicit out-of-distribution probe experiment (e.g. apply method to Eemian or Pliocene paleoclimate where forcing differs) OR a stop-criterion based on regime detection (e.g. require pre-1980 baseline period for stationarity)

## Combined scoring

**Total = Universal /25 + Climate-subfield /20 = /45**

For reporting, **always show both layers separately** so we know which axis a plan is strong/weak on.

## Two-pass procedure (same as base AUDIT_RUBRIC_v3)

### Pass 1: claim enumeration (mandatory)

Opus extracts:
- `claim_list` — every verifiable claim (full-RHS equation, numbered hyperparameter, named CMIP model with prior number, deterministic operator). Each with verbatim quote ≤25 words.
- `concern_list` — every reviewer-style concern that arises while reading: missing comparison, hand-waved mechanism, vague hyperparameter, unattributed novelty claim. Each with verbatim quote ≤25 words.

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
    "T1_necessity_vs_simpler_diagnostics": {"score": int, "justification": "..."},
    "T2_forced_vs_internal_disentanglement": {...},
    "T3_data_compute_operational": {...},
    "T4_assumption_violation_analysis": {...}
  },
  "universal_total": int (5-25),
  "subfield_total": int (4-20),
  "weighted_total_norm20": float (0-20),
  "raw_total_45": int
}
```

## Anchor calibration evidence (verbatim from Hébert & Laepple, for Opus reference)

- **T1 (Necessity)** anchor 4-5: paper Methods + Results sections explicitly contrast their CO₂-congruent variance decomposition with prior diagnostics (raw variance ratios, lat-band averages); the analytical inversion is the *only* method that separates internal variability from forced compensation in a model-by-model fashion.
- **T2 (Disentanglement)** anchor 4-5: paper performs CO₂-forced regression removal AND internal variance reconstruction AND ensemble-spread analysis as triple-cross-checked decomposition.
- **T3 (Compute/Data)** anchor 4-5: paper lists CMIP6 model ensemble (N models × M members), HadCRUT5 instrumental period 1850-2020, and effective sample size after temporal smoothing.
- **T4 (Assumption violations)** anchor 4-5: paper explicitly discusses Arctic amplification, aerosol-forcing gradients, and the limitation that the latitudinal land-ocean ratio assumption may break under non-stationary regional forcing.

These are anchor exemplars for Opus's calibration, not exhaustive. Audit Opus is instructed to read each plan's claims against these anchors and score 1-5 per pass-2 procedure.

## Locked-state evidence

- Authored: 2026-04-28 PM
- Authored before any σ_climate/ξ_climate run on Climate Variability (verifiable via git timestamp)
- T1-T4 anchors derived from source paper structure (per CROSS_DOMAIN_SOURCE_PAPERS_v1.md:57-69 verified evidence)
- U1-U5 unchanged from AUDIT_RUBRIC_v3.md (committed earlier)
