# Audit Rubric v3 — Non-Self-Consistent DFA for Transition-Metal Catalysis (Comp-Chem)

*Domain-specific rubric for cross-paper audit on Shi & Berkelbach (arxiv 2602.14962) — "Practical and accurate density functionals for transition-metal heterogeneous catalysis". U1-U5 verbatim from `AUDIT_RUBRIC_v3.md:28-66`. T1-T5 chemistry-tuned (5 axes — one more than TTT-D's 4) per plan §8.11.*

*Status: LOCKED for cross-paper audit. Authored 2026-04-28 PM, BEFORE any σ_nsc_dfa/ξ_nsc_dfa run. Future edits require explicit retraction notice.*

## User-locked decisions (carried over from `AUDIT_RUBRIC_v3.md`)

- **Total dimensions**: 10 (5 universal + 5 subfield-specific) — DFT methodology has both quantitative-transferability and qualitative-failure axes that are sufficiently distinct to warrant 5 T-axes
- **Anchor format**: Option B — anchors quote source-paper benchmarks where verifiable
- **Weighting**: equal-rank (each dim 1-5; total /50 raw)
- **Anchor strictness**: strict (5 = excellent, rare)
- **Anti-pattern penalty**: NONE — anchors encode "no substance = low score"

**Note on dimension count**: per plan §8.11 ("T-axis count is NOT fixed at 4 — design from source paper evidence"), this rubric uses 5 T-axes (T1-T5) because the NSC-DFA paper's contribution decomposes naturally into 5 distinct reviewer-grade axes that should each be scored independently. Total raw = 50 (not the 45 of standard 9-dim TTT-D rubric); normalized to /20 still uses (Universal × 0.55 + Subfield × 0.45) but with subfield denominator = 25 (5 dims × 5).

## Two-layer rubric (Universal + Comp-Chem-subfield-specific)

### Universal layer (5 dimensions, each 1-5 → /25)

#### U1. Soundness — Are claims supported by evidence (math or empirical)?
- 1: Claims unsupported (no equations, no benchmark numbers, no derivation)
- 2: Some claims supported but key claims hand-waved
- 3: Major claims supported by either inline math (full functional form) or named benchmark numbers — but not both
- 4: Claims supported by both inline functional form AND specific MAD/SME values; minor hand-waving acceptable
- 5: All non-trivial claims supported with rigor; assumptions stated; robustness across multiple benchmarks discussed

#### U2. Significance — Is this a real problem with real impact?
- 1: Toy problem, no clear practical value
- 2: Real problem but plan addresses only narrow slice (e.g. one transition metal)
- 3: Real problem, plan addresses central question (transition-metal adsorption accuracy), but unclear advance over BEEF-vdW
- 4: Real problem, plan claims sub-chemical-accuracy across CE39 with credible argument
- 5: Real problem, plan would advance SOTA on multiple benchmarks (CE39 + SBH17 + SE20) AND solve qualitative puzzles (CO/Pt(111))

#### U3. Originality — Novel beyond toolkit recombination?
- 1: Standard hybrid functional recombination ("PBE0 + vdW + new mixing fraction") with no insight
- 2: Minor adaptation of known functional family
- 3: Non-obvious combination with at least one mechanism-level insight (e.g. NSC framework idea)
- 4: Substantive modification: NSC framework + parameter-economy + transferability without retuning
- 5: Genuinely novel framing: NSC-DFA as a systematic route from GGA to higher-rung accuracy without SCF instabilities or per-system retuning

#### U4. Clarity — Is the plan well-reasoned and structurally clear?
- 1: Disorganized; key sections missing or vague
- 2: Functional form named but mechanism unspecified
- 3: Functional form + fitting protocol named but reasoning chain has gaps
- 4: Functional form + fitting + benchmark protocols specified with rationale; reasoning coherent end-to-end
- 5: Clear non-specialist-readable plan; each step (orbital generation → EXX → RPA → mixing) justified

#### U5. Reproducibility — Are compute, hyperparameters, code/data, and statistics specified?
- 1: No mention of computational details
- 2: 1 of {functional form, k-point grid, plane-wave cutoff} mentioned vaguely
- 3: 2-3 of those mentioned with some specificity
- 4: All stated with concrete values (k-points, plane-wave cutoff in eV, ω-screening, INCAR templates, software versions)
- 5: All stated + open-source workflow (QuAcc) commitment + benchmark data availability + cross-validation of fitted parameters

### Comp-Chem-subfield-specific layer (5 dimensions, each 1-5 → /25)

Adapted from NSC-DFA paper structure + standard catalysis-DFT reviewer expectations (J. Chem. Phys. / J. Catal. / Phys. Rev. X).

#### T1. Necessity vs hybrid functional benchmarks (multi-functional comparison set)

The paper compares against 13 functionals spanning all rungs of Jacob's ladder. New functional methodology is rejected when comparison set is narrow (e.g. only PBE) — must demonstrate improvement against ≥1 hybrid (PBE0/B3LYP/HSE06), ≥1 double-hybrid (XYG3), ≥1 RPA-based (RPA@PBE), and the catalysis-specific BEEF-vdW.

- 1: Plan does not address comparison vs prior functionals
- 2: Plan asserts improvement but compares only to ≤2 functionals
- 3: Plan compares to ≥1 hybrid + ≥1 GGA (≥3 functionals total) with concrete MAD numbers
- 4: Plan compares to ≥1 each of {GGA, hybrid, double-hybrid, RPA, vdW} (≥5 functionals)
- 5: Plan benchmarks against ≥10 functionals across all rungs of Jacob's ladder, including BEEF-vdW (catalysis-tuned baseline) AND a self-consistent hybrid (PBE0 or HSE06) on the same metallic systems

#### T2. EXX vs RPA correlation contribution decomposition

The double-hybrid form has two scalar parameters α (EXX mixing) and β (RPA correlation mixing). Reviewers will demand demonstration of which contribution drives the accuracy gain — without this, the double-hybrid could be replaced by hybrid-only with no loss.

- 1: Plan treats the functional as a black box (no separate α + β)
- 2: Plan introduces α and β but no contribution analysis
- 3: Plan proposes one contribution decomposition (e.g. fix β=0 hybrid-only baseline + report MAD difference)
- 4: Plan proposes both contribution decompositions (fix β=0 hybrid; fix α=fixed hybrid-only RPA-only) with relative MAD attribution
- 5: 4 + decomposition done across multiple benchmarks (CE39 + SBH17 + SE20), revealing which functional contribution dominates which benchmark

#### T3. Compute / cost accounting in operational units

Self-consistent hybrid functionals on metallic slabs are 10-100× more expensive than GGAs. Reviewers will demand wall-clock per system or CPU-hours per benchmark to justify "practical" claim.

- 1: No compute statement
- 2: Compute mentioned in non-operational units ("computationally efficient")
- 3: Compute stated in one operational unit (CPU-hours per system, OR fold-vs-EXX-only baseline)
- 4: Compute stated in multiple operational units with per-component breakdown (orbital gen + EXX eval + RPA correlation per system) AND per-system per-benchmark
- 5: Full operational accounting + sensitivity analysis (k-point grid scaling, plane-wave cutoff convergence, basis-set extrapolation tail) showing "no more than 2× the cost of a single EXX evaluation"

#### T4. Qualitative failure modes (CO/Pt(111) puzzle, graphene/Ni(111), SCF convergence)

Standard semilocal DFAs predict the wrong CO/Pt(111) site (FCC hollow vs experimental top). A new functional must reproduce this experimental site-selectivity to clear the qualitative-failure-mode bar that BEEF-vdW + PBE both fail. Additionally, hybrid functionals on metallic slabs face SCF convergence pathologies; NSC framework sidesteps these but must address them.

- 1: No analysis of qualitative failure modes
- 2: Acknowledges potential limitations but no concrete failure-mode probe
- 3: Names ≥1 qualitative test (CO/Pt(111) site OR graphene/Ni(111) chemisorption) — one of two
- 4: Both qualitative tests included AND SCF convergence pathologies discussed for self-consistent hybrid analog
- 5: 4 + plan includes additional probe (e.g. CO on multiple TM surfaces beyond Pt; or comparison to experimental selectivity numbers) + transferability discussion of NSC framework limitations

#### T5. Out-of-distribution probe + transferability across multiple benchmark sets

A practical DFA framework must work beyond its training set. The 2-parameter fit on a 7-reaction CE39 subset is a small training set — transferability to held-out CE39 + SBH17 + SE20 + S66 + NBH56 is the load-bearing claim.

- 1: Plan trains and evaluates on the same set
- 2: Plan trains on subset A and evaluates on subset A (no held-out evaluation)
- 3: Plan trains on subset of CE39 and evaluates on full CE39 (one held-out)
- 4: Plan trains on small CE39 subset and evaluates on full CE39 + ≥2 other benchmarks (SBH17, SE20, S66, or NBH56)
- 5: Plan trains on 7-reaction CE39 subset and evaluates on full CE39 (32 held-out) + all 4 other benchmarks (SBH17 + SE20 + S66 + NBH56) AND reports per-benchmark MAD AND identifies the benchmark with worst transferability

## Combined scoring

**Total = Universal /25 + Comp-Chem-subfield /25 = /50**

For reporting, **always show both layers separately**. Note: this rubric uses 5 T-axes (not 4), so total raw is /50, not /45 like TTT-D. Cross-paper comparison should use normalized /20: (Universal × 0.55 / 25) + (Subfield × 0.45 / 25), keeping universal weight slightly higher.

## Two-pass procedure (same as base AUDIT_RUBRIC_v3)

### Pass 1: claim enumeration (mandatory)

Opus extracts:
- `claim_list` — every verifiable claim (functional form, mixing parameters, k-point grid, plane-wave cutoff, named benchmark with prior MAD value, qualitative-failure-mode test). Each with verbatim quote ≤25 words.
- `concern_list` — every reviewer-style concern (missing functional comparison, hand-waved cost, untested transferability). Each with verbatim quote ≤25 words.

### Pass 2: score each dimension with reference to enumerated claims/concerns

For each of U1-U5 and T1-T5:
- Provide score (1-5)
- Provide 1-2 sentence justification referencing specific claims or concerns
- No anti-pattern penalty floor

### Output format

```json
{
  "claim_list": [{"kind": "functional"|"hparam"|"benchmark"|"operator", "quote": "..."}],
  "concern_list": [{"kind": "missing_comparison"|"hand_waved"|"vague_hparam"|"unattributed_novelty", "quote": "..."}],
  "universal_scores": {
    "U1_soundness": {"score": int, "justification": "..."},
    "U2_significance": {...}, "U3_originality": {...}, "U4_clarity": {...}, "U5_reproducibility": {...}
  },
  "subfield_scores": {
    "T1_necessity_vs_prior_functionals": {"score": int, "justification": "..."},
    "T2_EXX_vs_RPA_decomposition": {...},
    "T3_compute_operational": {...},
    "T4_qualitative_failure_modes": {...},
    "T5_transferability_across_benchmarks": {...}
  },
  "universal_total": int (5-25),
  "subfield_total": int (5-25),
  "weighted_total_norm20": float (0-20),
  "raw_total_50": int
}
```

## Anchor calibration evidence (verbatim from Shi & Berkelbach paper, for Opus reference)

- **T1 (Necessity)** anchor 4-5: "Figure 1d benchmarks 13 functionals — PBE, RPBE, BEEF-vdW, r²SCAN-rVV10, PBE0, B3LYP, HSE06, M06, XYG3, RPA@PBE, optPBE-vdW, rev-vdW-DF2, revTPSS — on CE39."
- **T2 (Decomposition)** anchor 4-5: "The double-hybrid form has α (EXX mixing) and β (RPA correlation mixing) fitted on a 7-reaction subset of CE39."
- **T3 (Compute)** anchor 4-5: "900 CPU core-hours on average for each system in the CE39 dataset… dhBEEF-vdW@BEEF-vdW is on average less than twice the cost of a single EXX evaluation."
- **T4 (Failure modes)** anchor 4-5: "Direct use of unscreened hybrid functionals to periodic metal slabs is questionable, due to their incorrect description of the band structure" + reproduction of CO/Pt(111) top vs hollow site selectivity puzzle.
- **T5 (Transferability)** anchor 4-5: "fitting only two scalar mixing parameters on a 7-reaction subset of CE39 — achieving the first sub-chemical-accuracy (MAD 11.8 kJ/mol) DFA on 39 experimental transition-metal adsorption energies" + balanced performance on SBH17 + SE20 + S66 + NBH56.

## Locked-state evidence

- Authored: 2026-04-28 PM
- Authored before any σ_nsc_dfa/ξ_nsc_dfa run on NSC-DFA (verifiable via git timestamp)
- T1-T5 anchors derived from source paper evidence (per `CROSS_DOMAIN_SOURCE_PAPERS_v1.md:179-186`)
- U1-U5 unchanged from `AUDIT_RUBRIC_v3.md` (committed earlier)
- 5 T-axes used (not 4) per plan §8.11 — DFT methodology demands separate transferability + qualitative-failure-mode axes
