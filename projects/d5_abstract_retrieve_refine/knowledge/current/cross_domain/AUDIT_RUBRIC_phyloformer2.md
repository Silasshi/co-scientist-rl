# Audit Rubric v3 — Phyloformer 2 / Likelihood-Free Phylogenetic Inference (Bio)

*Domain-specific rubric for cross-paper audit on Phyloformer 2 (Blassel et al., arxiv 2510.12976). U1-U5 verbatim from AUDIT_RUBRIC_v3.md:28-66 (universal NeurIPS/ICLR/ICML/NSF axes). T1-T4 PASS-as-is per CROSS_DOMAIN_SOURCE_PAPERS_v1.md catalog — Phyloformer 2 already has reviewer-grade T1-T4 evidence with verified quotes from the source paper.*

*Status: LOCKED for cross-paper audit. Authored 2026-04-28 PM, BEFORE any σ_phyloformer2/ξ_phyloformer2 run. Future edits require explicit retraction notice.*

## User-locked decisions (carried over from AUDIT_RUBRIC_v3.md)

- **Total dimensions**: 9 (5 universal + 4 subfield-specific)
- **Anchor format**: Option B — quote real reviewer comments where available; for Phyloformer 2, T1-T4 anchors quote the paper's own benchmarks/ablations
- **Weighting**: equal-rank (each dim 1-5; total /45 raw or /20 normalized)
- **Anchor strictness**: strict (5 = excellent, rare)
- **Anti-pattern penalty**: NONE — anchors encode "no substance = low score"

## Goal

Audit cross-paper plans (for the Phyloformer 2 research goal) on the same 9-dim hybrid rubric design as TTT-D, with the 4 subfield axes adapted to evolutionary genomics / probabilistic ML / phylogenetic inference. Universal axes carry over verbatim — they reflect what reviewers always weight regardless of domain.

## Two-layer rubric (Universal + Bio-subfield-specific)

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
- 4: All 3 stated with concrete values (e.g. compute budget in $/GPU-hr; LoRA rank=N; n=K seeds)
- 5: All stated + statistical methodology (CIs, p-values, multi-seed aggregation) + open-model commitment

### Bio-subfield-specific layer (4 dimensions, each 1-5 → /20)

Drawn from Phyloformer 2 (arxiv 2510.12976) source-paper evidence verified in `CROSS_DOMAIN_SOURCE_PAPERS_v1.md:88-93`. T1-T4 PASS-as-is (i.e. the same axes the source-paper authors themselves benchmarked on).

#### T1. Necessity vs likelihood-based / classical phylogenetic baselines

The most common rejection pattern for new phylogenetic methods: "you don't actually outperform the existing maximum-likelihood + Bayesian methods on the standard simulated data + topological-distance metrics." Phyloformer 2 paper Figure 2a + Section 4.1 benchmark vs **6 baselines: IQ-TREE, FastTree, FastME, Phyloformer-v1, RevBayes (MCMC), CherryML** across 10–200 taxa.

- 1: Plan does not address comparison vs existing phylogenetic baselines at all
- 2: Plan asserts neural method helps but no comparison to even one likelihood-based baseline
- 3: Plan mentions ≥1 likelihood-based baseline (IQ-TREE / FastTree / RevBayes / FastME) but doesn't quantify gap
- 4: Plan specifies ≥3 baselines incl. ≥1 from each tier {distance-based: FastTree/FastME, ML: IQ-TREE/PhyML, Bayesian: RevBayes/MrBayes} with prior numbers and explains expected delta
- 5: Plan explicitly characterizes the regime where the proposed method is *necessary* (e.g. specific reward sparsity / specific compute budget / specific evolutionary models like Cherry / SelReg / Potts where likelihood is intractable)

#### T2. Disentanglement — Architecture vs training-protocol vs prior-knowledge contribution

Phyloformer 2 paper has explicit ablation `PF2_MAE` that replaces BayesNJ canonical merge order with mean-absolute-error distance loss → "most of the topological accuracy gain is due to the BayesNJ loss". This is the canonical "what fraction of the gain comes from architecture vs loss" disentanglement that reviewers always demand.

- 1: Plan treats {neural network architecture, training loss, sequence-pair encoding} as one black box
- 2: Plan acknowledges architectural choice (e.g. transformer vs CNN) but no ablation proposed
- 3: Plan proposes one of {architecture ablation / loss ablation / training-data ablation}
- 4: Plan proposes ≥2 disentanglement ablations with expected results
- 5: Plan proposes full disentanglement matrix (architecture × loss × pre-training × encoding) with explicit hypothesis on which factor dominates

#### T3. Compute / cost accounting in operational units

Phyloformer 2 paper states: "by one order of magnitude compared to FastTree, two compared to IQ-TREE"; memory ceiling 200 taxa on V100 16GB; Figure 2b/2c. This is a paper-level reviewer expectation: state wall-clock per-query (seconds for amortized inference, hours for retraining) in a unit comparable to existing tools.

- 1: No compute statement
- 2: Compute mentioned in non-operational units (e.g. "compute-efficient" or just "FLOP / parameter count")
- 3: Compute stated in one operational unit (e.g. wall-clock minutes per query for amortized inference, OR GPU-hours for training)
- 4: Compute stated in multiple operational units with method-specific accounting (e.g. amortized inference: X seconds per tree; one-shot training: Y GPU-hours; memory: Z GB at K taxa)
- 5: Full operational accounting + sensitivity analysis (e.g. "wall-clock scales as O(N²) in taxa; memory ceiling at K taxa; vs IQ-TREE's O(N⁴) likelihood evaluation")

#### T4. Distribution shift / model-misspecification analysis

Phyloformer 2 paper's strongest reviewer-defense move: explicitly evaluating on **Cherry / SelReg / Potts models that break the i.i.d.-position assumption** (canonical likelihood-method failure mode), with honest distribution-shift caveats in Conclusion. The bio analog of TTT-D's reward-hacking analysis: any neural posterior estimation can fail when training and test simulation models diverge.

- 1: No analysis of model robustness or simulation-vs-empirical mismatch
- 2: Acknowledges potential simulation-empirical mismatch but no concrete pathway analysis
- 3: Names ≥1 specific model misspecification (e.g. site-rate-heterogeneity, codon usage bias, branch-length non-stationarity) OR ≥1 specific saturation regime (high taxa count / long branches) — one of the two
- 4: Both analyzed: misspecification pathways identified + saturation behavior characterized; mitigations proposed
- 5: 4 + plan includes an explicit out-of-distribution probe experiment (e.g. evaluate on co-evolutionary models like Cherry/SelReg/Potts whose likelihood is intractable) OR a stop-criterion based on quality detection

## Combined scoring

**Total = Universal /25 + Bio-subfield /20 = /45**

For reporting, **always show both layers separately** so we know which axis a plan is strong/weak on.

## Two-pass procedure (same as base AUDIT_RUBRIC_v3)

### Pass 1: claim enumeration (mandatory)

Opus extracts:
- `claim_list` — every verifiable claim (full-RHS equation, numbered hyperparameter, named benchmark with prior number, deterministic operator). Each with verbatim quote ≤25 words.
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
    "T1_necessity_vs_phylo_baselines": {"score": int, "justification": "..."},
    "T2_arch_vs_loss_disentanglement": {...},
    "T3_compute_operational": {...},
    "T4_misspecification_analysis": {...}
  },
  "universal_total": int (5-25),
  "subfield_total": int (4-20),
  "weighted_total_norm20": float (0-20),
  "raw_total_45": int
}
```

## Anchor calibration evidence (verbatim from Phyloformer 2 paper, for Opus reference)

- **T1 (Necessity)** anchor 4-5 quote: *"In particular, our method excels under sequence evolution models breaking the i.i.d. position assumption such as Potts, SelReg or Cherry, where likelihoods are intractable and only approximations or alternative methods can be used."*
- **T2 (Disentanglement)** anchor 4-5 quote: *"PF2_MAE replaces the BayesNJ loss with a mean-absolute-error distance loss, which still allows BayesNJ inference at test time but suppresses the canonical merge-order signal during training. We find that PF2_MAE recovers most of the topological accuracy gain."*
- **T3 (Compute)** anchor 4-5 quote: *"In terms of wall-clock time, our method is faster by one order of magnitude compared to FastTree and two compared to IQ-TREE."*
- **T4 (Misspecification)** anchor 4-5 quote: *"Critically, the advantage of likelihood-free inference over likelihood-based methods becomes substantial under sequence evolution models with intractable likelihoods, such as those incorporating site-coupling or selection..."*

These are not exhaustive — they are anchor exemplars for Opus's calibration. Audit Opus is instructed to read each plan's claims against these anchors and score 1-5 with justification per pass-2 procedure.

## Locked-state evidence

- Authored: 2026-04-28 PM
- Authored before any σ_phyloformer2/ξ_phyloformer2 run on Phyloformer 2 (verifiable via git timestamp on this file vs run dirs)
- T1-T4 anchors quote source paper directly; not domain-author judgment
- U1-U5 unchanged from AUDIT_RUBRIC_v3.md (committed earlier)
