# Audit Rubric v3 — D5 Realignment Hybrid Design

*Finalized 2026-04-25 based on REVIEWER_STANDARDS_v1.md + user decisions. Status: LOCKED for Phase 2 build.*

## User-locked decisions (2026-04-25)

- **Total dimensions**: 9 (5 universal + 4 subfield-specific) ✓
- **Subfield anchor format**: Option B — quote real reviewer comments from MTTT/Voyager/SCoRe/ReST-EM/etc. for calibration ✓
- **Weighting**: equal-rank items (each of 9 dimensions weighted equally; total /45 raw or /20 normalized = total/45 × 20)
- **Anchor strictness**: strict (5 = excellent, rare; reflects reviewer-form scale where "Strong Accept" is rare)
- **Anti-pattern penalty**: NONE — anchors themselves encode "no substance = low score"

## Goal

Replace v1 audit (4×1-5 → /20 with surface-form bias) and v2 audit (anti-pattern penalty floor saturation) with a hybrid design where:

- **Universal dimensions** (5 items) reflect what reviewers always weight (NeurIPS/ICLR/ICML/NSF universal)
- **TTT-Discover-specific dimensions** (4 items) reflect what test-time-search subfield reviewers specifically care about
- Each dimension is concrete, with substance-not-surface anchors
- Output is a structured score Opus must justify with quotes

## Two-layer rubric (Universal + Subfield-specific)

### Universal layer (5 dimensions, each 1-5 → /25 universal)

Drawn from NeurIPS Quality, ICML soundness, ICLR Q3, NSF intellectual merit. Anchors are reviewer-driven, not surface-form.

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
- 2: Sections present but mechanisms unspecified ("apply policy gradient", "use small learning rate")
- 3: Mechanisms named but reasoning chain has gaps
- 4: Mechanisms specified with rationale; reasoning chain coherent end-to-end
- 5: Clear non-specialist-readable plan; each step justified; no jargon without explanation

#### U5. Reproducibility — Are compute, hyperparameters, code/data, and statistics specified?

- 1: No mention of compute, hyperparameters, statistical methodology
- 2: 1 of {compute, hparams, eval methodology} mentioned vaguely
- 3: 2 of 3 mentioned with some specificity
- 4: All 3 stated with concrete values (e.g. compute budget in $/GPU-hr; LoRA rank=N; n=K seeds)
- 5: All stated + statistical methodology (CIs, p-values, multi-seed aggregation) + open-model commitment

### TTT-Discover-specific layer (4 dimensions, each 1-5 → /20 subfield)

Drawn from OpenReview records of MTTT, ReST-MCTS, ReST-EM, Voyager, SCoRe, Guided-ReST, DeepEvolve. These are the subfield-specific make-or-break questions reviewers consistently raise for test-time-discovery / search-with-LLMs / self-improvement papers.

#### T1. Necessity of framing — Does the plan demonstrate that test-time RL is *necessary*, not just sufficient?

The most common rejection pattern (MTTT, Voyager): "this paper claims X but tests don't actually require X." For TTT-Discover, the question is "is per-problem RL adaptation necessary, or does best-of-N on a frozen model match it?"

- 1: Plan does not address whether test-time RL is needed at all
- 2: Plan asserts test-time RL helps but no comparison to in-context-only baseline
- 3: Plan mentions a frozen-LLM baseline but doesn't quantify gap
- 4: Plan specifies a frozen-LLM best-of-N baseline with prior numbers and explains why test-time RL exceeds it
- 5: Plan explicitly characterizes the regime where test-time RL is *necessary* (e.g. specific reward sparsity, specific compute budget) and where it would NOT help

#### T2. Disentanglement — Does the plan separate LLM-prior contribution from method contribution?

Voyager critique: "Voyager does not really learn, ChatGPT does." For TTT-Discover, this means: how much of the result comes from gpt-oss-120b's pretraining vs from the test-time adaptation?

- 1: Plan treats LLM+method as one black box
- 2: Plan acknowledges base-model dependence but no ablation proposed
- 3: Plan proposes one of {frozen baseline / different base model / pretraining-only baseline}
- 4: Plan proposes ≥2 disentanglement ablations with expected results
- 5: Plan proposes full disentanglement matrix (frozen vs adapted × different bases) with explicit hypothesis on which factor dominates

#### T3. Compute / cost accounting in operational units

Guided-ReST rejected partly for FLOP-only accounting. Reviewers want $/GPU-hours/wall-clock.

- 1: No compute statement
- 2: Compute mentioned in non-operational units (e.g. "compute-efficient" or just FLOP)
- 3: Compute stated in one operational unit (e.g. GPU-hours or $)
- 4: Compute stated in multiple operational units with method-specific accounting (e.g. $X for training, $Y per inference, ~K rollouts at $Z each)
- 5: Full operational accounting + sensitivity analysis (e.g. "cost scales linearly with X; halving Y doubles cost")

#### T4. Reward-hacking / saturation analysis

SCoRe accepted *because* of reward-hacking analysis. ReST-EM cited for missing saturation characterization.

- 1: No analysis of reward-objective robustness or iteration-limit
- 2: Acknowledges potential reward hacking or saturation but no concrete pathway analysis
- 3: Names ≥1 specific Goodhart pathway OR ≥1 specific saturation mechanism; one of the two
- 4: Both analyzed: Goodhart pathways identified + saturation behavior characterized; mitigations proposed
- 5: 4 + the plan includes an explicit adversarial-probe experiment or a stop-criterion based on saturation detection

## Combined scoring

**Total = Universal /25 + TTT-specific /20 = /45**

Or normalized: total /20 weighted average (universal × 0.55 + subfield × 0.45 to keep universal slightly higher weight, since universal covers more general reviewer concerns).

For reporting, **always show both layers separately** so we know which axis a plan is strong/weak on.

## Two-pass procedure (similar to v2 but tightened)

### Pass 1: claim enumeration (mandatory)

Opus extracts:
- `claim_list` — every verifiable claim (full-RHS equation, numbered hyperparameter, named benchmark with prior number, deterministic operator). Each with verbatim quote ≤25 words.
- `concern_list` — every reviewer-style concern that arises while reading: missing comparison, hand-waved mechanism, vague hyperparameter, unattributed novelty claim. Each with verbatim quote ≤25 words.

`scaffold_list` from v2 is GENERALIZED to `concern_list` — concerns include but are not limited to "Pattern N" labels.

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
    "T1_necessity": {"score": int, "justification": "..."},
    "T2_disentanglement": {...}, "T3_compute": {...}, "T4_reward_hacking": {...}
  },
  "universal_total": int (5-25),
  "subfield_total": int (4-20),
  "weighted_total_norm20": float (0-20)
}
```

## Why hybrid (not pure universal, not pure subfield-specific)

| Approach | Pros | Cons |
|---|---|---|
| Pure universal (4×1-5) | Reusable across goals; matches v1 | Surface-form bias confirmed; can't catch subfield-specific anti-patterns |
| Pure subfield-specific | Highly tuned to TTT-Discover; catches "necessity" / "disentanglement" / "compute" cleanly | Can't generalize to other goals; need rewrite per goal |
| **Hybrid** | Universal layer reusable; subfield layer customized; explicit two-axis reporting reveals where plans fail | More complex prompt; more tokens per audit |

Per user's stated preference ("肯定是由 universal 和类似一些 domain 或者 goal specific 的这两部分组成的"), hybrid is the user-confirmed direction.

## Implementation plan (Phase 2)

After Phase 1C, Phase 2 will produce:

1. **`src/co_scientist/shared/audit_prompt_v3.py`** — final v3 prompt + parser. 9-dim hybrid, Option B anchors with embedded reviewer quotes (e.g. T1 anchor 1 cites MTTT review verbatim). Output JSON with `claim_list` + `concern_list` + `universal_scores` + `subfield_scores` + `universal_total` + `subfield_total` + `weighted_total_norm20`.

2. **Daemon integration**: extend `opus_critic_audit_daemon_v2_addendum.md` with v3 branch. `prompt_version: "v3"` triggers new prompt.

3. **Reuse infrastructure**: `OpusAuditClient.collect_all()` already extracts `per_dim_scores` keyed on `math/novelty/realism/rigor` (v1) or `per_dim_scores` (v2). v3 replaces with `universal_scores` (5 keys) + `subfield_scores` (4 keys); aggregator updated accordingly.

## Required reviewer-quote material for Option B anchors

When the audit_prompt_v3.py is built, embed these verbatim quotes from REVIEWER_STANDARDS_v1.md Part B as anchor calibrators:

- **T1 (Necessity)** anchor 1 quote: *"This paper motivates from the TTT perspective, but no TTT experiments are performed."* — MTTT reviewer DvfS
- **T2 (Disentanglement)** anchor 1 quote: *"Voyager does not really learn, ChatGPT does."* — Voyager reviewer eudD
- **T3 (Compute)** anchor 2 quote: *"No actual latency or compute cost accounting; claims based only on token budgets."* — Guided-ReST reviewer xv2X
- **T4 (Reward-hacking)** anchor 4 quote: *"Reward shaping with α>1 could incentivize the model to introduce minor errors in first steps to enable correction."* — SCoRe reviewer e4kn (positive: this analysis was rewarded)

Each anchor explicitly cites the source paper + venue so Opus calibrates against real review-language norms.

## Pairwise cross-validation discipline (added 2026-04-27)

Absolute audit alone is insufficient for **close-cluster** comparisons (gap < ~3
points). Phase 0.6 had a precedent inversion: μ-v2 vs δ scored **+2.13** on
absolute audit but **0-8** on pairwise — surface-form bias from oracle's "Pattern X"
template made μ-v2 look rigorous while δ's reference-paraphrased concrete specs
actually won when forced into discrimination.

**Discipline going forward**: any paper-claim headline that depends on a
close-cluster gap must ALSO be cross-checked by Opus pairwise tournament (≥8
position-randomized pairs per matchup). See:

- `paper_materials/methodology/M4_audit_isolated_methodology.md` § "Cross-validation discipline"
- `src/co_scientist/d5_abstract_retrieve_refine/mu_v4_pairwise_v1.py` (reusable runner;
  fork for any baseline-vs-baseline comparison)
- `src/co_scientist/shared/opus_pairwise_subagent.py` (file-bus client)

Cost per matchup: ~$4 (8 Opus pairwise calls). Wall: ~5 min with parallel subagent
fan-out. Inclusion threshold for paper: any close-cluster claim where gap < 3 / 45.

**Verified case study**: μ-v4's +2.75 over σ was cross-checked this way and
PASSED at 6-2 vs σ (PRIMARY clears ≥6/8 threshold), 7-1 vs δ, 7-1 vs α. See
`paper_materials/experiments/E6_mu_v4_pairwise_corroboration.json`.

**Wide-gap comparisons** (e.g., ε vs σ at +8.37, β vs μ-v4 at -5.25) do NOT
require pairwise cross-check — absolute-audit gap is too large for surface bias
to plausibly flip.
