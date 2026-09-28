# Archive: D5 Phase 0.6 work (pre-realignment)

**Archived 2026-04-25.** All run dirs in this folder are from before the D5 realignment.

## Why archived

The user identified that the prior work was misaligned with the intended D5 pipeline:

1. **Oracle abstraction structure was wrong** — `smoke_v3_oracle_abstraction.md` was a single-round
   derivation-scaffolding extraction from one reference plan, not a multi-round bibliography-grounded
   citation-cited extraction. This means smoke_v3_A and μ runs tested a different oracle than what the
   real pipeline would produce.

2. **δ / ε were treated as baselines** when they're actually semi-oracle conditions (full reference plan
   in prompt). All comparisons μ-vs-δ etc. are unfair.

3. **Audit prompt v1/v2 had surface-form bias / floor saturation** — neither produces reliable absolute
   scores within the bottom tier.

4. **No reviewer-standard grounding** — the audit dimensions were invented top-down rather than
   derived from what real reviewers focus on.

The realignment redesigns oracle structure (6 categories: Insights/Methodology/Theory/Math/Empirical/Failure-modes,
Opus-generated as ceiling estimate, multi-round cumulative, real bibliography citations) and audit rubric
(9 dimensions hybrid: 5 universal + 4 TTT-Discover-specific, anchored to real reviewer quotes from MTTT/Voyager/
SCoRe/etc.).

See:
- `projects/d5_abstract_retrieve_refine/knowledge/current/REVIEWER_STANDARDS_v1.md` — synthesis of NeurIPS/ICLR/
  ICML reviewer guidelines + 7 OpenReview adjacent papers
- `projects/d5_abstract_retrieve_refine/knowledge/current/ORACLE_DESIGN_v1.md` — finalized oracle design
- `projects/d5_abstract_retrieve_refine/knowledge/current/AUDIT_RUBRIC_v3.md` — finalized audit rubric

## DO NOT reference these archived runs as authoritative

The numbers in archived runs (μ proxy = 8.88/20, "D5 dead" or "D5 alive" verdicts, etc.) are based on
the wrong oracle and unfair baselines. Don't cite them as evidence about D5's value.

The pairwise tournament data (`_archive_pre_realignment_2026_04_25/2026_04_25_pairwise_v1/`) IS still useful
as a methodology demonstration — it showed that pairwise + audit can disagree, which informed the audit
v3 design. But the per-baseline rankings should not be trusted.

## What's preserved

- `2026_04_25_mu_baseline_v1/` — μ training run (10 iters, SDPO + critic) with old oracle
- `2026_04_25_alpha_baseline_v1/` — α (Opus distillation) with old config
- `2026_04_25_beta_baseline_v1/` — β (ref SFT)
- `2026_04_25_pairwise_v1/` — pairwise tournament (12 matchups across the run lifetime)
- `2026_04_25_audit_v2_validation/` — audit v2 attempt (failed strict gate)
- `2026_04_25_mu_dryrun_v1/` — μ 1-iter dry run
- `2026_04_baseline_delta_epsilon/` — δ + ε 30B/235B + reference-in-prompt
- `2026_04_smoke_pathway_v1/v2/v3/v2_rerun_div/v3_rerun_div/` — early smoke pathway tests with old oracle
- `sampling_fix_comparison.md` — Phase 0.5c sampling-fix doc

Code (`src/co_scientist/d5_abstract_retrieve_refine/`) is NOT archived — the trainer skeleton, file-bus
clients, etc. are reusable infrastructure. New realignment runs will write `_v2` versions where structure
changes.
