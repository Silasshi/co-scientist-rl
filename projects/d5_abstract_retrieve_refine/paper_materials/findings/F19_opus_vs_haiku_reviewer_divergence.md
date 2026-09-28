# F19: Opus vs Haiku reviewer divergence at close margins

**Filed**: 2026-04-29 PM
**Status**: New finding from re-running Opus pairwise tournament with Haiku reviewer

## Claim

Pairwise cell ranking is **reviewer-dependent at close margins**. Both reviewers agree on the wide-margin matchup (L2 vs μ-v4: ~88% agreement, both pick L2) but disagree on the close-margin matchup (ε vs L2: only 37.5% agreement, rankings invert). Specifically:

- **Opus ranking**: ε > L2 > μ-v4
- **Haiku ranking**: **L2 > ε** > μ-v4 (L2 vs ε flipped)

This is a paper-grade methodological finding: any v8/v9 conclusion drawn from a single reviewer's verdict at <5pt margin is reviewer-bound.

## Setup

Same 24 matchups (3 cell pairs × 8 plans). Re-ran 16 Opus matchups (μ-v4 vs L2 + ε vs L2) under Haiku reviewer. The other 8 (μ-v4 vs ε) only have Haiku verdicts (Opus rate-limited).

**Total verdicts**:
- Opus: 16 matchups (mu_v4_vs_L2 × 8, epsilon_vs_L2 × 8)
- Haiku: 24 matchups (all 3 pairs × 8)
- Compared: 16 (the overlap)

## Results

### Per-matchup agreement

| Matchup | Opus | Haiku | Agree? |
|---|---|---|:-:|
| μ-v4 vs L2 pair 0 | L2 | L2 | ✓ |
| μ-v4 vs L2 pair 1 | L2 | L2 | ✓ |
| μ-v4 vs L2 pair 2 | L2 | L2 | ✓ |
| μ-v4 vs L2 pair 3 | L2 | L2 | ✓ |
| μ-v4 vs L2 pair 4 | L2 | L2 | ✓ |
| μ-v4 vs L2 pair 5 | L2 | μ-v4 | ✗ |
| μ-v4 vs L2 pair 6 | L2 | L2 | ✓ |
| μ-v4 vs L2 pair 7 | L2 | L2 | ✓ |
| ε vs L2 pair 0 | ε | L2 | ✗ |
| ε vs L2 pair 1 | ε | ε | ✓ |
| ε vs L2 pair 2 | ε | L2 | ✗ |
| ε vs L2 pair 3 | ε | ε | ✓ |
| ε vs L2 pair 4 | tie | L2 | ~ |
| ε vs L2 pair 5 | L2 | ε | ✗ |
| ε vs L2 pair 6 | L2 | L2 | ✓ |
| ε vs L2 pair 7 | ε | L2 | ✗ |

**Tally**:
- Agree: 10/16 (62.5%)
- Tie partial: 1/16
- Flip: 5/16 (31%)

### Per-pair tally divergence

| Pair | Opus tally | Haiku tally | Inversion? |
|---|---|---|:-:|
| μ-v4 vs L2 | L2: 8, μ-v4: 0 | L2: 7, μ-v4: 1 | ✗ (no — both agree L2 wins) |
| ε vs L2 | ε: 5, tie: 1, L2: 2 | **L2: 5**, ε: 3 | ✓ (YES — winner inverts) |

### Cell-level ranking

| Reviewer | Ranking |
|---|---|
| Opus | ε > L2 > μ-v4 |
| Haiku | **L2 > ε** > μ-v4 |

## Mechanism hypotheses

**1. Haiku over-weights surface specificity**
Haiku verdicts repeatedly cite "L2 specifies LoRA r=64, β=8, GPU-hours, statistical methodology" as decisive. L2 plans are dense with explicit hyperparameters because they were trained under verifier reward + KL anchor toward π_oracle, which produces "checklist-style" outputs. Haiku's reasoning treats this as decisive substance even when ε has tighter mechanism-level coherence.

**2. Opus catches mechanism-level depth**
Opus verdicts on ε wins cite mechanism-level qualities: "ε's UCB formula + adaptive 5% rule is concrete and novel" or "ε reads as a concrete method with traceable cost model". Haiku's evaluation cannot reliably distinguish novel-mechanism-with-concrete-equations from toolkit-recombination-with-named-hyperparameters.

**3. ε's 235B + ref-plan output is dense, holistic**
ε plans are deep, integrated outputs from 235B + reference-plan-in-context. They lack the "checklist appearance" but carry more semantic depth. Haiku misses this; Opus catches it.

## Implications

**Reviewer-robust claim**: L2 sweeps μ-v4 (7-8 wins under both reviewers, 87.5% agreement). v9 KL anchor lifts ceiling above v1-v7 era ceiling. **This claim survives reviewer choice.**

**Reviewer-dependent claim**: L2 vs ε ranking. **This claim does NOT survive reviewer choice** — Opus says ε > L2, Haiku says L2 > ε.

**Paper framing pivot**: previous finding was "L2 approaches but does not equal ε ceiling (5-3 deficit vs ε under Opus)". Updated framing: **"L2 vs ε is at reviewer noise floor — at-or-near ε under Opus, surpassing ε under Haiku. Reviewer choice flips the verdict."**

**For audit-rubric design**: the only reliable claim about RL-trained 30B vs 235B+ref-plan is "they are within reviewer-noise". Anyone wanting a definitive verdict needs:
1. Multiple reviewers (Opus + Haiku + 3rd party) with explicit aggregation rule
2. OR a non-reviewer-based metric (e.g., hit-rate on gold equations + citations + empirical anchors as in F18 multi-axis grounding)

## Connection to F18

F18 (KL anchor stabilizes + lifts) was based on the Opus pairwise verdict (ε > L2 > μ-v4). Under Haiku verdict (L2 > ε > μ-v4), F18's "lifts ceiling" claim becomes **stronger** (L2 even surpasses 235B + ref-plan ceiling under Haiku). But neither claim is reviewer-robust.

**Updated F18 + F19 combined claim**:
- L2 unambiguously beats μ-v4 under both reviewers ✓
- L2 vs ε is at reviewer noise floor ⚠
- Final paper-grade ranking requires multi-reviewer aggregation or non-subjective metric

## Cost

- Opus 16 matchups: ~$0.80
- Haiku 24 matchups (16 reaudit + 8 mu_v4_vs_ε): ~$0.10
- **Total: ~$0.90** for full reviewer-divergence characterization

## Files

- Opus verdicts: `runs/2026_04_29_pairwise_3way/verdict_*.json` (24 files: 16 Opus + 8 Haiku for mu_v4_vs_ε)
- Haiku reaudit: `runs/2026_04_29_pairwise_3way/haiku_reaudit/verdict_*.json` (16 files)
- Matchup definitions: `runs/2026_04_29_pairwise_3way/matchups.json`
- Joint analysis: `RESULTS_pairwise_3way_tournament.md`
- This finding: `paper_materials/findings/F19_opus_vs_haiku_reviewer_divergence.md`
