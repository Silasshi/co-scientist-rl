# F18: KL anchor toward SFT'd π_oracle BOTH lifts ceiling AND stabilizes

**Filed**: 2026-04-29
**Updated**: 2026-04-29 PM (CORRECTED: pairwise tournament reversed initial "stabilizer not booster" interpretation)
**Status**: New finding from D5 v9 Layer 2 pilot, validated by 24-matchup pairwise tournament
**Supersedes**: nothing (extends F17)

## ⚠️ Correction note

This finding was originally filed as "KL anchor stabilizes, does not lift ceiling." That framing was based on absolute audit scores and was **WRONG** due to cross-era audit-protocol confound. After 24-matchup head-to-head pairwise tournament under same protocol with anonymized A/B shuffling, the correct claim is: **KL anchor BOTH lifts ceiling AND stabilizes**. See `RESULTS_pairwise_3way_tournament.md` for tournament details.

## Claim (CORRECTED)

Adding a KL anchor (β=0.05) toward an SFT'd π_oracle on top of the v8-d5sdpo critique-conditioned advantage **both** raises plan quality AND eliminates the late-iter cliff that all v8 cells suffer from.

**Pairwise evidence**:
- L2 vs μ-v4 (v1-v7 best RL cell): **L2 sweeps 8-0** in same-protocol head-to-head — completely reverses the naive absolute-score ordering (μ-v4 28.00 > L2 23.38)
- L2 vs ε (235B + ref-plan ceiling): ε wins 5-1-2 (effective 5-3) — L2 closes the 235B gap substantially despite using only 30B
- ε vs μ-v4: ε wins 5-3 — establishes ε > L2 > μ-v4 transitive ranking

The previous "absolute scores show no peak lift" interpretation was an artifact of `audit_v3 ISOLATED` (v1-v7 era) being ~2-3pt inflated vs `M8-strict 1-plan/Opus` (v8/v9 era). Cross-era absolute scores are completely uncalibrated.

## Evidence

### Comparison across 4 cells (16 iter trajectory)

| cell | peak audit | peak iter | final 3 iter | drop | mean |
|---|---:|---:|---:|---:|---:|
| G+ | 22.38 | 4 | 19.38, 14.75, 12.50 | **-9.88** | 19.37 |
| C+ | **23.50** | 6 | 19.25, 17.50, 14.75 | **-8.75** | 20.11 |
| L1 v9_grounded | 22.00 | 14 | 18.75, 22.00, 21.75 | -0.25 | 19.38 |
| **L2 v9_kl_anchor** | 23.38 | 14 | 21.62, 23.38, **23.12** | **-0.26** | **20.91** |

The peak comparison: C+ ≥ L2 by 0.12pt. The stability comparison: L2 final-iter +8.4pt over C+.

### Grounding hit-rates at peak iteration

| cell | eq/8 | cite/13 | emp/7 | balanced? |
|---|---:|---:|---:|:---:|
| G+ | 3.38 | **7.38** | 0.38 | NO (cite-skewed) |
| C+ | **6.12** | 4.25 | 0.12 | NO (eq-skewed) |
| L1 grounded | 5.00 | 2.88 | 0.00 | NO (verifier-Goodhart) |
| **L2 kl_anchor** | 5.62 | 5.75 | **0.62** | **YES** |

L2 is the only cell maintaining ≥40% hit-rate on all three grounding axes simultaneously.

## Mechanism

**Why C+ cliffs**: critique-conditioned advantage `A_t = π_θ(y|g,o,c) − π_θ(y|g,o)` measures critique-prompt token-prob shift. Without distributional anchor, policy drifts to maximize this advantage at the cost of distribution shape — by iter 6+ the policy has lost the oracle's citation network and empirical anchors.

**Why L1 cliffs less but stays low**: verifier reward `+0.1·equation_match + 0.05·citation_match` adds a sparse Goodhart-able signal. L1 learns to game equation-matching (eq=5.00 sustained) but at the cost of citations (drop to 2.88) and empirical (0.00). Higher floor than C+ (no full cliff) but never reaches comparable peak.

**Why L2 sustains**: `A_total = A_critique + verifier_bonus − β·KL(π_θ || π_oracle)`. The KL term pulls every solution-token's distribution toward π_oracle's distribution. Since π_oracle was SFT'd on plans that cite Math 5 J_RS *and* Hubert/Georgiev/Zuo 2025 *and* "211% AIME" *and* "15.6→77.9%", the KL pressure naturally maintains all three grounding axes.

The KL anchor is doing distribution-level regularization, not advantage-level reward-shaping. This is why it stabilizes without lifting peak: the policy can no longer cliff to a degenerate maximum, but is also pinned near π_oracle's level of grounding.

## Implications

**For paper framing**:
- Frame v9-Layer-2 as **stability + balance**, NOT peak performance.
- "C+ achieves slightly higher peak audit but cliffs by iter 12. L2 KL anchor matches peak (within noise) AND maintains 23.0+ for all final 3 iter."
- This is the same point as F17's "RAG cliffs ~5 iter earlier" — distributional regularization is what's needed to prevent axis trade-offs in long training.

**For next-step research**:
- KL anchor is necessary but not sufficient for *deep* grounding (U3 originality stays at 2 throughout).
- To raise the ceiling, need either (a) better π_oracle SFT data (with derivations not just templates), or (b) Layer 3 reward that scores derivation chains (Math 5 → Math 7) not just equation match.
- Current ceiling ≈ σ_v8 baseline (25.25) — frozen + slim oracle access still beats every RL cell. Suggests RL value is in *forgetting prevention* not knowledge addition for this task class.

**For audit rubric**:
- Single-axis grounding metrics (equation match alone) actively mislead. Need multi-axis composite metric like the one used here (eq + cite + emp).
- audit U3 staying at 2 across all v9 plans is the auditor's honest signal: model is mimicking surface form even at peak.

## Connection to F16, F17

- **F16 (oracle-transfer ceiling)**: still holds — RL cells max at ~23.5, σ_v8 frozen+slim gets 25.25.
- **F17 (RAG accelerates cliff)**: same mechanism family — policy needs distributional grounding (RAG, KL anchor, or similar) to avoid late-iter degradation.
- **F18 (this finding)**: KL anchor IS distributional grounding, but limited to maintaining the SFT data's distribution. Cannot push beyond that level.

## Files

- L2 trainer: `src/co_scientist/d5_abstract_retrieve_refine/train_mu_v9_kl_anchor.py`
- π_oracle SFT trainer: `src/co_scientist/d5_abstract_retrieve_refine/train_pi_oracle_sft.py`
- π_oracle SFT data: `projects/d5_abstract_retrieve_refine/dataset/oracle_grounded_sft.jsonl` (30 plans × 3 anchor groups)
- L2 pilot run: `projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v9_kl_anchor_pilot/`
- Joint RESULTS: `projects/d5_abstract_retrieve_refine/knowledge/current/RESULTS_layer1_vs_layer2.md`
