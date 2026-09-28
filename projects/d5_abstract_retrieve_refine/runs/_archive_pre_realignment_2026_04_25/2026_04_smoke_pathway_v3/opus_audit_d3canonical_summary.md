# D3 Canonical Depth Audit — D5 Smoke Pathway v3

**Auditor**: Opus 4.7 (claude-opus-4-7)
**Date**: 2026-04-23
**Prompt spec**: `src/co_scientist/ttt_discover/opus_eval_agent.py` lines 30-77 (4 dims × 1-5)
**Scored blind**: yes — no reference to v1 or v2 audit files during scoring.
**Plans**: 16 (8 B_baseline, 8 A_with_abstraction) from `buffer.jsonl`
**Oracle abstraction**: `data/smoke_v3_oracle_abstraction.md` (derivation-steps, no formula hints)

---

## Headline

| Condition | math | novelty | realism | rigor | **total** |
|---|---|---|---|---|---|
| B_baseline (n=8) | 1.000 | 1.000 | 2.875 | 1.875 | **6.75** |
| A_with_abstraction (n=8) | 2.250 | 2.000 | 3.000 | 2.000 | **9.25** |
| **Δ (A − B)** | +1.25 | +1.00 | +0.125 | +0.125 | **+2.50** |

**v3 Δ = +2.50**, versus **v1 Δ = +1.88** (pattern labels only) and **v2 Δ = +5.875** (pattern + explicit formula hints).

---

## Cross-version context

v1 isolated pattern labels (no derivation scaffolding, no formula hints): +1.88
v2 added derivation steps PLUS specific formula hints (log-sum-exp with β, Q+c·P·f(T,n(s)) form, KL=ln 2 budget, explicit target/sampler probability ratio): +5.875
v3 keeps derivation-step scaffolding but REMOVES the specific functional forms: **+2.50**

Removing formula hints eliminates roughly **(5.875 − 2.50) / 5.875 ≈ 57%** of v2's lift. v3 sits much closer to v1 (pattern-only) than to v2 (pattern + forms).

**Interpretation**: Derivation scaffolding without formula hints transfers SOME methodological reasoning — primarily the "shape" of structured derivation: enumerate failure modes, align objective with metric, identify sharpness parameter, add KL/divergence stability, correct IS mismatch, use a 3-domain deterministic-reward evaluation. But most of v2's quantitative lift came from the specific formula forms, not from the derivation-step structure.

---

## Answers to the 5 qualitative questions

### Q1. Does A > B in v3, and by how much?

Yes. **A − B = +2.50** on total (A=9.25, B=6.75). The gap is driven by math (+1.25) and novelty (+1.0); realism (+0.125) and rigor (+0.125) are essentially flat.

### Q2. Where does v3 fall relative to v1 and v2?

| Version | A total | B total | Δ |
|---|---|---|---|
| v1 (pattern labels only) | — | — | +1.88 |
| v2 (pattern + formula hints) | ~11 | ~5 | +5.875 |
| **v3 (derivation steps, no forms)** | **9.25** | **6.75** | **+2.50** |

v3 sits close to v1 and far below v2. **v3 A ≈ 9.25 is NOT ≈ v2 A (~11)** — v3 A regresses roughly 2 points toward v1 territory. Hypothesis verdict: v2's lift came **largely from leaked forms**, NOT from derivation-step structure alone.

### Q3. Do v3 A plans write objective equations?

**Yes, 2 of 8** (same count as v2, same 2/8 rate). The equations:
- **Plan 12 (A s4)**: `J(θ) = E_{τ~π_θ}[max_r R(τ)]` — max-of-reward functional, an adapted form
- **Plan 13 (A s5)**: softmax-over-rewards with temperature τ, plus an explicit limit statement "as τ → 0, the objective prioritizes the single best action" — this includes brief limit analysis

The other 6 A plans describe the softmax/KL/IS constructions in prose only (no written equations), which I scored math=2 (one standard formula mentioned by name but not written). v2's equations were more specific functional forms (log-sum-exp, Q+c·P·f); v3's equations are the more generic "max over τ" or "softmax with τ" — forms the model can plausibly derive from "sharpness parameter controlling alignment with max evaluation metric" without needing specific hints.

**Count comparison**: v1 had 0/8, v2 had 2/8, v3 has 2/8. Equation count transfers; specificity of equations regresses.

### Q4. Is realism still higher than v1's floor of 2.0?

Yes — v3 A realism = 3.0, well above v1's floor of 2.0. But v3 B realism is already 2.875, so the A-vs-B realism lift is only **+0.125**. Realism is driven by "all numerical claims consistent" (template-level coherence), not by formula specificity, so it stays high for both arms in v3. **Realism is NOT the signal that discriminates scaffolding vs no-scaffolding in v3.**

### Q5. Does scaffolding without formula hints transfer methodological reasoning?

**Partial transfer.** The lift is +2.50, compared to +5.875 when formulas are included (v2) and +1.88 with pattern labels only (v1). Removing the specific functional forms removes roughly 57% of v2's lift.

What DOES transfer from derivation scaffolding:
- Explicit failure-mode enumeration (Pattern 1) — A plans list 3-4 concrete failure modes; B plans give generic "frozen vs fine-tuning" dichotomy.
- Metric-aligned objective narrative (Pattern 2 shape) — A plans motivate a max-aligned objective from the metric; B plans default to "gradient descent on reward" without alignment argument.
- Pattern 4 warm-start / Pattern 5 divergence-budget / Pattern 6 IS-correction — named with stability considerations, though typically without written formulas.
- Pattern 7 3-domain evaluation with human / prior AI / compute-matched BoN baselines.

What does NOT transfer:
- Specific functional forms (log-sum-exp, adaptive β from KL budget, Q+c·P selection, exact IS ratio) — these need to be shown.
- Novelty ceiling (A novelty = 2.0): none of v3's A plans cross into "substantive modification with mechanistic insight" (novelty=4). They combine known techniques under explicit pattern scaffolding — the level 3 anchor "non-obvious combination with clear justification" is marginal.
- No depth on adaptive-β derivations, no MAX-vs-MEAN limit proofs, no concrete PUCT-style deviation.

**Implication for training**: if the goal is pure derivation scaffolding (no leaked forms), expect lift in the v1 range (+1-3 points) not the v2 range (+5-6 points). Phase 1+ on derivation-only training will plateau near v3 numbers unless curriculum length increases substantially. Two practical options: (a) accept formula hints as part of the training contract; (b) budget for a longer derivation-only curriculum and accept smaller per-epoch deltas.

---

## Verdict

**v3 hypothesis outcome**: v2's +5.875 lift was **primarily driven by formula hints, not by derivation-step structure alone**. Removing formula hints while keeping reasoning-step structure yields only +2.50, much closer to v1 (+1.88) than to v2 (+5.875). Derivation scaffolding without formula hints transfers roughly **2× above pattern-only** but far below the full-hint ceiling. Expect Phase 1+ training on pure derivation scaffolding to plateau near this level.

---

## Notable observations (not asked, but worth logging)

- **Template collapse within A**: 4 of 8 A plans (indices 9, 10, 14, 15) are text-identical copies of the same template. Temperature 0.7 with seed 200 + shared prompt collapses onto one "safe" pattern-labeled output for more than half the samples. Only plans 8, 11, 12, 13 show meaningful variation.
- **Template collapse within B**: plans 1, 4, 7 are also text-identical. So 3/8 B plans are the same template.
- **Effective diversity**: A has ~4 distinct plans, B has ~5 distinct plans. If we score distinct plans only the deltas would be approximately unchanged because the duplicates are all mid-range.
- **Best A plan (total=10)**: plans 12 and 13 — the two plans that wrote explicit equations AND at least one limit/boundary statement.
- **Worst A plans (total=9)**: everything else — they describe formal constructs (softmax, KL, IS) in prose without writing them, earning math=2.
- **Worst B plan (total=5)**: plan 0 (truncated at the start of Methodology). If excluded, B mean rises to 7.00 and Δ falls to +2.25 — still nowhere near v2's +5.875.
