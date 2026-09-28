# Signal Set v9 (Signal Hardening)

**Last updated**: 2026-04-18
**Code**: `src/co_scientist/shared/ten_signal_reward.py`
**Grader model**: Qwen/Qwen3-30B-A3B (default), Qwen/Qwen3-235B-A22B-Instruct-2507 (SA_arithmetic)
**Grader mode**: separate_call (one grader call per signal per plan)
**Grader repeats**: N=2 (median)
**Grader max_tokens**: 4096

---

## Motivation: Opus Cross-Family Audit

CR-v7 paper runs (2026-04-18) showed training grader (Qwen3-30B) saturates
at 0.975 while Opus 4.7 external depth audit scores 7-8/20 (reference 20/20).
Spearman ρ(training grader, Opus) = -0.16. Gap analysis:

| Opus Dimension | Gap | v8.1 Signal | Failure Mode |
|---|---|---|---|
| Math formalism | -3.75/5 | S2_rigor | Checks baselines, not formulas. CR-v7 gamed S2 2→5 via headers |
| Algorithmic novelty | -3.375/5 | S1/S9 | Check justification, not insight |
| Implementation realism | -2.625/5 | S5/S7 | S7 counts names, no math validation |
| Empirical rigor | -3.0/5 | S2 partial | Checks criterion syntax, not prior-number grounding |

v9 adds two signals targeting the largest gaps: S2a_formalism (math) and
SA_arithmetic (implementation realism). Weights rebalanced so depth signals
have commensurate weight with structural signals.

---

## Version History

| Version | Date | Key change | Ref aggregate |
|---------|------|-----------|---------------|
| v8.1 | 2026-04-15 | S4 GATE loosened + Occam weights | 0.857 |
| **v9** | **2026-04-18** | **S2a_formalism + SA_arithmetic + rebalance** | **TBD** |

---

## Layer 0: Hard Gates (unchanged from v8.1)

| Gate | What it checks | Threshold |
|---|---|---|
| **G1: Goal-Contrast Margin** | Is the plan specific to THIS goal? | margin ≥ 0.10 |
| **G2: Claim Verification** | Are factual claims real? | fabrication ratio < 15% |

If either gate fails → aggregate reward = 0.

## Layer 1: Gradient Signals (1-5 integer scale)

### v9 Weights

| # | ID | Name | Weight | Grader | Status |
|---|---|---|---|---|---|
| S1 | S1_depth | Reasoning Depth | 0.07 | 30B | Active |
| S2 | S2_rigor | Baseline Rigor | 0.08 | 30B | Active |
| **S2a** | **S2a_formalism** | **Mathematical Formalism** | **0.10** | **30B** | **NEW v9** |
| S3 | S3_positioning | Positioning | 0.10 | 30B | Active |
| S4 | S4_significance | Significance | 0.00 | — | Disabled |
| S5 | S5_feasibility | Feasibility Evaluability | 0.10 | 30B | Active |
| S6 | S6_risk_awareness | Mature Risk Awareness | 0.10 | 30B | Active |
| S7 | S7_specificity | Implementation Specificity | 0.10 | 30B | Active |
| S8 | S8_scope | Scope-Generalization | 0.13 | 30B | Active |
| S9 | S9_focus | Research Focus | 0.14 | 30B | Active |
| **SA** | **SA_arithmetic** | **Arithmetic Consistency** | **0.08** | **235B** | **NEW v9** |

**Key weight shift**: Depth signals (S2a + SA + S1) combined = 0.25.
Structural signals (S8 + S9) combined = 0.27. Previously: depth = 0.16
vs structural = 0.37.

### Weight change rationale

| Signal | v8.1 | v9 | Change |
|---|---|---|---|
| S1_depth | 0.03 | 0.07 | +0.04: addresses novelty gap |
| S2_rigor | 0.13 | 0.08 | -0.05: coverage split with S2a |
| S2a_formalism | — | 0.10 | NEW: #1 Opus gap |
| S3_positioning | 0.12 | 0.10 | -0.02 |
| S5_feasibility | 0.10 | 0.10 | unchanged |
| S6_risk_awareness | 0.13 | 0.10 | -0.03 |
| S7_specificity | 0.12 | 0.10 | -0.02 |
| S8_scope | 0.18 | 0.13 | -0.05: was overweighted |
| S9_focus | 0.19 | 0.14 | -0.05: was overweighted |
| SA_arithmetic | — | 0.08 | NEW: implementation realism gap |

---

## New Signal: S2a_formalism (Mathematical Formalism)

**What it measures**: Count of NON-TRIVIAL mathematical formulas in the plan.

**Why added**: Opus gap -3.75/5. S2_rigor measures baseline fairness
(FAIR/STRAWMAN counting) but does not check whether the plan contains
actual mathematical content. CR-v7 gamed S2 from 2/5 to 5/5 by adding
"Hypothesis" section headers without any formulas.

**Rubric (deterministic count → score)**:
- 1: 0 non-trivial formulas
- 2: 1 standard textbook formula (plain REINFORCE, cross-entropy, KL)
- 3: 1 adapted/novel formula for this plan
- 4: 2+ formulas with ≥1 non-textbook
- 5: 3+ formulas forming a coherent derivation chain

**CoT scaffolding**: Explicit line-by-line scan. Each formula classified as
TAUTOLOGICAL (L=-R), HYPERPARAMETER-ONLY (lr=3e-5), or NON-TRIVIAL.
Deterministic table from N_nontrivial count.

**Strict rules**:
- Hyperparameter values are NEVER equations
- Section headers with math words are NOT formulas
- LaTeX formatting of a number is NOT a formula

**Grader**: Qwen3-30B (formula counting is lexical/syntactic).

---

## New Signal: SA_arithmetic (Arithmetic Consistency)

**What it measures**: Internal consistency of numerical claims (parameter
counts, compute budgets, citations, units).

**Why added**: Opus gap -2.625/5. No existing signal validates arithmetic.
Opus found: LoRA params off by 5x, FLOPs off by 10,000x, hallucinated
arXiv URLs — all scored S7=5 by Qwen3-30B because S7 counts named
commitments, not validates them.

**Rubric (deterministic count → score)**:
- 1: ≥3 arithmetic errors
- 2: 2 errors
- 3: 1 error, or too few claims to verify (cap at 3)
- 4: 0 errors, ≥3 verifiable claims
- 5: 0 errors, ≥5 verifiable claims, all consistent

**CoT scaffolding**: List up to 8 numerical claims. For each, do
order-of-magnitude check (7B model → ~7e9 params, forward pass →
2×N×T FLOPs/token). Mark CONSISTENT / INCONSISTENT / UNVERIFIABLE.

**Grader**: Qwen3-235B-A22B (needs stronger math reasoning for
order-of-magnitude checks). Routed via `grader_model_override` field
in SignalSpec.

---

## Implementation Details

### grader_model_override routing

`SignalSpec` dataclass gained a `grader_model_override: str | None` field.
When set, `train_buffer_ttt.py::launch_gradient_futures` routes that
signal's grading calls to a separate `grader_client_alt` (created at
startup in `train_cr_v7.py::main()`).

### strip_critiques (B4_stripped baseline)

`train_cr_v7.py::Config` gained `strip_critiques: bool`. When True,
`build_whole_plan_revision_prompt` replaces the per-signal feedback
block with a single aggregate-score line. Used for the B4_stripped
ablation to isolate critique-conditioning contribution.

### Aggregation (unchanged)

```python
aggregate = Σ weight_i × normalize(score_i)
normalize(score) = (score - 1) / 4   # maps 1→0.0, 5→1.0
```
