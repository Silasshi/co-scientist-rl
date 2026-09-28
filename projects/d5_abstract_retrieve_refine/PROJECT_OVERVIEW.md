# D5 — Abstract-Retrieve-Refine: Project Overview

*Top-level summary. Read this FIRST. Last updated: 2026-04-27.*

## TL;DR

D5 trains a 30B model to write high-quality long-form research plans by abstracting
methodological insights from a paper's bibliography and using them at generation time.
**Phase 2 has produced a clean positive result**: μ-v4 plan-level **D5 in-house
off-policy IS-loss variant** (TEACHER samples; STUDENT lp recomputed under
non-critique input; importance_sampling loss — historically labeled "SDPO" but
neither canonical Hübotter SDPO 2601.20802 nor canonical OPSD 2601.18734; see
`knowledge/current/CANONICAL_NAMING_REFERENCE.md` and `knowledge/current/RUN_REGISTRY.md`)
+ Opus critic trained for 5 iters lifts Qwen3-30B-A3B from σ baseline 25.25 →
**28.00 / 45 (+2.75)** on the 9-dim isolated audit, with the iter-4 checkpoint
as production. The 235B upper-bound (ε frozen + reference plan in context) is
33.62. Multi-round instability of this in-house variant prevents sustained gains
past iter 5.

**Cross-validated (2026-04-27)**: Opus 4.7 anonymized pairwise tournament corroborates
the close-cluster ranking — μ-v4 wins **20/24 (83.3%)** across {σ, δ, α}: 6-2 vs σ
(PRIMARY clears the ≥6/8 PASS threshold), 7-1 vs δ, 7-1 vs α. Position-bias check
passes. The +2.75 absolute lift is not an audit-surface-form artifact.
See `paper_materials/findings/F2` § "Cross-validation".

**Cross-goal Phase 4a (FINALIZED 2026-04-28 via D-series n=24 stress test)**: μ-v4 LoRA
tested on 3 forward-citation TTT-Discover follow-up papers. Original 2-plan/subagent
audit suggested "Strong-mixed (2/3 pass)" verdict. After **3 progressive corroboration
stages** the final verdict is **DEFINITIVELY NULL transfer**:

| Stage | meta_ttl | tool_v_ttrl |
|---|---|---|
| Original 2-plan audit (n=8, 2026-04-26) | μ' wins +2.13 | μ' wins big +3.00 |
| Strict 1-plan + pairwise (n=8, 2026-04-27) | σ' edge / TIE | μ' weak (within noise) |
| Strict + pairwise (n=24, 2026-04-28) | **σ' DIRECTIONAL CONFIRMED** (σ' 14/9/1, Δ -0.96) | **TRUE NULL** (12/10/2, Δ -0.58) |

The original 2-plan batching introduced ~2-3 point anchoring inflation; n=24 stress
test rules out "underpowered noise" alternative. **Forward-citation transfer to follow-up
papers in the same TTT-family subfield does NOT exist** at the n=24 confidence level.
See `paper_materials/findings/F7` (final) and `paper_materials/methodology/M8_audit_pairwise_divergence.md`.

**Continual chain NULL (2026-04-26 → CORROBORATED 2026-04-27, Phase 5)**: tested
"field-expert via continual chain" hypothesis (continual training using D5 in-house
off-policy IS-loss variant — see RUN_REGISTRY.md) across 4-cell grid (anchor
saturation × Adam state × anchor_ce regularization). **All 4 cells fail H1 forward
learning**.
v3 (no regularization) Δ +0.62 vs μ-v4 (within noise); α (anchor_ce 0.1) Δ -2.62.
Phase 5 **NULL** triple-corroborated 2026-04-27: pairwise (4-4 TIE) + strict 1-plan
(Δ +1.87 within noise) + in-loop /20 trajectory (4-cell decline) all agree. The α
anchor_ce verdict softens under strict (Δ +0.26 TIE) but the iter-2 cliff in /20
trajectory remains the load-bearing failure signal. Phase 6 future work: plan-level
replay buffer / EWC with Fisher importance / LoRA-per-paper merge. **Phase 4b
cross-domain stretch test formally DEPRECATED**. See `paper_materials/findings/F9`.

**NEW methodology contribution (M8, 2026-04-27)**: 2-plan/subagent batching introduces
systematic anchoring that inflates close-cluster Δ by ~2-3 points relative to either
strict 1-plan/subagent calibration OR holistic pairwise tournament. H1 anchoring
hypothesis CONFIRMED on 2/4 close-cluster goals. Standing protocol going forward:
require pairwise OR strict corroboration for any verdict at Δ < 5/45. See
`paper_materials/methodology/M8_audit_pairwise_divergence.md`.

```
Audit /45                        ε ceiling 33.62
   35 ┤                             ●
      │
      │
   28 ┤  ●  μ-v4 iter 4 = 28.00 (+2.75 over σ) ← production
      │
   25 ┤  ── σ baseline 25.25 ── (frozen + oracle, no training)
      │     δ 25.12 / α 25.00 (other 30B baselines, ≈ σ)
      │     μ-v2 23.88 / β 22.75
      │
   16 ┤  ξ 15.62 (frozen + goal only, no oracle)
```

## Phase-by-phase summary

### Phase 1 (realignment, 2026-04-25): reviewer-grounded design

User identified that all pre-realignment runs were misaligned (oracle = single-round
extraction; δ/ε mistreated as baselines; audit had surface-form bias). Archived all
prior runs. Designed:

- **Oracle structure** (`ORACLE_DESIGN_v1.md`): 6 categories (Insights/Methodology/
  Theory/Math/Empirical/Failure-modes), Opus-generated as ceiling estimate, multi-round
  cumulative, real bibliography citations
- **Audit rubric** (`AUDIT_RUBRIC_v3.md`): 9-dim hybrid (5 universal + 4 TTT-Discover-
  specific), anchored to real OpenReview reviewer quotes (MTTT/Voyager/SCoRe/etc.)
- **Reviewer standards synthesis** (`REVIEWER_STANDARDS_v1.md`): 11 sources covering
  NeurIPS/ICLR/ICML reviewer forms + 7 OpenReview adjacent papers
- **Run-confirmation protocol** (`RUN_CONFIRMATION_TEMPLATE.md`): pre-run user
  approval gate

### Phase 2A-bis (2026-04-26): bibliography rebuild from LaTeX

PDF parsing originally dropped 17/87 references silently. Switched to LaTeX-source
pipeline (parse main.tex's `\cite{}` keys + main.bib database). Result: 86 papers
resolved (38 with full text), 4 marked non-paper, 4 explicitly failed. Files:

- `data/bibliography/resolved_v2.jsonl` (86 papers)
- `data/source_paper/v2.md` (LaTeX-extracted clean source paper)
- `data/bibliography/full_text/{arxiv_id}.md` (38 full texts via arxiv PDF + pypdf)

### Phase 2B (oracle build): 4-round Opus extraction

Multi-round Opus extraction over 62 papers from the bibliography (filtered relevance
≥1 in Round 0). 580 typed items in `oracle_v2.md`. Slim variant (`oracle_v2_slim.md`)
fits in 4096-token policy budget. Commit `67ce1b9`.

### Phase 2D (realigned trainers): v2 versions

Forked v1 trainers to v2 with new oracle path: `train_mu_v2.py`, `train_alpha_v2.py`,
`train_beta_v2.py`, `train_sigma_v2.py`, `train_xi_v2.py`, etc. All consume
`oracle_v2_slim.md`. Commit `3f1e848`.

### Phase 2E (7-baseline battery): trained at lr=1e-5

7 baselines all at max_tokens=4096: ξ → σ → δ → ε → α → β → μ-v2. Commit `b10d71a`.

### Phase 2F (audit_v3 isolated): 9-dim per-plan ranking

Methodology innovation: 8 balanced batches × 8 parallel Opus subagents, per-plan
isolation prevents within-batch anchoring. Establishes σ=25.25 as the production
30B reference baseline (`runs/2026_04_26_phase2F_audit_isolated/`). Code:
`src/co_scientist/d5_abstract_retrieve_refine/audit_v3_isolated.py`.

| Baseline | Mean /45 | Setup |
|---|---:|---|
| ε | 33.62 | frozen Qwen3-235B + reference plan in context |
| **σ** | **25.25** | **frozen Qwen3-30B + slim oracle (no training)** |
| δ | 25.12 | frozen Qwen3-30B + reference plan in context |
| α | 25.00 | Opus distillation 5 epochs × 16 plans |
| μ-v2 | 23.88 | plan-level in-house off-policy IS-loss + critic, lr=1e-5, 10 iters |
| β | 22.75 | direct SFT on reference plan |
| ξ | 15.62 | frozen Qwen3-30B + goal only |

### Phase 2 lr ablation (2026-04-27): the key result

(All three runs use the D5 in-house off-policy IS-loss variant — TEACHER samples;
STUDENT lp recomputed; importance_sampling loss. Historically self-labeled "SDPO"
but distinct from canonical Hübotter / OPSD. See `RUN_REGISTRY.md`.)

Three lr regimes tested:
- **μ-v2 lr=1e-5**: flat 23.88 (no learning, no collapse)
- **μ-v3 lr=2e-4**: peak iter 0 = 24.88, collapse iter 6 = 14.38 (catastrophic)
- **μ-v4 lr=5e-5** (geometric median): peak **iter 4 = 28.00** (+2.75 over σ),
  cliff iter 5 = 15.62 (-12.4 in single iter), collapse iter 6+ < 10

iter-4 weights are the **production μ-v4 checkpoint**. See
`paper_materials/findings/F2_mu_v4_iter4_peak.md` for full details.

## What we found / confirmed

1. **σ baseline (frozen + oracle inference) is the dominant lift** (+9.63 over ξ).
   Inference-time scaffolding does most of the work. (F1)

2. **Plan-level in-house off-policy IS-loss variant + Opus critic CAN add +2.75 over σ**
   at the right lr (5e-5), for early iters (peak iter 4). First 30B-trained variant to
   beat σ. (F2) [historically labeled "SDPO"; see CANONICAL_NAMING_REFERENCE.md]

3. **Multi-round instability of the in-house variant is empirical**. Three lr regimes
   tested; all eventually destabilize if run too long. (F3) [Note: not directly
   comparable to canonical SDPO multi-round behavior since we do not faithfully
   implement Hübotter SDPO — we have not tested whether canonical SDPO/OPSD has the
   same property.]

4. **PUCT formula transfers to TEACHER (PICK) distribution but not to STUDENT (EVAL)
   distribution** at any iter. The in-house variant's advantage signal measures
   stylistic divergence, not content tokens — the diagnosed root cause of multi-round
   instability in this variant. (F4)

5. **Per-plan isolated audit + balanced batches + anonymization is the audit
   methodology of record** for D5. Batched 4-dim audit hides per-iter differences via
   anchor saturation. (F5, F6)

## Outstanding questions for Phase 3 (retrieve-then-generate)

1. **Does dynamic retrieval (per-section) replace static slim_oracle?** Or does it
   augment it? Slim oracle currently gives the model a curated structured input;
   retrieval would build that input dynamically per-section.

2. **Can the SDPO_RECIPE_v1 (D5 in-house off-policy IS-loss variant) be reused for
   retrieval-augmented generation?** Or does the critique format need to change
   (e.g. critique includes retrieval-quality feedback)?

3. **Cross-goal generalization**: does iter-4 μ-v4 weight transfer to other research
   goals? (Candidate post-2026 follow-ups: TTRL, MiGrATe, ThetaEvolve, 1-shot
   RLVR, TTC-RL, LatentSeek — all share TTT-Discover's setup.)

## Production artifacts

- **μ-v4 iter-4 checkpoint**: `runs/2026_04_27_mu_v4/checkpoints.jsonl` row for
  batch=4 (Tinker state path; `kind: "both"` includes sampler weights)
- **σ baseline plans (control)**: `runs/2026_04_26_sigma_v2/eval_rollouts.jsonl`
- **Slim oracle**: `data/oracles/oracle_v2_2026_04_26_build/slim.md`
- **Source paper (clean LaTeX-extracted)**: `data/source_paper/v2.md`
- **9-dim audit rubric**: `knowledge/current/AUDIT_RUBRIC_v3.md`
- **Recipe (locked, D5 in-house off-policy IS-loss variant)**: `knowledge/current/SDPO_RECIPE_v1.md`
- **Naming / paper attribution truth**: `knowledge/current/CANONICAL_NAMING_REFERENCE.md`
- **Per-run setup truth table**: `knowledge/current/RUN_REGISTRY.md`
- **Reference papers (PDFs)**: `knowledge/papers/` (TTT-Discover 2601.16175 + Hübotter SDPO 2601.20802 + Zhao OPSD 2601.18734)
- **Paper materials folder**: `paper_materials/` (start here when writing the paper)

## What NOT to do

- ❌ Don't run μ training past iter 5 — multi-round instability is empirical
- ❌ Don't trust 4-dim daemon audit alone — saturates at anchor; always run 9-dim
  isolated for paper-grade conclusions
- ❌ Don't switch base model from Qwen3-30B-A3B without re-running σ baseline first
- ❌ Don't change oracle structure without re-auditing on the new oracle —
  small content changes can shift σ baseline by ±2 points
- ❌ Don't run two daemons concurrently — race conditions on file writes
- ❌ Don't cite μ-v3 (collapsed) numbers as authoritative in the paper

## Next phase (Phase 3) — separate plan to come

Phase 3: retrieve-then-generate. Build dynamic per-section retrieval over the 86-paper
bibliography; replace static slim_oracle with retrieved chunks; reuse SDPO_RECIPE_v1
(D5 in-house off-policy IS-loss variant) for fine-tuning, OR optionally migrate to
canonical SDPO (Hübotter 2601.20802) / OPSD (Zhao 2601.18734) — algorithmic decision
remains with the user. Plan doc TBD after this Phase 2 documentation pass.

## Document map

```
projects/d5_abstract_retrieve_refine/
├── PROJECT_OVERVIEW.md           ← YOU ARE HERE
├── STATUS.md                     ← live state pointer
├── DECISIONS.md                  ← append-only log
├── README.md                     ← repo-style entry point
├── CONVENTIONS.md                ← naming + folder rules
├── SCHEMA.md                     ← data formats
├── knowledge/
│   └── current/
│       ├── SDPO_RECIPE_v1.md     ← reusable training recipe
│       ├── AUDIT_RUBRIC_v3.md    ← 9-dim audit rubric (Option B anchors)
│       ├── ORACLE_DESIGN_v1.md   ← oracle structure + 6 categories
│       ├── REVIEWER_STANDARDS_v1.md  ← reviewer-form synthesis
│       ├── RUN_CONFIRMATION_TEMPLATE.md
│       └── PHASE_PLAN_v2.md      ← phase plan (Phase 2 marked COMPLETE)
├── paper_materials/              ← paper-writing canonical store
│   ├── README.md
│   ├── findings/                 ← 6 finding docs (F1-F6)
│   ├── experiments/              ← raw data dumps (E1-E5)
│   ├── figures_data/             ← matplotlib-ready CSVs (G1-G4)
│   ├── plan_samples/             ← 5 exemplar plans for paper
│   ├── methodology/              ← M1-M5 self-contained methodology docs
│   ├── related_work_excerpts/
│   └── next_steps/               ← N1-N3 Phase 3 open questions
├── data/                         ← bibliography + oracle + source paper
├── dataset/                      ← research_goal.txt, reference_solution.txt
├── runs/                         ← experiment run dirs
└── scripts/                      ← daemon specs, helper scripts
```
