# Direction 5: Abstract-Retrieve-Refine

**Status**: **Phase 2 COMPLETE (2026-04-27)** — μ-v4 iter-4 = 28.00/45 (+2.75 over σ).
First 30B-trained variant to beat σ baseline. Production checkpoint locked.
**Read first**: [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md), then [`STATUS.md`](STATUS.md), then [`paper_materials/`](paper_materials/) for paper-writing assets.
**Target**: ICLR 2027 main track
**Code**: `src/co_scientist/d5_abstract_retrieve_refine/`
**Recipe**: [`knowledge/current/SDPO_RECIPE_v1.md`](knowledge/current/SDPO_RECIPE_v1.md) (locked, reusable for Phase 3) — documents the **D5 in-house off-policy IS-loss variant** used by μ-v2/v3/v4. For paper-grade ArXiv attribution see [`knowledge/current/CANONICAL_NAMING_REFERENCE.md`](knowledge/current/CANONICAL_NAMING_REFERENCE.md); for per-run setup truth see [`knowledge/current/RUN_REGISTRY.md`](knowledge/current/RUN_REGISTRY.md); reference paper PDFs at [`knowledge/papers/`](knowledge/papers/).

## Overview

Train a language model to produce long-form research plans through a multi-round pipeline that mimics how a researcher works: select which references to read, read them, abstract their insights, iterate with cross-round memory, and finally compose a plan.

Training uses self-distillation with privileged observations — the student learns to internalize reviewer feedback so that its un-reviewed behavior approaches its reviewed-conditioned behavior. Privileged information for the reviewer is the source paper itself (factual ground truth).

> **Algorithm naming caveat**: D5's μ-v2/v3/v4/v6/κ-v1 trainers (historically self-labeled "SDPO") implement a **D5 in-house off-policy IS-loss variant** that is structurally distinct from canonical Hübotter SDPO (arxiv 2601.20802 — student on-policy + KL loss) and from canonical Zhao OPSD (arxiv 2601.18734 — student on-policy + JS divergence or sampled-token reverse-KL). The closest D5 implementation to a canonical paper is `train_mu_v7_opd.py` (opd_mode=True), which corresponds to OPSD's sampled-token policy-gradient variant (Zhao Table 3). See [`knowledge/current/CANONICAL_NAMING_REFERENCE.md`](knowledge/current/CANONICAL_NAMING_REFERENCE.md) for the truthful per-trainer table.

## Research hypothesis

The key difference between an expert and a novice scientist's plan is not surface writing quality but the capacity for **cross-instance abstraction** — extracting transferable patterns from prior work and applying them to a new problem. Current LLM+RL training regimes either (a) do not iterate (single-shot) or (b) iterate raw text without abstraction (FLARE / IRCoT). Multi-round abstraction iteration with self-distillation internalization should produce a model that can generate a plan as if it had privileged access to reference material.

## Relationship to prior directions

- **D3 TTT-Discover**: reuses the 12-goal dataset and Opus depth audit infrastructure. D5 is a new training method on top of D3's evaluation substrate.
- **D4 Grant Proposal**: tabled. User view: "research plan 写好之后 grant proposal 是水到渠成". Grant proposal as downstream application is future work.
- **D1/D2**: not reused.

## Pipeline (per instance)

```
Inputs: research goal G, source paper S with bibliography B = [papers b_1..b_N] with titles T = [t_1..t_N]

ROUND i (i = 1, 2, 3):
  Step A — paper selection (learned):
    Model sees (G, T, selection_history, prior_abstractions) → outputs selected subset S_i ⊂ T
  Step B — abstraction (learned):
    Retrieve full content of S_i. Model sees (G, content, selection_history, prior_abstractions) → outputs abstraction A_i
  Step C — review (not learned, provides privileged info for SDPO):
    Reviewer sees (trajectory so far, source paper S full content). Writes feedback review_i.

Distillation loss (per round, over selection + abstraction tokens):
  Canonical form: L = KL(p_student(·|no review) || p_student(·|with review_i))
  D5 in-house implementation: TEACHER samples under (·|with review_i); STUDENT lp recomputed under (·|no review); A_t = clamp((t_lp − s_lp)·scale, ±5); importance_sampling loss (off-policy IS-loss variant, distinct from canonical SDPO/OPSD; see CANONICAL_NAMING_REFERENCE.md)

FINAL PLAN:
  Model generates plan P from (G, all content, [A_1, A_2, A_3]).
  Supervised CE loss against source paper S's own research plan.
```

## Novel contributions (paper claims)

1. **Domain-grounded self-distillation**: reference paper content as privileged info (factual, not evaluative). D5 in-house off-policy IS-loss variant; closest published comparable is OPSD sampled-token policy-gradient (Zhao 2601.18734 Table 3) but with privileged-info source = paper not `y*` ground-truth CoT.
2. **Multi-round retrieval + learned paper selection + abstraction**: distinct from FLARE/IRCoT (raw text iteration) and RLAD (single round, no retrieval)
3. **Reference paper's plan as gold supervised target** for final plan (eliminates noisy reward)
4. **Cross-goal transfer on long-form writing**: 8 train, 4 held-out

## Key Differences from Related Work

| Work | What it has | What it lacks |
|---|---|---|
| Hübotter SDPO (arxiv 2601.20802) | self-distillation mechanism (canonical: student on-policy + KL loss + top-K vocab approx) | no retrieval, no long-form, privileged info is environmental feedback |
| Zhao OPSD (arxiv 2601.18734) | on-policy self-distillation with privileged answer y* (canonical: student on-policy + JS or sampled-token reverse-KL) | math/CoT focus, single round, privileged info is ground-truth answer not source paper |
| RLAD (2510.02263) | abstraction + RL, two-stage | math-only, single round, no retrieval |
| FLARE (2305.06983) | multi-round retrieve + generate | iterates sentence not abstraction, no self-distillation |
| IRCoT | iterative retrieve-then-reason | QA, no abstraction, no self-distillation |
| ExpeL (AAAI 2024) | insight extraction + retrieval | inference-time only, trajectory pool not external papers |
| DR Tulu (2511.19399) | retrieval + RL + long-form | single-round retrieve-then-generate, evolving rubrics not self-distillation |

## Code

| File | Purpose |
|---|---|
| `__init__.py` | Module marker |
| `extract_cite_keys_v1.py` | Phase 2A-bis-1: regex-extract `\cite{}` from main.tex (→ 90 cite keys) |
| `parse_bibtex_v1.py` | Phase 2A-bis-2: parse main.bib, cross-ref cite keys (→ 90/90 entries) |
| `build_bibliography_v2.py` | Phase 2A-bis-3..7: S2 batch + OpenAlex with `_verify_match` |
| `extract_source_paper_v2.py` | Phase 2A-bis-6: LaTeX → markdown, replaces corrupted PDF v1 |
| `expand_full_text_v1.py`, `extract_arxiv_latex_v1.py` | Phase 2A-bis-8/9: full-text expansion |
| `oracle_prompts_v2.py`, `build_oracle_v2.py`, `build_slim_oracle_v2.py` | Phase 2B: oracle extraction prompts + driver (4-round Opus over 62 papers) |
| `train_xi_v2.py`, `train_sigma_v2.py`, `train_delta_v2.py`, `train_epsilon_v2.py` | Phase 2D realigned baselines (frozen variants) |
| `train_alpha_v2.py`, `train_beta_v2.py` | Phase 2D realigned trained baselines (Opus distill / ref SFT) |
| `mu_prompts_v2.py` | Realigned μ prompts (oracle-aware, no reference-leak) |
| `train_mu_v2.py` | Phase 2E: plan-level in-house off-policy IS-loss variant + critic + oracle (lr=1e-5, 10 iter, 1 grad step). Historically labeled "SDPO"; see RUN_REGISTRY.md. |
| `train_mu_v3.py` | Phase 2 ablation: in-house off-policy IS-loss @ lr=2e-4, 4 grad steps × 20 iter — collapsed |
| `train_mu_v4.py` | Phase 2 ablation: in-house off-policy IS-loss @ lr=5e-5 + every-iter checkpoint + audit-drop early stop. Production checkpoint at iter 4. |
| `audit_v3_isolated.py` | Phase 2F: balanced-batch anonymized 8 × 8 parallel Opus subagent audit |
| `audit_v3_shuffled.py`, `audit_v3_runner.py` | audit_v3 v2 attempt (deprecated by isolated) |
| `pairwise_prefs_v1.py`, `pairwise_v2.py`, `fair_pairwise_v1.py` | pre-realignment pairwise tournament (now secondary) |

Shared modules (to be created in `src/co_scientist/shared/`):
| File | Purpose |
|---|---|
| `per_goal_database.py` | FAISS + paper content loader |
| `paper_selection.py` | Selection action head + parsing |
| `abstraction_extractor.py` | Abstraction generator |
| `reviewer_subagent_client.py` | Opus reviewer via file-bus |
| `sdpo_distillation.py` | Self-distillation loss (currently: D5 in-house off-policy IS-loss; canonical SDPO/OPSD migration optional per user decision) |
| `final_plan_supervised.py` | CE loss against reference plan |

## Dataset

**Single research goal**: TTT-Discover (arxiv 2601.16175, Yuksekgonul et al. Feb 2026).

D5-owned `dataset/` directory (copied from D3 `analysis/sanity_check/` on 2026-04-25, no longer symlinked to D4):

- `dataset/research_goal.txt` — TTT-Discover research goal (~830 chars, Goel-style)
- `dataset/reference_solution.txt` — TTT-Discover gold reference plan (~1200 words, 6 sections: Problem / Background / Hypothesis / Methodology / Evaluation / Limitations)
- `dataset/perturbations/` — 18 perturbation variants of the reference plan (for Phase 4 robustness testing, optional)
- `dataset/ttt_discover_bibliography/` — TTT-Discover paper's references (Phase 0b will populate via S2 API)
- `data/bibliography/resolved_v2.jsonl` — clean Phase 2A-bis bibliography: 86/90 cite keys resolved (38 with full text), built 2026-04-26 from LaTeX source (`main.tex` + `main.bib`)
- `data/source_paper/v2.md` — clean LaTeX-extracted source paper (~76,300 chars), replaces corrupted PDF v1
- `data/bibliography/full_text/*.md` — 28 arxiv references converted from LaTeX source to markdown
- `data/oracle_v2.md` / `oracle_v2_slim.md` — Phase 2B 4-round Opus extraction over 62 papers, 580 typed items

Single-goal commitment per 2026-04-24 decision: D5 paper is single-task mechanism study, not cross-goal generalization. Future work: extend to additional D3-style goals if mechanism validates.

## Dependencies

- Tinker + Qwen3-30B-A3B (policy)
- Opus 4.5 via OpenRouter (reviewer Phase 2, 10% spot-check Phase 3+)
- FAISS vectorstore
- arXiv + Semantic Scholar APIs
- Existing `src/co_scientist/shared/paper_retrieval.py`, `eval_core.py`

## Read-First Documents (for future sessions in this direction)

1. `DIRECTIONS.md` — project-wide map
2. `projects/d5_abstract_retrieve_refine/STATUS.md` — live state
3. **`knowledge/current/PHASE_PLAN_v2.md`** — strategic chronological roadmap (Phase 0 → 5, where we are, decision gates)
4. **`SCHEMA.md`** — file placement rules (where does this file go?)
5. Last 3 entries of `DECISIONS.md`
6. `CONVENTIONS.md` before creating files
7. `~/.claude/plans/giggly-jumping-hollerith.md` — original 2026-04-23/24 master plan (frozen reference)

## Owner

Yuhong Shi
