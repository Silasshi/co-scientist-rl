# D6 Grant Proposal ERR — Status

**Last updated**: 2026-04-30  
**Phase**: 0 → 1 transition (oracle build + pilot pending)

## Current State

All three paper arms (baseline, GRPO, OPD) plus the deferred OPD-grounded / OPD-KL-anchor scaffolds are wired up. Dataset in domain-organized structure (3 goals).
Prompt system centralized: every D6 production prompt now loads from `.md`
files under `projects/d6_grant_proposal/prompts/` via `prompt_loader.py`.
Generation prompt re-aligned to the empirical reference distribution
(target 1500 / max 2000 words, 8-heading reference template); review
prompt extended to an 8-child `<critique>` schema; new pairwise prompt
in place (caller deferred).
Oracle, gold files, and pilot training not yet run.

## What exists

| Artifact | Status |
|---|---|
| Dataset — `ai/02_foundational_rl` (pilot) | ✅ Migrated from D4 |
| Dataset — `ai/01_foundopt` | ✅ Mirrored 2026-04-30; retained as signal-validation goal source |
| Dataset — `natural_science/08_ecosystem_dynamics` | ✅ Migrated from D4 |
| Dataset — `social_science/12_climate_displacement` | ✅ Migrated from D4 |
| `signals.py` | ✅ Re-exports D4's 12-signal rubric |
| `prompts.py` | ✅ Loader-backed (templates in `prompts/`) |
| `prompt_loader.py` | ✅ load_prompt / render_prompt / reload_prompts |
| `grounding.py` | ✅ Equation/citation matching (regex+Jaccard, no sympy) |
| `extract_oracle.py` | ✅ Loader-backed |
| `extract_oracle_slim.py` | ✅ Written (domain-organized paths) |
| `extract_gold.py` | ✅ Written (equations+citations from reference_proposal.md) |
| `audit.py` | ✅ Loader-backed |
| `train_baseline.py` | ✅ Frozen baseline (no training) |
| `train_opd.py` | ✅ OPD (simplified 2026-04-30): sol-mask + privileged-Opus offset + PPO-clip |
| `train_grpo.py` | ✅ GRPO ablation: no reviewer, reward-only |
| `train_opd_grounded.py` | ✅ OPD-grounded (out of paper scope): OPD + grounding bonus |
| `train_opd_kl_anchor.py` | ✅ OPD-KL-anchor scaffold (blocked on oracle SFT) |
| `prompts/` (production source of truth) | ✅ generation/ review/ audit/ pairwise/ oracle/ legacy/ |
| `prompts/pairwise/head_to_head.md` | ✅ 12-dim verdict; caller deferred |
| `tests/test_d6_prompt_loader.py` | ✅ 24 tests — loader mechanics + content invariants |
| Oracle for any goal | ❌ Not built yet |
| Gold file for any goal | ❌ Not built yet |
| Pilot training run | ❌ Not run yet |
| Pairwise caller (`pairwise.py`) | ❌ Deferred — needs run data first |

## Phase 1 Sequence (ordered)

```
1. Build oracle + slim oracle (ai/02_foundational_rl)
   PYTHONPATH=src python -m co_scientist.d6_grant_proposal.extract_oracle \
       goal_domain=ai goal_name=02_foundational_rl
   PYTHONPATH=src python -m co_scientist.d6_grant_proposal.extract_oracle_slim \
       goal_domain=ai goal_name=02_foundational_rl

2. Build gold file (for OPD-grounded)
   PYTHONPATH=src python -m co_scientist.d6_grant_proposal.extract_gold \
       goal_domain=ai goal_name=02_foundational_rl

3. Run frozen baseline (frozen model)
   PYTHONPATH=src python -m co_scientist.d6_grant_proposal.train_baseline \
       goal_domain=ai goal_name=02_foundational_rl

4. Run OPD pilot [concurrent with GRPO]
   PYTHONPATH=src python -m co_scientist.d6_grant_proposal.train_opd \
       goal_domain=ai goal_name=02_foundational_rl \
       log_path=projects/d6_grant_proposal/runs/2026_04_30_pilot_foundrl_opd

5. Run GRPO ablation [concurrent with OPD]
   PYTHONPATH=src python -m co_scientist.d6_grant_proposal.train_grpo \
       goal_domain=ai goal_name=02_foundational_rl \
       log_path=projects/d6_grant_proposal/runs/2026_04_30_pilot_foundrl_grpo

6. Decision: audit OPD vs GRPO vs baseline
   → OPD ≫ GRPO: reviewer signal is load-bearing (expected)
   → OPD ≫ baseline+0.02: paper claim corroborated; OPD-grounded remains optional/out of scope

7. [Out of scope; only if needed] Run OPD-grounded pilot
   PYTHONPATH=src python -m co_scientist.d6_grant_proposal.train_opd_grounded \
       goal_domain=ai goal_name=02_foundational_rl \
       log_path=projects/d6_grant_proposal/runs/2026_04_30_pilot_foundrl_opd_grounded
```

## Selected Goals

| Domain | Goal | Role | Key Features |
|---|---|---|---|
| AI/CS | `02_foundational_rl` | **Pilot** (2026-04-30 revised) | RL theory, convergence bounds, G12 formalism |
| AI/CS | `01_foundopt` | Signal-validation source only | FoundOpt grant proposals; Opus-scored panel data lives here |
| Natural Science | `08_ecosystem_dynamics` | Cross-goal eval (Phase 5) | Bifurcation math, CSD, ecology theory |
| Social Science | `12_climate_displacement` | Cross-goal eval (Phase 5) | Intl law, governance, G12a (not G12) |

## Key Decisions

- Trainer family: OPD (simplified) (sol-mask + privileged-Opus offset + PPO-clip) — trust-region α removed 2026-04-30 for ICLR-workshop simplicity. OPD-grounded (+ grounding), OPD-KL-anchor (+ KL anchor) retained on disk but out of paper scope.
- GRPO: reward-only ablation (no Opus reviewer), runs concurrent with OPD
- OPD-KL-anchor blocked on oracle SFT (separate training run, Phase 2)
- Oracle source: `reference_proposal.md` (not LaTeX arxiv paper)
- Score scale: D4 12-signal weighted mean (NOT D5's /45)

## Deferred

- OPD-KL-anchor (oracle SFT required)
- RAG retrieval stage (grant-domain corpus not defined)
- Pairwise eval script (needs run data first)
- Multi-goal expansion (blocked on ai/02_foundational_rl pilot)
