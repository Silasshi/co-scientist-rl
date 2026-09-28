# Run Confirmation — μ-v8 d5sdpo (critique-on-current redesign + 4 design imports + 900-word footer)

**Date filed**: 2026-04-28 (revised from earlier draft after user identified critique-lag bug)
**Status**: AWAITING USER "yes go" approval per RUN_CONFIRMATION_TEMPLATE
**Plan**: `/home/silas/.claude/plans/snug-dreaming-yao.md` (μ-v8 d5sdpo)
**Trainer**: `src/co_scientist/d5_abstract_retrieve_refine/train_mu_v8_d5sdpo.py` (NEW, ported)

> **Algorithm framing (truthful)**: μ-v8 d5sdpo is a **D5 in-house SDPO-flavor
> variant**: on-policy IS-loss / PPO-clip with critique-on-current-rollout.
> Closest published comparable: **OPSD (Zhao 2601.18734) Table 3 sampled-token
> policy-gradient** + **Hübotter SDPO (2601.20802) Appendix A.2 trust-region
> anchor**. NOT a faithful reimplementation of either canonical algorithm
> (Tinker exposes scalar logprob only, not full-vocab/top-K KL or full-vocab JS).
> Disambiguation: `CANONICAL_NAMING_REFERENCE.md`, `RUN_REGISTRY.md`.

---

## Run identity

- **Run-dir naming**: `2026_04_29_mu_v8_d5sdpo_<cell>` where `<cell>` ∈ {sigma_v8, base, A, B, C, D, E, F, G}
- **Greek-letter family**: μ-v8 (training) + σ_v8 (re-baseline)
- **Trainer file**: `src/co_scientist/d5_abstract_retrieve_refine/train_mu_v8_d5sdpo.py` (fork of `train_mu_v7_opd.py` + 4 changes)
- **Prompt module**: `src/co_scientist/d5_abstract_retrieve_refine/mu_prompts_v8.py` (NEW; 900/1100-word footer)
- **Estimated wall-clock**: ~5 working days
- **Estimated cost**: ~$615 (full grid)
- **Comparator anchors**:
  - **σ_v8** (NEW; this run produces it) — anchor for v8 grid (length-matched)
  - μ-v7-opd-full iter 4 (RUN_REGISTRY:38 strict /45 = 26.62; 750-word footer)
  - μ-v4 production (RUN_REGISTRY:31 nominal 28.00 M8-confounded; 750-word footer)
  - σ legacy = 25.25 (RUN_REGISTRY:155; 750-word footer; **NOT length-matched to v8**)

## Pipeline diagram (v8, critique-on-current)

```
[research_goal.txt] ─┐
                     ├─→ student_input = goal + slim_oracle (NO critique, 900-word footer)
[slim_oracle (v2)] ──┘
                     │
                     ▼
            STUDENT samples 8 plans on-policy
                     │
                     ▼
            ┌──────────────────────────────────┐
            │ PICK 1 plan (rng over decoded)  │  ← critique here is on CURRENT rollout
            │     ↓                            │
            │ critic.submit + BLOCKING wait   │  ← +60-180s per iter
            │     ↓                            │
            │ critique_current = parsed XML   │
            └────────┬─────────────────────────┘
                     │
                     ▼
[prev_critique]   teacher_input = goal + slim_oracle + critique_current
   (DROPPED)              │
                          ▼
            teacher_lp = sampling_client.compute_logprobs(teacher_input + student_seq.tokens)
                          │
                          ▼
   [if α>0]   frozen_lp = initial_teacher_client.compute_logprobs(teacher_input + student_seq.tokens)
                          │
                          ▼
            teacher_lp_eff = (1-α) · teacher_lp + α · frozen_lp        ← Change 3 trust-region
                          │
                          ▼
            A_t = clamp((teacher_lp_eff − student_lp) · scale, ±5)
                          │
                          ▼
   [if mask]  A_t *= build_solution_token_mask(student_seq.tokens)     ← Change 2 solution mask
                          │
                          ▼
            datum = (student_input + student_seq.tokens, A_t, student_lp)
                          │
                          ▼
            forward_backward(loss_fn ∈ {ppo (clip 0.8/1.2), importance_sampling})  ← Change 4
                          │
                          ▼
            EVAL audit (async)
```

Key contrast vs v7-opd: critic is BEFORE teacher_lp compute (Change 1). v7-opd has it AFTER training using prev_critique. The `prev_critique` variable is dropped from v8.

## Verbatim prompts (v8 footer; same body as v1)

```
build_student_prompt_v8(goal, oracle):

I will provide you a research scenario and a set of methodological patterns
extracted from a high-quality research plan on this same scenario. Use the
patterns to guide the structure and reasoning of your plan, but do not copy
the phrasing literally.

Scenario: {goal}

# Methodological patterns to apply

{oracle_abstraction}

Write your research plan inside <solution>...</solution> tags. Target 900
words, max 1100 words. Use structured sections (Problem Statement, Background,
Hypothesis, Methodology, Evaluation, Limitations).

build_teacher_prompt_v8(goal, oracle, critique_xml):  same body + critique block
```

Footer change verified: `dataset/reference_solution.txt` = 962 words; σ baseline plans 8/8 stayed within 750-cap (mean 618), so policy was footer-bound. v8 footer (900/1100) matches reference and lets policy generate at reference scale.

## What model sees / doesn't (per iter)

| Variable | Student input | Teacher input | Critic input | Source |
|---|:---:|:---:|:---:|---|
| goal | YES | YES | YES | `dataset/research_goal.txt` |
| slim oracle | YES | YES | NO | `data/oracles/oracle_v2_2026_04_26_build/slim.md` |
| **critique on CURRENT rollout** | NO | **YES (NEW in v8)** | (output of critic) | from blocking critic call this iter |
| reference plan | NO | NO | NO | (held; not used in any loss) |
| source paper full | NO | NO | YES (privileged) | `data/source_paper/v2.md` |

## 12-cell ablation grid (axes: mask × α[3-level] × loss_fn; critique-on-current ON for ALL)

All 12 runs share: lr=5e-5, LoRA r=64, n_plans=8, n_grad_steps=4, n_iter=16, max_tokens=4096, temp=1.0, top_p=0.95, seed=42, audit_drop_threshold=5.0, opd_mode=on-policy student.

α is now a **3-level factor** ({0, 0.05, 0.1}) per Decision 3 (pre-registered upfront, not conditional on G's signal — avoids garden-of-forking-paths). α=0.01 (rubric_reward/sdpo's value) excluded — too small at our once-per-iter cadence.

| Cell | mask | α | loss_fn | Hypothesis tested |
|---|:---:|:---:|---|---|
| **base** (control) | False | 0 | importance_sampling | "v8 = v7-opd + critique-fix + word fix" — isolates critique-fix lift alone |
| **A** | True | 0 | importance_sampling | + mask only |
| **B** | False | 0.05 | importance_sampling | + trust-region α=0.05 |
| **B+** | False | **0.1** | importance_sampling | + trust-region α=0.1 (stronger anchor) |
| **C** | True | 0.05 | importance_sampling | + mask + trust-region α=0.05 |
| **C+** | True | **0.1** | importance_sampling | + mask + trust-region α=0.1 |
| **D** | False | 0 | ppo (ε=0.2) | + PPO only |
| **E** | True | 0 | ppo (ε=0.2) | + mask + PPO |
| **F** | False | 0.05 | ppo (ε=0.2) | + trust-region α=0.05 + PPO |
| **F+** | False | **0.1** | ppo (ε=0.2) | + trust-region α=0.1 + PPO |
| **G** | True | 0.05 | ppo (ε=0.2) | full v8 (mask + α=0.05 + PPO) |
| **G+** | True | **0.1** | ppo (ε=0.2) | full v8 with stronger anchor |

Pre-registered cliff-iter hypothesis: monotonic in α — α=0 cliffs iter 7 (v7-opd reproduction); α=0.05 cliffs iter ≥9; α=0.1 cliffs iter ≥11. If α=0.1 collapses (anchor kills learning) → upper bound established.

External comparators (NOT in grid; cited from RUN_REGISTRY):
- **σ_v8** (this run produces it) — frozen Qwen3-30B + slim_oracle + v8 footer
- v7-opd-full iter 4 (lagged critique + 750-word footer + IS-loss + no imports)

## Sequencing — 4 sequential batches (NO parallel runs per server constraint)

Revised for 12-cell grid (α 3-level expansion).

**Batch 0 — verification (1 day, $20)**
- σ_v8 re-baseline (8 frozen plans + audit, ~30 min, $15)
- v8 G smoke 1-iter (verify infra, ~10 min, $5):
  - Check critic fires BEFORE teacher_lp (log order: "STUDENT sampled" → "critic round-trip" → "TEACHER logprobs")
  - Check `initial_teacher_client.create_sampling_client(base_model=...)` works with sync API
  - Check `frozen_lp != teacher_lp` (interp is non-trivial)
  - Check `solution_mask_density > 0.5` per plan
  - Check `forward_backward(loss_fn="ppo", loss_fn_config={...})` accepts clip config
  - Check metrics row writes with all v8 fields

**Batch 1 — corner anchors (1.5 days, ~$85)**
- v8 base (no imports)
- G (all imports, α=0.05)
- G+ (all imports, α=0.1)
- v7-opd-full vs σ_v8 pairwise control matchup (Decision 2; uses existing v7-opd-full plans, $10)

After Batch 1: pairwise base vs v7-opd-full + G vs v7-opd-full + G vs G+ + v7-opd-full vs σ_v8. **Pre-registered Batch 3 drop check** (Decision 1): if `base` wins ≥6/8 vs v7-opd-full AND `G` wins ≥6/8 vs `base` → critique-fix is the dominant lever, SKIP Batch 3 (saves $170).

**Batch 2 — single-import + α=0.1 IS-loss (1 day, ~$120)**
- A (mask only)
- B (trust α=0.05 only)
- B+ (trust α=0.1 only)
- D (PPO only)

**Batch 3 — interactions (1.5 days, ~$170) — droppable per Decision 1**
- C (mask + trust α=0.05)
- C+ (mask + trust α=0.1)
- E (mask + PPO)
- F (trust α=0.05 + PPO)
- F+ (trust α=0.1 + PPO)

**Day 5 — full audit + factorial decomp + RUN_REGISTRY update + finding doc draft (~$385)**

## Eval metric + comparison rule

**Primary**: pairwise tournament, anonymized 1-pair/Opus subagent (M8-compliant, n=8)

| Matchup | n | Threshold |
|---|---:|---|
| **G vs v7-opd-full** (PRIMARY paper headline) | 8 | ≥6/8 wins |
| **base vs v7-opd-full** (critique-fix isolation, gates Batch 3 drop) | 8 | ≥6/8 to claim critique-fix dominant |
| **G vs base** (imports-vs-critique-fix marginal, gates Batch 3 drop) | 8 | ≥6/8 to claim imports add value |
| **v7-opd-full vs σ_v8** (Decision 2 control; length-matched comparator) | 8 | directional; reframes "+1.38 over σ legacy" |
| each of {A, B, B+, C, C+, D, E, F, F+, G+} vs σ_v8 | 8 each | directional |
| G vs G+ (α main effect at top corner) | 8 | directional |

Total pairs: ~110 × Opus subagent ≈ $250 audit.

**Secondary**: strict 1-plan/subagent /45 audit on iter 4 + iter peak per cell. Use as triangulation per M8 rule.

**Cliff diagnostic**: per-iter audit trajectory. F's cliff iter ≥ 9 (vs v7-opd-full iter 7) corroborates trust-region mechanism.

## Decision matrix (factorial decomposition post-runs)

| Effect | Method | If significant → |
|---|---|---|
| critique-fix (base vs v7-opd-full) | direct pairwise | "moving critic into critical path delivers paper-fidelity AND empirical lift" |
| length-matched v7-opd lift (Decision 2) | v7-opd-full vs σ_v8 | reframes "+1.38 over σ legacy" claim; if TIE, length confound was the lift |
| mask main effect | mean of {A,C,C+,E,G,G+} − mean of {base,B,B+,D,F,F+} pairwise vs σ_v8 | "F11 token-budget noise mask robust" |
| α main effect (3-level) | mean over α=0 cells, α=0.05 cells, α=0.1 cells; check monotonicity | "Hübotter A.2 trust-region delays cliff monotonically up to α≤0.1" or "α≥0.05 collapses (anchor too strong)" |
| PPO main effect | mean of {D,E,F,F+,G,G+} − mean of {base,A,B,B+,C,C+} | "PPO clip > IS-loss for stability" |
| α × cliff-iter | per-cell cliff iter (audit_drop_threshold trigger) regressed on α | direct test of A.2 mechanism prediction |
| 2-way interactions | residuals after main | report only if interaction > main effect magnitude |

Production-checkpoint policy: lock the cell with highest pairwise win rate vs σ_v8 (M8 close-cluster tie-breaker if tied within Δ=1).

## Cost breakdown (revised for 12-cell grid + Decisions 1/2/3)

| Item | $ |
|---|---:|
| σ_v8 re-baseline (8 plans + audit) | 15 |
| 12 × 16-iter Tinker training | ~360 |
| Critic calls in critical path (16 × 12 = 192) | ~60 |
| Pairwise audit (110 pairs incl Decision 2 + α-sweep matchups) | ~250 |
| Strict /45 audit (12 cells × 2 iters × 8 plans) | ~190 |
| **Total** | **~875** |

Pre-registered cost-reduction levers:
- **Batch 3 drop** (Decision 1): if `base` wins ≥6/8 vs v7-opd-full AND `G` wins ≥6/8 vs `base` → skip Batch 3 (5 cells × ~$50 + audit on those = saves ~$170 → revised total $705).

## Risks identified

1. **Critic call in critical path may time out** (~30-60s typical, 900s limit). Mitigation: keep `COLD_START_CRITIQUE_NONE` fallback for genuine timeouts (do NOT regress to lagged design); track timeout rate per run via metric `critique_was_cold_start`.

2. **Frozen base sampling client unverified on D5 sync Tinker API**. Mitigation: Batch 0 smoke explicitly verifies `service_client.create_sampling_client(base_model=...)` works and `frozen_lp != teacher_lp`. Abort full grid if smoke fails.

3. **Solution mask edge cases** — Qwen3 thinking-mode sometimes places content inside `<think>` outside `<solution>`. `find_solution_content_span` handles this (excludes solution spans overlapping think). Smoke checks `mean_solution_mask_density > 0.5` per plan.

4. **Cost overrun if M8 close-cluster on PRIMARY**. Pre-authorize n=16 rerun on PRIMARY only (+$60).

5. **Daemon scheduling**: 8 runs × 16 iter × ~$1 critic = 128 critic calls + 128 audit calls. Daemon must be up for ~5 days. Mitigation: existing `opus_critic_audit_daemon.md` validated for v7-opd-full.

6. **Word-count change σ confound**: σ_v8 (900-footer) vs σ legacy (750-footer) audit /45 not directly comparable. Solved by anchoring v8 grid to σ_v8, not σ legacy.

## Out of scope (per plan)

- ❌ Canonical Hübotter top-K KL (Tinker doesn't expose vocab logits)
- ❌ Canonical OPSD full-vocab JS (same)
- ❌ Replay buffer (Phase 6α — separate plan)
- ❌ Migration to OPSD privileged y* (separate user decision)
- ❌ Touching v7-opd / v4 / v6 trainers (preserve checkpoints)

## Verification checklist (before launching Batch 1)

- [x] Trainer file written (`train_mu_v8_d5sdpo.py`) — done
- [x] Prompt module written (`mu_prompts_v8.py`) — done
- [x] AST parse + import OK — verified
- [x] `pytest tests/test_replay_buffer_v1.py` 10/10 pass — verified
- [ ] σ_v8 re-baseline run + audit (Batch 0)
- [ ] 1-iter G smoke verifies critic-before-teacher_lp ordering, frozen base client, solution mask density, ppo loss config
- [ ] Daemon launched in separate Claude window (`opus_critic_audit_daemon.md`)
- [ ] Run-dir naming locked

## Open decisions — RESOLVED 2026-04-28 (per user "按照片里的说法来做")

| # | Decision | Resolution |
|---|---|---|
| 1 | Pre-authorize Batch 3 drop on Batch 1 decisive win | **YES, pre-registered** — gated on `base` ≥6/8 vs v7-opd-full AND `G` ≥6/8 vs `base`. Saves ~$170 if triggered. ML-design standard: pre-register conditional drops, don't decide ad-hoc. |
| 2 | Add v7-opd-full vs σ_v8 matchup ($10) | **YES, added to Batch 1** — separates critique-fix lift from word-footer-length lift. Reframes "v7-opd-full +1.38 over σ legacy" if length confound was the driver. |
| 3 | α sweep | **YES, pre-registered upfront 3-level {0, 0.05, 0.1}** — NOT conditional on G's cliff signal (garden-of-forking-paths). Adds 4 cells (B+, C+, F+, G+ at α=0.1). +$200. Pre-registered hypothesis: cliff-iter monotonic in α. |

**Total revised cost**: ~$875 (with Batch 3 drop trigger: ~$705)

## User confirmation

Awaiting "yes go" to launch Batch 0 (σ_v8 re-baseline + G smoke 1-iter, ~$20).
