# D5 Run Registry — truthful per-run setup record

*Created 2026-04-27 PM. **Single source of truth for what each historical run actually was**, derived from `config.json` + `code.diff` + actual trainer source code (NOT from docstrings or memory). Comparability across runs requires reading from this doc, not from older STATUS / finding docs.*

## Ground rule

Every column below is derived from one of:
1. The run's saved `config.json` (chz config snapshot)
2. The run's `code.diff` (which trainer was modified at run time)
3. The run's `launch.log` first line (`python -m co_scientist.d5_abstract_retrieve_refine.train_<X>`)
4. Actual `train_<X>.py` algorithm code at the version checked out at run time

If a column cell says **NOT VERIFIABLE FROM ARTIFACTS**, it means the artifact is missing or insufficient to determine that field. Do not guess.

**Truthful algorithm labels** (no claims to canonical SDPO/OPSD that aren't verified):
- `D5 in-house off-policy IS-loss` — TEACHER samples; STUDENT lp recomputed; IS-loss. Used by μ-v2/v3/v4/v6/κ-v1 family. NEITHER canonical SDPO nor OPSD.
- `D5 in-house on-policy IS-loss` — STUDENT samples on-policy; TEACHER lp recomputed; IS-loss. Used by μ-v7-opd (`opd_mode=True`). Closest to OPSD sampled-token policy-gradient (Zhao 2601.18734 Table 3) but NOT canonical KL-form.
- `SFT` — pure supervised fine-tuning. No advantage. No sampling. Used by α/β.
- `frozen inference` — no training. Used by σ/τ.
- `continual transfer of <prior_run> LoRA` — picks up `<prior_run>`'s LoRA state via `init_state_path`; algorithm = same as prior run's class.

For paper-faithful canonical algorithms, see `CANONICAL_NAMING_REFERENCE.md`.

## Master table — all training + inference runs

| Run dir | Trainer | Algorithm (truthful) | Sampler | Goal | Init state | LR | LoRA r | n_iter (planned/actual) | Cliff? | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026_04_25_mu_baseline_v1 | train_mu_baseline_v1.py | D5 in-house off-policy IS-loss (early version) | TEACHER | TTT-D | fresh | NOT VERIFIABLE | NOT VERIFIABLE | NOT VERIFIABLE | NOT VERIFIABLE | μ baseline; pre-realignment ARCHIVED |
| 2026_04_26_mu_v2 | train_mu_v2.py | D5 in-house off-policy IS-loss | TEACHER | TTT-D | fresh | 1e-5 | 64 | 10 / 10 | No | Phase 2 baseline; n_grad_steps_per_iter NOT SET in config |
| 2026_04_27_mu_v3 | train_mu_v3.py | D5 in-house off-policy IS-loss | TEACHER | TTT-D | fresh | 2e-4 | 64 | 20 / 20 | No (collapse iter 6) | LR ablation; collapsed at iter 6 audit per F3 |
| 2026_04_27_mu_v4 | train_mu_v4.py | D5 in-house off-policy IS-loss | TEACHER | TTT-D | fresh | 5e-5 | 64 | 20 / 8 (early-stop) | Yes (iter 5 cliff -12.4) | **Production** through 2026-04-27. audit_drop_threshold=3.0; production checkpoint = iter 4 |
| 2026_04_28_mu_v5_anchor_ce_v1 | train_mu_v4.py (chz override) | continual transfer of mu_v4 iter 2 LoRA via D5 in-house off-policy IS-loss | TEACHER | tool_v_ttrl | mu_v4 iter 2 | 5e-5 | 64 | 5 / 1 | Yes | anchor_ce_weight=0.1; reset_optimizer_state=True; cliff iter 1 |
| 2026_04_28_mu_v5_tool_v_cliff_v1 | train_mu_v4.py | continual transfer of mu_v4 iter 4 LoRA | TEACHER | tool_v_ttrl | mu_v4 iter 4 | 5e-5 | 64 | 5 / 2 | Yes (cliff iter 1) | reset_optimizer_state=False; F9 4-cell |
| 2026_04_28_mu_v5_tool_v_cliff_v2 | train_mu_v4.py | continual transfer of mu_v4 iter 4 LoRA | TEACHER | tool_v_ttrl | mu_v4 iter 4 | 5e-5 | 64 | 5 / 2 | Yes (cliff iter 1) | reset_optimizer_state=True; F9 4-cell |
| 2026_04_28_mu_v5_iter2anchor_5iter | train_mu_v4.py | continual transfer of mu_v4 iter 2 LoRA | TEACHER | tool_v_ttrl | mu_v4 iter 2 | 5e-5 | 64 | 5 / NOT VERIFIABLE | NOT VERIFIABLE | reset_optimizer_state=False |
| 2026_04_28_kappa_v1_smoke | train_kappa_v1.py | D5 in-house off-policy IS-loss (distillation-level, 3-round) | TEACHER | TTT-D | fresh | 5e-5 | 64 | 2 / 0 (smoke) | N/A | n_rounds=3, n_instances=8; smoke test only |
| 2026_04_29_mu_v7_opd_smoke | train_mu_v7_opd.py | D5 in-house on-policy IS-loss (closest to OPSD sampled-token, Zhao 2601.18734) | STUDENT | TTT-D | fresh | 5e-5 | 64 | 4 / 4 | No | opd_mode=True; smoke run; cold-start critique iter 0 (daemon late) |
| 2026_04_29_mu_v7_opd_full | train_mu_v7_opd.py | D5 in-house on-policy IS-loss | STUDENT | TTT-D | fresh | 5e-5 | 64 | 16 / 16 | Yes (cliff iter 7) | opd_mode=True; full run; cliff delayed +2 vs μ-v4 |
| 2026_04_25_alpha_baseline_v1 | train_alpha_baseline_v1.py | SFT (pre-realignment) | n/a | TTT-D | fresh | NOT VERIFIABLE | NOT VERIFIABLE | NOT VERIFIABLE | N/A | ARCHIVED pre-realignment |
| 2026_04_26_alpha_v2 | train_alpha_v2.py | SFT (Opus distillation; goal+oracle_v2) | n/a | TTT-D | fresh | 1e-5 | 64 | 5 epochs / 5 | No | n_eval_plans=8; max_tokens=2048 |
| 2026_04_26_alpha_v2b | train_alpha_v2.py | SFT (Opus distillation) | n/a | TTT-D | fresh | 1e-5 | 64 | 5 epochs / 5 | No | max_tokens=4096 (vs 2048 in v2) |
| 2026_04_25_beta_baseline_v1 | train_beta_baseline_v1.py | SFT (pre-realignment) | n/a | TTT-D | fresh | NOT VERIFIABLE | NOT VERIFIABLE | NOT VERIFIABLE | N/A | ARCHIVED pre-realignment |
| 2026_04_26_beta_v2 | train_beta_v2.py | SFT (direct ref imitation; no oracle) | n/a | TTT-D | fresh | 1e-5 | 64 | 10 / NOT VERIFIABLE | NOT VERIFIABLE | max_tokens=2048; single ref_plan example |
| 2026_04_26_beta_v2b | train_beta_v2.py | SFT (direct ref imitation) | n/a | TTT-D | fresh | 1e-5 | 64 | 10 / NOT VERIFIABLE | NOT VERIFIABLE | max_tokens=4096 (vs 2048) |
| 2026_04_29_mu_v8_d5sdpo_base | train_mu_v8_d5sdpo.py | D5 in-house on-policy IS-loss + critique-on-current | STUDENT | TTT-D | fresh | 5e-5 | 64 | 16 / 7 (KILLED iter 6) | Yes (cliff iter 5-6) | mask=F α=0 IS; F15 base. Peak 20.00 iter 3-4; final iter 6 = 10.00 |
| 2026_04_29_mu_v8_d5sdpo_G | train_mu_v8_d5sdpo.py | D5 in-house on-policy IS-loss + critique-on-current + mask + α=0.05 + PPO clip | STUDENT | TTT-D | fresh | 5e-5 | 64 | 16 / 10 (KILLED iter 9) | Yes (cliff iter 7-8) | mask=T α=0.05 PPO ε=0.2; F15. Peak 20.00 iter 3; final iter 9 = 10.25 |
| 2026_04_29_mu_v8_d5sdpo_Gplus | train_mu_v8_d5sdpo.py | D5 in-house on-policy IS-loss + critique-on-current + mask + α=0.1 + PPO clip | STUDENT | TTT-D | fresh | 5e-5 | 64 | 16 / 13 (DONE flag at iter 12; audit_drop_threshold=10.0) | Yes (slow cliff iter 11) | mask=T α=0.1 PPO ε=0.2; **F15 G+ peak 22.38 iter 4-5**; pairwise 8-0 vs σ_v8, 3-5 vs v7-opd-full, 8-0 vs μ-v4, 7-1 vs G |
| 2026_04_29_mu_v8_d5sdpo_E | train_mu_v8_d5sdpo.py | D5 in-house on-policy IS-loss + critique-on-current + mask + α=0 + PPO clip | STUDENT | TTT-D | fresh | 5e-5 | 64 | 16 / 10 (KILLED iter 9) | Yes (cliff iter 7-8) | mask=T α=0 PPO ε=0.2; F15. **Peak 22.75 iter 5** (highest peak in α=0); final iter 9 = 9.75 |
| 2026_04_29_mu_v8_d5sdpo_Cplus | train_mu_v8_d5sdpo.py | D5 in-house on-policy IS-loss + critique-on-current + mask + α=0.1 + IS | STUDENT | TTT-D | fresh | 5e-5 | 64 | 16 / 12 (KILLED iter 11) | Yes (cliff iter 10) | mask=T α=0.1 IS; F15. **Peak 23.50 iter 6** (highest peak across all v8 cells, no pairwise) |
| 2026_04_29_mu_v8_d5sdpo_Fplus | train_mu_v8_d5sdpo.py | D5 in-house on-policy IS-loss + critique-on-current + α=0.1 + PPO clip (no mask) | STUDENT | TTT-D | fresh | 5e-5 | 64 | 16 / 10 (KILLED iter 9) | Yes (cliff iter 8-9) | mask=F α=0.1 PPO ε=0.2; F15. Peak 23.00 iter 5-6; final iter 9 = 13.38 |
| 2026_04_29_mu_v8_d5sdpo_A | train_mu_v8_d5sdpo.py | D5 in-house on-policy IS-loss + critique-on-current + mask + α=0 + IS | STUDENT | TTT-D | fresh | 5e-5 | 64 | 16 / 2 (FAILED iter 1, critic timeout 900s + later SIGTERM) | n/a (failed before training landed) | mask=T α=0 IS; F15. Iter 0=19.38, iter 1=19.13. Critic timeout broke training; A still pending rerun for mask-only attribution |

## Inference-only runs (no training; included for completeness)

| Run dir | Generator | Sampler weights | Goal | n_plans | Notes |
|---|---|---|---|---|---|
| 2026_04_26_xi_v2 | train_xi.py (frozen) | base Qwen3-30B | TTT-D | 8 | ξ baseline = goal only, NO oracle |
| 2026_04_26_sigma_v2 | train_sigma.py (frozen) | base Qwen3-30B | TTT-D | 8 | σ baseline = goal + slim_oracle (no critique) |
| 2026_04_26_delta_v2 | (frozen inference) | base Qwen3-30B | TTT-D | 8 | δ = goal + reference plan in context |
| 2026_04_26_epsilon_v2 | (frozen inference) | base Qwen3-235B | TTT-D | 8 | ε ceiling = 235B + reference plan |
| 2026_04_28_xgoal_meta_ttl_mu | (μ-v4 LoRA inference) | mu_v4 iter 4 | meta_ttl | 8 | F7 cross-goal test |
| 2026_04_28_xgoal_tool_verification_ttrl_mu | (μ-v4 LoRA inference) | mu_v4 iter 4 | tool_v_ttrl | 8 | F7 cross-goal test |
| 2026_04_28_xgoal_tt_control_mu | (μ-v4 LoRA inference) | mu_v4 iter 4 | tt_control | 8 | F7 cross-goal test |
| 2026_04_28_xgoal_<goal>_mu_n24 | (μ-v4 LoRA inference, n=24) | mu_v4 iter 4 | meta_ttl / tool_v_ttrl | 16 (combined to 24 with above) | F7 D-series stress test |
| 2026_04_29_xgoal_<goal>_mu_opd | (μ-v7-opd LoRA inference) | mu_v7_opd iter 4 | meta_ttl / tool_v_ttrl / tt_control | 8 | F14 Phase 4a redo |
| 2026_04_28_tau_v* | train_tau_v1.py (frozen) | base Qwen3-30B | TTT-D | varies | 3-round distillation inference; main session work (F8) |
| 2026_04_28_sigma_v4_v1 | train_sigma_v4_v1.py | base or μ-v4 | TTT-D | varies | σ_v4 baseline w/ plan_v4 prompt (F8 control) |
| 2026_04_29_f4_attribution_offline | f4_attribution_offline.py | mu_v4 iter 2 | TTT-D | (8 buffer plans) | Stage A: re-compute t_lp, s_lp on existing μ-v4 buffer plans; not training |
| 2026_04_29_f4_stage_b_critique_probe | f4_stage_b_critique_probe.py | mu_v7_opd iter 4 | TTT-D | 8 (×2 arms) | Stage B: 2-arm sampling probe; not training |
| 2026_04_29_mu_v8_d5sdpo_sigma_v8 | baseline_frozen_v2.py (footer_version=v8) | base Qwen3-30B + slim_oracle, 900/1100 footer | TTT-D | 8 | **σ_v8 anchor for v8 grid (length-matched, strict 1-plan/Opus)**; mean 19.38/45; mean 811 words; replaces σ legacy 25.25 as v8 anchor |
| 2026_04_29_reference_audit_v3 | (frozen ref-plan re-audit, strict 1-plan/Opus) | n/a | TTT-D | 1 | Reference plan audit-v3 = 35/45 (universal 21 + subfield 14); ceiling reference for v8 cells |
| 2026_04_29_mu_v8_d5sdpo_Gplus_pairwise | mu_v8_d5sdpo_pairwise_v1.py | G+ iter 4 vs σ_v8 / v7-opd-full / μ-v4 / G | TTT-D | 4 matchups × 8 pairs | Phase 1 pairwise corroboration; verdicts G+ 8-0 vs σ_v8, **3-5 vs v7-opd-full (G+ LOSES)**, 8-0 vs μ-v4, 7-1 vs G |

## Per-run detail records (load-bearing runs)

### 2026_04_27_mu_v4 (production through 2026-04-27)

- **Trainer**: `src/co_scientist/d5_abstract_retrieve_refine/train_mu_v4.py`
- **Algorithm (truthful)**: D5 in-house off-policy IS-loss
  - Sample: TEACHER under `teacher_input = build_teacher_prompt(goal, oracle, prev_critique)`
  - Logprob signal: `student_lp = compute_logprobs(student_input + teacher_seq_tokens)` where `student_input = build_student_prompt(goal, oracle)` — NO critique
  - Advantage: `A_t = clamp((teacher_lp − student_lp) · 1.0, ±5)`
  - Datum: `build_sdpo_datum(student_prompt_tokens=student_input_tokens, gen_tokens=teacher_seq_tokens, ...)` — student prefix + teacher tokens (off-manifold for student EVAL)
  - Loss: `forward_backward(loss_fn="importance_sampling")`
- **Goal**: `dataset/research_goal.txt` (TTT-Discover, Hübotter 2026-01)
- **Oracle**: `data/oracles/oracle_v2_2026_04_26_build/slim.md`
- **Reference plan**: `dataset/reference_solution.txt` — NOT used in main loss (anchor_ce_weight=0)
- **Critic**: Opus subagent, has source paper as privileged info. Iter 0 = cold-start critique.
- **Config**: lr=5e-5, LoRA rank 64, n_plans 8, n_grad_steps_per_iter 4, max_tokens 4096, temp 1.0, top_p 0.95, seed 42, audit_drop_threshold=3.0
- **Iters**: planned 20, actual ~8 (early-stopped at iter 5 cliff)
- **Cliff**: iter 5 audit drop -12.4 (per F3 trajectory: iter 4 = 28.00 nominal → iter 5 = 15.62)
- **Production checkpoint**: state at end of iter 4 (= start of iter 5; row batch=4 / 5 per checkpoints.jsonl). Sampler path: `tinker://6b9d996d-8079-556d-82ac-560247551960:train:0/sampler_weights/iter_0004`
- **Cited in**: F2_mu_v4_iter4_peak.md, F3_sdpo_multi_round_instability.md, F4_critique_token_blindness.md, F11/F12/F13/F14, PROJECT_OVERVIEW.md, STATUS.md
- **Comparable to**: μ-v2 / μ-v3 (same algorithm, different lr); μ-v7-opd (same algorithm family but on-policy variant — F13/F14 paper-grade pairwise comparison)
- **Caveats**: 
  - 28.00/45 nominal audit was M8-confounded inline-batched; strict 1-plan/Opus likely ~24-25
  - "SDPO" label in this run's outputs is mislabeling — actual algorithm is D5 in-house off-policy IS-loss (not canonical Hübotter SDPO 2601.20802)

### 2026_04_29_mu_v7_opd_smoke

- **Trainer**: `src/co_scientist/d5_abstract_retrieve_refine/train_mu_v7_opd.py`
- **Algorithm (truthful)**: D5 in-house on-policy IS-loss (closest to OPSD sampled-token policy-gradient, Zhao 2601.18734 Table 3)
  - Sample: STUDENT on-policy under `student_input = goal + oracle (no critique)`
  - Logprob signal: `teacher_lp = compute_logprobs(teacher_input + student_seq_tokens)` where `teacher_input = goal + oracle + prev_critique`
  - Advantage: `A_t = clamp((teacher_lp − student_lp) · 1.0, ±5)` — sources swapped from v4
  - Datum: `student_prompt + student_seq_tokens` — on-manifold by construction
  - Loss: `forward_backward(loss_fn="importance_sampling")` (same Tinker call as v4)
- **Goal**: TTT-D (same as v4)
- **Oracle**: same slim_v2
- **Reference plan**: NOT used in main loss
- **Critic**: Opus subagent; iter 0 critic timed out (daemon attached late) → cold-start fallback. Iter 1-3 real Opus critique.
- **Config**: opd_mode=True, lr=5e-5, LoRA r=64, n_plans=8, n_grad_steps=4, audit_drop_threshold=5.0, seed=42
- **Iters**: planned 4, actual 4 (completed)
- **Cliff**: No (4-iter window too short to confirm absence)
- **Cited in**: F13_mu_opd_canonical_architecture.md (smoke pairwise 15/16); HANDOFF
- **Comparable to**: μ-v4 (same algorithm family, on-policy fix; pairwise μ-v7-opd-smoke vs μ-v4 = 15/16 on TTT-D)
- **Caveats**:
  - "OPD" label is informal; actual is D5 in-house on-policy IS-loss approximation of OPSD sampled-token form
  - Inline-batched audit dispatcher claimed Task tool unavailable; M8 confound likely on /45 numbers (27.75 reported); pairwise 15/16 anonymized is the M8-resistant ranking

### 2026_04_29_mu_v7_opd_full

- **Trainer**: `train_mu_v7_opd.py` (same as smoke)
- **Algorithm**: same as smoke — D5 in-house on-policy IS-loss
- **Goal / Oracle / Critic**: same setup as smoke
- **Config**: opd_mode=True, lr=5e-5, LoRA r=64, n_plans=8, n_grad_steps=4, audit_drop_threshold=5.0, seed=42
- **Iters**: planned 16, actual 16 (completed)
- **Cliff**: Yes — iter 7 audit drop (per 4-dim daemon: iter 4 = 10.50 → iter 7 = 4.38). +2 iter delayed vs μ-v4 (which cliffed iter 5).
- **Production candidate checkpoint**: state at end of iter 4 (sampler `iter_0004`)
- **Cited in**: F13 + F14 + HANDOFF
- **Comparable to**:
  - μ-v4 production (paper-grade pairwise 8-0 vs μ-v4 iter 4)
  - μ-v7-opd-smoke (same trainer + config; full = longer iter; both reach similar peak at iter 4)
- **Caveats**: same M8 caveat as smoke; pairwise is the authoritative ranker

### Continual transfer runs (Phase 5 4-cell grid)

All four cliff_v* / anchor_ce_v1 / iter2anchor runs use `train_mu_v4.py` with `init_state_path` set to a μ-v4 LoRA snapshot. They are NOT independent training runs — they continue from existing weights. Algorithm is **same as μ-v4** (D5 in-house off-policy IS-loss) just with different anchor + reset_optimizer_state combinations:

| Run | Anchor (init_state_path) | reset_optimizer | anchor_ce_weight | Cliff iter | F9 cell |
|---|---|---|---|---|---|
| anchor_ce_v1 | μ-v4 iter 2 | True | 0.1 | iter 1 | (anchor_saturation × Adam state × anchor_ce) cell |
| cliff_v1 | μ-v4 iter 4 | False | 0 | iter 1 | cell |
| cliff_v2 | μ-v4 iter 4 | True | 0 | iter 1 | cell |
| iter2anchor_5iter | μ-v4 iter 2 | False | 0 | NOT VERIFIABLE | cell |

All 4 cells failed forward learning per F9 NULL. **Caveat**: F9's NULL verdict is for D5 in-house off-policy IS-loss continual chains; canonical SDPO/OPSD continual was NOT tested.

## Cross-reference: which run is cited where

| Run | Cited in | Caveats for citation |
|---|---|---|
| mu_v4 | F2, F3, F4, F11, F12, F13, F14, PROJECT_OVERVIEW, STATUS, DECISIONS | All citations should qualify "SDPO" → "D5 in-house off-policy IS-loss" |
| mu_v7_opd_smoke | F13, F14, HANDOFF | "OPD" / "canonical SDPO" labels in cited docs are informal; truthful = D5 on-policy IS-loss (closest to OPSD sampled-token) |
| mu_v7_opd_full | F13, F14, HANDOFF | Same caveat |
| Phase 5 cliff_v* | F9, DECISIONS | F9 NULL is for in-house off-policy IS-loss continual; canonical not tested |
| α-v2 / β-v2 | F1 baseline ranking, PROJECT_OVERVIEW | SFT — not in SDPO/OPSD family at all |
| σ_v2 / δ_v2 / ε_v2 / ξ | All baseline tables | Frozen inference — not training |

## Anchor numbers (DO NOT re-baseline; cite from this registry)

| Anchor | /45 | Source | Caveat |
|---|---:|---|---|
| ξ (frozen + goal only) | 15.62 | runs/2026_04_26_phase2F_audit_isolated/ | strict isolated (M8-compliant) |
| σ (frozen + slim_oracle) | 25.25 | same | strict isolated |
| δ (frozen + reference plan) | 25.12 | same | strict isolated |
| α (Opus distill SFT) | 25.00 | same | strict isolated |
| μ-v2 | 23.88 | same | strict isolated |
| β (ref-only SFT) | 22.75 | same | strict isolated |
| **μ-v4 28.00** | 28.00 | NOMINAL (inline-batched audit, M8-confounded) | **CAVEAT: strict isolated likely 24-25; pairwise vs σ at the time was 6-2 directional** |
| ε (frozen 235B + reference) | 33.62 | same | strict isolated |
| **μ-v7-opd iter 4** | **26.62** | runs/2026_04_29_mu_v7_opd_full_audit_v3/ | **strict 1-plan/Opus isolated (post-M8 fix); +1.38 over σ; pairwise 23/24 vs μ-v4 production** |
| **σ_v8** (frozen + slim_oracle, 900/1100-word footer) | **19.38** | runs/2026_04_29_mu_v8_d5sdpo_sigma_v8/ | strict 1-plan/Opus length-matched anchor for v8 grid; Δ −5.88 vs σ legacy due to methodology + length confounds |
| reference plan (TTT-Discover) | 35.00 | runs/2026_04_29_reference_audit_v3/ | strict 1-plan/Opus isolated; ceiling reference (universal 21 + subfield 14) |
| μ-v8-d5sdpo G+ iter 4-5 peak | 22.38 | runs/2026_04_29_mu_v8_d5sdpo_Gplus/audit_responses/iter_004.json | mask=T α=0.1 PPO; pairwise 8-0 vs σ_v8, 3-5 vs v7-opd-full (G+ LOSES), 8-0 vs μ-v4, 7-1 vs G |
| μ-v8-d5sdpo C+ iter 6 peak | 23.50 | runs/2026_04_29_mu_v8_d5sdpo_Cplus/audit_responses/iter_006.json | mask=T α=0.1 IS; **highest v8 peak**, NO pairwise corroboration; production candidate pending pairwise |
| μ-v8-d5sdpo F+ iter 5-6 peak | 23.00 | runs/2026_04_29_mu_v8_d5sdpo_Fplus/audit_responses/iter_005.json | mask=F α=0.1 PPO; cliff iter 8-9; NO pairwise |
| μ-v8-d5sdpo E iter 5 peak | 22.75 | runs/2026_04_29_mu_v8_d5sdpo_E/audit_responses/iter_005.json | mask=T α=0 PPO; cliff iter 7-8; NO pairwise |

## How to use this registry

When citing a run in any new doc:
1. Look up the run dir in this table
2. Use the truthful "Algorithm" label (don't use "SDPO" without qualifier unless citing the actual Hübotter paper)
3. Cite caveats (M8 confound on μ-v4 28.00; pairwise vs audit /45 distinction)
4. If comparing two runs, ensure they're in the same algorithm family OR explicitly note the cross-family comparison

When designing a new run:
1. Reference the closest existing run from this registry as comparator
2. Choose algorithm label: "D5 in-house off-policy IS-loss" (use train_mu_v4 family) or "D5 in-house on-policy IS-loss" (use train_mu_v7_opd) or canonical OPSD/SDPO (need new trainer; not yet implemented)
3. Document setup precisely; future you will need this in the next registry update

## See also

- `CANONICAL_NAMING_REFERENCE.md` — paper attribution (Hübotter SDPO 2601.20802 vs Zhao OPSD 2601.18734)
- `SDPO_RECIPE_v1.md` — the recipe doc (now correctly framed as "D5 in-house off-policy IS-loss")
- `paper_materials/findings/F*.md` — finding docs; reference this registry for run-level details
