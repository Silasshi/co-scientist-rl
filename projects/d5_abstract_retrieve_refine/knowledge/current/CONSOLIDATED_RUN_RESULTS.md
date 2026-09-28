# D5 Consolidated Run Results — config + audit + findings (per cell)

**Created**: 2026-04-29 PM
**Updated**: 2026-04-29 PM (added v1-v7 era runs + protocol-confound disclosure)
**Purpose**: Single comprehensive comparison table of all D5 training+inference runs, with config, length-bucketed audit means, and per-run findings.

## ⚠️ CRITICAL: two non-comparable audit eras

| Era | Length target / max | Audit protocol | Audit script |
|---|---|---|---|
| **v1-v7 era (2026-04-25 to 04-28)** | 600 / 750 words | `audit_v3 ISOLATED` — **8 batches × 8 parallel Opus subagents, batch-anchored** | `audit_v3_isolated.py` |
| **v8 / v9 era (2026-04-29)** | 900 / 1100 words | `M8-strict` — **1 plan / 1 Opus subagent, no batch anchoring** | (file-bus subagent dispatch) |

**Confound**: per memory `feedback_audit_close_cluster_corroboration` (2026-04-27): "**2-plan/subagent batch anchoring inflates Δ ~2-3 points** vs strict 1-plan/subagent". v1-v7 absolute scores were measured under the inflated batch-anchored protocol; v8/v9 under the deflated strict protocol.

**Implication**: the cross-era comparison is **CONFOUNDED**. v1-v7 peak (μ-v4 28.00 / ε 33.62) is NOT directly comparable to v8/v9 peak (C+ 23.50 / L2 23.38). Net headline gap of ~5pt is at least partly an audit-protocol artifact, not a genuine quality regression.

To properly compare:
- Either re-run all v1-v7 cells under M8-strict 1-plan/Opus protocol (~$30 × ~10 cells = $300)
- Or re-run all v8/v9 cells under audit_v3 ISOLATED batch-anchored protocol (~$60 / cell × ~10 cells = $600)
- Or re-audit a calibration set (e.g., μ-v4 iter 4) under BOTH protocols and derive a calibration constant

This doc clearly separates the two eras below.

**Sources**:
- Config: each run's `config.json` + `code.diff` + `launch.log`
- Audit (v8/v9): each run's `audit_responses/iter_???.json` joined with `eval_rollouts.jsonl` for plan length
- Audit (v1-v7): `runs/2026_04_26_phase2F_audit_isolated/audit_v3_isolated_summary.md` + `project_d5_baseline_ranking` memo + `project_d5_phase2_progress` memo
- Truthful algorithm labels: see `RUN_REGISTRY.md` master table
- Detailed config snapshots: see `RUN_REGISTRY.md` per-run sections

For per-run granular history (LR ablations, kappa smoke, etc.) see `RUN_REGISTRY.md`. This doc focuses on **paper-relevant cells**.

---

## Era 1: v1-v7 (audit_v3 ISOLATED, target 600 / max 750 words)

These runs use `audit_v3_isolated.py` which dispatches 8 batches × 8 parallel anonymized Opus subagents with batch anchoring. Per `feedback_audit_close_cluster_corroboration` memo, batch anchoring inflates scores by ~2-3pt vs strict 1-plan-isolation. Treat absolute /45 numbers in this era as **inflated by ~2-3pt**.

### v1-v7 master comparison table

| Cell | Trainer | Algorithm (truthful) | Sampler | Init | LR | LoRA r | n_iter | Audit /45 (audit_v3 ISOLATED) | Mean word count | Peak iter | Cliff? | Status |
|---|---|---|:-:|---|:-:|:-:|:-:|---:|---:|:-:|:-:|---|
| **ε** | (frozen inference) | base Qwen3-235B + reference plan | n/a | base | n/a | n/a | 1 | **33.62** | n/a | static | n/a | ceiling reference (235B); out of reach for 30B-trained variants |
| **σ** | train_sigma.py | frozen Qwen3-30B + slim_oracle (no critique) | n/a | base | n/a | n/a | 1 | **25.25** | ~580 | static | n/a | **canonical 30B baseline** anchor for v1-v7 era |
| δ | (frozen inference) | base Qwen3-30B + reference plan in context | n/a | base | n/a | n/a | 1 | 25.12 | ~580 | static | n/a | ref-plan-in-context ceiling for 30B |
| α (alpha_v2) | train_alpha_v2.py | SFT (Opus distillation; goal+oracle_v2) | n/a | base | 1e-5 | 64 | 5 epochs | 25.00 | 592 | static | n/a | naive distillation; matches σ |
| α (alpha_v2b) | train_alpha_v2.py | SFT (max_tokens=4096) | n/a | base | 1e-5 | 64 | 5 epochs | (similar to v2) | 578 | static | n/a | length-budget variant of α |
| **μ-v2** | train_mu_v2.py | D5 in-house off-policy IS-loss (lr=1e-5) | TEACHER | base | 1e-5 | 64 | 10 | **23.88** | 579 | n/a | No | early SDPO; underperforms σ by 1.37pt |
| β (beta_v2) | train_beta_v2.py | SFT (direct ref imitation; no oracle) | n/a | base | 1e-5 | 64 | 10 | **22.75** | 562 | static | n/a | direct ref-plan imitation |
| ξ | train_xi.py (frozen) | frozen Qwen3-30B + goal only (NO oracle) | n/a | base | n/a | n/a | 1 | **15.62** | n/a | static | n/a | floor reference (no oracle) |
| **μ-v3** | train_mu_v3.py | D5 in-house off-policy IS-loss (lr=2e-4) | TEACHER | base | 2e-4 | 64 | 20 | **collapsed iter 6: 24.88 → 14.38** | 1402 | iter 6 | **YES** | KILLED; Chinese bleed, JSON leaks; lr too high |
| **μ-v4** | train_mu_v4.py | D5 in-house off-policy IS-loss (lr=5e-5) | TEACHER | base | 5e-5 | 64 | 20/8 (early-stop) | **iter 4 = 28.00** ★ | 881 | iter 4 | YES (iter 5 cliff -12.4) | **production** through 2026-04-27; pairwise 6-2 vs σ |
| μ-v4 replan | train_mu_v4.py (re-eval) | μ-v4 iter 4 LoRA, plan_v4 prompt | inference | μ-v4 iter 4 | n/a | 64 | 1 | (similar 28) | 996 | static | n/a | F8 control |
| μ-v5 anchor_ce | train_mu_v4.py + anchor_ce | continual transfer of μ-v4 iter 2 LoRA | TEACHER | μ-v4 iter 2 | 5e-5 | 64 | 5/1 | (cliff iter 1) | 872 | iter 0 | YES | F9 4-cell |
| μ-v5 tool_v cliff_v1 | train_mu_v4.py | continual transfer of μ-v4 iter 4 LoRA | TEACHER | μ-v4 iter 4 | 5e-5 | 64 | 5/2 | (cliff iter 1) | 986 | iter 0 | YES | reset_optimizer=False |
| μ-v5 tool_v cliff_v2 | train_mu_v4.py | continual transfer of μ-v4 iter 4 LoRA | TEACHER | μ-v4 iter 4 | 5e-5 | 64 | 5/2 | (cliff iter 1) | 991 | iter 0 | YES | reset_optimizer=True |
| **μ-v7-opd-full** | train_mu_v7_opd.py | D5 in-house on-policy IS-loss (~OPSD Zhao 2601.18734) | STUDENT | base | 5e-5 | 64 | 16 | (cliff iter 7; mid-pilot ~20) | 1695 | iter ~6 | YES | opd_mode=True; **5-3 pairwise vs G+** (v7 narrowly wins) |
| μ-v7-opd-smoke | train_mu_v7_opd.py | D5 in-house on-policy IS-loss | STUDENT | base | 5e-5 | 64 | 4 | (smoke) | 699 | n/a | No | smoke run; cold-start critique iter 0 |

### v1-v7 era findings

**1. μ-v4 28.00 was THE peak achievement of the entire D5 line.**
- Beat σ (25.25) by +2.75pt — first SDPO variant to demonstrably beat frozen+slim oracle
- Pairwise 6-2 vs σ corroborated; pairwise 8-0 vs μ-v2/μ-v3 corroborated
- This was paper-headline material as of 2026-04-27

**2. lr ablation: 5e-5 is sweet spot.**
- μ-v2 (1e-5): no convergence; mean_adv ~0.4 throughout
- μ-v3 (2e-4): catastrophic collapse iter 6 (Chinese bleed, JSON leaks)
- μ-v4 (5e-5): geometric median, peak iter 4 then cliff iter 5

**3. μ-v3 is a pre-existing cliff data point (iter 6, -10.5pt).**
The cliff problem was already visible in v1-v7. μ-v4's audit-drop early-stop guard (`audit_drop_threshold=3.0`) is what enabled the production checkpoint at iter 4.

**4. Transfer to other goals (Phase 5) all cliffed at iter 1.**
3 continual-transfer runs (mu_v5_anchor_ce, mu_v5_tool_v_cliff_v1/v2) all cliffed at iter 1 regardless of optimizer reset, regardless of base checkpoint (iter 2 vs iter 4). Cross-goal LoRA transfer is unstable.

**5. SFT family (α, β, alpha_v2b, beta_v2b) plateaus at ~22-25pt.**
α=25.00 matches σ; β=22.75 below σ. Direct SFT cannot beat frozen+slim oracle.

**6. ξ (no oracle) is the floor at 15.62.**
~10pt below σ — confirms oracle-in-prompt provides ~10pt of audit value at zero training cost.

---

## Era 2: v8 + v9 (M8-strict 1-plan/Opus, target 900 / max 1100 words)

These runs use `M8-strict` 1-plan/1-Opus dispatch (no batch anchoring). Per `feedback_audit_close_cluster_corroboration`, this protocol is ~2-3pt deflated vs v1-v7 era. Length target shifted from 600/750 to 900/1100.

### v8/v9 master comparison table

All audit scores M8-strict 1-plan/1-Opus, n=8 plans/iter, scale /45 (sum of 5 universal /25 + 4 subfield /20). Length buckets are word counts.

| Cell | Trainer | Algorithm (truthful) | mask | α | loss | LR | LoRA r | n_iter | overall mean | 600-750 (n / mean) | 750-900 (n / mean) | 900-1100 (n / mean) | >1100 (n / mean) | peak | peak iter | final 3 iter | drop | Status |
|---|---|---|:-:|:-:|:-:|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|:-:|---|---:|---|
| **σ_v8 anchor** | baseline_frozen_v2.py | frozen inference + slim oracle, 900/1100 footer | n/a | n/a | n/a | n/a | n/a | 1 (static) | **19.38** | — | — | — | — | 19.38 | static | — | — | DONE |
| base | train_mu_v8_d5sdpo.py | D5 in-house on-policy IS-loss + critique | F | 0 | IS | 5e-5 | 64 | 7/16 | 17.59 | 1 / 21.0 | 12 / 19.0 | 5 / 19.2 | 37 / 17.1 | 20.00 | 3-4 | 10.00, 12.50, … | -10.0 | KILLED iter 6 |
| G | train_mu_v8_d5sdpo.py | + mask + α=0.05 PPO | T | 0.05 | PPO | 5e-5 | 64 | 10/16 | 17.07 | 1 / 17.0 | 8 / 19.2 | 11 / 19.1 | 58 / 16.7 | 20.00 | 3 | 10.25, … | -9.75 | KILLED iter 9 |
| **G+** | train_mu_v8_d5sdpo.py | + mask + α=0.1 PPO | T | 0.1 | PPO | 5e-5 | 64 | 13/16 | **19.37** | 1 / 15.0 | 17 / 18.7 | 27 / 17.7 | **59 / 20.4** | 22.38 | 4-5 | 19.38, 14.75, 12.50 | -9.88 cliff | DONE; production checkpoint = iter 4 |
| E | train_mu_v8_d5sdpo.py | + mask + α=0 PPO | T | 0 | PPO | 5e-5 | 64 | 10/16 | 18.49 | 0 / N/A | 11 / 19.7 | 9 / 20.3 | 60 / 18.0 | 22.75 | 5 | 9.75, … | -13.0 | KILLED iter 9 |
| **C+** | train_mu_v8_d5sdpo.py | + mask + α=0.1 IS | T | 0.1 | IS | 5e-5 | 64 | 12/16 | **20.11** | 5 / 14.8 | 26 / 18.7 | 20 / **20.9** | 44 / 21.4 | **23.50** | 6 | 19.25, 17.50, 14.75 | -8.75 cliff | KILLED iter 11; **highest peak**, no pairwise |
| F+ | train_mu_v8_d5sdpo.py | + α=0.1 PPO (no mask) | F | 0.1 | PPO | 5e-5 | 64 | 10/16 | 19.59 | 0 / N/A | 12 / 19.7 | 17 / 18.3 | 33 / 21.1 | 23.00 | 5-6 | 13.38, … | -9.62 | KILLED iter 9 |
| Gplus_rag | train_mu_v8_d5sdpo.py | G+ + RAG retrieval | T | 0.1 | PPO | 5e-5 | 64 | 8/16 | 18.56 | 8 / 19.1 | 9 / 18.8 | 25 / 18.7 | 20 / 18.3 | 21.25 | 5 | 21.25, 15.88, 16.88 | -4.37 | KILLED iter 7 (F17 RAG cliff +5 iter earlier) |
| **L1 v9_grounded** | train_mu_v9_grounded.py | G+ + verifier-grounded reward (eq +0.1, cite +0.05) | T | 0.1 | PPO | 5e-5 | 64 | 16/16 | **19.38** | 6 / **22.2** | 26 / 18.7 | 33 / 18.9 | 62 / 19.6 | 22.00 | 14 | 18.75, 22.00, 21.75 | -0.25 stable | DONE; mid-pilot cliff iter 9-12 |
| **L2 v9_kl_anchor** | train_mu_v9_kl_anchor.py | L1 + KL anchor toward π_oracle (β=0.05) | T | 0.1 | PPO | 5e-5 | 64 | 16/16 | **20.91** | 12 / **22.2** | 38 / 20.8 | 32 / 20.3 | 42 / 21.2 | **23.38** | 14 | 21.62, 23.38, **23.12** | **-0.26 stable** | DONE; only stable RL cell + balanced grounding |

**Reading the length-bucket columns**: for each cell, plans are bucketed by word count. The `n / mean` cell shows how many plans fell into that bucket and the audit mean of those plans. Plans <600 words are excluded (rare, n=0-4 across all cells).

## Length-bucket findings (cross-cell)

**1. Long plans (>1100) score higher than 900-1100 in most cells** — counter-intuitive given the goal's 600-750 word target hint.
- G+: >1100 = 20.4 vs 900-1100 = 17.7 (+2.7pt for longer)
- C+: >1100 = 21.4 vs 900-1100 = 20.9 (+0.5pt)
- F+: >1100 = 21.1 vs 900-1100 = 18.3 (+2.8pt)
- L2: >1100 = 21.2 vs 900-1100 = 20.3 (+0.9pt)

**Implication**: M8-strict Opus reviewer rewards verbosity/comprehensiveness over the goal's word-budget signal. This is a confound for any cell-vs-cell comparison done at fixed-iter.

**2. The 600-750 bucket is small but informative**:
- L1: 6 / 22.2 (highest 600-750 mean)
- **L2: 12 / 22.2** (most short plans + high mean → only cell that scores well at goal-target length)
- C+: 5 / 14.8 (worst — when C+ plans are short, they're shallow)
- G+: 1 / 15.0 (only 1 short plan in entire run)

**Implication**: L2 KL anchor toward SFT'd π_oracle pulls plan length toward shorter, denser outputs while maintaining audit score. This is plausibly because SFT data was 30 plans averaging ~1062 words — the KL anchor regularizes length toward the SFT target.

**3. L2 mean word count = 1004 vs all others ~1120-1150**: KL anchor reduced verbosity. Since long plans score higher (point 1), this is a *cost* — but L2 still has higher overall audit (20.91) because it gains on the 600-1100 ranges where short plans previously scored low.

## Per-run findings synthesis

### σ_v8 anchor (frozen + slim oracle)
- **Setup**: Qwen3-30B-A3B base, no training, prompt = goal + slim_oracle + 900/1100 word footer
- **Audit**: 19.38/45 (single iter, no trajectory)
- **Note**: separate isolated-audit ranking memo (`project_d5_baseline_ranking`) shows σ at 25.25 under different audit protocol — the 19.38 here uses M8-strict 1-plan/Opus length-matched protocol consistent with v8 cells
- **Finding**: All RL cells have peak audit ABOVE σ_v8 anchor (19.38), except baseline (peak 20.00). This means RL adds value over frozen+slim. But σ at 25.25 (isolated, not length-matched) is unbeaten — frozen+slim with privileged oracle still tops if we don't penalize length.

### base v8-d5sdpo (mask=F α=0 IS)
- **Audit**: peak 20.00 iter 3-4, cliff iter 5-6 (final 10.00)
- **Finding F15-1**: Hard cliff appears even without mask/α/PPO modifications — base critique-conditioned advantage alone already destabilizes.

### G v8-d5sdpo (mask=T α=0.05 PPO)
- **Audit**: peak 20.00 iter 3, cliff iter 7-8 (final 10.25)
- **Finding F15-2**: α=0.05 too small to stabilize; mask helps but cliff still occurs.

### G+ v8-d5sdpo (mask=T α=0.1 PPO)
- **Audit**: peak 22.38 iter 4-5, slow cliff iter 11 (final 12.50)
- **Pairwise**: 8-0 vs σ_v8, **3-5 vs v7-opd-full (G+ LOSES)**, 8-0 vs μ-v4, 7-1 vs G
- **Finding F15-3**: G+ is best v8 cell with paper-defensible pairwise verdict. Production checkpoint = iter 4. Cliff still occurs but later.

### E v8-d5sdpo (mask=T α=0 PPO)
- **Audit**: peak 22.75 iter 5 (highest in α=0 family), cliff iter 7-8 (final 9.75)
- **Finding F15-4**: PPO clipping with mask but no entropy bonus — high peak but cliffs hard. α=0 doesn't help stability.

### C+ v8-d5sdpo (mask=T α=0.1 IS)
- **Audit**: **peak 23.50 iter 6 — highest peak across all v8 cells**, cliff iter 10 (final 14.75)
- **No pairwise** corroboration — promotion to production blocked
- **Finding F15-5**: IS-loss + mask + α=0.1 combination produces highest peak BUT cliffs as hard as G+ (-8.75pt drop). The peak/stability trade-off is unfavorable for deployment.

### F+ v8-d5sdpo (α=0.1 PPO no mask)
- **Audit**: peak 23.00 iter 5-6, cliff iter 8-9 (final 13.38)
- **Finding F15-6**: Mask removal at α=0.1 PPO doesn't degrade peak much vs G+ (23.00 vs 22.38) but cliffs slightly earlier. Mask is helpful but not load-bearing.

### Gplus_rag (G+ + RAG retrieval, F17)
- **Audit**: peak 21.25 iter 5, cliff iter 7 (final 16.88)
- **Finding F17**: RAG retrieval ACCELERATES the cliff by ~5 iter vs G+ (G+ cliff iter 11 vs Gplus_rag iter 7). Oracle breadth via RAG is anti-correlated with audit reward — providing more grounded retrieval makes the policy cliff faster, not slower. **Major paper signal**: RAG ≠ grounding.

### mu_v7_opd_full (D5 in-house on-policy IS-loss, OPSD-style)
- **Audit**: peak ~20 iter 4, cliff iter 7
- **Pairwise**: 5-3 vs G+ (v7 narrowly wins) → G+ "v8 supremacy" headline FALSIFIED
- **Finding**: on-policy IS-loss matches v8-d5sdpo cells in peak quality but cliffs at similar iter — algorithm choice (off-policy vs on-policy) doesn't dominate stability

### L1 v9_grounded (G+ + verifier-grounded reward)
- **Setup**: sparse per-token bonus +0.1 on equation-matched spans, +0.05 on citation-matched spans (gold = 8 eqs + 13 cites)
- **Audit**: peak 22.00 iter 14, mid-pilot cliff iter 9-12 (low 16.50), recovers final 21.75
- **Multi-axis grounding at peak**: eq=5.00/8 (62%), cite=2.88/13 (22%), emp=0.00/7 (0%)
- **Finding**: verifier reward causes Goodhart specifically on equation-matching. Citation hit-rate collapses to 0.0 by iter 11 while equation density holds — the policy learns to game the eq-match signal at the cost of citation/empirical grounding. **Surface-form grounding without distribution-level anchor is brittle**.

### L2 v9_kl_anchor (L1 + KL anchor toward π_oracle SFT'd, F18)
- **Setup**: L1 reward + per-token `−β·KL(π_θ || π_oracle)` with β=0.05, π_oracle = SFT'd on 30 grounded plans (LoRA r=64, 3 epochs)
- **Audit**: peak 23.38 iter 14, **mean 20.91 highest of all RL cells**, NO cliff, final 23.12
- **Multi-axis grounding at peak**: eq=5.62/8 (70%), cite=5.75/13 (44%), emp=0.62/7 (9%) — only cell sustained on all 3 axes
- **Finding F18**: KL anchor is a regularizer, not peak booster. C+ peak (23.50) ≥ L2 peak (23.38), but L2 final iter +8.4pt over C+. Distribution-level grounding via SFT'd π_oracle prevents both (a) the late-iter cliff suffered by all v8 cells and (b) single-axis Goodhart suffered by L1.

## Cliff vs stability summary

| Cell | Peak | Final | Drop | Stable? |
|---|---:|---:|---:|:-:|
| base | 20.00 | 10.00 | -10.0 | No |
| G | 20.00 | 10.25 | -9.75 | No |
| G+ | 22.38 | 12.50 | -9.88 | No |
| E | 22.75 | 9.75 | -13.0 | No |
| C+ | 23.50 | 14.75 | -8.75 | No |
| F+ | 23.00 | 13.38 | -9.62 | No |
| Gplus_rag | 21.25 | 16.88 | -4.37 | Partial |
| mu_v7_opd_full | ~20 | ~5 | ~-15 | No |
| **L1 v9_grounded** | 22.00 | 21.75 | -0.25 | **Yes** |
| **L2 v9_kl_anchor** | **23.38** | **23.12** | **-0.26** | **Yes** |

**Pattern**: All v8 RL cells cliff. Both v9 cells stable. The verifier-grounded reward in L1 + KL anchor in L2 are the ONLY stability mechanisms tested in D5. The KL anchor is the stronger of the two (L2 mean +1.53 over L1).

## Open questions / next steps

1. **Self-review ablation** (next session): re-audit all cells with Qwen3-30B-A3B as reviewer instead of Opus. Plan in `PLAN_self_review_ablation.md`. Goal: test if cell ranking is reviewer-dependent.
2. **L2 pairwise corroboration**: L2 iter 14 vs G+ iter 4 (paper claim verification, ~$10).
3. **L2 vs C+ pairwise**: L2 iter 14 vs C+ iter 6 (peak vs peak, ~$10). Tests "stability ≠ ceiling" claim.
4. **Length-controlled audit re-run**: prompt Opus to score plans within 600-750 word range only, see if cell ranking inverts (since C+/G+ benefit from >1100 plans).
5. **Layer 3 derivation reward**: if user wants to push past surface mimicry, would need reward signal on derivation chains (Math 5 → Math 7) not just static equation match.

## Cross-era comparison (v1-v7 vs v8/v9)

### Headline numbers

| Era | Best RL cell | Peak audit | Audit protocol | Length target |
|---|---|---:|---|---|
| v1-v7 | μ-v4 iter 4 | **28.00** | audit_v3 ISOLATED (batch-anchored) | 600/750 |
| v8 | C+ iter 6 | 23.50 | M8-strict 1-plan/Opus | 900/1100 |
| v9 | L2 iter 14 | 23.38 | M8-strict 1-plan/Opus | 900/1100 |

**Naive headline**: v1-v7 peak (28.00) >> v8/v9 peak (23.50, 23.38) by ~5pt.

### Decomposing the 5pt gap

The naive 5pt gap is NOT entirely a quality regression. Three confounds:

**Confound 1 — Audit protocol (~+2-3pt for batch-anchored)**:
Per `feedback_audit_close_cluster_corroboration` memo: batch anchoring inflates strict-isolated scores by 2-3pt. v1-v7 used batch-anchored, v8/v9 used isolated. → contributes +2-3pt to the gap.

**Confound 2 — Length target (~uncertain, but plausibly +1-2pt for shorter)**:
- v1-v7 600/750 era plans avg 562-881 words
- v8/v9 900/1100 era plans avg 1004-1150 words
- v8/v9 length-bucket data shows long plans (>1100) score 1-3pt HIGHER than 900-1100 in M8-strict, suggesting Opus rewards verbosity. But under the v1-v7 600-target prompt, plans were shorter — and we don't have v1-v7 plans evaluated in 900-1100 form.
- **Direction unclear**: longer plans might score higher (verbosity bonus) but compressing into 600 also forces denser content which may score higher per-word.

**Confound 3 — Different rubric weighting / TTT subfield emphasis (uncertain)**:
v1-v7 audit_v3 was 9-dim universal+subfield rubric; v8 era added M8-strict pairwise corroboration discipline AND tightened anchor descriptions. The rubric labels are the same but the strict-anchor scoring is harsher.

**Net**: of the ~5pt gap, ~2-3pt is protocol, ~1-2pt is unclear length effect, ~0-1pt may be genuine regression. **Cannot conclude v1-v7 was actually better** without protocol re-calibration.

### What's invariant across eras

**Stability problem is in BOTH eras**:
- v1-v7: μ-v3 cliffs iter 6, μ-v4 cliffs iter 5, μ-v5 transfers cliff iter 1, μ-v7-opd cliffs iter 7
- v8: base/G/G+/E/C+/F+ all cliff iter 5-11
- v9: only L1 + L2 are stable (KL anchor + verifier-grounded reward)

The cliff is a **5-month invariant** of the D5 line, NOT an artifact of any specific protocol or prompt. It cleared only with v9's KL anchor mechanism.

**ε ceiling at 33.62 (v1-v7)**:
235B + reference-plan-in-context still topples everything. This was never beaten under any protocol. Establishes the upper bound for what 30B can hope to imitate.

### Pre-registered re-calibration plan

To rigorously test whether v9 KL anchor lifts μ-v4's 28.00 ceiling, would need either:

1. **Re-audit μ-v4 iter 4 + L2 iter 14 under M8-strict 1-plan/Opus** (~$10): direct apples-to-apples comparison at same audit protocol. Pre-registered: if μ-v4 still scores ≥ L2 under M8-strict, v9 has not lifted ceiling, paper claim must be reduced to "stability only".
2. **Re-audit μ-v4 iter 4 + L2 iter 14 under audit_v3 ISOLATED** (~$15): inverse direction — measure L2 under inflated batch-anchored protocol. Pre-registered: if L2 ≥ 28.00 under batch-anchored, v9 lifted ceiling under v1-v7's protocol.

Recommended: **Path 1** — anchor L2 to v1-v7's strongest result (μ-v4 28.00) via the new strict protocol. Cheaper, more defensible.

This is in scope for next session per `PLAN_self_review_ablation.md` Phase 4 (re-run with same calibration anchors).

## Cross-references

- Master truthful config table: `RUN_REGISTRY.md` (this doc supersedes for paper-relevant cells)
- Pre-realignment alpha/beta SFT: `RUN_REGISTRY.md` rows 39-44
- Continual transfer Phase 5 4-cell grid: `RUN_REGISTRY.md` rows 32-35 + `project_d5_phase5_continual_sdpo` memo
- v1-v7 baseline audit isolated: `runs/2026_04_26_phase2F_audit_isolated/audit_v3_isolated_summary.md`
- v1-v7 lr ablation: `project_d5_sdpo_lr_ablation` memo
- Findings: `paper_materials/findings/F15` (v8 grid), `F16` (oracle-transfer), `F17` (RAG cliff), `F18` (KL anchor)
- Joint 4-way comparison: `RESULTS_layer1_vs_layer2.md`
- ABC experiment pre-reg + verdict: `EXPERIMENT_PLAN_oracle_transfer_ABC.md` + `RESULTS_oracle_transfer_ABC.md`
