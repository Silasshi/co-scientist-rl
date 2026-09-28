# D6 Grant Proposal — Master Plan

> **Status legend:** ✅ done · ⏳ in progress · ⬜ pending · ⏸ deferred · ❌ out of scope
>
> **Last updated:** 2026-04-30 (pilot-goal revert)
> **Owner:** Yuhong Shi (Oxford Physics, 2nd year)
> **Target venue:** ICLR workshop
> **Pilot goal:** `ai/02_foundational_rl`

---

## 0. About this document

This is the **single source of truth** for D6 strategy and progress. It must be updated after each completed step.

**Companion documents (do not duplicate; reference them):**

| Doc | Role | Path |
|---|---|---|
| STATUS.md | Live state — quick what-exists table | `projects/d6_grant_proposal/STATUS.md` |
| DECISIONS.md | Append-only decision log with rationales | `projects/d6_grant_proposal/DECISIONS.md` |
| CONVENTIONS.md | Naming + path rules | `projects/d6_grant_proposal/CONVENTIONS.md` |
| pipeline_docs/INDEX.md | Module-by-module pipeline map | `projects/d6_grant_proposal/pipeline_docs/INDEX.md` |
| Approved execution plan | Phase-by-phase plan (from `/plan` mode) | `~/.claude/plans/federated-mixing-donut.md` |
| Signal set v1 | 12-signal rubric definition | `projects/d6_grant_proposal/knowledge/current/SIGNAL_SET_v1.md` |

When the master plan and a companion document disagree, **this document wins**; update the companion next.

---

## 1. Project goal

> Train Qwen3-30B-A3B to generate high-quality **research grant proposals**, and demonstrate that:
> - **GRPO (reward-based RL)** is insufficient — it reward-hacks the rubric even when the rubric is well-aligned with strong-model judgments
> - **On-policy distillation (OPD) via the OPD trainer** performs better — the privileged-Opus reviewer's per-rollout feedback is more robust than a scalar reward

**Why grant proposals (vs research plans):** D4 had this task; D5 produced the OPD pipeline. D6 = D4 task + D5 pipeline. Grant proposals are richer than abstract research plans (NIH-style 8-section structure, deliverables, risk mitigation, budgets) and underexplored vs. research planning in the LLM literature.

---

## 2. Paper narrative

The 4-step story the paper must tell:

### Step 1 — Validate the signal

Show that the 12-signal rubric (`shared/grant_signal_reward.py`, evaluated by Qwen3-30B-A3B in production GRPO) correlates with strong-model (Opus 4.7) judgments **on a fixed, quality-stratified test set**.

> **Operative target:** Spearman ρ between Qwen-30B aggregate and Opus aggregate, on 15 held-out FoundOpt proposals stratified by quality.
>
> **Stated target:** ρ ≥ 0.9 (per user spec).
>
> **Methodological tension** (see §9): The strongest grader on this dataset (Qwen3-235B) only reaches ρ=0.741. Qwen3-30B currently reaches ρ=0.40. Reaching ≥0.9 with Qwen3-30B requires either (a) signal redesign, (b) a stronger grader, or (c) lowering the target. **This is an open methodology decision before Step 2 can begin.**

### Step 2 — GRPO failure case

Train the policy with GRPO using the validated signal as reward. Show that:
- Reward goes up over iterations
- Audit-grader (Opus) score does **not** go up — or goes down
- Goodhart traces visible: template collapse, rubric-feature inflation, etc.

The headline finding: **scalar-reward RL reward-hacks even a well-aligned rubric.**

### Step 3 — OPD (OPD) main contribution

Train the policy with the OPD SDPO loop. Three components only:
- Each rollout sent to Opus 4.7 reviewer with `reference_proposal.md` as **privileged** context (policy never sees this)
- Reviewer emits structured `<critique>` per rollout
- Policy updates on teacher–student logprob delta with: solution-only mask, advantage clip ±5, PPO-clip ε=0.2

(Trust-region α-blending toward the frozen base policy was removed 2026-04-30; see DECISIONS.md.)

Show: **Opus audit score rises**, no template collapse, robust per-signal lift.

### Step 4 — Generalisation (later)

After Steps 1–3 succeed on `ai/02_foundational_rl`, transfer the trained policy to `natural_science/08_ecosystem_dynamics` and `social_science/12_climate_displacement` (audit-only, no retraining). Tests whether OPD's gains hold cross-domain.

---

## 3. Scope (hard constraints)

### ✅ IN scope (paper arms + supporting infrastructure)

| Component | Module | Stage |
|---|---|---|
| Frozen-model lower bound (σ measurement, **not** a training pipeline) | `train_baseline.py` | sanity baseline |
| GRPO failure-case arm | `train_grpo.py` | paper arm 2 |
| OPD main arm (simplified: sol-mask + privileged-Opus offset + PPO-clip) | `train_opd.py` | paper arm 3 |
| 12-signal grader | `shared/grant_signal_reward.py` (re-exported via `signals.py`) | reward + audit support |
| Oracle pipeline (Extract → Slim) | `extract_oracle.py`, `extract_oracle_slim.py` | training input |
| Audit (Opus 4.7, isolated batch) | `audit.py` | evaluation |
| Signal-validation dataset + analysis | `build_validation_dataset.py` (✅), `validate_signals.py` (⬜) | Step 1 evidence |
| Pairwise close-cluster corroboration | `pairwise.py` (⬜, deferred) | M8 protocol |
| Prompts | `projects/d6_grant_proposal/prompts/*.md` via `prompt_loader.py` | all stages |

### ❌ OUT of scope (retained on disk, NOT executed)

| Component | Rationale |
|---|---|
| `train_opd_grounded.py` (OPD-grounded) | Grounding bonus deferred; risks Goodhart cliff (D5 F18) |
| `train_opd_kl_anchor.py` (OPD-KL-anchor) | Scaffold only; needs oracle SFT that doesn't exist |
| `extract_gold.py`, `grounding.py` | OPD-grounded dependencies; not used by OPD or GRPO |
| All D5 `train_mu_v*.py` other than the SDPO recipe | D5 is research plans, not grant proposals |
| All D4 `train_*.py` (CR-v7, IBT, etc.) | Replaced by D6 OPD pipeline |
| RAG retrieval | Grant-domain corpus undefined; deferred to future work |

**Disposition:** these files stay on disk so git history is preserved and they're available if Steps 1–4 fail and we need to revisit. They are NOT on the active execution path. Pipeline docs flag them as deferred.

---

## 4. Conceptual structure ↔ actual paths

The user's proposed conceptual structure (`training/grpo/`, `training/opd_v8/`, etc.) is a useful **reading map** but does not match the on-disk layout. The actual layout follows the project-wide `co_scientist.<direction>` Python import convention and was just consolidated in Phase 0. Mapping:

| Conceptual bucket | Actual paths |
|---|---|
| `training/grpo/` | `src/co_scientist/d6_grant_proposal/train_grpo.py` + `projects/d6_grant_proposal/configs/pilot_foundrl_grpo.json` |
| `training/opd_v8/` | `src/co_scientist/d6_grant_proposal/train_opd.py` + `projects/d6_grant_proposal/configs/pilot_foundrl_v8.json` |
| `data/` | `projects/d6_grant_proposal/dataset/` (inputs) + `projects/d6_grant_proposal/data/` (oracle outputs) |
| `prompts/` | `projects/d6_grant_proposal/prompts/{generation,review,audit,pairwise,oracle}/*.md` |
| `evaluation/` | `src/co_scientist/d6_grant_proposal/{audit.py, validate_signals.py (⬜), pairwise.py (⬜)}` + `projects/d6_grant_proposal/dataset/signal_validation/` |
| `pipeline_docs/` | `projects/d6_grant_proposal/pipeline_docs/INDEX.md + 13 module docs` (✅) |
| `knowledge/` | `projects/d6_grant_proposal/knowledge/D6_master_plan.md` (this file) + `current/SIGNAL_SET_v1.md` |

Why split between `src/` and `projects/`: the codebase convention is one Python namespace per direction (`co_scientist.d6_grant_proposal`) and one project root per direction (`projects/d6_grant_proposal/`). All directions follow this. Reorganising D6 into a flat `D6_grant_proposal/` would break the import machinery and diverge from D1–D5.

---

## 5. Step-by-step execution plan

### Step 0 — Pipeline consolidation ✅ (2026-04-30)

**Objective:** Clean codebase + documentation aligned to a single pilot goal.

**Actions:**
- Mirrored `ai/01_foundopt` goal asset into D6 dataset (research_goal.md, reference_proposal.md, alt_goals.json, weights.json) — kept on disk after revert
- Deleted `prompts/legacy/` (4 D4-era archives)
- Initially repointed STATUS.md / README.md / DECISIONS.md / CONVENTIONS.md / 3 JSON pilot configs from `02_foundational_rl` → `01_foundopt`, **then reverted to `02_foundational_rl`** (later 2026-04-30 — venue retargeted to ICLR workshop). See DECISIONS.md "Pilot goal reverted" entry.
- Renamed configs: `pilot_foundopt_*` → `pilot_foundrl_*` (final state)
- Wrote `pipeline_docs/INDEX.md` + 13 module docs
- Updated 2 prompt-loader tests; full suite 24/24 passing
- Aligned all `Config` dataclass defaults + docstring usage examples

**Output:** Cleaned codebase + `pipeline_docs/`. See `STATUS.md` for the artifact table.

**Success criteria (all met, post-revert):**
- All 13 D6 modules import cleanly
- `Config().goal_name == "02_foundational_rl"` for all trainers
- 24/24 prompt-loader tests pass
- 3 pilot configs `pilot_foundrl_*.json` resolve `goal_name=02_foundational_rl`

---

### Step 1 — Signal-validation dataset ✅ (2026-04-30)

**Objective:** Materialise a standalone D6 dataset of Opus-scored grant proposals so signal correlation can be measured without depending on D4-v2 paths.

**Actions:**
- Wrote `src/co_scientist/d6_grant_proposal/build_validation_dataset.py`
- Read 15 stratified FoundOpt proposals + Opus 4.7 grades from `projects/grant_proposal_v2/analysis/grader_panel_v8/`
- Emitted to `projects/d6_grant_proposal/dataset/signal_validation/foundopt_opus/`

**Output:**
```
foundopt_opus/
├── README.md                       # dataset card with cross-grader benchmarks
├── proposals/plan_{00..14}.md      # frontmatter + body
├── opus_grades.jsonl               # 180 rows (15 plans × 12 signals; G3+G5 null)
├── opus_depth.jsonl                # 15 holistic-audit rows (rubric-agnostic)
├── metadata.jsonl                  # 15 per-plan summary rows
└── panel_baselines/                # 8 cross-grader files (Qwen-235B/30B/4B, GPT-OSS-120B/20B, DeepSeek, Llama, Kimi)
```

**Success criteria (all met):**
- 180 grade rows with 30 G3/G5 nulls + 150 scored
- plan_00 body byte-equal to source
- Reference anchors plan_00/01/02 are identical
- Quality buckets track Opus aggregate (ref 4.20 / high 4.08 / mid 3.02 / low 2.60) ✓

**Sanity check:** Qwen-30B (the production GRPO grader) panel ρ vs Opus = 0.40 — already in the panel-baselines file. Step 2 will re-measure under the production GRPO scoring path.

---

### Step 2 — Signal-validation analysis ⏳ (next)

**Objective:** Quantify the production GRPO grader's correlation with Opus on the Step 1 dataset.

**Actions:**
- Write `src/co_scientist/d6_grant_proposal/validate_signals.py`
  - Loads 15 proposals from Step 1 dataset
  - Calls `shared/grant_signal_reward.{build_grader_prompt, build_single_call_prompt, parse_scores, weighted_aggregate}` end-to-end via the same Tinker code path that `train_grpo.py` uses (NOT a panel-style helper)
  - Outputs per-signal Spearman ρ vs Opus, plan-level pooled ρ, MAE, within-1 agreement, aggregate ρ
- Write `projects/d6_grant_proposal/analysis/signal_validation/qwen30b_vs_opus.ipynb` to visualize

**Output:**
- `projects/d6_grant_proposal/analysis/signal_validation/qwen30b_vs_opus.jsonl` (per-signal correlations)
- `projects/d6_grant_proposal/analysis/signal_validation/qwen30b_vs_opus.ipynb` (analysis notebook)
- One paragraph for the paper: "Qwen3-30B-A3B aggregate correlates with Opus 4.7 at ρ = ___ (n=15, Spearman). [Discussion of methodological gap.]"

**Success criteria — pre-registered (one of two paths must be chosen before this step starts):**

| Path | Aggregate ρ outcome | Action |
|---|---|---|
| **A. Imperfect-signal narrative** | ≥ 0.4 (matches panel result; Qwen-30B is the actual GRPO grader) | Proceed to Step 3 GRPO; paper says "GRPO reward-hacks even a moderately-aligned rubric" |
| **B. Strong-signal narrative (user spec)** | ≥ 0.9 | Requires switching grader / redesigning signals before Step 3 (open question §9) |

**Notes:**
- Reproduce the v8 panel number (Qwen-30B mean ρ ≈ 0.40) within ±0.05 as a sanity gate. Failure here means the production scoring path differs from the panel script — debug before reading any new ρ as a result.
- Spearman SE ≈ 0.22 at n=15. Differences <0.05 are noise.

---

### Step 3a — GRPO training ⬜

**Objective:** Establish the failure-case arm.

**Actions:**
- Run `train_grpo.py goal_domain=ai goal_name=02_foundational_rl log_path=runs/2026_05_*_pilot_foundrl_grpo_seed{0,1,2}` × 3 seeds
- 25 iterations per seed
- Audit at iter {0, 5, 10, 15, 20, 25} via `audit.py`

**Output:**
- 3× `runs/.../{buffer.jsonl, metrics.jsonl, checkpoints.jsonl}`
- 18 audit checkpoints (3 seeds × 6 iters) under `audit_responses/`
- Per-seed audit aggregate trajectory plot

**Success criteria:**
- All 3 seeds complete 25 iterations without crashes
- Reward (Qwen-30B aggregate) trajectory recorded per seed
- Audit (Opus) trajectory recorded per seed

**Predicted result (pre-registered hypothesis):** Reward goes up; Opus audit aggregate stays flat or declines after iter 5–10 (Goodhart). Per-signal breakdown shows specific signals inflating while others crash.

---

### Step 3b — OPD training ⬜ (concurrent with Step 3a)

**Objective:** Establish the main contribution arm.

**Actions:**
- Run `train_opd.py goal_domain=ai goal_name=02_foundational_rl log_path=runs/2026_05_*_pilot_foundrl_opd_seed{0,1,2}` × 3 seeds
- Same 25 iterations, same audit checkpoints as Step 3a
- Reviewer = Opus 4.7 via subagent file-bus, sees `reference_proposal.md` (privileged)

**Output:** Same structure as Step 3a, plus `critic_requests/` + `critic_responses/` per run.

**Success criteria — pre-registered:**
- All 3 seeds complete 25 iterations
- Audit aggregate at iter 25 ≥ baseline + 0.04, averaged over 3 seeds
- Audit aggregate at iter 25 ≥ GRPO + 0.04 (head-to-head margin)
- M8 close-cluster pairwise corroboration if Δ < 0.05

**Predicted result:** Opus audit aggregate rises 5–15 iters in then plateaus or continues to climb; per-signal breakdown shows balanced lift (no axis collapse).

---

### Step 3c — Audit + pairwise comparison ⬜

**Objective:** Settle OPD vs GRPO vs baseline in one auditable artifact.

**Actions:**
- Final audit batch: frozen baseline samples × 3, GRPO iter-25 × 3 seeds, OPD iter-25 × 3 seeds — all anonymized into `audit.py` with 8 parallel Opus subagents
- If Δ < 0.05/1.0 between OPD and GRPO → run `pairwise.py` (✅ prompt at `prompts/pairwise/head_to_head.md`; caller currently deferred — write before this step)
- Aggregate per-arm 12-signal scores

**Output:**
- `projects/d6_grant_proposal/analysis/pilot_comparison.md` — one-page table + per-signal breakdown + reproducibility notes
- Pre-registered metric file checked into git (the decision rule, locked before Step 3 started)

**Success criteria:** Decision rule is auditable from the file; the comparison table shows the expected ordering or doesn't (and the paper writes whichever happened).

---

### Step 4 — Cross-goal generalisation ⏸ (post-pilot)

**Objective:** Test whether the OPD-trained policy transfers to non-AI domains.

**Actions (only after Step 3c shows OPD ≫ GRPO ≫ baseline):**
- Audit the OPD-trained policy on `natural_science/08_ecosystem_dynamics` and `social_science/12_climate_displacement` — **no retraining**
- Compare against frozen-model baseline on the same goals

**Output:** A 2-row addition to `pilot_comparison.md` with cross-goal audit numbers.

**Success criteria:** OPD ≥ baseline + 0.02 on at least one transfer goal → "transfers cross-domain" claim. Otherwise → flag as future work and keep paper single-goal.

---

## 5b. OPD simplification (2026-04-30)

The trainer recipe shipped for the paper has **three components** — sol-mask, privileged-Opus teacher-student logprob offset, PPO-clip. The trust-region α=0.05 interpolation toward the frozen base policy that came from D5's TTT-Discover ablation grid was removed: it was not validated on grant proposals, and an extra "regularizer toward base policy" line in the methods section adds reviewer surface area without a paper claim attached. See `DECISIONS.md` ("2026-04-30 Trust-region removed from OPD trainer") for full rationale. Git history preserves the previous code path under tag `pre-trust-region-removal`. Pipeline diagrams: `knowledge/PIPELINE_GRPO_vs_OPD.md`.

## 6. RL-first strategy

All training and audits happen on `ai/02_foundational_rl` until Step 3c declares success. Signal validation (Step 1/2) runs on FoundOpt proposals because that's where the Opus-scored ground-truth data exists — see §9.5 for the resulting cross-goal caveat. Cross-domain transfer (`natural_science`, `social_science`) is **explicitly deferred** to Step 4. Reasoning:
- Single-goal results give clean signal; cross-goal noise compounds
- D5's OPD single-goal validation succeeded; multi-goal compound failures from D4 history motivated the SDPO pivot
- The signal-validation dataset is FoundOpt-only — measuring foundational_rl-specific signal correlation requires a separate Opus-grading pass (~$30; deferred unless §9.5's deferred-validation assumption breaks)

---

## 7. Minimal experiment list (paper budget)

| Arm | Trainer | Reviewer | Reward source | Seeds | n_iter | Audit | Cost rank |
|---|---|---|---|---|---|---|---|
| **baseline** | `train_baseline` | — | none (frozen sample) | 1 | 0 | 1× post-hoc | low |
| **GRPO** | `train_grpo` | — | Qwen-30B 12-signal weighted mean | 3 | 25 | 6× per seed | medium |
| **OPD** | `train_opd` | Opus 4.7 (privileged) | teacher–student logprob delta + sol-mask + PPO-clip | 3 | 25 | 6× per seed | high |

**Total runs:** 1 baseline + 3 GRPO + 3 OPD = 7 training runs, 36 audit batches, 1 pairwise tournament if needed.

**Out-of-budget:** baseline × 3 seeds (deterministic enough that 1 is fine), OPD-grounded & OPD-KL-anchor (out of scope), per-iter Opus audits (only 6 audit checkpoints — not every iter), cross-goal training (audit-only).

---

## 8. Status board

| Step | Status | Date | Owner | Artifact |
|---|---|---|---|---|
| Step 0 — Pipeline consolidation | ✅ | 2026-04-30 | Yuhong | `pipeline_docs/`, cleaned configs |
| Step 1 — Signal-validation dataset | ✅ | 2026-04-30 | Yuhong | `dataset/signal_validation/foundopt_opus/` |
| Step 2 — Signal-validation analysis | ⏳ | — | — | `validate_signals.py` + notebook |
| Step 3a — GRPO training | ⬜ | — | — | `runs/.../grpo_seed*/` × 3 |
| Step 3b — OPD training | ⬜ | — | — | `runs/.../opd_seed*/` × 3 |
| Step 3c — Audit + pairwise | ⬜ | — | — | `analysis/pilot_comparison.md` |
| Step 4 — Cross-goal transfer | ⏸ | (post-pilot) | — | — |

---

## 9. Open methodological questions (resolve before next step)

### 9.1 ρ ≥ 0.9 target vs imperfect-signal narrative

**The tension:** Step 1's stated success criterion is Spearman ρ ≥ 0.9 between the production GRPO grader (Qwen3-30B-A3B) and Opus 4.7. Step 2's pre-registered prediction is ρ ≈ 0.40 (panel benchmark). These are 5× apart.

**Why it matters:** The paper narrative works either way, but they're different narratives:

- **High-ρ narrative ("strong signal, reward-hacking"):** "We have a rubric that correlates with Opus at ρ ≥ 0.9 on i.i.d. data. Yet GRPO reward-hacks it under adversarial policy gradient." — Cleaner Goodhart story; needs ρ ≥ 0.9 first.
- **Low-ρ narrative ("imperfect signal, ceiling"):** "Even setting aside Goodhart, the rubric correlates only at ρ=0.40. GRPO trains against a noisy proxy and plateaus; OPD doesn't depend on the rubric." — Doesn't require ρ ≥ 0.9; aligns with the existing panel data.

**Three forks:**
1. **Lower the target to ρ ≥ 0.7** (achievable with Qwen3-235B at ~6× cost). Paper claim becomes: "moderate-strength signal still reward-hacks." Mid-cost, mid-narrative.
2. **Switch GRPO grader to Qwen3-235B.** Cost-feasible if the paper is willing to absorb 6× grader cost (~$1.70/run vs $0.30/run per FINAL_SUMMARY). Decision: keep narrative simple.
3. **Keep Qwen-30B at ρ=0.40 and reframe to imperfect-signal narrative.** Lowest cost; matches existing infrastructure; weaker Goodhart claim.

**Recommendation (ML research scientist persona):** Fork 1 or Fork 3. Fork 2's cost profile undermines the "we use a cheap grader" framing that motivates 30B in the first place. Fork 1 (235B grader, ρ ≥ 0.7) is the cleanest if compute allows. Fork 3 is the safe fallback if compute is tight. **Decision needed before Step 2 starts** — this changes which signal we measure.

### 9.2 frozen baseline status

Is baseline a paper arm or just a sanity check? If sanity, n=1 seed is fine. If paper arm with statistical claims, need 3 seeds (small but measurable variance from sampling temperature).

**Recommendation:** baseline is a sanity baseline, n=1 sufficient. The paper claim is OPD vs GRPO; baseline ≪ both is the lower-bound check, not a comparand.

### 9.3 Pairwise caller deferral

`pairwise.py` is currently a missing module (only the prompt exists at `prompts/pairwise/head_to_head.md`). Required by M8 close-cluster protocol if Step 3c yields Δ < 0.05. **Write before Step 3c, after Step 3a/b produce data.**

### 9.4 Number of seeds (3 vs 5)

3 seeds gives a SE of ~0.03 on the audit aggregate at the noise floor — exactly at the decision threshold. 5 would tighten this to ~0.02. Compute trade: 5 seeds doubles training cost from 6 runs to 10. **Recommendation: stick with 3. If Step 3c is a close call, add 2 more seeds before re-deciding.**

### 9.5 Cross-goal signal-validation mismatch (NEW after pilot-goal revert)

**The mismatch:** Pilot trains on `ai/02_foundational_rl`. The signal-validation dataset (`foundopt_opus/`) is `ai/01_foundopt`. The Opus-scored panel data only exists for FoundOpt — D4-v2's `grader_panel_v8` sampled FoundOpt proposals exclusively.

**Implication:** Step 2's ρ measurement is for *FoundOpt proposals graded by the GRPO grader vs Opus*. The paper claim "the GRPO reward signal is imperfect" generalises to `02_foundational_rl` only under the assumption that the rubric's correlation profile is goal-stable. This is plausible (the rubric is goal-agnostic; signals like "Mathematical Formalism" or "Risk Awareness" are not optimisation-specific) but **unverified for foundational_rl**.

**Three forks:**
1. **Accept the assumption, flag in paper.** Lowest-cost path. Paper text adds: "We measured grader-vs-Opus correlation on FoundOpt; we assume the rubric's calibration generalises to RL convergence proposals because [rubric is goal-agnostic, signal definitions don't depend on the goal text, etc.]."
2. **Collect a foundational_rl signal-validation dataset.** Sample 15 stratified `02_foundational_rl` proposals from existing D4 runs; grade with Opus 4.7 on 10 v8 signals. Cost ≈ 150 Opus calls (~$30) + 1–2 hours dispatch + analysis. Cleanest narrative.
3. **Re-grade the FoundOpt proposals AND collect new foundational_rl proposals.** Compare ρ across goals. Most expensive but produces "rubric calibration is goal-stable at ρ_diff < 0.1" as a paper claim.

**Recommendation (ML research scientist persona):** Fork 2 if reviewers will likely push on this; Fork 1 if not. The cost of Fork 2 is dwarfed by the training-run budget; running it first hardens the §9.1 ρ-target decision against goal drift. Decision needed before Step 2 finalises.

---

## 10. Out-of-scope code (retained, deprecated)

The following files exist on disk but are explicitly NOT on the active execution path:

| File | What it does | Why deprecated for this paper |
|---|---|---|
| `train_opd_grounded.py` | OPD + per-token citation/equation grounding bonus | D5 finding F18: cliffs mid-training (axis collapse). Compute-positive only after OPD confirmed. |
| `train_opd_kl_anchor.py` | OPD-grounded + KL anchor toward oracle SFT (NotImplementedError) | Blocked on oracle SFT that doesn't exist |
| `extract_gold.py` | Extracts equations + citations from reference (OPD-grounded input) | Not used by OPD / GRPO / baseline |
| `grounding.py` | Equation Jaccard + citation regex matcher (OPD-grounded mechanic) | Not used by OPD / GRPO / baseline |

**Disposition:** retained for reproducibility and possible future ablation. Pipeline docs (`pipeline_docs/train_opd_grounded.md`, `pipeline_docs/train_opd_kl_anchor.md`, `pipeline_docs/extract_gold.md`, `pipeline_docs/grounding.md`) flag them as DEFERRED. Never delete without git-history justification.

---

## Update protocol

When a step transitions:
1. Update the row in §8 **Status board** (status, date, owner, artifact path)
2. Add a one-line summary under the relevant Step section in §5
3. If a new methodological question opens, add it to §9; if one resolves, move it to a "Resolved" sub-section at the bottom of §9
4. Don't duplicate STATUS.md / DECISIONS.md content — point to them
5. Bump the "Last updated" line at the top
