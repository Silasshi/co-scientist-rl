# Grant Proposal Generation Decisions

Append-only log. Newest first.

---

### 2026-04-22: SDPO revision RL in CR-v7

**Decision**: Add SDPO (Self-Distillation Policy Optimization, Hübotter et al. 2026) mode to `train_cr_v7.py` for scores-only revision RL. Abandon LSE-v1 fresh-gen RL.

**Context**: Full-critique revision works (B4 Opus 16/20) but model is instruction-following, not learning. Scores-only revision creates RL headroom (MAIN-B4 Δ fresh_r +0.072 vs +0.011 for full_critique). But previous RL approaches had data efficiency problems (aggregate = 1 scalar advantage per revision, per-signal HER = context mismatch).

**SDPO mechanism**: Student generates revision seeing only scores. Self-teacher (same model) re-evaluates student's revision with full critique context. Per-token advantage = log(teacher_prob / student_prob) — dense signal telling which tokens the teacher agrees/disagrees with. Plugs into existing `importance_sampling` loss by replacing constant advantages with per-token varying advantages.

**Why SDPO over alternatives**:
- vs SFT distillation: SDPO is on-policy (student generates, teacher evaluates), SFT is off-policy (imitate teacher output)
- vs aggregate REINFORCE: SDPO gives N per-token advantages vs 1 constant advantage — orders of magnitude more informative
- vs per-signal HER: zero context mismatch (student prompt = loss prompt)
- SDPO paper shows +5.8% compute overhead only (one `compute_logprobs` call per revision)

**Implementation**: 3 new config flags (`sdpo`, `sdpo_scale`, `sdpo_clip_advantage`), teacher prompt built alongside student prompt in PHASE 2, SDPO advantage computed in PHASE 3. Also added `aggregate` mode as ablation baseline. All backward-compatible.

**Risk**: Plan revision may not have token-level credit as precise as code (SDPO paper domain). C3 vs C4 comparison will test this.

---

### 2026-04-22: Abandon LSE-v1 fresh-gen RL

**Decision**: Abandon LSE-v1 fresh-gen RL direction. Fresh-gen RL on 4B showed +0.027 over frozen (noise level).

**Pivot**: Focus on revision RL instead — teach model to revise from scores-only feedback via SDPO.

---

### 2026-04-22: Fork LSE-v1 trainer from CR-v7

**Decision**: Create `train_lse_v1.py` as a separate trainer, not modify CR-v7.

**Context**: RL in CR-v7 provides zero benefit (B4 ≥ MAIN across all metrics). Pairwise DPO test showed Qwen3-30B has 100% position bias with CPR prompt, and same-model DPO = same Goodhart as REINFORCE. LSE paper (arxiv 2603.18620) showed Qwen3-4B can be RL-trained effectively with improvement-based reward and large batches.

**Key changes**: 4B policy + 30B grader (breaks self-Goodhart), n_fresh 4→16 (reduces REINFORCE variance), GRPO advantage (replaces entropic weights), no HER revision RL (eliminates distribution mismatch).

**Why not modify CR-v7**: All existing runs (8 goals × 25 iters × MAIN/B4) must remain reproducible. CR-v7 serves as the established baseline. LSE-v1 is experimental.

---

## 2026-04-21: Corrected Qwen-Opus correlation understanding

**Decision**: Qwen reward is directionally correct (cross-condition ρ=+0.908). Previous "Goodhart" framing was overstated. RL failure is template collapse, not reward misspecification.

**Evidence**: 9-plan validation spanning MAIN/B4/A_fresh best/median/worst (Qwen 0.34-0.855, Opus 4-12). The ρ=-0.16 from D3 was within-condition fine ranking (all plans at Qwen 0.78-0.82). Cross-condition, Qwen IS reliable.

**Implication**: Don't change rubrics (strict variants all WORSE). Focus on template collapse prevention (GAPO) and critique-revise architecture.

---

## 2026-04-21: Multi-prompt strict rubric validation — NEGATIVE result

**Decision**: Do NOT add strict rubric variants. Standard rubrics are already well-designed.

**Evidence**: Tested strict variants for G12 (applied-only), G11 (domain-matched), G6 (alternative-aware), G4 (composability), G13 (method-specific risks) on 9 Opus-labeled plans. ALL had worse Opus correlation than standard:
- G12: ρ 0.902 → 0.552
- G11: ρ 0.806 → 0.522
- G6: ρ 0.599 → 0.552

Root cause: strict variants floor too many plans to 1/5, destroying ranking ability.

**Files**: `analysis/validate_multi_prompt.py`, `analysis/d4v7_foundopt_ablation/multi_prompt_validation.json`

---

## 2026-04-21: GAPO diversity RL implementation

**Decision**: Implement GAPO (Group-Aware Policy Optimization) with NN-distance diversity bonus in signal space. Replace KMeans (hyperparameter k) with nearest-neighbor distance (zero hyperparameters).

**Design**: `R_adjusted = R_quality × diversity_bonus`, where diversity_bonus = nn_dist / median(nn_dist), clipped to [0.1, 2.0]. Near-duplicates (nn_dist≈0) get ×0.1, unique plans get up to ×2.0.

**Literature**: GAPO (Anschel et al., EMNLP 2025), DRA-GRPO (Chen et al., 2025), KL mode collapse fix (GX-Chen et al., 2025).

**Files**: `train_cr_v7.py` — `gapo_diversity_adjusted_rewards()` function + `diversity_method` config.

---

## 2026-04-21: Multi-goal B4 paper experiments — 6 goals

**Decision**: Run B4 (critique-revise, no RL) on 6 diverse goals with unified config. 10 iterations each (buf_max plateaus by iter 5-10).

**Goals**: 01_foundopt (AI), 05_causal_healthcare (biomedical), 07_chemo_toxicity (biomedical), 08_ecosystem_dynamics (ecology), 09_sentencing_disparities (criminal justice), 12_climate_displacement (climate policy).

**Rubric fixes**: 07_chemo_toxicity G12→G12a + G3↑0.14; 12_climate_displacement G13↑0.12 (sum fix 0.98→1.00).

**Owner**: Yuhong

---

## 2026-04-20: D4-v7 — 12 hybrid signals + 8-goal dataset plan

**Decision**: Expand from 10 to 12 signals by adding D3's S2a_formalism and S6_risk_awareness. Down-weight saturated structural signals (G1/G2/G5/G10 → 0.04 each). Build 8-goal dataset with per-goal weight configs.

**Reason**: Opus eval on D4-v6 FoundOpt run showed G6_reasoning_depth (from D3 S1) was Goodharted — model writes syntactically complex "justifications" without semantic content. However, even imperfect depth signals (G6=5) correlated weakly with quality (7/20 vs 5/20 on Opus). Missing signals (formalism, risk) were explicitly flagged by Opus as "no math content" and "no risk awareness". Saturated signals (G1/G2/G5/G10 all 5/5 from iter 0) contributed zero gradient — weight redistribution to depth signals maximizes RL learning signal.

**Dataset**: 8 goals (6 AI/ML from EPSRC outlines + 2 non-AI) with hierarchical rubric: universal signal definitions + per-domain/per-goal weight overrides.

**Files affected**: `grant_signal_reward.py` (G12/G13 added, weights redistributed), `dataset/` (new directory structure)

**Owner**: Yuhong

---

## 2026-04-20: D4-v5 signal redesign — 10 signals for full grant proposals

**Decision**: Redesign D4-v4 (8 EPSRC-specific signals) into D4-v5 (10 agency-neutral signals) based on NIH reviewer feedback from Clayton R21 pilot.

**Changes from v4**:
- G2 Objectives Clarity → **Specific Aims Clarity** (measurable aims + expected outcomes)
- G4 Named Methods → **broadened** to include non-STEM methods (legal databases, analytical frameworks)
- G7 Beneficiaries **DELETED** (EPSRC-specific)
- G8 Impact Mechanisms → **Deliverable Clarity** (UK positioning removed)
- G9 Scope Feasibility **NEW** (counting scope red flags — from reviewer critique "scope seems too broad")
- G10 Approach Coverage **NEW** (aims vs methods coverage ratio — from "unclear what applicants would do")
- G11 Preliminary Evidence **NEW** (PI's own prior work — NIH reviewers weight this heavily)

**Reason**: D4-v4 signals were EPSRC-specific (UK positioning in G8, beneficiaries in G7). NIH R21 pilot exposed this: G4 scored 1/5 on a well-funded proposal because rubric only listed STEM algorithm examples. Reviewer feedback gave concrete dimensions for 3 new signals.

**Validation**: 5 perturbations on Clayton R21 pilot. 6/7 target drops ≥1 point. One noise failure (P1→G5, grader variance ±1 across 4 runs). Reference scores ≥3 on all 10 signals.

**Files affected**: `src/co_scientist/shared/grant_signal_reward.py`, `projects/grant_proposal/knowledge/current/SIGNAL_DESIGN_NOTES.md`, `pilots/nih_r21_pediatric_genomics/perturbations/`

**Owner**: Yuhong

---

## 2026-04-19: D4-v4 signal set — 8 orthogonal outline-native signals

**Decision**: After 4 iterations (v1→v4), settled on 8 narrow signals for EPSRC AI outline evaluation. Each signal checks ONE dimension via simple counting.

**Reason**: v1 (D3-adapted, 12 signals) had D3 bias. v2 (agency-first, 6 signals) and v3 (EPSRC, 5 signals) were too coarse — grader noise too high. v4 splits into 8 orthogonal signals: problem definition (G1-G2), technical grounding (G3-G4), novelty (G5-G6), impact (G7-G8).

**Validation**: 18 funded EPSRC AI outlines. Aggregate range 0.035-0.735. Top outlines (FoundOpt 0.700, MOSAIC 0.735) correctly identified. Good discrimination between specific vs generic outlines.

**Key insight**: Outline quality = technical specificity (cited papers, named methods, concrete gaps), NOT algorithmic detail. The grader can reliably count named references and classify specific vs generic claims in 400-700 word documents.

**Files affected**: `src/co_scientist/shared/grant_signal_reward.py`, `projects/grant_proposal/knowledge/current/SIGNAL_DESIGN_NOTES.md`

**Owner**: Yuhong

---

## 2026-04-19: Focus on EPSRC AI outlines as primary dataset

**Decision**: Use 18 funded EPSRC AI Hub outlines as the primary dataset for signal design and training. Outline format (~400-700 words) matches D3 research plan format.

**Reason**: (a) Outlines ≈ research plans in length and structure. (b) All 18 are funded (high quality reference). (c) AI domain aligns with our expertise. (d) EPSRC review criteria are public.

**Alternatives considered**: NIH full proposals (too long, section extraction issues), NSF proposals (no complete publicly available ones), Open Grants mix (heterogeneous quality/format).

**Owner**: Yuhong

---

## 2026-04-19: Dataset construction — rubrics + proposals

**Decision**: Build structured dataset with rubrics from 6 agencies + proposals from Open Grants (302) + NIH samples (97 PDFs).

**Files**: `data/rubrics/` (8 files, 6 countries), `data/proposals/` (Open Grants CSV + NIH PDFs), `data/scripts/` (download scripts)

**Owner**: Yuhong

---

## 2026-04-19: Created D4 (grant_proposal) direction

**Decision**: Fork D3's CR-v7 pipeline + v9 signal system for grant proposal generation.

**Reason**: Grant proposals have explicit public rubrics (NSF merit review, NIH study section criteria), making signal design more principled. Larger revision search space. Professor + Stanford interest. Target NeurIPS 2026.

**Alternatives considered**:
- (a) Extend D3 with a grant-proposal goal — rejected: signal rubrics need fundamental redesign for grant context, not just re-weighting
- (b) Start from scratch — rejected: CR-v7 pipeline is validated and should be reused

**Files affected**:
- Created: `src/co_scientist/shared/grant_signal_reward.py` (forked from `ten_signal_reward.py`)
- Created: `src/co_scientist/grant_proposal/` (train_cr_v7.py, train_buffer_ttt.py, train_critique_revise.py, opus_eval_agent.py)
- Created: `projects/grant_proposal/` (STATUS.md, README.md, CONVENTIONS.md, DECISIONS.md, knowledge/, analysis/, data/)
- Modified: `DIRECTIONS.md`, `CLAUDE.md`, `shared/docs/STATUS.md`

**Owner**: Yuhong
