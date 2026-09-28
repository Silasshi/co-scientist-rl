# D6 Grant Proposal ERR — Decisions

## 2026-04-29 D6 project creation

**Decision**: Use D5's option-(c) HER SDPO training loop for the grant proposal task.

**Rationale**: D4's CR-v7 pipeline (critique-revise + REINFORCE) failed due to template
collapse and Goodhart on the Qwen grader. D5's μ-v4 SDPO beat its σ frozen baseline
by +2.75/45 (pairwise 20/24). Grant proposal is relatively underexplored vs. research
planning; applying D5's stronger pipeline to D4's richer multi-goal task is the natural
next step.

---

**Decision**: Oracle source = `reference_proposal.md` (not an arxiv LaTeX paper).

**Rationale**: D4 goals do not have associated source papers with LaTeX available.
The reference proposal IS the gold standard — it's what the Opus critic should be
grounded in. In D5, the source paper was privileged because it was the TRUTH the
student was trained to approximate. Here, the reference proposal plays the same role.

---

**Decision**: Score scale = D4 12-signal weighted mean, NOT D5's /45.

**Rationale**: D4 findings (B4 = 19/40 on Opus, cross-domain gap ~19 points) are all
expressed in the D4 scale. Keeping D4's scale makes D6 results directly comparable
to D4's historical best. D5's /45 is TTT-Discover-specific (9 dimensions anchored to
OpenReview reviewer standards).

---

**Decision**: Pilot on `01_foundopt` only before expanding to all 12 goals.

**Rationale**: TTT-Discover (D5) used a single goal throughout. This gave clean signal.
Expanding to 12 goals before validating the pipeline on one goal risks compound
failures that are hard to diagnose. `01_foundopt` is the most thoroughly analyzed D4
goal and has the richest D4 ablation history for comparison.

---

**Decision**: No bibliography for Phase 1 pilot.

**Rationale**: D5's bibliography was built from the source paper's citation list.
D4 reference proposals do not systematically cite their own sources. Building a
per-goal bibliography from scratch requires significant infrastructure. Phase 1
tests the core SDPO loop with oracle-only scaffolding. If oracle-only stalls (as
D5's σ stalled before μ-v4), then bibliography RAG becomes Phase 2.

---

## 2026-04-30 Prompt centralization (D6 break from D5/D4 convention)

**Decision**: Externalize every D6 production prompt to
`projects/d6_grant_proposal/prompts/*.md` and load via a new
`prompt_loader.py`. D5 and D4 keep their hardcoded `.py` prompts.

**Rationale**: D6 prompts are paper artifacts (the grant-proposal task is the
paper's headline application) and benefit from being first-class diff-able
files: the paper can show "the actual prompt" without drift. With two
sources of truth (a hardcoded Python string and a `.md` documentation copy
extracted from it), the latter inevitably skews from production.

**Loader scope**: D6-only (`src/co_scientist/d6_grant_proposal/prompt_loader.py`).
D5 and D4 are not retrofitted. If a second direction adopts the pattern,
promote to `shared/`.

**Refactor invariant**: this commit must be a no-op. The byte-equality
tests in `tests/test_d6_prompt_loader.py` re-implement every public prompt
builder inline at the pre-refactor revision and assert the new
`render_prompt`-backed builders return identical bytes.

**Out-of-scope, deliberately not externalized**: `build_single_call_prompt`
in `shared/grant_signal_reward.py` is also used by D4 and shadowing it in
D6 is a separate decision tracked in CONVENTIONS.md.

**Follow-up (separate commit)**: content correctness fixes — generation
prompt length and section structure currently mismatch the reference
distribution; review prompt's `<critique>` schema is the original 5-child
form; pairwise prompt does not yet exist. These ship in a follow-up commit
under "one variable per diff" so a pilot regression is attributable.

---

## 2026-04-30 Prompt content fixes + new pairwise prompt

**Decision**: Realign generation prompt to the empirical reference
distribution; extend the reviewer `<critique>` schema; add a pairwise
comparison prompt. Loader infrastructure unchanged from the previous
commit.

**Generation prompt — length**: was 900 words / max 1100. The 15
reference proposals (3 D6 + 12 D4) are 1318–2038 words, mean 1639,
median 1582. Policy was being asked to under-shoot the oracle by ~40%.
Raised to **target 1500 words, max 2000 words**.

**Generation prompt — section structure**: was `Problem Statement / Gap
and Motivation / Research Hypothesis / Methodology / Evaluation Plan /
Deliverables and Impact` — a structure no reference proposal uses. All
15 references use the standard NIH-style 8-heading template:

```
## Specific Aims
## Research Strategy
   ### Significance
   ### Approach
   ### Evaluation Plan
   ### Timeline
   ### Deliverables
   ### Risk Mitigation
```

Replaced. Policy is now graded against the structure it was told to produce.

**Generation prompt — `{external_knowledge}` slot**: added an optional
`{external_knowledge}` template variable for future RAG integration.
`build_student_prompt` and `build_teacher_prompt` accept it as a default-
empty kwarg; when empty the section renders the placeholder
`(none provided for this run)`.

**Reviewer `<critique>` schema**: extended from 5 children to 8 by adding
`<clarity>`, `<strengths>`, `<weaknesses>`. The teacher distillation
gradient is unchanged because only `<improvement_directive>` is read by
the teacher prompt's conditioning channel (`prompts_v1.py:62` — verified
2026-04-30). The new children enrich human-readable debug output and
supply the merit / clarity / impact dimensions named in the user spec.

**New pairwise prompt** (`prompts/pairwise/head_to_head.md`): per-signal
A | B | TIE verdict over the 12 audit dimensions plus final verdict,
margin (STRONG | MODERATE | NARROW), and ≤120-word justification. Mirrors
D5's `mu_v4_pairwise_v1.py` design but on the D4 12-signal rubric.

**Pairwise caller deferred**: a `pairwise_v1.py` analogous to D5's
`mu_v4_pairwise_v1.py` is needed to randomize A/B order, dispatch Opus
subagents, and aggregate verdicts. Out of scope for this commit; ships
when the pilot has produced enough rollouts to corroborate.

**Verification**: 24-test suite (`tests/test_d6_prompt_loader.py`)
replaces the byte-equality fixtures with structural invariants:
8 reference headings, 8 critique children, 12 active signals in
audit/pairwise, no top-level `.md` files outside role subfolders, etc.
All 24 pass.

**Pre-flight pilot decision rule**: if the V8 01_foundopt audit
aggregate at iter 5 under this commit drops by more than 0.04 vs the
pre-change baseline, treat as regression and revert. The /60 raw → 0.0–1.0
weighted-mean comparison is the slice; baseline is the most recent V8
01_foundopt iter-5 aggregate already in `runs/`.

---

## 2026-04-30 Pilot goal locked to ai/01_foundopt (revert from 02_foundational_rl drift)

**Decision**: Pilot trains on `ai/01_foundopt`. STATUS.md, CONVENTIONS.md,
and `configs/pilot_*.json` had drifted to `ai/02_foundational_rl`; reverted.

**Rationale**: The D4-v2 `grader_panel_v8` dataset (15 grant proposals × 10
signals × Opus-4.7 scores, plus holistic depth audit) is FoundOpt-only.
Training on `01_foundopt` lets Phase 1 signal validation (washing this
dataset into `dataset/signal_validation/v8_opus_foundopt/`) and Phase 3
training share a goal — the imperfect-signal narrative ("Qwen-30B grader
mean ρ=0.40 with Opus → GRPO trains against a noisy proxy") becomes
quantitative. With `02_foundational_rl` we'd be making the same claim on
faith.

**Cost**: rename 3 configs (`pilot_foundrl_*` → `pilot_foundopt_*`) and
copy `01_foundopt` goal asset from D4 (already done — this commit).

**Cross-goal evaluation** (Phase 5): `02_foundational_rl`,
`08_ecosystem_dynamics`, `12_climate_displacement` retained for
audit-only transfer test after the pilot beats σ.

**Reverts**: the 2026-04-30 prompt-content commit's pilot decision rule
("V8 02_foundational_rl audit aggregate at iter 5") — the audit run that
pre-dates this decision is no longer the baseline. New baseline:
`runs/2026_04_30_pilot_foundopt_mu_v8` iter-5 aggregate, computed on
first execution.

**Sets up**: Phase 1 of the execution roadmap
(`/home/silas/.claude/plans/federated-mixing-donut.md`).

---

## 2026-04-30 (later) Pilot goal reverted to ai/02_foundational_rl

**Decision**: Pilot trains on `ai/02_foundational_rl`. Reverts the earlier
2026-04-30 entry that locked the pilot to `01_foundopt`.

**Rationale**: Venue retargeted from NeurIPS 2026 → ICLR workshop. RL-leaning
content fits ICLR better than generic optimisation theory. `02_foundational_rl`
is on-topic for the workshop track.

**Cost — accepts a goal mismatch with signal validation**: the Phase 1
signal-validation dataset (`dataset/signal_validation/v8_opus_foundopt/`) is
**FoundOpt-only** — it was washed from D4-v2 `grader_panel_v8` which only
sampled FoundOpt proposals. There is no equivalent Opus-scored data for
`02_foundational_rl` in the repo. Two implications:
1. Phase 2 signal validation measures Qwen-30B-vs-Opus correlation on
   FoundOpt proposals, then the Phase 3 GRPO arm trains against the same
   12-signal grader on `02_foundational_rl` proposals. The implicit assumption
   is that signal correlation is goal-stable (which is plausible — the rubric
   is goal-agnostic — but unverified). The paper must flag this.
2. To close the mismatch we'd need a separate Opus-grading pass on a stratified
   set of `02_foundational_rl` proposals (~$30 in Opus calls + 1-2 hours).
   Deferred unless signal validation on FoundOpt produces an unexpected number.

**Files affected**: 9 source `.py` files, 3 JSON configs (renamed
`pilot_foundopt_*` → `pilot_foundrl_*`), STATUS.md, README.md, CONVENTIONS.md,
9 pipeline_docs, the master plan, and the signal-validation dataset README.
The `dataset/signal_validation/v8_opus_foundopt/` artifact is unchanged
(provenance is still FoundOpt; only its role label moved from "pilot-anchored"
to "cross-goal validation").

**Retains**: `dataset/ai/01_foundopt/` (research_goal.md, reference_proposal.md,
alt_goals.json, weights.json) — kept on disk in case the FoundOpt grading
pipeline is ever needed for ablation or paper appendix.

**Sets up**: Phase 1 dataset stays put; Phase 2 needs a one-paragraph cross-goal
caveat; Phase 3 trains on `02_foundational_rl`.

---

## 2026-04-30 (later) Trust-region removed from V8

**Decision**: Delete the trust-region α-blend from `train_mu_v8.py` and
make the simplified V8 (sol-mask + privileged-Opus teacher–student logprob
offset + PPO-clip) the paper recipe. No dormant flag — the shipped code
IS the recipe.

**Rationale**:
1. **Venue alignment.** ICLR workshop reviewers will read the methods
   section once. Every additional component is an additional claim that
   has to be motivated and ablated. The trust region was a "drift
   regularizer toward the frozen base policy" — explainable but not
   load-bearing for the paper's core claim ("OPD beats GRPO on grant
   proposals via privileged critique").
2. **Provenance.** The α=0.05 setting was the best cell from D5's
   TTT-Discover (research-plan) V8 grid. Whether that transfers to grant
   proposals is unverified. Carrying it forward as a default is
   hyperparameter inheritance, not a tested choice.
3. **Cross-arm symmetry.** GRPO has no analogous frozen-policy anchor.
   Keeping a one-sided regularizer in V8 muddles the V8-vs-GRPO
   comparison.
4. **Compute.** The trust region added one extra logprob inference call
   per rollout per iter (the frozen-base forward pass). Removing it ~halves
   the wall-clock cost of the logprob compute stage.

**What's removed from `train_mu_v8.py`**: the `Config.trust_region_alpha`
field, the `initial_teacher_client` setup, the frozen-base logprob
compute block, the `t_lp_eff = (1-α)·t_lp + α·f_lp` interpolation, and
the corresponding `mean_t_lp_eff` / `wall_frozen_logprob_sec` /
`trust_region_alpha` keys in `buffer.jsonl` / `metrics.jsonl`.

**What's kept**:
- Solution-only mask (still gates the gradient to the `<solution>` body)
- PPO-clip loss with ε=0.2
- Privileged-Opus reviewer + teacher–student logprob offset (the
  defining feature of OPD on this task)
- Per-token advantage clip ±5 (`opd_anchor_clip`) — that's safety, not
  drift control

**Recovery**: pre-removal code is preserved under git tag
`pre-trust-region-removal` (to be added by user at next commit). Bringing
it back if a reviewer demands the ablation = `git revert <one commit>`
plus restoring the config key.

**Cost**: ~40 lines removed from `train_mu_v8.py`, no behavioral change
to GRPO / σ / V9 trainers, doc sweep across STATUS.md / README.md /
master plan §3 + §5 + §7 / pipeline_docs / project memory / CLAUDE.md.

**Pre-flight pilot decision rule**: the σ vs V8 comparison from
`D6_master_plan.md:248-251` is unchanged (V8 ≥ σ + 0.04 over 3 seeds at
iter 25). The pre-removal V8 has not been run on D6 yet — there is no
baseline to compare against, so this is a clean slate, not a regression
risk.

**Sets up**: Phase 3 V8 pilot can use the simplified recipe directly.
The pipeline diagram doc `knowledge/PIPELINE_GRPO_vs_OPD.md` will
illustrate the simplified recipe, not the pre-removal one.

---

## 2026-04-30 (later still) D6 naming standardization (drop μ/σ/v1 conventions)

**Decision**: Replace all D6-internal legacy names with paper-aligned vocabulary.
Three principles: pipeline role over version number; plain English over Greek;
no `_v1` suffix unless a `_v2` actually exists.

**Vocabulary**:

| Concept | Old | New |
|---|---|---|
| OPD trainer (paper recipe) | μ-v8 / mu / V8 | OPD |
| Frozen baseline | σ / sigma | baseline |
| GRPO ablation | grpo_v1 | grpo |
| Oracle artifact dir | `oracle_v1/` | `oracle/` |
| Gold artifact dir (deferred) | `gold_v1/` | `gold/` |
| Signal-validation dataset dir | `v8_opus_foundopt/` | `foundopt_opus/` |
| Module suffix `_v1` (signals/prompts/audit/grounding) | dropped |

**File renames** (29 paths via `git mv`):

Source code (13):
`train_mu_v8.py → train_opd.py`,
`train_grpo_v1.py → train_grpo.py`,
`train_sigma_v1.py → train_baseline.py`,
`train_mu_v9_grounded.py → train_opd_grounded.py`,
`train_mu_v9_kl_anchor.py → train_opd_kl_anchor.py`,
`build_oracle_v1.py → extract_oracle.py`,
`build_slim_oracle_v1.py → extract_oracle_slim.py`,
`build_gold_v1.py → extract_gold.py`,
`build_signal_validation_v1.py → build_validation_dataset.py`,
`audit_v1.py → audit.py`,
`signals_v1.py → signals.py`,
`prompts_v1.py → prompts.py`,
`grounding_v1.py → grounding.py`.

Pipeline docs (10): same stems as their source files (e.g.,
`pipeline_docs/train_mu_v8.md → pipeline_docs/train_opd.md`).

Configs (3):
`pilot_foundrl_v8.json → pilot_foundrl_opd.json`,
`pilot_foundrl_grpo_v1.json → pilot_foundrl_grpo.json`,
`pilot_foundrl_v9_grounded.json → pilot_foundrl_opd_grounded.json`.
**New**: `pilot_foundrl_baseline.json` for symmetry (the baseline arm
previously had no config).

Dataset dir (1):
`dataset/signal_validation/v8_opus_foundopt/ → dataset/signal_validation/foundopt_opus/`.

Plus path-string updates in code (`oracle_v1/` → `oracle/`,
`gold_v1/` → `gold/`, `v8_opus_foundopt/` → `foundopt_opus/`) and
prose updates throughout `STATUS.md`, `README.md`, `CONVENTIONS.md`,
`knowledge/D6_master_plan.md`, `knowledge/PIPELINE_GRPO_vs_OPD.md`,
all of `pipeline_docs/`, `prompts/README.md`,
`scripts/reviewer_subagent.md`, project root `CLAUDE.md`,
`DIRECTIONS.md`, project memory file, two test files.

**Rationale**:
1. **Paper alignment.** The ICLR workshop reviewer should read "OPD beats
   GRPO over a frozen baseline" — that vocabulary should match the file
   names so a reader who clones the repo finds the trainer they expect.
   "μ-v8" and "σ" are D5/D4 internal artefacts; D6 owns its own names.
2. **No vacuous versioning.** D6 has exactly one of each component. The
   `_v1` suffix advertises "there's a v2 somewhere" but there isn't.
   Drop it. Reintroduce `_v1` / `_v2` only when a true v2 actually
   ships alongside an existing v1.
3. **Greek-letter retirement.** The σ / μ convention came from a D5
   ablation grid (mu, sigma, delta, alpha, beta, xi, epsilon). D6 has
   three arms; calling them by what they are (`baseline`, `grpo`, `opd`)
   removes a lookup step from every reader.
4. **Run-name convention.** `2026_04_30_pilot_foundrl_opd` is now the
   shape of every new run dir; the `mu_v8` / `sigma_v1` patterns are
   gone. `runs/` is empty so no historical artefacts had to be
   preserved.

**Out-of-scope (kept untouched)**:
- Python namespace `co_scientist.d6_grant_proposal` (project-wide
  convention).
- All `Config` dataclass fields, all function signatures (e.g.,
  `build_student_prompt`, `extract_solution`, `parse_critique_xml`,
  `build_sdpo_datum`).
- D5 / D4 / shared module names — D6 imports `mu_prompts_v1` from D5
  for re-exported helpers; D5 owns its own naming.
- `prompts/{generation,review,audit,pairwise,oracle}/*.md` filenames
  (already organized by role, not version).
- `knowledge/current/SIGNAL_SET_v1.md` — the `_v1` here is meaningful
  (it's a versioned signal-set artefact; if a `SIGNAL_SET_v2.md`
  ships the suffix becomes load-bearing).

**Recovery**: pre-rename code is preserved under git tag
`pre-d6-rename` (created 2026-04-30 at commit
`450c0b10`). Bringing legacy names back if needed = `git revert`
the rename commit.

**Verification**: 30/30 tests pass, all 13 D6 modules import cleanly,
post-rename grep audit returns zero residuals in D6 paths (only
legitimate D5 cross-references and historical mentions in this very
DECISIONS entry remain).

**Sets up**: future code touches D6 with the new names; the master
plan and pipeline diagram doc are the canonical references.
