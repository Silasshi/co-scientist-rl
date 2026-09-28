# D6: Grant Proposal ERR

**Task**: Generate high-quality grant proposals across 12 cross-domain research goals.

**Pipeline**: Extract-Retrieve-Refine (ERR) via option-(c) HER SDPO, adapted from D5.

**Key difference from D4**: Replaces the CR-v7 critique-revise pipeline with D5's SDPO-based
training loop. The Opus critic receives `reference_proposal.md` as privileged context
(replacing D5's LaTeX source paper). No RL gradient — pure SDPO distillation.

## Dataset

12 research goals across 7 domains:

| Domain | Goals |
|---|---|
| AI/ML | 01_foundopt, 02_foundational_rl, 03_probabilistic_ai, 04_trustworthy_ai, 06_neurosymbolic |
| Biomedical | 05_causal_healthcare, 07_chemo_toxicity |
| Ecology | 08_ecosystem_dynamics |
| Criminal Justice | 09_sentencing_disparities |
| Education | 10_teacher_effectiveness |
| Social Science / Policy | 11_housing_first, 12_climate_displacement |

Each goal has:
- `research_goal.md`: Problem statement (~200-300 words)
- `reference_proposal.md`: Gold-standard reference (~750 words)
- `alt_goals.json`: Alternative goals for hard-gate contrast
- `weights.json`: Per-goal signal weight overrides

## Evaluation

**12-signal rubric** (`shared/grant_signal_reward.py`):

| Signal | Name | Weight |
|---|---|---|
| G1 | Problem Specificity | 0.04 |
| G2 | Specific Aims | 0.04 |
| G3 | Technical Evidence | 0.10 |
| G4 | Research Focus | 0.12 |
| G5 | Gap Identification | 0.04 |
| G6 | Reasoning Depth | 0.12 |
| G8 | Deliverable Clarity | 0.08 |
| G9 | Scope Feasibility | 0.08 |
| G10 | Approach Coverage | 0.04 |
| G11 | Evidence Rigor | 0.12 |
| G12 | Mathematical Formalism | 0.12 |
| G13 | Risk Awareness | 0.10 |

Domain overrides: G12 → G12a (Analytical Framework) for non-STEM domains.

**Aggregate**: weighted mean of normalized scores (1–5 → 0–1 per signal).

## Training

On-policy distillation (OPD) via SDPO option-(c) HER (same family as D5
`train_mu_v4.py`). The simplified ICLR-workshop recipe has three components:

- **STUDENT context**: `goal + slim_oracle` — the only context the policy ever sees
- **TEACHER context**: `goal + slim_oracle + critique` — used only to compute the per-token logprob target
- **Privileged-Opus reviewer**: Opus 4.7 sees the student rollout PLUS `reference_proposal.md` as privileged ground truth and emits a structured `<critique>`
- **Advantage**: `clamp(teacher_lp - student_lp, ±5)` per token
- **Solution-only mask**: zeros the advantage outside the last `<solution>...</solution>` block
- **Loss**: PPO-clip with ε=0.2

Trust-region α-blending toward the frozen base policy was removed 2026-04-30 — see DECISIONS.md.

Pilot: `ai/02_foundational_rl`, OPD trainer, lr=5e-5. See `STATUS.md` for the canonical run sequence.

## Key Commands

```bash
# Build oracle for the pilot goal
PYTHONPATH=src python -m co_scientist.d6_grant_proposal.extract_oracle \
    goal_domain=ai goal_name=02_foundational_rl

# Build slim oracle
PYTHONPATH=src python -m co_scientist.d6_grant_proposal.extract_oracle_slim \
    goal_domain=ai goal_name=02_foundational_rl

# Run OPD pilot (main contribution arm)
PYTHONPATH=src python -m co_scientist.d6_grant_proposal.train_opd \
    goal_domain=ai goal_name=02_foundational_rl \
    log_path=projects/d6_grant_proposal/runs/2026_04_30_pilot_foundrl_opd

# Run GRPO ablation (concurrent with OPD)
PYTHONPATH=src python -m co_scientist.d6_grant_proposal.train_grpo \
    goal_domain=ai goal_name=02_foundational_rl \
    log_path=projects/d6_grant_proposal/runs/2026_04_30_pilot_foundrl_grpo

# Run frozen baseline baseline
PYTHONPATH=src python -m co_scientist.d6_grant_proposal.train_baseline \
    goal_domain=ai goal_name=02_foundational_rl

# Audit a checkpoint
PYTHONPATH=src python -m co_scientist.d6_grant_proposal.audit \
    goal_domain=ai goal_name=02_foundational_rl
```

## Relationship to Other Directions

- **D4**: Same task + dataset; CR-v7 pipeline (abandoned due to Goodhart/template collapse)
- **D5**: Source of ERR pipeline + SDPO trainer; single research-plan goal (TTT-Discover)
- **D6**: D4 task + D5 pipeline. Pilot on `02_foundational_rl` (ICLR workshop venue); signal-validation dataset uses FoundOpt proposals — see `knowledge/D6_master_plan.md` §9.5 for the cross-goal caveat.
