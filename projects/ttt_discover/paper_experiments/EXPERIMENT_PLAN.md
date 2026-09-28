# Paper Experiment Plan (v2)

> **Note 2026-04-15**: Signals renamed in v6 — `S5_stability` → `S5_feasibility`, `S6_failure_interp` → `S6_risk_awareness`. Rubrics now aligned with 6-criteria framework (see `shared/docs/RESEARCH_PLAN_TEMPLATE.md`). A2/A3 disabled_signals should use new IDs.


**Date**: 2026-04-13
**Model**: Qwen3-30B-A3B (policy = grader = same model, 3B active MoE)
**Method**: CR-v5 paragraph-level editing + raw delta advantage
**Base trainer**: `src/co_scientist/ttt_discover/train_critique_revise.py`
**Signal set**: v5 (9 gradient signals + 2 hard gates, separate_call mode)
**Iterations**: 50 per run
**Grader mode**: separate_call, N=2 repeats
**Runs folder**: `projects/ttt_discover/runs/2026_04_30b_paper_experiments/`

---

## Shared Config (all runs unless overridden)

```python
model_name = "Qwen/Qwen3-30B-A3B"
grader_model_name = "Qwen/Qwen3-30B-A3B"   # policy = grader = same

n_iterations = 50
grader_repeats = 2

# Default (MAIN)
n_fresh = 4
n_revise = 4
n_revision_candidates = 2        # Best-of-2
paragraph_level_edit = True       # CR-v5 paragraph-level
use_ucb = True
ucb_c = 1.0
cold_start_iters = 1
lora_rank = 32
save_every = 5

# Signals: v5 (all 9 active)
# S1_depth(0.10), S2_rigor(0.12), S3_positioning(0.10), S4_significance(0.13),
# S5_stability(0.08), S6_failure_interp(0.11), S7_specificity(0.08),
# S8_scope(0.15), S9_focus(0.13)

# RL
learning_rate = 4e-5
delta_scale = 5.0
skip_rl_update = False
skip_hard_gates = False
disabled_signals = ""
train_on_fresh = False
```

---

## Experiment Matrix

### MAIN — Full Pipeline

| ID | Config | Purpose |
|---|---|---|
| **MAIN** | All defaults (paragraph edit, UCB, Best-of-2, skip-signal, hard gates, all 9 signals) | **The proposed method** |

### 4 Baselines

| ID | Name | Config overrides | Purpose | Priority |
|---|---|---|---|---|
| **B1** | Zero-shot | `n_iterations=1, n_fresh=50, n_revise=0, skip_rl_update=True` | Lower bound: base model raw output | P1 |
| **B2** | Best-of-N | `n_iterations=1, n_fresh=50, n_revise=0, skip_rl_update=True` (report max) | Sampling ceiling: best of 50 random plans | P1 |
| **B3** | Buffer-TTT (entropic) | `n_revise=0, n_fresh=8, train_on_fresh=True` | Old method baseline: undirected RL | P1 |
| **B4** | No training | `skip_rl_update=True` | In-context learning ceiling: buffer conditioning only | **P0** |

### 9 Ablations

| ID | Name | Config overrides | Tests what | Priority |
|---|---|---|---|---|
| **A1** | No hard gates | `skip_hard_gates=True` | Are hard gates (Layer 0) necessary? | P2 |
| **A2** | No S9_focus | `disabled_signals="S9_focus"` | Is anti-Goodhart signal necessary? | **P1** |
| **A3** | No S1_depth | `disabled_signals="S1_depth"` | Does reasoning depth matter? | P2 |
| **A4** | Best-of-1 | `n_revision_candidates=1` | Does Best-of-2 improve revision? | P2 |
| **A5** | No skip-signal | `min_revision_attempts=999` | Does signal cascade matter? | P2 |
| **A6** | No UCB | `use_ucb=False` | Does UCB exploration help? | P2 |
| **A8** | No revision | `n_revise=0, n_fresh=8, train_on_fresh=True` | Is revision mechanism necessary? | **P1** |
| **A9** | Diversity seeding | `diversity_seeding=True` | Does approach seeding + negative conditioning improve exploration? | P1 |

---

## Key Hypotheses

| Hypothesis | Compared runs | Expected |
|---|---|---|
| Training helps beyond in-context learning | MAIN vs B4 | MAIN > B4 (key gap) |
| Critique-revise > undirected RL | MAIN vs B3 | MAIN >> B3 |
| Self-iteration > pure sampling | MAIN vs B2 | MAIN > B2 |
| Hard gates prevent reward hacking | MAIN vs A1 | MAIN >= A1 |
| Anti-Goodhart signal is necessary | MAIN vs A2 | MAIN has better plan quality despite possibly lower auto score |
| Reasoning depth matters | MAIN vs A3 | MAIN > A3 on human/LLM eval |
| Best-of-2 improves revision success | MAIN vs A4 | MAIN > A4 on revision rate |
| Signal cascade enables adaptive targeting | MAIN vs A5 | MAIN > A5 |
| UCB > top-K for buffer selection | MAIN vs A6 | MAIN >= A6 |
| Structured prompt contributes | MAIN vs A7 | MAIN > A7 |
| Revision mechanism is core innovation | MAIN vs A8 | MAIN >> A8 |
| Diversity seeding improves exploration | MAIN vs A9 | A9 >= MAIN (higher buffer_max via broader exploration) |

---

## Run Order (parallelization)

**Phase 1** (P0 + P1 — most critical):
- MAIN, B4, A2, A8 — can run in parallel (4 runs)

**Phase 2** (P1 baselines):
- B1, B2, B3 — B1/B2 are 1-iteration (fast), B3 needs entropic training

**Phase 3** (P2 ablations):
- A1, A3, A4, A5, A6, A7 — all single-flag changes, can parallelize

**Phase 4** (after Phase 1 MAIN completes, for fair comparison):
- A9 — diversity seeding (must use same pipeline as MAIN, only adds diversity to fresh gen)

---

## Success Metrics

For each run, track:
1. **buffer_max** (peak aggregate reward)
2. **buffer_max trajectory** (when did it plateau?)
3. **Mean reward (last 10 iters)**
4. **Revision success rate** (peak and overall, if applicable)
5. **Per-signal means** (which signals improved most?)
6. **Best plan text** (for LLM-as-judge evaluation)

---

## Results

| Run | buffer_max | Mean (last 10) | Peak Rev% | Overall Rev% | Status |
|---|---|---|---|---|---|
| MAIN | — | — | — | — | PENDING |
| B1 | — | — | n/a | n/a | PENDING |
| B2 | — | — | n/a | n/a | PENDING |
| B3 | — | — | n/a | n/a | PENDING |
| B4 | — | — | — | — | PENDING |
| A1 | — | — | — | — | PENDING |
| A2 | — | — | — | — | PENDING |
| A3 | — | — | — | — | PENDING |
| A4 | — | — | — | — | PENDING |
| A5 | — | — | — | — | PENDING |
| A6 | — | — | — | — | PENDING |
| A7 | — | — | — | — | PENDING |
| A8 | — | — | n/a | n/a | PENDING |
| A9 | — | — | — | — | PENDING |
