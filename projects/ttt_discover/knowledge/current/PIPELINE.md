# Current Pipeline: CR-v7 with Per-Signal REINFORCE + Fresh-Plan Training

**Last updated**: 2026-04-19
**Code**: `src/co_scientist/ttt_discover/train_cr_v7.py`
**Primary model**: Qwen3-30B-A3B (policy = grader)
**Alt grader**: Qwen3-235B-A22B-Instruct-2507 (for SA_arithmetic only)
**Signal set**: v9 (see `SIGNAL_SET_v9.md`)
**Goal**: Goel-style 127-word research scenario

## Pipeline Overview

Three-phase loop per iteration:

### Phase 1: Fresh Generation (Exploration)
- Generate `n_fresh=4` plans from π(·|goal, buffer_context)
- Buffer context: top-K plans from buffer (UCB selection)
- Grade on all 10 active signals (v9) with `grader_repeats=1`
- With `train_on_fresh=True`: compute entropic REINFORCE gradient
  - w_k = exp(β·R_k) / Σ exp(β·R_j), β via KL budget (γ = ln 2)
  - This is RL's unique advantage — B4 cannot improve fresh quality

### Phase 2: Critique-Conditioned Revision (Exploitation)
- Select `n_revise=4` parents from buffer via UCB
- For each parent:
  - Grader emits per-signal critique c_i alongside score s_i
  - Build revision prompt: goal + plan + all K per-signal critiques
  - Generate `n_revision_candidates` (1 or 2) revised plans
  - Grade revisions, keep best by Δ_aggregate
- Add best revision to buffer

### Phase 3: Per-Signal REINFORCE (RL Update)
- For each revision with |Δ_i| > threshold on signal i:
  - Per-signal advantage: A_i = Δ_i · δ_scale, Δ_i = (s_rev_i - s_parent_i)/4
  - HER-style context relabelling: compute logπ(a | context_i) where
    context_i contains only critique_i (not all K)
  - Emit Datum with advantage A_i
- Loss: L = -Σ_i A_i · Σ_t logπ(a_t | context_i)
- Importance sampling ratio = 1 (REINFORCE form)
- LoRA rank 32, lr 4e-5, Adam

## Config Flags

| Flag | Default | Description |
|---|---|---|
| `skip_rl_update` | False | B4 ablation: skip all RL updates |
| `train_on_fresh` | False | Enable entropic REINFORCE on fresh plans |
| `strip_critiques` | False | B4_stripped: revision prompt shows only aggregate score |
| `n_revision_candidates` | 2 | BoN: candidates per parent revision |
| `grader_model_alt` | "" | Alt grader model for signals with grader_model_override |
| `disabled_signals` | "" | Comma-separated signal IDs to disable |

## Best Results (2026-04-19)

| Condition | Qwen buffer_max | Opus /20 |
|---|---|---|
| MAIN_fresh_bon1 (RL+fresh, BoN=1) | **0.927** | **10** |
| B4_bon1 (no RL, BoN=1) | 0.840 | 8 |
| MAIN_v9 (RL rev-only, BoN=2) | 0.950 | 9 |
| B4_v9 (no RL, BoN=2) | 0.950 | 9 |
| B4_stripped (no critique) | 0.763 | 6 |
| Reference (expert) | ~0.748 | 20 |

## Component Hierarchy

```
critique conditioning >> RL (with fresh training) >> BoN >> nothing
```

RL's unique contribution: improving the exploration policy (fresh plans).
Critique conditioning saturates exploitation. Only RL can improve exploration.
