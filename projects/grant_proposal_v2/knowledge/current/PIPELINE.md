# Current Pipeline: CR-v7 with Per-Signal REINFORCE + Fresh-Plan Training

**Last updated**: 2026-04-20
**Code**: `src/co_scientist/grant_proposal/train_cr_v7.py`
**Primary model**: Qwen3-30B-A3B (policy) / configurable grader (default same as policy)
**Alt grader**: Qwen3-235B-A22B-Instruct-2507 (for SA_arithmetic only)
**Signal set**: D4-v7 signals (`grant_signal_reward.py`)
**Goal**: Grant solicitation (NSF/NIH/ERC format)

> **Note**: This pipeline is architecturally identical to D3's CR-v7 pipeline
> (`src/co_scientist/ttt_discover/train_cr_v7.py`). The only differences are:
> (1) signal definitions are in `grant_signal_reward.py` instead of `ten_signal_reward.py`,
> (2) the pipeline operates on grant proposals instead of research plans, and
> (3) reference data comes from Open Grants winning proposals instead of arxiv-derived plans.

## Pipeline Overview

Three-phase loop per iteration:

### Phase 1: Fresh Generation (Exploration)
- Generate `n_fresh=4` proposals from pure policy (no buffer context)
- Grade on all active signals (D4-v7) with `grader_repeats=1` (temperature=0 → deterministic)
- With `train_on_fresh=True`: compute entropic REINFORCE gradient (novelty_weight=0)
  - w_k = exp(beta*R_k) / sum exp(beta*R_j), beta via KL budget (gamma = ln 2)
  - This is RL's unique advantage — B4 cannot improve fresh quality

### Phase 2: Critique-Conditioned Revision (Exploitation)
- Select `n_revise=4` parents from buffer via UCB
- For each parent:
  - Grader emits per-signal critique c_i alongside score s_i
  - Build revision prompt: goal + proposal + all K per-signal critiques
  - Generate `n_revision_candidates` (1 or 2) revised proposals
  - Grade revisions, keep best by Delta_aggregate
- Add best revision to buffer

### Phase 3: Per-Signal REINFORCE (RL Update)
- For each revision with |Delta_i| > threshold on signal i:
  - Per-signal advantage: A_i = Delta_i * delta_scale, Delta_i = (s_rev_i - s_parent_i)/(score_max - 1)
  - HER-style context relabelling: compute log pi(a | context_i) where
    context_i contains only critique_i (not all K)
  - Emit Datum with advantage A_i
- Loss: L = -sum_i A_i * sum_t log pi(a_t | context_i)
- Importance sampling ratio = 1 (REINFORCE form)
- LoRA rank 32, lr 4e-5, Adam

## Config Flags

| Flag | Default | Description |
|---|---|---|
| `skip_rl_update` | False | B4 ablation: skip all RL updates |
| `train_on_fresh` | False | Enable entropic REINFORCE on fresh proposals |
| `strip_critiques` | False | B4_stripped: revision prompt shows only aggregate score |
| `scores_only` | False | Show signal names + scores only, no critique text (reduced leakage) |
| `fresh_use_context` | False | Fresh plans see buffer context (D3-style; default off = pure policy) |
| `novelty_weight` | 0.0 | Novelty bonus on fresh plans (disabled) |
| `grader_repeats` | 1 | Grading calls per signal (temperature=0 → 1 is sufficient) |
| `grader_max_tokens` | 8192 | Token budget for grader (accommodates Qwen thinking) |
| `n_revision_candidates` | 1 | BoN: candidates per parent revision |
| `grader_model_alt` | "" | Alt grader model for signals with grader_model_override |
| `disabled_signals` | "" | Comma-separated signal IDs to disable |
