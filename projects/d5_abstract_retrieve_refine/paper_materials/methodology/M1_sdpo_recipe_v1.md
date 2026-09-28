# SDPO Recipe v1 — μ-v4 production configuration (locked 2026-04-27)

**Status**: locked. Reuse this recipe verbatim for Phase 3 abstraction-stage SDPO and
any future plan-level SDPO experiments.

## Architecture

```
                   ┌──────────────────────┐
   research goal ──┤  Teacher prompt      │── teacher samples 8 plans
                   │  = goal + oracle +   │   (n_plans=8 per iter)
   slim oracle ────┤    prev_critique     │
                   └──────────────────────┘            │
                                                       ▼
                   ┌─────────────────────┐    ┌──────────────────┐
   slim oracle ────┤  Student prompt     │    │  Pick 1 plan for │
       (NO        ┤  = goal + oracle    │    │  critique step   │
   critique)      │  (no critique)       │    └────────┬─────────┘
                   └─────────────────────┘             │
                            │                          ▼
                            │            ┌──────────────────────┐
                            │            │ Critic (Opus subagent) │
                            │            │  reads PICK + source  │
                            │            │  paper, writes        │
                            │            │  critique XML        │
                            │            └────────┬─────────────┘
                            │                     │
                            ▼                     │
            ┌──────────────────────────┐          │
            │ Student logprobs on the   │          │
            │ same 8 teacher tokens     │          │
            └────────┬──────────────────┘          │
                     │                              │
                     ▼                              ▼
            ┌──────────────────────────────────────┐
            │ A_t = clamp(scale*(t_lp - s_lp),     │
            │             [-clip, +clip])          │
            │ build SDPO datums; LoRA grad step    │
            └──────────────────────────────────────┘
```

## Hyperparameters (μ-v4 production, locked)

```python
# Optimization
learning_rate = 5e-5            # geometric median between v2's 1e-5 (no learning)
                                #   and v3's 2e-4 (immediate collapse)
n_iter = 20                     # but expect peak at iter 3-4, collapse iter 5+
                                #   stop early via per-iter audit (see below)
n_grad_steps_per_iter = 4       # microbatch chunk_size = 2; finer-grained Adam updates
n_plans = 8                     # samples per iter
lora_rank = 64
lora_alpha = 128                # 2× rank by convention
lora_dropout = 0.0

# SDPO advantage
sdpo_scale = 1.0                # multiplier on (teacher_lp - student_lp)
sdpo_clip_advantage = 5.0       # clamp per-token A_t to [-5, +5]
loss_fn = "importance_sampling" # via Tinker training_client.forward_backward

# Audit
eval_every = 1                  # CRITICAL: per-iter audit visibility
                                #   eval_every=3 gave us 4-dim daemon trajectories
                                #   that hid the peak via anchor saturation
save_every = 1                  # checkpoint every iter for rollback
audit_drop_threshold = 5.0      # SINGLE-iter drop ≥ 5 triggers early stop
                                #   (was 3.0 over-2-iter-rolling — too lax;
                                #   v4's 12.4 single-iter drop missed it)

# Adam
beta1 = 0.9
beta2 = 0.95
eps = 1e-8
```

## Per-token advantage formula

```python
# At each token position t in teacher's generated plan:
teacher_lp_t = log P(token_t | teacher_context, prev_tokens)
              # teacher_context = goal + oracle + prev_critique
student_lp_t = log P(token_t | student_context, prev_tokens)
              # student_context = goal + oracle (no critique)

A_t = clamp(sdpo_scale * (teacher_lp_t - student_lp_t),
            -sdpo_clip_advantage, +sdpo_clip_advantage)
```

The per-token advantage is **positive** when teacher's context (with critique)
makes a token more likely than student's context (without). Gradient pushes student
policy toward producing that token even without critique.

**Known limitation** (F4 critique-token-blindness): the gradient is dominated by
stylistic-token advantages because content tokens are rare in pretraining (low
absolute logprob in both contexts → small absolute advantage even when ratio is high).
This is why μ-v4 EVAL plans never include literal PUCT formula even though teacher
PICK plans do at iter 1-3.

## Datum construction

From `src/co_scientist/d5_abstract_retrieve_refine/mu_prompts_v2.py:build_sdpo_datum`:

```python
def build_sdpo_datum(student_input_tokens, teacher_gen_tokens, student_lps, advantages):
    # Concatenate prompt + teacher's generation
    all_tokens = student_input_tokens + teacher_gen_tokens
    prompt_len = len(student_input_tokens)

    # Logprobs and advantages aligned with teacher's generated portion
    full_logprobs = [0.0] * (prompt_len - 1) + student_lps
    full_advantages = [0.0] * (prompt_len - 1) + advantages

    return Datum(
        tokens=all_tokens,
        logprobs=full_logprobs,
        advantages=full_advantages,
    )
```

`Tinker.forward_backward(datums, loss_fn="importance_sampling")` does the rest. This
implementation is "option (c) HER relabelling" from Hübotter 2026: stored logprob =
recomputed logprob under the same context → IS ratio = 1 → reduces to per-signal REINFORCE.

## Stop criterion: best-of-early-iter

This is the most important practical lesson from the 3-lr ablation:

1. **Run for 8-10 iters max** with `eval_every=1` and `save_every=1`
2. **9-dim isolated audit per iter** (not 4-dim daemon — see F6)
3. **Production checkpoint = argmax(audit_mean over iters 0..N where collapse
   hasn't fired)**; typically iter 3-5
4. **Single-iter audit drop ≥ 5** triggers immediate halt + rollback to prior iter
5. Do NOT trust later-iter weights even if they "look stable" — may be on the
   pre-cliff plateau; verify with audit before deploying

## Critic prompt template

The Opus critic subagent receives goal + source paper (privileged) + 1 plan. It
outputs `<critique>` XML:

```
<critique>
  <idea_alignment>(1-3 sentences, recognize what's right)</idea_alignment>
  <missing_components>(1-3 sentences, name absent components specifically)</missing_components>
  <incorrect_assumptions>(1-3 sentences)</incorrect_assumptions>
  <feasibility>(1-3 sentences)</feasibility>
  <improvement_directive>(≤120 words, concrete actionable directive)</improvement_directive>
</critique>
```

DO NOT reveal paper name/author/arxiv ID in the critique. The critic has privileged
access to the source paper (`data/source_paper/v2.md`) but the model
being critiqued must not see it.

## File-bus pattern (no OpenRouter, no ANTHROPIC_API_KEY)

All Opus calls go via subagent file-bus:
- Trainer writes `critic_requests/iter_NNN.json` and `audit_requests/iter_NNN.json`
- A persistent Opus daemon (separate Claude Code session) polls and writes
  `critic_responses/iter_NNN.json` / `audit_responses/iter_NNN.json`
- Trainer blocks on response file existence (with timeout)

Daemon spec: see `projects/d5_abstract_retrieve_refine/scripts/opus_critic_audit_daemon.md`.

**Critical**: ONE daemon per run. Two daemons race on writing response files and can
produce malformed JSON. Daemon should use `json.dumps()` for output (don't string-concat
embedded XML — escape failures observed in v3 + v4).

## Reproducibility checklist

To reproduce μ-v4 iter-4 weights:

1. **Inputs**:
   - Goal: `dataset/research_goal.txt`
   - Slim oracle: `data/oracles/oracle_v2_2026_04_26_build/slim.md`
   - Source paper (privileged for critic): `data/source_paper/v2.md`
   - Bibliography (provenance): `data/bibliography/resolved_v2.jsonl` (86 papers)

2. **Code**:
   - Trainer: `src/co_scientist/d5_abstract_retrieve_refine/train_mu_v4.py`
   - Prompt builders: `src/co_scientist/d5_abstract_retrieve_refine/mu_prompts_v2.py`
   - Critic client: `src/co_scientist/shared/opus_critic_subagent.py`
   - Audit client: `src/co_scientist/shared/opus_audit_subagent.py`

3. **Run command**:
   ```bash
   PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.train_mu_v4 \
       log_path=projects/d5_abstract_retrieve_refine/runs/<DATE>_mu_v4 \
       config.learning_rate=5e-5 \
       config.n_grad_steps_per_iter=4 \
       config.eval_every=1 \
       config.save_every=1
   ```

4. **Daemon launch** (separate Claude Code session): see daemon spec in
   `projects/d5_abstract_retrieve_refine/scripts/opus_critic_audit_daemon.md`

5. **Stop criterion**: monitor 9-dim isolated audit per iter (not the 4-dim daemon
   audit). When isolated audit mean drops ≥ 5 in a single iter, halt and use
   prior-iter checkpoint.

## Known failure modes (DO NOT DO)

- **Don't run past iter 5-7** — multi-round instability is empirical (see F3)
- **Don't trust 4-dim daemon audit alone** — saturates at anchor 7 (F6)
- **Don't run two daemons concurrently** — race conditions on file writes
- **Don't use lr=2e-4** — collapses by iter 6
- **Don't use lr=1e-5** — no learning, advantage stuck at 0.4 with no progress

## Reuse for Phase 3 (abstraction-stage SDPO)

Phase 3 retrieve-then-generate may want to fine-tune the abstraction step (not just
plan generation). The same SDPO recipe applies:

- **Teacher context**: goal + retrieved chunks + critique
- **Student context**: goal + retrieved chunks (no critique)
- **Critic**: same Opus + privileged source paper, but the critique format may need
  to target retrieval quality (e.g. "missing key paper X") rather than plan content
- **Same hyperparameters**: lr=5e-5, 4 grad steps, eval_every=1, save_every=1
- **Same stop criterion**: best-of-early-iter via 9-dim isolated audit
- **Same expected failure mode**: multi-round instability

The architecture is task-agnostic; only the prompt template + critique format need
adaptation.
