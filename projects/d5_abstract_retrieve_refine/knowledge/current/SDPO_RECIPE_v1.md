# SDPO Recipe v1 — μ-v4 production configuration (locked 2026-04-27)

> **⚠ READ THIS FIRST** (2026-04-27 PM, revised): the body of this document
> describes what we now know is a **D5 in-house off-policy IS-loss variant**,
> NOT canonical SDPO. An earlier revision claimed it was "option (c) HER from
> Hübotter 2026" — WebFetch of arxiv 2601.20802 confirmed the Hübotter SDPO
> paper does not discuss any such variant. That label was D5 internal
> speculation.
>
> **Two ground-truth references (read these BEFORE this body)**:
> 1. `CANONICAL_NAMING_REFERENCE.md` — paper-grade ArXiv attribution of canonical
>    SDPO (Hübotter 2601.20802) and canonical OPSD (Zhao 2601.18734); maps each
>    D5 trainer to the truthful description from raw code
> 2. `RUN_REGISTRY.md` — per-run setup table (every historical training run + 
>    its actual algorithm + config + comparability anchors)
>
> **What this body documents**: the recipe for the D5 in-house off-policy
> IS-loss variant used by μ-v2/v3/v4/v6/κ-v1. This is retained as historical
> reference for those runs. For NEW canonical training, see CANONICAL_NAMING_REFERENCE
> + decide between Hübotter SDPO (top-K KL) vs Zhao OPSD (sampled-token).
>
> The SUPERSESSION addendum at the end of this file documents
> `train_mu_v7_opd.py` (closest D5 implementation to OPSD sampled-token
> policy-gradient variant; wins 45/48 pairwise vs the in-house off-policy
> variant per F11/F12/F13/F14).

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

`Tinker.forward_backward(datums, loss_fn="importance_sampling")` does the rest.

> **⚠ CORRECTION (2026-04-27 PM)**: an earlier version of this file claimed
> this is "option (c) HER relabelling from Hübotter 2026". WebFetch verification
> of the actual Hübotter paper (`arxiv.org/abs/2601.20802`) confirmed the paper
> does NOT discuss option (a)/(b)/(c) variants nor "HER relabelling". That label
> was D5 internal speculation. The correct truthful description: **this is a D5
> in-house off-policy IS-loss variant**. TEACHER samples plans (with critique
> context); STUDENT lp recomputed on those tokens (under no-critique context);
> stored logprob = recomputed student logprob → IS ratio = 1 → reduces to
> per-signal REINFORCE on student-context. **Crucially**: this is NEITHER
> canonical Hübotter SDPO (which requires student-on-policy + KL loss) NOR
> canonical Zhao OPSD (2601.18734, also student-on-policy + JS or sampled-token
> reverse-KL). For canonical algorithms see `CANONICAL_NAMING_REFERENCE.md` +
> `RUN_REGISTRY.md`.

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

## Recipe outcome (added 2026-04-27)

This recipe applied to μ-v4 produced:

- **audit_v3 ISOLATED** iter 4 mean = **28.00 / 45** (+2.75 over σ baseline 25.25);
  first 30B-trained variant to beat σ frozen+oracle baseline
- **Opus pairwise corroboration**: 6-2 vs σ (PRIMARY), 7-1 vs δ, 7-1 vs α →
  **20/24 = 83.3%** anonymized win rate against the close-cluster competitors
- Position-bias check passes (50/62/75% A-pos rates)

Both absolute and pairwise metrics agree — recipe is **validated for Phase 3 reuse**.

When applied to Phase 3 (retrieve-then-generate), expected behavior:

- Same lr=5e-5 / 4 grad steps / iter-4 peak likely
- Multi-round instability (iter 5+ cliff) likely persists; Hübotter 2026 §4 failure
  mode is architectural, not config-specific
- `save_every=1` + audit-drop ≥ 3 early stop guard remains essential
- Consider tightening to audit-drop ≥ 2 if retrieval introduces faster cliffs

See `paper_materials/findings/F2_mu_v4_iter4_peak.md` § "Cross-validation: pairwise
corroboration" + `paper_materials/experiments/E6_mu_v4_pairwise_corroboration.json`
for the full validation evidence.

## Continual SDPO warning (added 2026-04-26 from F9)

⚠️ **IMPORTANT for future sessions**: this recipe is for **single-paper SDPO** only.
Phase 5 (DECISIONS 2026-04-26 (c)/(d)/(e)) tested **continual** SDPO — chaining
P1 → P2 → P3 from a μ-v4 anchor onto follow-up papers (Tool-V, Meta-TTL).
**All 4 cells in (anchor saturation × Adam state × anchor_ce regularization) grid
FAILED H1 forward learning**. Future sessions: do NOT naively re-run continual
chains with hyperparameter tweaks.

Empirical findings:
- Adam state inheritance amplifies cliff (-3.5 with full Adam vs -2.1 with fresh
  Adam at iter 1) but fresh-Adam doesn't prevent cliff
- Non-saturated anchor (μ-v4 iter 2) delays cliff (iter 4 vs iter 1) but still
  degrades over 4 iters (-2.71 cumulative)
- Per-token CE anchor at weight 0.1 ACCELERATES collapse (iter 2 cliff vs iter 4
  for unregularized) — two-gradient conflict with Tool-V SDPO signal
- Best v5 vs μ-v4 isolated /45: v3 Δ +0.62 (within n=8 noise); α Δ -2.62 (negative)

Mitigations are **non-local** (not hyperparameter tweaks):
- Plan-level replay buffer (mix prior plans into batches, NOT output regularization)
- EWC penalty with Fisher Information (per-parameter importance)
- LoRA-per-paper + weighted merge (parallel single-paper SDPO + ensemble; Phase
  4a's σ' + μ' is the implicit validation of this path)

See `paper_materials/findings/F9_continual_sdpo_field_expert.md` for full 4-cell
diagnostic + per-token-CE-wrong-abstraction sub-finding. See
`knowledge/current/CONTINUAL_SDPO_FAILURE_MODES_v1.md` for one-page developer
reference.

**DO**: Use this recipe verbatim for any new TTT-family goal (single-paper SDPO
from base Qwen3-30B + 5-iter cap + lr=5e-5 + 4 grad steps).

**DO NOT**: Chain SDPO from a μ-v4 anchor to a new paper's critic without first
implementing one of the 3 mitigations above. Phase 6 will test ONE such mitigation
(likely plan-level replay buffer); pre-registration in DECISIONS 2026-04-26 (f) TBD.

---

## ADDENDUM (2026-04-27 PM, REVISED): D5 in-house on-policy variant

The above recipe documents a **D5 in-house off-policy IS-loss variant** (TEACHER samples plans with critique context; STUDENT lp recomputed; IS-loss). This is NEITHER canonical Hübotter SDPO (2601.20802; student on-policy + KL loss) nor canonical Zhao OPSD (2601.18734; student on-policy + JS or sampled-token reverse-KL). Stage A + Stage B (2026-04-27) isolated the off-manifold gradient property of this variant.

**`train_mu_v7_opd.py`** (`opd_mode=True`) is a **D5 in-house on-policy IS-loss variant** — the closest D5 implementation to OPSD's sampled-token policy-gradient form (Zhao 2601.18734, Table 3, ~2% gap from canonical full-vocab JS). One-architecture swap:

| | Old (D5 in-house off-policy IS-loss) | New (D5 in-house on-policy IS-loss) |
|---|---|---|
| Trainer file | `train_mu_v4.py` | `train_mu_v7_opd.py` |
| `opd_mode` config | n/a | `True` (default) |
| Sampler | TEACHER under teacher_input | STUDENT under student_input (on-policy) |
| Logprob signal | `student_lp = compute_logprobs(student_input + teacher_tokens)` | `teacher_lp = compute_logprobs(teacher_input + student_tokens)` |
| Datum prefix | student_input | student_input (unchanged) |
| Datum gen tokens | teacher-sampled (off-manifold) | student-sampled (on-manifold) |
| Advantage formula | `A = clamp(scale·(t_lp − s_lp), [−5, 5])` | SAME formula, sources swapped |
| Loss form | IS-loss with stored s_lp (off-manifold) | IS-loss with on-policy s_lp ≈ OPSD sampled-token policy-gradient |
| Closest paper alignment | NEITHER paper | OPSD sampled-token (Zhao 2601.18734 Table 3) — modulo IS-loss vs JS distinction |

**Empirical results** (F13/F14 — these were obtained from D5 in-house variants; not paper-faithful canonical algorithms):
- Pairwise μ-v7-opd vs μ-v4: **23/24 wins** combined on TTT-D + 22/24 cross-goal (smoke 15/16 + paper-grade 8-0 + xgoal)
- Audit /45 strict isolated peak (iter 4): 26.62 (+1.38 over σ)
- F3 cliff delayed +2 iter (5 → 7)
- J_β oracle-present content recovered (μ-v4 0/8, μ-v7-opd 3/8 at iter 5)

**Empirical claims should be reported as outcomes of "D5 in-house on-policy IS-loss" beating "D5 in-house off-policy IS-loss"**, NOT as canonical-OPSD-beats-canonical-SDPO comparisons (we have NOT implemented canonical KL-form of either paper). For paper-faithful canonical algorithms, see `CANONICAL_NAMING_REFERENCE.md`.

**Use `opd_mode=False` for ablation comparison** (reverts to D5 in-house off-policy variant).

Phase 5 continual chain trainers (cliff_v1/v2/v3/α) and replay variants (train_mu_v6_replay) inherit the D5 in-house off-policy IS-loss pattern from μ-v4. Re-forking them as `*_opd.py` would migrate to D5 in-house on-policy IS-loss.
