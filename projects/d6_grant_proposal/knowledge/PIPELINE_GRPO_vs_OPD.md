# D6 Pipelines — GRPO vs OPD

**Last updated:** 2026-04-30
**Source files (the things this doc describes):**

- `src/co_scientist/d6_grant_proposal/train_grpo.py` — GRPO trainer
- `src/co_scientist/d6_grant_proposal/train_opd.py` — OPD trainer (simplified recipe)
- `src/co_scientist/d6_grant_proposal/prompts.py` — student / teacher / critic prompt builders
- `projects/d6_grant_proposal/prompts/generation/{student,teacher}.md` — actual prompt templates
- `projects/d6_grant_proposal/prompts/review/critic_instructions.md` — Opus critic instructions
- `src/co_scientist/shared/grant_signal_reward.py` — 12-signal Qwen-30B grader (used by GRPO reward AND by frozen baseline scoring; NOT used by OPD)

If anything in this doc disagrees with those source files, the source files win.

---

## 1. Side-by-side overview

| Aspect | **GRPO** (`train_grpo.py`) | **OPD** (`train_opd.py`) |
|---|---|---|
| Reward / target signal | Qwen3-30B-A3B 12-signal weighted mean (scalar in `[0,1]`) | Per-token logprob delta from a **same-architecture teacher** conditioned on a **privileged-Opus critique** |
| External judge | None during training | Opus 4.7 reviewer; sees `reference_proposal.md` as privileged ground truth |
| Privileged information | None | `reference_proposal.md` flows through Opus → `<critique>` → teacher context |
| Group size per iter | `n_rollouts_per_group` (4 by default) | `n_plans` (8 by default); critic sees ONE randomly chosen plan per iter |
| Advantage formula | `A_i = (R_i − mean(R)) / (std(R) + ε)`, broadcast to every gen token of plan i | `A[t] = clamp(lp_teacher[t] − lp_student[t], ±5)`, per token, independent across tokens |
| Loss | PPO-clipped surrogate, ε=0.2 | PPO-clipped surrogate, ε=0.2 |
| Solution-only mask | Yes (kept for cross-arm comparability) | Yes (paper recipe) |
| Frozen-base anchor | None | None (trust-region α removed 2026-04-30) |
| Per-iter LLM calls | 4 sampler + 4 grader (frozen Qwen-30B) | 8 sampler + 8 teacher logprob + 1 Opus critic |
| What the gradient is "trying to imitate" | A high-reward sibling rollout (within-group) | The teacher's distribution under the same prompt + a critique that names the rollout's weaknesses |

The headline difference: **GRPO learns from a scalar grade. OPD learns from a structured per-rollout critique that was generated using ground truth the policy can never see.**

---

## 2. GRPO pipeline

```
                          ┌──────────────────────┐
                          │  research_goal.md    │
                          │  slim_oracle (3k tok)│   <-- inputs (read once)
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  build_student_prompt│   prompts.py:36
                          │  goal + oracle       │   = student.md template
                          └──────────┬───────────┘
                                     │
                                     │  Iter t (loop, t=0..N-1)
                                     │
        ┌────────────────────────────▼─────────────────────────────┐
        │ A. SAMPLE 4 rollouts from current LoRA-adapted policy    │  train_grpo.py:294-302
        │    (sampler_path = save_weights_for_sampler at iter t)   │
        └────────────────────────────┬─────────────────────────────┘
                                     │
                                     │  4× plan tokens, 4× sampler logprobs
                                     │
        ┌────────────────────────────▼─────────────────────────────┐
        │ B. GRADE each plan via FROZEN Qwen3-30B-A3B 12-signal    │  train_grpo.py:314-326
        │    rubric → R_i ∈ [0,1] (weighted mean of 1-5 → 0-1)     │  uses shared/grant_signal_reward.py
        └────────────────────────────┬─────────────────────────────┘
                                     │
                                     │  4 scalar rewards R_0..R_3
                                     │
        ┌────────────────────────────▼─────────────────────────────┐
        │ C. GROUP-RELATIVE ADVANTAGE                              │  train_grpo.py:333-335
        │    A_i = (R_i − mean(R)) / (std(R) + 1e-8), clipped ±5   │  train_grpo.py:360-361
        │    Broadcast: per_tok_adv = [A_i] * len(gen_tokens_i)    │
        └────────────────────────────┬─────────────────────────────┘
                                     │
                                     │  per-token advantages
                                     │
        ┌────────────────────────────▼─────────────────────────────┐
        │ D. SOLUTION-ONLY MASK                                    │  train_grpo.py:366-373
        │    Zero advantage outside last <solution>...</solution>  │  build_solution_token_mask_from_tokens
        │    (so gradient flows only on the proposal body)         │
        └────────────────────────────┬─────────────────────────────┘
                                     │
        ┌────────────────────────────▼─────────────────────────────┐
        │ E. TRAIN STEP — PPO-clipped surrogate, ε=0.2             │  train_grpo.py:413-430
        │    forward_backward → optim_step (Adam, lr=5e-5)         │
        └────────────────────────────┬─────────────────────────────┘
                                     │
                                     │  (optional) eval rollouts → audit
                                     │
                                     ▼
                              [next iter t+1]
```

### Step-by-step description

1. **Inputs.** `research_goal.md` and the slim oracle (`data/{domain}/{goal}/oracle/slim.md`) are read once. The oracle is a ~3k-token markdown abstraction of `reference_proposal.md` — methodological patterns only, no privileged details. (`train_grpo.py:185-194`.)

2. **Student prompt.** `build_student_prompt(goal, oracle)` renders `prompts/generation/student.md`, which asks the policy to write the proposal inside `<solution>...</solution>` with the 8-section NIH-style template. This same prompt is used by the policy for both rollouts and eval. (`prompts.py:36-47`.)

3. **Sample.** At each iter, save the current LoRA weights for the sampler, create a sampling client at that path, and draw `n_rollouts_per_group` plans (default 4). (`train_grpo.py:285-299`.)

4. **Grade.** For each plan, build a `single_call` grader prompt over the active 12 signals (G12 → G12a for `social_science` domains) and call the **frozen** Qwen3-30B-A3B grader (no LoRA). Parse the per-signal scores and compute the weighted mean → scalar `R_i`. (`train_grpo.py:149-174, 314-326`.)

5. **Group-relative advantage.** Standard GRPO formula: subtract the group mean, divide by group std, clip to ±5. The same scalar advantage broadcasts to every generated token of that plan. If the group is uniform (all rewards equal), advantage = 0 and the iter is skipped. (`train_grpo.py:329-339`.)

6. **Solution-only mask.** Reuse the OPD mask helper (`build_solution_token_mask_from_tokens`) so the gradient only flows on tokens inside the last `<solution>` body, and never on tokens inside `<think>` blocks. (`train_grpo.py:366-373`, mask logic at `train_opd.py:165-226`.)

7. **PPO-clip update.** Build a Tinker `Datum` per plan, run `forward_backward` with `loss_fn="ppo"` and `clip_low/high = 1.0 ∓ 0.2`, then `optim_step` with Adam (`β1=0.9, β2=0.95, lr=5e-5`). (`train_grpo.py:413-430`.)

8. **(Optional) Eval rollout.** At every `eval_every` iter, sample additional plans for downstream audit (`audit.py` reads them later). Audit is **not** part of the gradient. (`train_grpo.py:432-450`.)

### What the policy is being pushed toward

GRPO pushes the policy to imitate sibling rollouts that scored above the group mean according to the 12-signal Qwen-30B rubric. The reward is goal-agnostic (signals are defined the same way for every goal); the only goal-conditioning comes through the prompt and the oracle.

---

## 3. OPD pipeline (simplified, post-2026-04-30)

```
                          ┌──────────────────────┐
                          │  research_goal.md    │
                          │  slim_oracle (3k tok)│
                          │  reference_proposal  │  <- privileged for reviewer ONLY
                          │     .md (privileged) │
                          └──┬─────────┬─────────┘
                             │         │
                             │         │
              ┌──────────────▼──┐   ┌──▼─────────────────┐
              │ student.md      │   │ critic_instructions│
              │ (goal+oracle)   │   │ .md (8-child       │
              │                 │   │ <critique> schema) │
              └──────┬──────────┘   └──┬─────────────────┘
                     │                 │
                     │  Iter t (loop)  │
                     ▼                 │
        ┌────────────────────────────┐ │
        │ A. STUDENT ROLLOUT         │ │  train_opd.py:367-381
        │    Sample 8 plans under    │ │
        │    student context         │ │
        └──────────────┬─────────────┘ │
                       │               │
                       │ pick 1 plan   │
                       ▼               │
        ┌────────────────────────────┐ │
        │ B. OPUS CRITIC (BLOCKING)  │ │  train_opd.py:393-422
        │    Sees plan + reference   │ │  prompts.py:77-97
        │    proposal (privileged)   │◄┘
        │    Emits 8-child <critique>│
        │    "DO NOT mention ref     │
        │    proposal in output"     │
        └──────────────┬─────────────┘
                       │ critique XML (text)
                       ▼
        ┌────────────────────────────┐
        │ C. TEACHER CONTEXT         │   train_opd.py:424-428
        │    goal + oracle +         │   teacher.md template
        │    critique                │
        └──────────────┬─────────────┘
                       │
        ┌──────────────▼─────────────┐
        │ D. TEACHER LOGPROBS        │   train_opd.py:431-440
        │    For each of the 8       │   compute_logprobs on
        │    student rollouts:       │   teacher_prefix + student_tokens
        │    log p_teacher(token_t)  │
        └──────────────┬─────────────┘
                       │ teacher_lp[t]  (one per token, per plan)
                       ▼
        ┌────────────────────────────┐
        │ E. PER-TOKEN ADVANTAGE     │   train_opd.py:463-468
        │    A[t] = clip(             │
        │      teacher_lp[t]          │
        │      − student_lp[t], ±5)   │
        │    sdpo_scale = 1.0         │
        └──────────────┬─────────────┘
                       │
        ┌──────────────▼─────────────┐
        │ F. SOLUTION-ONLY MASK      │   train_opd.py:470-485
        │    Zero advantage outside  │
        │    last <solution>...</...>│
        └──────────────┬─────────────┘
                       │
        ┌──────────────▼─────────────┐
        │ G. TRAIN STEP              │   train_opd.py:508-528
        │    PPO-clip ε=0.2          │
        │    forward_backward →      │
        │    optim_step (Adam,       │
        │    lr=5e-5)                │
        └──────────────┬─────────────┘
                       │
                       │ (optional) eval rollouts → audit
                       ▼
                 [next iter t+1]
```

### Step-by-step description

1. **Inputs.** Same `research_goal.md` and slim oracle as GRPO, plus `reference_proposal.md`. The reference proposal is loaded into RAM at startup and used **only** by the Opus reviewer — the policy never sees it. (`train_opd.py:247-249`.)

2. **Student rollout.** Sample `n_plans` (8 by default) under the same student prompt GRPO uses (`prompts/generation/student.md`). The student context contains `goal + oracle`, no critique. (`train_opd.py:367-381`.)

3. **Pick one plan, send to Opus.** Choose one valid decoded plan at random, build a critic-request payload (`prompts.py:77-97`) that contains:
   - `goal` (also in the prompt)
   - `source_paper_md` = `reference_proposal.md` (PRIVILEGED — the only place this flows)
   - `plans` = `[{plan_id, text}]` (just the chosen rollout)
   - `instructions` = `prompts/review/critic_instructions.md`

   The instructions tell Opus 4.7 to use the reference as ground truth but **not** to mention it in the output. The reviewer emits an 8-child `<critique>` block (`<idea_alignment>`, `<missing_components>`, `<incorrect_assumptions>`, `<feasibility>`, `<clarity>`, `<strengths>`, `<weaknesses>`, `<improvement_directive>`).

   Critic call is **blocking** with a 900-second timeout; on timeout we fall back to the cold-start critique placeholder. (`train_opd.py:393-422`, `prompts/review/critic_instructions.md`.)

4. **Build teacher context.** `build_teacher_prompt(goal, oracle, critique_xml)` renders `prompts/generation/teacher.md`, which is the same as `student.md` plus a "Critique of prior attempt (to address)" section containing the full 8-child block. Per the prompt's own note, only the `<improvement_directive>` text actually shifts the conditional distribution; the other children are debug context for human readers. (`train_opd.py:424-428`, `prompts.py:50-63`.)

5. **Teacher logprobs.** For each of the 8 student rollouts, run `compute_logprobs(teacher_prefix + student_tokens)` on the **same LoRA-adapted policy** (the teacher is just the student under a different prompt — there is no separate teacher network). Slice off the prefix and keep the per-token logprobs over the rollout. (`train_opd.py:431-440`.)

6. **Per-token advantage.** For each token in each plan: `A[t] = clamp(teacher_lp[t] − student_lp[t], ±5)`. Tokens that the teacher rates higher than the student get positive advantage; tokens the teacher rates lower get negative. The clip is for safety against logprob outliers, not as a regularizer. (`train_opd.py:463-468`.)

7. **Solution-only mask.** Same mask helper as GRPO (`build_solution_token_mask_from_tokens`); zeros the advantage for any token outside the last `<solution>...</solution>` body, so gradient never flows on goal/oracle/critique tokens or on `<think>` content. (`train_opd.py:470-485`.)

8. **PPO-clip update.** Same loss as GRPO — a Tinker `Datum` per plan, `loss_fn="ppo"` with `clip_low/high = 1.0 ∓ 0.2`, then `optim_step` with Adam. The Datum's `model_input` is the student prompt + generation tokens (the teacher prefix is **not** in the model input — the teacher's role was only to produce the per-token target via `compute_logprobs`). (`train_opd.py:508-528`, `prompts.py:21-25` re-exporting `build_sdpo_datum`.)

9. **(Optional) Eval rollout.** Same as GRPO — extra samples for `audit.py` to score later. Not part of the gradient. (`train_opd.py:546-573`.)

### What the policy is being pushed toward

OPD pushes the policy to **become its own teacher**: the same network, conditioned on a privileged-Opus critique that named this iter's specific weaknesses, would assign higher probability to certain tokens than the no-critique student does. The gradient pushes the student's distribution toward that "with-critique self." Across many iters, the student internalizes the kinds of edits Opus repeatedly suggests — without ever seeing the reference proposal directly.

The privileged information enters the loop **once per iter** and only via the reviewer. It never reaches the policy weights through any channel except the per-token logprob delta induced by the resulting critique.

---

## 4. Why we expect OPD to beat GRPO on this task

This section is hypothesis, not result. It states the prediction so the pilot's outcome is interpretable.

GRPO's reward is the 12-signal Qwen3-30B-A3B grader. The available evidence (`projects/grant_proposal_v2/analysis/grader_panel_v8/analysis/correlation_matrix.jsonl:6`) shows that grader correlates with Opus 4.7 only at mean per-signal Spearman ρ = **0.399** on a 15-proposal stratified panel, and is especially weak on the depth signals: G6 reasoning_depth ρ = **−0.229**, G9 scope_feasibility ρ = **0.005**. Optimizing this reward under adversarial policy gradient is therefore optimizing a **noisy proxy** for what Opus would call a good grant proposal. The standard Goodhart prediction follows: reward goes up, audit-Opus score does not.

OPD does not depend on the rubric. The teacher's per-token logprob comes from a same-architecture model conditioned on a structured critique that was written by Opus while looking at a reference proposal. The signal carries reference-anchored corrections (e.g., "name the specific RL convergence-bound technique you'd use") that the rubric cannot encode in five-point scales. We expect OPD to lift the Opus audit score because the gradient was, indirectly, derived from Opus's own judgment about this specific rollout against this specific reference.

The pilot decision rule (D6_master_plan.md §5, Step 3c) is **OPD ≥ baseline + 0.04 over 3 seeds at iter 25 on Opus audit aggregate, AND OPD ≥ GRPO + 0.04**. M8 close-cluster pairwise corroboration kicks in if Δ < 0.05.

---

## 5. What is NOT in this pipeline (and where to find the deferred bits)

- **Equation/citation grounding bonus** — exists in `train_opd_grounded.py` but is out of scope for the workshop paper. See `D6_master_plan.md` §10.
- **KL anchor toward an SFT-tuned oracle** — scaffold only in `train_opd_kl_anchor.py`; needs an SFT pass that doesn't exist. Out of scope.
- **RAG retrieval over a grant-domain corpus** — corpus undefined. Future work.
- **Trust-region α-blend** — was in the pre-2026-04-30 OPD trainer (then named `train_mu_v8.py`); removed. See `DECISIONS.md` "2026-04-30 (later) Trust-region removed from V8". Recoverable via `git revert`.
- **GRPO with a stronger grader (Qwen3-235B, mean ρ=0.741)** — open methodological gate, see `D6_master_plan.md` §9.1. Decision pending before Step 2 starts.

---

## 6. Cross-references

- `STATUS.md` — what's built, what's pending
- `DECISIONS.md` — append-only decision log including the trust-region removal
- `D6_master_plan.md` — single source of truth for D6 strategy
- `pipeline_docs/INDEX.md` + `pipeline_docs/{train_opd,train_grpo,train_baseline}.md` — module-level docs
- `CONVENTIONS.md` — naming, paths, score-scale rules
