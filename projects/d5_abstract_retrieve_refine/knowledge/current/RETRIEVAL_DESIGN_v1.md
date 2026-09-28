# Retrieval Design v1 — D5 Phase 3 (走线 A)

*Locked 2026-04-27. Authoritative single-source for Phase 3 distillation pipeline
design. Plan ref: `~/.claude/plans/peaceful-tickling-wolf.md`. Cross-session
namespace lock: `DECISIONS.md` 2026-04-27 entry + 2026-04-26 Session B entry.*

## Status

Phase 3 (走线 A) extends Phase 2 by introducing **dynamic per-batch
distillation** of the oracle store as a **learnable action**, replacing Phase 2's
static slim-oracle scaffolding (σ baseline). Two new baselines: **τ** (frozen
+ distillation, no training) and **κ** (SDPO+critic trained on distillation
tokens).

Phase 2 production: μ-v4 iter 4 = 28.00 / 45 (+2.75 over σ 25.25), pairwise
corroborated 20/24. Phase 3 expects τ ≈ σ or τ > σ (Gate 1), and κ-peak ≥ τ + 2
AND ≥ σ + 3 (Gate 3 STRONG SUCCESS).

## Naming + cross-session partition

| Symbol | Meaning | Phase |
|---|---|---|
| σ | frozen Qwen3-30B-A3B + slim oracle (Opus pre-curated 87 items) | Phase 2 baseline (anchor) |
| μ-v4 | Phase 2 SDPO+critic on σ-style prompt | Phase 2 production (locked, do not retrain) |
| **τ** | frozen Qwen3-30B-A3B + 3-round distillation from medium oracle (181 items) | **Phase 3 baseline (Path Y)** |
| **κ** | SDPO+critic trained on τ-style 3-round distillation prompt | **Phase 3 production goal (Path Y)** |
| σ', μ', δ' | Phase 4a forward-citation cross-goal validation (Session B 走线 B) | NOT this design |

Session A (this design) does not touch Session B's `cross_goal/` subtree,
`baseline_frozen_v2.py`, `audit_v3_isolated_2way.py`, `paper_retrieval.py`,
F7 paper slot. See DECISIONS.md 2026-04-27 entry for full bilateral lock.

## Pipeline

### τ baseline (frozen, no training)

```
Setup: medium oracle (181 items, 6-cat tagged) shuffled (seed=τ_seed=42)
       → 3 batches (60 / 60 / 61)

Per goal G (TTT-Discover):
  Round 1: Qwen3-30B(G + batch_1) → A_1   (~700 tokens distilled list, no critic)
  Round 2: Qwen3-30B(G + batch_2) → A_2   (no critic, no A_1 memory)
  Round 3: Qwen3-30B(G + batch_3) → A_3   (no critic, no A_1/A_2 memory)

  PLAN: Qwen3-30B(G + [A_1, A_2, A_3]) → research plan
        instruction: "去重 + 整合三段 distillation"

  No SDPO, no CE, no LoRA — pure inference (Tinker only)

Eval: audit_v3 ISOLATED (8 plans × 9 dim) → τ mean / 45
```

### κ training (trained, SDPO + critic on each A_i)

```
Setup: same medium oracle → 3 batches (60/60/61); seed=κ_seed=43
       (distinct from τ seed for ablation independence)

Per iter (n_iter=20, peak expected iter 3-5):
  Per instance i ∈ 1..8:
    Round j ∈ 1..3:
      Step A — distillation (LEARNED, fresh base Qwen3-30B + LoRA):
        teacher prompt:  G + batch_j + critique_{i,j-1}*  (* round 1 has none)
        student prompt:  G + batch_j  (no critique)
        teacher samples A_i^j; student logprobs on same tokens
      Step B — review (Opus subagent, privileged source paper):
        Opus reads (G + A_i^j + source_paper.md) → critique_{i,j} (XML)

    Step C — plan generation (after 3 rounds):
      input: G + [A_i^1, A_i^2, A_i^3]
      teacher samples plan; student logprobs
      Reference plan CE supervision (μ-v4 recipe 同款)

  SDPO loss (per μ-v4 recipe):
    on distillation tokens A_i^j (teacher vs student logprobs, advantage clip 5.0)
    on plan tokens (no critic, just CE on reference plan)

  After grad step: audit checkpoint (every iter, save_every=1)
  Early stop trigger: audit-drop ≥ 2 over 2 consecutive audits (μ-v4 was ≥3)

Production: best-of-early-iter (iter 3-5 highest audit)
```

## Q1-Q9 lock summary

| Q | Decision | Rationale |
|---|---|---|
| Q1 | τ baseline first; Gate 1 must pass before launching κ | catch distillation-pipeline degradation early |
| Q2 | κ from fresh base Qwen3-30B (NOT warm-start μ-v4) | clean & comparable; μ-v4 prompt distribution differs |
| Q3 | **Path Y**: oracle source = `oracle_v2 medium.md` (181 items, 6-cat) | physical context constraint (final.md 116K tokens > 40K window); medium = same pool as σ's slim subset → clean σ vs τ comparison |
| Q3.5 | size-matched: τ output ~2200 tokens ≈ σ slim ~2100 tokens | fair token budget |
| Q4 | D-2 distillation (NOT selection) | learnable abstraction is the paper claim, selection is a degenerate sub-case; SDPO signal richer |
| Q4.1 | 3 batches (60/60/61), shuffled with `seed=κ_seed=43` (and `τ_seed=42`) | per-batch token budget 12K, fits 40K context with 24K margin |
| Q4.2 | 3 rounds **INDEPENDENT** — round j does NOT see A_1..A_{j-1} | avoid anchoring bias + SDPO cross-round signal pollution; matches Phase 2 architectural parallel (μ-v4 累积是 LoRA 不是 prompt) |
| Q4.3 | Plan generation step DOES see [A_1, A_2, A_3] concat | covers full oracle exposure; instruction includes "去重+整合" mitigates I-mode redundancy |
| Q5 | Critic on each A_i (option A); fallback to C if Gate 3 fails | option B (plan-only critique) gives no SDPO signal to distillation step → equivalent to Phase 2; option A targets the actual learnable action |
| Q5.1 | Critic format: XML (missing_critical / noise / faithfulness / improvement); Opus subagent + privileged source paper | reuses Phase 2 critic shape, adapts fields to distillation |
| Q6 | τ baseline = frozen + NO critic + 3 round independent + plan from concat | parallel σ pipeline, only oracle-source variable differs |
| Q7 | Single goal: TTT-Discover only | cross-goal validation = Session B 走线 B; Phase 3 = mechanism check |
| Q8 | Budget: ~15 hr Tinker + Opus via subagent only (zero OpenRouter) | per `feedback_minimize_openrouter.md` strengthening 2026-04-27 |
| Q9 | κ training: μ-v4 recipe (lr=5e-5, 4 grad steps, n_iter=20, save_every=1, eval_every=1) | recipe locked in `SDPO_RECIPE_v1.md`, validated 28.00/45 |
| Q9.1 | Early stop: audit-drop ≥ 2 over 2 audits (μ-v4 was ≥3, Phase 3 紧化) | retrieval may introduce faster cliffs |
| Q9.2 | Production: best-of-early-iter (iter 3-5 highest audit checkpoint) | μ-v4 same selection rule |
| Q10 | Word target: **600-750** (matches Phase 2 `_PLAN_FOOTER`); Step 7 adds τ-1200 ablation | clean cross-phase comparability vs Phase 2; reference plan ~1200 words → known T3/U5 ceiling (N3.L2); ablation quantifies length sensitivity for Phase 4 decision |

## Decision Gates

### Gate 1: τ baseline verdict (after τ run + audit)

| τ vs σ (σ=25.25) | Verdict | Action |
|---|---|---|
| τ ≥ 26.25 | CONFIDENT — pipeline > Opus slim | proceed κ training, expect κ-peak ≥ 27 |
| 24.25 ≤ τ ≤ 26.25 | NEUTRAL — pipeline ≈ slim | proceed κ training, mandatory pairwise verification |
| τ < 23.25 | STOP — pipeline degraded | DO NOT train κ; debug distillation prompt / batch / shuffle seed |

### Gate 2: κ training mid-iter monitors

| Signal | Trigger | Action |
|---|---|---|
| Healthy | mean_adv ∈ [0.2, 0.6], pos_frac ∈ [0.6, 0.9], audit not dropping | continue |
| mean_adv collapse | mean_adv < 0.1 for 3 consecutive iters | early stop, take prev checkpoint |
| audit drop trigger | audit-drop ≥ 2 over 2 audits | early stop, roll back to prev checkpoint |
| catastrophic crash | audit drops > 5 in 1 iter | EMERGENCY STOP, prev iter is production |

### Gate 3: κ-peak verdict (after audit)

| κ-peak vs τ vs σ | Verdict | Action |
|---|---|---|
| κ-peak ≥ τ + 2 AND κ-peak ≥ σ + 3 | STRONG SUCCESS | proceed Gate 4, paper headline confirmed |
| τ + 1 ≤ κ-peak < τ + 2 | MODERATE | proceed Gate 4; need ≥6/8 vs τ AND ≥6/8 vs σ to claim success |
| τ - 1 ≤ κ-peak < τ + 1 | TRAINING NEUTRAL | SDPO no gain; consider Q5 fallback option C retry, or pivot framing to "pipeline-as-contribution" |
| κ-peak < τ - 1 | TRAINING FAILED | retry option C; if still fail, ship τ + ablation only |

### Gate 4: κ pairwise corroboration

(`kappa_pairwise_v1.py`, same protocol as `mu_v4_pairwise_v1.py`)

| Matchup | Threshold | Verdict |
|---|---|---|
| κ vs σ (PRIMARY) | κ wins ≥ 6/8 | PASS — paper headline confirmed |
| κ vs σ | κ wins 4-5/8 | FLAG — n=16 follow-up |
| κ vs σ | κ wins ≤ 3/8 | INVERSION — Phase 3 claim retracted |
| κ vs τ | κ wins ≥ 6/8 | confirms training adds value over frozen distillation |
| κ vs τ | κ wins ≤ 3/8 | inverts — Phase 3 contribution = pipeline only, not SDPO |
| κ vs μ-v4 | informative side check | not paper-blocking |

## Q10 word-target rationale (keep 600/750, defer 1200 to ablation)

Discovered 2026-04-27 mid-Step-2a: Phase 2 prompts (`mu_prompts_v1._PLAN_FOOTER`,
`baseline_frozen_v2._PLAN_FOOTER`, `mu_prompts_v2._PLAN_FOOTER`) all set
`Target 600 words, max 750 words` — **half of reference plan (~1200 words)**. No
documented rationale; carried over from D3 conventions.

**Implication**: T3 (compute accounting) and U5 (reproducibility) audit
dimensions are systematically capped at this length — model lacks room to commit
operational specifics. Phase 2 mean T3 across baselines stays ≤ 2.5/5; even
ε (235B+ref) at T3=4.00 / σ at T3=2.00 / μ-v4 at T3=2.1 (already in N3.L2).

**Why we do NOT re-baseline Phase 2 at 1200**:
1. Phase 2 mechanism claims (μ-v4 +2.75 over σ, 8-baseline ranking, pairwise
   20/24, Hübotter §4 instability) are **internally valid at 600/750** —
   relative comparisons under fixed target are sound. Width is a constraint, not a bug.
2. Re-baseline cost: ~30 hr Tinker + ~$200 Opus subagent + 4-5 days wall +
   invalidates Session B 走线 B (μ-v4 sampler dependency) + rewrites F1-F6 docs
3. Estimated absolute lift if we DID re-baseline: T3 +0.4-0.7, U5 +0.4-0.6;
   σ rises in tandem with μ-v4 → headline gap unchanged or slightly compressed
4. Better experimental design: ablate length sensitivity once with Phase 3 τ
   (cheap), THEN decide if Phase 4 needs full re-baseline

**Decision**: Phase 3 (τ + κ) keeps **600/750 target verbatim** (Phase 2 footer
strings reused unchanged). Step 7 adds **τ-1200 ablation** as final
deliverable: 1 extra τ run with `distillation_max_tokens=1500` and
`plan_max_tokens=4096` (no change), `_PLAN_FOOTER` swapped to "Target 1200,
max 1500", + 1 audit ($5, ~1 hr). If τ_1200 ≫ τ_600 (gap > 1.5), trigger
Phase 4 baseline re-runs. If τ_1200 ≈ τ_600 (gap < 0.5), Phase 2/3 numbers
stay valid; just disclose as Limitations item.

**Paper Limitations text**: see N3.L2 (existing) + new L11 stub: "All Phase 2/3
plans capped at 600-750 words for cross-phase comparability with Phase 2 σ
baseline. Reference plan is ~1200 words; this systematically caps T3/U5
dimensions. Phase 4 ablation (τ-1200) tested length sensitivity → [verdict TBD]."

## Q3 reframe rationale (medium > final, ML-scientist take)

Original Q3 lock specified `oracle_v2 final.md` (580 items) as Path Y source.
Discovered 2026-04-27 during Step 0 implementation:

- final.md = 116K tokens; Qwen3-30B-A3B context = 40K tokens
- Cannot fit even one batch (200/200/180 → ~38K tokens each, leaves no room
  for goal/output/critique)
- The oracle build doc itself self-discloses this: "Slim variant filtered from
  full ttt_discover_oracle_v2.md (580 items, ~115K tokens, **too large for
  Qwen3-30B-A3B 40K context**)"

Reframe: switched to `medium.md` (181 items, ~37K tokens, Opus relevance ≥2
filter). 3 batches × 60 items × ~12K tokens fits 40K context with 24K margin.

**This is not a compromise — it's the more correct experimental design**:

1. **Clean σ vs τ comparison**: σ uses slim (87 items, relevance ≥3 subset of
   medium); τ uses full medium (181 items, relevance ≥2). Both pull from the
   SAME pool — σ is one specific Opus-curated subset, τ is the 30B's dynamic
   subset. **Same base, only selection method differs**. final would confound
   "selection method" with "65% rel=1 noise filtering capability".

2. **Information theory**: 181 × 200 tokens ≈ 37K of content; selection space
   C(181, 90) ≈ 10^53 — plenty for 30B to demonstrate selection capability.
   final's extra 400 items are Opus rel=1 distractors, not signal.

3. **Build experiments incrementally**: Phase 3 = clean mechanism check (medium
   pool); Phase 4 stretch = noise filter test (final or raw papers). Don't
   conflate two ML questions in one experiment.

4. **Methodological honesty**: paper writes "We retrieve from `oracle_v2_medium`,
   the relevance-filtered pool of 181 items (the largest set fitting Qwen3-30B-A3B's
   40K context). Broader-pool noise filtering is orthogonal future work." Reviewers
   accept this.

## Files (per namespace lock)

| File | Status |
|---|---|
| `src/co_scientist/d5_abstract_retrieve_refine/oracle_batch_helper_v1.py` | DONE (Step 0) |
| `src/co_scientist/d5_abstract_retrieve_refine/kappa_prompts_v1.py` | TODO Step 2a |
| `src/co_scientist/d5_abstract_retrieve_refine/train_tau_v1.py` | TODO Step 2a |
| `src/co_scientist/d5_abstract_retrieve_refine/train_kappa_v1.py` | TODO Step 3 |
| `src/co_scientist/d5_abstract_retrieve_refine/kappa_pairwise_v1.py` | TODO Step 3 (skeleton) / Step 6 (use) |
| `paper_materials/findings/F8_phase3_distillation_pathway.md` | TODO Step 7a |
| `paper_materials/experiments/E7_phase3_kappa_results.json` | TODO Step 7a |
| `paper_materials/methodology/M6_distillation_pipeline.md` | TODO Step 7a |
| `paper_materials/next_steps/N4_phase4_full_eval.md` | TODO Step 7a (conditional Gate 4 PASS) |

## Cost ledger (Tinker + Opus subagent only; zero OpenRouter)

| Step | Tinker | Opus subagent | Wall |
|---|---:|---:|---:|
| 0 (pre-flight) | $0 | $0 | 30 min |
| 1 (run conf τ) | $0 | $0 | 10 min |
| 2 (τ + audit) | free | ~$5 | 1 hr |
| 3 (Gate 1 + κ scaffold) | $0 | $0 | 2 hr |
| 4 (run conf κ) | $0 | $0 | 10 min |
| 5 (κ training) | free (~15 hr GPU) | ~$150 (24 critic + 5 audit × 5 iter) | 15 hr |
| 6 (κ pairwise) | $0 | ~$15 (3 matchups × 8 pairs) | 30 min |
| 7 (doc sync) | $0 | $0 | 2 hr |
| **Total** | **free** | **~$170** | **~21 hr (~3 days wall)** |
