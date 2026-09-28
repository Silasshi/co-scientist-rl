# F12 — Prefix-prior lock-in: the real mechanism behind D5's NULL results

> **⚠ TERMINOLOGY NOTE (2026-04-27 PM)**: F12 references "SDPO" — this is the
> D5 in-house off-policy IS-loss variant (μ-v4 family) and the in-house
> on-policy IS-loss variant (μ-v7-opd, closest to OPSD Zhao 2601.18734
> sampled-token PG). NOT canonical Hübotter SDPO. Mechanism claim
> ("prefix-prior lock-in is the real bottleneck") is established for both D5
> in-house variants; whether canonical SDPO's full-vocab/top-K KL form has the
> same property is not tested. Disambiguation:
> `knowledge/current/CANONICAL_NAMING_REFERENCE.md`.

## Headline

After F4 gradient-flooding was falsified (F11), Stage B critique-conditioned EVAL
probe + slim_oracle keyword analysis isolated the real mechanism: **prefix-prior
lock-in**. The in-house IS-loss variants transfer structural / stylistic capability
robustly, but specific-content acquisition (which algorithm name, which formula,
which baseline) is bottlenecked by what's in the student's actual sampling prefix at
EVAL time. Content present in the oracle prefix can be learned; content present only
in critique text (teacher-only context) cannot.

**Smoking-gun evidence**:

| Algorithm | slim_oracle mentions | μ-v4 EVAL recall | μ-OPD EVAL recall (iter 5) |
|---|---:|---:|---:|
| SIFT (in oracle) | 9 | high | high |
| GRPO (in oracle) | 23 | high | high |
| **PUCT** (only in critique text) | **0** | **0/64 across 8 iters** | **0/128 across 16 iters** |
| MCTS (only in critique text) | 0 | 0 | 0 |
| UCB (only in critique text) | 0 | 0 | 0 |
| **J_β** (entropic; in oracle methodology) | present | **0/64 across 8 iters** | **3/8 at iter 5 (recovered)** |

The pattern is unambiguous: regardless of how loud the teacher critique mentions PUCT (it does — Opus critique iter 1 said "PUCT-style initial-state selection with the correct functional form Q(s)+c·P(s)·sqrt(1+T)/(1+n(s))"), and regardless of whether SDPO gradient flows through PUCT positions in teacher rollouts (it does, at proportional magnitude per F11), **the student EVAL distribution defers to oracle prefix tokens** for specific algorithm choice.

## Mechanism (refined from F4's gradient-flooding hypothesis)

When student samples at EVAL time:
- Prefix = goal + oracle (no critique by construction)
- Sampling trajectory enters tokens reachable from `p(token | goal+oracle, prefix_<t)`
- Oracle prominently mentions SIFT (9×) and GRPO (23×) → these are high-prior in the relevant conditional-distribution sub-trees
- PUCT, never in oracle, never enters the high-prior sub-tree of the student's actual sampling distribution
- SDPO gradient on teacher tokens (which contain PUCT under critique context) updates parameters such that `p(PUCT | teacher_prefix_with_critique)` rises — but this prefix never appears at EVAL time. The shift to `p(PUCT | student_prefix_no_critique)` is too small to escape oracle-anchored modes (SIFT, GRPO).

Stage B (`runs/2026_04_29_f4_stage_b_critique_probe/`) tested whether adding a generic critique cue at EVAL would unlock PUCT. Result: **selection_mention 1/8 → 6/8** (model responds to critique cue) BUT **PUCT 0/8 → 0/8** (still defers to oracle algorithms — chose SIFT 8/8 and GRPO 8/8). Critique cue activates the "talk about action-selection" sub-tree but specific algorithm choice is governed by oracle prior, not critique tokens.

## How F11 + F12 unify D5's NULL results

| NULL | Original F4 explanation | F11+F12 reframe |
|---|---|---|
| F3 multi-round cliff (iter 5) | Stylistic tokens flood gradient → over-overshoot collapse | Off-manifold gradient toward teacher tokens (option c HER specific) accumulates, model drifts off student manifold → cliff |
| F7 cross-goal NULL (Phase 4a) | Style transfers, content doesn't (no mechanism specified) | Each new paper has its own oracle; oracle-prefix-determined content doesn't transfer |
| F9 continual NULL (Phase 5 4-cell grid) | Saturation + Adam dynamics | Continual SDPO compounds prefix-prior lock-in (each chained training stage redirects student distribution to a prefix that doesn't contain new paper's content) |

The unified mechanism: **SDPO option (c) HER's gradient is off-manifold; the (small) shift toward critique-conditioned teacher tokens is overwhelmed at sample time by the oracle-anchored student prefix.**

## Why canonical OPD (F13) helps

By switching sampler from teacher to student (`opd_mode=True`), the gradient flows through tokens the student actually produces at EVAL — on-manifold updates accumulate without off-manifold drift, AND oracle-present content (J_β) becomes learnable because the student's actual sampling trajectory does enter J_β-adjacent sub-trees. Oracle-absent content (PUCT) remains bound by prefix-prior lock-in even with canonical OPD — the lock-in mechanism is independent of training architecture once F2 (off-manifold) is fixed.

## Implications for D5's broader narrative

1. **Inference-time prefix engineering is a primary lever** (matches main session's F8 v2 finding for τ_v4 inference pipeline).
2. **Training-architecture choice affects content acquisition for oracle-PRESENT content** (F13: μ-OPD recovers J_β where (c) HER doesn't).
3. **Phase 3 retrieve-then-generate (extending oracle prefix dynamically) is the lever for oracle-ABSENT content** (e.g., PUCT) — main session's territory.
4. The full pipeline = retrieval (extend oracle prefix) + canonical OPD (on-policy gradient through prefix-induced trajectories) — both directions necessary for full content recovery.

## Data pointers

- Stage A (F11 falsification): `runs/2026_04_29_f4_attribution_offline/`
- **Stage B**: `runs/2026_04_29_f4_stage_b_critique_probe/` (script `src/co_scientist/d5_abstract_retrieve_refine/f4_stage_b_critique_probe.py`)
- Slim oracle keyword counts: `data/oracles/oracle_v2_2026_04_26_build/slim.md` (grep `SIFT`/`GRPO`/`PUCT`/`MCTS`)
- F4 original: `paper_materials/findings/F4_critique_token_blindness.md` (mechanism falsified)
- F13 (OPD fix): `paper_materials/findings/F13_mu_opd_canonical_architecture.md`
