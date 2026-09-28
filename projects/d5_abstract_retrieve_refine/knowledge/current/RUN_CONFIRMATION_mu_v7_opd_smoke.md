# Run Confirmation — μ-OPD Smoke (Day 1 of μ-OPD plan)

**Date filed**: 2026-04-27 PM
**Plan**: `/home/silas/.claude/plans/snug-dreaming-yao.md` (μ-OPD redesign)
**Status**: AWAITING USER "yes go" approval

> **⚠ TERMINOLOGY NOTE (added 2026-04-27 PM)** — this doc was filed with
> "canonical SDPO/OPD" / "Hübotter option (c) HER" framing. After paper-grade
> verification (WebFetch of arxiv 2601.20802 Hübotter and 2601.18734 Zhao):
> - μ-v7-opd (this run's trainer) is a **D5 in-house on-policy IS-loss
>   variant**; closest published comparable is OPSD (Zhao 2601.18734) Table 3
>   sampled-token policy-gradient variant. NOT canonical Hübotter SDPO.
> - "option (c) HER" applied to μ-v4 was D5 internal speculation; the Hübotter
>   paper has no such variant. μ-v4 is a **D5 in-house off-policy IS-loss
>   variant**.
> The empirical claim of this run (probe whether on-policy sampling rescues F1
> prefix-prior lock-in) is unchanged. Disambiguation:
> `knowledge/current/CANONICAL_NAMING_REFERENCE.md`.

## Why this run

Tests whether canonical SDPO/OPD architecture (sample student on-policy, teacher
provides logprob supervision) breaks F1 (prefix-prior lock-in) and F2 (off-manifold
gradient) — both of which were diagnosed by Stage A + Stage B as upstream causes of
F3/F4/F5/F7/F9 NULLs in the prior Hübotter option (c) HER variant.

Key diagnostic: PUCT presence in EVAL plans across iters. μ-v4 (option c) was 0/8
across iters 0-7. If μ-OPD recovers PUCT presence ≥ 1/8 by iter 4, F1 mechanism is
on-manifold-fixable via on-policy sampling alone.

## Run identity

- **Run name**: `mu_v7_opd_smoke_2026_04_29`
- **Greek-letter label**: μ-v7 (OPD variant; production μ-v4 unchanged)
- **Run dir**: `projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v7_opd_smoke/`
- **Estimated wall-clock**: ~5-6 hr (4 iter × ~75 min/iter, similar to μ-v4 cadence)
- **Estimated cost**: **~$30** (Tinker compute $20 + 4 critic calls $5 + 4 audit calls × 8 plans = ~$5)

## Pipeline diagram (OPD architecture)

```
goal+oracle (NO critique) ────────┐
                                  ├──> [Qwen3-30B-A3B + LoRA r=64, lr=5e-5] ──> 8 plans (ON-POLICY student)
                                  │                                                  │
                                                                                     ▼
goal+oracle+prev_critique ────────┐                            student_lp (from seq.logprobs, on-policy)
                                  │                                                  │
                                  ├──> [compute_logprobs(teacher_input + plan)] ──> teacher_lp per token
                                  └─                                                 │
                                                                                     ▼
                                                  A_t = clamp((t_lp − s_lp) · scale, [−5, 5])
                                                                                     │
                                                                                     ▼
                                              forward_backward(student_input + plan, A_t, IS) × 4 microbatches
                                                                                     │
                                                                                     ▼
                                                              critic call (Opus, file-bus)
                                                                                     │
                                                                                     ▼
                                                              eval audit (8 plans, async)
```

Difference from μ-v4 (option c HER):
- v4 sampled TEACHER under teacher_input (off-trajectory for student); supervised STUDENT_lp
- v7 OPD samples STUDENT under student_input (ON-trajectory); supervises with TEACHER_lp
- Same advantage formula `A_t = clamp(scale·(t_lp − s_lp), [−5, 5])` — only the source of t_lp/s_lp swapped

## Verbatim prompt template (UNCHANGED from μ-v4)

```python
# build_student_prompt:
"I will provide you a research scenario and a set of methodological"
" patterns extracted from a high-quality research plan on this same"
" scenario. Use the patterns to guide..."
+ "\n\nScenario: {goal}"
+ "\n\n# Methodological patterns to apply\n\n{oracle}"
+ _PLAN_FOOTER

# build_teacher_prompt: above + "\n\n# Critique of prior attempt..." + critique_xml
```

## What model sees / doesn't

| Variable | In prompt? | Source |
|---|---|---|
| goal | YES | `dataset/research_goal.txt` (TTT-D) |
| oracle abstraction (v2 slim) | YES | `data/oracles/oracle_v2_2026_04_26_build/slim.md` |
| reference plan | NO | (anchor_ce_weight=0.0; not used) |
| previous-iter critique | YES (in teacher_input only; student_input never has it) | `critic_responses/iter_N.json` (real Opus critic via daemon) |
| source paper full text | NO (only critic sees it) | `data/source_paper/v2.md` |

## Eval metric + comparison rule

**Primary**: `audit_v3_isolated.py` 9-dim mean /45, n=8 EVAL plans/iter × 4 iter.

**Diagnostic**: multi-regex content density on each EVAL plan, reusing `f4_stage_b_critique_probe.py` regex set:
- PUCT, Q(s,a), J_β, V(s), concrete_hparam, selection_mention

**Comparison anchors** (do NOT re-baseline):
- ξ baseline 15.62 (frozen + goal only) — μ-OPD must beat (sanity)
- σ baseline 25.25 (frozen + slim oracle) — μ-OPD smoke must beat (basic learning)
- μ-v4 production 28.00 (option c, peak iter 4) — μ-OPD smoke target to match or beat
- ε ceiling 33.62 (frozen 235B + reference) — out of reach for 30B

## Decision rule (pre-registered)

| μ-OPD smoke best iter audit /45 | PUCT presence | Verdict | Action |
|---:|:---:|---|---|
| ≥ 28.0 | ≥ 1/8 | STRONG PASS | Full μ-OPD launch + redo Phase 4a/5 |
| 26.0-27.9 | ≥ 1/8 | PASS | Full μ-OPD launch; gate Phase 4a/5 on full result |
| ≥ 26.0 | 0/8 | PARTIAL PASS | Full μ-OPD launch but flag mechanism still binds; ML scientist reconsult |
| 25.5-25.9 | any | MARGINAL | Reconsult; may need SFT warm-start fallback |
| < 25.5 | any | FAIL | On-policy garbage early iter; SFT warm-start fallback (50 SFT steps on μ-v4 buffer first) |

Diagnostics that must be sane before declaring PASS:
- mean_adv per iter is finite (no NaN); ideally 0.05-0.5 range
- pos_frac ≈ 0.5-0.8 (advantage signal balanced)
- mean_abs_adv > 0.1 (non-trivial gradient signal)
- audit_drop_threshold (5.0) NOT triggered (no early stop)

## Cost breakdown

- Tinker compute: 4 iter × ~12 min × ~$0.20/min = ~$10
- Tinker logprobs (teacher on student tokens, 8 plans × 4 iter): ~$5
- Opus critic subagent: 4 calls × ~$1 = ~$4
- Opus audit subagent: 4 iter × 8 parallel × ~$0.30 = ~$10
- **Total: ~$30**

## Launch command

```bash
source shared/tools/use_api_profile.sh new
PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.train_mu_v7_opd \
    log_path=projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v7_opd_smoke \
    n_iter=4 \
    eval_every=1 \
    save_every=1 \
    n_grad_steps_per_iter=4 \
    learning_rate=5e-5 \
    opd_mode=True \
    audit_drop_threshold=5.0 \
    today_date=2026_04_29 \
    seed=42
```

**Daemon required**: separate Claude Code window running
`projects/d5_abstract_retrieve_refine/scripts/opus_critic_audit_daemon.md`
(handles critic calls + async audits via file-bus). ONE daemon per run dir.

## Risks (carried from plan)

| Risk | Severity | Mitigation |
|---|---|---|
| On-policy garbage early iter | MEDIUM | Smoke gate FAIL → SFT warm-start fallback prepared |
| Teacher logprob compute on long prefix (~20k tokens) OOM/slow | MEDIUM | Same magnitude as v4 worked fine; parallel batching unchanged |
| Iter-0 advantage too small (s_lp ≈ on-policy lp under same model) | LOW | Iter-0 small adv is by design; check iter 1+ has signal |
| F3 cliff persists (SDPO-family-wide) | MEDIUM-HIGH | 4-iter smoke too short to confirm/refute; full run will tell |
| audit_v3 measures style not content → μ-OPD audit may be ≈ μ-v4 even if PUCT recovers | LOW-MEDIUM | Multi-regex content density is the secondary diagnostic; PUCT presence is the load-bearing signal |

## Pre-launch sanity checklist

- [x] `train_mu_v7_opd.py` syntax validated (`ast.parse` passed)
- [x] Module imports clean (chz Config defaults verified)
- [x] Diff vs v4 = +123 net lines (within plan's 120-150 estimate)
- [x] `opd_mode=True` default; `opd_anchor_clip=5.0`
- [x] Reuses `OpusCriticClient`, `OpusAuditClient`, `build_sdpo_datum`, `audit_v3_isolated.py` unchanged
- [x] Run dir avoids main session's `tau_v4_bon_*` namespace
- [ ] Daemon launched in separate Claude Code window — **AWAITING user setup**
- [ ] User "yes go" approval — **AWAITING**

---

**Action requested**: User approves + starts daemon, replies "yes go" or specifies changes.
