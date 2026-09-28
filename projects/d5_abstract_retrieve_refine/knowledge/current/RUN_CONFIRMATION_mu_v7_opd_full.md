# Run Confirmation — μ-OPD Full 16-Iter Run (Day 2-3)

**Date filed**: 2026-04-27 PM (after smoke + pairwise STRONG PASS)
**Plan**: `/home/silas/.claude/plans/snug-dreaming-yao.md` (μ-OPD redesign, task-017)
**Status**: AWAITING USER "yes go" approval

> **⚠ TERMINOLOGY NOTE (added 2026-04-27 PM)** — this doc uses "OPD" /
> "canonical SDPO/OPD" terminology. After paper-grade verification:
> μ-v7-opd is a **D5 in-house on-policy IS-loss variant**; closest published
> comparable is OPSD (Zhao 2601.18734) Table 3 sampled-token policy-gradient
> variant. NOT canonical Hübotter SDPO (which uses full-vocab/top-K KL).
> Disambiguation: `knowledge/current/CANONICAL_NAMING_REFERENCE.md`.

## Why this run

Smoke (4 iter) + pairwise (16 pairs) gave **STRONG PASS**:
- Audit /45 trajectory monotonic 20.25 → 27.75 (no cliff in 4-iter window)
- Pairwise OPD vs μ-v4 iter 3: **8-0**
- Pairwise OPD vs μ-v4 iter 4 production: **7-1**
- Combined: 15/16 OPD wins (93.8%)

Full run has THREE primary deliverables:
1. **F3 cliff test**: μ-v4 cliffed at iter 5 (-12.4 audit). Does μ-OPD avoid this through iter 8+? If yes → spotlight F3-fix finding.
2. **Production checkpoint**: identify peak iter (analogous to μ-v4 iter 4) to use as μ-OPD weights for downstream Phase 4a / Phase 5 redo.
3. **Larger sample for paper**: 16 iter × 8 EVAL plans = 128 total EVAL plans for content-density analysis (vs smoke 32). PUCT/J_β recovery rate scales with sample.

## Run identity

- **Run name**: `mu_v7_opd_full_2026_04_29`
- **Greek-letter label**: μ-v7 (OPD full production candidate)
- **Run dir**: `projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v7_opd_full/`
- **Estimated wall-clock**: ~10-14 hr (16 iter × ~40-50 min/iter accounting for critic round-trip)
- **Estimated cost**: **~$80-100** (Tinker compute $40 + 16 critic calls × $1.5 = $24 + 16 audit calls × $1 = $16)

## Pipeline diagram (UNCHANGED from smoke; opd_mode=True)

```
goal+oracle (NO critique) ────────┐
                                  ├──> [Qwen3-30B + LoRA r=64, lr=5e-5] ──> 8 plans (ON-POLICY student)
                                  │                                                  │
                                                                                     ▼
goal+oracle+prev_critique ────────┐                            student_lp (from seq.logprobs)
                                  │                                                  │
                                  ├──> [compute_logprobs(teacher_input + plan)] ──> teacher_lp per token
                                                                                     │
                                                                                     ▼
                                                  A_t = clamp((t_lp − s_lp) · scale, [−5, 5])
                                                                                     │
                                                                                     ▼
                                              forward_backward(student_input + plan, A_t, IS) × 4 microbatches
```

## Configuration

```python
# Same as smoke EXCEPT n_iter and audit_drop_threshold
n_iter = 16              # extends beyond smoke 4 iter
eval_every = 1           # per-iter audit
save_every = 1           # per-iter checkpoint for rollback to peak
audit_drop_threshold = 5.0   # early-stop if single-iter audit drop ≥ 5 (catches cliff)
n_grad_steps_per_iter = 4
learning_rate = 5e-5
opd_mode = True
opd_anchor_clip = 5.0
seed = 42                # match smoke seed for reproducibility on iter 0-3 check
```

**No ProRL ref-reset added**: ML scientist counter-argument noted ref-reset interval has no smoke evidence and `reset_optimizer_state=True` is itself disruptive. Default = trust audit_drop_threshold to catch cliff. If cliff DOES appear at iter 5-7 (matching μ-v4 location), that's the F3-is-SDPO-family-wide finding (also publishable).

## What model sees / doesn't (UNCHANGED from smoke)

| Variable | In prompt? | Source |
|---|---|---|
| goal | YES (both teacher + student inputs) | `dataset/research_goal.txt` |
| oracle abstraction (v2 slim) | YES | `data/oracles/oracle_v2_2026_04_26_build/slim.md` |
| reference plan | NO (anchor_ce_weight=0.0) | (not used) |
| prev critique | YES (teacher_input only); student_input never has it | `critic_responses/iter_N.json` |
| source paper | NO (only critic sees it) | `data/source_paper/v2.md` |

## Eval metric + decision rule (post-run)

**Primary**: `audit_v3_isolated.py` 9-dim mean /45 per iter, identify peak.

Re-run audit on full eval_rollouts via STRICT 1-plan/subagent dispatch (lesson from smoke: dispatcher subagent claimed Task tool unavailable; this time use the same per-pair-Agent pattern that worked for pairwise).

**Diagnostic**: multi-regex content density — track PUCT / J_β / Q(s,a) / hparam / selection_mention across 128 EVAL plans. Headline metric for paper: "EVAL plan content recovery rate at peak iter."

**Pairwise corroboration**: AFTER full run, 8 anonymized pairs μ-OPD peak vs μ-v4 iter 4 production. M8 strict 1-pair/subagent.

### Decision matrix (post full run)

| Outcome | Action |
|---|---|
| Peak iter audit ≥ 30 + PUCT ≥ 3/8 + no cliff through iter 8 | **SPOTLIGHT signal** — Phase 4a / 5 redo + paper integration as PRIMARY result |
| Peak iter audit 28-30 + PUCT 1-2/8 + cliff broken | PASS — F3 fix is the paper finding; full Phase 4a redo |
| Peak iter audit ≈ 27-28 + same cliff at iter 5-7 | PARTIAL — F1 fixed (signal validated by pairwise), F3 NOT fixed (SDPO-family-wide); paper as 2-finding result |
| Peak iter audit < 27 OR collapse | UNEXPECTED — reconsult ML scientist |

## Cost breakdown

- Tinker compute: 16 iter × ~25 min × ~$0.20/min = ~$40 (some iter savings due to caching)
- Tinker compute_logprobs (teacher on student tokens): ~$10
- Opus critic: 16 calls × ~$1.50 = $24
- Opus audit: 16 calls × 8 plans × ~$0.30 = ~$10 (async; daemon parallel)
- Opus pairwise post-run: 8 strict 1-pair = ~$3
- **Total: ~$87**

## Launch command

```bash
source shared/tools/use_api_profile.sh new
mkdir -p projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v7_opd_full
PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.train_mu_v7_opd \
    log_path=projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v7_opd_full \
    n_iter=16 \
    eval_every=1 \
    save_every=1 \
    n_grad_steps_per_iter=4 \
    learning_rate=5e-5 \
    opd_mode=True \
    audit_drop_threshold=5.0 \
    today_date=2026_04_29 \
    seed=42
```

**Daemon REQUIRED** (will spawn as background Agent in this session, same pattern as smoke).

## Risks (carried forward + new)

| Risk | Severity | Mitigation |
|---|---|---|
| F3 cliff appears at iter 5-7 (matching μ-v4) | MEDIUM-HIGH | audit_drop_threshold=5 halts run; salvage iter 4 checkpoint as production. Result is still publishable (dual finding: F1 fixed + F3 SDPO-family-wide) |
| Trainer + daemon both running 12+hr — one might die mid-run | MEDIUM | Resume-from-checkpoint logic in train_mu_v7_opd.py is intact (inherited from v4); daemon Agent has 6hr hard limit, may need re-launch mid-run if training exceeds it |
| Audit dispatcher (post-run, 128 plans) hits same M8 anchoring confound as smoke | LOW | Use the per-pair-Agent dispatch pattern that worked for pairwise (one Agent per plan, parallel) — bypasses dispatcher's nested-Task-tool issue |
| PUCT presence stays low (1-2/8 across iter) → not as strong as hoped | LOW-MEDIUM | Acceptable: pairwise already validates ranking; PUCT is one diagnostic among many. Multi-regex density average is the paper-reportable number |
| iter 5-7 are wasted compute if cliff hits early | LOW | audit_drop_threshold catches it; saves the rest of the run cost |

## Pre-launch sanity checklist

- [x] Smoke + pairwise both PASSED (8-0 + 7-1)
- [x] train_mu_v7_opd.py syntax + import verified (Day 0)
- [x] Code unchanged since smoke (no edits between smoke and full)
- [x] Run dir avoids main session namespace
- [ ] Daemon Agent spawned in background — **WILL DO at launch**
- [ ] User "yes go" approval — **AWAITING**

---

**Action requested**: User approves, I spawn daemon + launch trainer + monitor.
