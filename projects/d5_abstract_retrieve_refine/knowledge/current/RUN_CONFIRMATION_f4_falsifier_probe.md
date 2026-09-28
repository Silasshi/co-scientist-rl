# Run Confirmation — F4 Falsifier Probe v2 (Track 0 of D4 plan)

**Date filed**: 2026-04-27 PM (v2 revision after ML-scientist subagent feedback)
**Plan**: `/home/silas/.claude/plans/snug-dreaming-yao.md` Track 0 (killer Q #1)
**Status**: AWAITING USER "yes go" approval (Stage A → gates Stage B)

## Why v2 (revision summary)

ML-scientist subagent flagged the v1 design (single 3-iter run, cold-start critique always, PUCT-only outcome) as having **decision-breaking confounds**:
1. **Cold-start-always confound**: replacing μ-v4's iter 1+ real Opus critique with cold-start changes the teacher distribution itself. A null result becomes ambiguous between "F4 is right" and "cold-start critique gives no useful gradient signal anywhere".
2. **3 iter borderline**: PICK acquired PUCT at iter 1 in μ-v4, but EVAL never did across 0-7. EVAL acquisition under masking might lag 1-2 iters.
3. **PUCT-only outcome metric is fragile**: 8 plans × 1 regex = noisy binary outcome.

v2 fixes all three + adds an **even cheaper pre-test** (offline gradient attribution on existing μ-v4 buffer logs, $3 / 30 min) that may resolve the question entirely without a new training run.

---

## Stage A — Offline F4 Attribution (`f4_attribution_offline.py`)

**Question**: in μ-v4 iter-2 trained plans, what fraction of total |advantage| mass actually lands at content-token positions (vs stylistic)?

If F4 is mechanically correct, content_grad_mass_frac should be tiny (<1%) — gradient flows through the ~88% of tokens that are stylistic, with content tokens contributing essentially nothing. If it's already 5-10%+, F4 is wrong: gradient *does* reach content tokens; the EVAL learning failure has a different cause and D3a doesn't help.

### Run identity

- **Run name**: `f4_attribution_offline_2026_04_29`
- **Run dir**: `projects/d5_abstract_retrieve_refine/runs/2026_04_29_f4_attribution_offline/`
- **Estimated wall-clock**: ~30 min (16 logprob calls × 1-2 min each)
- **Estimated cost**: **~$3** (Tinker compute only; ZERO Opus, ZERO training)

### Pipeline diagram

```
μ-v4 iter-2 sampler weights ──┐
                              ├──> [compute_logprobs(teacher_prompt + plan_tokens)] ──> teacher_lp
goal+oracle+iter-1-critique ──┘                                                            │
                                                                                           ▼
μ-v4 iter-2 sampler weights ──┐                                              advantage = clamp((t_lp − s_lp) · scale, [−clip, clip])
                              ├──> [compute_logprobs(student_prompt + plan_tokens)] ──> student_lp
goal+oracle (no critique) ────┘                                                            │
                                                                                           ▼
                                              classify each token via is_content_token regex superset
                                                                                           │
                                                                                           ▼
                                              compute content_grad_mass_frac per plan + aggregate

8 plans from μ-v4 iter=2 buffer.jsonl
```

### Decision rule

| Mean content_grad_mass_frac (n=8) | Verdict | Action |
|---:|---|---|
| < **1%** | F4_STRONGLY_SUPPORTED | Proceed Stage B (paired probe) |
| 1-5% | F4_SUPPORTED | Proceed Stage B (paired probe) |
| 5-10% | F4_MARGINAL | Reconsult ML scientist subagent; may need different mask granularity |
| ≥ **10%** | F4_WEAKENED | DO NOT pursue D3a; expand D2-only and rethink |

### Launch command

```bash
source shared/tools/use_api_profile.sh new
PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.f4_attribution_offline \
    out_path=projects/d5_abstract_retrieve_refine/runs/2026_04_29_f4_attribution_offline \
    target_iter=2 \
    mu_v4_run=projects/d5_abstract_retrieve_refine/runs/2026_04_27_mu_v4
```

(Default `iter_sampler_path` = μ-v4 iter-2 weights, hardcoded from `runs/2026_04_27_mu_v4/checkpoints.jsonl` row `{"iter": 2, ...}`.)

**No daemon needed.** No critic, no audit subagent. Pure logprob compute + classification + statistics. Output: `runs/.../f4_attribution_offline/f4_attribution_summary.json`.

### Risks (Stage A)

| Risk | Mitigation |
|---|---|
| Re-tokenizing `raw_tokens_text` may differ slightly from original BPE | Acceptable: signed-advantage signs and orders-of-magnitude are stable across BPE variants. The PRIMARY metric is mass FRACTION, robust to small token-count drift. |
| `iter_sampler_path` may be GC'd from Tinker | Verify accessibility before launch (1 logprob call). If GC'd, fall back to iter-1 sampler (one iter earlier; same conclusions). |

---

## Stage B — F4 Probe v2 Paired A/B Run (gated on Stage A verdict)

Only execute if Stage A verdict = F4_STRONGLY_SUPPORTED or F4_SUPPORTED.

### Run identity

- **Run name (Arm A)**: `f4_probe_v2_armA_mask_on_2026_04_30`
- **Run name (Arm B)**: `f4_probe_v2_armB_mask_off_2026_04_30`
- **Run dir**: `projects/d5_abstract_retrieve_refine/runs/2026_04_30_f4_probe_v2_<arm>/`
- **Estimated wall-clock**: ~90 min for both arms in parallel (4 iter × ~10-15 min/iter; needs daemon)
- **Estimated cost**: **~$20** (Tinker $10 + 4 critic calls × 2 arms × ~$1.50 + 4 audit calls × 2 arms × ~$0.50)

### Pipeline diagram

```
goal+oracle+real-Opus-critique ──┐
                                 ├──> [base Qwen3-30B + LoRA r=64, lr=5e-5, 4 iter] ──> Arm A (mask=ON)
                                 │                                                        │
                                 │                                                        ▼ multi-regex content density (PUCT|Q(s,a)|J_β|V(s)|exp(β|hparams)
                                 │
                                 └──> [same recipe, same seed, mask=OFF] ──> Arm B (mask=OFF, vanilla μ-v4 control)
                                                                              │
                                                                              ▼ multi-regex content density

per-token diagnostics: mean |adv| at masked vs unmasked positions
```

### Decision rule (Arm A vs Arm B content-token density delta)

| (A density − B density) / B density | Verdict | Action |
|---:|---|---|
| ≥ **+25%** | GO D3a | Mask demonstrably routes gradient through content tokens, lifts EVAL recall |
| +10% to +25% | INCONCLUSIVE | ML-scientist subagent reconsult |
| ≤ **+10%** | KILL D3a | Mask doesn't help; D2 is the surviving hedge |

### Multi-regex outcome (replaces PUCT-only)

```python
CONTENT_REGEXES = [
    r"PUCT|Q\(s,?\s*a\)",                 # core MCTS objects
    r"J_?β|J_beta|exp\(\s*β",             # entropic objective
    r"V\(s\)",                             # value function
    r"LoRA\s+rank\s*=?\s*\d+",             # hparam: LoRA rank
    r"\d+\s*[×x]\s*\d+",                   # batch/rollout counts (50×512)
    r"\bβ\s*=\s*[\d.]+",                   # β = constant
    r"KL\s*budget|γ\s*=\s*ln\s*2",         # KL budget reference
]
def content_density(plan_text: str) -> float:
    n = sum(1 for rx in CONTENT_REGEXES if re.search(rx, plan_text))
    return n / len(CONTENT_REGEXES)  # in [0, 1]
```

Reported per plan + aggregate per arm.

### Per-token gradient diagnostics (added per subagent feedback)

Trainer logs per iter:
- `mean_abs_adv_content` (at mask positions, even when mask is ON they get clamped to 0 — so this is meaningful for Arm B only; for Arm A this is the gradient that DID flow through)
- `mean_abs_adv_stylistic`
- `content_grad_mass_frac` (post-clamp, post-mask)

### Launch commands (Stage B, both arms in parallel)

```bash
# Arm A (mask ON) — needs daemon running for critic + audit
PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.train_mu_v4 \
    log_path=projects/d5_abstract_retrieve_refine/runs/2026_04_30_f4_probe_v2_armA_mask_on \
    n_iter=4 eval_every=1 save_every=1 \
    f4_probe_content_only_mask=True \
    audit_drop_threshold=99.0 \
    today_date=2026_04_30 seed=42

# Arm B (mask OFF, vanilla μ-v4) — same daemon
PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.train_mu_v4 \
    log_path=projects/d5_abstract_retrieve_refine/runs/2026_04_30_f4_probe_v2_armB_mask_off \
    n_iter=4 eval_every=1 save_every=1 \
    f4_probe_content_only_mask=False \
    audit_drop_threshold=99.0 \
    today_date=2026_04_30 seed=42
```

(Daemon spec: `projects/d5_abstract_retrieve_refine/scripts/opus_critic_audit_daemon.md`. ONE daemon per run dir; running both arms requires two daemons in two separate Claude Code windows — careful re: subagent contention. Alternative: run sequentially.)

### Risks (Stage B)

| Risk | Mitigation |
|---|---|
| Two daemons in parallel race on file writes | **Mitigation**: run sequentially (Arm A first; Arm B after). Adds ~90 min wall but eliminates race. |
| Mask-OFF arm produces same content_density as μ-v4 historical | Acceptable — that's the within-experiment baseline. |
| 4 iter still insufficient for EVAL acquisition | If Arm A density at iter 4 ≈ Arm B density AND no monotonic trajectory delta, that's a real null. |

---

## Total cost (both stages)

| Stage | Cost | Wall | When |
|---|---:|---|---|
| A: Offline attribution | $3 | 30 min | Day 1 PM |
| B: Paired A/B (if A passes) | $20 | 90 min sequential / 60 min parallel | Day 2 |
| **Total** | **$23** | **2-3 hr** | Day 1 PM + Day 2 |

vs original v1: $8 / 30-50 min, but with decision-breaking confounds.

## Pre-launch sanity checklist (Stage A only — Stage B awaits A's result)

- [x] `f4_attribution_offline.py` syntax validated
- [x] Module imports clean (chz Config defaults verified)
- [x] μ-v4 iter-2 sampler path resolved from `checkpoints.jsonl`
- [x] iter-1 critic_response file exists + parseable (3443 chars critique)
- [x] 8 iter=2 plans confirmed in buffer.jsonl
- [ ] User "yes go" approval — **AWAITING for Stage A**

---

**Action requested**: User approves Stage A. Stage B awaits Stage A's verdict.
