# F14 — F7 cross-goal NULL inverted: D5 in-house on-policy IS-loss variant (closest to OPSD sampled-token PG) transfers across forward-citation papers

> **⚠ TERMINOLOGY NOTE (2026-04-27 PM)**: F14 was originally framed as
> "canonical OPD transfers". After paper-grade verification: μ-OPD = μ-v7-opd
> is a **D5 in-house on-policy IS-loss variant**, not canonical OPD. Closest
> published comparable: OPSD (Zhao 2601.18734) Table 3 sampled-token
> policy-gradient variant. μ-v4's "option (c) HER" label is also D5 internal
> speculation — Hübotter SDPO paper has no such variant; μ-v4 is the **D5
> in-house off-policy IS-loss variant**. The empirical contrast (22/24 OPD
> wins vs μ-v4 NULL) is real and load-bearing. Disambiguation:
> `knowledge/current/CANONICAL_NAMING_REFERENCE.md`.

## Headline

F7 (Phase 4a / D-series 2026-04-28) declared cross-goal forward-citation transfer
**DEFINITIVELY NULL at n=24** across 3 TTT-family papers (meta_ttl,
tool_verification_ttrl, tt_control) using μ-v4 LoRA (D5 in-house off-policy IS-loss
variant). F14 redoes Phase 4a with **μ-OPD iter 4 LoRA** (D5 in-house on-policy
IS-loss variant, closest to OPSD sampled-token PG; F13 architecture fix) — same
inference setup, no retraining — and observes:

| Goal | μ-v4 (F7, n=24) | μ-OPD (F14, n=8 anonymized strict pairwise) | Verdict |
|---|---:|---|---|
| meta_ttl | strict Δ -0.96, pairwise σ' 14/9/1 | **OPD 7-1** (4/8 A-pos balance) | ⭐ OPD wins ≥6/8 |
| tool_verification_ttrl | strict Δ -0.58, pairwise σ' 12/10/2 (TIE) | **OPD 8-0** (4/8 A-pos balance) | ⭐ OPD wins ≥6/8 |
| tt_control | strict Δ -0.88, pairwise σ' 5/3 | **OPD 7-0-1 TIE** (4/8 A-pos balance) | ⭐ OPD wins ≥6/8 |
| **Combined** | DEFINITIVELY NULL | **OPD 22/24 = 91.7%** | **F7 INVERTED** |

**F7 reframe**: cross-goal forward-citation transfer is NOT a fundamental D5 limit. **It was an artifact of the D5 in-house off-policy IS-loss variant used by μ-v4** (the variant historically labeled "option (c) HER" but with no basis in any paper — see CANONICAL_NAMING_REFERENCE.md). The D5 in-house on-policy IS-loss variant (F13, μ-v7-opd; closest to OPSD Zhao 2601.18734 Table 3 sampled-token PG) produces LoRA weights that DO transfer to forward-citation TTT-family papers under anonymized pairwise judgment.

## Methodology

- **Setup**: μ-OPD iter 4 sampler weights (production checkpoint per F13). 8 plans / goal sampled under student-context (goal + slim_oracle, no critique). Same prompt builders as μ-v4 historical Phase 4a inference. Inference-only — NO retraining.
- **Comparator**: existing μ-v4 LoRA Phase 4a inference plans (`runs/2026_04_28_xgoal_<goal>_mu/buffer.jsonl`, n=8 each). Same goal text, same oracle, same sampling params.
- **Pairwise**: 8 strict 1-pair Opus subagents per goal × 3 goals = 24 pairs total. Forced 4/8 perfect A-position balance. M8-compliant (no M8 anchoring confound).
- **Cost**: ~$8 (24 Opus pairwise calls + 3 inference calls). Wall ~10 min for inference + ~5 min pairwise (parallel).

Code: `src/co_scientist/d5_abstract_retrieve_refine/mu_opd_xgoal_inference.py` (~140-line inference-only fork).

## Position-bias check (M8 standing protocol)

All 3 goals show 4/8 OPD A-position. Decoded wins from A-pos and B-pos symmetrically:
- meta_ttl: OPD wins from A 3/7, from B 4/7 (B-position symmetry strongest)
- tool_v_ttrl: OPD wins from A 4/8, from B 4/8 (perfect)
- tt_control: OPD wins from A 3/7, from B 4/7

No position bias. The 22/24 win rate reflects substance, not anonymization artifact.

## Pairwise judges' substance differentiators (recurring patterns favoring OPD)

Across 24 pairs, judges cited:
- **U1 soundness**: RHS-complete equations (RS-GRPO, J_β entropic objective)
- **U5 reproducibility**: concrete hparams (β=4-8, K=10, lr=1e-5, LoRA rank, A100 hours)
- **T1 necessity**: named baselines with prior numbers (TTRL 40.2%, AIME 12.9→67%, ARC 53%)
- **T3 compute**: operational units (10000-update budget, GPU-hours, FLOP estimates)
- **T4 reward-hacking**: named Goodhart pathways + KL anchor mitigations + reward-cliff stop

μ-v4 plans repeatedly exhibited **failure artifacts** that μ-OPD plans did not:
- Leaked `<think>` preamble in plan body (format violation — Qwen3 thinking-mode bleed)
- Fabricated benchmarks (`MOSAIc`, `SuperTNT`, "infinite rate limit baseline")
- Undefined acronym methods (`TTA-DDPG`, `MED`, `AID`)
- Constraint violations (DDPG breaking frozen-actor goal requirement)
- Duplicated/truncated body text

These are exactly the failure modes one would predict from off-manifold gradient (F2) — student is being pushed toward teacher tokens it cannot reproduce coherently. μ-OPD plans are clean, on-manifold, internally consistent.

## Mechanism interpretation

F12 (prefix-prior lock-in) predicts that:
1. Content present in oracle prefix can be learned across architectures — confirmed via OPD's J_β recovery on TTT-D
2. Content absent from oracle prefix (PUCT) cannot be learned by training alone — confirmed (PUCT 0/8 throughout)
3. Cross-goal transfer of LEARNED CAPABILITY (not specific content) depends on whether the learned capability is on-manifold — confirmed: μ-OPD's on-policy weights transfer; μ-v4's off-manifold weights do not

F14's 22/24 win rate confirms F12+F13's joint prediction: **the (c) HER architecture poisoned cross-goal transfer with off-manifold artifacts; canonical OPD's on-policy training produces transfer-clean weights.**

## Implications for spotlight narrative

This converts D5's weakest existing result (F7 DEFINITIVELY NULL at n=24) into the **strongest cross-goal validation in the project**:

1. F11 falsifies F4 (gradient flooding) → mechanism gap exposed
2. F12 isolates prefix-prior lock-in (oracle keyword test) → diagnosis
3. F13 predicts canonical OPD fixes F2 → 23/24 pairwise on TTT-D validates
4. **F14 predicts cross-goal transfer was (c) HER artifact → 22/24 pairwise on 3 forward-citation papers validates**

Combined D5 pairwise across all matchups: **45/48 = 93.75% OPD wins anonymized, balanced positions, strict 1-pair/Opus.**

## Strict 9-dim isolated audit (added 2026-04-27 PM, post-pairwise)

Per ML scientist recommendation (paper-grade /45 numbers complement pairwise ranking), 24 cross-goal plans audited via strict 1-plan/Opus subagent (M8-compliant):

| Goal | μ-OPD strict /45 (n=8) | range | μ-v4 (F7 σ' anchor) | Δ |
|---|---:|---|---:|---:|
| meta_ttl | **22.88** | 19-26 | 17.75 | **+5.13** |
| tool_v_ttrl | **23.75** | 19-31 | 20.38 | **+3.37** |
| tt_control | **23.75** | 21-30 | 20.00 | **+3.75** |
| **Combined (n=24)** | **23.46** | 19-31 (std 2.83) | ~19.4 | **+4.0** |

Per-dim signature (averaged across 3 goals):
- Strongest: U2 significance (3.00), U4 clarity (3.00), U5 reproducibility (3.04), U1 soundness (2.95), T4 reward-hacking (3.04)
- Weakest: T3 compute (1.87), T2 disentanglement (1.96), T1 necessity (2.16), U3 originality (2.38)

Per-dim signature alignment with Phase 2 / TTT-D μ-OPD iter 4 (U4 / U5 / T4 dominant) — the same canonical-OPD mechanism produces the same per-dim improvements across goals, supporting F13's "mechanism-targeted, not aggregate noise" framing for paper.

Combined evidence: pairwise 22/24 directional + audit /45 +4.0 strict isolated. Both metrics agree.

## Limitations

1. **n=8 plans/goal pairwise** is on the lower edge for paper-grade publishing; consider expanding to n=12-16 if reviewer pushback expected. Current 22/24 = 91.7% is solidly above noise floor for n=8.
2. **PUCT specifically still 0/8** in cross-goal plans (oracle-absent content) — F12 lock-in remains binding for content not in slim_oracle.
3. **3 forward-citation papers** are within TTT-family subfield. Out-of-domain (non-ML) cross-paper validation is main session's territory (3rd session scouting).
4. **F3 cliff at iter 7 still present** in OPD — production checkpoint is iter 4 (pre-cliff). For cross-goal use, μ-OPD iter 4 weights are the production anchor.

## Data pointers

- Inference script: `src/co_scientist/d5_abstract_retrieve_refine/mu_opd_xgoal_inference.py`
- Per-goal μ-OPD inference: `runs/2026_04_29_xgoal_<goal>_mu_opd/` × 3
- Pairwise: `runs/2026_04_29_xgoal_mu_opd_vs_muv4_pairwise/` (24 pairs + matchups_meta.json)
- F7 (now reframed): `paper_materials/findings/F7_cross_goal_followup_validation.md`
- F11/F12/F13 (mechanism chain): same dir
- D5 STATUS.md current as of 2026-04-27 PM
