# D5 PHASE_PLAN_v2 — Strategic Roadmap

*Last refreshed: 2026-04-27. **Phase 2 COMPLETE (μ-v4 iter 4 = 28.00, +2.75 over σ)**.
Phase 3 (retrieve-then-generate) plan TBD — see `paper_materials/next_steps/N1_phase3_retrieve_then_generate.md`.*

> ⚠️ **As of 2026-04-27, the canonical executive summary is now [`../../PROJECT_OVERVIEW.md`](../../PROJECT_OVERVIEW.md)**. This PHASE_PLAN doc is preserved as historical roadmap context but the live "what's next" view lives in PROJECT_OVERVIEW.md and STATUS.md.

---

## §1 — How to read this doc

**Purpose**: a single chronological + strategic view of D5 from project start to ICLR 2027 paper draft. Every phase's status, completed work, decision gate, and next branch is on one page. Read this BEFORE diving into code or runs in any new session.

**Read order**: §2 (Where we are now) anchors the rest. Then read whichever phase section is currently active or just-completed. §11 (decision gates table) and §13 (file pointers) are reference utilities.

**When to update this doc**: when a phase or sub-phase completes, when a decision gate fires, or when a branch is taken (e.g. Phase 3 P vs F). Do NOT update for every iter — that's `STATUS.md`'s job.

**Doc relationships**:

| Doc | Role | Update cadence |
|---|---|---|
| `STATUS.md` | Live tactical state (current run, current iter, day-to-day) | Every run start/end, daily |
| `DECISIONS.md` | Append-only decision log (newest first) | Per non-obvious decision |
| `PHASE_PLAN_v2.md` (this file) | Strategic chronological roadmap | Per phase / decision-gate |
| `~/.claude/plans/giggly-jumping-hollerith.md` | Original 2026-04-23/24 master plan | Frozen historical reference |
| `knowledge/current/HANDOFF_2026_04_25.md` | Realignment-day handoff snapshot | Frozen |

---

## §2 — Where we are NOW (anchor)

**Calendar position**: 2026-04-26/27. **Phase 2 + Phase 4a + Phase 5 ALL COMPLETE.** Phase 4b cross-domain stretch test formally **DEPRECATED** per 2026-04-26 reframe (TTT-RL specialist doesn't switch fields; field-expert metaphor breaks). Phase 3 (Session A retrieve-then-generate) running parallel — independent paper finding (F8 slot).

**3-finding paper narrative locked**:
- **F2/F4**: μ-v4 single-paper SDPO works (+2.75 over σ; pairwise 20/24 corroborated; production checkpoint `runs/2026_04_27_mu_v4/checkpoints.jsonl` row batch=4)
- **F7**: Phase 4a forward-citation transfer **Strong-mixed** (2/3 pass; aggregate Δ +1.42; per-dim signature U5/T3/T4 reproduces from Phase 2)
- **F9**: Phase 5 continual SDPO chain **NULL across 4-cell grid** (anchor saturation × Adam state × anchor_ce regularization). Sub-finding: per-token CE anchor is the WRONG abstraction; two-gradient conflict accelerates collapse. Phase 6 future work directions: plan-level replay buffer / EWC-Fisher / LoRA-per-paper merge.
- Methodology: audit_v3_isolated balanced-batch anonymized 9-dim rubric

**Latest quantitative anchor** (audit_v3 ISOLATED on 8 baselines, /45 — Phase 2 reference):

| ε (235B+ref) | **μ-v4 iter 4** | **σ (frozen+oracle)** | δ | α | μ-v2 | β | ξ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 33.62 | **28.00** ✅ | **25.25** | 25.12 | 25.00 | 23.88 | 22.75 | 15.62 |

Cross-validation discipline established: any close-cluster (gap < 3) audit headline must be cross-checked by Opus pairwise tournament before paper claim. See `paper_materials/methodology/M4_audit_isolated_methodology.md` § "Cross-validation discipline".

**Phase 5 verdict (2026-04-26)**: continual SDPO from any μ-v4 anchor at lr=5e-5 fails across 4 grid cells. v3 best-iter Δ vs μ-v4 isolated /45 = +0.62 (within n=8 noise); α best-iter Δ = -2.62 (regularization backfires). See `paper_materials/findings/F9_continual_sdpo_field_expert.md` for full diagnostic + `knowledge/current/CONTINUAL_SDPO_FAILURE_MODES_v1.md` for one-page developer reference.

**Active phases**:
- Phase 3 (Session A) retrieve-then-generate — parallel, F8 slot reserved
- **Phase 6** (Session B next session): SDPO root-cause investigation + literature search + 1 plan-level mitigation experiment (replay buffer / EWC / LoRA-merge)
- **Phase 7** paper draft: DEFERRED until Phase 6 verdict

**Target deadline**: ICLR 2027 main track, ~2026-10. Buffer ≈ 5-6 months.

---

## §3 — Phase 0: scaffolding + pathway smoke + sanity gates — ✅ DONE

Pre-realignment work (2026-04-23 to 2026-04-25). All archived under `runs/_archive_pre_realignment_2026_04_25/`.

| Sub-phase | Description | Status | Anchor |
|---|---|---|---|
| 0a | D5 directory + harness scaffolding | ✅ | commit `26e0ba1` |
| 0b v1 | Bibliography fetch via PDF parsing | ✅ but deprecated | superseded by Phase 2A-bis (LaTeX) |
| 0c | Smoke pathway v1/v2/v3: oracle abstraction beats goal-only baseline at inference | ✅ | commits `58d7f19` / `a9b2185` / `2181826` |
| 0.5 | Sanity gates: dataset migration, Qwen contamination clear, sampling fix | ✅ | commit `ff0344f` |
| 0.6 | Baseline pilot battery (μ/α/β/δ/ε at v1) | ✅ but deprecated | superseded by Phase 2E (v2 baselines) |

**What survived**: smoke pathway claim ("oracle abstraction at inference helps frozen 30B") was withdrawn under v1 absolute audit, then revalidated under fair pairwise. Lives on as a sub-claim in Phase 2 narrative. All numbers from v1 audit are NOT authoritative.

---

## §4 — REALIGNMENT (2026-04-25)

**Trigger**: user identified 4 structural misalignments in pre-realignment work:

1. Oracle was single-round single-paper, not multi-round bibliography-grounded
2. δ/ε mistreated as fair baselines (they have reference plan in prompt → semi-oracle)
3. Audit prompts had surface-form bias (v1) and floor saturation (v2)
4. No reviewer-standard grounding for any audit dimension

**3 design docs locked** (all in `knowledge/current/`):

- `REVIEWER_STANDARDS_v1.md` — synthesis of NeurIPS/ICLR/ICML 2025 reviewer forms + 7 OpenReview adjacent papers (MTTT/ReST-MCTS/Voyager/SCoRe/Guided-ReST/DeepEvolve)
- `ORACLE_DESIGN_v1.md` — 6 categories × 3 rounds × cumulative memory; Opus generates as ceiling estimate
- `AUDIT_RUBRIC_v3.md` — 9-dim hybrid (5 universal + 4 subfield), anchors with verbatim OpenReview reviewer quotes

**Pre-run protocol binding**: every run from now on requires `RUN_CONFIRMATION_TEMPLATE.md` filled & approved before launch. Non-skippable.

**Greek-letter naming**: ξ (frozen + goal), σ (frozen + oracle), μ (trained + oracle), α (Opus distill), β (ref-SFT), δ (frozen + ref plan), ε (235B + ref plan).

---

## §5 — Phase 1: design doc production — ✅ DONE

Realignment Phase 1 = the 3 design docs in §4. Completed in one day (2026-04-25). No code, no runs.

---

## §6 — Phase 2: build + 7-baseline ranking + first SDPO ablation — 🟢 IN PROGRESS

The realignment subdivided Phase 2 into 6 sub-phases (A-bis, B, D, E, F, G):

| Sub-phase | Description | Status | Anchor |
|---|---|---|---|
| 2A-bis | Bibliography rebuild from LaTeX (`main.tex` + `main.bib`); 86/90 cite keys resolved, 38 with full text | ✅ | commits `5f2d878` / `7860126` |
| 2B | `oracle_v2.md` via 4-round Opus extraction over 62 papers (580 typed items); slim variant for 4096-token budget | ✅ | commit `67ce1b9` |
| 2C | (skipped — folded into 2B/2D) | — | — |
| 2D | Realigned v2 trainers for all 7 baselines | ✅ | commit `3f1e848` |
| 2E | Full 7-baseline battery trained at `max_tokens=4096` | ✅ | commit `b10d71a` |
| 2F | audit_v3 ISOLATED methodology + 7-baseline ranking | ✅ | commit `e00bd8c` |
| **2G** | **First SDPO lr ablation (μ-v2 → v3 → v4)** | 🟢 IN FLIGHT | runs `2026_04_27_mu_v{2,3,4}` |

**2G detail** (the active experiment):

| Run | lr | n_grad/iter | n_iter | Outcome | Verdict |
|---|---:|---:|---:|---|---|
| μ-v2 | 1e-5 | 1 | 10 | mean_adv ~0.4-0.5 healthy; teacher PUCT 0/8 → 8/8; student PUCT 0/8 全程 | **23.88 / 45 < σ 25.25**. Gradient budget under-sized; signal computed but never moved policy |
| μ-v3 | 2e-4 | 4 | 20 | mean_adv 0.5 → 0.05 looked like convergence but iter-6 audit dropped 24.88 → 14.38 | **KILLED**. lr too aggressive for LoRA-r64; distribution co-collapse |
| μ-v4 | **5e-5** | 4 | 20 | iter 0-2: mean_adv in 0.18-0.48 sweet zone | **In flight**. Pass ≥ 26 / Accept 24-26 / Fail ≤ 22 |

Geometric median lr `5e-5 = √(1e-5 × 2e-4)`. Per-iter checkpointing + audit-drop early-stop guard added per μ-v3 lessons.

**User constraint** (recorded in `feedback_d5_rl_aesthetic.md` memory): keep SDPO architecture, only tune hyperparameters. CR-v7+SFT pivot was rejected. After v4, if pivot needed, it should be at the *mechanism* level (retrieve-refine), not within SDPO-vs-SFT.

---

## §7 — Decision gate: μ-v4 verdict — ⏸ GATED on §6's 2G

When μ-v4 finishes (or hits early-stop), run isolated 9-dim ensemble audit on its final EVAL plans. Compare absolute mean to σ 25.25.

| Verdict | Mean /45 | Implication | Next phase |
|---|---|---|---|
| **Pass** | ≥ 26 | SDPO at lr=5e-5 internalizes critique; 30B + training > 30B + oracle alone | §8 Branch P (full training) |
| **Acceptable** | 24-26 | SDPO neutral or marginal; training matches σ | §8 evaluate ROI: if cross-goal validation cheap and shows lift on 1-2 other goals → continue Branch P; else pivot |
| **Fail** | ≤ 22 | SDPO dead at all 3 lr regimes (1e-5, 5e-5, 2e-4 spanning 20×) | §8 Branch F (retrieve-then-generate as paper main contribution) |

Three lr points spanning 20× is sufficient evidence either way. Do NOT propose more lr variants if v4 fails.

---

## §8 — Phase 3: full training OR pivot — ⏳ PENDING (branched on §7)

### Branch P — SDPO full training (μ-v4 Pass / Acceptable)

This matches original master plan's Phase 3.

- **Scale**: 30+ iter × 8 train rollouts + 4 held-out × 3 retrieve-abstract rounds
- **Reviewer**: Opus 4.5 throughout (or self-review if held-out 50-abstraction Opus-vs-self ρ > 0.7 — check at Phase 3 mid-point)
- **Ablations**:
  - P1 vs P2: distill abstraction tokens only vs (selection + abstraction)
  - Round count: 1, 2, 3, 4
  - Abstraction size: 50 vs 150 vs 300 words
  - Reviewer: Opus vs self vs reference-plan-only
  - SDPO vs plain SFT against teacher (isolate privileged-info mechanism)
  - With vs without cross-round history visibility
- **Mid-phase mandatory δ-baseline checkpoint**: at Phase 3 mid-point, compare D5 trained vs δ (frozen 30B + ref in prompt) on Opus audit. Gate Phase 3 → 4: must have D5 > δ + 2 pts.
- **Cost ceiling**: $2000

### Branch F — Retrieve-then-generate pivot (μ-v4 Fail)

D5 paper's main contribution shifts from SDPO mechanism to **retrieve-refine architecture + audit_v3 isolated methodology**.

- **Drop**: SDPO loss, plan critic, teacher/student split
- **Keep**: Multi-round retrieval, structured buffer, oracle-style abstraction inference-time scaffolding
- **New core mechanism**: model-selected paper retrieval (still learned action) + cumulative buffer accumulation across rounds + final plan generation. Loss = supervised CE on reference plan tokens (or pure inference-time, no training).
- **Paper narrative**: "Across 3 lr regimes spanning 20×, plan-level SDPO never beats frozen+oracle inference. The publishable contribution is the retrieve-refine architecture — and audit_v3 isolated as the methodology fix that exposed SDPO's failure on long-form generation."
- **Cost**: lower than Branch P (~$500 — most cost was SDPO Opus reviewer)

**Either branch ends at the same Phase 4 gate**: cross-goal validation + 5 anti-distillation baselines.

---

## §9 — Phase 4: evaluation + anti-distillation baselines + cross-goal — ⏳ PENDING

Fixed regardless of §8 branch. Layered eval framework.

**Layer 1 — Mechanism evidence (sub-capability):**
- Selection precision@5 per round (model-selected ∩ source paper's actual citations)
- Abstraction quality on Opus 4-axis (specificity / grounding / applicability / novelty)
- Pathway ablation: handcrafted oracle abstractions → frozen model. If oracle-frozen ≥ trained → composition skill = 0; trained value is in abstraction.

**Layer 2 — Primary metric (final plan):**
- Opus 4.7 depth audit /40 on held-out eval rollouts
- Opus pairwise vs `reference_solution.txt` (target: ≥ 40% preference for trained plan)
- Required effect size: trained ≥ strongest baseline + 3 pts, bootstrap 95% CI non-overlapping

**5 mandatory anti-distillation baselines**:

| Baseline | Status (Phase 2E coverage) | Phase 4 action |
|---|---|---|
| α naive Opus-SFT | ✅ done at v2 (25.00) | Re-evaluate under Phase 4 audit if Phase 3 changes prompt format |
| β direct ref-SFT | ✅ done (22.75) | Same |
| γ RLAIF Opus-reward | ❌ not yet | Phase 4 to add (~$300) |
| δ frozen 30B + ref | ✅ done (25.12) | Re-evaluate alongside trained model |
| ε frozen 235B + ref | ✅ done (33.62) | Already known ceiling; reframe as "competitive within X pts" |

**Required comparison table for paper Results section** — see `~/.claude/plans/giggly-jumping-hollerith.md` § Phase 4b for exact format.

**Anti-distillation differentiators**: also run FLARE / IRCoT / DR Tulu replicas adapted to research-plan generation. Differentiates D5 mechanism from iter-retrieve-only methods.

**Cross-goal validation (paper-critical)**: pick 2-3 D3-style research papers post-Qwen-cutoff. Run trained model OR retrieve-refine pipeline (whichever Branch in §8). Required: Branch's main effect generalizes to ≥ 2/3 held-out goals. Cost: ~$30-50 OpenRouter, ~1-2 days. **Without this, the paper is single-goal and ICLR-risky.**

---

## §10 — Phase 5 (new numbering): Continual SDPO chain — ✅ COMPLETE NULL (2026-04-26)

⚠️ **Numbering note**: Originally §10 was "Phase 5 = paper draft". After the 2026-04-26 user reframe ("field-expert via continual SDPO"), the active research phase BECAME continual SDPO (testing μ-v4 → μ-v5 → μ-v6 chain). Original "paper draft" phase moved to Phase 7. Phase 4a and Phase 4b (now DEPRECATED, see §15) sit between Phase 3 and Phase 5 in the new sequence.

**Phase 5 design (locked DECISIONS 2026-04-26 (c) and (d))**: chain μ-v4 → μ-v5 (Tool-V SDPO) → μ-v6 (Meta-TTL SDPO); tt_control held out for H3 expansion test; pre-registered 4 hypotheses (H1 forward learning / H2 retention / H3 expansion / H4 aggregate field-expert).

**Result (DECISIONS 2026-04-26 (e))**: **NULL across 4-cell grid** (anchor saturation × Adam state × anchor_ce regularization).

| Run | Anchor | Adam | anchor_ce | iter 0 → 4 (in-loop /20) | Failure |
|---|---|---|---:|---:|---|
| cliff_v1 | μ-v4 iter 4 | full | 0 | 8.38 → cliff | iter-1 cliff -3.50 |
| cliff_v2 | μ-v4 iter 4 | fresh | 0 | 9.00 → cliff | iter-1 cliff -2.13 |
| v3 | μ-v4 iter 2 | fresh | 0 | 9.375 → 6.67 | 4-iter slow decline -2.71 |
| α | μ-v4 iter 2 | fresh | **0.1** | **10.375 → cliff** | iter-2 cliff -3.0 (regularization BACKFIRES) |

**Best v5 vs μ-v4 isolated /45 on Tool-V**: v3 iter 0 Δ +0.62 (within noise); α iter 0 Δ -2.62 (negative). Both fail H1 +1.0 threshold.

**Sub-finding (load-bearing for paper)**: per-token CE anchor is the WRONG ABSTRACTION for continual SDPO. Two competing gradients (TTT-D anchor + Tool-V SDPO) fight rather than consolidate. See `paper_materials/findings/F9_continual_sdpo_field_expert.md` for full 4-cell diagnostic + per-token-CE analysis. See `knowledge/current/CONTINUAL_SDPO_FAILURE_MODES_v1.md` for developer-facing one-page reference.

**Phase 6 future work directions explicit in F9**:
- Plan-level replay buffer (mix prior plans into batches, not output regularization)
- EWC penalty with Fisher Information (per-parameter importance)
- LoRA-per-paper + weighted merge (parallel single-paper SDPO + ensemble)

---

## §10b — Phase 7: paper draft — ⏳ DEFERRED (target ICLR 2027 ~2026-10)

**Status**: explicitly paused per user 2026-04-26 ("不用那么早开始写paper"); resume after Phase 6 verdict.

8 sections, 10 pages + unlimited appendix:

1. Abstract
2. Intro — motivate via scientist workflow; SDPO + privileged-info reviewer + forward-citation transfer
3. Related Work — cite Hübotter 2026 SDPO, RLAD, FLARE, IRCoT, ExpeL, DR Tulu, RLAIF, Reflexion, Self-Refine, Constitutional AI; continual learning canon (Kirkpatrick EWC, Lopez-Paz GEM, Rolnick Replay)
4. Method — full mechanism spec (μ-v4 SDPO + Phase 4a transfer protocol)
5. Experiments — baselines + ablations table; F2/F4 + F7 + F9 + (F10 if Phase 6 succeeds)
6. Analysis — mechanism breakdown, forward-citation transfer signature, continual NULL diagnostic, audit_v3_isolated methodology
7. Discussion + Limitations — single-goal-family scope, 30B constraint, eval-grader cost, continual mitigations as future work
8. References

**Naming convention** (binding, see `feedback_paper_naming.md`):
- Use Collaborator / Researcher / Evaluator
- Never use student / tutor / teacher in paper prose

---

## §11 — Decision gates summary

| Gate | Phase | Trigger | Threshold | Branch |
|---|---|---|---|---|
| Smoke pathway | 0c | absolute audit Δ(A-B) | +0.075 / non-overlap CI | Validated → Phase 1 |
| Sanity gates | 0.5 | 3 sub-checks | All clear | → Phase 0.6 |
| Baseline kill matrix | 0.6 | μ vs δ/α/β | μ > all + 1 pt | Architecture survives |
| audit_v3 v2 absolute | 2F | directional concordance | ≥ 6/7 | FAILED → pivot to ISOLATED |
| audit_v3 ISOLATED | 2F | within-tier discrimination | distinct totals > 50% | PASS → canonical metric |
| **μ-v4 verdict** | **2G** | **isolated audit /45 vs σ 25.25** | **≥ 26 / 24-26 / ≤ 22** | **§8 Branch P selected (μ-v4 iter 4 = 28.00 PASS)** |
| Mid-Phase 3 δ check | 3 | trained vs δ | D5 > δ + 2 pts | Continue / re-design (Session A active) |
| **Phase 4a forward-citation** | **4a** | **μ' - σ' on 3 follow-ups** | **2/3 ≥ +1.0 AND aggregate ≥ +0.5** | **Strong-mixed PASS (2/3, Δ +1.42)** |
| **Phase 5 H1 (continual)** | **5** | **best v5 isolated /45 vs μ-v4 + no cliff** | **Δ ≥ +1.0 AND no inter-iter drop ≥ 2** | **❌ NULL across 4-cell; per-token CE wrong abstraction** |
| **Phase 5α anchor_ce** | **5α** | **mitigation rescue test** | **same as 5 H1** | **❌ FAIL — accelerates collapse** |
| Cross-goal | 4 | trained on 2-3 held-out | 2/3 same direction | Phase 4a Strong-mixed |
| Effect size | 4 | trained vs strongest baseline | + 3 pts, CI non-overlap | Phase 7 to compute |
| Paper draft | 5 | 10-page complete | LaTeX compiles | ICLR submission |

---

## §12 — Open questions (snapshot)

1. **γ RLAIF baseline** still pending — Phase 4 must add ~$300 ($300/run × 1 run)
2. **Bibliography full-text vs abstracts**: smoke decomposition suggests abstracts-only is ~90% of effect. Phase 3 default = abstracts; expand if trained plateaus < 10/20
3. **Cross-goal selection** (Phase 4): 2-3 D3-style papers post-Qwen-cutoff. Candidates: TBD; check D3 paper-list for already-curated post-cutoff goals first
4. **Phase 2b reviewer transition** (Opus → self): n=8 too small for ρ-based decision. Phase 3 mid-point bumps to n=32 held-out abstractions
5. **Single-goal ICLR risk**: even if cross-goal lifts, 2-3 goals is small. Mitigation requires explicit "scope" statement in Limitations section

---

## §13 — Critical file pointers per phase

**Phase 0 (archived)**:
- `runs/_archive_pre_realignment_2026_04_25/` — all v1 work
- `src/co_scientist/d5_abstract_retrieve_refine/smoke_pathway_v1.py` — pathway test orchestrator

**Realignment (Phase 1)**:
- `knowledge/current/REVIEWER_STANDARDS_v1.md`
- `knowledge/current/ORACLE_DESIGN_v1.md`
- `knowledge/current/AUDIT_RUBRIC_v3.md`
- `knowledge/current/RUN_CONFIRMATION_TEMPLATE.md`

**Phase 2A-bis (bibliography)**:
- `src/co_scientist/d5_abstract_retrieve_refine/extract_cite_keys_v1.py` (tex → 90 cite keys)
- `src/co_scientist/d5_abstract_retrieve_refine/parse_bibtex_v1.py` (bib → resolved)
- `src/co_scientist/d5_abstract_retrieve_refine/build_bibliography_v2.py` (S2 batch + OpenAlex)
- `src/co_scientist/d5_abstract_retrieve_refine/extract_source_paper_v2.py` (LaTeX → md)
- `data/bibliography/resolved_v2.jsonl` (86 entries)
- `data/source_paper/v2.md` (~76,300 chars)
- `data/bibliography/full_text/*.md` (38 LaTeX→md papers)

**Phase 2B (oracle)**:
- `src/co_scientist/d5_abstract_retrieve_refine/oracle_prompts_v2.py`
- `src/co_scientist/d5_abstract_retrieve_refine/build_oracle_v2.py`
- `data/oracle_v2.md` (full) / `data/oracle_v2_slim.md` (4096-token fit)

**Phase 2D (realigned trainers)**:
- `src/co_scientist/d5_abstract_retrieve_refine/train_{xi,sigma,delta,epsilon,alpha,beta,mu}_v2.py`
- `src/co_scientist/d5_abstract_retrieve_refine/mu_prompts_v2.py`

**Phase 2E (baseline runs)**:
- `runs/2026_04_26_xi_v2/` / `_sigma_v2/` / `_delta_v2/` / `_epsilon_v2/` / `_alpha_v2b/` / `_beta_v2b/` / `_mu_v2/`

**Phase 2F (audit)**:
- `src/co_scientist/d5_abstract_retrieve_refine/audit_v3_isolated.py`
- `runs/2026_04_26_phase2F_audit_isolated/audit_v3_isolated_summary.md` (canonical 7-baseline ranking)
- `src/co_scientist/shared/audit_prompt_v3.py` (rubric prompt builder)

**Phase 2G (SDPO lr ablation, in flight)**:
- `src/co_scientist/d5_abstract_retrieve_refine/train_mu_v3.py` (lr=2e-4, killed)
- `src/co_scientist/d5_abstract_retrieve_refine/train_mu_v4.py` (lr=5e-5, in flight)
- `runs/2026_04_27_mu_v3/` (collapsed)
- `runs/2026_04_27_mu_v3_midtrain_audit/` (mid-train PUCT diagnostic)
- `runs/2026_04_27_mu_v4/` (in flight — DO NOT modify)

**Subagent / shared infrastructure**:
- `src/co_scientist/shared/opus_subagent_grader.py` — Opus file-bus client
- `src/co_scientist/shared/paper_retrieval.py` — S2 + OpenAlex (extended in 2A-bis)
- `src/co_scientist/shared/audit_prompt_v3.py` — 9-dim rubric prompt
- `projects/d5_abstract_retrieve_refine/specs/subagents/opus_oracle_subagent.md` — oracle build subagent spec

---

## §15 — Phase 4b cross-domain stretch — DEPRECATED (2026-04-26)

**Original framing** (in PHASE_PLAN draft, never executed): test μ-v4 transfer to a medium-OOD goal where oracle is rebuilt for a different subfield (e.g., ViT or World Models).

**Reason for deprecation** (user 2026-04-26 reframe): D5's research narrative is "domain-expert via continual SDPO with privileged-info reviewers", NOT cross-goal/cross-domain generalization. A real scientist deeply specialized in TTT-RL doesn't suddenly switch to a totally different field; the field-expert metaphor breaks. Phase 4a (forward-citation follow-ups in same subfield) and Phase 5 (continual chain in same subfield) are the correct test articulations of the metaphor; Phase 4b would be **out of scope** per the metaphor.

**What this means**: Do NOT propose cross-domain transfer experiments in Phase 6+. Cross-DOMAIN ≠ cross-PAPER-IN-SAME-DOMAIN. Phase 4a's forward-citation transfer is already the legitimate generalization test.

**Recorded in**: DECISIONS 2026-04-26 (c) and (e); PROJECT_OVERVIEW.md "Phase 4b deprecated"; F9 "Out of scope" section.

---

## §16 — Phase 6 (next): SDPO root-cause investigation + plan-level mitigation

**Status**: ⏳ PLAN ONLY (next session per user 2026-04-26 "不用那么早开始写paper, 先...同步docs").

**Scope**:
- B.1 Literature investigation: Hübotter 2026 SDPO follow-ups; continual DPO/RLHF survey; LLM continual fine-tuning canon (Kirkpatrick EWC, Lopez-Paz GEM, Rolnick Replay, Buzzega DER)
- B.2 Root-cause hypothesis synthesis (4 falsifiable hypotheses; each maps to a mitigation)
- B.3 Phase 6α experiment: pick ONE highest-info-gain mitigation. Likely **plan-level replay buffer** (standard CL technique; addresses critic-sparsity hypothesis; cleanest path to "did continual rescue?" answer)
- B.4 Phase 7 paper draft re-evaluation after Phase 6 verdict

**Plan**: `~/.claude/plans/adaptive-churning-perlis.md` Phase B section. Pre-registration to be locked in DECISIONS 2026-04-26 (f) before launch.

---

## §14 — Update log for this doc

- 2026-04-26: created. Folded `~/.claude/plans/giggly-jumping-hollerith.md` (master plan) + `HANDOFF_2026_04_25.md` (realignment) + Phase 2 actual progress + STATUS.md / DECISIONS.md latest entries.
