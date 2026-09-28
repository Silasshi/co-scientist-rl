# F8 — Phase 3 inference-time distillation pipeline: comparable to σ baseline + 3 failure modes characterized

## Headline (REVISED 2026-04-27 PM after Step 2a/B/C/E re-validation)

**Phase 3 inference-time 3-round distillation pipeline on open-source Qwen3-30B
produces research plans STATISTICALLY COMPARABLE to a hand-curated Opus-extracted
oracle baseline (no fine-tuning). Pairwise corroboration on patched + freshly-
sampled plans (Step E, 2026-04-27): **τ_v4_clean vs σ_v4 = 4-4 TIE** (50% A-position
balanced — not position bias). The original Step 2.5 verdict (τ_v4 7-1 STRONG)
was a sampling artifact: it was conducted on lucky-draw τ_v4 plans (instance 0
single-shot 27 vs K=8 reseed mean 24.88; instance 2 single-shot 29 vs K=4 reseed
mean 23.50). Cross-instance τ_v4_clean mean = 24.50 / 45 (vs claimed 26.50).**

**Three paper-relevant failure modes characterized**:

1. **Self-review degeneration** (M7, Smoke A): Qwen3-30B with source paper as
   privileged info loses 0-8 to Opus-as-critic. SDPO + 30B-self-critic NOT viable;
   needs ≥235B critic. (Unchanged from original F8.)

2. **Single-shot evaluation inflates apparent lift** (NEW, Step 2a/B/C):
   within-instance σ_within = 1.17 (instance 0 K=8) and 0.50 (instance 2 K=4) —
   variance is heterogeneous across instances. Single-sample-per-instance audit at
   n=8 has cross-instance σ ≈ 2.18 → SE on mean = 0.77, 95% CI ≈ ±1.5. Original
   "τ_v4 26.50 beats σ 25.25 by +1.25" was within this CI — pairwise 7-1 was on
   specific lucky draws on both sides. **Implication for the field**: BoN-style
   inference-time scaling literature reporting single-shot improvements at this
   magnitude needs error-bar reporting; the 1-2 point band looks like signal but
   is noise floor.

3. **Citation-tag leakage from multi-round distillation** (NEW, Step 2a discovery):
   distillation step organizes oracle items by index labels ("Methodology 17",
   "Theory 5", "Math 11") that the plan generator copies through as if they were
   real citations. Original τ_v4 plans averaged ~16 such tags per plan; judges
   penalized U1 soundness + U4 clarity dimensions. Patched plan prompt with explicit
   anti-leakage instruction reduced leakage by ~85% (18 total across 8 plans, vs
   ~120 originally) but did not eliminate it — model attention to anti-leakage
   instruction is partial. **Open-source future work**: fix at distillation prompt
   level (use natural-language descriptors instead of indexed tags).

## Trajectory (4 distillation prompt iterations)

Source: `runs/2026_04_28_tau_v{1,2,3,4}/` + `runs/2026_04_28_tau_v{1,2,3,4}_audit/`

| Run | Setup change vs prior | Mean /45 | vs σ Phase 2 (25.25) |
|---|---|---:|---:|
| τ_v1 | Initial(distill_max_tokens=1024, original prompt) | 18.12 | -7.13 |
| τ_v2 | distill_max_tokens 1024 → 3500 | 19.00 | -6.25 |
| τ_v3 | + verbatim-preserve distillation prompt(equations/hparams in backticks) | 20.38 | -4.87 |
| **τ_v4** | + plan-prompt v4(mandatory T1+T2+T3 instructions) | **26.50** | **+1.25** |

τ_v4 is the production checkpoint for Phase 3 inference-time pipeline. **First
non-fine-tuned 30B variant to beat σ baseline**.

Per-dim breakdown τ_v4 vs σ Phase 2 (key dims):
- T1 Necessity-of-TTT: τ_v4 3.62 vs σ 3.12(+0.50)
- T2 Disentanglement: τ_v4 3.50 vs σ 2.38(**+1.12**)
- T3 Compute accounting: τ_v4 3.12 vs σ 2.00(**+1.12**)
- U5 Reproducibility: τ_v4 3.12 vs σ 2.62(+0.50)
- U3 Originality: τ_v4 2.00 vs σ 2.38(-0.38, regression)

The plan-prompt v4 instructions (mandatory disentanglement ablation + compute units
+ quantified frozen-LLM gap) are the dominant variable; distillation pipeline adds
incremental value on top of that.

## Step 2a/B/C/E re-validation (2026-04-27 PM) — definitive correction to original headline

After F8 was first committed (commit 68e08b1), a routine BoN-sanity check (Step 2a:
1 instance × K=8 plan reseeds, ~$4) was launched to validate the within-instance
sampling variance assumption before committing $16-32 to a full BoN ablation.
Three findings emerged that **invalidated the original F8 headline**:

### Step 2a (1 inst × K=8 reseed of original prompt, ~$4): within-instance σ + lucky-draw discovery

- Instance 0, K=8 plan reseeds (same prompt, same distillations, same temp/top_p,
  only RNG seed varies): mean **24.88**, σ **1.17**, range 23-27
- **Original τ_v4 production single-shot for instance 0 = 27/45**, but K=8 reseed
  mean is 24.88. Production was a high draw on the within-instance distribution.
- Across all 8 τ_v4 production single-shots: mean 26.50, σ 1.41 — implies
  same lucky-draw artifact at the population level too. True cross-instance mean
  estimate from K=8 reseed on instance 0 alone (24.88) is 1.62 BELOW production.
- **Citation bug discovered**: 4 of 8 K=8 sanity judges flagged plans containing
  "Math 11", "Methodology 24", "Math 10" — internal oracle-batch index labels from
  the distillation step that the plan generator treats as real citations. Source
  distillations contain 16-17 such tags each.

### Step A (plan-prompt patch): citation hygiene instruction

`_KAPPA_PLAN_FOOTER_V4` patched to add explicit anti-leakage instruction:

> "**Citation hygiene**: The distilled abstractions above use internal index
> labels like 'Methodology 17', 'Theory 5', 'Math 11', 'Algorithm 3', 'Background 2'
> — these are bookkeeping tags from the source organization, NOT real citations.
> Use the conceptual content but DO NOT reproduce these tags in your plan. Cite
> real prior work by first-author + year (e.g., 'Hübotter 2026') or by descriptive
> name only."

Applied to plan prompt only (not distillation prompt) to preserve the τ_v3→τ_v4
verbatim-preserve gains.

### Step B (8 inst × 1 plan with PATCHED prompt, ~$4): cross-instance re-baseline

Source: `runs/2026_04_30_tau_v4_clean/` + `runs/2026_04_30_tau_v4_clean_audit/`

- **τ_v4_clean cross-instance mean: 24.50 / 45** (vs claimed F8 26.50 — drop 2.0)
- σ across 8 = **2.18** (cross-instance noise; SE on mean = 0.77, 95% CI [22.96, 26.04])
- σ Phase 2 baseline (25.25) **falls inside** 95% CI of τ_v4_clean
- Tag-leakage residual: 18 tags total across 8 plans (vs ~120 in original
  unpatched τ_v4). Patch ~85% effective; not fully eliminated.

Per-dim shift (τ_v4_clean - τ_v4 production):

| Dim | orig | clean | Δ |
|---|---:|---:|---:|
| U1 soundness | 2.75 | 2.38 | -0.38 |
| U2 significance | 3.00 | 3.12 | +0.12 |
| U3 originality | 2.00 | 2.00 | 0.00 |
| U4 clarity | 3.00 | 3.00 | 0.00 |
| U5 reproducibility | 3.12 | 2.75 | -0.38 |
| T1 necessity | 3.62 | 3.12 | -0.50 |
| T2 disentanglement | 3.50 | 3.00 | -0.50 |
| T3 compute | 3.12 | 3.12 | 0.00 |
| T4 reward_hacking | 2.38 | 2.00 | -0.38 |

**Pattern**: drop is uniform across ~5 dims rather than concentrated in any single
dim. Most consistent interpretation = sampling variance (2.0-point shift ≈ 1 SD
across 8-instance batch). Bug fix did NOT lift U1/U4 as expected — likely because
judges don't penalize tags as harshly as the 8/8 anecdotal flagging suggested,
or because tag effects are confounded with sampling noise at this n.

### Step C (1 inst × K=4 reseed with PATCHED prompt, ~$2): σ_within generalization spot-check

Source: `runs/2026_04_30_tau_v4_bon_sanity_inst2/`

- Instance 2, K=4 plan reseeds with patched prompt: mean **23.50**, σ **0.50**,
  range 23-24 (n=4)
- σ_within varies 2.3× between instances (instance 0 K=8 σ=1.17; instance 2 K=4 σ=0.50)
- Instance 2's τ_v4 production single-shot was **29/45** (the highest of all 8 in
  τ_v4 production); K=4 reseed mean is 23.50 — gap of **5.5 points**, even bigger
  lucky-draw than instance 0.

### Step E (8 pairs τ_v4_clean vs σ_v4, ~$32): pairwise rerun

Source: `runs/2026_04_30_step_e_pairwise/` (seed=52, distinct from prior 51).
8 pair-isolated Opus subagents (one per pair, full per-pair isolation per
subagent task isolation rule). Position-randomized 4-4.

| | τ_v4_clean | σ_v4 |
|---|---:|---:|
| Wins | **4** | **4** |
| Ties | 0 | |
| A-position | 4/8 (50%) — no position bias |

**Verdict: TIE** — clean falsification of the original Step 2.5 7-1 STRONG verdict.

Cross-pair pattern from judge rationales:
- τ_v4_clean wins (pair_00, pair_05) when its plan happens to specify explicit
  hparams + open-weight baseline
- σ_v4 wins (pair_04, pair_07, pair_06) when τ_v4_clean's plan still leaks tags
  ("Math 9"/"Methodology 10/14/22" called out by judge) OR uses closed-weight GPT-4
  baseline (violates goal's open-weight constraint)
- **Pair-level outcomes correlated with sample-specific quality, not method-level
  signal** — consistent with the "single-shot evaluation inflates apparent lift"
  failure mode.

### Combined evidence picture (Phase 3 final)

| Comparison | Original (F8 v1) | Re-validated (F8 v2) |
|---|---|---|
| τ_v4 absolute mean | 26.50 | **24.50** (drop 2.0) |
| σ Phase 2 baseline | 25.25 | (single-shot, may also be inflated) |
| pairwise τ vs σ | 7-1 STRONG | **4-4 TIE** |
| within-inst σ | unknown | 1.17 (inst 0) / 0.50 (inst 2) |
| citation bug | unknown | discovered + ~85% fixed |
| self-review (M7, Smoke A) | 8-0 dispelled | unchanged ✓ |

## Cross-validation: Step 2.5 pairwise (3 close-cluster matchups)

Source: `runs/2026_04_28_step2_5_pairwise/`. Per cross-validation discipline (M4):
close-cluster gap < 3 / 45 requires pairwise corroboration. Per `feedback_quality_first.md`
(2026-04-27 binding rule), close-cluster comparisons MUST not be skipped.

8 position-randomized pairs per matchup × 3 matchups = 24 Opus pairwise judge calls
(seed=51). Each subagent judged anonymously without baseline label.

| Matchup | Audit gap | **Pairwise verdict** | Significance |
|---|---:|---|---|
| τ_v4 vs σ_v4 (slim+plan_v4) | +0.75 | **τ_v4 wins 7-1** | STRONG |
| τ_v4 vs μ-v4-replan (μ-v4 LoRA + plan_v4) | +1.75 | tie 4-4 | distillation ≈ SDPO LoRA at plan_v4 |
| σ_v4 vs μ-v4 Phase 2 prod | -2.25 | μ-v4 wins 7-1 | μ-v4 weights still dominant ceiling |

**Critical correction from absolute audit alone**: τ_v4 vs σ_v4 absolute Δ=+0.75 was
within 1σ noise floor (std 1.5-2.2 at n=8). Initial interpretation was "noise — not
worth pairwise". The pairwise 7-1 STRONG result corrects that interpretation: the
distillation pipeline's contribution IS real, just below the absolute-audit
discrimination threshold.

## Negative finding: Qwen3-30B self-review fails (Smoke A)

Source: `runs/2026_04_28_smoke_a_critique_corr/`

**Test**: 8 τ_v4 plans × (Qwen3-30B-base critique via Tinker + Opus critique via
subagent), both with source paper as privileged info. Then 8 Opus-as-judge
subagents compare per-pair which critique is more useful (more specific named
items, more accurate ground-truth, more actionable directive).

| Result | Count |
|---|---:|
| **Opus critique preferred** | **8/8 (100%)** |
| Qwen3 critique preferred | 0/8 |
| Tie | 0/8 |

A-position win rate balanced (4/8) → not position bias, real content quality gap.

### Qualitative pattern (from judge rationales)

Across all 8 plans, Opus critique consistently identified:
- **Specific algorithmic objects**: J_β formula, KL budget γ=ln 2, PUCT formula
  `c·P(s)·√(1+T)/(1+n(s))`, max-of-children Q with rank prior, top-2 children,
  archive cap 1000, lineage backprop
- **Concrete hyperparameters**: LoRA rank 32, 50 steps × 512 rollouts, batch 8×64,
  KL coefficient 0.1 vs 0.01
- **Real evaluation domains** (not hallucinations): Erdős minimum-overlap (with
  numbers 0.380924→0.380876), GPUMode TriMul, AtCoder, single-cell denoising
  (OpenProblems), AHC039/AHC058
- **Infrastructure specifics**: gpt-oss-120b on Tinker, ~$500/run
- **Ground-truth corrections**: adaptive β(s) via KL constraint (NOT constant β=8
  the distillations claimed); LoRA rank 32 (NOT 64/4 the distillations claimed)

Qwen3-30B critique mentions same conceptual terms ("entropic objective", "PUCT",
"LoRA") but does NOT extract specific formulas, numerical values, or domain names.
Some critiques (plan 4, plan 7) leaked raw `<think>` thinking-mode tokens into the
critique XML output.

### Implication: SDPO + self-distillation framework not viable with 30B critic

The SDPO advantage signal in distillation framework requires teacher critique to
reflect real source-paper grounding (not surface paraphrase). With 30B critic
producing generic critique, the SDPO gradient would train the model toward
"paraphrase-style imitation" rather than "absorb source-paper specifics" — exactly
the failure mode F4 (critique-token-blindness) but worse, because the critique
itself is content-blind.

This is NOT the same as a "self-collapse" / sycophancy bias (Challenge 1). The
issue is **Privileged Info Comprehension** (Challenge 2): 30B critic cannot read
the source paper as carefully as Opus.

## What this means for Phase 3 contribution claim

**Final paper-grade Phase 3 claim** (REVISED 2026-04-27 PM):

> "We construct an inference-time abstract-retrieve-refine pipeline (3 distillation
> rounds + plan-prompt template with mandatory disentanglement/compute/baseline
> instructions) on open-source Qwen3-30B-A3B. The pipeline produces research plans
> **statistically comparable** to a hand-curated Opus-extracted oracle baseline
> with **NO fine-tuning** required (τ_v4_clean = 24.50/45 vs σ Phase 2 = 25.25/45;
> pairwise 4-4 TIE on patched + freshly-sampled plans, Step E). Both methods sit
> well below the SDPO-trained ceiling μ-v4 = 28.00. The trajectory τ_v1 18 →
> τ_v4 26.50 (in original 'lucky-draw' production) demonstrates inference-time
> prompt engineering progressively recovers ~6.5 audit points; the
> distillation-vs-baseline absolute Δ is within the n=8 single-shot noise floor.
>
> We characterize **three failure modes** relevant to multi-round retrieve-distill
> pipelines:
> (1) **Self-review degeneration**: Qwen3-30B with source paper as privileged
>     info loses 0-8 to Opus-as-critic on critique specificity; SDPO + 30B-self-
>     critic NOT viable (M7).
> (2) **Single-shot evaluation inflation**: at σ_within ≈ 1.17 and σ_between ≈ 2.18,
>     n=8 single-shot pairwise comparisons can produce 7-1 STRONG verdicts that
>     are sampling artifacts (our own Step 2.5 verdict was falsified by Step E
>     re-run on patched + fresh samples).
> (3) **Citation-tag leakage**: distillation organization labels (e.g., 'Methodology
>     17') propagate into plans as if they were real citations; partially fixable
>     at plan-prompt level (~85% reduction observed) but full fix requires
>     distillation-prompt redesign (open future work).
>
> All three are characterized with mechanism + quantitative evidence. The negative-
> result-with-mechanism pattern (rather than 'beats X' positive claim) is the
> Phase 3 contribution."

## Limitations (REVISED 2026-04-27 PM)

1. **Single goal** (TTT-Discover) — cross-goal validation deferred. Same as L1 in N3.
2. **n=8 plans per condition** with std 1.5-2.5 and σ_within ≈ 1.17 → single-shot
   absolute Δ < 2 / 45 is below noise floor; cannot reliably distinguish methods
   in this range without K=4-8 reseeds per instance. The Step 2.5 → Step E
   trajectory (7-1 STRONG → 4-4 TIE) is the cleanest empirical demonstration of
   this within the project; methodology lesson worth a paper-section discussion.
3. **Word-target 600-750** (per L11 in N3) caps T3/U5 dimensions structurally.
4. **SDPO + real Opus critic in distillation framework (κ_opus) NOT YET TESTED**.
   Prior κ smoke used cold-start critique (cost-saving compromise, INVALIDATED).
   Self-critic variant (κ_self) cancelled after Smoke A. Whether real-Opus-critic
   κ_opus would improve over τ_v4_clean 24.50 is **open**; lower priority now
   given τ_v4 ≈ σ + ceiling at μ-v4 28.00.
5. **plan-prompt-v4 dominance vs distillation contribution**: most of τ_v4_clean's
   24.50 vs τ_v1 18.12 gain (+6.4 trajectory) is from plan-prompt instructions,
   not distillation pipeline. σ + plan_v4 alone = 25.75 (essentially at τ_v4_clean);
   the distillation step's NET contribution at single-shot n=8 is **NOT detectable**
   above the noise floor.
6. **Citation-tag leakage NOT fully fixed**: 18 tag-leaks across 8 patched plans
   (vs ~120 originally). Future work: redesign distillation prompt to use natural-
   language descriptors (~5-10 lines change in `kappa_prompts_v1.py:_DISTILL_FOOTER`).
7. **σ Phase 2 baseline (25.25) was also single-shot n=8** and likely subject to
   same lucky-draw inflation. True σ population mean is unknown without K=4-8
   reseed; if σ population mean is 24-25, conclusions hold. If higher, τ_v4 might
   even slightly underperform — testable but deferred (cost ~$4).

## Data pointers

| What | Where |
|---|---|
| τ_v4 production plans (orig, with bug) | `runs/2026_04_28_tau_v4/buffer.jsonl` (8 plans) |
| **τ_v4_clean** (PATCHED, fresh sample, n=8) | `runs/2026_04_30_tau_v4_clean/` + `_audit/` |
| BoN sanity inst 0 K=8 (σ_within = 1.17) | `runs/2026_04_29_tau_v4_bon_sanity/` + `_audit/` |
| BoN sanity inst 2 K=4 (σ_within = 0.50) | `runs/2026_04_30_tau_v4_bon_sanity_inst2/` + `_audit/` |
| **Step E pairwise rerun (4-4 TIE)** | `runs/2026_04_30_step_e_pairwise/step_e_pairwise_summary.md` |
| τ_v1/v2/v3/v_C | `runs/2026_04_28_tau_v{1,2,3}/`, `runs/2026_04_28_tau_C/` |
| σ_v4 + μ-v4-replan controls | `runs/2026_04_28_sigma_v4/`, `runs/2026_04_28_mu_v4_replan/` |
| Step 2.5 pairwise summary | `runs/2026_04_28_step2_5_pairwise/step2_5_pairwise_summary.md` |
| Smoke A summary | `runs/2026_04_28_smoke_a_critique_corr/` |
| INVALIDATED κ smoke (cold-start, do not cite) | `runs/2026_04_28_kappa_v1_smoke/` |
| Trainer code (production) | `src/co_scientist/d5_abstract_retrieve_refine/train_tau_v1.py` |
| Plan v4 prompt | `kappa_prompts_v1.py` `_KAPPA_PLAN_FOOTER_V4` |
| Distillation prompts | `kappa_prompts_v1.py` `_DISTILL_FOOTER` (verbatim-preserve) |
| Methodology lesson | `paper_materials/methodology/M7_self_review_failure.md` |
| Raw evidence dump | `paper_materials/experiments/E8_phase3_distillation_evidence.json` |
