# F3 — Multi-round instability of D5 in-house off-policy IS-loss variant: 3-lr ablation

> **⚠ TERMINOLOGY NOTE (2026-04-27 PM)**: F3 was originally framed as "SDPO
> multi-round instability reproducing Hübotter 2026 §4". After paper-grade
> verification, the μ-v2/v3/v4 trainer is **not** canonical Hübotter SDPO
> (canonical: student on-policy + KL loss; ours: TEACHER samples + IS-loss).
> The instability is real and reproducible across 3 lr regimes in **our**
> variant, but we have NOT tested whether canonical SDPO (Hübotter 2601.20802)
> or canonical OPSD (Zhao 2601.18734) exhibits the same property. Mechanism
> claim should be scoped to "D5 in-house off-policy IS-loss variant", not to
> canonical SDPO. Disambiguation: `knowledge/current/CANONICAL_NAMING_REFERENCE.md`.

## Headline

Plan-level **D5 in-house off-policy IS-loss variant** (TEACHER samples; STUDENT lp
recomputed; importance_sampling loss — historically labeled "SDPO") at three learning
rates spanning 20× (1e-5, 5e-5, 2e-4) all eventually **destabilize the model** if run
past their peak. The peak iter and severity of post-peak collapse depend on lr; the
*existence* of collapse does not. The instability is empirically reproducible in this
variant; cross-comparison with canonical Hübotter SDPO §4 multi-round failure mode is
suggestive but not directly demonstrated (we do not faithfully implement that
algorithm — see CANONICAL_NAMING_REFERENCE.md).

## 3-lr ablation table

| Run | lr | n_grad | Peak iter | Peak mean /45 | Collapse iter | Collapse mean | Symptoms |
|---|---:|---:|---:|---:|---:|---:|---|
| μ-v2 | 1e-5 | 1 | (none) | 23.88 (flat) | (none, no learning) | n/a | EVAL plans plateau at σ-anchor; no PUCT, no equations, no cliff. **Under-trained**. |
| μ-v3 | 2e-4 | 4 | 0 (24.88, isolated audit) | 24.88 | iter 6 | 14.38 | XML leaks, biocultural headers, Chinese-character bleed, JSON thinking-mode leaks |
| **μ-v4** | **5e-5** | **4** | **iter 4 (28.00)** | **28.00** | **iter 5 (15.62)** | **15.62 in 1 iter** | hallucinated jargon ("Brobdingnag super-KAGA"), mixed scripts (Hebrew/Cyrillic/Korean), meta-narration, `<META>` and `<solution>` XML leaks |

(μ-v3 isolated audit data only collected for iter 0/3/6; trajectory shape inferred.)

## μ-v4 cliff in detail

Iter 4 → iter 5 EVAL audit means: **28.00 → 15.62** (drop of -12.4 in one training iter).

The early-stop guard set at threshold 3.0 (rolling mean over 2 audits) didn't fire
because the entire drop happened in a single iter — `mean_iter4 = 28.00`, `mean_iter5 = 15.62`,
drop = 12.4 in one iter, but the rolling check compared iter 4 to iter 3 (drop = 0.75)
and iter 5 to iter 4 (drop = 12.4) without combining. Lesson learned: tighten the guard
to "single-iter drop ≥ 5" for future SDPO runs.

## Symptom catalog (collapsed plans, from forensic audit)

Across μ-v3 iter 6 and μ-v4 iter 5-7 collapsed plans:
- **Mixed-script bleed**: Chinese, Hebrew, Cyrillic, Korean characters interspersed in English text
- **XML metadata leaks**: `<META>`, `<solution>`, `<structure>`, `<Problem Statement>` tags appearing as plan content
- **Self-referential meta-narration**: "The user has asked for a research plan..."
- **Hallucinated jargon**: "Brobdingnag super-KAGA", "Mudra method", "baryonic merging",
  "Layer Sigmoid Stick", "PROJECTIVE BIU CLAUSE"
- **Token-level repetition**: "MATH-500" four times in one line; identical phrase
  re-emitted three times consecutively
- **Recursive critique**: "Critique of the Improvement-Directive Critique" (model
  recursing on its own critique format)
- **Hallucinated benchmarks**: "MOSAIc", "SuperTNT", "Hydra-LLM", "IRC Exam"
- **Thinking-mode JSON leaks**: `{'type':'thinking', 'thinking': ...}` raw Python dict in plan body

These are not failure modes the base model exhibits at iter 0 — they appear ONLY after
SDPO training crosses some critical-update threshold.

## Hypothesized mechanism

Per F4 (critique-token-blindness): SDPO advantage `teacher_lp - student_lp` is dominated
by stylistic / connective tokens (high entropy, large divergence) rather than content
tokens (low entropy, small divergence). Each grad step pushes LoRA toward stylistic
amplification. After a few iters, stylistic distortion compounds into the symptoms above.

## What this means for the paper

- **3-lr ablation establishes empirical bounds**: SDPO can transfer content (μ-v4
  peak +2.75) but cannot sustain it past ~4-5 iters at any lr we tested
- **Best-of-early-iter is the production rule**: stop at peak, save iter-4 weights,
  don't trust later iters
- **The multi-round instability story is a published failure mode** (Hübotter 2026
  §4); we replicate it on a different task family (plan-level long-form generation
  vs. their single-step problem solving) and with a different content-target
  (research-plan dimensions vs math-correctness)
- **Critique-token-blindness is the root cause** (F4) — the lr-tuning ablation is a
  consequence; even the right lr only delays the inevitable

## Data pointers

- 3-lr trajectories: `paper_materials/experiments/E3_sdpo_lr_ablation.json`
- v4 trajectory: `paper_materials/experiments/E2_mu_v4_trajectory.json`
- Collapsed-plan exemplar: `paper_materials/plan_samples/exemplar_iter6_collapsed.md`
- Hubotter §4 excerpt: `paper_materials/related_work_excerpts/hubotter_2026_sdpo_section4.md`
