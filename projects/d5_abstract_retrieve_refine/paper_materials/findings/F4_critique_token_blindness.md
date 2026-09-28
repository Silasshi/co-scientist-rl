# F4 — Critique-token-blindness: D5 in-house IS-loss advantage measures style, not content

> **⚠ TERMINOLOGY NOTE (2026-04-27 PM)**: F4 evidence is from D5 in-house
> off-policy IS-loss runs (μ-v2/v3/v4, historically labeled "SDPO" but
> structurally TEACHER samples + IS-loss; not canonical Hübotter SDPO and not
> canonical OPSD). The blindness mechanism may behave differently in canonical
> on-policy SDPO/OPSD variants — F11 / F12 / F13 explore this with
> `train_mu_v7_opd.py` (closest D5 implementation to OPSD sampled-token
> policy-gradient variant per Zhao 2601.18734 Table 3). Disambiguation:
> `knowledge/current/CANONICAL_NAMING_REFERENCE.md`.

## Headline

Across all three lr regimes of the D5 in-house off-policy IS-loss variant, the Opus
critic's improvement_directive (e.g. "Add PUCT formula
`Q(s,a) = E[r] + c·P(s,a)·sqrt(N)/(1+n(s))`") successfully transfers to the
**TEACHER distribution** (PICK plans contain PUCT at iter 1-3) but **never to the
STUDENT EVAL distribution** (0/8 EVAL plans contain PUCT at any iter, all three lr
regimes).

## Empirical evidence

Source: `paper_materials/experiments/E4_pick_plan_content_evolution.md` and
`paper_materials/experiments/E5_eval_plan_content_evolution.md`

### Teacher (PICK) plans, μ-v4

| Iter | PUCT formula present | J_β formula | Advantage form `A=w−1−λlog(π/π₀)` | Q=max child |
|---:|:---:|:---:|:---:|:---:|
| 0 | 0/1 | 0/1 | 0/1 | 0/1 |
| 1 | 1/1 | 1/1 | 1/1 | 1/1 |
| 2 | 1/1 | 1/1 | 1/1 | 1/1 |
| 3 | 1/1 | 1/1 | 1/1 | 1/1 |
| 4 | (Goodhart bloat: LC-15, RSA-2048) | — | — | — |
| 5+ | (word salad, see F3) | — | — | — |

Iter 1-3 PICK plans **fully absorbed** the source-paper content the critic asked for.

### Student (EVAL) plans, μ-v4

| Iter | PUCT formula present | per-state β(s) | J_β formula | Mean audit |
|---:|:---:|:---:|:---:|---:|
| 0 | 0/8 | 0/8 | 0/8 | 24.00 |
| 1 | 0/8 | 0/8 | 0/8 | 24.75 |
| 2 | 0/8 | 0/8 | 0/8 | 24.88 |
| 3 | 0/8 | 0/8 | 0/8 | 27.25 |
| 4 | 0/8 | 0/8 | 1/8 | 28.00 |

Even at the **iter-4 peak**, only 1 in 8 EVAL plans included an entropic objective
formula, and **none** included PUCT or per-state β(s). The audit gain came from
**other dimensions** (better reproducibility specs, more named baselines, more concrete
risk-awareness analysis) — not from the specific formal content the critic was pushing
for.

## Mechanism

### What SDPO measures

The advantage at each token is `A_t = teacher_lp(token | teacher_ctx) - student_lp(token | student_ctx)`,
where `teacher_ctx = goal + oracle + critique` and `student_ctx = goal + oracle`.

Per-token advantage is large when:
- Teacher's context (with critique) makes a token highly probable
- Student's context (without critique) makes it improbable

### Where the gradient actually goes

The ABSOLUTE advantage `|A_t|` is dominated by tokens where the two contexts disagree
strongly per-token:

- **Stylistic tokens** (transitions like "However,", "Furthermore,", hedges like
  "approximately", "it is reasonable to assume"): teacher writes them often when
  responding to a critique pointing out structural gaps; student writes them less.
  Per-token entropy is high → divergence is high → advantage is high.
- **Content tokens** (PUCT formula tokens like "Q", "(", "s", ",", "a", ")", "=", " ",
  "E", ...): both teacher and student assign LOW probability because PUCT is a rare
  symbol sequence in pretraining. Teacher's probability is 5x student's, but both
  are still tiny (e.g. 0.01 vs 0.002), so per-token advantage is small.

The grad signal **integrates per-token advantages across all token positions**.
Stylistic tokens are 100x more numerous than content tokens. Even if content-token
advantages are positive, they're drowned by the volume of stylistic-token advantages.

Result: LoRA learns stylistic rephrasing of plan structure, not insertion of specific
formulas.

## Why this fails for content but works (eventually) for style

EVAL plan iter 4 audit gains (broad-spectrum +0.3-0.7 per dim) come from:
- **Style improvements** that approximate content properties (e.g. mentioning "we
  control for reward hacking" → +0.5 on T4 even without an actual probe)
- **Verbose elaboration** of the original plan's claims (longer, more reproducible-
  looking) → +0.7 on U5

These are real improvements but they are **not** "the model learned PUCT". They are
"the model learned to write more carefully under the same content constraints".

## What this means for the paper

- **The +2.75 lift is real but mechanistically different** from what the critic
  asked for. Paper should be honest: SDPO + critic teaches the model to *appear*
  more rigorous, not to *include* specific content.
- **Adding token-level critique signal** (insert literal PUCT formula in critique
  output) might fix this — but converts the loss form toward SFT-on-revisions
  (CR-v7 architecture). Not the elegant SDPO mechanism we wanted to keep.
- **Dynamic retrieval (Phase 3)** could potentially fix this differently: if the
  oracle content itself contains PUCT, the student-context prompt has PUCT
  available, and SDPO doesn't need to teach it — it just needs to teach the student
  to USE oracle content already in its prompt.

## Data pointers

- PICK plan evolution: `paper_materials/experiments/E4_pick_plan_content_evolution.md`
- EVAL plan evolution: `paper_materials/experiments/E5_eval_plan_content_evolution.md`
- PICK iter 3 with PUCT exemplar: `paper_materials/plan_samples/exemplar_pick_iter3_picked_PUCT.md`
- Discussion in Phase 3 design: `paper_materials/next_steps/N1_phase3_retrieve_then_generate.md`
