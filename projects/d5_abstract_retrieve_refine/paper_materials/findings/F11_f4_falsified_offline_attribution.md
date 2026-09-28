# F11 — F4 gradient-flooding mechanism FALSIFIED via offline attribution

> **⚠ TERMINOLOGY NOTE (2026-04-27 PM)**: F11 references "SDPO advantage" —
> this is shorthand for the D5 in-house off-policy IS-loss variant's advantage
> signal `A_t = clamp((t_lp − s_lp)·scale, ±5)`. The variant is structurally
> distinct from canonical Hübotter SDPO (2601.20802) and canonical OPSD (Zhao
> 2601.18734); see `knowledge/current/CANONICAL_NAMING_REFERENCE.md`. The
> falsification is valid for the in-house variant and (by construction —
> off-policy datum + scalar-logprob A_t) was directly measurable; whether the
> same falsification holds for canonical SDPO/OPSD's vocab-level loss forms is
> not tested here.

## Headline

The original F4 finding ("critique-token-blindness via gradient flooding") claimed:
the D5 in-house off-policy IS-loss advantage is dominated by stylistic tokens (~100×
more numerous than content tokens) so that content-token gradient mass approaches 0%.
This was used to explain three D5 NULL results (F3 multi-round cliff, F7 cross-goal
NULL, F9 continual NULL).

**Stage A offline attribution falsifies F4 quantitatively** (n=8, μ-v4 iter-2 buffer plans replayed through saved iter-2 sampler weights):

```
Mean content_grad_mass_frac = 15.83% (range 12.99% - 19.35%)
Content density            = 18.55%  (so content tokens get ~85% of proportional gradient mass)
Mean |adv| at content      = 0.40
Mean |adv| at stylistic    = 0.49
Ratio                      = 0.83  (NEAR-PROPORTIONAL, NOT flooded)
```

F4 predicted content_grad_mass_frac < 0.5%. Observed 15.83%. **F4's quantitative mechanism is wrong by ~30×.**

## What's still real

The empirical PICK-vs-EVAL gap F4 cited IS real:
- Teacher PICK plans contain literal PUCT formula 1/1 across iters 1-3 (μ-v4 buffer)
- Student EVAL plans contain PUCT 0/8 across ALL iters 0-7 (μ-v4 eval_rollouts)

But the explanation is NOT gradient flooding. The gradient does reach content tokens at proportional magnitude. Some other mechanism prevents student EVAL from acquiring PUCT. Stage B (F12) tested and ruled out conditional-context gating; the surviving mechanism is **prefix-prior lock-in**.

## Methodology

- **Setup**: Replay 8 μ-v4 iter-2 buffer plans through saved iter-2 sampler weights. For each plan, recompute `teacher_lp` (under goal+oracle+iter-1-critique) and `student_lp` (under goal+oracle, no critique). Per-token advantage A_t = clamp((t_lp − s_lp) · 1.0, [−5, 5]) — same formula as live training.
- **Classification**: Each token classified content/stylistic via regex superset (digits, math operators, Greek glyphs+names, callable forms like Q(s)/J(θ)).
- **Aggregate**: `content_grad_mass_frac = sum(|adv| at content positions) / sum(|adv| at all positions)`.
- **Cost**: ~$3 (Tinker logprob compute only; NO Opus, NO training).
- **Wall**: ~30 min.

Full code: `src/co_scientist/d5_abstract_retrieve_refine/f4_attribution_offline.py`. Output: `runs/2026_04_29_f4_attribution_offline/f4_attribution_summary.json`.

## Implication for paper

F4 was load-bearing for D5's prior NULL-result narrative. Its falsification is a **truth-in-reporting commitment** — the paper must:
1. Cite F4's original PICK-vs-EVAL evidence (real empirical pattern).
2. Document the gradient-flooding hypothesis it spawned (now disproven).
3. Cite Stage A measurement (15.83% content mass) as the falsifier.
4. Pivot to Stage B + F12 (prefix-prior lock-in) + F13 (canonical OPD on-policy fix) for the actual mechanism + fix narrative.

The reframe is **stronger** than F4: prefix-prior lock-in unifies F3/F7/F9 NULLs more cleanly (each NULL involves student-context-only sampling failing to reach content reachable only via teacher-context); and canonical OPD architecturally fixes F2 (off-manifold gradient) which F4 conflated with flooding.

## Data pointers

- Stage A script: `src/co_scientist/d5_abstract_retrieve_refine/f4_attribution_offline.py`
- Stage A summary: `runs/2026_04_29_f4_attribution_offline/f4_attribution_summary.json`
- F4 (original, now falsified): `paper_materials/findings/F4_critique_token_blindness.md`
- F12 (Stage B mechanism): `paper_materials/findings/F12_prefix_prior_lock_in.md`
- F13 (canonical OPD fix): `paper_materials/findings/F13_mu_opd_canonical_architecture.md`
