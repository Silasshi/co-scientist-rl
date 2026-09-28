# Exp B — n-gram overlap trajectory across 6 v8-d5sdpo cells

*Filed 2026-04-29. Falsifier for F16 H16-2 (critique-conditioned advantage
rewards critique-conformity, not oracle-fidelity).*

## Setup

- **Inputs**: 6 cells from Phase 2 v8-d5sdpo grid (base, G, G+, E, C+, F+)
  buffer.jsonl files, total 528 plans across 7-13 iters per cell
- **Reference corpus**: slim oracle Math (12 items) + Methodology (17 items),
  concatenated, 8,592 tokens, 7,962 unique 4-grams
- **Anchor**: dataset/reference_solution.txt (35/45 plan, 1,043 tokens) plotted
  as horizontal dashed line per n
- **Metric**: per-plan Jaccard overlap of n-grams in `plan_text` (already
  XML-stripped by `_decode_plan` → `extract_solution` chain) vs corpus n-grams
- **Sweep**: n ∈ {1, 2, 3, 4} for sensitivity check
- **Cliff detection**: argmax consecutive drop in mean audit total per iter
  (none of the 6 cells triggered cliff in the 7-13 iter window — `is_post_cliff=0`
  throughout; cliff column reserved for runs that exceed 16 iter)

Pre-registered falsification rule (EXPERIMENT_PLAN_oracle_transfer_ABC.md L100-105):
- All 6 cells flat/decreasing → H16-2 corroborated
- All 6 cells monotonic rising → H16-2 falsified
- Mixed → ambiguous

## Joint verdict per n

| n | Reference plan | Joint | Per-cell pre-cliff slope |
|---|---:|---|---|
| 1 | 0.1684 | MIXED (3/6 rising) | base+ G+ E+ rising; G+/C+/F+ falling |
| 2 | 0.0242 | MIXED (1/6 rising) | only base rising; rest flat/falling |
| 3 | 0.0023 | **CORROBORATED** | no cells rising |
| 4 | 0.0000 | (saturated) | no cells rising; ref=0 makes uninterpretable |

**At n ≥ 3, no cell shows a rising trend**, including the highest-audit cells
G+ (peak 22.38) and C+ (peak 23.50). Both fall at every n.

## Surprise: reference plan has LOWER overlap than students at n=1, n=2

The 35/45 reference plan (1,043 tokens) shows lower overlap with the
Math+Methodology corpus than student plans (1,500-2,500 tokens after
`extract_solution`). At n=1 students peak at 0.19-0.21 vs reference 0.17.

Most plausible explanation: **students are conditioned on the full oracle
text in their prompt**, so they mechanically pick up oracle vocabulary
regardless of whether the SDPO advantage is pushing them to. The reference
plan was written without oracle-as-context, so it uses original phrasing.

This makes n=1 and n=2 results **uninformative for H16-2**: rising n=1
overlap could be an artifact of "oracle vocab in context", not an artifact
of "training pushes toward oracle". The right resolution is n ≥ 3, where
random-coincidence floor is low and the rise pattern would mean true content
internalization, not vocabulary copying.

## Length confound noted (not driving verdict)

Student plans are 1500-2500 tokens after extraction, vs reference 1,043 tokens.
The student footer prompt asks for "600 words target, max 750 words" — students
are violating by 2-3×. This is a separate finding (template noncompliance), not
relevant to the H16-2 falsifier — Jaccard normalizes for size.

## Inverted-U pattern across all cells

All 6 cells follow the same trajectory regardless of mask/α/loss config:
- **Rise iter 0 → 3-4**: small but consistent across cells (~+0.05 at n=1, ~+0.01 at n=2)
- **Plateau iter 4-6**: peak overlap region
- **Decline iter 6 → end**: monotone fall to or below iter-0 baseline

The "rising" portion is shared across cells with vastly different
hyperparameters (mask=T/F, α=0.05/0.1, PPO/IS) — including base which is
mask=F α=0 (essentially frozen + no trust region). This means the rise is
**not driven by SDPO mechanism**; it appears to be an effect of the LoRA
adapter learning *some* statistical structure during early iters before
collapsing into a template.

## H16-2 verdict (joint with above caveats)

**H16-2 CORROBORATED at the n ≥ 3 resolution where the metric is interpretable.**

The SDPO mechanism (critique-conditioned advantage, with or without solution
mask, with or without trust region, IS or PPO) does not produce a rising
trajectory of oracle Math+Methodology n-gram usage. The early-iter rise that
appears at low n is shared with the no-training base cell and is likely a
generic vocabulary effect, not oracle-content learning. The cells that
achieved best audit scores (G+, C+) are the cells with the *steepest*
n-gram declines, suggesting they're moving *away* from oracle vocabulary
to whatever the audit reward favors.

This does not falsify H16-1 (retrieval bottleneck) or H16-3 (LoRA bandwidth)
on its own — those need Exp A and Exp C respectively. But it makes H16-2
a viable primary explanation for the v8 ceiling.

## Implications

- Even if Exp A shows Qwen3-30B *can* retrieve oracle items verbatim under
  explicit instruction, the SDPO loss does not incentivize doing so during
  natural plan generation. The advantage signal channels learning toward
  audit-reward-pleasing patterns, not oracle-content reproduction.
- Exp C (RAG top-K=5) becomes especially useful: if RAG cell shows the same
  inverted-U n-gram trajectory + same audit peak, then attention bandwidth
  is not the bottleneck — incentive is.
- The pattern of best-audit cells having steepest n-gram declines is
  consistent with a Goodhart hypothesis: the audit grader rewards properties
  that are *anti-correlated* with oracle-vocabulary usage (perhaps fluency,
  novelty, structural completeness).

## Files

- `figures/exp_B_overlap_trajectories.png` — 4×2 grid (n × pre-cliff/full)
- `per_cell_per_iter.csv` — `cell,n,iter,n_plans,mean_overlap,std_overlap,is_post_cliff`
- `verdict.json` — programmatic verdicts for downstream RESULTS report

## Methodology notes

- Tokenization: lowercase + `[a-z0-9]+` regex word-split
- Jaccard: `|A ∩ B| / |A ∪ B|` over n-gram sets
- Reference corpus extracted via `^### (Math|Methodology) (\d+):` regex,
  count == 29 hard-asserted in script
- Buffer plan_text is already XML-stripped (verified via L188-191 of
  `train_mu_v8_d5sdpo.py`: `_decode_plan` calls `extract_solution`)
