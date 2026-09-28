# D1 Rubric vs. Opus Depth Audit — Calibration Check

Opus 4.7 blind score on 5 plans pulled from D1 (rubric_reward) run.
Prompt: D3 canonical depth audit (`src/co_scientist/ttt_discover/opus_eval_agent.py:DEPTH_AUDIT_PROMPT`, 4 dims x 1-5, max 20).

## Structural caveat (read first)

The input file labels this group with a single `goal_inferred = "FP8 training stability of SwiGLU / Dynamic Clamping"`, but reconstruction from each plan's `grader_output` rubric criteria shows the 5 plans actually come from **five different research goals**:

| sample_idx | reconstructed goal (from grader rubric)                         | rubric score |
|------------|-----------------------------------------------------------------|--------------|
| 4          | FP8 SwiGLU via Smooth-SwiGLU per-channel scaling                | 0.314        |
| 1          | Holistic environmental impact of a 13B LLM (carbon + water)     | 0.600        |
| 5          | Relaxing strict equalized-odds fairness constraint              | 0.757        |
| 6          | Relaxing strict equalized-odds fairness constraint (SAME as 5)  | 0.834        |
| 7          | Model-based representations for model-free RL (no planning)     | 0.949        |

Plans 5 and 6 are the only pair with an identical rubric, so they're the only valid direct rubric comparison. Everything else is across-goal and measures "match to goal-specific expected solution", which is not a pure plan-quality signal.

## Blind Opus scores

| sample_idx | rubric (D1) | math | novelty | realism | rigor | Opus total / 20 |
|------------|-------------|------|---------|---------|-------|-----------------|
| 4          | 0.314       | 2    | 3       | 2       | 2     | 9               |
| 1          | 0.600       | 1    | 1       | 2       | 2     | 6               |
| 5          | 0.757       | 2    | 1       | 1       | 2     | 6               |
| 6          | 0.834       | 2    | 2       | 3       | 3     | 10              |
| 7          | 0.949       | 2    | 2       | 1       | 2     | 7               |

## Correlation

- Pearson(rubric, opus_total) = **-0.167**
- Spearman(rubric, opus_total) = **+0.154**
- Both near zero on n=5 (well inside noise).

Ordering agreement: rubric ascending (4 < 1 < 5 < 6 < 7) maps to opus totals (9, 6, 6, 10, 7). Only one pair (5 -> 6) is monotone.

## Same-rubric case (plans 5 vs 6)

Plans 5 and 6 share the identical equalized-odds rubric. D1 ranks 6 > 5 (0.834 > 0.757), Opus also ranks 6 > 5 (10 > 6). **Within a shared rubric, D1 and Opus agree directionally.** Plan 6 avoids plan 5's key realism error (ResNet-18 on tabular Adult Income), uses real algorithms (NSGA-II) and libraries (AIF360), and has coherent ablations. This is a positive, narrowly scoped calibration signal.

## Across-goal case (all 5)

D1's highest-rubric plan (idx 7, MBPO-LD, 0.949) scores 7/20 on Opus — **below** plans 6 (10/20) and 4 (9/20). It contains the clearest factual fabrications in the set:
- Invented baseline "SBPI" (not a real algorithm),
- Claims MBPO runs at 4.1M steps/sec on A100 (roughly 1000x too high),
- Self-contradiction "6k stars, 200+ stars",
- Claims to be first model-free method with latent dynamics while ignoring Dreamer / SLAC / PlaNet.

D1's lowest-rubric plan (idx 4, Dynamic Clamping, 0.314) scores 9/20 on Opus. It was rubric-penalized not for poor quality but for proposing Dynamic Clamping instead of the rubric's expected solution (Smooth-SwiGLU per-channel scaling). Its intrinsic depth is comparable to mid-rubric plans.

## Verdict (paper-facing)

1. **Across-goal D1 rubric is not a cross-goal plan-quality measure.** On this slice, Pearson -0.17 / Spearman +0.15 between rubric and depth. The rubric rewards solution-match within each goal; aggregating it across goals mixes match-to-expected-answer with plan quality.
2. **Within a shared rubric D1 is not noise** (plans 5 vs 6 show directional agreement with Opus), but it can still mis-rank on internal quality: plan 7 got 0.949 on its own rubric despite obvious realism failures, which means the rubric is vulnerable to polished template-filling when goal-match keywords are present.
3. **Implication for the paper.** Any D1 vs. universal-reward (D3 / D4) comparison should either (a) pool only within-goal deltas or (b) explicitly label D1's across-goal mean as a goal-match rate, not a plan-depth score. Using D1's aggregate rubric score as a proxy for plan quality in a cross-goal comparison is misleading.
4. **n=5 caveat.** Sample size is too small to generalize numerically, but sample_idx=7 alone is sufficient to demonstrate that D1's within-goal rubric can assign its maximum score to a plan with clear factual fabrications.

## Plan-level notes

- **idx 4 (rubric 0.314, opus 9):** On-goal for FP8 stability, but rubric expected Smooth-SwiGLU. Plan proposes Dynamic Clamping with a clamp formula tied to sigmoid saturation. Fabricates "LLaMA-7B 8-bit GPT-3 variant" and "NVIDIA FP8 APIs v2.2+".
- **idx 1 (rubric 0.600, opus 6):** Holistic LCA. No equations, standard methodology. Training emissions (850 kg CO2e for 13B x 1024 A100 x 10 epochs) are roughly 2 orders of magnitude too low; percentages don't sum to 100; citations likely fabricated.
- **idx 5 (rubric 0.757, opus 6):** Fairness Pareto. ResNet-18 on tabular Adult Income is incoherent. "20x20 grid over scalar lambda" is nonsensical. Arbitrary per-method lambdas undermine the "unified framework" framing.
- **idx 6 (rubric 0.834, opus 10):** Same rubric as idx 5, genuinely better. COMPAS+Adult, NSGA-II, AIF360, coherent FAT formula, plausible ablations.
- **idx 7 (rubric 0.949, opus 7):** MBPO-LD. Fabricated "SBPI" baseline, implausible 4.1M steps/s MBPO, self-contradictory repo stats, ignores Dreamer/SLAC/PlaNet. Polished prose, lowest realism.

Source data: `projects/d5_abstract_retrieve_refine/analysis/d1_rubric_validation/data/d1_5plans_group215_2.json`
Structured output: `../data/d1_opus_audit.json`
