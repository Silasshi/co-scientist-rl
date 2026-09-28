# D5 Smoke Test — Opus Depth Audit Summary

**Run**: `2026_04_smoke_pathway_v1`  
**Audit date**: 2026-04-23  
**Auditor**: Opus 4.7 (blind to condition during scoring)


## Scoring rubric

- **depth** (1-10): mechanistic grounding, argument coherence, non-generic engagement with the core.
- **methods** (1-10): concreteness of training procedure — objective, update rule, data flow, hyperparameters.
- **feasibility** (1-10): doability on open-weight models within fixed compute; compute-matched baselines; realism.
- **grounding** (1-10): citations, numbers, named datasets/benchmarks. (Expected uniformly low, per user hypothesis.)


## Summary (group means over n=8 per condition)

| metric | B_baseline | A_with_abstraction | Δ (A − B) |
|---|---|---|---|
| depth | 3.375 | 5.0 | +1.625 |
| methods | 3.375 | 4.25 | +0.875 |
| feasibility | 3.5 | 4.5 | +1.000 |
| grounding | 2.0 | 2.0 | +0.000 |
| total | 12.25 | 15.75 | +3.500 |


## Per-plan scores

| condition | sample_idx | depth | methods | feasibility | grounding | total | comment |
|---|---|---|---|---|---|---|---|
| B_baseline | 0 | 3 | 3 | 3 | 2 | 11 | Generic search-update cycle with vague 'reward as loss' phrasing; no specific objective, model, or hyperparameters; truncated at Evaluation. |
| B_baseline | 1 | 3 | 3 | 4 | 2 | 12 | Names MCTS and a compute budget (1000 passes) but phrases updates as 'minimize negative reward'; evaluation domains (protein folding, quantum circuit design) are named but without baselines or datasets. |
| B_baseline | 2 | 3 | 3 | 3 | 2 | 11 | Policy-gradient/reward-weighted MLE named generically; no concrete formulas or hyperparameters; domain list generic. |
| B_baseline | 3 | 3 | 3 | 3 | 2 | 11 | Writes a toy SGD rule (theta += eta * grad r) that is not actually a valid RL objective; domain list generic; no baselines. |
| B_baseline | 4 | 4 | 4 | 4 | 2 | 14 | Mentions LoRA, EWC, 100 gradient steps budget and significance testing; still generic on objective and no concrete model or numeric targets. |
| B_baseline | 5 | 3 | 3 | 3 | 2 | 11 | Duplicate of plan idx=2 — same generic policy-gradient framing with no specifics. |
| B_baseline | 6 | 4 | 4 | 4 | 2 | 14 | Duplicate of plan idx=4 — LoRA, EWC, 100-step budget mentioned; still no concrete objective or model. |
| B_baseline | 7 | 4 | 4 | 4 | 2 | 14 | Duplicate of plan idx=4 / idx=6 — same LoRA/EWC framing. |
| A_with_abstraction | 0 | 5 | 4 | 5 | 2 | 16 | Explicitly enumerates 4 failure modes (objective mismatch, horizon, exploration collapse, average vs max) and proposes state buffer + tree-search + adaptive hyperparameters + importance sampling; compute-matched Best-of-N baseline named; still no formulas/model/numbers. |
| A_with_abstraction | 1 | 5 | 5 | 5 | 2 | 17 | Cleanly lists 5 components (buffer, tree-search-over-past-states, adaptive hyperparameters, importance-sampled gradients, max-focused training); evaluation names protein folding/chemistry/theorem proving with Best-of-N + sample efficiency + generalization tests. |
| A_with_abstraction | 2 | 5 | 5 | 5 | 2 | 17 | 4-component method (buffer, tree-search with lineage diversity, adaptive per-state hyperparameters w/ divergence threshold, importance sampling); evaluates on 3 deterministic-reward domains with ablations. |
| A_with_abstraction | 3 | 5 | 4 | 4 | 2 | 15 | Shorter version of same structure — max-reward objective, cumulative state buffers, tree-search over past states, adaptive hyperparameters with divergence budget, importance sampling; less elaboration than plans 9/10. |
| A_with_abstraction | 4 | 5 | 4 | 5 | 2 | 16 | 3-component method (state buffering, tree-search with ancestry overlap, adaptive hyperparameter tuning with divergence budget) + importance sampling; ablation studies and compute-matched baselines mentioned. |
| A_with_abstraction | 5 | 5 | 4 | 4 | 2 | 15 | Compact version of the same 4-component pipeline (cumulative buffer, tree-search-over-past-states, adaptive hyperparameters, importance-sampling correction); domains named (molecular design, physics sims) but shallow. |
| A_with_abstraction | 6 | 5 | 4 | 4 | 2 | 15 | Three-phase formulation (initial exploration, state accumulation, iterative refinement) with tree-search, divergence-budgeted adaptive hyperparameters, importance sampling, and explicit max objective. |
| A_with_abstraction | 7 | 5 | 4 | 4 | 2 | 15 | Duplicate of plan idx=14 — same three-phase formulation and components. |


## Qualitative verdict

The oracle abstraction clearly helps: mean total score rises from 12.25/40 (B_baseline) to 15.75/40 (A_with_abstraction), a delta of +3.5 points driven almost entirely by depth (+1.625), feasibility (+1.0), and methods (+0.875). Grounding is flat at 2.0 in both conditions, matching the hypothesis that a retrieval-free frozen Qwen3-30B cannot manufacture citations or specific numerical baselines regardless of prompt scaffolding — so this dimension is a genuine control. No A plan scored below 15; no B plan scored above 14. The conditions are cleanly separable with this n=8 per side.

Qualitatively, A plans adopt the abstraction patterns almost literally. Every A plan names four signature components — cumulative state buffer, tree-search over past states, adaptive per-state hyperparameters under a divergence budget, and importance-sampling correction — and every A plan frames the problem with 3-4 enumerated failure modes followed by a max-aligned (rather than mean-aligned) training objective. That vocabulary is almost never present in B plans, which instead default to 'gradient descent on reward as loss' or 'policy gradient' without articulating the objective mismatch or the horizon problem. Plans 9 and 10 are the strongest A outputs (17/40) because they add ablation structure and sample-efficiency / generalization sub-metrics on top of the pattern vocabulary; plans 8 and 12 follow at 16. The best B plans (idx=4/6/7, all duplicates, at 14/40) earn methodological concreteness only by naming LoRA + EWC + 100-step budget — none of which come from the abstraction and none of which address the core discovery-objective mismatch.

Two caveats for interpretation. First, the A plans improve by importing the abstraction's vocabulary wholesale rather than by deriving it from first principles — this is structural transfer, not mechanistic reasoning, so the observed lift measures what a frozen model can restyle from a well-formed prior, not what it understands. No A plan writes the entropic J_beta objective, the PUCT formula, the rank-based prior, or specifies a concrete model (gpt-oss-120b), LoRA rank, learning rate, or numeric result target — all of which are in the reference solution. Second, decoding appears partially deterministic or seed-collided: B idx=2/5 are identical; B idx=4/6/7 are identical; A idx=14/15 are identical. This collapses effective sample diversity from 16 to ~11 unique plans. The A > B gap survives this (means computed over all 8 per side are still clean), but future runs should verify the sampler is truly diverse before drawing stronger conclusions from per-sample comparisons.


## Data quality notes

- Decoding produced duplicate outputs: B idx=2 == B idx=5; B idx=4 == B idx=6 == B idx=7; A idx=14 == A idx=15. Effective unique plans: ~11 of 16.
- The A > B gap remains clean under the full n=8 means, but per-sample variance is understated by the duplication. Future runs should verify sampler diversity (seed handling, temperature).
