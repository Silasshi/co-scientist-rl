# Research directions

The project ran six directions between Dec 2025 and Apr 2026, all now **concluded**. Each has code under `src/co_scientist/<dir>/` (featured) and `experiments/<dir>/` (iteration history), and records under `projects/<dir>/`: configs, run summaries, `DECISIONS.md`, `STATUS.md` and findings. Inside the per-direction `DECISIONS.md` / `STATUS.md` files, "active" and target-venue lines are historical.

| ID | Directory | Question | Outcome |
|---|---|---|---|
| D1 | `rubric_reward` | Does rubric-reward GRPO (Goel et al.) work at 30B, and can reward shaping, self-distillation, two-stage refinement, rubric dropout, multi-turn training or a learned selector beat it? | Reproduction 0.560 → 0.693 held-out. Nothing beat it: only two variants had formal evals (0.654, 0.657), the rest were screened on training reward. Selection-gap analysis: oracle BoN-8 0.849 vs self-selection 0.694 |
| D2 | `ibt` | Does iterative, dialogue-style brainstorming (4B policy, 30B grader) improve plans beyond single-pass GRPO? | Inconclusive; the hint and RL effects could not be cleanly separated |
| D3 | `ttt_discover` | Can a goal-agnostic multi-signal reward with test-time RL replace goal-specific rubrics? | Critique conditioning gave +0.19; RL gave +0.087 only when training the exploration policy. Signal hardening fixed a reward that ranked plans against the independent judge within a saturated band |
| D4 | `grant_proposal`, `grant_proposal_v2` | Does the D3 pipeline transfer to grant proposals with public funder rubrics? | RL pushed the proxy reward to 0.970 while the independent judge gave 11 / 20; critique–revise without RL got 16 / 20. Diagnosed as template collapse (proxy and judge still agree across conditions, ρ = +0.91) |
| D5 | `d5_abstract_retrieve_refine` | Can self-distillation with a privileged "oracle" of the source paper teach cross-paper abstraction? | Small, single-goal gain: 28.00 vs 25.25 / 45, corroborated 20 / 24 pairwise. Later variants: one pre-registered claim falsified, one headline retracted, verifier-grounded reward Goodharted, KL anchor stabilised training but did not raise the peak |
| D6 | `d6_grant_proposal` | The D5 on-policy distillation recipe on grant proposals, with a reward-only GRPO control | Implemented with the control; the project concluded before the pilot sequence finished |

See the [README](README.md#key-findings) for the cross-direction findings and [`docs/algorithms/`](docs/algorithms/) for the methods.
