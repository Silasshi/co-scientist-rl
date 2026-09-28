# Research Directions

This project has 7 research directions. Each direction has its own code, configs, runs, analysis, and knowledge under `projects/{direction}/`. Shared resources (papers, tools, env) live under `shared/`.

## Directions

| ID | Directory | Description | Status |
|---|---|---|---|
| D1 | `rubric_reward` | Reproduce & extend "Training AI Scientists Using Rubric Rewards". Single-scalar rubric grader + GRPO/SDPO/reward shaping/doublegeneration/multi-turn/selector. | Best: 0.693. Ceiling reached. |
| D2 | `ibt` | Iterative Brainstorming Training. Iterative dialogue-style improvement with 4B policy + 30B grader. | Run 7: 106 batches. Under investigation. |
| D3 | `ttt_discover` | TTT-Discover-style per-goal optimization with multi-signal reward. Universal plan quality signals. | Signal v8.1 + CR-v5 pipeline: AUC 0.767 on 60 refs + 162 perturbations. 3 paper runs pending launch. |
| D4 | `grant_proposal` | Grant proposal generation using CR-v7 pipeline + forked multi-signal reward. Public rubrics (NSF/NIH/ERC). | 58 runs complete. B4 best on Opus (19/40). RL Goodharts. Legacy — see D4v2. |
| D4v2 | `grant_proposal_v2` | Investigation phase: diagnosing Qwen-Opus quality inversion. Same code as D4. | Tabled (pivot to D5). |
| **D5** | `d5_abstract_retrieve_refine` | SDPO-style self-distillation with privileged observations (source paper as privileged info) for long-form research plan generation, via multi-round model-selected retrieval + abstraction. | **Active. Phase 2 COMPLETE — μ-v4 iter 4 = 28.00/45 (+2.75 over σ), first 30B-trained variant to beat σ frozen+oracle baseline. Pairwise corroborated 20/24 (vs σ 6-2, vs δ 7-1, vs α 7-1). 8-baseline audit_v3 ISOLATED ranking: ε 33.62 ≫ μ-v4 28.00 ≫ σ 25.25 ≈ δ ≈ α > μ-v2 23.88 > β > ξ. Phase 3 (2026-04-27 PM, F8 v2): τ_v4 inference-time pipeline = 24.50/45 (n=8, statistically comparable to σ); original 7-1 STRONG headline retracted on patched + fresh plans (Step E pairwise 4-4 TIE). See `projects/d5_abstract_retrieve_refine/DECISIONS.md:7-44`. Target: ICLR 2027.** |
| **D6** | `d6_grant_proposal` | D5 ERR pipeline applied to D4 grant proposal task. Oracle = reference_proposal.md; V8 trainer (sol-mask + trust-region + PPO-clip); V9-L1 grounded (+citation/eq bonus); GRPO ablation (no reviewer); σ baseline; 3 domain goals (AI/CS, NatSci, SocSci). | **Active. Phase 2 complete (V8, V9-L1, GRPO, σ, gold extraction all implemented). Pilot sequence (oracle build → σ → V8 ‖ GRPO → V9-L1) pending on ai/02_foundational_rl.** |

## Code layout

```
src/co_scientist/
  shared/              # Cross-direction: api_profiles, eval_core, openrouter_client, seven_signal_reward
  rubric_reward/       # D1 trainers: grpo/, sdpo/, reward_shaping/, multiturn/, refinement/, selector/
  ibt/                 # D2 trainers: train_ibt, train_ibt_v2, baselines, self_calibration
  ttt_discover/        # D3 trainers: train_7signal_vector
  grant_proposal/      # D4 trainers: train_cr_v7, train_buffer_ttt, train_critique_revise, opus_eval_agent
  d5_abstract_retrieve_refine/  # D5 trainers: train_mu_v4 (production SDPO), train_sigma_v2/delta/alpha/beta/xi/epsilon (baselines), audit_v3_isolated, mu_v4_pairwise_v1
  d6_grant_proposal/             # D6 trainers: train_opd, train_opd_grounded, train_grpo, train_baseline, extract_oracle, extract_gold, audit
  eval/                # Shared eval scripts: eval_bon, eval_cpr, eval_ibt, eval_multiturn, etc.
```

## Project layout

```
projects/
  rubric_reward/            # D1 configs, runs, analysis, knowledge, data
  ibt/                      # D2 configs, runs, analysis, knowledge, data
  ttt_discover/             # D3 configs, runs, analysis, knowledge, data, tools
  grant_proposal/           # D4 configs, runs, analysis, knowledge, data
  grant_proposal_v2/        # D4v2 analysis + runs (tabled)
  d5_abstract_retrieve_refine/  # D5 configs, runs, analysis, knowledge (dataset symlinks to grant_proposal/dataset)
  d6_grant_proposal/             # D6 configs, runs, analysis, knowledge, dataset/{domain}/{goal}/, data/{domain}/{goal}/oracle/

shared/                # Cross-direction: docs, papers, knowledge, tools, env
external_repos/        # Cloned external repos (OpenClaw-RL, etc.)
```

## Adding a new direction

1. Create `src/co_scientist/{new_direction}/` with `__init__.py`
2. Create `projects/{new_direction}/` with subdirs: configs/, runs/, analysis/, knowledge/, data/
3. Add an entry to this table
4. Import shared code via `from co_scientist.shared.xxx import yyy`
