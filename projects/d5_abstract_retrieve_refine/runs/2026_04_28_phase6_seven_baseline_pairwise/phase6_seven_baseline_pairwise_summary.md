# D5 D3 — 7-Baseline Pairwise Corroboration (M8 broader validation)

*Generated 2026-04-26 20:31:46*  
*Seed: 70, n_pairs/matchup: 8*  
*Per DECISIONS 2026-04-28 (g). Strict 1 subagent/pair task isolation.*

## 3 Matchups (all close-cluster from Phase 2F audit_v3 isolated)

Phase 2F 2-plan/subagent audit Δ:
- σ 25.25 vs δ 25.12: Δ +0.13
- σ 25.25 vs α 25.00: Δ +0.25
- δ 25.12 vs α 25.00: Δ +0.12

Pre-registered binding (DECISIONS 2026-04-28 (g)):
- All 3 matchups within ±5/8 → 7-baseline close cluster IS true null; M8 cross-goal effect doesn't generalize
- ≥2/3 matchups break ≥6/8 in same baseline's favor → 2-plan audit DID rank correctly; M8 effect is cross-goal-specific
- Mixed → per-matchup nuance documented

| Matchup | la wins | lb wins | ties | A-pos rate | Winner |
|---|---:|---:|---:|---:|---|
| **sigma_vs_delta** | 6 (sigma) | 2 (delta) | 0 | 2/8 (25%) | **sigma** |
| **sigma_vs_alpha** | 3 (sigma) | 5 (alpha) | 0 | 7/8 (88%) | **alpha** |
| **delta_vs_alpha** | 3 (delta) | 5 (alpha) | 0 | 7/8 (88%) | **alpha** |

## Per-matchup verdict (vs Phase 2F audit Δ)

### σ vs δ close-cluster
- Audit Δ: +0.13 / 45 (effective tie)
- Pairwise: 6 sigma / 2 delta / 0 TIE
- **PASS** — sigma wins 6-2 ≥ 6/8 threshold.

### σ vs α close-cluster
- Audit Δ: +0.25 / 45 (effective tie)
- Pairwise: 3 sigma / 5 alpha / 0 TIE
- **INCONCLUSIVE** — 3-5 (within ±5/8 of tie).

### δ vs α close-cluster
- Audit Δ: +0.12 / 45 (effective tie)
- Pairwise: 3 delta / 5 alpha / 0 TIE
- **INCONCLUSIVE** — 3-5 (within ±5/8 of tie).


## Position-bias check

A-side win rate per matchup (should be 25-75% — outside this range = position-bias confound):

- sigma_vs_delta: 25% (✓)
- sigma_vs_alpha: 88% (⚠️ POSITION BIAS)
- delta_vs_alpha: 88% (⚠️ POSITION BIAS)

## Per-pair rationales

### sigma_vs_delta

- **sigma_vs_delta_pair_00**: B
    > Plan B dominates on the operational/specificity dimensions the rubric prioritizes: it provides a full RHS-complete weighted policy gradient ($\nabla J(\theta) = \mathbb{E}_{\tau \sim B}[w(r_\tau)\nabla_\theta \log \pi_\theta(a|s,d)]$ with $w(r)=\text{sigmoid}(\alpha r)$), names the base model (Llama
- **sigma_vs_delta_pair_01**: B
    > Plan B dominates on methodological rigor and goal alignment. It anchors its design in named prior work with real systems as baselines (MiGrATe, TTRL/Hubert et al. 2025, AlphaEvolve/Georgiev et al. 2025, RS-GRPO/Jiang et al. 2025, Akyurek et al. 2024) and explicitly justifies RS-GRPO's exponential ut
- **sigma_vs_delta_pair_02**: B
    > Plan B (TTRL-Discover) wins on the higher-weight dimensions despite having weaker hyperparameter specificity. On Significance/Soundness, Plan B's choice of RS-GRPO with an exponential-utility objective (Jiang et al., 2025) is directly motivated by the goal's 'single best solution' evaluation criteri
- **sigma_vs_delta_pair_03**: B
    > Plan B wins on goal-alignment, soundness, and originality, while Plan A only wins on hparam specificity. The goal explicitly requires optimizing for the single best solution, and Plan B operationalizes this via risk-sensitive RL with an exponential-utility GRPO objective (Jiang et al., 2025) — a pri
- **sigma_vs_delta_pair_04**: A
    > Plan A dominates on goal alignment, soundness, and methodological rigor. Most importantly, A explicitly targets the goal's 'single best solution' evaluation by adopting the risk-sensitive RL exponential-utility objective from Jiang et al. 2025 — a direct mathematical match to evaluating max-quality 
- **sigma_vs_delta_pair_05**: B
    > Plan B dominates on methodological rigor and reproducibility. B names concrete prior methods with author/year (GRPO from Guo 2025, SIFT from Hubotter 2024, risk-sensitive RL from Jiang 2025, per-instance reset from Li 2025) and specifies a concrete compute budget (10-20 gradient steps per problem) p
- **sigma_vs_delta_pair_06**: A
    > Plan A wins on soundness, methodological rigor, and feasibility. A grounds each component in specific cited prior work with attributed mechanisms (Hubert et al. 2025 for synthetic curricula a la AlphaProof, Jiang et al. 2025 for RS-GRPO with exponential-utility objectives, Georgiev et al. 2025 for p
- **sigma_vs_delta_pair_07**: B
    > Plan B writes an RHS-complete reward-weighted policy gradient objective L(theta) = -E_tau[w(tau) * log pi_theta(a|s,d)] with w(tau) = exp(beta*R(tau)) / sum exp(beta*R(tau')), and explains how beta amplifies high-reward trajectories — directly addressing the goal's 'single best solution' criterion. 
### sigma_vs_alpha

- **sigma_vs_alpha_pair_00**: A
    > Plan A has stronger methodological specificity and broader scholarly grounding. It names the concrete RL algorithm (GRPO with exponential-utility objective from Jiang et al. 2025), explicitly invokes SIFT (Hubotter et al. 2024) as a diversity-aware selector for past attempts, prescribes parameter re
- **sigma_vs_alpha_pair_01**: A
    > Both plans assemble the same toolkit (RS-GRPO/Jiang 2025, program-space search/Georgiev 2025, synthetic-variant curriculum/Hubert 2025) so they trade closely on Originality and Soundness. Plan A wins on Methodological Rigor and Specificity: it names concrete benchmarks with citations (AIME, MATH-500
- **sigma_vs_alpha_pair_02**: A
    > Plan A and Plan B propose structurally similar TTT pipelines (synthetic variant curriculum + risk-sensitive GRPO with exponential utility + per-problem updates), but A is more specific and methodologically rigorous on the dimensions that matter for reproducibility and validation. A names a concrete 
- **sigma_vs_alpha_pair_03**: B
    > Plan B edges out on operational specificity and methodological structure. Plan B explicitly names LoRA adapters as the parameter-update vehicle (a load-bearing choice for fixed-budget TTT that Plan A omits with vague 'lightweight parameter updates'), splits methodology into 5 numbered components (Pr
- **sigma_vs_alpha_pair_04**: A
    > Plan A is narrowly better on methodological rigor and specificity. It names concrete benchmark suites (AIME, MATH-500, matrix multiplication, physics reasoning) where Plan B only says 'math proofs, physics simulations'; it names the optimizer (GRPO, Guo et al. 2025) explicitly rather than the generi
- **sigma_vs_alpha_pair_05**: A
    > Both plans converge on the same architectural recipe (TTRL + program-space search + synthetic curriculum + SIFT diversity buffering + risk-sensitive objective from Jiang 2025), so the discrimination must come from specificity and goal alignment. Plan A wins on evaluation alignment: it commits to the
- **sigma_vs_alpha_pair_06**: A
    > Both plans recombine the same toolkit (synthetic-variant curriculum, RS-GRPO with exponential utility per Jiang et al. 2025, program-space search per Georgiev et al. 2025, SIFT data selection per Hubotter et al. 2024) and target identical math benchmarks (AIME2024, MATH-500), so the differentiator i
- **sigma_vs_alpha_pair_07**: A
    > Plan A dominates on reproducibility and methodological rigor by citing real author+year references for every named component (Hubert et al. 2025 TTRL, Jiang et al. 2025 RS-GRPO with exponential-utility, Hubotter et al. 2024 SIFT, Novikov et al. 2025 AlphaEvolve, Georgiev et al. 2025, Guo et al. 2025
### delta_vs_alpha

- **delta_vs_alpha_pair_00**: B
    > Plan B (delta) wins on technical specificity and reproducibility: it provides an RHS-complete weighted policy-gradient equation (nabla J = E_{tau~B}[w(r_tau) nabla log pi]) with a concrete weight function w(r) = sigmoid(alpha*r), names the base model (Llama-3-8B), specifies LoRA rank 64, gives an en
- **delta_vs_alpha_pair_01**: A
    > Plan A is substantially more rigorous and goal-aligned. It names specific methods with citations and roles (GRPO with exponential-utility objective per Jiang et al. 2025; SIFT example selection per Hubotter et al. 2024; program-space search per Georgiev et al. 2025; synthetic per-problem curriculum 
- **delta_vs_alpha_pair_02**: A
    > Plan A wins on methodological rigor and originality. A names specific prior work with concrete numbers (Hubert et al. 2025 AlphaProof TTRL, Georgiev et al. 2025 program-space on a 67-problem benchmark, Jiang et al. 2025 GRPO with exponential-utility objective, Hubotter et al. 2024 SIFT diversity, Zu
- **delta_vs_alpha_pair_03**: A
    > Plan A wins on more than half the rubric dimensions despite Plan B's stronger reproducibility numerics. On Soundness and Methodological rigor, A grounds each component in specific prior work (Akyurek 2024 for hypothesis-class extension, Jiang 2025 for risk-sensitive exponential-utility GRPO, Hubotte
- **delta_vs_alpha_pair_04**: A
    > Plan A wins on Soundness, Reproducibility, and Specificity, which the rubric explicitly weights via its 'RHS-complete equations beat method-naming' and 'operational compute beats vague compute' tiebreakers. A provides a full reward-weighted policy-gradient update with RHS-complete loss (nabla L = E_
- **delta_vs_alpha_pair_05**: A
    > Plan A dominates on specificity, soundness, and goal alignment. A grounds every claim in named prior work (Akyurek et al. 2024 for the frozen-ICL bound, Hubert/Zuo 2025 for prior TTT limitations, Hubotter et al. 2024 SIFT for diversity buffering, Jiang et al. 2025 for the risk-sensitive objective, G
- **delta_vs_alpha_pair_06**: A
    > Plan A (TTRL-Discover) wins on Soundness, Methodological rigor, and Feasibility/Significance. It cites real, dated prior work (Akyurek 2024, Hubert 2025, Surina 2025, Hubotter 2024, Jiang 2025, Zuo 2025, Georgiev 2025) with concrete prior numbers (+62% pass@1 for TTRL from Zuo 2025) and named benchm
- **delta_vs_alpha_pair_07**: A
    > Plan A specifies a complete reward-weighted update objective with full RHS (L(theta) = -E_tau[w(tau) log pi_theta(a|s,d)] with w(tau) = exp(beta R(tau))/sum exp(beta R(tau'))) and concrete reproducibility details (Llama-3-8B + LoRA, 100 steps per problem, 1000-evaluation budget, named baselines: fro
