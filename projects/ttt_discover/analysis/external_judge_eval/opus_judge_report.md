# External Judge Evaluation (Claude Opus 4.7) — D3 Paper Depth Audit

*Evaluator*: Claude Opus 4.7 (1M context) — CROSS-FAMILY judge relative to Qwen3-30B training grader
*Date*: 2026-04-17
*Scope*: 8 generated plans (5 pilot pre-stitching-fix + 3 paper-run post-fix) vs 1 human-expert reference
*Purpose*: Quantify the Qwen3-30B grader's ceiling on depth dimensions it cannot measure

---

## Reference plan baseline (Goel 2025 expert level)

The reference decomposes the problem into three concrete failure modes of standard test-time RL (mean-vs-max objective mismatch; horizon collapse; two-level exploration collapse) and proposes targeted fixes with explicit math.

Representative quotes:

> "J_β(θ) = E_{s∼reuse(H)}[log E_{a∼π_θ(·|d,s)}[exp(β · R(s,a))]] ... As β → ∞, the inner expectation concentrates all probability mass on the maximum-reward action" (entropic objective)

> "Q(s) + c · P(s) · √(1+T) / (1+n(s)) where Q(s) is the MAXIMUM reward among states generated from s (not the mean, which is the key deviation from AlphaZero-style PUCT)" (2-character algorithmic novelty)

> "gpt-oss-120b as the base policy via the Tinker API with LoRA rank 32 and Adam learning rate 4e-5 ... 50 training steps per problem. A full training run costs approximately $500 on Tinker." (feasible budget)

> "an improved upper bound on Erdős' minimum overlap below the prior AlphaEvolve result of 0.380924 ... AtCoder score exceeding 566,997 ... single-cell denoising correlation around 0.71, compared to MAGIC at 0.64." (named baselines with specific prior numbers)

Scores (calibration only):
- Math formalism: 5/5
- Algorithmic novelty: 5/5
- Implementation realism: 5/5
- Empirical rigor: 5/5

---

## Per-plan scores

### Plan: pilot pair_0_1_iter1 (score 0.675)
Training grader score: 0.675 on Qwen3-30B

#### Mathematical formalism: 1/5
> "Use stochastic gradient descent (SGD) or Adam with learning rate scheduling. Apply weight decay to prevent overfitting."

Reasoning: No formulas at all. No objective written; no β, no KL, no Q-value, no PUCT.

#### Algorithmic novelty: 2/5
> "Interleaved Search and Learning (ISL) – A framework where the LLM alternates between: Search Phase ... Learning Phase"

Reasoning: "Interleave search and learning" is the problem statement rebranded as contribution. Curriculum learning + diversity penalties are off-the-shelf. No MAX-PUCT or entropic-objective analogue.

#### Implementation realism: 3/5
> "Freeze all layers except a small subset (e.g., last few transformer blocks) to reduce compute overhead."

Reasoning: Partial-layer finetuning is plausible. No O(n²) error. But no specific base model, no $/hr budget, no Tinker-LoRA acknowledgment; "100–1,000 passes" is arbitrary.

#### Empirical rigor: 2/5
> "Randomized search with fixed prompts (e.g., GPT-3.5-turbo with no parameter updates) and evolutionary algorithms (e.g., genetic programming with 100 generations)."

Reasoning: Domains listed (protein folding, material design, chemistry) without specific benchmarks or prior AI numbers. Threshold "≥ 0.8 reward score in protein folding" is invented.

---

### Plan: pilot pair_2_3_iter3 (score 0.760)
Training grader score: 0.760 on Qwen3-30B

#### Mathematical formalism: 1/5
> "Use CMA-ES with a population size of 20 ... Weight decay (λ=1e-4) and gradient clipping (max norm=1.0)"

Reasoning: Zero formulas. Hyperparameter numbers are not math.

#### Algorithmic novelty: 1/5
> "This revision reduces technique count from 7+ to 2 core methods (CMA-ES and gradient ascent)"

Reasoning: Plan advertises *subtraction* as novelty. CMA-ES + gradient ascent is the most generic pairing possible.

#### Implementation realism: 1/5
> "For non-differentiable rewards: Use CMA-ES to perturb weights, evaluate rewards, and update the population."

Reasoning: FATAL ERROR. CMA-ES maintains and inverts n×n covariance Σ. For a 7B-param LLM, n ≈ 7×10^9 → Σ has ≈5×10^19 entries, infeasible by orders of magnitude. Plan says "perturb model weights directly" — not LoRA-restricted. CMA-ES is validated on ≤O(10^4)-dim problems, not 10^9-dim.

#### Empirical rigor: 2/5
> "Frozen LLMs (e.g., LLaMA-7B with static parameters). CMA-ES without LLM guidance (as a pure evolutionary approach)."

Reasoning: Named baselines but no prior numbers, no concrete benchmark, no failure criterion. Thresholds (0.8, 0.6) are arbitrary.

---

### Plan: pilot pair_4_5_iter4 (score 0.815) — *stitching artifact present*

Training grader score: 0.815 on Qwen3-30B

Note: Lines 30–66 contain a richer Implementation Steps section citing specific tools (AutoDock Vina v1.2.3, RDKit 2023.03.1, pycma v3.1.0, ZINC-15). Lines 51–66 then repeat the older generic version. Evaluation Metrics appears twice. **Stitching artifact — evaluated on richer block; duplicates not counted.**

#### Mathematical formalism: 1/5
> "CMA-ES (pycma v3.1.0) with population size=20, initial step size=0.1, and 50 iterations. Perturb model weights directly via parameter-level mutations (σ=0.05)."

Reasoning: The "specificity boost" (0.760 → 0.815) is entirely tool-name and hyperparameter injection. Still zero formulas.

#### Algorithmic novelty: 1/5
Reasoning: Identical skeleton to pair_2_3. Molecular-design instantiation is domain substitution, not algorithmic contribution.

#### Implementation realism: 1/5
> "Use CMA-ES (pycma v3.1.0) ... Perturb model weights directly via parameter-level mutations (σ=0.05)."

Reasoning: Same fatal CMA-ES-on-weights as pair_2_3. Also: "Limit total FLOPs to 1.2e10" — this is ≈10 ms of A100 compute, inconsistent with "20 pop × 50 iter × 100 evals" on a 7B model (≈10^18 FLOPs, **8 orders of magnitude** above stated budget). The PyPI version pin does not fix the O(n²)-memory infeasibility.

#### Empirical rigor: 2/5
> "Compare against a state-of-the-art scientific LLM (e.g., BioMol-1 or PharmaBERT) with static parameters"

Reasoning: BioMol-1 and PharmaBERT named without prior numbers. No published AutoDock score target. Wilcoxon n=10 applied to invented thresholds. Names ZINC-15/PubChem but no comparable prior AI number like AlphaEvolve 0.380924.

---

### Plan: pilot pair_6_7_iter6 (score 0.815)
Training grader score: 0.815 on Qwen3-30B

#### Mathematical formalism: 1/5
> "Use gradient ascent on a subset of parameters (e.g., last 20% of layers) to minimize $ \mathcal{L} = -R(s) $, where $ R(s) $ is the solution's reward."

Reasoning: The single formula is the "trivial tautology" flagged in the rubric. L = -R is definitional. No policy-gradient expansion.

#### Algorithmic novelty: 2/5
> "CMA-ES with a hybrid gradient approximation to balance exploration and exploitation."

Reasoning: "Hybrid gradient approximation" is undefined. Diversity cos<0.8 is standard. No MAX-Q insight.

#### Implementation realism: 2/5
> "Non-Differentiable Rewards: Deploy CMA-ES with a hybrid gradient approximation"

Reasoning: Same CMA-ES concern, partially mitigated by "20% of layers" (still millions of params — still infeasible for classical CMA-ES without sep-CMA or adapter restriction, neither specified). AdamW lr=1e-4 plausible.

#### Empirical rigor: 2/5
> "State-of-the-art methods for sparse reward settings (e.g., PPO with reward shaping, Trajectory Optimization with reward extrapolation)."

Reasoning: Names Materials Project and PubChem but no prior numbers. "≥10% improvement" is relative without absolute anchor.

---

### Plan: pilot pair_8_9_iter9 (score 0.782) — B4 in-context variant
Training grader score: 0.782 on Qwen3-30B

#### Mathematical formalism: 2/5
> "Optimize $ \mathcal{L} = -\sum_{s} \text{Reward}(s) \cdot \log P(s) + \lambda \cdot \text{Regularization}(w) $ ... Use REINFORCE with adaptive baselines or meta-learning (e.g., MAML)"

Reasoning: Closest to real objective among generated plans — essentially REINFORCE with generic regularizer. But `Regularization(w)` is placeholder, no temperature, no KL budget, no baseline-variance term. One correctly-stated standard formula = 2/5.

#### Algorithmic novelty: 2/5
> "dynamic reward-adaptive learning (DRAL) ... adaptive learning rate: Adjust the learning rate based on the variance of recent rewards to stabilize training."

Reasoning: Adaptive LR on reward variance is reasonable heuristic, not non-obvious. MAML is mis-applied — MAML needs a task distribution; this is single-task test-time.

#### Implementation realism: 3/5
> "Limit gradient updates to 5–10 steps per problem and batch sizes to 16–64 solutions. Use gradient checkpointing and parallelized evaluation ... LoRA or adapter layers to update only a subset"

Reasoning: Feasibility-consistent. No CMA-ES error. LoRA correctly invoked. Missing $/hr, base-model identity, Tinker acknowledgment, but nothing fatal. Best feasibility among pilot plans.

#### Empirical rigor: 2/5
> "Frozen LLMs: No parameter updates during search. Random Search: No learning, only sampling. Human-in-the-Loop: Manual refinement (for context)."

Reasoning: Default baseline triple, no named prior numbers. Domains generic ("Mathematical Proofs", "Physics Simulations") without specific open problems.

---

### Plan: MAIN_v6_minimal (score 0.768)
Training grader score: 0.768 on Qwen3-30B. Signal vector shows S2_rigor=2 — training grader detected the math weakness.

#### Mathematical formalism: 1/5
Reasoning: Zero formulas in 826 words. "Dynamic Weight Decay: Adjust based on reward variance", "REINFORCE or finite differences" — names without equations. No confidence-weighting formula for multi-fidelity. Training grader's own S2=2 confirms it.

#### Algorithmic novelty: 2/5
> "Multi-Fidelity Reward Handling: Low-Fidelity Proxies ... High-Fidelity Evaluations"

Reasoning: Multi-fidelity BO is well-established (Kandasamy 2017). "Contextual prompt engineering: dynamically refine prompts based on reward history" — refinement rule undefined. No MAX-Q-style insight.

#### Implementation realism: 3/5
> "Low-Rank Adapter Layers (LoRA): Update only a subset of weights ... limit parameter updates to 15 steps and evaluations to 3,000 candidates (10% high-fidelity)."

Reasoning: LoRA correct, explicit budget (15 updates, 3,000 candidates), no CMA-ES error. Missing $/hr, base-model identity beyond "LLaMA or Mistral", Tinker acknowledgment.

#### Empirical rigor: 2/5
> "Solution Quality: Compare the best solution's reward to baselines (frozen LLMs, random search, execution feedback)."

Reasoning: Three baseline categories, zero prior numbers, zero named open problems. "Discover novel theorems" without specific conjecture identified.

---

### Plan: B4_v6_minimal (score 0.785)
Training grader score: 0.785 on Qwen3-30B (highest among paper runs). Signal vector S2_rigor=2.

#### Mathematical formalism: 2/5
> "Loss Function: Optimize $ \mathcal{L} = -\sum_{s} \text{Reward}(s) \cdot \log P(s) + \lambda \cdot \text{Regularization}(w) $"

Reasoning: One standard REINFORCE-with-regularizer formula. `Regularization(w)` is placeholder. No β, no KL, no limit analysis. One valid non-trivial equation.

#### Algorithmic novelty: 2/5
> "dynamic reward shaping ... adaptive exploration-exploitation control ... multi-fidelity evaluation."

Reasoning: Laundry list, none operationalized. MAML mis-cited (single-task setting). No MAX-Q analogue.

#### Implementation realism: 3/5
> "Parameter-Efficient Updates: Use LoRA or adapter layers ... Limit gradient updates to 5–10 steps per problem and batch sizes to 16–64 solutions."

Reasoning: LoRA correct, feasible numbers, gradient checkpointing. No CMA-ES error. GPT-4 cited as base — problematic since closed-weight and goal requires open-weight.

#### Empirical rigor: 2/5
> "Frozen LLMs: No parameter updates during search. Random Search: No learning, only sampling. Human-in-the-Loop: Manual refinement (for context)."

Reasoning: Same triple as MAIN. No prior AI number, no specific open problem.

---

### Plan: B1_v6_minimal (zero-shot baseline, score 0.597)
Training grader score: 0.597 on Qwen3-30B. Signal vector S2_rigor=1 — training grader correctly detected no math.

#### Mathematical formalism: 1/5
Reasoning: Zero formulas. "Use the reward as the loss function" is prose.

#### Algorithmic novelty: 1/5
> "Use Bayesian optimization or evolutionary strategies to guide search toward high-reward regions."

Reasoning: Option-list prose ("or"). Enumerates standard techniques without committing. Zero non-obvious proposals.

#### Implementation realism: 3/5
> "Freeze most layers of the LLM, only fine-tuning a small 'adaptation head' (e.g., via LoRA). Use model distillation to compress the adapted model if needed."

Reasoning: LoRA + gradient checkpointing + partial-layer freezing — all feasible. No fatal errors. Only "5–10 steps" as quantification.

#### Empirical rigor: 2/5
> "Frozen LLM (no parameter updates). LLM with external feedback (e.g., human-in-the-loop). Random search or heuristic-based methods."

Reasoning: Generic triple, no prior numbers. Domains abstract — no Erdős, no AlphaEvolve, no MAGIC.

---

## Aggregate ranking (out of 20)

| Rank | Plan | Qwen3-30B score | Opus total/20 | Math | Novelty | Realism | Rigor |
|---|---|---|---|---|---|---|---|
| 1 | Reference (Goel-expert) | 0.748 (prior pilot) | 20/20 | 5 | 5 | 5 | 5 |
| 2 | B4_v6_minimal | 0.785 | 9/20 | 2 | 2 | 3 | 2 |
| 2 | pilot pair_8_9_iter9 | 0.782 | 9/20 | 2 | 2 | 3 | 2 |
| 4 | MAIN_v6_minimal | 0.768 | 8/20 | 1 | 2 | 3 | 2 |
| 4 | pilot pair_0_1_iter1 | 0.675 | 8/20 | 1 | 2 | 3 | 2 |
| 6 | pilot pair_6_7_iter6 | 0.815 | 7/20 | 1 | 2 | 2 | 2 |
| 6 | B1_v6_minimal (zero-shot) | 0.597 | 7/20 | 1 | 1 | 3 | 2 |
| 8 | pilot pair_4_5_iter4 | 0.815 | 5/20 | 1 | 1 | 1 | 2 |
| 8 | pilot pair_2_3_iter3 | 0.760 | 5/20 | 1 | 1 | 1 | 2 |

Notes on ranking:
- Reference included only as calibration, not in generated-plan ranking.
- Qwen3-30B ranks pilot pair_4_5 and pilot pair_6_7 (both 0.815) at the top of the 8 generated plans; Opus ranks them at 6 and 8.
- Opus ranks B4 and pilot pair_8_9 at the top because they alone write down one non-trivial equation.
- pilot pair_4_5 (training-grader co-top at 0.815) sits at Opus rank 8 — explicit inversion due to fatal CMA-ES error.

---

## Cross-grader inversion analysis

### 1. Plans beating the reference on Qwen3-30B but losing on Opus

The prior-pilot reference score on Qwen3-30B was 0.748. The following 4 generated plans **exceed it on the training grader**:

| Plan | Qwen3-30B | Ref-Qwen3-30B | Opus/20 | Ref-Opus/20 | Inversion? |
|---|---|---|---|---|---|
| pilot pair_4_5 | 0.815 | 0.748 | 5/20 | 20/20 | YES (extreme: Δ 15/20) |
| pilot pair_6_7 | 0.815 | 0.748 | 7/20 | 20/20 | YES (Δ 13/20) |
| B4_v6_minimal | 0.785 | 0.748 | 9/20 | 20/20 | YES (Δ 11/20) |
| pilot pair_8_9 | 0.782 | 0.748 | 9/20 | 20/20 | YES (Δ 11/20) |

**4 out of 8 generated plans are ranked above the reference by the training grader but far below it by the independent judge.** This is the canonical signal-ceiling evidence: Qwen3-30B assigns higher reward to plans that an external judge sees as demonstrably shallower than the reference, in every one of the four depth dimensions the training grader cannot directly measure.

pilot pair_4_5 is the most extreme inversion (0.815 Qwen3 vs 5/20 Opus) — it has a fatal CMA-ES-on-LLM-weights feasibility bug the training grader completely missed.

### 2. Rank correlation between Qwen3-30B and Opus

Ranks on 8 generated plans (1 = best; midrank on ties):

| Plan | Qwen3-rank | Opus-rank | d² |
|---|---|---|---|
| pilot pair_4_5 | 1.5 | 8.5 | 49 |
| pilot pair_6_7 | 1.5 | 6.5 | 25 |
| B4 | 3 | 1.5 | 2.25 |
| pilot pair_8_9 | 4 | 1.5 | 6.25 |
| MAIN | 5 | 4.5 | 0.25 |
| pilot pair_2_3 | 6 | 8.5 | 6.25 |
| pilot pair_0_1 | 7 | 4.5 | 6.25 |
| B1 (zero-shot) | 8 | 6.5 | 2.25 |

Σd² ≈ 97.5; Spearman ρ = 1 − 6·97.5/(8·63) ≈ −0.16. In practical terms: **the training grader and Opus disagree on the ordering of generated plans (slight negative correlation).** The training-grader top-2 plans (both pilot, 0.815) are tied for bottom by Opus; the training-grader bottom-2 (pair_0_1 at 0.675 and B1 zero-shot at 0.597) sit in the upper half of the Opus ranking because they simply do less damage (no CMA-ES fantasy).

### 3. Which dimension is Qwen3-30B most blind to?

Averaged across 8 generated plans:

| Dimension | Opus mean/5 | Gap to reference (5) | Gap rank (1 = worst) |
|---|---|---|---|
| **Mathematical formalism** | **1.25** | **−3.75** | **1 (most blind)** |
| Algorithmic novelty | 1.625 | −3.375 | 2 |
| Empirical rigor | 2.000 | −3.000 | 3 |
| Implementation realism | 2.375 | −2.625 | 4 |

**Mathematical formalism is the dimension the training grader is most blind to.** 7 of 8 plans score 1/5 — no formulas or only tautological `L = -R`. The 2 plans that do score 2/5 (B4 and pilot pair_8_9) only manage one standard REINFORCE equation with a placeholder regularizer; neither approaches the reference's J_β + ∇J_β + adaptive KL + MAX-PUCT formula quartet.

Interestingly, **the Qwen3-30B grader's own S2_rigor signal partially detects this** — it assigns S2=1 or S2=2 to every paper-run plan (1 for B1, 2 for MAIN and B4). So the issue is not that Qwen3-30B cannot see the gap — it is that S2 has weight 1/8 of the aggregate, so eight plans with {S2=2, S7=5, S8=5, others=4+} still aggregate to 0.75+. The blindness is at the aggregation level, not the detection level. See "Summary" below.

Algorithmic novelty (gap 3.375) is the second-largest blind spot: the training grader happily rewards category labels ("Hybrid Test-Time Adaptation", "Dynamic Reward-Adaptive Learning") without evaluating whether a non-obvious modification exists.

---

## Key technical errors spotted

The training grader missed the following fatal or near-fatal feasibility issues:

**1. CMA-ES on full LLM weights (pilot pair_2_3, score 0.760)**
> "For non-differentiable rewards: Use CMA-ES to perturb weights, evaluate rewards, and update the population."
CMA-ES stores a full n×n covariance matrix and performs O(n³) decomposition per iteration. On a 7B-parameter LLM, Σ alone is ≈5×10^19 entries. This is physically infeasible on any hardware. The plan does not invoke sep-CMA-ES or any restricted-parameter variant; it says "weights" without qualification.

**2. CMA-ES on full weights + version numbers (pilot pair_4_5, score 0.815)**
> "CMA-ES (pycma v3.1.0) with population size=20, initial step size=0.1, and 50 iterations. Perturb model weights directly via parameter-level mutations (σ=0.05)."
Same error as (1), with false precision from a PyPI version pin. Training grader rewarded the version specificity with +0.055 vs pair_2_3; Opus flagged it as a fatal error either way.

**3. FLOP-budget inconsistency (pilot pair_4_5, score 0.815)**
> "Limit total FLOPs to 1.2e10 using PyTorch 2.0 with mixed-precision training (FP16)."
1.2×10^10 FLOPs ≈ 10 ms of A100 wall-time at half-precision. The same plan describes 20 × 50 × 100 = 100,000 forward passes through a LLaMA-7B (≈10^13 FLOPs per forward pass). Implied compute is roughly 10^18 FLOPs — **8 orders of magnitude** above the stated budget. The training grader rewarded the number anyway.

**4. CMA-ES hybrid gradient approximation (pilot pair_6_7, score 0.815)**
> "Deploy CMA-ES with a hybrid gradient approximation to balance exploration and exploitation."
The object "CMA-ES with hybrid gradient approximation" is not defined, and no paper is cited. If intended to mean CMA-ES on full LLM weights, same fatal error as (1). If intended to mean adapter-only CMA-ES, the plan does not say so.

**5. MAML mis-citation for single-task test-time (pilot pair_8_9 and B4)**
> "Use REINFORCE with adaptive baselines or meta-learning (e.g., MAML) for low-variance updates."
MAML requires a distribution of tasks during meta-training. The research goal explicitly describes single-problem test-time adaptation. Citing MAML here is a non-sequitur. The training grader rewarded the name without checking applicability.

**6. GPT-4 named as open-weight base (B4)**
> "Start with a pretrained LLM (e.g., LLaMA, GPT-4) and a problem-specific reward function."
GPT-4 is closed-weight; the goal specifies open-weight models. The training grader did not flag this.

All six errors slipped past the Qwen3-30B grader. Only (6) is detectable by surface keyword match; the other five require mathematical or engineering-realism reasoning the grader does not perform.

---

## Summary (≤ 5 bullets)

- **4 of 8 generated plans score higher than the human-expert reference on the Qwen3-30B training grader (0.782–0.815 vs ref 0.748) while scoring 5–9 out of 20 on Opus's 4-dimension audit (ref = 20/20).** Canonical signal-ceiling inversion confirmed on four independently-generated plans, not a single outlier.

- **Mathematical formalism is the training grader's largest blind spot (Opus mean 1.25/5 across 8 plans vs reference 5/5, gap 3.75).** The grader's S2_rigor signal correctly assigns 1–2 to most generated plans, but its 1/8 aggregation weight cannot overcome inflated S7_specificity (5/5 from version-pinned tool names) and S8_scope scores. The ceiling is in the **aggregation rule, not the detection of depth**.

- **The training grader rewards surface specificity regardless of correctness.** "CMA-ES on LLM weights" (physically infeasible) plus a PyPI version pin earns 0.815 — the highest training score in the pilot. A FLOP budget 8 orders of magnitude off goes unflagged. Version numbers, dataset names, and $-amounts raise the reward whether or not they are mathematically consistent with the rest of the plan.

- **Rank correlation between training-grader and Opus is slightly negative (Spearman ρ ≈ −0.16 on the 8 generated plans).** The two 0.815-tied top plans on training grader sit at Opus ranks 6.5 and 8.5 (last place); the two lowest training scores (pair_0_1 at 0.675 and B1 at 0.597) sit at Opus ranks 4.5 and 6.5 because they avoid the CMA-ES fantasy errors their "better" siblings commit.

- **The signal-ceiling is not fixable by more training; it requires a different reward structure.** No amount of Qwen3-30B-guided RL or critique-revise iteration can push a generated plan past the reference on dimensions (math formalism, algorithmic novelty) the grader's rubric does not encode. Candidates: (a) formula-detection signal with LaTeX parsing + dimensional-consistency check; (b) explicit novelty signal against a retrieved nearest-prior-work embedding; (c) grader-panel ensemble including a stronger cross-family model. Without one of these, further training at v8.1 will keep improving Qwen3-30B scores while plan depth plateaus.
