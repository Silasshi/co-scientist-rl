"""Generate perturbation variants of the reference plan.

Unified naming (after signal set revision):
  - P_HG1_goal_contrast   → damages Hard Gate 1 (Goal-Contrast Margin)
  - P_HG2_claim_verif     → damages Hard Gate 2 (Claim Verification)
  - P_S1_mechanism        → damages Signal S1 (Mechanism Plausibility)
  - P_S2_rigor            → damages Signal S2 (Evidence Rigor)
  - P_S3_positioning      → damages Signal S3 (Positioning)
  - P_S4_significance     → damages Signal S4 (Significance)
  - P_S5_stability        → damages Signal S5 (Stability & Reproducibility)
  - P_S6_failure_interp   → damages Signal S6 (Failure Interpretability)
  - P_S7_specificity      → damages Signal S7 (Implementation Specificity)
  - P_S8_scope            → damages Signal S8 (Scope-Generalization)

Mixed perturbations:
  - M01_first5_bad        → damages HG1 + HG2 + S1 + S2 + S3
  - M02_last5_bad         → damages S4 + S5 + S6 + S7 + S8
  - M03_alternating       → damages HG1 + S1 + S3 + S5 + S7 (odd-indexed)

Design principle: each perturbation applies a targeted, localized damage to
exactly ONE signal while leaving the others as intact as possible. This lets
the sanity check measure whether the grader correctly localizes the damage.
"""

from pathlib import Path

HERE = Path(__file__).parent
PERT_DIR = HERE / "perturbations"
REF_PATH = PERT_DIR / "00_reference.txt"
REFERENCE = REF_PATH.read_text()


def apply_replacements(text: str, replacements: list[tuple[str, str]]) -> str:
    out = text
    for old, new in replacements:
        if old not in out:
            raise ValueError(f"Substring not found: {old[:80]!r}")
        out = out.replace(old, new)
    return out


def apply_replacements_lenient(text: str, replacements: list[tuple[str, str]]) -> tuple[str, int, int]:
    """Like apply_replacements but skips failed replacements (for mixed perturbations
    where earlier replacements may have modified text expected by later ones)."""
    out = text
    applied = 0
    skipped = 0
    seen = set()
    for old, new in replacements:
        if old in seen:
            skipped += 1
            continue
        seen.add(old)
        if old not in out:
            skipped += 1
            continue
        out = out.replace(old, new)
        applied += 1
    return out, applied, skipped


# ============================================================
# P_HG1: Damage Goal-Contrast Margin (hard gate 1)
# ============================================================
# Replace goal-specific technical commitments with generic text that could
# apply to any RL/optimization research.

P_HG1_REPLACEMENTS = [
    (
        "*Core training objective.* We define the entropic objective J_β(θ) = E_{s∼reuse(H)}[log E_{a∼π_θ(·|d,s)}[exp(β · R(s,a))]], where d is the problem description, s is the initial state sampled from buffer H, and β is a temperature constant. As β → ∞, the inner expectation concentrates on the maximum-reward action. The policy gradient takes a reward-weighted log-likelihood form ∇J_β(θ) = E[w_β(a) · ∇log π_θ(a|s,d)] with weights w_β(a) = exp(β·R)/Z. We set β adaptively per state via a KL budget γ = ln 2 to prevent instability.",
        "*Core training objective.* We use a reinforcement learning objective that rewards high-quality solutions and discourages low-quality ones. The training signal is derived from the environment's reward function and shaped to improve sample efficiency. We use standard policy gradient methods with appropriate regularization to prevent training instability.",
    ),
    (
        "*State reuse via PUCT.* We maintain a buffer H of all (state, action, next-state, reward) tuples. To select a warm-start state, we use the score Q(s) + c · P(s) · √(1+T) / (1+n(s)), where Q(s) is the MAXIMUM reward among s's descendants (not mean), P(s) is a rank-based prior, n(s) counts expansions, T is total expansions, and c is the exploration coefficient. We block the full lineage of any selected state from concurrent selection to preserve diversity.",
        "*Search strategy.* We use an efficient exploration mechanism that balances exploring new directions with exploiting promising ones. Our approach leverages insights from past work on search and optimization to find good solutions efficiently.",
    ),
]


# ============================================================
# P_HG2: Damage Claim Verification (hard gate 2)
# ============================================================
# Insert fabricated citations, numbers, method names.

P_HG2_REPLACEMENTS = [
    (
        "**Background**\nPrior approaches divide into two families. Evolutionary search methods (AlphaEvolve, OpenEvolve, ThetaEvolve) accumulate candidate solutions in a buffer and use domain-specific heuristics to construct new prompts for a frozen LLM. These can achieve state-of-the-art results on mathematics and kernel engineering, but they never update the LLM itself — the model cannot internalize insights discovered during the search. The second family applies standard RL algorithms (PPO, GRPO) at test time on the single target problem. These do update the model but inherit the wrong objective (mean rather than max) and still suffer from the short-horizon problem because each rollout starts fresh. Recent work (ThetaEvolve) began combining these ideas, but without addressing the objective mismatch.",
        "**Background**\nPrior approaches divide into two families. Evolutionary search methods (AlphaEvolve [Smith et al., NeurIPS 2024, 0.3814 Erdős bound], OpenEvolve [Chang & Liu, ICML 2024], ThetaEvolve [Zhang et al., ICLR 2025, 0.72 single-cell correlation]) accumulate candidate solutions in a buffer and use domain-specific heuristics. The GenericRL baseline [Park et al., 2024] achieves 94.2% sample efficiency on standard benchmarks [DiscoveryBench v2, 847 problems]. Recent work on EntropicDiscover [Chen et al., NeurIPS 2024, cited 1204 times] showed 38.3% improvement over PPO baselines using the OrthoReward decomposition method [Watanabe et al., 2024].",
    ),
    (
        "**Expected Results**\n(a) Erdős' bound below 0.380924 (AlphaEvolve); (b) TriMul runtime up to 2× faster than best human; (c) AtCoder score exceeding 566,997 (best human); (d) single-cell denoising correlation around 0.71 vs MAGIC at 0.64.",
        "**Expected Results**\nBased on the ScaleDiscover scaling laws [Kumar et al., 2024, arXiv:2407.18293], we project: (a) Erdős' bound of 0.380612 ± 0.000023, surpassing AlphaEvolve's 0.380924 by 8.7σ; (b) TriMul runtime of 1047μs on H100 (2.8× speedup vs best human), validated on the KernelBench-Pro dataset [Ahmed et al., 2024, 5,247 kernels]; (c) AtCoder score of 634,291 on AHC039, ranking 1st out of 2,847 participants; (d) single-cell denoising correlation of 0.834 on the DenoiseBench-XL dataset, compared to MAGIC [van Dijk et al., 2018] at 0.64 and DeepImpute [Arisdakessian et al., 2019] at 0.71.",
    ),
]


# ============================================================
# P_S1: Damage Mechanism Plausibility  [NEW]
# ============================================================
# Remove the causal reasoning and theoretical justifications.
# Replace with pure "we will try X" assertions.
# The plan keeps its specificity, structure, significance — but loses the
# "why should this work" arguments.

P_S1_REPLACEMENTS = [
    (
        "**Hypothesis**\nTwo targeted modifications to standard test-time RL, combined, enable new state-of-the-art discovery: (1) replacing the mean-reward RL objective with an entropic objective that exponentially weights high-reward samples, directly aligning the training signal with the discovery objective; and (2) replacing naive initial-state sampling with a PUCT-inspired tree search over the accumulated buffer, where the Q-value is the maximum (not mean) reward among a state's descendants. The first modification fixes the objective mismatch and discourages exploration collapse at the policy level. The second fixes the short-horizon problem by enabling warm starts from promising past states, and prevents exploration collapse at the trajectory level via a rank-based exploration bonus.",
        "**Hypothesis**\nWe propose to apply two modifications to standard test-time RL: (1) use an entropic objective instead of the mean-reward RL objective; and (2) use a PUCT-inspired tree search over an accumulated buffer to select initial states. We will try these two modifications together and expect the results to improve over baselines.",
    ),
    (
        "*Core training objective.* We define the entropic objective J_β(θ) = E_{s∼reuse(H)}[log E_{a∼π_θ(·|d,s)}[exp(β · R(s,a))]], where d is the problem description, s is the initial state sampled from buffer H, and β is a temperature constant. As β → ∞, the inner expectation concentrates on the maximum-reward action. The policy gradient takes a reward-weighted log-likelihood form ∇J_β(θ) = E[w_β(a) · ∇log π_θ(a|s,d)] with weights w_β(a) = exp(β·R)/Z. We set β adaptively per state via a KL budget γ = ln 2 to prevent instability.",
        "*Core training objective.* We define the entropic objective J_β(θ) = E_{s∼reuse(H)}[log E_{a∼π_θ(·|d,s)}[exp(β · R(s,a))]], with a temperature constant β. The policy gradient is ∇J_β(θ) = E[w_β(a) · ∇log π_θ(a|s,d)] with weights w_β(a) = exp(β·R)/Z. We set β to a fixed value of 1.0.",
    ),
    (
        "*State reuse via PUCT.* We maintain a buffer H of all (state, action, next-state, reward) tuples. To select a warm-start state, we use the score Q(s) + c · P(s) · √(1+T) / (1+n(s)), where Q(s) is the MAXIMUM reward among s's descendants (not mean), P(s) is a rank-based prior, n(s) counts expansions, T is total expansions, and c is the exploration coefficient. We block the full lineage of any selected state from concurrent selection to preserve diversity.",
        "*State reuse via PUCT.* We maintain a buffer H of all (state, action, next-state, reward) tuples. To select a warm-start state, we use the score Q(s) + c · P(s) · √(1+T) / (1+n(s)), where Q(s), P(s), n(s), T, and c are computed from buffer statistics.",
    ),
]


# ============================================================
# P_S2: Damage Evidence Rigor
# ============================================================
# Remove baselines and ablations. Make the success criterion vague.

P_S2_REPLACEMENTS = [
    (
        "**Evaluation**\nWe evaluate on four domains: (1) mathematical open problems (Erdős' minimum overlap and autocorrelation inequalities); (2) GPU kernel engineering (TriMul on A100 and H100); (3) algorithm design (AtCoder AHC039); (4) single-cell denoising in biology. For each, we report against the best-known human result, the best prior AI result (typically AlphaEvolve), and a Best-of-25600 baseline matched on sample budget.",
        "**Evaluation**\nWe evaluate on four domains: (1) mathematical open problems; (2) GPU kernel engineering; (3) algorithm design; (4) single-cell denoising. We will run our method on each problem and report the best reward achieved. The quality of the results will demonstrate the effectiveness of our approach.",
    ),
    (
        "*Training loop.* At each step: (1) sample 512 rollouts as 8 groups of 64 sharing an initial state selected via PUCT; (2) generate actions containing thinking tokens and executable code; (3) parse and execute to produce reward; (4) add to buffer; (5) apply one gradient step on the entropic objective with importance sampling correction.",
        "*Training loop.* At each step, the model generates candidate solutions, which are evaluated against the environment's reward function. The policy is updated to produce better solutions over time. No ablation studies are needed since the method is evaluated end-to-end.",
    ),
]


# ============================================================
# P_S3: Damage Positioning
# ============================================================
# Remove related work, replace with "no one has done this" claims.

P_S3_REPLACEMENTS = [
    (
        "**Background**\nPrior approaches divide into two families. Evolutionary search methods (AlphaEvolve, OpenEvolve, ThetaEvolve) accumulate candidate solutions in a buffer and use domain-specific heuristics to construct new prompts for a frozen LLM. These can achieve state-of-the-art results on mathematics and kernel engineering, but they never update the LLM itself — the model cannot internalize insights discovered during the search. The second family applies standard RL algorithms (PPO, GRPO) at test time on the single target problem. These do update the model but inherit the wrong objective (mean rather than max) and still suffer from the short-horizon problem because each rollout starts fresh. Recent work (ThetaEvolve) began combining these ideas, but without addressing the objective mismatch.",
        "**Background**\nThe problem of using AI to discover state-of-the-art solutions to scientific problems is a new and underexplored area. Nobody has systematically tackled this challenge before. Existing approaches to reinforcement learning are focused on different goals and do not apply here. Our work is the first to address the fundamental challenges of discovery at test time.",
    ),
]


# ============================================================
# P_S4: Damage Significance
# ============================================================
# Remove "why this matters" content. Vague expected results.

P_S4_REPLACEMENTS = [
    (
        "**Expected Results**\n(a) Erdős' bound below 0.380924 (AlphaEvolve); (b) TriMul runtime up to 2× faster than best human; (c) AtCoder score exceeding 566,997 (best human); (d) single-cell denoising correlation around 0.71 vs MAGIC at 0.64.",
        "**Expected Results**\nWe expect the method to produce reasonable results across the four evaluation domains. The outputs should be comparable to other approaches and the training process should be stable. We do not anticipate dramatic breakthroughs but expect the method to perform adequately.",
    ),
    (
        "**Problem Statement**\nStandard reinforcement learning algorithms applied at test time to a single discovery problem fail for three concrete reasons. First, their objective function maximizes average reward, while discovery cares only about the maximum reward achieved. Second, starting each attempt from scratch limits the effective horizon, so the policy cannot reach complex multi-step solutions. Third, naive exploration heuristics either collapse to safe actions (at the policy level) or over-exploit a few promising states (at the trajectory level). A discovery method must fix all three problems simultaneously.",
        "**Problem Statement**\nWe investigate the application of reinforcement learning to test-time discovery problems. This is one of several possible approaches to apply learning to hard tasks. Various technical challenges arise in implementation. Whether the problem is important is debatable; we pursue it as an interesting technical exercise.",
    ),
]


# ============================================================
# P_S5: Damage Stability & Reproducibility
# ============================================================
# Remove mentions of variance, seeds. Claim deterministic results.

P_S5_REPLACEMENTS = [
    (
        "**Limitations**\nOur method overfits to a single problem by design and produces no generalizable policy. It requires continuous verifiable rewards, which may not exist for open-ended discovery settings. Adaptive-β computation adds wall-clock overhead. Poor tuning of the PUCT exploration coefficient may cause over-exploitation or failure to find promising states. Single-seed results; variance across seeds is not characterized in this iteration but will be addressed in follow-up work.",
        "**Limitations**\nOur method overfits to a single problem by design and produces no generalizable policy. It requires continuous verifiable rewards, which may not exist for open-ended discovery settings.",
    ),
    (
        "**Expected Results**\n(a) Erdős' bound below 0.380924 (AlphaEvolve); (b) TriMul runtime up to 2× faster than best human; (c) AtCoder score exceeding 566,997 (best human); (d) single-cell denoising correlation around 0.71 vs MAGIC at 0.64.",
        "**Expected Results**\n(a) Erdős' bound of exactly 0.380876; (b) TriMul runtime of exactly 1161μs on H100; (c) AtCoder score of exactly 567,062; (d) single-cell denoising correlation of exactly 0.71. These results are deterministic outcomes of applying our method.",
    ),
]


# ============================================================
# P_S6: Damage Failure Interpretability
# ============================================================
# Remove Limitations section, add overconfident claims.

P_S6_REPLACEMENTS = [
    (
        "**Limitations**\nOur method overfits to a single problem by design and produces no generalizable policy. It requires continuous verifiable rewards, which may not exist for open-ended discovery settings. Adaptive-β computation adds wall-clock overhead. Poor tuning of the PUCT exploration coefficient may cause over-exploitation or failure to find promising states. Single-seed results; variance across seeds is not characterized in this iteration but will be addressed in follow-up work.",
        "**Conclusion**\nOur method is robust and will reliably produce improvements on all problems we apply it to. If results fall short of expectations, this will be due to insufficient compute or problems that are simply too hard. The method itself is sound and does not require fallback paths or diagnostic procedures.",
    ),
]


# ============================================================
# P_S7: Damage Implementation Specificity
# ============================================================
# Replace concrete details (numbers, specific methods) with vague alternatives.

P_S7_REPLACEMENTS = [
    (
        "*Implementation.* We use gpt-oss-120b via Tinker API with LoRA rank 32, Adam lr 4e-5, context window 32,768 tokens, and run for 50 training steps per problem.",
        "*Implementation.* We use a large open-source language model with parameter-efficient fine-tuning. Standard optimization settings and a sufficient number of training iterations are used. The context window is set large enough to accommodate the inputs.",
    ),
    (
        "*Training loop.* At each step: (1) sample 512 rollouts as 8 groups of 64 sharing an initial state selected via PUCT; (2) generate actions containing thinking tokens and executable code; (3) parse and execute to produce reward; (4) add to buffer; (5) apply one gradient step on the entropic objective with importance sampling correction.",
        "*Training loop.* At each step, we generate rollouts, evaluate them in the environment, add them to the buffer, and update the model weights. The training loop continues until convergence.",
    ),
]


# ============================================================
# P_S8: Damage Scope-Generalization Alignment
# ============================================================
# Overclaim scope (universal applicability) while keeping narrow evaluation.

P_S8_REPLACEMENTS = [
    (
        "**Problem Statement**\nStandard reinforcement learning algorithms applied at test time to a single discovery problem fail for three concrete reasons. First, their objective function maximizes average reward, while discovery cares only about the maximum reward achieved. Second, starting each attempt from scratch limits the effective horizon, so the policy cannot reach complex multi-step solutions. Third, naive exploration heuristics either collapse to safe actions (at the policy level) or over-exploit a few promising states (at the trajectory level). A discovery method must fix all three problems simultaneously.",
        "**Problem Statement**\nOur method solves the general problem of scientific discovery across ALL domains of science, engineering, and mathematics. We present a universal framework that works for any discovery task, including theoretical physics, drug discovery, materials science, astronomy, linguistics, social sciences, and economics. The three shortcomings of prior RL methods we identify are universal failure modes that our approach fixes in complete generality.",
    ),
    (
        "**Limitations**\nOur method overfits to a single problem by design and produces no generalizable policy. It requires continuous verifiable rewards, which may not exist for open-ended discovery settings. Adaptive-β computation adds wall-clock overhead. Poor tuning of the PUCT exploration coefficient may cause over-exploitation or failure to find promising states. Single-seed results; variance across seeds is not characterized in this iteration but will be addressed in follow-up work.",
        "**Limitations**\nBecause our method is universally applicable to all discovery problems across all scientific domains, the limitations are minimal. Our results on four specific domains fully establish the method's generality and can be safely extrapolated to any other scientific discovery task.",
    ),
]


# ============================================================
# Mixed perturbations
# ============================================================

# M01: Hard gates + S1 + S2 + S3 bad (first 5)
M01_REPLACEMENTS = (
    P_HG1_REPLACEMENTS + P_HG2_REPLACEMENTS + P_S1_REPLACEMENTS + P_S2_REPLACEMENTS + P_S3_REPLACEMENTS
)

# M02: S4 + S5 + S6 + S7 + S8 bad (last 5)
M02_REPLACEMENTS = (
    P_S4_REPLACEMENTS + P_S5_REPLACEMENTS + P_S6_REPLACEMENTS + P_S7_REPLACEMENTS + P_S8_REPLACEMENTS
)

# M03: HG1 + S1 + S3 + S5 + S7 bad (odd-indexed)
M03_REPLACEMENTS = (
    P_HG1_REPLACEMENTS + P_S1_REPLACEMENTS + P_S3_REPLACEMENTS + P_S5_REPLACEMENTS + P_S7_REPLACEMENTS
)


# ============================================================
# Generate all files
# ============================================================

SINGLE_PERTURBATIONS = [
    ("P_HG1_goal_contrast.txt", P_HG1_REPLACEMENTS),
    ("P_HG2_claim_verif.txt", P_HG2_REPLACEMENTS),
    ("P_S1_mechanism.txt", P_S1_REPLACEMENTS),
    ("P_S2_rigor.txt", P_S2_REPLACEMENTS),
    ("P_S3_positioning.txt", P_S3_REPLACEMENTS),
    ("P_S4_significance.txt", P_S4_REPLACEMENTS),
    ("P_S5_stability.txt", P_S5_REPLACEMENTS),
    ("P_S6_failure_interp.txt", P_S6_REPLACEMENTS),
    ("P_S7_specificity.txt", P_S7_REPLACEMENTS),
    ("P_S8_scope.txt", P_S8_REPLACEMENTS),
]

MIXED_PERTURBATIONS = [
    ("M01_first5_bad.txt", M01_REPLACEMENTS),
    ("M02_last5_bad.txt", M02_REPLACEMENTS),
    ("M03_alternating.txt", M03_REPLACEMENTS),
]


def main():
    print(f"Reference plan: {len(REFERENCE)} chars, {len(REFERENCE.split())} words")
    print()
    print("=== Single perturbations (strict) ===")

    for filename, replacements in SINGLE_PERTURBATIONS:
        try:
            perturbed = apply_replacements(REFERENCE, replacements)
        except ValueError as e:
            print(f"FAIL {filename}: {e}")
            continue
        out_path = PERT_DIR / filename
        out_path.write_text(perturbed)
        delta_chars = len(perturbed) - len(REFERENCE)
        delta_words = len(perturbed.split()) - len(REFERENCE.split())
        print(f"OK  {filename}: {len(perturbed)} chars ({delta_chars:+d}), words ({delta_words:+d})")

    print()
    print("=== Mixed perturbations (lenient — skip already-modified sections) ===")
    for filename, replacements in MIXED_PERTURBATIONS:
        perturbed, applied, skipped = apply_replacements_lenient(REFERENCE, replacements)
        out_path = PERT_DIR / filename
        out_path.write_text(perturbed)
        delta_chars = len(perturbed) - len(REFERENCE)
        delta_words = len(perturbed.split()) - len(REFERENCE.split())
        print(
            f"OK  {filename}: applied={applied}, skipped={skipped}, "
            f"{len(perturbed)} chars ({delta_chars:+d}), words ({delta_words:+d})"
        )


if __name__ == "__main__":
    main()
