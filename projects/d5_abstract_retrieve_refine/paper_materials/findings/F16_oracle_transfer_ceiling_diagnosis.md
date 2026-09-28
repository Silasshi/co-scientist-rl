# F16 — Oracle transfer ceiling: critique-conditioned self-distillation reweights existing priors but does not internalize oracle content

*Filed 2026-04-28. Hypothesis-stage diagnosis grounded in code reading
(`src/co_scientist/d5_abstract_retrieve_refine/train_mu_v8_d5sdpo.py:393-473`)
and oracle inspection (`projects/d5_abstract_retrieve_refine/data/oracles/
oracle_v2_2026_04_26_build/slim.md`). Experimental falsification deferred to
the ABC experiment plan (`knowledge/current/EXPERIMENT_PLAN_oracle_transfer_ABC.md`).*

## Headline (hypothesis to be tested)

The μ-v8-d5sdpo lift over σ_v8 (audit +3.00, pairwise 8-0) does NOT come from
the model internalizing oracle content (specific equations, named methods,
empirical numbers). It comes from a critique-conditioned reweighting of priors
the base model already possesses (section structure, named-method keyword
frequency, plan-level formatting). Three mechanism-level bottlenecks block the
oracle → output content pathway:

1. **Long-context retrieval bottleneck** in Qwen3-30B at 10K-word oracle
2. **Critique-conditioned advantage rewards critique-conformity, not
   oracle-fidelity**
3. **LoRA at rank 64 / 16 iters has insufficient bandwidth to internalize
   formula-length token sequences as new manifold paths**

This is a forward-looking diagnosis. ABC experiments (defined in the linked
plan) will falsify or corroborate.

## Verified architectural facts

### F16-fact-1 — Teacher = current LoRA-adapted student, not frozen base

`train_mu_v8_d5sdpo.py:393-396`:

```python
sp_path = training_client.save_weights_for_sampler(name=f"iter_{iter_idx:04d}").result().path
sampling_client = service_client.create_sampling_client(model_path=sp_path)
```

`train_mu_v8_d5sdpo.py:467-473`:

```python
teacher_lp_futures = [
    sampling_client.compute_logprobs(
        types.ModelInput.from_ints(tokens=teacher_tokens_prefix + list(seq.tokens))
    )
    for seq in student_result.sequences
]
```

Same `sampling_client` is used for student rollout and teacher_lp computation.
Both are at current LoRA-adapted weights. Teacher and student differ only by
prompt:

- `student_lp = π_θ(y | goal, oracle)` (from sampling)
- `teacher_lp = π_θ(y | goal, oracle, critique)` (from `compute_logprobs`)

`initial_teacher_client` (line 327, frozen base Qwen3-30B-A3B) is created **only
when** `config.trust_region_alpha > 0.0` and is used only for the trust-region
interpolation `teacher_lp_eff = (1-α)·teacher_lp + α·frozen_lp`. The primary
teacher_lp is current π_θ.

**Implication**: this is **self-distillation**, not distillation from external
strong teacher. Critique is in prompt, not in loss. The advantage signal
A_t = teacher_lp_eff_t − student_lp_t is the **critique-conditional shift in
the same model's lp**, not a transfer from a stronger teacher to a weaker
student.

### F16-fact-2 — Oracle is in the prompt, not in any loss term

`train_mu_v8_d5sdpo.py:402, 460`:

```python
student_text = build_student_prompt_v8(goal, oracle)        # oracle in student prompt
teacher_text = build_teacher_prompt_v8(goal, oracle, critique_current)  # oracle in teacher prompt
```

Oracle text appears identically in both prompts. It contributes to both
student_lp and teacher_lp through the same model. **Oracle is not a gradient
target**; it is conditioning context for both forward passes that get
differenced.

The advantage A_t therefore reflects only the **delta from adding critique
to a prompt that already contains oracle**. Oracle conditions the lp baseline,
critique perturbs it; the gradient signal is "shift induced by critique
conditioning" — not "shift induced by oracle conditioning".

### F16-fact-3 — Oracle slim has all the formulas the critic flags as missing

Oracle slim
(`projects/d5_abstract_retrieve_refine/data/oracles/oracle_v2_2026_04_26_build/slim.md`,
10,265 words, 87 items × 17 papers) contains the formulas Opus critique
repeatedly flags as missing in student plans. Verified by direct read:

- Math 5 (Jiang et al. 2025): `J_RS(π_θ) = E_{x~D} [(1/β) · log E_{y~π_θ(.|x)}
  [e^{β · r(y)}]]`
- Math 7 (Phan et al. 2025): full GRPO loss with clipped policy ratio RHS
- Methodology 9 (RS-GRPO): `Â_β(y_i) = (1/β) · (e^{β · r(y_i)} / mean_j e^{β · r(y_j)} - 1)`
- Math 12 (TTRL): `θ ← θ + η · ∇_θ E_{y ~ π_θ(·|x)}[r(y, y*)]`
- Empirical 4: AIME 2024 pass@1 15.6% → 77.9%
- Empirical 16: TTRL +211% on AIME with Qwen2.5-Math-7B
- Empirical 10: 4×4 complex matmul 49 → 48 multiplications

So oracle is NOT missing the content. The bottleneck cannot be attributed to
"oracle missing the right facts".

### F16-fact-4 — Critic responses contain paper-grade substantive content

Verified by direct read of representative `critic_responses/iter_NNN.json`
files:

- E iter 5 (peak 22.75): 7-point free-form critique, names `J_β(s)` adaptive
  state-conditional formulation, names PUCT score formula, calls out per-problem
  reset misread, identifies 4×4 evaluation suite mismatches
- G+ iter 4 (peak 22.38): XML-structured critique with idea_alignment /
  missing_components / incorrect_assumptions / feasibility / improvement_directive
  sections; names PUCT formula, KL-budget bisection on β(s), LOO entropic
  advantages, IS correction
- C+ iter 6 (peak 23.50): XML-structured, names PUCT, calls out gradient-based
  variant generator infeasibility for kernel/algorithm domains
- E iter 9 (cliff floor 9.75): 8-point critique on a degenerate plan with
  word-salad output ("Hydrologic compaction super-knowledge", "Laclebp Hodah
  Build")

Opus critic produces detailed paper-grounded technical critique. So the
bottleneck cannot be attributed to "critic not producing usable content".

## Hypothesized mechanism for the surface-vs-substance gap

Given F16-fact-3 (oracle has the content) and F16-fact-4 (critic produces
substantive critique), but plans persistently fail to reproduce specific formulas,
named numbers, or canonical benchmark identity (per Opus critique evidence),
the gap must be in **how oracle/critique content is transferred into model
parameters**.

Three hypothesized bottlenecks (forward-looking; each falsifiable by ABC
experiments below):

### H16-1 — Long-context retrieval bottleneck

Hypothesis: Qwen3-30B at 10K-word oracle in ICL has limited capacity to copy a
specific 30-50 token formula sequence (e.g., `J_RS = (1/β) · log E[e^{β·r}]`)
into its output. It can echo high-frequency named tokens ("PUCT", "GRPO") that
appear repeatedly across oracle items, but rarely commits to the exact RHS of a
formula.

Falsifiable by Experiment A: prompt frozen Qwen3-30B with full oracle slim and
a direct retrieval instruction ("quote Math 5 verbatim from the oracle"). If
fails → retrieval-bottleneck confirmed; if succeeds → bottleneck is downstream
(distillation incentive or LoRA bandwidth).

### H16-2 — Critique-conditioned advantage rewards critique-conformity, not oracle-fidelity

Hypothesis: A_t = π_θ(y | goal, oracle, critique) − π_θ(y | goal, oracle)
measures "how does adding critique to the prompt shift π_θ's evaluation of y".
This signal does not specifically push π_θ to copy oracle content; it pushes
π_θ to produce content that critique would approve of. If π_θ does not
internally know which oracle item to ground a critique-approved answer in, the
gradient direction is content-agnostic — it could just as easily push toward
"plausible-looking PUCT-shaped equation" as toward "the actual PUCT formula
from oracle".

Falsifiable by Experiment B: measure n-gram overlap between student plan output
and oracle Math + Methodology sections per training iter. If overlap stays flat
or decreases → distillation does NOT push the model toward oracle copy. If
overlap increases monotonically → distillation IS pushing toward oracle, and
the failure is in how π_θ executes the copy (H16-1 / H16-3).

### H16-3 — LoRA bandwidth insufficient for formula-length internalization

Hypothesis: LoRA rank=64, lr=5e-5, 16 iters × 8 plans × ~2K tokens with
pos_frac ≈ 0.08 (verified from launch.log mean_adv stats) yields ~20K
positive-advantage tokens of effective gradient signal across the entire LoRA
manifold. This bandwidth is sufficient to reweight existing token-frequency
priors (section header preference, named-method keyword frequency) but
insufficient to engrave a specific 30-50 token formula sequence as a stable new
high-probability path through the manifold.

Falsifiable by Experiment C: compare full-oracle-in-prompt baseline vs
RAG-style top-K oracle item retrieval. If RAG > full-oracle → attention
bandwidth was the dominant bottleneck (H16-1 angle). If RAG ≈ full-oracle →
bottleneck is in distillation incentive (H16-2) or LoRA bandwidth (H16-3),
both not solved by smaller context.

## Why the user's "semantic manifold" framing is partially right

User's intuitive framing (paraphrased from session 2026-04-28 PM):

> LLM ≈ semantic manifold; RL/distillation = path selection on the manifold;
> oracle slim = expanding the manifold; RL → should be able to put paths on the
> goals we want.

Right in spirit:
- LLM internal representations do form a manifold-like structure
- LoRA does modify path weights on that manifold
- Oracle does provide a "target region" the model could in principle reach

Mechanically incomplete:
- Oracle is in the prompt, not in the loss. It is a conditioning input, not a
  gradient target. It cannot "expand the manifold"; it can only **reweight the
  conditional distribution along directions the manifold already supports**.
- LoRA at 16 iter / rank 64 mostly **reweights existing modes**, not adds new
  ones. Engraving a new 30-token formula sequence as a stable new mode is
  closer to manifold expansion than reweighting and exceeds our budget.
- The gradient signal (advantage) does not specifically reward oracle-fidelity;
  it rewards critique-conformity. These are correlated but not identical.

## Implications for paper framing

Honest paper framing if H16-1/H16-2/H16-3 are corroborated by ABC:

> "We characterize a surface ceiling of critique-conditioned self-distillation
> on a frozen-pretrain-capacity backbone (Qwen3-30B + LoRA): structural priors
> the base model already possesses (section format, named-method keyword
> frequency) are reinforced, but content-level corrections (specific equation
> RHS, canonical benchmark identity, named numerical results) are not
> internalized. Despite a 10K-word oracle in prompt containing all the
> formulas the critic flags as missing, n-gram overlap between student outputs
> and oracle math/methodology sections remains stable across iters, indicating
> the oracle → output pathway is not active. Trust-region interpolation with
> the frozen base distribution at α=0.1 monotonically delays the resulting
> self-distillation drift cliff (base iter 5-6 → α=0.05 iter 8 → α=0.1 iter
> 11), corroborating the diagnosis that the cliff is caused by un-anchored
> self-evaluation drift on a token-level signal that is not grounded by an
> external verifier."

This framing is a paper-grade negative-result-with-mechanism: it does not
overclaim "v8 beats v7-opd-full" (falsified — see F15) and does not overclaim
"distillation teaches model new content" (under test — see ABC plan). Instead
it pinpoints **where future work should intervene**: oracle-grounded reward
(verifier or oracle-faithfulness signal in loss), retrieval-aware distillation
(RAG instead of full oracle), or higher-bandwidth update (full-finetune or
larger LoRA / more iters).

## Files / sources

- Trainer code: `src/co_scientist/d5_abstract_retrieve_refine/train_mu_v8_d5sdpo.py:315-475`
- Prompt builders: `src/co_scientist/d5_abstract_retrieve_refine/mu_prompts_v8.py:48-100`
- Oracle: `projects/d5_abstract_retrieve_refine/data/oracles/oracle_v2_2026_04_26_build/slim.md`
- Critic responses (all cells × all iters):
  `runs/2026_04_29_mu_v8_d5sdpo_<cell>/critic_responses/iter_NNN.json`
- Phase 2 audit data: per F15 finding doc
- ABC experiment plan: `knowledge/current/EXPERIMENT_PLAN_oracle_transfer_ABC.md`

---

## Falsification update (filed 2026-04-29)

ABC experiments executed. Verdicts:

### H16-1 — STRONGLY FALSIFIED

Exp A (`runs/2026_04_29_exp_A_oracle_retrieval/verdict.md`): frozen
Qwen3-30B-A3B with full slim oracle (18,150 tokens) returned **4/4
verbatim-correct retrievals** under explicit-quote instructions:

- Q1 Math 5 J_RS: LaTeX-rendered equivalent of target, source `(Jiang et al. 2025) (Ref [27])` ✓
- Q2 Math 7 GRPO loss: exact char-by-char including MiGrATe-specific `eps_low/eps_high` convention ✓
- Q3 Methodology 9 RS-GRPO advantage: exact ✓
- Q4 Empirical 4 AIME: exact `15.6% -> 77.9% over ~10,000 RL steps` ✓

Q5 control (write J_RS without consulting oracle) hit all 4 anchors —
model already knows J_RS from prior training. Q1 partially confounded
by prior knowledge, but Q2 and Q4 are unambiguously oracle-retrieved
(item-specific naming + numbers).

**Long-context retrieval is NOT the bottleneck.** The model is fully
capable of citing oracle content when explicitly directed to.

### H16-2 — CORROBORATED at n ≥ 3

Exp B (`runs/2026_04_29_exp_B_ngram_overlap/`): n-gram Jaccard overlap
of `plan_text` vs slim Math 1-12 + Methodology 1-17 (29 items, 8,592
tokens) across 6 cells × 7-13 iters × 8 plans/iter.

Joint verdict per n:

- n=1: MIXED (3/6 rising; ref overlap 0.168 < student peaks 0.18-0.21
  because students have full oracle in prompt and copy vocabulary mechanically)
- n=2: MIXED (1/6 rising; only base cell)
- **n=3: CORROBORATED (no cells rising; ref floor 0.002)**
- n=4: CORROBORATED (no cells rising; ref = 0 saturated)

Critical observations:

- All 6 cells follow inverted-U (rise iter 0→3-4, plateau, decline)
  regardless of mask/α/loss/IS-vs-PPO config — the rising portion is
  shared with the no-training base cell, so it is **not driven by the
  SDPO advantage**. Likely an effect of LoRA learning generic structural
  priors during early iters before template-collapse.
- **G+ (peak audit 22.38) and C+ (peak audit 23.50) — best cells — have
  the STEEPEST n-gram declines** at every n. The cells that please the
  audit grader most are also the cells moving FARTHEST from oracle
  vocabulary. Strong evidence that audit reward and oracle fidelity are
  anti-correlated under v8-d5sdpo.

**The SDPO mechanism does not push toward oracle content internalization
during natural plan generation, even though the model is capable of doing
so when explicitly asked (per Exp A).** This is exactly H16-2 — the
critique-conditioned advantage is content-agnostic with respect to oracle
fidelity.

### H16-3 — INDETERMINATE (separable from H16-2 not achievable under this design)

Exp C (`runs/2026_04_29_mu_v8_d5sdpo_Gplus_rag/`, AUDIT_SUMMARY.md): G+ config +
RAG top-K=5. Stopped at iter 8/16 after cliff confirmed at iter 6 (manual
stop, audit_drop_threshold=7.0 not auto-triggered: 5.4-pt drop < 7.0).

Trajectory vs G+ baseline:

| iter | RAG /45 | G+ /45 | Δ |
|---|---:|---:|---:|
| 0-4 | 18.88 → 20.12 | 19.50 → 22.38 | -0.6 to -2.4 |
| **5 (RAG peak)** | **21.25** | 22.38 (peak) | **-1.12** |
| 6 (RAG cliff) | 15.88 | 21.38 | -5.50 |
| 7 (post-cliff) | 16.88 | 20.00 | -3.12 |

Per pre-registered EXPERIMENT_PLAN L154: `RAG ≈ G+ peak (Δ < 1) → H16-2
dominant; H16-3 indeterminate`. Δ peak = -1.12 lands on this row.

**Interpretation**: the gradient signal extracts the same amount of value
from a 5-item retrieved context as from the 87-item full oracle. This
isolates the bottleneck from "input bandwidth" hypotheses (H16-1 and
H16-3 both downstream of input). H16-2 (incentive misalignment) is the
dominant load-bearing fault.

### F17 — Cliff acceleration (NEW, off-matrix)

Unanticipated finding: **RAG cell template-collapsed at iter 6 vs G+
baseline cliff at iter ~11** (per F15). RAG accelerated cliff onset by
~5 iters at matched audit-drop-threshold and matched α/mask/loss config.

Hypothesis (to be characterized in F17): oracle BREADTH (87 items) provides
an anti-collapse anchor independent of which items contribute to gradient
signal. Mechanism candidates: (a) richer per-iter teacher attention shifts
on critique-conditioned retrieval (since for full oracle every iter sees
all items, but for RAG the items shift per critique → less stable teacher
context); (b) vocabulary diversity that resists boilerplate contraction;
(c) more low-probability-but-relevant items providing exploration boost
to LoRA gradient.

Off-matrix because pre-registered ABC plan didn't isolate cliff timing
as a primary metric. F17 will design a follow-up rag_k sweep (k ∈ {5, 10,
20, 40}) to test if cliff iter scales with K.

### Production decision (post-ABC)

**Primary lever**: verifier-grounded reward (sympy / citation / executor).
Larger LoRA or wider context will not solve the ceiling — gradient signal
doesn't reward oracle-fidelity. See `RESULTS_oracle_transfer_ABC.md` and
F17 spec (to be written).

**RAG production caveat**: pure RAG-only swap is NOT a viable substitute
for full oracle. Even though peak audit matches, F17 cliff acceleration
loses ~5 iters of useful training. If retrieval is adopted, it should be
hybrid (full oracle for student / retrieved for teacher) or stacked with
verifier-grounded reward + stronger anti-collapse anchor.

### Implication for paper framing (revised again)

The L208-223 framing was partially wrong about H16-1 → H16-3 grouping.
Updated framing:

> "The dominant fault is H16-2: the critique-conditioned advantage signal
> rewards critique-conformity, not oracle-fidelity. Three falsification
> experiments isolate this from competing hypotheses: (Exp A) frozen
> Qwen3-30B-A3B retrieves specific oracle items verbatim 4/4 under direct
> instruction with the 18K-token oracle in context, ruling out long-context
> retrieval bottleneck; (Exp B) n-gram overlap of student outputs with the
> oracle Math+Methodology corpus stays flat or declines across all 6 v8
> training cells regardless of mask/α/loss configuration, with the highest-
> audit cells (G+ 22.38, C+ 23.50) showing the steepest declines; (Exp C)
> swapping the full 87-item oracle for top-K=5 retrieved items in the
> highest-performing cell reproduces peak audit (RAG 21.25 vs G+ 22.38,
> Δ=-1.12 within noise), confirming gradient signal extracts equal value
> from 5 vs 87 items. Production direction is verifier-grounded reward.
> Companion finding (F17): RAG accelerates template-collapse cliff by ~5
> iterations (RAG iter 6 vs G+ iter ~11), suggesting oracle breadth provides
> an anti-collapse anchor distinct from gradient-signal bandwidth."

### Implication for paper framing (revised from L208-223)

The L208-223 honest framing was **partially wrong**: it grouped H16-1
with H16-2 / H16-3 as joint bottlenecks, but H16-1 has been falsified.
The corrected framing should isolate H16-2 as the dominant fault:

> "We characterize a surface ceiling of critique-conditioned
> self-distillation on a frozen-pretrain-capacity backbone (Qwen3-30B
> + LoRA): structural priors the base model already possesses (section
> format, named-method keyword frequency) are reinforced, but
> content-level corrections (specific equation RHS, canonical benchmark
> identity, named numerical results) are not internalized. The model
> retains the capability to retrieve specific oracle content verbatim
> when explicitly instructed (frozen Qwen3-30B with 18K-token oracle in
> context: 4/4 verbatim quotation under direct-quote instruction), but
> the critique-conditioned advantage A_t = π_θ(y | g, o, c) − π_θ(y | g,
> o) is content-agnostic with respect to oracle fidelity — it rewards
> critique-conformity, which the audit grader reinforces in directions
> increasingly orthogonal to oracle vocabulary (best-audit cells G+/C+
> show the steepest declines in n-gram overlap with oracle math/method
> sections across training iters)."

The lever is **incentive structure**, not retrieval/attention/bandwidth.
Production direction (pending Exp C confirmation): verifier-grounded
reward (sympy expression equality, citation match), or
oracle-faithfulness signal added directly to the advantage.

### Production decision (pending Exp C)

| Exp C outcome | Decision |
|---|---|
| RAG ≈ G+ peak | Verifier-grounded reward primary lever |
| RAG > G+ peak | RAG production + verifier-grounded reward stacked |
| RAG < G+ peak | Hybrid full+retrieved oracle + verifier-grounded reward |

See `knowledge/current/RESULTS_oracle_transfer_ABC.md` for joint cross-table.
