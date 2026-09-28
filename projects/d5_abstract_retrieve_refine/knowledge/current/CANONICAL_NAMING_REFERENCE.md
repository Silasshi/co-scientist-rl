# D5 Canonical Naming Reference — paper-grounded ArXiv attribution

*Created 2026-04-27 PM. **Revised 2026-04-27 PM** after WebFetch from arxiv verified that earlier "option (c) HER from Hübotter 2026" claim has NO basis in any paper — that label was D5-internal speculation.*

**Source of truth ranking**:
1. arxiv paper algorithm sections (verified via WebFetch from `arxiv.org/html/<id>`)
2. Raw `train_*.py` source code (sample / compute_logprobs / forward_backward calls)
3. `RUN_REGISTRY.md` for run-level setup
4. **NOT** docstrings or memory or older md docs

## Two papers — paper-grade attribution

| Term | ArXiv | Title | First author | Algorithm essence (verified from paper algorithm section) |
|------|-------|-------|--------------|-------------------------------------------------------------|
| **SDPO** | **2601.20802** | "Reinforcement Learning via Self-Distillation" | Hübotter et al. | Algorithm 1: STUDENT samples on-policy `y ~ π_θ(·|x)`; teacher = same model conditioned on environmental feedback `f`; loss = `KL(student ‖ stopgrad(teacher_with_f))`; uses top-K vocab approximation (full vocab too memory-heavy); EMA / trust-region teacher regularization variants. **Paper does NOT discuss "option (a)/(b)/(c)" or "HER relabelling"**. |
| **OPSD** | **2601.18734** | "Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models" | Zhao et al. | STUDENT samples on-policy `ŷ ~ p_S(·|x)`; teacher = same model conditioned on **privileged answer `y*`** (ground-truth CoT from dataset); default loss = generalized Jensen-Shannon (β=0.5); **sampled-token policy-gradient variant** (Table 3, ~2% gap from full-vocab) computes A_n = log(p_T) − log(p_S) per sampled token, no full vocab needed. |

## Two terms we should NOT use without qualifier

| Term | Why ambiguous |
|------|---------------|
| **"OPD"** | Loose synonym we used internally for "On-Policy Distillation". Not the paper name of either of the two papers above (OPSD has S; SDPO is different). Use **"OPSD (Zhao 2601.18734)"** or **"Hübotter SDPO (2601.20802)"** with explicit ArXiv ID. |
| **"option (c) HER" / "Hindsight Experience Replay relabelling"** | Originally from `SDPO_RECIPE_v1.md:121-122` claiming this is "from Hübotter 2026". **WebFetch of arxiv 2601.20802 confirms paper does NOT discuss any such variant.** This was a D5 internal speculation that propagated into doc claims. Don't use it as if it's a paper concept; if needed, refer to D5's actual variant as "D5 in-house off-policy IS-loss variant". |

## What D5 trainers actually implement (vs what they were called)

Audited from raw code in `src/co_scientist/d5_abstract_retrieve_refine/train_*.py`:

| Trainer | Self-label in docstring/log | **Truthful description (from code)** | Closest paper alignment |
|---------|---------------------------|--------------------------------------|-------------------------|
| `train_mu_v2.py` | "SDPO" | TEACHER samples plans under teacher_input (`goal+oracle+critique`); STUDENT lp recomputed under student_input (`goal+oracle`); `A_t = clamp((t_lp − s_lp)·1.0, ±5)`; `forward_backward(loss="importance_sampling")` on student_prefix + teacher_tokens. **Off-policy IS-loss.** | NEITHER SDPO NOR OPSD. Both papers REQUIRE student on-policy sampling. This trainer samples teacher → fundamentally different family. |
| `train_mu_v3.py` | "SDPO" | Same as v2; lr=2e-4 (v2 had 1e-5). | Same as v2 — neither paper. |
| `train_mu_v4.py` (production through 2026-04-27) | "SDPO" | Same as v2/v3; lr=5e-5; n_grad_steps=4. | Same as v2/v3 — neither paper. |
| `train_mu_v6_replay.py` | "SDPO + replay buffer" | v4 main step + additional replay step (sample n historical plans, recompute student lp under current weights, build replay datum with constant `replay_anchor_weight` advantage, separate forward_backward). | Main step: same as v4 (neither paper). Replay step: structurally OPSD-flavor anchor (constant positive advantage = soft CE on historical). Hybrid. |
| `train_kappa_v1.py` | "distillation-level SDPO" | Same off-policy IS-loss pattern, applied per round in 3-round distillation chain. | Same family as v4 — neither paper. |
| `train_mu_v7_opd.py` (`opd_mode=True`, default) | "OPD" / "canonical SDPO flavor" | STUDENT samples on-policy under student_input; TEACHER lp recomputed under teacher_input; `A_t = clamp((t_lp − s_lp)·1.0, ±5)`; `forward_backward(loss="importance_sampling")` on student_prefix + student_tokens. **On-policy IS-loss.** | **Closest to OPSD (Zhao 2601.18734) sampled-token policy-gradient variant** (Table 3, ~2% gap from full-vocab JS). Student on-policy ✓, per-token A = teacher_lp − student_lp ✓, but uses IS-loss form, not direct JS minimization. **NOT** canonical KL-form Hübotter SDPO either (full-vocab/top-K KL not implemented; we use scalar logprob). |
| `train_mu_v7_opd.py` (`opd_mode=False`) | (legacy fallback) | Same as v4 (TEACHER samples). | Neither paper. |
| `train_alpha_v2.py` | (no SDPO label) | Pure SFT on Opus-distilled plans (5 epochs × 16 plans). | Not in SDPO/OPSD family — no sampling, no advantage, just CE on labels. |
| `train_beta_v2.py` | (no SDPO label) | Pure SFT on single reference_solution.txt. | Not in SDPO/OPSD family. |
| `train_sigma_v4_v1.py` | (control) | Frozen inference (base or μ-v4 LoRA + slim oracle + plan_v4 prompt). | Not training. |
| `train_tau_v1.py` | (control) | Frozen 3-round distillation inference. | Not training. |

**Critical takeaway**: every D5 trainer that historically self-labeled as "SDPO" implements an algorithm that is structurally NEITHER canonical Hübotter SDPO NOR canonical Zhao OPSD. The trainers ARE useful and produced measurable results (e.g. F2's μ-v4 +2.75 audit), but those results should be reported as outcomes of "D5 in-house variants", not as outcomes of canonical SDPO/OPSD.

## File:line code evidence for load-bearing trainers

### `train_mu_v4.py` — D5 in-house off-policy IS-loss variant
- L343-348: `sampling_client.sample(prompt=teacher_input, num_samples=8, ...)` — TEACHER samples
- L361-368: `compute_logprobs(student_tokens + list(seq.tokens))` — STUDENT lp on teacher tokens
- L384-389: `a = clamp((t_lp − s_lp) · sdpo_scale, ±sdpo_clip_advantage)`
- L391-396: `build_sdpo_datum(student_prompt_tokens=student_tokens, gen_tokens=teacher_seq_tokens, ...)`
- L419: `forward_backward(loss_fn="importance_sampling")`

### `train_mu_v7_opd.py` (`opd_mode=True`) — D5 in-house on-policy IS-loss variant
- L424: `sampling_client.sample(prompt=student_input, ...)` — STUDENT samples on-policy
- L450-456: `compute_logprobs(teacher_tokens_prefix + list(seq.tokens))` — TEACHER lp on student tokens
- L506-510: `a = clamp((t_lp − s_lp) · sdpo_scale, ±opd_anchor_clip)` — same formula, sources swapped
- L517-522: `build_sdpo_datum(student_prompt_tokens=student_tokens, gen_tokens=student_seq_tokens, ...)`
- L548: `forward_backward(loss_fn="importance_sampling")`

## What our IS-loss formula corresponds to in OPSD (Zhao 2601.18734)

OPSD's sampled-token policy-gradient form (their Table 3 alternative to full-vocab JS):

```
∇L_OPSD ≈ E_{ŷ ~ p_S}[ A_n · ∇log p_S(ŷ_n | x, ŷ_<n) ]
where A_n = log p_T(ŷ_n | x, y*, ŷ_<n) − log p_S(ŷ_n | x, ŷ_<n)
```

Our `train_mu_v7_opd.py` (opd_mode=True):
```
A_t = clamp((teacher_lp − student_lp)·scale, ±5)
loss = importance_sampling: -mean(A_t · log_pi_student(ŷ_t | student_prefix))
gradient = -A_t · ∇log_pi_student
update direction = +A_t · ∇log_pi_student
```

These are gradient-equivalent up to the `clamp(±5)` truncation and `scale` factor. So `train_mu_v7_opd.py` is the **closest D5 implementation to OPSD sampled-token policy-gradient variant**, with these caveats:
- Not full-vocab JS (the canonical OPSD default)
- Not Hübotter top-K KL either
- Teacher-context in our case = `goal + oracle + critique` (Opus-derived) — both papers are flexible here; OPSD specifically uses ground-truth `y*` directly, our Opus critique is a derived signal

## Ground rule for future sessions

| Pattern in old D5 doc | What it likely means | What you should write instead |
|-----------------------|---------------------|-------------------------------|
| "SDPO" near μ-v2/v3/v4/v6/κ-v1 | D5 in-house off-policy IS-loss variant | "D5 in-house off-policy IS-loss (TEACHER samples)" with link to RUN_REGISTRY |
| "canonical SDPO" or "SDPO" with paper citation Hübotter 2026 | Canonical SDPO from arxiv 2601.20802 (paper) | "Hübotter SDPO (2601.20802)" |
| "OPD" or "On-Policy Distillation" | Probably means our μ-v7-opd implementation, but ambiguous | "μ-v7-opd (D5 in-house on-policy IS-loss; closest to OPSD 2601.18734 sampled-token variant)" |
| "OPSD" | Canonical OPSD from arxiv 2601.18734 (Zhao paper) | Same — keep with explicit ArXiv ID |
| "option (c) HER" or "Hübotter option (c)" | D5 internal speculation; NOT IN ANY PAPER | "D5 in-house off-policy IS-loss variant" with note that prior label was unfounded |
| Result claims like "+2.75 SDPO" | Real audit result from D5 in-house variant | "+2.75 from D5 in-house off-policy IS-loss variant (μ-v4)" |

Variable names `sdpo_scale`, `sdpo_clip_advantage`, `opd_mode` etc. are NOT renamed (saved checkpoints depend on them). They apply equally to all D5 in-house IS-loss variants.

## See also

- `RUN_REGISTRY.md` — per-run setup truth table (every historical training run + its actual algorithm + config)
- `SDPO_RECIPE_v1.md` — recipe doc; READ-FIRST header + L121-122 corrected to remove "(c) HER from Hübotter" claim
- `paper_materials/findings/F11..F14*.md` — mechanism findings; updated to use truthful labels
