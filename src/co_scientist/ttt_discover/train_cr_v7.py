"""CR-v7: Per-signal context REINFORCE trainer (standalone).

Extracted from `train_critique_revise.py` 2026-04-17 for clarity. This
file contains ONLY the active CR-v7 pipeline — no CR-v4 (whole-plan
rewrite), CR-v5 (paragraph-level edit), or CR-v6 (locus-based revision)
legacy paths. Shared utilities (buffer management, grader plumbing,
fresh-plan prompts, UCB selection) are imported from
`train_buffer_ttt.py` and `train_critique_revise.py` rather than
duplicated.

## Pipeline (per iter)

N_SIGNALS below = len(GRADIENT_SIGNALS) — nominally 9 (S1..S9), or 8
when S4_significance is ablated via config.disabled_signals.

1. **Fresh phase**: `sampling_client.sample(num_samples=n_fresh)`
   generates n_fresh plans under `build_research_plan_prompt`.

2. **Grade fresh**: each fresh plan is graded with
   `emit_critique=True`. Grader emits score + reasoning + prose
   critique per signal. Results (incl. `per_signal_critiques`) stored
   in buffer.

3. **UCB select**: `ucb_select` picks `n_revise` parents from buffer.

4. **Revise sampling (parallel)**: fire all `n_revise` sampling
   futures first; await in second pass. Each parent uses
   `build_whole_plan_revision_prompt` containing ALL N_SIGNALS
   per-signal critiques. Each parent produces `n_revision_candidates`
   (BoN=2) whole-plan rewrites.

5. **Grade revisions**: same `emit_critique=True` pipeline. New
   critiques go into buffer for future iters.

6. **RL loss (per-signal REINFORCE, HER-style relabelling)**: for
   each revision `a`, compute N_SIGNALS-dim Δ vector
   `(s'_i - s_i) / 4`. For each signal with |Δ_i| ≥ delta_threshold:
   - Build `context_i = build_per_signal_context(plan, critique_i,
     signal_name, score, goal)` — same structure as sampling context
     but with ONLY critique_i in the feedback block.
   - Recompute `logπ(a | context_i)` via batched
     `sampling_client.compute_logprobs_async` (SDPO pattern).
   - Emit `types.Datum` with advantage `A_i = Δ_i · delta_scale`.

7. **Training step**: `forward_backward(datums,
   loss_fn="importance_sampling")`. Option (c): stored logprob =
   recomputed logprob → ratio ≡ 1 → per-signal REINFORCE on the
   context_i distribution.

## Buffer schema (every iter stores n_fresh + n_revise entries)

```
iteration, plan_text, signal_vector[N_SIGNALS], aggregate_reward,
hard_gate_passed, per_repeat_scores,
per_signal_critiques[N_SIGNALS],  # ← CR-v7 field
entry_type ∈ {fresh, revision}, parent_idx, delta_reward, ...
```

## References

- Plan: `~/.claude/plans/loss-function-importance-sampling-ppo-p-adaptive-wilkinson.md`
- Decisions: `projects/ttt_discover/DECISIONS.md` (2026-04-17 CR-v7 entry)
- Pipeline doc: `projects/ttt_discover/knowledge/current/PIPELINE.md`

Usage:
    source shared/tools/use_api_profile.sh new
    python src/co_scientist/ttt_discover/train_cr_v7.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.ten_signal_reward import (
    SIGNALS as GRADIENT_SIGNALS,
    SIGNAL_WEIGHTS as GRADIENT_WEIGHTS,
    aggregate_reward as gradient_aggregate,
)
from co_scientist.ttt_discover.train_buffer_ttt import (
    BufferEntry,
    PlanFutures,
    TenSignalReward,
    build_research_plan_prompt,
    collect_plan_reward,
    launch_plan_reward,
    select_context,
)
from co_scientist.ttt_discover.train_critique_revise import (
    MAX_PLAN_CHARS_IN_PROMPT,
    detect_duplicated_content,
    parse_critique_and_plan,
    ucb_select,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)


# =============================================================================
# Active research goal (loaded from disk)
# =============================================================================

_HERE = Path(__file__).resolve()
_GOAL_PATH = _HERE.parents[3] / "projects" / "ttt_discover" / "analysis" / "sanity_check" / "research_goal.txt"
TARGET_GOAL = _GOAL_PATH.read_text().strip() if _GOAL_PATH.exists() else ""


# =============================================================================
# Config (CR-v7 only; no locus / paragraph / c3_* fields)
# =============================================================================


@chz.chz
class Config:
    # --- Infrastructure ---
    base_url: str | None = None
    api_profile: str | None = "new"
    log_path: str = "/home/silas/co-scientist-project/projects/ttt_discover/runs/_archive/cr_v7/default"
    model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"  # policy = grader

    # --- Training loop ---
    n_iterations: int = 25
    n_fresh: int = 4                  # fresh plans per iter (exploration)
    n_revise: int = 4                 # parents selected for revision (exploitation)
    n_revision_candidates: int = 2    # BoN per parent
    grader_repeats: int = 2           # grader calls per signal per plan (median)

    # --- Buffer / context selection ---
    K_exploit: int = 3
    K_explore: int = 2
    cold_start_iters: int = 1         # first N iters: fresh-only, no revisions
    use_ucb: bool = True
    ucb_c: float = 1.0

    # --- Per-signal REINFORCE ---
    delta_threshold: float = 1e-3     # skip per-signal datum if |Δ_i| below
    delta_scale: float = 5.0          # per-signal advantage multiplier

    # --- Baselines ---
    skip_rl_update: bool = False      # B4: in-context only, no gradient
    skip_hard_gates: bool = False     # A1: ablate goal-contrast + claim verif
    strip_critiques: bool = False     # B4_stripped: revision prompt shows only aggregate score

    # --- Signal ablation ---
    disabled_signals: str = ""        # comma-separated IDs (e.g. "S4_significance")

    # --- Fresh-plan training (optional) ---
    train_on_fresh: bool = False      # B3/A8: entropic training on fresh plans
    kl_budget: float = 0.693          # ln 2 (for train_on_fresh)
    beta_max: float = 20.0

    # --- Grader critique emission (required for CR-v7) ---
    emit_signal_critique: bool = True

    # --- Alt grader for signals with grader_model_override (v9: SA_arithmetic) ---
    grader_model_alt: str = ""        # e.g. "Qwen/Qwen3-235B-A22B"; empty = no alt

    # --- Optimization ---
    learning_rate: float = 4e-5
    clip_eps: float = 0.2
    lora_rank: int = 32

    # --- Sampling ---
    max_length: int = 32768
    max_tokens: int = 4096            # whole-plan rewrite (CR-v7 default = 4096)
    grader_max_tokens: int = 4096
    grader_hard_gate_max_tokens: int = 8192
    temperature: float = 1.0
    grader_temperature: float = 0.0

    # --- Format constraints ---
    max_word_count: int = 750
    min_words: int = 30

    # --- Logging / checkpoint ---
    save_every: int = 5
    today_date: str = time.strftime("%Y-%m-%d", time.localtime())
    seed: int = 0


# =============================================================================
# CR-v7 prompts
# =============================================================================


def build_whole_plan_revision_prompt(
    goal: str,
    plan_text: str,
    signal_vector: dict[str, int | None],
    critiques: dict[str, str],
    strip_critiques: bool = False,
) -> str:
    """Whole-plan rewrite conditioned on all per-signal critiques.

    Structure: # Research Goal / # Current Plan / # Per-signal feedback
    (N_SIGNALS lines) / # Output Format. `build_per_signal_context`
    mirrors this exactly except the feedback block contains 1 line —
    keeps logπ(a | context_i) drift bounded to critique content, not
    scaffolding.

    When `strip_critiques=True` (B4_stripped baseline), the feedback
    block is replaced with a single aggregate-score line. This isolates
    the contribution of per-signal critique conditioning.
    """
    if len(plan_text) > MAX_PLAN_CHARS_IN_PROMPT:
        plan_text = plan_text[:MAX_PLAN_CHARS_IN_PROMPT] + "\n\n[... rest truncated ...]"

    if strip_critiques:
        agg = gradient_aggregate(signal_vector)
        feedback_block = f"- Overall score: {agg:.3f}/1.0"
    else:
        feedback_lines = []
        for spec in GRADIENT_SIGNALS:
            score = signal_vector.get(spec.id)
            critique = (critiques or {}).get(spec.id, "") or ""
            score_str = f"{score}/5" if score is not None else "?/5"
            if critique:
                feedback_lines.append(f"- {spec.name} ({score_str}): {critique}")
            else:
                feedback_lines.append(f"- {spec.name} ({score_str}): [no critique]")
        feedback_block = "\n".join(feedback_lines)

    return (
        f"# Research Goal\n{goal}\n\n"
        f"# Current Plan\n{plan_text.strip()}\n\n"
        f"# Per-signal feedback\n{feedback_block}\n\n"
        f"# Output Format\n"
        f"Rewrite the plan addressing the feedback above.\n\n"
        f"<think>\n"
        f"...your reasoning...\n"
        f"</think>\n"
        f"<solution>\n"
        f"...your rewritten plan...\n"
        f"</solution>"
    )


def build_per_signal_context(
    plan_text: str,
    critique_i: str,
    signal_name: str,
    score: int | None,
    goal: str,
) -> str:
    """Single-critique context used at LOSS time (never sampled).

    The policy action `a` was sampled under `build_whole_plan_revision_prompt`
    (all 8 critiques). At loss time we recompute `logπ(a | context_i)` under
    this single-critique context — HER-style relabelling that gives per-signal
    gradient decoupling.
    """
    if len(plan_text) > MAX_PLAN_CHARS_IN_PROMPT:
        plan_text = plan_text[:MAX_PLAN_CHARS_IN_PROMPT] + "\n\n[... rest truncated ...]"
    critique = critique_i.strip() if critique_i else "[no critique]"
    score_str = f"{score}/5" if score is not None else "?/5"
    return (
        f"# Research Goal\n{goal}\n\n"
        f"# Current Plan\n{plan_text.strip()}\n\n"
        f"# Per-signal feedback\n- {signal_name} ({score_str}): {critique}\n\n"
        f"# Output Format\n"
        f"Rewrite the plan addressing the feedback above.\n\n"
        f"<think>\n"
        f"...your reasoning...\n"
        f"</think>\n"
        f"<solution>\n"
        f"...your rewritten plan...\n"
        f"</solution>"
    )


# =============================================================================
# Main training loop
# =============================================================================


def main(config: Config):
    os.makedirs(config.log_path, exist_ok=True)
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    # Signal ablation — must propagate to BOTH the shared reward module AND
    # train_buffer_ttt, which binds `GRADIENT_SIGNALS = ten_signal_reward.SIGNALS`
    # at import time. In-place list mutation (`[:] = ...`) updates every
    # module that imported the list by reference; plain rebinding would
    # miss import-time aliases.
    global GRADIENT_SIGNALS, GRADIENT_WEIGHTS
    if config.disabled_signals:
        disabled = {s.strip() for s in config.disabled_signals.split(",") if s.strip()}
        from co_scientist.shared import ten_signal_reward as _tsr
        from co_scientist.ttt_discover import train_buffer_ttt as _tbt
        active_signals = [s for s in _tsr.SIGNALS if s.id not in disabled]
        active_weights = {s.id: _tsr.SIGNAL_WEIGHTS[s.id] for s in active_signals}
        total_w = sum(active_weights.values())
        if total_w > 0:
            active_weights = {k: v / total_w for k, v in active_weights.items()}
        # Mutate in place so previously-imported aliases (in train_buffer_ttt
        # and elsewhere) see the ablated list.
        _tsr.SIGNALS[:] = active_signals
        _tsr.SIGNAL_WEIGHTS.clear()
        _tsr.SIGNAL_WEIGHTS.update(active_weights)
        # Explicitly rebind the alias in train_buffer_ttt (it used
        # `from ... import SIGNALS as GRADIENT_SIGNALS` which is a
        # separate name binding, not a mutation target).
        _tbt.GRADIENT_SIGNALS = _tsr.SIGNALS
        _tbt.GRADIENT_WEIGHTS = _tsr.SIGNAL_WEIGHTS
        GRADIENT_SIGNALS = _tsr.SIGNALS
        GRADIENT_WEIGHTS = _tsr.SIGNAL_WEIGHTS
        logger.info(f"Signal ablation: disabled {disabled}, {len(active_signals)} active")
        logger.info(f"Reweighted: {active_weights}")

    logger.info(f"TARGET GOAL ({len(TARGET_GOAL)} chars)")
    logger.info(
        f"CR-v7 plan: n_iter={config.n_iterations}, n_fresh={config.n_fresh}, "
        f"n_revise={config.n_revise}, BoN={config.n_revision_candidates}, "
        f"grader_repeats={config.grader_repeats}"
    )
    logger.info(
        f"Paradigm: Per-signal Context REINFORCE | "
        f"Model: {config.model_name} | Grader: {config.grader_model_name} | "
        f"UCB: {config.use_ucb} | HardGates: {not config.skip_hard_gates} | "
        f"SkipRL: {config.skip_rl_update} | TrainFresh: {config.train_on_fresh}"
    )

    rng = random.Random(config.seed)
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Renderer: {renderer_name}")

    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )

    last_checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path)
    if last_checkpoint is None:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank,
        )
        start_iter = 0
        logger.info(f"Fresh start: LoRA rank={config.lora_rank}")
    else:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            last_checkpoint["state_path"]
        )
        start_iter = last_checkpoint["batch"] + 1
        logger.info(f"Resuming from iteration {start_iter}")

    grader_client = service_client.create_sampling_client(
        base_model=config.grader_model_name,
    )
    grader_client_alt = None
    renderer_alt = None
    if config.grader_model_alt:
        grader_client_alt = service_client.create_sampling_client(
            base_model=config.grader_model_alt,
        )
        alt_tokenizer = get_tokenizer(config.grader_model_alt)
        alt_renderer_name = model_info.get_recommended_renderer_name(config.grader_model_alt)
        renderer_alt = renderers.get_renderer(alt_renderer_name, alt_tokenizer)
        logger.info(f"Alt grader: {config.grader_model_alt} (for signals with grader_model_override)")
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    buffer: list[dict] = []
    buffer_path = os.path.join(config.log_path, "buffer.jsonl")
    if os.path.exists(buffer_path):
        with open(buffer_path) as f:
            for line in f:
                buffer.append(json.loads(line))
        logger.info(f"Resumed buffer with {len(buffer)} entries")

    train_log_path = os.path.join(config.log_path, "train", "training_logs.jsonl")
    os.makedirs(os.path.dirname(train_log_path), exist_ok=True)
    iter_summary_path = os.path.join(config.log_path, "train", "iter_summary.jsonl")

    # =========================================================================
    for iter_idx in range(start_iter, config.n_iterations):
        t_start = time.time()

        if config.save_every > 0 and iter_idx > 0 and iter_idx % config.save_every == 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{iter_idx:04d}_{config.today_date}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": iter_idx},
            )

        # Save weights for sampler
        sampling_result = training_client.save_weights_for_sampler(
            name=f"iter_{iter_idx:04d}",
        ).result()
        sampling_client = service_client.create_sampling_client(
            model_path=sampling_result.path,
        )

        # =====================================================================
        # PHASE 1: Generate n_fresh plans
        # =====================================================================
        if iter_idx < config.cold_start_iters:
            context = []
        else:
            import dataclasses as _dc
            _be_fields = {f.name for f in _dc.fields(BufferEntry)}
            valid_buf = [
                BufferEntry(**{k: v for k, v in e.items() if k in _be_fields})
                for e in buffer if e.get("hard_gate_passed")
            ]
            context = select_context(valid_buf, config.K_exploit, config.K_explore, rng)

        # Batched fresh-plan sampling: all n_fresh plans drawn in one
        # `.sample(num_samples=n_fresh)` call (server-side parallel).
        fresh_samples = []
        fresh_prompt_tokens = []
        prompt_text = build_research_plan_prompt(goal=TARGET_GOAL, context=context)
        model_input = renderer.build_generation_prompt(
            [{"role": "user", "content": prompt_text}]
        )
        shared_prompt_ints = model_input.to_ints()
        fresh_future = sampling_client.sample(
            prompt=model_input,
            num_samples=config.n_fresh,
            sampling_params=sampling_params,
        )
        fresh_result = fresh_future.result()
        for seq in fresh_result.sequences:
            text = renderer.parse_response(seq.tokens)[0]["content"]
            if "<solution>" in text and "</solution>" not in text:
                text = text.rstrip() + "\n</solution>"
            fresh_samples.append({"tokens": seq.tokens, "logprobs": seq.logprobs, "text": text})
            fresh_prompt_tokens.append(shared_prompt_ints)

        logger.info(f"Iter {iter_idx}: generated {len(fresh_samples)} fresh plans")

        # Launch grading for fresh plans (parallel, fire-and-forget)
        fresh_futures: list[PlanFutures | None] = []
        for sample in fresh_samples:
            if len(sample["text"].strip().split()) < config.min_words:
                fresh_futures.append(None)
                continue
            fresh_futures.append(
                launch_plan_reward(
                    plan=sample["text"], goal=TARGET_GOAL,
                    grader_client=grader_client, renderer=renderer,
                    n_repeats=config.grader_repeats,
                    grader_max_tokens=config.grader_max_tokens,
                    hg_max_tokens=config.grader_hard_gate_max_tokens,
                    temperature=config.grader_temperature,
                    skip_hard_gates=config.skip_hard_gates,
                    emit_critique=config.emit_signal_critique,
                    grader_client_alt=grader_client_alt, renderer_alt=renderer_alt,
                )
            )

        # =====================================================================
        # PHASE 2: Critique-Revise from buffer
        # =====================================================================
        revise_samples = []
        revise_futures: list[PlanFutures | None] = []

        if iter_idx >= config.cold_start_iters and len(buffer) >= config.n_revise:
            if config.use_ucb:
                total_selections = sum(e.get("n_selected", 0) for e in buffer)
                candidates = ucb_select(buffer, config.n_revise, total_selections, config.ucb_c)
                for buf_idx, _ in candidates:
                    buffer[buf_idx]["n_selected"] = buffer[buf_idx].get("n_selected", 0) + 1
            else:
                valid_entries = [(i, e) for i, e in enumerate(buffer) if e.get("hard_gate_passed")]
                valid_entries.sort(key=lambda x: x[1]["aggregate_reward"], reverse=True)
                candidates = valid_entries[:config.n_revise]

            # Pass 1 — fire all revise sampling futures concurrently
            v7_pending: list[tuple[int, dict, dict, list[int], "tinker.Future"]] = []
            for buf_idx, parent in candidates:
                parent_critiques = parent.get("per_signal_critiques", {}) or {}
                revise_prompt = build_whole_plan_revision_prompt(
                    goal=TARGET_GOAL,
                    plan_text=parent["plan_text"],
                    signal_vector=parent["signal_vector"],
                    critiques=parent_critiques,
                    strip_critiques=config.strip_critiques,
                )
                revise_input = renderer.build_generation_prompt(
                    [{"role": "user", "content": revise_prompt}]
                )
                revise_tokens_ints = revise_input.to_ints()
                future = sampling_client.sample(
                    prompt=revise_input,
                    num_samples=config.n_revision_candidates,
                    sampling_params=sampling_params,
                )
                v7_pending.append((buf_idx, parent, parent_critiques, revise_tokens_ints, future))

            # Pass 2 — await all, process candidates
            for buf_idx, parent, parent_critiques, revise_tokens_ints, future in v7_pending:
                revise_result = future.result()
                for cand_idx, seq in enumerate(revise_result.sequences):
                    raw_text = renderer.parse_response(seq.tokens)[0]["content"]
                    critique_str, revised_plan = parse_critique_and_plan(raw_text)
                    if not revised_plan:
                        revised_plan = raw_text
                    if detect_duplicated_content(revised_plan):
                        logger.warning(
                            f"Iter {iter_idx} cand {cand_idx}: duplicated content, "
                            "reverting to parent"
                        )
                        revised_plan = parent["plan_text"]
                        critique_str = (critique_str or "") + " [REJECTED: dup]"
                    if "<solution>" in revised_plan and "</solution>" not in revised_plan:
                        revised_plan = revised_plan.rstrip() + "\n</solution>"
                    plan_for_grading = revised_plan
                    if "<solution>" not in plan_for_grading:
                        plan_for_grading = f"<solution>\n{plan_for_grading}\n</solution>"

                    revise_samples.append({
                        "parent_idx": buf_idx,
                        "parent_reward": parent["aggregate_reward"],
                        "parent_plan_text": parent["plan_text"],
                        "parent_signal_vector": dict(parent["signal_vector"]),
                        "parent_critiques": dict(parent_critiques),
                        "critique": critique_str or "",
                        "text": plan_for_grading,
                        "tokens": seq.tokens,
                        "logprobs": seq.logprobs,
                        "revise_tokens_ints": revise_tokens_ints,
                        "candidate_idx": cand_idx,
                    })
                    if len(plan_for_grading.strip().split()) >= config.min_words:
                        revise_futures.append(
                            launch_plan_reward(
                                plan=plan_for_grading, goal=TARGET_GOAL,
                                grader_client=grader_client, renderer=renderer,
                                n_repeats=config.grader_repeats,
                                grader_max_tokens=config.grader_max_tokens,
                                hg_max_tokens=config.grader_hard_gate_max_tokens,
                                temperature=config.grader_temperature,
                                skip_hard_gates=config.skip_hard_gates,
                                emit_critique=config.emit_signal_critique,
                                grader_client_alt=grader_client_alt, renderer_alt=renderer_alt,
                            )
                        )
                    else:
                        revise_futures.append(None)

            logger.info(
                f"Iter {iter_idx}: generated {len(revise_samples)} revision candidates "
                f"for {len(candidates)} parents (best-of-{config.n_revision_candidates})"
            )

        # =====================================================================
        # Collect all grading results
        # =====================================================================
        logger.info(f"Iter {iter_idx}: collecting grading results...")
        t_collect = time.time()

        fresh_rewards: list[TenSignalReward | None] = []
        for fut in fresh_futures:
            if fut is None:
                fresh_rewards.append(None)
                continue
            try:
                fresh_rewards.append(collect_plan_reward(fut, tokenizer, skip_hard_gates=config.skip_hard_gates))
            except Exception as e:
                logger.error(f"Fresh plan grading failed: {type(e).__name__}")
                fresh_rewards.append(None)

        revise_rewards: list[TenSignalReward | None] = []
        for fut in revise_futures:
            if fut is None:
                revise_rewards.append(None)
                continue
            try:
                revise_rewards.append(collect_plan_reward(fut, tokenizer, skip_hard_gates=config.skip_hard_gates))
            except Exception as e:
                logger.error(f"Revision grading failed: {type(e).__name__}")
                revise_rewards.append(None)

        logger.info(f"Iter {iter_idx}: collection complete in {time.time() - t_collect:.0f}s")

        # =====================================================================
        # Add to buffer (fresh + best-of-N revisions)
        # =====================================================================
        for i, (sample, reward) in enumerate(zip(fresh_samples, fresh_rewards)):
            if reward is None:
                continue
            entry = {
                "iteration": iter_idx,
                "plan_text": sample["text"],
                "signal_vector": {k: v for k, v in reward.signal_vector.items() if v is not None},
                "hard_gate_passed": reward.hard_gate_passed,
                "hard_gate_failed_name": reward.hard_gate_failed_name,
                "aggregate_reward": reward.aggregate_reward,
                "goal_contrast_diag": reward.goal_contrast_diag,
                "claim_verification_diag": reward.claim_verification_diag,
                "per_repeat_scores": reward.per_repeat_scores,
                "per_signal_critiques": dict(reward.per_signal_critiques),
                "word_count": len(sample["text"].strip().split()),
                "entry_type": "fresh",
                "parent_idx": None,
                "delta_reward": None,
            }
            buffer.append(entry)
            with open(buffer_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Best-of-N filter per parent: keep the candidate with highest delta
        from collections import defaultdict
        parent_candidates: dict[int, list[tuple[dict, TenSignalReward, float]]] = defaultdict(list)
        for rsample, rreward in zip(revise_samples, revise_rewards):
            if rreward is None:
                continue
            delta = rreward.aggregate_reward - rsample["parent_reward"]
            parent_candidates[rsample["parent_idx"]].append((rsample, rreward, delta))

        revision_deltas = []
        best_revisions: list[tuple[dict, TenSignalReward, float]] = []
        for parent_idx, cands in parent_candidates.items():
            best_cand = max(cands, key=lambda x: x[2])
            best_revisions.append(best_cand)
            rsample, rreward, delta = best_cand
            revision_deltas.append(delta)

            entry = {
                "iteration": iter_idx,
                "plan_text": rsample["text"],
                "signal_vector": {k: v for k, v in rreward.signal_vector.items() if v is not None},
                "hard_gate_passed": rreward.hard_gate_passed,
                "hard_gate_failed_name": rreward.hard_gate_failed_name,
                "aggregate_reward": rreward.aggregate_reward,
                "goal_contrast_diag": rreward.goal_contrast_diag,
                "claim_verification_diag": rreward.claim_verification_diag,
                "per_repeat_scores": rreward.per_repeat_scores,
                "per_signal_critiques": dict(rreward.per_signal_critiques),
                "word_count": len(rsample["text"].strip().split()),
                "entry_type": "revision",
                "parent_idx": rsample["parent_idx"],
                "critique_text": rsample.get("critique"),
                "delta_reward": delta,
                "n_candidates": len(cands),
                "best_of_n_deltas": sorted([d for _, _, d in cands], reverse=True),
            }
            buffer.append(entry)
            with open(buffer_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Count positive-delta revisions across ALL candidates (not just
        # the best-of-N survivors) — reported in iter_summary for
        # visibility into how often the revise step actually helps.
        all_valid_revisions = [
            (rs, rr, rr.aggregate_reward - rs["parent_reward"])
            for rs, rr in zip(revise_samples, revise_rewards)
            if rr is not None
        ]
        n_positive_revisions = sum(1 for _, _, d in all_valid_revisions if d > 0)

        # =====================================================================
        # PHASE 3: Per-signal REINFORCE loss construction
        # =====================================================================
        # Option (c) HER-style relabelling: stored logprob = recomputed
        # under context_i, not the sampling context. IS ratio = exp(new -
        # stored) = 1 → importance_sampling loss reduces to
        # L_i = -A_i · logπ(a | context_i) — per-signal REINFORCE on the
        # context_i distribution. Bias is intentional: we want the
        # single-critique gradient direction driven by the outcome of an
        # all-hints rewrite.
        training_datums = []
        beta = 0.0

        valid_fresh = [
            (s, r, i) for i, (s, r) in enumerate(zip(fresh_samples, fresh_rewards)) if r is not None
        ]

        # Ablation: train_on_fresh (entropic weights on fresh plans)
        if config.train_on_fresh and len(valid_fresh) >= 2:
            from co_scientist.ttt_discover.train_buffer_ttt import entropic_weights
            fresh_rewards_arr = np.array([r.aggregate_reward for _, r, _ in valid_fresh])
            fresh_weights, beta = entropic_weights(
                fresh_rewards_arr, config.kl_budget, config.beta_max,
            )
            for (sample, reward, orig_idx), weight in zip(valid_fresh, fresh_weights):
                prompt_tokens = [int(t) for t in fresh_prompt_tokens[orig_idx]]
                gen_tokens = [int(t) for t in sample["tokens"]]
                full_seq = prompt_tokens + gen_tokens
                ob_len = len(prompt_tokens) - 1
                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=full_seq[:-1]),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(
                            torch.tensor(full_seq[1:], dtype=torch.long)
                        ),
                        "logprobs": TensorData.from_torch(
                            torch.tensor(
                                [0.0] * ob_len + sample["logprobs"], dtype=torch.float,
                            )
                        ),
                        "advantages": TensorData.from_torch(
                            torch.tensor(
                                [0.0] * ob_len + [float(weight)] * len(sample["logprobs"]),
                                dtype=torch.float,
                            )
                        ),
                    },
                )
                training_datums.append(datum)

        # Per-signal REINFORCE on revision candidates
        if all_valid_revisions:
            # Flat work list: (sid, delta_i, ctx_tokens, gen_tokens) × non-trivial signals
            per_datum_work: list[tuple[str, float, list[int], list[int]]] = []
            for rsample, rreward, _ in all_valid_revisions:
                if not rreward.hard_gate_passed:
                    continue
                parent_sv = rsample["parent_signal_vector"]
                parent_critiques = rsample["parent_critiques"]
                # Truncate once per revision; build_per_signal_context would
                # otherwise run the same check 8× on the same plan.
                parent_plan = rsample["parent_plan_text"]
                if len(parent_plan) > MAX_PLAN_CHARS_IN_PROMPT:
                    parent_plan = parent_plan[:MAX_PLAN_CHARS_IN_PROMPT] + "\n\n[... rest truncated ...]"
                gen_tokens = list(rsample["tokens"])
                if not gen_tokens:
                    continue
                for spec in GRADIENT_SIGNALS:
                    sid = spec.id
                    s_parent = parent_sv.get(sid)
                    s_rev = rreward.signal_vector.get(sid)
                    if s_parent is None or s_rev is None:
                        continue
                    delta_i = (s_rev - s_parent) / 4.0
                    if abs(delta_i) < config.delta_threshold:
                        continue
                    ctx_text = build_per_signal_context(
                        plan_text=parent_plan,
                        critique_i=parent_critiques.get(sid, ""),
                        signal_name=spec.name,
                        score=s_parent,
                        goal=TARGET_GOAL,
                    )
                    ctx_tokens = renderer.build_generation_prompt(
                        [{"role": "user", "content": ctx_text}]
                    ).to_ints()
                    if len(ctx_tokens) < 1:
                        # Defensive: empty context → negative ob_len would
                        # silently mis-align tensors.
                        continue
                    per_datum_work.append((sid, delta_i, ctx_tokens, gen_tokens))

            if per_datum_work:
                # Single asyncio.gather over all (revision × signal) logprob calls.
                model_inputs = [
                    types.ModelInput.from_ints(tokens=ctx + gen)
                    for (_, _, ctx, gen) in per_datum_work
                ]

                async def _batch() -> list[list[float]]:
                    return await asyncio.gather(
                        *[sampling_client.compute_logprobs_async(s) for s in model_inputs]
                    )

                try:
                    logprobs_all = asyncio.run(_batch())
                except RuntimeError:
                    logprobs_all = [
                        sampling_client.compute_logprobs(mi) for mi in model_inputs
                    ]

                for (sid, delta_i, ctx_tokens, gen_tokens), full_logprobs in zip(
                    per_datum_work, logprobs_all,
                ):
                    # compute_logprobs returns one logprob per input token
                    # (same length as input). Slice to the generation portion:
                    # positions [len(ctx) .. len(ctx)+len(gen)-1] each hold
                    # logπ(gen_tokens[k] | ctx + gen_tokens[:k]).
                    # Matches SDPO convention at train_sdpo.py:3460-3463.
                    prompt_len = len(ctx_tokens)
                    gen_logprobs = full_logprobs[prompt_len : prompt_len + len(gen_tokens)]
                    if len(gen_logprobs) != len(gen_tokens):
                        logger.warning(
                            f"CR-v7: logprob length mismatch for {sid}: "
                            f"got {len(gen_logprobs)}, expected {len(gen_tokens)}; skipping."
                        )
                        continue
                    # Length-normalized clamp: skip datum when mean per-token
                    # logprob < -10 (prob < 4.5e-5 avg → off-distribution).
                    mean_logprob = sum(gen_logprobs) / len(gen_logprobs)
                    if mean_logprob < -10.0:
                        continue

                    A_i = float(delta_i) * config.delta_scale
                    full_seq = ctx_tokens + gen_tokens
                    ob_len = prompt_len - 1
                    datum = types.Datum(
                        model_input=types.ModelInput.from_ints(tokens=full_seq[:-1]),
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(
                                torch.tensor(full_seq[1:], dtype=torch.long)
                            ),
                            "logprobs": TensorData.from_torch(
                                torch.tensor(
                                    [0.0] * ob_len + list(gen_logprobs),
                                    dtype=torch.float,
                                )
                            ),
                            "advantages": TensorData.from_torch(
                                torch.tensor(
                                    [0.0] * ob_len + [A_i] * len(gen_tokens),
                                    dtype=torch.float,
                                )
                            ),
                        },
                    )
                    training_datums.append(datum)

        # Apply update
        if training_datums and not config.skip_rl_update:
            try:
                fwd_bwd_future = training_client.forward_backward(
                    training_datums, loss_fn="importance_sampling",
                )
                optim_step_future = training_client.optim_step(adam_params)
                _ = fwd_bwd_future.result()
                _ = optim_step_future.result()
            except Exception:
                logger.exception(f"Iter {iter_idx}: training step failed")

        # =====================================================================
        # Logging
        # =====================================================================
        all_rewards = (
            [r.aggregate_reward for _, r, _ in valid_fresh]
            + [rr.aggregate_reward for rs, rr in zip(revise_samples, revise_rewards) if rr is not None]
        )
        buffer_max = max((e["aggregate_reward"] for e in buffer if e.get("hard_gate_passed")), default=0.0)

        per_signal_means = {}
        all_valid_rewards = (
            [r for _, r, _ in valid_fresh]
            + [rr for rr in revise_rewards if rr is not None]
        )
        for spec in GRADIENT_SIGNALS:
            scores = [r.signal_vector.get(spec.id) for r in all_valid_rewards
                      if r.hard_gate_passed and r.signal_vector.get(spec.id) is not None]
            if scores:
                per_signal_means[f"signal/{spec.id}/mean"] = float(np.mean(scores))

        iter_summary = {
            "iter": iter_idx,
            "time/total": time.time() - t_start,
            "fresh/count": len(valid_fresh),
            "fresh/reward_mean": float(np.mean([r.aggregate_reward for _, r, _ in valid_fresh])) if valid_fresh else 0,
            "revise/count": len(revise_samples),
            "revise/n_positive": n_positive_revisions,
            "revise/mean_delta": float(np.mean(revision_deltas)) if revision_deltas else 0,
            "revise/max_delta": float(max(revision_deltas)) if revision_deltas else 0,
            "reward/mean": float(np.mean(all_rewards)) if all_rewards else 0,
            "reward/max": float(max(all_rewards)) if all_rewards else 0,
            "reward/buffer_max": buffer_max,
            "beta": beta,
            "buffer/size": len(buffer),
            "datums/count": len(training_datums),
            **per_signal_means,
        }
        with open(iter_summary_path, "a") as f:
            f.write(json.dumps(iter_summary) + "\n")

        with open(train_log_path, "a") as f:
            for i, (sample, reward) in enumerate(zip(fresh_samples, fresh_rewards)):
                if reward is None:
                    continue
                f.write(json.dumps({
                    "iter": iter_idx, "type": "fresh", "sample_idx": i,
                    "text": sample["text"],
                    "aggregate_reward": reward.aggregate_reward,
                    "signal_vector": reward.signal_vector,
                    "hard_gate_passed": reward.hard_gate_passed,
                }, ensure_ascii=False) + "\n")
            for i, (rsample, rreward) in enumerate(zip(revise_samples, revise_rewards)):
                if rreward is None:
                    continue
                f.write(json.dumps({
                    "iter": iter_idx, "type": "revision", "sample_idx": i,
                    "text": rsample["text"],
                    "aggregate_reward": rreward.aggregate_reward,
                    "signal_vector": rreward.signal_vector,
                    "hard_gate_passed": rreward.hard_gate_passed,
                    "parent_reward": rsample["parent_reward"],
                    "critique": rsample.get("critique", ""),
                    "delta": rreward.aggregate_reward - rsample["parent_reward"],
                }, ensure_ascii=False) + "\n")

        ml_logger.log_metrics(iter_summary, step=iter_idx)
        logger.info(
            f"Iter {iter_idx}: reward mean={iter_summary['reward/mean']:.3f} "
            f"max={iter_summary['reward/max']:.3f} buffer_max={buffer_max:.3f} "
            f"| revisions: {n_positive_revisions}/{len(revise_samples)} positive "
            f"(mean_delta={iter_summary['revise/mean_delta']:.3f}) "
            f"| time={iter_summary['time/total']:.0f}s"
        )

    # Final checkpoint (guard against resume-complete case where the
    # for-loop never executes and iter_idx is unbound).
    if start_iter < config.n_iterations:
        checkpoint_utils.save_checkpoint(
            training_client=training_client,
            name=f"{iter_idx:04d}_final_{config.today_date}",
            log_path=config.log_path,
            kind="both",
            loop_state={"batch": iter_idx},
        )
    ml_logger.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    chz.entrypoint(main)
