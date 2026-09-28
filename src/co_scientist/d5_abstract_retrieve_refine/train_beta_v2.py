"""D5 β trainer v2 — direct SFT on reference plan, eval+train use goal+oracle_v2.

Tests: does Qwen3-30B-A3B LoRA SFT directly on (goal → reference_solution.txt)
beat μ baseline? This is the "architecture is overhead" check —
if β ≥ μ, plan-level SDPO + critic adds nothing over direct imitation.

Single training example (the TTT-Discover reference plan). Train for
n_iter LoRA-SGD steps with importance_sampling loss against the reference
plan tokens. Eval n=8 plans from STUDENT context (goal only) every K iters.

NO oracle abstraction (β doesn't see it — would conflate the test).
NO critique (β has no reviewer). NO RL (just supervised CE).

Usage:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.train_beta_v2 \
        log_path=projects/d5_abstract_retrieve_refine/runs/2026_05_xx_beta_v2 \
        n_iter=10 eval_every=2

Decision rule (post-run):
- β ≥ μ (8.88) → "architecture is overhead, paper not publishable"
- β < μ → SDPO/architecture has marginal value
"""
from __future__ import annotations

import json
import logging
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
from co_scientist.shared.opus_audit_subagent import OpusAuditClient
from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v2 import (
    build_audit_request_payload,
    extract_solution,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)


REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    api_profile: str | None = "new"
    base_url: str | None = None

    log_path: str = (
        "projects/d5_abstract_retrieve_refine/runs/2026_05_xx_beta_v2"
    )
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"
    reference_plan_path: str = (
        "projects/d5_abstract_retrieve_refine/dataset/reference_solution.txt"
    )
    oracle_abstraction_path: str = (
        "projects/d5_abstract_retrieve_refine/data/oracles/oracle_v2_2026_04_26_build/slim.md"
    )

    model_name: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 64
    learning_rate: float = 1e-5

    # Single SFT example (the reference plan), trained over n_iter steps
    n_iter: int = 10

    # Sampling for eval (Phase 0.5c fix)
    n_eval_plans: int = 8
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.95
    eval_every: int = 2

    save_every: int = 5
    today_date: str = "2026_04_25"

    # SFT advantage weight (constant per token; matches train_cr_v7.py:1113 anchor pattern)
    sft_weight: float = 1.0


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


_PLAN_FOOTER = (
    "\n\nWrite your research plan inside <solution>...</solution> tags."
    " Target 600 words, max 750 words. Use structured sections (Problem"
    " Statement, Background, Hypothesis, Methodology, Evaluation, Limitations)."
)


def build_student_prompt(goal: str, oracle: str) -> str:
    """β-v2 eval/student context: goal + oracle (slim).

    v2 difference vs v1: SFT target + eval BOTH see oracle_v2_slim. The β-v2
    question is "does direct ref-SFT match σ (frozen + same oracle)?"
    """
    return (
        "I will provide you a research scenario and a set of methodological"
        " patterns extracted from a high-quality research plan on this same"
        " scenario. Use the patterns to guide the structure and reasoning of"
        " your plan, but do not copy the phrasing literally."
        f"\n\nScenario: {goal}"
        f"\n\n# Methodological patterns to apply\n\n{oracle}"
        + _PLAN_FOOTER
    )


def _decode_plan(seq, tokenizer, renderer) -> str:
    parsed = renderer.parse_response(seq.tokens)
    content = parsed[0].get("content", "") if parsed else ""
    return extract_solution(content) if content else tokenizer.decode(seq.tokens)


def main(config: Config):
    log_dir = _resolve(config.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    ml_log.setup_logging(
        log_dir=str(log_dir),
        wandb_project=None, wandb_name=None, config=config,
        do_configure_logging_module=True,
    )

    goal = _resolve(config.goal_path).read_text().strip()
    ref_plan = _resolve(config.reference_plan_path).read_text().strip()
    oracle = _resolve(config.oracle_abstraction_path).read_text().strip()
    logger.info("β v2. goal=%d c, ref=%d c, oracle(slim)=%d c", len(goal), len(ref_plan), len(oracle))
    logger.info("log_dir = %s", log_dir)

    cfg_path = log_dir / "config.json"
    cfg_path.write_text(json.dumps({k: getattr(config, k) for k in dir(config)
                                     if not k.startswith("_") and not callable(getattr(config, k))},
                                    default=str, indent=2))

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )
    last_ckpt = checkpoint_utils.get_last_checkpoint(str(log_dir))
    if last_ckpt is None:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank,
        )
        start_iter = 0
    else:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            last_ckpt["state_path"]
        )
        start_iter = last_ckpt.get("batch", 0) + 1

    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=renderer.get_stop_sequences(),
    )

    audit_client = OpusAuditClient(log_path=log_dir)

    # Build the SFT datum ONCE (reused every iter — same example, multiple updates).
    # Use renderer.build_supervised_example which returns (ModelInput, weights tensor).
    student_text = build_student_prompt(goal, oracle)
    student_input = renderer.build_generation_prompt(
        [{"role": "user", "content": student_text}]
    )

    target_text = f"<solution>\n{ref_plan}\n</solution>"
    full_convo = [
        {"role": "user", "content": student_text},
        {"role": "assistant", "content": target_text},
    ]
    sup_input, sup_weights = renderer.build_supervised_example(full_convo)
    full_seq = sup_input.to_ints()
    weights = sup_weights.tolist() if hasattr(sup_weights, "tolist") else list(sup_weights)
    n_target = int(sum(1 for w in weights if w > 0))
    logger.info(
        "SFT datum built via build_supervised_example: full_seq=%d, n_target=%d (weight>0)",
        len(full_seq), n_target,
    )

    # importance_sampling loss with stored logprobs=0 and advantage = sft_weight on
    # tokens marked trainable, 0 elsewhere. Shift-by-1 (target_tokens = full_seq[1:]).
    advantages = [config.sft_weight * float(w) for w in weights[1:]]
    sft_logprobs = [0.0] * (len(full_seq) - 1)
    sft_datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens=full_seq[:-1]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(
                torch.tensor(full_seq[1:], dtype=torch.long)
            ),
            "logprobs": TensorData.from_torch(
                torch.tensor(sft_logprobs, dtype=torch.float)
            ),
            "advantages": TensorData.from_torch(
                torch.tensor(advantages, dtype=torch.float)
            ),
        },
    )

    metrics_path = log_dir / "metrics.jsonl"
    eval_path = log_dir / "eval_rollouts.jsonl"
    checkpoints_path = log_dir / "checkpoints.jsonl"

    for iter_idx in range(start_iter, config.n_iter):
        t0 = time.time()

        if config.save_every > 0 and iter_idx % config.save_every == 0 and iter_idx > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{iter_idx:06d}_{config.today_date}",
                log_path=str(log_dir),
                kind="state",
                loop_state={"batch": iter_idx},
            )

        sp_path = training_client.save_weights_for_sampler(
            name=f"iter_{iter_idx:04d}"
        ).result().path
        sampling_client = service_client.create_sampling_client(model_path=sp_path)
        with open(checkpoints_path, "a") as f:
            f.write(json.dumps({"iter": iter_idx, "sampler_path": sp_path,
                                 "ts": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")

        # ----- SFT STEP: 1 datum, 1 forward_backward, 1 optim_step -----
        t_step0 = time.time()
        fb_fut = training_client.forward_backward([sft_datum], loss_fn="importance_sampling")
        os_fut = training_client.optim_step(adam_params)
        fb_fut.result()
        os_fut.result()
        t_step = time.time() - t_step0
        logger.info("Iter %d: SFT step in %.1fs", iter_idx, t_step)

        # ----- EVAL AUDIT (async, every eval_every iters) -----
        if iter_idx % config.eval_every == 0:
            t_eval0 = time.time()
            eval_result = sampling_client.sample(
                prompt=student_input,
                num_samples=config.n_eval_plans,
                sampling_params=sampling_params,
            ).result()
            eval_plans = []
            for k, seq in enumerate(eval_result.sequences):
                pid = f"iter_{iter_idx:03d}_eval_{k}"
                ptext = _decode_plan(seq, tokenizer, renderer)
                eval_plans.append({"plan_id": pid, "text": ptext})
            audit_client.submit(
                iter_idx,
                build_audit_request_payload(iter_idx=iter_idx, goal=goal, plans=eval_plans),
            )
            with open(eval_path, "a") as f:
                for ep in eval_plans:
                    f.write(json.dumps({"iter": iter_idx, **ep}) + "\n")
            logger.info("Iter %d: eval audit submitted (%d plans, %.1fs sample, async)",
                         iter_idx, len(eval_plans), time.time() - t_eval0)

        with open(metrics_path, "a") as f:
            f.write(json.dumps({
                "iter": iter_idx,
                "wall_total_sec": time.time() - t0,
                "wall_train_sec": t_step,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            }) + "\n")
        logger.info("Iter %d done in %.1fs", iter_idx, time.time() - t0)

    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=f"final_{config.today_date}",
        log_path=str(log_dir),
        kind="both",
        loop_state={"batch": config.n_iter},
    )
    logger.info("Saved final checkpoint")

    logger.info("Collecting audit responses (waiting up to 30 min)...")
    audit_summary = audit_client.collect_all(
        out_path=log_dir / "audit_log.jsonl",
        timeout_sec=1800.0, poll_interval_sec=15.0,
    )
    valid = [s for s in audit_summary if s.get("mean_total") is not None]
    if valid:
        last3 = valid[-3:]
        beta_proxy = float(np.mean([s["mean_total"] for s in last3]))
        logger.info("==================== β PROXY ====================")
        logger.info("β proxy (mean of last %d audit means): %.2f /20", len(last3), beta_proxy)
        logger.info("μ baseline:    8.88")
        logger.info("δ baseline:    6.75")
        logger.info("ε baseline:   12.13")
        logger.info("If β ≥ 8.88 → architecture is overhead, paper not publishable")
        logger.info("================================================")
    logger.info("β baseline run complete.")


if __name__ == "__main__":
    config = chz.entrypoint(Config)
    main(config)
