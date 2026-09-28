"""D5 v9 Layer 2 — SFT π_oracle on N oracle-grounded plans.

Builds a frozen reference policy π_oracle (LoRA on top of base Qwen3-30B-A3B)
that has been SFT'd to emit oracle-grounded plans. Used by `train_mu_v9_kl_anchor.py`
as the KL anchor target: `A_total = A_critique + λ·A_grounding − β·KL(π_θ || π_oracle)`.

Pattern adapted from `train_alpha_v2.py:105-243` — same `_build_sft_datum` +
`forward_backward(loss_fn="importance_sampling")` recipe. Uses v9-grounded's
student prompt (goal + full oracle slim) so SFT'd π_oracle is compatible
with v9 trainer's per-token logprob computation under the same student prompt.

Input: `projects/d5_abstract_retrieve_refine/dataset/oracle_grounded_sft.jsonl`
with rows `{plan_id, anchor_group, plan_text}` (30 plans from Phase 2 Opus
generation).

Output: tinker:// uri of final sampler weights, written to
`<log_path>/pi_oracle_final_sampler_uri.txt` for downstream consumption by
v9 KL anchor trainer.

Usage:
    source shared/tools/use_api_profile.sh new
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.train_pi_oracle_sft \
        sft_data_path=projects/d5_abstract_retrieve_refine/dataset/oracle_grounded_sft.jsonl \
        log_path=projects/d5_abstract_retrieve_refine/runs/2026_04_29_pi_oracle_sft \
        n_epochs=3 lora_rank=64 learning_rate=5e-5
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

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
from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v8 import (
    build_student_prompt_v8,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)

REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    api_profile: str | None = "new"
    base_url: str | None = None

    log_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_29_pi_oracle_sft"
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"
    oracle_abstraction_path: str = (
        "projects/d5_abstract_retrieve_refine/data/oracles/oracle_v2_2026_04_26_build/slim.md"
    )
    sft_data_path: str = (
        "projects/d5_abstract_retrieve_refine/dataset/oracle_grounded_sft.jsonl"
    )

    model_name: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 64
    learning_rate: float = 5e-5

    n_epochs: int = 3
    sft_weight: float = 1.0

    save_every: int = 1
    today_date: str = "2026_04_29"


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _build_sft_datum(goal: str, oracle: str, plan_text: str, renderer, tokenizer, sft_weight: float):
    """Build SFT datum from (goal+oracle → grounded_plan) using renderer's supervised builder."""
    student_text = build_student_prompt_v8(goal, oracle)
    target_text = f"<solution>\n{plan_text}\n</solution>"
    convo = [
        {"role": "user", "content": student_text},
        {"role": "assistant", "content": target_text},
    ]
    sup_input, sup_weights = renderer.build_supervised_example(convo)
    full_seq = sup_input.to_ints()
    weights = sup_weights.tolist() if hasattr(sup_weights, "tolist") else list(sup_weights)
    advantages = [sft_weight * float(w) for w in weights[1:]]
    logprobs = [0.0] * (len(full_seq) - 1)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=full_seq[:-1]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(
                torch.tensor(full_seq[1:], dtype=torch.long)
            ),
            "logprobs": TensorData.from_torch(
                torch.tensor(logprobs, dtype=torch.float)
            ),
            "advantages": TensorData.from_torch(
                torch.tensor(advantages, dtype=torch.float)
            ),
        },
    )


def main(config: Config):
    log_dir = _resolve(config.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    ml_log.setup_logging(
        log_dir=str(log_dir), wandb_project=None, wandb_name=None,
        config=config, do_configure_logging_module=True,
    )

    goal = _resolve(config.goal_path).read_text().strip()
    oracle = _resolve(config.oracle_abstraction_path).read_text().strip()
    sft_path = _resolve(config.sft_data_path)
    assert sft_path.exists(), f"SFT data not found: {sft_path}"

    sft_records = []
    with open(sft_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            sft_records.append(json.loads(line))
    assert sft_records, f"Empty SFT data file: {sft_path}"
    logger.info("π_oracle SFT — goal=%d c, oracle=%d c, %d plans", len(goal), len(oracle), len(sft_records))
    logger.info("log_dir = %s", log_dir)

    cfg_path = log_dir / "config.json"
    cfg_path.write_text(json.dumps(
        {k: getattr(config, k) for k in dir(config)
         if not k.startswith("_") and not callable(getattr(config, k))},
        default=str, indent=2,
    ))

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name, rank=config.lora_rank,
    )

    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    # Build SFT datums up front
    sft_datums = []
    for i, rec in enumerate(sft_records):
        plan_text = rec.get("plan_text") or rec.get("text") or ""
        if not plan_text.strip():
            logger.warning("Plan %d empty; skipping (id=%s)", i, rec.get("plan_id"))
            continue
        d = _build_sft_datum(goal, oracle, plan_text, renderer, tokenizer, config.sft_weight)
        sft_datums.append(d)
    logger.info("Built %d SFT datums (target tokens per datum vary; full ~900-word plans)", len(sft_datums))

    metrics_path = log_dir / "metrics.jsonl"
    checkpoints_path = log_dir / "checkpoints.jsonl"

    final_sp_path = ""
    for epoch_idx in range(config.n_epochs):
        t0 = time.time()

        # Save sampler weights at start of epoch (so we always have a recoverable checkpoint)
        sp_path = training_client.save_weights_for_sampler(
            name=f"pi_oracle_epoch_{epoch_idx:04d}"
        ).result().path
        final_sp_path = sp_path
        with open(checkpoints_path, "a") as f:
            f.write(json.dumps({"epoch": epoch_idx, "sampler_path": sp_path,
                                "ts": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")

        # SFT forward+backward over all 30 datums in one batch (small dataset)
        t_step0 = time.time()
        fb_fut = training_client.forward_backward(sft_datums, loss_fn="importance_sampling")
        os_fut = training_client.optim_step(adam_params)
        fb_fut.result()
        os_fut.result()
        t_step = time.time() - t_step0
        logger.info("Epoch %d: SFT step (%d datums) in %.1fs", epoch_idx, len(sft_datums), t_step)

        with open(metrics_path, "a") as f:
            f.write(json.dumps({
                "epoch": epoch_idx, "n_datums": len(sft_datums),
                "wall_total_sec": time.time() - t0, "wall_train_sec": t_step,
                "sampler_path": sp_path,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            }) + "\n")

    # Final sampler save (after last epoch)
    final_sp_path = training_client.save_weights_for_sampler(
        name="pi_oracle_final"
    ).result().path
    with open(checkpoints_path, "a") as f:
        f.write(json.dumps({"epoch": "final", "sampler_path": final_sp_path,
                            "ts": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")

    # Write the final sampler URI to a discoverable file for downstream KL trainer
    uri_file = log_dir / "pi_oracle_final_sampler_uri.txt"
    uri_file.write_text(final_sp_path + "\n")
    logger.info("π_oracle SFT complete. Final sampler: %s", final_sp_path)
    logger.info("URI saved to: %s", uri_file)


if __name__ == "__main__":
    main(chz.entrypoint(Config))
