"""
Self-Calibration Training Pipeline (bestversion-scale).

Architecture matches best_ver.py:
  - Batch processing: 64 goals per batch
  - Full dataset: all training goals, multiple epochs
  - 3-phase async: generate all → grade all → update

Added on top of bestversion:
  - Dual evaluation: self-eval (blind) + rubric-eval (with rubric)
  - Calibration-weighted GRPO: amplify advantage where model overestimates
  - Per-desideratum gap tracking: identify blind spots

Single model (Qwen3-30B-A3B) for all roles: generation, self-evaluation, grading.

Usage:
  # Train (same scale as bestversion)
  python train_self_calibration.py api_profile=NEW run_eval=False

  # Eval
  python train_self_calibration.py api_profile=NEW run_eval=True eval_epoch=-1
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
import datasets
import tinker
from co_scientist.shared.api_profiles import create_service_client
from tinker import types
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from datasets import load_dataset

from co_scientist.rubric_reward.grpo.best_ver import (
    build_grader_prompt,
    build_research_plan_prompt,
    compute_rubric_reward_from_xml,
)
from co_scientist.ibt.train_ibt import (
    compute_reward,
    create_training_datum,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


# ============================================================
# Config (matches bestversion defaults)
# ============================================================

@chz.chz
class Config:
    base_url: str | None = None
    api_profile: str | None = None
    log_path: str = "/home/silas/co-scientist-project/runs/2026/4/ibt/7"
    model_name: str = "Qwen/Qwen3-30B-A3B"

    run_eval: bool = False
    eval_epoch: int = -1

    ml_data: bool = True
    arxiv_data: bool = False
    pubmed_data: bool = False

    batch_size: int = 64
    group_size: int = 8
    learning_rate: float = 1e-5
    clip_eps: float = 0.2

    lora_rank: int = 64
    save_every: int = 30
    max_tokens: int = 2048
    grader_max_tokens: int = 8192
    temperature: float = 1.0
    grader_temperature: float = 0.0

    max_word_count: int = 750
    target_word_count: int = 600
    scale_length_bonus: float = 120.0
    scaling_factor: float = 0.08
    min_words: int = 30

    w_calibration: float = 0.5

    today_date: str = time.strftime("%Y-%m-%d", time.localtime())


# ============================================================
# Self-Evaluation (blind — no rubric, no reference)
# ============================================================

DESIDERATA_NAMES = [
    "thoroughness", "specificity", "soundness",
    "justification", "efficiency", "ethics", "coherence",
]


def build_blind_self_eval_prompt(scenario: str, plan: str) -> str:
    """Self-evaluation: model scores its own plan on 7 quality dimensions.
    NO rubric items, NO reference solution — pure self-assessment.
    """
    return f"""You are a researcher who just wrote a research plan. Now step back and critically evaluate your own plan's quality.

# Research Scenario
{scenario}

# Your Research Plan
{plan}

# Self-Evaluation Instructions
Score your plan on each of the following 7 quality dimensions. For each dimension, assign a score from 0 to 3:

  0 = NOT SATISFIED: The plan clearly fails on this dimension.
  1 = WEAKLY SATISFIED: The plan touches on this but is vague or superficial.
  2 = PARTIALLY SATISFIED: Reasonable but with notable gaps or weaknesses.
  3 = FULLY SATISFIED: Clear, concrete, and convincing. No major issues.

Be honest and self-critical. Do not inflate your scores.

Return your evaluation in this exact XML format:

<self_eval>
  <thoroughness>[0-3] Does the plan address all key aspects and constraints of the scenario?</thoroughness>
  <specificity>[0-3] Are the methods detailed with concrete parameters, datasets, and steps — not vague?</specificity>
  <soundness>[0-3] Is the plan logically sound with no overlooked flaws that would undermine it?</soundness>
  <justification>[0-3] Is each methodological choice well-motivated with clear rationale?</justification>
  <efficiency>[0-3] Is the plan practical and efficient without unnecessary complexity?</efficiency>
  <ethics>[0-3] Are there any potential negative consequences or ethical concerns?</ethics>
  <coherence>[0-3] Are all parts of the plan consistent and well-integrated with each other?</coherence>
</self_eval>"""


def parse_self_eval_scores(xml_text: str) -> list[float]:
    """Parse blind self-evaluation into 7 scores in [0, 1]."""
    level_map = {0: 0.0, 1: 0.2, 2: 0.6, 3: 1.0}
    scores = []
    parsed_count = 0
    for name in DESIDERATA_NAMES:
        match = re.search(rf"<{name}>[^<]*?(\d)[^<]*?</{name}>", xml_text, re.IGNORECASE | re.DOTALL)
        if match:
            level = int(match.group(1))
            scores.append(level_map.get(min(level, 3), 0.0))
            parsed_count += 1
        else:
            scores.append(0.0)
    if parsed_count < 7:
        logger.warning(f"Self-eval: only parsed {parsed_count}/7 scores")
    return scores


def extract_rubric_per_desideratum(xml_text: str) -> list[float]:
    """Extract per-desideratum means from rubric XML (10 items × 7 desiderata → 7 means)."""
    level_map = {0: 0.0, 1: 0.2, 2: 0.6, 3: 1.0}
    item_blocks = re.findall(r"<item\s+num=.*?>.*?</item>", xml_text, flags=re.DOTALL)
    all_desiderata = [[] for _ in range(7)]
    for item_xml in item_blocks:
        levels = re.findall(r"<level>(\d+)</level>", item_xml)
        levels = [int(l) for l in levels if l.isdigit()]
        if len(levels) < 7:
            levels = levels + [0] * (7 - len(levels))
        levels = levels[:7]
        mapped = [level_map.get(l, 0.0) for l in levels]
        for d in range(7):
            all_desiderata[d].append(mapped[d])
    return [float(np.mean(d)) if d else 0.0 for d in all_desiderata]


def compute_calibration_weight(self_per_d, rubric_per_d, alpha):
    """Weight = 1 + alpha * mean(overestimation). Higher when model has blind spots."""
    overest = [max(0.0, s - r) for s, r in zip(self_per_d, rubric_per_d)]
    return 1.0 + alpha * np.mean(overest)


# ============================================================
# Main (bestversion-scale architecture)
# ============================================================

def main(config: Config):
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path, wandb_project=None, wandb_name=None,
        config=config, do_configure_logging_module=True)

    is_eval = config.run_eval
    os.makedirs(config.log_path, exist_ok=True)

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    if config.ml_data:
        data = load_dataset("facebook/research-plan-gen", "ml")
    elif config.arxiv_data:
        data = load_dataset("facebook/research-plan-gen", "arxiv")
    elif config.pubmed_data:
        data = load_dataset("facebook/research-plan-gen", "pubmed")
    assert isinstance(data, datasets.DatasetDict)
    dataset = data["test"] if is_eval else data["train"]
    n_train_batches = len(dataset) // config.batch_size

    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile)

    # Resume logic (same as bestversion)
    last_checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path)
    if (config.eval_epoch == 0 and is_eval) or (not is_eval and last_checkpoint is None):
        resume_info = False
    elif config.eval_epoch > 0 and is_eval:
        checkpoints = checkpoint_utils.load_checkpoints_file(config.log_path)
        ckpts = [c for c in checkpoints if "state_path" in c]
        resume_info = ckpts[config.eval_epoch - 1] if ckpts else False
    else:
        resume_info = last_checkpoint

    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"])
        actual_batch = resume_info["batch"] + 1
        start_batch = actual_batch % n_train_batches
        logger.info(f"Resuming from batch {resume_info['batch']}, actual_batch={actual_batch}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank)
        start_batch = 0
        actual_batch = 0

    # Grader uses same model (self-improvement)
    grader_client = service_client.create_sampling_client(base_model=config.model_name)

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens, stop=renderer.get_stop_sequences(),
        temperature=config.temperature)
    grader_params = tinker.types.SamplingParams(
        max_tokens=config.grader_max_tokens, stop=renderer.get_stop_sequences(),
        temperature=config.grader_temperature)
    self_eval_params = tinker.types.SamplingParams(
        max_tokens=2048, stop=renderer.get_stop_sequences(),
        temperature=config.grader_temperature)
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)

    if is_eval:
        sr = training_client.save_weights_for_sampler(name=f"Eval_{actual_batch-1}").result()
        sampling_client = service_client.create_sampling_client(model_path=sr.path)

    logger.info(f"{'EVAL' if is_eval else 'TRAIN'}: {n_train_batches} batches, "
                f"batch={config.batch_size}, G={config.group_size}, w_cal={config.w_calibration}")

    # ============================================================
    # Main loop (bestversion architecture)
    # ============================================================
    for batch_idx in range(start_batch, n_train_batches):
        t_batch = time.time()
        real_batch = actual_batch + (batch_idx - start_batch) if not is_eval else actual_batch

        if not is_eval and config.save_every > 0 and batch_idx > 0 and batch_idx % config.save_every == 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{real_batch:06d}_{config.today_date}",
                log_path=config.log_path, kind="state",
                loop_state={"batch": real_batch})

        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(dataset))
        batch_rows = dataset.select(range(batch_start, batch_end))

        if not is_eval:
            sr = training_client.save_weights_for_sampler(name=f"{real_batch:06d}").result()
            sampling_client = service_client.create_sampling_client(model_path=sr.path)

        logger.info(f"Batch {real_batch}: {len(batch_rows)} goals × G={config.group_size}")

        # --- PHASE 1: LAUNCH ALL GENERATIONS ---
        policy_futures = []
        policy_prompts_tokens = []
        for goal in batch_rows["Goal"]:
            prompt_text = build_research_plan_prompt(scenario=goal)
            model_input = renderer.build_generation_prompt(
                [{"role": "user", "content": prompt_text}])
            policy_prompts_tokens.append(model_input.to_ints())
            policy_futures.append(sampling_client.sample(
                prompt=model_input, num_samples=config.group_size,
                sampling_params=sampling_params))

        # --- PHASE 2: COLLECT PLANS, LAUNCH DUAL EVALUATION ---
        batch_groups = []
        all_self_futures = []
        all_rubric_futures = []

        for i, pf in enumerate(policy_futures):
            result = pf.result()
            goal = batch_rows["Goal"][i]
            rubric = batch_rows["Rubric"][i]
            ref_sol = batch_rows["Reference solution"][i]

            group_self_f, group_rubric_f, group_samples = [], [], []

            for seq in result.sequences:
                plan = renderers.get_text_content(renderer.parse_response(seq.tokens)[0])
                if "<solution>" in plan and "</solution>" not in plan:
                    plan = plan.rstrip() + "\n</solution>"

                group_samples.append({
                    "tokens": seq.tokens, "logprobs": seq.logprobs, "text": plan})

                # Self-eval (blind) — uses base model to ensure stable XML output
                se = renderer.build_generation_prompt(
                    [{"role": "user", "content": build_blind_self_eval_prompt(goal, plan)}])
                group_self_f.append(grader_client.sample(se, num_samples=1, sampling_params=self_eval_params))

                # Rubric-eval
                gr = renderer.build_generation_prompt(
                    [{"role": "user", "content": build_grader_prompt(
                        scenario=goal, rubric_items=rubric,
                        proposed_plan=plan, reference_solution=ref_sol)}])
                group_rubric_f.append(grader_client.sample(gr, num_samples=1, sampling_params=grader_params))

            batch_groups.append({
                "prompt_tokens": policy_prompts_tokens[i],
                "samples": group_samples})
            all_self_futures.append(group_self_f)
            all_rubric_futures.append(group_rubric_f)

        # --- PHASE 3: COLLECT GRADES, COMPUTE ADVANTAGES, UPDATE ---
        training_datums = []
        b_rubric, b_self, b_cal, b_overest = [], [], [], []
        b_rewards, b_advantages, b_wc = [], [], []
        batch_logs = []
        dropped, num_total, num_valid = 0, 0, 0

        for gi in range(len(batch_groups)):
            group = batch_groups[gi]
            group_rewards, valid_samples = [], []

            for j in range(len(group["samples"])):
                num_total += 1
                plan = group["samples"][j]["text"]
                wc = len(plan.split())

                if wc < config.min_words or plan.strip() in {"<think>", "</think>", "<solution>", "</solution>"}:
                    dropped += 1
                    continue

                # Rubric eval
                rr = all_rubric_futures[gi][j].result()
                r_xml = renderers.get_text_content(renderer.parse_response(rr.sequences[0].tokens)[0])
                rubric_score = compute_rubric_reward_from_xml(r_xml)
                rubric_per_d = extract_rubric_per_desideratum(r_xml)

                # Self eval
                se_result = all_self_futures[gi][j].result()
                s_xml = renderers.get_text_content(renderer.parse_response(se_result.sequences[0].tokens)[0])
                self_per_d = parse_self_eval_scores(s_xml)
                self_score = float(np.mean(self_per_d))

                # Reward (same formula as bestversion)
                reward, wc = compute_reward(plan, rubric_score, config)

                # Calibration
                cal_err = float(np.mean([abs(s - r) for s, r in zip(self_per_d, rubric_per_d)]))
                overest = float(np.mean([max(0, s - r) for s, r in zip(self_per_d, rubric_per_d)]))
                cal_w = compute_calibration_weight(self_per_d, rubric_per_d, config.w_calibration)

                group_rewards.append(reward)
                valid_samples.append({
                    "info": group["samples"][j], "reward": reward,
                    "cal_weight": cal_w})
                num_valid += 1

                b_rubric.append(rubric_score)
                b_self.append(self_score)
                b_cal.append(cal_err)
                b_overest.append(overest)
                b_rewards.append(reward)
                b_wc.append(wc)

                batch_logs.append({
                    "batch_idx": real_batch, "group_idx": gi, "sample_idx": j,
                    "rubric_score": rubric_score, "self_score": self_score,
                    "cal_error": cal_err, "overest": overest,
                    "final_reward": reward, "word_count": wc})

            # GRPO
            if not group_rewards or len(valid_samples) < 2:
                continue
            mean_r = np.mean(group_rewards)
            advs = [r - mean_r for r in group_rewards]
            b_advantages.extend(advs)

            if all(a == 0.0 for a in advs):
                continue

            if not is_eval:
                pt = [int(t) for t in group["prompt_tokens"]]
                for k, vs in enumerate(valid_samples):
                    si = vs["info"]
                    w_adv = vs["cal_weight"] * advs[k]
                    datum = create_training_datum(
                        prompt_tokens=pt,
                        generated_tokens=[int(t) for t in si["tokens"]],
                        logprobs=si["logprobs"],
                        advantages=w_adv,
                    )
                    training_datums.append(datum)

        # Log
        summary = {
            "batch_idx": real_batch,
            "rubric/sample_mean_all": float(np.mean(b_rubric)) if b_rubric else 0.0,
            "rubric/sample_std_all": float(np.std(b_rubric)) if b_rubric else 0.0,
            "self_eval/mean": float(np.mean(b_self)) if b_self else 0.0,
            "calibration/error": float(np.mean(b_cal)) if b_cal else 0.0,
            "calibration/overest": float(np.mean(b_overest)) if b_overest else 0.0,
            "reward/sample_mean_valid": float(np.mean(b_rewards)) if b_rewards else 0.0,
            "advantage_mean": float(np.mean(b_advantages)) if b_advantages else 0.0,
            "length_mean": float(np.mean(b_wc)) if b_wc else 0.0,
            "samples/generated": num_total, "samples/valid": num_valid,
        }

        subdir = "evaluation" if is_eval else "train"
        os.makedirs(os.path.join(config.log_path, subdir), exist_ok=True)
        with open(os.path.join(config.log_path, subdir, "batch_summary.jsonl"), "a") as f:
            f.write(json.dumps(summary) + "\n")
        if batch_logs:
            with open(os.path.join(config.log_path, subdir, "training_logs.jsonl"), "a") as f:
                for item in batch_logs:
                    f.write(json.dumps(item) + "\n")

        if is_eval:
            logger.info(f"EVAL {real_batch}: rubric={summary['rubric/sample_mean_all']:.3f}")
            continue

        if not training_datums:
            logger.warning("No datums. Skipping.")
            continue

        try:
            training_client.forward_backward(
                training_datums, loss_fn="ppo",
                loss_fn_config={"clip_low_threshold": 1-config.clip_eps,
                               "clip_high_threshold": 1+config.clip_eps}).result()
            training_client.optim_step(adam_params).result()
        except Exception:
            logger.exception("Training failed")
            continue

        logger.info(
            f"Batch {real_batch}: rubric={summary['rubric/sample_mean_all']:.3f}, "
            f"self={summary['self_eval/mean']:.3f}, cal_err={summary['calibration/error']:.3f}, "
            f"datums={len(training_datums)}, t={time.time()-t_batch:.0f}s")

        ml_logger.log_metrics({
            "progress/batch": real_batch,
            "rubric/mean": summary["rubric/sample_mean_all"],
            "self_eval/mean": summary["self_eval/mean"],
            "calibration/error": summary["calibration/error"],
            "calibration/overest": summary["calibration/overest"],
        }, step=real_batch)

    if not is_eval:
        checkpoint_utils.save_checkpoint(
            training_client=training_client,
            name=f"{real_batch:06d}_final_{config.today_date}",
            log_path=config.log_path, kind="state",
            loop_state={"batch": real_batch})


if __name__ == "__main__":
    chz.nested_entrypoint(main)
