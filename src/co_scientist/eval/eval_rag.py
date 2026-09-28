"""
RAG evaluation script for co-scientist research plan models.

For each test goal:
  1. Retrieve k nearest training goals by TF-IDF cosine similarity
  2. Use their reference solutions as few-shot examples in the generation prompt
  3. Generate plans from bestversion checkpoint
  4. Grade with the canonical grader
  5. Compare to bestversion baseline (0.693 rubric without RAG)

Usage examples:
  python eval_rag.py checkpoint_run_path=runs/2026/2/withA1,A2/2\\(ml\\)

  python eval_rag.py checkpoint_run_path=runs/2026/2/withA1,A2/2\\(ml\\) rag_k=5

  python eval_rag.py checkpoint_run_path=none checkpoint_batch=0
"""

import logging
import math
import time
import re
import numpy as np
from collections import Counter
import json
import os
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
import datasets
import tinker
from co_scientist.shared.api_profiles import create_service_client
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from datasets import load_dataset

from co_scientist.shared.eval_core import (
    build_research_plan_prompt,
    build_grader_prompt,
    compute_rubric_reward_from_xml,
    check_format_compliance,
    resolve_checkpoint,
)


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


# ============================================================
# TF-IDF helpers  (numpy only, same pattern as rubric_predictability_gate.py)
# ============================================================

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_tfidf_matrix(
    documents: list[str],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Build a TF-IDF matrix from a list of documents.
    Returns (matrix [n_docs x vocab_size], vocabulary list, idf vector).
    """
    doc_tokens = [_tokenize(doc) for doc in documents]

    # Build vocabulary (document frequency)
    vocab_counter: Counter = Counter()
    for tokens in doc_tokens:
        vocab_counter.update(set(tokens))

    # Filter: keep tokens in >= 2 docs and <= 90% of docs
    n_docs = len(documents)
    min_df = 2
    max_df = int(0.9 * n_docs)
    vocab = sorted(
        w for w, c in vocab_counter.items() if min_df <= c <= max_df
    )
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    # IDF
    idf = np.zeros(len(vocab), dtype=np.float32)
    for w in vocab:
        df = vocab_counter[w]
        idf[word_to_idx[w]] = math.log((n_docs + 1) / (df + 1)) + 1  # smooth IDF

    # TF-IDF matrix
    matrix = np.zeros((n_docs, len(vocab)), dtype=np.float32)
    for doc_idx, tokens in enumerate(doc_tokens):
        if not tokens:
            continue
        tf_counter = Counter(tokens)
        for w, count in tf_counter.items():
            if w in word_to_idx:
                tf = count / len(tokens)
                matrix[doc_idx, word_to_idx[w]] = tf * idf[word_to_idx[w]]

    # L2 normalize rows
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    return matrix, vocab, idf


def tfidf_transform_single(
    text: str,
    vocab: list[str],
    idf: np.ndarray,
) -> np.ndarray:
    """Transform a single query text into a TF-IDF vector using existing vocab/idf."""
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    tokens = _tokenize(text)
    vec = np.zeros(len(vocab), dtype=np.float32)
    if not tokens:
        return vec
    tf_counter = Counter(tokens)
    for w, count in tf_counter.items():
        if w in word_to_idx:
            tf = count / len(tokens)
            vec[word_to_idx[w]] = tf * idf[word_to_idx[w]]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def retrieve_nearest(
    query_vec: np.ndarray,
    train_matrix: np.ndarray,
    k: int,
) -> np.ndarray:
    """Return indices of k nearest training docs by cosine similarity."""
    similarities = train_matrix @ query_vec  # both L2-normalized
    indices = np.argsort(-similarities)[:k]
    return indices


# ============================================================
# Config
# ============================================================

@chz.chz
class Config:
    # ---- Checkpoint selection ----
    checkpoint_run_path: str = "runs/2026/2/withA1,A2/2(ml)"
    checkpoint_batch: int = -1
    eval_output_path: str = ""

    # ---- Model / API ----
    base_url: str | None = None
    api_profile: str | None = None
    model_name: str = "Qwen/Qwen3-30B-A3B"
    grader_model_name: str = "Qwen/Qwen3-30B-A3B"

    # ---- Dataset ----
    ml_data: bool = True

    # ---- Sampling ----
    batch_size: int = 64
    group_size: int = 8
    max_tokens: int = 2048
    grader_max_tokens: int = 8192
    temperature: float = 1.0
    grader_temperature: float = 0.0

    # ---- Reward / format ----
    max_word_count: int = 750
    target_word_count: int = 600
    scale_length_bonus: float = 120.0
    scaling_factor: float = 0.08
    min_words: int = 30

    # ---- LoRA ----
    lora_rank: int = 64

    # ---- RAG parameters ----
    rag_k: int = 3
    rag_max_words: int = 500


# ============================================================
# Main
# ============================================================

def main(config: Config):
    # --- Resolve output path ---
    if config.eval_output_path:
        output_dir = config.eval_output_path
    elif config.checkpoint_run_path.lower() != "none":
        output_dir = os.path.join(config.checkpoint_run_path, "eval_rag")
    else:
        output_dir = os.path.join(os.getcwd(), "eval_rag_base_model")
    os.makedirs(output_dir, exist_ok=True)

    ml_logger = ml_log.setup_logging(
        log_dir=output_dir,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    # --- Load dataset (both train and test splits) ---
    logger.info("Loading dataset (train + test splits)...")
    if config.ml_data:
        data = load_dataset("facebook/research-plan-gen", "ml")
    assert isinstance(data, datasets.DatasetDict)
    train_dataset = data["train"]
    test_dataset = data["test"]

    logger.info(
        f"Train set: {len(train_dataset)} examples | "
        f"Test set: {len(test_dataset)} examples"
    )

    # --- Build TF-IDF index on training goals (once) ---
    logger.info("Building TF-IDF index on training goals...")
    t_index = time.time()
    train_goals = train_dataset["Goal"]
    train_refs = train_dataset["Reference solution"]
    tfidf_matrix, vocab, idf_vector = build_tfidf_matrix(train_goals)
    logger.info(
        f"TF-IDF index built: {tfidf_matrix.shape[0]} docs, "
        f"{tfidf_matrix.shape[1]} vocab terms, "
        f"took {time.time() - t_index:.1f}s"
    )

    n_eval_batches = len(test_dataset) // config.batch_size
    logger.info(
        f"Test set: {len(test_dataset)} examples → "
        f"{n_eval_batches} batches of {config.batch_size}"
    )

    # --- Tokenizer / renderer ---
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # --- Service client ---
    service_client = create_service_client(
        base_url=config.base_url,
        api_profile=config.api_profile,
    )

    # --- Checkpoint ---
    resume_info, batch_label = resolve_checkpoint(
        config.checkpoint_run_path, config.checkpoint_batch
    )

    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
        loaded_batch = resume_info["batch"]
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        loaded_batch = -1

    # --- Prepare fixed sampler (weights frozen for entire eval) ---
    logger.info(f"Saving weights for eval sampler (checkpoint=b{batch_label})...")
    sampling_result = training_client.save_weights_for_sampler(
        name=f"eval_rag_b{batch_label}"
    ).result()
    sampling_client = service_client.create_sampling_client(
        model_path=sampling_result.path
    )

    grader_client = service_client.create_sampling_client(
        base_model=config.grader_model_name
    )

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )

    # --- Per-eval-run output files ---
    logs_path = os.path.join(output_dir, f"eval_b{batch_label}_logs.jsonl")
    summary_path = os.path.join(output_dir, f"eval_b{batch_label}_summary.jsonl")

    logger.info(
        f"Starting RAG eval | checkpoint=b{batch_label} | rag_k={config.rag_k} | "
        f"{n_eval_batches} batches | output → {output_dir}"
    )

    all_rubric_scores: list[float] = []
    all_rewards: list[float] = []
    all_format_penalties: list[float] = []

    # ============================================================
    # Eval loop
    # ============================================================
    for batch_idx in range(n_eval_batches):
        t_start = time.time()
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(test_dataset))
        batch_rows = test_dataset.select(range(batch_start, batch_end))

        logger.info(
            f"Eval batch {batch_idx + 1}/{n_eval_batches} "
            f"(goals {batch_start}–{batch_end - 1})"
        )

        # --- Phase 1: retrieve examples & launch policy generations ---
        policy_futures = []
        batch_retrieved_indices = []  # one per goal in this batch

        for goal in batch_rows["Goal"]:
            # Retrieve k nearest training goals
            query_vec = tfidf_transform_single(goal, vocab, idf_vector)
            nearest_indices = retrieve_nearest(query_vec, tfidf_matrix, config.rag_k)
            batch_retrieved_indices.append(nearest_indices)

            # Build few-shot examples from retrieved training data
            examples = []
            for idx in nearest_indices:
                ref_solution = train_refs[idx]
                words = ref_solution.split()
                if len(words) > config.rag_max_words:
                    ref_solution = " ".join(words[:config.rag_max_words]) + "..."
                examples.append({
                    "scenario": train_goals[idx],
                    "solution": ref_solution,
                })

            prompt_text = build_research_plan_prompt(
                scenario=goal, examples=examples
            )
            convo = [{"role": "user", "content": prompt_text}]
            model_input = renderer.build_generation_prompt(convo)
            policy_futures.append(
                sampling_client.sample(
                    prompt=model_input,
                    num_samples=config.group_size,
                    sampling_params=sampling_params,
                )
            )

        # --- Phase 2: collect plans & launch graders ---
        batch_groups_data = []
        all_grader_futures = []

        for i, p_future in enumerate(policy_futures):
            result = p_future.result()
            goal = batch_rows["Goal"][i]
            rubric = batch_rows["Rubric"][i]
            ref_sol = batch_rows["Reference solution"][i]

            group_grader_futures = []
            group_samples_info = []

            for group_result in result.sequences:
                proposed_plan = renderers.get_text_content(
                    renderer.parse_response(group_result.tokens)[0]
                )
                if "<solution>" in proposed_plan and "</solution>" not in proposed_plan:
                    proposed_plan = proposed_plan.rstrip() + "\n</solution>"

                grader_prompt_text = build_grader_prompt(
                    scenario=goal,
                    rubric_items=rubric,
                    proposed_plan=proposed_plan,
                    reference_solution=ref_sol,
                )
                grader_input = renderer.build_generation_prompt(
                    [{"role": "user", "content": grader_prompt_text}]
                )
                g_future = grader_client.sample(
                    grader_input,
                    num_samples=1,
                    sampling_params=tinker.types.SamplingParams(
                        max_tokens=config.grader_max_tokens,
                        temperature=config.grader_temperature,
                    ),
                )
                group_grader_futures.append(g_future)
                group_samples_info.append({"text": proposed_plan})

            batch_groups_data.append({
                "goal": goal,
                "samples_info": group_samples_info,
            })
            all_grader_futures.append(group_grader_futures)

        # --- Phase 3: collect grades & compute metrics ---
        batch_rubric_scores: list[float] = []
        batch_rewards: list[float] = []
        batch_format_penalties: list[float] = []
        batch_logs: list[dict] = []
        dropped = 0
        num_total = 0
        num_valid = 0

        for group_idx, group_futures in enumerate(all_grader_futures):
            group_data = batch_groups_data[group_idx]

            for j, g_future in enumerate(group_futures):
                num_total += 1
                g_result = g_future.result()
                parsed_msg, _ = renderer.parse_response(g_result.sequences[0].tokens)
                xml_text = renderers.get_text_content(parsed_msg)

                rubric_score = compute_rubric_reward_from_xml(xml_text)
                plan_text = group_data["samples_info"][j]["text"]
                word_count = len(plan_text.strip().split())

                is_compliant = check_format_compliance(plan_text, config.max_word_count)
                excess = max(0, word_count - config.max_word_count)
                format_penalty = 0.0 if is_compliant else 0.2 + 0.0005 * excess
                length_bonus = np.exp(
                    -((word_count - config.target_word_count) / config.scale_length_bonus) ** 2
                )
                final_reward = rubric_score + config.scaling_factor * length_bonus - format_penalty

                batch_rubric_scores.append(rubric_score)
                batch_format_penalties.append(format_penalty)

                if word_count < config.min_words:
                    dropped += 1
                    continue
                if plan_text.strip() in {"<think>", "</think>", "<solution>", "</solution>"}:
                    dropped += 1
                    continue

                batch_rewards.append(final_reward)
                num_valid += 1

                batch_logs.append({
                    "eval_batch_idx": batch_idx,
                    "checkpoint_batch": batch_label,
                    "group_idx": group_idx,
                    "sample_idx": j,
                    "goal": group_data["goal"],
                    "policy_output": plan_text,
                    "grader_output": xml_text,
                    "rubric_score": rubric_score,
                    "format_penalty": format_penalty,
                    "final_reward": final_reward,
                    "word_count": word_count,
                    "is_compliant": is_compliant,
                    # RAG-specific
                    "rag_k": config.rag_k,
                    "retrieved_indices": batch_retrieved_indices[group_idx].tolist(),
                })

        # Accumulate across batches
        all_rubric_scores.extend(batch_rubric_scores)
        all_rewards.extend(batch_rewards)
        all_format_penalties.extend(batch_format_penalties)

        # Per-batch summary
        batch_summary = {
            "eval_batch_idx": batch_idx,
            "checkpoint_batch": batch_label,
            "rubric/mean": float(np.mean(batch_rubric_scores)) if batch_rubric_scores else 0.0,
            "rubric/std": float(np.std(batch_rubric_scores)) if batch_rubric_scores else 0.0,
            "reward/mean": float(np.mean(batch_rewards)) if batch_rewards else 0.0,
            "reward/std": float(np.std(batch_rewards)) if batch_rewards else 0.0,
            "format_penalty/mean": float(np.mean(batch_format_penalties)) if batch_format_penalties else 0.0,
            "samples/total": num_total,
            "samples/valid": num_valid,
            "samples/dropped": dropped,
            "time_s": round(time.time() - t_start, 1),
        }
        logger.info(
            f"  rubric={batch_summary['rubric/mean']:.4f}  "
            f"reward={batch_summary['reward/mean']:.4f}  "
            f"format_pen={batch_summary['format_penalty/mean']:.4f}  "
            f"valid={num_valid}/{num_total}"
        )

        # Write batch logs
        with open(logs_path, "a") as f:
            for item in batch_logs:
                f.write(json.dumps(item) + "\n")

        with open(summary_path, "a") as f:
            f.write(json.dumps(batch_summary) + "\n")

    # --- Final aggregate summary ---
    aggregate = {
        "checkpoint_batch": batch_label,
        "loaded_batch": loaded_batch,
        "n_eval_batches": n_eval_batches,
        "n_samples_total": len(all_rubric_scores),
        "n_samples_valid": len(all_rewards),
        "rubric/mean": float(np.mean(all_rubric_scores)) if all_rubric_scores else 0.0,
        "rubric/std": float(np.std(all_rubric_scores)) if all_rubric_scores else 0.0,
        "reward/mean": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "reward/std": float(np.std(all_rewards)) if all_rewards else 0.0,
        "format_penalty/mean": float(np.mean(all_format_penalties)) if all_format_penalties else 0.0,
        "rag_k": config.rag_k,
        "rag_max_words": config.rag_max_words,
        "comparison": {
            "bestversion_rubric": 0.693,
            "reference_rubric": 0.867,
        },
    }

    aggregate_path = os.path.join(output_dir, f"eval_b{batch_label}_aggregate.json")
    with open(aggregate_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"RAG EVAL COMPLETE  checkpoint=b{batch_label}  rag_k={config.rag_k}")
    logger.info(f"  rubric mean : {aggregate['rubric/mean']:.4f}")
    logger.info(f"  reward mean : {aggregate['reward/mean']:.4f}")
    logger.info(f"  format pen  : {aggregate['format_penalty/mean']:.4f}")
    logger.info(f"  valid/total : {aggregate['n_samples_valid']}/{aggregate['n_samples_total']}")
    logger.info(f"  comparison  : bestversion={aggregate['comparison']['bestversion_rubric']:.3f}  reference={aggregate['comparison']['reference_rubric']:.3f}")
    logger.info(f"  results     : {output_dir}")
    logger.info("=" * 60)

    ml_logger.close()


if __name__ == "__main__":
    chz.nested_entrypoint(main)
