"""
Extract CPR (Contrastive Plan Ranking) preference pairs from bestversion training logs.

Reads training logs, reconstructs (goal, plan, score) triples via dataset alignment,
creates pairwise preference pairs, splits by goal (not by pair), and saves to JSONL.

Usage:
  python tools/extract_cpr_pairs.py
  python tools/extract_cpr_pairs.py --epoch_filter all --score_gap_threshold 0.10
  python tools/extract_cpr_pairs.py --log_path runs/2026/2/withA1,A2/2(ml)/train/training_logs.jsonl
"""

import argparse
import json
import os
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Extract CPR preference pairs from training logs")
    parser.add_argument(
        "--log_path",
        type=str,
        default="runs/2026/2/withA1,A2/2(ml)/train/training_logs.jsonl",
        help="Path to training logs JSONL file",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="ml",
        help="Dataset split name for facebook/research-plan-gen",
    )
    parser.add_argument(
        "--epoch_filter",
        type=str,
        default="last",
        choices=["last", "all"],
        help="Which epoch(s) to include: 'last' (batches >= 107) or 'all'",
    )
    parser.add_argument(
        "--score_gap_threshold",
        type=float,
        default=0.05,
        help="Minimum score gap to create a preference pair",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size used in training (for goal reconstruction)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for A/B assignment and train/val split",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # Resolve log path relative to project root
    project_root = Path(__file__).resolve().parent.parent
    log_path = project_root / args.log_path
    output_dir = project_root / args.output_dir

    print(f"Loading training logs from: {log_path}")
    if not log_path.exists():
        raise FileNotFoundError(f"Training log not found: {log_path}")

    # Load dataset for goal text reconstruction
    print(f"Loading dataset: facebook/research-plan-gen ({args.dataset_split}) train split...")
    dataset = load_dataset("facebook/research-plan-gen", args.dataset_split)["train"]

    # Step 1: Read training logs and reconstruct triples
    print("Reading training logs...")
    entries = []
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line)
            entries.append(entry)
    print(f"  Total log entries: {len(entries)}")

    # Determine epoch boundary for filtering
    max_batch = max(e["batch_idx"] for e in entries)
    n_train_batches = len(dataset) // args.batch_size
    print(f"  Max batch_idx: {max_batch}, batches per epoch: {n_train_batches}")

    # Step 2: Epoch filtering
    if args.epoch_filter == "last":
        # Last epoch = batches >= n_train_batches (i.e., batch 107+ for 215-batch / 2-epoch run)
        epoch_start = n_train_batches
        filtered = [e for e in entries if e["batch_idx"] >= epoch_start]
        print(f"  Epoch filter: last (batch_idx >= {epoch_start})")
    else:
        filtered = entries
        print(f"  Epoch filter: all")
    print(f"  Entries after epoch filter: {len(filtered)}")

    # Step 3: Reconstruct (goal, plan, score) triples
    triples = []
    for entry in filtered:
        goal_idx = entry["batch_idx"] * args.batch_size + entry["group_idx"]
        # Wrap around for multi-epoch
        goal_idx = goal_idx % len(dataset)

        triples.append({
            "goal": dataset[goal_idx]["Goal"],
            "plan": entry["policy_output"],
            "score": entry["rubric_score"],
            "batch_idx": entry["batch_idx"],
            "group_idx": entry["group_idx"],
            "sample_idx": entry["sample_idx"],
        })

    # Step 4: Group by goal (batch_idx, group_idx)
    grouped = defaultdict(list)
    for t in triples:
        key = (t["batch_idx"], t["group_idx"])
        grouped[key].append(t)

    print(f"  Unique goals (batch_idx, group_idx): {len(grouped)}")

    # Step 5: Create preference pairs
    pairs = []
    skipped_ties = 0
    for key, samples in grouped.items():
        for s_i, s_j in combinations(samples, 2):
            score_gap = abs(s_i["score"] - s_j["score"])
            if score_gap < args.score_gap_threshold:
                skipped_ties += 1
                continue

            winner = s_i if s_i["score"] > s_j["score"] else s_j
            loser = s_j if s_i["score"] > s_j["score"] else s_i

            # Randomly assign to position A or B (prevent position bias)
            if random.random() < 0.5:
                plan_a, plan_b, label = winner["plan"], loser["plan"], "A"
            else:
                plan_a, plan_b, label = loser["plan"], winner["plan"], "B"

            pairs.append({
                "goal": s_i["goal"],
                "plan_a": plan_a,
                "plan_b": plan_b,
                "label": label,
                "score_gap": round(score_gap, 6),
                "batch_idx": key[0],
                "group_idx": key[1],
            })

    print(f"  Total pairs created: {len(pairs)}")
    print(f"  Skipped near-ties: {skipped_ties}")

    # Step 6: Train/val split by goal (not by pair)
    unique_goals = list(grouped.keys())
    random.shuffle(unique_goals)
    split_idx = int(0.8 * len(unique_goals))
    train_goals = set(map(tuple, unique_goals[:split_idx]))
    val_goals = set(map(tuple, unique_goals[split_idx:]))

    train_pairs = [p for p in pairs if (p["batch_idx"], p["group_idx"]) in train_goals]
    val_pairs = [p for p in pairs if (p["batch_idx"], p["group_idx"]) in val_goals]

    # Step 7: Save to JSONL
    os.makedirs(output_dir, exist_ok=True)
    train_path = output_dir / "cpr_pairs_train.jsonl"
    val_path = output_dir / "cpr_pairs_val.jsonl"

    # Strip internal keys before saving
    def clean_pair(p):
        return {
            "goal": p["goal"],
            "plan_a": p["plan_a"],
            "plan_b": p["plan_b"],
            "label": p["label"],
            "score_gap": p["score_gap"],
        }

    with open(train_path, "w") as f:
        for p in train_pairs:
            f.write(json.dumps(clean_pair(p)) + "\n")

    with open(val_path, "w") as f:
        for p in val_pairs:
            f.write(json.dumps(clean_pair(p)) + "\n")

    # Step 8: Summary stats
    label_counts_train = defaultdict(int)
    for p in train_pairs:
        label_counts_train[p["label"]] += 1
    label_counts_val = defaultdict(int)
    for p in val_pairs:
        label_counts_val[p["label"]] += 1

    train_gaps = [p["score_gap"] for p in train_pairs]
    val_gaps = [p["score_gap"] for p in val_pairs]

    print("\n" + "=" * 60)
    print("CPR PAIR EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Total pairs:       {len(pairs)}")
    print(f"  Train pairs:       {len(train_pairs)}  (A={label_counts_train['A']}, B={label_counts_train['B']})")
    print(f"  Val pairs:         {len(val_pairs)}  (A={label_counts_val['A']}, B={label_counts_val['B']})")
    print(f"  Train goals:       {len(train_goals)}")
    print(f"  Val goals:         {len(val_goals)}")
    if train_gaps:
        print(f"  Score gap (train): mean={sum(train_gaps)/len(train_gaps):.4f}, "
              f"min={min(train_gaps):.4f}, max={max(train_gaps):.4f}")
    if val_gaps:
        print(f"  Score gap (val):   mean={sum(val_gaps)/len(val_gaps):.4f}, "
              f"min={min(val_gaps):.4f}, max={max(val_gaps):.4f}")
    print(f"  Saved to:          {train_path}")
    print(f"                     {val_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
