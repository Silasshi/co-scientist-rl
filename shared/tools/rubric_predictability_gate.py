"""
Rubric predictability gate experiment for RC-GRPO.

Tests whether rubric items are predictable from goal text using:
  Test 1: Retrieval overlap (kNN on TF-IDF goal vectors, ROUGE-L of retrieved rubrics)
  Test 2: Rubric item clustering (K-means on TF-IDF rubric item vectors)
  Test 3: Rubric vocabulary statistics

Uses numpy + scipy only (no sklearn/sentence-transformers dependency).

Usage:
  python tools/rubric_predictability_gate.py --output gate_report.json
"""

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from datasets import load_dataset


# ============================================================
# TF-IDF implementation (no sklearn dependency)
# ============================================================

def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_tfidf_matrix(documents: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Build a TF-IDF matrix from a list of documents.
    Returns (matrix [n_docs x vocab_size], vocabulary list).
    """
    # Tokenize all documents
    doc_tokens = [tokenize(doc) for doc in documents]

    # Build vocabulary
    vocab_counter: Counter = Counter()
    for tokens in doc_tokens:
        vocab_counter.update(set(tokens))  # document frequency

    # Filter: keep tokens that appear in >= 2 docs and <= 90% of docs
    n_docs = len(documents)
    min_df = 2
    max_df = int(0.9 * n_docs)
    vocab = sorted(
        w for w, c in vocab_counter.items() if min_df <= c <= max_df
    )
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    # Build TF-IDF
    idf = np.zeros(len(vocab))
    for w in vocab:
        df = vocab_counter[w]
        idf[word_to_idx[w]] = math.log((n_docs + 1) / (df + 1)) + 1  # smooth IDF

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

    return matrix, vocab


# ============================================================
# kNN helpers
# ============================================================

def knn_indices(query_vecs: np.ndarray, index_vecs: np.ndarray, k: int) -> np.ndarray:
    """
    Find k nearest neighbors using cosine distance.
    Returns array of shape [n_queries, k] with indices into index_vecs.
    """
    # Process in chunks to avoid memory issues
    chunk_size = 200
    n_queries = query_vecs.shape[0]
    all_indices = np.zeros((n_queries, k), dtype=np.int64)

    for start in range(0, n_queries, chunk_size):
        end = min(start + chunk_size, n_queries)
        dists = cdist(query_vecs[start:end], index_vecs, metric="cosine")
        all_indices[start:end] = np.argsort(dists, axis=1)[:, :k]

    return all_indices


# ============================================================
# ROUGE-L implementation
# ============================================================

def lcs_length(x: list[str], y: list[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    # Use space-optimized LCS
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 score."""
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens) if hyp_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ============================================================
# K-means implementation (no sklearn dependency)
# ============================================================

def kmeans(X: np.ndarray, k: int, max_iter: int = 100, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple K-means clustering.
    Returns (labels, centroids).
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]

    # Initialize centroids with k-means++
    centroids = np.zeros((k, X.shape[1]), dtype=np.float32)
    centroids[0] = X[rng.randint(n)]
    for c in range(1, k):
        dists = cdist(X, centroids[:c], metric="euclidean").min(axis=1)
        probs = dists ** 2
        probs /= probs.sum()
        centroids[c] = X[rng.choice(n, p=probs)]

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        # Assign (chunked to save memory)
        chunk_size = 500
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            dists = cdist(X[start:end], centroids, metric="euclidean")
            labels[start:end] = np.argmin(dists, axis=1)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for c_idx in range(k):
            members = X[labels == c_idx]
            if len(members) > 0:
                new_centroids[c_idx] = members.mean(axis=0)
            else:
                new_centroids[c_idx] = X[rng.randint(n)]

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return labels, centroids


# ============================================================
# Test 1: Retrieval Overlap
# ============================================================

def test_retrieval_overlap(goals: list[str], rubrics: list[list[str]], goal_matrix, n_folds: int = 5, k: int = 5) -> dict:
    """
    5-fold cross-validation: for each held-out goal, find k nearest training goals,
    concatenate their rubric items, compute ROUGE-L against actual rubric items.
    """
    print(f"\n{'='*60}")
    print("TEST 1: Retrieval Overlap (kNN on goal embeddings)")
    print(f"{'='*60}")

    n = len(goals)

    # Prepare rubric texts
    rubric_texts = [" ".join(items) for items in rubrics]

    # Create fold indices
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    fold_size = n // n_folds

    rouge_scores = []
    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        train_vecs = goal_matrix[train_idx]
        test_vecs = goal_matrix[test_idx]

        # Find k nearest neighbors
        nn_idx = knn_indices(test_vecs, train_vecs, k)

        fold_scores = []
        for i, test_i in enumerate(test_idx):
            # Get neighbor rubrics
            neighbor_global_idx = train_idx[nn_idx[i]]
            retrieved = " ".join(rubric_texts[j] for j in neighbor_global_idx)
            actual = rubric_texts[test_i]

            score = rouge_l_f1(actual, retrieved)
            fold_scores.append(score)

        fold_mean = np.mean(fold_scores)
        print(f"  Fold {fold+1}/{n_folds}: ROUGE-L = {fold_mean:.4f} ({len(test_idx)} test goals)")
        rouge_scores.extend(fold_scores)

    mean_rouge = float(np.mean(rouge_scores))
    std_rouge = float(np.std(rouge_scores))
    passed = mean_rouge > 0.3

    print(f"\n  Mean ROUGE-L: {mean_rouge:.4f} +/- {std_rouge:.4f}")
    print(f"  Threshold: > 0.3")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        "test": "retrieval_overlap",
        "mean_rouge_l": round(mean_rouge, 4),
        "std_rouge_l": round(std_rouge, 4),
        "n_goals": n,
        "n_folds": n_folds,
        "k_neighbors": k,
        "threshold": 0.3,
        "passed": passed,
    }


# ============================================================
# Test 2: Rubric Item Clustering
# ============================================================

def test_rubric_clustering(
    goals: list[str],
    rubrics: list[list[str]],
    goal_matrix,
    all_items: list[str],
    item_goal_map: list[int],
    n_item_clusters: int = 30,
    n_goal_clusters: int = 20,
) -> dict:
    """
    Cluster rubric items and goals, report concentration and within-goal-cluster overlap.
    """
    print(f"\n{'='*60}")
    print("TEST 2: Rubric Item Clustering")
    print(f"{'='*60}")

    print(f"  Total rubric items: {len(all_items)}")
    print(f"  Unique rubric items (exact match): {len(set(all_items))}")

    # Build TF-IDF for rubric items
    print(f"  Building TF-IDF for rubric items...")
    t0 = time.time()
    item_matrix, item_vocab = build_tfidf_matrix(all_items)
    print(f"  TF-IDF built: {item_matrix.shape} ({time.time()-t0:.1f}s)")

    # K-means on rubric items
    print(f"  Clustering {len(all_items)} items into {n_item_clusters} clusters...")
    t0 = time.time()
    item_labels, _ = kmeans(item_matrix, n_item_clusters)
    print(f"  Clustering done ({time.time()-t0:.1f}s)")

    cluster_sizes = Counter(int(l) for l in item_labels)
    sorted_sizes = sorted(cluster_sizes.values(), reverse=True)
    top_10_count = sum(sorted_sizes[:10])
    concentration = top_10_count / len(all_items)

    print(f"  Top-10 cluster concentration: {concentration:.2%} ({top_10_count}/{len(all_items)})")
    print(f"  Largest clusters: {sorted_sizes[:5]}")

    # Goal clustering and within-cluster rubric overlap
    print(f"\n  Clustering goals into {n_goal_clusters} clusters...")
    goal_labels, _ = kmeans(goal_matrix, n_goal_clusters)

    # Measure rubric overlap within goal clusters
    within_cluster_overlaps = []
    for c in range(n_goal_clusters):
        cluster_goal_indices = [i for i, l in enumerate(goal_labels) if l == c]
        if len(cluster_goal_indices) < 2:
            continue

        # Collect rubric items for each goal in this cluster
        cluster_rubric_sets = []
        for gi in cluster_goal_indices:
            item_tokens = set()
            for item in rubrics[gi]:
                item_tokens.update(tokenize(item))
            cluster_rubric_sets.append(item_tokens)

        # Pairwise Jaccard similarity
        n_goals_in_cluster = len(cluster_rubric_sets)
        pairwise_sims = []
        for i in range(n_goals_in_cluster):
            for j in range(i + 1, n_goals_in_cluster):
                a, b = cluster_rubric_sets[i], cluster_rubric_sets[j]
                if a or b:
                    jaccard = len(a & b) / len(a | b) if (a | b) else 0.0
                    pairwise_sims.append(jaccard)

        if pairwise_sims:
            within_cluster_overlaps.append(float(np.mean(pairwise_sims)))

    mean_within_overlap = float(np.mean(within_cluster_overlaps)) if within_cluster_overlaps else 0.0
    print(f"  Mean within-cluster rubric overlap (Jaccard): {mean_within_overlap:.4f}")

    concentration_passed = concentration > 0.6
    print(f"\n  Concentration threshold: > 60%")
    print(f"  Result: {'PASS' if concentration_passed else 'FAIL'}")

    return {
        "test": "rubric_clustering",
        "total_items": len(all_items),
        "unique_items_exact": len(set(all_items)),
        "n_item_clusters": n_item_clusters,
        "top_10_concentration": round(concentration, 4),
        "concentration_threshold": 0.6,
        "n_goal_clusters": n_goal_clusters,
        "mean_within_cluster_rubric_overlap": round(mean_within_overlap, 4),
        "passed": concentration_passed,
    }


# ============================================================
# Test 3: Rubric Vocabulary Statistics
# ============================================================

def test_vocabulary_stats(rubrics: list[list[str]], all_items: list[str]) -> dict:
    """
    Count unique rubric items, common patterns, frequent keywords.
    """
    print(f"\n{'='*60}")
    print("TEST 3: Rubric Vocabulary Statistics")
    print(f"{'='*60}")

    total = len(all_items)
    unique_exact = len(set(all_items))

    # Items per goal distribution
    items_per_goal = [len(items) for items in rubrics]
    mean_items = float(np.mean(items_per_goal))
    std_items = float(np.std(items_per_goal))

    print(f"  Total rubric items: {total}")
    print(f"  Unique items (exact match): {unique_exact}")
    print(f"  Items per goal: {mean_items:.1f} +/- {std_items:.1f}")

    # Keyword frequency analysis
    keywords_of_interest = [
        "scalability", "ethical", "baseline", "evaluation", "data",
        "model", "experiment", "statistical", "reproducibility", "bias",
        "methodology", "analysis", "hypothesis", "sample", "control",
        "variable", "metric", "validation", "performance", "comparison",
    ]

    keyword_counts = {}
    all_text_lower = " ".join(all_items).lower()
    for kw in keywords_of_interest:
        count = all_text_lower.count(kw)
        if count > 0:
            keyword_counts[kw] = count

    # Sort by frequency
    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: -x[1])
    print(f"\n  Keyword frequencies in rubric items:")
    for kw, count in sorted_keywords[:15]:
        print(f"    {kw}: {count} ({count/total*100:.1f}% of items)")

    # Most common rubric items (exact match)
    item_counter = Counter(all_items)
    most_common = item_counter.most_common(10)
    print(f"\n  Most common rubric items (exact match):")
    for item, count in most_common:
        truncated = item[:80] + "..." if len(item) > 80 else item
        print(f"    [{count}x] {truncated}")

    # Token vocabulary size (lowercased words across all items)
    all_tokens = set()
    for item in all_items:
        all_tokens.update(tokenize(item))
    vocab_size = len(all_tokens)

    print(f"\n  Rubric token vocabulary size: {vocab_size}")
    print(f"  Ratio unique_items/total: {unique_exact/total:.4f}")

    return {
        "test": "vocabulary_stats",
        "total_items": total,
        "unique_items_exact": unique_exact,
        "unique_ratio": round(unique_exact / total, 4),
        "items_per_goal_mean": round(mean_items, 2),
        "items_per_goal_std": round(std_items, 2),
        "token_vocabulary_size": vocab_size,
        "top_keywords": dict(sorted_keywords[:15]),
        "most_common_items": [
            {"item": item[:100], "count": count} for item, count in most_common[:5]
        ],
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Rubric predictability gate experiment")
    parser.add_argument("--output", type=str, default="gate_report.json", help="Output JSON report path")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors")
    args = parser.parse_args()

    print("Loading dataset (facebook/research-plan-gen, ML split, train)...")
    data = load_dataset("facebook/research-plan-gen", "ml")
    dataset = data["train"]
    print(f"Loaded {len(dataset)} training goals")

    goals = dataset["Goal"]
    rubrics = dataset["Rubric"]

    # Pre-compute shared data
    print("Building goal TF-IDF matrix (shared across tests)...")
    t0 = time.time()
    goal_matrix, vocab = build_tfidf_matrix(goals)
    print(f"  TF-IDF built: {goal_matrix.shape} matrix, {len(vocab)} vocab terms ({time.time()-t0:.1f}s)")

    all_items = []
    item_goal_map = []
    for goal_idx, items in enumerate(rubrics):
        for item in items:
            all_items.append(item)
            item_goal_map.append(goal_idx)

    # Run all three tests
    t_total = time.time()

    result_1 = test_retrieval_overlap(goals, rubrics, goal_matrix, n_folds=args.n_folds, k=args.k)
    result_2 = test_rubric_clustering(goals, rubrics, goal_matrix, all_items, item_goal_map)
    result_3 = test_vocabulary_stats(rubrics, all_items)

    # Go/No-Go decision
    all_passed = result_1["passed"] and result_2["passed"]
    # Test 3 is informational (no hard pass/fail)

    total_time = time.time() - t_total

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_s": round(total_time, 1),
        "tests": [result_1, result_2, result_3],
        "go_decision": all_passed,
        "summary": (
            "GO: Rubric items are predictable from goal text. Proceed with RC-GRPO."
            if all_passed
            else "NO-GO: Rubric items are NOT sufficiently predictable. RC-GRPO unlikely to help."
        ),
    }

    # Write report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"GATE DECISION: {'GO' if all_passed else 'NO-GO'}")
    print(f"{'='*60}")
    print(f"  Test 1 (Retrieval ROUGE-L > 0.3): {'PASS' if result_1['passed'] else 'FAIL'} ({result_1['mean_rouge_l']:.4f})")
    print(f"  Test 2 (Cluster concentration > 60%): {'PASS' if result_2['passed'] else 'FAIL'} ({result_2['top_10_concentration']:.2%})")
    print(f"  Test 3 (Vocabulary stats): informational")
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
