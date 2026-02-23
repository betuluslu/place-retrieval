from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


def recall_at_k(
    ranked_gallery_labels: Sequence[Sequence[str]],
    true_labels: Sequence[Sequence[str]],
    k: int,
) -> float:
    """
    Computes Recall@K for retrieval.
    For each query, we check if at least ONE correct label
    appears in the top-K retrieved results.

    Args:
        ranked_gallery_labels[i]:
            Ordered predictions for query i (best -> worst)
        true_labels[i]:
            List of correct labels for query i (multi-positive)
        k:
            Top-K threshold (e.g., K=1,5,10)

    Returns:
        Recall@K score between 0 and 1
    """
    assert len(ranked_gallery_labels) == len(true_labels)
    hit = 0
    total = 0
    for preds, trues in zip(ranked_gallery_labels, true_labels):
        if not trues:
            # open-set query (no valid match in gallery) -> exclude from recall
            continue
        total += 1
        topk = preds[:k]
        if any(p in trues for p in topk):
            hit += 1
    return (hit / total) if total > 0 else 0.0


def average_precision(
    ranked_gallery_labels: Sequence[str],
    true_labels: Sequence[str],
) -> float:
    """
    Computes Average Precision (AP) for a single query.

    Mathematical idea:
    AP = sum( Precision@k * relevance(k) ) / number_of_relevant_items

    - Supports multi-positive labels
    - Returns 0 if query is open-set (no true labels)
    """

    if not true_labels:
        return 0.0

    true_set = set(true_labels)
    num_relevant = len(true_set)

    hit_count = 0
    precision_sum = 0.0

    for i, pred in enumerate(ranked_gallery_labels, start=1):
        if pred in true_set:
            hit_count += 1
            precision_at_i = hit_count / i
            precision_sum += precision_at_i

    # normalize by number of relevant items (NOT by hits!)
    return precision_sum / num_relevant


def mean_average_precision(
    ranked_gallery_labels: Sequence[Sequence[str]],
    true_labels: Sequence[Sequence[str]],
) -> float:
    """
    Computes mean Average Precision (mAP) over all VALID queries.

    - Open-set queries are EXCLUDED from mAP
    - Only queries with true matches are evaluated

    Args:
        ranked_gallery_labels:
            List of ranked predictions for each query
        true_labels:
            List of ground-truth labels for each query

    Returns:
        mAP score (float)
    """

    ap_list = []

    for preds, trues in zip(ranked_gallery_labels, true_labels):
        if not trues:  # open-set query
            continue
        ap = average_precision(preds, trues)
        ap_list.append(ap)

    if len(ap_list) == 0:
        return 0.0

    return float(sum(ap_list) / len(ap_list))