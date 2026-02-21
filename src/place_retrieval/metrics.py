from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


def recall_at_k(
    ranked_gallery_labels: Sequence[Sequence[str]],
    true_labels: Sequence[Sequence[str]],
    k: int,
) -> float:
    """
    ranked_gallery_labels[i] = top-ranked gallery labels for query i (ordered best->worst)
    true_labels[i] = list/set of acceptable labels for query i (multi-positive supported)
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
    Proper AP:
    AP = sum(P@k * rel(k)) / (#relevant)
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
    Proper mAP: mean of AP over VALID queries only (exclude open-set)
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