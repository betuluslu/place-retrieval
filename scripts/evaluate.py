import argparse
import json
from pathlib import Path

import numpy as np

from place_retrieval.metrics import mean_average_precision, recall_at_k


def _class_dir_from_relpath(relpath: str) -> str:
    # landmarks/<class_dir>/(gallery|query)/file.jpg
    parts = relpath.replace("\\", "/").split("/")
    return parts[1]


def main():
    """
    Evaluates the retrieval system.

    - Loads gallery and query embeddings from cache
    - Computes similarity between query and gallery
    - Ranks gallery images for each query
    - Calculates Recall@K and mAP
    - Detects UNKNOWN (open-set) queries using a threshold
    """

    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, default="data/dataset")  # kept for compatibility
    p.add_argument("--cache-dir", type=str, default="cache/embeddings_resnet50")
    p.add_argument("--k-list", type=str, default="1,5,10")
    p.add_argument("--unknown-threshold", type=float, default=0.50)
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)

    gallery_embs = np.load(cache_dir / "gallery_embeddings.npy")
    query_embs = np.load(cache_dir / "query_embeddings.npy")
    gallery_paths = json.loads((cache_dir / "gallery_paths.json").read_text())
    query_paths = json.loads((cache_dir / "query_paths.json").read_text())

    # Build mapping: class_dir -> set of gallery ITEMS (paths)
    class_to_gallery_items = {}
    for gp in gallery_paths:
        c = _class_dir_from_relpath(gp)
        class_to_gallery_items.setdefault(c, set()).add(gp)

    # True items per query: all gallery images from same class_dir (multi-positive!)
    true_items = []
    open_set_count = 0
    for qp in query_paths:
        qc = _class_dir_from_relpath(qp)
        items = sorted(list(class_to_gallery_items.get(qc, [])))
        if not items:
            open_set_count += 1
        true_items.append(items)

    true_is_unknown = [len(items) == 0 for items in true_items]

    # Ranked list per query: gallery ITEMS ordered by similarity
    ranked_items_all = []
    unknown_pred_count = 0

    # UNKNOWN confusion metrics
    tp = fp = fn = 0

    for i in range(len(query_embs)):
        q = query_embs[i]
        sims = gallery_embs @ q  # cosine because embeddings are L2-normalized
        order = np.argsort(-sims)

        ranked_items = [gallery_paths[j] for j in order.tolist()]
        ranked_items_all.append(ranked_items)

        top1_sim = float(sims[order[0]])
        pred_is_unknown = top1_sim < args.unknown_threshold

        if pred_is_unknown:
            unknown_pred_count += 1

        # Confusion accounting for UNKNOWN (open-set)
        if true_is_unknown[i] and pred_is_unknown:
            tp += 1
        elif (not true_is_unknown[i]) and pred_is_unknown:
            fp += 1
        elif true_is_unknown[i] and (not pred_is_unknown):
            fn += 1

    # Convert k-list string (e.g. "1,5,10") into integer list [1, 5, 10]

    k_list = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]

    unknown_precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    unknown_recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    metrics = {
        "total_queries": len(query_embs),
        "open_set_queries": open_set_count,
        "unknown_threshold": args.unknown_threshold,
        "unknown_predicted_count": unknown_pred_count,
        "unknown_metrics": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": unknown_precision,
            "recall": unknown_recall,
        },
        "recall": {},
        "mAP": None,
    }

    for k in k_list:
        metrics["recall"][f"R@{k}"] = recall_at_k(ranked_items_all, true_items, k=k)

    metrics["mAP"] = mean_average_precision(ranked_items_all, true_items)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()