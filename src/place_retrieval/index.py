from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    """
    This class stores the search results for a single query image.

    indices: indices of the top-K gallery images
    scores: cosine similarity scores for top-K matches
    paths: relative file paths of the matched gallery images

    We keep everything together so the system can easily
    display results and debug retrieval outputs.
    """
    indices: np.ndarray  # (K,)
    scores: np.ndarray   # (K,) cosine similarity
    paths: List[str]     # (K,) gallery relpaths


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Applies L2 normalization to embeddings.

    Why is this important?
    - After normalization, dot product == cosine similarity
    - Makes similarity comparison more stable
    - Standard practice in image retrieval systems

    Args:
        x: embedding matrix of shape (N, D)

    Returns:
        L2-normalized embeddings with same shape (N, D)
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def cosine_topk(
    gallery_embs: np.ndarray,   # (N, D) L2-normalized
    query_emb: np.ndarray,      # (D,)   L2-normalized
    k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds Top-K most similar gallery images using cosine similarity.

    IMPORTANT:
    Because embeddings are L2-normalized,
    cosine similarity = dot product.

    So instead of computing expensive cosine distance,
    we simply do:
        sims = gallery @ query

    This is very fast and fully vectorized.

    Args:
        gallery_embs: all gallery embeddings (N images)
        query_emb: single query embedding (D dimension)
        k: number of top results to return

    Returns:
        idx_sorted: indices of top-K most similar images
        scores: corresponding cosine similarity scores
    """

    sims = gallery_embs @ query_emb  # (N,)
    k = min(k, sims.shape[0])
    # fast top-k (unsorted), then sort
    idx = np.argpartition(-sims, kth=k - 1)[:k]
    idx_sorted = idx[np.argsort(-sims[idx])]
    return idx_sorted, sims[idx_sorted]


def search_one(
    gallery_embs: np.ndarray,
    gallery_paths: List[str],
    query_emb: np.ndarray,
    top_k: int = 5,
) -> SearchResult:
    """
    Performs retrieval for a single query image.

    Full flow:
    1. Compute cosine similarity with all gallery embeddings
    2. Select top-K most similar images
    3. Map indices to actual file paths
    4. Return structured search result

    This function is the core of the retrieval system.
    It is used during:
    - Demo search
    - Evaluation (Recall@K, mAP)
    - Qualitative result inspection
    """
    idx, scores = cosine_topk(gallery_embs, query_emb, k=top_k)
    paths = [gallery_paths[i] for i in idx.tolist()]
    return SearchResult(indices=idx, scores=scores, paths=paths)