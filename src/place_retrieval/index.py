from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    indices: np.ndarray  # (K,)
    scores: np.ndarray   # (K,) cosine similarity
    paths: List[str]     # (K,) gallery relpaths


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def cosine_topk(
    gallery_embs: np.ndarray,   # (N, D) L2-normalized
    query_emb: np.ndarray,      # (D,)   L2-normalized
    k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    # cosine sim = dot product because vectors are normalized
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
    idx, scores = cosine_topk(gallery_embs, query_emb, k=top_k)
    paths = [gallery_paths[i] for i in idx.tolist()]
    return SearchResult(indices=idx, scores=scores, paths=paths)