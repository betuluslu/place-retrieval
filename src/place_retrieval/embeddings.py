from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from place_retrieval.data import ImageRecord

LOGGER = logging.getLogger("place_retrieval.embeddings")


@dataclass(frozen=True)
class EmbeddingBatch:
    paths: List[str]
    embeddings: np.ndarray  # shape: (B, D), float32, L2-normalized


def _build_backbone(device: torch.device) -> Tuple[nn.Module, int]:
    """
    Baseline: ResNet50 backbone without classifier head.
    Returns (model, embedding_dim).
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # remove classification head => output becomes (B, 2048)
    model.fc = nn.Identity()
    model.eval().to(device)
    return model, 2048


def _preprocess() -> transforms.Compose:
    weights = models.ResNet50_Weights.DEFAULT
    return weights.transforms()


@torch.no_grad()
def extract_embeddings(
    dataset_root: Path,
    records: List[ImageRecord],
    batch_size: int = 32,
    device_str: str = "cpu",
) -> EmbeddingBatch:
    device = torch.device(device_str)

    model, dim = _build_backbone(device)
    preprocess = _preprocess()

    all_paths: List[str] = []
    all_embs: List[np.ndarray] = []

    # simple batching
    batch_imgs = []
    batch_paths = []

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)  # (B,3,H,W)
        feats = model(x)  # (B, 2048)
        feats = feats.float()
        # L2 normalize
        feats = feats / (feats.norm(p=2, dim=1, keepdim=True) + 1e-12)
        feats_np = feats.cpu().numpy().astype(np.float32)

        all_paths.extend(batch_paths)
        all_embs.append(feats_np)

        batch_imgs.clear()
        batch_paths.clear()

    for r in records:
        img_path = dataset_root / r.relpath
        img = Image.open(img_path).convert("RGB")
        tensor = preprocess(img)  # torch float tensor
        batch_imgs.append(tensor)
        batch_paths.append(r.relpath)

        if len(batch_imgs) >= batch_size:
            flush()

    flush()

    embs = np.concatenate(all_embs, axis=0) if all_embs else np.zeros((0, dim), dtype=np.float32)
    return EmbeddingBatch(paths=all_paths, embeddings=embs)