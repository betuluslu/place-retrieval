import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=str, default="cache/embeddings_resnet50")
    p.add_argument("--out", type=str, default="cache/index_resnet50.npz")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gallery_embs = np.load(cache_dir / "gallery_embeddings.npy")
    gallery_paths = json.loads((cache_dir / "gallery_paths.json").read_text())

    np.savez_compressed(out_path, gallery_embeddings=gallery_embs, gallery_paths=np.array(gallery_paths, dtype=object))
    print(f"Saved index: {out_path} (N={gallery_embs.shape[0]}, D={gallery_embs.shape[1]})")


if __name__ == "__main__":
    main()