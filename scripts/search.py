import argparse
import json
from pathlib import Path

import numpy as np

from place_retrieval.index import search_one


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=str, default="cache/embeddings_resnet50")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--query-idx", type=int, default=None, help="If set, searches only that query index.")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)

    gallery_embs = np.load(cache_dir / "gallery_embeddings.npy")
    query_embs = np.load(cache_dir / "query_embeddings.npy")

    gallery_paths = json.loads((cache_dir / "gallery_paths.json").read_text())
    query_paths = json.loads((cache_dir / "query_paths.json").read_text())

    if args.query_idx is not None:
        qi = args.query_idx
        q_emb = query_embs[qi]
        res = search_one(gallery_embs, gallery_paths, q_emb, top_k=args.top_k)
        print(f"QUERY[{qi}] = {query_paths[qi]}")
        for rank, (path, score) in enumerate(zip(res.paths, res.scores), start=1):
            print(f"{rank:02d}. {score:.4f}  {path}")
        return

    # otherwise loop all queries (preview first 10)
    n_show = min(10, len(query_paths))
    for qi in range(n_show):
        q_emb = query_embs[qi]
        res = search_one(gallery_embs, gallery_paths, q_emb, top_k=args.top_k)
        print("=" * 60)
        print(f"QUERY[{qi}] = {query_paths[qi]}")
        for rank, (path, score) in enumerate(zip(res.paths, res.scores), start=1):
            print(f"{rank:02d}. {score:.4f}  {path}")


if __name__ == "__main__":
    main()