import argparse
import json
import logging
from pathlib import Path

import numpy as np

from place_retrieval.data import load_manifest
from place_retrieval.embeddings import extract_embeddings

logging.basicConfig(level=logging.INFO)


def main():
    """
    Extracts image embeddings and saves them to cache files.

    - Loads dataset records from manifest.csv
    - Splits images into gallery and query
    - Extracts embeddings using the model
    - Saves embeddings and paths to disk (.npy + .json)

    If cache files already exist, it skips extraction to save time.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="data/dataset")
    parser.add_argument("--out-dir", type=str, default="cache/embeddings_resnet50")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")  # later: "cuda"
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    """
    Load all image records from manifest.csv.
    Each record contains:
    - relpath
    - split (gallery or query)
    - class info
    """

    records = load_manifest(dataset_root)

    # split
    gallery = [r for r in records if r.split == "gallery"]
    query = [r for r in records if r.split == "query"]

    # cache files
    gallery_emb_path = out_dir / "gallery_embeddings.npy"
    gallery_meta_path = out_dir / "gallery_paths.json"
    query_emb_path = out_dir / "query_embeddings.npy"
    query_meta_path = out_dir / "query_paths.json"

    if gallery_emb_path.exists() and gallery_meta_path.exists():
        logging.info("Gallery cache found, skipping extraction: %s", out_dir)
    else:
        logging.info("Extracting gallery embeddings (%d images)...", len(gallery))
        g = extract_embeddings(dataset_root, gallery, batch_size=args.batch_size, device_str=args.device)
        np.save(gallery_emb_path, g.embeddings)
        gallery_meta_path.write_text(json.dumps(g.paths, indent=2))
        logging.info("Saved: %s", gallery_emb_path)

    if query_emb_path.exists() and query_meta_path.exists():
        logging.info("Query cache found, skipping extraction: %s", out_dir)
    else:
        logging.info("Extracting query embeddings (%d images)...", len(query))
        q = extract_embeddings(dataset_root, query, batch_size=args.batch_size, device_str=args.device)
        np.save(query_emb_path, q.embeddings)
        query_meta_path.write_text(json.dumps(q.paths, indent=2))
        logging.info("Saved: %s", query_emb_path)


if __name__ == "__main__":
    main()