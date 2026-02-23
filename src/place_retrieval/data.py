from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from PIL import Image, UnidentifiedImageError


LOGGER = logging.getLogger("place_retrieval.data")


@dataclass(frozen=True)
class ImageRecord:
    """
    This class represents a single image entry from manifest.csv.

    Why we use this:
    - Keeps dataset information clean and structured
    - Easier to debug and track images
    - Avoids using raw dicts everywhere
    """
    image_id: str
    relpath: str
    split: str          # "gallery" or "query"
    class_dir: str
    class_name: str
    landmark_id: str

    @property
    def key(self) -> Tuple[str, str]:
        """
        Unique key for an image.
        Useful to detect duplicates across splits.
        """
        return (self.split, self.relpath)


def _md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute MD5 hash of a file.

    Why we do this:
    - To detect duplicate images between gallery and query
    - Prevent data leakage (VERY important for evaluation)
    """
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _safe_open_image(path: Path) -> Optional[Image.Image]:
    """
    Safely opens an image.

    What this function handles:
    - Missing files
    - Corrupted images
    - Unsupported formats
    - Grayscale to RGB conversion

    Returns:
        PIL Image if valid, otherwise None
    """
    try:
        img = Image.open(path)
        img.load()  # force decode
    except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
        LOGGER.warning("Skipping unreadable image: %s (%s)", path, e)
        return None

    # Normalize mode to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def load_manifest(dataset_root: Path) -> List[ImageRecord]:
    """
    Loads manifest.csv and converts each row into ImageRecord objects.

    Expected structure:
    dataset_root/
        manifest.csv
        landmarks/...

    Why this is important:
    - Central source of truth for dataset
    - Prevents hardcoding file paths
    - Ensures reproducibility
    """
    manifest_path = dataset_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv not found at {manifest_path}")

    df = pd.read_csv(manifest_path)

    required = {"image_id", "relpath", "split", "class_dir", "class_name", "landmark_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest.csv missing columns: {sorted(missing)}")

    records: List[ImageRecord] = []
    for row in df.to_dict(orient="records"):
        split = str(row["split"]).strip().lower()
        if split not in {"gallery", "query"}:
            LOGGER.warning("Unknown split=%s for relpath=%s (skipping)", split, row.get("relpath"))
            continue

        records.append(
            ImageRecord(
                image_id=str(row["image_id"]),
                relpath=str(row["relpath"]),
                split=split,
                class_dir=str(row["class_dir"]),
                class_name=str(row["class_name"]),
                landmark_id=str(row["landmark_id"]),
            )
        )
    return records


def validate_dataset(
    dataset_root: Path,
    min_side: int = 32,
    check_cross_split_duplicate_md5: bool = True,
) -> dict:
    """
    Validates the dataset quality and integrity.

    This function checks:
    - Missing image files
    - Unreadable or corrupted images
    - Very small images (low quality)
    - Data leakage (same image in gallery and query)

    This is CRITICAL for a robust ML pipeline.
    """
    records = load_manifest(dataset_root)

    total = len(records)
    by_split = {"gallery": 0, "query": 0}
    missing_files = 0
    unreadable = 0
    too_small = 0

    # quick check: identical relpath present in both splits
    relpath_to_splits = {}
    for r in records:
        relpath_to_splits.setdefault(r.relpath, set()).add(r.split)
    cross_split_same_relpath = [p for p, s in relpath_to_splits.items() if len(s) > 1]

    md5_by_split = {"gallery": {}, "query": {}}

    for r in records:
        by_split[r.split] += 1
        img_path = dataset_root / r.relpath
        if not img_path.exists():
            missing_files += 1
            LOGGER.warning("Missing file: %s", img_path)
            continue

        img = _safe_open_image(img_path)
        if img is None:
            unreadable += 1
            continue

        w, h = img.size
        if min(w, h) < min_side:
            too_small += 1
            LOGGER.warning("Too small image (min_side=%s): %s (%sx%s)", min_side, img_path, w, h)
            continue

        if check_cross_split_duplicate_md5:
            try:
                md5 = _md5_file(img_path)
                md5_by_split[r.split][img_path.as_posix()] = md5
            except OSError as e:
                LOGGER.warning("Failed hashing %s (%s)", img_path, e)

    cross_split_md5_duplicates = []
    if check_cross_split_duplicate_md5:
        gallery_md5 = set(md5_by_split["gallery"].values())
        query_md5 = set(md5_by_split["query"].values())
        overlap = gallery_md5.intersection(query_md5)
        if overlap:
            # list up to 20 examples
            for md5 in list(overlap)[:20]:
                g_files = [p for p, v in md5_by_split["gallery"].items() if v == md5]
                q_files = [p for p, v in md5_by_split["query"].items() if v == md5]
                cross_split_md5_duplicates.append({"md5": md5, "gallery": g_files, "query": q_files})

    summary = {
        "total_records": total,
        "split_counts": by_split,
        "missing_files": missing_files,
        "unreadable_images": unreadable,
        "too_small_images": too_small,
        "cross_split_same_relpath": cross_split_same_relpath[:20],
        "cross_split_md5_duplicates": cross_split_md5_duplicates,
    }
    return summary