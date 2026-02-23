import json
import logging
from pathlib import Path

from place_retrieval.data import validate_dataset

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    """
    Validates the dataset before running the pipeline.

    - Reads the dataset folder
    - Checks image files using the validation function
    - Detects missing, unreadable, or small images
    - Prints a summary report in JSON format
    """

    dataset_root = Path("data/dataset")
    summary = validate_dataset(dataset_root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))