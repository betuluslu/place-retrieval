
# Image-Based Place Recognition & Retrieval System

## Overview
This project implements a lightweight image-based place recognition system that retrieves the most similar location for a given query image using visual embeddings and cosine similarity search. The system is designed to be robust, reproducible, and suitable for real-world scenarios such as robotics, surveillance, and visual localization.

The pipeline includes:
- Dataset validation
- Embedding extraction (pretrained CNN)
- Similarity search (Top-K retrieval)
- Open-set (UNKNOWN) location handling
- Evaluation with Recall@K and mAP
- Unit-tested metric implementation


## Overview

1. Dataset Handling
- Dataset is organized using a manifest.csv file
- Split: gallery (reference images) and query (search images)
- Robust validation includes:
  • Missing file detection
  • Corrupted image skipping
  • Grayscale to RGB conversion
  • Small resolution filtering
- Metadata (file names, GPS, etc.) is NOT used for recognition (pixel-only approach)


2. Embedding Extraction
Backbone: ResNet50 (pretrained, PyTorch)
- Classification head removed → 2048-D feature vectors
- L2 normalization applied to embeddings
- Batched inference for efficiency
- Cached embeddings (.npy) to avoid recomputation

Why ResNet50?
- Strong baseline for visual retrieval
- Fast and lightweight
- No training required (as specified in assignment)


3. Similarity Search
- Cosine similarity (dot product on normalized embeddings)
- Top-K retrieval implemented using NumPy
- Efficient ranking via vectorized operations


4. Open-Set (UNKNOWN) Recognition
Some query images may not exist in the gallery.

Decision rule:
IF Top-1 cosine similarity < τ → Predict UNKNOWN

Final chosen threshold:
τ = 0.50

This value was selected via empirical tuning to balance false UNKNOWN and missed UNKNOWN predictions.


5. Evaluation Metrics
Evaluation is performed at item-level with multi-positive support:
- Recall@K (K = 1, 5, 10)
- Mean Average Precision (mAP)
- Open-set UNKNOWN Precision & Recall

Important Notes:
- All gallery images of the same location are treated as positives
- Open-set queries (no gallery match) are excluded from Recall@K and mAP
- Metrics are validated with unit tests (pytest)


## Results (Provided Dataset)

| Metric | Score |
|--------|-------|
| Recall@1 | 1.00 |
| Recall@5 | 1.00 |
| Recall@10 | 1.00 |
| mAP | ~0.90 |
| UNKNOWN Precision | 1.00 |
| UNKNOWN Recall | 1.00 |

**Additional Stats:**
- Total Queries: 29  
- Open-set Queries: 9  
- Unknown Threshold (τ): 0.50

## Project Structure
```
place-retrieval/
 ├── src/place_retrieval/
 │   ├── data.py
 │   ├── embeddings.py
 │   ├── index.py
 │   └── metrics.py
 ├── scripts/
 │   ├── validate_dataset.py
 │   ├── extract_embeddings.py
 │   ├── build_index.py
 │   ├── search.py
 │   └── evaluate.py
 ├── tests/
 │   └── test_metrics.py
 ├── .gitignore
 ├── README.md
 └── requirements.txt
```


## Installation

Create a virtual environment:
```bash
python -m venv venv
```

### Windows
```bash
venv\Scripts\activate
```

### Linux / Mac
```bash
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```


## How to Run the Pipeline
This section provides step-by-step commands to reproduce the full retrieval pipeline from dataset validation to evaluation.

### 1. Validate Dataset
```bash
python scripts/validate_dataset.py
```


### 2. Extract Embeddings (Cached)
```bash
python scripts/extract_embeddings.py --device cpu
```

### 3. Build Index
```bash
python scripts/build_index.py
```

### 4. Run Similarity Search (Example Query)
```bash
python scripts/search.py --query-idx 0 --top-k 5
```

### 5. Evaluate System Performance
```bash
python scripts/evaluate.py --unknown-threshold 0.50
```

### 6. Run Unit Tests
```bash
pytest -q
```


## Engineering & Design Highlights
- Modular and clean project architecture (src/scripts/tests)
- Reproducible pipeline with cached embeddings
- Robust dataset validation (real-world edge cases)
- Proper metric implementation (multi-positive mAP)
- Open-set recognition support
- Unit-tested evaluation logic


## Limitations & Future Improvements
- Performance may degrade under extreme viewpoint or lighting changes
- Small dataset may cause optimistic Recall scores
- Possible Improvements:
  • Using DINOv2 or CLIP embeddings
  • FAISS for scalable ANN search
  • Hard negative mining
  • ONNX optimization for faster inference


## Notes
- Dataset and cached embeddings are excluded via .gitignore to keep the repository lightweight and reproducible.
- The focus of this assignment is correctness, robustness, and engineering quality rather than heavy model training.

