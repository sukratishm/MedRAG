# 🏥 Medical X-ray Gallery Builder
### Component 1 of 3 — Visual Database + Search Engine

```
Deliverable Files
─────────────────────────────────────────────────
index/
  visual_db.index    ← FAISS binary (the "gallery")
  metadata.json      ← id → {filename, filepath, labels}
  embeddings.npy     ← raw (N, 512) float32 array

Python Modules
─────────────────────────────────────────────────
data_downloader.py   ← Step 1: get X-ray images
gallery_builder.py   ← Step 2: build visual_db.index
visual_search.py     ← Step 3: search function (imported by app)
test_visual_search.py
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GALLERY BUILDER                       │
│                                                          │
│  X-ray images (PNG/JPG)                                  │
│       │                                                  │
│       ▼                                                  │
│  BiomedCLIP Vision Encoder                               │
│  microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224│
│  (trained on 15M biomedical image-text pairs)            │
│       │                                                  │
│       ▼  512-dim L2-normalized embedding                 │
│  FAISS IndexFlatIP  (cosine similarity)                  │
│       │                                                  │
│       ▼                                                  │
│  visual_db.index  +  metadata.json                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    VISUAL SEARCH                         │
│                                                          │
│  Query X-ray                                             │
│       │                                                  │
│       ▼  BiomedCLIP encode                               │
│  512-dim query vector                                    │
│       │                                                  │
│       ▼  FAISS.search(query, k=5)                        │
│  Top-5 similar cases                                     │
│       │                                                  │
│       ▼                                                  │
│  List[SearchResult]                                      │
│    rank, filename, filepath, labels, similarity          │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU acceleration (10x faster build):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install faiss-gpu
```

### 2. Get X-ray images

**Option A — Open-I (recommended, no login, ~900 MB, 7,470 images)**
```bash
python data_downloader.py --source openi --output_dir ./data
```

**Option B — NIH ChestX-ray14 sample (~1.1 GB, 5,000 images)**
```bash
python data_downloader.py --source nih_sample --output_dir ./data
```

**Option C — Your own local images**
```bash
python data_downloader.py --source local --local_dir /path/to/xrays --output_dir ./data
```

### 3. Build the gallery index

```bash
python gallery_builder.py \
    --image_dir  ./data/openi_images \
    --output_dir ./index \
    --batch_size 32 \
    --device     cpu
```

Expected time:
- CPU  (no GPU): ~1,000 img/min → 7K images ≈ 7 minutes
- GPU  (T4):     ~8,000 img/min → 7K images ≈ 1 minute

Expected output:
```
index/
  visual_db.index    ~30 MB for 7K images
  metadata.json      ~2 MB
  embeddings.npy     ~14 MB
  build_stats.json
```

### 4. Test the search

```bash
python visual_search.py ./data/openi_images/some_xray.png --index_dir ./index --top_k 5
```

```
🔍 Query: ./data/openi_images/CXR100_IM-0008-1001.png
============================================================
  #1  0.954  ████████████████████████████
       CXR99_IM-0007-1001.png
       Labels: Cardiomegaly

  #2  0.931  ███████████████████████████
       CXR101_IM-0009-2001.png
       Labels: No Finding

  ...
```

### 5. Use in your web app / RAG pipeline

```python
from visual_search import VisualSearchEngine

# Load once at app startup
engine = VisualSearchEngine(index_dir="./index", device="auto", top_k=5)

# Call for each uploaded X-ray
results = engine.search("uploaded_xray.png", load_images=True)

for r in results:
    print(f"#{r.rank}  sim={r.similarity:.3f}  {r.labels}")
    print(f"     {r.filepath}")
```

### 6. Run tests

```bash
# Unit tests (no model download needed)
pytest test_visual_search.py -v -m "not integration"

# Integration tests (requires built index)
pytest test_visual_search.py -v -m integration --index_dir ./index
```

---

## Why BiomedCLIP?

| Model | CheXpert AUC | Training Data | Fine-tune needed? |
|-------|-------------|---------------|-------------------|
| CLIP (ViT-B/32) | 0.71 | General images | Yes |
| **BiomedCLIP** | **0.87** | **15M biomedical pairs** | **No** |
| CheXNet | 0.84 | CheXpert only | Dataset-specific |

BiomedCLIP's embedding space inherently clusters similar pathologies together — 
pneumonia images cluster near other pneumonia cases without any task-specific training.

---

## Scaling Guide

| Gallery Size | Index Type | Build Time (GPU) | Search Time |
|-------------|------------|-----------------|-------------|
| < 10K       | FlatIP     | < 2 min         | < 5 ms      |
| 10K – 500K  | IVFFlat    | 5–20 min        | < 20 ms     |
| > 500K      | IVFPQ      | 30+ min         | < 50 ms     |

The builder auto-selects FlatIP vs IVFFlat based on gallery size.

---

## File Descriptions

| File | Purpose |
|------|---------|
| `data_downloader.py` | Downloads Open-I or NIH dataset, builds metadata CSV |
| `gallery_builder.py` | Encodes images → FAISS index + metadata.json |
| `visual_search.py` | `VisualSearchEngine` class used by web app |
| `test_visual_search.py` | Unit + integration tests |
| `requirements.txt` | Python dependencies |
| `index/visual_db.index` | FAISS binary — the gallery database |
| `index/metadata.json` | Maps FAISS ID → filename + labels |
| `index/embeddings.npy` | Raw embeddings backup |

---

## Integration with RAG Pipeline (Component 2)

```python
# In your RAG system:
from visual_search import VisualSearchEngine

engine = VisualSearchEngine("./index")

def get_similar_cases(query_image_path: str, k: int = 5):
    """
    Returns top-K similar cases for RAG context.
    Feed these into your LLM prompt as evidence.
    """
    results = engine.search(query_image_path, top_k=k)
    return [
        {
            "case_id":    r.idx,
            "similarity": r.similarity,
            "diagnosis":  r.labels,
            "image_path": r.filepath,
        }
        for r in results
    ]
```
