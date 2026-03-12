"""
gallery_builder.py
──────────────────
Builds the visual search database for Medical X-ray RAG.

Pipeline:
  1. Load all X-ray images from --image_dir
  2. Encode each image → 512-dim vector via BiomedCLIP
  3. Normalize + store in FAISS IndexFlatIP (cosine similarity via dot product)
  4. Save:   visual_db.index   (FAISS binary)
             metadata.json     (filename → {path, labels, idx})
             embeddings.npy    (raw numpy array, optional backup)

BiomedCLIP:
  microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
  Trained on 15M biomedical image-caption pairs from PubMed Central.
  Zero-shot performance on CheXpert = 0.85+ AUC (no fine-tuning needed).

Usage:
    python gallery_builder.py \
        --image_dir  ./data/openi_images \
        --output_dir ./index \
        --batch_size 64 \
        --device     cpu

    # Resume interrupted build:
    python gallery_builder.py --image_dir ./data/openi_images --resume

Output files:
    ./index/visual_db.index    ← FAISS binary index
    ./index/metadata.json      ← id → {filename, filepath, labels}
    ./index/embeddings.npy     ← (N, 512) float32 array
    ./index/build_stats.json   ← timing + counts
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import faiss
import open_clip
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
BIOMEDCLIP_MODEL  = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
EMBED_DIM         = 512
SUPPORTED_EXTS    = {".png", ".jpg", ".jpeg", ".dcm"}
INDEX_FILE        = "visual_db.index"
METADATA_FILE     = "metadata.json"
EMBEDDINGS_FILE   = "embeddings.npy"
STATS_FILE        = "build_stats.json"


# ── Dataset ────────────────────────────────────────────────────────────────────
class XRayDataset(Dataset):
    """
    Lazy-loading dataset for chest X-ray images.
    Applies BiomedCLIP preprocessing (resize 224, normalize).
    Skips corrupt/unreadable files gracefully.
    """

    def __init__(
        self,
        image_paths: list[Path],
        transform,
        metadata_csv_path: Optional[Path] = None,
    ):
        self.paths = image_paths
        self.transform = transform
        self.label_map: dict[str, str] = {}

        # Optional: load NIH/CheXpert labels CSV
        if metadata_csv_path and metadata_csv_path.exists():
            import pandas as pd
            df = pd.read_csv(metadata_csv_path)
            if "filename" in df.columns and "labels" in df.columns:
                self.label_map = dict(zip(df["filename"], df["labels"].fillna("Unknown")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
            label = self.label_map.get(path.name, "Unknown")
            return tensor, str(path), label, True   # (tensor, path, label, valid)
        except (UnidentifiedImageError, OSError, Exception) as e:
            log.warning(f"Skipping corrupt image: {path.name}  ({e})")
            # Return a zero tensor so DataLoader batch stays uniform
            dummy = torch.zeros(3, 224, 224)
            return dummy, str(path), "CORRUPT", False


def collate_skip_corrupt(batch):
    """Custom collate: filter out corrupt images before batching."""
    valid = [(t, p, l) for t, p, l, ok in batch if ok]
    if not valid:
        return None
    tensors, paths, labels = zip(*valid)
    return torch.stack(tensors), list(paths), list(labels)


# ── Model loader ───────────────────────────────────────────────────────────────
def load_biomedclip(device: str):
    """
    Load BiomedCLIP vision encoder from HuggingFace hub.
    Returns (model, transform) where model outputs 512-dim image embeddings.
    """
    log.info("Loading BiomedCLIP from HuggingFace hub (first run downloads ~350 MB)...")
    try:
        model, _, transform = open_clip.create_model_and_transforms(
            BIOMEDCLIP_MODEL
        )
        model = model.to(device).eval()
        log.info(f"BiomedCLIP loaded  ✓  device={device}")
        return model, transform
    except Exception as e:
        log.error(f"Failed to load BiomedCLIP: {e}")
        log.error("Ensure open-clip-torch is installed:  pip install open-clip-torch")
        raise


# ── Embedding engine ───────────────────────────────────────────────────────────
@torch.no_grad()
def encode_batch(model, image_tensors: torch.Tensor, device: str) -> np.ndarray:
    """Encode a batch of image tensors → L2-normalized embeddings (N, 512)."""
    image_tensors = image_tensors.to(device)
    features = model.encode_image(image_tensors)
    # L2 normalize → cosine similarity = dot product
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)


# ── FAISS index builder ────────────────────────────────────────────────────────
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build FAISS IndexFlatIP (inner product = cosine similarity after L2-norm).
    For galleries > 100K images, swap to IndexIVFFlat for 10x faster search.
    """
    n, d = embeddings.shape
    log.info(f"Building FAISS index  ({n:,} vectors × {d} dims)")

    if n < 10_000:
        # Exact search — best for < 10K images
        index = faiss.IndexFlatIP(d)
    else:
        # Approximate search — needed for large galleries
        nlist = min(256, n // 39)   # IVF rule: nlist ≈ sqrt(N)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        log.info(f"Training IVF index with nlist={nlist}...")
        index.train(embeddings)
        index.nprobe = 16  # search 16 cells at query time (accuracy vs speed)

    index.add(embeddings)
    log.info(f"FAISS index built  ✓  total vectors: {index.ntotal:,}")
    return index


# ── Resume support ─────────────────────────────────────────────────────────────
def load_checkpoint(output_dir: Path) -> tuple[np.ndarray | None, dict | None, int]:
    """Load partial embeddings + metadata if build was interrupted."""
    emb_ckpt = output_dir / "embeddings_checkpoint.npy"
    meta_ckpt = output_dir / "metadata_checkpoint.json"

    if emb_ckpt.exists() and meta_ckpt.exists():
        embeddings = np.load(emb_ckpt)
        with open(meta_ckpt) as f:
            metadata = json.load(f)
        start_idx = len(metadata)
        log.info(f"[RESUME] Found checkpoint with {start_idx:,} images. Continuing...")
        return embeddings, metadata, start_idx

    return None, None, 0


def save_checkpoint(output_dir: Path, embeddings: np.ndarray, metadata: dict):
    """Save incremental checkpoint every N batches."""
    np.save(output_dir / "embeddings_checkpoint.npy", embeddings)
    with open(output_dir / "metadata_checkpoint.json", "w") as f:
        json.dump(metadata, f)


# ── Main pipeline ──────────────────────────────────────────────────────────────
def build_gallery(
    image_dir: Path,
    output_dir: Path,
    batch_size: int = 64,
    device: str = "auto",
    metadata_csv: Optional[Path] = None,
    resume: bool = False,
    checkpoint_every: int = 500,
):
    """
    Full pipeline: images → BiomedCLIP embeddings → FAISS index.
    
    Args:
        image_dir:        Directory containing X-ray images (scanned recursively)
        output_dir:       Where to save visual_db.index + metadata.json
        batch_size:       Images per GPU/CPU batch (lower if OOM)
        device:           "cuda", "cpu", or "auto"
        metadata_csv:     Optional CSV with columns: filename, labels
        resume:           Resume from last checkpoint if available
        checkpoint_every: Save checkpoint every N images
    """
    t_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve device ─────────────────────────────────────────────────────────
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps"  if torch.backends.mps.is_available() else "cpu"
        )
    log.info(f"Device: {device}")

    # ── Collect image paths ────────────────────────────────────────────────────
    all_images = sorted([
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTS
    ])
    if not all_images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    log.info(f"Found {len(all_images):,} images in {image_dir}")

    # ── Resume checkpoint ──────────────────────────────────────────────────────
    existing_emb, existing_meta, start_idx = (None, None, 0)
    if resume:
        existing_emb, existing_meta, start_idx = load_checkpoint(output_dir)

    images_to_process = all_images[start_idx:]
    log.info(f"Images to process: {len(images_to_process):,}")

    # ── Load BiomedCLIP ────────────────────────────────────────────────────────
    model, transform = load_biomedclip(device)

    # ── Dataset + DataLoader ───────────────────────────────────────────────────
    dataset = XRayDataset(images_to_process, transform, metadata_csv)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device == "cuda"),
        collate_fn=collate_skip_corrupt,
        prefetch_factor=2 if device == "cuda" else None,
    )

    # ── Accumulate embeddings ──────────────────────────────────────────────────
    all_embeddings: list[np.ndarray] = []
    all_metadata: dict = existing_meta or {}   # id (int) → {filename, filepath, labels}
    global_idx = start_idx
    skipped = 0

    log.info("Encoding images with BiomedCLIP...")
    for batch in tqdm(loader, desc="Encoding", unit="batch", ncols=80):
        if batch is None:
            continue
        tensors, paths, labels = batch
        batch_emb = encode_batch(model, tensors, device)

        for i, (path, label) in enumerate(zip(paths, labels)):
            all_embeddings.append(batch_emb[i])
            all_metadata[str(global_idx)] = {
                "filename": Path(path).name,
                "filepath": path,
                "labels":   label,
                "idx":      global_idx,
            }
            global_idx += 1

        # Periodic checkpoint
        if global_idx % checkpoint_every < batch_size:
            combined_emb = np.vstack(
                [existing_emb] + all_embeddings
                if existing_emb is not None else all_embeddings
            )
            save_checkpoint(output_dir, combined_emb, all_metadata)
            log.info(f"  Checkpoint saved at {global_idx:,} images")

    if not all_embeddings:
        raise RuntimeError("No valid images were encoded. Check image directory.")

    # ── Stack all embeddings ───────────────────────────────────────────────────
    new_embeddings = np.vstack(all_embeddings)
    if existing_emb is not None:
        final_embeddings = np.vstack([existing_emb, new_embeddings])
    else:
        final_embeddings = new_embeddings

    log.info(f"Embeddings shape: {final_embeddings.shape}")

    # ── Build + save FAISS index ───────────────────────────────────────────────
    index = build_faiss_index(final_embeddings)
    index_path = output_dir / INDEX_FILE
    faiss.write_index(index, str(index_path))
    log.info(f"FAISS index saved → {index_path}  ({index_path.stat().st_size / 1e6:.1f} MB)")

    # ── Save metadata ──────────────────────────────────────────────────────────
    meta_path = output_dir / METADATA_FILE
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    log.info(f"Metadata saved   → {meta_path}")

    # ── Save raw embeddings (optional, useful for offline analysis) ────────────
    emb_path = output_dir / EMBEDDINGS_FILE
    np.save(emb_path, final_embeddings)
    log.info(f"Embeddings saved → {emb_path}  ({emb_path.stat().st_size / 1e6:.1f} MB)")

    # ── Clean up checkpoints ───────────────────────────────────────────────────
    for ckpt in ["embeddings_checkpoint.npy", "metadata_checkpoint.json"]:
        ckpt_path = output_dir / ckpt
        if ckpt_path.exists():
            ckpt_path.unlink()

    # ── Build stats ────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    stats = {
        "total_images":   index.ntotal,
        "skipped":        skipped,
        "embed_dim":      EMBED_DIM,
        "model":          BIOMEDCLIP_MODEL,
        "index_type":     type(index).__name__,
        "build_time_sec": round(elapsed, 1),
        "throughput_img_per_sec": round(index.ntotal / elapsed, 1),
        "index_size_mb":  round(index_path.stat().st_size / 1e6, 2),
        "device":         device,
    }
    with open(output_dir / STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    log.info("=" * 55)
    log.info(f"✅ Gallery build complete!")
    log.info(f"   Images indexed : {index.ntotal:,}")
    log.info(f"   Build time     : {elapsed:.0f}s  ({stats['throughput_img_per_sec']} img/s)")
    log.info(f"   Index size     : {stats['index_size_mb']} MB")
    log.info(f"   Output dir     : {output_dir.resolve()}")
    log.info("=" * 55)

    return index, all_metadata


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS visual search index from chest X-ray images"
    )
    parser.add_argument(
        "--image_dir", type=Path, required=True,
        help="Root directory containing X-ray images (searched recursively)"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("./index"),
        help="Where to save visual_db.index + metadata.json (default: ./index)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for encoding. Reduce to 16 if CPU RAM < 8 GB (default: 64)"
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu", "mps"], default="auto",
        help="Compute device (default: auto-detect)"
    )
    parser.add_argument(
        "--metadata_csv", type=Path, default=None,
        help="Optional CSV with columns: filename, labels"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint if build was interrupted"
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=500,
        help="Save checkpoint every N images (default: 500)"
    )
    args = parser.parse_args()

    build_gallery(
        image_dir=args.image_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        batch_size=args.batch_size,
        device=args.device,
        metadata_csv=args.metadata_csv,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
