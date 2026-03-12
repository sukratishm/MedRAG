"""
visual_search.py
────────────────
Search function for the Medical X-ray RAG system.

Input:  A chest X-ray image (file path or PIL Image or numpy array)
Output: Top-K most similar cases from the gallery database

This is the module imported by your web app and RAG pipeline.

Usage:
    from visual_search import VisualSearchEngine

    engine = VisualSearchEngine(
        index_dir="./index",
        device="auto"
    )

    results = engine.search("./query_xray.png", top_k=5)
    # returns List[SearchResult]
    for r in results:
        print(f"{r.rank}. {r.filename}  sim={r.similarity:.3f}  labels={r.labels}")
"""

import json
import time
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Union, Optional

import faiss
import torch
import open_clip
from PIL import Image, UnidentifiedImageError

log = logging.getLogger(__name__)

# ── Constants (must match gallery_builder.py) ──────────────────────────────────
BIOMEDCLIP_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
INDEX_FILE       = "visual_db.index"
METADATA_FILE    = "metadata.json"


# ── Result dataclass ───────────────────────────────────────────────────────────
@dataclass
class SearchResult:
    """One similar case returned by the search engine."""
    rank:        int           # 1 = most similar
    idx:         int           # Internal FAISS index ID
    filename:    str           # Image filename
    filepath:    str           # Absolute path to the image
    labels:      str           # Diagnosis labels (from metadata)
    similarity:  float         # Cosine similarity [0, 1]
    image:       Optional[object] = field(default=None, repr=False)
    # ↑ Optionally loaded PIL Image (set load_images=True in search())

    def to_dict(self) -> dict:
        return {
            "rank":       self.rank,
            "idx":        self.idx,
            "filename":   self.filename,
            "filepath":   self.filepath,
            "labels":     self.labels,
            "similarity": round(float(self.similarity), 4),
        }


# ── Search Engine ──────────────────────────────────────────────────────────────
class VisualSearchEngine:
    """
    Thread-safe visual search engine for chest X-ray similarity retrieval.

    Architecture:
        Query image
            │
            ▼
        BiomedCLIP vision encoder  →  512-dim embedding (L2 normalized)
            │
            ▼
        FAISS IndexFlatIP          →  cosine similarity search
            │
            ▼
        Top-K results + metadata

    Attributes:
        index_dir (Path): Directory containing visual_db.index + metadata.json
        device    (str):  Compute device for BiomedCLIP
        top_k     (int):  Default number of results to return
    """

    def __init__(
        self,
        index_dir: Union[str, Path],
        device: str = "auto",
        top_k: int = 5,
    ):
        self.index_dir = Path(index_dir).resolve()
        self.top_k = top_k
        self._model = None
        self._transform = None
        self._index = None
        self._metadata: dict = {}

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Eager load
        self._load_index()
        self._load_model()
        log.info(f"VisualSearchEngine ready  (index={self._index.ntotal:,} images, device={self.device})")

    # ── Private loaders ────────────────────────────────────────────────────────
    def _load_index(self):
        """Load FAISS index + metadata from disk."""
        index_path = self.index_dir / INDEX_FILE
        meta_path  = self.index_dir / METADATA_FILE

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}\n"
                "Run:  python gallery_builder.py --image_dir ./data --output_dir ./index"
            )
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        log.info(f"Loading FAISS index from {index_path}...")
        self._index = faiss.read_index(str(index_path))

        # For IVF indexes, set nprobe for recall/speed tradeoff
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = 16

        log.info(f"Index loaded  ({self._index.ntotal:,} vectors, dim={self._index.d})")

        with open(meta_path) as f:
            self._metadata = json.load(f)

    def _load_model(self):
        """Load BiomedCLIP vision encoder."""
        log.info("Loading BiomedCLIP encoder...")
        model, _, transform = open_clip.create_model_and_transforms(BIOMEDCLIP_MODEL)
        self._model = model.to(self.device).eval()
        self._transform = transform
        log.info("BiomedCLIP loaded  ✓")

    # ── Embedding ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode a single PIL image → L2-normalized 512-dim embedding.
        Returns shape (1, 512) float32 numpy array.
        """
        tensor = self._transform(image).unsqueeze(0).to(self.device)
        features = self._model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    # ── Public API ─────────────────────────────────────────────────────────────
    def search(
        self,
        query: Union[str, Path, Image.Image, np.ndarray],
        top_k: Optional[int] = None,
        load_images: bool = False,
        exclude_perfect_match: bool = False,
    ) -> list[SearchResult]:
        """
        Find the top-K most similar X-ray images to a query.

        Args:
            query:                 File path, PIL Image, or RGB numpy array
            top_k:                 Number of results (overrides default)
            load_images:           Load PIL Images into SearchResult.image
            exclude_perfect_match: Skip results with similarity ≥ 0.9999
                                   (use when query is in the gallery itself)

        Returns:
            List[SearchResult] ordered by descending similarity
        """
        t0 = time.perf_counter()
        k = top_k or self.top_k

        # ── Load query image ───────────────────────────────────────────────────
        if isinstance(query, (str, Path)):
            query_path = Path(query)
            if not query_path.exists():
                raise FileNotFoundError(f"Query image not found: {query_path}")
            try:
                img = Image.open(query_path).convert("RGB")
            except (UnidentifiedImageError, OSError) as e:
                raise ValueError(f"Cannot open image: {query_path}  ({e})")

        elif isinstance(query, np.ndarray):
            img = Image.fromarray(query.astype(np.uint8))

        elif isinstance(query, Image.Image):
            img = query.convert("RGB")

        else:
            raise TypeError(f"Unsupported query type: {type(query)}")

        # ── Encode ─────────────────────────────────────────────────────────────
        query_emb = self._embed_image(img)   # (1, 512)

        # ── FAISS search ───────────────────────────────────────────────────────
        search_k = k + 1 if exclude_perfect_match else k
        similarities, indices = self._index.search(query_emb, search_k)
        similarities = similarities[0]   # (k,)
        indices      = indices[0]        # (k,)

        # ── Build results ──────────────────────────────────────────────────────
        results: list[SearchResult] = []
        rank = 1
        for sim, idx in zip(similarities, indices):
            if idx < 0:                                    # FAISS returns -1 for empty slots
                continue
            if exclude_perfect_match and float(sim) >= 0.9999:
                continue                                   # skip exact self-match

            meta = self._metadata.get(str(idx), {})
            filepath = meta.get("filepath", "")

            result = SearchResult(
                rank=rank,
                idx=int(idx),
                filename=meta.get("filename", f"image_{idx}"),
                filepath=filepath,
                labels=meta.get("labels", "Unknown"),
                similarity=float(sim),
            )

            if load_images and filepath and Path(filepath).exists():
                try:
                    result.image = Image.open(filepath).convert("RGB")
                except Exception:
                    pass   # image loading is best-effort

            results.append(result)
            rank += 1
            if len(results) >= k:
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.debug(f"Search completed in {elapsed_ms:.1f} ms  →  {len(results)} results")
        return results

    def search_batch(
        self,
        queries: list[Union[str, Path, Image.Image]],
        top_k: Optional[int] = None,
    ) -> list[list[SearchResult]]:
        """
        Batch search for multiple query images.
        More efficient than calling search() in a loop.
        """
        k = top_k or self.top_k
        embeddings = []

        for q in queries:
            if isinstance(q, (str, Path)):
                img = Image.open(q).convert("RGB")
            elif isinstance(q, np.ndarray):
                img = Image.fromarray(q.astype(np.uint8))
            else:
                img = q.convert("RGB")
            embeddings.append(self._embed_image(img)[0])

        batch_emb = np.stack(embeddings)            # (N, 512)
        sims_batch, idxs_batch = self._index.search(batch_emb, k)

        all_results = []
        for sims, idxs in zip(sims_batch, idxs_batch):
            results = []
            for rank, (sim, idx) in enumerate(zip(sims, idxs), start=1):
                if idx < 0:
                    continue
                meta = self._metadata.get(str(idx), {})
                results.append(SearchResult(
                    rank=rank,
                    idx=int(idx),
                    filename=meta.get("filename", f"image_{idx}"),
                    filepath=meta.get("filepath", ""),
                    labels=meta.get("labels", "Unknown"),
                    similarity=float(sim),
                ))
            all_results.append(results)

        return all_results

    def get_stats(self) -> dict:
        """Return index statistics."""
        return {
            "total_images": self._index.ntotal,
            "embed_dim":    self._index.d,
            "index_type":   type(self._index).__name__,
            "device":       self.device,
            "index_dir":    str(self.index_dir),
        }

    def __repr__(self) -> str:
        return (
            f"VisualSearchEngine("
            f"images={self._index.ntotal:,}, "
            f"device={self.device}, "
            f"index_dir={self.index_dir})"
        )


# ── Standalone CLI ─────────────────────────────────────────────────────────────
def main():
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser(
        description="Search for similar X-ray images"
    )
    parser.add_argument("query_image", type=Path, help="Path to query X-ray image")
    parser.add_argument(
        "--index_dir", type=Path, default=Path("./index"),
        help="Directory with visual_db.index (default: ./index)"
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    engine = VisualSearchEngine(
        index_dir=args.index_dir,
        device=args.device,
        top_k=args.top_k,
    )

    print(f"\n🔍 Query: {args.query_image}")
    print("=" * 60)
    results = engine.search(args.query_image, exclude_perfect_match=True)

    for r in results:
        bar = "█" * int(r.similarity * 30)
        print(f"  #{r.rank}  {r.similarity:.3f}  {bar}")
        print(f"       {r.filename}")
        print(f"       Labels: {r.labels}")
        print()

    print(f"Index stats: {engine.get_stats()}")


if __name__ == "__main__":
    main()
