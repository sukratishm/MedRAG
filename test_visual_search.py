"""
test_visual_search.py
─────────────────────
Unit + integration tests for the gallery builder pipeline.

Run:
    # Fast unit tests (no model needed):
    pytest test_visual_search.py -v -m "not integration"

    # Full integration test (requires built index):
    pytest test_visual_search.py -v --index_dir ./index --image_dir ./data
"""

import json
import tempfile
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image


# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture
def dummy_index_dir(tmp_path):
    """Create a minimal fake FAISS index + metadata for unit tests."""
    import faiss

    d = 512
    n = 20
    embeddings = np.random.randn(n, d).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= norms

    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    faiss.write_index(index, str(tmp_path / "visual_db.index"))

    metadata = {
        str(i): {
            "filename": f"image_{i:04d}.png",
            "filepath": str(tmp_path / f"image_{i:04d}.png"),
            "labels":   "Pneumonia" if i % 3 == 0 else "No Finding",
            "idx":      i,
        }
        for i in range(n)
    }
    with open(tmp_path / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # Create dummy PNG files
    for i in range(n):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(tmp_path / f"image_{i:04d}.png")

    return tmp_path, embeddings


@pytest.fixture
def dummy_xray_image(tmp_path) -> Path:
    """Create a fake grayscale X-ray image."""
    img_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    img = Image.fromarray(img_array, mode="L").convert("RGB")
    path = tmp_path / "test_xray.png"
    img.save(path)
    return path


# ── Unit tests ─────────────────────────────────────────────────────────────────
class TestSearchResult:
    def test_to_dict(self):
        from visual_search import SearchResult
        r = SearchResult(rank=1, idx=5, filename="img.png",
                         filepath="/data/img.png", labels="Pneumonia",
                         similarity=0.87654)
        d = r.to_dict()
        assert d["rank"] == 1
        assert d["similarity"] == 0.8765   # rounded to 4 decimal places
        assert d["labels"] == "Pneumonia"
        assert "image" not in d            # PIL image not serialized


class TestFAISSIndex:
    """Test FAISS index properties independent of BiomedCLIP."""

    def test_build_flat_index(self):
        import faiss
        d, n = 512, 100
        emb = np.random.randn(n, d).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        index = faiss.IndexFlatIP(d)
        index.add(emb)
        assert index.ntotal == n

    def test_search_returns_correct_k(self):
        import faiss
        d, n = 512, 50
        emb = np.random.randn(n, d).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        index = faiss.IndexFlatIP(d)
        index.add(emb)

        query = emb[0:1]  # use first vector as query
        sims, idxs = index.search(query, k=5)
        assert sims.shape == (1, 5)
        assert idxs.shape == (1, 5)
        # Self-match should be first with similarity ≈ 1.0
        assert abs(sims[0][0] - 1.0) < 1e-5
        assert idxs[0][0] == 0

    def test_cosine_similarity_via_dot_product(self):
        """L2-normalized dot product = cosine similarity."""
        import faiss
        d = 512
        # Two identical vectors should have similarity 1.0
        v = np.random.randn(1, d).astype(np.float32)
        v /= np.linalg.norm(v)

        index = faiss.IndexFlatIP(d)
        index.add(v)

        sims, _ = index.search(v, k=1)
        assert abs(sims[0][0] - 1.0) < 1e-5

    def test_ivf_index_for_large_gallery(self):
        """IVF index works for large galleries (>10K vectors)."""
        import faiss
        d, n = 512, 10_000
        emb = np.random.randn(n, d).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        nlist = 64
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(emb)
        index.add(emb)
        index.nprobe = 8

        assert index.ntotal == n
        # Check that search still works
        sims, idxs = index.search(emb[0:1], k=5)
        assert idxs[0][0] == 0   # self should be top result


class TestMetadataBuilding:
    def test_metadata_keys(self, dummy_index_dir):
        _, embeddings = dummy_index_dir
        meta_path = dummy_index_dir[0] / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        assert "0" in meta
        entry = meta["0"]
        assert "filename" in entry
        assert "filepath" in entry
        assert "labels" in entry
        assert "idx" in entry

    def test_metadata_count_matches_index(self, dummy_index_dir):
        import faiss
        index_dir = dummy_index_dir[0]
        index = faiss.read_index(str(index_dir / "visual_db.index"))
        with open(index_dir / "metadata.json") as f:
            meta = json.load(f)
        assert index.ntotal == len(meta)


class TestVisualSearchEngine:
    """Tests using mocked BiomedCLIP to avoid model download."""

    def _get_engine_with_mock_model(self, index_dir):
        """Create engine with BiomedCLIP mocked out."""
        from visual_search import VisualSearchEngine
        import faiss

        with patch("visual_search.open_clip.create_model_and_transforms") as mock_create:
            mock_model = MagicMock()
            mock_transform = MagicMock(return_value=MagicMock(
                unsqueeze=lambda _: MagicMock(to=lambda _: MagicMock())
            ))
            mock_create.return_value = (mock_model, None, mock_transform)

            engine = VisualSearchEngine(index_dir=index_dir, device="cpu")

            # Mock the embed function to return a random normalized vector
            def fake_embed(img):
                v = np.random.randn(1, 512).astype(np.float32)
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                return v

            engine._embed_image = fake_embed
            return engine

    def test_search_returns_k_results(self, dummy_index_dir):
        index_dir = dummy_index_dir[0]
        engine = self._get_engine_with_mock_model(index_dir)

        dummy_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        results = engine.search(dummy_img, top_k=5)
        assert len(results) == 5

    def test_results_sorted_by_similarity(self, dummy_index_dir):
        index_dir = dummy_index_dir[0]
        engine = self._get_engine_with_mock_model(index_dir)

        dummy_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        results = engine.search(dummy_img, top_k=5)
        sims = [r.similarity for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_results_have_required_fields(self, dummy_index_dir):
        index_dir = dummy_index_dir[0]
        engine = self._get_engine_with_mock_model(index_dir)

        dummy_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        results = engine.search(dummy_img, top_k=3)
        for r in results:
            assert hasattr(r, "rank")
            assert hasattr(r, "filename")
            assert hasattr(r, "filepath")
            assert hasattr(r, "labels")
            assert hasattr(r, "similarity")
            assert 0.0 <= r.similarity <= 1.0

    def test_ranks_are_sequential(self, dummy_index_dir):
        index_dir = dummy_index_dir[0]
        engine = self._get_engine_with_mock_model(index_dir)

        dummy_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        results = engine.search(dummy_img, top_k=5)
        for i, r in enumerate(results, start=1):
            assert r.rank == i

    def test_file_not_found_raises(self, dummy_index_dir):
        index_dir = dummy_index_dir[0]
        engine = self._get_engine_with_mock_model(index_dir)
        with pytest.raises(FileNotFoundError):
            engine.search("/nonexistent/image.png")

    def test_batch_search(self, dummy_index_dir):
        index_dir = dummy_index_dir[0]
        engine = self._get_engine_with_mock_model(index_dir)

        imgs = [
            Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            for _ in range(3)
        ]
        batch_results = engine.search_batch(imgs, top_k=5)
        assert len(batch_results) == 3
        assert all(len(r) == 5 for r in batch_results)

    def test_get_stats(self, dummy_index_dir):
        index_dir = dummy_index_dir[0]
        engine = self._get_engine_with_mock_model(index_dir)
        stats = engine.get_stats()
        assert "total_images" in stats
        assert stats["total_images"] == 20
        assert stats["embed_dim"] == 512

    def test_to_dict_serializable(self, dummy_index_dir):
        """Search results must be JSON serializable for API responses."""
        index_dir = dummy_index_dir[0]
        engine = self._get_engine_with_mock_model(index_dir)

        dummy_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        results = engine.search(dummy_img, top_k=3)
        payload = [r.to_dict() for r in results]
        assert json.dumps(payload)   # raises if not serializable


# ── Integration tests (require real index) ─────────────────────────────────────
@pytest.mark.integration
class TestIntegration:
    """Run with: pytest -m integration --index_dir ./index --image_dir ./data"""

    @pytest.fixture(autouse=True)
    def setup(self, request):
        self.index_dir = Path(request.config.getoption("--index_dir", default="./index"))
        self.image_dir = Path(request.config.getoption("--image_dir", default="./data"))

    def test_real_search(self):
        from visual_search import VisualSearchEngine
        engine = VisualSearchEngine(self.index_dir, device="cpu")
        stats = engine.get_stats()
        assert stats["total_images"] > 0
        print(f"\nIndex contains {stats['total_images']:,} images")

    def test_search_with_real_image(self):
        from visual_search import VisualSearchEngine
        engine = VisualSearchEngine(self.index_dir, device="cpu")

        # Find first image in data dir
        images = list(self.image_dir.rglob("*.png"))[:1]
        if not images:
            pytest.skip("No test images found")

        results = engine.search(images[0], top_k=5, exclude_perfect_match=True)
        assert len(results) > 0
        assert results[0].similarity <= 1.0
        print(f"\nTop result: {results[0].filename}  sim={results[0].similarity:.3f}")


# ── Pytest config ──────────────────────────────────────────────────────────────
def pytest_addoption(parser):
    parser.addoption("--index_dir", action="store", default="./index")
    parser.addoption("--image_dir", action="store", default="./data")
