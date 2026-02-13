"""
Tests for latentlens.index — ContextualIndex and Neighbor.

Uses synthetic data (no GPU or model downloads required).
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from latentlens.index import ContextualIndex, Neighbor


def _make_synthetic_index(
    layers=(1, 8, 16), num_embeddings=100, hidden_dim=64, seed=42
):
    """Create a ContextualIndex with random normalized embeddings."""
    gen = torch.Generator().manual_seed(seed)
    layers_data = {}
    for layer in layers:
        emb = torch.randn(num_embeddings, hidden_dim, generator=gen)
        emb = F.normalize(emb, dim=-1)
        metadata = [
            {
                "token_str": f"tok_{i}",
                "token_id": i,
                "caption": f"caption containing tok_{i}",
                "position": i % 20 + 2,
            }
            for i in range(num_embeddings)
        ]
        layers_data[layer] = {"embeddings": emb, "metadata": metadata}
    return ContextualIndex(layers_data)


class TestNeighbor:
    def test_fields(self):
        n = Neighbor(
            token_str=" dog",
            similarity=0.85,
            caption="a large dog",
            position=3,
            token_id=42,
            contextual_layer=16,
        )
        assert n.token_str == " dog"
        assert n.similarity == 0.85
        assert n.contextual_layer == 16

    def test_defaults(self):
        n = Neighbor(token_str="x", similarity=0.5)
        assert n.position == -1
        assert n.token_id == -1
        assert n.contextual_layer == -1


class TestContextualIndex:
    def test_available_layers(self):
        index = _make_synthetic_index(layers=(4, 1, 16))
        assert index.available_layers == [1, 4, 16]

    def test_hidden_dim(self):
        index = _make_synthetic_index(hidden_dim=128)
        assert index.hidden_dim == 128

    def test_len(self):
        index = _make_synthetic_index(layers=(1, 8), num_embeddings=50)
        assert len(index) == 100  # 50 per layer * 2 layers

    def test_repr(self):
        index = _make_synthetic_index()
        r = repr(index)
        assert "ContextualIndex" in r
        assert "300" in r  # 100 * 3 layers

    def test_search_returns_correct_shape(self):
        index = _make_synthetic_index()
        query = F.normalize(torch.randn(5, 64), dim=-1)
        results = index.search(query, top_k=3)
        assert len(results) == 5
        for neighbors in results:
            assert len(neighbors) == 3
            for n in neighbors:
                assert isinstance(n, Neighbor)

    def test_search_single_vector(self):
        """search() should accept a 1-D query and return a list with one entry."""
        index = _make_synthetic_index()
        query = F.normalize(torch.randn(64), dim=-1)
        results = index.search(query, top_k=2)
        assert len(results) == 1
        assert len(results[0]) == 2

    def test_search_sorted_by_similarity(self):
        index = _make_synthetic_index()
        query = F.normalize(torch.randn(3, 64), dim=-1)
        results = index.search(query, top_k=5)
        for neighbors in results:
            sims = [n.similarity for n in neighbors]
            assert sims == sorted(sims, reverse=True), "Results not sorted by descending similarity"

    def test_search_cross_layer_merge(self):
        """Results should come from multiple contextual layers."""
        index = _make_synthetic_index(layers=(1, 8, 16), num_embeddings=200)
        query = F.normalize(torch.randn(10, 64), dim=-1)
        results = index.search(query, top_k=10)
        all_ctx_layers = set()
        for neighbors in results:
            for n in neighbors:
                all_ctx_layers.add(n.contextual_layer)
        # With random data and enough queries, results should span multiple layers
        assert len(all_ctx_layers) > 1, (
            f"Expected results from multiple contextual layers, got {all_ctx_layers}"
        )

    def test_search_subset_layers(self):
        """search(layers=...) should restrict to those contextual layers."""
        index = _make_synthetic_index(layers=(1, 8, 16))
        query = F.normalize(torch.randn(3, 64), dim=-1)
        results = index.search(query, top_k=5, layers=[1, 8])
        for neighbors in results:
            for n in neighbors:
                assert n.contextual_layer in (1, 8)

    def test_search_similarity_range(self):
        """Cosine similarities should be in [-1, 1]."""
        index = _make_synthetic_index()
        query = F.normalize(torch.randn(5, 64), dim=-1)
        results = index.search(query, top_k=5)
        for neighbors in results:
            for n in neighbors:
                assert -1.0 <= n.similarity <= 1.0 + 1e-6

    def test_save_and_reload_roundtrip(self):
        index = _make_synthetic_index(layers=(1, 16), num_embeddings=30, hidden_dim=32)
        query = F.normalize(torch.randn(3, 32), dim=-1)
        original_results = index.search(query, top_k=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            index.save(tmpdir)

            # Verify files were created
            assert (Path(tmpdir) / "metadata.json").exists()
            assert (Path(tmpdir) / "layer_1" / "embeddings_cache.pt").exists()
            assert (Path(tmpdir) / "layer_16" / "embeddings_cache.pt").exists()

            loaded = ContextualIndex.from_directory(tmpdir)
            assert loaded.available_layers == [1, 16]
            assert loaded.hidden_dim == 32

            loaded_results = loaded.search(query, top_k=5)

        # Verify identical results
        assert len(loaded_results) == len(original_results)
        for orig, load in zip(original_results, loaded_results):
            assert len(orig) == len(load)
            for o, l in zip(orig, load):
                assert o.token_str == l.token_str
                assert abs(o.similarity - l.similarity) < 1e-4

    def test_from_directory_with_layer_filter(self):
        index = _make_synthetic_index(layers=(1, 8, 16))
        with tempfile.TemporaryDirectory() as tmpdir:
            index.save(tmpdir)
            loaded = ContextualIndex.from_directory(tmpdir, layers=[1, 16])
            assert loaded.available_layers == [1, 16]

    def test_from_directory_missing_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No embeddings_cache.pt"):
                ContextualIndex.from_directory(tmpdir)

    def test_to_device(self):
        index = _make_synthetic_index()
        index.to("cpu")
        assert index.device == torch.device("cpu")

    @pytest.mark.slow
    def test_from_pretrained_qwen2vl(self):
        """Download Qwen2-VL embeddings from HuggingFace and verify structure."""
        index = ContextualIndex.from_pretrained(
            "McGill-NLP/latentlens-qwen2vl-embeddings",
            layers=[27],
        )
        assert 27 in index.available_layers
        assert index.hidden_dim == 3584  # Qwen2-VL hidden size
        query = F.normalize(torch.randn(1, 3584), dim=-1)
        results = index.search(query, top_k=3)
        assert len(results) == 1
        assert len(results[0]) == 3
