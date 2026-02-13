"""
End-to-end tests simulating a real user who:
  1. pip install latentlens
  2. Points to a .txt file + any HuggingFace model name
  3. Gets a searchable contextual embedding index

Tests cover causal LMs, VLMs, and different architectures.
Each test validates the full pipeline: load corpus → build index → save → reload → search.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

CORPUS_PATH = Path(__file__).parent / "corpus_100.txt"


def _build_and_validate(model_name, corpus, layers, batch_size=8, device=None):
    """Build index from corpus, validate structure, save/reload, search."""
    import latentlens

    index = latentlens.build_index(
        model_name,
        corpus=corpus,
        layers=layers,
        max_contexts_per_token=10,
        batch_size=batch_size,
        device=device,
        show_progress=False,
    )

    # Validate structure
    assert len(index.available_layers) == len(layers)
    for layer in layers:
        assert layer in index.available_layers
    assert index.hidden_dim > 0
    assert len(index) > 0

    # Save and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        index.save(tmpdir)

        # Verify files exist
        for layer in layers:
            cache_file = Path(tmpdir) / f"layer_{layer}" / "embeddings_cache.pt"
            assert cache_file.exists(), f"Missing cache file for layer {layer}"

        reloaded = latentlens.ContextualIndex.from_directory(tmpdir)
        assert reloaded.available_layers == index.available_layers
        assert reloaded.hidden_dim == index.hidden_dim

    # Search with random query
    query = F.normalize(torch.randn(5, index.hidden_dim), dim=-1)
    results = index.search(query, top_k=3)

    assert len(results) == 5
    for neighbors in results:
        assert len(neighbors) == 3
        for n in neighbors:
            assert isinstance(n, latentlens.Neighbor)
            assert n.token_str  # non-empty
            assert -1.0 <= n.similarity <= 1.0 + 1e-6
            assert n.contextual_layer in layers

    return index


class TestUserE2E:
    """Simulates: pip install latentlens → build_index(model, corpus.txt) → search."""

    # ── Causal LMs (small, CPU-friendly) ────────────────────────────────

    def test_distilgpt2_from_txt_file(self):
        """User points to a .txt file and distilgpt2."""
        _build_and_validate("distilgpt2", str(CORPUS_PATH), layers=[1, 3, 5], device="cpu")

    def test_tiny_gpt2(self):
        """Smallest possible GPT-2 variant."""
        _build_and_validate("sshleifer/tiny-gpt2", str(CORPUS_PATH), layers=[1, 2], device="cpu")

    def test_pythia_70m(self):
        """GPT-NeoX architecture (EleutherAI Pythia)."""
        _build_and_validate("EleutherAI/pythia-70m", str(CORPUS_PATH), layers=[1, 3, 5], device="cpu")

    def test_opt_125m(self):
        """OPT architecture (Meta)."""
        _build_and_validate("facebook/opt-125m", str(CORPUS_PATH), layers=[1, 5, 11], device="cpu")

    # ── VLMs (need GPU) ──────────────────────────────────────────────────

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_qwen2_vl_2b(self):
        """Qwen2-VL-2B — real VLM, uses AutoModel fallback."""
        _build_and_validate(
            "Qwen/Qwen2-VL-2B-Instruct", str(CORPUS_PATH), layers=[1, 8, 27],
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_smolvlm_256m(self):
        """SmolVLM-256M — Idefics3 architecture, nested text_config."""
        _build_and_validate(
            "HuggingFaceTB/SmolVLM-256M-Instruct", str(CORPUS_PATH), layers=[1, 8, 15],
        )

    def test_corpus_as_list(self):
        """User passes a Python list instead of a file path."""
        texts = ["the dog barked loudly", "a cat sat on the mat", "birds sing in trees"]
        _build_and_validate("distilgpt2", texts, layers=[1, 5], device="cpu")

    def test_auto_layers_default(self):
        """User omits layers= entirely → auto_layers picks sensible defaults."""
        import latentlens

        index = latentlens.build_index(
            "distilgpt2",
            corpus=["hello world", "foo bar baz"] * 10,
            device="cpu",
            show_progress=False,
            max_contexts_per_token=5,
        )
        # distilgpt2 has 6 layers → auto_layers(6) = [1, 2, 4, 5]
        assert index.available_layers == [1, 2, 4, 5]

    def test_save_reload_search_identical(self):
        """Save → reload → search gives identical results."""
        import latentlens

        index = _build_and_validate("distilgpt2", str(CORPUS_PATH), layers=[1, 5])

        query = F.normalize(torch.randn(3, index.hidden_dim), dim=-1)
        original = index.search(query, top_k=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            index.save(tmpdir)
            reloaded = latentlens.ContextualIndex.from_directory(tmpdir)
            reloaded_results = reloaded.search(query, top_k=5)

        for orig_neighbors, reload_neighbors in zip(original, reloaded_results):
            for o, r in zip(orig_neighbors, reload_neighbors):
                assert o.token_str == r.token_str
                assert abs(o.similarity - r.similarity) < 1e-4
                assert o.contextual_layer == r.contextual_layer

    def test_metadata_has_captions(self):
        """Each neighbor should have the source caption it came from."""
        import latentlens

        index = latentlens.build_index(
            "distilgpt2",
            corpus=str(CORPUS_PATH),
            layers=[1],
            device="cpu",
            show_progress=False,
            max_contexts_per_token=10,
        )
        query = F.normalize(torch.randn(1, index.hidden_dim), dim=-1)
        results = index.search(query, top_k=3)

        for n in results[0]:
            assert n.caption, "Neighbor should have non-empty caption"
            assert n.position >= 2, "Position should be >= 2 (BOS and pos 1 skipped)"
            assert n.token_id >= 0, "Token ID should be non-negative"

    def test_prefix_dedup_across_corpus(self):
        """Identical texts in corpus → embeddings stored only once (prefix dedup)."""
        import latentlens

        # Exact duplicate texts — every prefix is identical
        texts = [
            "the cat sat on the mat",
            "the cat sat on the mat",
            "the cat sat on the mat",
        ]
        index = latentlens.build_index(
            "distilgpt2",
            corpus=texts,
            layers=[1],
            device="cpu",
            show_progress=False,
            max_contexts_per_token=100,
        )
        layer_data = index._layers_data[1]
        # With 3 identical texts, dedup should yield the same count as 1 text
        count_from_3 = len(layer_data["metadata"])

        index_single = latentlens.build_index(
            "distilgpt2",
            corpus=["the cat sat on the mat"],
            layers=[1],
            device="cpu",
            show_progress=False,
            max_contexts_per_token=100,
        )
        count_from_1 = len(index_single._layers_data[1]["metadata"])

        assert count_from_3 == count_from_1, (
            f"Dedup failed: 3 identical texts gave {count_from_3} entries, "
            f"1 text gave {count_from_1}"
        )
