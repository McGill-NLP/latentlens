"""
Tests for latentlens.extract — build_index, auto_layers, load_corpus.

CPU-only tests use synthetic data.  GPU tests are marked @pytest.mark.slow.
"""

import tempfile
from pathlib import Path

import pytest

from latentlens.extract import auto_layers, load_corpus


class TestAutoLayers:
    def test_32_layers(self):
        assert auto_layers(32) == [1, 2, 4, 8, 16, 24, 30, 31]

    def test_28_layers(self):
        assert auto_layers(28) == [1, 2, 4, 8, 16, 24, 26, 27]

    def test_12_layers(self):
        result = auto_layers(12)
        assert result == [1, 2, 4, 8, 10, 11]

    def test_small_model(self):
        result = auto_layers(6)
        assert result == [1, 2, 4, 5]

    def test_sorted_and_unique(self):
        for n in [12, 24, 28, 32, 48]:
            layers = auto_layers(n)
            assert layers == sorted(set(layers))
            assert all(0 < l < n for l in layers)


class TestLoadCorpus:
    def test_list_passthrough(self):
        texts = ["hello world", "foo bar"]
        assert load_corpus(texts) == texts

    def test_txt_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("first sentence\n")
            f.write("second sentence\n")
            f.write("\n")  # empty line should be skipped
            f.write("third sentence\n")
            path = f.name
        result = load_corpus(path)
        assert result == ["first sentence", "second sentence", "third sentence"]
        Path(path).unlink()

    def test_csv_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("text,label\n")
            f.write("a dog runs,1\n")
            f.write("a cat sits,0\n")
            path = f.name
        result = load_corpus(path)
        assert result == ["a dog runs", "a cat sits"]
        Path(path).unlink()

    def test_csv_no_header(self):
        """CSV where the first row is data (not a recognizable header)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("the quick brown fox,2\n")
            f.write("jumped over the fence,3\n")
            path = f.name
        result = load_corpus(path)
        assert result == ["the quick brown fox", "jumped over the fence"]
        Path(path).unlink()

    def test_path_object(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\n")
            path = Path(f.name)
        assert load_corpus(path) == ["hello"]
        path.unlink()


class TestBuildIndex:
    @pytest.mark.slow
    def test_build_index_small_corpus(self):
        """Build an index from a small corpus (requires GPU + model download)."""
        from latentlens.extract import build_index

        texts = [
            "a dog sits on the grass",
            "the cat jumped over the fence",
            "birds fly in the blue sky",
        ] * 10  # 30 texts

        index = build_index(
            "allenai/OLMo-7B-1024-preview",
            corpus=texts,
            layers=[1, 8],
            max_contexts_per_token=5,
            batch_size=8,
            show_progress=False,
        )

        assert 1 in index.available_layers
        assert 8 in index.available_layers
        assert len(index) > 0

    @pytest.mark.slow
    def test_prefix_dedup(self):
        """Same prefix stored only once."""
        from latentlens.extract import build_index

        # Two texts sharing a prefix — "the cat" at positions 0-2
        texts = ["the cat sat on the mat", "the cat jumped high"]

        index = build_index(
            "allenai/OLMo-7B-1024-preview",
            corpus=texts,
            layers=[1],
            max_contexts_per_token=100,
            batch_size=2,
            show_progress=False,
        )

        # Count how many times "cat" appears (with prefix "the cat")
        # Should be exactly 1 (prefix dedup)
        layer_data = index._layers_data[1]
        cat_entries = [
            m for m in layer_data["metadata"]
            if m["token_str"].strip() == "cat" and m["position"] == 2
        ]
        assert len(cat_entries) == 1

    @pytest.mark.slow
    def test_soft_cap(self):
        """max_contexts_per_token is enforced."""
        from latentlens.extract import build_index

        # Many unique contexts for "the"
        texts = [f"the {word} runs fast" for word in [
            "dog", "cat", "bird", "fish", "horse", "cow", "pig",
            "sheep", "goat", "duck", "bear", "wolf", "fox",
        ]]

        index = build_index(
            "allenai/OLMo-7B-1024-preview",
            corpus=texts,
            layers=[1],
            max_contexts_per_token=3,
            batch_size=4,
            show_progress=False,
        )

        layer_data = index._layers_data[1]
        from collections import Counter
        token_counts = Counter(m["token_str"] for m in layer_data["metadata"])
        for count in token_counts.values():
            assert count <= 3
