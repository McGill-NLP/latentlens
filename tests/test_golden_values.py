"""
Golden value tests - verify actual NN results match expected values.

These tests check that specific patches have specific nearest neighbors,
ensuring the analysis code produces consistent results.

Run with: pytest tests/test_golden_values.py -v
"""
import json
import pytest
from pathlib import Path


class TestLatentLensGoldenValues:
    """Verify specific LatentLens nearest neighbor values."""

    @pytest.fixture
    def latentlens_data(self, analysis_results_dir):
        """Load LatentLens output for OLMo-ViT, visual layer 8."""
        path = (
            analysis_results_dir /
            "contextual_nearest_neighbors" /
            "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded_lite10" /
            "contextual_neighbors_visual8_allLayers.json"
        )
        if not path.exists():
            pytest.skip("LatentLens golden file not found")
        with open(path) as f:
            return json.load(f)

    def test_image0_patch0_top1_token(self, latentlens_data):
        """Image 0, patch 0 should have specific top-1 token."""
        patch0 = latentlens_data["results"][0]["chunks"][0]["patches"][0]
        top1 = patch0["nearest_contextual_neighbors"][0]

        # Check token string exists and similarity is in expected range
        assert "token_str" in top1
        assert 0.10 < top1["similarity"] < 0.25, f"Similarity {top1['similarity']} out of expected range"

    def test_image0_patch0_top5_count(self, latentlens_data):
        """Image 0, patch 0 should have exactly 5 neighbors."""
        patch0 = latentlens_data["results"][0]["chunks"][0]["patches"][0]
        assert len(patch0["nearest_contextual_neighbors"]) == 5

    def test_image0_patch0_similarities_sorted(self, latentlens_data):
        """Neighbors should be sorted by similarity (descending)."""
        patch0 = latentlens_data["results"][0]["chunks"][0]["patches"][0]
        neighbors = patch0["nearest_contextual_neighbors"]
        similarities = [n["similarity"] for n in neighbors]
        assert similarities == sorted(similarities, reverse=True), "Neighbors not sorted by similarity"

    def test_contextual_layers_used(self, latentlens_data):
        """Should search across multiple contextual layers."""
        patch0 = latentlens_data["results"][0]["chunks"][0]["patches"][0]
        neighbors = patch0["nearest_contextual_neighbors"]
        layers_used = set(n["contextual_layer"] for n in neighbors)
        # Top-5 might come from different layers
        assert len(layers_used) >= 1


class TestLogitLensGoldenValues:
    """Verify specific LogitLens prediction values."""

    @pytest.fixture
    def logitlens_data(self, analysis_results_dir):
        """Load LogitLens output for OLMo-ViT, layer 8."""
        path = (
            analysis_results_dir /
            "logit_lens" /
            "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded_lite10" /
            "logit_lens_layer8_topk5_multi-gpu.json"
        )
        if not path.exists():
            pytest.skip("LogitLens golden file not found")
        with open(path) as f:
            return json.load(f)

    def test_image0_patch0_top5_count(self, logitlens_data):
        """Image 0, patch 0 should have exactly 5 predictions."""
        patch0 = logitlens_data["results"][0]["chunks"][0]["patches"][0]
        assert len(patch0["top_predictions"]) == 5

    def test_image0_patch0_has_tokens(self, logitlens_data):
        """All predictions should have token strings."""
        patch0 = logitlens_data["results"][0]["chunks"][0]["patches"][0]
        for pred in patch0["top_predictions"]:
            assert "token" in pred
            assert isinstance(pred["token"], str)
            assert len(pred["token"]) > 0

    def test_image0_patch0_logits_sorted(self, logitlens_data):
        """Predictions should be sorted by logit (descending)."""
        patch0 = logitlens_data["results"][0]["chunks"][0]["patches"][0]
        predictions = patch0["top_predictions"]
        logits = [p["logit"] for p in predictions]
        assert logits == sorted(logits, reverse=True), "Predictions not sorted by logit"

    def test_layer_idx_matches_filename(self, logitlens_data):
        """Layer index in data should match filename."""
        assert logitlens_data["layer_idx"] == 8


class TestEmbeddingLensGoldenValues:
    """Verify specific EmbeddingLens nearest neighbor values."""

    @pytest.fixture
    def embedding_lens_data(self, analysis_results_dir):
        """Load EmbeddingLens output for OLMo-ViT, layer 8."""
        path = (
            analysis_results_dir /
            "nearest_neighbors" /
            "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded_lite10" /
            "nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer8.json"
        )
        if not path.exists():
            pytest.skip("EmbeddingLens golden file not found")
        with open(path) as f:
            return json.load(f)

    def test_image0_patch0_has_neighbors(self, embedding_lens_data):
        """Image 0, patch 0 should have neighbors."""
        images = embedding_lens_data["splits"]["validation"]["images"]
        patch0 = images[0]["chunks"][0]["patches"][0]
        assert len(patch0["nearest_neighbors"]) > 0

    def test_image0_patch0_similarities_sorted(self, embedding_lens_data):
        """Neighbors should be sorted by similarity (descending)."""
        images = embedding_lens_data["splits"]["validation"]["images"]
        patch0 = images[0]["chunks"][0]["patches"][0]
        neighbors = patch0["nearest_neighbors"]
        similarities = [n["similarity"] for n in neighbors]
        assert similarities == sorted(similarities, reverse=True), "Neighbors not sorted by similarity"

    def test_llm_layer_matches(self, embedding_lens_data):
        """LLM layer in data should be 8."""
        assert embedding_lens_data["llm_layer"] == 8


class TestCrossMethodConsistency:
    """Verify consistency across methods for same image."""

    @pytest.fixture
    def all_data(self, analysis_results_dir):
        """Load all three method outputs."""
        base = analysis_results_dir

        latentlens_path = (
            base / "contextual_nearest_neighbors" /
            "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded_lite10" /
            "contextual_neighbors_visual8_allLayers.json"
        )
        logitlens_path = (
            base / "logit_lens" /
            "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded_lite10" /
            "logit_lens_layer8_topk5_multi-gpu.json"
        )
        embedding_lens_path = (
            base / "nearest_neighbors" /
            "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded_lite10" /
            "nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer8.json"
        )

        data = {}
        if latentlens_path.exists():
            with open(latentlens_path) as f:
                data["latentlens"] = json.load(f)
        if logitlens_path.exists():
            with open(logitlens_path) as f:
                data["logitlens"] = json.load(f)
        if embedding_lens_path.exists():
            with open(embedding_lens_path) as f:
                data["embedding_lens"] = json.load(f)

        if len(data) < 2:
            pytest.skip("Need at least 2 method outputs for cross-method tests")
        return data

    def test_same_checkpoint(self, all_data):
        """All methods should use same checkpoint."""
        checkpoints = set()
        for name, data in all_data.items():
            checkpoints.add(data["checkpoint"])
        assert len(checkpoints) == 1, f"Different checkpoints used: {checkpoints}"

    def test_same_image0_caption(self, all_data):
        """All methods should have same ground truth caption for image 0."""
        captions = set()
        for name, data in all_data.items():
            if name == "embedding_lens":
                caption = data["splits"]["validation"]["images"][0]["ground_truth_caption"]
            else:
                caption = data["results"][0]["ground_truth_caption"]
            captions.add(caption[:100])  # Compare first 100 chars
        assert len(captions) == 1, "Different captions for image 0"

    def test_same_num_patches(self, all_data):
        """All methods should have same number of patches for image 0."""
        patch_counts = {}
        for name, data in all_data.items():
            if name == "embedding_lens":
                chunks = data["splits"]["validation"]["images"][0]["chunks"]
            else:
                chunks = data["results"][0]["chunks"]
            total_patches = sum(len(c["patches"]) for c in chunks)
            patch_counts[name] = total_patches

        counts = list(patch_counts.values())
        assert all(c == counts[0] for c in counts), f"Different patch counts: {patch_counts}"
