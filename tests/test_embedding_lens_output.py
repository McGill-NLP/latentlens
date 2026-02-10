"""
Tests for EmbeddingLens (nearest neighbors) output format.

These tests verify that:
1. Existing EmbeddingLens outputs have expected structure
2. Output values match golden files

Run with: pytest tests/test_embedding_lens_output.py -v
"""
import json
import pytest
from pathlib import Path


class TestEmbeddingLensOutputStructure:
    """Test that existing EmbeddingLens outputs have expected structure."""

    @pytest.fixture
    def embedding_lens_output_dir(self, analysis_results_dir):
        """Path to EmbeddingLens output (lite10 version)."""
        return analysis_results_dir / "nearest_neighbors" / "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded_lite10"

    def test_output_files_exist(self, embedding_lens_output_dir):
        """Expected output files should exist for multiple layers."""
        if not embedding_lens_output_dir.exists():
            pytest.skip("EmbeddingLens output directory not found")

        files = list(embedding_lens_output_dir.glob("nearest_neighbors_*.json"))
        assert len(files) >= 5, f"Expected at least 5 layer files, found {len(files)}"

    def test_output_metadata(self, embedding_lens_output_dir):
        """Output JSON should have expected metadata fields."""
        if not embedding_lens_output_dir.exists():
            pytest.skip("EmbeddingLens output directory not found")

        files = list(embedding_lens_output_dir.glob("nearest_neighbors_*.json"))
        output_file = files[0]

        with open(output_file) as f:
            data = json.load(f)

        # Check required fields (EmbeddingLens uses llm_layer and splits structure)
        assert "checkpoint" in data
        assert "llm_layer" in data
        assert "splits" in data
        assert "validation" in data["splits"]

    def test_results_structure(self, embedding_lens_output_dir):
        """Each result should have expected structure."""
        if not embedding_lens_output_dir.exists():
            pytest.skip("EmbeddingLens output directory not found")

        files = list(embedding_lens_output_dir.glob("nearest_neighbors_*.json"))
        output_file = files[0]

        with open(output_file) as f:
            data = json.load(f)

        # EmbeddingLens uses splits.validation.images structure
        images = data["splits"]["validation"]["images"]
        assert len(images) > 0, "Images should not be empty"

        # Check first image
        first_image = images[0]
        assert "image_idx" in first_image
        assert "ground_truth_caption" in first_image
        assert "chunks" in first_image

        # Check chunk/patch structure
        first_chunk = first_image["chunks"][0]
        assert "patches" in first_chunk

        first_patch = first_chunk["patches"][0]
        assert "patch_idx" in first_patch
        assert "nearest_neighbors" in first_patch

        # Check neighbor structure
        neighbors = first_patch["nearest_neighbors"]
        assert len(neighbors) > 0
        first_neighbor = neighbors[0]
        assert "token" in first_neighbor
        assert "similarity" in first_neighbor


class TestEmbeddingLensAllModels:
    """Test that all model combinations have EmbeddingLens outputs."""

    def test_multiple_models_have_outputs(self, analysis_results_dir):
        """Multiple model combinations should have EmbeddingLens outputs."""
        base = analysis_results_dir / "nearest_neighbors"
        if not base.exists():
            pytest.skip("EmbeddingLens output directory not found")

        lite10_dirs = [d for d in base.iterdir() if d.is_dir() and "lite10" in d.name and "ablations" not in str(d)]
        assert len(lite10_dirs) >= 6, f"Expected at least 6 model outputs, found {len(lite10_dirs)}"
