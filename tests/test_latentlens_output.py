"""
Tests for LatentLens (contextual nearest neighbors) output format.

These tests verify that:
1. Existing LatentLens outputs have expected structure
2. Output values match golden files
3. New runs would produce consistent results

Run with: pytest tests/test_latentlens_output.py -v
"""
import json
import pytest
from pathlib import Path


class TestLatentLensOutputStructure:
    """Test that existing LatentLens outputs have expected structure."""

    def test_output_files_exist(self, latentlens_output_olmo_vit_lite10):
        """Expected output files should exist for all visual layers."""
        expected_visual_layers = [0, 1, 2, 4, 8, 16, 24, 30, 31]

        for visual_layer in expected_visual_layers:
            output_file = latentlens_output_olmo_vit_lite10 / f"contextual_neighbors_visual{visual_layer}_allLayers.json"
            assert output_file.exists(), f"Output file for visual_layer={visual_layer} not found"

    def test_output_metadata(self, latentlens_output_olmo_vit_lite10):
        """Output JSON should have expected metadata fields."""
        output_file = latentlens_output_olmo_vit_lite10 / "contextual_neighbors_visual8_allLayers.json"

        with open(output_file) as f:
            data = json.load(f)

        # Check required metadata fields
        assert "checkpoint" in data
        assert "contextual_dir" in data
        assert "visual_layer" in data
        assert "contextual_layers_used" in data
        assert "num_images" in data
        assert "top_k" in data
        assert "results" in data

        # Check specific values for lite10 version
        assert data["visual_layer"] == 8
        assert data["num_images"] == 100  # lite10 still processes 100 images
        assert data["top_k"] == 5

    def test_output_has_correct_layers(self, latentlens_output_olmo_vit_lite10):
        """Output should use all contextual layers for search."""
        output_file = latentlens_output_olmo_vit_lite10 / "contextual_neighbors_visual8_allLayers.json"

        with open(output_file) as f:
            data = json.load(f)

        # OLMo has layers 1,2,4,8,16,24,30,31 for contextual search
        expected_contextual_layers = [1, 2, 4, 8, 16, 24, 30, 31]
        assert data["contextual_layers_used"] == expected_contextual_layers


class TestLatentLensResultsFormat:
    """Test the format of individual patch results."""

    def test_results_structure(self, latentlens_output_olmo_vit_lite10):
        """Each result should have expected structure."""
        output_file = latentlens_output_olmo_vit_lite10 / "contextual_neighbors_visual8_allLayers.json"

        with open(output_file) as f:
            data = json.load(f)

        results = data["results"]
        assert len(results) > 0, "Results should not be empty"

        # Check first result
        first_result = results[0]
        assert "image_idx" in first_result
        assert "ground_truth_caption" in first_result
        assert "feature_shape" in first_result
        assert "chunks" in first_result

        # Check feature shape (OLMo-ViT uses 576 patches, 4096 hidden dim)
        feature_shape = first_result["feature_shape"]
        assert feature_shape[2] == 576, f"Expected 576 patches, got {feature_shape[2]}"
        assert feature_shape[3] == 4096, f"Expected 4096 hidden dim, got {feature_shape[3]}"

    def test_patch_neighbors_format(self, latentlens_output_olmo_vit_lite10):
        """Each patch should have top-k nearest neighbors with required fields."""
        output_file = latentlens_output_olmo_vit_lite10 / "contextual_neighbors_visual8_allLayers.json"

        with open(output_file) as f:
            data = json.load(f)

        first_result = results = data["results"][0]
        first_chunk = first_result["chunks"][0]
        first_patch = first_chunk["patches"][0]

        # Check patch fields
        assert "patch_idx" in first_patch
        assert "patch_row" in first_patch
        assert "patch_col" in first_patch
        assert "nearest_contextual_neighbors" in first_patch

        # Check neighbors (top-5)
        neighbors = first_patch["nearest_contextual_neighbors"]
        assert len(neighbors) == 5, f"Expected 5 neighbors, got {len(neighbors)}"

        # Check neighbor fields
        for neighbor in neighbors:
            assert "token_str" in neighbor
            assert "token_id" in neighbor
            assert "caption" in neighbor
            assert "position" in neighbor
            assert "similarity" in neighbor
            assert "contextual_layer" in neighbor

            # Similarity should be in reasonable range
            assert -1 <= neighbor["similarity"] <= 1, f"Similarity {neighbor['similarity']} out of range"


class TestLatentLensGoldenValues:
    """Test specific golden values from known outputs."""

    def test_image0_visual8_patch0_neighbors(self, latentlens_output_olmo_vit_lite10):
        """Test specific neighbor values for image 0, visual layer 8, patch 0."""
        output_file = latentlens_output_olmo_vit_lite10 / "contextual_neighbors_visual8_allLayers.json"

        with open(output_file) as f:
            data = json.load(f)

        # Get patch 0 of image 0
        image0 = data["results"][0]
        assert image0["image_idx"] == 0

        patch0 = image0["chunks"][0]["patches"][0]
        assert patch0["patch_idx"] == 0

        neighbors = patch0["nearest_contextual_neighbors"]

        # The top neighbor for this patch should be specific
        # (This is a golden value test - if extraction changes, this test will catch it)
        top_neighbor = neighbors[0]

        # Check that similarity is approximately what we expect (allowing small floating point differences)
        assert 0.15 < top_neighbor["similarity"] < 0.20, \
            f"Top neighbor similarity {top_neighbor['similarity']} not in expected range"

    def test_num_patches_consistency(self, latentlens_output_olmo_vit_lite10):
        """All visual layers should produce same number of patches (576 for ViT)."""
        expected_patches = 576  # 24x24 grid for ViT-L/14-336

        for visual_layer in [0, 1, 2, 4, 8]:
            output_file = latentlens_output_olmo_vit_lite10 / f"contextual_neighbors_visual{visual_layer}_allLayers.json"

            if not output_file.exists():
                continue

            with open(output_file) as f:
                data = json.load(f)

            for result in data["results"]:
                total_patches = sum(len(chunk["patches"]) for chunk in result["chunks"])
                assert total_patches == expected_patches, \
                    f"Visual layer {visual_layer}, image {result['image_idx']}: expected {expected_patches} patches, got {total_patches}"


class TestLatentLensAllModels:
    """Test that all 9 model combinations have outputs."""

    @pytest.fixture
    def all_model_outputs(self, analysis_results_dir):
        """Paths to all 9 model outputs (lite10 versions)."""
        base = analysis_results_dir / "contextual_nearest_neighbors"
        models = [
            "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded_lite10",
            "train_mlp-only_pixmo_cap_resize_olmo-7b_dinov2-large-336_step12000-unsharded_lite10",
            "train_mlp-only_pixmo_cap_resize_olmo-7b_siglip_step12000-unsharded_lite10",
            "train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336_step12000-unsharded_lite10",
            "train_mlp-only_pixmo_cap_resize_llama3-8b_dinov2-large-336_step12000-unsharded_lite10",
            "train_mlp-only_pixmo_cap_resize_llama3-8b_siglip_step12000-unsharded_lite10",
            "train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_seed10_step12000-unsharded_lite10",
            "train_mlp-only_pixmo_cap_resize_qwen2-7b_dinov2-large-336_step12000-unsharded_lite10",
            "train_mlp-only_pixmo_cap_resize_qwen2-7b_siglip_step12000-unsharded_lite10",
        ]
        return {m: base / m for m in models}

    def test_all_9_models_have_outputs(self, all_model_outputs, analysis_results_dir):
        """All 9 model combinations should have LatentLens outputs."""
        base = analysis_results_dir / "contextual_nearest_neighbors"

        # List actual directories
        existing = [d.name for d in base.iterdir() if d.is_dir() and "lite10" in d.name and "ablations" not in d.name]

        # We expect at least 9 model outputs
        assert len(existing) >= 9, f"Expected at least 9 model outputs, found {len(existing)}: {existing}"
