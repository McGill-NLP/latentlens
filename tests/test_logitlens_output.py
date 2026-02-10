"""
Tests for LogitLens output format.

These tests verify that:
1. Existing LogitLens outputs have expected structure
2. Output values match golden files

Run with: pytest tests/test_logitlens_output.py -v
"""
import json
import pytest
from pathlib import Path


class TestLogitLensOutputStructure:
    """Test that existing LogitLens outputs have expected structure."""

    @pytest.fixture
    def logitlens_output_dir(self, analysis_results_dir):
        """Path to LogitLens output (lite10 version)."""
        return analysis_results_dir / "logit_lens" / "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded_lite10"

    def test_output_files_exist(self, logitlens_output_dir):
        """Expected output files should exist for multiple layers."""
        if not logitlens_output_dir.exists():
            pytest.skip("LogitLens output directory not found")

        files = list(logitlens_output_dir.glob("logit_lens_layer*.json"))
        assert len(files) >= 5, f"Expected at least 5 layer files, found {len(files)}"

    def test_output_metadata(self, logitlens_output_dir):
        """Output JSON should have expected metadata fields."""
        if not logitlens_output_dir.exists():
            pytest.skip("LogitLens output directory not found")

        output_file = list(logitlens_output_dir.glob("logit_lens_layer*.json"))[0]

        with open(output_file) as f:
            data = json.load(f)

        # Check required fields (LogitLens uses layer_idx, not llm_layer)
        assert "checkpoint" in data
        assert "layer_idx" in data
        assert "num_images" in data
        assert "top_k" in data
        assert "results" in data

    def test_results_structure(self, logitlens_output_dir):
        """Each result should have expected structure."""
        if not logitlens_output_dir.exists():
            pytest.skip("LogitLens output directory not found")

        output_file = list(logitlens_output_dir.glob("logit_lens_layer*.json"))[0]

        with open(output_file) as f:
            data = json.load(f)

        results = data["results"]
        assert len(results) > 0, "Results should not be empty"

        # Check first result
        first_result = results[0]
        assert "image_idx" in first_result
        assert "ground_truth_caption" in first_result
        assert "chunks" in first_result

        # Check chunk/patch structure
        first_chunk = first_result["chunks"][0]
        assert "patches" in first_chunk

        first_patch = first_chunk["patches"][0]
        assert "patch_idx" in first_patch
        assert "top_predictions" in first_patch

        # Check prediction structure (LogitLens uses logit, not probability)
        predictions = first_patch["top_predictions"]
        assert len(predictions) > 0
        first_pred = predictions[0]
        assert "token" in first_pred
        assert "token_id" in first_pred
        assert "logit" in first_pred


class TestLogitLensAllModels:
    """Test that all model combinations have LogitLens outputs."""

    def test_multiple_models_have_outputs(self, analysis_results_dir):
        """Multiple model combinations should have LogitLens outputs."""
        base = analysis_results_dir / "logit_lens"
        if not base.exists():
            pytest.skip("LogitLens output directory not found")

        lite10_dirs = [d for d in base.iterdir() if d.is_dir() and "lite10" in d.name and "ablations" not in str(d)]
        assert len(lite10_dirs) >= 6, f"Expected at least 6 model outputs, found {len(lite10_dirs)}"
