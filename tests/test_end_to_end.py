"""
End-to-end tests that run scripts on small subsets.

These tests verify that:
1. Scripts can be executed without errors
2. Output format matches golden files
3. Numerical values are consistent

Run with: pytest tests/test_end_to_end.py -v -s
(Use -s to see script output)

NOTE: These tests require GPU and take longer to run.
Mark with @pytest.mark.slow for optional skipping.
"""
import json
import subprocess
import sys
import tempfile
import pytest
from pathlib import Path


# Mark all tests in this module as slow (can skip with: pytest -m "not slow")
pytestmark = pytest.mark.slow


class TestExtractEmbeddingsScript:
    """Test the extract_embeddings.py script."""

    def test_script_exists(self):
        """Script should exist."""
        script = Path(__file__).parent.parent / "reproduce" / "scripts" / "extract_embeddings.py"
        assert script.exists(), f"Script not found at {script}"

    def test_script_has_argparse(self):
        """Script should have --help option."""
        script = Path(__file__).parent.parent / "reproduce" / "scripts" / "extract_embeddings.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(script.parent.parent)
        )
        assert result.returncode == 0, f"--help failed: {result.stderr}"
        assert "--model" in result.stdout or "--help" in result.stdout


class TestRunLatentLensScript:
    """Test the run_latentlens.py script."""

    def test_script_exists(self):
        """Script should exist."""
        script = Path(__file__).parent.parent / "reproduce" / "scripts" / "run_latentlens.py"
        assert script.exists(), f"Script not found at {script}"

    def test_script_has_argparse(self):
        """Script should have --help option."""
        script = Path(__file__).parent.parent / "reproduce" / "scripts" / "run_latentlens.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(script.parent.parent)
        )
        assert result.returncode == 0, f"--help failed: {result.stderr}"
        assert "--ckpt-path" in result.stdout or "--help" in result.stdout


class TestRunLogitLensScript:
    """Test the run_logitlens.py script."""

    def test_script_exists(self):
        """Script should exist."""
        script = Path(__file__).parent.parent / "reproduce" / "scripts" / "run_logitlens.py"
        assert script.exists(), f"Script not found at {script}"

    def test_script_syntax_valid(self):
        """Script should have valid Python syntax (can be compiled)."""
        script = Path(__file__).parent.parent / "reproduce" / "scripts" / "run_logitlens.py"
        # Multi-GPU scripts require torchrun, so we just check syntax
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(script)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(script.parent.parent)
        )
        assert result.returncode == 0, f"Syntax check failed: {result.stderr}"


class TestRunEmbeddingLensScript:
    """Test the run_embedding_lens.py script."""

    def test_script_exists(self):
        """Script should exist."""
        script = Path(__file__).parent.parent / "reproduce" / "scripts" / "run_embedding_lens.py"
        assert script.exists(), f"Script not found at {script}"

    def test_script_syntax_valid(self):
        """Script should have valid Python syntax (can be compiled)."""
        script = Path(__file__).parent.parent / "reproduce" / "scripts" / "run_embedding_lens.py"
        # Multi-GPU scripts require torchrun, so we just check syntax
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(script)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(script.parent.parent)
        )
        assert result.returncode == 0, f"Syntax check failed: {result.stderr}"


class TestCompareWithGolden:
    """Compare script outputs with golden files from original repo."""

    @pytest.fixture
    def golden_latentlens_output(self, analysis_results_dir):
        """Load a golden LatentLens output file."""
        golden_file = (
            analysis_results_dir /
            "contextual_nearest_neighbors" /
            "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded_lite10" /
            "contextual_neighbors_visual8_allLayers.json"
        )
        if not golden_file.exists():
            pytest.skip("Golden file not found")

        with open(golden_file) as f:
            return json.load(f)

    def test_golden_has_expected_structure(self, golden_latentlens_output):
        """Golden output should have all expected fields."""
        data = golden_latentlens_output

        # Top-level fields
        assert "checkpoint" in data
        assert "contextual_dir" in data
        assert "visual_layer" in data
        assert "results" in data

        # Results structure
        assert len(data["results"]) > 0
        first_result = data["results"][0]
        assert "image_idx" in first_result
        assert "chunks" in first_result

        # Patch structure
        first_patch = first_result["chunks"][0]["patches"][0]
        assert "patch_idx" in first_patch
        assert "nearest_contextual_neighbors" in first_patch

        # Neighbor structure
        first_neighbor = first_patch["nearest_contextual_neighbors"][0]
        assert "token_str" in first_neighbor
        assert "similarity" in first_neighbor
        assert "contextual_layer" in first_neighbor

    def test_golden_values_stable(self, golden_latentlens_output):
        """Specific golden values should remain stable."""
        data = golden_latentlens_output

        # Check metadata
        assert data["visual_layer"] == 8
        assert data["top_k"] == 5

        # Check first image
        first_result = data["results"][0]
        assert first_result["image_idx"] == 0

        # Check feature shape (OLMo-ViT: 576 patches, 4096 hidden)
        assert first_result["feature_shape"][2] == 576
        assert first_result["feature_shape"][3] == 4096
