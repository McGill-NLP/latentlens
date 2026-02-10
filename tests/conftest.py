"""
Pytest fixtures and configuration for LatentLens tests.
"""
import os
import pytest
from pathlib import Path


# Paths to original repo data (for golden file comparisons)
# Set ORIGINAL_REPO_ROOT and optionally MOLMO_DATA_DIR env vars to run these tests.
_repo_root = os.environ.get("ORIGINAL_REPO_ROOT", None)
ORIGINAL_REPO_ROOT = Path(_repo_root) if _repo_root else None

_molmo_data = os.environ.get("MOLMO_DATA_DIR", None)
MOLMO_DATA = Path(_molmo_data) if _molmo_data else (ORIGINAL_REPO_ROOT / "molmo_data" if ORIGINAL_REPO_ROOT else None)

ANALYSIS_RESULTS = ORIGINAL_REPO_ROOT / "analysis_results" if ORIGINAL_REPO_ROOT else None


@pytest.fixture
def original_repo_root():
    """Path to the original molmo repo."""
    if ORIGINAL_REPO_ROOT is None or not ORIGINAL_REPO_ROOT.exists():
        pytest.skip("ORIGINAL_REPO_ROOT env var not set or path does not exist")
    return ORIGINAL_REPO_ROOT


@pytest.fixture
def molmo_data_dir():
    """Path to molmo_data directory with checkpoints and contextual embeddings."""
    if MOLMO_DATA is None or not MOLMO_DATA.exists():
        pytest.skip("MOLMO_DATA_DIR env var not set (and ORIGINAL_REPO_ROOT/molmo_data not found)")
    return MOLMO_DATA


@pytest.fixture
def analysis_results_dir():
    """Path to analysis_results directory with existing outputs."""
    if ANALYSIS_RESULTS is None or not ANALYSIS_RESULTS.exists():
        pytest.skip("ORIGINAL_REPO_ROOT env var not set (or analysis_results not found)")
    return ANALYSIS_RESULTS


@pytest.fixture
def contextual_embeddings_olmo(molmo_data_dir):
    """Path to OLMo contextual embeddings."""
    return molmo_data_dir / "contextual_llm_embeddings_vg" / "allenai_OLMo-7B-1024-preview"


@pytest.fixture
def latentlens_output_olmo_vit_lite10(analysis_results_dir):
    """Path to existing LatentLens output (lite10 version for quick tests)."""
    return analysis_results_dir / "contextual_nearest_neighbors" / "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded_lite10"


@pytest.fixture
def checkpoint_olmo_vit(molmo_data_dir):
    """Path to OLMo-ViT checkpoint."""
    return molmo_data_dir / "checkpoints" / "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336" / "step12000-unsharded"
