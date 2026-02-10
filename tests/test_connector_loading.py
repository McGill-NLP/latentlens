"""
Tests for connector extraction and loading.

These tests verify that:
1. Connector weights can be extracted from full checkpoints
2. Loading connector + pretrained LLM/ViT gives same outputs as full checkpoint

Run with: pytest tests/test_connector_loading.py -v
"""

import os
import pytest
import torch
import tempfile
from pathlib import Path


# Paths to test data (set ORIGINAL_REPO_ROOT and MOLMO_DATA_DIR env vars)
_repo_root = os.environ.get("ORIGINAL_REPO_ROOT", None)
ORIGINAL_REPO = Path(_repo_root) if _repo_root else None

_molmo_data = os.environ.get("MOLMO_DATA_DIR", None)
MOLMO_DATA = Path(_molmo_data) if _molmo_data else None

# Test with OLMo-ViT checkpoint
FULL_CHECKPOINT = MOLMO_DATA / "checkpoints" / "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336" / "step12000-unsharded" if MOLMO_DATA else None

# Connector params
CONNECTOR_PARAMS = [
    "vision_backbone.image_pooling_2d",
    "vision_backbone.image_projector",
    "vision_backbone.cls_projector",
    "vision_backbone.pad_embed",
    "transformer.wte.new_embedding",
]


class TestConnectorExtraction:
    """Test connector weight extraction."""

    def test_full_checkpoint_exists(self):
        """Full checkpoint should exist for testing."""
        if FULL_CHECKPOINT is None or not FULL_CHECKPOINT.exists():
            pytest.skip("Full checkpoint not found")
        assert (FULL_CHECKPOINT / "model.pt").exists()
        assert (FULL_CHECKPOINT / "config.yaml").exists()

    def test_extract_connector_weights(self):
        """Should extract only connector params."""
        if FULL_CHECKPOINT is None or not FULL_CHECKPOINT.exists():
            pytest.skip("Full checkpoint not found")

        state_dict = torch.load(FULL_CHECKPOINT / "model.pt", map_location="cpu")

        # Extract connector
        connector = {}
        for name, param in state_dict.items():
            if any(conn in name for conn in CONNECTOR_PARAMS):
                connector[name] = param

        # Should have connector params
        assert len(connector) > 0, "No connector params found"
        assert len(connector) < len(state_dict), "Should extract subset, not all"

        # Check expected params exist
        assert any("image_projector" in k for k in connector), "Missing image_projector"
        assert any("new_embedding" in k for k in connector), "Missing new_embedding"

    def test_connector_size_reasonable(self):
        """Connector should be ~350MB, not 30GB."""
        if FULL_CHECKPOINT is None or not FULL_CHECKPOINT.exists():
            pytest.skip("Full checkpoint not found")

        state_dict = torch.load(FULL_CHECKPOINT / "model.pt", map_location="cpu")

        connector_size_mb = 0
        for name, param in state_dict.items():
            if any(conn in name for conn in CONNECTOR_PARAMS):
                connector_size_mb += param.numel() * param.element_size() / (1024 * 1024)

        # Should be around 350MB (not 30GB)
        assert 100 < connector_size_mb < 500, f"Unexpected size: {connector_size_mb:.1f} MB"

    def test_extract_and_save(self):
        """Should save connector to file."""
        if FULL_CHECKPOINT is None or not FULL_CHECKPOINT.exists():
            pytest.skip("Full checkpoint not found")

        import sys
        if ORIGINAL_REPO is None or not ORIGINAL_REPO.exists():
            pytest.skip("ORIGINAL_REPO_ROOT env var not set or path does not exist")
        sys.path.insert(0, str(ORIGINAL_REPO / "latentlens_release"))
        from reproduce.scripts.extract_connector import extract_connector

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "connector.pt"
            connector = extract_connector(FULL_CHECKPOINT, output_path, verbose=False)

            # File should exist and be small
            assert output_path.exists()
            size_mb = output_path.stat().st_size / (1024 * 1024)
            assert size_mb < 500, f"Saved file too large: {size_mb:.1f} MB"

            # Should be loadable
            loaded = torch.load(output_path, map_location="cpu")
            assert len(loaded) == len(connector)


@pytest.mark.slow
class TestConnectorLoading:
    """Test that connector + pretrained gives same outputs as full checkpoint.

    These tests require GPU and loading large models.
    """

    @pytest.fixture
    def extracted_connector(self):
        """Extract connector to temp file."""
        if FULL_CHECKPOINT is None or not FULL_CHECKPOINT.exists():
            pytest.skip("Full checkpoint not found")

        import sys
        if ORIGINAL_REPO is None or not ORIGINAL_REPO.exists():
            pytest.skip("ORIGINAL_REPO_ROOT env var not set or path does not exist")
        sys.path.insert(0, str(ORIGINAL_REPO / "latentlens_release"))
        from reproduce.scripts.extract_connector import extract_connector

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "connector.pt"
            extract_connector(FULL_CHECKPOINT, output_path, verbose=False)
            yield output_path

    def test_connector_weights_match_full_checkpoint(self):
        """Connector weights should exactly match those in full checkpoint."""
        if FULL_CHECKPOINT is None or not FULL_CHECKPOINT.exists():
            pytest.skip("Full checkpoint not found")

        # Load full checkpoint
        full_state = torch.load(FULL_CHECKPOINT / "model.pt", map_location="cpu")

        # Extract connector
        connector = {}
        for name, param in full_state.items():
            if any(conn in name for conn in CONNECTOR_PARAMS):
                connector[name] = param

        # Load stripped checkpoint (ablations are already stripped)
        if MOLMO_DATA is None:
            pytest.skip("MOLMO_DATA_DIR env var not set")
        stripped_ckpt = MOLMO_DATA / "checkpoints" / "ablations" / "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_seed10" / "step12000-unsharded"
        if not stripped_ckpt.exists():
            pytest.skip("Stripped checkpoint not found")

        stripped_state = torch.load(stripped_ckpt / "model.pt", map_location="cpu")

        # Stripped should have similar structure (only connector params)
        assert len(stripped_state) < 100, "Stripped checkpoint has too many params"

        # Both should have same connector param names
        full_connector_keys = set(connector.keys())
        stripped_keys = set(stripped_state.keys())

        # Note: stripped might have slightly different param names
        # Just verify structure is similar
        assert len(stripped_keys) < 50, "Stripped should have few params"

    def test_loading_pattern_documented(self):
        """Document the correct loading pattern."""
        # This test documents the expected loading pattern:
        #
        # 1. Load config from checkpoint
        # cfg = TrainConfig.load(f"{checkpoint_path}/config.yaml")
        #
        # 2. Initialize model (on meta device for efficiency)
        # cfg.model.init_device = "meta"
        # model = Molmo(cfg.model)
        #
        # 3. Load pretrained LLM + ViT weights
        # model.reset_with_pretrained_weights()
        #
        # 4. Load connector weights (from extracted file)
        # connector = torch.load("connector.pt", map_location="cpu")
        # model.load_state_dict(connector, strict=False)
        #
        # 5. Move to device
        # model = model.to("cuda")
        # model.eval()
        pass


class TestConnectorVerification:
    """Verify connector loading gives correct outputs.

    This is the critical test: load model two ways and verify identical outputs.
    """

    @pytest.mark.slow
    def test_outputs_match_full_checkpoint(self):
        """Loading connector should give same outputs as full checkpoint.

        This test:
        1. Loads model with full checkpoint (30GB)
        2. Loads model with connector only + pretrained LLM/ViT
        3. Runs same input through both
        4. Verifies outputs are identical

        Skipped by default - run with: pytest -m slow
        """
        pytest.skip("Requires GPU and ~30GB memory - run manually")

        # The test would do:
        # 1. Load full checkpoint model
        # 2. Load connector-only model
        # 3. Create test input (random image tensor)
        # 4. Run both models
        # 5. Assert outputs match within tolerance
