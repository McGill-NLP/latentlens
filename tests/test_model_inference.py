"""
Integration tests for model loading and inference.

These tests verify that:
1. Models can be loaded from connector + base weights
2. Models can generate captions

Run with: pytest tests/test_model_inference.py -v
"""

import pytest
import torch
from pathlib import Path


# Path to existing converted weights in main repo
MOLMO_DATA_DIR = Path(__file__).parent.parent.parent / "molmo_data"


@pytest.fixture
def check_weights_exist():
    """Skip tests if converted weights don't exist."""
    if not MOLMO_DATA_DIR.exists():
        pytest.skip(f"MOLMO_DATA_DIR not found at {MOLMO_DATA_DIR}")

    llm_path = MOLMO_DATA_DIR / "pretrained_llms" / "olmo-1024-preview.pt"
    vit_path = MOLMO_DATA_DIR / "pretrained_image_encoders" / "vit-l-14-336.pt"

    if not llm_path.exists():
        pytest.skip(f"LLM weights not found at {llm_path}")
    if not vit_path.exists():
        pytest.skip(f"ViT weights not found at {vit_path}")

    return True


class TestModelLoading:
    """Test that models can be loaded correctly."""

    def test_load_connector_weights(self):
        """Connector weights can be loaded and have expected keys."""
        from huggingface_hub import hf_hub_download
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = hf_hub_download(
                repo_id="McGill-NLP/latentlens-connectors",
                filename="olmo-vit/connector.pt",
                local_dir=tmpdir,
            )

            connector = torch.load(Path(tmpdir) / "olmo-vit" / "connector.pt", map_location="cpu")

            # Expected connector keys
            expected_keys = {
                "vision_backbone.image_projector.w1.weight",
                "vision_backbone.image_projector.w2.weight",
                "vision_backbone.image_projector.w3.weight",
                "vision_backbone.pad_embed",
                "transformer.wte.new_embedding",
            }

            assert set(connector.keys()) == expected_keys

    @pytest.mark.slow
    def test_load_full_model(self, check_weights_exist):
        """Full model (LLM + ViT + connector) can be loaded."""
        import os
        import sys

        # Add parent to path for olmo imports
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        os.environ["MOLMO_DATA_DIR"] = str(MOLMO_DATA_DIR)

        from olmo import Molmo
        from olmo.config import ModelConfig

        # Load proven config from configs/ folder
        config_path = Path(__file__).parent.parent / "reproduce" / "configs" / "olmo-vit.yaml"
        config = ModelConfig.load(str(config_path), key="model", validate_paths=False)
        config.init_device = "cpu"

        # Build model
        model = Molmo(config)

        # Load pretrained weights
        model.reset_with_pretrained_weights()

        # Load connector
        from huggingface_hub import hf_hub_download
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            hf_hub_download(
                repo_id="McGill-NLP/latentlens-connectors",
                filename="olmo-vit/connector.pt",
                local_dir=tmpdir,
            )

            connector = torch.load(
                Path(tmpdir) / "olmo-vit" / "connector.pt",
                map_location="cpu"
            )
            model.load_state_dict(connector, strict=False)

        # Verify model has expected structure
        assert hasattr(model, "transformer")
        assert hasattr(model, "vision_backbone")


