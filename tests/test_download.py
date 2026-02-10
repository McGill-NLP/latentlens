"""
Tests for model download functionality.

Run with: pytest tests/test_download.py -v
"""

import pytest
import torch
import tempfile
from pathlib import Path


class TestConnectorDownload:
    """Test downloading connectors from HuggingFace."""

    def test_download_single_connector(self):
        """Can download a single connector."""
        from huggingface_hub import hf_hub_download

        with tempfile.TemporaryDirectory() as tmpdir:
            # Download olmo-vit connector
            path = hf_hub_download(
                repo_id="McGill-NLP/latentlens-connectors",
                filename="olmo-vit/connector.pt",
                local_dir=tmpdir,
            )

            # Verify file exists
            connector_path = Path(tmpdir) / "olmo-vit" / "connector.pt"
            assert connector_path.exists(), f"Connector not found at {connector_path}"

            # Verify it's loadable
            connector = torch.load(connector_path, map_location="cpu")
            assert len(connector) == 5, f"Expected 5 params, got {len(connector)}"

            # Check expected keys
            assert any("image_projector" in k for k in connector)
            assert any("new_embedding" in k for k in connector)

    def test_connector_has_correct_shapes(self):
        """Downloaded connector has expected shapes."""
        from huggingface_hub import hf_hub_download

        with tempfile.TemporaryDirectory() as tmpdir:
            path = hf_hub_download(
                repo_id="McGill-NLP/latentlens-connectors",
                filename="olmo-vit/connector.pt",
                local_dir=tmpdir,
            )

            connector = torch.load(Path(tmpdir) / "olmo-vit" / "connector.pt", map_location="cpu")

            # OLMo-ViT specific shapes
            assert connector["transformer.wte.new_embedding"].shape == torch.Size([128, 4096])
            assert connector["vision_backbone.pad_embed"].shape == torch.Size([2, 2048])

    def test_all_connectors_listed(self):
        """All 9 connectors are available."""
        from huggingface_hub import HfApi

        api = HfApi()
        files = list(api.list_repo_files("McGill-NLP/latentlens-connectors"))

        expected_models = [
            "olmo-vit", "olmo-dino", "olmo-siglip",
            "llama-vit", "llama-dino", "llama-siglip",
            "qwen-vit", "qwen-dino", "qwen-siglip",
        ]

        for model in expected_models:
            assert f"{model}/connector.pt" in files, f"Missing {model}/connector.pt"


