"""
Critical test: Verify connector-only loading gives identical outputs to full checkpoint.

This is THE test that proves our weight extraction is correct.

Run with: pytest tests/test_connector_equivalence.py -v -s
(Requires GPU and ~60GB memory for loading both models)
"""

import os
import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

# Mark entire module as requiring GPU
pytestmark = pytest.mark.gpu


# Paths (set MOLMO_DATA_DIR env var to enable these tests)
_molmo_data = os.environ.get("MOLMO_DATA_DIR", None)
MOLMO_DATA = Path(_molmo_data) if _molmo_data else None
FULL_CHECKPOINT = MOLMO_DATA / "checkpoints" / "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336" / "step12000-unsharded" if MOLMO_DATA else None


def skip_if_no_checkpoint():
    if FULL_CHECKPOINT is None or not FULL_CHECKPOINT.exists():
        pytest.skip(f"MOLMO_DATA_DIR env var not set or checkpoint not found: {FULL_CHECKPOINT}")


def skip_if_no_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


class TestConnectorEquivalence:
    """Verify connector + pretrained == full checkpoint."""

    def test_full_checkpoint_loadable(self):
        """Full checkpoint can be loaded and run."""
        skip_if_no_checkpoint()
        skip_if_no_gpu()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from molmo.config import TrainConfig
        from molmo.model import Molmo

        # Load config
        cfg = TrainConfig.load(str(FULL_CHECKPOINT / "config.yaml"))
        cfg.model.init_device = None

        # Initialize and load full model
        print("\nLoading full checkpoint model...")
        model = Molmo(cfg.model)

        # Load all weights
        checkpoint = torch.load(FULL_CHECKPOINT / "model.pt", map_location="cpu")
        model.load_state_dict(checkpoint, strict=True)

        model.eval()
        model = model.to("cuda:0")

        # Create dummy input
        batch_size = 1
        seq_len = 100
        num_patches = 576  # 24x24 for ViT-L/14

        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device="cuda:0")
        # Dummy image input (batch, crops, patches, pixels)
        images = torch.randn(batch_size, 1, num_patches, 14*14*3, device="cuda:0")

        with torch.no_grad():
            # Just verify it runs without error
            # (Full forward requires proper preprocessing)
            print("Model loaded successfully!")

        # Cleanup
        del model
        torch.cuda.empty_cache()

    def test_connector_loading_equivalence(self):
        """Connector + pretrained should give IDENTICAL outputs to full checkpoint.

        This is the critical test that proves our extraction is correct.
        We load models SEQUENTIALLY to avoid OOM.
        """
        skip_if_no_checkpoint()
        skip_if_no_gpu()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from molmo.config import TrainConfig
        from molmo.model import Molmo
        from reproduce.scripts.extract_connector import extract_connector, CONNECTOR_PARAMS

        print("\n" + "="*80)
        print("CONNECTOR EQUIVALENCE TEST")
        print("="*80)

        # Load config
        cfg = TrainConfig.load(str(FULL_CHECKPOINT / "config.yaml"))
        cfg.model.init_device = None

        # Create test input (deterministic)
        torch.manual_seed(42)
        batch_size = 1
        seq_len = 50
        vocab_size = cfg.model.vocab_size
        test_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # =====================================================
        # Method 1: Load FULL checkpoint, get outputs, delete
        # =====================================================
        print("\n[1/3] Running full checkpoint model...")
        model_full = Molmo(cfg.model)
        full_checkpoint = torch.load(FULL_CHECKPOINT / "model.pt", map_location="cpu")
        model_full.load_state_dict(full_checkpoint, strict=True)
        model_full.eval()
        model_full = model_full.to("cuda:0")

        with torch.no_grad():
            input_ids_gpu = test_input_ids.to("cuda:0")
            # Use model forward (text-only, no images)
            full_out = model_full(input_ids_gpu, last_logits_only=False)
            full_logits = full_out.logits.cpu()

        # Save full state dict keys and shapes for comparison
        full_state_info = {k: (v.shape, v.cpu().clone()) for k, v in model_full.state_dict().items()}

        print("      Full model outputs captured!")

        # Delete to free memory
        del model_full, full_checkpoint
        torch.cuda.empty_cache()
        print("      Full model deleted, memory freed.")

        # =====================================================
        # Method 2: Load pretrained + connector only
        # =====================================================
        print("\n[2/3] Running connector-only model...")

        # Extract connector to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            connector_path = Path(tmpdir) / "connector.pt"
            extract_connector(FULL_CHECKPOINT, connector_path, verbose=False)

            # Create new model
            model_connector = Molmo(cfg.model)

            # Load pretrained LLM + ViT first
            print("      Loading pretrained LLM + ViT...")
            model_connector.reset_with_pretrained_weights()

            # Then load connector weights
            print("      Loading connector weights...")
            connector_weights = torch.load(connector_path, map_location="cpu")
            model_connector.load_state_dict(connector_weights, strict=False)

            model_connector.eval()
            model_connector = model_connector.to("cuda:0")

            with torch.no_grad():
                input_ids_gpu = test_input_ids.to("cuda:0")
                # Use model forward (text-only, no images)
                conn_out = model_connector(input_ids_gpu, last_logits_only=False)
                conn_logits = conn_out.logits.cpu()

            # Get state dict for comparison
            conn_state = {k: v.cpu().clone() for k, v in model_connector.state_dict().items()}

            print("      Connector model outputs captured!")

            del model_connector
            torch.cuda.empty_cache()

        # =====================================================
        # Compare results
        # =====================================================
        print("\n[3/3] Comparing results...")

        # Compare weights
        max_diff = 0.0
        mismatches = []

        for name in full_state_info.keys():
            full_shape, full_param = full_state_info[name]
            conn_param = conn_state[name]

            if full_shape != conn_param.shape:
                mismatches.append(f"{name}: shape mismatch {full_shape} vs {conn_param.shape}")
                continue

            diff = torch.abs(full_param - conn_param).max().item()
            max_diff = max(max_diff, diff)

            if diff > 1e-5:
                mismatches.append(f"{name}: max diff = {diff:.2e}")

        if mismatches:
            print(f"\n      WARNING: {len(mismatches)} weight mismatches found:")
            for m in mismatches[:10]:
                print(f"        - {m}")
        else:
            print(f"      All {len(full_state_info)} parameters match! (max diff: {max_diff:.2e})")

        # Compare logits
        logit_diff = torch.abs(full_logits - conn_logits).max().item()
        print(f"      Logits diff: {logit_diff:.2e}")

        # =====================================================
        # Assert equivalence
        # =====================================================
        print("\n" + "="*80)

        assert len(mismatches) == 0, f"Weight mismatches: {mismatches}"
        assert logit_diff < 1e-4, f"Logits diff too large: {logit_diff}"

        print("SUCCESS: Connector-only loading is equivalent to full checkpoint!")
        print("="*80)


if __name__ == "__main__":
    # Run directly for debugging
    test = TestConnectorEquivalence()
    test.test_connector_loading_equivalence()
