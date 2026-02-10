#!/usr/bin/env python3
"""
Extract connector weights from a full checkpoint.

The connector (MLP projector + embeddings) is the only trained component.
LLM and ViT weights are frozen and loaded from pretrained sources.

Usage:
    python scripts/extract_connector.py --checkpoint /path/to/step12000-unsharded --output connectors/olmo-vit.pt

Output: A small (~350MB) file containing only the connector weights.
"""

import argparse
import torch
from pathlib import Path


# Connector parameters to extract
CONNECTOR_PARAMS = [
    "vision_backbone.image_pooling_2d",
    "vision_backbone.image_projector",
    "vision_backbone.cls_projector",
    "vision_backbone.pad_embed",
    "transformer.wte.new_embedding",
]


def is_connector_param(param_name: str) -> bool:
    """Check if parameter is part of the connector."""
    return any(conn in param_name for conn in CONNECTOR_PARAMS)


def extract_connector(checkpoint_path: Path, output_path: Path, verbose: bool = True):
    """Extract connector weights from a full checkpoint."""
    model_path = checkpoint_path / "model.pt"
    config_path = checkpoint_path / "config.yaml"

    if not model_path.exists():
        raise FileNotFoundError(f"No model.pt found at {model_path}")

    if verbose:
        print(f"Loading checkpoint from {checkpoint_path}...")

    state_dict = torch.load(model_path, map_location="cpu")

    # Extract connector weights
    connector_weights = {}
    total_size_mb = 0

    for name, param in state_dict.items():
        if is_connector_param(name):
            connector_weights[name] = param
            size_mb = param.numel() * param.element_size() / (1024 * 1024)
            total_size_mb += size_mb
            if verbose:
                print(f"  {name}: {list(param.shape)} ({size_mb:.2f} MB)")

    if verbose:
        print(f"\nExtracted {len(connector_weights)} parameters ({total_size_mb:.2f} MB)")

    # Save connector weights
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(connector_weights, output_path)

    # Also copy config.yaml if it exists
    if config_path.exists():
        import shutil
        config_output = output_path.parent / "config.yaml"
        shutil.copy(config_path, config_output)
        if verbose:
            print(f"Copied config.yaml to {config_output}")

    if verbose:
        actual_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\nSaved to {output_path} ({actual_size_mb:.1f} MB)")

    return connector_weights


def main():
    parser = argparse.ArgumentParser(description="Extract connector weights from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory (containing model.pt)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for connector weights (.pt file)")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    extract_connector(
        Path(args.checkpoint),
        Path(args.output),
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
