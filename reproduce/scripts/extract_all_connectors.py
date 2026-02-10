#!/usr/bin/env python3
"""
Extract connector weights from all 9 main model checkpoints.

Creates a directory structure ready for HuggingFace upload:
    connectors/
    ├── olmo-vit/
    │   ├── connector.pt
    │   └── config.yaml
    ├── olmo-dino/
    │   └── ...
    └── ...

Usage:
    python scripts/extract_all_connectors.py --output connectors/
"""

import argparse
import os
import shutil
import torch
from pathlib import Path

# Base path for checkpoints (override with MOLMO_CHECKPOINTS_DIR env variable)
CHECKPOINTS_BASE = Path(os.environ.get("MOLMO_CHECKPOINTS_DIR", "./checkpoints"))

# The 9 main model configurations
MODELS = {
    "olmo-vit": "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336",
    "olmo-dino": "train_mlp-only_pixmo_cap_resize_olmo-7b_dinov2-large-336",
    "olmo-siglip": "train_mlp-only_pixmo_cap_resize_olmo-7b_siglip",
    "llama-vit": "train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336",
    "llama-dino": "train_mlp-only_pixmo_cap_resize_llama3-8b_dinov2-large-336",
    "llama-siglip": "train_mlp-only_pixmo_cap_resize_llama3-8b_siglip",
    "qwen-vit": "train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_seed10",
    "qwen-dino": "train_mlp-only_pixmo_cap_resize_qwen2-7b_dinov2-large-336",
    "qwen-siglip": "train_mlp-only_pixmo_cap_resize_qwen2-7b_siglip",
}

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


def extract_connector(checkpoint_path: Path, output_dir: Path, model_name: str):
    """Extract connector from a checkpoint."""
    model_path = checkpoint_path / "model.pt"
    config_path = checkpoint_path / "config.yaml"

    if not model_path.exists():
        print(f"  ERROR: {model_path} not found")
        return False

    print(f"  Loading {model_name}...")
    state_dict = torch.load(model_path, map_location="cpu")

    # Extract connector weights
    connector = {}
    total_size_mb = 0

    for name, param in state_dict.items():
        if is_connector_param(name):
            connector[name] = param
            total_size_mb += param.numel() * param.element_size() / (1024 * 1024)

    # Create output directory
    model_output = output_dir / model_name
    model_output.mkdir(parents=True, exist_ok=True)

    # Save connector
    connector_path = model_output / "connector.pt"
    torch.save(connector, connector_path)

    # Copy config
    if config_path.exists():
        shutil.copy(config_path, model_output / "config.yaml")

    # Create a README for the model
    readme_content = f"""# LatentLens Connector: {model_name}

This contains the trained MLP connector weights for the {model_name} configuration.

## Model Info

- **LLM**: {model_name.split('-')[0].upper()}
- **Vision Encoder**: {model_name.split('-')[1].upper()}
- **Connector Size**: {total_size_mb:.1f} MB
- **Parameters**: {len(connector)}

## Usage

```python
from molmo import LatentLens

# Load model with connector
model = LatentLens.from_pretrained("McGill-NLP/latentlens-{model_name}")

# Or load connector manually
import torch
connector = torch.load("connector.pt")
```

## Files

- `connector.pt` - Trained connector weights
- `config.yaml` - Model configuration (for reference)

## Citation

```bibtex
@article{{krojer2026latentlens,
  title={{LatentLens: Revealing Highly Interpretable Visual Tokens in LLMs}},
  author={{Krojer, Benno and Nayak, Shravan and others}},
  year={{2026}}
}}
```
"""
    with open(model_output / "README.md", "w") as f:
        f.write(readme_content)

    actual_size = connector_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {connector_path} ({actual_size:.1f} MB, {len(connector)} params)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Extract all connectors")
    parser.add_argument("--output", type=str, default="connectors",
                        help="Output directory")
    parser.add_argument("--models", type=str, nargs="*",
                        help="Specific models to extract (default: all)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_extract = args.models if args.models else list(MODELS.keys())

    print("=" * 60)
    print("EXTRACTING CONNECTORS")
    print("=" * 60)

    success = 0
    failed = 0

    for model_name in models_to_extract:
        if model_name not in MODELS:
            print(f"\nUnknown model: {model_name}")
            failed += 1
            continue

        checkpoint_dir = MODELS[model_name]
        checkpoint_path = CHECKPOINTS_BASE / checkpoint_dir / "step12000-unsharded"

        print(f"\n[{model_name}]")

        if extract_connector(checkpoint_path, output_dir, model_name):
            success += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"Done! Extracted {success}/{success + failed} connectors")
    print(f"Output: {output_dir.absolute()}")
    print("=" * 60)

    # Print summary
    print("\nConnector sizes:")
    for model_name in sorted(MODELS.keys()):
        connector_file = output_dir / model_name / "connector.pt"
        if connector_file.exists():
            size_mb = connector_file.stat().st_size / (1024 * 1024)
            print(f"  {model_name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
