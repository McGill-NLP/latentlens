#!/bin/bash
# Step 1: Download pretrained checkpoints
#
# Downloads connector weights from HuggingFace, then downloads and converts
# base LLMs and vision encoders to Molmo format.
#
# Usage:
#   ./reproduce/step1_download.sh                    # Download all
#   ./reproduce/step1_download.sh --connectors-only  # Connectors only (~3GB)

set -e

echo "============================================"
echo "Step 1: Download Data"
echo "============================================"
echo ""

# Change to repo root
cd "$(dirname "$0")/.."

CONNECTORS_ONLY=false
if [ "$1" = "--connectors-only" ]; then
    CONNECTORS_ONLY=true
fi

# --- Part 1: Download connector weights from HuggingFace ---
echo "Downloading connector weights..."
mkdir -p checkpoints
MODELS=("olmo-vit" "olmo-dino" "olmo-siglip" "llama-vit" "llama-dino" "llama-siglip" "qwen-vit" "qwen-dino" "qwen-siglip")

for model in "${MODELS[@]}"; do
    echo "  Downloading $model connector..."
    huggingface-cli download McGill-NLP/latentlens-connectors "$model/connector.pt" "$model/config.yaml" --local-dir checkpoints
done

# Rename connector.pt -> model.pt for each checkpoint (what the scripts expect)
for model in "${MODELS[@]}"; do
    if [ -f "checkpoints/$model/connector.pt" ] && [ ! -f "checkpoints/$model/model.pt" ]; then
        cp "checkpoints/$model/connector.pt" "checkpoints/$model/model.pt"
    fi
done

if [ "$CONNECTORS_ONLY" = true ]; then
    echo ""
    echo "Connectors downloaded to checkpoints/"
    echo "Skipping base models (--connectors-only)."
    exit 0
fi

# --- Part 2: Download and convert base models ---
# The base LLMs and vision encoders are downloaded from HuggingFace and
# converted to Molmo's weight format. This takes ~30-60 min depending on
# network speed and creates ~50GB of .pt files.

echo ""
echo "Downloading and converting base models to Molmo format..."
echo "  This downloads from HuggingFace and converts weights."
echo "  Output: pretrained/ directory (~50GB total)"
echo ""

python reproduce/scripts/convert_pretrained.py --all --output-dir .

echo ""
echo "Step 1 complete!"
echo ""
echo "Directory structure:"
echo "  checkpoints/       - Connector weights + configs (9 models)"
echo "  pretrained/        - Converted base models (3 LLMs + 3 ViTs)"
echo ""
