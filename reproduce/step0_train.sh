#!/bin/bash
# Step 0: Train ViT-LLM connectors from scratch
#
# Trains the connector MLP for each of the 9 model combinations.
# Requires base models to be downloaded first (step1_download.sh without --connectors-only).
#
# Usage:
#   ./reproduce/step0_train.sh                    # Train all 9 models
#   ./reproduce/step0_train.sh --model olmo-vit   # Train a single model
#
# Environment variables:
#   NPROC_PER_NODE  Number of GPUs per node (default: 4)
#   CUDA_VISIBLE_DEVICES  Which GPUs to use (default: all)
#
# Hardware requirements:
#   - 4× A100 80GB (or equivalent)
#   - ~3-5 hours per model (12,000 steps)

set -e

# Change to repo root
cd "$(dirname "$0")/.."

NPROC=${NPROC_PER_NODE:-4}
SINGLE_MODEL=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            SINGLE_MODEL="$2"
            shift 2
            ;;
        --model=*)
            SINGLE_MODEL="${1#*=}"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Map short model names to config save_folder names
declare -A SAVE_FOLDERS
SAVE_FOLDERS["olmo-vit"]="baseline_pixmo-captions_resize_olmo-7b_vit-l-14-336"
SAVE_FOLDERS["olmo-dino"]="baseline_pixmo-captions_resize_olmo-7b_dinov2-large-336"
SAVE_FOLDERS["olmo-siglip"]="baseline_pixmo-captions_resize_olmo-7b_siglip"
SAVE_FOLDERS["llama-vit"]="baseline_pixmo-captions_resize_llama3-8b_vit-l-14-336"
SAVE_FOLDERS["llama-dino"]="baseline_pixmo-captions_resize_llama3-8b_dinov2-large-336"
SAVE_FOLDERS["llama-siglip"]="baseline_pixmo-captions_resize_llama3-8b_siglip"
SAVE_FOLDERS["qwen-vit"]="baseline_pixmo-captions_resize_qwen2-7b_vit-l-14-336_seed10"
SAVE_FOLDERS["qwen-dino"]="baseline_pixmo-captions_resize_qwen2-7b_dinov2-large-336"
SAVE_FOLDERS["qwen-siglip"]="baseline_pixmo-captions_resize_qwen2-7b_siglip"

ALL_MODELS=("olmo-vit" "olmo-dino" "olmo-siglip" "llama-vit" "llama-dino" "llama-siglip" "qwen-vit" "qwen-dino" "qwen-siglip")

if [ -n "$SINGLE_MODEL" ]; then
    MODELS=("$SINGLE_MODEL")
    # Validate model name
    valid=false
    for m in "${ALL_MODELS[@]}"; do
        if [ "$m" = "$SINGLE_MODEL" ]; then
            valid=true
            break
        fi
    done
    if [ "$valid" = false ]; then
        echo "Error: Unknown model '$SINGLE_MODEL'"
        echo "Valid models: ${ALL_MODELS[*]}"
        exit 1
    fi
else
    MODELS=("${ALL_MODELS[@]}")
fi

echo "============================================"
echo "Step 0: Train Connectors"
echo "============================================"
echo ""
echo "Models to train: ${MODELS[*]}"
echo "GPUs per node: $NPROC"
echo ""

# Check that base models exist
if [ ! -d "pretrained" ]; then
    echo "Error: pretrained/ directory not found."
    echo "Run step1_download.sh first (without --connectors-only) to download base models."
    exit 1
fi

train_model() {
    local model=$1
    local config="reproduce/configs/${model}.yaml"
    local save_name="${SAVE_FOLDERS[$model]}"
    local checkpoint_dir="./checkpoints/${save_name}"

    if [ ! -f "$config" ]; then
        echo "Error: Config not found: $config"
        return 1
    fi

    echo "--------------------------------------------"
    echo "Training: $model"
    echo "Config:   $config"
    echo "--------------------------------------------"

    torchrun --nproc_per_node="$NPROC" \
        reproduce/scripts/train_connector.py "$config"

    echo ""
    echo "Training complete for $model"

    # Find the latest unsharded checkpoint
    local latest_step
    latest_step=$(ls -d "${checkpoint_dir}"/step*-unsharded 2>/dev/null | sort -t'p' -k2 -n | tail -1)

    if [ -z "$latest_step" ]; then
        echo "Warning: No unsharded checkpoint found in ${checkpoint_dir}"
        echo "Skipping connector extraction for $model"
        return 0
    fi

    # Extract connector weights
    echo "Extracting connector weights for $model from ${latest_step}..."
    python scripts/extract_connector.py \
        --checkpoint "$latest_step" \
        --output "connectors/${model}.pt"
    echo "Connector extracted: connectors/${model}.pt"
    echo ""
}

for model in "${MODELS[@]}"; do
    train_model "$model"
done

echo ""
echo "============================================"
echo "Training complete!"
echo "============================================"
echo ""
echo "Trained connectors saved to connectors/"
echo ""
echo "Next steps:"
echo "  1. Run step2_extract_contextual.sh to extract contextual embeddings"
echo "  2. Run step3_run_analysis.sh to run interpretability analysis"
echo ""
