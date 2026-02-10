#!/bin/bash
# Step 2: Extract contextual embeddings from LLMs
#
# This creates the text embedding caches needed for LatentLens analysis.
# Each LLM takes ~4 hours on a single GPU.
#
# Usage:
#   ./reproduce/step2_extract_contextual.sh           # Run all 4 LLMs
#   ./reproduce/step2_extract_contextual.sh olmo      # Run only OLMo
#   ./reproduce/step2_extract_contextual.sh qwen2vl   # Run only Qwen2-VL

set -e

# Change to repo root
cd "$(dirname "$0")/.."

echo "============================================"
echo "Step 2: Extract Contextual Embeddings"
echo "============================================"

# Directory for outputs
OUTPUT_BASE="contextual_embeddings"
mkdir -p "$OUTPUT_BASE"

# Function to extract embeddings for a model
extract_embeddings() {
    local model=$1
    local output_name=$2
    local layers=$3

    echo ""
    echo "Extracting embeddings for $model..."
    echo "  Output: $OUTPUT_BASE/$output_name"
    echo "  Layers: $layers"
    echo ""

    python reproduce/scripts/extract_embeddings.py \
        --model "$model" \
        --layers $layers \
        --output-dir "$OUTPUT_BASE/$output_name" \
        --num-captions -1
}

# Model configurations
# OLMo-7B, LLaMA3-8B: 32 layers, analyze 1 2 4 8 16 24 30 31
# Qwen2-7B, Qwen2-VL: 28 layers, analyze 1 2 4 8 16 24 26 27

run_olmo() {
    extract_embeddings "allenai/OLMo-7B-1024-preview" "olmo-7b" "1 2 4 8 16 24 30 31"
}

run_llama() {
    extract_embeddings "meta-llama/Meta-Llama-3-8B" "llama3-8b" "1 2 4 8 16 24 30 31"
}

run_qwen() {
    extract_embeddings "Qwen/Qwen2-7B" "qwen2-7b" "1 2 4 8 16 24 26 27"
}

run_qwen2vl() {
    # Qwen2-VL uses a different architecture with its own vision processing.
    # For Qwen2-VL embedding extraction, see the Qwen2-VL documentation.
    echo ""
    echo "NOTE: Qwen2-VL embedding extraction requires a separate script."
    echo "  Qwen2-VL uses a different architecture (native vision processing)"
    echo "  and is not supported by the standard extract_embeddings.py."
    echo "  See the Qwen2-VL documentation for embedding extraction."
    echo ""
}

# Parse command line argument
if [ $# -eq 0 ]; then
    # Run all models
    echo "Running all 4 LLMs (this will take ~16 hours total)..."
    run_olmo
    run_llama
    run_qwen
    run_qwen2vl
else
    case $1 in
        olmo)
            run_olmo
            ;;
        llama)
            run_llama
            ;;
        qwen)
            run_qwen
            ;;
        qwen2vl)
            run_qwen2vl
            ;;
        *)
            echo "Unknown model: $1"
            echo "Usage: $0 [olmo|llama|qwen|qwen2vl]"
            exit 1
            ;;
    esac
fi

echo ""
echo "Step 2 complete!"
echo "Contextual embeddings saved to: $OUTPUT_BASE/"
