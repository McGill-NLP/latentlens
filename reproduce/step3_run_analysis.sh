#!/bin/bash
# Step 3: Run interpretability analyses
#
# This runs LatentLens, LogitLens, and EmbeddingLens on all model combinations.
#
# GPU modes:
#   - Single GPU (default): python reproduce/scripts/run_*.py
#   - Multi GPU: Set NPROC_PER_NODE env var (e.g., NPROC_PER_NODE=4 ./reproduce/step3_run_analysis.sh)
#
# Usage:
#   ./reproduce/step3_run_analysis.sh                    # Run all (single GPU)
#   ./reproduce/step3_run_analysis.sh latentlens         # Run only LatentLens
#   ./reproduce/step3_run_analysis.sh logitlens          # Run only LogitLens
#   ./reproduce/step3_run_analysis.sh embedding_lens     # Run only EmbeddingLens
#   NPROC_PER_NODE=4 ./reproduce/step3_run_analysis.sh   # Run all on 4 GPUs
#
# Time estimates (single GPU, A100 80GB, per model, 9 layers, 100 images):
#   LatentLens:     ~45 min  (preloads images to GPU, iterates caches)
#   LogitLens:      ~22 min  (model forward pass per image per layer)
#   EmbeddingLens:  ~22 min  (single forward pass per image, all layers extracted at once)
#   Total per model: ~90 min
#   Total all 9 models: ~13.5 hours

set -e

# Change to repo root
cd "$(dirname "$0")/.."

echo "============================================"
echo "Step 3: Run Interpretability Analysis"
echo "============================================"

# Configuration
NUM_IMAGES=100
OUTPUT_BASE="results"
CHECKPOINT_BASE="checkpoints"
CONTEXTUAL_BASE="contextual_embeddings"

# GPU configuration
# Set NPROC_PER_NODE=N to use multi-GPU with torchrun
NPROC=${NPROC_PER_NODE:-0}  # 0 means single-GPU mode
MASTER_PORT=${MASTER_PORT:-29500}

if [ "$NPROC" -gt 1 ]; then
    RUN_CMD="torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT"
    echo "GPU mode: Multi-GPU ($NPROC GPUs via torchrun)"
else
    RUN_CMD="python"
    echo "GPU mode: Single GPU"
    echo "  (Set NPROC_PER_NODE=4 for multi-GPU)"
fi
echo ""

# Model combinations
# Format: checkpoint_name:contextual_name:layers:hf_model_dir
# hf_model_dir is the HuggingFace model ID with "/" replaced by "_"
# (extract_embeddings.py creates a subdir named after the HF model)
MODELS_32=(
    "olmo-vit:olmo-7b:0,1,2,4,8,16,24,30,31:allenai_OLMo-7B-1024-preview"
    "olmo-dino:olmo-7b:0,1,2,4,8,16,24,30,31:allenai_OLMo-7B-1024-preview"
    "olmo-siglip:olmo-7b:0,1,2,4,8,16,24,30,31:allenai_OLMo-7B-1024-preview"
    "llama-vit:llama3-8b:0,1,2,4,8,16,24,30,31:meta-llama_Meta-Llama-3-8B"
    "llama-dino:llama3-8b:0,1,2,4,8,16,24,30,31:meta-llama_Meta-Llama-3-8B"
    "llama-siglip:llama3-8b:0,1,2,4,8,16,24,30,31:meta-llama_Meta-Llama-3-8B"
)

MODELS_28=(
    "qwen-vit:qwen2-7b:0,1,2,4,8,16,24,26,27:Qwen_Qwen2-7B"
    "qwen-dino:qwen2-7b:0,1,2,4,8,16,24,26,27:Qwen_Qwen2-7B"
    "qwen-siglip:qwen2-7b:0,1,2,4,8,16,24,26,27:Qwen_Qwen2-7B"
)

run_latentlens() {
    echo ""
    echo "Running LatentLens analysis..."
    echo "  (Single GPU only — uses image preloading for efficiency)"

    for model_spec in "${MODELS_32[@]}" "${MODELS_28[@]}"; do
        IFS=':' read -r ckpt ctx layers hf_dir <<< "$model_spec"

        echo "  Processing $ckpt..."
        python reproduce/scripts/run_latentlens.py \
            --ckpt-path "$CHECKPOINT_BASE/$ckpt" \
            --contextual-dir "$CONTEXTUAL_BASE/$ctx/$hf_dir" \
            --visual-layer "$layers" \
            --num-images "$NUM_IMAGES" \
            --output-dir "$OUTPUT_BASE/latentlens/$ckpt"
    done
}

run_logitlens() {
    echo ""
    echo "Running LogitLens analysis..."

    for model_spec in "${MODELS_32[@]}" "${MODELS_28[@]}"; do
        IFS=':' read -r ckpt ctx layers hf_dir <<< "$model_spec"

        echo "  Processing $ckpt..."
        $RUN_CMD reproduce/scripts/run_logitlens.py \
            --ckpt-path "$CHECKPOINT_BASE/$ckpt" \
            --layers "$layers" \
            --num-images "$NUM_IMAGES" \
            --output-dir "$OUTPUT_BASE/logitlens/$ckpt"
    done
}

run_embedding_lens() {
    echo ""
    echo "Running EmbeddingLens analysis..."

    for model_spec in "${MODELS_32[@]}" "${MODELS_28[@]}"; do
        IFS=':' read -r ckpt ctx layers hf_dir <<< "$model_spec"

        echo "  Processing $ckpt..."
        $RUN_CMD reproduce/scripts/run_embedding_lens.py \
            --ckpt-path "$CHECKPOINT_BASE/$ckpt" \
            --llm_layer "$layers" \
            --num-images "$NUM_IMAGES" \
            --output-base-dir "$OUTPUT_BASE/embedding_lens/$ckpt"
    done
}

# Parse command line argument
if [ $# -eq 0 ]; then
    # Run all analyses
    run_latentlens
    run_logitlens
    run_embedding_lens
else
    case $1 in
        latentlens)
            run_latentlens
            ;;
        logitlens)
            run_logitlens
            ;;
        embedding_lens)
            run_embedding_lens
            ;;
        *)
            echo "Unknown analysis: $1"
            echo "Usage: $0 [latentlens|logitlens|embedding_lens]"
            exit 1
            ;;
    esac
fi

echo ""
echo "Step 3 complete!"
echo "Results saved to: $OUTPUT_BASE/"
