#!/bin/bash
# Run all reproduction steps
#
# This script runs the full pipeline to reproduce paper results.
# Total time: ~24-48 hours depending on hardware.
#
# Usage:
#   ./reproduce/run_all.sh                # Run everything (download pretrained connectors)
#   ./reproduce/run_all.sh --train        # Train connectors from scratch instead of downloading
#   ./reproduce/run_all.sh --skip-step1   # Skip download (if data is ready)
#   ./reproduce/run_all.sh --skip-step2   # Skip embedding extraction (if cached)

set -e

echo "============================================"
echo "LatentLens: Full Reproduction Pipeline"
echo "============================================"
echo ""
echo "This will run all steps to reproduce paper results."
echo "Estimated time: 24-48 hours"
echo ""

# Parse arguments
SKIP_STEP1=false
SKIP_STEP2=false
TRAIN_FROM_SCRATCH=false

for arg in "$@"; do
    case $arg in
        --train)
            TRAIN_FROM_SCRATCH=true
            ;;
        --skip-step1)
            SKIP_STEP1=true
            ;;
        --skip-step2)
            SKIP_STEP2=true
            ;;
    esac
done

# Change to repo root
cd "$(dirname "$0")/.."

# Step 0/1: Train or Download
if [ "$TRAIN_FROM_SCRATCH" = true ]; then
    echo "Training connectors from scratch (step 0)..."
    # Download base models (LLMs + ViTs) but not pretrained connectors
    ./reproduce/step1_download.sh
    ./reproduce/step0_train.sh
elif [ "$SKIP_STEP1" = false ]; then
    ./reproduce/step1_download.sh
else
    echo "Skipping Step 1 (download)"
fi

# Step 2: Extract contextual embeddings
if [ "$SKIP_STEP2" = false ]; then
    ./reproduce/step2_extract_contextual.sh
else
    echo "Skipping Step 2 (contextual embeddings)"
fi

# Step 3: Run analyses
./reproduce/step3_run_analysis.sh

echo ""
echo "============================================"
echo "Reproduction complete!"
echo "============================================"
echo ""
echo "Results:"
echo "  - Analysis results: results/"
echo ""
