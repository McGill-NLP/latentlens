#!/bin/bash
# Run all reproduction steps
#
# This script runs the full pipeline to reproduce paper results.
# Total time: ~24-48 hours depending on hardware.
#
# Usage:
#   ./reproduce/run_all.sh                # Run everything
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

for arg in "$@"; do
    case $arg in
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

# Step 1: Download
if [ "$SKIP_STEP1" = false ]; then
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
