# LLM Judge Evaluation Scripts
#
# This module contains scripts for evaluating the interpretability of
# visual tokens using an LLM judge (GPT-4o or similar).
#
# Main scripts:
#   - evaluate_interpretability.py: Run LLM judge on analysis results
#   - aggregate_results.py: Combine results across models/layers
#
# The evaluation methodology:
#   1. For each patch, show the LLM the image with a red bounding box
#   2. Provide the top-5 nearest neighbor tokens from analysis
#   3. LLM judges if tokens are semantically related to the patch
#   4. Compute % interpretable across patches
