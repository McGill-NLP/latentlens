#!/usr/bin/env python3
"""
Aggregate evaluation results across models and layers.

This script reads evaluation results from multiple model/layer combinations
and produces a summary table matching the paper's main results.

Usage:
    python aggregate_results.py --eval-dir evaluation/ --output results/aggregated.json

The output format matches results/paper_results.json for comparison.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_evaluation_results(eval_dir):
    """Load all evaluation results from directory structure."""
    eval_dir = Path(eval_dir)
    results = defaultdict(lambda: defaultdict(dict))

    # Expected structure: eval_dir/{analysis_type}/{model}/evaluation_results.json
    for analysis_type_dir in eval_dir.iterdir():
        if not analysis_type_dir.is_dir():
            continue

        analysis_type = analysis_type_dir.name  # e.g., "latentlens", "logitlens", "nn"

        for model_dir in analysis_type_dir.iterdir():
            if not model_dir.is_dir():
                continue

            results_file = model_dir / "evaluation_results.json"
            if not results_file.exists():
                continue

            model_name = model_dir.name  # e.g., "olmo-vit"

            with open(results_file) as f:
                data = json.load(f)

            for layer_result in data:
                layer = str(layer_result["layer"])
                fraction = layer_result.get("interpretable_fraction", 0.0)
                # Convert to percentage
                results[analysis_type][model_name][layer] = round(fraction * 100, 2)

    return results


def compare_to_paper(results, paper_results_path):
    """Compare aggregated results to paper's published numbers."""
    with open(paper_results_path) as f:
        paper = json.load(f)

    print("\n=== Comparison to Paper Results ===\n")

    # Map analysis types
    type_map = {
        "latentlens": "contextual",
        "contextual": "contextual",
        "logitlens": "logitlens",
        "nn": "nn",
        "embedding_lens": "nn",
    }

    for analysis_type, models in results.items():
        paper_key = type_map.get(analysis_type, analysis_type)
        if paper_key not in paper:
            print(f"Skipping {analysis_type}: not in paper results")
            continue

        print(f"\n{analysis_type.upper()}")
        print("-" * 60)

        # Map short model names to paper_results.json names
        model_name_map = {
            "olmo-vit": "olmo-7b+vit-l-14-336",
            "olmo-dino": "olmo-7b+dinov2-large-336",
            "olmo-siglip": "olmo-7b+siglip",
            "llama-vit": "llama3-8b+vit-l-14-336",
            "llama-dino": "llama3-8b+dinov2-large-336",
            "llama-siglip": "llama3-8b+siglip",
            "qwen-vit": "qwen2-7b+vit-l-14-336",
            "qwen-dino": "qwen2-7b+dinov2-large-336",
            "qwen-siglip": "qwen2-7b+siglip",
        }

        for model_name, layers in models.items():
            paper_model = model_name_map.get(model_name, model_name)
            if paper_model not in paper[paper_key]:
                print(f"  {model_name}: not in paper results")
                continue

            print(f"\n  {model_name}:")
            for layer, value in sorted(layers.items(), key=lambda x: int(x[0])):
                paper_value = paper[paper_key].get(paper_model, {}).get(layer)
                if paper_value is not None:
                    diff = value - paper_value
                    status = "OK" if abs(diff) < 5 else "DIFF"
                    print(f"    Layer {layer:>2}: {value:5.1f}% (paper: {paper_value:5.1f}%, diff: {diff:+5.1f}%) [{status}]")
                else:
                    print(f"    Layer {layer:>2}: {value:5.1f}% (paper: N/A)")


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument("--eval-dir", required=True, help="Directory with evaluation results")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--compare", default=None, help="Path to paper_results.json for comparison")

    args = parser.parse_args()

    results = load_evaluation_results(args.eval_dir)

    # Convert to regular dict for JSON serialization
    output = {}
    for analysis_type, models in results.items():
        output[analysis_type] = {}
        for model, layers in models.items():
            output[analysis_type][model] = dict(layers)

    # Save
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Aggregated results saved to: {args.output}")

    # Print summary
    print("\n=== Summary ===\n")
    for analysis_type, models in output.items():
        print(f"{analysis_type}:")
        for model, layers in models.items():
            avg = sum(layers.values()) / len(layers) if layers else 0
            print(f"  {model}: avg {avg:.1f}% across {len(layers)} layers")

    # Compare to paper if requested
    if args.compare:
        compare_to_paper(results, args.compare)


if __name__ == "__main__":
    main()
