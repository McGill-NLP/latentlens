#!/usr/bin/env python3
"""
Evaluate interpretability of visual tokens using an LLM judge.

This script takes analysis results from LatentLens, LogitLens, or EmbeddingLens
and uses an LLM to evaluate whether the top-5 nearest neighbor tokens/words are
semantically related to each image patch.

The method is auto-detected from the JSON structure:
  - LatentLens:    results[].chunks[].patches[].nearest_contextual_neighbors[]
  - LogitLens:     results[].chunks[].patches[].top_predictions[]
  - EmbeddingLens: splits.validation.images[].chunks[].patches[].nearest_neighbors[]

For LatentLens, subword tokens are expanded to full words using the caption context
(e.g., "ing" from "rendering" → "rendering"). For LogitLens and EmbeddingLens,
raw vocabulary tokens are used directly.

Usage:
    # Evaluate LatentLens results
    python evaluate_interpretability.py \
        --results-dir output/latentlens/olmo-vit/ \
        --images-dir /path/to/images \
        --output-dir evaluation/latentlens/olmo-vit

    # Evaluate LogitLens results
    python evaluate_interpretability.py \
        --results-dir output/logitlens/olmo-vit/ \
        --images-dir /path/to/images \
        --output-dir evaluation/logitlens/olmo-vit

    # Override API key (default: reads OPENAI_API_KEY env var)
    python evaluate_interpretability.py \
        --results-dir output/latentlens/olmo-vit/ \
        --images-dir /path/to/images \
        --output-dir evaluation/test \
        --api-key sk-...

Requirements:
    - OpenAI API key (set OPENAI_API_KEY env var, or pass --api-key)
    - Analysis results JSON files from run_latentlens.py, run_logitlens.py, or run_embedding_lens.py

API Cost Estimate:
    - ~$0.01 per patch evaluation (GPT-5 with images)
    - 100 patches × 9 layers × 9 models = ~$80-100 for full reproduction
"""

import os
import sys
import io
import json
import argparse
import base64
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

from utils import (
    process_image_with_mask,
    calculate_square_bbox_from_patch,
    draw_bbox_on_image,
    sample_valid_patch_positions,
)
from prompts import IMAGE_PROMPT_WITH_CROP


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_image(pil_image):
    """Encode PIL image to base64 data URL."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"


# ---------------------------------------------------------------------------
# Subword → full-word expansion (LatentLens only)
# Copied from llm_judge/run_single_model_with_viz_contextual.py
# ---------------------------------------------------------------------------

def extract_full_word_from_token(sentence: str, token: str) -> str:
    """
    Extract the full word containing the token from the sentence.
    If the token is a subword within a larger word (e.g., "ing" in "rendering"),
    expand to return the entire containing word. Case-insensitive match.
    If not found, fall back to returning the token itself.
    """
    if not sentence:
        return token.strip() if token else ""

    token = token.strip() if token else ""
    if not token:
        return ""

    low_sent = sentence.lower()
    low_tok = token.lower()
    if not low_tok:
        return token

    idx = low_sent.find(low_tok)
    if idx == -1:
        return token

    def is_word_char(ch: str) -> bool:
        return ch.isalnum() or ch == '_'

    start = idx
    end = idx + len(low_tok)

    token_has_space = any(ch.isspace() for ch in token)

    if not token_has_space:
        left_is_word = start > 0 and is_word_char(sentence[start - 1])
        right_is_word = end < len(sentence) and is_word_char(sentence[end])

        if left_is_word or right_is_word:
            exp_start = start
            exp_end = end
            while exp_start > 0 and is_word_char(sentence[exp_start - 1]):
                exp_start -= 1
            while exp_end < len(sentence) and is_word_char(sentence[exp_end]):
                exp_end += 1

            expanded = sentence[exp_start:exp_end]
            if not any(ch.isspace() for ch in expanded):
                return expanded

    return sentence[start:end]


def extract_words_from_contextual(nearest_list):
    """
    Take top-5 nearest contextual neighbors and extract the full words.
    Each entry has fields like token_str, caption, position, similarity.
    Returns a list of expanded words.
    """
    words = []
    for entry in nearest_list[:5]:
        token = entry.get('token_str', '')
        caption = entry.get('caption', '')
        word = extract_full_word_from_token(caption, token)
        if word:
            words.append(word)
    return words


# ---------------------------------------------------------------------------
# Method detection and word extraction
# ---------------------------------------------------------------------------

def detect_method(data):
    """Auto-detect analysis method from JSON structure.

    Returns one of: 'latentlens', 'logitlens', 'embeddinglens'
    """
    # EmbeddingLens has splits.validation.images structure
    if "splits" in data:
        return "embeddinglens"

    # Both LatentLens and LogitLens have results[] — check first patch
    if "results" in data and data["results"]:
        first_img = data["results"][0]
        patches = []
        if "chunks" in first_img and first_img["chunks"]:
            patches = first_img["chunks"][0].get("patches", [])
        if patches:
            first_patch = patches[0]
            if "nearest_contextual_neighbors" in first_patch:
                return "latentlens"
            if "top_predictions" in first_patch:
                return "logitlens"

    raise ValueError(
        "Cannot detect method from JSON structure. "
        "Expected LatentLens (nearest_contextual_neighbors), "
        "LogitLens (top_predictions), or EmbeddingLens (splits.validation)."
    )


def get_images_from_data(data, method):
    """Extract list of image entries from analysis JSON.

    Each entry is a dict with at least 'image_idx' and 'chunks'.
    """
    if method == "embeddinglens":
        return data["splits"]["validation"]["images"]
    else:
        return data["results"]


def extract_words_for_patch(patch, method):
    """Extract candidate words from a single patch entry.

    Returns list of strings (top-5 words/tokens).
    """
    if method == "latentlens":
        neighbors = patch.get("nearest_contextual_neighbors", [])
        return extract_words_from_contextual(neighbors)
    elif method == "logitlens":
        predictions = patch.get("top_predictions", [])
        return [p["token"] for p in predictions[:5]]
    elif method == "embeddinglens":
        neighbors = patch.get("nearest_neighbors", [])
        return [n["token"] for n in neighbors[:5]]
    return []


# ---------------------------------------------------------------------------
# LLM judge call
# ---------------------------------------------------------------------------

def get_llm_judgment(client, image_with_bbox, cropped_image, candidate_words,
                     model="gpt-5", api_provider="openai"):
    """
    Call LLM to judge if candidate words are interpretable for the image region.

    Returns:
        dict with keys: interpretable, concrete_words, abstract_words, global_words, reasoning
    """
    prompt = IMAGE_PROMPT_WITH_CROP.format(candidate_words=str(candidate_words))

    main_url = encode_image(image_with_bbox)
    crop_url = encode_image(cropped_image) if cropped_image is not None else None

    if api_provider == "openrouter":
        content = [
            {"type": "image_url", "image_url": {"url": main_url}},
            {"type": "text", "text": prompt},
        ]
        if crop_url is not None:
            content.append({"type": "image_url", "image_url": {"url": crop_url}})

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=1000,
        )
        response_text = resp.choices[0].message.content
    else:
        # OpenAI Responses API (matches paper evaluation)
        content = [
            {"type": "input_image", "image_url": main_url},
            {"type": "input_text", "text": prompt},
        ]
        if crop_url is not None:
            content.append({"type": "input_image", "image_url": crop_url})

        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
            reasoning={"effort": "low"},
            text={"verbosity": "low"},
        )
        response_text = resp.output_text

    # Parse JSON from response
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}') + 1
    if start_idx != -1 and end_idx > start_idx:
        json_str = response_text[start_idx:end_idx]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    return {
        "interpretable": False,
        "concrete_words": [],
        "abstract_words": [],
        "global_words": [],
        "reasoning": f"Could not parse response: {response_text[:200]}"
    }


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def load_analysis_results(results_dir, layer):
    """Load analysis results JSON for a specific layer.

    Tries multiple naming conventions:
      - *layer{N}_*.json or *layer{N}.json  (LogitLens, EmbeddingLens)
      - *visual{N}_*.json or *visual{N}.json (LatentLens)

    Uses exact layer matching to avoid e.g. *layer2* matching layer24.
    """
    results_dir = Path(results_dir)
    import re

    # Patterns that anchor the layer number (not followed by another digit)
    layer_re = re.compile(rf'(?:layer|visual){layer}(?:\D|$)')

    files = sorted(f for f in results_dir.glob("*.json") if layer_re.search(f.name))
    if files:
        print(f"  Loading: {files[0].name}")
        with open(files[0]) as f:
            return json.load(f)

    raise FileNotFoundError(f"No results found for layer {layer} in {results_dir}")


def find_image_path(images_dir, image_idx):
    """Find image file for a given image index.

    Tries common naming patterns: {idx:05d}.jpg, {idx}.jpg, {idx:05d}.png, etc.
    """
    images_dir = Path(images_dir)
    for pattern in [
        f"{image_idx:05d}.jpg",
        f"{image_idx}.jpg",
        f"{image_idx:05d}.png",
        f"{image_idx}.png",
    ]:
        path = images_dir / pattern
        if path.exists():
            return str(path)
    return None


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_model(args):
    """Evaluate interpretability for a single model across layers."""
    # API client
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: No API key. Set OPENAI_API_KEY env var or pass --api-key.")
        sys.exit(1)

    if args.api_provider == "openrouter":
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    else:
        client = OpenAI(api_key=api_key)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    all_results = []

    for layer in args.layers:
        print(f"\n{'='*60}")
        print(f"Evaluating layer {layer}...")

        # Load analysis results for this layer
        try:
            analysis_data = load_analysis_results(results_dir, layer)
        except FileNotFoundError as e:
            print(f"  Skipping layer {layer}: {e}")
            continue

        # Auto-detect method
        method = detect_method(analysis_data)
        print(f"  Method: {method}")

        # Extract image entries
        images = get_images_from_data(analysis_data, method)

        # Determine grid size from first image's patches
        first_patches = []
        if images and "chunks" in images[0] and images[0]["chunks"]:
            first_patches = images[0]["chunks"][0].get("patches", [])
        if first_patches:
            max_row = max(p.get("patch_row", 0) for p in first_patches)
            max_col = max(p.get("patch_col", 0) for p in first_patches)
            grid_size = max(max_row + 1, max_col + 1)
        else:
            grid_size = 24  # fallback
        patch_size = 512.0 / grid_size
        bbox_size = 3

        print(f"  Grid: {grid_size}x{grid_size}, {len(images)} images available")

        layer_results = {
            "layer": layer,
            "method": method,
            "grid_size": grid_size,
            "patches": [],
            "interpretable_count": 0,
            "total_count": 0,
        }

        # Build a flat list of (image_idx, patch) candidates
        # Index patches by (row, col) for efficient lookup after sampling
        images_processed = 0

        for img_entry in images:
            if layer_results["total_count"] >= args.num_patches:
                break

            image_idx = img_entry.get("image_idx", img_entry.get("image_index", 0))

            # Find image file
            img_path = find_image_path(args.images_dir, image_idx)
            if img_path is None:
                continue

            # Get patches for this image
            patches = []
            if "chunks" in img_entry and img_entry["chunks"]:
                patches = img_entry["chunks"][0].get("patches", [])
            if not patches:
                continue

            # Build patch lookup by (row, col)
            patch_map = {}
            for p in patches:
                row = p.get("patch_row", 0)
                col = p.get("patch_col", 0)
                patch_map[(row, col)] = p

            # Process image and sample valid patch positions
            processed_img, img_mask = process_image_with_mask(img_path, model_name=args.model_name)
            sampled_positions = sample_valid_patch_positions(
                img_mask, bbox_size=bbox_size,
                num_samples=args.num_samples_per_image, grid_size=grid_size
            )
            if not sampled_positions:
                continue

            for patch_row, patch_col in sampled_positions:
                if layer_results["total_count"] >= args.num_patches:
                    break

                # Get center patch (same convention as original scripts)
                center_row = patch_row + bbox_size // 2
                center_col = patch_col + bbox_size // 2
                patch = patch_map.get((center_row, center_col))
                if patch is None:
                    continue

                # Extract words
                words = extract_words_for_patch(patch, method)
                if not words:
                    continue

                # Draw bbox and crop
                bbox = calculate_square_bbox_from_patch(
                    patch_row, patch_col, patch_size=patch_size, size=bbox_size
                )
                img_with_bbox = draw_bbox_on_image(processed_img, bbox)

                left = int(patch_col * patch_size)
                top = int(patch_row * patch_size)
                right = int((patch_col + bbox_size) * patch_size)
                bottom = int((patch_row + bbox_size) * patch_size)
                left = max(0, left); top = max(0, top)
                right = min(right, processed_img.size[0])
                bottom = min(bottom, processed_img.size[1])
                cropped = processed_img.crop((left, top, right, bottom))

                # Call LLM judge
                judgment = get_llm_judgment(
                    client, img_with_bbox, cropped, words,
                    model=args.api_model, api_provider=args.api_provider,
                )

                # Determine interpretability
                concrete = judgment.get("concrete_words", [])
                abstract = judgment.get("abstract_words", [])
                global_w = judgment.get("global_words", [])
                is_interpretable = len(concrete) > 0 or len(abstract) > 0 or len(global_w) > 0

                result = {
                    "image_idx": image_idx,
                    "patch_row": center_row,
                    "patch_col": center_col,
                    "candidate_words": words,
                    "judgment": judgment,
                    "interpretable": is_interpretable,
                }
                layer_results["patches"].append(result)
                layer_results["total_count"] += 1
                if is_interpretable:
                    layer_results["interpretable_count"] += 1

                status = "PASS" if is_interpretable else "FAIL"
                print(f"  img {image_idx} ({center_row},{center_col}): {words} -> {status}")

            images_processed += 1

            # Incremental save after each image
            _save_results(output_dir, all_results + [layer_results])

        # Compute fraction
        total = layer_results["total_count"]
        if total > 0:
            layer_results["interpretable_fraction"] = (
                layer_results["interpretable_count"] / total
            )
        else:
            layer_results["interpretable_fraction"] = 0.0

        pct = layer_results["interpretable_fraction"] * 100
        print(f"\n  Layer {layer}: {pct:.1f}% interpretable "
              f"({layer_results['interpretable_count']}/{total})")

        all_results.append(layer_results)

    # Final save
    _save_results(output_dir, all_results)

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for r in all_results:
        pct = r["interpretable_fraction"] * 100
        print(f"  Layer {r['layer']:>2}: {pct:5.1f}% "
              f"({r['interpretable_count']}/{r['total_count']})")

    output_file = output_dir / "evaluation_results.json"
    print(f"\nResults saved to: {output_file}")


def _save_results(output_dir, all_results):
    """Save evaluation results to JSON (called incrementally)."""
    output_file = Path(output_dir) / "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate interpretability of visual tokens using an LLM judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # LatentLens (subword tokens expanded to full words)
  python evaluate_interpretability.py \\
      --results-dir output/latentlens/olmo-vit/ \\
      --images-dir /path/to/pixmo_cap/validation/ \\
      --output-dir evaluation/latentlens/olmo-vit

  # LogitLens (raw vocabulary tokens)
  python evaluate_interpretability.py \\
      --results-dir output/logitlens/olmo-vit/ \\
      --images-dir /path/to/pixmo_cap/validation/ \\
      --output-dir evaluation/logitlens/olmo-vit

  # With OpenRouter instead of OpenAI
  python evaluate_interpretability.py \\
      --results-dir output/logitlens/olmo-vit/ \\
      --images-dir /path/to/images \\
      --output-dir evaluation/test \\
      --api-provider openrouter \\
      --api-model google/gemini-2.0-flash-exp
        """,
    )
    parser.add_argument("--results-dir", required=True,
                        help="Directory with analysis result JSONs (one per layer)")
    parser.add_argument("--images-dir", required=True,
                        help="Directory with source images (named {idx:05d}.jpg)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for evaluation results")
    parser.add_argument("--layers", type=int, nargs="+",
                        default=[0, 1, 2, 4, 8, 16, 24, 30, 31],
                        help="Layers to evaluate (default: %(default)s)")
    parser.add_argument("--num-patches", type=int, default=100,
                        help="Total number of patches to evaluate per layer")
    parser.add_argument("--num-samples-per-image", type=int, default=1,
                        help="Number of patches to sample per image")
    parser.add_argument("--model-name", default=None,
                        help="Model name for preprocessing (e.g. 'qwen2vl' for center-crop, "
                             "'qwen2-7b_vit-l-14-336_seed10' for resize-and-pad). "
                             "If not set, defaults to resize-and-pad.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # API configuration
    api_group = parser.add_argument_group("API configuration")
    api_group.add_argument("--api-key", default=None,
                           help="API key (default: reads OPENAI_API_KEY env var)")
    api_group.add_argument("--api-provider", default="openai",
                           choices=["openai", "openrouter"],
                           help="API provider (default: openai)")
    api_group.add_argument("--api-model", default="gpt-5",
                           help="Model to use for evaluation (default: gpt-5)")

    args = parser.parse_args()

    # Validate API key availability
    if not args.api_key and not os.environ.get("OPENAI_API_KEY"):
        print("Error: No API key provided.")
        print("  Set OPENAI_API_KEY environment variable, or pass --api-key.")
        sys.exit(1)

    evaluate_model(args)


if __name__ == "__main__":
    main()
