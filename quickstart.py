#!/usr/bin/env python3
"""
LatentLens Quickstart — Interpreting visual tokens in Qwen2-VL

Demonstrates the core LatentLens idea: visual token representations become
interpretable when compared against contextual text embeddings (nearest-neighbor
search in the LLM's own representation space).

No `latentlens` package imports needed — LatentLens is a *method*, not a framework.
This script uses only standard `transformers` and `torch`.

Requirements:
    - GPU with >=24GB VRAM (Qwen2-VL-7B in float16)
    - pip install transformers torch huggingface_hub pillow

Usage:
    python quickstart.py                              # uses bundled example.png
    python quickstart.py --image path/to/image.jpg    # your own image
    python quickstart.py --layers 2,8,16,27 --top-k 10

On first run, contextual embeddings (~2GB per layer) are downloaded from HuggingFace.
"""

import argparse
import math
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ── Constants ────────────────────────────────────────────────────────────────

HF_EMBEDDINGS_REPO = "McGill-NLP/latentlens-qwen2vl-embeddings"
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
IMAGE_PAD_TOKEN_ID = 151655  # <|image_pad|> in Qwen2-VL's vocabulary
FIXED_RESOLUTION = 448  # → 16×16 grid = 256 vision tokens
AVAILABLE_LAYERS = [1, 2, 4, 8, 16, 24, 26, 27]  # layers with pre-computed embeddings


# ── Helpers ──────────────────────────────────────────────────────────────────

def download_contextual_embeddings(layer, cache_dir=None):
    """Download pre-computed contextual text embeddings for one layer."""
    path = hf_hub_download(
        repo_id=HF_EMBEDDINGS_REPO,
        filename=f"layer_{layer}/embeddings_cache.pt",
        repo_type="model",
        cache_dir=cache_dir,
    )
    return path


def load_contextual_cache(path, device):
    """Load a contextual embedding cache and return normalized embeddings + metadata."""
    cache = torch.load(path, map_location="cpu", weights_only=False)
    embeddings = cache["embeddings"].to(device)  # keep native float16
    embeddings = F.normalize(embeddings, dim=-1)
    metadata = cache["metadata"]  # list of dicts: {token_str, token_id, caption, position}
    return embeddings, metadata


def preprocess_image(image_path):
    """Load image, center-crop to square, resize to FIXED_RESOLUTION."""
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    image = image.crop((left, top, left + min_dim, top + min_dim))
    image = image.resize((FIXED_RESOLUTION, FIXED_RESOLUTION), Image.LANCZOS)
    return image


def find_vision_token_range(input_ids):
    """Return (start, end, count) of vision token positions."""
    ids = input_ids.cpu().numpy()
    if ids.ndim == 2:
        ids = ids[0]
    positions = np.where(ids == IMAGE_PAD_TOKEN_ID)[0]
    if len(positions) == 0:
        return None, None, 0
    return int(positions[0]), int(positions[-1]) + 1, len(positions)


def search_nearest_neighbors_batch(features_by_layer, ctx_paths, device, top_k):
    """
    For selected vision tokens across multiple LLM layers, find the top-k nearest
    contextual text embeddings by searching across ALL contextual layers and merging.

    Loads each contextual cache exactly once (not once per LLM layer), searching
    all LLM layers' features against it before unloading.

    Args:
        features_by_layer: dict of {llm_layer: [num_selected, hidden_dim]} normalized features
        ctx_paths: dict of {layer: path} for all contextual embedding caches
        device: torch device for computation
        top_k: number of nearest neighbors to return per token

    Returns:
        dict of {llm_layer: list of [(token_str, similarity, caption, ctx_layer), ...]}
    """
    ctx_layers = sorted(ctx_paths.keys())
    llm_layers = sorted(features_by_layer.keys())

    # candidates[llm_layer][ctx_layer] = (vals, idxs) on CPU
    candidates = {ll: {} for ll in llm_layers}
    ctx_metadata_cache = {}

    # Phase 1: Load each contextual cache once, search all LLM layers against it
    for cl in ctx_layers:
        embeddings, metadata = load_contextual_cache(ctx_paths[cl], device)
        ctx_metadata_cache[cl] = metadata
        for ll in llm_layers:
            similarity = torch.matmul(features_by_layer[ll], embeddings.T)
            vals, idxs = torch.topk(similarity, k=top_k, dim=-1)
            candidates[ll][cl] = (vals.cpu(), idxs.cpu())
            del similarity
        del embeddings
        torch.cuda.empty_cache()

    # Phase 2: Merge across contextual layers for each LLM layer
    num_ctx = len(ctx_layers)
    results_by_layer = {}
    for ll in llm_layers:
        all_vals = torch.stack([candidates[ll][cl][0] for cl in ctx_layers])  # [num_ctx, num_sel, top_k]
        all_idxs = torch.stack([candidates[ll][cl][1] for cl in ctx_layers])

        results = []
        for tok_idx in range(all_vals.shape[1]):
            flat_vals = all_vals[:, tok_idx, :].flatten()
            flat_idxs = all_idxs[:, tok_idx, :].flatten()
            layer_ids = torch.arange(num_ctx).unsqueeze(1).expand(-1, top_k).flatten()
            global_top_vals, global_top_pos = torch.topk(flat_vals, k=top_k)

            neighbors = []
            for k_idx in range(top_k):
                pos = global_top_pos[k_idx].item()
                sim = global_top_vals[k_idx].item()
                cl_idx = layer_ids[pos].item()
                emb_idx = flat_idxs[pos].item()
                ctx_layer = ctx_layers[cl_idx]
                meta = ctx_metadata_cache[ctx_layer][emb_idx]
                neighbors.append((meta["token_str"], sim, meta["caption"], ctx_layer))
            results.append(neighbors)
        results_by_layer[ll] = results
    return results_by_layer


def _find_token_in_caption(caption, token_str):
    """
    Find the best position of token_str in caption, respecting word boundaries.

    BPE tokens starting with " " (space) should match at word boundaries, not
    inside other words. E.g., " door" should match " door" in "white door", NOT
    the "door" inside "doorway".

    Returns character index in caption, or -1 if not found.
    """
    token_clean = token_str.strip().lower()
    caption_lower = caption.lower()

    # If original BPE token had a leading space, prefer word-boundary match
    starts_word = token_str.startswith(" ")

    if starts_word:
        # Try matching at word boundaries first: look for " token" or at position 0
        # Search for space + token where the char AFTER is not alpha (end of match = word boundary or end)
        search_with_space = " " + token_clean
        pos = 0
        while pos < len(caption_lower):
            idx = caption_lower.find(search_with_space, pos)
            if idx == -1:
                break
            # Match starts after the space
            match_start = idx + 1
            match_end = match_start + len(token_clean)
            # Accept if at end of string or next char is not alphanumeric (word boundary)
            if match_end >= len(caption_lower) or not caption_lower[match_end].isalpha():
                return match_start
            pos = idx + 1

        # Also try at position 0 (caption starts with the token)
        if caption_lower.startswith(token_clean):
            match_end = len(token_clean)
            if match_end >= len(caption_lower) or not caption_lower[match_end].isalpha():
                return 0

    # Fallback: first occurrence (for continuation tokens or if word-boundary match failed)
    idx = caption_lower.find(token_clean)
    return idx


def smart_truncate_around_token(caption, token_str, max_len=50):
    """
    Truncate caption while keeping the token visible.
    Returns (prefix, token, suffix) for separate rendering.

    Ported from create_layer_evolution_annotation_set.py, with improved
    word-boundary matching for BPE tokens.
    """
    if not caption or not token_str:
        return ("", token_str.strip() if token_str else "", "")

    token_clean = token_str.strip()
    idx = _find_token_in_caption(caption, token_str)

    if idx == -1:
        # Token not found verbatim — show caption prefix + token appended
        budget = max_len - len(token_clean) - 6
        if budget > 0 and len(caption) > budget:
            trunc = caption[:budget]
            last_space = trunc.rfind(" ")
            if last_space > 10:
                trunc = trunc[:last_space]
            return (trunc + "... ", token_clean, "")
        return (caption + " ", token_clean, "")

    prefix = caption[:idx]
    token = caption[idx : idx + len(token_clean)]
    suffix = caption[idx + len(token_clean) :]

    if len(prefix) + len(token) + len(suffix) <= max_len:
        return (prefix, token, suffix)

    # Truncate — prioritize context around token
    available = max_len - len(token) - 6
    if available < 4:
        return ("", token, "")

    prefix_budget = min(len(prefix), available * 3 // 5)
    suffix_budget = min(len(suffix), available - prefix_budget)
    prefix_budget = min(len(prefix), available - suffix_budget)

    if len(prefix) > prefix_budget:
        trunc = prefix[-prefix_budget:]
        first_space = trunc.find(" ")
        if 0 < first_space < len(trunc) - 3:
            trunc = trunc[first_space + 1 :]
        prefix = "..." + trunc

    if len(suffix) > suffix_budget:
        trunc = suffix[:suffix_budget]
        last_space = trunc.rfind(" ")
        if last_space > 3:
            trunc = trunc[:last_space]
        suffix = trunc + "..."

    return (prefix, token, suffix)


def create_visualization(image, results, num_vision_tokens, sample_indices, output_path, layer):
    """
    Create a PNG showing the image with green bounding boxes on selected
    vision tokens, with numbered labels and caption context (token
    highlighted in yellow) listed alongside.

    Args:
        image:             PIL Image (the preprocessed input image)
        results:           list of [(token_str, sim, caption, ctx_layer), ...] per patch_idx
        num_vision_tokens: total number of vision tokens
        sample_indices:    list of patch indices to display
        output_path:       where to save the PNG
        layer:             layer index (for title)
    """
    n_show = len(sample_indices)
    # ── Grid geometry (read from data, never hardcode) ───────────────────
    grid_size = int(math.sqrt(num_vision_tokens))
    display_size = 512
    img_resized = image.resize((display_size, display_size), Image.LANCZOS)
    patch_size = display_size / grid_size  # e.g. 512/16 = 32px

    # ── Canvas: image on left, label column on right ─────────────────────
    label_width = 380
    margin = 20
    canvas_w = display_size + margin + label_width + margin
    # Each label needs ~40px for two lines (caption context + similarity)
    label_block_h = 45
    labels_total_h = n_show * label_block_h + 30
    canvas_h = max(display_size + 2 * margin + 30, labels_total_h + margin + 30)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    # Load fonts (same as create_layer_evolution_annotation_set.py)
    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
        )
        label_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11
        )
        bold_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11
        )
    except OSError:
        title_font = label_font = bold_font = ImageFont.load_default()

    # Title
    title = f"LatentLens — Layer {layer} — {n_show} sampled tokens"
    draw.text((margin, margin // 2), title, fill="black", font=title_font)

    img_y_offset = margin + 25
    canvas.paste(img_resized, (margin, img_y_offset))

    # ── Semi-transparent overlay ─────────────────────────────────────────
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    label_x = display_size + 2 * margin
    label_y_start = img_y_offset + 5
    green = "#228B22"

    for rank, patch_idx in enumerate(sample_indices):
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        neighbors = results[patch_idx]
        if not neighbors:
            continue
        token_str, sim, caption, _ctx_layer = neighbors[0]

        # ── Patch bbox (following create_layer_evolution_annotation_set.py) ──
        x1 = margin + int(col * patch_size)
        y1 = img_y_offset + int(row * patch_size)
        x2 = margin + int((col + 1) * patch_size)
        y2 = img_y_offset + int((row + 1) * patch_size)

        # Semi-transparent green fill
        overlay_draw.rectangle([x1, y1, x2, y2], fill=(34, 139, 34, 50))
        # Green outline
        draw.rectangle([x1, y1, x2, y2], outline=green, width=2)

        # Rank number inside the box — larger font, centered
        rank_text = str(rank + 1)
        rank_font = bold_font  # use bold for visibility
        rb = draw.textbbox((0, 0), rank_text, font=rank_font)
        rw, rh = rb[2] - rb[0], rb[3] - rb[1]
        pad = 3
        rx = x1 + (x2 - x1 - rw) // 2
        ry = y1 + (y2 - y1 - rh) // 2
        draw.rectangle(
            [rx - pad, ry - pad, rx + rw + pad, ry + rh + pad],
            fill=(255, 255, 255, 200),
        )
        draw.text((rx, ry), rank_text, fill=green, font=rank_font)

        # ── Label with caption context + yellow-highlighted token ────────
        label_y = label_y_start + rank * label_block_h

        # Line 1: "N. sim=0.XX"
        header = f"{rank + 1}.  sim={sim:.2f}"
        draw.text((label_x, label_y), header, fill="#444444", font=label_font)

        # Line 2: caption with yellow-highlighted token
        # (following create_layer_evolution_annotation_set.py pattern)
        prefix, tok, suffix = smart_truncate_around_token(caption, token_str)
        text_y = label_y + 18
        x_pos = label_x + 10

        if prefix:
            draw.text((x_pos, text_y), prefix, fill="#444444", font=label_font)
            pb = draw.textbbox((x_pos, text_y), prefix, font=label_font)
            x_pos = pb[2]

        if tok:
            # Yellow highlight behind token (same as create_layer_evolution_annotation_set.py)
            tb = draw.textbbox((x_pos, text_y), tok, font=bold_font)
            draw.rectangle(
                [tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1], fill="#FFFF00"
            )
            draw.text((x_pos, text_y), tok, fill="black", font=bold_font)
            x_pos = tb[2]

        if suffix:
            draw.text((x_pos, text_y), suffix, fill="#444444", font=label_font)


    # Composite overlay
    canvas_rgba = canvas.convert("RGBA")
    canvas_rgba = Image.alpha_composite(canvas_rgba, overlay)
    canvas = canvas_rgba.convert("RGB")

    canvas.save(output_path, quality=95)
    return output_path


def format_results(results_by_layer, sample_indices, grid_size, layers, top_k):
    """Pretty-print nearest-neighbor results with caption context."""
    for layer in layers:
        results = results_by_layer[layer]
        print(f"\n{'─' * 70}")

        if layer <= 2:
            stage = "early — representations not yet specialized"
        elif layer <= 8:
            stage = "middle — concepts emerging"
        else:
            stage = "late — highly interpretable"
        print(f"Layer {layer} ({stage}):")

        for patch_idx in sample_indices:
            row = patch_idx // grid_size
            col = patch_idx % grid_size
            neighbors = results[patch_idx]
            if neighbors:
                tok, sim, caption, _cl = neighbors[0]
                prefix, token, suffix = smart_truncate_around_token(caption, tok)
                context = f"{prefix}[{token}]{suffix}"
                rest = ", ".join(
                    f"{t.strip()!r} ({s:.2f})" for t, s, *_ in neighbors[1:top_k]
                )
                print(f"  ({row:>2},{col:>2}): {context}  ({sim:.2f})  | {rest}")
            else:
                print(f"  ({row:>2},{col:>2}): (no results)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LatentLens quickstart: interpret visual tokens in Qwen2-VL"
    )
    default_image = str(Path(__file__).parent / "example.png")
    parser.add_argument(
        "--image", type=str, default=default_image,
        help="Path to an input image (default: bundled example.png)"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="2,8,27",
        help=f"Comma-separated LLM layers to extract vision features from (available: {AVAILABLE_LAYERS})",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of nearest neighbors per token"
    )
    parser.add_argument(
        "--seed", type=int, default=10, help="Random seed for token sampling"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for output PNG visualization (default: <image_stem>_latentlens.png)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory for downloaded embeddings",
    )
    args = parser.parse_args()

    layers = [int(l.strip()) for l in args.layers.split(",")]
    for layer in layers:
        if layer not in AVAILABLE_LAYERS:
            print(f"Error: layer {layer} not available. Choose from {AVAILABLE_LAYERS}")
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: running on CPU will be very slow")

    print("=" * 60)
    print("LatentLens Quickstart — Qwen2-VL-7B-Instruct")
    print("=" * 60)
    print(f"Image:  {args.image}")
    print(f"Layers: {layers}")
    print(f"Top-k:  {args.top_k}")
    print()

    # ── Step 1: Download contextual embeddings (ALL layers) ────────────────
    # LatentLens searches across all contextual layers and merges globally,
    # so we always need all available layers regardless of which LLM layers
    # we extract vision features from.
    print("Step 1/4: Downloading contextual embeddings (all layers)...")
    ctx_paths = {}
    for layer in AVAILABLE_LAYERS:
        print(f"  Layer {layer}...", end=" ", flush=True)
        path = download_contextual_embeddings(layer, cache_dir=args.cache_dir)
        ctx_paths[layer] = path
        print("done")
    print()

    # ── Step 2: Load Qwen2-VL ────────────────────────────────────────────
    print("Step 2/4: Loading Qwen2-VL-7B-Instruct...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Fix resolution so we get a consistent 16×16 grid
    fixed_pixels = FIXED_RESOLUTION * FIXED_RESOLUTION
    processor.image_processor.min_pixels = fixed_pixels
    processor.image_processor.max_pixels = fixed_pixels
    processor.image_processor.do_resize = False
    print(f"  Model loaded (float16, {FIXED_RESOLUTION}x{FIXED_RESOLUTION} → 16x16 grid)")
    print()

    # ── Step 3: Process image and extract hidden states ──────────────────
    print("Step 3/4: Processing image...")
    image = preprocess_image(args.image)

    inputs = processor(
        images=[image],
        text="<|image_pad|>Describe this image in detail.",
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    vision_start, vision_end, num_vision = find_vision_token_range(inputs["input_ids"])
    if num_vision == 0:
        print("Error: no vision tokens found in input. Check image and processor.")
        sys.exit(1)
    print(f"  Vision tokens: {num_vision} (positions {vision_start}–{vision_end - 1})")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple of 29 tensors (embedding + 28 layers)
    print(f"  Hidden states extracted from {len(hidden_states)} layers")
    print()

    # ── Step 4: Nearest-neighbor search (cross-layer merge) ─────────────
    # Pick display indices upfront so we only search the tokens we'll show.
    # Same 10 tokens for both text output and visualization.
    grid_size = int(math.sqrt(num_vision))
    random.seed(args.seed)
    selected = sorted(random.sample(range(num_vision), min(10, num_vision)))
    sel_to_patch = {i: patch_idx for i, patch_idx in enumerate(selected)}

    print(f"Step 4/4: Searching {len(selected)} tokens across {len(AVAILABLE_LAYERS)} contextual layers...")

    # Extract only the selected vision features for each LLM layer
    features_by_layer = {}
    for layer in layers:
        hs = hidden_states[layer]
        all_vision = hs[:, vision_start:vision_end, :].squeeze(0)  # [num_vision, dim]
        sel_feats = all_vision[selected]  # [num_selected, dim]
        features_by_layer[layer] = F.normalize(sel_feats.float(), dim=-1).half()

    # Free model hidden states
    del hidden_states, outputs
    torch.cuda.empty_cache()

    # Batch search: loads each contextual cache once for all LLM layers
    sel_results_by_layer = search_nearest_neighbors_batch(
        features_by_layer, ctx_paths, device, args.top_k
    )

    # Expand back to full-index results (sparse: only selected indices have data)
    results_by_layer = {}
    for layer in layers:
        full = [[] for _ in range(num_vision)]
        for sel_i, neighbors in enumerate(sel_results_by_layer[layer]):
            full[sel_to_patch[sel_i]] = neighbors
        results_by_layer[layer] = full

    # ── Display results ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS — Top nearest neighbors for sampled vision tokens")
    print("=" * 60)
    format_results(results_by_layer, selected, grid_size, layers, args.top_k)

    # ── Create visualization PNG ─────────────────────────────────────
    # Visualize the earliest requested layer (most surprising/compelling result)
    vis_layer = layers[0]
    if args.output:
        output_path = args.output
    else:
        image_stem = Path(args.image).stem
        output_path = str(Path(args.image).parent / f"{image_stem}_latentlens.png")

    create_visualization(
        image, results_by_layer[vis_layer], num_vision, selected, output_path, vis_layer
    )
    print(f"\nVisualization saved to: {output_path}")

    print(f"\n{'─' * 60}")
    print("Done! Each token's nearest neighbors show what the LLM")
    print("'sees' at that layer — early layers are noisy, late layers")
    print("are highly interpretable (content words matching the image).")
    print()
    print("For full analysis across 300 images with LLM-judge evaluation,")
    print("see the reproduction instructions in README.md.")


if __name__ == "__main__":
    main()
