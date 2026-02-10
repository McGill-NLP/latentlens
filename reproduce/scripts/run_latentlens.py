#!/usr/bin/env python3
"""
LatentLens: Contextual Nearest Neighbors Analysis

This script finds nearest neighbors in contextual text embeddings for visual tokens
at each layer of the model, implementing the LatentLens method.

Algorithm:
    1. Load model
    2. Preload all images directly to GPU (one-time disk read)
    3. For each contextual cache:
        Load cache
        For each image (from GPU, no disk or RAM transfer!):
            Forward pass → get all visual layer features
            Search against cache → store candidates
        Unload cache
    4. Merge candidates → pick global top-k
    5. Save results

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_latentlens.py \\
        --ckpt-path <path> --contextual-dir <dir> --visual-layer 0,1,8,24 --num-images 100

Part of the LatentLens package: https://github.com/McGill-NLP/latentlens
"""

import argparse
import gc
import json
import time
import math
import os
import numpy as np
import torch
from pathlib import Path

from molmo.config import ModelConfig, TrainConfig
from molmo.data import build_mm_preprocessor
from molmo.model import Molmo
from molmo.util import resource_path
from molmo.data.pixmo_datasets import PixMoCap


def find_available_layers(contextual_dir):
    """Find all layer directories with caches."""
    contextual_path = Path(contextual_dir)
    layers = []
    for layer_dir in contextual_path.iterdir():
        if layer_dir.is_dir() and layer_dir.name.startswith("layer_"):
            cache_file = layer_dir / "embeddings_cache.pt"
            if cache_file.exists():
                layer_idx = int(layer_dir.name.split("_")[1])
                layers.append(layer_idx)
    return sorted(layers)


def patch_idx_to_row_col(patch_idx, patches_per_chunk):
    """Convert patch index to row and column coordinates."""
    grid_size = int(math.sqrt(patches_per_chunk))
    row = patch_idx // grid_size
    col = patch_idx % grid_size
    return row, col


def preload_images(dataset, preprocessor, prompt, image_indices, device):
    """Preload and preprocess all images directly to GPU. No RAM→GPU transfer during processing."""
    cached_data = []
    num_images = len(image_indices)

    for i, img_idx in enumerate(image_indices):
        if i % 10 == 0:
            print(f"    {i}/{num_images} (image {img_idx})...", flush=True)
        example_data = dataset.get(img_idx, np.random)
        
        caption_text = ""
        if "message_list" in example_data and len(example_data["message_list"]) > 0:
            caption_text = example_data["message_list"][0].get("text", "")
        
        example = {"image": example_data["image"], "messages": [prompt]}
        batch = preprocessor(example, rng=np.random)
        
        # Store preprocessed data directly on GPU!
        cached_data.append({
            'images': torch.tensor(batch.get("images")).to(device),
            'image_masks': torch.tensor(batch.get("image_masks")).to(device) if batch.get("image_masks") is not None else None,
            'input_tokens': torch.tensor(batch["input_tokens"]).to(device),
            'image_input_idx': torch.tensor(batch.get("image_input_idx")).to(device) if batch.get("image_input_idx") is not None else None,
            'caption': caption_text
        })
    
    return cached_data


def extract_visual_features_from_cache(model, cached_batch, use_n_token_only, visual_layers, device, precision_dtype=torch.float16):
    """
    Extract features from pre-cached batch (already on GPU, no transfer needed).
    Returns: dict[visual_layer] -> features [num_patches, hidden_dim], and metadata
    """
    features_by_layer = {}

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=precision_dtype):
            # Tensors already on GPU from preload!
            images = cached_batch['images'].unsqueeze(0)
            image_masks = cached_batch['image_masks'].unsqueeze(0) if cached_batch['image_masks'] is not None else None
            
            need_layer_0 = 0 in visual_layers
            need_llm_layers = any(l > 0 for l in visual_layers)
            
            # Layer 0: vision backbone output
            if need_layer_0:
                feats_l0, _ = model.vision_backbone(images, image_masks, return_tokens_before_MLP=True)
                if type(use_n_token_only) == int and use_n_token_only != -1:
                    feats_l0 = feats_l0[:, :, :use_n_token_only, :]
                elif type(use_n_token_only) == list and len(use_n_token_only) > 0:
                    feats_l0 = feats_l0[:, :, use_n_token_only, :]
                
                B, num_chunks, patches_per_chunk, hidden_dim = feats_l0.shape
                features_by_layer[0] = torch.nn.functional.normalize(feats_l0.view(-1, hidden_dim), dim=-1).float()
                del feats_l0
            
            # LLM layers: one forward pass for all
            if need_llm_layers:
                # Tensors already on GPU from preload!
                input_ids = cached_batch['input_tokens'].unsqueeze(0)
                image_input_idx = cached_batch['image_input_idx'].unsqueeze(0) if cached_batch['image_input_idx'] is not None else None
                
                output = model(
                    input_ids=input_ids,
                    images=images,
                    image_masks=image_masks,
                    image_input_idx=image_input_idx,
                    output_hidden_states=True,
                    last_logits_only=False,
                )
                hidden_states = output.hidden_states
                
                B = image_input_idx.shape[0]
                num_chunks = image_input_idx.shape[1]
                patches_per_chunk = image_input_idx.shape[2]
                hidden_dim = hidden_states[0].shape[-1]
                
                flat_pos = image_input_idx.view(B, -1)
                valid_mask = flat_pos >= 0
                
                for vl in visual_layers:
                    if vl == 0:
                        continue
                    layer_idx = min(vl, len(hidden_states) - 1)
                    hs = hidden_states[layer_idx]
                    
                    feats = torch.zeros((B, num_chunks, patches_per_chunk, hidden_dim), device=hs.device, dtype=hs.dtype)
                    for b in range(B):
                        valid = flat_pos[b][valid_mask[b]]
                        if valid.numel() > 0:
                            feats.view(B, -1, hidden_dim)[b, valid_mask[b], :] = hs[b, valid.long(), :]
                    
                    features_by_layer[vl] = torch.nn.functional.normalize(feats.view(-1, hidden_dim), dim=-1).float()
                
                del hidden_states, output
            
            del images, image_masks
            torch.cuda.empty_cache()
    
    num_patches = features_by_layer[visual_layers[0]].shape[0]
    metadata = {
        'num_chunks': num_chunks,
        'patches_per_chunk': patches_per_chunk,
        'hidden_dim': hidden_dim,
        'num_patches': num_patches
    }
    
    return features_by_layer, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--contextual-dir", type=str, required=True)
    parser.add_argument("--visual-layer", type=str, default="24")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--image-indices", type=str, default=None,
                       help="Comma-separated list of specific image indices to process (e.g., '100,102,108'). If provided, --num-images is ignored.")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="analysis_results/contextual_nearest_neighbors")
    parser.add_argument("--use-bf16", action="store_true", help="Use bfloat16 instead of float16 (for models with extreme norms)")
    args = parser.parse_args()

    # Parse image indices if provided
    if args.image_indices:
        image_indices = [int(i.strip()) for i in args.image_indices.split(",")]
    else:
        image_indices = list(range(args.num_images))
    
    device = torch.device("cuda")
    visual_layers = [int(l.strip()) for l in args.visual_layer.split(",")]
    ctx_layers = find_available_layers(args.contextual_dir)
    
    print("=" * 70)
    print("CONTEXTUAL NN (WITH IMAGE PRELOADING)")
    print("=" * 70)
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Contextual dir: {args.contextual_dir}")
    print(f"Visual layers: {visual_layers}")
    print(f"Contextual layers: {ctx_layers}")
    num_images = len(image_indices)
    print(f"Images: {num_images}" + (f" (indices: {image_indices[:5]}...)" if len(image_indices) > 5 else f" (indices: {image_indices})"))
    print(f"Total forward passes: {len(ctx_layers)} caches × {num_images} images = {len(ctx_layers) * num_images}")
    print()
    
    # ===== LOAD MODEL =====
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    load_start = time.time()
    
    cfg = TrainConfig.load(f"{args.ckpt_path}/config.yaml")
    cfg.model.init_device = "cpu"
    model = Molmo(cfg.model)
    
    ckpt_file = f"{args.ckpt_path}/model.pt"
    ckpt_size_gb = os.path.getsize(ckpt_file) / (1024**3)
    
    if ckpt_size_gb < 1.0:
        print(f"  Stripped checkpoint ({ckpt_size_gb:.2f} GB) - loading pretrained...")
        model.reset_with_pretrained_weights()
    
    print(f"  Loading weights...")
    weights = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(weights, strict=False)
    del weights
    gc.collect()
    
    if args.use_bf16:
        print(f"  Moving to GPU (bf16)...")
        model = model.to(dtype=torch.bfloat16).cuda().eval()
    else:
        print(f"  Moving to GPU (fp16)...")
        model = model.half().cuda().eval()
    torch.cuda.empty_cache()
    
    print(f"✓ Model loaded in {time.time() - load_start:.1f}s")
    print()
    
    # Create preprocessor and dataset
    model_config = ModelConfig.load(resource_path(args.ckpt_path, "config.yaml"), key="model", validate_paths=False)
    model_config.system_prompt_kind = "none"
    preprocessor = build_mm_preprocessor(model_config, for_inference=True, shuffle_messages=False, is_training=False, require_image_features=True)
    use_n_token_only = model_config.vision_backbone.use_n_token_only
    
    dataset = PixMoCap(split=args.split, mode="captions")
    prompt = "Describe this image in detail."
    
    # ===== PRELOAD IMAGES =====
    print("=" * 70)
    print("PRELOADING IMAGES TO GPU")
    print("=" * 70)
    preload_start = time.time()
    
    print(f"  Loading {num_images} images directly to GPU...", flush=True)
    cached_images = preload_images(dataset, preprocessor, prompt, image_indices, device)

    preload_time = time.time() - preload_start
    print(f"✓ Images preloaded in {preload_time:.1f}s ({num_images/preload_time:.1f} img/s)")
    print()
    
    # Output setup
    ckpt_name = args.ckpt_path.split("/")[-2] + "_" + args.ckpt_path.split("/")[-1]
    output_dir = Path(args.output_dir) / ckpt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== EXTRACT VISUAL FEATURES (one forward pass per image) =====
    print("=" * 70)
    print("EXTRACTING VISUAL FEATURES")
    print("=" * 70)
    extract_start = time.time()
    precision_dtype = torch.bfloat16 if args.use_bf16 else torch.float16

    # features_cache[i][vl] = normalized features tensor (num_patches, hidden_dim) on CPU
    features_cache = []
    shape_info = None

    for i, img_idx in enumerate(image_indices):
        if i % 5 == 0 or i == num_images - 1:
            elapsed = time.time() - extract_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Image {i + 1}/{num_images} (idx={img_idx}) ({rate:.1f} img/s)", flush=True)

        features_by_layer, meta = extract_visual_features_from_cache(
            model, cached_images[i], use_n_token_only, visual_layers, device, precision_dtype
        )

        if shape_info is None:
            shape_info = meta

        # Move features to CPU to free GPU for cache loading later
        features_cache.append({vl: feats.cpu() for vl, feats in features_by_layer.items()})
        del features_by_layer

    extract_time = time.time() - extract_start
    print(f"✓ Features extracted in {extract_time:.1f}s ({num_images / extract_time:.1f} img/s)")
    print()

    # Free model from GPU — only caches + search from here
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ===== SEARCH AGAINST CONTEXTUAL CACHES =====
    candidates = {vl: {img: {} for img in image_indices} for vl in visual_layers}
    ctx_metadata_cache = {}

    search_start = time.time()

    for ctx_idx, ctx_layer in enumerate(ctx_layers):
        print("=" * 70)
        print(f"CONTEXTUAL LAYER {ctx_layer} ({ctx_idx + 1}/{len(ctx_layers)})")
        print("=" * 70)

        # Load cache to GPU
        print(f"  Loading cache to GPU...", end=" ", flush=True)
        cache_start = time.time()
        cache_file = Path(args.contextual_dir) / f"layer_{ctx_layer}" / "embeddings_cache.pt"
        cache_data = torch.load(cache_file, map_location='cpu', weights_only=False)
        embeddings = cache_data['embeddings'].to(device)
        metadata = cache_data['metadata']
        embeddings_norm = torch.nn.functional.normalize(embeddings, dim=-1)
        ctx_metadata_cache[ctx_layer] = metadata
        del cache_data
        print(f"done ({time.time() - cache_start:.1f}s, {embeddings.shape[0]} embeddings on GPU)")

        # Search all images (no forward pass — just matmul!)
        print(f"  Searching {num_images} images...")
        img_start = time.time()

        for i, img_idx in enumerate(image_indices):
            for vl in visual_layers:
                feats = features_cache[i][vl].to(device)
                similarity = torch.matmul(feats, embeddings_norm.T)
                top_vals, top_idxs = torch.topk(similarity, k=args.top_k, dim=-1)
                candidates[vl][img_idx][ctx_layer] = (top_vals, top_idxs)
                del similarity, feats

        img_time = time.time() - img_start
        print(f"  ✓ Done: {num_images} images in {img_time:.1f}s ({num_images/img_time:.1f} img/s)")

        # Unload cache from GPU
        del embeddings, embeddings_norm
        gc.collect()
        torch.cuda.empty_cache()

    search_time = time.time() - search_start
    print()
    print(f"✓ All caches searched in {search_time:.1f}s ({search_time/60:.1f} min)")
    print(f"  (Feature extraction: {extract_time:.1f}s + Search: {search_time:.1f}s)")
    print()
    
    # ===== BUILD RESULTS =====
    print("=" * 70)
    print("BUILDING RESULTS")
    print("=" * 70)
    
    build_start = time.time()
    num_patches = shape_info['num_patches']
    num_chunks = shape_info['num_chunks']
    patches_per_chunk = shape_info['patches_per_chunk']
    hidden_dim = shape_info['hidden_dim']
    
    all_results = {vl: [] for vl in visual_layers}
    
    for i, img_idx in enumerate(image_indices):
        if i % 20 == 0:
            print(f"  Image {i + 1}/{num_images} (idx={img_idx})...", flush=True)
        
        for vl in visual_layers:
            all_vals = torch.stack([candidates[vl][img_idx][cl][0] for cl in ctx_layers])
            all_idxs = torch.stack([candidates[vl][img_idx][cl][1] for cl in ctx_layers])
            
            chunks_results = []
            for chunk_idx in range(num_chunks):
                chunk_results = {
                    "chunk_name": "Full Image" if chunk_idx == 0 else f"Chunk {chunk_idx}",
                    "patches": []
                }
                
                for local_patch_idx in range(patches_per_chunk):
                    global_patch_idx = chunk_idx * patches_per_chunk + local_patch_idx
                    
                    patch_vals = all_vals[:, global_patch_idx, :]
                    patch_idxs = all_idxs[:, global_patch_idx, :]
                    
                    flat_vals = patch_vals.flatten()
                    flat_idxs = patch_idxs.flatten()
                    ctx_ids = torch.arange(len(ctx_layers)).unsqueeze(1).expand(-1, args.top_k).flatten()
                    
                    global_top_vals, global_top_pos = torch.topk(flat_vals, k=args.top_k)
                    
                    nearest = []
                    for k_idx in range(args.top_k):
                        pos = global_top_pos[k_idx].item()
                        sim = global_top_vals[k_idx].item()
                        ctx_idx = ctx_ids[pos].item()
                        emb_idx = flat_idxs[pos].item()
                        
                        ctx_layer = ctx_layers[ctx_idx]
                        meta = ctx_metadata_cache[ctx_layer][emb_idx]
                        nearest.append({
                            'token_str': meta['token_str'],
                            'token_id': meta['token_id'],
                            'caption': meta['caption'],
                            'position': meta['position'],
                            'similarity': sim,
                            'contextual_layer': ctx_layer
                        })
                    
                    row, col = patch_idx_to_row_col(local_patch_idx, patches_per_chunk)
                    chunk_results["patches"].append({
                        "patch_idx": local_patch_idx,
                        "patch_row": row,
                        "patch_col": col,
                        "nearest_contextual_neighbors": nearest
                    })
                
                chunks_results.append(chunk_results)
            
            all_results[vl].append({
                "image_idx": img_idx,
                "ground_truth_caption": cached_images[i]['caption'],
                "feature_shape": [1, num_chunks, patches_per_chunk, hidden_dim],
                "llm_layer_used": vl,
                "chunks": chunks_results
            })
    
    build_time = time.time() - build_start
    print(f"✓ Results built in {build_time:.1f}s")
    print()
    
    # ===== SAVE =====
    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    total_time = preload_time + extract_time + search_time + build_time
    
    for vl in visual_layers:
        output_file = output_dir / f"contextual_neighbors_visual{vl}_allLayers.json"
        
        output_data = {
            'checkpoint': args.ckpt_path,
            'contextual_dir': args.contextual_dir,
            'visual_layer': vl,
            'contextual_layers_used': ctx_layers,
            'split': args.split,
            'num_images': num_images,
            'image_indices': image_indices,
            'top_k': args.top_k,
            'processing_time_seconds': total_time,
            'results': all_results[vl]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"  ✓ {output_file.name}")
    
    print()
    print("=" * 70)
    print("✓ DONE!")
    print(f"  Preload: {preload_time:.1f}s | Extract: {extract_time:.1f}s | Search: {search_time:.1f}s | Build: {build_time:.1f}s")
    print(f"  Total: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
