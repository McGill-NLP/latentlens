"""
LogitLens Analysis for Visual Tokens

This script analyzes what the model "thinks" at each layer by applying the final
language model head (ln_f + ff_out) to intermediate hidden states. For each visual
token at each layer, we extract the top-5 predicted vocabulary tokens.

This gives us insight into how visual token representations evolve through the
transformer layers and when they start to align with specific vocabulary tokens.

Usage:
    # Single GPU (auto-detected):
    python scripts/run_logitlens.py --ckpt-path <path> --num-images 100

    # Multi-GPU with torchrun:
    torchrun --nproc_per_node=2 scripts/run_logitlens.py --ckpt-path <path> --num-images 100

Part of the LatentLens package: https://github.com/McGill-NLP/latentlens
"""

import argparse
import json
import math
import os
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
from tqdm import tqdm

from molmo.config import ModelConfig, TrainConfig
from molmo.data import build_mm_preprocessor
from molmo.model import Molmo
from molmo.util import resource_path
from molmo.data.pixmo_datasets import PixMoCap

# Check if we're in distributed mode (launched via torchrun)
DISTRIBUTED_MODE = "RANK" in os.environ

if DISTRIBUTED_MODE:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
    from molmo.torch_util import get_local_rank, get_world_size
else:
    # Single-GPU mode - define stubs
    def get_local_rank():
        return 0
    def get_world_size():
        return 1


def unwrap_model(model):
    """Get the underlying model, unwrapping FSDP if needed."""
    return model.module if hasattr(model, 'module') else model


def patch_idx_to_row_col(patch_idx, patches_per_chunk):
    """Convert patch index to row and column coordinates."""
    grid_size = int(math.sqrt(patches_per_chunk))
    row = patch_idx // grid_size
    col = patch_idx % grid_size
    return row, col


def clear_gpu_memory():
    """Clear CUDA cache to free up memory."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def decode_token(tokenizer, idx):
    """Decode a token and ensure it's a proper Unicode string."""
    token = tokenizer.decode([int(idx)])
    # Convert to actual characters by encoding and decoding through utf-8
    return token.encode('utf-8').decode('utf-8')


def process_images_logit_lens(model, preprocessor, dataset, num_images, prompt, 
                               use_n_token_only, top_k=5, llm_layers=None):
    """
    Process images and extract logit lens top-k predictions for visual tokens.
    
    Args:
        model: The Molmo model
        preprocessor: Data preprocessor
        dataset: Dataset to process
        num_images: Number of images to process
        prompt: Text prompt for the model
        use_n_token_only: Which visual tokens to use
        top_k: Number of top predictions to extract
        llm_layers: List of layer indices to analyze (None = all layers)
    
    Returns:
        List of results, one per image
    """
    # Distribute images across processes
    world_size = get_world_size()
    local_rank = get_local_rank()
    
    images_per_process = num_images // world_size
    start_idx = local_rank * images_per_process
    end_idx = start_idx + images_per_process
    
    # Handle remainder for last process
    if local_rank == world_size - 1:
        end_idx = num_images
    
    if local_rank == 0:
        print(f"Process {local_rank}: Processing images {start_idx} to {end_idx-1}")
    
    results = []
    device = torch.device(f"cuda:{local_rank}")
    
    for i in tqdm(range(start_idx, end_idx), desc=f"Rank {local_rank}"):
        example_data = dataset.get(i, np.random)
        
        # Extract ground truth caption
        caption_text = ""
        if "message_list" in example_data and len(example_data["message_list"]) > 0:
            message = example_data["message_list"][0]
            caption_text = message.get("text", "")
        
        # Create example
        example = {
            "image": example_data["image"],
            "messages": [prompt]
        }
        
        # Preprocess
        batch = preprocessor(example, rng=np.random)
        
        # Initialize results
        image_results = {
            "image_idx": i,
            "ground_truth_caption": caption_text,
            "layers": []  # Will store results per layer
        }
        
        # Run inference with hidden states
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                # Move to GPU
                input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).to(device)
                images_tensor = torch.tensor(batch.get("images")).unsqueeze(0).to(device)
                image_masks_tensor = torch.tensor(batch.get("image_masks")).unsqueeze(0).to(device) if batch.get("image_masks") is not None else None
                image_input_idx_tensor = torch.tensor(batch.get("image_input_idx")).unsqueeze(0).to(device) if batch.get("image_input_idx") is not None else None
                
                # Forward pass to get hidden states
                output = model(
                    input_ids=input_ids,
                    images=images_tensor,
                    image_masks=image_masks_tensor,
                    image_input_idx=image_input_idx_tensor,
                    output_hidden_states=True,
                    last_logits_only=False,
                )
                
                hidden_states = output.hidden_states  # Tuple of tensors, one per layer
                
                # Process each layer (or specified layers)
                layers_to_process = llm_layers if llm_layers is not None else range(len(hidden_states))
                
                for layer_idx in layers_to_process:
                    if layer_idx >= len(hidden_states):
                        continue
                    
                    # Get hidden state at this layer
                    hs = hidden_states[layer_idx]  # [B, seq_len, d_model]
                    
                    # Extract visual token positions
                    B = hs.shape[0]
                    num_chunks = image_input_idx_tensor.shape[1]
                    patches_per_chunk = image_input_idx_tensor.shape[2]
                    d_model = hs.shape[-1]
                    
                    # Gather visual token features
                    visual_features = torch.zeros(
                        (B, num_chunks, patches_per_chunk, d_model),
                        device=hs.device,
                        dtype=hs.dtype
                    )
                    
                    flat_positions = image_input_idx_tensor.view(B, -1)
                    valid_mask = flat_positions >= 0
                    
                    for b in range(B):
                        valid_pos = flat_positions[b][valid_mask[b]]
                        if valid_pos.numel() == 0:
                            continue
                        gathered = hs[b, valid_pos.long(), :]
                        visual_features.view(B, -1, d_model)[b, valid_mask[b], :] = gathered
                    
                    # Apply layer norm and output projection (logit lens)
                    visual_features_normed = unwrap_model(model).transformer.ln_f(visual_features)
                    logits = unwrap_model(model).transformer.ff_out(visual_features_normed)
                    
                    if unwrap_model(model).config.scale_logits:
                        logits = logits / math.sqrt(unwrap_model(model).config.d_model)
                    
                    # Get top-k predictions for each visual token
                    topk_values, topk_indices = torch.topk(logits, k=top_k, dim=-1)
                    
                    # Convert to CPU
                    topk_values = topk_values.cpu()
                    topk_indices = topk_indices.cpu()
                    
                    # Store results for this layer
                    layer_results = {
                        "layer_idx": layer_idx,
                        "chunks": []
                    }
                    
                    for chunk_idx in range(num_chunks):
                        chunk_results = {
                            "chunk_name": "Full Image" if chunk_idx == 0 else f"Chunk {chunk_idx}",
                            "patches": []
                        }
                        
                        for patch_idx in range(patches_per_chunk):
                            # Get top-k for this patch
                            patch_topk_values = topk_values[0, chunk_idx, patch_idx].numpy()
                            patch_topk_indices = topk_indices[0, chunk_idx, patch_idx].numpy()
                            
                            # Decode tokens
                            top_predictions = []
                            for val, idx in zip(patch_topk_values, patch_topk_indices):
                                token_str = decode_token(preprocessor.tokenizer, idx)
                                top_predictions.append({
                                    "token": token_str,
                                    "token_id": int(idx),
                                    "logit": float(val)
                                })
                            
                            # Add row/col info
                            row, col = patch_idx_to_row_col(patch_idx, patches_per_chunk)
                            
                            patch_results = {
                                "patch_idx": patch_idx,
                                "patch_row": row,
                                "patch_col": col,
                                "top_predictions": top_predictions
                            }
                            
                            chunk_results["patches"].append(patch_results)
                        
                        layer_results["chunks"].append(chunk_results)
                    
                    image_results["layers"].append(layer_results)
                
                # Clear tensors
                del input_ids, images_tensor, image_masks_tensor, image_input_idx_tensor
                del hidden_states, visual_features, logits, topk_values, topk_indices
                clear_gpu_memory()
        
        results.append(image_results)
        clear_gpu_memory()
    
    # Gather results from all processes
    if DISTRIBUTED_MODE:
        all_results = [None] * world_size
        dist.all_gather_object(all_results, results)

        # Combine results on main process
        if local_rank == 0:
            combined_results = []
            for process_results in all_results:
                combined_results.extend(process_results)
            return combined_results
        else:
            return None
    else:
        # Single-GPU mode: just return results directly
        return results


def main():
    # Initialize distributed training if in distributed mode
    if DISTRIBUTED_MODE:
        dist.init_process_group(backend="nccl")
        local_rank = get_local_rank()
        world_size = get_world_size()
        torch.cuda.set_device(f"cuda:{local_rank}")
        device = torch.device(f"cuda:{local_rank}")
    else:
        # Single-GPU mode
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser(description="Logit Lens Analysis for Visual Tokens (Multi-GPU)")
    parser.add_argument("--ckpt-path", type=str, required=True,
                       help="Path to Molmo checkpoint to analyze")
    parser.add_argument("--num-images", type=int, default=100,
                       help="Number of images to process")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"],
                       help="Dataset split to use")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of top predictions to extract (default: 5)")
    parser.add_argument("--layers", type=str, default=None,
                       help="Comma-separated list of layer indices to analyze (e.g., '0,8,16,23'). Default: all layers")
    parser.add_argument("--output-dir", type=str, default="analysis_results/logit_lens",
                       help="Output directory for results")
    args = parser.parse_args()
    
    # Parse layer indices
    if args.layers is not None:
        llm_layers = [int(x) for x in args.layers.split(',')]
    else:
        llm_layers = None
    
    if local_rank == 0:
        print(f"{'='*80}")
        print(f"Logit Lens Analysis for Visual Tokens (Multi-GPU)")
        print(f"{'='*80}\n")
        print(f"Checkpoint: {args.ckpt_path}")
        print(f"Dataset split: {args.split}")
        print(f"Number of images: {args.num_images}")
        print(f"Top-k predictions: {args.top_k}")
        print(f"Layers to analyze: {llm_layers if llm_layers else 'all'}")
        print(f"Running on {world_size} processes")
        print()
    
    # Load Molmo model on CPU first
    # Works with both full checkpoints and stripped (MLP-only) checkpoints
    if local_rank == 0:
        print(f"Loading Molmo model from {args.ckpt_path}...")
    
    cfg = TrainConfig.load(f"{args.ckpt_path}/config.yaml")
    cfg.model.init_device = None  # Override init_device to avoid meta tensors
    
    model = Molmo(cfg.model)
    
    # Load pretrained weights (LLM + ViT)
    model.reset_with_pretrained_weights()
    
    # Load checkpoint weights (works with both full and stripped checkpoints)
    checkpoint_weights = torch.load(f"{args.ckpt_path}/model.pt", map_location="cpu")
    model.load_state_dict(checkpoint_weights, strict=False)
    
    if local_rank == 0:
        print(f"Loaded {len(checkpoint_weights)} parameters from checkpoint")
    
    model.eval()

    # Wrap model with FSDP for sharding (distributed mode only)
    if DISTRIBUTED_MODE:
        if local_rank == 0:
            print("Wrapping model with FSDP for sharding...")

        wrap_policy = model.get_fsdp_wrap_policy("by_block_and_size")

        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            ),
            auto_wrap_policy=wrap_policy,
            device_id=local_rank,
            use_orig_params=True,
        )

        if local_rank == 0:
            print(f"Model wrapped with FSDP on device: {device}\n")
    else:
        # Single-GPU mode: just move to device
        model = model.to(device).to(torch.float16)
        print(f"Model moved to {device} (single-GPU mode)\n")
    
    # Create preprocessor
    if "hf:" in args.ckpt_path:
        model_config = unwrap_model(model).config
    else:
        model_config = ModelConfig.load(resource_path(args.ckpt_path, "config.yaml"), key="model", validate_paths=False)
    
    model_config.system_prompt_kind = "none"
    preprocessor = build_mm_preprocessor(
        model_config,
        for_inference=True,
        shuffle_messages=False,
        is_training=False,
        require_image_features=True
    )
    
    use_n_token_only = model_config.vision_backbone.use_n_token_only
    
    # Load dataset
    if local_rank == 0:
        print(f"Loading PixMo-Cap {args.split} split...")
    dataset = PixMoCap(split=args.split, mode="captions")
    if local_rank == 0:
        print()
    
    # Wait for all processes to be ready
    if DISTRIBUTED_MODE:
        dist.barrier()

    # Process images (distributed across GPUs)
    prompt = "Describe this image in detail."
    if local_rank == 0:
        print(f"Processing {args.num_images} images across {world_size} GPU(s)...")
    results = process_images_logit_lens(
        model, preprocessor, dataset, args.num_images, prompt, use_n_token_only,
        top_k=args.top_k, llm_layers=llm_layers
    )

    # Wait for all processes to finish
    if DISTRIBUTED_MODE:
        dist.barrier()
    
    # Save results (only on main process)
    if local_rank == 0:
        # Setup output directory
        ckpt_name = args.ckpt_path.split("/")[-2] + "_" + args.ckpt_path.split("/")[-1]
        output_dir = Path(args.output_dir) / ckpt_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Group results by layer and save separately
        print(f"\nSaving results to {output_dir}...")
        
        # Collect all unique layer indices from results
        all_layer_indices = set()
        for image_result in results:
            for layer_data in image_result['layers']:
                all_layer_indices.add(layer_data['layer_idx'])
        
        all_layer_indices = sorted(all_layer_indices)
        print(f"Found {len(all_layer_indices)} layers to save: {all_layer_indices}")
        
        # Save each layer separately
        for layer_idx in all_layer_indices:
            # Extract data for this layer from all images
            layer_results = []
            for image_result in results:
                # Find the layer data for this specific layer
                layer_data = None
                for ld in image_result['layers']:
                    if ld['layer_idx'] == layer_idx:
                        layer_data = ld
                        break
                
                if layer_data is not None:
                    layer_results.append({
                        'image_idx': image_result['image_idx'],
                        'ground_truth_caption': image_result['ground_truth_caption'],
                        'chunks': layer_data['chunks']
                    })
            
            # Save to file
            output_file = output_dir / f"logit_lens_layer{layer_idx}_topk{args.top_k}_multi-gpu.json"
            
            output_data = {
                'checkpoint': args.ckpt_path,
                'split': args.split,
                'num_images': args.num_images,
                'num_processes': world_size,
                'top_k': args.top_k,
                'layer_idx': layer_idx,
                'results': layer_results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"  Saved layer {layer_idx} -> {output_file.name}")
        
        print(f"\n✓ Saved {len(all_layer_indices)} layer files to {output_dir}")
        
        # Print some example results
        print(f"\n{'='*80}")
        print(f"EXAMPLE RESULTS (First image, first patch, first layer)")
        print(f"{'='*80}\n")
        
        if results and len(results) > 0:
            first_image = results[0]
            print(f"Image {first_image['image_idx']}")
            print(f"Ground truth: {first_image['ground_truth_caption'][:100]}...\n")
            
            if first_image['layers'] and len(first_image['layers']) > 0:
                first_layer = first_image['layers'][0]
                print(f"Layer {first_layer['layer_idx']}:")
                
                if first_layer['chunks'] and len(first_layer['chunks']) > 0:
                    first_chunk = first_layer['chunks'][0]
                    if first_chunk['patches'] and len(first_chunk['patches']) > 0:
                        first_patch = first_chunk['patches'][0]
                        print(f"  Patch {first_patch['patch_idx']} (row={first_patch['patch_row']}, col={first_patch['patch_col']}):")
                        print(f"  Top-{args.top_k} predictions:")
                        for i, pred in enumerate(first_patch['top_predictions'], 1):
                            print(f"    {i}. '{pred['token']}' (logit={pred['logit']:.3f})")
                        print()
        
        print(f"\n✓ Analysis complete!")
        print(f"Results saved to: {output_dir}")

    # Wait for main process to finish saving
    if DISTRIBUTED_MODE:
        dist.barrier()


if __name__ == "__main__":
    main()
