"""EmbeddingLens: Nearest Neighbor Analysis in LLM Embedding Space

This script analyzes visual tokens by finding their nearest neighbors in the LLM's
input embedding matrix. For each visual token at each layer, we extract the top-5
most similar vocabulary tokens.

Usage:
    # Single GPU (auto-detected):
    python scripts/run_embedding_lens.py --ckpt-path <path> --num-images 100

    # Multi-GPU with torchrun:
    torchrun --nproc_per_node=2 scripts/run_embedding_lens.py --ckpt-path <path> --num-images 100

Part of the LatentLens package: https://github.com/McGill-NLP/latentlens
"""
import logging
import sys
import json
import re
import os
from pathlib import Path
import argparse
import math

import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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

# Add translation library (robust to env/version issues)
try:
    from googletrans import Translator
    TRANSLATION_AVAILABLE = True
except Exception as e:
    TRANSLATION_AVAILABLE = False
    print(f"Warning: translation disabled (googletrans import error: {e}). To enable, install compatible versions, e.g.:\n"
          f"  pip install 'googletrans==4.0.0rc1' 'httpx<0.24' 'httpcore<1.0'")

from molmo.config import ModelConfig, TrainConfig
from molmo.data import build_mm_preprocessor
from molmo.model import Molmo
from molmo.util import resource_path
from molmo.data.pixmo_datasets import PixMoCap, PixMoPoints
from molmo.data.preprocessor import resize_and_pad, siglip_resize_and_pad

log = logging.getLogger(__name__)

# Global translator instance
_translator = None
_translation_cache = {}
_cache_file = Path("translation_cache.json")
_cache_dirty = False
_cache_save_counter = 0

def load_translation_cache():
    """Load translation cache from file."""
    global _translation_cache
    if _cache_file.exists():
        try:
            with open(_cache_file, 'r', encoding='utf-8') as f:
                _translation_cache = json.load(f)
            print(f"Loaded {len(_translation_cache)} cached translations from {_cache_file}")
        except Exception as e:
            print(f"Could not load translation cache: {e}")
            _translation_cache = {}
    else:
        _translation_cache = {}

def save_translation_cache():
    """Save translation cache to file."""
    global _cache_dirty
    if _cache_dirty:
        try:
            with open(_cache_file, 'w', encoding='utf-8') as f:
                json.dump(_translation_cache, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(_translation_cache)} translations to cache")
            _cache_dirty = False
        except Exception as e:
            print(f"Could not save translation cache: {e}")

def get_translator():
    """Get or create a global translator instance."""
    global _translator
    if _translator is None and TRANSLATION_AVAILABLE:
        _translator = Translator()
        # Load cache when we first create the translator
        load_translation_cache()
    return _translator

def contains_chinese(text):
    """Check if text contains Chinese characters."""
    if not text:
        return False
    # Check for Chinese characters (CJK Unified Ideographs)
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

def translate_chinese_token(token_text):
    """Translate Chinese tokens to English, return list of possible translations."""
    global _cache_dirty, _cache_save_counter
    
    if not TRANSLATION_AVAILABLE or not contains_chinese(token_text):
        return [token_text]
    
    # Check cache first
    if token_text in _translation_cache:
        cached_result = _translation_cache[token_text]
        if cached_result != token_text:  # If we have a translation
            return [token_text, cached_result]
        else:
            return [token_text]
    
    translator = get_translator()
    if translator is None:
        return [token_text]
    
    try:
        # Clean the token text
        clean_text = token_text.strip()
        if not clean_text:
            return [token_text]
        
        # Translate to English - use 'zh-cn' instead of 'zh' for Chinese
        result = translator.translate(clean_text, src='zh-cn', dest='en')
        if result and result.text:
            translated = result.text.lower().strip()
            
            # Cache the result
            _translation_cache[token_text] = translated
            _cache_dirty = True
            _cache_save_counter += 1
            
            # Save cache every 100 translations
            if _cache_save_counter >= 100:
                save_translation_cache()
                _cache_save_counter = 0
            
            # Return both original and translated versions
            if translated != clean_text.lower():
                print(f"Translated '{token_text}' -> '{translated}'")
                return [token_text, translated]
            else:
                return [token_text]
        else:
            print(f"Translation failed for '{token_text}': No result")
            # Cache the failure (store original token)
            _translation_cache[token_text] = token_text
            _cache_dirty = True
            return [token_text]
    except Exception as e:
        print(f"Translation failed for '{token_text}': {e}")
        # Cache the failure (store original token)
        _translation_cache[token_text] = token_text
        _cache_dirty = True
        return [token_text]

def decode_token(tokenizer, idx):
    """Decode a token and ensure it's a proper Unicode string."""
    token = tokenizer.decode([int(idx)])
    # Convert to actual characters by encoding and decoding through utf-8
    return token.encode('utf-8').decode('utf-8')

def clear_gpu_memory():
    """Clear CUDA cache to free up memory."""
    import gc
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    # Force another garbage collection after CUDA cleanup
    gc.collect()

def normalize_text_for_matching(text):
    """Normalize text for fuzzy matching: lowercase, remove punctuation, split by whitespace, filter stopwords."""
    # Define common stopwords to exclude from matching
    stopwords = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 
        'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with', 'the', 'this', 'these', 
        'those', 'they', 'them', 'their', 'there', 'then', 'than', 'but', 'or', 'so', 'if', 'when', 
        'where', 'why', 'how', 'what', 'who', 'which', 'can', 'could', 'would', 'should', 'may', 'might'
    }
    
    # Define visual/task words that indicate interface or visual elements
    visual_task_words = {
        'image', 'photo', 'picture', 'img', 'display', 'frame', 'screen', 'view', 'window', 'panel', 'icon', 'cursor', 'pixel', 'resolution', 'monitor', 'camera',
        'lens', 'focus', 'zoom', 'crop', 'filter', 'edit', 'save', 'load', 'file', 'folder', 'document',
        'page', 'layout', 'design', 'graphic', 'visual', 'render', 'preview', 'thumbnail', 'gallery',
        'album', 'slide', 'presentation', 'video', 'clip', 'frame', 'shot', 'capture', 'record'
    }
    
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters, but preserve Unicode letters
    # This regex keeps Unicode letters (\w), digits, and spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Split by whitespace and filter out empty strings and stopwords
    words = [word.strip() for word in text.split() 
             if word.strip() and word.strip() not in stopwords and len(word.strip()) > 1]
    return words, visual_task_words

def check_match_type(token_words, caption_words, visual_task_words):
    """Check what type of match this is: interpretable, visual_task, or none.
    
    Returns:
        tuple: (match_type, matches_list)
        match_type: 'interpretable', 'visual_task', or 'none'
        matches_list: list of match dictionaries
    """
    matches = []
    has_interpretable = False
    has_visual_task = False
    
    for token_word in token_words:
        # Check for visual/task matches first
        if token_word in visual_task_words:
            matches.append({
                "token_word": token_word,
                "match_type": "visual_task",
                "matched_word": token_word
            })
            has_visual_task = True
        
        # Check for interpretable matches with caption
        for caption_word in caption_words:
            if token_word == caption_word:
                matches.append({
                    "token_word": token_word,
                    "match_type": "interpretable", 
                    "matched_word": caption_word
                })
                has_interpretable = True
    
    # Determine overall match type (visual_task takes precedence)
    if has_visual_task:
        match_type = 'visual_task'
    elif has_interpretable:
        match_type = 'interpretable'
    else:
        match_type = 'none'
    
    return match_type, matches

def check_match_type_with_translation(token_text, caption_words, visual_task_words):
    """Enhanced matching that includes Chinese translation support.
    
    Args:
        token_text: Original token text (may contain Chinese)
        caption_words: List of normalized words from the caption
        visual_task_words: Set of visual/task interface words
    
    Returns:
        tuple: (match_type, matches_list)
        match_type: 'interpretable', 'visual_task', or 'none'
        matches_list: list of match dictionaries
    """
    # Get all possible translations of the token
    token_translations = translate_chinese_token(token_text)
    
    # Normalize each translation
    all_token_words = []
    for translation in token_translations:
        normalized_words, _ = normalize_text_for_matching(translation)
        all_token_words.extend(normalized_words)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_token_words = []
    for word in all_token_words:
        if word not in seen:
            seen.add(word)
            unique_token_words.append(word)
    
    # Use the existing check_match_type function
    return check_match_type(unique_token_words, caption_words, visual_task_words)


def patch_idx_to_row_col(patch_idx, patches_per_chunk):
    """Convert patch index to row and column coordinates.
    
    Args:
        patch_idx: Index of the patch (0-based)
        patches_per_chunk: Total number of patches (e.g., 144 for 12x12, 576 for 24x24)
    
    Returns:
        tuple: (row, col) where row and col are 0-based coordinates
    
    Example:
        For 24x24 grid (576 patches): patch_idx=27 -> row=1, col=3
        For 12x12 grid (144 patches): patch_idx=27 -> row=2, col=3
    """
    grid_size = int(math.sqrt(patches_per_chunk))
    row = patch_idx // grid_size
    col = patch_idx % grid_size
    return row, col


def create_interpretability_plot(overall_statistics, output_path):
    """Create a bar plot showing overall interpretability statistics."""
    if not overall_statistics:
        log.warning("No statistics available for plotting")
        return
    
    # Create overall interpretability plot
    interpretable = overall_statistics.get("interpretable_visual_tokens", 0)
    total = overall_statistics.get("total_visual_tokens", 1)
    non_interpretable = total - interpretable
    
    plt.figure(figsize=(8, 6))
    categories = ['Interpretable', 'Non-interpretable']
    values = [interpretable, non_interpretable]
    colors = ['steelblue', 'lightcoral']
    
    bars = plt.bar(categories, values, color=colors, alpha=0.7)
    
    # Add percentage labels on bars
    for bar, value in zip(bars, values):
        percentage = (value / total * 100) if total > 0 else 0
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01, 
                f'{value}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Number of Visual Tokens')
    plt.title(f'Visual Token Interpretability Analysis\n(Total: {total} tokens)')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Interpretability plot saved to {output_path}")

def create_token_position_plot(token_position_stats, output_path, top_n=20):
    """Create a bar plot showing top token positions by interpretability percentage."""
    if not token_position_stats:
        log.warning("No token position statistics available for plotting")
        return
    
    # Determine patches_per_chunk from the available patch indices
    max_patch_idx = max(token_position_stats.keys()) if token_position_stats else 0
    # Estimate grid size - assume it's a perfect square
    # Add 1 because patch indices are 0-based
    estimated_total_patches = max_patch_idx + 1
    estimated_grid_size = int(math.sqrt(estimated_total_patches))
    
    # Check if it's likely a perfect square grid (144 or 576 are common)
    if estimated_grid_size * estimated_grid_size == estimated_total_patches:
        patches_per_chunk = estimated_total_patches
    elif estimated_total_patches <= 144:
        patches_per_chunk = 144  # 12x12 grid
    elif estimated_total_patches <= 576:
        patches_per_chunk = 576  # 24x24 grid
    else:
        # For larger grids, find the next perfect square
        estimated_grid_size = int(math.ceil(math.sqrt(estimated_total_patches)))
        patches_per_chunk = estimated_grid_size * estimated_grid_size
    
    # Sort by interpretability percentage and take top N
    sorted_positions = sorted(token_position_stats.items(), 
                            key=lambda x: x[1].get("interpretability_percentage", 0), 
                            reverse=True)[:top_n]
    
    # Create labels with patch_idx and row/col information
    position_labels = []
    for pos, _ in sorted_positions:
        row, col = patch_idx_to_row_col(pos, patches_per_chunk)
        position_labels.append(f"{pos}\n(r{row},c{col})")
    
    percentages = [stats.get("interpretability_percentage", 0) for _, stats in sorted_positions]
    
    plt.figure(figsize=(15, 6))  # Made wider to accommodate longer labels
    bars = plt.bar(range(len(position_labels)), percentages, color='forestgreen', alpha=0.7)
    
    plt.xlabel('Token Position (patch_idx and row,col coordinates)')
    plt.ylabel('Interpretability Percentage (%)')
    plt.title(f'Top {top_n} Token Positions by Interpretability')
    plt.xticks(range(len(position_labels)), position_labels, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, percentage) in enumerate(zip(bars, percentages)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Token position plot saved to {output_path}")


def is_already_translated(token_text):
    """Check if a token has already been translated (contains 'translated from' pattern)."""
    return " (translated from " in token_text

def get_display_token(original_token):
    """Get display version of token, showing translation with original Chinese in brackets."""
    # If already translated, return as-is
    if is_already_translated(original_token):
        return original_token
        
    if not TRANSLATION_AVAILABLE or not contains_chinese(original_token):
        return original_token
    
    # Check cache first
    if original_token in _translation_cache:
        cached_result = _translation_cache[original_token]
        if cached_result != original_token:  # If we have a translation
            return f"{cached_result} (translated from {original_token})"
        else:
            return original_token
    
    # If not in cache, try to translate and cache it
    translator = get_translator()
    if translator is None:
        return original_token
    
    try:
        clean_text = original_token.strip()
        if not clean_text:
            return original_token
        
        result = translator.translate(clean_text, src='zh-cn', dest='en')
        if result and result.text:
            translated = result.text.lower().strip()
            
            # Cache the result
            _translation_cache[original_token] = translated
            global _cache_dirty, _cache_save_counter
            _cache_dirty = True
            _cache_save_counter += 1
            
            # Save cache every 100 translations
            if _cache_save_counter >= 100:
                save_translation_cache()
                _cache_save_counter = 0
            
            if translated != clean_text.lower():
                print(f"Translated '{original_token}' -> '{translated}'")
                return f"{translated} (translated from {original_token})"
            else:
                return original_token
        else:
            # Cache the failure
            _translation_cache[original_token] = original_token
            _cache_dirty = True
            return original_token
    except Exception as e:
        print(f"Translation failed for '{original_token}': {e}")
        # Cache the failure
        _translation_cache[original_token] = original_token
        _cache_dirty = True
        return original_token

def process_split(model, preprocessor, dataset, num_images, prompt, use_n_token_only, token_embeddings, split_name="", llm_layer: int = 0, generate_captions: bool = False):
    """Process a dataset split and return results with distributed processing."""
    # Distribute images across processes
    world_size = get_world_size()
    local_rank = get_local_rank()
    
    print(f"[DEBUG Rank {local_rank}] Entered process_split, num_images={num_images}, world_size={world_size}")
    
    images_per_process = num_images // world_size
    start_idx = local_rank * images_per_process
    end_idx = start_idx + images_per_process
    
    # Handle remainder for last process
    if local_rank == world_size - 1:
        end_idx = num_images
    
    print(f"[DEBUG Rank {local_rank}] Will process images {start_idx} to {end_idx-1}")
    if local_rank == 0:
        print(f"Process {local_rank}: Processing images {start_idx} to {end_idx-1}")
    
    split_results = []
    
    # Initialize statistics tracking
    total_visual_tokens = 0
    interpretable_visual_tokens = 0
    visual_task_visual_tokens = 0
    token_position_stats = {}
    
    print(f"[DEBUG Rank {local_rank}] About to start loop over {end_idx - start_idx} images")
    
    # Process assigned images for this process
    for i in tqdm(range(start_idx, end_idx), desc=f"Rank {local_rank}"):
        if i == start_idx:
            print(f"[DEBUG Rank {local_rank}] Starting iteration i={i}")
        example_data = dataset.get(i, np.random)
        if i == start_idx:
            print(f"[DEBUG Rank {local_rank}] Got example_data for i={i}")
        
        # Extract ground truth caption
        # For PixMoCap: extract from "text" field
        # For PixMoPoints: extract from "label" field (combine all labels)
        caption_text = ""
        if "message_list" in example_data and len(example_data["message_list"]) > 0:
            if "text" in example_data["message_list"][0]:
                # PixMoCap format: has "text" field
                message = example_data["message_list"][0]
                caption_text = message.get("text", "")
            elif "label" in example_data["message_list"][0]:
                # PixMoPoints format: has "label" field
                # Combine all labels from all messages
                labels = [msg.get("label", "") for msg in example_data["message_list"] if "label" in msg]
                caption_text = " ".join(labels)
        
        # Normalize caption words for matching
        caption_words, visual_task_words = normalize_text_for_matching(caption_text)
        
        # Create example with the provided prompt
        example = {
            "image": example_data["image"],
            "messages": [prompt]
        }

        # Preprocess example
        batch = preprocessor(example, rng=np.random)

        # Initialize image results
        image_results = {
            "image_idx": i,
            "ground_truth_caption": caption_text,
            "chunks": []
        }

        # Run inference
        if i == start_idx:
            print(f"[DEBUG Rank {local_rank}] About to run inference for i={i}")
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                # Move data to GPU
                device = torch.device(f"cuda:{local_rank}")
                if i == start_idx:
                    print(f"[DEBUG Rank {local_rank}] Moving tensors to device {device}")
                images_tensor = torch.tensor(batch.get("images")).unsqueeze(0).to(device)
                image_masks_tensor = torch.tensor(batch.get("image_masks")).unsqueeze(0).to(device) if batch.get("image_masks") is not None else None
                
                if i == start_idx:
                    print(f"[DEBUG Rank {local_rank}] About to call model forward with llm_layer={llm_layer}")
                # Select which features to use for analysis:
                #  - llm_layer == 0: use vision backbone projected features (existing behavior)
                #  - llm_layer > 0: use hidden states from the specified LLM layer at visual token positions
                if llm_layer == 0:
                    image_features, tokens_before_MLP = model.vision_backbone(images_tensor, image_masks_tensor, return_tokens_before_MLP=True)
                    if i == start_idx:
                        print(f"[DEBUG Rank {local_rank}] Got image_features from vision_backbone")
                    # Handle use_n_token_only properly
                    if type(use_n_token_only) == int and use_n_token_only != -1:
                        image_features = image_features[:, :, :use_n_token_only, :]
                    elif type(use_n_token_only) == list and len(use_n_token_only) > 0:
                        image_features = image_features[:, :, use_n_token_only, :]
                else:
                    # Forward through the LLM to get hidden states
                    input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).to(device)
                    image_input_idx_tensor = torch.tensor(batch.get("image_input_idx")).unsqueeze(0).to(device) if batch.get("image_input_idx") is not None else None
                    output = model(
                        input_ids=input_ids,
                        images=images_tensor,
                        image_masks=image_masks_tensor,
                        image_input_idx=image_input_idx_tensor,
                        output_hidden_states=True,
                        last_logits_only=False,
                    )
                    hidden_states = output.hidden_states  # Tuple[Tensor]
                    # Bound-check layer index against available hidden states
                    max_layer_index = len(hidden_states) - 1  # includes final ln_f state
                    assert image_input_idx_tensor is not None, "image_input_idx is required to extract visual tokens from LLM layers"
                    
                    # Check if requested layer is valid
                    if llm_layer > max_layer_index:
                        raise ValueError(
                            f"Requested layer {llm_layer} is out of bounds. "
                            f"Model has {len(hidden_states)} hidden states (layers 0-{max_layer_index}). "
                            f"Please use a layer index between 0 and {max_layer_index}."
                        )
                    
                    # Helper to gather features from a specific layer index
                    def gather_visual_features_from_layer(hs_tensor):
                        B = hs_tensor.shape[0]
                        num_chunks_local = image_input_idx_tensor.shape[1]
                        patches_per_chunk_local = image_input_idx_tensor.shape[2]
                        d_model_local = hs_tensor.shape[-1]
                        feats = torch.zeros((B, num_chunks_local, patches_per_chunk_local, d_model_local), device=hs_tensor.device, dtype=hs_tensor.dtype)
                        flat_positions_local = image_input_idx_tensor.view(B, -1)
                        valid_mask_local = flat_positions_local >= 0
                        for b in range(B):
                            valid_pos = flat_positions_local[b][valid_mask_local[b]]
                            if valid_pos.numel() == 0:
                                continue
                            gathered = hs_tensor[b, valid_pos.long(), :]
                            feats.view(B, -1, d_model_local)[b, valid_mask_local[b], :] = gathered
                        return feats

                    layer_index = llm_layer
                    hs = hidden_states[layer_index]
                    image_features = gather_visual_features_from_layer(hs)
                image_results["feature_shape"] = list(image_features.shape)
                image_results["llm_layer_used"] = llm_layer
                
                # Use the pre-loaded token embeddings (passed as parameter)
                # Reshape image features to combine batch and chunks dimensions
                batch_size, num_chunks, patches_per_chunk, hidden_dim = image_features.shape
                image_features_reshaped = image_features.view(-1, patches_per_chunk, hidden_dim)
                
                # Normalize the embeddings for cosine similarity
                image_features_norm = torch.nn.functional.normalize(image_features_reshaped, dim=-1)
                token_embeddings_norm = torch.nn.functional.normalize(token_embeddings, dim=-1)
                
                # Compute cosine similarity for each patch
                similarity = torch.matmul(image_features_norm, token_embeddings_norm.T)
                
                # Get top-5 most similar tokens for each patch
                top_k = 5
                top_values, top_indices = torch.topk(similarity, k=top_k, dim=-1)
                
                # Clear intermediate tensors
                del similarity, image_features_norm, token_embeddings_norm
                clear_gpu_memory()

                # Generate caption if requested
                if generate_captions:
                    # Important: Ensure FSDP state is clean before generate
                    torch.cuda.synchronize()
                    if DISTRIBUTED_MODE:
                        dist.barrier()
                    input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).to(device)
                    image_input_idx_tensor = torch.tensor(batch.get("image_input_idx")).unsqueeze(0).to(device) if batch.get("image_input_idx") is not None else None
                    
                    # Regenerate images_tensor and image_masks_tensor for generation (they were deleted earlier)
                    images_tensor_gen = torch.tensor(batch.get("images")).unsqueeze(0).to(device)
                    image_masks_tensor_gen = torch.tensor(batch.get("image_masks")).unsqueeze(0).to(device) if batch.get("image_masks") is not None else None
                    
                    output = model.generate(
                        input_ids=input_ids,
                        images=images_tensor_gen,
                        image_masks=image_masks_tensor_gen,
                        image_input_idx=image_input_idx_tensor,
                        max_steps=200,
                        min_steps=4,
                        is_distributed=False
                    )
                    token_ids = output.token_ids[:, 0].detach().cpu().numpy()
                    decoded = preprocessor.tokenizer.decode(token_ids[0])
                    image_results["generated_response"] = decoded
                    
                    # Clear generation tensors
                    del input_ids, image_input_idx_tensor, images_tensor_gen, image_masks_tensor_gen, output
                    clear_gpu_memory()

                # Clear GPU tensors
                del images_tensor, image_masks_tensor, image_features
                clear_gpu_memory()

                # Store results for each chunk
                for chunk_idx in range(num_chunks):
                    chunk_results = {
                        "chunk_name": "Full Image" if chunk_idx == 0 else f"Chunk {chunk_idx}",
                        "patches": []
                    }
                    
                    for patch_idx in range(patches_per_chunk):
                        patch_values = top_values[chunk_idx, patch_idx].cpu().numpy().tolist()
                        patch_indices = top_indices[chunk_idx, patch_idx].cpu().numpy().tolist()
                        patch_tokens = [decode_token(preprocessor.tokenizer, idx) for idx in patch_indices]
                        
                        # Initialize token position stats if not exists
                        if patch_idx not in token_position_stats:
                            row, col = patch_idx_to_row_col(patch_idx, patches_per_chunk)
                            token_position_stats[patch_idx] = {
                                "total_occurrences": 0,
                                "interpretable_occurrences": 0,
                                "visual_task_occurrences": 0,
                                "interpretability_percentage": 0.0,
                                "visual_task_percentage": 0.0,
                                "patch_row": row,
                                "patch_col": col
                            }
                        
                        token_position_stats[patch_idx]["total_occurrences"] += 1
                        
                        # Check for matches using three-category system
                        overall_match_type = 'none'
                        all_matches = []
                        
                        for j, (token_text, similarity_score) in enumerate(zip(patch_tokens, patch_values)):
                            # Check what type of match this token has (with translation support)
                            match_type, matches = check_match_type_with_translation(token_text, caption_words, visual_task_words)
                            
                            if match_type != 'none':
                                # Update overall match type (visual_task takes precedence over interpretable)
                                if match_type == 'visual_task' or (overall_match_type != 'visual_task' and match_type == 'interpretable'):
                                    overall_match_type = match_type
                                
                                # Add detailed match info
                                for match in matches:
                                    all_matches.append({
                                        "token": get_display_token(token_text),
                                        "token_word": match["token_word"],
                                        "matched_word": match["matched_word"],
                                        "match_type": match["match_type"],
                                        "similarity": float(similarity_score)
                                    })
                        
                        # Update statistics
                        total_visual_tokens += 1
                        if overall_match_type == 'interpretable':
                            interpretable_visual_tokens += 1
                            token_position_stats[patch_idx]["interpretable_occurrences"] += 1
                        elif overall_match_type == 'visual_task':
                            visual_task_visual_tokens += 1
                            token_position_stats[patch_idx]["visual_task_occurrences"] += 1
                        
                        # Add row/col information based on patch_idx
                        row, col = patch_idx_to_row_col(patch_idx, patches_per_chunk)
                        
                        patch_results = {
                            "patch_idx": patch_idx,
                            "patch_row": row,
                            "patch_col": col,
                            "nearest_neighbors": [
                                {"token": get_display_token(token), "similarity": float(sim)}
                                for token, sim in zip(patch_tokens, patch_values)
                            ],
                            "match_type": overall_match_type,
                            "caption_match": overall_match_type == 'interpretable',
                            "visual_task_match": overall_match_type == 'visual_task'
                        }
                        
                        # Add detailed matches if any were found
                        if all_matches:
                            patch_results["matches"] = all_matches
                        
                        chunk_results["patches"].append(patch_results)
                    
                    image_results["chunks"].append(chunk_results)

        split_results.append(image_results)
        
        # Clear memory after each image to prevent OOM
        clear_gpu_memory()

    # Gather results from all processes
    all_split_results = [None] * world_size
    all_token_position_stats = [None] * world_size

    # Use all_gather to collect results from all processes
    if DISTRIBUTED_MODE:
        dist.all_gather_object(all_split_results, split_results)
        dist.all_gather_object(all_token_position_stats, token_position_stats)
    else:
        all_split_results = [split_results]
        all_token_position_stats = [token_position_stats]
    
    # Combine results on main process
    if local_rank == 0:
        # Flatten the gathered results
        combined_results = []
        for process_results in all_split_results:
            combined_results.extend(process_results)
        
        # Combine token position statistics
        combined_token_position_stats = {}
        for process_stats in all_token_position_stats:
            for patch_idx, stats in process_stats.items():
                if patch_idx not in combined_token_position_stats:
                    combined_token_position_stats[patch_idx] = {
                        "total_occurrences": 0,
                        "interpretable_occurrences": 0,
                        "visual_task_occurrences": 0,
                        "interpretability_percentage": 0.0,
                        "visual_task_percentage": 0.0,
                        "patch_row": stats.get("patch_row", 0),
                        "patch_col": stats.get("patch_col", 0)
                    }
                combined_token_position_stats[patch_idx]["total_occurrences"] += stats["total_occurrences"]
                combined_token_position_stats[patch_idx]["interpretable_occurrences"] += stats["interpretable_occurrences"]
                combined_token_position_stats[patch_idx]["visual_task_occurrences"] += stats["visual_task_occurrences"]
        
        # Calculate interpretability percentages for token positions
        for pos_stats in combined_token_position_stats.values():
            total = pos_stats["total_occurrences"]
            interpretable = pos_stats["interpretable_occurrences"]
            visual_task = pos_stats["visual_task_occurrences"]
            pos_stats["interpretability_percentage"] = (interpretable / total * 100) if total > 0 else 0.0
            pos_stats["visual_task_percentage"] = (visual_task / total * 100) if total > 0 else 0.0
        
        # Calculate overall statistics
        total_visual_tokens = sum(pos_stats["total_occurrences"] for pos_stats in combined_token_position_stats.values())
        interpretable_visual_tokens = sum(pos_stats["interpretable_occurrences"] for pos_stats in combined_token_position_stats.values())
        visual_task_visual_tokens = sum(pos_stats["visual_task_occurrences"] for pos_stats in combined_token_position_stats.values())
        
        overall_statistics = {
            "total_visual_tokens": total_visual_tokens,
            "interpretable_visual_tokens": interpretable_visual_tokens,
            "visual_task_visual_tokens": visual_task_visual_tokens,
            "interpretability_percentage": (interpretable_visual_tokens / total_visual_tokens * 100) if total_visual_tokens > 0 else 0,
            "visual_task_percentage": (visual_task_visual_tokens / total_visual_tokens * 100) if total_visual_tokens > 0 else 0,
            "token_position_statistics": combined_token_position_stats
        }
        return combined_results, overall_statistics
    else:
        # Non-main processes return None
        return None, None

def process_split_all_layers(model, preprocessor, dataset, num_images, prompt, use_n_token_only, token_embeddings, split_name="", target_layers=None, generate_captions: bool = False):
    """Process a dataset split for ALL layers in a single pass over images.

    Instead of doing one forward pass per (image, layer) pair, does one forward
    pass per image and extracts features for all requested layers. This gives
    ~Nx speedup where N is the number of layers.

    Args:
        target_layers: List of LLM layer indices to process (e.g., [0, 1, 2, 4, 8, 16, 24, 30, 31])

    Returns:
        On rank 0: dict mapping layer_idx -> (results_list, statistics_dict)
        On other ranks: None
    """
    if target_layers is None:
        target_layers = [0]

    world_size = get_world_size()
    local_rank = get_local_rank()

    images_per_process = num_images // world_size
    start_idx = local_rank * images_per_process
    end_idx = start_idx + images_per_process
    if local_rank == world_size - 1:
        end_idx = num_images

    if local_rank == 0:
        print(f"Process {local_rank}: Processing images {start_idx} to {end_idx-1} for {len(target_layers)} layers")

    device = torch.device(f"cuda:{local_rank}")

    # Determine if we need LLM forward pass (any layer > 0)
    needs_llm_forward = any(layer > 0 for layer in target_layers)
    has_layer_zero = 0 in target_layers
    non_zero_layers = [l for l in target_layers if l > 0]

    # Per-layer accumulators
    per_layer_results = {layer: [] for layer in target_layers}
    per_layer_stats = {}
    for layer in target_layers:
        per_layer_stats[layer] = {
            "total_visual_tokens": 0,
            "interpretable_visual_tokens": 0,
            "visual_task_visual_tokens": 0,
            "token_position_stats": {},
        }

    for i in tqdm(range(start_idx, end_idx), desc=f"Rank {local_rank}"):
        example_data = dataset.get(i, np.random)

        # Extract ground truth caption
        caption_text = ""
        if "message_list" in example_data and len(example_data["message_list"]) > 0:
            if "text" in example_data["message_list"][0]:
                message = example_data["message_list"][0]
                caption_text = message.get("text", "")
            elif "label" in example_data["message_list"][0]:
                labels = [msg.get("label", "") for msg in example_data["message_list"] if "label" in msg]
                caption_text = " ".join(labels)

        caption_words, visual_task_words = normalize_text_for_matching(caption_text)

        example = {
            "image": example_data["image"],
            "messages": [prompt]
        }
        batch = preprocessor(example, rng=np.random)

        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                images_tensor = torch.tensor(batch.get("images")).unsqueeze(0).to(device)
                image_masks_tensor = torch.tensor(batch.get("image_masks")).unsqueeze(0).to(device) if batch.get("image_masks") is not None else None

                # Collect features per layer from minimal forward passes
                layer_features = {}  # layer_idx -> image_features tensor

                # Layer 0: vision backbone only (no LLM forward pass)
                if has_layer_zero:
                    vb_features, _ = model.vision_backbone(images_tensor, image_masks_tensor, return_tokens_before_MLP=True)
                    if type(use_n_token_only) == int and use_n_token_only != -1:
                        vb_features = vb_features[:, :, :use_n_token_only, :]
                    elif type(use_n_token_only) == list and len(use_n_token_only) > 0:
                        vb_features = vb_features[:, :, use_n_token_only, :]
                    layer_features[0] = vb_features

                # Layers > 0: single LLM forward pass, extract all layers
                if non_zero_layers:
                    input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).to(device)
                    image_input_idx_tensor = torch.tensor(batch.get("image_input_idx")).unsqueeze(0).to(device) if batch.get("image_input_idx") is not None else None
                    output = model(
                        input_ids=input_ids,
                        images=images_tensor,
                        image_masks=image_masks_tensor,
                        image_input_idx=image_input_idx_tensor,
                        output_hidden_states=True,
                        last_logits_only=False,
                    )
                    hidden_states = output.hidden_states
                    max_layer_index = len(hidden_states) - 1
                    assert image_input_idx_tensor is not None, "image_input_idx is required to extract visual tokens from LLM layers"

                    # Helper to gather visual features from a hidden state tensor
                    def gather_visual_features(hs_tensor):
                        B = hs_tensor.shape[0]
                        num_chunks_local = image_input_idx_tensor.shape[1]
                        patches_per_chunk_local = image_input_idx_tensor.shape[2]
                        d_model_local = hs_tensor.shape[-1]
                        feats = torch.zeros((B, num_chunks_local, patches_per_chunk_local, d_model_local), device=hs_tensor.device, dtype=hs_tensor.dtype)
                        flat_positions_local = image_input_idx_tensor.view(B, -1)
                        valid_mask_local = flat_positions_local >= 0
                        for b in range(B):
                            valid_pos = flat_positions_local[b][valid_mask_local[b]]
                            if valid_pos.numel() == 0:
                                continue
                            gathered = hs_tensor[b, valid_pos.long(), :]
                            feats.view(B, -1, d_model_local)[b, valid_mask_local[b], :] = gathered
                        return feats

                    # Extract features for all requested non-zero layers
                    for layer_idx in non_zero_layers:
                        if layer_idx > max_layer_index:
                            raise ValueError(
                                f"Requested layer {layer_idx} is out of bounds. "
                                f"Model has {len(hidden_states)} hidden states (layers 0-{max_layer_index})."
                            )
                        layer_features[layer_idx] = gather_visual_features(hidden_states[layer_idx])

                    del hidden_states, output, input_ids, image_input_idx_tensor

                # Generate caption once (shared across layers)
                generated_caption = None
                if generate_captions:
                    torch.cuda.synchronize()
                    if DISTRIBUTED_MODE:
                        dist.barrier()
                    input_ids_gen = torch.tensor(batch["input_tokens"]).unsqueeze(0).to(device)
                    image_input_idx_gen = torch.tensor(batch.get("image_input_idx")).unsqueeze(0).to(device) if batch.get("image_input_idx") is not None else None
                    images_tensor_gen = torch.tensor(batch.get("images")).unsqueeze(0).to(device)
                    image_masks_tensor_gen = torch.tensor(batch.get("image_masks")).unsqueeze(0).to(device) if batch.get("image_masks") is not None else None
                    gen_output = model.generate(
                        input_ids=input_ids_gen,
                        images=images_tensor_gen,
                        image_masks=image_masks_tensor_gen,
                        image_input_idx=image_input_idx_gen,
                        max_steps=200, min_steps=4, is_distributed=False
                    )
                    token_ids = gen_output.token_ids[:, 0].detach().cpu().numpy()
                    generated_caption = preprocessor.tokenizer.decode(token_ids[0])
                    del input_ids_gen, image_input_idx_gen, images_tensor_gen, image_masks_tensor_gen, gen_output
                    clear_gpu_memory()

                # Now compute NN for each layer using its features
                for target_layer in target_layers:
                    image_features = layer_features[target_layer]
                    batch_size, num_chunks, patches_per_chunk, hidden_dim = image_features.shape
                    image_features_reshaped = image_features.view(-1, patches_per_chunk, hidden_dim)

                    image_features_norm = torch.nn.functional.normalize(image_features_reshaped, dim=-1)
                    token_embeddings_norm = torch.nn.functional.normalize(token_embeddings, dim=-1)
                    similarity = torch.matmul(image_features_norm, token_embeddings_norm.T)

                    top_k = 5
                    top_values, top_indices = torch.topk(similarity, k=top_k, dim=-1)

                    del similarity, image_features_norm

                    # Build image results for this layer
                    image_results = {
                        "image_idx": i,
                        "ground_truth_caption": caption_text,
                        "feature_shape": list(image_features.shape),
                        "llm_layer_used": target_layer,
                        "chunks": []
                    }
                    if generated_caption is not None:
                        image_results["generated_response"] = generated_caption

                    stats = per_layer_stats[target_layer]
                    token_position_stats = stats["token_position_stats"]

                    for chunk_idx in range(num_chunks):
                        chunk_results = {
                            "chunk_name": "Full Image" if chunk_idx == 0 else f"Chunk {chunk_idx}",
                            "patches": []
                        }

                        for patch_idx in range(patches_per_chunk):
                            patch_values = top_values[chunk_idx, patch_idx].cpu().numpy().tolist()
                            patch_indices = top_indices[chunk_idx, patch_idx].cpu().numpy().tolist()
                            patch_tokens = [decode_token(preprocessor.tokenizer, idx) for idx in patch_indices]

                            if patch_idx not in token_position_stats:
                                row, col = patch_idx_to_row_col(patch_idx, patches_per_chunk)
                                token_position_stats[patch_idx] = {
                                    "total_occurrences": 0,
                                    "interpretable_occurrences": 0,
                                    "visual_task_occurrences": 0,
                                    "interpretability_percentage": 0.0,
                                    "visual_task_percentage": 0.0,
                                    "patch_row": row,
                                    "patch_col": col
                                }

                            token_position_stats[patch_idx]["total_occurrences"] += 1

                            overall_match_type = 'none'
                            all_matches = []

                            for j, (token_text, similarity_score) in enumerate(zip(patch_tokens, patch_values)):
                                match_type, matches = check_match_type_with_translation(token_text, caption_words, visual_task_words)
                                if match_type != 'none':
                                    if match_type == 'visual_task' or (overall_match_type != 'visual_task' and match_type == 'interpretable'):
                                        overall_match_type = match_type
                                    for match in matches:
                                        all_matches.append({
                                            "token": get_display_token(token_text),
                                            "token_word": match["token_word"],
                                            "matched_word": match["matched_word"],
                                            "match_type": match["match_type"],
                                            "similarity": float(similarity_score)
                                        })

                            stats["total_visual_tokens"] += 1
                            if overall_match_type == 'interpretable':
                                stats["interpretable_visual_tokens"] += 1
                                token_position_stats[patch_idx]["interpretable_occurrences"] += 1
                            elif overall_match_type == 'visual_task':
                                stats["visual_task_visual_tokens"] += 1
                                token_position_stats[patch_idx]["visual_task_occurrences"] += 1

                            row, col = patch_idx_to_row_col(patch_idx, patches_per_chunk)

                            patch_results = {
                                "patch_idx": patch_idx,
                                "patch_row": row,
                                "patch_col": col,
                                "nearest_neighbors": [
                                    {"token": get_display_token(token), "similarity": float(sim)}
                                    for token, sim in zip(patch_tokens, patch_values)
                                ],
                                "match_type": overall_match_type,
                                "caption_match": overall_match_type == 'interpretable',
                                "visual_task_match": overall_match_type == 'visual_task'
                            }
                            if all_matches:
                                patch_results["matches"] = all_matches

                            chunk_results["patches"].append(patch_results)

                        image_results["chunks"].append(chunk_results)

                    per_layer_results[target_layer].append(image_results)

                    del top_values, top_indices

                # Clean up shared tensors
                del images_tensor, image_masks_tensor, layer_features
                clear_gpu_memory()

    # Gather results from all processes and build per-layer output
    combined_per_layer = {}

    for target_layer in target_layers:
        local_results = per_layer_results[target_layer]
        local_stats = per_layer_stats[target_layer]

        all_split_results = [None] * world_size
        all_token_position_stats = [None] * world_size

        if DISTRIBUTED_MODE:
            dist.all_gather_object(all_split_results, local_results)
            dist.all_gather_object(all_token_position_stats, local_stats["token_position_stats"])
        else:
            all_split_results = [local_results]
            all_token_position_stats = [local_stats["token_position_stats"]]

        if local_rank == 0:
            combined_results = []
            for process_results in all_split_results:
                combined_results.extend(process_results)

            combined_token_position_stats = {}
            for process_stats in all_token_position_stats:
                for patch_idx, pstats in process_stats.items():
                    if patch_idx not in combined_token_position_stats:
                        combined_token_position_stats[patch_idx] = {
                            "total_occurrences": 0,
                            "interpretable_occurrences": 0,
                            "visual_task_occurrences": 0,
                            "interpretability_percentage": 0.0,
                            "visual_task_percentage": 0.0,
                            "patch_row": pstats.get("patch_row", 0),
                            "patch_col": pstats.get("patch_col", 0)
                        }
                    combined_token_position_stats[patch_idx]["total_occurrences"] += pstats["total_occurrences"]
                    combined_token_position_stats[patch_idx]["interpretable_occurrences"] += pstats["interpretable_occurrences"]
                    combined_token_position_stats[patch_idx]["visual_task_occurrences"] += pstats["visual_task_occurrences"]

            for pos_stats in combined_token_position_stats.values():
                total = pos_stats["total_occurrences"]
                pos_stats["interpretability_percentage"] = (pos_stats["interpretable_occurrences"] / total * 100) if total > 0 else 0.0
                pos_stats["visual_task_percentage"] = (pos_stats["visual_task_occurrences"] / total * 100) if total > 0 else 0.0

            total_vt = sum(ps["total_occurrences"] for ps in combined_token_position_stats.values())
            interp_vt = sum(ps["interpretable_occurrences"] for ps in combined_token_position_stats.values())
            vtask_vt = sum(ps["visual_task_occurrences"] for ps in combined_token_position_stats.values())

            overall_statistics = {
                "total_visual_tokens": total_vt,
                "interpretable_visual_tokens": interp_vt,
                "visual_task_visual_tokens": vtask_vt,
                "interpretability_percentage": (interp_vt / total_vt * 100) if total_vt > 0 else 0,
                "visual_task_percentage": (vtask_vt / total_vt * 100) if total_vt > 0 else 0,
                "token_position_statistics": combined_token_position_stats
            }
            combined_per_layer[target_layer] = (combined_results, overall_statistics)

    if local_rank == 0:
        return combined_per_layer
    else:
        return None


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
    
    parser = argparse.ArgumentParser(description="Analyze PixMoCap interpretability (Multi-GPU version)")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to checkpoint to analyze")
    parser.add_argument("--force-rerun", action="store_true", help="Force re-running analysis even if results file exists")
    parser.add_argument("--preprocessing_mode", type=str, choices=["padding", "warping"], 
                       help="Preprocessing mode: 'padding' for black padding, 'warping' for direct resize, or omit for config default")
    parser.add_argument("--llm_layer", type=str, default="0",
                       help="LLM layer index(es) to extract features from. Single layer (e.g., '0') or comma-separated list (e.g., '0,8,16'). 0 = vision backbone features")
    parser.add_argument("--generate-captions", action="store_true",
                       help="Generate and save captions for each image (adds processing time)")
    parser.add_argument("--output-base-dir", type=str, default="analysis_results/nearest_neighbors",
                       help="Base directory for saving results (default: analysis_results/nearest_neighbors)")
    parser.add_argument("--num-images", type=int, default=300,
                       help="Number of validation images to process (default: 300)")
    args = parser.parse_args()
    
    # Hardcoded parameters
    checkpoint_path = args.ckpt_path
    
    # Load config to detect dataset type (more reliable than path detection)
    # Check if this is a Pixmo Points or Pixmo Captions checkpoint
    is_pixmo_points = False
    points_kind = "basic"  # Default for points
    points_counting = False  # Default: pointing mode (not counting)
    
    # Load config to check dataset field
    cfg_path = f"{checkpoint_path}/config.yaml"
    temp_cfg = TrainConfig.load(cfg_path)
    
    if hasattr(temp_cfg, 'data') and hasattr(temp_cfg.data, 'dataset'):
        dataset_name = temp_cfg.data.dataset
        dataset_name_lower = dataset_name.lower()
        
        # Use the same mapping as get_dataset_by_name in olmo/data/__init__.py
        # This ensures we use the exact same parameters as during training
        if dataset_name in ["pointing_high_freq", "pixmo_points_high_freq"]:
            is_pixmo_points = True
            points_kind = "high_frequency"
            points_counting = False
        elif dataset_name in ["point_count_high_freq", "pixmo_points_high_freq_counting"]:
            is_pixmo_points = True
            points_kind = "high_frequency"
            points_counting = True
        elif dataset_name in ["pointing", "pixmo_points"]:
            is_pixmo_points = True
            points_kind = "basic"
            points_counting = False
        elif dataset_name in ["point_count", "pixmo_points_counting"]:
            is_pixmo_points = True
            points_kind = "basic"
            points_counting = True
        elif "cap" in dataset_name_lower:
            is_pixmo_points = False
        else:
            # Fallback: if dataset name contains "points" but doesn't match above patterns
            if "points" in dataset_name_lower:
                is_pixmo_points = True
                # Try to infer from dataset name
                if "counting" in dataset_name_lower:
                    points_counting = True
                if "high" in dataset_name_lower:
                    points_kind = "high_frequency"
    else:
        # Fallback to path-based detection if config doesn't have dataset field
        checkpoint_path_lower = checkpoint_path.lower()
        if "pixmo_points" in checkpoint_path_lower or "pixmo-points" in checkpoint_path_lower:
            is_pixmo_points = True
        elif "pixmo_cap" in checkpoint_path_lower or "pixmo-cap" in checkpoint_path_lower:
            is_pixmo_points = False
        else:
            if local_rank == 0:
                print(f"Warning: Could not detect dataset type from config. Defaulting to Pixmo Captions")
    
    # Set prompt and dataset parameters based on detected type
    if is_pixmo_points:
        prompt = "Point to the objects in this image."
        dataset_name = "PixMoPoints"
        output_suffix = "pixmo_points"
        dataset_mode = "points"
    else:
        prompt = "Describe this image in detail."
        dataset_name = "PixMoCap"
        output_suffix = "pixmo_cap"
        dataset_mode = "captions"
    
    num_train_images = 0
    num_val_images = args.num_images
    
    # Parse layers to process
    target_layers = [int(layer.strip()) for layer in args.llm_layer.split(",")]
    
    if local_rank == 0:
        print(f"Auto-detected dataset type from config: {'Pixmo Points' if is_pixmo_points else 'Pixmo Captions'}")
        print(f"Dataset: {dataset_name}")
        if is_pixmo_points:
            print(f"Points kind: {points_kind}, counting: {points_counting}")
        else:
            print(f"Dataset mode: {dataset_mode}")
        print(f"Prompt: {prompt}")
        print(f"Running on {world_size} processes")
        print(f"LLM layers to process: {target_layers}")
        print(f"Generate captions: {args.generate_captions}")
        print(f"Processing layers sequentially to avoid OOM")
        print()

    # Setup base results directory (only on main process)
    if local_rank == 0:
        ckpt_name = checkpoint_path.split("/")[-2] + "_" + checkpoint_path.split("/")[-1]
        # Add preprocessing mode to folder name if specified
        if args.preprocessing_mode:
            ckpt_name += f"_{args.preprocessing_mode}"
        results_dir = Path(args.output_base_dir) / ckpt_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Check which layers need to be processed
        layers_to_process = []
        for target_layer in target_layers:
            layer_suffix = f"_layer{target_layer}"
            output_file = results_dir / f"nearest_neighbors_analysis_{output_suffix}_multi-gpu{layer_suffix}.json"
            if output_file.exists() and not args.force_rerun:
                print(f"Layer {target_layer}: Results already exist, skipping")
            else:
                layers_to_process.append(target_layer)
        
        if not layers_to_process:
            print("All requested layers already processed. Use --force-rerun to reprocess.")
            if DISTRIBUTED_MODE:
                # Signal other processes to exit
                num_layers_tensor = torch.tensor([0], device=device)
                dist.broadcast(num_layers_tensor, src=0)
            return
        else:
            print(f"Will process {len(layers_to_process)} layer(s): {layers_to_process}")
            if DISTRIBUTED_MODE:
                # Broadcast the actual layers to process (not just the count)
                layers_tensor = torch.tensor(layers_to_process + [0] * (len(target_layers) - len(layers_to_process)), device=device)
                num_layers_tensor = torch.tensor([len(layers_to_process)], device=device)
                dist.broadcast(num_layers_tensor, src=0)
                dist.broadcast(layers_tensor, src=0)
    else:
        # This branch only runs for non-zero ranks (distributed mode only)
        assert DISTRIBUTED_MODE, "Non-zero local_rank requires distributed mode"
        results_dir = None
        # Receive number of layers to process
        num_layers_tensor = torch.tensor([0], device=device)
        dist.broadcast(num_layers_tensor, src=0)
        if num_layers_tensor.item() == 0:
            return
        # Receive the actual layers to process
        layers_tensor = torch.tensor([0] * len(target_layers), device=device)
        dist.broadcast(layers_tensor, src=0)
        num_layers = num_layers_tensor.item()
        layers_to_process = layers_tensor[:num_layers].tolist()

    # Wait for main process to set up directories
    if DISTRIBUTED_MODE:
        dist.barrier()

    if local_rank == 0:
        print("\n=== Loading model (once for all layers) ===")

    # Load model with FSDP
    if local_rank == 0:
        print(f"Loading model from {checkpoint_path}")
    
    # Load model on CPU first
    # Works with both full checkpoints and stripped (MLP-only) checkpoints
    cfg = TrainConfig.load(f"{checkpoint_path}/config.yaml")
    cfg.model.init_device = None  # Override init_device to avoid meta tensors
    
    model = Molmo(cfg.model)
    
    # Check checkpoint size to determine if it's a full or stripped checkpoint
    # Full checkpoints (with trained LLM/ViT) are ~29GB and contain all weights
    # Stripped checkpoints (connector-only) are ~200MB and only contain connector weights
    import os
    checkpoint_file = f"{checkpoint_path}/model.pt"
    checkpoint_size_gb = os.path.getsize(checkpoint_file) / (1024**3)
    
    # If checkpoint is > 1GB, it's a full checkpoint with all weights
    is_full_checkpoint = checkpoint_size_gb > 1.0
    
    if not is_full_checkpoint:
        # Small checkpoint - only contains connector weights, need to load pretrained LLM/ViT
        if local_rank == 0:
            print(f"Detected stripped checkpoint ({checkpoint_size_gb:.2f} GB)")
            print("Loading pretrained weights (LLM + ViT)...")
        model.reset_with_pretrained_weights()
    else:
        if local_rank == 0:
            print(f"Detected full checkpoint ({checkpoint_size_gb:.2f} GB)")
            print("Skipping pretrained weights loading (checkpoint contains all weights)")
    
    # Load checkpoint weights
    if local_rank == 0:
        print("Loading checkpoint weights...")
    checkpoint_weights = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(checkpoint_weights, strict=False)
    
    # Free checkpoint memory immediately
    num_params = len(checkpoint_weights)
    del checkpoint_weights
    import gc
    gc.collect()
    
    if local_rank == 0:
        print(f"Loaded {num_params} parameter tensors from checkpoint")
    
    model.eval()
    
    # Wrap model for inference
    if DISTRIBUTED_MODE:
        # Synchronize all ranks before FSDP wrapping to avoid race conditions
        dist.barrier()

        # Wrap model with FSDP for sharding
        if local_rank == 0:
            print("Wrapping model with FSDP for sharding...")

        # Get FSDP wrap policy from the model
        wrap_policy = model.get_fsdp_wrap_policy("by_block_and_size")

        # Wrap model in FSDP
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

        print(f"Model wrapped with FSDP on device: {device}")
    else:
        # Single-GPU mode: move model to device directly
        print(f"Moving model to {device} (single-GPU mode)...")
        model = model.to(device).to(torch.float16)
        print(f"Model on device: {device}")

    # Create preprocessor
    if "hf:" in checkpoint_path:
        model_config = model.config
    else:
        model_config = ModelConfig.load(resource_path(checkpoint_path, "config.yaml"), key="model", validate_paths=False)
    # Override system prompt kind to avoid length conditioning
    model_config.system_prompt_kind = "none"
    preprocessor = build_mm_preprocessor(
        model_config,
        for_inference=True,
        shuffle_messages=False,
        is_training=False,
        require_image_features=True
    )

    # Override resize behavior based on preprocessing mode
    if args.preprocessing_mode == "padding":
        # Store original resize method
        original_resize = preprocessor.mm_preprocessor.resize
        
        # Override the resize_image method to force black padding
        def resize_image_with_black_padding(image, output_size, is_training, rng):
            # Force use of resize_and_pad with black padding regardless of resize mode
            return resize_and_pad(
                image, output_size, pad_value=0, rng=rng, is_training=is_training,
                resize_method="torch-bilinear")
        
        preprocessor.mm_preprocessor.resize_image = resize_image_with_black_padding
    elif args.preprocessing_mode == "warping":
        # Store original resize method
        original_resize = preprocessor.mm_preprocessor.resize
        
        # Override the resize_image method to force direct resize (no padding)
        def resize_image_with_warping(image, output_size, is_training, rng):
            # Force direct resize without padding (stretches image to fill target size)
            # Use SigLIP method for consistent warping behavior
            return siglip_resize_and_pad(image, output_size)
        
        preprocessor.mm_preprocessor.resize_image = resize_image_with_warping

    use_n_token_only = model_config.vision_backbone.use_n_token_only

    # Load token embeddings once for all layers
    if local_rank == 0:
        print("Loading cached token embeddings...")
    model_identifier = model.config.tokenizer.identifier
    if "qwen" in model_identifier.lower():
        cached_embeddings_path = "analysis_results/cached_text_embeddings/Qwen_Qwen2-7B/layer_0_static_vocab.npy"
    elif "dolma" in model_identifier.lower():
        cached_embeddings_path = "analysis_results/cached_text_embeddings/allenai_OLMo-7B-1024-preview/layer_0_static_vocab.npy"
    elif "llama" in model_identifier.lower():
        cached_embeddings_path = "analysis_results/cached_text_embeddings/meta-llama_Meta-Llama-3-8B/layer_0_static_vocab.npy"
    
    # Try loading cached embeddings, falling back to model's own embeddings
    token_embeddings = None
    # Try both relative to CWD and relative to script location
    search_paths = [
        Path(cached_embeddings_path),
        Path(__file__).parent.parent.parent / cached_embeddings_path,
    ]
    for p in search_paths:
        if p.exists():
            token_embeddings = torch.from_numpy(np.load(str(p))).to(device)
            if local_rank == 0:
                print(f"Loaded cached token embeddings from {p}, shape: {token_embeddings.shape}")
            break

    if token_embeddings is None:
        if local_rank == 0:
            print(f"Cached embeddings not found, extracting from model's embedding layer...")
        # model.transformer.wte.embedding is a Parameter with shape [vocab_size, d_model]
        token_embeddings = model.transformer.wte.embedding.detach().clone().to(device)
        if local_rank == 0:
            print(f"Loaded token embeddings from model, shape: {token_embeddings.shape}")

    # Process all layers efficiently (one forward pass per image)
    if local_rank == 0:
        print(f"\n{'='*60}")
        print(f"Processing {len(layers_to_process)} layers in single pass over images")
        print(f"Layers: {layers_to_process}")
        print(f"{'='*60}")

    # Train split
    train_per_layer = None
    if num_train_images > 0:
        if local_rank == 0:
            print("Processing train split...")
        if is_pixmo_points:
            train_dataset = PixMoPoints(split="train", kind=points_kind, counting=points_counting)
        else:
            train_dataset = PixMoCap(split="train", mode=dataset_mode)
        train_per_layer = process_split_all_layers(
            model, preprocessor, train_dataset, num_train_images, prompt,
            use_n_token_only, token_embeddings, "train", layers_to_process, args.generate_captions
        )
        del train_dataset
        clear_gpu_memory()
        if DISTRIBUTED_MODE:
            dist.barrier()

    # Validation split
    if local_rank == 0:
        print("Processing validation split...")
    if is_pixmo_points:
        val_dataset = PixMoPoints(split="validation", kind=points_kind, counting=points_counting)
    else:
        val_dataset = PixMoCap(split="validation", mode=dataset_mode)
    val_per_layer = process_split_all_layers(
        model, preprocessor, val_dataset, num_val_images, prompt,
        use_n_token_only, token_embeddings, "validation", layers_to_process, args.generate_captions
    )
    del val_dataset
    clear_gpu_memory()
    if DISTRIBUTED_MODE:
        dist.barrier()

    # Save per-layer results (only on main process)
    if local_rank == 0:
        empty_statistics = {
            "total_visual_tokens": 0,
            "interpretable_visual_tokens": 0,
            "visual_task_visual_tokens": 0,
            "interpretability_percentage": 0,
            "visual_task_percentage": 0,
            "token_position_statistics": {}
        }

        for target_layer in layers_to_process:
            # Get train results for this layer
            if train_per_layer and target_layer in train_per_layer:
                train_images, train_statistics = train_per_layer[target_layer]
            else:
                train_images = []
                train_statistics = empty_statistics

            # Get val results for this layer
            val_images, val_statistics = val_per_layer[target_layer]

            # Combine statistics
            combined_total_tokens = train_statistics["total_visual_tokens"] + val_statistics["total_visual_tokens"]
            combined_interpretable_tokens = train_statistics["interpretable_visual_tokens"] + val_statistics["interpretable_visual_tokens"]
            combined_visual_task_tokens = train_statistics["visual_task_visual_tokens"] + val_statistics["visual_task_visual_tokens"]
            combined_token_position_stats = {}
            for split_stats in [train_statistics, val_statistics]:
                for pos, stats in split_stats["token_position_statistics"].items():
                    if pos not in combined_token_position_stats:
                        combined_token_position_stats[pos] = {
                            "total_occurrences": 0,
                            "interpretable_occurrences": 0,
                            "visual_task_occurrences": 0,
                            "interpretability_percentage": 0.0,
                            "visual_task_percentage": 0.0,
                            "patch_row": stats.get("patch_row", 0),
                            "patch_col": stats.get("patch_col", 0)
                        }
                    combined_token_position_stats[pos]["total_occurrences"] += stats["total_occurrences"]
                    combined_token_position_stats[pos]["interpretable_occurrences"] += stats["interpretable_occurrences"]
                    combined_token_position_stats[pos]["visual_task_occurrences"] += stats["visual_task_occurrences"]
            for pos_stats in combined_token_position_stats.values():
                total = pos_stats["total_occurrences"]
                pos_stats["interpretability_percentage"] = (pos_stats["interpretable_occurrences"] / total * 100) if total > 0 else 0.0
                pos_stats["visual_task_percentage"] = (pos_stats["visual_task_occurrences"] / total * 100) if total > 0 else 0.0

            # Create results dictionary (same format as before)
            all_results = {
                "checkpoint": checkpoint_path,
                "prompt": prompt,
                "dataset": dataset_name,
                "num_processes": world_size,
                "preprocessing_mode": args.preprocessing_mode,
                "llm_layer": target_layer,
                "splits": {
                    "train": {
                        "num_images": num_train_images,
                        "images": train_images,
                        "statistics": train_statistics
                    },
                    "validation": {
                        "num_images": num_val_images,
                        "images": val_images,
                        "statistics": val_statistics
                    }
                },
                "overall_statistics": {
                    "total_visual_tokens": combined_total_tokens,
                    "interpretable_visual_tokens": combined_interpretable_tokens,
                    "visual_task_visual_tokens": combined_visual_task_tokens,
                    "interpretability_percentage": (combined_interpretable_tokens / combined_total_tokens * 100) if combined_total_tokens > 0 else 0,
                    "visual_task_percentage": (combined_visual_task_tokens / combined_total_tokens * 100) if combined_total_tokens > 0 else 0,
                    "train_statistics": train_statistics,
                    "validation_statistics": val_statistics,
                    "token_position_statistics": combined_token_position_stats
                }
            }

            # Save outputs
            layer_suffix = f"_layer{target_layer}"
            per_layer_output_file = results_dir / f"nearest_neighbors_analysis_{output_suffix}_multi-gpu{layer_suffix}.json"
            with open(per_layer_output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"✓ Layer {target_layer}: Results saved to {per_layer_output_file}")
            plot_file = results_dir / f"nearest_neighbors_analysis_{output_suffix}_multi-gpu{layer_suffix}_summary_plot.png"
            create_interpretability_plot(all_results["overall_statistics"], plot_file)
            token_plot_file = results_dir / f"token_position_interpretability_plot_multi-gpu{layer_suffix}.png"
            create_token_position_plot(all_results["overall_statistics"]["token_position_statistics"], token_plot_file)

        save_translation_cache()
        print("="*60)
        print(f"✓ All {len(layers_to_process)} layer(s) processed successfully!")
        print("="*60)

    # Wait for all processes to finish
    if DISTRIBUTED_MODE:
        dist.barrier()

if __name__ == "__main__":
    main() 