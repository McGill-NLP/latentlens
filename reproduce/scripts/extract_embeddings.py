# Note: This script extracts contextual text embeddings from standard HuggingFace
# causal LMs (OLMo, LLaMA, Qwen2, etc.). For Qwen2-VL, which uses a different
# architecture with native vision processing, see the Qwen2-VL documentation
# for embedding extraction.

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import random
from collections import defaultdict

# Import ml_dtypes early so numpy can deserialize float8 arrays
try:
    import ml_dtypes
except ImportError:
    print("Warning: ml_dtypes not available. Float8 embeddings will not load correctly.")
    ml_dtypes = None

# Constants
NUM_CAPTIONS = 1000000  # Number of LAION captions to process
BATCH_SIZE =64  # Batch size for processing
SAVE_FREQUENCY = 1000  # Save progress every N captions
MAX_CAPTIONS_PER_TOKEN = 20  # Maximum number of embeddings to store per token (reservoir sampling)
EMBEDDING_DTYPE = 'float8'  # Data type for saving embeddings: 'float32', 'float16', or 'float8' (requires ml_dtypes)
# Layers to extract - matching general_and_nearest_neighbors_pixmo_cap_multi-gpu.py
# Default: first few (1,2,3,4) then every 4th (8,12,...) up to last block
# For 32-layer model: [1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 31]
LAYERS_TO_EXTRACT = None  # Auto-computed from model.config.num_hidden_layers

# Dry run mode - only store metadata, not actual embeddings
DRY_RUN_MODE = False  # Set to True to only store token metadata

def convert_from_stored_dtype(embedding_array):
    """Convert embedding from stored dtype back to float32 for computation."""
    dtype_str = str(embedding_array.dtype)
    
    # Check if it's a void type (raw bytes from float8)
    if dtype_str.startswith('|V') or dtype_str.startswith('V'):
        # This is float8 saved as raw bytes, need to reinterpret
        if ml_dtypes is not None:
            # Reinterpret the raw bytes as float8, then convert to float32
            embedding_fp8 = embedding_array.view(ml_dtypes.float8_e4m3fn)
            return embedding_fp8.astype(np.float32)
        else:
            raise ImportError("ml_dtypes required to load float8 embeddings")
    elif 'float8' in dtype_str:
        if ml_dtypes is not None:
            return embedding_array.astype(np.float32)
        else:
            return embedding_array.astype(np.float32)
    elif embedding_array.dtype == np.float16:
        return embedding_array.astype(np.float32)
    elif embedding_array.dtype == np.float32:
        return embedding_array
    else:
        # Try generic conversion
        return embedding_array.astype(np.float32)

def convert_embedding_dtype(embedding, target_dtype='float16'):
    """
    Convert embedding to target dtype for storage optimization.
    
    Args:
        embedding: numpy array
        target_dtype: 'float32', 'float16', or 'float8'
    
    Returns:
        numpy array in target dtype
    """
    if target_dtype == 'float32':
        return embedding.astype(np.float32)
    elif target_dtype == 'float16':
        return embedding.astype(np.float16)
    elif target_dtype == 'float8':
        # True fp8 requires ml_dtypes library
        if ml_dtypes is not None:
            # float8_e4m3fn has limited range: ~[-448, 448]
            # Clip outliers to prevent NaN (typically affects <1% of values)
            FLOAT8_MAX = 448.0
            embedding_clipped = np.clip(embedding, -FLOAT8_MAX, FLOAT8_MAX)
            
            # Check if clipping was needed
            num_clipped = np.sum(np.abs(embedding) > FLOAT8_MAX)
            if num_clipped > 0:
                # Only print warning occasionally (not for every embedding)
                import random
                if random.random() < 0.01:  # 1% of the time
                    print(f"  Note: Clipped {num_clipped}/{embedding.size} values to float8 range (normal for outliers)")
            
            # Use e4m3fn (4-bit exponent, 3-bit mantissa) which is good for ML
            return embedding_clipped.astype(ml_dtypes.float8_e4m3fn)
        else:
            print("Warning: ml_dtypes not available for fp8, falling back to float16")
            return embedding.astype(np.float16)
    else:
        raise ValueError(f"Unsupported dtype: {target_dtype}")

def load_progress(output_dir):
    """Load existing progress from checkpoint."""
    progress_file = output_dir / "progress.json"
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return None

def save_progress(output_dir, layer_dirs, captions_processed, total_captions, token_seen_counts=None):
    """Save current progress to checkpoint."""
    # Save progress metadata
    progress = {
        'captions_processed': captions_processed,
        'total_captions': total_captions,
        'layer_counters': {str(layer_idx): info['counter'] for layer_idx, info in layer_dirs.items()}
    }
    
    # Save random state for reproducible reservoir sampling across resumes
    progress['random_state'] = list(random.getstate()[1])
    progress['random_gauss_next'] = random.getstate()[2]

    # Save token_seen_counts for reservoir sampling if provided
    if token_seen_counts is not None:
        progress['token_seen_counts'] = {
            str(layer_idx): {token: counts for token, counts in layer_counts.items()}
            for layer_idx, layer_counts in token_seen_counts.items()
        }
    
    progress_file = output_dir / "progress.json"
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    
    # Save each layer's token embeddings
    for layer_idx, layer_info in layer_dirs.items():
        layer_dir = layer_info['layer_dir']
        token_dict = layer_info['token_dict']
        output_file = layer_dir / "token_embeddings.json"
        
        with open(output_file, 'w') as f:
            json.dump(token_dict, f, indent=2)
    
    print(f"  Progress saved: {captions_processed}/{total_captions} captions processed")

def load_existing_embeddings(layer_dirs, output_dir, layers_to_extract):
    """Load existing token embeddings if resuming."""
    progress = load_progress(output_dir)
    if progress is None:
        return 0, None
    
    print(f"Resuming from previous run...")
    print(f"  Previously processed: {progress['captions_processed']} captions")
    
    # Load existing token embeddings for each layer
    for layer_idx, layer_info in layer_dirs.items():
        layer_dir = layer_info['layer_dir']
        token_file = layer_dir / "token_embeddings.json"
        
        if token_file.exists():
            with open(token_file, 'r') as f:
                saved_token_dict = json.load(f)
            
            # Restore the dict structure expected during extraction
            token_dict = {}
            for token_str, data in saved_token_dict.items():
                # Check if data is already in dict format (with 'preferred'/'fallback' keys)
                if isinstance(data, dict) and 'preferred' in data:
                    # Already in the correct format
                    token_dict[token_str] = data
                elif isinstance(data, list):
                    # Convert from list format to dict format
                    # Split embeddings back into preferred and fallback based on position
                    preferred = [emb for emb in data if isinstance(emb, dict) and emb.get('position', 0) >= 10]
                    fallback = [emb for emb in data if isinstance(emb, dict) and 2 <= emb.get('position', 0) < 10]
                    token_dict[token_str] = {
                        'preferred': preferred,
                        'fallback': fallback,
                        'combined': []
                    }
                else:
                    # Unknown format, skip
                    print(f"  Warning: Skipping token '{token_str}' - unexpected format: {type(data)}")
                    continue
            
            layer_info['token_dict'] = token_dict
            print(f"  Loaded layer {layer_idx}: {len(token_dict)} unique tokens")
        
        # Restore counter
        if str(layer_idx) in progress['layer_counters']:
            layer_info['counter'] = progress['layer_counters'][str(layer_idx)]
    
    # Restore token_seen_counts for reservoir sampling
    token_seen_counts = None
    if 'token_seen_counts' in progress:
        token_seen_counts = {
            layer_idx: defaultdict(lambda: {'preferred_count': 0, 'fallback_count': 0})
            for layer_idx in layers_to_extract
        }
        for layer_idx_str, layer_counts in progress['token_seen_counts'].items():
            layer_idx = int(layer_idx_str)
            if layer_idx in token_seen_counts:
                for token, counts in layer_counts.items():
                    token_seen_counts[layer_idx][token] = counts
        print(f"  Restored reservoir sampling counts")

    # Restore random state for reproducible reservoir sampling across resumes
    if 'random_state' in progress:
        state_tuple = (3, tuple(progress['random_state']), progress.get('random_gauss_next'))
        random.setstate(state_tuple)
        print(f"  Restored random state for reproducible reservoir sampling")

    return progress['captions_processed'], token_seen_counts

def load_vg_phrases(vg_file_path, num_phrases=NUM_CAPTIONS):
    """Load phrases from Visual Genome phrases file (included in repo)."""
    if not Path(vg_file_path).exists():
        raise FileNotFoundError(
            f"Visual Genome phrases file not found: {vg_file_path}\n"
            f"This file should be at reproduce/vg_phrases.txt"
        )

    if num_phrases is None:
        print(f"Loading ALL phrases from {vg_file_path}...")
    else:
        print(f"Loading {num_phrases} phrases from {vg_file_path}...")

    phrases = []
    with open(vg_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if num_phrases is not None and i >= num_phrases:
                break

            phrase = line.strip()
            if phrase:  # Only add non-empty phrases
                phrases.append(phrase)

            if (i + 1) % 100000 == 0:
                print(f"Loaded {i + 1} phrases...")

    print(f"Loaded {len(phrases)} phrases from VG phrases file")
    return phrases

def load_tsv_captions(tsv_file_path, num_captions=NUM_CAPTIONS):
    """DEPRECATED: CC captions are no longer used. Use VG phrases instead."""
    raise NotImplementedError("Conceptual Captions support removed. Use Visual Genome phrases (the default).")
    
    print(f"Loaded {len(captions)} captions from LAION")
    return captions

def analyze_tokenizer_output(tokenizer, captions, batch_size, output_dir, start_offset=0):
    """
    Analyze tokenizer output without running the LLM model.
    This is much faster and gives insights into token diversity and caption patterns.
    """
    
    print(f"\n🔍 TOKENIZER ANALYSIS MODE: No LLM inference, just tokenization analysis")
    
    # Skip already processed captions
    if start_offset > 0:
        captions = captions[start_offset:]
        print(f"Skipping first {start_offset} captions (already processed)")
    
    total_batches = (len(captions) + batch_size - 1) // batch_size
    print(f"Processing {len(captions)} captions in {total_batches} batches of size {batch_size}")
    
    # Statistics tracking
    token_stats = {}  # token -> count
    caption_lengths = []
    token_position_stats = {}  # position -> token_counts
    vocabulary = set()
    
    # Process captions in batches
    for batch_idx in tqdm(range(0, len(captions), batch_size), desc="Analyzing captions"):
        batch_captions = captions[batch_idx:batch_idx + batch_size]
        
        # Tokenize batch
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": 512  # Reasonable limit for captions
        }
        
        try:
            tokenized = tokenizer(batch_captions, **tokenizer_kwargs)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
        except Exception as e:
            print(f"Tokenization error in batch {batch_idx}: {e}")
            continue
        
        # Process each sentence in the batch
        for sent_idx, caption in enumerate(batch_captions):
            # Get valid token positions (exclude padding)
            valid_positions = (attention_mask[sent_idx] == 1).nonzero(as_tuple=True)[0]
            token_ids = input_ids[sent_idx][valid_positions]
            
            # Track caption length
            caption_lengths.append(len(token_ids))
            
            # For each valid token position
            for pos_idx, token_pos in enumerate(valid_positions):
                token_id = token_ids[pos_idx].item()
                token_str = tokenizer.decode([token_id])
                
                # Update vocabulary
                vocabulary.add(token_str)
                
                # Update token frequency
                if token_str not in token_stats:
                    token_stats[token_str] = 0
                token_stats[token_str] += 1
                
                # Update position statistics
                if pos_idx not in token_position_stats:
                    token_position_stats[pos_idx] = {}
                if token_str not in token_position_stats[pos_idx]:
                    token_position_stats[pos_idx][token_str] = 0
                token_position_stats[pos_idx][token_str] += 1
    
    # Calculate statistics
    total_tokens = sum(token_stats.values())
    unique_tokens = len(token_stats)
    avg_caption_length = np.mean(caption_lengths)
    median_caption_length = np.median(caption_lengths)
    
    print(f"\n📊 TOKENIZER ANALYSIS RESULTS:")
    print(f"  Total captions processed: {len(captions)}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Unique tokens: {unique_tokens:,}")
    print(f"  Average caption length: {avg_caption_length:.1f} tokens")
    print(f"  Median caption length: {median_caption_length:.1f} tokens")
    print(f"  Vocabulary size: {len(vocabulary):,}")
    
    # Save results
    results = {
        'total_captions': len(captions),
        'total_tokens': total_tokens,
        'unique_tokens': unique_tokens,
        'vocabulary_size': len(vocabulary),
        'caption_length_stats': {
            'mean': float(avg_caption_length),
            'median': float(median_caption_length),
            'min': int(np.min(caption_lengths)),
            'max': int(np.max(caption_lengths)),
            'std': float(np.std(caption_lengths))
        },
        'top_tokens': sorted(token_stats.items(), key=lambda x: x[1], reverse=True)[:100],
        'token_position_stats': {str(k): v for k, v in token_position_stats.items()},
        'data_source': 'TSV image captions (tokenizer analysis only)',
        'analysis_type': 'tokenizer_only_dry_run'
    }
    
    # Save to file
    output_file = output_dir / "tokenizer_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    return total_tokens, len(captions)

def extract_contextual_embeddings(model, tokenizer, captions, layers_to_extract, batch_size, layer_dirs, device, output_dir, start_offset=0, max_captions_per_token=MAX_CAPTIONS_PER_TOKEN, token_seen_counts_init=None, embedding_dtype=EMBEDDING_DTYPE, apply_position_filter=True):
    """
    Extract contextual embeddings for all tokens across captions.
    
    Position filtering (from map_token2caption.py):
    - If apply_position_filter=True: Only considers tokens at position >= 2 (skips positions 0 and 1)
    - If apply_position_filter=True: Prefers tokens at position >= 10, but falls back to position >= 2 if needed
    - If apply_position_filter=False: Includes all token positions (0, 1, 2, ...)
    - Uses reservoir sampling to limit storage to max_captions_per_token per token
    
    Storage optimization:
    - Embeddings are converted to specified dtype (float16, float8, etc.) to reduce storage
    
    Matching the layer extraction logic from general_and_nearest_neighbors_pixmo_cap_multi-gpu.py:
    - Uses output_hidden_states=True to get all hidden states
    - hidden_states[0] = input embeddings (before first block)
    - hidden_states[i] = output of block i-1 (for i >= 1)
    - hidden_states[n_layers] = output after final block and layer norm
    """
    
    total_embedding_count = 0
    captions_processed = start_offset
    total_tokens_seen = 0  # Total tokens across all positions
    filtered_tokens_processed = 0  # Tokens at position >= 2
    
    # Skip already processed captions
    if start_offset > 0:
        captions = captions[start_offset:]
        print(f"\nSkipping first {start_offset} captions (already processed)")
    
    total_batches = (len(captions) + batch_size - 1) // batch_size
    print(f"\nProcessing {len(captions)} captions in {total_batches} batches of size {batch_size}")
    if apply_position_filter:
        print(f"Position filtering: skipping positions 0-1, preferring position >= 10")
    else:
        print(f"Position filtering: DISABLED - including all token positions")
    print(f"Reservoir sampling: max {max_captions_per_token} embeddings per token per layer")
    print(f"Storage dtype: {embedding_dtype} (saves {2 if embedding_dtype == 'float16' else 4 if embedding_dtype == 'float8' else 1}x less storage than float32)")
    
    # Track how many times each token has been seen for reservoir sampling
    # Structure: {layer_idx: {token_str: {'preferred_count': int, 'fallback_count': int}}}
    if token_seen_counts_init is not None:
        token_seen_counts = token_seen_counts_init
        print(f"Resuming with existing reservoir sampling counts")
    else:
        token_seen_counts = {layer_idx: defaultdict(lambda: {'preferred_count': 0, 'fallback_count': 0}) 
                             for layer_idx in layers_to_extract}
    
    # Process captions in batches
    for batch_idx in tqdm(range(0, len(captions), batch_size), desc="Processing batches"):
        batch_captions = captions[batch_idx:batch_idx + batch_size]
        
        # Tokenize batch
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "return_token_type_ids": False,
            "truncation": True,
            "max_length": 512,
            "padding": True if batch_size > 1 else False
        }
        
        encodings = tokenizer(batch_captions, **tokenizer_kwargs)
        
        # Move encodings to GPU
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        # Get input_ids for decoding tokens (after moving to GPU)
        input_ids = encodings['input_ids']
        
        # Get model outputs with all hidden states
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                last_logits_only=False,
            )
        
        # Get all hidden states
        # hidden_states[0] = input embeddings
        # hidden_states[i] = output of block i-1 (for i >= 1)
        all_hidden_states = outputs.hidden_states
        
        # Process each caption in the batch
        for sent_idx, caption in enumerate(batch_captions):
            # Get non-padding token positions
            attention_mask = encodings['attention_mask'][sent_idx]
            valid_positions = torch.where(attention_mask == 1)[0].cpu().numpy()
            
            # Get token IDs for this caption
            token_ids = input_ids[sent_idx][valid_positions]
            
            # Track tokens seen in this caption
            total_tokens_seen += len(token_ids)
            
            # For each valid token position
            for pos_idx, token_pos in enumerate(valid_positions):
                # POSITION FILTERING: Skip positions 0 and 1 (if enabled)
                if apply_position_filter and pos_idx < 2:
                    continue
                
                filtered_tokens_processed += 1
                token_id = token_ids[pos_idx].item()
                token_str = tokenizer.decode([token_id])
                
                # Determine if this is preferred (pos >= 10) or fallback (2 <= pos < 10)
                # If position filtering is disabled, treat all positions as preferred
                if apply_position_filter:
                    is_preferred = pos_idx >= 10
                else:
                    is_preferred = True  # All positions are treated equally when filter is disabled
                
                # CRITICAL FIX: Make reservoir sampling decision ONCE for ALL layers
                # Use the first layer to track counts and make decisions
                first_layer = layers_to_extract[0]
                first_layer_info = layer_dirs[first_layer]
                first_token_dict = first_layer_info['token_dict']
                
                # Initialize token_dict entry for first layer
                if token_str not in first_token_dict:
                    first_token_dict[token_str] = {
                        'preferred': [],
                        'fallback': [],
                        'combined': []
                    }
                
                # Get counts from first layer (shared decision making)
                counts = token_seen_counts[first_layer][token_str]
                
                # Decide which reservoir and get count
                if is_preferred:
                    counts['preferred_count'] += 1
                    reservoir_type = 'preferred'
                    count = counts['preferred_count']
                else:
                    counts['fallback_count'] += 1
                    reservoir_type = 'fallback'
                    count = counts['fallback_count']
                
                # Make the sampling decision ONCE (applies to all layers)
                first_reservoir = first_token_dict[token_str][reservoir_type]
                should_store = False
                replace_idx = None
                
                if len(first_reservoir) < max_captions_per_token:
                    # Reservoir not full, always add
                    should_store = True
                else:
                    # Reservoir sampling: random chance to replace existing item
                    j = random.randint(0, count - 1)
                    if j < max_captions_per_token:
                        should_store = True
                        replace_idx = j
                
                # If we decided to store, do it for ALL layers
                if should_store:
                    for layer_idx in layers_to_extract:
                        # Get the hidden state for this token from the specified layer
                        hidden_state = all_hidden_states[layer_idx][sent_idx, token_pos].cpu().numpy()
                        
                        # CRITICAL: Check for NaN IMMEDIATELY - FAIL LOUDLY if detected
                        if np.isnan(hidden_state).any():
                            error_msg = (
                                f"\n{'='*80}\n"
                                f"CRITICAL ERROR: NaN detected in hidden state!\n"
                                f"{'='*80}\n"
                                f"Layer: {layer_idx}\n"
                                f"Caption: {caption}\n"
                                f"Token: '{token_str}' (ID: {token_id})\n"
                                f"Position: {pos_idx}\n"
                                f"NaN count: {np.isnan(hidden_state).sum()}/{hidden_state.size}\n"
                                f"NaN indices: {np.where(np.isnan(hidden_state))[0][:20].tolist()}\n"
                                f"{'='*80}\n"
                                f"This indicates a numerical instability in the model forward pass.\n"
                                f"Possible causes:\n"
                                f"  - Model weights have NaN/Inf\n"
                                f"  - Numerical overflow in float32 (try mixed precision)\n"
                                f"  - Specific caption triggers instability\n"
                                f"{'='*80}\n"
                            )
                            raise ValueError(error_msg)
                        
                        # Get the directory info for this layer
                        layer_info = layer_dirs[layer_idx]
                        token_dict = layer_info['token_dict']
                        
                        # Initialize token_dict entry for this layer
                        if token_str not in token_dict:
                            token_dict[token_str] = {
                                'preferred': [],
                                'fallback': [],
                                'combined': []
                            }
                        
                        reservoir = token_dict[token_str][reservoir_type]
                        counter = layer_info['counter']
                        
                        if not DRY_RUN_MODE:
                            # Normal mode: save embedding to disk
                            embeddings_dir = layer_info['embeddings_dir']
                            embedding_path = embeddings_dir / f"emb_{counter:08d}.npy"
                            
                            # Convert to target dtype for storage optimization
                            hidden_state_converted = convert_embedding_dtype(hidden_state, embedding_dtype)
                            
                            # Double-check: verify no NaN after dtype conversion
                            # (convert back to check since float8 can't be checked directly)
                            if embedding_dtype in ['float8', 'float16']:
                                check_array = convert_from_stored_dtype(hidden_state_converted)
                                if np.isnan(check_array).any():
                                    raise ValueError(
                                        f"NaN introduced during dtype conversion to {embedding_dtype}!\n"
                                        f"Layer {layer_idx}, token '{token_str}', caption: {caption}"
                                    )
                            
                            np.save(embedding_path, hidden_state_converted)
                            
                            # Create entry
                            entry = {
                                'embedding_path': str(embedding_path.relative_to(layer_info['layer_dir'])),
                                'caption': caption,
                                'position': pos_idx,
                                'token_id': token_id,
                                'dtype': str(hidden_state_converted.dtype)
                            }
                        else:
                            # Dry run mode: only store metadata
                            entry = {
                                'caption': caption,
                                'position': pos_idx,
                                'token_id': token_id,
                                'embedding_shape': hidden_state.shape,
                                'embedding_dtype': str(hidden_state.dtype)
                            }
                        
                        # Add or replace in reservoir
                        # CRITICAL: Check if replace_idx is valid for THIS layer's reservoir
                        # (in case this layer was deleted/corrupted and is being rebuilt)
                        if replace_idx is not None and replace_idx < len(reservoir):
                            reservoir[replace_idx] = entry
                        else:
                            # Either no replacement, or reservoir not full yet on this layer
                            reservoir.append(entry)
                        
                        # Increment counter
                        layer_info['counter'] += 1
                        total_embedding_count += 1
        
        # Update captions processed counter
        captions_processed += len(batch_captions)
        
        # Save progress periodically
        if captions_processed % SAVE_FREQUENCY < batch_size or captions_processed >= start_offset + len(captions):
            total_captions = start_offset + len(captions)
            print(f"\nSaving progress at {captions_processed}/{total_captions} captions...")
            save_progress(output_dir, layer_dirs, captions_processed, total_captions, token_seen_counts)
    
    print(f"\n📊 Extraction Statistics:")
    print(f"  Total tokens seen: {total_tokens_seen:,}")
    print(f"  Tokens filtered (position >= 2): {filtered_tokens_processed:,}")
    print(f"  Embeddings stored (after reservoir sampling): {total_embedding_count:,}")
    print(f"\nExtracted {total_embedding_count} contextual embeddings across {len(layers_to_extract)} layers")
    for layer_idx in layers_to_extract:
        num_unique = len(layer_dirs[layer_idx]['token_dict'])
        num_embeddings = layer_dirs[layer_idx]['counter']
        print(f"  Layer {layer_idx}: {num_embeddings} embeddings for {num_unique} unique tokens")
    
    return total_embedding_count, captions_processed

def extract_from_reference_layer(model, tokenizer, reference_layer_idx, target_layers, output_dir, device, embedding_dtype=EMBEDDING_DTYPE, use_data_parallel=False, save_frequency=5000):
    """
    Extract embeddings from target layers using the exact same tokens/positions as a reference layer.
    This ensures perfect alignment of embeddings across layers.
    
    Args:
        save_frequency: Save JSON files every N captions processed (default: 5000)
    """
    
    print(f"\n{'='*80}")
    print(f"REFERENCE LAYER MODE: Extracting from layers {target_layers}")
    print(f"Using reference layer {reference_layer_idx} for token/position selection")
    print(f"JSON save frequency: every {save_frequency} captions")
    print(f"{'='*80}\n")
    
    # Load reference layer's token embeddings
    reference_layer_dir = output_dir / f"layer_{reference_layer_idx}"
    reference_file = reference_layer_dir / "token_embeddings.json"
    
    if not reference_file.exists():
        raise FileNotFoundError(f"Reference layer file not found: {reference_file}")
    
    print(f"Loading reference embeddings from {reference_file}...")
    with open(reference_file, 'r') as f:
        reference_token_dict = json.load(f)
    
    # Build list of unique (caption, position, token_id, token_str) tuples to process
    caption_positions = {}  # caption -> list of (position, token_id, token_str)
    total_embeddings = 0
    
    for token_str, embeddings_list in reference_token_dict.items():
        # Handle both list and dict formats
        if isinstance(embeddings_list, dict):
            embeddings_list = embeddings_list.get('preferred', []) + embeddings_list.get('fallback', [])
        
        for emb in embeddings_list:
            if not isinstance(emb, dict):
                continue
            caption = emb['caption']
            position = emb['position']
            token_id = emb['token_id']
            
            if caption not in caption_positions:
                caption_positions[caption] = []
            caption_positions[caption].append((position, token_id, token_str))
            total_embeddings += 1
    
    print(f"Found {len(caption_positions)} unique captions with {total_embeddings} total token positions")
    
    # Create layer directories for target layers
    layer_dirs = {}
    for layer_idx in target_layers:
        layer_dir = output_dir / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        embeddings_dir = layer_dir / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        layer_dirs[layer_idx] = {
            'layer_dir': layer_dir,
            'embeddings_dir': embeddings_dir,
            'counter': 0,
            'token_dict': {}
        }
    
    # Check if model is already wrapped with DataParallel (it should be from main())
    # Don't wrap it again to avoid double-wrapping issues
    if use_data_parallel:
        if isinstance(model, torch.nn.DataParallel):
            print(f"Model already using DataParallel")
        else:
            print(f"WARNING: use_data_parallel=True but model not wrapped. This shouldn't happen.")
    
    # Process captions in batches for efficiency
    print(f"\nProcessing {len(caption_positions)} captions in batches...")
    captions_processed = 0
    sanity_check_done = False
    
    # Convert to list for batching
    caption_items = list(caption_positions.items())
    batch_size = BATCH_SIZE  # Match the batch size used in normal mode
    
    for batch_start in tqdm(range(0, len(caption_items), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(caption_items))
        batch = caption_items[batch_start:batch_end]
        
        # Prepare batch
        batch_captions = [caption for caption, _ in batch]
        batch_positions_lists = [positions_list for _, positions_list in batch]
        
        # Tokenize the batch
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "return_token_type_ids": False,
            "truncation": True,
            "max_length": 512,
            "padding": True  # Pad to same length in batch
        }
        
        encodings = tokenizer(batch_captions, **tokenizer_kwargs)
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        # Get model outputs for the batch
        with torch.no_grad():
            outputs = model(
                input_ids=encodings['input_ids'],
                attention_mask=encodings.get('attention_mask'),
                output_hidden_states=True,
                last_logits_only=False,
            )
        
        all_hidden_states = outputs.hidden_states
        
        # Process each caption in the batch
        for batch_idx, (caption, positions_list) in enumerate(batch):
            input_ids = encodings['input_ids'][batch_idx]
            
            # Extract embeddings for each position
            for position, token_id, token_str in positions_list:
                # Verify token_id matches (safety check)
                actual_token_id = input_ids[position].item()
                if actual_token_id != token_id:
                    print(f"  ⚠️  WARNING: Token mismatch at position {position}!")
                    print(f"     Expected: {token_id} ({token_str})")
                    print(f"     Got: {actual_token_id} ({tokenizer.decode([actual_token_id])})")
                    print(f"     Caption: {caption[:100]}...")
                    continue  # Skip this position
                
                # Extract from each target layer
                for layer_idx in target_layers:
                    hidden_state = all_hidden_states[layer_idx][batch_idx, position].cpu().numpy()
                    
                    # CRITICAL: Check for NaN IMMEDIATELY - FAIL LOUDLY if detected
                    if np.isnan(hidden_state).any():
                        error_msg = (
                            f"\n{'='*80}\n"
                            f"CRITICAL ERROR: NaN detected in hidden state! (Reference Mode)\n"
                            f"{'='*80}\n"
                            f"Layer: {layer_idx}\n"
                            f"Caption: {caption}\n"
                            f"Token: '{token_str}' (ID: {token_id})\n"
                            f"Position: {position}\n"
                            f"NaN count: {np.isnan(hidden_state).sum()}/{hidden_state.size}\n"
                            f"NaN indices: {np.where(np.isnan(hidden_state))[0][:20].tolist()}\n"
                            f"{'='*80}\n"
                            f"This indicates a numerical instability in the model forward pass.\n"
                            f"{'='*80}\n"
                        )
                        raise ValueError(error_msg)
                    
                    layer_info = layer_dirs[layer_idx]
                    counter = layer_info['counter']
                    
                    # Save embedding
                    embeddings_dir = layer_info['embeddings_dir']
                    embedding_path = embeddings_dir / f"emb_{counter:08d}.npy"
                    
                    # Convert to target dtype
                    hidden_state_converted = convert_embedding_dtype(hidden_state, embedding_dtype)
                    
                    # Double-check: verify no NaN after dtype conversion
                    if embedding_dtype in ['float8', 'float16']:
                        check_array = convert_from_stored_dtype(hidden_state_converted)
                        if np.isnan(check_array).any():
                            # Debug: show the values that caused overflow
                            abs_max = np.abs(hidden_state).max()
                            abs_min = np.abs(hidden_state[hidden_state != 0]).min() if np.any(hidden_state != 0) else 0
                            raise ValueError(
                                f"NaN introduced during dtype conversion to {embedding_dtype}!\n"
                                f"Layer {layer_idx}, token '{token_str}', caption: {caption}\n"
                                f"Hidden state range before conversion: [{hidden_state.min():.4f}, {hidden_state.max():.4f}]\n"
                                f"Absolute max: {abs_max:.4f}, Absolute min (non-zero): {abs_min:.8f}\n"
                                f"float8_e4m3fn range: ~[-448, 448]\n"
                                f"Values outside float8 range: {np.sum(np.abs(hidden_state) > 448)} / {hidden_state.size}\n"
                                f"→ Hidden state values are too large for float8!"
                            )
                    
                    np.save(embedding_path, hidden_state_converted)
                    
                    # Add to token dict
                    token_dict = layer_info['token_dict']
                    if token_str not in token_dict:
                        token_dict[token_str] = []
                    
                    token_dict[token_str].append({
                        'embedding_path': str(embedding_path.relative_to(layer_info['layer_dir'])),
                        'caption': caption,
                        'position': position,
                        'token_id': token_id,
                        'dtype': str(hidden_state_converted.dtype)
                    })
                    
                    layer_info['counter'] += 1
            
            captions_processed += 1
        
        # Periodic JSON saving
        if captions_processed % save_frequency == 0:
            print(f"\n💾 Saving progress at {captions_processed}/{len(caption_positions)} captions...")
            for layer_idx, layer_info in layer_dirs.items():
                layer_dir = layer_info['layer_dir']
                token_dict = layer_info['token_dict']
                output_file = layer_dir / "token_embeddings.json"
                
                with open(output_file, 'w') as f:
                    json.dump(token_dict, f, indent=2)
                
                num_emb = sum(len(embs) for embs in token_dict.values())
                print(f"  Layer {layer_idx}: {len(token_dict)} tokens, {num_emb} embeddings")
        
        # Sanity check once after we have at least 2 tokens
        if not sanity_check_done and len(layer_dirs[target_layers[0]]['token_dict']) >= 2:
            print(f"\n🔍 SANITY CHECK: Verifying embedding alignment and distinctness...")
            # Load a reference embedding
            ref_layer_dir = output_dir / f"layer_{reference_layer_idx}"
            available_tokens = list(layer_dirs[target_layers[0]]['token_dict'].keys())
            
            if len(available_tokens) >= 2:
                first_token = available_tokens[0]
                second_token = available_tokens[1]
                
                # Get first embedding for this token from reference layer
                ref_json_file = ref_layer_dir / "token_embeddings.json"
                with open(ref_json_file, 'r') as f:
                    ref_json = json.load(f)
                
                # Check 1: Same token across different layers (should be similar)
                if first_token in ref_json:
                    # Handle both list and dict (preferred/fallback) formats
                    ref_data = ref_json[first_token]
                    if isinstance(ref_data, dict):
                        # Dict format with 'preferred' and 'fallback'
                        ref_emb_list = ref_data.get('preferred', []) + ref_data.get('fallback', [])
                    else:
                        # List format
                        ref_emb_list = ref_data
                    
                    if len(ref_emb_list) > 0:
                        ref_emb_info = ref_emb_list[0]
                        ref_emb_path = ref_layer_dir / ref_emb_info['embedding_path']
                        ref_emb = np.load(ref_emb_path)
                        
                        # Compare to target layer
                        target_emb_info = layer_dirs[target_layers[0]]['token_dict'][first_token][0]
                        target_emb_path = layer_dirs[target_layers[0]]['layer_dir'] / target_emb_info['embedding_path']
                        target_emb = np.load(target_emb_path)
                        
                        # Convert to float32 for comparison (handles all dtypes including float8)
                        ref_emb_fp32 = convert_from_stored_dtype(ref_emb)
                        target_emb_fp32 = convert_from_stored_dtype(target_emb)
                        
                        # Compute cosine similarity
                        cos_sim_cross_layer = np.dot(ref_emb_fp32, target_emb_fp32) / (np.linalg.norm(ref_emb_fp32) * np.linalg.norm(target_emb_fp32))
                        
                        print(f"\n  CHECK 1: Same token across layers (should be MORE similar than random tokens)")
                        print(f"  Token: '{first_token}'")
                        print(f"  Caption: {ref_emb_info['caption'][:60]}...")
                        print(f"  Position: {ref_emb_info['position']}")
                        print(f"  Cosine similarity (layer {reference_layer_idx} vs {target_layers[0]}): {cos_sim_cross_layer:.4f}")
                        
                        # Check 2: Different/unrelated tokens across the same two layers (baseline)
                        if second_token in ref_json and second_token in layer_dirs[target_layers[0]]['token_dict']:
                            # Get second token from reference layer
                            ref_data_2 = ref_json[second_token]
                            if isinstance(ref_data_2, dict):
                                ref_emb_list_2 = ref_data_2.get('preferred', []) + ref_data_2.get('fallback', [])
                            else:
                                ref_emb_list_2 = ref_data_2
                            
                            if len(ref_emb_list_2) > 0:
                                ref_emb_info_2 = ref_emb_list_2[0]
                                ref_emb_path_2 = ref_layer_dir / ref_emb_info_2['embedding_path']
                                ref_emb_2 = np.load(ref_emb_path_2)
                                ref_emb_2_fp32 = convert_from_stored_dtype(ref_emb_2)
                                
                                # Get second token from target layer
                                target_emb_info_2 = layer_dirs[target_layers[0]]['token_dict'][second_token][0]
                                target_emb_path_2 = layer_dirs[target_layers[0]]['layer_dir'] / target_emb_info_2['embedding_path']
                                target_emb_2 = np.load(target_emb_path_2)
                                target_emb_2_fp32 = convert_from_stored_dtype(target_emb_2)
                                
                                # Baseline: Compare unrelated tokens across layers (token A from ref layer vs token B from target layer)
                                cos_sim_baseline = np.dot(ref_emb_fp32, target_emb_2_fp32) / (np.linalg.norm(ref_emb_fp32) * np.linalg.norm(target_emb_2_fp32))
                                
                                print(f"\n  CHECK 2: Unrelated tokens across layers (baseline - should be LESS similar)")
                                print(f"  Token from layer {reference_layer_idx}: '{first_token}'")
                                print(f"  Token from layer {target_layers[0]}: '{second_token}'")
                                print(f"  Cosine similarity: {cos_sim_baseline:.4f}")
                                
                                # Compare the two similarities
                                print(f"\n  COMPARISON:")
                                print(f"  Same token across layers:     {cos_sim_cross_layer:.4f}")
                                print(f"  Unrelated tokens across layers: {cos_sim_baseline:.4f}")
                                
                                if cos_sim_cross_layer > cos_sim_baseline:
                                    print(f"  ✓ SANITY CHECK PASSED: Same token is more similar ({cos_sim_cross_layer:.4f}) than unrelated tokens ({cos_sim_baseline:.4f})!")
                                else:
                                    print(f"  ❌ WARNING: Unrelated tokens ({cos_sim_baseline:.4f}) are more similar than same token ({cos_sim_cross_layer:.4f})!")
                
                sanity_check_done = True
    
    print(f"\n✓ Extraction complete!")
    for layer_idx in target_layers:
        num_unique = len(layer_dirs[layer_idx]['token_dict'])
        num_embeddings = layer_dirs[layer_idx]['counter']
        print(f"  Layer {layer_idx}: {num_embeddings} embeddings for {num_unique} unique tokens")
    
    # Final save of all JSONs
    print(f"\n💾 Saving final JSON files...")
    for layer_idx, layer_info in layer_dirs.items():
        layer_dir = layer_info['layer_dir']
        token_dict = layer_info['token_dict']
        output_file = layer_dir / "token_embeddings.json"
        
        with open(output_file, 'w') as f:
            json.dump(token_dict, f, indent=2)
        
        file_size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"  Layer {layer_idx}: {output_file} ({file_size_mb:.1f} MB)")
    
    print(f"\n✓ All JSON files saved successfully!")
    
    return layer_dirs

def save_token_embeddings(layer_dirs, output_dir, model_name, layers_extracted, num_captions, max_captions_per_token=MAX_CAPTIONS_PER_TOKEN, embedding_dtype=EMBEDDING_DTYPE, dataset_name="vg", apply_position_filter=False):
    """
    Save token embeddings dictionaries to JSON files, one per layer.
    Combines preferred and fallback lists before saving.
    """
    
    print(f"\nCombining preferred (pos >= 10) and fallback (2 <= pos < 10) entries...")
    
    # Calculate total statistics
    total_unique_tokens = set()
    total_embeddings = 0
    layer_stats = {}
    
    for layer_idx, layer_info in layer_dirs.items():
        token_dict = layer_info.get('token_dict', {})
        
        # Combine preferred and fallback lists for each token
        for token_str, data in token_dict.items():
            # Check if data is already a list (from reference layer extraction) or dict (from normal extraction)
            if isinstance(data, list):
                # Already in final format from reference layer extraction
                combined = data
            elif isinstance(data, dict):
                # Dict format from normal extraction - combine preferred and fallback
                preferred = data.get('preferred', [])
                fallback = data.get('fallback', [])
                
                # Start with preferred entries
                combined = preferred.copy()
                
                # If we need more to reach max_captions_per_token, add fallback
                if len(combined) < max_captions_per_token:
                    needed = max_captions_per_token - len(combined)
                    combined.extend(fallback[:needed])
            else:
                # Unexpected format, skip
                print(f"Warning: Unexpected data format for token {token_str}: {type(data)}")
                continue
            
            # Replace with final combined list
            token_dict[token_str] = combined
        
        num_embeddings = sum(len(embs) for embs in token_dict.values())
        num_unique = len(token_dict)
        total_embeddings += num_embeddings
        total_unique_tokens.update(token_dict.keys())
        layer_stats[f"layer_{layer_idx}"] = {
            'num_unique_tokens': num_unique,
            'num_embeddings': num_embeddings
        }
    
    # Calculate storage savings
    dtype_size_map = {'float32': 4, 'float16': 2, 'float8': 1}
    storage_multiplier = dtype_size_map.get(embedding_dtype, 4) / 4.0
    
    # Create global metadata
    dataset_source = 'Visual Genome phrases'
    
    if apply_position_filter:
        position_filtering_info = {
            'enabled': True,
            'description': 'Only tokens at position >= 2 are included',
            'preferred_positions': 'position >= 10',
            'fallback_positions': '2 <= position < 10',
            'note': 'Preferred positions are prioritized; fallback positions supplement if needed'
        }
    else:
        position_filtering_info = {
            'enabled': False,
            'description': 'All token positions are included (0, 1, 2, ...)',
            'note': 'All positions are treated equally without preference'
        }
    
    metadata = {
        'model_name': model_name,
        'layers_extracted': layers_extracted,
        'num_captions_processed': num_captions,
        'max_captions_per_token': max_captions_per_token,
        'total_unique_tokens_across_all_layers': len(total_unique_tokens),
        'total_embeddings_across_all_layers': total_embeddings,
        'layer_statistics': layer_stats,
        'embedding_dtype': embedding_dtype,
        'dataset': dataset_name,
        'data_source': dataset_source,
        'storage_optimization': {
            'dtype': embedding_dtype,
            'bytes_per_value': dtype_size_map.get(embedding_dtype, 4),
            'storage_vs_fp32': f"{storage_multiplier:.1%}",
            'note': 'Embeddings are stored in reduced precision to save disk space'
        },
        'position_filtering': position_filtering_info,
        'sampling_method': 'Reservoir sampling to limit embeddings per token',
        'note': 'Layer indices follow model.forward() hidden_states convention: hidden_states[i] = output of block i-1',
        'structure': 'Each layer has its own subdirectory with token_embeddings.json and embeddings/ folder',
        'dry_run_mode': DRY_RUN_MODE,
        'note_dry_run': 'Dry run mode stores only token metadata, not actual embeddings' if DRY_RUN_MODE else None
    }
    
    # Save global metadata
    print(f"\nSaving global metadata...")
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save per-layer token embeddings
    for layer_idx, layer_info in layer_dirs.items():
        layer_dir = layer_info['layer_dir']
        token_dict = layer_info.get('token_dict', {})
        output_file = layer_dir / "token_embeddings.json"
        
        print(f"Saving layer {layer_idx} token embeddings to {output_file}...")
        with open(output_file, "w") as f:
            json.dump(token_dict, f, indent=2)
        
        num_emb = sum(len(embs) for embs in token_dict.values())
        print(f"  Saved {len(token_dict)} unique tokens with {num_emb} embeddings")
    
    print(f"\n✓ All token embeddings saved")
    print(f"Global metadata: {metadata['total_unique_tokens_across_all_layers']} unique tokens, {metadata['total_embeddings_across_all_layers']} total embeddings")

def main():
    # Declare global variables first
    global DRY_RUN_MODE
    
    parser = argparse.ArgumentParser(
        description="Create contextual embeddings from captions",
        epilog="""
Examples:
  # Process Llama with specific layers
  python %(prog)s --model meta-llama/Meta-Llama-3-8B --layers 1 2 4 8 16 24
  
  # Use negative indices for last layers (model-agnostic)
  python %(prog)s --model Qwen/Qwen2-7B --layers -1 -2 -4
  
  # Mix positive and negative indices
  python %(prog)s --model allenai/OLMo-7B-1024-preview --layers 1 2 4 -1
  
  # Process multiple models
  python %(prog)s --model Qwen/Qwen2-7B --model allenai/OLMo-7B-1024-preview --layers 8 16 24
  
  # Use reference layer to extract from additional layers
  python %(prog)s --model Qwen/Qwen2-7B --layers 1 2 4 --reference-layer 8
  
  # Process limited number of captions
  python %(prog)s --model Qwen/Qwen2-7B --layers 8 16 24 --num-captions 100000
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true", 
                       help="Dry run mode: only store token metadata, not actual embeddings")
    parser.add_argument("--num-captions", type=int, default=NUM_CAPTIONS,
                       help=f"Number of captions to process (default: {NUM_CAPTIONS}). Set to -1 or None for all captions")
    parser.add_argument("--max-captions-per-token", type=int, default=MAX_CAPTIONS_PER_TOKEN,
                       help=f"Maximum number of embeddings to store per token (default: {MAX_CAPTIONS_PER_TOKEN})")
    parser.add_argument("--embedding-dtype", type=str, default=EMBEDDING_DTYPE,
                       choices=['float32', 'float16', 'float8'],
                       help=f"Data type for saving embeddings (default: {EMBEDDING_DTYPE}). float16 saves 2x, float8 saves 4x storage vs float32")
    parser.add_argument("--output-dir", type=str, default="molmo_data/contextual_llm_embeddings",
                       help="Base output directory (default: analysis_results/contextual_llm_embeddings)")
    parser.add_argument("--reference-layer", type=int, default=None,
                       help="Reference layer to copy token/caption/position selections from. If specified, extracts embeddings for the exact same tokens that were saved in the reference layer.")
    parser.add_argument("--vg-file", type=str, default="reproduce/vg_phrases.txt",
                       help="Path to Visual Genome phrases file")
    parser.add_argument("--model", type=str, action='append', 
                       help="Model name to process (can be specified multiple times). Examples: allenai/OLMo-7B-1024-preview, Qwen/Qwen2-7B, meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--layers", type=int, nargs='+', default=None,
                       help="Layers to extract (e.g., --layers 1 2 4 or --layers 8 16 24). Supports negative indices: -1 for last layer, -2 for second-to-last, etc. If not specified, uses LAYERS_TO_EXTRACT constant or auto-computes")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reservoir sampling (default: 42). Use same seed for reproducibility")
    args = parser.parse_args()
    
    # Set global dry run mode and use num_captions from args
    DRY_RUN_MODE = args.dry_run
    num_captions = None if args.num_captions == -1 else args.num_captions
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    # VG phrases: no position filtering needed (short phrases, all positions useful)
    apply_position_filter = False

    if DRY_RUN_MODE:
        print("DRY RUN MODE: Only storing token metadata, not actual embeddings")
        print(f"   Processing {'ALL' if num_captions is None else num_captions} phrases from {args.vg_file}")
        print()

    print(f"Dataset: Visual Genome phrases")
    if apply_position_filter:
        print(f"Position filtering: DISABLED - including all token positions")
    print(f"Reservoir sampling: max {args.max_captions_per_token} embeddings per token per layer")
    print(f"Embedding dtype: {args.embedding_dtype} ({'2x' if args.embedding_dtype == 'float16' else '4x' if args.embedding_dtype == 'float8' else '1x'} storage savings vs float32)")
    print(f"Number of captions: {'ALL' if num_captions is None else f'{num_captions:,}'}")
    print()
    
    # Determine which models to process
    if args.model:
        model_names = args.model
    else:
        # Default models if none specified
        model_names = [
            "allenai/OLMo-7B-1024-preview"
        ]
        print(f"⚠️  No --model specified, using default: {model_names}")
        print(f"   Use --model to specify model(s), e.g., --model Qwen/Qwen2-7B\n")
    
    for model_name in model_names:
        print(f"\n{'='*80}")
        print(f"Processing model: {model_name}")
        print(f"{'='*80}\n")
        
        # Load tokenizer (always needed)
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Set padding token if not present (needed for Llama models)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set pad_token to eos_token: {tokenizer.eos_token}")
        
        if not DRY_RUN_MODE:
            # Load model only if not in dry run mode
            print("Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float32, 
                trust_remote_code=True
            )
            
            # Move model to GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
            model = model.to(device)
            
            # Use DataParallel for multi-GPU if available
            use_data_parallel = False
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                num_gpus = torch.cuda.device_count()
                print(f"Using DataParallel with {num_gpus} GPUs")
                print(f"Note: Batch size {BATCH_SIZE} will be split across {num_gpus} GPUs ({BATCH_SIZE // num_gpus} per GPU)")
                model = torch.nn.DataParallel(model)
                use_data_parallel = True
            
            model.eval()
        else:
            # Dry run mode - no model needed
            model = None
            device = None
            use_data_parallel = False
        
        # Create output directory
        base_output = Path(args.output_dir)
        output_dir = base_output / model_name.replace("/", "_")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}\n")
        
        # Load Visual Genome phrases (auto-downloads if missing)
        captions = load_vg_phrases(args.vg_file, num_phrases=num_captions)
        
        if DRY_RUN_MODE:
            # Dry run: only tokenizer analysis
            print("Running tokenizer-only analysis...")
            total_tokens, captions_processed = analyze_tokenizer_output(
                tokenizer, captions, BATCH_SIZE, output_dir, start_offset=0
            )
            print(f"\n✅ Tokenizer analysis complete!")
            print(f"   Processed {captions_processed} captions")
            print(f"   Found {total_tokens:,} total tokens")
        else:
            # Normal mode: full LLM processing
            # Compute layers to extract (access .module.config if using DataParallel)
            model_config = model.module.config if use_data_parallel else model.config
            n_layers = model_config.num_hidden_layers
            
            # Use layers from command-line args first, then LAYERS_TO_EXTRACT constant, then auto-compute
            if args.layers is not None:
                # Convert negative indices to positive (e.g., -1 -> n_layers-1)
                layers_to_extract = []
                for layer_idx in args.layers:
                    if layer_idx < 0:
                        positive_idx = n_layers + layer_idx
                        if positive_idx < 0:
                            raise ValueError(f"Layer index {layer_idx} is out of bounds for model with {n_layers} layers")
                        layers_to_extract.append(positive_idx)
                    else:
                        if layer_idx >= n_layers:
                            raise ValueError(f"Layer index {layer_idx} is out of bounds for model with {n_layers} layers (0-{n_layers-1})")
                        layers_to_extract.append(layer_idx)
                
                print(f"Model has {n_layers} layers")
                print(f"Extracting from layers (command-line specified): {layers_to_extract}")
                if any(idx < 0 for idx in args.layers):
                    print(f"  (converted from: {args.layers})")
                print()
            elif LAYERS_TO_EXTRACT is not None:
                layers_to_extract = LAYERS_TO_EXTRACT
                print(f"Model has {n_layers} layers")
                print(f"Extracting from layers (LAYERS_TO_EXTRACT constant): {layers_to_extract}\n")
            else:
                # Default: [1, 2, 4, 8, 16, 24, n-2, n-1]
                # Covers early/mid/late layers; matches the paper's analysis grid.
                layers_to_extract = sorted(set(
                    [l for l in [1, 2, 4, 8, 16, 24] if l < n_layers]
                    + [n_layers - 2, n_layers - 1]
                ))
                print(f"Model has {n_layers} layers")
                print(f"Extracting from layers (default): {layers_to_extract}\n")
            
            # Check if using reference layer mode
            if args.reference_layer is not None:
                # Reference layer mode: extract exact same tokens/positions from target layers
                layer_dirs = extract_from_reference_layer(
                    model, tokenizer, args.reference_layer, layers_to_extract,
                    output_dir, device, embedding_dtype=args.embedding_dtype,
                    use_data_parallel=use_data_parallel
                )
                
                # Save results
                print("\nSaving results...")
                save_token_embeddings(layer_dirs, output_dir, model_name, layers_to_extract,
                                    num_captions=0, max_captions_per_token=args.max_captions_per_token,
                                    embedding_dtype=args.embedding_dtype,
                                    dataset_name="vg", apply_position_filter=apply_position_filter)

                build_caches_for_output(output_dir, layers_to_extract)

                print(f"\n✓ Completed processing for {model_name}")
                print(f"Results saved to: {output_dir}\n")
                continue
            
            # Create layer directories
            layer_dirs = {}
            for layer_idx in layers_to_extract:
                layer_dir = output_dir / f"layer_{layer_idx}"
                layer_dir.mkdir(parents=True, exist_ok=True)
                embeddings_dir = layer_dir / "embeddings"
                embeddings_dir.mkdir(parents=True, exist_ok=True)
                layer_dirs[layer_idx] = {
                    'layer_dir': layer_dir,
                    'embeddings_dir': embeddings_dir,
                    'counter': 0,
                    'token_dict': {}
                }
            
            # Load existing progress if resuming
            start_offset, token_seen_counts = load_existing_embeddings(layer_dirs, output_dir, layers_to_extract)
            
            # Check if already completed
            if start_offset >= len(captions):
                print(f"\nAll {len(captions)} captions already processed!")
                print(f"Delete {output_dir / 'progress.json'} to restart from scratch.")
                continue
            
            # Extract contextual embeddings from captions
            total_embeddings, captions_processed = extract_contextual_embeddings(
                model, tokenizer, captions, layers_to_extract, 
                BATCH_SIZE, layer_dirs, device, output_dir, start_offset, 
                max_captions_per_token=args.max_captions_per_token,
                token_seen_counts_init=token_seen_counts,
                embedding_dtype=args.embedding_dtype,
                apply_position_filter=apply_position_filter
            )
            
            print(f"\n\nTotal embeddings extracted: {total_embeddings}")
            
            # Final save
            print("\nSaving final results...")
            save_token_embeddings(layer_dirs, output_dir, model_name, layers_to_extract,
                                captions_processed, max_captions_per_token=args.max_captions_per_token,
                                embedding_dtype=args.embedding_dtype,
                                dataset_name="vg", apply_position_filter=apply_position_filter)

            build_caches_for_output(output_dir, layers_to_extract)

        print(f"\n✓ Completed processing for {model_name}")
        print(f"Results saved to: {output_dir}\n")

def build_cache_for_layer(layer_dir):
    """Build embeddings_cache.pt from individual .npy files in a layer directory.

    Consolidates the many small .npy embedding files into a single .pt file
    containing a stacked tensor + metadata list. This is required by
    run_latentlens.py for efficient GPU-based nearest neighbor search.

    Args:
        layer_dir: Path to layer directory containing token_embeddings.json and embeddings/

    Returns:
        (success: bool, message: str)
    """
    from concurrent.futures import ThreadPoolExecutor

    layer_dir = Path(layer_dir)
    cache_file = layer_dir / "embeddings_cache.pt"

    token_embeddings_file = layer_dir / "token_embeddings.json"
    if not token_embeddings_file.exists():
        return False, "token_embeddings.json not found"

    with open(token_embeddings_file) as f:
        token_dict = json.load(f)

    # Prepare loading tasks
    loading_tasks = []
    for token_str, embeddings_info in token_dict.items():
        if isinstance(embeddings_info, dict):
            embeddings_info = embeddings_info.get('preferred', []) + embeddings_info.get('fallback', [])
        for emb_info in embeddings_info:
            if not isinstance(emb_info, dict):
                continue
            embedding_path = layer_dir / emb_info['embedding_path']
            loading_tasks.append((embedding_path, token_str, emb_info))

    # Load embeddings with thread pool for parallel I/O
    all_embeddings = []
    metadata_list = []

    def _load_one(args_tuple):
        emb_path, tok_str, emb_info = args_tuple
        if not emb_path.exists():
            return None
        embedding = np.load(emb_path, allow_pickle=False, mmap_mode='r')
        emb_f32 = convert_from_stored_dtype(embedding).copy()
        return {
            'embedding': emb_f32,
            'metadata': {
                'token_str': tok_str,
                'token_id': emb_info['token_id'],
                'caption': emb_info['caption'],
                'position': emb_info['position'],
            }
        }

    with ThreadPoolExecutor(max_workers=32) as executor:
        for result in executor.map(_load_one, loading_tasks):
            if result is None:
                continue
            all_embeddings.append(result['embedding'])
            metadata_list.append(result['metadata'])

    if len(all_embeddings) == 0:
        return False, "No embeddings loaded"

    # Stack into single tensor
    embeddings_matrix = torch.from_numpy(np.stack(all_embeddings, axis=0))

    # Build token_to_indices mapping
    token_to_indices = defaultdict(list)
    for idx, meta in enumerate(metadata_list):
        token_to_indices[meta['token_str']].append(idx)
    token_to_indices = dict(token_to_indices)

    # Save atomically
    cache_data = {
        'embeddings': embeddings_matrix,
        'metadata': metadata_list,
        'token_to_indices': token_to_indices,
    }
    temp_file = cache_file.with_suffix('.pt.tmp')
    torch.save(cache_data, temp_file)
    temp_file.rename(cache_file)

    return True, f"Cached {len(metadata_list)} embeddings ({len(token_to_indices)} unique tokens)"


def build_caches_for_output(output_dir, layers_to_extract):
    """Build embeddings_cache.pt for all extracted layers.

    Called automatically after extraction completes so that
    run_latentlens.py can load the embeddings without a separate step.
    """
    print("\nBuilding embedding caches (for run_latentlens.py)...")
    for layer_idx in layers_to_extract:
        layer_dir = output_dir / f"layer_{layer_idx}"
        if not layer_dir.exists():
            continue
        cache_file = layer_dir / "embeddings_cache.pt"
        if cache_file.exists():
            print(f"  layer_{layer_idx}: cache already exists, skipping")
            continue
        print(f"  layer_{layer_idx}: building cache...", end=" ", flush=True)
        success, msg = build_cache_for_layer(layer_dir)
        if success:
            print(f"done ({msg})")
        else:
            print(f"FAILED ({msg})")


if __name__ == "__main__":
    main()
