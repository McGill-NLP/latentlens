"""
Build a :class:`~latentlens.index.ContextualIndex` from a text corpus and a
HuggingFace causal LM.

The key difference from the paper's ``extract_embeddings.py`` is **prefix
deduplication** instead of reservoir sampling: in a causal LM the hidden state
at position *i* depends only on tokens 0..i, so identical prefixes produce
identical embeddings.  We hash the prefix and skip duplicates, which is both
simpler and avoids storing redundant entries.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm

from latentlens.index import ContextualIndex
from latentlens.models import SUPPORTED_MODELS, get_hidden_states, get_num_hidden_layers, load_model


def auto_layers(num_hidden_layers: int) -> list[int]:
    """
    Return a sensible default set of layers to extract, matching the paper's
    analysis grid.

    Covers early/mid/late layers: ``[1, 2, 4, 8, 16, 24, n-2, n-1]``
    (clamped to the model's layer count).

    Parameters
    ----------
    num_hidden_layers : int
        Total number of transformer blocks (e.g., 32 for LLaMA-3-8B).
    """
    base = [l for l in [1, 2, 4, 8, 16, 24] if l < num_hidden_layers]
    top = [num_hidden_layers - 2, num_hidden_layers - 1]
    return sorted(set(base + top))


def load_corpus(source: Union[str, Path, list[str]]) -> list[str]:
    """
    Load a text corpus.

    Parameters
    ----------
    source : str, Path, or list[str]
        * ``list[str]`` — returned as-is.
        * ``.txt`` file — one sentence per line.
        * ``.csv`` file — first column of each row (header skipped if present).

    Returns
    -------
    list[str]
    """
    if isinstance(source, list):
        return source

    path = Path(source)
    if path.suffix == ".csv":
        texts: list[str] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            # If the first field looks like actual text (not a column name), keep it
            if header and not header[0].strip().lower().startswith(("text", "sentence", "caption", "id", "index")):
                texts.append(header[0].strip())
            for row in reader:
                if row and row[0].strip():
                    texts.append(row[0].strip())
        return texts
    else:
        # Default: one line per sentence (.txt or any other extension)
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]


def build_index(
    model_name: str,
    corpus: Union[str, Path, list[str]],
    layers: Optional[Sequence[int]] = None,
    max_contexts_per_token: int = 50,
    batch_size: int = 32,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    show_progress: bool = True,
) -> ContextualIndex:
    """
    Build a :class:`ContextualIndex` by running a causal LM on a text corpus.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID (e.g., ``"allenai/OLMo-7B-1024-preview"``).
    corpus : str, Path, or list[str]
        Text data — a file path (``.txt`` or ``.csv``) or a list of strings.
    layers : sequence of int, optional
        Which LLM layers to extract.  Defaults to :func:`auto_layers`.
    max_contexts_per_token : int
        Soft cap on how many unique contexts are stored per token string
        (first-come-first-served).
    batch_size : int
        Number of texts per forward pass.
    device : str or torch.device, optional
        Defaults to ``"cuda"`` if available.
    dtype : torch.dtype
        Model weight dtype (default ``torch.float32``).
    show_progress : bool
        Show a ``tqdm`` progress bar.

    Returns
    -------
    ContextualIndex
    """
    # ── Load model ────────────────────────────────────────────────────────
    model, tokenizer = load_model(model_name, device=device, dtype=dtype)
    dev = next(model.parameters()).device

    # ── Determine layers ──────────────────────────────────────────────────
    n_layers = get_num_hidden_layers(model)
    if layers is None:
        if model_name in SUPPORTED_MODELS:
            layers_to_extract = SUPPORTED_MODELS[model_name]["default_layers"]
        else:
            layers_to_extract = auto_layers(n_layers)
    else:
        layers_to_extract = sorted(layers)

    # ── Load corpus ───────────────────────────────────────────────────────
    texts = load_corpus(corpus)

    # ── Storage ───────────────────────────────────────────────────────────
    # Per-layer lists of embeddings and metadata
    layer_embeddings: dict[int, list[torch.Tensor]] = defaultdict(list)
    layer_metadata: dict[int, list[dict]] = defaultdict(list)

    seen_prefixes: set[int] = set()  # hash of prefix token IDs
    token_counts: dict[str, int] = defaultdict(int)  # unique contexts per token

    # ── Process corpus in batches ─────────────────────────────────────────
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Building index", unit="batch")

    for batch_start in iterator:
        batch_texts = texts[batch_start : batch_start + batch_size]

        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        input_ids = encodings["input_ids"].to(dev)
        attention_mask = encodings["attention_mask"].to(dev)

        # Forward pass — returns tuple of (n_layers + 1) tensors
        hidden_states = get_hidden_states(model, input_ids, attention_mask)

        # Process each sentence in the batch
        for sent_idx in range(input_ids.shape[0]):
            sent_ids = input_ids[sent_idx]
            mask = attention_mask[sent_idx]
            valid_len = mask.sum().item()

            for pos in range(2, valid_len):  # skip BOS and position 1
                # Prefix deduplication
                prefix = tuple(sent_ids[:pos + 1].tolist())
                prefix_hash = hash(prefix)
                if prefix_hash in seen_prefixes:
                    continue

                token_id = sent_ids[pos].item()
                token_str = tokenizer.decode([token_id])

                # Soft cap on contexts per token
                if token_counts[token_str] >= max_contexts_per_token:
                    continue

                seen_prefixes.add(prefix_hash)
                token_counts[token_str] += 1

                caption = batch_texts[sent_idx] if sent_idx < len(batch_texts) else ""
                meta = {
                    "token_str": token_str,
                    "token_id": token_id,
                    "caption": caption,
                    "position": pos,
                }

                # Store embedding for ALL extracted layers (shared decision)
                for layer_idx in layers_to_extract:
                    # hidden_states[0] = input embeddings, [i] = block i output
                    emb = hidden_states[layer_idx][sent_idx, pos, :].cpu()
                    layer_embeddings[layer_idx].append(emb)
                    layer_metadata[layer_idx].append(meta)

        del hidden_states
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # ── Assemble and normalize ────────────────────────────────────────────
    layers_data: dict[int, dict] = {}
    for layer_idx in layers_to_extract:
        if not layer_embeddings[layer_idx]:
            continue
        emb_tensor = torch.stack(layer_embeddings[layer_idx])  # [N, D]
        emb_tensor = F.normalize(emb_tensor.float(), dim=-1)
        layers_data[layer_idx] = {
            "embeddings": emb_tensor,
            "metadata": layer_metadata[layer_idx],
        }

    return ContextualIndex(layers_data)
