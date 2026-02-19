"""
ContextualIndex: load, search, and persist contextual embedding caches.

The search algorithm merges nearest neighbors across all contextual layers
(the core LatentLens insight — hidden states are compared against text
embeddings from every layer, then globally ranked).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F


@dataclass
class Neighbor:
    """A single nearest-neighbor result from contextual embedding search."""

    token_str: str
    similarity: float
    caption: str = ""
    position: int = -1
    token_id: int = -1
    contextual_layer: int = -1


class ContextualIndex:
    """
    Searchable index of contextual text embeddings across multiple LLM layers.

    Each layer stores a matrix of L2-normalized embeddings and per-row metadata
    (token string, source caption, position, token ID).

    The :meth:`search` method queries all layers simultaneously and returns
    globally ranked neighbors — the key LatentLens operation.

    Examples
    --------
    >>> index = ContextualIndex.from_pretrained("McGill-NLP/latentlens-qwen2vl-embeddings")
    >>> results = index.search(hidden_states, top_k=5)
    """

    def __init__(self, layers_data: dict[int, dict]) -> None:
        """
        Parameters
        ----------
        layers_data : dict[int, dict]
            Maps layer index to ``{"embeddings": Tensor[N, D], "metadata": list[dict]}``.
            Embeddings should already be L2-normalized.
        """
        self._layers_data = layers_data

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def available_layers(self) -> list[int]:
        return sorted(self._layers_data.keys())

    @property
    def hidden_dim(self) -> int:
        for layer_data in self._layers_data.values():
            return layer_data["embeddings"].shape[1]
        raise ValueError("Index is empty — no layers loaded")

    @property
    def device(self) -> torch.device:
        for layer_data in self._layers_data.values():
            return layer_data["embeddings"].device
        raise ValueError("Index is empty — no layers loaded")

    def __len__(self) -> int:
        """Total number of embeddings across all layers."""
        return sum(ld["embeddings"].shape[0] for ld in self._layers_data.values())

    def __repr__(self) -> str:
        layers = self.available_layers
        counts = [self._layers_data[l]["embeddings"].shape[0] for l in layers]
        total = sum(counts)
        return (
            f"ContextualIndex(layers={layers}, "
            f"total_embeddings={total:,}, hidden_dim={self.hidden_dim})"
        )

    # ── Search ────────────────────────────────────────────────────────────

    def search(
        self,
        query: torch.Tensor,
        top_k: int = 5,
        layers: Optional[Sequence[int]] = None,
    ) -> list[list[Neighbor]]:
        """
        Find the top-k nearest contextual text neighbors for each query vector.

        Searches across all contextual layers and returns globally ranked
        results (cross-layer merge).

        Parameters
        ----------
        query : Tensor of shape ``[num_tokens, hidden_dim]``
            Query vectors (e.g., hidden states from an LLM layer).
            Automatically L2-normalized if not already.
        top_k : int
            Number of neighbors per query token.
        layers : sequence of int, optional
            Subset of contextual layers to search. Defaults to all available.

        Returns
        -------
        list[list[Neighbor]]
            ``results[i]`` is a list of ``top_k`` :class:`Neighbor` objects for
            query token ``i``, sorted by descending similarity.
        """
        if query.ndim == 1:
            query = query.unsqueeze(0)

        query = F.normalize(query.float(), dim=-1)

        search_layers = sorted(layers) if layers is not None else self.available_layers

        device = query.device
        num_tokens = query.shape[0]

        # Phase 1: per-layer top-k candidates
        # Store (values, indices) per contextual layer on CPU
        candidates: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        metadata_cache: dict[int, list[dict]] = {}

        for cl in search_layers:
            layer_data = self._layers_data[cl]
            embeddings = layer_data["embeddings"].to(device)
            metadata_cache[cl] = layer_data["metadata"]

            similarity = torch.matmul(query, embeddings.T)  # [num_tokens, N]
            k = min(top_k, similarity.shape[1])
            vals, idxs = torch.topk(similarity, k=k, dim=-1)
            candidates[cl] = (vals.cpu(), idxs.cpu())
            del similarity

        # Phase 2: cross-layer merge
        ctx_layers = sorted(candidates.keys())
        num_ctx = len(ctx_layers)

        all_vals = torch.stack([candidates[cl][0] for cl in ctx_layers])  # [C, T, k]
        all_idxs = torch.stack([candidates[cl][1] for cl in ctx_layers])

        actual_k = all_vals.shape[2]
        layer_ids = (
            torch.arange(num_ctx).unsqueeze(1).expand(-1, actual_k).flatten()
        )

        results: list[list[Neighbor]] = []
        for tok_idx in range(num_tokens):
            flat_vals = all_vals[:, tok_idx, :].flatten()
            flat_idxs = all_idxs[:, tok_idx, :].flatten()
            merge_k = min(top_k, flat_vals.shape[0])
            global_top_vals, global_top_pos = torch.topk(flat_vals, k=merge_k)

            neighbors: list[Neighbor] = []
            for k_idx in range(merge_k):
                pos = global_top_pos[k_idx].item()
                sim = global_top_vals[k_idx].item()
                cl_idx = layer_ids[pos].item()
                emb_idx = flat_idxs[pos].item()
                ctx_layer = ctx_layers[cl_idx]
                meta = metadata_cache[ctx_layer][emb_idx]
                neighbors.append(
                    Neighbor(
                        token_str=meta.get("token_str", ""),
                        similarity=sim,
                        caption=meta.get("caption", ""),
                        position=meta.get("position", -1),
                        token_id=meta.get("token_id", -1),
                        contextual_layer=ctx_layer,
                    )
                )
            results.append(neighbors)

        return results

    # ── I/O ───────────────────────────────────────────────────────────────

    def save(
        self,
        path: Union[str, Path],
        storage_dtype: torch.dtype = torch.float16,
    ) -> None:
        """
        Save the index to a directory with one sub-directory per layer.

        Each layer is saved as ``layer_N/embeddings_cache.pt`` containing
        ``{"embeddings": Tensor, "metadata": list[dict]}``, compatible with
        the cache format used by ``quickstart.py`` and ``extract_embeddings.py``.

        Parameters
        ----------
        path : str or Path
            Output directory.
        storage_dtype : torch.dtype
            Dtype for stored embeddings (default ``torch.float16`` for 2x
            savings).  Embeddings are cast back to float32 on load.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta_info = {"layers": self.available_layers, "hidden_dim": self.hidden_dim}
        (path / "metadata.json").write_text(json.dumps(meta_info, indent=2))

        for layer, layer_data in self._layers_data.items():
            layer_dir = path / f"layer_{layer}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "embeddings": layer_data["embeddings"].cpu().to(storage_dtype),
                    "metadata": layer_data["metadata"],
                },
                layer_dir / "embeddings_cache.pt",
            )

    @classmethod
    def from_directory(
        cls,
        path: Union[str, Path],
        layers: Optional[Sequence[int]] = None,
    ) -> ContextualIndex:
        """
        Load an index from a directory of ``layer_N/embeddings_cache.pt`` files.

        Parameters
        ----------
        path : str or Path
            Directory containing per-layer cache files.
        layers : sequence of int, optional
            Specific layers to load. If ``None``, loads all available.
        """
        path = Path(path)
        layers_data: dict[int, dict] = {}

        # Discover available layer directories
        layer_dirs = sorted(path.glob("layer_*"))
        for layer_dir in layer_dirs:
            layer_num = int(layer_dir.name.split("_")[1])
            if layers is not None and layer_num not in layers:
                continue
            cache_file = layer_dir / "embeddings_cache.pt"
            if not cache_file.exists():
                continue
            cache = torch.load(cache_file, map_location="cpu", weights_only=False)
            embeddings = F.normalize(cache["embeddings"].float(), dim=-1)
            layers_data[layer_num] = {
                "embeddings": embeddings,
                "metadata": cache["metadata"],
            }

        if not layers_data:
            raise FileNotFoundError(
                f"No embeddings_cache.pt files found in {path}. "
                "Expected directory structure: layer_N/embeddings_cache.pt"
            )

        return cls(layers_data)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        layers: Optional[Sequence[int]] = None,
        cache_dir: Optional[str] = None,
    ) -> ContextualIndex:
        """
        Download and load an index from the HuggingFace Hub.

        Parameters
        ----------
        repo_id : str
            HuggingFace repository ID (e.g., ``"McGill-NLP/latentlens-qwen2vl-embeddings"``).
        layers : sequence of int, optional
            Specific layers to download. If ``None``, downloads all listed in
            the repo's ``metadata.json``, or tries a default set.
        cache_dir : str, optional
            Local cache directory for HuggingFace downloads.
        """
        from huggingface_hub import hf_hub_download, HfApi

        # Try to discover available layers from metadata.json
        if layers is None:
            try:
                meta_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="metadata.json",
                    repo_type="model",
                    cache_dir=cache_dir,
                )
                with open(meta_path) as f:
                    meta = json.load(f)
                layers = meta["layers"]
            except Exception:
                # Fall back to listing repo files
                api = HfApi()
                files = api.list_repo_files(repo_id, repo_type="model")
                layers = sorted(
                    int(f.split("/")[0].split("_")[1])
                    for f in files
                    if f.startswith("layer_") and f.endswith("embeddings_cache.pt")
                )

        if not layers:
            raise ValueError(
                f"Could not determine available layers for {repo_id}. "
                "Pass `layers` explicitly."
            )

        layers_data: dict[int, dict] = {}
        for layer in layers:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=f"layer_{layer}/embeddings_cache.pt",
                repo_type="model",
                cache_dir=cache_dir,
            )
            cache = torch.load(path, map_location="cpu", weights_only=False)
            embeddings = F.normalize(cache["embeddings"].float(), dim=-1)
            layers_data[layer] = {
                "embeddings": embeddings,
                "metadata": cache["metadata"],
            }

        return cls(layers_data)

    # ── Device management ─────────────────────────────────────────────────

    def to(self, device: Union[str, torch.device]) -> ContextualIndex:
        """Move all embeddings to the specified device. Returns self."""
        device = torch.device(device)
        for layer_data in self._layers_data.values():
            layer_data["embeddings"] = layer_data["embeddings"].to(device)
        return self
