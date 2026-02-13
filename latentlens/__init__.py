"""
LatentLens: Interpret continuous token representations via contextual nearest neighbors.

Build a contextual embedding index from any HuggingFace causal LM, then
search it with your own hidden states to understand what those representations encode.
"""

__version__ = "0.1.0"

from latentlens.index import ContextualIndex, Neighbor
from latentlens.extract import auto_layers, build_index, load_corpus
from latentlens.models import SUPPORTED_MODELS, get_hidden_states, load_model

__all__ = [
    "ContextualIndex",
    "Neighbor",
    "auto_layers",
    "build_index",
    "load_corpus",
    "load_model",
    "get_hidden_states",
    "SUPPORTED_MODELS",
]
