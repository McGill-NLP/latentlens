"""
Model loading and hidden-state extraction for any HuggingFace model.

Works with causal LMs, VLMs (Qwen2-VL, LLaVA, ...), speech LLMs,
video LLMs, or any model that supports ``output_hidden_states=True``.

The loader tries ``AutoModelForCausalLM`` first (most common), then
falls back to ``AutoModel`` for architectures that don't register as
causal LMs (e.g., VLMs with conditional generation heads).
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# VLM auto-class: prefer AutoModelForImageTextToText (transformers >= 4.55),
# fall back to AutoModelForVision2Seq (transformers 4.36–4.54)
try:
    from transformers import AutoModelForImageTextToText as _VLMAutoClass
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq as _VLMAutoClass
    except ImportError:
        _VLMAutoClass = None


# Known model configurations.  ``num_hidden_layers`` and ``hidden_size`` are
# informational (the actual values are read from model.config at runtime).
# ``default_layers`` is used by :func:`~latentlens.extract.auto_layers` when
# no explicit layer list is given.
SUPPORTED_MODELS: dict[str, dict] = {
    "allenai/OLMo-7B-1024-preview": {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "default_layers": [1, 2, 4, 8, 16, 24, 30, 31],
    },
    "meta-llama/Meta-Llama-3-8B": {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "default_layers": [1, 2, 4, 8, 16, 24, 30, 31],
    },
    "Qwen/Qwen2-7B": {
        "num_hidden_layers": 28,
        "hidden_size": 3584,
        "default_layers": [1, 2, 4, 8, 16, 24, 26, 27],
    },
}


def load_model(
    model_name: str,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    trust_remote_code: bool = True,
) -> tuple:
    """
    Load a HuggingFace model and its tokenizer.

    Tries ``AutoModelForCausalLM`` first (standard LLMs), then falls back
    to ``AutoModel`` for VLMs, speech LLMs, video LLMs, and other
    architectures that don't register as causal LMs.

    Sets the model to eval mode and ensures a pad token is defined (required
    for batched tokenization).

    Parameters
    ----------
    model_name : str
        Any HuggingFace model ID — causal LMs, VLMs, speech models, etc.
    device : str or torch.device, optional
        Target device. Defaults to ``"cuda"`` if available, else ``"cpu"``.
    dtype : torch.dtype
        Model weight dtype (default ``torch.float32``).
    trust_remote_code : bool
        Passed to ``from_pretrained`` (required for OLMo and Qwen models).

    Returns
    -------
    (model, tokenizer)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try loading in order: CausalLM (standard LLMs) → VLM auto-class
    # (Qwen2-VL, LLaVA, Molmo, ...) → AutoModel (catch-all)
    load_kwargs = dict(torch_dtype=dtype, trust_remote_code=trust_remote_code)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except (ValueError, KeyError, TypeError):
        if _VLMAutoClass is not None:
            try:
                model = _VLMAutoClass.from_pretrained(model_name, **load_kwargs)
            except (ValueError, KeyError, TypeError):
                model = AutoModel.from_pretrained(model_name, **load_kwargs)
        else:
            model = AutoModel.from_pretrained(model_name, **load_kwargs)
    model = model.to(device).eval()

    return model, tokenizer


def get_num_hidden_layers(model) -> int:
    """
    Get the number of hidden layers from a model config.

    Handles both standard LLMs (``config.num_hidden_layers``) and VLMs
    where the LLM config is nested under ``config.text_config``.
    """
    config = model.config
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    if hasattr(config, "text_config") and hasattr(config.text_config, "num_hidden_layers"):
        return config.text_config.num_hidden_layers
    raise AttributeError(
        f"Cannot determine num_hidden_layers from {type(config).__name__}. "
        "Pass `layers` explicitly to build_index()."
    )


def get_hidden_states(
    model,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, ...]:
    """
    Run a forward pass and return all hidden states (including the input embedding layer).

    For VLMs, pass model-specific inputs (e.g., ``pixel_values``,
    ``image_grid_thw``) as keyword arguments — they are forwarded to
    the model's ``forward()`` method.

    Parameters
    ----------
    model : PreTrainedModel
        A HuggingFace model in eval mode (LLM, VLM, etc.).
    input_ids : Tensor of shape ``[batch, seq_len]``, optional
        Tokenized input IDs.
    attention_mask : Tensor, optional
        Attention mask (1 = real token, 0 = padding).
    **kwargs
        Additional keyword arguments passed to ``model.forward()``
        (e.g., ``pixel_values`` for VLMs).

    Returns
    -------
    tuple[Tensor, ...]
        ``hidden_states[0]`` is the input embedding, ``hidden_states[i]`` for
        ``i >= 1`` is the output of transformer block ``i-1``.  Each tensor has
        shape ``[batch, seq_len, hidden_dim]``.
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
    return outputs.hidden_states
