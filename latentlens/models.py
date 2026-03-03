"""
Model loading and hidden-state extraction for supported HuggingFace causal LMs.

Users can also pass any HuggingFace model name — the helpers here handle
pad-token setup, eval mode, and the ``output_hidden_states=True`` forward pass.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    Load a HuggingFace causal LM and its tokenizer.

    Sets the model to eval mode and ensures a pad token is defined (required
    for batched tokenization).

    Parameters
    ----------
    model_name : str
        HuggingFace model ID (e.g., ``"allenai/OLMo-7B-1024-preview"``).
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

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=trust_remote_code
    )
    model = model.to(device).eval()

    return model, tokenizer


def get_hidden_states(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, ...]:
    """
    Run a forward pass and return all hidden states (including the input embedding layer).

    Parameters
    ----------
    model : PreTrainedModel
        A HuggingFace causal LM (in eval mode).
    input_ids : Tensor of shape ``[batch, seq_len]``
        Tokenized input IDs.
    attention_mask : Tensor, optional
        Attention mask (1 = real token, 0 = padding).

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
        )
    return outputs.hidden_states
