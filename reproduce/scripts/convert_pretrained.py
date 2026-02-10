#!/usr/bin/env python3
"""
Convert HuggingFace pretrained models to Molmo format.

Downloads base LLMs and vision encoders from HuggingFace, converts their
state dicts to Molmo's expected format, and saves as single .pt files.

This is needed because Molmo uses a custom weight layout that differs from
the HuggingFace model format.

Usage:
    # Convert a single model
    python scripts/convert_pretrained.py openai --output-dir pretrained/
    python scripts/convert_pretrained.py olmo_1024_preview --output-dir pretrained/

    # Convert all models needed for reproduction
    python scripts/convert_pretrained.py --all --output-dir pretrained/

    # List available models
    python scripts/convert_pretrained.py --list
"""

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import AutoModel, AutoModelForCausalLM, CLIPModel, SiglipModel


# ============================================================================
# Dict flatten/unflatten utilities (replaces flax.traverse_util dependency)
# ============================================================================

def _flatten_dict(d, parent_key="", sep="."):
    """Flatten a nested dict into a flat dict with dotted keys."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def _unflatten_dict(d, sep="."):
    """Unflatten a flat dict with dotted keys into a nested dict."""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result

from molmo.config import (
    VisionBackboneConfig, ModelConfig, BlockType, LayerNormType,
    AttentionType, TokenizerConfig,
)


# ============================================================================
# Vision backbone configs (from original launch_scripts/utils.py)
# ============================================================================

CLIP_VISION_BACKBONE = VisionBackboneConfig(
    image_model_type="openai",
    image_default_input_size=(336, 336),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1024,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=23,
    image_head_dim=64,
    image_mlp_dim=4096,
    image_mlp_activations="quick_gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-5,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
)

SIGLIP_VISION_BACKBONE = VisionBackboneConfig(
    image_model_type="siglip",
    image_default_input_size=(378, 378),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1152,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=27,
    image_head_dim=72,
    image_mlp_dim=4304,
    image_mlp_activations="gelu_pytorch_tanh",
    image_dropout_rate=0.0,
    image_num_pos=729,
    image_norm_eps=1e-6,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
    resize_mode="siglip",
)

DINOV2_VISION_BACKBONE = VisionBackboneConfig(
    image_model_type="dino",
    image_default_input_size=(336, 336),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1024,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=24,
    image_head_dim=64,
    image_mlp_dim=4096,
    image_mlp_activations="gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-6,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
    resize_mode="dino",
)


# ============================================================================
# LLM configs (from original launch_scripts/utils.py)
# ============================================================================

OLMO_1024_PREVIEW = ModelConfig(
    d_model=4096,
    n_heads=32,
    n_kv_heads=None,
    clip_qkv=None,
    n_layers=32,
    mlp_ratio=4,
    mlp_hidden_size=22016,
    activation_type="swiglu",
    block_type="sequential",
    block_group_size=1,
    rope=True,
    rope_full_precision=True,
    rope_theta=500000,
    attention_dropout=0.0,
    attention_layer_norm=True,
    layer_norm_type="rms",
    layer_norm_with_affine=True,
    layer_norm_eps=1.0e-06,
    attention_layer_norm_with_affine=True,
    max_sequence_length=4096,
    include_bias=False,
    bias_for_layer_norm=False,
    scale_logits=False,
    vocab_size=100278,
    embedding_size=100352,
    additional_vocab_size=128,
    weight_tying=False,
    attention_type=AttentionType.sdpa,
    init_device="meta",
    init_fn="normal",
    init_std=0.02,
    init_cutoff_factor=3.0,
    precision="amp_bf16",
    norm_after=True,
    tokenizer=TokenizerConfig(identifier="allenai/dolma2-tokenizer"),
    embedding_dropout=0,
    image_pooling_2d="attention_meanq",
    image_padding_embed="pad_and_partial_pad",
)

QWEN2_7B = ModelConfig(
    vocab_size=152064,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=True,
    weight_tying=False,
    include_bias=False,
    embedding_size=152064,
    d_model=3584,
    mlp_hidden_size=18944 * 2,
    n_layers=28,
    additional_vocab_size=128,
    n_heads=28,
    n_kv_heads=4,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(identifier="Qwen/Qwen2-7B"),
    image_pooling_2d="attention_meanq",
    image_padding_embed="pad_and_partial_pad",
)

LLAMA_3_8B = ModelConfig(
    vocab_size=128256,
    additional_vocab_size=128,
    max_sequence_length=8192,
    residual_dropout=0.0,
    embedding_dropout=0.0,
    response_residual_dropout=0.0,
    attention_dropout=0.0,
    rope=True,
    qkv_bias=False,
    weight_tying=False,
    include_bias=False,
    embedding_size=128256,
    d_model=4096,
    mlp_hidden_size=28672,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    rope_theta=500000.0,
    layer_norm_eps=1e-5,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(identifier="meta-llama/Meta-Llama-3-8B"),
    image_pooling_2d="attention_meanq",
    image_padding_embed="pad_and_partial_pad",
    block_type="llama",
)


# ============================================================================
# Model registries
# ============================================================================

VISION_BACKBONES: Dict[str, VisionBackboneConfig] = {
    "openai": CLIP_VISION_BACKBONE,
    "siglip": SIGLIP_VISION_BACKBONE,
    "dinov2_large_336": DINOV2_VISION_BACKBONE,
}

LLMS: Dict[str, ModelConfig] = {
    "olmo_1024_preview": OLMO_1024_PREVIEW,
    "qwen2_7b": QWEN2_7B,
    "llama3_8b": LLAMA_3_8B,
}

VIT_HF_SOURCES = {
    "openai": "openai/clip-vit-large-patch14-336",
    "siglip": "google/siglip-so400m-patch14-384",
    "dinov2_large_336": "facebook/dinov2-large",
}

LLM_HF_SOURCES = {
    "olmo_1024_preview": "allenai/OLMo-7B-1024-preview",
    "qwen2_7b": "Qwen/Qwen2-7B",
    "llama3_8b": "meta-llama/Meta-Llama-3-8B",
}

# Output filenames (what config.yaml vit_load_path/llm_load_path expect)
OUTPUT_FILENAMES = {
    "openai": "pretrained/vit-l-14-336.pt",
    "siglip": "pretrained/siglip-so400m-14-384.pt",
    "dinov2_large_336": "pretrained/dinov2-large-336.pt",
    "olmo_1024_preview": "pretrained/olmo-1024-preview.pt",
    "qwen2_7b": "pretrained/qwen2-7b.pt",
    "llama3_8b": "pretrained/llama3-8b.pt",
}

ALL_MODELS = list(VISION_BACKBONES.keys()) + list(LLMS.keys())


# ============================================================================
# Conversion functions (from original scripts/convert_hf_to_molmo.py)
# ============================================================================

def interpolate_position_embeddings(
    position_embeddings: torch.Tensor,
    num_patches: int,
    dim: int,
    patch_size: int,
    height: int,
    width: int,
    num_prefix_tokens: int = 1,
) -> torch.Tensor:
    from torch import nn as torch_nn

    num_positions = position_embeddings.shape[1] - num_prefix_tokens
    if num_patches == num_positions and height == width:
        return position_embeddings
    class_pos_embed = position_embeddings[:, :num_prefix_tokens]
    patch_pos_embed = position_embeddings[:, num_prefix_tokens:]
    height = height // patch_size
    width = width // patch_size
    height, width = height + 0.1, width + 0.1
    patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
    target_dtype = patch_pos_embed.dtype
    patch_pos_embed = torch_nn.functional.interpolate(
        patch_pos_embed.to(dtype=torch.float32),
        scale_factor=(float(height / math.sqrt(num_positions)), float(width / math.sqrt(num_positions))),
        mode="bicubic",
        align_corners=False,
    ).to(dtype=target_dtype)
    if int(height) != patch_pos_embed.shape[-2] or int(width) != patch_pos_embed.shape[-1]:
        raise ValueError("Width or height does not match with the interpolated position embeddings")
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed, patch_pos_embed), dim=1)


def convert_state_dict_clip(state_dict, vision_config: VisionBackboneConfig) -> Dict[str, Any]:
    state_dict = _unflatten_dict(state_dict, sep=".")

    resblocks = {}
    for layer in range(vision_config.image_num_layers):
        layer_dict = state_dict["encoder"]["layers"][str(layer)]
        q, k, v, o = [
            layer_dict["self_attn"][f"{x}_proj"].pop("weight")
            for x in ["q", "k", "v", "out"]
        ]
        q_b, k_b, v_b, o_b = [
            layer_dict["self_attn"][f"{x}_proj"].pop("bias")
            for x in ["q", "k", "v", "out"]
        ]

        w1, w2 = [layer_dict["mlp"][f"{x}"].pop("weight") for x in ["fc1", "fc2"]]
        w1_b, w2_b = [layer_dict["mlp"][f"{x}"].pop("bias") for x in ["fc1", "fc2"]]

        mapped_layer_dict = {
            "attention": {
                "wq": dict(weight=q, bias=q_b),
                "wk": dict(weight=k, bias=k_b),
                "wv": dict(weight=v, bias=v_b),
                "wo": dict(weight=o, bias=o_b),
            },
            "feed_forward": {
                "w1": dict(weight=w1, bias=w1_b),
                "w2": dict(weight=w2, bias=w2_b),
            },
            "attention_norm": {
                "weight": layer_dict["layer_norm1"].pop("weight"),
                "bias": layer_dict["layer_norm1"].pop("bias"),
            },
            "ffn_norm": {
                "weight": layer_dict["layer_norm2"].pop("weight"),
                "bias": layer_dict["layer_norm2"].pop("bias"),
            }
        }
        resblocks[str(layer)] = mapped_layer_dict

    if str(vision_config.image_num_layers) in state_dict["encoder"]["layers"]:
        del state_dict["encoder"]["layers"][str(vision_config.image_num_layers)]

    height, width = vision_config.image_default_input_size
    num_patches = vision_config.image_num_pos - 1
    position_embedding = state_dict["embeddings"]["position_embedding"].pop("weight")
    position_embedding = interpolate_position_embeddings(
        position_embedding.unsqueeze(0), num_patches,
        position_embedding.shape[-1], vision_config.image_patch_size, height, width,
    )

    patch_embedding = state_dict["embeddings"]["patch_embedding"].pop(
        "weight"
    ).permute(0, 2, 3, 1).reshape(vision_config.image_emb_dim, -1)

    pre_ln = {
        "weight": state_dict["pre_layrnorm"].pop("weight"),
        "bias": state_dict["pre_layrnorm"].pop("bias"),
    }

    out = {
        "class_embedding": state_dict["embeddings"].pop("class_embedding"),
        "positional_embedding": position_embedding[0],
        "patch_embedding": dict(weight=patch_embedding),
        "pre_ln": pre_ln,
        "transformer": dict(resblocks=resblocks),
    }
    out = _flatten_dict(out, sep=".")
    del state_dict["post_layernorm"]
    for k in _flatten_dict(state_dict):
        raise ValueError("Unused parameter:", k)
    return out


def convert_state_dict_siglip(state_dict, vision_config: VisionBackboneConfig) -> Dict[str, Any]:
    state_dict = _unflatten_dict(state_dict, sep=".")

    resblocks = {}
    for layer in range(vision_config.image_num_layers):
        layer_dict = state_dict["encoder"]["layers"][str(layer)]
        q, k, v, o = [
            layer_dict["self_attn"][f"{x}_proj"].pop("weight")
            for x in ["q", "k", "v", "out"]
        ]
        q_b, k_b, v_b, o_b = [
            layer_dict["self_attn"][f"{x}_proj"].pop("bias")
            for x in ["q", "k", "v", "out"]
        ]

        w1, w2 = [layer_dict["mlp"][f"{x}"].pop("weight") for x in ["fc1", "fc2"]]
        w1_b, w2_b = [layer_dict["mlp"][f"{x}"].pop("bias") for x in ["fc1", "fc2"]]

        mapped_layer_dict = {
            "attention": {
                "wq": dict(weight=q, bias=q_b),
                "wk": dict(weight=k, bias=k_b),
                "wv": dict(weight=v, bias=v_b),
                "wo": dict(weight=o, bias=o_b),
            },
            "feed_forward": {
                "w1": dict(weight=w1, bias=w1_b),
                "w2": dict(weight=w2, bias=w2_b),
            },
            "attention_norm": {
                "weight": layer_dict["layer_norm1"].pop("weight"),
                "bias": layer_dict["layer_norm1"].pop("bias"),
            },
            "ffn_norm": {
                "weight": layer_dict["layer_norm2"].pop("weight"),
                "bias": layer_dict["layer_norm2"].pop("bias"),
            }
        }
        resblocks[str(layer)] = mapped_layer_dict

    height, width = vision_config.image_default_input_size
    num_patches = vision_config.image_num_pos
    position_embedding = state_dict["embeddings"]["position_embedding"].pop("weight")
    position_embedding = interpolate_position_embeddings(
        position_embedding.unsqueeze(0), num_patches,
        position_embedding.shape[-1], vision_config.image_patch_size, height, width,
        num_prefix_tokens=0,
    )

    patch_embedding = state_dict["embeddings"]["patch_embedding"].pop(
        "weight"
    ).permute(0, 2, 3, 1).reshape(vision_config.image_emb_dim, -1)
    patch_embedding_b = state_dict["embeddings"]["patch_embedding"].pop("bias")

    out = {
        "positional_embedding": position_embedding[0],
        "patch_embedding": dict(weight=patch_embedding, bias=patch_embedding_b),
        "transformer": dict(resblocks=resblocks),
    }
    out = _flatten_dict(out, sep=".")
    del state_dict["post_layernorm"]
    del state_dict["head"]
    for k in _flatten_dict(state_dict):
        raise ValueError("Unused parameter:", k)
    return out


def convert_state_dict_dino(state_dict, vision_config: VisionBackboneConfig) -> Dict[str, Any]:
    state_dict = _unflatten_dict(state_dict, sep=".")

    resblocks = {}
    for layer in range(vision_config.image_num_layers):
        layer_dict = state_dict["encoder"]["layer"][str(layer)]
        q, k, v = [
            layer_dict["attention"]["attention"][f"{x}"].pop("weight")
            for x in ["query", "key", "value"]
        ]
        q_b, k_b, v_b = [
            layer_dict["attention"]["attention"][f"{x}"].pop("bias")
            for x in ["query", "key", "value"]
        ]
        o = layer_dict["attention"]["output"]["dense"].pop("weight")
        o_b = layer_dict["attention"]["output"]["dense"].pop("bias")

        w1, w2 = [layer_dict["mlp"][f"{x}"].pop("weight") for x in ["fc1", "fc2"]]
        w1_b, w2_b = [layer_dict["mlp"][f"{x}"].pop("bias") for x in ["fc1", "fc2"]]

        mapped_layer_dict = {
            "attention": {
                "wq": dict(weight=q, bias=q_b),
                "wk": dict(weight=k, bias=k_b),
                "wv": dict(weight=v, bias=v_b),
                "wo": dict(weight=o, bias=o_b),
            },
            "feed_forward": {
                "w1": dict(weight=w1, bias=w1_b),
                "w2": dict(weight=w2, bias=w2_b),
            },
            "attention_norm": {
                "weight": layer_dict["norm1"].pop("weight"),
                "bias": layer_dict["norm1"].pop("bias"),
            },
            "ffn_norm": {
                "weight": layer_dict["norm2"].pop("weight"),
                "bias": layer_dict["norm2"].pop("bias"),
            },
            "lambda1": layer_dict["layer_scale1"].pop("lambda1"),
            "lambda2": layer_dict["layer_scale2"].pop("lambda1"),
        }
        resblocks[str(layer)] = mapped_layer_dict

    height, width = vision_config.image_default_input_size
    num_patches = vision_config.image_num_pos - 1
    position_embedding = state_dict["embeddings"].pop("position_embeddings")
    position_embedding = interpolate_position_embeddings(
        position_embedding, num_patches,
        position_embedding.shape[-1], vision_config.image_patch_size, height, width,
    )

    patch_embedding = state_dict["embeddings"]["patch_embeddings"]["projection"].pop(
        "weight"
    ).permute(0, 2, 3, 1).reshape(vision_config.image_emb_dim, -1)
    patch_embedding_b = state_dict["embeddings"]["patch_embeddings"]["projection"].pop("bias")

    out = {
        "class_embedding": state_dict["embeddings"].pop("cls_token").reshape(-1),
        "positional_embedding": position_embedding[0],
        "patch_embedding": dict(weight=patch_embedding, bias=patch_embedding_b),
        "transformer": dict(resblocks=resblocks),
    }
    out = _flatten_dict(out, sep=".")
    del state_dict["layernorm"]
    del state_dict["embeddings"]["mask_token"]
    for k in _flatten_dict(state_dict):
        raise ValueError("Unused parameter:", k)
    return out


def convert_state_dict_olmo_1024_preview(state_dict, config: ModelConfig, block_type: BlockType) -> Dict[str, Any]:
    state_dict = _unflatten_dict(state_dict, sep=".")
    assert len(state_dict) == 2
    lmhead = state_dict["lm_head"]
    state_dict = state_dict["model"]

    blocks = {}
    for layer in range(config.n_layers):
        layer_dict = state_dict["layers"][str(layer)]
        q, k, v, o = [layer_dict["self_attn"][f"{k}_proj"].pop("weight") for k in ["q", "k", "v", "o"]]
        mlp_gate = layer_dict["mlp"]["gate_proj"].pop("weight")
        mlp_up = layer_dict["mlp"]["up_proj"].pop("weight")
        mlp_down = layer_dict["mlp"]["down_proj"].pop("weight")

        assert block_type == BlockType.sequential

        mapped_layer_dict = {
            "ff_proj": {"weight": torch.cat([mlp_up, mlp_gate], 0)},
            "ff_out": {"weight": mlp_down},
            "ff_norm": {"weight": layer_dict["post_feedforward_layernorm"].pop("weight")},
            "attn_norm": {"weight": layer_dict["post_attention_layernorm"].pop("weight")},
            "att_proj": dict(weight=torch.cat((q, k, v), dim=0)),
            "attn_out": dict(weight=o),
            "q_norm": {"weight": layer_dict["self_attn"]["q_norm"].pop("weight")},
            "k_norm": {"weight": layer_dict["self_attn"]["k_norm"].pop("weight")},
        }
        blocks[str(layer)] = mapped_layer_dict

    out = _flatten_dict(dict(transformer=dict(blocks=blocks)), sep=".")
    assert list(lmhead) == ["weight"]
    out.update({
        "transformer.wte.embedding": state_dict["embed_tokens"].pop("weight"),
        "transformer.ln_f.weight": state_dict["norm"].pop("weight"),
        "transformer.ff_out.weight": lmhead.pop("weight"),
    })
    for k in _flatten_dict(state_dict):
        raise ValueError("Unused parameter:", k)
    return out


def convert_state_dict_qwen2(state_dict, config: ModelConfig, block_type: BlockType) -> Dict[str, Any]:
    """Also used for LLaMA 3 (identical structure)."""
    state_dict = _unflatten_dict(state_dict, sep=".")
    assert len(state_dict) == 2
    lmhead = state_dict["lm_head"]
    state_dict = state_dict["model"]

    blocks = {}
    for layer in range(config.n_layers):
        layer_dict = state_dict["layers"][str(layer)]
        q, k, v, o = [layer_dict["self_attn"][f"{k}_proj"].pop("weight") for k in ["q", "k", "v", "o"]]
        if config.qkv_bias:
            q_b, k_b, v_b = [layer_dict["self_attn"][f"{k}_proj"].pop("bias") for k in ["q", "k", "v"]]
        else:
            q_b, k_b, v_b = None, None, None

        mlp_gate = layer_dict["mlp"]["gate_proj"].pop("weight")
        mlp_up = layer_dict["mlp"]["up_proj"].pop("weight")
        mlp_down = layer_dict["mlp"]["down_proj"].pop("weight")

        if block_type == BlockType.llama:
            mapped_layer_dict = {
                "q_proj": dict(weight=q, bias=q_b),
                "k_proj": dict(weight=k, bias=k_b),
                "v_proj": dict(weight=v, bias=v_b),
                "attn_out": dict(weight=o),
                "ff_norm": {"weight": layer_dict["post_attention_layernorm"].pop("weight")},
                "attn_norm": {"weight": layer_dict["input_layernorm"].pop("weight")},
                "ff_proj1": dict(weight=mlp_gate),
                "ff_proj2": dict(weight=mlp_up),
                "ff_out": dict(weight=mlp_down),
            }
        elif block_type == BlockType.sequential:
            mapped_layer_dict = {
                "ff_proj": {"weight": torch.cat([mlp_up, mlp_gate], 0)},
                "ff_out": {"weight": mlp_down},
                "ff_norm": {"weight": layer_dict["post_attention_layernorm"].pop("weight")},
                "attn_norm": {"weight": layer_dict["input_layernorm"].pop("weight")},
                "att_proj": dict(
                    weight=torch.cat((q, k, v), dim=0),
                    bias=None if q_b is None else torch.cat((q_b, k_b, v_b), dim=0)
                ),
                "attn_out": dict(weight=o),
            }
        else:
            raise NotImplementedError(block_type)
        blocks[str(layer)] = mapped_layer_dict

    out = _flatten_dict(dict(transformer=dict(blocks=blocks)), sep=".")
    assert list(lmhead) == ["weight"]
    out.update({
        "transformer.wte.embedding": state_dict["embed_tokens"].pop("weight"),
        "transformer.ln_f.weight": state_dict["norm"].pop("weight"),
        "transformer.ff_out.weight": lmhead.pop("weight"),
    })
    for k in _flatten_dict(state_dict):
        raise ValueError("Unused parameter:", k)
    return out


CONVERT_FNS = {
    "openai": convert_state_dict_clip,
    "siglip": convert_state_dict_siglip,
    "dinov2_large_336": convert_state_dict_dino,
    "olmo_1024_preview": convert_state_dict_olmo_1024_preview,
    "qwen2_7b": convert_state_dict_qwen2,
    "llama3_8b": convert_state_dict_qwen2,  # LLaMA 3 has identical structure to Qwen2
}


# ============================================================================
# Main conversion logic
# ============================================================================

def convert_vit(model_name: str, output_dir: Path, cache_dir: str = None):
    """Download and convert a vision encoder."""
    hf_source = VIT_HF_SOURCES[model_name]
    v_cfg = VISION_BACKBONES[model_name]
    convert_fn = CONVERT_FNS[model_name]
    output_filename = OUTPUT_FILENAMES[model_name]

    output_path = output_dir / output_filename
    if output_path.exists():
        print(f"  Already exists: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {hf_source}...")
    model = AutoModel.from_pretrained(
        hf_source, torch_dtype=torch.float32, cache_dir=cache_dir,
    )
    if isinstance(model, (CLIPModel, SiglipModel)):
        model = model.vision_model

    state_dict = model.state_dict()
    del model

    print(f"  Converting to Molmo format...")
    converted = convert_fn(state_dict, v_cfg)
    del state_dict

    print(f"  Saving to {output_path}...")
    torch.save(converted, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Done ({size_mb:.0f} MB)")
    return output_path


def convert_llm(model_name: str, output_dir: Path, cache_dir: str = None):
    """Download and convert an LLM."""
    hf_source = LLM_HF_SOURCES[model_name]
    cfg = LLMS[model_name]
    convert_fn = CONVERT_FNS[model_name]
    output_filename = OUTPUT_FILENAMES[model_name]

    output_path = output_dir / output_filename
    if output_path.exists():
        print(f"  Already exists: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {hf_source}...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_source, torch_dtype=torch.float32,
        trust_remote_code=True, cache_dir=cache_dir,
    )

    state_dict = model.state_dict()
    del model

    print(f"  Converting to Molmo format...")
    converted = convert_fn(state_dict, cfg, cfg.block_type)
    del state_dict

    print(f"  Saving to {output_path}...")
    torch.save(converted, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Done ({size_mb:.0f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to Molmo format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  Vision encoders: openai, siglip, dinov2_large_336
  LLMs:            olmo_1024_preview, qwen2_7b, llama3_8b

Examples:
  python scripts/convert_pretrained.py openai --output-dir pretrained/
  python scripts/convert_pretrained.py --all --output-dir pretrained/
"""
    )
    parser.add_argument("model", nargs="?", help="Model to convert")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Base output directory (default: current dir)")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HuggingFace cache directory")
    parser.add_argument("--all", action="store_true",
                        help="Convert all models needed for reproduction")
    parser.add_argument("--list", action="store_true",
                        help="List available models")

    args = parser.parse_args()

    if args.list:
        print("Vision encoders:")
        for name, hf in VIT_HF_SOURCES.items():
            print(f"  {name:25s} -> {hf}")
        print("\nLLMs:")
        for name, hf in LLM_HF_SOURCES.items():
            print(f"  {name:25s} -> {hf}")
        print(f"\nOutput files (relative to --output-dir):")
        for name, path in OUTPUT_FILENAMES.items():
            print(f"  {name:25s} -> {path}")
        return

    output_dir = Path(args.output_dir)

    if args.all:
        models = ALL_MODELS
    elif args.model:
        models = [args.model]
    else:
        parser.print_help()
        return

    print("=" * 60)
    print("CONVERTING PRETRAINED MODELS TO MOLMO FORMAT")
    print("=" * 60)

    for model_name in models:
        print(f"\n[{model_name}]")
        if model_name in VISION_BACKBONES:
            convert_vit(model_name, output_dir, args.cache_dir)
        elif model_name in LLMS:
            convert_llm(model_name, output_dir, args.cache_dir)
        else:
            print(f"  Unknown model: {model_name}")
            print(f"  Available: {', '.join(ALL_MODELS)}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
