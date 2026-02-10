"""
Data utilities for LatentLens.

Provides image preprocessing and dataset loading utilities.

Derived from Molmo (https://github.com/allenai/molmo)
Copyright 2024 Allen Institute for AI
Licensed under the Apache License, Version 2.0
"""
import logging

from .preprocessor import (
    load_image,
    resize_and_pad,
    siglip_resize_and_pad,
    dino_resize_and_pad,
    MultiModalPreprocessor,
    Preprocessor,
)
from .data_formatter import DataFormatter
from .pixmo_datasets import PixMoCap

log = logging.getLogger(__name__)


def build_mm_preprocessor(
    model_config,
    for_inference=False,
    shuffle_messages=True,
    is_training=False,
    require_image_features=False
):
    """Build a multimodal preprocessor from model config."""
    v_cfg = model_config.vision_backbone
    h, w = model_config.llm_patches_per_crop()
    if not model_config.image_padding_embed:
        image_padding_mask = None
    elif model_config.fix_image_padding:
        image_padding_mask = 2
    else:
        image_padding_mask = 1

    return Preprocessor(
        DataFormatter(
            prompt_templates=model_config.prompt_type,
            message_format=model_config.message_formatting,
            system_prompt=model_config.system_prompt_kind,
            always_start_with_space=model_config.always_start_with_space,
            default_inference_len=model_config.default_inference_len
        ),
        MultiModalPreprocessor(
            tokenizer=model_config.get_tokenizer(),
            normalize=str(v_cfg.image_model_type),
            crop_mode=model_config.crop_mode,
            max_crops=model_config.max_crops,
            overlap_margins=model_config.overlap_margins,
            resize=v_cfg.resize_mode,
            use_col_tokens=model_config.use_col_tokens,
            base_image_input_size=v_cfg.image_default_input_size,
            image_pooling_w=model_config.image_pooling_w,
            image_pooling_h=model_config.image_pooling_h,
            image_token_length_w=w,
            image_token_length_h=h,
            image_patch_size=v_cfg.image_patch_size,
            image_padding_mask=image_padding_mask,
            pad_value=model_config.pad_value,
            loss_token_weighting=model_config.multi_annotation_weighting,
        ),
        for_inference=for_inference,
        shuffle_messages=shuffle_messages,
        is_training=is_training,
        require_image_features=require_image_features,
    )


__all__ = [
    "load_image",
    "resize_and_pad",
    "siglip_resize_and_pad",
    "dino_resize_and_pad",
    "MultiModalPreprocessor",
    "Preprocessor",
    "DataFormatter",
    "PixMoCap",
    "build_mm_preprocessor",
]
