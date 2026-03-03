"""
Data utilities for LatentLens.

Provides image preprocessing and dataset loading utilities.

Derived from Molmo (https://github.com/allenai/molmo)
Copyright 2024 Allen Institute for AI
Licensed under the Apache License, Version 2.0
"""
import logging

import numpy as np
from torch.utils.data import DataLoader, DistributedSampler

from .preprocessor import (
    load_image,
    resize_and_pad,
    siglip_resize_and_pad,
    dino_resize_and_pad,
    MultiModalPreprocessor,
    Preprocessor,
)
from .collator import MMCollator
from .data_formatter import DataFormatter
from .dataset import DeterministicDataset
from .iterable_dataset_mixture import IterableDatasetMixture
from .pixmo_datasets import (
    PixMoCap, PixMoCapQa, PixMoCount, PixMoDocs, PixMoPoints,
    PixMoPointsLeftRight, PixMoPointsSpatial, PixMoPointsTopBottom,
    PixMoPointExplanations, PixMoAskModelAnything, PixMoPointsEval,
    TokenImageDataset, ColorImageDataset, ColorMosaicDataset,
)
from ..torch_util import get_global_rank, get_world_size

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


def build_torch_mm_eval_dataloader(
    batch_size, seed, model_config, data_config, pad_batches, max_steps=None
):
    preprocessor = build_mm_preprocessor(
        model_config, for_inference=data_config.for_inference, shuffle_messages=data_config.shuffle_messages,
        require_image_features=pad_batches
    )
    logging.info(f"Loading eval dataset: {data_config.dataset}/{data_config.split}")
    dataset = get_dataset_by_name(data_config.dataset, data_config.split)
    n_pad = 0
    if pad_batches:
        global_batch_size = batch_size * get_world_size()
        n_steps = (len(dataset) + global_batch_size - 1) // global_batch_size
        if max_steps:
            n_steps = min(n_steps, max_steps)
        if n_steps * global_batch_size > len(dataset):
            n_pad = (n_steps * global_batch_size) - len(dataset)

    dataset = DeterministicDataset(
        dataset=dataset,
        seed=seed,
        preprocessor=preprocessor,
        n_pad=n_pad
    )

    sampler = DistributedSampler(
        dataset,
        drop_last=data_config.drop_last,
        shuffle=data_config.shuffle,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=MMCollator(
            data_config.sequence_length,
            max_crops=model_config.get_max_crops(),
            include_metadata=True,
            pad=data_config.pad,
        ),
        num_workers=data_config.num_workers,
        sampler=sampler,
        pin_memory=data_config.pin_memory,
        prefetch_factor=None if data_config.num_workers == 0 else data_config.prefetch_factor,
        persistent_workers=False if data_config.num_workers == 0 else data_config.persistent_workers,
        timeout=data_config.timeout,
    )


def build_eval_dataloader(train_config, data_config, batch_size, max_steps=None):
    seed = data_config.seed if data_config.seed is not None else train_config.seed
    if data_config.multi_modal in ["torch"]:
        return build_torch_mm_eval_dataloader(
            batch_size, seed, train_config.model, data_config,
            pad_batches=train_config.fsdp is not None and not data_config.drop_last,
            max_steps=max_steps
        )
    else:
        raise NotImplementedError(data_config.multi_modal)


def build_train_dataloader(train_config, device=None):
    if device is None:
        device = "cpu"
    assert train_config.device_train_batch_size is not None
    seed = train_config.data.seed if train_config.data.seed is not None else train_config.seed
    log.info(f"Building train dataloader with seed: {seed}")
    data_config = train_config.data
    if train_config.data.multi_modal in ["torch", "torch_hf"]:
        preprocessor = build_mm_preprocessor(
            train_config.model, shuffle_messages=data_config.shuffle, is_training=True, require_image_features=True)
        if data_config.dataset:
            datasets = [get_dataset_by_name(
                data_config.dataset, data_config.split, config=train_config)]
            rates = [1]
        else:
            if data_config.mixture:
                mixture = {}
                for name, rate in data_config.mixture.items():
                    logging.info(f"Loading train dataset {name}/{data_config.split}")
                    mixture[name] = (get_dataset_by_name(name, data_config.split), rate)
            else:
                mixture = {}
                for root_size_mixture in data_config.root_size_mixture:
                    group_datasets = {}
                    for name, as_size in root_size_mixture.mixture.items():
                        logging.info(f"Loading train dataset {name}/{data_config.split}")
                        dataset = get_dataset_by_name(name, data_config.split)
                        if as_size is not None:
                            size = as_size
                        else:
                            size = len(dataset)
                        group_datasets[name] = (dataset, np.sqrt(size))
                    total_rate = sum(x[1] for x in group_datasets.values())
                    mixture.update({name: (ds, r/total_rate*root_size_mixture.rate)
                                     for name, (ds, r) in group_datasets.items()})

            total_rate = sum(x[1] for x in mixture.values())
            mixture = sorted(mixture.items(), key=lambda x: x[0])
            rates = [rate/total_rate for (_, (_, rate)) in mixture]
            datasets = [ds for (_, (ds, _)) in mixture]
            logging.info("Sampling rates:")
            names = list(x[0] for x in mixture)
            for ix in np.argsort(rates)[::-1]:
                logging.info(f"{names[ix]}: {100*rates[ix]:0.2f}")
        datasets = [DeterministicDataset(ds, preprocessor, data_config.seed) for ds in datasets]
        assert train_config.epoch == 0 or train_config.epoch is None

        dataset = IterableDatasetMixture(
            datasets=datasets,
            mixture_rates=rates,
            global_batch_size=train_config.global_train_batch_size,
            seed=data_config.seed,
            shuffle=data_config.shuffle,
        )
        return DataLoader(
            dataset,
            batch_size=train_config.device_train_batch_size,
            drop_last=train_config.data.drop_last,
            collate_fn=MMCollator(
                train_config.data.sequence_length, False,
                pad=data_config.pad, max_crops=train_config.model.get_max_crops()),
            num_workers=train_config.data.num_workers,
            pin_memory=train_config.data.pin_memory,
            prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
            persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
            timeout=train_config.data.timeout,
        )
    else:
        raise NotImplementedError(train_config.data.multi_modal)


def get_dataset_by_name(dataset_name, split, config=None):
    extra_args = config.data.extra_args if config and hasattr(config.data, 'extra_args') else {}

    if dataset_name == "token_image":
        return TokenImageDataset(split=split, config=config, **extra_args)
    elif dataset_name == "color_image":
        return ColorImageDataset(split=split, config=config, **extra_args)
    elif dataset_name == "color_mosaic":
        return ColorMosaicDataset(split=split, config=config, **extra_args)
    elif dataset_name == "android_control":
        return AndroidControl(split)
    elif dataset_name in ["scifi_document_qa", "pixmo_docs_other"]:
        return PixMoDocs("other", split=split)
    elif dataset_name in ["scifi_table_qa", "pixmo_docs_tables"]:
        return PixMoDocs("tables", split=split)
    elif dataset_name in ["scifi_diagram_qa", "pixmo_docs_diagrams"]:
        return PixMoDocs("diagrams", split=split)
    elif dataset_name in ["scifi_charts_qa", "pixmo_docs_charts"]:
        return PixMoDocs("charts", split=split)

    # PixMo-Pointing
    elif dataset_name in ["pointing_high_freq", "pixmo_points_high_freq"]:
        return PixMoPoints(kind="high_frequency", split=split, counting=False)
    elif dataset_name in ["point_count_high_freq", "pixmo_points_high_freq_counting"]:
        return PixMoPoints(kind="high_frequency", split=split, counting=True)
    elif dataset_name in ["pointing", "pixmo_points"]:
        return PixMoPoints(kind="basic", split=split, counting=False)
    elif dataset_name in ["point_count", "pixmo_points_counting"]:
        return PixMoPoints(kind="basic", split=split, counting=True)
    elif dataset_name in ["left_right", "pixmo_points_left_right"]:
        return PixMoPointsLeftRight(kind="basic", split=split)
    elif dataset_name in ["spatial", "pixmo_points_spatial"]:
        return PixMoPointsSpatial(kind="basic", split=split)
    elif dataset_name in ["top_bottom", "pixmo_points_top_bottom"]:
        return PixMoPointsTopBottom(kind="basic", split=split)

    # PixMo-Point-Explanations
    elif dataset_name in ["point_qa", "pixmo_pointing_explanations"]:
        return PixMoPointExplanations(split=split, split_groups=True)

    # PixMo-Count
    elif dataset_name in ["fast_flickr_count_qa_point_count", "pixmo_count_counting"]:
        return PixMoCount(split=split, counting=True)
    elif dataset_name in ["fast_flickr_count_qa_pointing", "pixmo_count"]:
        return PixMoCount(split=split, counting=False)

    # PixMo-AskModelAnything
    elif dataset_name in ["user_qa", "pixmo_ask_model_anything"]:
        return PixMoAskModelAnything(split=split)

    # PixMo-CapQa
    elif dataset_name in ["synthetic_qa_v3_as_user_qa", "pixmo_cap_qa"]:
        return PixMoCapQa(split=split)

    # PixMo-Cap
    if dataset_name in ["cockatoo_and_transcript_712k_sept6", "pixmo_cap_with_transcripts"]:
        return PixMoCap(split, mode="transcript_and_caption")
    if dataset_name in ["cockatoo_712k_sept6", "pixmo_cap"]:
        first_sentence_only = False if extra_args is None else extra_args.get("first_sentence_only", False)
        return PixMoCap(split, mode="captions", first_sentence_only=first_sentence_only)

    if dataset_name == "pointing_eval":
        assert split == "test"
        return PixMoPointsEval()

    # Academic datasets — lazy import (requires molmo.hf_datasets which
    # is only needed for evaluation, not for the 9 LatentLens training
    # configs which all use PixMoCap).
    from .academic_datasets import (
        ChartQa, ScienceQAImageOnly, TextVqa, OkVqa, DocQa,
        InfoQa, AOkVqa, Vqa2, PlotQa, FigureQa, DvQa, SceneTextQa,
        TabWMPDirectAnswer, AndroidControl, TallyQa, AI2D, CountBenchQa,
        RealWorldQa, MathVista, MMMU, ClockBench,
    )
    if dataset_name == "android_control":
        return AndroidControl(split)
    if dataset_name == "android_control_ll":
        return AndroidControl(split, mode="ll")
    if dataset_name == "chart_qa":
        return ChartQa(split, weighted=False)
    if dataset_name == "real_world_qa_no_instruction":
        assert split == "test"
        return RealWorldQa("no_instruction")
    if dataset_name == "chart_qa_weighted":
        return ChartQa(split, weighted=True)
    if dataset_name == "info_qa":
        return InfoQa(split)
    if dataset_name == "doc_qa":
        return DocQa(split)
    if dataset_name == "science_qa_img":
        return ScienceQAImageOnly(split)
    if dataset_name == "coco_2014_vqa_multi":
        return Vqa2(split, multi_question=True)
    if dataset_name == "coco_2014_vqa":
        return Vqa2(split, multi_question=False)
    if dataset_name == "text_vqa":
        return TextVqa(split)
    if dataset_name == "plot_qa":
        return PlotQa(split, in_memory=False)
    if dataset_name == "figure_qa":
        return FigureQa(dict(train="train", validation="validation1")[split])
    if dataset_name == "dv_qa":
        return DvQa(split, in_memory=False)
    if dataset_name == "okvqa":
        return OkVqa(split)
    if dataset_name in ["mmmu"]:
        return MMMU(split)
    if dataset_name in ["mmmu_test"]:
        return MMMU(split)
    if dataset_name == "a_okvqa_da":
        return AOkVqa(split=split, direct_answer=True)
    if dataset_name == "a_okvqa_mc":
        return AOkVqa(split=split, direct_answer=False)
    if dataset_name == "st_qa":
        return SceneTextQa(split=split)
    if dataset_name == "tabwmp_da":
        return TabWMPDirectAnswer(split=split, include_options=False)
    if dataset_name == "countbench_qa":
        assert split == "huggingface"
        return CountBenchQa()
    if dataset_name == "tally_qa":
        return TallyQa(split=split)
    if dataset_name == "ai2_diagram_v2_mix_transparent":
        return AI2D(split=split, boxes="both")
    if dataset_name == "clock_bench":
        return ClockBench(split=split)
    elif dataset_name == "math_vista_v2":
        if split == "validation":
            split = "testmini"
        return MathVista(split)
    raise NotImplementedError(dataset_name, split)


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
    "build_train_dataloader",
    "build_eval_dataloader",
    "get_dataset_by_name",
]
