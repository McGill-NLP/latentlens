"""
Evaluation utilities for training.

Derived from Molmo (https://github.com/allenai/molmo)
Copyright 2024 Allen Institute for AI
Licensed under the Apache License, Version 2.0
"""
from typing import Dict, List, Union

import torch
from torchmetrics import MeanMetric, Metric

from .loss_evaluator import LossDatasetEvaluator
from ..config import DatasetEvaluatorConfig, TrainConfig
from ..data import build_eval_dataloader
from ..torch_util import get_world_size, get_global_rank

__all__ = [
    "build_evaluator",
    "build_loss_evaluators",
    "build_inf_evaluators",
    "InfDatasetEvaluator",
]


class InfDatasetEvaluator:
    """Stub for inference evaluator — not used in connector-only training configs.

    All 9 LatentLens training configs have ``inf_evaluators: []``, so
    this class is never instantiated.  It exists only so that type
    annotations in ``train.py`` resolve without importing the full
    (heavy-dependency) inference evaluation stack.
    """

    label: str = ""

    def evaluate_model(self, *args, **kwargs):
        raise NotImplementedError(
            "InfDatasetEvaluator is a stub in the release repo. "
            "The 9 LatentLens training configs do not use inference evaluators."
        )


def build_evaluator(
    train_config: TrainConfig, eval_config: DatasetEvaluatorConfig, tokenizer, device: torch.device
) -> LossDatasetEvaluator:
    eval_loader = build_eval_dataloader(
        train_config,
        eval_config.data,
        eval_config.device_eval_batch_size or train_config.device_eval_batch_size,
    )

    def make_metric():
        return MeanMetric(nan_strategy="error").to(device)

    eval_metric: Union[Metric, Dict[str, Metric]]
    eval_metric = dict(
        Loss=make_metric(),
        Accuracy=make_metric(),
        ZLoss=make_metric()
    )
    return LossDatasetEvaluator(
        label=eval_config.label,
        eval_loader=eval_loader,
        eval_metric=eval_metric,
        subset_num_batches=eval_config.subset_num_batches or train_config.eval_subset_num_batches,
    )


def build_loss_evaluators(cfg: TrainConfig, device: torch.device) -> List[LossDatasetEvaluator]:
    evaluators = []
    tokenizer = cfg.model.get_tokenizer()
    if len(set(x.label for x in cfg.evaluators)) != len(cfg.evaluators):
        raise ValueError("Non-unique labels in evaluators")
    for eval_cfg in cfg.evaluators:
        evaluators.append(build_evaluator(cfg, eval_cfg, tokenizer, device))
    return evaluators


def build_inf_evaluators(cfg: TrainConfig, device: torch.device) -> List[InfDatasetEvaluator]:
    """Build inference evaluators.

    All 9 LatentLens configs have ``inf_evaluators: []``, so this
    always returns an empty list in practice.
    """
    if cfg.inf_evaluators:
        raise NotImplementedError(
            "Inference evaluators are not supported in the release repo. "
            "All 9 LatentLens training configs use inf_evaluators: []."
        )
    return []
