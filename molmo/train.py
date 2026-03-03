from __future__ import annotations

import cProfile
import gc
import logging
import math
import os
import random
import shutil
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from pstats import SortKey
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
try:
    import wandb
    from wandb.sdk.data_types.base_types.wb_value import WBValue
except ImportError:
    wandb = None  # type: ignore[assignment]
    WBValue = object  # type: ignore[misc,assignment]

from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from torchmetrics import Metric

from .aliases import PathOrStr
from .checkpoint import Checkpointer, FullCheckpointer, build_sharded_checkpointer
from .config import (
    BlockType,
    CheckpointType,
    SchedulerUnits,
    ShardedCheckpointerType,
    SpeedMonitorConfig,
    TrainConfig, BatchDivisor,
)
from .data.iterable_dataset_mixture import IterableDatasetMixture
from .eval import InfDatasetEvaluator
from .exceptions import OLMoConfigurationError
from .model import Molmo
from .optim import Optimizer, Scheduler
from .torch_util import (
    barrier,
    gc_cuda,
    get_fs_local_rank,
    get_global_rank,
    get_world_size,
    move_to_device,
    peak_gpu_memory,
    synchronize_flag,
    synchronize_value, get_local_world_size, )
from .util import upload

try:
    from megablocks.layers.moe import (
        batched_load_balancing_loss,
        clear_load_balancing_loss,
        get_load_balancing_loss,
    )
except ImportError:
    pass

__all__ = ["SpeedMonitor", "LRMonitor", "Trainer"]

log = logging.getLogger(__name__)


@dataclass
class BatchStatsMonitor:
    max_window_size: int = 20
    sync_nodes: bool = True
    _batch_stats: Deque[Dict[str, float]] = field(default_factory=lambda: deque([]))

    def log_batch(self, batch):
        input_ids = batch["input_ids"]
        non_masked = (input_ids >= 0).to(dtype=torch.float32)
        stats = {
            "batch/non_masked_tokens": non_masked.sum(-1).mean(),
            "batch/per_non_masked_tokens": non_masked.mean(),
            "batch/examples_truncated": non_masked[:, -1].mean()
        }
        if "loss_masks" in batch:
            mask = (batch["loss_masks"] > 0).to(dtype=torch.float32)
            stats["batch/loss_tokens"] = mask.sum(-1).mean()
            stats["batch/per_loss_tokens"] = mask.mean()

        self._batch_stats.append(stats)
        if len(self._batch_stats) > self.max_window_size:
            self._batch_stats.popleft()

    def reset(self) -> None:
        self._batch_stats.clear()

    def check(self, device):
        stats = defaultdict(list)
        for batch in self._batch_stats:
            for k, v in batch.items():
                stats[k].append(v)

        out = {}
        for k, v in stats.items():
            v = torch.stack(v).mean()
            if self.sync_nodes:
                v = v.to(device)
                dist.all_reduce(v)
                v.div_(get_world_size())
            out[k] = v.item()
        return out


@dataclass
class SpeedMonitor:
    cfg: SpeedMonitorConfig
    global_total_tokens: int = 0
    stats: Deque[Tuple[float, int, int]] = field(default_factory=lambda: deque([]))

    def batch_start(self, global_total_tokens: int, device_batch_num_tokens: int, device_batch_num_loss_tokens: int, record: bool = True) -> None:
        self.global_total_tokens = global_total_tokens
        if record:
            if len(self.stats) >= self.cfg.window_size:
                self.stats.popleft()
            self.stats.append((
                time.monotonic(),
                device_batch_num_tokens,
                device_batch_num_loss_tokens
            ))

    def reset(self) -> None:
        self.stats.clear()

    def check(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {"throughput/total_tokens": self.global_total_tokens}
        if self.stats:
            interval_seconds = time.monotonic() - self.stats[0][0]
            interval_batches = len(self.stats)
            interval_tokens = sum(x[1] for x in self.stats)
            interval_loss_tokens = sum(x[2] for x in self.stats)
            metrics["throughput/device/loss_tokens_per_second"] = interval_loss_tokens / interval_seconds
            metrics["throughput/device/tokens_per_second"] = interval_tokens / interval_seconds
            metrics["throughput/device/batches_per_second"] = interval_batches / interval_seconds
        return metrics


@dataclass
class LRMonitor:
    optim: torch.optim.Optimizer

    def check(self) -> Dict[str, float]:
        lrs = [group["lr"] for group in self.optim.param_groups]
        return {f"optim/learning_rate_group{idx}": lr for idx, lr in enumerate(lrs)}


def cross_entropy_loss(
    logits, labels, ignore_index: int = -100, reduction: str = "mean", compute_z_loss: bool = False, z_loss_scale: float = 1e-4,
):
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

    if not compute_z_loss:
        return loss, None

    z_squared = logits.logsumexp(-1).pow(2)
    if reduction == "mean":
        z_squared = (z_squared * (labels != ignore_index)).mean()
    elif reduction == "sum":
        z_squared = (z_squared * (labels != ignore_index)).sum()

    z_loss = z_loss_scale * z_squared

    return loss, z_loss


@dataclass
class DatasetMetrics:
    label: str
    eval_loader: DataLoader
    eval_metric: Union[Metric, Dict[str, Metric], List[Metric]]
    subset_num_batches: Optional[int] = None

    def reset_metrics(self) -> None:
        if isinstance(self.eval_metric, Metric):
            self.eval_metric.reset()
        else:
            for metric in self.eval_metric.values():
                metric.reset()

    def compute_metrics(self) -> Dict[str, float]:
        return {f"{self.label}/{k}": v.compute().item() for k, v in self.eval_metric.items()}

    def update_metrics(
        self,
        batch: Dict[str, Any],
        eval_out: Dict[str, torch.Tensor],
    ) -> None:
        total_weight = eval_out["total_weight"]
        self.eval_metric["Loss"].update(eval_out["total_loss"]/total_weight, total_weight)
        self.eval_metric["Accuracy"].update(eval_out["total_accuracy"]/total_weight, total_weight)
        self.eval_metric["ZLoss"].update(eval_out["total_zloss"]/total_weight, total_weight)


@dataclass
class Trainer:
    cfg: TrainConfig
    model: Molmo
    fsdp_model: FSDP
    optim: Optimizer
    scheduler: Scheduler
    train_loader: DataLoader
    device: torch.device
    evaluators: List[DatasetMetrics]
    inference_evaluators: List[InfDatasetEvaluator]
    epoch: Optional[int] = None
    global_step: int = 0

    global_train_examples_seen_this_epoch: int = 0
    """Tracks the global number of training examples seen in the current epoch for the purpose of restoring
    the data loader position on restarts."""

    global_train_tokens_seen: int = 0
    """Tracks the global total number of tokens trained on."""

    checkpoints: List[Path] = field(default_factory=list)
    unsharded_checkpoints: List[Path] = field(default_factory=list)
    ephemeral_checkpoints: List[Path] = field(default_factory=list)
    min_train_loss: float = float("inf")
    cur_train_loss: float = float("inf")
    _start_time: float = field(default_factory=time.time)
    _last_step_time: float = field(default_factory=time.time)
    _step_times: Deque[float] = field(default_factory=lambda: deque(maxlen=50))  # Track last 50 step times
    _gc_init_state: bool = True
    _inference_warmup: bool = True
    loss_fn: Callable[..., torch.Tensor] = field(default_factory=lambda: cross_entropy_loss)  # type: ignore
    last_sharded_checkpoint_step: Optional[int] = None
    last_unsharded_checkpoint_step: Optional[int] = None
    _node_src: int = None
    _node_group: Any = None
    _node_group_ranks: Any = None

    # Add a class variable to hold the vocab_embeddings
    vocab_embeddings: Optional[torch.Tensor] = None

    text_avg: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.cfg.fused_loss:
            import flash_attn
            from flash_attn.ops.triton.cross_entropy import (  # type: ignore
                cross_entropy_loss,
            )

            # The `ignored_index` parameter of `cross_entropy_loss` was changed to `ignore_index` in v2.5.8 with commit https://github.com/Dao-AILab/flash-attention/commit/ec6d22143b5d375e253b2ebfc563b26a43f43684
            ce_loss_use_ignore_index_param = version.parse(flash_attn.__version__) >= version.parse("2.5.8")

            def fused_loss_fn(
                logits, labels, ignore_index: int = -100, reduction: str = "mean", compute_z_loss: bool = False
            ):
                if ce_loss_use_ignore_index_param:
                    ignore_index_kwarg = {"ignore_index": ignore_index}
                else:
                    ignore_index_kwarg = {"ignored_index": ignore_index}

                loss, z_loss = cross_entropy_loss(
                    logits,
                    labels,
                    label_smoothing=0.0,
                    logit_scale=1.0,
                    lse_square_scale=self.cfg.softmax_auxiliary_loss_scale if self.cfg.softmax_auxiliary_loss else 0.0,
                    inplace_backward=False,
                    process_group=None,
                    **ignore_index_kwarg,
                )

                mask = labels != ignore_index

                if reduction == "mean":
                    loss = loss.sum() / mask.sum()
                elif reduction == "sum":
                    loss = loss.sum()
                else:
                    loss = loss

                if not compute_z_loss:
                    return loss, None

                if reduction == "mean":
                    z_loss = z_loss.sum() / mask.sum()
                elif reduction == "sum":
                    z_loss = z_loss.sum()
                else:
                    z_loss = z_loss

                return loss, z_loss

            self.loss_fn = fused_loss_fn


        if self.model.config.block_type == BlockType.moe:
            from .config import config_to_moe_args

            self.moe_args = config_to_moe_args(self.cfg.model)            

        # Load vocab_embeddings from a .npy file
        if False:  # Soft-remove legacy vocab embeddings loading
            if Trainer.vocab_embeddings is None:
                vocab_embeddings_np = np.load('embedding_matrix_qwen2_7b.npy')
                Trainer.vocab_embeddings = torch.tensor(vocab_embeddings_np, device=self.device)
                avg_norm = torch.norm(Trainer.vocab_embeddings, dim=-1).mean().item()
                print(f"Vocabulary embeddings average norm: {avg_norm}")
            # Calculate the average textual embedding
            self.text_avg = Trainer.vocab_embeddings.mean(dim=0)
        else:
            # Initialize dummy values to avoid attribute errors
            if Trainer.vocab_embeddings is None:
                Trainer.vocab_embeddings = None
            self.text_avg = None

    @property
    def dataset(self) -> IterableDataset:
        return self.train_loader

    @property
    def tokens_per_batch(self) -> int:
        return self.cfg.global_train_batch_size * self.cfg.model.max_sequence_length

    @property
    def batches_per_epoch(self) -> int:
        return self.dataset.total_size // self.cfg.global_train_batch_size

    @property
    def max_epochs(self) -> int:
        if isinstance(self.cfg.max_duration, str) and self.cfg.max_duration.endswith("ep"):
            return int(self.cfg.max_duration[:-2].strip())
        else:
            return 1

    @property
    def max_steps(self) -> int:
        if isinstance(self.cfg.max_duration, int):
            return self.cfg.max_duration
        elif isinstance(self.cfg.max_duration, str):
            if self.cfg.max_duration.endswith("T"):
                # convert to float *first* to handle scientific notation
                max_tokens = int(float(self.cfg.max_duration[:-1].strip()))
                tokens_remaining = max(max_tokens - self.global_train_tokens_seen, 0)
                steps_remaining = tokens_remaining // self.tokens_per_batch
                return self.global_step + steps_remaining
            elif self.cfg.max_duration.endswith("ep"):
                max_epochs = int(self.cfg.max_duration[:-2].strip())
                return max_epochs * self.batches_per_epoch
            else:
                # convert to float *first* to handle scientific notation
                return int(float(self.cfg.max_duration))
        else:
            raise TypeError(f"expected int or str for 'max_duration', found {type(self.cfg.max_duration)}")

    @property
    def max_tokens(self) -> int:
        if isinstance(self.cfg.max_duration, int):
            return (
                self.global_train_tokens_seen
                + max(self.cfg.max_duration - self.global_step, 0) * self.tokens_per_batch
            )
        elif isinstance(self.cfg.max_duration, str):
            if self.cfg.max_duration.endswith("T"):
                # convert to float *first* to handle scientific notation
                return int(float(self.cfg.max_duration[:-1].strip()))
            elif self.cfg.max_duration.endswith("ep"):
                max_epochs = int(self.cfg.max_duration[:-2].strip())
                return max_epochs * self.batches_per_epoch * self.tokens_per_batch
            else:
                # convert to float *first* to handle scientific notation
                return (
                    self.global_train_tokens_seen
                    + max(int(float(self.cfg.max_duration)) - self.global_step, 0) * self.tokens_per_batch
                )
        else:
            raise TypeError(f"expected int or str for 'max_duration', found {type(self.cfg.max_duration)}")

    @property
    def scheduler_current(self) -> int:
        if self.cfg.scheduler.units == SchedulerUnits.steps:
            return self.global_step
        elif self.cfg.scheduler.units == SchedulerUnits.tokens:
            return self.global_train_tokens_seen
        else:
            raise NotImplementedError(self.cfg.scheduler.units)

    @property
    def scheduler_max(self) -> int:
        if self.cfg.scheduler.units == SchedulerUnits.steps:
            return self.max_steps
        elif self.cfg.scheduler.units == SchedulerUnits.tokens:
            return self.max_tokens
        else:
            raise NotImplementedError(self.cfg.scheduler.units)

    def trainer_state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "global_train_examples_seen_this_epoch": self.global_train_examples_seen_this_epoch,
            "global_train_tokens_seen": self.global_train_tokens_seen,
            "world_size": get_world_size(),
            "checkpoints": self.checkpoints,
            "unsharded_checkpoints": self.unsharded_checkpoints,
            "ephemeral_checkpoints": self.ephemeral_checkpoints,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
            },
        }

    def load_trainer_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Checkpoint paths.
        self.checkpoints = [
            path
            for path in state_dict["checkpoints"]
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]
        self.unsharded_checkpoints = [
            path
            for path in state_dict["unsharded_checkpoints"]
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]
        self.ephemeral_checkpoints = [
            path
            for path in state_dict.get("ephemeral_checkpoints", [])
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]

        # Dataset / dataloader position.
        checkpoint_epoch = state_dict.get("epoch", 0)
        self.global_step = state_dict["global_step"]
        self.global_train_examples_seen_this_epoch = state_dict.get(
            "global_train_examples_seen_this_epoch",
            state_dict.get(  # for backwards compatibility
                "global_train_examples_seen",
                state_dict.get("global_data_step", self.global_step) * self.cfg.global_train_batch_size,
            ),
        )
        self.global_train_tokens_seen = state_dict.get(
            "global_train_tokens_seen",
            state_dict.get("global_data_step", self.global_step)  # for backwards compatibility
            * self.cfg.global_train_batch_size
            * self.cfg.model.max_sequence_length,
        )

        if not self.cfg.restore_dataloader:
            self.epoch = 0
            self.global_train_tokens_seen = 0
            self.global_train_examples_seen_this_epoch = 0
        elif self.epoch is None:
            self.epoch = checkpoint_epoch
        elif checkpoint_epoch != self.epoch:
            log.info(f"Starting new epoch (epoch = {self.epoch})")
            self.global_train_examples_seen_this_epoch = 0

        if self.cfg.fast_forward_batches:
            log.info(f"Fast-forwarding data loader by {self.cfg.fast_forward_batches:,d} steps")
            # Technically we don't "see" these batches that we fast-forward through, but we use
            # this variable to update the position of the dataset so we need to include them here.
            self.global_train_examples_seen_this_epoch += (
                self.cfg.fast_forward_batches * self.cfg.global_train_batch_size
            )
            # NOTE: on the other hand we don't add anything to 'self.global_train_tokens_seen' here because
            # that variable is meant to track the actual number of tokens trained on.

        if self.global_train_examples_seen_this_epoch > 0:
            assert isinstance(self.dataset.dataset, IterableDatasetMixture)
            log.info(f"Data loader will start at instance index {self.global_train_examples_seen_this_epoch:,d}")
            self.dataset.dataset.start_index = self.global_train_examples_seen_this_epoch

        # Reset learning rate and weight decay to the values from the config, not the checkpoint.
        log.info("Resetting learning rate...")
        if self.cfg.model.vision_backbone is not None:
            initial_lr_dict = {
                "connector": self.cfg.optimizer.connector_learning_rate,
                "vit": self.cfg.optimizer.vit_learning_rate,
                "llm": self.cfg.optimizer.llm_learning_rate,
            }
            weight_decay_dict = {
                "connector": self.cfg.optimizer.connector_weight_decay,
                "vit": self.cfg.optimizer.vit_weight_decay,
                "llm": self.cfg.optimizer.llm_weight_decay,
            }
            for group in self.optim.param_groups:
                group_name = group["group_name"]
                component_name = group_name.split("_")[0]
                new_learning_rate = self.scheduler.get_lr(
                    initial_lr_dict[component_name],
                    self.scheduler_current,
                    self.scheduler_max,
                    group_name,
                )
                group["lr"] = new_learning_rate
                if "weight_decay" in group and group["weight_decay"] > 0.0:
                    group["weight_decay"] = weight_decay_dict[component_name]
        else:
            new_learning_rate = self.scheduler.get_lr(
                self.cfg.optimizer.learning_rate, self.scheduler_current, self.scheduler_max
            )
            for group in self.optim.param_groups:
                group["lr"] = new_learning_rate
                group["initial_lr"] = self.cfg.optimizer.learning_rate
                if "weight_decay" in group and group["weight_decay"] > 0.0:
                    group["weight_decay"] = self.cfg.optimizer.weight_decay

        # RNG states.
        if "rng" in state_dict and state_dict.get("world_size", get_world_size()) == get_world_size():
            log.info("Restoring RNG states...")
            rng_state = state_dict["rng"]
            self.restore_rng_state(rng_state)
        else:
            log.warning(
                "Trainer will not restore RNG states since the RNG states in the checkpoint are missing or invalid. "
                "This typically happens when restoring from an unsharded checkpoint or a checkpoint that was saved "
                "with a different world size. If that's the case you can safely ignore this warning."
            )

    def restore_rng_state(self, rng_state: Dict[str, Any]) -> None:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        torch.cuda.set_rng_state(rng_state["cuda"])

    def _save_checkpoint(
        self, checkpointer: Checkpointer, checkpoint_type: CheckpointType
    ) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        if checkpoint_type == CheckpointType.sharded:
            suffix = ""
            current_checkpoints = self.checkpoints
            link_latest = get_fs_local_rank() == 0
            num_checkpoints_to_keep = self.cfg.save_num_checkpoints_to_keep
        elif checkpoint_type == CheckpointType.unsharded:
            suffix = "-unsharded"
            current_checkpoints = self.unsharded_checkpoints
            link_latest = get_global_rank() == 0
            num_checkpoints_to_keep = self.cfg.save_num_unsharded_checkpoints_to_keep
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            suffix = ""
            current_checkpoints = self.ephemeral_checkpoints
            link_latest = get_fs_local_rank() == 0
            num_checkpoints_to_keep = 1
        else:
            raise NotImplementedError(checkpoint_type)

        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)

        checkpoint_dir = Path(self.cfg.save_folder) / f"step{self.global_step}{suffix}"
        remote_checkpoint_dir: Optional[str] = None
        if self.cfg.remote_save_folder is not None:
            remote_checkpoint_dir = f"{self.cfg.remote_save_folder.rstrip('/')}/{checkpoint_dir.name}"
        current_checkpoints.append(checkpoint_dir)

        # Save the checkpoint.
        try:
            checkpointer.save_checkpoint(
                checkpoint_dir,
                self.fsdp_model,
                self.optim,
                self.trainer_state_dict(),
                upload_to=remote_checkpoint_dir,
            )
        except FileExistsError:
            raise OLMoConfigurationError(
                f"Checkpoint for step {self.global_step} already exists, use --save-overwrite to overwrite it"
            )

        if link_latest:
            # Link to 'latest'.
            latest_path = Path(self.cfg.save_folder) / f"latest{suffix}"
            latest_path.unlink(missing_ok=True)
            try:
                latest_path.symlink_to(checkpoint_dir.name, target_is_directory=True)
            except FileExistsError:
                # Same as above, caught when another (file-system) local rank 0 has already made the 'latest' symlink.
                # This can happen when nodes are saving to a common NFS drive but otherwise have distinct
                # file-systems.
                if latest_path.resolve().name != checkpoint_dir.name:
                    raise

        # Save multimodal dataset checkpoint
        if self.cfg.save_dataloader_state:
            data_ckpt_fname = checkpoint_dir / f"rank{get_global_rank()}_data.bin"
            self.dataset.save(data_ckpt_fname)

        # Remove old checkpoints.
        if num_checkpoints_to_keep > 0:
            while len(current_checkpoints) > num_checkpoints_to_keep:
                self.remove_checkpoint(0, checkpoint_type)

        barrier()

        if remote_checkpoint_dir is not None:
            return remote_checkpoint_dir, checkpoint_dir
        else:
            return checkpoint_dir, None

    def save_sharded_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = build_sharded_checkpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.sharded)
        self.last_sharded_checkpoint_step = self.global_step
        return result

    def save_ephemeral_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = build_sharded_checkpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.sharded_ephemeral)
        self.last_sharded_checkpoint_step = self.global_step
        return result

    def _remove_sharded_checkpoint(self, idx: int, checkpoints: List[Path]):
        oldest_checkpoint = checkpoints.pop(idx)
        barrier()
        if get_fs_local_rank() == 0 and oldest_checkpoint.is_dir():
            shutil.rmtree(oldest_checkpoint, ignore_errors=True)
            latest_path = Path(self.cfg.save_folder) / "latest"
            if latest_path.resolve() == oldest_checkpoint.resolve():
                latest_path.unlink()
        barrier()

    def remove_sharded_checkpoint(self, idx: int = 0):
        self._remove_sharded_checkpoint(idx, self.checkpoints)

    def remove_ephemeral_checkpoint(self, idx: int = 0):
        self._remove_sharded_checkpoint(idx, self.ephemeral_checkpoints)

    def restore_sharded_checkpoint(
        self,
        load_path: PathOrStr,
        local_cache: Optional[PathOrStr] = None,
        *,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        sharded_checkpointer: Optional[ShardedCheckpointerType] = None,
    ):
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)
        checkpointer = build_sharded_checkpointer(self.cfg, name=sharded_checkpointer)
        trainer_state = checkpointer.restore_checkpoint(
            load_path,
            self.fsdp_model,
            self.optim,
            local_cache=local_cache,
            load_optimizer_state=load_optimizer_state,
        )
        if load_trainer_state:
            self.load_trainer_state_dict(trainer_state)
        barrier()

    def save_unsharded_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = FullCheckpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.unsharded)
        self.last_unsharded_checkpoint_step = self.global_step
        return result

    def remove_unsharded_checkpoint(self, idx: int = 0):
        barrier()
        oldest_checkpoint = self.unsharded_checkpoints.pop(idx)
        if get_global_rank() == 0 and oldest_checkpoint.is_dir():
            shutil.rmtree(oldest_checkpoint, ignore_errors=True)
            latest_path = Path(self.cfg.save_folder) / "latest-unsharded"
            if latest_path.resolve() == oldest_checkpoint.resolve():
                latest_path.unlink()
        barrier()

    def restore_unsharded_checkpoint(
        self,
        load_path: PathOrStr,
        local_cache: Optional[PathOrStr] = None,
        *,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
    ):
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)
        checkpointer = FullCheckpointer(self.cfg)
        trainer_state = checkpointer.restore_checkpoint(
            load_path,
            self.fsdp_model,
            self.optim,
            local_cache=local_cache,
            load_optimizer_state=load_optimizer_state,
        )
        if load_trainer_state:
            self.load_trainer_state_dict(trainer_state)
        barrier()

    def save_checkpoint(
        self, checkpoint_type: CheckpointType = CheckpointType.sharded
    ) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        result: Tuple[PathOrStr, Optional[PathOrStr]]
        if checkpoint_type == CheckpointType.sharded:
            result = self.save_sharded_checkpoint()
        elif checkpoint_type == CheckpointType.unsharded:
            result = self.save_unsharded_checkpoint()
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            result = self.save_ephemeral_checkpoint()
        else:
            raise NotImplementedError(checkpoint_type)

        gc_cuda()
        return result

    def restore_checkpoint(
        self,
        load_path: PathOrStr,
        *,
        checkpoint_type: Optional[CheckpointType] = None,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        load_dataloader_state: bool = True,
        sharded_checkpointer: Optional[ShardedCheckpointerType] = None,
    ):
        if checkpoint_type == CheckpointType.unsharded or (
            checkpoint_type is None and str(load_path).rstrip("/").endswith("-unsharded")
        ):
            self.restore_unsharded_checkpoint(
                load_path,
                local_cache=local_cache,
                load_optimizer_state=load_optimizer_state,
                load_trainer_state=load_trainer_state,
            )
        elif checkpoint_type == CheckpointType.sharded or checkpoint_type is None:
            self.restore_sharded_checkpoint(
                load_path,
                local_cache=local_cache,
                load_optimizer_state=load_optimizer_state,
                load_trainer_state=load_trainer_state,
                sharded_checkpointer=sharded_checkpointer,
            )
        elif checkpoint_type is not None:
            raise NotImplementedError(checkpoint_type)

        if load_dataloader_state:
            # Restore multimodal dataset checkpoint
            logging.info("Loading dataloader state...")
            data_ckpt_fname = os.path.join(load_path, f"rank{get_global_rank()}_data.bin")
            self.dataset.restore(data_ckpt_fname)
            logging.info("Done")

        gc_cuda()

    def remove_checkpoint(self, idx: int = 0, checkpoint_type: CheckpointType = CheckpointType.sharded):
        if checkpoint_type == CheckpointType.sharded:
            self.remove_sharded_checkpoint(idx=idx)
        elif checkpoint_type == CheckpointType.unsharded:
            self.remove_unsharded_checkpoint(idx=idx)
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            self.remove_ephemeral_checkpoint(idx=idx)
        else:
            raise NotImplementedError(checkpoint_type)

    def move_to_device(self, batch, device):
        return move_to_device(batch, device)

    def get_labels(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Labels are just input IDs shifted to the left (first item is ignored).
        labels, label_mask, attention_mask, instance_mask = (
            batch["input_ids"].clone(),
            batch.get("label_mask"),
            batch.get("attention_mask"),
            batch.get("instance_mask"),
        )
        if label_mask is not None:
            labels.masked_fill_(~label_mask, -100)
        if attention_mask is not None:
            labels.masked_fill_(attention_mask == 0.0, -100)
        if instance_mask is not None:
            labels.masked_fill_(~instance_mask.unsqueeze(-1), value=-100)
        return labels[..., 1:].contiguous()

    def model_forward(
        self, batch: Dict[str, Any], loss_reduction: str = "mean", compute_z_loss: bool = False, return_logit_lenses: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        # shape: (batch_size, seq_len, vocab_size)
        with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
            output = self.fsdp_model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                attention_bias=batch.get("attention_bias"),
                response_mask=(batch["loss_masks"] > 0) if "loss_masks" in batch else None,
                images=batch.get("images"),
                image_masks=batch.get("image_masks"),
                image_input_idx=batch.get("image_input_idx"),
                subsegment_ids=batch.get("subsegment_ids"),
                position_ids=batch.get("position_ids"),
                return_logit_lenses=return_logit_lenses,
                output_hidden_states=return_logit_lenses,  # Add this line to ensure we collect hidden states
                loss_masks=batch.get("loss_masks")
            )
            logits = output.logits
            if return_logit_lenses:
                accuracies_per_layer = output.accuracies_per_layer
            else:
                accuracies_per_layer = None
        if "labels" in batch:
            assert "loss_masks" in batch
            assert loss_reduction == "none"
            loss_masks = batch["loss_masks"] * (batch["loss_masks"] > 0)
            labels = batch["labels"].long()
            labels.masked_fill_(~(loss_masks > 0), -100)
            labels = labels.view(-1)
            logits_for_loss = logits.to(torch.float32).view(-1, logits.size(-1)) # for numerical stability
        else:
            logits_for_loss = logits[..., :-1, :].contiguous()
            # shape: (batch_size * seq_len, vocab_size)
            logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
            # shape: (batch_size, seq_len)
            labels = self.get_labels(batch)
            # shape: (batch_size * seq_len,)
            labels = labels.view(-1)
        ce_loss, z_loss = self.loss_fn(
            logits_for_loss, labels, ignore_index=-100, reduction=loss_reduction,
            compute_z_loss=compute_z_loss, z_loss_scale=self.cfg.softmax_auxiliary_loss_scale,
        )
        bs = batch["input_ids"].shape[0]
        if loss_reduction == "none":
            # Reshape (batch_size * seq_len,) -> (batch_size, seq_len)
            ce_loss = ce_loss.view(bs, -1)
            if z_loss is not None:
                z_loss = z_loss.view(bs, -1)

        accuracy = torch.argmax(logits_for_loss, dim=-1) == labels
        
        if "labels" in batch:
            ce_loss = ce_loss * loss_masks
            if z_loss is not None:
                z_loss = z_loss * loss_masks
            accuracy = accuracy.view(bs, -1)
            accuracy = accuracy * loss_masks
        else:
            accuracy = (accuracy * (labels >= 0))
            accuracy = accuracy.view(bs, -1)
        return accuracy, ce_loss, z_loss, logits, accuracies_per_layer

    def train_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], List[float], Optional[List[float]]]:
        micro_batches = self.split_batch(batch)
        has_labels = "labels" in batch

        if has_labels:
            loss_masks = batch["loss_masks"] * (batch["loss_masks"] > 0)
            if self.cfg.batch_divisor == BatchDivisor.global_batch:
                batch_size_in_tokens = loss_masks.sum()
                dist.all_reduce(batch_size_in_tokens)
                batch_size_in_tokens.div_(get_world_size())
            elif self.cfg.batch_divisor == BatchDivisor.device_batch:
                batch_size_in_tokens = loss_masks.sum()
            else:
                raise ValueError()
        else:
            batch_size_in_tokens = batch["input_ids"].numel()

        del batch

        ce_batch_loss = torch.tensor(0.0, device=self.device)
        batch_accuracy = torch.tensor(0.0, device=self.device)
        z_batch_loss = None if not self.cfg.softmax_auxiliary_loss else torch.tensor(0.0, device=self.device)
        lb_batch_loss = None if self.model.config.block_type != BlockType.moe else torch.tensor(0.0, device=self.device)
        moe_z_batch_loss = None if not self.model.config.moe_zloss_weight else torch.tensor(0.0, device=self.device)
        expert_assignments = (
            None if self.model.config.block_type != BlockType.moe or not self.model.config.moe_log_expert_assignment
            else torch.zeros((self.model.config.n_layers, self.model.config.moe_num_experts))
        )

        summed_correct_per_pos = None
        summed_count_per_pos = None

        for micro_batch in micro_batches:
            # Determine if this is a metric logging step
            return_logit_lenses = self.global_step % self.cfg.wandb.log_interval == 0

            # Call the model forward function with return_logit_lenses=True during logging steps
            accuracy, ce_loss, z_loss, logits, accuracy_per_layer = self.model_forward(
                micro_batch, compute_z_loss=self.cfg.softmax_auxiliary_loss, loss_reduction="none" if has_labels else "sum", return_logit_lenses=return_logit_lenses
            )

            if has_labels:
                labels = micro_batch["labels"]
                mask = labels != -100  # [B, T]
                correct = accuracy.bool() & mask  # [B, T]
                
                correct_per_pos = correct.sum(0).float()  # [T]
                count_per_pos = mask.sum(0).float()       # [T]

                if summed_correct_per_pos is None:
                    summed_correct_per_pos = correct_per_pos
                    summed_count_per_pos = count_per_pos
                else:
                    summed_correct_per_pos += correct_per_pos
                    summed_count_per_pos += count_per_pos

                ce_loss_sum = ce_loss.sum(0)  # [T]
                if z_loss is not None:
                    z_loss_sum = z_loss.sum(0)

                ce_loss = ce_loss_sum.sum()
                accuracy = correct.sum()
                if z_loss is not None:
                    z_loss = z_loss_sum.sum()

            ce_loss = ce_loss / batch_size_in_tokens
            accuracy = accuracy / batch_size_in_tokens

            ce_batch_loss += ce_loss.detach()
            batch_accuracy += accuracy.detach()

            if self.cfg.softmax_auxiliary_loss:
                assert z_loss is not None
                assert z_batch_loss is not None
                z_loss = z_loss / batch_size_in_tokens
                loss = ce_loss + z_loss
                z_batch_loss += z_loss.detach()
            else:
                loss = ce_loss

            del logits

            if self.model.config.block_type == BlockType.moe:
                if self.model.config.moe_zloss_weight:
                    lb_loss, moe_z_loss = batched_load_balancing_loss(self.moe_args)
                    lb_loss = lb_loss / len(micro_batches)
                    moe_z_loss = moe_z_loss / len(micro_batches)
                elif self.model.config.moe_loss_weight:
                    lb_loss = batched_load_balancing_loss(self.moe_args) / len(micro_batches)
                if self.model.config.moe_log_expert_assignment:
                    if self.model.config.moe_zloss_weight:
                        tokens_per_expert, _, _ = zip(*get_load_balancing_loss())
                    else:
                        tokens_per_expert, _ = zip(*get_load_balancing_loss())
                    expert_assignments += torch.stack(tokens_per_expert, dim=0).cpu()
                clear_load_balancing_loss()
                if self.model.config.moe_loss_weight:
                    loss += lb_loss
                    lb_batch_loss += lb_loss.detach()
                if self.model.config.moe_zloss_weight:
                    loss += moe_z_loss
                    moe_z_batch_loss += moe_z_loss.detach()

            loss.backward()

        dist.all_reduce(summed_correct_per_pos)
        dist.all_reduce(summed_count_per_pos)
        if accuracy_per_layer is not None:
            # accuracy_per_layer shape: (n_layers, ?), with variable second dim
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, accuracy_per_layer.cpu())

            # Gathered is a list of (n_layers, ?)
            # Concatenate along dim=1, then take mean per layer
            max_layers = gathered[0].size(0)
            for t in gathered:
                assert t.size(0) == max_layers, "Mismatch in n_layers"

            per_layer_accuracy = torch.cat(gathered, dim=1).mean(dim=1).to(accuracy_per_layer.device)
        else:
            per_layer_accuracy = None


        per_position_accuracy = [
            correct.item() / count.item()
            for correct, count in zip(summed_correct_per_pos, summed_count_per_pos)
            if count.item() > 0
        ]


        return ce_batch_loss, z_batch_loss, batch_accuracy, lb_batch_loss, moe_z_batch_loss, expert_assignments, per_position_accuracy, per_layer_accuracy

    def compute_nearest_neighbor_alignment(self, visual_embeddings, token_ids, vocab_embeddings, step, k=5):
        if False:  # Soft-remove nearest neighbor alignment computation
            print(f"TRACE: visual_embeddings.shape: {visual_embeddings.shape}")
            print(f"TRACE: token_ids.shape: {token_ids.shape}")
            print(f"TRACE: vocab_embeddings.shape: {vocab_embeddings.shape}")
            print(f"TRACE: token_ids: {token_ids}")
            B, T, D = visual_embeddings.shape
            V = vocab_embeddings.shape[0]

            vocab_embeddings_norm = F.normalize(vocab_embeddings, p=2, dim=1)

            def compute(similarities, token_ids_expanded, prefix):
                correct_sim = similarities[torch.arange(similarities.size(0)), token_ids_expanded]
                ranks = (similarities >= correct_sim.unsqueeze(1)).sum(dim=1)
                top1 = (ranks == 1).float().mean()
                top5 = (ranks <= k).float().mean()
                mean_rank = ranks.float().mean()
                mrr = (1.0 / ranks.float()).mean()

                # Aggregate metrics across GPUs
                for tensor in [top1, top5, mean_rank, mrr]:
                    dist.all_reduce(tensor)
                    tensor /= get_world_size()

                return {
                    f"{prefix}/top1_accuracy": top1.item(),
                    f"{prefix}/top5_accuracy": top5.item(),
                    f"{prefix}/mean_rank": mean_rank.item(),
                    f"{prefix}/mrr": mrr.item(),
                }

            pooled = visual_embeddings.mean(dim=1)  # [B, D]
            pooled_norm = F.normalize(pooled, p=2, dim=1)
            sim_pooled = pooled_norm @ vocab_embeddings_norm.T  # [B, V]
            metrics_pooled = compute(sim_pooled, token_ids, "nn_alignment_pooled")

            flat = visual_embeddings.view(B * T, D)
            flat_norm = F.normalize(flat, p=2, dim=1)
            sim_all = flat_norm @ vocab_embeddings_norm.T  # [B*T, V]
            token_ids_expanded = token_ids.repeat_interleave(T)  # [B*T]
            metrics_all = compute(sim_all, token_ids_expanded, "nn_alignment_all_tokens")

            def compute_l2_metrics(embeddings, token_ids_expanded, prefix):
                # Compute L2 distances
                l2_distances = torch.cdist(embeddings, vocab_embeddings)
                # Extract the L2 distance for the correct token
                correct_l2 = l2_distances[torch.arange(l2_distances.size(0)), token_ids_expanded]
                # Calculate ranks based on L2 distance
                ranks = (l2_distances <= correct_l2.unsqueeze(1)).sum(dim=1)
                top1 = (ranks == 1).float().mean()
                top5 = (ranks <= k).float().mean()
                mean_rank = ranks.float().mean()
                mrr = (1.0 / ranks.float()).mean()

                # Aggregate metrics across GPUs
                for tensor in [top1, top5, mean_rank, mrr]:
                    dist.all_reduce(tensor)
                    tensor /= get_world_size()

                return {
                    f"{prefix}/top1_accuracy": top1.item(),
                    f"{prefix}/top5_accuracy": top5.item(),
                    f"{prefix}/mean_rank": mean_rank.item(),
                    f"{prefix}/mrr": mrr.item(),
                }

            # Compute L2 metrics for pooled embeddings
            metrics_pooled_l2 = compute_l2_metrics(pooled, token_ids, "nn_alignment_pooled_l2")

            # Compute L2 metrics for all tokens
            metrics_all_l2 = compute_l2_metrics(flat, token_ids_expanded, "nn_alignment_all_tokens_l2")

            # Combine all metrics
            metrics = {**metrics_pooled, **metrics_all, **metrics_pooled_l2, **metrics_all_l2}

            if step % 1000 == 0 and B > 0:
                for i in range(min(5, B)):
                    top_indices = sim_pooled[i].topk(5).indices.tolist()
                    log.info(f"NN Sample {i} (true token: {token_ids[i]}): Top matches: {top_indices}")

            return metrics
        else:
            # Return empty metrics when disabled
            return {}

    def evaluate_nn_alignment(self, batch):
        """
        Run a forward pass to extract visual embeddings and compute nearest neighbor metrics.
        
        Args:
            batch: A batch of data containing images and token IDs
            
        Returns:
            Dictionary of metrics
        """
        if False:  # Soft-remove nearest neighbor alignment evaluation
            if "token_id" not in batch:
                available_keys = list(batch.keys())
                error_msg = f"token_id missing from batch. Available keys: {available_keys}"
                print(error_msg)
                return {}
            
            print(f"Performing nearest neighbor alignment evaluation on batch with {batch['token_id'].shape[0]} examples")
            
            # Move batch to device if needed
            batch = self.move_to_device(batch, self.device)
            
            # Set model to eval mode temporarily
            was_training = self.fsdp_model.training
            self.fsdp_model.eval()
            
            # Extract visual embeddings with a forward pass
            with torch.no_grad():
                # Get the model's vocabulary embedding matrix
                vocab_embeddings = Trainer.vocab_embeddings
                
                # Run the forward pass with return_visual_embeddings=True to get visual embeddings directly
                output = self.fsdp_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    attention_bias=batch.get("attention_bias"),
                    response_mask=(batch["loss_masks"] > 0) if "loss_masks" in batch else None,
                    images=batch.get("images"),
                    image_masks=batch.get("image_masks"),
                    image_input_idx=batch.get("image_input_idx"),
                    subsegment_ids=batch.get("subsegment_ids"),
                    position_ids=batch.get("position_ids"),
                    return_visual_embeddings=True,
                    loss_masks=batch.get("loss_masks")
                )
                
                # Get the visual embeddings
                visual_embeddings = output.visual_embeddings
                
                if visual_embeddings is None:
                    error_msg = "CRITICAL ERROR: Model returned None for visual_embeddings. Check the model forward pass."
                    log.error(error_msg)
                    if self.global_step > 0:
                        raise ValueError(error_msg)
                    return {}
            
            # Restore model to its previous mode
            self.fsdp_model.train(was_training)
            
            # Compute nearest neighbor metrics using the extracted token IDs
            metrics = self.compute_nearest_neighbor_alignment(
                visual_embeddings, batch["token_id"], vocab_embeddings, self.global_step
            )
            
            log.info(f"Nearest neighbor alignment metrics: {metrics}")
            return metrics
        else:
            # Return empty metrics when disabled
            return {}

    def train_step(self, batch: Dict[str, Any], reduce_global_loss: bool = True) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # Record how many instances are going to be skipped (masked out).
        if (instance_mask := batch.get("instance_mask")) is not None:
            metrics["train/masked_instances_local_rank"] = (~instance_mask).sum().item()

        # Zero-gradients.
        self.optim.zero_grad(set_to_none=True)


        # Move tensors to the right device.
        batch = self.move_to_device(batch, self.device)

        # Run forward-backward pass.
        ce_batch_loss, z_batch_loss, batch_accuracy, lb_batch_loss, moe_z_batch_loss, expert_assignments, per_position_accuracy, per_layer_accuracy = self.train_batch(batch)

        # Check if we should run nearest neighbor evaluation
        if False:  # Soft-remove nearest neighbor evaluation check
            should_run_nn_eval = batch.get("token_id") is not None and hasattr(self.cfg, 'expensive_logging_interval') and self.global_step % self.cfg.expensive_logging_interval == 0

            # Validate required keys for nearest neighbor evaluation
            if should_run_nn_eval:
                if "images" not in batch or batch["images"] is None:
                    log.warning(f"Step {self.global_step}: Cannot run nearest neighbor evaluation - batch is missing 'images'")
                    should_run_nn_eval = False
                elif "token_id" not in batch:
                    log.error(f"Step {self.global_step}: Cannot run nearest neighbor evaluation - CRITICAL ERROR - batch is missing 'token_id'. Available keys: {list(batch.keys())}")
                    # Only raise an error after some initial steps to allow for warmup
                    if self.global_step > 10:
                        raise ValueError(f"CRITICAL ERROR: token_id missing from batch during nearest neighbor evaluation at step {self.global_step}")
                    should_run_nn_eval = False
                else:
                    log.info(f"Will run nearest neighbor evaluation at step {self.global_step} with {batch['token_id'].shape[0]} examples")

            # Compute nearest neighbor alignment metrics directly from batch if applicable
            if should_run_nn_eval:
                nn_metrics = self.evaluate_nn_alignment(batch)
                metrics.update(nn_metrics)

        # Collect loss, potentially reducing over all ranks.
        if reduce_global_loss:
            dist.reduce(ce_batch_loss, 0)
            ce_batch_loss.div_(get_world_size())
            if z_batch_loss is not None:
                dist.reduce(z_batch_loss, 0)
                z_batch_loss.div_(get_world_size())
            if batch_accuracy is not None:
                dist.reduce(batch_accuracy, 0)
                batch_accuracy.div_(get_world_size())
            if lb_batch_loss is not None:
                dist.reduce(lb_batch_loss, 0)
                lb_batch_loss.div_(get_world_size())
            if moe_z_batch_loss is not None:
                dist.reduce(moe_z_batch_loss, 0)
                moe_z_batch_loss.div_(get_world_size())

        # Clip gradient norms and collect param/gradient/optim metrics.
        should_log_optim_metrics_this_step = self.should_log_optim_metrics_this_step()
        optim_metrics = self.optim.clip_grads_and_collect_metrics(
            self.global_step,
            collect_param_metrics=should_log_optim_metrics_this_step,
            # passing this process group here ensures metrics are reduced correctly when we're using
            # HYBRID sharding.
            process_group=self.fsdp_model.process_group,
            multi_modal=self.cfg.model.vision_backbone is not None,
        )

        # Adjust the learning rate.
        if self.cfg.model.vision_backbone is not None:
            initial_lr_dict = {
                "connector": self.cfg.optimizer.connector_learning_rate,
                "vit": self.cfg.optimizer.vit_learning_rate,
                "llm": self.cfg.optimizer.llm_learning_rate,
            }
            for group in self.optim.param_groups:
                group_name = group["group_name"]
                component_name = group_name.split("_")[0]
                group["lr"] = self.scheduler.get_lr(
                    initial_lr_dict[component_name],
                    self.scheduler_current,
                    self.scheduler_max,
                    group_name,
                )
                group["max_grad_norm"] = self.scheduler.get_max_grad_norm(
                    self.cfg.max_grad_norm, self.scheduler_current, self.scheduler_max
                )
                group["max_grad_norm_ratio"] = self.scheduler.get_max_grad_norm(
                    self.cfg.max_grad_norm_ratio, self.scheduler_current, self.scheduler_max
                )
        else:
            for group in self.optim.param_groups:
                # TODO (epwalsh): if we want to enable different LRs or gradient clipping settings per group
                # we should pass `group["initial_lr"]` or `group["initial_max_grad_norm"]` here instead of
                # the corresponding values from `self.cfg`.
                group["lr"] = self.scheduler.get_lr(
                    self.cfg.optimizer.learning_rate, self.scheduler_current, self.scheduler_max
                )
                group["max_grad_norm"] = self.scheduler.get_max_grad_norm(
                    self.cfg.max_grad_norm, self.scheduler_current, self.scheduler_max
                )
                group["max_grad_norm_ratio"] = self.scheduler.get_max_grad_norm(
                    self.cfg.max_grad_norm_ratio, self.scheduler_current, self.scheduler_max
                )

        # Optimizer step.
        self.optim.step()

        # Collect metrics and check for NaN loss.
        # NOTE: this involves a bunch of host-device syncs so we wait until the last moment to do this.
        if torch.isnan(ce_batch_loss):
            raise ValueError("nan loss encountered")
        if z_batch_loss is not None and torch.isnan(z_batch_loss):
            raise ValueError("nan loss encountered")
        for key, value in optim_metrics.items():
            metrics[f"optim/{key}"] = value.item()
        self.cur_train_loss = ce_batch_loss.item()
        self.min_train_loss = min(self.min_train_loss, self.cur_train_loss)
        metrics["train/CrossEntropyLoss"] = self.cur_train_loss
        metrics["train/Perplexity"] = math.exp(self.cur_train_loss)
        metrics["train/Accuracy"] = batch_accuracy.item()
        if z_batch_loss is not None:
            metrics["train/ZLoss"] = z_batch_loss.item()
        if lb_batch_loss is not None:
            metrics["train/LoadBalancingLoss"] = lb_batch_loss.item()
            # Log assignment metrics.
            if expert_assignments is not None:
                for layer_idx, expert_assignments_layer in enumerate(expert_assignments):
                    total_tokens = expert_assignments_layer.sum().item()
                    for expert_idx, expert_assignment in enumerate(expert_assignments_layer):
                        metrics[f"train/TokensPercentage/layer{layer_idx}/expert{expert_idx}"] = (
                            expert_assignment.item() / total_tokens
                        ) * 100
                        metrics[
                            f"train/TokensTotal/layer{layer_idx}/expert{expert_idx}"
                        ] = expert_assignment.item()
        if moe_z_batch_loss is not None:
            metrics["train/MoEZLoss"] = moe_z_batch_loss.item()

        # Maybe collect post-step optimizer-specific metrics.
        if should_log_optim_metrics_this_step:
            optim_metrics = self.optim.get_post_step_metrics(
                self.fsdp_model, process_group=self.fsdp_model.process_group
            )
            for key, value in optim_metrics.items():
                metrics[f"optim/{key}"] = value.item()

        # Log per-position accuracy
        if len(per_position_accuracy) < 5:
            for pos, acc in enumerate(per_position_accuracy):
                metrics[f"train/PerPositionAccuracy/pos{pos}"] = acc

        # Log per-layer accuracy
        if per_layer_accuracy is not None:
            for layer_idx, layer_acc in enumerate(per_layer_accuracy):
                metrics[f"train/LayerAccuracy/layer{layer_idx}"] = layer_acc

        # Calculate modality gap for the current batch
        if False and self.should_log_this_step() and "images" in batch and batch["images"] is not None:  # Soft-remove modality gap calculation
            with torch.no_grad():
                output = self.fsdp_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    attention_bias=batch.get("attention_bias"),
                    response_mask=(batch["loss_masks"] > 0) if "loss_masks" in batch else None,
                    images=batch.get("images"),
                    image_masks=batch.get("image_masks"),
                    image_input_idx=batch.get("image_input_idx"),
                    subsegment_ids=batch.get("subsegment_ids"),
                    position_ids=batch.get("position_ids"),
                    return_visual_embeddings=True,
                    loss_masks=batch.get("loss_masks")
                )
                visual_embeddings = output.visual_embeddings
                sampled_visual_embeddings = output.sampled_visual_embeddings

                ####### Sample embeddings for PCA #######
                num_samples = 200  # Number of samples from each modality
                
                # Gather visual embeddings from all GPUs
                # print(f'DEBUG: visual_embeddings.shape: {visual_embeddings.shape}')
                gathered_visual = [torch.zeros_like(visual_embeddings) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_visual, visual_embeddings)
                # Concatenate along batch dimension
                visual_all = torch.cat(gathered_visual, dim=0)  # shape: (total_batch, patch_num, hidden_dim)
                
                # Flatten batch and patch dimensions
                batch_size, patch_num, hidden_dim = visual_all.shape
                visual_flat = visual_all.reshape(-1, hidden_dim)  # shape: (total_batch * patch_num, hidden_dim)
                
                # Sample from flattened visual embeddings
                if visual_flat.size(0) >= num_samples:
                    visual_indices = torch.randperm(visual_flat.size(0), device=visual_flat.device)[:num_samples]
                    visual_samples = visual_flat[visual_indices]
                else:
                    # If we have fewer samples than requested, use all of them
                    visual_samples = visual_flat
                    num_samples = visual_flat.size(0)
                
                # Sample from vocab embeddings
                vocab_indices = torch.randperm(Trainer.vocab_embeddings.size(0), device=Trainer.vocab_embeddings.device)[:num_samples]
                vocab_samples = Trainer.vocab_embeddings[vocab_indices]
                
                # Combine samples and move to CPU for PCA
                combined_samples = torch.cat([visual_samples, vocab_samples], dim=0).cpu().numpy()
                
                # Perform PCA
                from sklearn.decomposition import PCA
                pca = PCA(n_components=3)  # Get top 3 components
                transformed = pca.fit_transform(combined_samples)
                
                # Split back into visual and vocab components
                visual_pca = transformed[:num_samples]
                vocab_pca = transformed[num_samples:]

                # Only create and log plot on rank 0
                if get_global_rank() == 0:
                    try:
                        import matplotlib.pyplot as plt
                        fig = plt.figure(figsize=(10, 10))
                        ax = fig.add_subplot(111)

                        # Plot visual in one color, vocab in another
                        ax.scatter(visual_pca[:, 0], visual_pca[:, 1], c='blue', label="Visual", alpha=0.7)
                        ax.scatter(vocab_pca[:, 0], vocab_pca[:, 1], c='orange', label="Vocab", alpha=0.7)

                        ax.legend()
                        ax.set_title(f"PCA Visualization at Step {self.global_step}")
                        ax.set_xlabel("First Principal Component")
                        ax.set_ylabel("Second Principal Component")

                        # Add to metrics for wandb logging
                        metrics["pca_plot"] = wandb.Image(fig)
                        
                        # Close the figure to free memory
                        plt.close(fig)
                    except Exception as e:
                        log.warning(f"Failed to create PCA plot: {str(e)}")

                # Calculate average norm of visual embeddings
                visual_norms = torch.norm(visual_embeddings, dim=-1)  # [batch_size, seq_len]
                avg_visual_norm = visual_norms.mean()
                dist.all_reduce(avg_visual_norm)
                avg_visual_norm = avg_visual_norm / dist.get_world_size()

                metrics["VisualEmbeddings/average_norm"] = avg_visual_norm
                
                # Calculate std of norms to measure consistency
                std_visual_norm = visual_norms.std()
                dist.all_reduce(std_visual_norm)
                std_visual_norm = std_visual_norm / dist.get_world_size()
                metrics["VisualEmbeddings/norm_std"] = std_visual_norm

                # Calculate the average visual embedding for the batch
                visual_avg = visual_embeddings.mean(dim=(0,1))
                # Calculate the modality gap
                modality_gap_l2 = torch.norm(1/self.text_avg.shape[-1] *(visual_avg - self.text_avg))
                dist.all_reduce(modality_gap_l2)    
                modality_gap_l2 = modality_gap_l2 / dist.get_world_size()
                metrics["ModalityGap/L2_normalized"] = modality_gap_l2
                modality_gap_cosine = torch.nn.functional.cosine_similarity(visual_avg, self.text_avg, dim=0)
                dist.all_reduce(modality_gap_cosine)
                modality_gap_cosine = modality_gap_cosine / dist.get_world_size()
                metrics["ModalityGap/cosine"] = modality_gap_cosine

                # Calculate nearest vocab embedding distance for each visual token
                vocab_embeddings_norm = F.normalize(Trainer.vocab_embeddings, p=2, dim=1)
                visual_embeddings_norm = F.normalize(visual_embeddings.view(-1, visual_embeddings.size(-1)), p=2, dim=1)
                # Compute cosine similarities
                similarities = torch.mm(visual_embeddings_norm, vocab_embeddings_norm.T)
                # Find the maximum similarity for each visual token
                max_similarities, _ = similarities.max(dim=1)
                # Calculate the average maximum similarity
                avg_max_similarity = max_similarities.mean()
                dist.all_reduce(avg_max_similarity)
                avg_max_similarity = avg_max_similarity / dist.get_world_size()
                metrics["ModalityGap/NearestVocabCosine"] = avg_max_similarity

                # Calculate nearest vocab embedding L2 distance for each visual token
                # Reshape visual embeddings for batch processing
                visual_embeddings_flat = visual_embeddings.view(-1, visual_embeddings.size(-1))
                # Compute L2 distances
                l2_distances = torch.cdist(visual_embeddings_flat, Trainer.vocab_embeddings)
                # Find the minimum L2 distance for each visual token
                min_l2_distances, _ = l2_distances.min(dim=1)
                # Calculate the average minimum L2 distance
                avg_min_l2_distance = 1/self.text_avg.shape[-1] * min_l2_distances.mean()
                dist.all_reduce(avg_min_l2_distance)
                avg_min_l2_distance = avg_min_l2_distance / dist.get_world_size()
                metrics["ModalityGap/NearestVocabL2_normalized"] = avg_min_l2_distance


            # Calculate intra-modality similarity for visual tokens (layer 0)
            batch_size, num_patches, hidden_dim = visual_embeddings.shape
            if batch_size > 1:
                # Randomly select one patch from each image
                random_indices = torch.randint(0, num_patches, (batch_size,), device=visual_embeddings.device)
                selected_patches = visual_embeddings[torch.arange(batch_size), random_indices]  # [B, D]

                # Gather selected patches across GPUs
                gathered = [torch.zeros_like(selected_patches) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered, selected_patches)
                selected_patches = torch.cat(gathered, dim=0)  # [B_total, D]

                if selected_patches.shape[0] > 1:
                    selected_patches_norm = F.normalize(selected_patches, p=2, dim=1)
                    similarities = selected_patches_norm @ selected_patches_norm.T
                    mask = torch.triu(torch.ones_like(similarities), diagonal=1)
                    num_pairs = mask.sum()
                    if num_pairs > 0:
                        avg_similarity = (similarities * mask).sum() / num_pairs
                        metrics["VisualEmbeddings/intra_modality_cosine_similarity_layer0"] = avg_similarity.item()

            if sampled_visual_embeddings is not None:
                for layer_idx, embeddings in sampled_visual_embeddings.items():
                    embeddings_list = [torch.zeros_like(embeddings) for _ in range(dist.get_world_size())]
                    dist.all_gather(embeddings_list, embeddings)
                    embeddings = torch.cat(embeddings_list, dim=0)

                    if embeddings.shape[0] <= 1:
                        continue

                    # Normalize
                    embeddings_norm = F.normalize(embeddings, p=2, dim=1)

                    # Cosine sim matrix
                    sim_matrix = embeddings_norm @ embeddings_norm.T

                    # Take upper triangle (excluding diagonal)
                    mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1)
                    num_pairs = mask.sum()

                    if num_pairs > 0:
                        avg_sim = (sim_matrix * mask).sum() / num_pairs
                        metrics[f"VisualEmbeddings/intra_modality_cosine_similarity_layer{layer_idx}"] = avg_sim.item()


        return metrics

    def eval_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
            acc, ce_loss, z_loss, logits, _  = self.model_forward(batch, loss_reduction="none", compute_z_loss=True, return_logit_lenses=False)
        if "labels" in batch:
            loss_masks = batch["loss_masks"] * (batch["loss_masks"] > 0)
            batch_size_in_tokens = loss_masks.sum(-1)

            return dict(
                total_weight=batch_size_in_tokens.sum(),
                total_loss=ce_loss.sum(),
                total_accuracy=acc.sum(),
                total_zloss=z_loss.sum(),
                batch_loss=ce_loss.sum()/batch_size_in_tokens.sum(),
                batch_accuracy=acc.sum()/batch_size_in_tokens.sum(),
                batch_zloss=z_loss.sum()/batch_size_in_tokens.sum(),
                logits=logits
            )
        else:
            return dict(
                instance_loss=ce_loss.mean(-1),
                instance_aaccuracy=acc.mean(-1),
                batch_loss=ce_loss.mean(),
                batch_accuracy=acc.mean(),
                z_loss=z_loss.mean(),
                logits=logits
            )

    def eval_step(self, batch: Dict[str, Any], evaluator: DatasetMetrics) -> None:
        # Move tensors to the right device.
        batch = self.move_to_device(batch, self.device)

        # Run forward pass.
        with torch.no_grad():  # NOTE: 'torch.inference_mode()' doesn't work with 'torch.compile()'.
            eval_out = self.eval_batch(batch)

        # Update metrics.
        evaluator.update_metrics(
            batch, eval_out
        )  # batch includes all keys that the downstream evaluation needs

        barrier()

    def split_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        microbatch_size = self.cfg.device_train_microbatch_size
        batch_size = batch["input_ids"].shape[0]
        if batch_size <= microbatch_size:
            return [batch]
        else:
            micro_batches = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    micro_batches[key] = value.split(microbatch_size, dim=0)
                elif isinstance(value, list):
                    micro_batches[key] = [
                        value[microbatch_size * i : microbatch_size * i + microbatch_size]
                        for i in range(math.ceil(batch_size / microbatch_size))
                    ]
                else:
                    raise ValueError(f"unexpected item in batch: '{key}={value}'")
            return [
                {key: value[i] for key, value in micro_batches.items()}  # type: ignore
                for i in range(len(micro_batches["input_ids"]))
            ]

    def system_metrics(self) -> Dict[str, float]:
        metrics = {}
        if self.global_step < 3 or self.global_step % 10 == 0:
            peak_gpu_mb = peak_gpu_memory()
            if peak_gpu_mb is not None:
                metrics["System/Peak GPU Memory (MB)"] = peak_gpu_mb
        return metrics

    def log_metrics_to_console(self, prefix: str, metrics: Dict[str, float]):
        # Calculate time estimates
        current_time = time.time()
        step_time = current_time - self._last_step_time
        self._last_step_time = current_time

        # Add the current step time to the deque
        self._step_times.append(step_time)

        # Calculate average step time
        avg_step_time = sum(self._step_times) / len(self._step_times)

        # Estimate time remaining
        steps_remaining = self.max_steps - self.global_step
        estimated_time_remaining = avg_step_time * steps_remaining

        # Calculate elapsed time and total estimated time
        elapsed_time = current_time - self._start_time
        total_estimated_time = elapsed_time + estimated_time_remaining

        # Format time as readable strings
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))
        remaining_str = str(timedelta(seconds=int(estimated_time_remaining)))
        total_str = str(timedelta(seconds=int(total_estimated_time)))

        # Add time estimates to metrics
        time_metrics = {
            "time/elapsed": elapsed_str,
            "time/remaining": remaining_str,
            "time/total_estimate": total_str,
        }

        # Combine with existing metrics
        all_metrics = {**metrics, **time_metrics}

        # Log using existing logging mechanism
        log.info(
            f"{prefix}\n"
            + "\n".join(
                [
                    f"    {name}={value}"
                    for name, value in all_metrics.items()
                    if isinstance(value, (int, float, str))
                ]
            )
        )

    def should_log_optim_metrics_this_step(self) -> bool:
        if self.cfg.wandb is None:
            # We only log optimizer-specific metrics to W&B, since there are usually too many metrics
            # to log to the console.
            return False
        optim_log_interval = self.cfg.optimizer.metrics_log_interval
        if optim_log_interval is None:
            optim_log_interval = self.cfg.wandb.log_interval
        else:
            optim_log_interval = max(optim_log_interval, self.cfg.wandb.log_interval)
        return self.global_step % optim_log_interval == 0

    def should_log_this_step(self) -> bool:
        if self.global_step % self.cfg.console_log_interval == 0:
            return True
        elif self.cfg.wandb is not None and self.global_step % self.cfg.wandb.log_interval == 0:
            return True
        else:
            return False

    def inference_eval(self) -> Dict[str, Union[float, WBValue]]:
        self.optim.zero_grad(set_to_none=True)
        self.fsdp_model.eval()
        all_metrics = {}
        for evaluator in self.inference_evaluators:
            log.info(f"Running evaluation for '{evaluator.label}'...")
            dataset_metrics = evaluator.evaluate_model(
                self.fsdp_model,
                device=self.device,
                autocast_precision=self.cfg.autocast_precision,
                is_distributed=True,
                inference_warmup=self._inference_warmup,
                pbar=False
            )
            self._inference_warmup = False
            self.log_metrics_to_console(f"{evaluator.label}", dataset_metrics)
            all_metrics.update({f"{evaluator.label}/{k}": v for k, v in dataset_metrics.items()})
        return all_metrics

    def eval(self) -> Dict[str, Any]:
        # Zero gradients and set model to 'eval' mode.
        self.optim.zero_grad(set_to_none=True)
        self.fsdp_model.eval()
        warmed_up = False
        torch.cuda.empty_cache()

        eval_metrics = {}
        for evaluator in self.evaluators:
            if not warmed_up:
                # The first batch can take a while as the iterator compiles/warms up, this
                # can cause the nodes to think they got de-synced since some of the nodes
                # might take much longer to get it and start the forward pass then others.
                # To avoid this, we manually sync the nodes for the first batch
                barrier()
                warmed_up = True

            log.info(f"Running evaluation for '{evaluator.label}'...")

            # Reset metrics.
            evaluator.reset_metrics()

            # Initialize data loader iterator.
            eval_batches = iter(evaluator.eval_loader)

            # Adjust how many batches to evaluate on.
            num_eval_batches = (
                evaluator.subset_num_batches
                if evaluator.subset_num_batches is not None
                else self.cfg.eval_subset_num_batches
            )
            if num_eval_batches > 0:
                if isinstance(evaluator.eval_loader, torch.utils.data.IterableDataset):
                    pass  # No defined length
                else:
                    num_eval_batches = min(num_eval_batches, len(evaluator.eval_loader))
                eval_batches = islice(eval_batches, num_eval_batches)

            # Run model over batches.
            for eval_step, eval_batch in enumerate(eval_batches):
                self.eval_step(eval_batch, evaluator)

                # Log to console.
                if eval_step + 1 == num_eval_batches or (eval_step + 1) % self.cfg.console_log_interval == 0:
                    log.info(f"[eval_step={eval_step + 1}/{num_eval_batches}]")

            if hasattr(evaluator.eval_loader, "reset"):
                evaluator.eval_loader.reset()  # Reset the loader to free RAM

            # Get final metrics.
            metrics = evaluator.compute_metrics()
            eval_metrics.update(metrics)
            self.log_metrics_to_console(f"{evaluator.label}", metrics)

            del eval_batches

        return eval_metrics

    def check_if_cancelled(self) -> Tuple[bool, int]:
        should_cancel = False
        cancel_reason: Optional[str] = None
        extra_steps = 0
        if get_global_rank() == 0:
            if self.cfg.time_limit is not None and time.time() - self._start_time >= self.cfg.time_limit:
                # First check if we've reached the training time limit.
                should_cancel = True
                cancel_reason = "time limit reached"
                extra_steps = self.cfg.extra_steps_after_cancel
            elif wandb is not None and wandb.run is not None and (api_key := os.environ.get("WANDB_API_KEY")) is not None:
                # Finally, check if someone canceled the run from W&B by adding the 'cancel' / 'canceled' tag..
                # We won't see it in the run object. So we have to use the import/export API to check.
                from requests.exceptions import RequestException
                from wandb.errors import CommError

                try:
                    api = wandb.Api(api_key=api_key)
                    run = api.run(wandb.run.path)
                    for tag in run.tags or []:
                        if tag.lower() in {"cancel", "canceled", "cancelled"}:
                            should_cancel = True
                            cancel_reason = "Weights & Biases tag"
                            extra_steps = self.cfg.extra_steps_after_cancel
                            break
                except (RequestException, CommError):
                    log.info("Failed to check if W&B run is cancelled, continuing run.")

        run_canceled = synchronize_flag(should_cancel, self.device)
        if run_canceled:
            extra_steps = synchronize_value(extra_steps, self.device)
            if cancel_reason is None:
                if extra_steps > 0:
                    log.warning(f"Run canceled, stopping in {extra_steps} more steps...")
                else:
                    log.warning("Run canceled")
            else:
                if extra_steps > 0:
                    log.warning(f"Run canceled due to {cancel_reason}, stopping in {extra_steps} more steps...")
                else:
                    log.warning(f"Run canceled due to {cancel_reason}")

        return run_canceled, extra_steps

    def fit(self):
        if self.cfg.stop_after is not None:
            if self.cfg.stop_at is None:
                self.cfg.stop_at = self.global_step + self.cfg.stop_after
            else:
                self.cfg.stop_at = min(self.cfg.stop_at, self.global_step + self.cfg.stop_after)

        # Set default NN evaluation interval if not specified
        if False:  # Soft-remove NN evaluation interval setting
            if not hasattr(self.cfg, 'expensive_logging_interval'):
                self.cfg.expensive_logging_interval = 10

        self._start_time = time.time()
        self._gc_init_state = gc.isenabled()  # cache if garbage collection is enabled, reset on close.

        # Log vocabulary embedding statistics at start of training
        if False and wandb is not None and wandb.run is not None:  # Soft-remove vocab embedding logging
            vocab_norms = torch.norm(Trainer.vocab_embeddings, dim=-1)
            wandb.run.summary["VocabularyEmbeddings/average_norm"] = vocab_norms.mean().item()
            wandb.run.summary["VocabularyEmbeddings/norm_std"] = vocab_norms.std().item()

        # Disable automatic garbage collection, FSDP doesn't work well with it.
        if self.cfg.gen1_gc_interval is not None:
            gc.disable()

        if self.cfg.load_path is not None and self.global_step > 0 and self.cfg.eval_on_load:
            eval_metrics = self.eval()
            if wandb is not None and wandb.run is not None:
                wandb.log(eval_metrics, step=self.global_step)

        # Set model to 'train' mode.
        self.fsdp_model.train()

        # Initialize monitors.
        assert self.cfg.device_train_batch_size is not None
        speed_monitor = SpeedMonitor(self.cfg.speed_monitor)
        lr_monitor = LRMonitor(self.optim)
        batch_monitor = BatchStatsMonitor()

        # Log system metrics at the start of training.
        sys_metrics = self.system_metrics()
        if sys_metrics:
            self.log_metrics_to_console("Pre-train system metrics", sys_metrics)
            if wandb is not None and wandb.run is not None:
                wandb.log(sys_metrics, step=0)

        # Python Profiler stuff
        if self.cfg.python_profiling:
            python_profiler = cProfile.Profile()
        else:
            python_profiler = None

        # PyTorch Profiler stuff
        if self.cfg.torch_profiling and get_global_rank() == 0:
            from torch.profiler import schedule

            profiling_schedule = schedule(wait=1, warmup=5, active=3, repeat=1)

            def on_trace_ready(p):
                profiler_output_dir = Path(self.cfg.save_folder) / "profiler"
                profiler_output_dir.mkdir(exist_ok=True)

                output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=32)
                log.info(f"Profile by total GPU time at step {p.step_num}:\n{output}")
                output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=32)
                log.info(f"Profile by total CPU time at step {p.step_num}:\n{output}")

                p.export_chrome_trace(
                    str(trace_path := (profiler_output_dir / f"{p.step_num}.chrome_trace.json.gz"))
                )
                if self.cfg.remote_save_folder is not None:
                    upload_folder = f"{self.cfg.remote_save_folder.rstrip('/')}/profiler"
                    log.info(f"Tracing complete, uploading results to '{upload_folder}'...")
                    upload(trace_path, f"{upload_folder}/{trace_path.name}")

            from torch.profiler import ProfilerActivity

            torch_profiler = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=True,
                schedule=profiling_schedule,
                on_trace_ready=on_trace_ready,
            )
            del profiling_schedule
        else:
            import contextlib

            torch_profiler = contextlib.nullcontext()

        # Train.
        first_batch: bool = True
        cancel_initiated: bool = False
        stop_at: Optional[int] = self.cfg.stop_at
        save_checkpoints: bool = True

        warmed_up = False

        with torch_profiler as p:
            for epoch in range(self.epoch or 0, self.max_epochs):
                for batch in self.train_loader:
                    if not warmed_up:
                        # The first batch can take a while as the iterator compiles/warms up, this
                        # can cause the nodes to think they got de-synced since some of the nodes
                        # might take much longer to get it and start the forward pass then others.
                        # To avoid this, we manually sync the nodes for the first batch
                        barrier()
                        warmed_up = True

                    # Bookkeeping.
                    # NOTE: To track the global batch size / number of tokens per batch we make the assumption that all
                    # batches see the same number of tokens, which should be the case for language model pre-training
                    # (at least when drop_last=True).
                    # Alternatively we'd have to use a distributed all reduce over seq_len here, but I don't want that
                    # overhead. So for now I'm putting these assertions here so if the assumption is violated it will
                    # fail loudly.
                    batch_size, seq_len = batch["input_ids"].shape
                    assert seq_len <= self.cfg.model.max_sequence_length
                    assert (
                        batch_size == (self.cfg.global_train_batch_size // get_world_size()),
                        f"batch size is {batch_size}, but bs={self.cfg.global_train_batch_size} among {get_local_world_size()} world size"
                    )
                    global_batch_size = batch_size * get_world_size()  # assumes batch size equal across ranks
                    self.global_step += 1
                    
                    # DEBUG: Print token details for first batch (commented out for production)
                    # if first_batch and get_global_rank() == 0:
                    #     log.info("="*80)
                    #     log.info("DEBUG [olmo/train.py first_batch]: Training batch token details")
                    #     ...
                    
                    self.global_train_examples_seen_this_epoch += global_batch_size
                    self.global_train_tokens_seen += global_batch_size * seq_len
                    speed_monitor.batch_start(
                        self.global_train_tokens_seen,
                        batch_size * seq_len,  # num tokens in batch for this device
                        (batch["loss_masks"] > 0).sum(),
                        # We start monitoring speed after the first batch since the first
                        # batch might be an outlier due to compiling and other initialization overhead.
                        record=not first_batch,
                    )
                    batch_monitor.log_batch(batch)

                    should_log_this_step = self.should_log_this_step()

                    # Run train step on batch.
                    metrics = self.train_step(batch, reduce_global_loss=should_log_this_step)

                    # Maybe collect other metrics.
                    if should_log_this_step:
                        # Speed metrics.
                        metrics.update(speed_monitor.check())
                        # System metrics.
                        metrics.update(self.system_metrics())

                        # Learning rate metrics.
                        metrics.update(batch_monitor.check(self.device))

                        # Learning rate metrics.
                        metrics.update(lr_monitor.check())

                    # Log metrics to console.
                    if self.global_step % self.cfg.console_log_interval == 0:
                        if get_global_rank() == 0:
                            self.log_metrics_to_console(f"[step={self.global_step}/{self.max_steps}]", metrics)
                        else:
                            log.info(f"[step={self.global_step}/{self.max_steps}]")

                    # Log metrics to W&B.
                    if (
                        wandb is not None and wandb.run is not None
                        and self.cfg.wandb is not None
                        and self.global_step % self.cfg.wandb.log_interval == 0
                    ):
                        wandb.log(metrics, step=self.global_step)

                    # Check if/when run should be canceled.
                    if not cancel_initiated and self.global_step % self.cfg.canceled_check_interval == 0:
                        cancel_initiated, extra_steps = self.check_if_cancelled()
                        if cancel_initiated:
                            stop_at = (
                                self.global_step + extra_steps
                                if stop_at is None
                                else min(self.global_step + extra_steps, stop_at)
                            )

                    # Maybe save sharded checkpoint.
                    if save_checkpoints and (
                        cancel_initiated
                        or (
                            self.global_step % self.cfg.save_interval == 0
                            and self.cfg.save_num_checkpoints_to_keep != 0
                        )
                    ):
                        log.info("Saving checkpoint...")
                        checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded)
                        log.info(f"Checkpoint saved to {checkpoint_path}")

                        # Remove any ephemeral checkpoints.
                        while self.ephemeral_checkpoints:
                            self.remove_ephemeral_checkpoint()

                        # Reset speed monitor so that we don't count the time taken to save checkpoints.
                        speed_monitor.reset()

                        # If the run was just canceled this will be the final checkpoint.
                        if cancel_initiated:
                            save_checkpoints = False
                    elif (
                        self.cfg.save_interval_ephemeral is not None
                        and self.global_step % self.cfg.save_interval_ephemeral == 0
                    ):
                        log.info("Saving ephemeral checkpoint...")
                        checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded_ephemeral)
                        log.info(f"Checkpoint saved to {checkpoint_path}")

                        # Reset speed monitor so that we don't count the time taken to save checkpoints.
                        speed_monitor.reset()

                    # Maybe save unsharded checkpoint.
                    if (
                        save_checkpoints
                        and self.cfg.save_interval_unsharded is not None
                        and self.global_step % self.cfg.save_interval_unsharded == 0
                        and self.cfg.save_num_unsharded_checkpoints_to_keep != 0
                    ):
                        log.info("Saving unsharded checkpoint...")
                        checkpoint_path, _ = self.save_checkpoint(CheckpointType.unsharded)
                        log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

                        # Reset speed monitor so that we don't count the time taken to save checkpoints.
                        speed_monitor.reset()

                    # Maybe run evaluations.
                    last_step = stop_at and (self.global_step >= stop_at)
                    if not cancel_initiated and self.cfg.eval_interval > 0 and (
                        self.global_step % self.cfg.eval_interval == 0 or last_step):
                        eval_metrics = self.eval()

                        # Log metrics to W&B.
                        if wandb is not None and wandb.run is not None:
                            wandb.log(eval_metrics, step=self.global_step)

                        # Reset speed monitor so that we don't count the time taken to run evaluations.
                        speed_monitor.reset()

                        # Reset model to 'train' mode.
                        self.fsdp_model.train()

                    if not cancel_initiated and (
                        self.inference_evaluators and
                        self.cfg.inf_eval_interval and
                        (self.global_step % self.cfg.inf_eval_interval == 0 or last_step)
                    ):
                        eval_metrics = self.inference_eval()

                        # Log metrics to W&B.
                        if wandb is not None and wandb.run is not None:
                            wandb.log(eval_metrics, step=self.global_step)

                        # Reset speed monitor so that we don't count the time taken to run evaluations.
                        speed_monitor.reset()

                        # Reset model to 'train' mode.
                        self.fsdp_model.train()
                    
                    # DEBUG: Validation generation test (commented out for production)
                    # Uncomment for debugging generation during training
                    # if (self.global_step % 500 == 0 and hasattr(self.cfg.data, 'dataset') and self.cfg.data.dataset == "left_right"):
                    #     ... (validation generation code)

                    # End of batch.
                    first_batch = False
                    if p is not None:
                        p.step()

                    if stop_at is not None and self.global_step >= stop_at:
                        break

                    # Run generation 1 garbage collection.
                    if self.cfg.gen1_gc_interval is not None and self.global_step % self.cfg.gen1_gc_interval == 0:
                        gc.collect(1)

                    # Python Profiler stuff
                    # We do this now, at the bottom of this loop, so we capture the work of getting the next batch.
                    if python_profiler is not None:
                        if self.global_step == 5:
                            python_profiler.enable()
                        elif self.global_step == 8:
                            python_profiler.disable()
                            python_profiler.print_stats(sort=SortKey.CUMULATIVE)
                            python_profiler = None
                else:
                    log.info("Training epoch complete")
                    self.epoch = epoch + 1
                    self.global_train_examples_seen_this_epoch = 0
                    if self.epoch < self.max_epochs:
                        self.dataset.reshuffle()
                    continue

                break

        # Save final checkpoint.
        if save_checkpoints:
            final_checkpoint_path = None
            if (
                self.cfg.save_interval_unsharded is not None
                and self.last_unsharded_checkpoint_step != self.global_step
            ):
                log.info("Saving final unsharded model checkpoint...")
                checkpoint_path, _ = self.save_checkpoint(CheckpointType.unsharded)
                log.info(f"Unsharded checkpoint saved to {checkpoint_path}")
                final_checkpoint_path = checkpoint_path
            elif (
                self.cfg.save_num_checkpoints_to_keep != 0
                and self.last_sharded_checkpoint_step != self.global_step
            ):
                log.info("Saving final checkpoint...")
                checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded)
                log.info(f"Checkpoint saved to {checkpoint_path}")
                final_checkpoint_path = checkpoint_path
            
            # Cleanup all checkpoints except the final one
            if final_checkpoint_path is not None and get_global_rank() == 0:
                log.info(f"Cleaning up checkpoints, keeping only {final_checkpoint_path.name}...")
                save_folder = Path(self.cfg.save_folder)
                final_checkpoint_name = final_checkpoint_path.name
                
                # Count checkpoints before cleanup
                checkpoint_dirs = [d for d in save_folder.iterdir() 
                                 if d.is_dir() and d.name.startswith('step')]
                log.info(f"Found {len(checkpoint_dirs)} checkpoint directories")
                
                # Remove all checkpoint directories except the final one
                removed_count = 0
                for checkpoint_dir in checkpoint_dirs:
                    if checkpoint_dir.name != final_checkpoint_name:
                        log.info(f"Removing checkpoint: {checkpoint_dir.name}")
                        shutil.rmtree(checkpoint_dir, ignore_errors=True)
                        removed_count += 1
                
                # Also remove 'latest' and 'latest-unsharded' symlinks if they exist and don't point to final checkpoint
                for link_name in ['latest', 'latest-unsharded']:
                    link_path = save_folder / link_name
                    if link_path.exists():
                        if link_path.is_symlink() and link_path.resolve().name != final_checkpoint_name:
                            log.info(f"Removing symlink: {link_name}")
                            link_path.unlink()
                        elif not link_path.is_symlink() and link_path.name != final_checkpoint_name:
                            log.info(f"Removing directory: {link_name}")
                            shutil.rmtree(link_path, ignore_errors=True)
                
                log.info(f"Cleanup complete: removed {removed_count} checkpoint directories")
            
            barrier()

    def close(self, exit_code: int = 0) -> None:
        gc_cuda()

        if self._gc_init_state:
            gc.enable()
        else:
            gc.disable()
        if wandb is not None and wandb.run is not None:
            wandb.finish(exit_code=exit_code, quiet=True)

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        del exc_val, exc_tb
        self.close(0 if exc_type is None else 1)
