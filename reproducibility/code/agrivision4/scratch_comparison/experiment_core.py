"""Pure contracts for the controlled Agrivision4 scratch comparison."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    kind: str
    timm_name: str | None
    expected_parameters: int
    pretrained: bool = False


MODEL_SPECS = {
    "wavestarnet_big_transfer_v2": ModelSpec(
        key="wavestarnet_big_transfer_v2",
        display_name="WavestarNet-Big-Transfer-V2",
        kind="wavestarnet",
        timm_name=None,
        expected_parameters=2_956_577,
    ),
    "mobilenetv3_large_100": ModelSpec(
        key="mobilenetv3_large_100",
        display_name="MobileNetV3-Large",
        kind="timm",
        timm_name="mobilenetv3_large_100",
        expected_parameters=4_228_933,
    ),
    "efficientnet_b0": ModelSpec(
        key="efficientnet_b0",
        display_name="EfficientNet-B0",
        kind="timm",
        timm_name="efficientnet_b0",
        expected_parameters=4_034_449,
    ),
    "efficientnet_b3": ModelSpec(
        key="efficientnet_b3",
        display_name="EfficientNet-B3",
        kind="timm",
        timm_name="efficientnet_b3",
        expected_parameters=10_728_509,
    ),
    "mobilevit_s": ModelSpec(
        key="mobilevit_s",
        display_name="MobileViT-S",
        kind="timm",
        timm_name="mobilevit_s",
        expected_parameters=4_951_093,
    ),
    "swin_tiny_patch4_window7_224": ModelSpec(
        key="swin_tiny_patch4_window7_224",
        display_name="Swin-Tiny",
        kind="timm",
        timm_name="swin_tiny_patch4_window7_224",
        expected_parameters=27_535_503,
    ),
    "convnext_tiny": ModelSpec(
        key="convnext_tiny",
        display_name="ConvNeXt-Tiny",
        kind="timm",
        timm_name="convnext_tiny",
        expected_parameters=27_836_277,
    ),
}


def model_spec(key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[key]
    except KeyError as error:
        raise ValueError(f"unknown scratch model_key: {key}") from error


@dataclass(frozen=True)
class ScratchConfig:
    model_key: str
    run_mode: str
    seeds: tuple[int, ...] = (42, 2024, 3407)
    num_classes: int = 21
    image_size: int = 224
    train_batch_size: int = 32
    eval_batch_size: int = 64
    max_epochs: int = 100
    smoke_epochs: int = 1
    warmup_epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.05
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    min_lr_ratio: float = 0.05
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 0.001
    early_stopping_start_epoch: int = 10

    def __post_init__(self) -> None:
        model_spec(self.model_key)
        active_seeds(self.run_mode, self.seeds)
        if self.num_classes != 21:
            raise ValueError("controlled Agrivision4 scratch protocol requires 21 classes")
        if self.image_size != 224:
            raise ValueError("controlled Agrivision4 scratch protocol requires 224 px")
        if self.max_epochs != 100 or self.smoke_epochs != 1:
            raise ValueError("scratch protocol requires 100 full and 1 smoke epoch")
        if self.warmup_epochs != 5:
            raise ValueError("scratch protocol requires five warm-up epochs")
        if self.train_batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer hyperparameters are invalid")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if not 0.0 < self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in (0, 1]")
        if self.early_stopping_patience != 30:
            raise ValueError("scratch protocol requires early-stopping patience 30")
        if self.early_stopping_min_delta != 0.001:
            raise ValueError("scratch protocol requires early-stopping min_delta 0.001")
        if self.early_stopping_start_epoch != 10:
            raise ValueError("scratch protocol begins early-stopping monitoring at epoch 10")


def active_seeds(run_mode: str, seeds: Sequence[int]) -> list[int]:
    if tuple(seeds) != (42, 2024, 3407):
        raise ValueError("seeds must be exactly [42, 2024, 3407]")
    if run_mode in {"smoke", "screening"}:
        return [42]
    if run_mode == "development":
        return list(seeds)
    raise ValueError("run_mode must be smoke, screening, or development")


def epochs_for_mode(config: ScratchConfig) -> int:
    return config.smoke_epochs if config.run_mode == "smoke" else config.max_epochs


def should_step_optimizer(
    batch_index: int,
    *,
    total_batches: int,
    accumulation_steps: int,
) -> bool:
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    if batch_index < 1 or batch_index > total_batches:
        raise ValueError("batch_index must be within the loader")
    return batch_index % accumulation_steps == 0 or batch_index == total_batches


def warmup_epochs_for_mode(config: ScratchConfig) -> int:
    return 0 if config.run_mode == "smoke" else config.warmup_epochs


def assert_no_test_access(paths: Iterable[str | Path]) -> None:
    for value in paths:
        path = Path(value)
        forbidden = any(
            part.casefold().split(".", 1)[0] == "test"
            or part.casefold().startswith("test_")
            for part in path.parts
        )
        if forbidden:
            raise ValueError(f"test access is forbidden in development: {path}")


def class_weight_values(label_ids: Sequence[int], num_classes: int) -> list[float]:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    counts = [0] * num_classes
    for value in label_ids:
        label = int(value)
        if label not in range(num_classes):
            raise ValueError(f"unknown training label: {label}")
        counts[label] += 1
    if any(count <= 0 for count in counts):
        raise ValueError("each class requires training support")
    sample_count = sum(counts)
    raw = [math.sqrt(sample_count / count) for count in counts]
    mean = sum(raw) / len(raw)
    return [value / mean for value in raw]


@dataclass
class EarlyStoppingState:
    patience: int
    min_delta: float
    start_epoch: int
    best_macro_f1: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0
    stop_epoch: int | None = None

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be nonnegative")
        if self.start_epoch < 1:
            raise ValueError("start_epoch must be positive")

    def update(self, epoch: int, macro_f1: float) -> bool:
        if self.stop_epoch is not None:
            return True
        if epoch < self.start_epoch:
            return False
        score = float(macro_f1)
        if not math.isfinite(score):
            raise ValueError("early-stopping score must be finite")
        if self.best_macro_f1 is None or score > self.best_macro_f1 + self.min_delta:
            self.best_macro_f1 = score
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.stop_epoch = int(epoch)
            return True
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "start_epoch": self.start_epoch,
            "best_macro_f1": self.best_macro_f1,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
            "stop_epoch": self.stop_epoch,
        }


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: Any,
        *,
        total_epochs: int,
        warmup_epochs: int,
        min_lr_ratio: float,
    ) -> None:
        if total_epochs < 1 or warmup_epochs < 0 or warmup_epochs >= total_epochs:
            raise ValueError("scheduler epoch configuration is invalid")
        if not 0.0 < min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in (0, 1]")
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    def _scale(self, epoch: int) -> float:
        if epoch not in range(1, self.total_epochs + 1):
            raise ValueError("epoch is outside the scheduler budget")
        if self.warmup_epochs and epoch <= self.warmup_epochs:
            return epoch / self.warmup_epochs
        decay_epochs = self.total_epochs - self.warmup_epochs
        decay_index = epoch - self.warmup_epochs - 1
        progress = decay_index / max(1, decay_epochs - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def step(self, epoch: int) -> list[float]:
        scale = self._scale(epoch)
        values = [base_lr * scale for base_lr in self.base_lrs]
        for group, value in zip(self.optimizer.param_groups, values):
            group["lr"] = value
        return values


def _parameter_modules(model: Any) -> dict[int, tuple[str, Any, str]]:
    owners: dict[int, tuple[str, Any, str]] = {}
    for module_path, module in model.named_modules():
        for local_name, parameter in module.named_parameters(recurse=False):
            full_name = f"{module_path}.{local_name}" if module_path else local_name
            owners.setdefault(id(parameter), (full_name, module, local_name))
    return owners


def _uses_zero_decay(module: Any, local_name: str) -> bool:
    import torch.nn as nn

    normalization_types = (
        nn.modules.batchnorm._BatchNorm,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.modules.instancenorm._InstanceNorm,
    )
    class_name = module.__class__.__name__.casefold()
    return (
        local_name.casefold() in {"bias", "gamma", "layer_scale", "layerscale"}
        or isinstance(module, normalization_types)
        or "norm" in class_name
        or "layerscale" in class_name
    )


def build_optimizer_groups(
    model: Any,
    *,
    learning_rate: float,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    owners = _parameter_modules(model)
    buckets: dict[str, list[Any]] = {"decay": [], "no_decay": []}
    audit = []
    for parameter in trainable:
        if id(parameter) not in owners:
            raise ValueError("trainable parameter has no owning module")
        parameter_name, module, local_name = owners[id(parameter)]
        bucket = "no_decay" if _uses_zero_decay(module, local_name) else "decay"
        applied_decay = 0.0 if bucket == "no_decay" else weight_decay
        buckets[bucket].append(parameter)
        audit.append(
            {
                "parameter_name": parameter_name,
                "group": bucket,
                "lr": learning_rate,
                "weight_decay": applied_decay,
                "numel": parameter.numel(),
            }
        )
    grouped = buckets["decay"] + buckets["no_decay"]
    if len({id(parameter) for parameter in grouped}) != len(grouped):
        raise ValueError("optimizer parameter groups overlap")
    if {id(parameter) for parameter in grouped} != {id(parameter) for parameter in trainable}:
        raise ValueError("optimizer parameter groups do not cover the model")
    groups = [
        {
            "name": name,
            "params": parameters,
            "lr": learning_rate,
            "weight_decay": 0.0 if name == "no_decay" else weight_decay,
        }
        for name, parameters in buckets.items()
    ]
    return groups, sorted(audit, key=lambda row: row["parameter_name"])


def checkpoint_is_better(
    candidate: dict[str, float | int],
    best: dict[str, float | int] | None,
) -> bool:
    if best is None:
        return True
    candidate_f1 = float(candidate["macro_f1"])
    best_f1 = float(best["macro_f1"])
    if candidate_f1 > best_f1 + 1e-12:
        return True
    if abs(candidate_f1 - best_f1) <= 1e-12:
        candidate_nll = float(candidate["nll"])
        best_nll = float(best["nll"])
        if candidate_nll < best_nll - 1e-12:
            return True
        if abs(candidate_nll - best_nll) <= 1e-12:
            return int(candidate["epoch"]) < int(best["epoch"])
    return False
