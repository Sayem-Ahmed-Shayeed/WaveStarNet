"""Self-contained runtime for the controlled Agrivision4 scratch comparison."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as functional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from agrivision4.scratch_comparison.experiment_core import (
        EarlyStoppingState,
        ScratchConfig,
        WarmupCosineScheduler,
        active_seeds,
        assert_no_test_access,
        build_optimizer_groups,
        checkpoint_is_better,
        class_weight_values,
        epochs_for_mode,
        model_spec,
        warmup_epochs_for_mode,
    )
except ModuleNotFoundError:
    # Generated notebooks define these symbols in the preceding core cell.
    pass


DATASET_ID = "sayemahmedshayeed/agrivision4-clean-grouped-research-v2"
IMAGE_DATASET_ID = "sumitnath07/agrivision4-clean-grouped-v1"
NUM_CLASSES = 21
PROTOCOL_ID = "agrivision4_scratch_comparison_v1"
EXPECTED_WAVESTAR_ARCHITECTURE_SHA256 = (
    "cc58b8d8fe86b84b5e40072ad98804f37ae03c18300e0a2510bbdefec78331ed"
)
EXPECTED_WAVESTAR_BACKBONE_SHA256 = (
    "f642f9b1eb6ca21454566ae81d7bc76782bed77e9210b4c4edaa32c63596bf94"
)
METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_precision",
    "macro_recall",
    "weighted_precision",
    "weighted_recall",
    "nll",
    "brier",
    "ece_15bin",
)


@dataclass(frozen=True)
class DevelopmentBundle:
    frames: dict[str, Any]
    class_names: tuple[str, ...]
    class_to_id: dict[str, int]
    root: Path
    image_root: Path
    manifest_paths: dict[str, Path]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(payload: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("high")


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_transforms() -> tuple[Any, Any]:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    training = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.70, 1.00),
                ratio=(0.80, 1.25),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.20, 0.20, 0.15, 0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    evaluation = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return training, evaluation


class ManifestImageDataset(Dataset):
    def __init__(self, frame: Any, transform: Any):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row.absolute_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return (
            tensor,
            int(row.label_id),
            str(row.sample_id),
            str(row.component_id),
            str(row.packaged_path),
        )


def resolve_locked_manifests(
    input_root: str | Path,
    expected_hashes: dict[str, str],
) -> tuple[Path, dict[str, Path]]:
    root = Path(input_root).resolve()
    candidates: list[tuple[Path, dict[str, Path]]] = []
    train_paths = list(root.rglob("manifests/train.csv"))
    train_paths.extend(root.rglob("manifests/inner_cv/fold_0_train.csv"))
    for train_path in sorted(set(path.resolve() for path in train_paths)):
        if train_path.name == "fold_0_train.csv":
            validation_path = train_path.with_name("fold_0_validation.csv")
            mapping_path = train_path.parent.parent / "class_mapping.json"
            dataset_root = train_path.parent.parent.parent
        else:
            validation_path = train_path.with_name("validation.csv")
            mapping_path = train_path.parent / "class_mapping.json"
            dataset_root = train_path.parent.parent
        paths = {
            "train": train_path,
            "validation": validation_path,
            "class_mapping": mapping_path,
        }
        if not all(path.is_file() for path in paths.values()):
            continue
        assert_no_test_access(paths.values())
        actual = {name: sha256_file(path) for name, path in paths.items()}
        if actual == expected_hashes:
            candidates.append((dataset_root.resolve(), paths))
    unique = {
        tuple(str(paths[name]) for name in ("train", "validation", "class_mapping")):
        (dataset_root, paths)
        for dataset_root, paths in candidates
    }
    if len(unique) != 1:
        raise ValueError(
            f"expected one hash-matched locked manifest set, found {len(unique)}"
        )
    return next(iter(unique.values()))


def resolve_image_root(
    input_root: str | Path,
    *,
    validate_images: bool,
) -> Path:
    root = Path(input_root).resolve()
    directories = sorted(
        path.resolve()
        for path in root.rglob("agrivision4-clean-grouped-images")
        if path.is_dir()
    )
    if len(directories) == 1:
        return directories[0]
    if len(directories) > 1:
        raise ValueError(f"expected one clean image root, found {len(directories)}")
    if not validate_images:
        return root / "agrivision4-clean-grouped-images"
    archives = sorted(
        path.resolve()
        for path in root.rglob("agrivision4-clean-grouped-images.zip")
        if path.is_file()
    )
    if len(archives) != 1:
        raise ValueError(
            f"expected one clean image directory or archive, found {len(archives)} archives"
        )
    extraction_root = Path("/kaggle/working/_agrivision4_scratch_images").resolve()
    extraction_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archives[0]) as archive:
        for member in archive.infolist():
            candidate = (extraction_root / member.filename).resolve()
            try:
                candidate.relative_to(extraction_root)
            except ValueError as error:
                raise ValueError("image archive contains an unsafe path") from error
        archive.extractall(extraction_root)
    return extraction_root


def load_development_bundle(
    input_root: str | Path,
    *,
    expected_hashes: dict[str, str],
    validate_images: bool = True,
) -> DevelopmentBundle:
    root, paths = resolve_locked_manifests(input_root, expected_hashes)
    image_root = resolve_image_root(input_root, validate_images=validate_images)
    mapping = json.loads(paths["class_mapping"].read_text(encoding="utf-8"))
    class_to_id = {
        str(name): int(label_id)
        for name, label_id in mapping["class_to_id"].items()
    }
    id_to_class = {
        int(label_id): str(name)
        for label_id, name in mapping["id_to_class"].items()
    }
    if sorted(class_to_id.values()) != list(range(NUM_CLASSES)):
        raise ValueError("class IDs must be contiguous range(21)")
    class_names = tuple(id_to_class[index] for index in range(NUM_CLASSES))
    frames: dict[str, Any] = {}
    sample_ids: set[str] = set()
    component_owner: dict[str, str] = {}
    required_columns = {
        "sample_id",
        "component_id",
        "split",
        "label",
        "label_id",
        "packaged_path",
        "inner_role",
    }
    for split in ("train", "validation"):
        frame = pd.read_csv(paths[split])
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"{split} manifest missing columns: {sorted(missing)}")
        if frame.empty or frame["sample_id"].duplicated().any():
            raise ValueError(f"{split} manifest is empty or has duplicate sample IDs")
        current_ids = set(frame["sample_id"].astype(str))
        if sample_ids.intersection(current_ids):
            raise ValueError("sample IDs overlap across development splits")
        sample_ids.update(current_ids)
        absolute_paths = []
        for row in frame.itertuples(index=False):
            if str(row.split) != "modeling_pool" or str(row.inner_role) != split:
                raise ValueError(f"manifest row has wrong locked role: {row.sample_id}")
            if class_to_id.get(str(row.label)) != int(row.label_id):
                raise ValueError(f"class mapping mismatch: {row.sample_id}")
            component = str(row.component_id)
            owner = component_owner.get(component)
            if owner is not None and owner != split:
                raise ValueError("component IDs overlap across development splits")
            component_owner[component] = split
            candidate = (image_root / str(row.packaged_path)).resolve()
            try:
                candidate.relative_to(image_root)
            except ValueError as error:
                raise ValueError("packaged path escapes image root") from error
            if validate_images and not candidate.is_file():
                raise ValueError(f"packaged image is missing: {row.packaged_path}")
            absolute_paths.append(str(candidate))
        frame = frame.copy()
        frame["absolute_path"] = absolute_paths
        frames[split] = frame.sort_values("sample_id").reset_index(drop=True)
    for split, frame in frames.items():
        support = set(int(value) for value in frame["label_id"])
        if support != set(range(NUM_CLASSES)):
            raise ValueError(f"{split} does not contain all 21 classes")
    return DevelopmentBundle(
        frames,
        class_names,
        class_to_id,
        root,
        image_root,
        paths,
    )


def stratified_subset(frame: Any, per_class: int, seed: int) -> Any:
    if per_class < 1:
        raise ValueError("per_class must be positive")
    groups = [
        group.sample(n=min(per_class, len(group)), random_state=seed)
        for _, group in frame.groupby("label_id", sort=True)
    ]
    return (
        pd.concat(groups, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def build_loaders(
    frames: dict[str, Any],
    seed: int,
    config: ScratchConfig,
) -> dict[str, DataLoader]:
    training_transform, evaluation_transform = build_transforms()
    generator = torch.Generator().manual_seed(seed)
    use_cuda = torch.cuda.is_available()
    common = {
        "num_workers": 2,
        "pin_memory": use_cuda,
        "persistent_workers": True,
        "worker_init_fn": seed_worker,
    }
    return {
        "train": DataLoader(
            ManifestImageDataset(frames["train"], training_transform),
            batch_size=config.train_batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
            **common,
        ),
        "validation": DataLoader(
            ManifestImageDataset(frames["validation"], evaluation_transform),
            batch_size=config.eval_batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        ),
    }


def _wavestar_symbols() -> tuple[Any, Any, Any]:
    if all(
        name in globals()
        for name in (
            "create_wavestarnet",
            "architecture_fingerprint",
            "backbone_schema_fingerprint",
        )
    ):
        return (
            globals()["create_wavestarnet"],
            globals()["architecture_fingerprint"],
            globals()["backbone_schema_fingerprint"],
        )
    from agrivision4.models.wavestarnet_family import (
        architecture_fingerprint,
        backbone_schema_fingerprint,
        create_wavestarnet,
    )

    return create_wavestarnet, architecture_fingerprint, backbone_schema_fingerprint


def state_schema_sha256(model: nn.Module) -> str:
    schema = [
        (name, list(tensor.shape), str(tensor.dtype))
        for name, tensor in sorted(model.state_dict().items())
    ]
    payload = json.dumps(schema, separators=(",", ":"), sort_keys=False).encode()
    return sha256_bytes(payload)


def build_model(
    config: ScratchConfig,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    spec = model_spec(config.model_key)
    if spec.pretrained:
        raise ValueError("scratch model specification cannot enable pretraining")
    if spec.kind == "wavestarnet":
        factory, architecture_fingerprint, backbone_schema_fingerprint = (
            _wavestar_symbols()
        )
        model = factory(config.model_key, num_classes=config.num_classes)
        architecture_hash = architecture_fingerprint(
            config.model_key, config.num_classes
        )
        backbone_hash = backbone_schema_fingerprint(config.model_key)
        if architecture_hash != EXPECTED_WAVESTAR_ARCHITECTURE_SHA256:
            raise ValueError("WavestarNet architecture SHA-256 mismatch")
        if backbone_hash != EXPECTED_WAVESTAR_BACKBONE_SHA256:
            raise ValueError("WavestarNet backbone schema SHA-256 mismatch")
    else:
        import timm

        if spec.timm_name is None:
            raise ValueError("timm model specification is incomplete")
        model = timm.create_model(
            spec.timm_name,
            pretrained=False,
            num_classes=config.num_classes,
        )
        architecture_hash = state_schema_sha256(model)
        backbone_hash = None
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters != spec.expected_parameters:
        raise ValueError(
            f"parameter count mismatch: {actual_parameters} != {spec.expected_parameters}"
        )
    audit = {
        "model_key": spec.key,
        "display_name": spec.display_name,
        "initialization": "random",
        "pretrained": False,
        "checkpoint_loaded": False,
        "teacher_loaded": False,
        "parameters": actual_parameters,
        "architecture_sha256": architecture_hash,
        "backbone_schema_sha256": backbone_hash,
    }
    return model.to(device), audit


def class_weights(
    train_frame: Any,
    device: torch.device,
    *,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    labels = train_frame["label_id"]
    if hasattr(labels, "tolist"):
        labels = labels.tolist()
    values = class_weight_values(labels, num_classes)
    return torch.tensor(values, dtype=torch.float32, device=device)


def build_optimizer(
    model: nn.Module,
    config: ScratchConfig,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, bool, list[dict[str, Any]]]:
    groups, audit = build_optimizer_groups(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    fused = device.type == "cuda"
    kwargs = {
        "lr": config.learning_rate,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
    }
    try:
        optimizer = torch.optim.AdamW(groups, fused=fused, **kwargs)
    except (TypeError, RuntimeError):
        fused = False
        optimizer = torch.optim.AdamW(groups, **kwargs)
    return optimizer, fused, audit


def fixed_label_metrics(
    *,
    sample_ids: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: Any,
    num_classes: int,
) -> dict[str, Any]:
    if len(sample_ids) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("metric input lengths differ")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("duplicate sample IDs")
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if probability_array.shape != (len(y_true), num_classes):
        raise ValueError("probability matrix shape is invalid")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(y_true, y_pred):
        if int(target) not in range(num_classes) or int(prediction) not in range(num_classes):
            raise ValueError("metric label is outside the fixed class range")
        matrix[int(target), int(prediction)] += 1
    per_class = []
    for label_id in range(num_classes):
        true_positive = int(matrix[label_id, label_id])
        false_positive = int(matrix[:, label_id].sum() - true_positive)
        false_negative = int(matrix[label_id, :].sum() - true_positive)
        support = int(matrix[label_id].sum())
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class.append(
            {
                "label_id": label_id,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    supports = matrix.sum(axis=1).astype(np.float64)
    total = float(supports.sum())
    accuracy = float(np.trace(matrix) / total)
    macro_precision = float(np.mean([row["precision"] for row in per_class]))
    macro_recall = float(np.mean([row["recall"] for row in per_class]))
    macro_f1 = float(np.mean([row["f1"] for row in per_class]))
    weighted_precision = float(
        sum(row["precision"] * row["support"] for row in per_class) / total
    )
    weighted_recall = float(
        sum(row["recall"] * row["support"] for row in per_class) / total
    )
    weighted_f1 = float(
        sum(row["f1"] * row["support"] for row in per_class) / total
    )
    return {
        "accuracy": accuracy,
        "balanced_accuracy": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    sample_ids: list[str] = []
    component_ids: list[str] = []
    packaged_paths: list[str] = []
    targets: list[int] = []
    logits_parts = []
    with torch.inference_mode():
        for images, labels, batch_ids, batch_components, batch_paths in loader:
            logits = model(images.float().to(device, non_blocking=True)).float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite validation logits")
            sample_ids.extend(batch_ids)
            component_ids.extend(batch_components)
            packaged_paths.extend(batch_paths)
            targets.extend(labels.tolist())
            logits_parts.append(logits.cpu())
    logits = torch.cat(logits_parts, dim=0)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    probabilities = torch.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1)
    metrics = fixed_label_metrics(
        sample_ids=sample_ids,
        y_true=targets,
        y_pred=predictions.tolist(),
        probabilities=probabilities.numpy(),
        num_classes=NUM_CLASSES,
    )
    metrics["nll"] = float(functional.cross_entropy(logits, target_tensor).item())
    one_hot = functional.one_hot(target_tensor, num_classes=NUM_CLASSES).float()
    metrics["brier"] = float(
        (probabilities - one_hot).square().sum(dim=1).mean().item()
    )
    confidences = probabilities.max(dim=1).values
    correct = predictions.eq(target_tensor)
    ece = 0.0
    boundaries = torch.linspace(0.0, 1.0, 16)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        selected = confidences.gt(lower) & confidences.le(upper)
        if selected.any():
            ece += float(selected.float().mean()) * abs(
                float(correct[selected].float().mean())
                - float(confidences[selected].mean())
            )
    metrics["ece_15bin"] = ece
    payload = {
        "sample_ids": sample_ids,
        "component_ids": component_ids,
        "packaged_paths": packaged_paths,
        "targets": np.asarray(targets, dtype=np.int64),
        "predictions": predictions.numpy(),
        "logits": logits.numpy(),
        "probabilities": probabilities.numpy(),
    }
    return metrics, payload


def should_step_optimizer(
    batch_index: int,
    *,
    total_batches: int,
    accumulation_steps: int,
) -> bool:
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    if batch_index not in range(1, total_batches + 1):
        raise ValueError("batch_index is outside the epoch")
    return batch_index % accumulation_steps == 0 or batch_index == total_batches


def _accumulation_window_size(
    batch_index: int,
    total_batches: int,
    accumulation_steps: int,
) -> int:
    window_start = ((batch_index - 1) // accumulation_steps) * accumulation_steps + 1
    return min(accumulation_steps, total_batches - window_start + 1)


def finish_amp_optimizer_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    gradient_clip_norm: float,
) -> tuple[float, bool]:
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    norm_value = float(norm.detach().cpu() if isinstance(norm, torch.Tensor) else norm)
    skipped = not math.isfinite(norm_value)
    if not skipped:
        scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return norm_value, skipped


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: ScratchConfig,
) -> dict[str, float | int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    sample_count = 0
    correct = 0
    maximum_gradient_norm = 0.0
    skipped_steps = 0
    total_batches = len(loader)
    for batch_index, (images, labels, _, _, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        window_size = _accumulation_window_size(
            batch_index,
            total_batches,
            config.gradient_accumulation_steps,
        )
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images)
            full_loss = criterion(logits, labels)
            loss = full_loss / window_size
        if not torch.isfinite(full_loss) or not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite training output")
        scaler.scale(loss).backward()
        if should_step_optimizer(
            batch_index,
            total_batches=total_batches,
            accumulation_steps=config.gradient_accumulation_steps,
        ):
            gradient_norm, skipped = finish_amp_optimizer_step(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                gradient_clip_norm=config.gradient_clip_norm,
            )
            if math.isfinite(gradient_norm):
                maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
            skipped_steps += int(skipped)
        batch_size = labels.shape[0]
        loss_sum += float(full_loss.detach().item()) * batch_size
        sample_count += batch_size
        correct += int(logits.detach().argmax(dim=1).eq(labels).sum().item())
    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct / sample_count,
        "gradient_norm": maximum_gradient_norm,
        "amp_nonfinite_steps_skipped": skipped_steps,
        "optimizer_steps": math.ceil(total_batches / config.gradient_accumulation_steps),
    }


def save_validation_evaluation(
    metrics: dict[str, Any],
    payload: dict[str, Any],
    class_names: tuple[str, ...],
    seed_root: Path,
) -> None:
    scalar_metrics = {
        key: value
        for key, value in metrics.items()
        if key not in {"confusion_matrix", "per_class"}
    }
    write_json(scalar_metrics, seed_root / "validation_metrics.json")
    pd.DataFrame(metrics["confusion_matrix"]).to_csv(
        seed_root / "validation_confusion_matrix.csv",
        index=False,
        header=False,
    )
    report = [
        {**row, "class_name": class_names[int(row["label_id"])]}
        for row in metrics["per_class"]
    ]
    write_json(
        {"labels": list(range(NUM_CLASSES)), "classes": report},
        seed_root / "validation_classification_report.json",
    )
    prediction_frame = pd.DataFrame(
        {
            "sample_id": payload["sample_ids"],
            "component_id": payload["component_ids"],
            "packaged_path": payload["packaged_paths"],
            "true_label_id": payload["targets"],
            "predicted_label_id": payload["predictions"],
        }
    )
    for label_id in range(NUM_CLASSES):
        prediction_frame[f"logit_{label_id:02d}"] = payload["logits"][:, label_id]
        prediction_frame[f"prob_{label_id:02d}"] = payload["probabilities"][:, label_id]
    prediction_frame.to_csv(seed_root / "validation_predictions.csv", index=False)


def _make_scaler(device: torch.device) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
        init_scale=128.0,
        growth_interval=2000,
    )


def _strict_reload_and_parity(
    *,
    model: nn.Module,
    checkpoint_path: Path,
    fixed_images: torch.Tensor,
    config: ScratchConfig,
    device: torch.device,
) -> bool:
    model.eval()
    with torch.inference_mode():
        expected = model(fixed_images.float()).float().cpu()
    reloaded, _ = build_model(config, device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    reloaded.load_state_dict(state, strict=True)
    reloaded.eval()
    with torch.inference_mode():
        actual = reloaded(fixed_images.float()).float().cpu()
    parity = torch.allclose(expected, actual, rtol=1e-5, atol=1e-6)
    del reloaded
    if not parity:
        raise ValueError("checkpoint reload logits differ")
    return True


def run_seed(
    *,
    seed: int,
    frames: dict[str, Any],
    class_names: tuple[str, ...],
    device: torch.device,
    weights: torch.Tensor,
    config: ScratchConfig,
    output_root: Path,
) -> dict[str, Any]:
    seed_everything(seed)
    seed_root = output_root / f"seed_{seed}"
    if seed_root.exists():
        shutil.rmtree(seed_root)
    seed_root.mkdir(parents=True)
    total_epochs = epochs_for_mode(config)
    write_json(
        {
            **asdict(config),
            "seed": seed,
            "active_epochs": total_epochs,
            "selection": "validation_macro_f1_then_nll_then_earlier_epoch",
        },
        seed_root / "config.json",
    )
    loaders = build_loaders(frames, seed, config)
    model, initialization_audit = build_model(config, device)
    write_json(initialization_audit, seed_root / "initialization_audit.json")
    optimizer, fused, optimizer_audit = build_optimizer(model, config, device)
    write_json(optimizer_audit, seed_root / "optimizer_groups.json")
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs_for_mode(config),
        min_lr_ratio=config.min_lr_ratio,
    )
    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=config.label_smoothing,
    )
    scaler = _make_scaler(device)
    early_stopping = EarlyStoppingState(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        start_epoch=config.early_stopping_start_epoch,
    )
    checkpoint_path = seed_root / "best_checkpoint.pt"
    best: dict[str, float | int] | None = None
    history = []
    started = time.perf_counter()
    stop_reason = "maximum_epochs"
    for epoch in range(1, total_epochs + 1):
        learning_rates = scheduler.step(epoch)
        train_metrics = train_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            scaler,
            device,
            config,
        )
        validation_metrics, _ = evaluate_model(model, loaders["validation"], device)
        candidate = {
            "macro_f1": validation_metrics["macro_f1"],
            "nll": validation_metrics["nll"],
            "epoch": epoch,
        }
        improved = checkpoint_is_better(candidate, best)
        if improved:
            best = candidate
            torch.save(model.state_dict(), checkpoint_path)
        should_stop = False
        if config.run_mode != "smoke":
            should_stop = early_stopping.update(epoch, validation_metrics["macro_f1"])
        elapsed = time.perf_counter() - started
        average_epoch = elapsed / epoch
        remaining_epochs = max(0, total_epochs - epoch)
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "gradient_norm": train_metrics["gradient_norm"],
            "amp_nonfinite_steps_skipped": train_metrics[
                "amp_nonfinite_steps_skipped"
            ],
            "optimizer_steps": train_metrics["optimizer_steps"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_nll": validation_metrics["nll"],
            "learning_rate": learning_rates[0],
            "improved_checkpoint": improved,
            "early_stopping_bad_epochs": early_stopping.bad_epochs,
            "elapsed_seconds": elapsed,
            "estimated_remaining_seconds": average_epoch * remaining_epochs,
            "peak_gpu_memory_bytes": (
                torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
            ),
        }
        history.append(record)
        print(
            json.dumps({"event": "epoch_complete", "seed": seed, **record}, sort_keys=True),
            flush=True,
        )
        if should_stop:
            stop_reason = "early_stopping"
            break
    pd.DataFrame(history).to_csv(seed_root / "history.csv", index=False)
    write_json(
        {**early_stopping.as_dict(), "reason": stop_reason, "epochs_run": len(history)},
        seed_root / "early_stopping.json",
    )
    if best is None or not checkpoint_path.is_file():
        raise RuntimeError("training did not produce a validation-selected checkpoint")
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    fixed_images = next(iter(loaders["validation"]))[0][:2].to(device)
    parity = _strict_reload_and_parity(
        model=model,
        checkpoint_path=checkpoint_path,
        fixed_images=fixed_images,
        config=config,
        device=device,
    )
    validation_metrics, validation_payload = evaluate_model(
        model, loaders["validation"], device
    )
    save_validation_evaluation(
        validation_metrics,
        validation_payload,
        class_names,
        seed_root,
    )
    checkpoint_hash = sha256_file(checkpoint_path)
    (seed_root / "best_checkpoint.sha256").write_text(
        checkpoint_hash + "\n", encoding="utf-8"
    )
    result = {
        "seed": seed,
        **{name: float(validation_metrics[name]) for name in METRIC_NAMES},
        "best_epoch": int(best["epoch"]),
        "epochs_run": len(history),
        "stop_reason": stop_reason,
        "checkpoint_sha256": checkpoint_hash,
        "optimizer_fused": fused,
        "checkpoint_reload": True,
        "strict_state_dict": True,
        "logit_parity": parity,
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        ),
    }
    del loaders, optimizer, scaler, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def aggregate_seed_metrics(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"seed_count": len(results)}
    for metric in METRIC_NAMES:
        values = [float(row[metric]) for row in results]
        mean = statistics.mean(values)
        standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        margin = (
            4.3026527299 * standard_deviation / math.sqrt(len(values))
            if len(values) == 3
            else None
        )
        summary[metric] = {
            "mean": mean,
            "standard_deviation": standard_deviation,
            "confidence_interval_95": (
                [mean - margin, mean + margin] if margin is not None else None
            ),
            "values": values,
        }
    return summary


def environment_payload(device: torch.device) -> dict[str, Any]:
    import timm
    import torchvision

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def artifact_inventory(root: Path) -> list[dict[str, Any]]:
    inventory_path = root / "artifact_inventory.json"
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != inventory_path
    ]


def validate_artifact_tree(root: Path, active_seed_values: Sequence[int]) -> None:
    required_root = {
        "access_policy.json",
        "class_weights.json",
        "dataset_summary.json",
        "environment.json",
        "model_summary.json",
        "run_manifest.json",
        "validation_seed_metrics.csv",
        "validation_summary.json",
    }
    missing_root = [name for name in required_root if not (root / name).is_file()]
    if missing_root:
        raise ValueError(f"required root artifacts are missing: {missing_root}")
    required_seed = {
        "best_checkpoint.pt",
        "best_checkpoint.sha256",
        "config.json",
        "early_stopping.json",
        "history.csv",
        "initialization_audit.json",
        "optimizer_groups.json",
        "validation_classification_report.json",
        "validation_confusion_matrix.csv",
        "validation_metrics.json",
        "validation_predictions.csv",
    }
    for seed in active_seed_values:
        seed_root = root / f"seed_{seed}"
        missing = [name for name in required_seed if not (seed_root / name).is_file()]
        if missing:
            raise ValueError(f"required seed artifacts are missing: {missing}")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = [part.casefold() for part in path.relative_to(root).parts]
        if any(part == "test" or part.startswith("test_") for part in relative_parts):
            raise ValueError(f"test artifact is forbidden: {path}")


def _validate_embedded_sources(values: dict[str, Any]) -> None:
    expected_pairs = (
        ("EMBEDDED_MODEL_SOURCE_BYTES", "EXPECTED_MODEL_SOURCE_SHA256"),
        ("EMBEDDED_CORE_SOURCE_BYTES", "EXPECTED_CORE_SOURCE_SHA256"),
        ("EMBEDDED_RUNTIME_SOURCE_BYTES", "EXPECTED_RUNTIME_SOURCE_SHA256"),
    )
    for bytes_key, hash_key in expected_pairs:
        if bytes_key not in values or hash_key not in values:
            raise ValueError(f"embedded source attestation is missing: {bytes_key}")
        if sha256_bytes(values[bytes_key]) != values[hash_key]:
            raise ValueError(f"embedded source SHA-256 mismatch: {bytes_key}")


def main_from_globals(values: dict[str, Any]) -> None:
    _validate_embedded_sources(values)
    config = ScratchConfig(
        model_key=str(values["MODEL_KEY"]),
        run_mode=str(values["RUN_MODE"]),
        seeds=tuple(int(seed) for seed in values["SEEDS"]),
        max_epochs=int(values["MAX_EPOCHS"]),
        warmup_epochs=int(values["WARMUP_EPOCHS"]),
        learning_rate=float(values["LEARNING_RATE"]),
        weight_decay=float(values["WEIGHT_DECAY"]),
        label_smoothing=float(values["LABEL_SMOOTHING"]),
        gradient_clip_norm=float(values["GRADIENT_CLIP_NORM"]),
        gradient_accumulation_steps=int(values["GRADIENT_ACCUMULATION_STEPS"]),
        early_stopping_patience=int(values["EARLY_STOPPING_PATIENCE"]),
        early_stopping_min_delta=float(values["EARLY_STOPPING_MIN_DELTA"]),
        early_stopping_start_epoch=int(values["EARLY_STOPPING_START_EPOCH"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("controlled Kaggle run requires a CUDA GPU")
    output_root = Path("/kaggle/working") / str(values["OUTPUT_SAFE_NAME"])
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    started_at = utc_now()
    print(
        json.dumps(
            {
                "event": "run_start",
                "model_key": config.model_key,
                "run_mode": config.run_mode,
                "initialization": "random",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    bundle = load_development_bundle(
        "/kaggle/input",
        expected_hashes=dict(values["EXPECTED_MANIFEST_SHA256"]),
    )
    full_frames = bundle.frames
    frames = full_frames
    if config.run_mode == "smoke":
        frames = {
            "train": stratified_subset(full_frames["train"], 2, 42),
            "validation": stratified_subset(full_frames["validation"], 1, 43),
        }
    active_seed_values = active_seeds(config.run_mode, config.seeds)
    write_json(
        {
            "test_manifest_loaded": False,
            "test_loader_created": False,
            "test_metrics_computed": False,
            "selection_source": "fixed_validation_only",
        },
        output_root / "access_policy.json",
    )
    write_json(environment_payload(device), output_root / "environment.json")
    probe_model, model_audit = build_model(config, device)
    write_json(model_audit, output_root / "model_summary.json")
    del probe_model
    gc.collect()
    torch.cuda.empty_cache()
    write_json(
        {
            "protocol_id": PROTOCOL_ID,
            "kernel_id": values["KERNEL_ID"],
            "expected_version": values["EXPECTED_VERSION"],
            "model_key": config.model_key,
            "display_name": model_spec(config.model_key).display_name,
            "run_mode": config.run_mode,
            "initialization": "random",
            "pretrained": False,
            "active_seeds": active_seed_values,
            "dataset_id": DATASET_ID,
            "image_dataset_id": IMAGE_DATASET_ID,
            "started_at": started_at,
            "test_accessed": False,
            "source_sha256": {
                "model": values["EXPECTED_MODEL_SOURCE_SHA256"],
                "core": values["EXPECTED_CORE_SOURCE_SHA256"],
                "runtime": values["EXPECTED_RUNTIME_SOURCE_SHA256"],
            },
            "expected_manifest_sha256": values["EXPECTED_MANIFEST_SHA256"],
        },
        output_root / "run_manifest.json",
    )
    write_json(
        {
            "dataset_id": DATASET_ID,
            "image_dataset_id": IMAGE_DATASET_ID,
            "class_names": list(bundle.class_names),
            "development_split_counts": {
                split: len(frame) for split, frame in full_frames.items()
            },
            "active_split_counts": {
                split: len(frame) for split, frame in frames.items()
            },
            "manifest_sha256": {
                name: sha256_file(path)
                for name, path in bundle.manifest_paths.items()
            },
        },
        output_root / "dataset_summary.json",
    )
    weights = class_weights(full_frames["train"], device)
    write_json(
        {
            "method": "sqrt_inverse_frequency_mean_normalized",
            "weighted_sampler": False,
            "values": weights.detach().cpu().tolist(),
        },
        output_root / "class_weights.json",
    )
    results = [
        run_seed(
            seed=seed,
            frames=frames,
            class_names=bundle.class_names,
            device=device,
            weights=weights,
            config=config,
            output_root=output_root,
        )
        for seed in active_seed_values
    ]
    pd.DataFrame(results).to_csv(
        output_root / "validation_seed_metrics.csv", index=False
    )
    summary = aggregate_seed_metrics(results)
    summary.update(
        {
            "seeds": active_seed_values,
            "selection": "validation_macro_f1_then_nll_then_earlier_epoch",
            "test_accessed": False,
        }
    )
    write_json(summary, output_root / "validation_summary.json")
    validate_artifact_tree(output_root, active_seed_values)
    manifest_path = output_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"completed_at": utc_now(), "status": "complete"})
    write_json(manifest, manifest_path)
    write_json(artifact_inventory(output_root), output_root / "artifact_inventory.json")
    print(
        json.dumps({"event": "run_complete", "output": str(output_root)}, sort_keys=True),
        flush=True,
    )
