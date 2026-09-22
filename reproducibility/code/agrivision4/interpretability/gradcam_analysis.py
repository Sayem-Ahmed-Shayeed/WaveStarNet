"""Reproducible Grad-CAM analysis for the locked AgriVision4 validation set."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import platform
import random
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from agrivision4.models.wavestarnet_family import create_wavestarnet


REQUIRED_MANIFEST_COLUMNS = {
    "sample_id",
    "label",
    "label_id",
    "packaged_path",
}

WAVESTARNET_KEY = "wavestarnet_big_transfer_v2"
MOBILEVIT_KEY = "mobilevit_s"
CHECKPOINT_MEMBERS = {
    WAVESTARNET_KEY: (
        "results/scratch comparison/wavestarnet_big_transfer_v2/"
        "agrivision4_scratch_wavestarnet_transfer_v2_development_v01/"
        "seed_42/best_checkpoint.pt"
    ),
    MOBILEVIT_KEY: (
        "results/scratch comparison/mobilevit_s/"
        "agrivision4_scratch_mobilevit_s_development_v01/"
        "seed_42/best_checkpoint.pt"
    ),
}
REFERENCE_PREDICTION_MEMBERS = {
    key: member.replace("best_checkpoint.pt", "validation_predictions.csv")
    for key, member in CHECKPOINT_MEMBERS.items()
}


@dataclass(frozen=True)
class AnalysisConfig:
    repo_root: Path
    validation_csv: Path
    class_mapping_json: Path
    image_zip: Path
    checkpoint_zip: Path
    output_dir: Path
    seed: int = 42
    image_size: int = 224
    inline_examples_per_category: int = 4
    progress_interval: int = 10

    def __post_init__(self) -> None:
        for name in (
            "repo_root",
            "validation_csv",
            "class_mapping_json",
            "image_zip",
            "checkpoint_zip",
            "output_dir",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.seed != 42:
            raise ValueError("the locked analysis seed must be 42")
        if self.image_size != 224:
            raise ValueError("the locked analysis input size must be 224")
        if self.inline_examples_per_category < 1:
            raise ValueError("inline_examples_per_category must be positive")
        if self.progress_interval < 1:
            raise ValueError("progress_interval must be positive")


@dataclass(frozen=True)
class ValidationBundle:
    frame: pd.DataFrame
    class_names: tuple[str, ...]
    manifest_sha256: str
    class_mapping_sha256: str


@dataclass
class LoadedModel:
    key: str
    display_name: str
    model: nn.Module
    target_layers: list[tuple[str, nn.Module]]
    checkpoint_member: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class GradCAMResult:
    target_class: int
    logits: torch.Tensor
    fused_cam: torch.Tensor
    layer_cams: dict[str, torch.Tensor]


@dataclass(frozen=True)
class PredictionGradCAMResult:
    logits: torch.Tensor
    predicted_class: int
    cams: dict[str, GradCAMResult]


@dataclass(frozen=True)
class AnalysisSummary:
    predictions_csv: Path
    per_class_metrics_csv: Path
    case_selection_csv: Path
    summary_json: Path
    validation_count: int
    record_count: int


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def default_config(repo_root: Path) -> AnalysisConfig:
    root = Path(repo_root).resolve()
    manifest_root = (
        root
        / "agrivision4"
        / "research_protocol"
        / "agrivision4-locked-validation-v1"
        / "manifests"
    )
    return AnalysisConfig(
        repo_root=root,
        validation_csv=manifest_root / "validation.csv",
        class_mapping_json=manifest_root / "class_mapping.json",
        image_zip=(
            root
            / "agrivision4"
            / "cleaned_kaggle"
            / "agrivision4-clean-grouped-v1"
            / "agrivision4-clean-grouped-images.zip"
        ),
        checkpoint_zip=root / "agrivision4" / "results wavestarnet.zip",
        output_dir=(
            root
            / "agrivision4"
            / "results"
            / "interpretability"
            / "gradcam_validation_v1"
        ),
    )


class ZipImageStore:
    def __init__(self, archive_path: Path) -> None:
        self.archive_path = Path(archive_path)
        self._archive: zipfile.ZipFile | None = None

    def __enter__(self) -> "ZipImageStore":
        self._archive = zipfile.ZipFile(self.archive_path, "r")
        return self

    def open_rgb(self, packaged_path: str) -> Image.Image:
        if self._archive is None:
            raise RuntimeError("ZipImageStore must be used as a context manager")
        with self._archive.open(str(packaged_path)) as stream:
            image = Image.open(stream)
            image.load()
            return image.convert("RGB")

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None


def _parse_class_mapping(path: Path) -> tuple[dict[str, int], tuple[str, ...]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    class_to_id = {str(name): int(index) for name, index in payload["class_to_id"].items()}
    expected_ids = list(range(len(class_to_id)))
    if sorted(class_to_id.values()) != expected_ids:
        raise ValueError("class IDs must be contiguous from zero")
    id_to_class = {int(index): str(name) for index, name in payload["id_to_class"].items()}
    class_names = tuple(id_to_class[index] for index in expected_ids)
    if any(class_to_id[name] != index for index, name in enumerate(class_names)):
        raise ValueError("class_to_id and id_to_class disagree")
    return class_to_id, class_names


def load_locked_validation(config: AnalysisConfig) -> ValidationBundle:
    for path in (config.validation_csv, config.class_mapping_json, config.image_zip):
        if not path.is_file():
            raise FileNotFoundError(path)

    class_to_id, class_names = _parse_class_mapping(config.class_mapping_json)
    frame = pd.read_csv(config.validation_csv)
    missing_columns = REQUIRED_MANIFEST_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"validation manifest is missing columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError("validation manifest must not be empty")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("validation sample_id values must be unique")

    frame = frame.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["label"] = frame["label"].astype(str)
    frame["label_id"] = frame["label_id"].astype(int)
    frame["packaged_path"] = frame["packaged_path"].astype(str)
    for row in frame.itertuples(index=False):
        if row.label not in class_to_id:
            raise ValueError(f"unknown validation label: {row.label}")
        if class_to_id[row.label] != row.label_id:
            raise ValueError(f"label mapping mismatch for sample {row.sample_id}")

    with zipfile.ZipFile(config.image_zip, "r") as archive:
        members = set(archive.namelist())
    missing_members = sorted(set(frame["packaged_path"]).difference(members))
    if missing_members:
        preview = missing_members[:3]
        raise ValueError(f"{len(missing_members)} paths are missing from image ZIP: {preview}")

    frame = frame.sort_values(["sample_id", "packaged_path"], kind="stable").reset_index(drop=True)
    return ValidationBundle(
        frame=frame,
        class_names=class_names,
        manifest_sha256=sha256_file(config.validation_csv),
        class_mapping_sha256=sha256_file(config.class_mapping_json),
    )


def resolve_target_layers(model_key: str, model: nn.Module) -> list[tuple[str, nn.Module]]:
    if model_key == WAVESTARNET_KEY:
        stages = getattr(model, "stages", None)
        if stages is None or len(stages) != 4:
            raise ValueError("WavestarNet must expose exactly four stages")
        return [
            ("stage_2", stages[1]),
            ("stage_3", stages[2]),
            ("stage_4", stages[3]),
        ]
    if model_key == MOBILEVIT_KEY:
        final_conv = getattr(model, "final_conv", None)
        if not isinstance(final_conv, nn.Module):
            raise ValueError("MobileViT-S must expose final_conv")
        return [("final_conv", final_conv)]
    raise ValueError(f"unsupported analysis model: {model_key}")


def load_state_dict_strict(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    incompatible = model.load_state_dict(dict(state), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def _checkpoint_state(payload: bytes) -> Mapping[str, torch.Tensor]:
    loaded = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if isinstance(loaded, Mapping) and "state_dict" in loaded:
        loaded = loaded["state_dict"]
    if not isinstance(loaded, Mapping):
        raise TypeError("checkpoint must contain a state-dict mapping")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in loaded.items()):
        raise TypeError("checkpoint state dict contains invalid keys or values")
    return loaded


def _create_analysis_model(model_key: str, num_classes: int) -> nn.Module:
    if model_key == WAVESTARNET_KEY:
        return create_wavestarnet(model_key, num_classes=num_classes)
    if model_key == MOBILEVIT_KEY:
        import timm

        return timm.create_model(model_key, pretrained=False, num_classes=num_classes)
    raise ValueError(f"unsupported analysis model: {model_key}")


def load_analysis_models(config: AnalysisConfig) -> list[LoadedModel]:
    if not config.checkpoint_zip.is_file():
        raise FileNotFoundError(config.checkpoint_zip)
    bundle = load_locked_validation(config)
    display_names = {
        WAVESTARNET_KEY: "WavestarNet",
        MOBILEVIT_KEY: "MobileViT-S",
    }
    loaded_models: list[LoadedModel] = []
    with zipfile.ZipFile(config.checkpoint_zip, "r") as archive:
        available = set(archive.namelist())
        for model_key in (WAVESTARNET_KEY, MOBILEVIT_KEY):
            member = CHECKPOINT_MEMBERS[model_key]
            if member not in available:
                raise FileNotFoundError(f"checkpoint archive member not found: {member}")
            payload = archive.read(member)
            model = _create_analysis_model(model_key, len(bundle.class_names))
            load_state_dict_strict(model, _checkpoint_state(payload))
            model.eval()
            loaded_models.append(
                LoadedModel(
                    key=model_key,
                    display_name=display_names[model_key],
                    model=model,
                    target_layers=resolve_target_layers(model_key, model),
                    checkpoint_member=member,
                    checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
                )
            )
    return loaded_models


def normalize_cam(cam: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    value = cam.detach().float()
    if not torch.isfinite(value).all():
        raise ValueError("Grad-CAM contains non-finite values")
    minimum = value.min()
    maximum = value.max()
    span = maximum - minimum
    if float(span) <= epsilon:
        return torch.zeros_like(value)
    return (value - minimum) / span


def fuse_normalized_cams(
    cams: list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    output_size: tuple[int, int],
) -> torch.Tensor:
    if not cams:
        raise ValueError("at least one Grad-CAM map is required")
    normalized: list[torch.Tensor] = []
    for cam in cams:
        value = cam.detach().float()
        if value.ndim != 2:
            raise ValueError("each Grad-CAM map must be two-dimensional")
        if tuple(value.shape) != tuple(output_size):
            value = F.interpolate(
                value[None, None],
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        normalized.append(normalize_cam(value))
    return torch.stack(normalized, dim=0).mean(dim=0)


def compute_gradcam(
    model: nn.Module,
    target_layers: list[tuple[str, nn.Module]],
    input_tensor: torch.Tensor,
    *,
    target_class: int,
) -> GradCAMResult:
    if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
        raise ValueError("Grad-CAM requires an NCHW tensor with batch size one")
    if not target_layers:
        raise ValueError("at least one target layer is required")

    activations: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def capture(name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
            if not torch.is_tensor(output) or output.ndim != 4:
                raise ValueError(f"target layer {name} must produce an NCHW tensor")
            activations[name] = output

        return hook

    try:
        for name, layer in target_layers:
            if name in activations:
                raise ValueError(f"duplicate target-layer name: {name}")
            handles.append(layer.register_forward_hook(capture(name)))
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(input_tensor)
            if logits.ndim != 2 or logits.shape[0] != 1:
                raise ValueError("model must produce one two-dimensional logits tensor")
            if target_class not in range(logits.shape[1]):
                raise ValueError(f"target class is out of range: {target_class}")
            missing = [name for name, _ in target_layers if name not in activations]
            if missing:
                raise RuntimeError(f"target layers were not executed: {missing}")
            ordered_activations = tuple(activations[name] for name, _ in target_layers)
            gradients = torch.autograd.grad(
                logits[0, target_class],
                ordered_activations,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    layer_cams: dict[str, torch.Tensor] = {}
    for (name, _), activation, gradient in zip(
        target_layers,
        ordered_activations,
        gradients,
    ):
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=tuple(input_tensor.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        layer_cams[name] = normalize_cam(cam).cpu()

    fused_cam = fuse_normalized_cams(
        list(layer_cams.values()),
        output_size=tuple(input_tensor.shape[-2:]),
    ).cpu()
    return GradCAMResult(
        target_class=int(target_class),
        logits=logits.detach().cpu(),
        fused_cam=fused_cam,
        layer_cams=layer_cams,
    )


def _cams_from_gradients(
    target_layers: list[tuple[str, nn.Module]],
    activations: tuple[torch.Tensor, ...],
    gradients: tuple[torch.Tensor, ...],
    output_size: tuple[int, int],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    layer_cams: dict[str, torch.Tensor] = {}
    for (name, _), activation, gradient in zip(target_layers, activations, gradients):
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        layer_cams[name] = normalize_cam(cam).cpu()
    fused = fuse_normalized_cams(
        list(layer_cams.values()),
        output_size=output_size,
    ).cpu()
    return layer_cams, fused


def compute_prediction_gradcams(
    model: nn.Module,
    target_layers: list[tuple[str, nn.Module]],
    input_tensor: torch.Tensor,
    *,
    ground_truth_class: int,
) -> PredictionGradCAMResult:
    if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
        raise ValueError("Grad-CAM requires an NCHW tensor with batch size one")
    names = [name for name, _ in target_layers]
    if not names:
        raise ValueError("at least one target layer is required")
    if len(names) != len(set(names)):
        raise ValueError("target-layer names must be unique")

    activations_by_name: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def capture(name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
            if not torch.is_tensor(output) or output.ndim != 4:
                raise ValueError(f"target layer {name} must produce an NCHW tensor")
            activations_by_name[name] = output

        return hook

    try:
        for name, layer in target_layers:
            handles.append(layer.register_forward_hook(capture(name)))
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(input_tensor)
            if logits.ndim != 2 or logits.shape[0] != 1:
                raise ValueError("model must produce one two-dimensional logits tensor")
            if ground_truth_class not in range(logits.shape[1]):
                raise ValueError(f"ground-truth class is out of range: {ground_truth_class}")
            missing = [name for name in names if name not in activations_by_name]
            if missing:
                raise RuntimeError(f"target layers were not executed: {missing}")
            activations = tuple(activations_by_name[name] for name in names)
            predicted_class = int(logits.argmax(dim=1).item())
            target_classes = list(dict.fromkeys([predicted_class, int(ground_truth_class)]))
            results_by_class: dict[int, GradCAMResult] = {}
            for index, target_class in enumerate(target_classes):
                gradients = torch.autograd.grad(
                    logits[0, target_class],
                    activations,
                    retain_graph=index < len(target_classes) - 1,
                    create_graph=False,
                    allow_unused=False,
                )
                layer_cams, fused_cam = _cams_from_gradients(
                    target_layers,
                    activations,
                    gradients,
                    tuple(input_tensor.shape[-2:]),
                )
                results_by_class[target_class] = GradCAMResult(
                    target_class=target_class,
                    logits=logits.detach().cpu(),
                    fused_cam=fused_cam,
                    layer_cams=layer_cams,
                )
    finally:
        for handle in handles:
            handle.remove()

    return PredictionGradCAMResult(
        logits=logits.detach().cpu(),
        predicted_class=predicted_class,
        cams={
            "predicted": results_by_class[predicted_class],
            "ground_truth": results_by_class[int(ground_truth_class)],
        },
    )


def heatmap_diagnostics(cam: torch.Tensor) -> dict[str, float | bool]:
    array = cam.detach().cpu().float().numpy()
    finite = bool(np.isfinite(array).all())
    finite_values = array[np.isfinite(array)]
    if finite_values.size:
        minimum = float(finite_values.min())
        maximum = float(finite_values.max())
        mean = float(finite_values.mean())
        std = float(finite_values.std())
        constant = bool(finite and np.ptp(finite_values) <= 1e-12)
    else:
        minimum = maximum = mean = std = float("nan")
        constant = False
    return {
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "std": std,
        "nonzero_fraction": float(np.count_nonzero(array) / array.size),
        "finite": finite,
        "constant": constant,
    }


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def prepare_evaluation_image(
    image: Image.Image,
    image_size: int = 224,
) -> tuple[torch.Tensor, Image.Image]:
    if image_size != 224:
        raise ValueError("the locked evaluation image size must be 224")
    resized = TF.resize(
        image.convert("RGB"),
        256,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    cropped = TF.center_crop(resized, [image_size, image_size])
    tensor = TF.to_tensor(cropped)
    tensor = TF.normalize(
        tensor,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return tensor.unsqueeze(0), cropped


def _safe_stem(sample_id: str, packaged_path: str) -> str:
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in sample_id)
    digest = hashlib.sha256(packaged_path.encode("utf-8")).hexdigest()[:10]
    return f"{safe}_{digest}"


def _relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _save_png_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def _save_heatmap(cam: torch.Tensor, path: Path) -> None:
    array = np.clip(cam.detach().cpu().numpy(), 0.0, 1.0)
    _save_png_atomic(Image.fromarray(np.uint8(np.round(array * 255.0)), mode="L"), path)


def _save_overlay(base_image: Image.Image, cam: torch.Tensor, path: Path) -> None:
    import matplotlib

    image_array = np.asarray(base_image.convert("RGB"), dtype=np.float32) / 255.0
    cam_array = np.clip(cam.detach().cpu().numpy(), 0.0, 1.0)
    colored = matplotlib.colormaps["turbo"](cam_array)[..., :3]
    alpha = (0.15 + 0.40 * cam_array)[..., None]
    overlay = np.clip((1.0 - alpha) * image_array + alpha * colored, 0.0, 1.0)
    _save_png_atomic(Image.fromarray(np.uint8(np.round(overlay * 255.0)), mode="RGB"), path)


def _write_json_atomic(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _artifact_paths(
    output_dir: Path,
    model_key: str,
    role: str,
    stem: str,
) -> tuple[Path, Path]:
    return (
        output_dir / "heatmaps" / model_key / role / f"{stem}.png",
        output_dir / "overlays" / model_key / role / f"{stem}.png",
    )


def _record_artifacts_exist(record: dict[str, object], output_dir: Path) -> bool:
    required = (
        "predicted_heatmap",
        "ground_truth_heatmap",
        "predicted_overlay",
        "ground_truth_overlay",
    )
    if record.get("artifact_status") != "complete":
        return False
    try:
        paths = [output_dir / str(record[name]) for name in required]
    except KeyError:
        return False
    for path in paths:
        if not path.is_file():
            return False
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError):
            return False
    stage_paths = json.loads(str(record.get("stage_heatmaps", "{}")))
    return all((output_dir / value).is_file() for value in stage_paths.values())


def select_cases(predictions: pd.DataFrame, per_category: int = 4) -> pd.DataFrame:
    if per_category < 1:
        raise ValueError("per_category must be positive")
    required = {"sample_id", "model_key", "correct", "confidence", "label"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"case selection is missing columns: {sorted(missing)}")
    keys = (WAVESTARNET_KEY, MOBILEVIT_KEY)
    available = set(predictions["model_key"].astype(str))
    if not set(keys).issubset(available):
        raise ValueError("case selection requires WavestarNet and MobileViT-S")

    ordered = predictions.sort_values(["sample_id", "model_key"], kind="stable")
    metadata_columns = ["sample_id", "label"]
    if "packaged_path" in ordered.columns:
        metadata_columns.append("packaged_path")
    base = ordered[metadata_columns].drop_duplicates("sample_id").set_index("sample_id")
    correct = ordered.pivot(index="sample_id", columns="model_key", values="correct").astype(bool)
    confidence = ordered.pivot(index="sample_id", columns="model_key", values="confidence").astype(float)
    table = base.join(correct.add_prefix("correct__")).join(confidence.add_prefix("confidence__"))
    wave_correct = table[f"correct__{WAVESTARNET_KEY}"]
    mobile_correct = table[f"correct__{MOBILEVIT_KEY}"]
    wave_conf = table[f"confidence__{WAVESTARNET_KEY}"]
    mobile_conf = table[f"confidence__{MOBILEVIT_KEY}"]
    table["mean_confidence"] = (wave_conf + mobile_conf) / 2.0
    table["confidence_margin"] = (wave_conf - mobile_conf).abs()

    selected: list[dict[str, object]] = []

    def add(category: str, subset: pd.DataFrame, sort_column: str, ascending: bool) -> None:
        ranked = subset.reset_index().sort_values(
            [sort_column, "sample_id"], ascending=[ascending, True], kind="stable"
        ).head(per_category)
        for rank, row in enumerate(ranked.itertuples(index=False), start=1):
            selected.append(
                {
                    "category": category,
                    "rank": rank,
                    "sample_id": str(row.sample_id),
                    "label": str(row.label),
                    "selection_score": float(getattr(row, sort_column)),
                }
            )

    add("both_correct", table[wave_correct & mobile_correct], "mean_confidence", False)
    add("wavestarnet_only_correct", table[wave_correct & ~mobile_correct], "confidence_margin", False)
    add("mobilevit_only_correct", table[~wave_correct & mobile_correct], "confidence_margin", False)
    add("both_wrong", table[~wave_correct & ~mobile_correct], "mean_confidence", False)
    add("wavestarnet_low_confidence_correct", table[wave_correct], f"confidence__{WAVESTARNET_KEY}", True)
    add("mobilevit_low_confidence_correct", table[mobile_correct], f"confidence__{MOBILEVIT_KEY}", True)
    add("wavestarnet_high_confidence_error", table[~wave_correct], f"confidence__{WAVESTARNET_KEY}", False)
    add("mobilevit_high_confidence_error", table[~mobile_correct], f"confidence__{MOBILEVIT_KEY}", False)

    label_support = table.groupby("label", sort=True).size().sort_values(kind="stable")
    rare_labels = set(label_support.head(min(3, len(label_support))).index)
    add("rare_classes", table[table["label"].isin(rare_labels)], "mean_confidence", True)
    alternaria = table[table["label"].str.contains("alternaria", case=False, na=False)]
    add("alternaria", alternaria, "mean_confidence", True)

    model_correctness = pd.concat([wave_correct, mobile_correct], axis=1).mean(axis=1)
    difficulty = model_correctness.groupby(table["label"]).mean().sort_values(kind="stable")
    difficult_labels = set(difficulty.head(min(3, len(difficulty))).index)
    add("difficult_classes", table[table["label"].isin(difficult_labels)], "mean_confidence", False)

    columns = ["category", "rank", "sample_id", "label", "selection_score"]
    result = pd.DataFrame(selected, columns=columns)
    if result.empty:
        return result
    return result.drop_duplicates(["category", "sample_id"]).sort_values(
        ["category", "rank", "sample_id"], kind="stable"
    ).reset_index(drop=True)


def _metric_tables(
    predictions: pd.DataFrame,
    class_names: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    per_class_rows: list[dict[str, object]] = []
    aggregate: dict[str, dict[str, float]] = {}
    labels = list(range(len(class_names)))
    for model_key, frame in predictions.groupby("model_key", sort=True):
        y_true = frame["label_id"].astype(int).to_numpy()
        y_pred = frame["predicted_label_id"].astype(int).to_numpy()
        aggregate[str(model_key)] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        }
        precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        support = np.bincount(y_true, minlength=len(labels))
        for label_id, name in enumerate(class_names):
            per_class_rows.append(
                {
                    "model_key": str(model_key),
                    "label_id": label_id,
                    "label": name,
                    "support": int(support[label_id]),
                    "precision": float(precision[label_id]),
                    "recall": float(recall[label_id]),
                    "f1": float(f1[label_id]),
                }
            )
    return pd.DataFrame(per_class_rows), aggregate


def _save_confusion_matrices(
    predictions: pd.DataFrame,
    class_names: tuple[str, ...],
    output_dir: Path,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    outputs: dict[str, str] = {}
    labels = list(range(len(class_names)))
    for model_key, frame in predictions.groupby("model_key", sort=True):
        matrix = confusion_matrix(frame["label_id"], frame["predicted_label_id"], labels=labels)
        figure, axis = plt.subplots(figsize=(11, 9))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{model_key} validation confusion matrix")
        axis.set_xlabel("Predicted class ID")
        axis.set_ylabel("True class ID")
        axis.set_xticks(labels)
        axis.set_yticks(labels)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        path = output_dir / "figures" / f"confusion_matrix_{model_key}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        outputs[str(model_key)] = _relative_path(path, output_dir)
    return outputs


def _load_resume_records(path: Path, output_dir: Path) -> dict[tuple[str, str], dict[str, object]]:
    records: dict[tuple[str, str], dict[str, object]] = {}
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = (str(record["sample_id"]), str(record["model_key"]))
        if _record_artifacts_exist(record, output_dir):
            records[key] = record
    return records


def run_analysis(
    config: AnalysisConfig,
    *,
    limit: int | None = None,
    loaded_models: list[LoadedModel] | None = None,
) -> AnalysisSummary:
    seed_everything(config.seed)
    bundle = load_locked_validation(config)
    frame = bundle.frame if limit is None else bundle.frame.head(int(limit))
    if frame.empty:
        raise ValueError("analysis requires at least one validation image")
    models = load_analysis_models(config) if loaded_models is None else loaded_models
    if {model.key for model in models} != {WAVESTARNET_KEY, MOBILEVIT_KEY}:
        raise ValueError("analysis requires exactly WavestarNet and MobileViT-S")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "seed": config.seed,
        "image_size": config.image_size,
        "validation_count": int(len(frame)),
        "full_validation_count": int(len(bundle.frame)),
        "limit": limit,
        "model_keys": [model.key for model in models],
        "target_layers": {model.key: [name for name, _ in model.target_layers] for model in models},
    }
    _write_json_atomic(run_config, config.output_dir / "run_config.json")
    integrity = {
        "validation_manifest_sha256": bundle.manifest_sha256,
        "class_mapping_sha256": bundle.class_mapping_sha256,
        "image_zip_sha256": sha256_file(config.image_zip),
        "checkpoint_zip_sha256": sha256_file(config.checkpoint_zip),
        "checkpoints": {model.key: model.checkpoint_sha256 for model in models},
    }
    _write_json_atomic(integrity, config.output_dir / "integrity_report.json")

    expected_keys = {
        (str(row.sample_id), model.key)
        for row in frame.itertuples(index=False)
        for model in models
    }
    journal = config.output_dir / "progress_records.jsonl"
    records = {
        key: value
        for key, value in _load_resume_records(journal, config.output_dir).items()
        if key in expected_keys
    }
    total = len(frame) * len(models)
    completed = len(records)
    resumed_record_count = completed
    records_processed_this_invocation = 0
    start = time.perf_counter()
    with ZipImageStore(config.image_zip) as store, journal.open("a", encoding="utf-8") as journal_handle:
        for row in frame.itertuples(index=False):
            image = store.open_rgb(str(row.packaged_path))
            input_tensor, display_image = prepare_evaluation_image(image, config.image_size)
            stem = _safe_stem(str(row.sample_id), str(row.packaged_path))
            for loaded in models:
                key = (str(row.sample_id), loaded.key)
                if key in records:
                    continue
                try:
                    result = compute_prediction_gradcams(
                        loaded.model,
                        loaded.target_layers,
                        input_tensor,
                        ground_truth_class=int(row.label_id),
                    )
                    probabilities = torch.softmax(result.logits[0].float(), dim=0)
                    predicted_id = result.predicted_class
                    confidence = float(probabilities[predicted_id])
                    entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
                    artifact_values: dict[str, str] = {}
                    stage_values: dict[str, str] = {}
                    diagnostic_values: dict[str, dict[str, float | bool]] = {}
                    for role, cam_result in result.cams.items():
                        heatmap_path, overlay_path = _artifact_paths(
                            config.output_dir, loaded.key, role, stem
                        )
                        _save_heatmap(cam_result.fused_cam, heatmap_path)
                        _save_overlay(display_image, cam_result.fused_cam, overlay_path)
                        artifact_values[f"{role}_heatmap"] = _relative_path(
                            heatmap_path, config.output_dir
                        )
                        artifact_values[f"{role}_overlay"] = _relative_path(
                            overlay_path, config.output_dir
                        )
                        diagnostic_values[role] = heatmap_diagnostics(cam_result.fused_cam)
                        for layer_name, layer_cam in cam_result.layer_cams.items():
                            stage_path = (
                                config.output_dir
                                / "heatmaps"
                                / loaded.key
                                / "stages"
                                / role
                                / layer_name
                                / f"{stem}.png"
                            )
                            _save_heatmap(layer_cam, stage_path)
                            stage_values[f"{role}:{layer_name}"] = _relative_path(
                                stage_path, config.output_dir
                            )
                    record: dict[str, object] = {
                        "sample_id": str(row.sample_id),
                        "packaged_path": str(row.packaged_path),
                        "label": str(row.label),
                        "label_id": int(row.label_id),
                        "model_key": loaded.key,
                        "model_name": loaded.display_name,
                        "predicted_label_id": predicted_id,
                        "predicted_label": bundle.class_names[predicted_id],
                        "confidence": confidence,
                        "entropy": entropy,
                        "correct": bool(predicted_id == int(row.label_id)),
                        "logits": json.dumps(result.logits[0].tolist()),
                        "probabilities": json.dumps(probabilities.tolist()),
                        "stage_heatmaps": json.dumps(stage_values, sort_keys=True),
                        "heatmap_diagnostics": json.dumps(diagnostic_values, sort_keys=True),
                        "artifact_status": "complete",
                        **artifact_values,
                    }
                except Exception as error:
                    error_record = {
                        "sample_id": str(row.sample_id),
                        "packaged_path": str(row.packaged_path),
                        "model_key": loaded.key,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    with (config.output_dir / "errors.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(error_record, sort_keys=True) + "\n")
                    raise
                records[key] = record
                journal_handle.write(json.dumps(record, sort_keys=True) + "\n")
                journal_handle.flush()
                completed += 1
                records_processed_this_invocation += 1
                if completed % config.progress_interval == 0 or completed == total:
                    elapsed = time.perf_counter() - start
                    new_count = max(1, completed)
                    rate = new_count / max(elapsed, 1e-12)
                    remaining = max(0, total - completed) / max(rate, 1e-12)
                    print(
                        json.dumps(
                            {
                                "event": "gradcam_progress",
                                "completed": completed,
                                "total": total,
                                "elapsed_seconds": round(elapsed, 3),
                                "records_per_second": round(rate, 4),
                                "estimated_remaining_seconds": round(remaining, 3),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    missing_records = expected_keys.difference(records)
    if missing_records:
        raise RuntimeError(f"analysis is missing {len(missing_records)} prediction records")
    predictions = pd.DataFrame([records[key] for key in sorted(expected_keys)])
    predictions_csv = config.output_dir / "predictions.csv"
    _write_csv_atomic(predictions, predictions_csv)
    per_class, aggregate = _metric_tables(predictions, bundle.class_names)
    per_class_csv = config.output_dir / "per_class_metrics.csv"
    _write_csv_atomic(per_class, per_class_csv)
    cases = select_cases(predictions, config.inline_examples_per_category)
    cases_csv = config.output_dir / "case_selection.csv"
    _write_csv_atomic(cases, cases_csv)
    confusion_paths = _save_confusion_matrices(predictions, bundle.class_names, config.output_dir)
    agreement = float(
        predictions.pivot(index="sample_id", columns="model_key", values="predicted_label_id")
        .nunique(axis=1)
        .eq(1)
        .mean()
    )
    processing_elapsed_seconds = time.perf_counter() - start
    existing_summary_path = config.output_dir / "analysis_summary.json"
    if records_processed_this_invocation == 0 and existing_summary_path.is_file():
        previous_summary = json.loads(existing_summary_path.read_text(encoding="utf-8"))
        processing_elapsed_seconds = float(
            previous_summary.get("processing_elapsed_seconds", processing_elapsed_seconds)
        )
    png_artifact_count = sum(1 for _ in config.output_dir.rglob("*.png"))
    result_bytes = sum(
        path.stat().st_size for path in config.output_dir.rglob("*") if path.is_file()
    )
    summary_payload = {
        "validation_count": int(len(frame)),
        "record_count": int(len(predictions)),
        "records_processed_this_invocation": records_processed_this_invocation,
        "resumed_record_count": resumed_record_count,
        "processing_elapsed_seconds": processing_elapsed_seconds,
        "png_artifact_count": png_artifact_count,
        "result_bytes_before_inline_case_grids": result_bytes,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "timm": importlib.metadata.version("timm"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "matplotlib": importlib.metadata.version("matplotlib"),
        },
        "metrics": aggregate,
        "prediction_agreement": agreement,
        "confusion_matrices": confusion_paths,
        "constant_heatmap_records": int(
            predictions["heatmap_diagnostics"].map(
                lambda value: any(item["constant"] for item in json.loads(value).values())
            ).sum()
        ),
    }
    summary_json = existing_summary_path
    _write_json_atomic(summary_payload, summary_json)
    return AnalysisSummary(
        predictions_csv=predictions_csv,
        per_class_metrics_csv=per_class_csv,
        case_selection_csv=cases_csv,
        summary_json=summary_json,
        validation_count=len(frame),
        record_count=len(predictions),
    )


def verify_real_inputs(
    config: AnalysisConfig,
    *,
    sample_limit: int = 2,
) -> pd.DataFrame:
    if sample_limit < 1:
        raise ValueError("sample_limit must be positive")
    seed_everything(config.seed)
    bundle = load_locked_validation(config)
    models = load_analysis_models(config)
    reference_frames: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(config.checkpoint_zip, "r") as archive:
        for key, member in REFERENCE_PREDICTION_MEMBERS.items():
            reference_frames[key] = pd.read_csv(io.BytesIO(archive.read(member))).set_index(
                "sample_id"
            )

    verification_rows: list[dict[str, object]] = []
    with ZipImageStore(config.image_zip) as store:
        for row in bundle.frame.head(sample_limit).itertuples(index=False):
            tensor, _ = prepare_evaluation_image(
                store.open_rgb(str(row.packaged_path)), config.image_size
            )
            for loaded in models:
                result = compute_prediction_gradcams(
                    loaded.model,
                    loaded.target_layers,
                    tensor,
                    ground_truth_class=int(row.label_id),
                )
                probabilities = torch.softmax(result.logits[0].float(), dim=0).numpy()
                reference = reference_frames[loaded.key].loc[str(row.sample_id)]
                reference_probabilities = np.array(
                    [float(reference[f"prob_{index:02d}"]) for index in range(len(bundle.class_names))]
                )
                maximum_probability_error = float(
                    np.max(np.abs(probabilities - reference_probabilities))
                )
                if result.predicted_class != int(reference["predicted_label_id"]):
                    raise RuntimeError(
                        f"prediction mismatch for {loaded.key}/{row.sample_id}: "
                        f"actual={result.predicted_class}, "
                        f"reference={int(reference['predicted_label_id'])}"
                    )
                if maximum_probability_error > 1e-5:
                    raise RuntimeError(
                        f"probability mismatch for {loaded.key}/{row.sample_id}: "
                        f"max_abs_error={maximum_probability_error}"
                    )
                verification_rows.append(
                    {
                        "sample_id": str(row.sample_id),
                        "model_key": loaded.key,
                        "checkpoint_sha256": loaded.checkpoint_sha256,
                        "target_layers": ", ".join(name for name, _ in loaded.target_layers),
                        "input_shape": str(tuple(tensor.shape)),
                        "cam_shape": str(tuple(result.cams["predicted"].fused_cam.shape)),
                        "predicted_label_id": result.predicted_class,
                        "reference_prediction_match": True,
                        "maximum_probability_error": maximum_probability_error,
                        "predicted_cam_finite": heatmap_diagnostics(
                            result.cams["predicted"].fused_cam
                        )["finite"],
                        "ground_truth_cam_finite": heatmap_diagnostics(
                            result.cams["ground_truth"].fused_cam
                        )["finite"],
                    }
                )
    return pd.DataFrame(verification_rows)


def display_analysis_report(summary: AnalysisSummary) -> dict[str, object]:
    import matplotlib.pyplot as plt
    from IPython.display import display

    output_dir = summary.summary_json.parent
    payload = json.loads(summary.summary_json.read_text(encoding="utf-8"))
    predictions = pd.read_csv(summary.predictions_csv)
    per_class = pd.read_csv(summary.per_class_metrics_csv)
    cases = pd.read_csv(summary.case_selection_csv)

    metrics = pd.DataFrame(payload["metrics"]).T.reset_index(names="model_key")
    print("Aggregate validation metrics", flush=True)
    display(metrics.style.format(precision=4))

    print("Lowest per-class F1 values", flush=True)
    lowest = per_class.sort_values(["model_key", "f1", "support", "label"], kind="stable")
    display(lowest.groupby("model_key", sort=True).head(8).reset_index(drop=True))

    for model_key, relative in payload["confusion_matrices"].items():
        with Image.open(output_dir / relative) as image:
            figure, axis = plt.subplots(figsize=(10, 8))
            axis.imshow(image)
            axis.set_title(f"{model_key}: locked-validation confusion matrix")
            axis.axis("off")
            plt.show()
            plt.close(figure)

    if not cases.empty:
        prediction_index = predictions.set_index(["sample_id", "model_key"])
        for category, category_cases in cases.groupby("category", sort=True):
            rows = len(category_cases)
            figure, axes = plt.subplots(rows, 2, figsize=(10, max(3.2, 3.1 * rows)), squeeze=False)
            for row_index, case in enumerate(category_cases.itertuples(index=False)):
                for column, model_key in enumerate((WAVESTARNET_KEY, MOBILEVIT_KEY)):
                    prediction = prediction_index.loc[(str(case.sample_id), model_key)]
                    overlay_path = output_dir / str(prediction["predicted_overlay"])
                    with Image.open(overlay_path) as overlay:
                        axes[row_index, column].imshow(overlay)
                    title = (
                        f"{model_key}\ntrue: {prediction['label']}\n"
                        f"pred: {prediction['predicted_label']} "
                        f"({float(prediction['confidence']):.3f})"
                    )
                    axes[row_index, column].set_title(title, fontsize=8)
                    axes[row_index, column].axis("off")
            figure.suptitle(category.replace("_", " ").title(), fontsize=12)
            figure.tight_layout()
            figure_path = output_dir / "figures" / f"case_grid_{category}.png"
            figure.savefig(figure_path, dpi=180, bbox_inches="tight")
            plt.show()
            plt.close(figure)

    diagnostics = predictions["heatmap_diagnostics"].map(json.loads)
    diagnostic_summary = pd.DataFrame(
        {
            "model_key": predictions["model_key"],
            "predicted_constant": diagnostics.map(lambda item: item["predicted"]["constant"]),
            "ground_truth_constant": diagnostics.map(
                lambda item: item["ground_truth"]["constant"]
            ),
            "predicted_nonzero_fraction": diagnostics.map(
                lambda item: item["predicted"]["nonzero_fraction"]
            ),
            "ground_truth_nonzero_fraction": diagnostics.map(
                lambda item: item["ground_truth"]["nonzero_fraction"]
            ),
        }
    )
    print("Heatmap diagnostics by model", flush=True)
    display(
        diagnostic_summary.groupby("model_key", sort=True).agg(
            predicted_constant_maps=("predicted_constant", "sum"),
            ground_truth_constant_maps=("ground_truth_constant", "sum"),
            predicted_nonzero_fraction_mean=("predicted_nonzero_fraction", "mean"),
            ground_truth_nonzero_fraction_mean=("ground_truth_nonzero_fraction", "mean"),
        )
    )
    inventory = {
        "validation_images": summary.validation_count,
        "prediction_records": summary.record_count,
        "selected_inline_cases": int(len(cases)),
        "prediction_agreement": float(payload["prediction_agreement"]),
        "output_directory": str(output_dir),
    }
    payload["png_artifact_count"] = sum(1 for _ in output_dir.rglob("*.png"))
    payload["result_bytes"] = sum(
        path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
    )
    _write_json_atomic(payload, summary.summary_json)
    inventory["png_artifact_count"] = payload["png_artifact_count"]
    inventory["result_bytes"] = payload["result_bytes"]
    print(json.dumps(inventory, indent=2, sort_keys=True), flush=True)
    return inventory
