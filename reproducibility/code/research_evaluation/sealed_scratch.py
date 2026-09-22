"""Fail-closed final-holdout evaluation for PlantDoc and PlantWild scratch runs."""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import math
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import t as student_t
from torch.utils.data import DataLoader, Dataset

from agrivision4.evaluation.sealed_test import compute_classification_metrics


SEEDS = (42, 2024, 3407)
MODELS = (
    "wavestarnet_big_transfer_v2",
    "efficientnet_b0",
    "efficientnet_b3",
    "mobilenetv3_large_100",
    "mobilevit_s",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class EvaluationSpec:
    repo_root: Path
    dataset: str
    evaluation_id: str
    acknowledgement: str
    protocol_id: str
    dataset_root: Path
    test_manifest: Path
    train_manifest: Path
    validation_manifest: Path
    mapping_path: Path
    image_root: Path | None
    image_archive: Path | None
    image_path_column: str
    image_hash_column: str
    num_classes: int
    test_samples: int
    runtime_module: str
    core_module: str
    scratch_results_root: Path
    registry_path: Path
    output_root: Path

    @property
    def output_directory(self) -> Path:
        return self.output_root / self.evaluation_id

    @property
    def ledger_path(self) -> Path:
        return self.output_root / f".{self.evaluation_id}.access.json"


def build_spec(repo_root: str | Path, dataset: str) -> EvaluationSpec:
    root = Path(repo_root).resolve()
    if dataset == "plantdoc":
        return EvaluationSpec(
            repo_root=root,
            dataset=dataset,
            evaluation_id="plantdoc_scratch_final_holdout_v1",
            acknowledgement="RUN_PLANTDOC_SCRATCH_FINAL_HOLDOUT_ONCE",
            protocol_id="plantdoc_scratch_comparison_v1",
            dataset_root=root / "plantdoc/datasets/plantdoc-clean-grouped-v1",
            train_manifest=root / "plantdoc/datasets/plantdoc-clean-grouped-v1/manifests/train.csv",
            validation_manifest=root / "plantdoc/datasets/plantdoc-clean-grouped-v1/manifests/validation.csv",
            test_manifest=root / "plantdoc/datasets/plantdoc-clean-grouped-v1/manifests/test.csv",
            mapping_path=root / "plantdoc/datasets/plantdoc-clean-grouped-v1/manifests/class_mapping.json",
            image_root=None,
            image_archive=root / "plantdoc/datasets/plantdoc-clean-grouped-v1/plantdoc-clean-grouped-images.zip",
            image_path_column="packaged_path",
            image_hash_column="sha256",
            num_classes=27,
            test_samples=430,
            runtime_module="plantdoc.scratch_comparison.notebook_runtime",
            core_module="plantdoc.scratch_comparison.experiment_core",
            scratch_results_root=root / "plantdoc/results/scratch comparison",
            registry_path=root / "plantdoc/evaluation/plantdoc_scratch_checkpoint_registry.json",
            output_root=root / "plantdoc/results/sealed_test",
        )
    if dataset == "plantwild":
        return EvaluationSpec(
            repo_root=root,
            dataset=dataset,
            evaluation_id="plantwild_scratch_final_holdout_v1",
            acknowledgement="RUN_PLANTWILD_SCRATCH_FINAL_HOLDOUT_ONCE",
            protocol_id="plantwild_scratch_comparison_v1",
            dataset_root=root / "plantwild/audit/plantwild-v2-audit-v1",
            train_manifest=root / "plantwild/kaggle/plantwild-clean-training-v1/manifests/train.csv",
            validation_manifest=root / "plantwild/kaggle/plantwild-clean-training-v1/manifests/validation.csv",
            test_manifest=root / "plantwild/audit/plantwild-v2-audit-v1/manifests/test.csv",
            mapping_path=root / "plantwild/kaggle/plantwild-clean-training-v1/manifests/class_mapping.json",
            image_root=root / "plantwild/dataset/plantwild_v2",
            image_archive=None,
            image_path_column="source_relative_path",
            image_hash_column="file_sha256",
            num_classes=105,
            test_samples=1523,
            runtime_module="plantwild.scratch_comparison.notebook_runtime",
            core_module="plantwild.scratch_comparison.experiment_core",
            scratch_results_root=root / "plantwild/results/scratch comparison",
            registry_path=root / "plantwild/evaluation/plantwild_scratch_checkpoint_registry.json",
            output_root=root / "plantwild/results/sealed_test",
        )
    raise ValueError("dataset must be plantdoc or plantwild")


def _require(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} missing: {path}")


def _mapping(path: Path, num_classes: int) -> tuple[tuple[str, ...], dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    class_to_id = payload.get("class_to_id", payload)
    class_to_id = {str(label): int(index) for label, index in class_to_id.items()}
    if sorted(class_to_id.values()) != list(range(num_classes)):
        raise ValueError("class mapping is not contiguous")
    names = tuple(label for label, _ in sorted(class_to_id.items(), key=lambda item: item[1]))
    return names, class_to_id


def _manifest_fields(spec: EvaluationSpec) -> tuple[str, ...]:
    return ("sample_id", "component_id", "split", "label", "label_id", spec.image_path_column, spec.image_hash_column)


def _validate_manifest(
    frame: pd.DataFrame,
    *,
    name: str,
    expected_split: str,
    class_to_id: dict[str, int],
    spec: EvaluationSpec,
    expected_rows: int | None = None,
) -> None:
    missing = set(_manifest_fields(spec)) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing fields: {sorted(missing)}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"{name} row count mismatch")
    if frame.empty or frame["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"{name} has invalid sample IDs")
    if set(frame["split"].astype(str)) != {expected_split}:
        raise ValueError(f"{name} split role mismatch")
    if frame["component_id"].astype(str).duplicated().any():
        raise ValueError(f"{name} has duplicate components")
    for row in frame.itertuples(index=False):
        if class_to_id.get(str(row.label)) != int(row.label_id):
            raise ValueError(f"{name} class mapping mismatch: {row.sample_id}")


def _overlap(first: pd.DataFrame, second: pd.DataFrame, hash_column: str) -> dict[str, int]:
    return {
        column: len(set(first[column].astype(str)) & set(second[column].astype(str)))
        for column in ("sample_id", "component_id", hash_column)
    }


def _load_registry(spec: EvaluationSpec) -> dict[str, Any]:
    _require(spec.registry_path, "frozen checkpoint registry")
    registry = json.loads(spec.registry_path.read_text(encoding="utf-8"))
    if registry.get("evaluation_id") != spec.evaluation_id:
        raise ValueError("checkpoint registry evaluation ID mismatch")
    if registry.get("protocol_id") != spec.protocol_id:
        raise ValueError("checkpoint registry protocol mismatch")
    if tuple(registry.get("seeds", ())) != SEEDS:
        raise ValueError("checkpoint registry seed mismatch")
    if tuple(registry.get("models", ())) != MODELS:
        raise ValueError("checkpoint registry model set mismatch")
    return registry


def validate_protocol(spec: EvaluationSpec) -> dict[str, Any]:
    registry = _load_registry(spec)
    files = {
        "train": spec.train_manifest,
        "validation": spec.validation_manifest,
        "test": spec.test_manifest,
        "class_mapping": spec.mapping_path,
    }
    if spec.image_archive is not None:
        files["image_archive"] = spec.image_archive
    for name, path in files.items():
        _require(path, name)
    observed = {name: _sha256(path) for name, path in files.items() if name != "image_archive"}
    expected = registry["manifest_sha256"]
    if observed != expected:
        raise ValueError(f"frozen manifest identity mismatch: {observed}")
    class_names, class_to_id = _mapping(spec.mapping_path, spec.num_classes)
    train, validation, test = (pd.read_csv(files[name]) for name in ("train", "validation", "test"))
    _validate_manifest(train, name="train", expected_split="train", class_to_id=class_to_id, spec=spec)
    _validate_manifest(validation, name="validation", expected_split="validation", class_to_id=class_to_id, spec=spec)
    _validate_manifest(test, name="test", expected_split="test", class_to_id=class_to_id, spec=spec, expected_rows=spec.test_samples)
    if set(test["label_id"].astype(int)) != set(range(spec.num_classes)):
        raise ValueError("sealed test does not cover all classes")
    overlaps = {
        "train_vs_validation": _overlap(train, validation, spec.image_hash_column),
        "train_vs_test": _overlap(train, test, spec.image_hash_column),
        "validation_vs_test": _overlap(validation, test, spec.image_hash_column),
    }
    if any(value for pair in overlaps.values() for value in pair.values()):
        raise ValueError(f"split overlap detected: {overlaps}")
    if spec.image_archive is not None and _sha256(spec.image_archive) != registry["image_archive_sha256"]:
        raise ValueError("image archive identity mismatch")
    support = test.groupby("label_id").size()
    return {
        "test_samples": len(test),
        "class_count": len(class_names),
        "minimum_class_support": int(support.min()),
        "maximum_class_support": int(support.max()),
        "overlap": overlaps,
        "manifest_sha256": observed,
        "registry_sha256": _sha256(spec.registry_path),
    }


@dataclass(frozen=True)
class Checkpoint:
    model_key: str
    display_name: str
    seed: int
    parameters: int
    checkpoint_path: Path
    checkpoint_sha256: str
    architecture_sha256: str
    backbone_schema_sha256: str | None


def _artifact_hashes(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["path"]): str(item["sha256"]) for item in payload}


def discover_checkpoints(spec: EvaluationSpec) -> list[Checkpoint]:
    registry = _load_registry(spec)
    records: list[Checkpoint] = []
    for model_key in MODELS:
        frozen = registry["frozen_models"][model_key]
        run_root = spec.repo_root / frozen["run_root"]
        files = {
            "run_manifest_sha256": run_root / "run_manifest.json",
            "model_summary_sha256": run_root / "model_summary.json",
            "artifact_inventory_sha256": run_root / "artifact_inventory.json",
            "validation_seed_metrics_sha256": run_root / "validation_seed_metrics.csv",
        }
        for key, path in files.items():
            _require(path, f"{model_key} {key}")
            if _sha256(path) != frozen[key]:
                raise ValueError(f"{model_key} frozen {key} mismatch")
        manifest = json.loads(files["run_manifest_sha256"].read_text(encoding="utf-8"))
        summary = json.loads(files["model_summary_sha256"].read_text(encoding="utf-8"))
        inventory = _artifact_hashes(files["artifact_inventory_sha256"])
        validation = pd.read_csv(files["validation_seed_metrics_sha256"])
        if any((manifest.get("status") != "complete", manifest.get("protocol_id") != spec.protocol_id, manifest.get("model_key") != model_key, manifest.get("run_mode") != "development", manifest.get("initialization") != "random", bool(manifest.get("pretrained")), manifest.get("test_accessed") is not False)):
            raise ValueError(f"{model_key} does not satisfy frozen scratch contract")
        if tuple(manifest.get("active_seeds", ())) != SEEDS or len(validation) != len(SEEDS):
            raise ValueError(f"{model_key} seed contract mismatch")
        for seed in SEEDS:
            checkpoint = run_root / f"seed_{seed}/best_checkpoint.pt"
            sidecar = run_root / f"seed_{seed}/best_checkpoint.sha256"
            _require(checkpoint, f"{model_key} checkpoint")
            _require(sidecar, f"{model_key} checkpoint checksum")
            expected = str(frozen["checkpoints"][str(seed)])
            observed = _sha256(checkpoint)
            row = validation.loc[validation["seed"].astype(int) == seed]
            identities = {expected, observed, sidecar.read_text(encoding="utf-8").strip(), inventory.get(f"seed_{seed}/best_checkpoint.pt"), str(row.iloc[0]["checkpoint_sha256"]) if len(row) == 1 else None}
            if identities != {expected}:
                raise ValueError(f"{model_key} seed {seed} checkpoint identity mismatch")
            records.append(Checkpoint(model_key, str(summary["display_name"]), seed, int(summary["parameters"]), checkpoint, observed, str(summary["architecture_sha256"]), summary.get("backbone_schema_sha256")))
    return records


def _runtime(spec: EvaluationSpec) -> tuple[Any, Any]:
    return importlib.import_module(spec.core_module), importlib.import_module(spec.runtime_module)


def _strict_load(spec: EvaluationSpec, records: Sequence[Checkpoint]) -> list[dict[str, Any]]:
    core, runtime = _runtime(spec)
    audits = []
    for model_key in MODELS:
        model, audit = runtime.build_model(core.ScratchConfig(model_key=model_key, run_mode="development"), torch.device("cpu"))
        model_records = [record for record in records if record.model_key == model_key]
        if any(
            any(
                (
                    audit["parameters"] != record.parameters,
                    audit["architecture_sha256"] != record.architecture_sha256,
                    audit.get("backbone_schema_sha256")
                    != record.backbone_schema_sha256,
                )
            )
            for record in model_records
        ):
            raise ValueError(f"{model_key} current architecture does not match frozen checkpoints")
        for record in model_records:
            state = torch.load(record.checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            audits.append({"model_key": model_key, "seed": record.seed, "strict_state_dict": True, "parameters": audit["parameters"]})
        del model
        gc.collect()
    return audits


def run_preflight(spec: EvaluationSpec, *, strict_load: bool = True) -> dict[str, Any]:
    protocol = validate_protocol(spec)
    records = discover_checkpoints(spec)
    audits = _strict_load(spec, records) if strict_load else []
    return {"status": "preflight_passed", "sealed_inference_executed": False, "protocol": protocol, "checkpoint_count": len(records), "checkpoint_table": [record.__dict__ | {"checkpoint_path": str(record.checkpoint_path)} for record in records], "strict_load_audits": audits}


class _ImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: Any):
        self.frame, self.transform = frame.reset_index(drop=True), transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        with Image.open(row.absolute_path) as image:
            return self.transform(image.convert("RGB")), int(row.label_id)


def _prepare_test_images(spec: EvaluationSpec, work_root: Path) -> tuple[pd.DataFrame, tuple[str, ...]]:
    test = pd.read_csv(spec.test_manifest).sort_values("sample_id").reset_index(drop=True)
    class_names, _ = _mapping(spec.mapping_path, spec.num_classes)
    if spec.image_archive is not None:
        cache = work_root / "test_images"
        cache.mkdir()
        with zipfile.ZipFile(spec.image_archive) as archive:
            members = set(archive.namelist())
            if set(test[spec.image_path_column].astype(str)) - members:
                raise ValueError("test images missing from archive")
            paths = []
            for row in test.itertuples(index=False):
                target = cache / str(getattr(row, spec.image_path_column)).replace("/", "__")
                with archive.open(str(getattr(row, spec.image_path_column))) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
                if _sha256(target) != str(getattr(row, spec.image_hash_column)):
                    raise ValueError(f"test image hash mismatch: {row.sample_id}")
                paths.append(str(target))
    else:
        assert spec.image_root is not None
        paths = []
        for row in test.itertuples(index=False):
            target = (spec.image_root / str(getattr(row, spec.image_path_column))).resolve()
            try:
                target.relative_to(spec.image_root)
            except ValueError as error:
                raise ValueError("test path escapes image root") from error
            if not target.is_file() or _sha256(target) != str(getattr(row, spec.image_hash_column)):
                raise ValueError(f"test image hash mismatch: {row.sample_id}")
            paths.append(str(target))
    test["absolute_path"] = paths
    return test, class_names


def _aggregate(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_names = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall", "weighted_precision", "weighted_recall", "nll", "brier", "ece_15bin")
    for (key, name, parameters), group in seed_metrics.groupby(["model_key", "display_name", "parameters"], sort=False):
        row: dict[str, Any] = {"model_key": key, "display_name": name, "parameters": int(parameters), "seeds": ",".join(str(value) for value in sorted(group.seed)), "seed_count": len(group)}
        critical = float(student_t.ppf(0.975, len(group) - 1))
        for metric in metric_names:
            values = group[metric].to_numpy(dtype=float)
            mean, std = float(values.mean()), float(values.std(ddof=1))
            delta = critical * std / math.sqrt(len(values))
            row |= {f"{metric}_mean": mean, f"{metric}_std": std, f"{metric}_ci95_low": mean-delta, f"{metric}_ci95_high": mean+delta}
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(["macro_f1_mean", "accuracy_mean"], ascending=False).reset_index(drop=True)
    result.insert(0, "rank_by_macro_f1", np.arange(1, len(result)+1))
    return result


def _claim(spec: EvaluationSpec) -> dict[str, Any]:
    spec.output_root.mkdir(parents=True, exist_ok=True)
    payload = {"evaluation_id": spec.evaluation_id, "status": "started", "attempt_id": uuid.uuid4().hex, "started_at": _utc_now(), "output_directory": str(spec.output_directory)}
    try:
        fd = os.open(spec.ledger_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise FileExistsError(f"final holdout was already claimed: {spec.ledger_path}") from error
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def _transition(spec: EvaluationSpec, attempt_id: str, expected: str, status: str, details: dict[str, Any]) -> dict[str, Any]:
    current = json.loads(spec.ledger_path.read_text(encoding="utf-8"))
    if current.get("attempt_id") != attempt_id or current.get("status") != expected:
        raise ValueError("invalid sealed evaluation ledger transition")
    payload = current | details | {"status": status, "finished_at": _utc_now()}
    _write_json(spec.ledger_path, payload)
    return payload


def _inventory(root: Path) -> list[dict[str, Any]]:
    return [{"relative_path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in sorted(root.rglob("*")) if path.is_file() and path.name != "artifact_inventory.json"]


def load_existing_results(spec: EvaluationSpec) -> dict[str, Any]:
    output = spec.output_directory
    completed_path = output / "COMPLETED.json"
    inventory_path = output / "artifact_inventory.json"
    for path in (completed_path, inventory_path, output / "sealed_test_seed_metrics.csv", output / "sealed_test_aggregate.csv"):
        _require(path, "completed sealed-test artifact")
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    ledger = json.loads(spec.ledger_path.read_text(encoding="utf-8"))
    if completed.get("status") != "complete" or completed.get("evaluation_id") != spec.evaluation_id:
        raise ValueError("sealed output completion marker mismatch")
    if ledger.get("evaluation_id") != spec.evaluation_id or ledger.get("status") not in {"publishing", "complete"}:
        raise ValueError("sealed output ledger mismatch")
    if ledger.get("completed_marker_sha256") != _sha256(completed_path) or ledger.get("artifact_inventory_sha256") != _sha256(inventory_path):
        raise ValueError("sealed output ledger artifact identity mismatch")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    listed = set()
    for item in inventory:
        relative = str(item["relative_path"])
        path = (output / relative).resolve()
        try:
            path.relative_to(output.resolve())
        except ValueError as error:
            raise ValueError("unsafe sealed output path") from error
        if relative in listed or not path.is_file() or path.stat().st_size != int(item["bytes"]) or _sha256(path) != str(item["sha256"]):
            raise ValueError("sealed output inventory mismatch")
        listed.add(relative)
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file() and path.name != "artifact_inventory.json"}
    if actual != listed:
        raise ValueError("sealed output inventory membership mismatch")
    if ledger["status"] == "publishing":
        _transition(spec, str(ledger["attempt_id"]), "publishing", "complete", {"recovered_from_publishing": True})
    return {"status": "report_loaded", "sealed_inference_executed": False, "output_directory": output, "seed_metrics": pd.read_csv(output / "sealed_test_seed_metrics.csv"), "aggregate": pd.read_csv(output / "sealed_test_aggregate.csv"), "artifact_inventory": inventory}


def execute(spec: EvaluationSpec, *, run_mode: str, acknowledgement: str = "", device: str = "auto", batch_size: int = 64, num_workers: int = 0) -> dict[str, Any]:
    if run_mode == "preflight":
        return run_preflight(spec, strict_load=True)
    if run_mode == "report_existing":
        return load_existing_results(spec)
    if run_mode == "sealed_evaluation":
        return run_sealed_evaluation(spec, acknowledgement=acknowledgement, device=device, batch_size=batch_size, num_workers=num_workers)
    raise ValueError("run_mode must be preflight, sealed_evaluation, or report_existing")


def run_sealed_evaluation(spec: EvaluationSpec, *, acknowledgement: str, device: str = "auto", batch_size: int = 64, num_workers: int = 0) -> dict[str, Any]:
    if acknowledgement != spec.acknowledgement:
        raise PermissionError("sealed acknowledgement is absent or incorrect")
    if spec.output_directory.exists():
        raise FileExistsError("sealed output already exists")
    preflight = run_preflight(spec, strict_load=True)
    records = discover_checkpoints(spec)
    claim = _claim(spec)
    selected = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
    work_root: Path | None = None
    try:
        work_root = Path(tempfile.mkdtemp(prefix=f".{spec.evaluation_id}.working-", dir=spec.output_root))
        result_root = work_root / "result"; result_root.mkdir()
        frame, class_names = _prepare_test_images(spec, work_root)
        _, transform = _runtime(spec)[1].build_transforms()
        loader = DataLoader(_ImageDataset(frame, transform), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=selected.type == "cuda")
        rows = []
        core, runtime = _runtime(spec)
        for index, record in enumerate(records, 1):
            model, audit = runtime.build_model(core.ScratchConfig(model_key=record.model_key, run_mode="development"), selected)
            if audit["architecture_sha256"] != record.architecture_sha256:
                raise ValueError("architecture changed after preflight")
            state = torch.load(record.checkpoint_path, map_location=selected, weights_only=True); model.load_state_dict(state, strict=True); model.eval()
            started = time.perf_counter(); logits, targets = [], []
            with torch.inference_mode():
                for images, labels in loader:
                    output = model(images.float().to(selected, non_blocking=True)).float()
                    if not torch.isfinite(output).all(): raise FloatingPointError("non-finite logits")
                    logits.append(output.cpu()); targets.append(labels.cpu())
            logits_np, targets_np = torch.cat(logits).numpy(), torch.cat(targets).numpy()
            shifted = logits_np - logits_np.max(axis=1, keepdims=True); exp = np.exp(shifted); probabilities = exp / exp.sum(axis=1, keepdims=True)
            metrics, per_class, matrix = compute_classification_metrics(targets_np, probabilities, class_names=class_names, ece_bins=15)
            seed_root = result_root / record.model_key / f"seed_{record.seed}"; seed_root.mkdir(parents=True)
            row = metrics | {"model_key":record.model_key,"display_name":record.display_name,"seed":record.seed,"parameters":record.parameters,"runtime_seconds":time.perf_counter()-started,"checkpoint_sha256":record.checkpoint_sha256}; rows.append(row)
            _write_json(seed_root / "test_metrics.json", row); per_class.to_csv(seed_root / "test_per_class.csv", index=False); pd.DataFrame(matrix,index=class_names,columns=class_names).to_csv(seed_root / "test_confusion_matrix.csv", index_label="target")
            pred = probabilities.argmax(axis=1); prediction = frame[["sample_id","component_id","label","label_id"]].copy(); prediction["prediction"] = pred; prediction["correct"] = pred == targets_np; prediction["confidence"] = probabilities.max(axis=1); prediction.to_csv(seed_root / "test_predictions.csv", index=False); np.savez_compressed(seed_root / "test_arrays.npz", sample_ids=frame.sample_id.astype(str).to_numpy(), targets=targets_np, predictions=pred, logits=logits_np.astype(np.float32), probabilities=probabilities.astype(np.float32))
            print(json.dumps({"event":"checkpoint_evaluated","index":index,"total":len(records),"model_key":record.model_key,"seed":record.seed,"accuracy":metrics["accuracy"],"macro_f1":metrics["macro_f1"]}), flush=True)
            del model, state, logits, targets, logits_np, targets_np, probabilities; gc.collect()
            if selected.type == "cuda": torch.cuda.empty_cache()
        seed_metrics = pd.DataFrame(rows); aggregate = _aggregate(seed_metrics); seed_metrics.to_csv(result_root / "sealed_test_seed_metrics.csv", index=False); aggregate.to_csv(result_root / "sealed_test_aggregate.csv", index=False)
        _write_json(result_root / "preflight_report.json", preflight); _write_json(result_root / "COMPLETED.json", {"status":"complete","evaluation_id":spec.evaluation_id,"completed_at":_utc_now()}); _write_json(result_root / "artifact_inventory.json", _inventory(result_root))
        published = _transition(spec, claim["attempt_id"], "started", "publishing", {"completed_marker_sha256":_sha256(result_root/"COMPLETED.json"),"artifact_inventory_sha256":_sha256(result_root/"artifact_inventory.json")})
        os.replace(result_root, spec.output_directory)
        completed = _transition(spec, published["attempt_id"], "publishing", "complete", {})
        shutil.rmtree(work_root, ignore_errors=True)
    except BaseException as error:
        if spec.output_directory.exists(): raise
        try:
            status=json.loads(spec.ledger_path.read_text(encoding="utf-8")).get("status")
            _transition(spec, claim["attempt_id"], str(status), "failed", {"error_type":type(error).__name__,"error_message":str(error)})
        except BaseException: pass
        raise
    return {"status":"complete","sealed_inference_executed":True,"output_directory":spec.output_directory,"access_ledger":completed,"seed_metrics":seed_metrics,"aggregate":aggregate}
