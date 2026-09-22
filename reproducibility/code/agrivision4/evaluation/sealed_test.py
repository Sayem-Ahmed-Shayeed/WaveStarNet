"""Fail-closed evaluation of frozen scratch checkpoints on Research V2 holdout."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import t as student_t
from torch.utils.data import DataLoader, Dataset

from agrivision4.scratch_comparison.experiment_core import (
    ScratchConfig,
    model_spec,
)
from agrivision4.scratch_comparison.notebook_runtime import (
    METRIC_NAMES,
    build_model,
    build_transforms,
)


PROTOCOL_ID = "agrivision4_clean_grouped_research_v2"
EVALUATION_ID = "agrivision4_scratch_research_v2_final_holdout_v1"
SEALED_ACKNOWLEDGEMENT = "RUN_AGRIVISION4_RESEARCH_V2_FINAL_HOLDOUT_ONCE"
EXPECTED_SEEDS = (42, 2024, 3407)
EXPECTED_MODEL_KEYS = (
    "wavestarnet_big_transfer_v2",
    "efficientnet_b0",
    "efficientnet_b3",
    "mobilenetv3_large_100",
    "mobilevit_s",
)
EXPECTED_TRAIN_SHA256 = (
    "255477b324d5d868072b61f47715f63dbcdaea7c66a5ad5f2cb17e35840c05b2"
)
EXPECTED_VALIDATION_SHA256 = (
    "3d3f6d0764767178aa1701a12b9493ec10b7e8bbc38a28d3ba9f066083c8473a"
)
EXPECTED_CLASS_MAPPING_SHA256 = (
    "bc9f0fbb6a2e7bc24ffb01910207edfc764066c7316376588625b3b004536e18"
)
EXPECTED_FINAL_HOLDOUT_SHA256 = (
    "aa8f05ee626b7e1b95eb3bd8b411acbe6b76fc96f3ec4c3323c233899f866f05"
)
EXPECTED_IMAGE_ARCHIVE_SHA256 = (
    "df327b5d88a4a02cbfc4ba12c123724b3aa10f79c61bda088009b276f3b971ae"
)
EXPECTED_CHECKPOINT_REGISTRY_SHA256 = (
    "1a3788435c7b1062c882dda9fc0363cb86197b708d598afdeb1fa0d461d3dc21"
)
EXPECTED_SOURCE_SHA256 = {
    "core": "b25643f7832cdc7471b9402756b8d2fe2968beadf424e80b70789eede91ef616",
    "runtime": "c62c78b1037029d46b93e256a32eeb3582ed37252729321027e47497087236d0",
    "model": "1e600b84c2a901640ff93ec9ef8e255291cbb07f335d5217988ffe147c2ca53f",
}
NUM_CLASSES = 21
FINAL_HOLDOUT_SAMPLES = 579

RESULT_DIRECTORY_NAMES = {
    "wavestarnet_big_transfer_v2": "wavestarnet_big_transfer_v2",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b3": "efficientnet_b3",
    "mobilenetv3_large_100": "mobilenetv3_large_100",
    "mobilevit_s": "mobilevit_s",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(payload: Any, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


@dataclass(frozen=True)
class EvaluationConfig:
    repo_root: Path
    run_mode: str = "preflight"
    acknowledgement: str = ""
    device: str = "auto"
    batch_size: int = 32
    num_workers: int = 0
    verify_archive_hash: bool = True
    verify_image_hashes: bool = True
    output_directory: Path | None = None

    def __post_init__(self) -> None:
        root = Path(self.repo_root).expanduser().resolve()
        object.__setattr__(self, "repo_root", root)
        if self.run_mode not in {
            "preflight",
            "sealed_evaluation",
            "report_existing",
        }:
            raise ValueError(
                "run_mode must be preflight, sealed_evaluation, or report_existing"
            )
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        if self.output_directory is None:
            output = (
                root
                / "agrivision4/results/sealed_test"
                / EVALUATION_ID
            )
        else:
            output = Path(self.output_directory)
            if not output.is_absolute():
                output = root / output
        object.__setattr__(self, "output_directory", output.resolve())

    @property
    def protocol_root(self) -> Path:
        return (
            self.repo_root
            / "agrivision4/research_protocol/agrivision4-locked-validation-v1"
        )

    @property
    def train_manifest(self) -> Path:
        return self.protocol_root / "manifests/train.csv"

    @property
    def validation_manifest(self) -> Path:
        return self.protocol_root / "manifests/validation.csv"

    @property
    def class_mapping(self) -> Path:
        return self.protocol_root / "manifests/class_mapping.json"

    @property
    def final_holdout_manifest(self) -> Path:
        return self.protocol_root / "sealed/final_holdout.csv"

    @property
    def image_archive(self) -> Path:
        return (
            self.repo_root
            / "agrivision4/cleaned_kaggle/agrivision4-clean-grouped-v1"
            / "agrivision4-clean-grouped-images.zip"
        )

    @property
    def scratch_results_root(self) -> Path:
        return self.repo_root / "agrivision4/results/scratch comparison"

    @property
    def canonical_output_directory(self) -> Path:
        return (
            self.repo_root
            / "agrivision4/results/sealed_test"
            / EVALUATION_ID
        ).resolve()

    @property
    def access_ledger_path(self) -> Path:
        return (
            self.repo_root
            / "agrivision4/results/sealed_test"
            / f".{EVALUATION_ID}.access.json"
        ).resolve()

    @property
    def checkpoint_registry(self) -> Path:
        return (
            self.repo_root
            / "agrivision4/evaluation"
            / "agrivision4_scratch_checkpoint_registry.json"
        )


@dataclass(frozen=True)
class CheckpointRecord:
    model_key: str
    display_name: str
    seed: int
    checkpoint_path: Path
    expected_sha256: str
    actual_sha256: str
    parameters: int
    architecture_sha256: str
    backbone_schema_sha256: str | None
    run_root: Path
    test_accessed: bool

    def public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checkpoint_path"] = str(self.checkpoint_path)
        payload["run_root"] = str(self.run_root)
        return payload


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")


def _read_mapping(path: Path) -> tuple[tuple[str, ...], dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    class_to_id = {
        str(name): int(label_id)
        for name, label_id in payload["class_to_id"].items()
    }
    if sorted(class_to_id.values()) != list(range(NUM_CLASSES)):
        raise ValueError("class mapping IDs must be contiguous range(21)")
    id_to_class = {
        int(label_id): str(name)
        for label_id, name in payload["id_to_class"].items()
    }
    class_names = tuple(id_to_class[index] for index in range(NUM_CLASSES))
    if any(class_to_id[name] != index for index, name in enumerate(class_names)):
        raise ValueError("class mapping directions disagree")
    return class_names, class_to_id


def _validate_manifest(
    frame: pd.DataFrame,
    *,
    name: str,
    class_to_id: dict[str, int],
    expected_split: str,
    expected_rows: int | None,
) -> None:
    required = {
        "sample_id",
        "component_id",
        "split",
        "label",
        "label_id",
        "packaged_path",
        "sha256",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"{name} rows differ: {len(frame)} != {expected_rows}")
    if frame.empty:
        raise ValueError(f"{name} is empty")
    for column in ("sample_id", "component_id", "packaged_path", "sha256"):
        if frame[column].astype(str).duplicated().any():
            raise ValueError(f"{name} has duplicate {column}")
    if set(frame["split"].astype(str)) != {expected_split}:
        raise ValueError(f"{name} has an unexpected split role")
    for row in frame.itertuples(index=False):
        if class_to_id.get(str(row.label)) != int(row.label_id):
            raise ValueError(f"{name} class mapping mismatch: {row.sample_id}")


def _overlap(first: pd.DataFrame, second: pd.DataFrame) -> dict[str, int]:
    return {
        key: len(
            set(first[key].astype(str)).intersection(
                set(second[key].astype(str))
            )
        )
        for key in ("sample_id", "component_id", "sha256")
    }


def validate_protocol(
    config: EvaluationConfig,
    *,
    verify_archive_hash: bool | None = None,
) -> dict[str, Any]:
    paths = {
        "train": config.train_manifest,
        "validation": config.validation_manifest,
        "class_mapping": config.class_mapping,
        "final_holdout": config.final_holdout_manifest,
        "image_archive": config.image_archive,
    }
    for name, path in paths.items():
        _require_file(path, name)

    actual_hashes = {
        "train": sha256_file(paths["train"]),
        "validation": sha256_file(paths["validation"]),
        "class_mapping": sha256_file(paths["class_mapping"]),
        "final_holdout": sha256_file(paths["final_holdout"]),
    }
    expected_hashes = {
        "train": EXPECTED_TRAIN_SHA256,
        "validation": EXPECTED_VALIDATION_SHA256,
        "class_mapping": EXPECTED_CLASS_MAPPING_SHA256,
        "final_holdout": EXPECTED_FINAL_HOLDOUT_SHA256,
    }
    if actual_hashes != expected_hashes:
        raise ValueError(
            f"protocol hash mismatch: actual={actual_hashes}, expected={expected_hashes}"
        )

    class_names, class_to_id = _read_mapping(paths["class_mapping"])
    train = pd.read_csv(paths["train"])
    validation = pd.read_csv(paths["validation"])
    final_holdout = pd.read_csv(paths["final_holdout"])
    _validate_manifest(
        train,
        name="locked train",
        class_to_id=class_to_id,
        expected_split="modeling_pool",
        expected_rows=1810,
    )
    _validate_manifest(
        validation,
        name="locked validation",
        class_to_id=class_to_id,
        expected_split="modeling_pool",
        expected_rows=915,
    )
    _validate_manifest(
        final_holdout,
        name="Research V2 final holdout",
        class_to_id=class_to_id,
        expected_split="final_holdout",
        expected_rows=FINAL_HOLDOUT_SAMPLES,
    )
    if set(final_holdout["label_id"].astype(int)) != set(range(NUM_CLASSES)):
        raise ValueError("final holdout does not contain every class")

    overlap = {
        "train_vs_validation": _overlap(train, validation),
        "train_vs_final_holdout": _overlap(train, final_holdout),
        "validation_vs_final_holdout": _overlap(validation, final_holdout),
    }
    if any(value for pair in overlap.values() for value in pair.values()):
        raise ValueError(f"Research V2 partition overlap detected: {overlap}")

    should_hash_archive = (
        config.verify_archive_hash
        if verify_archive_hash is None
        else bool(verify_archive_hash)
    )
    archive_hash = None
    if should_hash_archive:
        archive_hash = sha256_file(paths["image_archive"])
        if archive_hash != EXPECTED_IMAGE_ARCHIVE_SHA256:
            raise ValueError("clean image archive SHA-256 mismatch")

    with zipfile.ZipFile(paths["image_archive"]) as archive:
        members = set(archive.namelist())
    missing_members = sorted(
        set(final_holdout["packaged_path"].astype(str)) - members
    )
    if missing_members:
        raise ValueError(
            f"{len(missing_members)} final-holdout images are absent from archive"
        )

    supports = final_holdout.groupby("label_id").size().sort_index()
    return {
        "protocol_id": PROTOCOL_ID,
        "train_sha256": actual_hashes["train"],
        "validation_sha256": actual_hashes["validation"],
        "class_mapping_sha256": actual_hashes["class_mapping"],
        "final_holdout_sha256": actual_hashes["final_holdout"],
        "image_archive_sha256": archive_hash,
        "image_archive_hash_verified": should_hash_archive,
        "modeling_pool_samples": len(train) + len(validation),
        "train_samples": len(train),
        "validation_samples": len(validation),
        "final_holdout_samples": len(final_holdout),
        "class_count": len(class_names),
        "minimum_class_support": int(supports.min()),
        "maximum_class_support": int(supports.max()),
        "overlap": overlap,
        "missing_archive_members": 0,
    }


def _single_run_root(model_root: Path) -> Path:
    manifests = sorted(model_root.glob("*/run_manifest.json"))
    if len(manifests) != 1:
        raise ValueError(
            f"expected one completed run under {model_root}, found {len(manifests)}"
        )
    return manifests[0].parent


def _load_checkpoint_registry(config: EvaluationConfig) -> dict[str, Any]:
    _require_file(config.checkpoint_registry, "frozen checkpoint registry")
    actual_hash = sha256_file(config.checkpoint_registry)
    if actual_hash != EXPECTED_CHECKPOINT_REGISTRY_SHA256:
        raise ValueError("frozen checkpoint registry SHA-256 mismatch")
    registry = json.loads(
        config.checkpoint_registry.read_text(encoding="utf-8")
    )
    if registry.get("evaluation_id") != EVALUATION_ID:
        raise ValueError("checkpoint registry evaluation ID mismatch")
    if registry.get("protocol_id") != "agrivision4_scratch_comparison_v1":
        raise ValueError("checkpoint registry protocol ID mismatch")
    if tuple(registry.get("seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("checkpoint registry seed set mismatch")
    frozen_models = registry.get("frozen_models", {})
    if tuple(frozen_models) != EXPECTED_MODEL_KEYS:
        raise ValueError("checkpoint registry model set or order mismatch")
    for model_key, payload in frozen_models.items():
        checkpoint_seeds = tuple(
            int(seed) for seed in payload.get("checkpoints", {})
        )
        if checkpoint_seeds != EXPECTED_SEEDS:
            raise ValueError(f"{model_key} frozen checkpoint seeds mismatch")
    return registry


def _artifact_hash_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = {
        str(item["path"]): str(item["sha256"])
        for item in payload
    }
    if len(mapping) != len(payload):
        raise ValueError(f"duplicate paths in development inventory: {path}")
    return mapping


def discover_checkpoints(config: EvaluationConfig) -> list[CheckpointRecord]:
    registry = _load_checkpoint_registry(config)
    records: list[CheckpointRecord] = []
    for model_key in EXPECTED_MODEL_KEYS:
        directory = RESULT_DIRECTORY_NAMES[model_key]
        run_root = _single_run_root(config.scratch_results_root / directory)
        manifest_path = run_root / "run_manifest.json"
        summary_path = run_root / "model_summary.json"
        inventory_path = run_root / "artifact_inventory.json"
        validation_metrics_path = run_root / "validation_seed_metrics.csv"
        _require_file(manifest_path, f"{model_key} run manifest")
        _require_file(summary_path, f"{model_key} model summary")
        _require_file(inventory_path, f"{model_key} artifact inventory")
        _require_file(
            validation_metrics_path,
            f"{model_key} validation seed metrics",
        )
        frozen = registry["frozen_models"][model_key]
        frozen_file_hashes = {
            "run_manifest_sha256": manifest_path,
            "model_summary_sha256": summary_path,
            "artifact_inventory_sha256": inventory_path,
            "validation_seed_metrics_sha256": validation_metrics_path,
        }
        for registry_key, path in frozen_file_hashes.items():
            if sha256_file(path) != str(frozen[registry_key]):
                raise ValueError(
                    f"{model_key} frozen {registry_key} mismatch"
                )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        inventory = _artifact_hash_map(inventory_path)
        validation_metrics = pd.read_csv(validation_metrics_path)
        if manifest.get("status") != "complete":
            raise ValueError(f"{model_key} run is not complete")
        if manifest.get("protocol_id") != "agrivision4_scratch_comparison_v1":
            raise ValueError(f"{model_key} protocol ID mismatch")
        if manifest.get("run_mode") != "development":
            raise ValueError(f"{model_key} is not a development run")
        if manifest.get("model_key") != model_key:
            raise ValueError(f"{model_key} run manifest model mismatch")
        if manifest.get("initialization") != "random" or manifest.get("pretrained"):
            raise ValueError(f"{model_key} is not a random-initialization run")
        if manifest.get("test_accessed") is not False:
            raise ValueError(f"{model_key} reports test access")
        if tuple(manifest.get("active_seeds", ())) != EXPECTED_SEEDS:
            raise ValueError(f"{model_key} active seed set mismatch")
        expected_manifests = manifest.get("expected_manifest_sha256", {})
        if expected_manifests != {
            "train": EXPECTED_TRAIN_SHA256,
            "validation": EXPECTED_VALIDATION_SHA256,
            "class_mapping": EXPECTED_CLASS_MAPPING_SHA256,
        }:
            raise ValueError(f"{model_key} development manifest hashes mismatch")
        if manifest.get("source_sha256") != EXPECTED_SOURCE_SHA256:
            raise ValueError(f"{model_key} source hashes mismatch")
        if summary.get("model_key") != model_key:
            raise ValueError(f"{model_key} model summary mismatch")
        spec = model_spec(model_key)
        if int(summary.get("parameters", -1)) != spec.expected_parameters:
            raise ValueError(f"{model_key} parameter count mismatch")
        if set(validation_metrics["seed"].astype(int)) != set(EXPECTED_SEEDS):
            raise ValueError(f"{model_key} validation seed rows mismatch")
        if len(validation_metrics) != len(EXPECTED_SEEDS):
            raise ValueError(f"{model_key} validation seed count mismatch")

        found_seeds: set[int] = set()
        for seed in EXPECTED_SEEDS:
            seed_root = run_root / f"seed_{seed}"
            checkpoint = seed_root / "best_checkpoint.pt"
            checksum_path = seed_root / "best_checkpoint.sha256"
            config_path = seed_root / "config.json"
            for path, description in (
                (checkpoint, "checkpoint"),
                (checksum_path, "checkpoint checksum"),
                (config_path, "seed configuration"),
            ):
                _require_file(path, f"{model_key} seed {seed} {description}")
            seed_config = json.loads(config_path.read_text(encoding="utf-8"))
            if seed_config.get("model_key") != model_key:
                raise ValueError(f"{model_key} seed {seed} configuration mismatch")
            if int(seed_config.get("seed", -1)) != seed:
                raise ValueError(f"{model_key} seed directory mismatch")
            if seed_config.get("run_mode") != "development":
                raise ValueError(f"{model_key} seed {seed} is not development")
            frozen_checksum = str(frozen["checkpoints"][str(seed)])
            sidecar_checksum = checksum_path.read_text(encoding="utf-8").strip()
            actual_checksum = sha256_file(checkpoint)
            inventory_checksum = inventory.get(
                f"seed_{seed}/best_checkpoint.pt"
            )
            validation_row = validation_metrics.loc[
                validation_metrics["seed"].astype(int) == seed
            ]
            validation_checksum = str(
                validation_row.iloc[0]["checkpoint_sha256"]
            )
            checkpoint_hashes = {
                "frozen_registry": frozen_checksum,
                "sidecar": sidecar_checksum,
                "actual_file": actual_checksum,
                "development_inventory": inventory_checksum,
                "validation_seed_metrics": validation_checksum,
            }
            if set(checkpoint_hashes.values()) != {frozen_checksum}:
                raise ValueError(
                    f"{model_key} seed {seed} independent checkpoint "
                    f"identities disagree: {checkpoint_hashes}"
                )
            records.append(
                CheckpointRecord(
                    model_key=model_key,
                    display_name=spec.display_name,
                    seed=seed,
                    checkpoint_path=checkpoint.resolve(),
                    expected_sha256=frozen_checksum,
                    actual_sha256=actual_checksum,
                    parameters=spec.expected_parameters,
                    architecture_sha256=str(summary["architecture_sha256"]),
                    backbone_schema_sha256=summary.get(
                        "backbone_schema_sha256"
                    ),
                    run_root=run_root.resolve(),
                    test_accessed=False,
                )
            )
            found_seeds.add(seed)
        if found_seeds != set(EXPECTED_SEEDS):
            raise ValueError(f"{model_key} seed set mismatch")
    if len(records) != len(EXPECTED_MODEL_KEYS) * len(EXPECTED_SEEDS):
        raise ValueError("checkpoint inventory size mismatch")
    return records


def _validate_source_files(config: EvaluationConfig) -> dict[str, str]:
    paths = {
        "core": config.repo_root
        / "agrivision4/scratch_comparison/experiment_core.py",
        "runtime": config.repo_root
        / "agrivision4/scratch_comparison/notebook_runtime.py",
        "model": config.repo_root / "agrivision4/models/wavestarnet_family.py",
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if actual != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            f"local scratch source differs from trained source: {actual}"
        )
    return actual


def _strict_checkpoint_preflight(
    records: Sequence[CheckpointRecord],
) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for model_key in EXPECTED_MODEL_KEYS:
        model_records = [record for record in records if record.model_key == model_key]
        config = ScratchConfig(model_key=model_key, run_mode="development")
        model, audit = build_model(config, torch.device("cpu"))
        reference = model_records[0]
        if audit["architecture_sha256"] != reference.architecture_sha256:
            raise ValueError(f"{model_key} architecture hash mismatch")
        if audit["backbone_schema_sha256"] != reference.backbone_schema_sha256:
            raise ValueError(f"{model_key} backbone schema hash mismatch")
        for record in model_records:
            state = torch.load(
                record.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(state, dict) or not all(
                isinstance(value, torch.Tensor) for value in state.values()
            ):
                raise ValueError(f"{model_key} seed {record.seed} state is invalid")
            model.load_state_dict(state, strict=True)
            audits.append(
                {
                    "model_key": model_key,
                    "seed": record.seed,
                    "strict_state_dict": True,
                    "parameters": audit["parameters"],
                    "architecture_sha256": audit["architecture_sha256"],
                    "checkpoint_sha256": record.actual_sha256,
                }
            )
            del state
        del model
        gc.collect()
    return audits


def run_preflight(
    config: EvaluationConfig,
    *,
    verify_model_loading: bool = True,
) -> dict[str, Any]:
    protocol = validate_protocol(config)
    source_hashes = _validate_source_files(config)
    records = discover_checkpoints(config)
    model_audits = (
        _strict_checkpoint_preflight(records) if verify_model_loading else []
    )
    return {
        "status": "preflight_passed",
        "sealed_inference_executed": False,
        "protocol": protocol,
        "source_sha256": source_hashes,
        "checkpoint_count": len(records),
        "model_count": len(EXPECTED_MODEL_KEYS),
        "seed_count": len(EXPECTED_SEEDS),
        "checkpoint_table": [record.public_dict() for record in records],
        "strict_load_audits": model_audits,
    }


def ensure_output_is_unused(output_directory: str | Path) -> None:
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(
            f"sealed-test output already exists and cannot be overwritten: {output}"
        )


def validate_sealed_execution_request(config: EvaluationConfig) -> None:
    if config.run_mode != "sealed_evaluation":
        raise ValueError("sealed execution requires sealed_evaluation mode")
    if config.acknowledgement != SEALED_ACKNOWLEDGEMENT:
        raise PermissionError(
            "sealed evaluation acknowledgement is absent or incorrect"
        )
    if not config.verify_archive_hash or not config.verify_image_hashes:
        raise ValueError(
            "sealed evaluation requires archive and per-image hash verification"
        )
    if Path(config.output_directory) != config.canonical_output_directory:
        raise ValueError(
            "sealed evaluation requires the canonical output directory"
        )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def claim_access_ledger(config: EvaluationConfig) -> dict[str, Any]:
    validate_sealed_execution_request(config)
    path = config.access_ledger_path
    path.parent.mkdir(parents=True, exist_ok=True)
    attempt_id = uuid.uuid4().hex
    payload = {
        "evaluation_id": EVALUATION_ID,
        "protocol_id": PROTOCOL_ID,
        "attempt_id": attempt_id,
        "status": "started",
        "started_at": utc_now(),
        "output_directory": str(config.canonical_output_directory),
        "final_holdout_sha256": EXPECTED_FINAL_HOLDOUT_SHA256,
        "checkpoint_registry_sha256": EXPECTED_CHECKPOINT_REGISTRY_SHA256,
        "acknowledgement": config.acknowledgement,
    }
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError as error:
        raise FileExistsError(
            "Research V2 final-holdout access has already been claimed; "
            f"inspect the canonical ledger: {path}"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return payload


def _transition_access_ledger(
    config: EvaluationConfig,
    *,
    attempt_id: str,
    status: str,
    expected_status: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    allowed_transitions = {
        ("started", "publishing"),
        ("started", "failed"),
        ("publishing", "complete"),
        ("publishing", "failed"),
    }
    if (expected_status, status) not in allowed_transitions:
        raise ValueError(
            f"invalid access ledger transition: {expected_status} -> {status}"
        )
    path = config.access_ledger_path
    if not path.is_file():
        raise FileNotFoundError("canonical access ledger is missing")
    current = json.loads(path.read_text(encoding="utf-8"))
    if current.get("attempt_id") != attempt_id:
        raise ValueError("access ledger attempt ID mismatch")
    if current.get("status") != expected_status:
        raise ValueError(
            f"access ledger is not in {expected_status} state"
        )
    transitioned = {
        **current,
        **details,
        "status": status,
    }
    timestamp = utc_now()
    if status == "publishing":
        transitioned["publishing_at"] = timestamp
    else:
        transitioned["finished_at"] = timestamp
    _write_json_atomic(path, transitioned)
    return transitioned


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _HoldoutDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: Any):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Any, int, str, str, str]:
        row = self.frame.iloc[index]
        with Image.open(str(row.absolute_path)) as image:
            tensor = self.transform(image.convert("RGB"))
        return (
            tensor,
            int(row.label_id),
            str(row.sample_id),
            str(row.component_id),
            str(row.packaged_path),
        )


def _materialize_holdout(
    config: EvaluationConfig,
    cache_root: Path,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    frame = pd.read_csv(config.final_holdout_manifest)
    class_names, class_to_id = _read_mapping(config.class_mapping)
    _validate_manifest(
        frame,
        name="Research V2 final holdout",
        class_to_id=class_to_id,
        expected_split="final_holdout",
        expected_rows=FINAL_HOLDOUT_SAMPLES,
    )
    cache_root.mkdir(parents=True, exist_ok=False)
    absolute_paths: list[str] = []
    with zipfile.ZipFile(config.image_archive) as archive:
        members = set(archive.namelist())
        for row in frame.sort_values("sample_id").itertuples(index=False):
            member = str(row.packaged_path)
            if member not in members:
                raise FileNotFoundError(f"archive member is missing: {member}")
            payload = archive.read(member)
            if config.verify_image_hashes:
                actual = hashlib.sha256(payload).hexdigest()
                if actual != str(row.sha256):
                    raise ValueError(f"image hash mismatch: {row.sample_id}")
            destination = (cache_root / member).resolve()
            try:
                destination.relative_to(cache_root.resolve())
            except ValueError as error:
                raise ValueError("unsafe packaged path in final holdout") from error
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            absolute_paths.append(str(destination))
    frame = frame.sort_values("sample_id").reset_index(drop=True)
    frame["absolute_path"] = absolute_paths
    return frame, class_names


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int,
) -> float:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == y_true
    total = len(y_true)
    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for index, (lower, upper) in enumerate(
        zip(boundaries[:-1], boundaries[1:])
    ):
        if index == 0:
            selected = (confidence >= lower) & (confidence <= upper)
        else:
            selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            ece += selected.sum() / total * abs(
                correct[selected].mean() - confidence[selected].mean()
            )
    return float(ece)


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
    *,
    class_names: Sequence[str],
    ece_bins: int = 15,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    targets = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    num_classes = len(class_names)
    if targets.ndim != 1 or len(targets) == 0:
        raise ValueError("targets must be a nonempty one-dimensional array")
    if probabilities.shape != (len(targets), num_classes):
        raise ValueError("probability matrix shape is invalid")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("probability rows must sum to one")
    if (targets < 0).any() or (targets >= num_classes).any():
        raise ValueError("target is outside the fixed class range")

    predictions = probabilities.argmax(axis=1)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    rows: list[dict[str, Any]] = []
    for label_id, class_name in enumerate(class_names):
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
        rows.append(
            {
                "label_id": label_id,
                "class_name": str(class_name),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    per_class = pd.DataFrame(rows)
    support = per_class["support"].to_numpy(dtype=np.float64)
    total = float(support.sum())
    clipped = np.clip(probabilities, 1e-12, 1.0)
    one_hot = np.eye(num_classes, dtype=np.float64)[targets]
    metrics = {
        "accuracy": float((predictions == targets).mean()),
        "balanced_accuracy": float(per_class["recall"].mean()),
        "macro_f1": float(per_class["f1"].mean()),
        "weighted_f1": float(np.average(per_class["f1"], weights=support)),
        "macro_precision": float(per_class["precision"].mean()),
        "macro_recall": float(per_class["recall"].mean()),
        "weighted_precision": float(
            np.average(per_class["precision"], weights=support)
        ),
        "weighted_recall": float(
            np.average(per_class["recall"], weights=support)
        ),
        "nll": float(-np.log(clipped[np.arange(len(targets)), targets]).mean()),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        f"ece_{ece_bins}bin": expected_calibration_error(
            targets, probabilities, bins=ece_bins
        ),
    }
    if total != len(targets):
        raise AssertionError("per-class support does not sum to sample count")
    return metrics, per_class, matrix


def aggregate_seed_metrics(rows: Iterable[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        raise ValueError("seed metrics are empty")
    required = {"model_key", "seed"}
    if not required.issubset(frame.columns):
        raise ValueError("seed metrics require model_key and seed")
    metric_columns = [
        name for name in METRIC_NAMES if name in frame.columns
    ]
    if not metric_columns:
        metric_columns = [
            column
            for column in frame.columns
            if column not in {"model_key", "display_name", "seed", "parameters"}
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
    aggregate_rows: list[dict[str, Any]] = []
    for model_key, group in frame.groupby("model_key", sort=False):
        if len(group) < 2:
            raise ValueError(f"{model_key} requires at least two seeds")
        row: dict[str, Any] = {
            "model_key": model_key,
            "display_name": (
                str(group["display_name"].iloc[0])
                if "display_name" in group
                else model_key
            ),
            "parameters": (
                int(group["parameters"].iloc[0])
                if "parameters" in group
                else None
            ),
            "seeds": ",".join(str(value) for value in sorted(group["seed"])),
            "seed_count": len(group),
        }
        critical = float(student_t.ppf(0.975, df=len(group) - 1))
        for metric in metric_columns:
            values = group[metric].astype(float).to_numpy()
            mean = float(values.mean())
            std = float(values.std(ddof=1))
            half_width = critical * std / math.sqrt(len(values))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = mean - half_width
            row[f"{metric}_ci95_high"] = mean + half_width
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    if "macro_f1_mean" in aggregate:
        aggregate = aggregate.sort_values(
            ["macro_f1_mean", "accuracy_mean"],
            ascending=[False, False],
        ).reset_index(drop=True)
        aggregate.insert(0, "rank_by_macro_f1", np.arange(1, len(aggregate) + 1))
    return aggregate


def _prediction_frame(
    frame: pd.DataFrame,
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    output = frame[
        ["sample_id", "component_id", "packaged_path", "label", "label_id"]
    ].copy()
    output["target"] = targets
    output["prediction"] = predictions
    output["prediction_label"] = [
        class_names[int(value)] for value in predictions
    ]
    output["correct"] = predictions == targets
    output["confidence"] = probabilities.max(axis=1)
    for label_id in range(probabilities.shape[1]):
        output[f"probability_{label_id:02d}"] = probabilities[:, label_id]
    return output


def _save_seed_outputs(
    seed_root: Path,
    *,
    frame: pd.DataFrame,
    class_names: tuple[str, ...],
    record: CheckpointRecord,
    metrics: dict[str, float],
    per_class: pd.DataFrame,
    matrix: np.ndarray,
    logits: np.ndarray,
    probabilities: np.ndarray,
    runtime_seconds: float,
) -> dict[str, Any]:
    seed_root.mkdir(parents=True, exist_ok=False)
    targets = frame["label_id"].to_numpy(dtype=np.int64)
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    scalar = {
        **metrics,
        "model_key": record.model_key,
        "display_name": record.display_name,
        "seed": record.seed,
        "parameters": record.parameters,
        "samples": len(frame),
        "runtime_seconds": runtime_seconds,
        "checkpoint_sha256": record.actual_sha256,
    }
    write_json(scalar, seed_root / "test_metrics.json")
    per_class.to_csv(seed_root / "test_per_class.csv", index=False)
    pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    ).to_csv(seed_root / "test_confusion_matrix.csv", index_label="target")
    predictions_frame = _prediction_frame(
        frame,
        targets,
        predictions,
        probabilities,
        class_names,
    )
    predictions_frame.to_csv(seed_root / "test_predictions.csv", index=False)
    np.savez_compressed(
        seed_root / "test_arrays.npz",
        sample_ids=frame["sample_id"].astype(str).to_numpy(),
        targets=targets,
        predictions=predictions,
        logits=logits.astype(np.float32),
        probabilities=probabilities.astype(np.float32),
    )
    return scalar


def _environment(device: torch.device) -> dict[str, Any]:
    import timm
    import torchvision

    return {
        "captured_at": utc_now(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "cpu_count": os.cpu_count(),
        "fp32_inference": True,
        "tta": False,
        "seed_ensemble": False,
    }


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    inventory = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "artifact_inventory.json":
            continue
        inventory.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return inventory


def _verify_completed_output(config: EvaluationConfig) -> list[dict[str, Any]]:
    output = config.canonical_output_directory
    completed_path = output / "COMPLETED.json"
    inventory_path = output / "artifact_inventory.json"
    for path, description in (
        (completed_path, "completion marker"),
        (inventory_path, "artifact inventory"),
        (output / "sealed_test_seed_metrics.csv", "seed metrics"),
        (output / "sealed_test_aggregate.csv", "aggregate metrics"),
    ):
        _require_file(path, f"sealed-test {description}")
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    if completed.get("status") != "complete":
        raise ValueError("sealed-test completion marker is not complete")
    if completed.get("evaluation_id") != EVALUATION_ID:
        raise ValueError("sealed-test completion marker evaluation ID mismatch")
    ledger = json.loads(
        config.access_ledger_path.read_text(encoding="utf-8")
    )
    if ledger.get("status") not in {"publishing", "complete"}:
        raise ValueError(
            "canonical access ledger is neither publishing nor complete"
        )
    if ledger.get("evaluation_id") != EVALUATION_ID:
        raise ValueError("canonical access ledger evaluation ID mismatch")
    if Path(ledger.get("output_directory", "")).resolve() != output:
        raise ValueError("canonical access ledger output mismatch")
    if ledger.get("final_holdout_sha256") != EXPECTED_FINAL_HOLDOUT_SHA256:
        raise ValueError("canonical access ledger holdout identity mismatch")
    if (
        ledger.get("checkpoint_registry_sha256")
        != EXPECTED_CHECKPOINT_REGISTRY_SHA256
    ):
        raise ValueError("canonical access ledger checkpoint registry mismatch")
    if ledger.get("completed_marker_sha256") != sha256_file(completed_path):
        raise ValueError("canonical completion marker identity mismatch")
    if ledger.get("artifact_inventory_sha256") != sha256_file(inventory_path):
        raise ValueError("canonical artifact inventory identity mismatch")

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    listed = set()
    for item in inventory:
        relative = str(item["relative_path"])
        if relative in listed:
            raise ValueError("duplicate path in sealed-test inventory")
        listed.add(relative)
        path = (output / relative).resolve()
        try:
            path.relative_to(output)
        except ValueError as error:
            raise ValueError("unsafe path in sealed-test inventory") from error
        _require_file(path, f"sealed-test artifact {relative}")
        if path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"sealed-test artifact byte mismatch: {relative}")
        if sha256_file(path) != str(item["sha256"]):
            raise ValueError(f"sealed-test artifact hash mismatch: {relative}")
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "artifact_inventory.json"
    }
    if actual != listed:
        raise ValueError("sealed-test inventory membership mismatch")
    if ledger["status"] == "publishing":
        _transition_access_ledger(
            config,
            attempt_id=str(ledger["attempt_id"]),
            expected_status="publishing",
            status="complete",
            details={
                "recovered_from_publishing": True,
                "recovered_at": utc_now(),
            },
        )
    return inventory


def load_existing_results(config: EvaluationConfig) -> dict[str, Any]:
    if config.run_mode != "report_existing":
        raise ValueError("existing report loading requires report_existing mode")
    inventory = _verify_completed_output(config)
    output = config.canonical_output_directory
    return {
        "status": "report_loaded",
        "sealed_inference_executed": False,
        "output_directory": output,
        "seed_metrics": pd.read_csv(output / "sealed_test_seed_metrics.csv"),
        "aggregate": pd.read_csv(output / "sealed_test_aggregate.csv"),
        "artifact_inventory": inventory,
    }


def _load_checkpoint(
    record: CheckpointRecord,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    scratch_config = ScratchConfig(
        model_key=record.model_key,
        run_mode="development",
    )
    model, audit = build_model(scratch_config, device)
    if audit["architecture_sha256"] != record.architecture_sha256:
        raise ValueError(f"{record.model_key} architecture mismatch")
    if audit["backbone_schema_sha256"] != record.backbone_schema_sha256:
        raise ValueError(f"{record.model_key} backbone schema mismatch")
    state = torch.load(
        record.checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    del state
    return model, audit


def _infer(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, targets, _, _, _ in loader:
            logits = model(images.float().to(device, non_blocking=True)).float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite sealed-test logits")
            logits_parts.append(logits.cpu())
            target_parts.append(targets.cpu())
    logits = torch.cat(logits_parts).numpy()
    targets = torch.cat(target_parts).numpy()
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    return targets.astype(np.int64), probabilities.astype(np.float64), logits


def run_sealed_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    validate_sealed_execution_request(config)
    output = config.canonical_output_directory
    ensure_output_is_unused(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    preflight = run_preflight(config, verify_model_loading=True)
    records = discover_checkpoints(config)
    device = resolve_device(config.device)
    access_claim = claim_access_ledger(config)
    work_root: Path | None = None
    try:
        work_root = Path(
            tempfile.mkdtemp(
                prefix=f".{output.name}.working-",
                dir=output.parent,
            )
        )
        result_root = work_root / "result"
        result_root.mkdir()
        image_cache = work_root / "_sealed_images"
        frame, class_names = _materialize_holdout(config, image_cache)
        _, evaluation_transform = build_transforms()
        loader = DataLoader(
            _HoldoutDataset(frame, evaluation_transform),
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.num_workers > 0,
        )
        seed_rows: list[dict[str, Any]] = []
        for index, record in enumerate(records, start=1):
            model, audit = _load_checkpoint(record, device)
            started = time.perf_counter()
            targets, probabilities, logits = _infer(model, loader, device)
            runtime_seconds = time.perf_counter() - started
            expected_targets = frame["label_id"].to_numpy(dtype=np.int64)
            if not np.array_equal(targets, expected_targets):
                raise ValueError("loader target order differs from final manifest")
            metrics, per_class, matrix = compute_classification_metrics(
                targets,
                probabilities,
                class_names=class_names,
                ece_bins=15,
            )
            row = _save_seed_outputs(
                result_root / record.model_key / f"seed_{record.seed}",
                frame=frame,
                class_names=class_names,
                record=record,
                metrics=metrics,
                per_class=per_class,
                matrix=matrix,
                logits=logits,
                probabilities=probabilities,
                runtime_seconds=runtime_seconds,
            )
            seed_rows.append(row)
            print(
                json.dumps(
                    {
                        "event": "checkpoint_evaluated",
                        "index": index,
                        "total": len(records),
                        "model_key": record.model_key,
                        "seed": record.seed,
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "runtime_seconds": runtime_seconds,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del model, targets, probabilities, logits
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        seed_metrics = pd.DataFrame(seed_rows)
        aggregate = aggregate_seed_metrics(seed_metrics)
        seed_metrics.to_csv(
            result_root / "sealed_test_seed_metrics.csv", index=False
        )
        aggregate.to_csv(
            result_root / "sealed_test_aggregate.csv", index=False
        )
        write_json(preflight["protocol"], result_root / "protocol_report.json")
        write_json(
            {
                "status": preflight["status"],
                "sealed_inference_executed_during_preflight": preflight[
                    "sealed_inference_executed"
                ],
                "protocol": preflight["protocol"],
                "source_sha256": preflight["source_sha256"],
                "checkpoint_count": preflight["checkpoint_count"],
                "strict_load_audits": preflight["strict_load_audits"],
            },
            result_root / "preflight_report.json",
        )
        pd.DataFrame(preflight["checkpoint_table"]).to_csv(
            result_root / "checkpoint_inventory.csv", index=False
        )
        shutil.copy2(
            config.checkpoint_registry,
            result_root / config.checkpoint_registry.name,
        )
        write_json(_environment(device), result_root / "environment.json")
        write_json(
            {
                "evaluation_id": EVALUATION_ID,
                "protocol_id": PROTOCOL_ID,
                "attempt_id": access_claim["attempt_id"],
                "started_at": access_claim["started_at"],
                "completed_at": utc_now(),
                "acknowledgement": config.acknowledgement,
                "final_holdout_sha256": EXPECTED_FINAL_HOLDOUT_SHA256,
                "test_samples": FINAL_HOLDOUT_SAMPLES,
                "models": list(EXPECTED_MODEL_KEYS),
                "seeds": list(EXPECTED_SEEDS),
                "model_selection_performed": False,
                "tta": False,
                "seed_ensemble": False,
                "inference_precision": "fp32",
            },
            result_root / "access_ledger.json",
        )
        write_json(
            {
                "status": "complete",
                "evaluation_id": EVALUATION_ID,
                "completed_at": utc_now(),
                "checkpoint_count": len(records),
                "model_count": len(EXPECTED_MODEL_KEYS),
                "seed_count": len(EXPECTED_SEEDS),
            },
            result_root / "COMPLETED.json",
        )
        write_json(
            _artifact_inventory(result_root),
            result_root / "artifact_inventory.json",
        )
        shutil.rmtree(image_cache)
        publishing_ledger = _transition_access_ledger(
            config,
            attempt_id=str(access_claim["attempt_id"]),
            expected_status="started",
            status="publishing",
            details={
                "output_directory": str(output),
                "completed_marker_sha256": sha256_file(
                    result_root / "COMPLETED.json"
                ),
                "artifact_inventory_sha256": sha256_file(
                    result_root / "artifact_inventory.json"
                ),
            },
        )
        os.replace(result_root, output)
        completed_ledger = _transition_access_ledger(
            config,
            attempt_id=str(publishing_ledger["attempt_id"]),
            expected_status="publishing",
            status="complete",
            details={
                "recovered_from_publishing": False,
            },
        )
        shutil.rmtree(work_root, ignore_errors=True)
    except BaseException as error:
        if output.exists():
            raise
        failed: Path | None = None
        if work_root is not None and work_root.exists():
            failed = output.parent / (
                f".{output.name}.failed-{int(time.time())}"
            )
            os.replace(work_root, failed)
        try:
            current_ledger = json.loads(
                config.access_ledger_path.read_text(encoding="utf-8")
            )
            current_status = str(current_ledger.get("status"))
            _transition_access_ledger(
                config,
                attempt_id=str(access_claim["attempt_id"]),
                expected_status=current_status,
                status="failed",
                details={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "failed_work_directory": (
                        str(failed) if failed is not None else None
                    ),
                },
            )
        except BaseException:
            # The already-created "started" ledger still blocks an unsafe retry.
            pass
        raise

    return {
        "status": "complete",
        "sealed_inference_executed": True,
        "output_directory": output,
        "access_ledger": completed_ledger,
        "seed_metrics": pd.read_csv(output / "sealed_test_seed_metrics.csv"),
        "aggregate": pd.read_csv(output / "sealed_test_aggregate.csv"),
        "artifact_inventory": json.loads(
            (output / "artifact_inventory.json").read_text(encoding="utf-8")
        ),
    }


def execute_from_values(values: dict[str, Any]) -> dict[str, Any]:
    config = EvaluationConfig(
        repo_root=Path(values["REPO_ROOT"]),
        run_mode=str(values["RUN_MODE"]),
        acknowledgement=str(values["ACKNOWLEDGEMENT"]),
        device=str(values["DEVICE"]),
        batch_size=int(values["BATCH_SIZE"]),
        num_workers=int(values["NUM_WORKERS"]),
        verify_archive_hash=bool(values["VERIFY_ARCHIVE_HASH"]),
        verify_image_hashes=bool(values["VERIFY_IMAGE_HASHES"]),
        output_directory=Path(values["OUTPUT_DIRECTORY"]),
    )
    if config.run_mode == "preflight":
        return run_preflight(config)
    if config.run_mode == "report_existing":
        return load_existing_results(config)
    return run_sealed_evaluation(config)
