"""One-time sealed test evaluation for PlantWild's WaveDown ablation."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from agrivision4.evaluation.sealed_test import compute_classification_metrics
from plantwild.scratch_comparison.notebook_runtime import build_transforms, sha256_file
from plantwild.wavedown_ablation.analysis import holm_adjust
from plantwild.wavedown_ablation.experiment_core import ABLATION_PARAMETER_COUNTS, SEEDS
from plantwild.wavedown_ablation.model import create_wavedown_ablation


EVALUATION_ID = "plantwild_wavestarnet_wavedown_final_holdout_v01"
ACKNOWLEDGEMENT = "RUN_PLANTWILD_WAVEDOWN_FINAL_HOLDOUT_ONCE"
TEST_SHA256 = "83395b392417e6d1a6036bd19b2555600ea8fb4b36fa7075c6da294294c16748"
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "nll", "brier", "ece_15bin")
VARIANTS = ("full_wavestarnet", "parameter_matched_conventional_downsampling")
PRIMARY_PAIRED_METRICS = ("accuracy", "macro_f1")
PAIRED_RESAMPLES = 10_000
PAIRED_ANALYSIS_SEED = 20260915


@dataclass(frozen=True)
class CheckpointRecord:
    variant: str
    seed: int
    path: Path
    sha256: str
    display_name: str


@dataclass(frozen=True)
class WaveDownSealedSpec:
    repo_root: Path
    test_manifest: Path
    class_mapping: Path
    image_root: Path
    ablation_archive: Path
    full_root: Path
    output_root: Path

    @property
    def output_directory(self) -> Path:
        return self.output_root / EVALUATION_ID

    @property
    def zip_path(self) -> Path:
        return self.output_root / f"{EVALUATION_ID}.zip"


def build_spec(repo_root: str | Path) -> WaveDownSealedSpec:
    root = Path(repo_root).resolve()
    return WaveDownSealedSpec(
        repo_root=root,
        test_manifest=root / "plantwild/audit/plantwild-v2-audit-v2/manifests/test.csv",
        class_mapping=root / "plantwild/audit/plantwild-v2-audit-v2/manifests/class_mapping.json",
        image_root=root / "plantwild/dataset/plantwild_v2",
        ablation_archive=root / "plantwild/results/wavedown ablation/plantwild wavedown ablation results.zip",
        full_root=root / "plantwild/results/scratch comparison/wavestarnet-v2/plantwild_scratch_wavestarnet_transfer_v2_development_v01",
        output_root=root / "plantwild/results/sealed_test",
    )


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {"relative_path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_inventory.json"
    ]


def validate_artifact_inventory(root: Path) -> None:
    expected = _json(root / "artifact_inventory.json")
    if expected != _inventory(root):
        raise ValueError("artifact inventory mismatch")


def _class_names(mapping_path: Path) -> tuple[str, ...]:
    mapping = _json(mapping_path)
    values = mapping.get("class_to_id", mapping)
    class_to_id = {str(name): int(label_id) for name, label_id in values.items()}
    if sorted(class_to_id.values()) != list(range(105)):
        raise ValueError("class mapping is not the fixed 105-class mapping")
    return tuple(name for name, _ in sorted(class_to_id.items(), key=lambda item: item[1]))


def _test_frame(spec: WaveDownSealedSpec) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if sha256_file(spec.test_manifest) != TEST_SHA256:
        raise ValueError("test manifest hash mismatch")
    names = _class_names(spec.class_mapping)
    frame = pd.read_csv(spec.test_manifest)
    required = {"sample_id", "component_id", "split", "label", "label_id", "source_relative_path", "file_sha256"}
    if required - set(frame.columns) or len(frame) != 1523 or frame.sample_id.duplicated().any():
        raise ValueError("sealed test manifest contract mismatch")
    if set(frame.split.astype(str)) != {"test"} or frame.component_id.duplicated().any():
        raise ValueError("sealed test split/component contract mismatch")
    if not np.array_equal(frame.label_id.to_numpy(int), np.array([names.index(label) for label in frame.label])):
        raise ValueError("sealed test labels do not match class mapping")
    frame = frame.copy()
    frame["absolute_path"] = frame.source_relative_path.map(lambda value: spec.image_root / str(value))
    missing = [str(path) for path in frame.absolute_path if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"sealed test images missing: {missing[:3]}")
    return frame, names


class _TestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: Any):
        self.frame, self.transform = frame.reset_index(drop=True), transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        with Image.open(row.absolute_path) as image:
            return self.transform(image.convert("RGB")), int(row.label_id)


def align_test_predictions(reference: pd.DataFrame, candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"sample_id", "component_id", "true_label_id", "predicted_label_id"}
    for name, frame in (("reference", reference), ("candidate", candidate)):
        if required - set(frame.columns) or frame.sample_id.duplicated().any():
            raise ValueError(f"{name} predictions are not uniquely identifiable")
    left, right = (frame.sort_values("sample_id").reset_index(drop=True) for frame in (reference, candidate))
    if left.sample_id.tolist() != right.sample_id.tolist():
        raise ValueError("test sample IDs do not match")
    if not np.array_equal(left.component_id, right.component_id):
        raise ValueError("test component IDs do not match")
    if not np.array_equal(left.true_label_id, right.true_label_id):
        raise ValueError("test ground-truth labels do not match")
    return left, right


def _archive_root(spec: WaveDownSealedSpec, work: Path) -> Path:
    with zipfile.ZipFile(spec.ablation_archive) as archive:
        roots = sorted({Path(name).parts[0] for name in archive.namelist() if name.endswith("run_manifest.json")})
        if len(roots) != 1:
            raise ValueError("ablation archive must contain exactly one result root")
        archive.extractall(work)
    return work / roots[0]


def _records(spec: WaveDownSealedSpec, ablation_root: Path) -> list[CheckpointRecord]:
    sources = {
        "full_wavestarnet": (spec.full_root, "WavestarNet"),
        "parameter_matched_conventional_downsampling": (ablation_root, "WavestarNet (parameter-matched conventional downsampling)"),
    }
    records = []
    for variant, (root, display_name) in sources.items():
        manifest = _json(root / "run_manifest.json")
        if manifest.get("status") != "complete" or tuple(manifest.get("active_seeds", ())) != SEEDS:
            raise ValueError(f"{variant} source run is not frozen and complete")
        for seed in SEEDS:
            path = root / f"seed_{seed}/best_checkpoint.pt"
            sidecar = root / f"seed_{seed}/best_checkpoint.sha256"
            if not path.is_file() or not sidecar.is_file():
                raise FileNotFoundError(f"missing {variant} seed {seed} checkpoint")
            observed = sha256_file(path)
            if observed != sidecar.read_text(encoding="utf-8").strip():
                raise ValueError(f"{variant} seed {seed} checkpoint hash mismatch")
            records.append(CheckpointRecord(variant, seed, path, observed, display_name))
    return records


def _strict_load(record: CheckpointRecord) -> dict[str, Any]:
    model = create_wavedown_ablation(record.variant).cpu()
    parameters = sum(value.numel() for value in model.parameters())
    if parameters != ABLATION_PARAMETER_COUNTS[record.variant]:
        raise ValueError("model parameter contract mismatch")
    state = torch.load(record.path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return {"variant": record.variant, "seed": record.seed, "parameters": parameters, "checkpoint_sha256": record.sha256, "strict_load": True}


def run_preflight(spec: WaveDownSealedSpec) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="plantwild-wavedown-preflight-") as temp:
        ablation_root = _archive_root(spec, Path(temp))
        frame, names = _test_frame(spec)
        records = _records(spec, ablation_root)
        audits = [_strict_load(record) for record in records]
    return {"status": "preflight_passed", "sealed_inference_executed": False, "test_samples": len(frame), "class_count": len(names), "test_manifest_sha256": TEST_SHA256, "strict_load_audits": audits}


def _metric_values(targets: np.ndarray, probabilities: np.ndarray, names: tuple[str, ...]) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    metrics, per_class, matrix = compute_classification_metrics(targets, probabilities, class_names=names, ece_bins=15)
    return {metric: float(metrics[metric]) for metric in METRICS}, per_class, matrix


def _stratified_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    indices = [rng.choice(np.flatnonzero(labels == label), size=np.count_nonzero(labels == label), replace=True) for label in np.unique(labels)]
    return np.concatenate(indices)


def _load_test_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
    except ValueError as error:
        if "Object arrays" in str(error):
            raise ValueError("test arrays contain unsafe object sample_ids") from error
        raise
    required = {"sample_ids", "targets", "probabilities"}
    if required - set(values) or values["sample_ids"].dtype.kind == "O":
        raise ValueError("test arrays contain invalid sample_ids")
    return values


def paired_metric_summary(full_root: Path, ablation_root: Path, *, resamples: int = 2000, seed: int = 20260915) -> pd.DataFrame:
    rng, rows = np.random.default_rng(seed), []
    for run_seed in SEEDS:
        left = _load_test_arrays(full_root / f"seed_{run_seed}/test_arrays.npz")
        right = _load_test_arrays(ablation_root / f"seed_{run_seed}/test_arrays.npz")
        if not (np.array_equal(left["sample_ids"], right["sample_ids"]) and np.array_equal(left["targets"], right["targets"])):
            raise ValueError("test arrays do not align")
        class_count = left["probabilities"].shape[1]
        names = tuple(str(index) for index in range(class_count))
        full, _, _ = _metric_values(left["targets"], left["probabilities"], names)
        ablation, _, _ = _metric_values(right["targets"], right["probabilities"], names)
        for metric in METRICS:
            rows.append({"seed": run_seed, "metric": metric, "full_wavestarnet": full[metric], "wavedown_ablation": ablation[metric], "full_minus_ablation": full[metric] - ablation[metric]})
    result = pd.DataFrame(rows)
    summary = []
    for metric, group in result.groupby("metric", sort=False):
        values = group.full_minus_ablation.to_numpy(float)
        summary.append({"metric": metric, "full_wavestarnet_mean": group.full_wavestarnet.mean(), "wavedown_ablation_mean": group.wavedown_ablation.mean(), "full_minus_ablation": values.mean(), "seed_sd": values.std(ddof=1)})
    return pd.DataFrame(summary)


def _primary_classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    classes: int,
) -> dict[str, float]:
    """Return the two predeclared paired endpoints from one confusion matrix."""
    matrix = np.bincount(
        targets * classes + predictions,
        minlength=classes * classes,
    ).reshape(classes, classes)
    support = matrix.sum(axis=1).astype(float)
    predicted = matrix.sum(axis=0).astype(float)
    true_positive = np.diag(matrix).astype(float)
    f1 = np.divide(
        2.0 * true_positive,
        support + predicted,
        out=np.zeros(classes, dtype=float),
        where=(support + predicted) > 0,
    )
    return {
        "accuracy": float(np.trace(matrix) / len(targets)),
        "macro_f1": float(f1.mean()),
    }


def paired_wavedown_statistics(
    full_root: Path,
    ablation_root: Path,
    *,
    resamples: int = PAIRED_RESAMPLES,
    seed: int = PAIRED_ANALYSIS_SEED,
) -> pd.DataFrame:
    """Compute report-only paired inference from saved sealed-test arrays.

    The reference-minus-candidate sign convention means a positive effect favors
    complete WavestarNet. Bootstrap resampling is class-stratified within every
    seed. The label-swap test swaps paired predictions sample-wise within every
    seed, then averages the three seed effects. Holm correction is deliberately
    limited to the two primary predictive endpoints in this confirmation.
    """
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    bundles: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, int, list[np.ndarray]]] = []
    rows: list[dict[str, float | int | str]] = []
    for run_seed in SEEDS:
        full = _load_test_arrays(full_root / f"seed_{run_seed}" / "test_arrays.npz")
        candidate = _load_test_arrays(ablation_root / f"seed_{run_seed}" / "test_arrays.npz")
        if not (
            np.array_equal(full["sample_ids"], candidate["sample_ids"])
            and np.array_equal(full["targets"], candidate["targets"])
        ):
            raise ValueError(f"sealed arrays do not align for seed {run_seed}")
        targets = full["targets"].astype(int, copy=False)
        full_prediction = full["predictions"].astype(int, copy=False)
        candidate_prediction = candidate["predictions"].astype(int, copy=False)
        classes = int(max(targets.max(), full_prediction.max(), candidate_prediction.max()) + 1)
        if classes != 105 or len(targets) != 1523:
            raise ValueError(f"sealed PlantWild contract mismatch for seed {run_seed}")
        groups = [np.flatnonzero(targets == label) for label in np.unique(targets)]
        if not groups or any(len(group) == 0 for group in groups):
            raise ValueError("class-stratified bootstrap groups are invalid")
        bundles.append((run_seed, targets, full_prediction, candidate_prediction, classes, groups))
        full_metrics = _primary_classification_metrics(targets, full_prediction, classes)
        candidate_metrics = _primary_classification_metrics(targets, candidate_prediction, classes)
        for metric in PRIMARY_PAIRED_METRICS:
            full_value = full_metrics[metric]
            candidate_value = candidate_metrics[metric]
            rows.append(
                {
                    "seed": run_seed,
                    "metric": metric,
                    "full_wavestarnet": full_value,
                    "parameter_matched_conventional": candidate_value,
                    "full_minus_conventional": full_value - candidate_value,
                }
            )

    seed_metrics = pd.DataFrame(rows)
    summaries: list[dict[str, float | int | str | bool]] = []
    observed = {
        metric: float(
            seed_metrics.loc[seed_metrics.metric == metric, "full_minus_conventional"].mean()
        )
        for metric in PRIMARY_PAIRED_METRICS
    }
    bootstrap = {metric: np.empty(resamples, dtype=float) for metric in PRIMARY_PAIRED_METRICS}
    null = {metric: np.empty(resamples, dtype=float) for metric in PRIMARY_PAIRED_METRICS}
    for index in range(resamples):
        bootstrap_effects = {metric: [] for metric in PRIMARY_PAIRED_METRICS}
        null_effects = {metric: [] for metric in PRIMARY_PAIRED_METRICS}
        for _, targets, full_prediction, candidate_prediction, classes, groups in bundles:
            indices = np.concatenate(
                [rng.choice(group, size=len(group), replace=True) for group in groups]
            )
            full_bootstrap = _primary_classification_metrics(targets[indices], full_prediction[indices], classes)
            candidate_bootstrap = _primary_classification_metrics(targets[indices], candidate_prediction[indices], classes)
            swap = rng.random(len(targets)) < 0.5
            random_full = np.where(swap, candidate_prediction, full_prediction)
            random_candidate = np.where(swap, full_prediction, candidate_prediction)
            full_null = _primary_classification_metrics(targets, random_full, classes)
            candidate_null = _primary_classification_metrics(targets, random_candidate, classes)
            for metric in PRIMARY_PAIRED_METRICS:
                bootstrap_effects[metric].append(full_bootstrap[metric] - candidate_bootstrap[metric])
                null_effects[metric].append(full_null[metric] - candidate_null[metric])
        for metric in PRIMARY_PAIRED_METRICS:
            bootstrap[metric][index] = float(np.mean(bootstrap_effects[metric]))
            null[metric][index] = float(np.mean(null_effects[metric]))

    for metric in PRIMARY_PAIRED_METRICS:
        observed_by_seed = seed_metrics.loc[
            seed_metrics.metric == metric,
            "full_minus_conventional",
        ].to_numpy(float)
        metric_rows = seed_metrics.loc[seed_metrics.metric == metric]
        summaries.append(
            {
                "metric": metric,
                "full_wavestarnet_mean": float(metric_rows.full_wavestarnet.mean()),
                "parameter_matched_conventional_mean": float(metric_rows.parameter_matched_conventional.mean()),
                "full_minus_conventional": observed[metric],
                "full_minus_conventional_pp": observed[metric] * 100.0,
                "seed_sd": float(observed_by_seed.std(ddof=1)),
                "bootstrap95_low": float(np.quantile(bootstrap[metric], 0.025)),
                "bootstrap95_high": float(np.quantile(bootstrap[metric], 0.975)),
                "bootstrap95_low_pp": float(np.quantile(bootstrap[metric], 0.025) * 100.0),
                "bootstrap95_high_pp": float(np.quantile(bootstrap[metric], 0.975) * 100.0),
                "paired_randomization_p_raw": (1 + int(np.count_nonzero(np.abs(null[metric]) >= abs(observed[metric])))) / (resamples + 1),
                "bootstrap_resamples": resamples,
                "randomization_resamples": resamples,
            }
        )
    result = pd.DataFrame(summaries)
    result["paired_randomization_p_holm"] = holm_adjust(result.paired_randomization_p_raw.tolist())
    result["holm_reject_0_05"] = result.paired_randomization_p_holm < 0.05
    return result


def per_class_f1_summary(result_root: Path, *, expected_class_count: int = 105) -> pd.DataFrame:
    grouped: dict[str, pd.DataFrame] = {}
    for variant in VARIANTS:
        frames = [pd.read_csv(result_root / variant / f"seed_{seed}" / "test_per_class.csv") for seed in SEEDS]
        joined = pd.concat(frames, ignore_index=True)
        grouped[variant] = joined.groupby(["label_id", "class_name"], as_index=False).agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"), support=("support", "first"))
    full = grouped["full_wavestarnet"].rename(columns={"f1_mean": "full_wavestarnet_f1_mean", "f1_std": "full_wavestarnet_f1_std"})
    ablation = grouped["parameter_matched_conventional_downsampling"].rename(columns={"f1_mean": "wavedown_ablation_f1_mean", "f1_std": "wavedown_ablation_f1_std"})
    result = full.merge(ablation, on=["label_id", "class_name", "support"], how="inner", validate="one_to_one")
    if len(result) != expected_class_count:
        raise ValueError(f"per-class reports do not align to all {expected_class_count} labels")
    result["full_minus_ablation_f1"] = result.full_wavestarnet_f1_mean - result.wavedown_ablation_f1_mean
    return result.sort_values("label_id").reset_index(drop=True)


def run_sealed_evaluation(spec: WaveDownSealedSpec, *, acknowledgement: str, device: str = "auto", batch_size: int = 64, num_workers: int = 0) -> dict[str, Any]:
    if acknowledgement != ACKNOWLEDGEMENT:
        raise PermissionError("sealed acknowledgement is absent or incorrect")
    if spec.output_directory.exists() or spec.zip_path.exists():
        raise FileExistsError("sealed output already exists")
    preflight = run_preflight(spec)
    selected = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
    spec.output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{EVALUATION_ID}-", dir=spec.output_root) as temp:
        work, result_root = Path(temp), Path(temp) / "result"
        result_root.mkdir()
        ablation_source = _archive_root(spec, work / "ablation-source")
        frame, names = _test_frame(spec)
        records = _records(spec, ablation_source)
        _, transform = build_transforms()
        loader = DataLoader(_TestDataset(frame, transform), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=selected.type == "cuda")
        rows = []
        for record in records:
            model = create_wavedown_ablation(record.variant).to(selected)
            model.load_state_dict(torch.load(record.path, map_location=selected, weights_only=True), strict=True)
            model.eval(); started = time.perf_counter(); logits, targets = [], []
            with torch.inference_mode():
                for images, labels in loader:
                    output = model(images.to(selected, non_blocking=True)).float()
                    if not torch.isfinite(output).all():
                        raise FloatingPointError("non-finite logits")
                    logits.append(output.cpu()); targets.append(labels.cpu())
            logits_np, targets_np = torch.cat(logits).numpy(), torch.cat(targets).numpy()
            shifted = logits_np - logits_np.max(axis=1, keepdims=True); exp = np.exp(shifted); probs = exp / exp.sum(axis=1, keepdims=True)
            metrics, per_class, matrix = _metric_values(targets_np, probs, names)
            seed_root = result_root / record.variant / f"seed_{record.seed}"; seed_root.mkdir(parents=True)
            prediction = frame[["sample_id", "component_id", "label", "label_id"]].rename(columns={"label_id": "true_label_id"}).copy()
            prediction["predicted_label_id"] = probs.argmax(axis=1); prediction["correct"] = prediction.predicted_label_id.eq(prediction.true_label_id); prediction["confidence"] = probs.max(axis=1)
            prediction.to_csv(seed_root / "test_predictions.csv", index=False)
            per_class.to_csv(seed_root / "test_per_class.csv", index=False)
            pd.DataFrame(matrix, index=names, columns=names).to_csv(seed_root / "test_confusion_matrix.csv", index_label="target")
            np.savez_compressed(seed_root / "test_arrays.npz", sample_ids=frame.sample_id.to_numpy(dtype=str), targets=targets_np, predictions=prediction.predicted_label_id.to_numpy(), logits=logits_np.astype(np.float32), probabilities=probs.astype(np.float32))
            row = metrics | {"variant": record.variant, "display_name": record.display_name, "seed": record.seed, "parameters": ABLATION_PARAMETER_COUNTS[record.variant], "checkpoint_sha256": record.sha256, "runtime_seconds": time.perf_counter() - started}
            _write_json(seed_root / "test_metrics.json", row); rows.append(row)
            print(json.dumps({"event": "checkpoint_evaluated", "variant": record.variant, "seed": record.seed, "macro_f1": row["macro_f1"]}), flush=True)
            del model, logits, targets
            if selected.type == "cuda": torch.cuda.empty_cache()
        seed_metrics = pd.DataFrame(rows); seed_metrics.to_csv(result_root / "test_seed_metrics.csv", index=False)
        aggregate = seed_metrics.groupby("variant", as_index=False)[list(METRICS)].agg(["mean", "std"]); aggregate.columns = ["_".join(column).strip("_") for column in aggregate.columns]; aggregate.to_csv(result_root / "test_aggregate.csv", index=False)
        paired = paired_metric_summary(result_root / "full_wavestarnet", result_root / "parameter_matched_conventional_downsampling"); paired.to_csv(result_root / "paired_comparison.csv", index=False)
        per_class = per_class_f1_summary(result_root); per_class.to_csv(result_root / "per_class_f1_comparison.csv", index=False)
        _write_json(result_root / "preflight_report.json", preflight)
        _write_json(result_root / "evaluation_manifest.json", {"evaluation_id": EVALUATION_ID, "completed_at": _utc_now(), "test_manifest_sha256": TEST_SHA256, "test_samples": len(frame), "seeds": list(SEEDS), "variants": list(VARIANTS), "test_accessed": True})
        _write_json(result_root / "COMPLETED.json", {"status": "complete", "evaluation_id": EVALUATION_ID, "completed_at": _utc_now()})
        _write_json(result_root / "artifact_inventory.json", _inventory(result_root)); validate_artifact_inventory(result_root)
        shutil.make_archive(str(spec.output_root / EVALUATION_ID), "zip", result_root)
        shutil.move(str(result_root), str(spec.output_directory))
    return {"status": "complete", "sealed_inference_executed": True, "output_directory": spec.output_directory, "zip_path": spec.zip_path, "seed_metrics": seed_metrics, "aggregate": aggregate, "paired": paired, "per_class": per_class}


def load_existing_results(spec: WaveDownSealedSpec) -> dict[str, Any]:
    validate_artifact_inventory(spec.output_directory)
    return {"status": "report_loaded", "sealed_inference_executed": False, "output_directory": spec.output_directory, "zip_path": spec.zip_path, "seed_metrics": pd.read_csv(spec.output_directory / "test_seed_metrics.csv"), "aggregate": pd.read_csv(spec.output_directory / "test_aggregate.csv"), "paired": pd.read_csv(spec.output_directory / "paired_comparison.csv"), "per_class": pd.read_csv(spec.output_directory / "per_class_f1_comparison.csv")}


def augment_existing_analysis(spec: WaveDownSealedSpec) -> Path:
    """Add derived report-only artifacts without reading test images or checkpoints."""
    root = spec.output_directory
    if not (root / "COMPLETED.json").is_file():
        raise FileNotFoundError("sealed evaluation is not complete")
    per_class_f1_summary(root).to_csv(root / "per_class_f1_comparison.csv", index=False)
    paired_statistics = paired_wavedown_statistics(
        root / "full_wavestarnet",
        root / "parameter_matched_conventional_downsampling",
    )
    paired_statistics.to_csv(root / "paired_test_statistics.csv", index=False)
    markdown_lines = [
        "# PlantWild WaveDown Post-Selection Confirmation",
        "",
        "This report-only analysis uses the saved three-seed test predictions.",
        "The full and parameter-matched conventional variants were specified before this PlantWild test-set confirmation.",
        "No PlantWild test-set result was used to select the WavestarNet architecture or training protocol.",
        "Positive effects favor full WavestarNet. Confidence intervals use 10,000 class-stratified paired bootstrap resamples.",
        "Paired label-swap randomization p-values are Holm-adjusted across Accuracy and Macro-F1 only.",
        "",
        paired_statistics.to_markdown(index=False),
        "",
    ]
    (root / "paired_test_statistics.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    _write_json(
        root / "paired_test_statistics_manifest.json",
        {
            "analysis_type": "post_selection_test_set_confirmation",
            "comparison": "full_wavestarnet_minus_parameter_matched_conventional_downsampling",
            "primary_metrics": list(PRIMARY_PAIRED_METRICS),
            "bootstrap_method": "10,000 class-stratified paired image-level resamples averaged across matching seeds",
            "paired_test": "10,000 sample-wise paired label-swap randomization resamples averaged across matching seeds",
            "holm_family": "Accuracy and Macro-F1 only",
            "random_seed": PAIRED_ANALYSIS_SEED,
            "test_manifest_sha256": TEST_SHA256,
            "test_images_loaded": False,
            "checkpoints_loaded": False,
        },
    )
    manifest = _json(root / "evaluation_manifest.json")
    manifest["report_analysis_completed_at"] = _utc_now()
    manifest["paired_test_statistics_completed_at"] = _utc_now()
    _write_json(root / "evaluation_manifest.json", manifest)
    _write_json(root / "artifact_inventory.json", _inventory(root))
    validate_artifact_inventory(root)
    shutil.make_archive(str(spec.output_root / EVALUATION_ID), "zip", root)
    return root / "per_class_f1_comparison.csv"


def execute(spec: WaveDownSealedSpec, *, run_mode: str, acknowledgement: str = "", device: str = "auto", batch_size: int = 64, num_workers: int = 0) -> dict[str, Any]:
    if run_mode == "preflight":
        return run_preflight(spec)
    if run_mode == "sealed_evaluation":
        return run_sealed_evaluation(spec, acknowledgement=acknowledgement, device=device, batch_size=batch_size, num_workers=num_workers)
    if run_mode == "report_existing":
        return load_existing_results(spec)
    raise ValueError("run_mode must be preflight, sealed_evaluation, or report_existing")
