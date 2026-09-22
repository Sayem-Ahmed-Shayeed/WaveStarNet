"""Paired statistical analysis for the AgriVision4 ablation study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


SEEDS = (42, 2024, 3407)
METRICS = ("macro_f1", "accuracy", "weighted_f1")
REFERENCE_ARCHIVE = "agrivision4/results wavestarnet.zip"
REFERENCE_PREFIX = (
    "results/scratch comparison/wavestarnet_big_transfer_v2/"
    "agrivision4_scratch_wavestarnet_transfer_v2_development_v01"
)
REQUIRED_COLUMNS = (
    "sample_id",
    "component_id",
    "true_label_id",
    "predicted_label_id",
)


@dataclass(frozen=True)
class PredictionBundle:
    sample_ids: np.ndarray
    true_labels: np.ndarray
    component_ids: np.ndarray
    predictions: np.ndarray
    source_hashes: dict[str, str]


def classification_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    *,
    num_classes: int,
) -> dict[str, float]:
    true = np.asarray(true_labels, dtype=np.int64)
    predicted = np.asarray(predicted_labels, dtype=np.int64)
    if true.shape != predicted.shape or true.ndim != 1:
        raise ValueError("true and predicted labels must be aligned one-dimensional arrays")
    if true.size == 0:
        raise ValueError("metrics require at least one observation")
    if true.min() < 0 or predicted.min() < 0:
        raise ValueError("labels must be non-negative")
    if true.max() >= num_classes or predicted.max() >= num_classes:
        raise ValueError("label exceeds num_classes")

    confusion = np.bincount(
        true * num_classes + predicted,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    true_support = confusion.sum(axis=1).astype(np.float64)
    predicted_support = confusion.sum(axis=0).astype(np.float64)
    true_positive = np.diag(confusion).astype(np.float64)
    denominator = true_support + predicted_support
    per_class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros(num_classes, dtype=np.float64),
        where=denominator > 0,
    )
    return {
        "accuracy": float(true_positive.sum() / true.size),
        "macro_f1": float(per_class_f1.mean()),
        "weighted_f1": float(np.dot(per_class_f1, true_support) / true.size),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("p-values must be a finite one-dimensional sequence")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be within [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[index])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(len(values), dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def align_prediction_frames(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for name, frame in (("reference", reference), ("candidate", candidate)):
        missing = set(REQUIRED_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"{name} predictions missing columns: {sorted(missing)}")
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"{name} predictions contain duplicate sample_id values")

    left = reference.loc[:, REQUIRED_COLUMNS].sort_values("sample_id").reset_index(drop=True)
    right = candidate.loc[:, REQUIRED_COLUMNS].sort_values("sample_id").reset_index(drop=True)
    if left["sample_id"].tolist() != right["sample_id"].tolist():
        raise ValueError("prediction sample sets do not match")
    if not np.array_equal(left["true_label_id"], right["true_label_id"]):
        raise ValueError("ground-truth labels do not match")
    if not np.array_equal(left["component_id"], right["component_id"]):
        raise ValueError("component identifiers do not match")
    return left, right


def stratified_bootstrap_indices(
    labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = np.asarray(labels)
    sampled = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        sampled.append(rng.choice(indices, size=len(indices), replace=True))
    result = np.concatenate(sampled)
    rng.shuffle(result)
    return result


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_prediction_csv(
    source: Path,
    seed: int,
    *,
    member_prefix: str | None = None,
) -> tuple[pd.DataFrame, str, str]:
    suffix = f"seed_{seed}/validation_predictions.csv"
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            if member_prefix is None:
                matches = [name for name in archive.namelist() if name.endswith(suffix)]
                if len(matches) != 1:
                    raise ValueError(
                        f"expected one {suffix} in {source}, found {len(matches)}"
                    )
                member = matches[0]
            else:
                member = f"{member_prefix}/{suffix}"
                if member not in archive.namelist():
                    raise FileNotFoundError(f"missing archive member: {member}")
            raw = archive.read(member)
        return pd.read_csv(io.BytesIO(raw)), f"{source}::{member}", _sha256(raw)

    path = source / suffix
    raw = path.read_bytes()
    return pd.read_csv(io.BytesIO(raw)), str(path), _sha256(raw)


def load_prediction_bundle(
    source: Path,
    *,
    seeds: Sequence[int] = SEEDS,
    member_prefix: str | None = None,
    reference_bundle: PredictionBundle | None = None,
) -> PredictionBundle:
    aligned_predictions = []
    source_hashes: dict[str, str] = {}
    canonical: pd.DataFrame | None = None
    for seed in seeds:
        frame, source_name, digest = _read_prediction_csv(
            source,
            seed,
            member_prefix=member_prefix,
        )
        source_hashes[source_name] = digest
        if canonical is None:
            canonical = frame
        else:
            canonical, frame = align_prediction_frames(canonical, frame)
        aligned_predictions.append(
            frame.sort_values("sample_id")["predicted_label_id"].to_numpy(np.int64)
        )
    assert canonical is not None
    canonical = canonical.sort_values("sample_id").reset_index(drop=True)

    if reference_bundle is not None:
        reference_frame = pd.DataFrame(
            {
                "sample_id": reference_bundle.sample_ids,
                "component_id": reference_bundle.component_ids,
                "true_label_id": reference_bundle.true_labels,
                "predicted_label_id": reference_bundle.predictions[0],
            }
        )
        reference_frame, canonical = align_prediction_frames(reference_frame, canonical)

    components = canonical["component_id"].astype(str).to_numpy()
    labels = canonical["true_label_id"].to_numpy(np.int64)
    component_labels = pd.DataFrame(
        {"component_id": components, "true_label_id": labels}
    ).groupby("component_id")["true_label_id"].nunique()
    if (component_labels > 1).any():
        raise ValueError("a validation component contains multiple ground-truth classes")

    return PredictionBundle(
        sample_ids=canonical["sample_id"].astype(str).to_numpy(),
        true_labels=labels,
        component_ids=components,
        predictions=np.stack(aligned_predictions),
        source_hashes=source_hashes,
    )


def _mean_seed_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    per_seed = [
        classification_metrics(labels, prediction, num_classes=num_classes)
        for prediction in predictions
    ]
    return {
        metric: float(np.mean([row[metric] for row in per_seed]))
        for metric in METRICS
    }


def _seed_metric_deltas(
    labels: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    num_classes: int,
) -> dict[str, np.ndarray]:
    output = {metric: [] for metric in METRICS}
    for ref_prediction, candidate_prediction in zip(reference, candidate):
        ref_metrics = classification_metrics(
            labels, ref_prediction, num_classes=num_classes
        )
        candidate_metrics = classification_metrics(
            labels, candidate_prediction, num_classes=num_classes
        )
        for metric in METRICS:
            output[metric].append(candidate_metrics[metric] - ref_metrics[metric])
    return {metric: np.asarray(values) for metric, values in output.items()}


def paired_resampling_analysis(
    reference: PredictionBundle,
    candidate: PredictionBundle,
    *,
    num_classes: int,
    resamples: int,
    random_seed: int,
) -> dict[str, dict[str, float | list[float]]]:
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if reference.predictions.shape != candidate.predictions.shape:
        raise ValueError("reference and candidate prediction arrays do not align")
    if not np.array_equal(reference.sample_ids, candidate.sample_ids):
        raise ValueError("reference and candidate sample identifiers do not align")
    if not np.array_equal(reference.true_labels, candidate.true_labels):
        raise ValueError("reference and candidate labels do not align")
    if not np.array_equal(reference.component_ids, candidate.component_ids):
        raise ValueError("reference and candidate components do not align")

    labels = reference.true_labels
    reference_mean = _mean_seed_metrics(labels, reference.predictions, num_classes)
    candidate_mean = _mean_seed_metrics(labels, candidate.predictions, num_classes)
    observed = {
        metric: candidate_mean[metric] - reference_mean[metric] for metric in METRICS
    }
    seed_deltas = _seed_metric_deltas(
        labels,
        reference.predictions,
        candidate.predictions,
        num_classes,
    )
    bootstrap = {metric: np.empty(resamples, dtype=np.float64) for metric in METRICS}
    null = {metric: np.empty(resamples, dtype=np.float64) for metric in METRICS}
    bootstrap_rng = np.random.default_rng(random_seed)
    permutation_rng = np.random.default_rng(random_seed + 1)

    for iteration in range(resamples):
        indices = stratified_bootstrap_indices(labels, bootstrap_rng)
        boot_reference = _mean_seed_metrics(
            labels[indices], reference.predictions[:, indices], num_classes
        )
        boot_candidate = _mean_seed_metrics(
            labels[indices], candidate.predictions[:, indices], num_classes
        )
        for metric in METRICS:
            bootstrap[metric][iteration] = boot_candidate[metric] - boot_reference[metric]

        swap = permutation_rng.random(len(labels)) < 0.5
        permuted_reference = np.where(swap, candidate.predictions, reference.predictions)
        permuted_candidate = np.where(swap, reference.predictions, candidate.predictions)
        null_reference = _mean_seed_metrics(labels, permuted_reference, num_classes)
        null_candidate = _mean_seed_metrics(labels, permuted_candidate, num_classes)
        for metric in METRICS:
            null[metric][iteration] = null_candidate[metric] - null_reference[metric]

    result: dict[str, dict[str, float | list[float]]] = {}
    degrees_of_freedom = reference.predictions.shape[0] - 1
    critical_t = float(student_t.ppf(0.975, degrees_of_freedom))
    for metric in METRICS:
        paired_values = seed_deltas[metric]
        paired_sd = float(np.std(paired_values, ddof=1))
        seed_margin = critical_t * paired_sd / math.sqrt(len(paired_values))
        raw_p = float(
            (1 + np.count_nonzero(np.abs(null[metric]) >= abs(observed[metric])))
            / (resamples + 1)
        )
        result[metric] = {
            "reference_mean": reference_mean[metric],
            "candidate_mean": candidate_mean[metric],
            "delta": observed[metric],
            "seed_deltas": paired_values.tolist(),
            "paired_seed_delta_sd": paired_sd,
            "paired_seed_t95_low": observed[metric] - seed_margin,
            "paired_seed_t95_high": observed[metric] + seed_margin,
            "bootstrap95_low": float(np.quantile(bootstrap[metric], 0.025)),
            "bootstrap95_high": float(np.quantile(bootstrap[metric], 0.975)),
            "paired_randomization_p_raw": raw_p,
        }
    return result


def _stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**32)


def _format_p(value: float) -> str:
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def _write_markdown(
    wide: pd.DataFrame,
    destination: Path,
    *,
    resamples: int,
) -> None:
    lines = [
        "# AgriVision4 Ablation Statistical Analysis",
        "",
        (
            "Paired comparisons use the same 915-image locked validation set and "
            "matching seeds (42, 2024, 3407). Confidence intervals are based on "
            f"{resamples:,} class-stratified paired bootstrap resamples. P-values use "
            f"{resamples:,} paired randomization resamples and are Holm-corrected within "
            "each metric family."
        ),
        "",
        "| Variant | Params | Macro-F1 (%) | Delta (pp) | Bootstrap 95% CI (pp) | Holm p | Accuracy (%) | Delta (pp) | Bootstrap 95% CI (pp) | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in wide.iterrows():
        lines.append(
            "| {display_name} | {parameters:,} | {macro_f1_candidate_mean_pct:.2f} | "
            "{macro_f1_delta_pp:+.2f} | [{macro_f1_bootstrap95_low_pp:+.2f}, "
            "{macro_f1_bootstrap95_high_pp:+.2f}] | {macro_p} | "
            "{accuracy_candidate_mean_pct:.2f} | {accuracy_delta_pp:+.2f} | "
            "[{accuracy_bootstrap95_low_pp:+.2f}, {accuracy_bootstrap95_high_pp:+.2f}] | "
            "{accuracy_p} |".format(
                **row,
                macro_p=_format_p(row["macro_f1_p_holm"]),
                accuracy_p=_format_p(row["accuracy_p_holm"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Positive deltas favor the ablation; negative deltas favor full WavestarNet.",
            "- Bootstrap intervals quantify validation-sample uncertainty for the observed three-seed mean.",
            "- Seed-level t intervals are retained in the CSV and quantify training-run uncertainty, but are imprecise with only three seeds.",
            "- Statistical significance does not replace practical effect-size interpretation.",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    repo_root: Path,
    output_dir: Path,
    *,
    resamples: int = 10_000,
    random_seed: int = 20_260_905,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_path = (
        repo_root / "agrivision4/results/ablation_study/final_ablation_comparison.csv"
    )
    comparison = pd.read_csv(comparison_path)
    reference_row = comparison.loc[comparison["experiment_group"] == "reference"]
    if len(reference_row) != 1:
        raise ValueError("expected exactly one reference row")
    reference_row = reference_row.iloc[0]

    reference = load_prediction_bundle(
        repo_root / REFERENCE_ARCHIVE,
        member_prefix=REFERENCE_PREFIX,
    )
    num_classes = int(reference.true_labels.max()) + 1
    if len(reference.true_labels) != int(reference_row["validation_samples"]):
        raise ValueError("reference validation sample count does not match comparison table")
    if len(np.unique(reference.component_ids)) != len(reference.component_ids):
        raise ValueError("component-cluster bootstrap is required for non-singleton components")

    long_rows: list[dict[str, object]] = []
    input_hashes = dict(reference.source_hashes)
    for _, row in comparison.loc[comparison["experiment_group"] != "reference"].iterrows():
        source = repo_root / str(row["source"])
        candidate = load_prediction_bundle(source, reference_bundle=reference)
        input_hashes.update(candidate.source_hashes)
        result = paired_resampling_analysis(
            reference,
            candidate,
            num_classes=num_classes,
            resamples=resamples,
            random_seed=_stable_seed(random_seed, str(row["ablation_key"])),
        )
        for metric, values in result.items():
            expected_reference = float(reference_row[f"{metric}_mean"])
            expected_candidate = float(row[f"{metric}_mean"])
            if not math.isclose(values["reference_mean"], expected_reference, abs_tol=5e-9):
                raise ValueError(f"reference {metric} does not reproduce the summary table")
            if not math.isclose(values["candidate_mean"], expected_candidate, abs_tol=5e-9):
                raise ValueError(
                    f"{row['ablation_key']} {metric} does not reproduce the summary table"
                )
            long_rows.append(
                {
                    "table_order": int(row["table_order"]),
                    "ablation_key": row["ablation_key"],
                    "display_name": row["display_name"],
                    "experiment_group": row["experiment_group"],
                    "parameters": int(row["parameters"]),
                    "parameter_delta": int(row["parameter_delta"]),
                    "metric": metric,
                    "reference_mean_pct": 100.0 * values["reference_mean"],
                    "candidate_mean_pct": 100.0 * values["candidate_mean"],
                    "delta_pp": 100.0 * values["delta"],
                    "seed_42_delta_pp": 100.0 * values["seed_deltas"][0],
                    "seed_2024_delta_pp": 100.0 * values["seed_deltas"][1],
                    "seed_3407_delta_pp": 100.0 * values["seed_deltas"][2],
                    "paired_seed_delta_sd_pp": 100.0 * values["paired_seed_delta_sd"],
                    "paired_seed_t95_low_pp": 100.0 * values["paired_seed_t95_low"],
                    "paired_seed_t95_high_pp": 100.0 * values["paired_seed_t95_high"],
                    "bootstrap95_low_pp": 100.0 * values["bootstrap95_low"],
                    "bootstrap95_high_pp": 100.0 * values["bootstrap95_high"],
                    "paired_randomization_p_raw": values["paired_randomization_p_raw"],
                    "resamples": resamples,
                    "resampling_unit": "sample_id (one image per component_id)",
                }
            )

    long = pd.DataFrame(long_rows).sort_values(["table_order", "metric"])
    long["p_holm"] = np.nan
    for metric in METRICS:
        mask = long["metric"] == metric
        long.loc[mask, "p_holm"] = holm_adjust(
            long.loc[mask, "paired_randomization_p_raw"].tolist()
        )
    long["holm_reject_0_05"] = long["p_holm"] < 0.05

    identity_columns = [
        "table_order",
        "ablation_key",
        "display_name",
        "experiment_group",
        "parameters",
        "parameter_delta",
    ]
    wide = long.loc[:, identity_columns].drop_duplicates().set_index(identity_columns)
    for metric in METRICS:
        subset = long.loc[long["metric"] == metric].set_index(identity_columns)
        for source_column, suffix in (
            ("reference_mean_pct", "reference_mean_pct"),
            ("candidate_mean_pct", "candidate_mean_pct"),
            ("delta_pp", "delta_pp"),
            ("paired_seed_delta_sd_pp", "paired_seed_delta_sd_pp"),
            ("paired_seed_t95_low_pp", "paired_seed_t95_low_pp"),
            ("paired_seed_t95_high_pp", "paired_seed_t95_high_pp"),
            ("bootstrap95_low_pp", "bootstrap95_low_pp"),
            ("bootstrap95_high_pp", "bootstrap95_high_pp"),
            ("paired_randomization_p_raw", "p_raw"),
            ("p_holm", "p_holm"),
            ("holm_reject_0_05", "holm_reject_0_05"),
        ):
            wide[f"{metric}_{suffix}"] = subset[source_column]
    wide = wide.reset_index().sort_values("table_order")

    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / "ablation_statistical_comparison_long.csv"
    wide_path = output_dir / "ablation_statistical_comparison.csv"
    markdown_path = output_dir / "ablation_statistical_comparison.md"
    long.to_csv(long_path, index=False, float_format="%.9f")
    wide.to_csv(wide_path, index=False, float_format="%.9f")
    _write_markdown(wide, markdown_path, resamples=resamples)

    repo_prefix = str(repo_root.resolve()) + "/"
    portable_input_hashes = {
        name.removeprefix(repo_prefix): digest
        for name, digest in input_hashes.items()
    }

    manifest = {
        "analysis_id": "agrivision4_ablation_statistics_v1",
        "reference_key": str(reference_row["ablation_key"]),
        "seeds": list(SEEDS),
        "sample_count": len(reference.true_labels),
        "component_count": len(np.unique(reference.component_ids)),
        "class_count": num_classes,
        "bootstrap_method": "class-stratified paired nonparametric bootstrap",
        "randomization_method": "paired sample-wise label-swap randomization",
        "multiple_testing": "Holm correction separately within each metric family",
        "resamples": resamples,
        "random_seed": random_seed,
        "analysis_script_sha256": _sha256(Path(__file__).read_bytes()),
        "input_prediction_sha256": dict(sorted(portable_input_hashes.items())),
        "output_sha256": {
            path.name: _sha256(path.read_bytes())
            for path in (long_path, wide_path, markdown_path)
        },
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return long, wide


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("agrivision4/results/ablation_study/statistics"),
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20_260_905)
    args = parser.parse_args(list(argv) if argv is not None else None)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = args.repo_root / output_dir
    _, wide = run_analysis(
        args.repo_root.resolve(),
        output_dir.resolve(),
        resamples=args.resamples,
        random_seed=args.random_seed,
    )
    print(wide.to_string(index=False))
    print(f"\nWrote statistical analysis to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
