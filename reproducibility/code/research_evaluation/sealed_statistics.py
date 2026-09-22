"""Reproducible statistical analysis for sealed multi-seed classification tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest


DEFAULT_SEEDS = (42, 2024, 3407)
METRICS = ("accuracy", "macro_f1")
WAVESTARNET_KEY = "wavestarnet_big_transfer_v2"
PREDICTION_COLUMNS = ("sample_id", "component_id", "label_id", "prediction")


@dataclass(frozen=True)
class PredictionBundle:
    sample_ids: np.ndarray
    component_ids: np.ndarray
    true_labels: np.ndarray
    predictions: np.ndarray
    seeds: tuple[int, ...]
    source_hashes: dict[str, str]


def classification_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    *,
    num_classes: int,
) -> dict[str, float]:
    true = np.asarray(true_labels, dtype=np.int64)
    predicted = np.asarray(predicted_labels, dtype=np.int64)
    if true.ndim != 1 or true.shape != predicted.shape or true.size == 0:
        raise ValueError("true and predicted labels must be aligned non-empty vectors")
    if num_classes <= 1:
        raise ValueError("num_classes must exceed one")
    if true.min() < 0 or predicted.min() < 0:
        raise ValueError("labels must be non-negative")
    if true.max() >= num_classes or predicted.max() >= num_classes:
        raise ValueError("label exceeds num_classes")

    confusion = np.bincount(
        true * num_classes + predicted,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted_support = confusion.sum(axis=0).astype(np.float64)
    true_positive = np.diag(confusion).astype(np.float64)
    denominator = support + predicted_support
    per_class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros(num_classes, dtype=np.float64),
        where=denominator > 0,
    )
    return {
        "accuracy": float(true_positive.sum() / true.size),
        "macro_f1": float(per_class_f1.mean()),
    }


def align_prediction_frames(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for name, frame in (("reference", reference), ("candidate", candidate)):
        missing = set(PREDICTION_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"{name} predictions missing columns: {sorted(missing)}")
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"{name} predictions contain duplicate sample_id values")

    left = reference.loc[:, PREDICTION_COLUMNS].sort_values("sample_id").reset_index(drop=True)
    right = candidate.loc[:, PREDICTION_COLUMNS].sort_values("sample_id").reset_index(drop=True)
    if left["sample_id"].tolist() != right["sample_id"].tolist():
        raise ValueError("prediction sample sets do not match")
    if not np.array_equal(left["label_id"], right["label_id"]):
        raise ValueError("ground-truth labels do not match")
    if not np.array_equal(left["component_id"].astype(str), right["component_id"].astype(str)):
        raise ValueError("component identifiers do not match")
    return left, right


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prediction_bundle(
    result_root: Path,
    model_key: str,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    reference: PredictionBundle | None = None,
) -> PredictionBundle:
    canonical: pd.DataFrame | None = None
    predictions: list[np.ndarray] = []
    source_hashes: dict[str, str] = {}
    for seed in seeds:
        path = result_root / model_key / f"seed_{seed}" / "test_predictions.csv"
        frame = pd.read_csv(path)
        source_hashes[str(path)] = _sha256(path)
        if canonical is None:
            canonical = frame
        else:
            canonical, frame = align_prediction_frames(canonical, frame)
        predictions.append(
            frame.sort_values("sample_id")["prediction"].to_numpy(np.int64)
        )

    assert canonical is not None
    canonical = canonical.sort_values("sample_id").reset_index(drop=True)
    bundle = PredictionBundle(
        sample_ids=canonical["sample_id"].astype(str).to_numpy(),
        component_ids=canonical["component_id"].astype(str).to_numpy(),
        true_labels=canonical["label_id"].to_numpy(np.int64),
        predictions=np.stack(predictions),
        seeds=tuple(int(seed) for seed in seeds),
        source_hashes=source_hashes,
    )
    if len(np.unique(bundle.component_ids)) != len(bundle.component_ids):
        raise ValueError("non-singleton components require a component-cluster bootstrap")
    if reference is not None:
        if not np.array_equal(reference.sample_ids, bundle.sample_ids):
            raise ValueError("prediction sample identifiers do not align")
        if not np.array_equal(reference.true_labels, bundle.true_labels):
            raise ValueError("prediction ground-truth labels do not align")
        if not np.array_equal(reference.component_ids, bundle.component_ids):
            raise ValueError("prediction component identifiers do not align")
        if reference.seeds != bundle.seeds:
            raise ValueError("prediction seeds do not align")
    return bundle


def stratified_bootstrap_indices(
    labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    sampled = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        sampled.append(rng.choice(indices, size=len(indices), replace=True))
    result = np.concatenate(sampled)
    rng.shuffle(result)
    return result


def _mean_seed_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    rows = [
        classification_metrics(labels, prediction, num_classes=num_classes)
        for prediction in predictions
    ]
    return {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in METRICS
    }


def bootstrap_model_interval(
    bundle: PredictionBundle,
    *,
    num_classes: int,
    resamples: int,
    random_seed: int,
) -> dict[str, dict[str, float]]:
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    point = _mean_seed_metrics(bundle.true_labels, bundle.predictions, num_classes)
    samples = {metric: np.empty(resamples, dtype=np.float64) for metric in METRICS}
    rng = np.random.default_rng(random_seed)
    for iteration in range(resamples):
        indices = stratified_bootstrap_indices(bundle.true_labels, rng)
        metrics = _mean_seed_metrics(
            bundle.true_labels[indices],
            bundle.predictions[:, indices],
            num_classes,
        )
        for metric in METRICS:
            samples[metric][iteration] = metrics[metric]
    return {
        metric: {
            "mean": point[metric],
            "bootstrap95_low": float(np.quantile(samples[metric], 0.025)),
            "bootstrap95_high": float(np.quantile(samples[metric], 0.975)),
        }
        for metric in METRICS
    }


def paired_bootstrap_comparison(
    reference: PredictionBundle,
    candidate: PredictionBundle,
    *,
    num_classes: int,
    resamples: int,
    random_seed: int,
) -> dict[str, dict[str, float]]:
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if reference.seeds != candidate.seeds:
        raise ValueError("prediction seeds do not align")
    if not np.array_equal(reference.sample_ids, candidate.sample_ids):
        raise ValueError("prediction sample identifiers do not align")
    if not np.array_equal(reference.true_labels, candidate.true_labels):
        raise ValueError("prediction ground-truth labels do not align")

    labels = reference.true_labels
    reference_point = _mean_seed_metrics(labels, reference.predictions, num_classes)
    candidate_point = _mean_seed_metrics(labels, candidate.predictions, num_classes)
    deltas = {metric: np.empty(resamples, dtype=np.float64) for metric in METRICS}
    rng = np.random.default_rng(random_seed)
    for iteration in range(resamples):
        indices = stratified_bootstrap_indices(labels, rng)
        ref = _mean_seed_metrics(labels[indices], reference.predictions[:, indices], num_classes)
        cand = _mean_seed_metrics(labels[indices], candidate.predictions[:, indices], num_classes)
        for metric in METRICS:
            deltas[metric][iteration] = cand[metric] - ref[metric]

    output: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = deltas[metric]
        lower_tail = (1 + np.count_nonzero(values <= 0.0)) / (resamples + 1)
        upper_tail = (1 + np.count_nonzero(values >= 0.0)) / (resamples + 1)
        output[metric] = {
            "reference_mean": reference_point[metric],
            "candidate_mean": candidate_point[metric],
            "delta": candidate_point[metric] - reference_point[metric],
            "bootstrap95_low": float(np.quantile(values, 0.025)),
            "bootstrap95_high": float(np.quantile(values, 0.975)),
            "paired_bootstrap_p_two_sided": float(min(1.0, 2.0 * min(lower_tail, upper_tail))),
        }
    return output


def exact_mcnemar(
    true_labels: np.ndarray,
    reference_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
) -> dict[str, int | float]:
    true = np.asarray(true_labels)
    reference_correct = np.asarray(reference_predictions) == true
    candidate_correct = np.asarray(candidate_predictions) == true
    if reference_correct.shape != candidate_correct.shape or true.ndim != 1:
        raise ValueError("McNemar inputs must be aligned one-dimensional arrays")
    candidate_only = int(np.count_nonzero(candidate_correct & ~reference_correct))
    reference_only = int(np.count_nonzero(reference_correct & ~candidate_correct))
    discordant = candidate_only + reference_only
    p_value = 1.0
    if discordant:
        p_value = float(
            binomtest(
                min(candidate_only, reference_only),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    return {
        "candidate_only_correct": candidate_only,
        "reference_only_correct": reference_only,
        "discordant": discordant,
        "p_exact_two_sided": p_value,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted_sorted[rank] = running
    adjusted = np.empty(len(values), dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def choose_strongest_baseline(aggregate: pd.DataFrame) -> str:
    eligible = aggregate.loc[~aggregate["model_key"].astype(str).str.startswith("wavestarnet")]
    if eligible.empty:
        raise ValueError("aggregate table contains no eligible baseline")
    return str(eligible.sort_values("macro_f1_mean", ascending=False).iloc[0]["model_key"])


def _stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**32)


def _write_markdown(
    intervals: pd.DataFrame,
    comparisons: pd.DataFrame,
    mcnemar: pd.DataFrame,
    destination: Path,
    *,
    resamples: int,
) -> None:
    lines = [
        "# Final Sealed-Test Statistical Analysis",
        "",
        (
            f"Performance intervals and paired differences use {resamples:,} "
            "class-stratified bootstrap resamples. Metrics are averaged over the three "
            "matched seeds before each bootstrap statistic is recorded."
        ),
        "",
        "## Model Confidence Intervals",
        "",
        "| Dataset | Model | Accuracy % (95% CI) | Macro-F1 % (95% CI) |",
        "|---|---|---:|---:|",
    ]
    for (dataset, model), group in intervals.groupby(["dataset", "display_name"], sort=False):
        values = group.set_index("metric")
        acc = values.loc["accuracy"]
        f1 = values.loc["macro_f1"]
        lines.append(
            f"| {dataset} | {model} | {acc['mean_pct']:.2f} "
            f"[{acc['bootstrap95_low_pct']:.2f}, {acc['bootstrap95_high_pct']:.2f}] | "
            f"{f1['mean_pct']:.2f} [{f1['bootstrap95_low_pct']:.2f}, "
            f"{f1['bootstrap95_high_pct']:.2f}] |"
        )
    lines.extend(
        [
            "",
            "## WavestarNet Versus Strongest Baseline",
            "",
            "| Dataset | Baseline | Metric | WavestarNet % | Baseline % | Delta pp (95% CI) | Paired-bootstrap p |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in comparisons.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['reference_display_name']} | {row['metric']} | "
            f"{row['candidate_mean_pct']:.2f} | {row['reference_mean_pct']:.2f} | "
            f"{row['delta_pp']:+.2f} [{row['bootstrap95_low_pp']:+.2f}, "
            f"{row['bootstrap95_high_pp']:+.2f}] | {row['paired_bootstrap_p_two_sided']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Exact McNemar Tests",
            "",
            "| Dataset | Seed | WavestarNet only correct | Baseline only correct | Exact p | Holm p (all tests) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in mcnemar.iterrows():
        lines.append(
            f"| {row['dataset']} | {int(row['seed'])} | "
            f"{int(row['candidate_only_correct'])} | {int(row['reference_only_correct'])} | "
            f"{row['p_exact_two_sided']:.4f} | {row['p_holm_all_tests']:.4f} |"
        )
    lines.extend(
        [
            "",
            "McNemar tests are seed-specific because pooling three predictions of the same "
            "image would violate the independent-pairs assumption. Positive deltas favor "
            "WavestarNet. Confidence intervals describe held-out sample uncertainty; they "
            "are not three-seed t intervals.",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    datasets: Sequence[tuple[str, Path]],
    output_dir: Path,
    *,
    resamples: int = 10_000,
    random_seed: int = 20_260_909,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interval_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    mcnemar_rows: list[dict[str, object]] = []
    input_hashes: dict[str, str] = {}

    for dataset_name, result_root in datasets:
        aggregate_path = result_root / "sealed_test_aggregate.csv"
        aggregate = pd.read_csv(aggregate_path)
        input_hashes[str(aggregate_path)] = _sha256(aggregate_path)
        baseline_key = choose_strongest_baseline(aggregate)
        display_names = dict(zip(aggregate["model_key"], aggregate["display_name"]))
        bundles: dict[str, PredictionBundle] = {}
        canonical: PredictionBundle | None = None
        for model_key in aggregate["model_key"].astype(str):
            bundle = load_prediction_bundle(
                result_root,
                model_key,
                reference=canonical,
            )
            if canonical is None:
                canonical = bundle
            bundles[model_key] = bundle
            input_hashes.update(bundle.source_hashes)

        assert canonical is not None
        num_classes = int(canonical.true_labels.max()) + 1
        for model_key, bundle in bundles.items():
            intervals = bootstrap_model_interval(
                bundle,
                num_classes=num_classes,
                resamples=resamples,
                random_seed=_stable_seed(random_seed, f"{dataset_name}:{model_key}:interval"),
            )
            summary_row = aggregate.loc[aggregate["model_key"] == model_key].iloc[0]
            for metric, values in intervals.items():
                expected = float(summary_row[f"{metric}_mean"])
                if not np.isclose(values["mean"], expected, atol=5e-9):
                    raise ValueError(
                        f"{dataset_name}/{model_key} {metric} does not reproduce aggregate"
                    )
                interval_rows.append(
                    {
                        "dataset": dataset_name,
                        "model_key": model_key,
                        "display_name": display_names[model_key],
                        "samples": len(bundle.true_labels),
                        "seed_count": len(bundle.seeds),
                        "metric": metric,
                        "mean_pct": 100.0 * values["mean"],
                        "bootstrap95_low_pct": 100.0 * values["bootstrap95_low"],
                        "bootstrap95_high_pct": 100.0 * values["bootstrap95_high"],
                        "resamples": resamples,
                    }
                )

        comparison = paired_bootstrap_comparison(
            bundles[baseline_key],
            bundles[WAVESTARNET_KEY],
            num_classes=num_classes,
            resamples=resamples,
            random_seed=_stable_seed(random_seed, f"{dataset_name}:paired"),
        )
        for metric, values in comparison.items():
            comparison_rows.append(
                {
                    "dataset": dataset_name,
                    "candidate_model_key": WAVESTARNET_KEY,
                    "candidate_display_name": display_names[WAVESTARNET_KEY],
                    "reference_model_key": baseline_key,
                    "reference_display_name": display_names[baseline_key],
                    "metric": metric,
                    "candidate_mean_pct": 100.0 * values["candidate_mean"],
                    "reference_mean_pct": 100.0 * values["reference_mean"],
                    "delta_pp": 100.0 * values["delta"],
                    "bootstrap95_low_pp": 100.0 * values["bootstrap95_low"],
                    "bootstrap95_high_pp": 100.0 * values["bootstrap95_high"],
                    "paired_bootstrap_p_two_sided": values["paired_bootstrap_p_two_sided"],
                    "resamples": resamples,
                }
            )
        reference = bundles[baseline_key]
        candidate = bundles[WAVESTARNET_KEY]
        for index, seed in enumerate(candidate.seeds):
            result = exact_mcnemar(
                candidate.true_labels,
                reference.predictions[index],
                candidate.predictions[index],
            )
            mcnemar_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "candidate_model_key": WAVESTARNET_KEY,
                    "reference_model_key": baseline_key,
                    **result,
                }
            )

    intervals = pd.DataFrame(interval_rows)
    comparisons = pd.DataFrame(comparison_rows)
    mcnemar = pd.DataFrame(mcnemar_rows)
    mcnemar["p_holm_all_tests"] = holm_adjust(mcnemar["p_exact_two_sided"].tolist())
    mcnemar["reject_0_05_after_holm"] = mcnemar["p_holm_all_tests"] < 0.05

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "intervals": output_dir / "sealed_test_bootstrap_confidence_intervals.csv",
        "comparisons": output_dir / "wavestarnet_vs_best_baseline_paired_bootstrap.csv",
        "mcnemar": output_dir / "wavestarnet_vs_best_baseline_mcnemar.csv",
        "report": output_dir / "sealed_test_statistical_analysis.md",
    }
    intervals.to_csv(paths["intervals"], index=False, float_format="%.9f")
    comparisons.to_csv(paths["comparisons"], index=False, float_format="%.9f")
    mcnemar.to_csv(paths["mcnemar"], index=False, float_format="%.9f")
    _write_markdown(intervals, comparisons, mcnemar, paths["report"], resamples=resamples)

    manifest = {
        "analysis_id": "final_sealed_test_statistics_v1",
        "datasets": [name for name, _ in datasets],
        "metrics": list(METRICS),
        "seeds": list(DEFAULT_SEEDS),
        "resamples": resamples,
        "random_seed": random_seed,
        "bootstrap_method": "class-stratified paired nonparametric bootstrap",
        "mcnemar_method": "seed-specific exact two-sided binomial McNemar test",
        "mcnemar_multiple_testing": "Holm correction across all dataset-seed tests",
        "analysis_script_sha256": _sha256(Path(__file__)),
        "input_sha256": dict(sorted(input_hashes.items())),
        "output_sha256": {path.name: _sha256(path) for path in paths.values()},
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return intervals, comparisons, mcnemar


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("summary/final_sealed_test_statistics"))
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20_260_909)
    args = parser.parse_args(list(argv) if argv is not None else None)
    output = args.output_dir if args.output_dir.is_absolute() else args.repo_root / args.output_dir
    dataset_roots = [
        (
            "AgriVision4",
            args.repo_root
            / "agrivision4/results/sealed_test/agrivision4_scratch_research_v2_final_holdout_v1",
        ),
        (
            "PlantDoc",
            args.repo_root / "plantdoc/results/sealed_test/plantdoc_scratch_final_holdout_v1",
        ),
        (
            "PlantWild",
            args.repo_root / "plantwild/results/sealed_test/plantwild_scratch_final_holdout_v1",
        ),
    ]
    run_analysis(
        dataset_roots,
        output,
        resamples=args.resamples,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
