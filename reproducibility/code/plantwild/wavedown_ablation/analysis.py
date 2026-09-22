"""Fail-closed paired validation analysis for the PlantWild WaveDown ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


SEEDS = (42, 2024, 3407)
METRICS = ("accuracy", "macro_f1", "weighted_f1", "nll")
REQUIRED_COLUMNS = ("sample_id", "component_id", "true_label_id", "predicted_label_id")


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite values in [0, 1]")
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        result[index] = running
    return result.tolist()


def align_seed_predictions(reference: pd.DataFrame, candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for name, frame in (("reference", reference), ("candidate", candidate)):
        missing = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing or frame["sample_id"].duplicated().any():
            raise ValueError(f"{name} predictions are not uniquely identifiable")
    left = reference.sort_values("sample_id").reset_index(drop=True)
    right = candidate.sort_values("sample_id").reset_index(drop=True)
    if left["sample_id"].tolist() != right["sample_id"].tolist():
        raise ValueError("validation sample IDs do not match")
    if not np.array_equal(left["component_id"], right["component_id"]):
        raise ValueError("validation component IDs do not match")
    if not np.array_equal(left["true_label_id"], right["true_label_id"]):
        raise ValueError("validation ground-truth labels do not match")
    return left, right


def _metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray, classes: int) -> dict[str, float]:
    matrix = np.bincount(y * classes + pred, minlength=classes * classes).reshape(classes, classes)
    support = matrix.sum(axis=1).astype(float)
    predicted = matrix.sum(axis=0).astype(float)
    tp = np.diag(matrix).astype(float)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(classes), where=(support + predicted) > 0)
    return {
        "accuracy": float(tp.sum() / len(y)),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, support) / len(y)),
        "nll": float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1)).mean()),
    }


def _probabilities(frame: pd.DataFrame, classes: int) -> np.ndarray:
    columns = [f"prob_{index:02d}" for index in range(classes)]
    if any(column not in frame for column in columns):
        raise ValueError("validation probabilities are incomplete")
    values = frame[columns].to_numpy(float)
    if not np.isfinite(values).all() or not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("validation probabilities are invalid")
    return values


def _read(root: Path, seed: int) -> pd.DataFrame:
    path = root / f"seed_{seed}" / "validation_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def paired_analysis(reference_root: Path, candidate_root: Path, *, resamples: int = 10_000, seed: int = 20260914) -> pd.DataFrame:
    rows = []
    raw_p = []
    rng = np.random.default_rng(seed)
    bundles = []
    for run_seed in SEEDS:
        reference, candidate = align_seed_predictions(_read(reference_root, run_seed), _read(candidate_root, run_seed))
        classes = int(max(reference.true_label_id.max(), candidate.true_label_id.max()) + 1)
        y = reference.true_label_id.to_numpy(int)
        reference_prediction = reference.predicted_label_id.to_numpy(int)
        candidate_prediction = candidate.predicted_label_id.to_numpy(int)
        reference_probability = _probabilities(reference, classes)
        candidate_probability = _probabilities(candidate, classes)
        bundles.append((y, reference_prediction, candidate_prediction, reference_probability, candidate_probability, classes))
        reference_metrics = _metrics(y, reference_prediction, reference_probability, classes)
        candidate_metrics = _metrics(y, candidate_prediction, candidate_probability, classes)
        for metric in METRICS:
            rows.append({"seed": run_seed, "metric": metric, "reference": reference_metrics[metric], "candidate": candidate_metrics[metric], "delta": candidate_metrics[metric] - reference_metrics[metric]})
    frame = pd.DataFrame(rows)
    summary = []
    for metric in METRICS:
        values = frame.loc[frame.metric == metric, "delta"].to_numpy(float)
        summary.append({"metric": metric, "reference_mean": float(frame.loc[frame.metric == metric, "reference"].mean()), "candidate_mean": float(frame.loc[frame.metric == metric, "candidate"].mean()), "delta": float(values.mean()), "sample_sd": float(values.std(ddof=1)), "delta_pp": float(values.mean() * 100) if metric != "nll" else np.nan})
    summary_frame = pd.DataFrame(summary)
    # Sample-level paired label-swap test and stratified paired bootstrap are completed per seed then averaged.
    for index, row in summary_frame.iterrows():
        metric = row.metric
        bootstrap, null = [], []
        for _ in range(resamples):
            seed_values, null_values = [], []
            for y, reference_prediction, candidate_prediction, rp, cp, classes in bundles:
                groups = [np.flatnonzero(y == label) for label in np.unique(y)]
                indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
                rm = _metrics(y[indices], reference_prediction[indices], rp[indices], classes)[metric]
                cm = _metrics(y[indices], candidate_prediction[indices], cp[indices], classes)[metric]
                seed_values.append(cm - rm)
                swap = rng.random(len(y)) < 0.5
                rpred = np.where(swap, candidate_prediction, reference_prediction)
                cpred = np.where(swap, reference_prediction, candidate_prediction)
                rprob, cprob = np.where(swap[:, None], cp, rp), np.where(swap[:, None], rp, cp)
                null_values.append(_metrics(y, cpred, cprob, classes)[metric] - _metrics(y, rpred, rprob, classes)[metric])
            bootstrap.append(float(np.mean(seed_values)))
            null.append(float(np.mean(null_values)))
        observed = float(row.delta)
        summary_frame.loc[index, "bootstrap95_low"] = float(np.quantile(bootstrap, 0.025))
        summary_frame.loc[index, "bootstrap95_high"] = float(np.quantile(bootstrap, 0.975))
        raw_p.append((1 + int(np.count_nonzero(np.abs(null) >= abs(observed)))) / (resamples + 1))
    summary_frame["paired_randomization_p_raw"] = raw_p
    summary_frame["paired_randomization_p_holm"] = holm_adjust(raw_p)
    return summary_frame


def write_analysis(reference_root: Path, candidate_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    result = paired_analysis(reference_root, candidate_root)
    result.to_csv(output_root / "paired_validation_analysis.csv", index=False)
    lines = ["# PlantWild WavestarNet WaveDown Ablation", "", "Validation-only paired ablation; this is not a PlantWild test-set result.", "", result.to_markdown(index=False)]
    (output_root / "paired_validation_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.make_archive(str(output_root / "plantwild_wavestarnet_wavedown_ablation"), "zip", output_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    write_analysis(args.reference_root, args.candidate_root, args.output_root)
