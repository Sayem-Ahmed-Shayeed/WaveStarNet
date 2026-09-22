"""Summarize completed AgriVision4 Grad-CAM and prediction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


WAVESTARNET_KEY = "wavestarnet_big_transfer_v2"
MOBILEVIT_KEY = "mobilevit_s"


def _agreement_categories(
    wave_correct: pd.Series,
    mobile_correct: pd.Series,
) -> pd.Series:
    values = np.select(
        [
            wave_correct & mobile_correct,
            wave_correct & ~mobile_correct,
            ~wave_correct & mobile_correct,
            ~wave_correct & ~mobile_correct,
        ],
        [
            "both_correct",
            "wavestarnet_only_correct",
            "mobilevit_only_correct",
            "both_wrong",
        ],
        default="unclassified",
    )
    categories = pd.Series(values, index=wave_correct.index, name="category")
    if categories.eq("unclassified").any():
        raise RuntimeError("some model-agreement rows could not be classified")
    return categories


def _spatial_metrics(path: Path) -> dict[str, float]:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    height, width = array.shape
    total = float(array.sum())
    flat = np.sort(array.ravel())
    top_count = max(1, int(np.ceil(flat.size * 0.10)))
    border = max(1, int(round(min(height, width) * 0.15)))
    border_mask = np.zeros_like(array, dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True
    border_area_fraction = float(border_mask.mean())

    values = {
        "mean_activation": float(array.mean()),
        "nonzero_fraction": float((array > 0).mean()),
        "top10_mass_fraction": float("nan"),
        "border_mass_fraction": float("nan"),
        "border_excess_ratio": float("nan"),
        "center_distance": float("nan"),
        "spatial_entropy": float("nan"),
    }
    if total <= 0.0:
        return values

    yy, xx = np.indices(array.shape)
    center_y = float((array * yy).sum() / total)
    center_x = float((array * xx).sum() / total)
    distribution = array.ravel() / total
    distribution = distribution[distribution > 0.0]
    border_mass_fraction = float(array[border_mask].sum() / total)
    values.update(
        {
            "top10_mass_fraction": float(flat[-top_count:].sum() / total),
            "border_mass_fraction": border_mass_fraction,
            "border_excess_ratio": border_mass_fraction / border_area_fraction,
            "center_distance": float(
                np.sqrt(
                    ((center_y - (height - 1) / 2.0) / height) ** 2
                    + ((center_x - (width - 1) / 2.0) / width) ** 2
                )
            ),
            "spatial_entropy": float(
                -(distribution * np.log(distribution)).sum() / np.log(array.size)
            ),
        }
    )
    return values


def summarize(result_root: Path) -> dict[str, Path]:
    root = Path(result_root)
    predictions = pd.read_csv(root / "predictions.csv")
    per_class = pd.read_csv(root / "per_class_metrics.csv")

    pivot = predictions.pivot(
        index="sample_id",
        columns="model_key",
        values=["correct", "predicted_label_id", "confidence"],
    )
    wave_correct = pivot[("correct", WAVESTARNET_KEY)].astype(bool)
    mobile_correct = pivot[("correct", MOBILEVIT_KEY)].astype(bool)
    categories = _agreement_categories(wave_correct, mobile_correct)
    agreement = categories.value_counts().rename_axis("category").reset_index(name="count")
    agreement["fraction"] = agreement["count"] / len(categories)
    agreement_path = root / "model_agreement_summary.csv"
    agreement.to_csv(agreement_path, index=False)

    confidence = (
        predictions.groupby(["model_key", "correct"], sort=True)
        .agg(
            count=("sample_id", "size"),
            mean_confidence=("confidence", "mean"),
            median_confidence=("confidence", "median"),
            mean_entropy=("entropy", "mean"),
        )
        .reset_index()
    )
    confidence_path = root / "confidence_error_summary.csv"
    confidence.to_csv(confidence_path, index=False)

    per_class_wide = per_class.pivot(
        index=["label_id", "label", "support"],
        columns="model_key",
        values=["precision", "recall", "f1"],
    ).reset_index()
    per_class_wide.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in per_class_wide.columns
    ]
    per_class_wide["f1_delta_wavestarnet_minus_mobilevit"] = (
        per_class_wide[f"f1_{WAVESTARNET_KEY}"]
        - per_class_wide[f"f1_{MOBILEVIT_KEY}"]
    )
    per_class_wide["recall_delta_wavestarnet_minus_mobilevit"] = (
        per_class_wide[f"recall_{WAVESTARNET_KEY}"]
        - per_class_wide[f"recall_{MOBILEVIT_KEY}"]
    )
    per_class_wide = per_class_wide.sort_values(
        "f1_delta_wavestarnet_minus_mobilevit", ascending=False, kind="stable"
    )
    per_class_path = root / "per_class_model_comparison.csv"
    per_class_wide.to_csv(per_class_path, index=False)

    spatial_rows: list[dict[str, object]] = []
    for prediction in predictions.itertuples(index=False):
        for role in ("predicted", "ground_truth"):
            spatial_rows.append(
                {
                    "sample_id": str(prediction.sample_id),
                    "model_key": str(prediction.model_key),
                    "role": role,
                    "correct": bool(prediction.correct),
                    "label": str(prediction.label),
                    **_spatial_metrics(root / str(getattr(prediction, f"{role}_heatmap"))),
                }
            )
    spatial = pd.DataFrame(spatial_rows)
    spatial_path = root / "gradcam_spatial_metrics.csv"
    spatial.to_csv(spatial_path, index=False)
    spatial_summary = (
        spatial.groupby(["model_key", "role", "correct"], sort=True)
        .agg(
            count=("sample_id", "size"),
            mean_activation=("mean_activation", "mean"),
            mean_nonzero_fraction=("nonzero_fraction", "mean"),
            mean_top10_mass_fraction=("top10_mass_fraction", "mean"),
            mean_border_excess_ratio=("border_excess_ratio", "mean"),
            mean_center_distance=("center_distance", "mean"),
            mean_spatial_entropy=("spatial_entropy", "mean"),
        )
        .reset_index()
    )
    spatial_summary_path = root / "gradcam_spatial_summary.csv"
    spatial_summary.to_csv(spatial_summary_path, index=False)

    confusion_rows: list[pd.DataFrame] = []
    for model_key, frame in predictions.groupby("model_key", sort=True):
        errors = frame.loc[~frame["correct"].astype(bool)]
        grouped = (
            errors.groupby(["label", "predicted_label"], sort=True)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "label", "predicted_label"], ascending=[False, True, True])
        )
        grouped.insert(0, "model_key", model_key)
        confusion_rows.append(grouped)
    confusions = pd.concat(confusion_rows, ignore_index=True)
    confusions_path = root / "top_error_pairs.csv"
    confusions.to_csv(confusions_path, index=False)

    outputs = {
        "agreement": agreement_path,
        "confidence": confidence_path,
        "per_class": per_class_path,
        "spatial": spatial_path,
        "spatial_summary": spatial_summary_path,
        "confusions": confusions_path,
    }
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    print("\nAgreement\n", agreement.to_string(index=False))
    print("\nConfidence\n", confidence.to_string(index=False))
    print("\nSpatial summary\n", spatial_summary.to_string(index=False))
    print("\nLargest WavestarNet F1 advantages\n", per_class_wide.head(8).to_string(index=False))
    print("\nLargest WavestarNet F1 deficits\n", per_class_wide.tail(8).to_string(index=False))
    print("\nTop error pairs\n", confusions.groupby("model_key", sort=True).head(8).to_string(index=False))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result_root",
        nargs="?",
        type=Path,
        default=Path("agrivision4/results/interpretability/gradcam_validation_v1"),
    )
    args = parser.parse_args()
    summarize(args.result_root)


if __name__ == "__main__":
    main()
