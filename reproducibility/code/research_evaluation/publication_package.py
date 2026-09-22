"""Build and freeze the publication-ready WavestarNet result package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


MODEL_ORDER = (
    "wavestarnet_big_transfer_v2",
    "mobilevit_s",
    "mobilenetv3_large_100",
    "efficientnet_b0",
    "efficientnet_b3",
)
DISPLAY_NAMES = {
    "wavestarnet_big_transfer_v2": "WavestarNet",
    "mobilevit_s": "MobileViT-S",
    "mobilenetv3_large_100": "MobileNetV3-Large",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b3": "EfficientNet-B3",
}
DATASET_ORDER = ("AgriVision4", "PlantDoc", "PlantWild")
CB_BLUE = "#0077BB"
CB_ORANGE = "#EE7733"
CB_TEAL = "#009988"
CB_GREY = "#777777"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_calibration_error(
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(confidence)
    error = 0.0
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (confidence >= low) & (
            confidence <= high if index == bins - 1 else confidence < high
        )
        if selected.any():
            error += (selected.sum() / total) * abs(
                correct[selected].mean() - confidence[selected].mean()
            )
    return float(error)


def merge_sealed_results(
    dataset: str,
    aggregate: pd.DataFrame,
    intervals: pd.DataFrame,
) -> pd.DataFrame:
    subset = intervals.loc[intervals["dataset"] == dataset]
    low = subset.pivot(index="model_key", columns="metric", values="bootstrap95_low_pct")
    high = subset.pivot(index="model_key", columns="metric", values="bootstrap95_high_pct")
    output = aggregate.copy()
    output["dataset"] = dataset
    if "rank_by_macro_f1" not in output:
        output["rank_by_macro_f1"] = (
            output["macro_f1_mean"].rank(method="first", ascending=False).astype(int)
        )
    if "seeds" not in output:
        output["seeds"] = "not_reported"
    output["publication_name"] = output["model_key"].map(DISPLAY_NAMES).fillna(output["display_name"])
    for metric in ("accuracy", "macro_f1"):
        output[f"{metric}_mean_pct"] = 100.0 * output[f"{metric}_mean"]
        output[f"{metric}_sd_pct"] = 100.0 * output[f"{metric}_std"]
        output[f"{metric}_bootstrap95_low_pct"] = output["model_key"].map(low[metric])
        output[f"{metric}_bootstrap95_high_pct"] = output["model_key"].map(high[metric])
    output["weighted_f1_mean_pct"] = 100.0 * output["weighted_f1_mean"]
    output["weighted_f1_sd_pct"] = 100.0 * output["weighted_f1_std"]
    output["training_initialization"] = "random"
    output["evaluation_split"] = "sealed_test"
    columns = [
        "dataset",
        "rank_by_macro_f1",
        "model_key",
        "publication_name",
        "parameters",
        "seeds",
        "seed_count",
        "accuracy_mean_pct",
        "accuracy_sd_pct",
        "accuracy_bootstrap95_low_pct",
        "accuracy_bootstrap95_high_pct",
        "macro_f1_mean_pct",
        "macro_f1_sd_pct",
        "macro_f1_bootstrap95_low_pct",
        "macro_f1_bootstrap95_high_pct",
        "weighted_f1_mean_pct",
        "weighted_f1_sd_pct",
        "training_initialization",
        "evaluation_split",
    ]
    return output.loc[:, columns].sort_values("rank_by_macro_f1").reset_index(drop=True)


def _matched_row(
    efficiency: pd.DataFrame,
    model_key: str,
    *,
    device: str,
    precision: str,
    batch_size: int,
) -> pd.Series:
    matched = efficiency.loc[
        efficiency["model_key"].eq(model_key)
        & efficiency["device"].eq(device)
        & efficiency["precision"].eq(precision)
        & efficiency["batch_size"].eq(batch_size)
    ]
    if len(matched) != 1:
        raise ValueError(
            f"expected one efficiency row for {model_key}/{device}/{precision}/bs{batch_size}"
        )
    return matched.iloc[0]


def select_efficiency_rows(
    efficiency: pd.DataFrame,
    model_keys: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for model_key in model_keys:
        cpu = _matched_row(efficiency, model_key, device="cpu", precision="fp32", batch_size=1)
        gpu = _matched_row(efficiency, model_key, device="cuda", precision="fp32", batch_size=1)
        throughput = _matched_row(
            efficiency, model_key, device="cuda", precision="fp16", batch_size=32
        )
        rows.append(
            {
                "model_key": model_key,
                "cpu_fp32_bs1_latency_ms": cpu["latency_mean_ms"],
                "t4_fp32_bs1_latency_ms": gpu["latency_mean_ms"],
                "t4_fp32_bs1_latency_p95_ms": gpu.get("latency_p95_ms", np.nan),
                "t4_fp16_bs32_throughput_images_s": throughput["throughput_images_s"],
                "t4_fp16_bs32_peak_allocated_mb": throughput["peak_gpu_allocated_mb"],
                "t4_fp16_bs32_peak_reserved_mb": throughput.get("peak_gpu_reserved_mb", np.nan),
            }
        )
    return pd.DataFrame(rows)


def build_freeze_rows(
    repo_root: Path,
    sources: Sequence[tuple[str, Path]],
) -> list[dict[str, object]]:
    root = repo_root.resolve()
    selected: dict[Path, str] = {}
    for category, source in sources:
        path = source if source.is_absolute() else root / source
        if not path.exists():
            raise FileNotFoundError(path)
        files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
        for item in files:
            selected.setdefault(item.resolve(), category)
    rows = []
    for path in sorted(selected, key=lambda item: str(item.relative_to(root))):
        rows.append(
            {
                "category": selected[path],
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _save_figure(fig: plt.Figure, destination: Path) -> None:
    fig.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _configure_matplotlib() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )


def _performance_figure(main: pd.DataFrame, destination: Path) -> None:
    # Each dataset is a panel; the two marker rows on every model make the
    # Accuracy/Macro-F1 comparison explicit without repeating long axis labels.
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6), sharey=True)
    metric_colors = {"accuracy": CB_BLUE, "macro_f1": CB_ORANGE}
    offsets = {"accuracy": 0.16, "macro_f1": -0.16}
    all_low = main[["accuracy_bootstrap95_low_pct", "macro_f1_bootstrap95_low_pct"]].to_numpy()
    all_high = main[["accuracy_bootstrap95_high_pct", "macro_f1_bootstrap95_high_pct"]].to_numpy()
    xmin = max(0.0, float(np.floor(np.nanmin(all_low) / 5.0) * 5.0 - 2.0))
    xmax = float(np.ceil(np.nanmax(all_high) / 5.0) * 5.0 + 2.0)

    for column, dataset in enumerate(DATASET_ORDER):
        ax = axes[column]
        data = main.loc[main["dataset"] == dataset].copy()
        data["order"] = data["model_key"].map({key: i for i, key in enumerate(MODEL_ORDER)})
        data = data.sort_values("order")
        y = np.arange(len(data), dtype=float)
        ax.axhspan(-0.48, 0.48, color=CB_BLUE, alpha=0.07, zorder=0)
        for index, (_, record) in enumerate(data.iterrows()):
            accuracy = float(record["accuracy_mean_pct"])
            macro_f1 = float(record["macro_f1_mean_pct"])
            ax.plot([accuracy, macro_f1], [y[index] + offsets["accuracy"], y[index] + offsets["macro_f1"]],
                    color="#C6CDD0", lw=0.8, zorder=1)
            for metric in ("accuracy", "macro_f1"):
                mean = float(record[f"{metric}_mean_pct"])
                low = float(record[f"{metric}_bootstrap95_low_pct"])
                high = float(record[f"{metric}_bootstrap95_high_pct"])
                ax.errorbar(
                    mean,
                    y[index] + offsets[metric],
                    xerr=[[mean - low], [high - mean]],
                    fmt="none",
                    ecolor=metric_colors[metric],
                    capsize=2.5,
                    elinewidth=1.1,
                    zorder=2,
                )
                marker = "o" if metric == "accuracy" else "D"
                ax.scatter(mean, y[index] + offsets[metric], marker=marker,
                           color=metric_colors[metric], s=24, zorder=3)
        ax.set_title(dataset)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(len(data) - 0.45, -0.55)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if column != 0:
            ax.tick_params(axis="y", labelleft=False)

    axes[0].set_yticks(np.arange(len(MODEL_ORDER)))
    axes[0].set_yticklabels([DISPLAY_NAMES[key] for key in MODEL_ORDER])
    for label in axes[0].get_yticklabels():
        if label.get_text() == "WavestarNet":
            label.set_fontweight("bold")
    axes[0].set_ylabel("Models")
    fig.supxlabel("Mean score (%)  •  whiskers show class-stratified bootstrap 95% CIs", y=0.01)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=CB_BLUE,
                   markeredgecolor=CB_BLUE, markersize=5, label="Accuracy"),
        plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=CB_ORANGE,
                   markeredgecolor=CB_ORANGE, markersize=5, label="Macro-F1"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, frameon=False, handletextpad=0.4, columnspacing=1.2)
    fig.subplots_adjust(left=0.19, right=0.99, top=0.84, bottom=0.20, wspace=0.14)
    _save_figure(fig, destination)


def _efficiency_figure(main: pd.DataFrame, efficiency: pd.DataFrame, destination: Path) -> None:
    performance = main.loc[main["dataset"] == "PlantWild", ["model_key", "macro_f1_mean_pct"]]
    data = efficiency.merge(performance, on="model_key", how="inner")
    data["model_label"] = data["model_key"].map(DISPLAY_NAMES)
    data["order"] = data["model_key"].map({key: i for i, key in enumerate(MODEL_ORDER)})
    data = data.sort_values("order")
    y = np.arange(len(data))
    colors = [CB_BLUE if key == MODEL_ORDER[0] else CB_GREY for key in data["model_key"]]
    panels = (
        ("macro_f1_mean_pct", "PlantWild Macro-F1 (%)", "{:.1f}"),
        ("parameters_m", "Parameters (M)", "{:.2f}"),
        ("t4_fp32_bs1_latency_ms", "T4 latency (ms)", "{:.1f}"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.7), sharey=True)
    for ax, (column, xlabel, number_format) in zip(axes, panels):
        values = data[column].to_numpy(dtype=float)
        ax.barh(y, values, color=colors, alpha=0.9, height=0.58)
        pad = max(0.02 * (values.max() - values.min()), 0.05)
        for index, value in enumerate(values):
            ax.text(value + pad, y[index], number_format.format(value), va="center", fontsize=7)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.invert_yaxis()
    axes[0].set_yticks(y, data["model_label"])
    axes[0].set_ylabel("Model")
    axes[0].get_yticklabels()[0].set_fontweight("bold")
    fig.subplots_adjust(left=0.20, right=0.99, top=0.96, bottom=0.20, wspace=0.28)
    _save_figure(fig, destination)


def _ablation_figure(ablation: pd.DataFrame, destination: Path) -> None:
    data = ablation.loc[ablation["experiment_group"] != "reference"].copy()
    short_labels = {
        "fixed_haar_wavedown": "Fixed Haar WaveDown",
        "capacity_matched_conventional_downsampling": "Capacity-matched conventional",
        "conventional_downsampling": "Lower-capacity conventional",
        "no_learned_gates": "No learned gates",
        "all_blocks_gated": "All blocks gated",
        "se_attention": "SE replacement",
        "additive_fusion": "Mean fusion",
        "stage4_only_head": "Stage-4-only head",
    }
    data["display_label"] = data["ablation_key"].map(short_labels)
    data = data.sort_values("macro_f1_delta_pp", ascending=True)
    y = np.arange(len(data), dtype=float)
    mean = data["macro_f1_delta_pp"].to_numpy()
    low = data["macro_f1_bootstrap95_low_pp"].to_numpy()
    high = data["macro_f1_bootstrap95_high_pp"].to_numpy()
    colors = [CB_TEAL if value >= 0 else CB_ORANGE for value in mean]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.barh(y, mean, left=np.minimum(mean, 0), color=colors, alpha=0.25, height=0.62, zorder=1)
    ax.errorbar(mean, y, xerr=np.vstack([mean - low, high - mean]), fmt="none",
                ecolor="#555555", capsize=2.5, elinewidth=1.1, zorder=2)
    ax.scatter(mean, y, c=colors, s=27, zorder=3)
    for index, value in enumerate(mean):
        ax.text(value + (0.12 if value >= 0 else -0.12), y[index], f"{value:+.2f}",
                va="center", ha="left" if value >= 0 else "right", fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, data["display_label"])
    ax.set_xlabel("Macro-F1 change from full WavestarNet (percentage points)")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    fig.subplots_adjust(left=0.29, right=0.99, top=0.98, bottom=0.18)
    _save_figure(fig, destination)


def _paired_figure(comparisons: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharex=True, sharey=True)
    metric_colors = {"accuracy": CB_TEAL, "macro_f1": CB_BLUE}
    metric_labels = {"accuracy": "Accuracy", "macro_f1": "Macro-F1"}
    all_low = comparisons["bootstrap95_low_pp"].to_numpy()
    all_high = comparisons["bootstrap95_high_pp"].to_numpy()
    xmin = float(np.floor(np.nanmin(all_low) * 2.0) / 2.0 - 0.25)
    xmax = float(np.ceil(np.nanmax(all_high) * 2.0) / 2.0 + 0.25)
    for column, dataset in enumerate(DATASET_ORDER):
        ax = axes[column]
        subset = comparisons.loc[comparisons["dataset"] == dataset]
        y_values = {"accuracy": 1.0, "macro_f1": 0.0}
        for metric in ("accuracy", "macro_f1"):
            row = subset.loc[subset["metric"] == metric].iloc[0]
            mean = float(row["delta_pp"])
            low = float(row["bootstrap95_low_pp"])
            high = float(row["bootstrap95_high_pp"])
            y = y_values[metric]
            ax.barh(y, mean, color=metric_colors[metric], alpha=0.75, height=0.42, zorder=1)
            ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt="none",
                        ecolor="#444444", capsize=2.5, elinewidth=1.1, zorder=2)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axhline(0.5, color="#DDDDDD", linewidth=0.6)
        ax.set_title(dataset)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.45, 1.45)
        ax.set_yticks([1.0, 0.0])
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
        if column == 0:
            ax.set_yticklabels([metric_labels["accuracy"], metric_labels["macro_f1"]])
            ax.set_ylabel("Metric")
        else:
            ax.tick_params(axis="y", labelleft=False)
    fig.supxlabel("WavestarNet minus MobileViT-S (percentage points)", y=0.02)
    handles = [
        Patch(facecolor=CB_TEAL, edgecolor="none", alpha=0.75, label="Accuracy"),
        Patch(facecolor=CB_BLUE, edgecolor="none", alpha=0.75, label="Macro-F1"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03),
               ncol=2, frameon=False, handletextpad=0.4, columnspacing=1.2)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.79, bottom=0.23, wspace=0.14)
    _save_figure(fig, destination)


def _copy_image_as_publication_figure(source: Path, destination: Path) -> None:
    shutil.copy2(source, destination.with_suffix(".png"))
    with Image.open(source) as image:
        image.convert("RGB").save(destination.with_suffix(".pdf"), resolution=300.0)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], headers: Sequence[str], formats: dict[str, str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(formats.get(column, "{}").format(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_publication_package(repo_root: Path, output_dir: Path) -> None:
    repo = repo_root.resolve()
    output = output_dir if output_dir.is_absolute() else repo / output_dir
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    stats_root = repo / "summary/final_sealed_test_statistics"
    intervals = pd.read_csv(stats_root / "sealed_test_bootstrap_confidence_intervals.csv")
    sealed_roots = {
        "AgriVision4": repo / "agrivision4/results/sealed_test/agrivision4_scratch_research_v2_final_holdout_v1",
        "PlantDoc": repo / "plantdoc/results/sealed_test/plantdoc_scratch_final_holdout_v1",
        "PlantWild": repo / "plantwild/results/sealed_test/plantwild_scratch_final_holdout_v1",
    }
    main = pd.concat(
        [
            merge_sealed_results(dataset, pd.read_csv(root / "sealed_test_aggregate.csv"), intervals)
            for dataset, root in sealed_roots.items()
        ],
        ignore_index=True,
    )
    main.to_csv(tables_dir / "table_1_sealed_test_results.csv", index=False, float_format="%.6f")

    efficiency_root = repo / "plantwild/results/controlled_runs/PlantWild Model-Only T4 Efficiency Benchmark/plantwild_t4_efficiency_v02"
    complexity = pd.read_csv(efficiency_root / "model_complexity.csv")
    timing = pd.read_csv(efficiency_root / "efficiency_results.csv")
    efficiency = select_efficiency_rows(timing, MODEL_ORDER).merge(
        complexity[
            [
                "model_key",
                "parameters",
                "serialized_state_dict_mb",
                "forward_gflops",
                "fma_equivalent_macs",
            ]
        ],
        on="model_key",
    )
    efficiency["model"] = efficiency["model_key"].map(DISPLAY_NAMES)
    efficiency["parameters_m"] = efficiency["parameters"] / 1e6
    efficiency["macs_g"] = efficiency["fma_equivalent_macs"] / 1e9
    efficiency.to_csv(tables_dir / "table_2_efficiency.csv", index=False, float_format="%.6f")

    ablation = pd.read_csv(repo / "agrivision4/results/ablation_study/final_ablation_comparison.csv")
    ablation_stats = pd.read_csv(repo / "agrivision4/results/ablation_study/statistics/ablation_statistical_comparison.csv")
    ablation = ablation.merge(
        ablation_stats[
            [
                "ablation_key",
                "macro_f1_bootstrap95_low_pp",
                "macro_f1_bootstrap95_high_pp",
                "macro_f1_p_holm",
                "accuracy_bootstrap95_low_pp",
                "accuracy_bootstrap95_high_pp",
                "accuracy_p_holm",
            ]
        ],
        on="ablation_key",
        how="left",
    )
    ablation.to_csv(tables_dir / "table_3_ablation_results.csv", index=False, float_format="%.6f")

    comparisons = pd.read_csv(stats_root / "wavestarnet_vs_best_baseline_paired_bootstrap.csv")
    comparisons.to_csv(tables_dir / "table_4_paired_statistical_comparison.csv", index=False, float_format="%.6f")
    mcnemar = pd.read_csv(stats_root / "wavestarnet_vs_best_baseline_mcnemar.csv")
    mcnemar.to_csv(tables_dir / "table_s1_mcnemar_tests.csv", index=False, float_format="%.6f")

    gradcam_root = repo / "agrivision4/results/interpretability/gradcam_validation_v1"
    agreement = pd.read_csv(gradcam_root / "model_agreement_summary.csv")
    confidence = pd.read_csv(gradcam_root / "confidence_error_summary.csv")
    spatial = pd.read_csv(gradcam_root / "gradcam_spatial_summary.csv")
    predictions = pd.read_csv(gradcam_root / "predictions.csv")
    gradcam_rows = []
    only_correct = {
        "wavestarnet_big_transfer_v2": int(agreement.loc[agreement["category"] == "wavestarnet_only_correct", "count"].iloc[0]),
        "mobilevit_s": int(agreement.loc[agreement["category"] == "mobilevit_only_correct", "count"].iloc[0]),
    }
    for model_key in ("wavestarnet_big_transfer_v2", "mobilevit_s"):
        model_predictions = predictions.loc[predictions["model_key"] == model_key]
        prediction_correct = model_predictions["correct"].astype(bool)
        incorrect = confidence.loc[(confidence["model_key"] == model_key) & (~confidence["correct"])].iloc[0]
        correct_spatial = spatial.loc[
            (spatial["model_key"] == model_key)
            & (spatial["role"] == "predicted")
            & (spatial["correct"])
        ].iloc[0]
        gradcam_rows.append(
            {
                "model_key": model_key,
                "model": DISPLAY_NAMES[model_key],
                "model_only_correct_cases": only_correct[model_key],
                "incorrect_predictions": int(incorrect["count"]),
                "high_confidence_errors_ge_0_90": int(
                    ((~prediction_correct) & (model_predictions["confidence"] >= 0.90)).sum()
                ),
                "ece_10bin": expected_calibration_error(
                    model_predictions["confidence"].to_numpy(),
                    prediction_correct.to_numpy(),
                    bins=10,
                ),
                "correct_map_top10_mass_fraction": correct_spatial["mean_top10_mass_fraction"],
                "correct_map_spatial_entropy": correct_spatial["mean_spatial_entropy"],
                "analysis_split": "AgriVision4 locked validation",
            }
        )
    gradcam = pd.DataFrame(gradcam_rows)
    gradcam.to_csv(tables_dir / "table_5_gradcam_error_analysis.csv", index=False, float_format="%.6f")

    _performance_figure(main, figures_dir / "figure_1_sealed_test_performance")
    _efficiency_figure(main, efficiency, figures_dir / "figure_2_accuracy_efficiency_tradeoff")
    _ablation_figure(ablation, figures_dir / "figure_3_ablation_effects")
    _paired_figure(comparisons, figures_dir / "figure_4_paired_effects")
    _copy_image_as_publication_figure(
        gradcam_root / "figures/case_grid_wavestarnet_only_correct.png",
        figures_dir / "figure_5_gradcam_model_disagreement",
    )
    _copy_image_as_publication_figure(
        gradcam_root / "figures/case_grid_both_wrong.png",
        figures_dir / "figure_6_shared_failure_cases",
    )

    main_print = main.copy()
    for metric in ("accuracy", "macro_f1"):
        main_print[f"{metric}_report"] = main_print.apply(
            lambda row: (
                f"{row[f'{metric}_mean_pct']:.2f} +/- {row[f'{metric}_sd_pct']:.2f} "
                f"[{row[f'{metric}_bootstrap95_low_pct']:.2f}, "
                f"{row[f'{metric}_bootstrap95_high_pct']:.2f}]"
            ),
            axis=1,
        )
    main_print["weighted_f1_report"] = main_print.apply(
        lambda row: f"{row['weighted_f1_mean_pct']:.2f} +/- {row['weighted_f1_sd_pct']:.2f}",
        axis=1,
    )
    ablation_print = ablation.copy()
    ablation_print["macro_f1_report"] = ablation_print.apply(
        lambda row: f"{100 * row['macro_f1_mean']:.2f} +/- {100 * row['macro_f1_sd']:.2f}",
        axis=1,
    )
    ablation_print["macro_f1_effect_report"] = ablation_print.apply(
        lambda row: "Reference" if pd.isna(row["macro_f1_bootstrap95_low_pp"]) else (
            f"{row['macro_f1_delta_pp']:+.2f} "
            f"[{row['macro_f1_bootstrap95_low_pp']:.2f}, "
            f"{row['macro_f1_bootstrap95_high_pp']:.2f}], "
            f"p={row['macro_f1_p_holm']:.3f}"
        ),
        axis=1,
    )

    table_markdown = ["# Publication Tables", ""]
    table_markdown.extend(
        [
            "## Table 1. Sealed-Test Performance",
            "",
            _markdown_table(
                main_print,
                ("dataset", "publication_name", "parameters", "accuracy_report", "macro_f1_report", "weighted_f1_report"),
                ("Dataset", "Model", "Parameters", "Accuracy: mean +/- SD [95% CI]", "Macro-F1: mean +/- SD [95% CI]", "Weighted-F1: mean +/- SD"),
                {"parameters": "{:,}"},
            ),
            "",
            "## Table 2. Model Complexity and Efficiency",
            "",
            _markdown_table(
                efficiency,
                ("model", "parameters_m", "forward_gflops", "macs_g", "serialized_state_dict_mb", "cpu_fp32_bs1_latency_ms", "t4_fp32_bs1_latency_ms", "t4_fp16_bs32_throughput_images_s", "t4_fp16_bs32_peak_allocated_mb"),
                ("Model", "Params (M)", "FLOPs (G)", "MACs (G)", "Size (MB)", "CPU ms", "T4 ms", "T4 img/s", "Peak MB"),
                {column: "{:.2f}" for column in ("parameters_m", "forward_gflops", "macs_g", "serialized_state_dict_mb", "cpu_fp32_bs1_latency_ms", "t4_fp32_bs1_latency_ms", "t4_fp16_bs32_throughput_images_s", "t4_fp16_bs32_peak_allocated_mb")},
            ),
            "",
            "## Table 3. AgriVision4 Ablation Study",
            "",
            _markdown_table(
                ablation_print,
                ("display_name", "parameters", "macro_f1_report", "macro_f1_effect_report", "accuracy_mean", "accuracy_delta_pp"),
                ("Variant", "Parameters", "Macro-F1 (%)", "Macro-F1 effect: delta [95% CI], Holm p", "Accuracy", "Delta (pp)"),
                {"parameters": "{:,}", "accuracy_mean": "{:.4f}", "accuracy_delta_pp": "{:+.2f}"},
            ),
            "",
            "## Table 4. Paired Sealed-Test Comparison Against MobileViT-S",
            "",
            _markdown_table(
                comparisons,
                ("dataset", "metric", "candidate_mean_pct", "reference_mean_pct", "delta_pp", "bootstrap95_low_pp", "bootstrap95_high_pp", "paired_bootstrap_p_two_sided"),
                ("Dataset", "Metric", "WavestarNet (%)", "MobileViT-S (%)", "Delta (pp)", "CI low", "CI high", "p"),
                {column: "{:.3f}" for column in ("candidate_mean_pct", "reference_mean_pct", "delta_pp", "bootstrap95_low_pp", "bootstrap95_high_pp", "paired_bootstrap_p_two_sided")},
            ),
            "",
            "## Table 5. Grad-CAM and Error Analysis",
            "",
            _markdown_table(
                gradcam,
                ("model", "model_only_correct_cases", "incorrect_predictions", "high_confidence_errors_ge_0_90", "ece_10bin", "correct_map_top10_mass_fraction", "correct_map_spatial_entropy"),
                ("Model", "Only correct", "Errors", "Errors >=0.90", "ECE", "Top-10% mass", "Spatial entropy"),
                {"model_only_correct_cases": "{:d}", "incorrect_predictions": "{:d}", "high_confidence_errors_ge_0_90": "{:d}", "ece_10bin": "{:.4f}", "correct_map_top10_mass_fraction": "{:.4f}", "correct_map_spatial_entropy": "{:.4f}"},
            ),
            "",
            "## Supplementary Table S1. Exact McNemar Tests",
            "",
            _markdown_table(
                mcnemar,
                ("dataset", "seed", "candidate_only_correct", "reference_only_correct", "p_exact_two_sided", "p_holm_all_tests", "reject_0_05_after_holm"),
                ("Dataset", "Seed", "WavestarNet only", "MobileViT-S only", "Exact p", "Holm p", "Reject"),
                {"seed": "{:d}", "candidate_only_correct": "{:d}", "reference_only_correct": "{:d}", "p_exact_two_sided": "{:.4f}", "p_holm_all_tests": "{:.4f}"},
            ),
            "",
        ]
    )
    (tables_dir / "publication_tables.md").write_text("\n".join(table_markdown), encoding="utf-8")

    captions = """# Figure Captions

**Figure 1. Sealed-test performance across three datasets.** Points show the mean across seeds 42, 2024, and 3407; error bars show 95% class-stratified bootstrap confidence intervals over held-out images. All models were trained from random initialization under the matched scratch protocol.

**Figure 2. Accuracy-efficiency trade-off on PlantWild.** Predictive performance is the three-seed sealed-test Macro-F1. Complexity and latency use model-only synthetic 224 x 224 input on the standardized benchmark. T4 latency is FP32, batch size 1. Lower horizontal values and higher vertical values are preferable.

**Figure 3. Component ablations relative to full WavestarNet.** Points show the three-seed mean Macro-F1 difference on the locked AgriVision4 validation split; error bars show class-stratified paired-bootstrap 95% confidence intervals. Negative values favor the complete model.

**Figure 4. Paired sealed-test effects against MobileViT-S.** Positive values favor WavestarNet. Error bars show class-stratified paired-bootstrap 95% confidence intervals. Every interval crosses zero, supporting statistical comparability rather than superiority.

**Figure 5. Representative AgriVision4 cases correctly classified only by WavestarNet.** Grad-CAM overlays compare the evidence used by both models. WavestarNet maps are descriptively more concentrated, but lesion-localization superiority is not claimed because pixel-level lesion annotations are unavailable.

**Figure 6. Representative high-confidence failures shared by both models.** Examples illustrate confusion between visually overlapping symptom categories, mild disease and healthy tissue, and occasional crop-level shortcut behavior.
"""
    (output / "figure_captions.md").write_text(captions, encoding="utf-8")

    readme = """# WavestarNet Publication Package V1

Status: **FROZEN**. No further model selection or hyperparameter tuning is permitted for these reported experiments.

## Protocol Boundaries

- Tables 1, 2, and 4 use final sealed-test results or model-only efficiency measurements.
- Table 3 uses only the locked AgriVision4 validation split for controlled architecture ablations.
- Table 5 and Figures 5-6 use only the locked AgriVision4 validation split for interpretability and error analysis.
- All predictive comparisons in Table 1 use random initialization and the matched three-seed scratch protocol.
- Transfer-learning and knowledge-distillation experiments are intentionally excluded from this primary package.

## Main Statistical Conclusion

WavestarNet is statistically comparable with MobileViT-S on all three sealed test sets. It is numerically higher on AgriVision4 and PlantDoc and numerically lower on PlantWild; all paired-bootstrap confidence intervals include zero, and no seed-specific McNemar test is significant after Holm correction.

## Rebuild

Run `python -m research_evaluation.publication_package --repo-root .` after verifying that the frozen inputs listed in `freeze_manifest.csv` are unchanged.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    (output / "FROZEN.md").write_text(
        "# Frozen Result Set\n\nFrozen on 2026-09-09 after sealed-test, statistical, efficiency, ablation, and Grad-CAM analyses. No further model tuning is part of this result set.\n",
        encoding="utf-8",
    )

    ablation_sources = [
        Path(path) for path in pd.read_csv(repo / "agrivision4/results/ablation_study/final_ablation_comparison.csv")["source"].unique()
    ]
    source_paths: list[tuple[str, Path]] = [
        *(('sealed_test', root) for root in sealed_roots.values()),
        ("statistical_analysis", stats_root),
        ("ablation_summary", repo / "agrivision4/results/ablation_study"),
        *(('ablation_source', repo / path) for path in ablation_sources),
        ("efficiency", efficiency_root),
        ("interpretability", gradcam_root),
        ("architecture", repo / "agrivision4/models/wavestarnet_family.py"),
        ("analysis_code", repo / "research_evaluation/sealed_statistics.py"),
        ("analysis_code", repo / "research_evaluation/publication_package.py"),
        ("publication_deliverable", output / "tables"),
        ("publication_deliverable", output / "figures"),
        ("publication_deliverable", output / "figure_captions.md"),
        ("publication_deliverable", output / "README.md"),
        ("publication_deliverable", output / "FROZEN.md"),
    ]
    rows = build_freeze_rows(repo, source_paths)
    manifest_csv = output / "freeze_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_csv, index=False)
    category_counts = pd.DataFrame(rows).groupby("category").agg(files=("relative_path", "size"), bytes=("bytes", "sum")).reset_index()
    freeze = {
        "freeze_id": "wavestarnet_publication_results_v1",
        "status": "frozen_no_further_tuning",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(rows),
        "total_bytes": int(sum(int(row["bytes"]) for row in rows)),
        "manifest_csv_sha256": sha256_file(manifest_csv),
        "category_summary": category_counts.to_dict(orient="records"),
    }
    (output / "freeze_manifest.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("summary/publication_package_v1"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    build_publication_package(args.repo_root, args.output_dir)


if __name__ == "__main__":
    main()
