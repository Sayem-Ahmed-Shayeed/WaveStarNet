"""Fixed-label classification metrics for publication-quality evaluation."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def classification_metrics(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    labels: Sequence[Hashable],
    sample_ids: Sequence[Hashable] | None = None,
) -> dict[str, Any]:
    """Calculate metrics over an explicit, fixed label set."""
    true_values = list(y_true)
    pred_values = list(y_pred)
    ordered_labels = list(labels)

    if not ordered_labels or len(set(ordered_labels)) != len(ordered_labels):
        raise ValueError("labels must be a non-empty sequence of unique values")
    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred must have the same length")
    if not true_values:
        raise ValueError("y_true and y_pred must not be empty")

    if sample_ids is not None:
        ids = list(sample_ids)
        if len(ids) != len(true_values):
            raise ValueError("sample_ids and predictions must have the same length")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate sample_id values are not allowed")

    label_to_index = {label: index for index, label in enumerate(ordered_labels)}
    unknown = (set(true_values) | set(pred_values)) - set(ordered_labels)
    if unknown:
        rendered = ", ".join(sorted(map(str, unknown)))
        raise ValueError(f"values outside explicit labels: {rendered}")

    size = len(ordered_labels)
    confusion = [[0 for _ in range(size)] for _ in range(size)]
    for target, prediction in zip(true_values, pred_values):
        confusion[label_to_index[target]][label_to_index[prediction]] += 1

    per_class: dict[str, dict[str, float | int | str]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    supports: list[int] = []

    for index, label in enumerate(ordered_labels):
        true_positive = confusion[index][index]
        support = sum(confusion[index])
        predicted = sum(row[index] for row in confusion)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1_scores.append(float(f1))
        supports.append(int(support))
        per_class[str(label)] = {
            "label": str(label),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }

    total = len(true_values)
    correct = sum(confusion[index][index] for index in range(size))

    def weighted(values: Sequence[float]) -> float:
        return float(sum(value * support for value, support in zip(values, supports)) / total)

    return {
        "labels": [str(label) for label in ordered_labels],
        "n_samples": total,
        "accuracy": float(correct / total),
        "balanced_accuracy": _mean(recalls),
        "macro_precision": _mean(precisions),
        "macro_recall": _mean(recalls),
        "macro_f1": _mean(f1_scores),
        "weighted_precision": weighted(precisions),
        "weighted_recall": weighted(recalls),
        "weighted_f1": weighted(f1_scores),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }
