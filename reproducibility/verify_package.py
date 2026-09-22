#!/usr/bin/env python3
"""Validate the compact publication reproducibility package."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = {
    "efficientnet_b0",
    "efficientnet_b3",
    "mobilenetv3_large_100",
    "mobilevit_s",
    "wavestarnet_big_transfer_v2",
}
SEEDS = {"42", "2024", "3407"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(path: str) -> None:
    target = ROOT / path
    if not target.exists():
        raise AssertionError(f"missing required artifact: {path}")


def main() -> int:
    files = [path for path in ROOT.rglob("*") if path.is_file()]
    forbidden = [path for path in files if path.suffix.lower() in {".pt", ".zip", ".pyc"}]
    forbidden += [path for path in files if "__pycache__" in path.parts]
    if forbidden:
        raise AssertionError(f"forbidden Git artifact: {forbidden[0].relative_to(ROOT)}")

    too_large = [path for path in files if path.stat().st_size > 50 * 1024 * 1024]
    if too_large:
        raise AssertionError(f"file exceeds 50 MiB: {too_large[0].relative_to(ROOT)}")

    for notebook in ROOT.rglob("*.ipynb"):
        parsed = json.loads(notebook.read_text(encoding="utf-8"))
        if "cells" not in parsed or "nbformat" not in parsed:
            raise AssertionError(f"invalid notebook structure: {notebook.relative_to(ROOT)}")

    with (ROOT / "checksums.sha256").open(encoding="utf-8") as handle:
        for line in handle:
            expected, relative = line.rstrip("\n").split("  ", 1)
            path = ROOT / relative
            if not path.is_file() or digest(path) != expected:
                raise AssertionError(f"checksum mismatch: {relative}")

    workspace = ROOT.parent
    source_tree_available = (workspace / "agrivision4").is_dir()
    with (ROOT / "artifact_manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    for row in manifest_rows:
        source = row["source_path"]
        packaged = ROOT / row["repository_path"]
        if source == "generated_for_reproducibility_package" or source.startswith("adapted:"):
            continue
        if not source_tree_available:
            continue
        if source.startswith("zip:"):
            archive_name, member_name = source[4:].split("!/", 1)
            with zipfile.ZipFile(workspace / archive_name) as archive:
                member_digest = hashlib.sha256(archive.read(member_name)).hexdigest()
            if member_digest != row["sha256"]:
                raise AssertionError(f"ZIP-member provenance mismatch: {source}")
        else:
            source_path = workspace / source
            if not source_path.is_file() or digest(source_path) != digest(packaged):
                raise AssertionError(f"source-copy mismatch: {row['repository_path']} <- {source}")

    required = [
        "agrivision4/test_evaluation/results/agrivision4_scratch_research_v2_final_holdout_v1/sealed_test_aggregate.csv",
        "plantdoc/test_evaluation/results/plantdoc_scratch_final_holdout_v1/sealed_test_aggregate.csv",
        "plantwild/test_evaluation/results/plantwild_scratch_final_holdout_v1/sealed_test_aggregate.csv",
        "agrivision4/ablation/results/summary/final_ablation_comparison.csv",
        "agrivision4/ablation/results/summary/statistics/ablation_statistical_comparison.csv",
        "plantwild/wavedown_confirmation/results/plantwild_wavestarnet_wavedown_final_holdout_v01/paired_test_statistics.csv",
        "plantwild/efficiency/results/plantwild_t4_efficiency_v02/efficiency_results.csv",
        "agrivision4/interpretability/results/gradcam_validation_v1/gradcam_spatial_summary.csv",
    ]
    for path in required:
        require(path)

    for dataset, evaluation in {
        "agrivision4": "agrivision4_scratch_research_v2_final_holdout_v1",
        "plantdoc": "plantdoc_scratch_final_holdout_v1",
        "plantwild": "plantwild_scratch_final_holdout_v1",
    }.items():
        aggregate = ROOT / dataset / "test_evaluation" / "results" / evaluation / "sealed_test_seed_metrics.csv"
        with aggregate.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        model_field = "model_key" if rows and "model_key" in rows[0] else "model"
        observed_models = {row[model_field] for row in rows}
        observed_seeds = {str(row["seed"]) for row in rows}
        if observed_models != MODELS or observed_seeds != SEEDS or len(rows) != 15:
            raise AssertionError(
                f"unexpected test coverage for {dataset}: models={observed_models}, seeds={observed_seeds}, rows={len(rows)}"
            )

    print(f"verified_files={len(files)}")
    print("datasets=agrivision4,plantdoc,plantwild")
    print("test_rows=15_per_dataset")
    print(f"source_provenance={'verified' if source_tree_available else 'not_available_in_clone'}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
