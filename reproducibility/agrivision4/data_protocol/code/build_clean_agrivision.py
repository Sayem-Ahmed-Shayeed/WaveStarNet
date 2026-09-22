"""Build and validate the conservative AgriVision 4 cleaned derivative."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

try:
    from tools.agrivision_cleaning import (
        ComponentExclusion,
        ComponentMembership,
        ImageRecord,
        RetainedImage,
        connected_components,
        exact_duplicate_edges,
        filter_supported_classes,
        near_duplicate_edges,
        normalized_rgb_thumbnail,
        perceptual_signature,
        resolve_components,
        stratified_split,
    )
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    from agrivision_cleaning import (  # type: ignore[no-redef]
        ComponentExclusion,
        ComponentMembership,
        ImageRecord,
        RetainedImage,
        connected_components,
        exact_duplicate_edges,
        filter_supported_classes,
        near_duplicate_edges,
        normalized_rgb_thumbnail,
        perceptual_signature,
        resolve_components,
        stratified_split,
    )


MANIFEST_NAME = "agri_vision4_split_manifest.csv"
IMAGE_ZIP_NAME = "agrivision4-clean-grouped-images.zip"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    total_images: int
    split_counts: dict[str, int]
    package_sha256: str


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_basename(source_path: str) -> str:
    basename = Path(source_path).name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("_")
    return safe or "image"


def _zip_write_bytes(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    archive.writestr(info, payload)


def _load_source_rows(archive: zipfile.ZipFile) -> list[dict[str, str]]:
    try:
        manifest_bytes = archive.read(MANIFEST_NAME)
    except KeyError as error:
        raise ValueError(f"source archive is missing {MANIFEST_NAME}") from error
    rows = list(csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8-sig"))))
    required = {"sample_id", "relative_path", "crop", "label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"source manifest must contain {sorted(required)}")
    paths = [row["relative_path"] for row in rows]
    sample_ids = [row["sample_id"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("source manifest contains duplicate relative_path values")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("source manifest contains duplicate sample_id values")
    return sorted(rows, key=lambda row: row["relative_path"])


def _inventory_source(
    archive: zipfile.ZipFile, rows: list[dict[str, str]]
) -> list[ImageRecord]:
    archive_names = set(archive.namelist())
    records: list[ImageRecord] = []
    for index, row in enumerate(rows):
        relative_path = row["relative_path"]
        if relative_path not in archive_names:
            raise ValueError(f"source manifest image is missing: {relative_path}")
        payload = archive.read(relative_path)
        try:
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                width, height = image.size
                signature = perceptual_signature(image)
                thumbnail = normalized_rgb_thumbnail(image)
        except Exception as error:
            raise ValueError(f"unreadable source image: {relative_path}") from error
        records.append(
            ImageRecord(
                index=index,
                archive_path=relative_path,
                label=row["label"],
                crop=row["crop"],
                sha256=hashlib.sha256(payload).hexdigest(),
                md5=hashlib.md5(payload).hexdigest(),
                byte_size=len(payload),
                width=width,
                height=height,
                perceptual_hash=signature,
                rgb_thumbnail=thumbnail,
            )
        )
        if len(records) % 250 == 0:
            print(f"Inventoried {len(records)}/{len(rows)} images", flush=True)
    return records


def _manifest_rows(
    retained: list[RetainedImage],
    split: dict[str, list[RetainedImage]],
    class_to_id: dict[str, int],
) -> list[dict[str, Any]]:
    component_split = {
        item.component_id: split_name
        for split_name, values in split.items()
        for item in values
    }
    rows = []
    for sample_index, item in enumerate(
        sorted(retained, key=lambda value: (value.record.label, value.component_id))
    ):
        record = item.record
        class_id = class_to_id[record.label]
        packaged_path = (
            f"images/{class_id:02d}/"
            f"{item.component_id}_{_safe_basename(record.archive_path)}"
        )
        rows.append(
            {
                "sample_id": f"clean_{sample_index:06d}",
                "component_id": item.component_id,
                "split": component_split[item.component_id],
                "label": record.label,
                "label_id": class_id,
                "crop": record.crop,
                "packaged_path": packaged_path,
                "source_archive_path": record.archive_path,
                "sha256": record.sha256,
                "md5": record.md5,
                "width": record.width,
                "height": record.height,
                "byte_size": record.byte_size,
            }
        )
    return rows


def _membership_rows(
    records: list[ImageRecord], memberships: list[ComponentMembership]
) -> list[dict[str, Any]]:
    by_index = {record.index: record for record in records}
    return [
        {
            "component_id": membership.component_id,
            "source_index": membership.record_index,
            "source_archive_path": by_index[membership.record_index].archive_path,
            "label": by_index[membership.record_index].label,
            "sha256": by_index[membership.record_index].sha256,
            "md5": by_index[membership.record_index].md5,
            "canonical": membership.canonical,
            "retained": membership.retained,
            "reason": membership.reason,
        }
        for membership in memberships
    ]


def _exclusion_rows(
    records: list[ImageRecord], exclusions: list[ComponentExclusion]
) -> list[dict[str, Any]]:
    by_index = {record.index: record for record in records}
    rows = []
    for exclusion in exclusions:
        members = [by_index[index] for index in exclusion.member_indices]
        rows.append(
            {
                "component_id": exclusion.component_id,
                "reason": exclusion.reason,
                "labels": "|".join(exclusion.labels),
                "member_count": len(members),
                "source_archive_paths": "|".join(
                    sorted(record.archive_path for record in members)
                ),
            }
        )
    return rows


def _write_readme(
    root: Path,
    summary: dict[str, Any],
    source_sha256: str,
    max_hamming: int,
    max_mad: float,
    minimum_components_per_class: int,
) -> None:
    text = f"""# AgriVision 4 Clean Grouped V1

This private derivative was built from the supplied AgriVision 4 original-image archive.

## Cleaning

- Source images inventoried: {summary['source_images']}
- Canonical images retained: {summary['retained_images']}
- Conflicting-label images excluded: {summary['excluded_conflicting_images']}
- Under-supported canonical images excluded: {summary['excluded_under_supported_images']}
- Retained benchmark classes: {summary['class_count']}
- Minimum independent components per retained class: {minimum_components_per_class}
- Exact duplicate edges: {summary['exact_duplicate_edges']}
- Confirmed near-copy edges: {summary['near_duplicate_edges']}
- Near-copy thresholds: perceptual Hamming <= {max_hamming}, RGB MAD <= {max_mad}
- Split seed/fractions: 42 / 70-15-15

Every multi-label duplicate component is excluded. Every retained single-label component contributes one unchanged canonical source file. The test split must not be evaluated with legacy checkpoints because they may have trained on reassigned images.

## Provenance and license warning

The source package identifies Mendeley dataset `8t6k37ztxc`, but the exact upstream version, license, and redistribution terms were not supplied. This Kaggle dataset must remain private until those terms are verified. Kaggle license metadata is `other`; it is not a new license grant.

Source archive SHA-256: `{source_sha256}`

See `audit/` for exclusions, components, thresholds, distributions, and hashes. See `manifests/` for the fixed class mapping and splits.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def _copy_code(root: Path) -> None:
    code_dir = root / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    sources = (
        Path(__file__),
        Path(__file__).with_name("agrivision_cleaning.py"),
        Path(__file__).with_name("research_metrics.py"),
    )
    for source in sources:
        shutil.copy2(source, code_dir / source.name)


def build_dataset(
    source_zip: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    max_hamming: int = 4,
    max_mad: float = 0.001,
    minimum_components_per_class: int = 20,
) -> Path:
    source_path = Path(source_zip).resolve()
    output_path = Path(output_dir).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path.exists():
        raise FileExistsError(f"output directory already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.building-", dir=output_path.parent)
    )

    try:
        source_sha256 = _sha256_file(source_path)
        with zipfile.ZipFile(source_path) as source_archive:
            source_rows = _load_source_rows(source_archive)
            records = _inventory_source(source_archive, source_rows)
            exact_edges = exact_duplicate_edges(records)
            print("Screening conservative near copies", flush=True)
            near_edges = near_duplicate_edges(records, max_hamming, max_mad)
            components = connected_components(records, exact_edges + near_edges)
            canonical_retained, exclusions, memberships = resolve_components(
                records, components
            )
            retained, under_supported = filter_supported_classes(
                canonical_retained, minimum_components_per_class
            )
            split = stratified_split(retained, seed=seed)

            labels = sorted({item.record.label for item in retained})
            class_to_id = {label: index for index, label in enumerate(labels)}
            manifest_rows = _manifest_rows(retained, split, class_to_id)

            image_zip = temporary / IMAGE_ZIP_NAME
            with zipfile.ZipFile(image_zip, "w") as target_archive:
                row_by_component = {row["component_id"]: row for row in manifest_rows}
                for item in sorted(retained, key=lambda value: value.component_id):
                    row = row_by_component[item.component_id]
                    _zip_write_bytes(
                        target_archive,
                        row["packaged_path"],
                        source_archive.read(item.record.archive_path),
                    )

        manifest_fields = list(manifest_rows[0])
        _write_csv(temporary / "manifests/all.csv", manifest_fields, manifest_rows)
        for split_name in ("train", "validation", "test"):
            _write_csv(
                temporary / f"manifests/{split_name}.csv",
                manifest_fields,
                (row for row in manifest_rows if row["split"] == split_name),
            )
        _write_json(
            temporary / "manifests/class_mapping.json",
            {
                "class_to_id": class_to_id,
                "id_to_class": {str(value): key for key, value in class_to_id.items()},
            },
        )

        membership_rows = _membership_rows(records, memberships)
        _write_csv(
            temporary / "audit/duplicate_components.csv",
            list(membership_rows[0]),
            membership_rows,
        )
        exclusion_rows = _exclusion_rows(records, exclusions)
        _write_csv(
            temporary / "audit/excluded_conflicting_components.csv",
            ["component_id", "reason", "labels", "member_count", "source_archive_paths"],
            exclusion_rows,
        )
        under_supported_rows = [
            {
                "label": label,
                "independent_components": count,
                "minimum_required": minimum_components_per_class,
                "canonical_paths": "|".join(
                    sorted(
                        item.record.archive_path
                        for item in canonical_retained
                        if item.record.label == label
                    )
                ),
            }
            for label, count in under_supported.items()
        ]
        _write_csv(
            temporary / "audit/excluded_under_supported_classes.csv",
            [
                "label",
                "independent_components",
                "minimum_required",
                "canonical_paths",
            ],
            under_supported_rows,
        )
        edge_rows = [
            {
                "left_index": edge.left,
                "right_index": edge.right,
                "left_path": records[edge.left].archive_path,
                "right_path": records[edge.right].archive_path,
                "left_label": records[edge.left].label,
                "right_label": records[edge.right].label,
                "hamming": edge.hamming,
                "rgb_mad": f"{edge.rgb_mad:.10f}",
            }
            for edge in near_edges
        ]
        _write_csv(
            temporary / "audit/near_duplicate_pairs.csv",
            [
                "left_index",
                "right_index",
                "left_path",
                "right_path",
                "left_label",
                "right_label",
                "hamming",
                "rgb_mad",
            ],
            edge_rows,
        )

        class_counts = Counter(row["label"] for row in manifest_rows)
        class_rows = []
        for label in labels:
            class_rows.append(
                {
                    "label": label,
                    "label_id": class_to_id[label],
                    "total": class_counts[label],
                    "train": sum(
                        row["label"] == label and row["split"] == "train"
                        for row in manifest_rows
                    ),
                    "validation": sum(
                        row["label"] == label and row["split"] == "validation"
                        for row in manifest_rows
                    ),
                    "test": sum(
                        row["label"] == label and row["split"] == "test"
                        for row in manifest_rows
                    ),
                }
            )
        _write_csv(
            temporary / "audit/class_distribution.csv",
            ["label", "label_id", "total", "train", "validation", "test"],
            class_rows,
        )
        split_counts = Counter(row["split"] for row in manifest_rows)
        _write_csv(
            temporary / "audit/split_distribution.csv",
            ["split", "count"],
            ({"split": name, "count": split_counts[name]} for name in ("train", "validation", "test")),
        )

        excluded_indices = {
            index for exclusion in exclusions for index in exclusion.member_indices
        }
        summary = {
            "source_archive": source_path.name,
            "source_archive_sha256": source_sha256,
            "source_images": len(records),
            "canonical_images_before_support_filter": len(canonical_retained),
            "retained_images": len(retained),
            "excluded_conflicting_components": len(exclusions),
            "excluded_conflicting_images": len(excluded_indices),
            "excluded_under_supported_classes": len(under_supported),
            "excluded_under_supported_images": sum(under_supported.values()),
            "minimum_components_per_class": minimum_components_per_class,
            "duplicate_components": sum(len(component) > 1 for component in components),
            "duplicate_images_discarded": (
                len(records) - len(excluded_indices) - len(canonical_retained)
            ),
            "exact_duplicate_edges": len(exact_edges),
            "near_duplicate_edges": len(near_edges),
            "component_count": len(components),
            "class_count": len(labels),
            "seed": seed,
            "fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "split_counts": {name: split_counts[name] for name in ("train", "validation", "test")},
            "near_duplicate_thresholds": {
                "signature_size": 32,
                "max_hamming": max_hamming,
                "max_rgb_mad": max_mad,
            },
        }
        _write_json(temporary / "audit/audit_summary.json", summary)
        (temporary / "audit/source_archive_sha256.txt").write_text(
            f"{source_sha256}  {source_path.name}\n", encoding="utf-8"
        )
        _write_json(
            temporary / "dataset-metadata.json",
            {
                "title": f"AgriVision 4 Clean Grouped {len(labels)}-Class V1",
                "id": "sumitnath07/agrivision4-clean-grouped-v1",
                "licenses": [{"name": "other"}],
                "description": (
                    "Private conservative duplicate-cleaned AgriVision 4 derivative "
                    f"with a {len(labels)}-class deterministic benchmark manifest."
                ),
            },
        )
        _write_readme(
            temporary,
            summary,
            source_sha256,
            max_hamming,
            max_mad,
            minimum_components_per_class,
        )
        _copy_code(temporary)
        validation = validate_package(temporary)
        _write_json(temporary / "audit/package_validation.json", asdict(validation))
        os.replace(temporary, output_path)
        return output_path
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_package(root: str | Path, verify_hashes: bool = True) -> ValidationReport:
    package_root = Path(root).resolve()
    required = (
        package_root / "README.md",
        package_root / "dataset-metadata.json",
        package_root / "manifests/all.csv",
        package_root / "manifests/train.csv",
        package_root / "manifests/validation.csv",
        package_root / "manifests/test.csv",
        package_root / "manifests/class_mapping.json",
        package_root / "audit/audit_summary.json",
        package_root / IMAGE_ZIP_NAME,
    )
    missing_files = [str(path) for path in required if not path.is_file()]
    if missing_files:
        raise ValueError(f"missing package files: {missing_files}")

    all_rows = _read_csv(package_root / "manifests/all.csv")
    if not all_rows:
        raise ValueError("all manifest is empty")
    for key in ("sample_id", "component_id", "packaged_path", "sha256"):
        values = [row[key] for row in all_rows]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {key} in all manifest")

    split_rows = {
        name: _read_csv(package_root / f"manifests/{name}.csv")
        for name in ("train", "validation", "test")
    }
    all_ids = {row["sample_id"] for row in all_rows}
    split_id_sets = {
        name: {row["sample_id"] for row in rows} for name, rows in split_rows.items()
    }
    if set.union(*split_id_sets.values()) != all_ids:
        raise ValueError("split manifests do not cover the all manifest")
    if any(
        split_id_sets[left] & split_id_sets[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("split manifests overlap")
    for name, rows in split_rows.items():
        if any(row["split"] != name for row in rows):
            raise ValueError(f"{name} manifest contains a different split")

    class_mapping = json.loads(
        (package_root / "manifests/class_mapping.json").read_text(encoding="utf-8")
    )
    class_to_id = {key: int(value) for key, value in class_mapping["class_to_id"].items()}
    for row in all_rows:
        if row["label"] not in class_to_id or int(row["label_id"]) != class_to_id[row["label"]]:
            raise ValueError("manifest label mapping mismatch")

    image_zip = package_root / IMAGE_ZIP_NAME
    expected_paths = {row["packaged_path"] for row in all_rows}
    with zipfile.ZipFile(image_zip) as archive:
        packaged_paths = set(archive.namelist())
        missing_images = expected_paths - packaged_paths
        if missing_images:
            raise ValueError(f"missing packaged image: {sorted(missing_images)[0]}")
        extra_images = packaged_paths - expected_paths
        if extra_images:
            raise ValueError(f"unexpected packaged image: {sorted(extra_images)[0]}")
        if verify_hashes:
            for row in all_rows:
                digest = hashlib.sha256(archive.read(row["packaged_path"])).hexdigest()
                if digest != row["sha256"]:
                    raise ValueError(f"packaged image hash mismatch: {row['packaged_path']}")

    summary = json.loads(
        (package_root / "audit/audit_summary.json").read_text(encoding="utf-8")
    )
    split_counts = {name: len(rows) for name, rows in split_rows.items()}
    if summary["retained_images"] != len(all_rows) or summary["split_counts"] != split_counts:
        raise ValueError("audit summary counts do not match manifests")

    return ValidationReport(
        ok=True,
        total_images=len(all_rows),
        split_counts=split_counts,
        package_sha256=_sha256_file(image_zip),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hamming", type=int, default=4)
    parser.add_argument("--max-mad", type=float, default=0.001)
    parser.add_argument("--minimum-components-per-class", type=int, default=20)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.validate_only:
        report = validate_package(args.output_dir)
    else:
        if args.source_zip is None:
            raise SystemExit("--source-zip is required unless --validate-only is used")
        output = build_dataset(
            args.source_zip,
            args.output_dir,
            seed=args.seed,
            max_hamming=args.max_hamming,
            max_mad=args.max_mad,
            minimum_components_per_class=args.minimum_components_per_class,
        )
        report = ValidationReport(
            **json.loads(
                (output / "audit/package_validation.json").read_text(encoding="utf-8")
            )
        )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
