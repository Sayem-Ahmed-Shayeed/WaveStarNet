"""Deterministic, non-destructive audit for the local PlantWild v2 release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.fft import dctn


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MANIFEST_FIELDS = [
    "sample_id",
    "component_id",
    "split",
    "label",
    "label_id",
    "source_relative_path",
    "source_engine",
    "file_sha256",
    "pixel_sha256",
    "width",
    "height",
    "byte_size",
    "quality_flags",
]


@dataclass(frozen=True)
class AuditConfig:
    source_root: Path
    output_root: Path
    source_archive: Path | None = None
    seed: int = 42
    split_fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)
    minimum_components_per_class: int = 20
    enable_near_duplicates: bool = True
    probable_phash_hamming: int = 6
    confirmed_phash_hamming: int = 4
    confirmed_rgb_mad: float = 0.045
    confirmed_gray_correlation: float = 0.985
    confirmed_aspect_relative_delta: float = 0.03
    probable_rgb_mad: float = 0.08
    probable_gray_correlation: float = 0.96
    probable_aspect_relative_delta: float = 0.08


@dataclass
class ImageItem:
    index: int
    relative_path: str
    label: str
    source_engine: str
    file_sha256: str
    pixel_sha256: str
    byte_size: int
    width: int
    height: int
    mode: str
    phash: int
    rgb_thumbnail: np.ndarray = field(repr=False)
    gray_thumbnail: np.ndarray = field(repr=False)
    quality: dict[str, float] = field(repr=False)
    quality_flags: tuple[str, ...]

    @property
    def pixel_count(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class PairEdge:
    left: int
    right: int
    kind: str
    evidence: str
    hamming: int = 0
    rgb_mad: float = 0.0
    gray_correlation: float = 0.0
    aspect_relative_delta: float = 0.0


def parse_source_engine(filename: str) -> str:
    value = Path(filename).stem.casefold()
    for engine in ("google", "bing", "baidu"):
        if re.search(rf"(?:^|[^a-z]){engine}(?:[^a-z]|$)", value):
            return engine
    return "unspecified"


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


def _perceptual_hash(gray: np.ndarray) -> int:
    resized = np.asarray(
        Image.fromarray(gray.astype(np.uint8), mode="L").resize(
            (32, 32), Image.Resampling.LANCZOS
        ),
        dtype=np.float32,
    )
    low_frequency = dctn(resized, type=2, norm="ortho")[:8, :8]
    threshold = float(np.median(low_frequency.ravel()[1:]))
    value = 0
    for enabled in (low_frequency >= threshold).ravel():
        value = (value << 1) | int(enabled)
    return value


def _entropy(gray: np.ndarray) -> float:
    histogram = np.bincount(gray.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / histogram.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _quality_metrics(rgb: np.ndarray, gray: np.ndarray) -> tuple[dict[str, float], tuple[str, ...]]:
    height, width = gray.shape
    gray64 = np.asarray(
        Image.fromarray(gray, mode="L").resize((64, 64), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    laplacian = (
        -4.0 * gray64[1:-1, 1:-1]
        + gray64[:-2, 1:-1]
        + gray64[2:, 1:-1]
        + gray64[1:-1, :-2]
        + gray64[1:-1, 2:]
    )
    abs_laplacian = np.abs(laplacian)
    border_width = 10
    border_mask = np.zeros_like(abs_laplacian, dtype=bool)
    border_mask[:border_width, :] = True
    border_mask[-border_width:, :] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True
    center_mask = ~border_mask
    border_edge = float(abs_laplacian[border_mask].mean())
    center_edge = float(abs_laplacian[center_mask].mean())

    column_change = np.mean(np.abs(np.diff(gray64, axis=1)), axis=0)
    row_change = np.mean(np.abs(np.diff(gray64, axis=0)), axis=1)
    median_change = float(np.median(np.concatenate([column_change, row_change]))) + 1e-6
    seam_ratio = float(max(column_change.max(), row_change.max()) / median_change)

    rgb_float = rgb.astype(np.float32)
    maximum = rgb_float.max(axis=2)
    minimum = rgb_float.min(axis=2)
    saturation = np.zeros_like(maximum, dtype=np.float32)
    np.divide(maximum - minimum, maximum, out=saturation, where=maximum > 0)
    green_fraction = float(
        np.mean(
            (rgb_float[:, :, 1] > rgb_float[:, :, 0] * 1.05)
            & (rgb_float[:, :, 1] > rgb_float[:, :, 2] * 1.05)
        )
    )

    metrics = {
        "aspect_ratio": float(width / height),
        "megapixels": float(width * height / 1_000_000.0),
        "brightness_mean": float(gray.mean()),
        "contrast_std": float(gray.std()),
        "entropy_bits": _entropy(gray),
        "blur_laplacian_variance": float(laplacian.var()),
        "border_to_center_edge_ratio": float(border_edge / (center_edge + 1e-6)),
        "seam_ratio": seam_ratio,
        "saturation_mean": float(saturation.mean()),
        "green_fraction": green_fraction,
    }
    flags: list[str] = []
    if min(width, height) < 224:
        flags.append("low_resolution")
    if min(width, height) < 128:
        flags.append("very_low_resolution")
    if metrics["aspect_ratio"] < 0.5 or metrics["aspect_ratio"] > 2.0:
        flags.append("extreme_aspect_ratio")
    if metrics["brightness_mean"] < 30:
        flags.append("very_dark")
    if metrics["brightness_mean"] > 225:
        flags.append("very_bright")
    if metrics["contrast_std"] < 20:
        flags.append("low_contrast")
    if metrics["entropy_bits"] < 3.5:
        flags.append("low_entropy")
    if metrics["blur_laplacian_variance"] < 35:
        flags.append("possible_blur")
    if metrics["border_to_center_edge_ratio"] > 2.5 and border_edge > 18:
        flags.append("possible_border_overlay")
    if metrics["seam_ratio"] > 7.0:
        flags.append("possible_collage_seam")
    if metrics["green_fraction"] < 0.02:
        flags.append("low_green_fraction_review")
    return metrics, tuple(sorted(flags))


def _inventory(source_root: Path) -> tuple[list[ImageItem], list[dict[str, str]]]:
    candidates = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    )
    if not candidates:
        raise ValueError(f"no images found under {source_root}")

    items: list[ImageItem] = []
    unreadable: list[dict[str, str]] = []
    for source_index, path in enumerate(candidates):
        relative = path.relative_to(source_root).as_posix()
        if len(path.relative_to(source_root).parts) != 2:
            unreadable.append({"source_relative_path": relative, "error": "unexpected_path_depth"})
            continue
        payload = path.read_bytes()
        try:
            with Image.open(path) as image:
                image.load()
                original_mode = image.mode
                transposed = ImageOps.exif_transpose(image).convert("RGB")
                rgb = np.asarray(transposed, dtype=np.uint8)
        except Exception as error:
            unreadable.append(
                {"source_relative_path": relative, "error": f"{type(error).__name__}: {error}"}
            )
            continue
        height, width = rgb.shape[:2]
        gray = np.asarray(Image.fromarray(rgb, mode="RGB").convert("L"), dtype=np.uint8)
        rgb_thumbnail = np.asarray(
            Image.fromarray(rgb, mode="RGB").resize((64, 64), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        gray_thumbnail = np.asarray(
            Image.fromarray(gray, mode="L").resize((64, 64), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        quality, flags = _quality_metrics(rgb, gray)
        pixel_digest = hashlib.sha256()
        pixel_digest.update(f"{width}x{height}:RGB\n".encode("ascii"))
        pixel_digest.update(rgb.tobytes())
        items.append(
            ImageItem(
                index=len(items),
                relative_path=relative,
                label=path.parent.name,
                source_engine=parse_source_engine(path.name),
                file_sha256=hashlib.sha256(payload).hexdigest(),
                pixel_sha256=pixel_digest.hexdigest(),
                byte_size=len(payload),
                width=width,
                height=height,
                mode=original_mode,
                phash=_perceptual_hash(gray),
                rgb_thumbnail=rgb_thumbnail,
                gray_thumbnail=gray_thumbnail,
                quality=quality,
                quality_flags=flags,
            )
        )
        if len(items) % 500 == 0:
            print(f"Inventoried {len(items)}/{len(candidates)} images", flush=True)
    return items, unreadable


def _group_edges(items: list[ImageItem]) -> list[PairEdge]:
    pair_evidence: dict[tuple[int, int], set[str]] = defaultdict(set)
    for attribute, evidence in (
        ("file_sha256", "identical_file_sha256"),
        ("pixel_sha256", "identical_decoded_pixels"),
    ):
        groups: dict[str, list[int]] = defaultdict(list)
        for item in items:
            groups[getattr(item, attribute)].append(item.index)
        for indices in groups.values():
            ordered = sorted(indices)
            for right in ordered[1:]:
                pair_evidence[(ordered[0], right)].add(evidence)
    return [
        PairEdge(left, right, "exact", "|".join(sorted(evidence)))
        for (left, right), evidence in sorted(pair_evidence.items())
    ]


class _BKNode:
    def __init__(self, value: int, index: int) -> None:
        self.value = value
        self.indices = [index]
        self.children: dict[int, _BKNode] = {}

    def add(self, value: int, index: int) -> None:
        node = self
        while True:
            distance = (node.value ^ value).bit_count()
            if distance == 0:
                node.indices.append(index)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(value, index)
                return
            node = child

    def query(self, value: int, radius: int) -> list[int]:
        results: list[int] = []
        stack = [self]
        while stack:
            node = stack.pop()
            distance = (node.value ^ value).bit_count()
            if distance <= radius:
                results.extend(node.indices)
            lower, upper = distance - radius, distance + radius
            stack.extend(child for edge, child in node.children.items() if lower <= edge <= upper)
        return results


def _pair_metrics(left: ImageItem, right: ImageItem) -> tuple[int, float, float, float]:
    hamming = (left.phash ^ right.phash).bit_count()
    rgb_mad = float(np.mean(np.abs(left.rgb_thumbnail - right.rgb_thumbnail)))
    left_gray = left.gray_thumbnail.ravel()
    right_gray = right.gray_thumbnail.ravel()
    left_centered = left_gray - left_gray.mean()
    right_centered = right_gray - right_gray.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    correlation = float(np.dot(left_centered, right_centered) / denominator) if denominator else 0.0
    left_aspect = left.width / left.height
    right_aspect = right.width / right.height
    aspect_delta = float(abs(left_aspect - right_aspect) / max(left_aspect, right_aspect))
    return hamming, rgb_mad, correlation, aspect_delta


def _near_edges(
    items: list[ImageItem], exact_edges: list[PairEdge], config: AuditConfig
) -> tuple[list[PairEdge], list[PairEdge]]:
    if not config.enable_near_duplicates or not items:
        return [], []
    exact_pairs = {(edge.left, edge.right) for edge in exact_edges}
    exact_pairs |= {(right, left) for left, right in exact_pairs}
    root = _BKNode(items[0].phash, items[0].index)
    confirmed: list[PairEdge] = []
    probable: list[PairEdge] = []
    for item in items[1:]:
        for left_index in sorted(set(root.query(item.phash, config.probable_phash_hamming))):
            if (left_index, item.index) in exact_pairs:
                continue
            left = items[left_index]
            hamming, mad, correlation, aspect_delta = _pair_metrics(left, item)
            edge = PairEdge(
                left.index,
                item.index,
                "near",
                "phash_rgb_gray_aspect",
                hamming,
                mad,
                correlation,
                aspect_delta,
            )
            is_confirmed = (
                hamming <= config.confirmed_phash_hamming
                and mad <= config.confirmed_rgb_mad
                and correlation >= config.confirmed_gray_correlation
                and aspect_delta <= config.confirmed_aspect_relative_delta
            )
            is_probable = (
                hamming <= config.probable_phash_hamming
                and mad <= config.probable_rgb_mad
                and correlation >= config.probable_gray_correlation
                and aspect_delta <= config.probable_aspect_relative_delta
            )
            if is_confirmed:
                confirmed.append(edge)
            elif is_probable:
                probable.append(edge)
        root.add(item.phash, item.index)
    key = lambda edge: (edge.left, edge.right)
    return sorted(confirmed, key=key), sorted(probable, key=key)


def _components(item_count: int, edges: list[PairEdge]) -> list[tuple[int, ...]]:
    parent = list(range(item_count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for edge in edges:
        union(edge.left, edge.right)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(item_count):
        groups[find(index)].append(index)
    return sorted((tuple(values) for values in groups.values()), key=lambda values: values[0])


def _component_id(members: list[ImageItem]) -> str:
    payload = "\n".join(sorted(item.relative_path for item in members))
    return "cmp_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _split_counts(total: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    if len(fractions) != 3 or any(value < 0 for value in fractions):
        raise ValueError("split fractions must contain three non-negative values")
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        raise ValueError("split fractions must sum to one")
    raw = [total * value for value in fractions]
    counts = [math.floor(value) for value in raw]
    order = sorted(range(3), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    if total >= 3:
        for required in (1, 2):
            if counts[required] == 0:
                donor = max(range(3), key=lambda index: (counts[index], -index))
                counts[donor] -= 1
                counts[required] += 1
    return tuple(counts)  # type: ignore[return-value]


def _assign_splits(
    canonical: list[tuple[str, ImageItem]], config: AuditConfig
) -> dict[str, str]:
    by_label: dict[str, list[tuple[str, ImageItem]]] = defaultdict(list)
    for component_id, item in canonical:
        by_label[item.label].append((component_id, item))
    assignments: dict[str, str] = {}
    for label in sorted(by_label):
        values = sorted(by_label[label], key=lambda value: value[0])
        class_seed = int.from_bytes(
            hashlib.sha256(f"{config.seed}:{label}".encode("utf-8")).digest()[:8], "big"
        )
        random.Random(class_seed).shuffle(values)
        counts = _split_counts(len(values), config.split_fractions)
        offset = 0
        for split, count in zip(("train", "validation", "test"), counts):
            for component_id, _ in values[offset : offset + count]:
                assignments[component_id] = split
            offset += count
    return assignments


def _edge_rows(edges: list[PairEdge], items: list[ImageItem]) -> list[dict[str, Any]]:
    rows = []
    for edge in edges:
        left, right = items[edge.left], items[edge.right]
        rows.append(
            {
                "left_index": edge.left,
                "right_index": edge.right,
                "left_path": left.relative_path,
                "right_path": right.relative_path,
                "left_label": left.label,
                "right_label": right.label,
                "same_label": left.label == right.label,
                "kind": edge.kind,
                "evidence": edge.evidence,
                "phash_hamming": edge.hamming,
                "rgb_mad": f"{edge.rgb_mad:.8f}",
                "gray_correlation": f"{edge.gray_correlation:.8f}",
                "aspect_relative_delta": f"{edge.aspect_relative_delta:.8f}",
            }
        )
    return rows


def _inventory_rows(items: list[ImageItem]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        row: dict[str, Any] = {
            "source_index": item.index,
            "source_relative_path": item.relative_path,
            "label": item.label,
            "source_engine": item.source_engine,
            "file_sha256": item.file_sha256,
            "pixel_sha256": item.pixel_sha256,
            "phash_hex": f"{item.phash:016x}",
            "width": item.width,
            "height": item.height,
            "mode": item.mode,
            "byte_size": item.byte_size,
            "quality_flags": "|".join(item.quality_flags),
        }
        row.update({key: f"{value:.8f}" for key, value in item.quality.items()})
        rows.append(row)
    return rows


def _write_contact_sheets(
    source_root: Path, output_root: Path, queue_rows: list[dict[str, Any]], limit: int = 300
) -> int:
    selected = queue_rows[:limit]
    if not selected:
        return 0
    cell_width, cell_height, columns, rows_per_sheet = 220, 190, 5, 5
    font = ImageFont.load_default()
    sheets = 0
    for start in range(0, len(selected), columns * rows_per_sheet):
        batch = selected[start : start + columns * rows_per_sheet]
        canvas = Image.new("RGB", (columns * cell_width, rows_per_sheet * cell_height), "white")
        draw = ImageDraw.Draw(canvas)
        for position, row in enumerate(batch):
            x = (position % columns) * cell_width
            y = (position // columns) * cell_height
            path = source_root / row["source_relative_path"]
            with Image.open(path) as image:
                image.load()
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((cell_width - 10, 130), Image.Resampling.LANCZOS)
                canvas.paste(image, (x + (cell_width - image.width) // 2, y + 2))
            label = str(row["label"])[:30]
            reason = str(row["review_reason"])[:32]
            draw.text((x + 4, y + 136), f"{start + position:04d} {label}", fill="black", font=font)
            draw.text((x + 4, y + 151), reason, fill="black", font=font)
            draw.text((x + 4, y + 166), Path(row["source_relative_path"]).name[:30], fill="black", font=font)
        sheet_path = output_root / "review" / f"contact_sheet_{sheets + 1:02d}.jpg"
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(sheet_path, quality=90)
        sheets += 1
    return sheets


def _write_duplicate_pair_sheets(
    source_root: Path,
    output_root: Path,
    pair_rows: list[dict[str, Any]],
) -> int:
    if not pair_rows:
        return 0
    pair_width, pair_height, columns, rows_per_sheet = 560, 220, 2, 4
    image_width, image_height = 260, 150
    font = ImageFont.load_default()
    sheets = 0
    for start in range(0, len(pair_rows), columns * rows_per_sheet):
        batch = pair_rows[start : start + columns * rows_per_sheet]
        canvas = Image.new(
            "RGB",
            (columns * pair_width, rows_per_sheet * pair_height),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for position, row in enumerate(batch):
            x = (position % columns) * pair_width
            y = (position // columns) * pair_height
            for side_index, side in enumerate(("left", "right")):
                path = source_root / str(row[f"{side}_path"])
                with Image.open(path) as image:
                    image.load()
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    image.thumbnail((image_width, image_height), Image.Resampling.LANCZOS)
                    image_x = x + side_index * (pair_width // 2)
                    image_x += ((pair_width // 2) - image.width) // 2
                    canvas.paste(image, (image_x, y + 2))
                label = str(row[f"{side}_label"])[:38]
                filename = Path(str(row[f"{side}_path"])).name[:38]
                text_x = x + side_index * (pair_width // 2) + 4
                draw.text((text_x, y + 154), label, fill="black", font=font)
                draw.text((text_x, y + 169), filename, fill="black", font=font)
            evidence = str(row["evidence"])[:70]
            metrics = (
                f"{row['kind']} h={row['phash_hamming']} "
                f"mad={row['rgb_mad']} corr={row['gray_correlation']}"
            )
            draw.text((x + 4, y + 187), evidence, fill="black", font=font)
            draw.text((x + 4, y + 202), metrics, fill="black", font=font)
        sheet_path = (
            output_root
            / "review"
            / f"cross_label_duplicate_sheet_{sheets + 1:02d}.jpg"
        )
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(sheet_path, quality=90)
        sheets += 1
    return sheets


def _tree_inventory_digest(items: list[ImageItem]) -> str:
    digest = hashlib.sha256()
    for item in sorted(items, key=lambda value: value.relative_path):
        digest.update(item.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.file_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate(
    config: AuditConfig,
    items: list[ImageItem],
    manifest_rows: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    by_component: dict[str, set[str]] = defaultdict(set)
    split_by_component: dict[str, set[str]] = defaultdict(set)
    split_by_label: dict[str, set[str]] = defaultdict(set)
    item_by_path = {item.relative_path: item for item in items}
    for row in manifest_rows:
        relative = row["source_relative_path"]
        source = config.source_root / relative
        if not source.is_file():
            errors.append(f"missing manifest source: {relative}")
            continue
        item = item_by_path.get(relative)
        if item is None or item.file_sha256 != row["file_sha256"]:
            errors.append(f"manifest hash mismatch: {relative}")
        by_component[row["component_id"]].add(row["label"])
        split_by_component[row["component_id"]].add(row["split"])
        split_by_label[row["label"]].add(row["split"])
    for component_id, labels in by_component.items():
        if len(labels) != 1:
            errors.append(f"retained component has multiple labels: {component_id}")
    for component_id, splits in split_by_component.items():
        if len(splits) != 1:
            errors.append(f"component crosses splits: {component_id}")
    required_splits = {"train", "validation", "test"}
    for label, splits in split_by_label.items():
        if splits != required_splits:
            errors.append(f"class missing split coverage: {label}: {sorted(splits)}")
    conflict_ids = {row["component_id"] for row in conflicts}
    retained_ids = set(by_component)
    overlap = sorted(conflict_ids & retained_ids)
    if overlap:
        errors.append(f"conflicting components retained: {overlap[:5]}")
    sample_ids = [row["sample_id"] for row in manifest_rows]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("duplicate sample IDs")
    return {
        "ok": not errors,
        "errors": errors,
        "manifest_images": len(manifest_rows),
        "manifest_classes": len(split_by_label),
        "manifest_components": len(by_component),
        "split_counts": dict(sorted(Counter(row["split"] for row in manifest_rows).items())),
    }


def build_audit(config: AuditConfig) -> dict[str, Any]:
    source_root = Path(config.source_root).resolve()
    output_root = Path(config.output_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty audit directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    items, unreadable = _inventory(source_root)
    source_digest = _tree_inventory_digest(items)
    exact_edges = _group_edges(items)
    confirmed_near, probable_near = _near_edges(items, exact_edges, config)
    components = _components(len(items), exact_edges + confirmed_near)

    canonical_before_support: list[tuple[str, ImageItem]] = []
    memberships: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    component_sizes: list[int] = []
    for indices in components:
        members = [items[index] for index in indices]
        component_id = _component_id(members)
        labels = sorted({item.label for item in members})
        component_sizes.append(len(members))
        if len(labels) > 1:
            conflicts.append(
                {
                    "component_id": component_id,
                    "labels": "|".join(labels),
                    "member_count": len(members),
                    "source_relative_paths": "|".join(sorted(item.relative_path for item in members)),
                    "reason": "conflicting_labels",
                }
            )
            for item in members:
                memberships.append(
                    {
                        "component_id": component_id,
                        "source_index": item.index,
                        "source_relative_path": item.relative_path,
                        "label": item.label,
                        "canonical": False,
                        "retained": False,
                        "reason": "conflicting_labels",
                    }
                )
            continue
        canonical = min(
            members,
            key=lambda item: (-item.pixel_count, -item.byte_size, item.relative_path),
        )
        canonical_before_support.append((component_id, canonical))
        for item in members:
            is_canonical = item.index == canonical.index
            memberships.append(
                {
                    "component_id": component_id,
                    "source_index": item.index,
                    "source_relative_path": item.relative_path,
                    "label": item.label,
                    "canonical": is_canonical,
                    "retained": is_canonical,
                    "reason": "canonical" if is_canonical else "duplicate_of_canonical",
                }
            )

    support_before = Counter(item.label for _, item in canonical_before_support)
    excluded_classes = {
        label: count
        for label, count in sorted(support_before.items())
        if count < config.minimum_components_per_class
    }
    canonical = [
        value for value in canonical_before_support if value[1].label not in excluded_classes
    ]
    assignments = _assign_splits(canonical, config)
    class_to_id = {label: index for index, label in enumerate(sorted({item.label for _, item in canonical}))}

    manifest_rows: list[dict[str, Any]] = []
    for sample_index, (component_id, item) in enumerate(
        sorted(canonical, key=lambda value: (value[1].label, value[0]))
    ):
        manifest_rows.append(
            {
                "sample_id": f"plantwild_{sample_index:06d}",
                "component_id": component_id,
                "split": assignments[component_id],
                "label": item.label,
                "label_id": class_to_id[item.label],
                "source_relative_path": item.relative_path,
                "source_engine": item.source_engine,
                "file_sha256": item.file_sha256,
                "pixel_sha256": item.pixel_sha256,
                "width": item.width,
                "height": item.height,
                "byte_size": item.byte_size,
                "quality_flags": "|".join(item.quality_flags),
            }
        )

    inventory_rows = _inventory_rows(items)
    inventory_fields = list(inventory_rows[0]) if inventory_rows else []
    edge_fields = [
        "left_index",
        "right_index",
        "left_path",
        "right_path",
        "left_label",
        "right_label",
        "same_label",
        "kind",
        "evidence",
        "phash_hamming",
        "rgb_mad",
        "gray_correlation",
        "aspect_relative_delta",
    ]
    _write_csv(output_root / "audit" / "source_inventory.csv", inventory_fields, inventory_rows)
    _write_csv(
        output_root / "audit" / "unreadable_images.csv",
        ["source_relative_path", "error"],
        unreadable,
    )
    exact_rows = _edge_rows(exact_edges, items)
    confirmed_near_rows = _edge_rows(confirmed_near, items)
    probable_near_rows = _edge_rows(probable_near, items)
    _write_csv(
        output_root / "audit" / "exact_duplicate_pairs.csv",
        edge_fields,
        exact_rows,
    )
    _write_csv(
        output_root / "audit" / "confirmed_near_duplicate_pairs.csv",
        edge_fields,
        confirmed_near_rows,
    )
    _write_csv(
        output_root / "audit" / "probable_near_duplicate_review.csv",
        edge_fields,
        probable_near_rows,
    )
    cross_label_rows = []
    for row in exact_rows + confirmed_near_rows:
        if row["same_label"]:
            continue
        cross_label_rows.append(
            {
                **row,
                "exclusion_decision": "exclude_entire_conflicting_component",
                "review_status": "automatic_integrity_exclusion_pending_expert_relabeling",
            }
        )
    cross_label_fields = edge_fields + ["exclusion_decision", "review_status"]
    _write_csv(
        output_root / "review" / "cross_label_duplicate_pairs.csv",
        cross_label_fields,
        cross_label_rows,
    )
    duplicate_pair_sheet_count = _write_duplicate_pair_sheets(
        source_root,
        output_root,
        cross_label_rows,
    )
    _write_csv(
        output_root / "audit" / "duplicate_component_membership.csv",
        ["component_id", "source_index", "source_relative_path", "label", "canonical", "retained", "reason"],
        sorted(memberships, key=lambda row: (row["component_id"], row["source_index"])),
    )
    _write_csv(
        output_root / "audit" / "excluded_conflicting_components.csv",
        ["component_id", "labels", "member_count", "source_relative_paths", "reason"],
        sorted(conflicts, key=lambda row: row["component_id"]),
    )
    _write_csv(
        output_root / "audit" / "excluded_under_supported_classes.csv",
        ["label", "independent_components", "minimum_required"],
        [
            {
                "label": label,
                "independent_components": count,
                "minimum_required": config.minimum_components_per_class,
            }
            for label, count in excluded_classes.items()
        ],
    )

    class_rows = []
    retained_counts = Counter(row["label"] for row in manifest_rows)
    source_counts = Counter(item.label for item in items)
    for label in sorted(source_counts):
        class_rows.append(
            {
                "label": label,
                "source_images": source_counts[label],
                "independent_components": support_before.get(label, 0),
                "retained_benchmark_images": retained_counts.get(label, 0),
                "benchmark_eligible": label in class_to_id,
            }
        )
    _write_csv(
        output_root / "audit" / "class_distribution.csv",
        ["label", "source_images", "independent_components", "retained_benchmark_images", "benchmark_eligible"],
        class_rows,
    )
    split_distribution = []
    split_counter = Counter((row["split"], row["label"]) for row in manifest_rows)
    for split in ("train", "validation", "test"):
        for label in sorted(class_to_id):
            split_distribution.append(
                {"split": split, "label": label, "count": split_counter[(split, label)]}
            )
    _write_csv(
        output_root / "audit" / "split_distribution.csv",
        ["split", "label", "count"],
        split_distribution,
    )
    engine_rows = [
        {"source_engine": engine, "count": count}
        for engine, count in sorted(Counter(item.source_engine for item in items).items())
    ]
    _write_csv(
        output_root / "audit" / "source_engine_distribution.csv",
        ["source_engine", "count"],
        engine_rows,
    )

    flagged = [item for item in items if item.quality_flags]
    _write_csv(
        output_root / "audit" / "quality_flags.csv",
        inventory_fields,
        _inventory_rows(flagged),
    )
    queue_by_path: dict[str, dict[str, Any]] = {}
    for item in sorted(flagged, key=lambda value: (-len(value.quality_flags), value.relative_path)):
        queue_by_path[item.relative_path] = {
            "source_relative_path": item.relative_path,
            "label": item.label,
            "review_reason": "|".join(item.quality_flags),
            "review_status": "pending_human_or_expert_review",
        }
    for label in sorted(source_counts):
        representatives = sorted((item for item in items if item.label == label), key=lambda item: item.file_sha256)
        if representatives:
            item = representatives[0]
            existing = queue_by_path.get(item.relative_path)
            reason = "deterministic_class_representative"
            if existing:
                existing["review_reason"] += "|" + reason
            else:
                queue_by_path[item.relative_path] = {
                    "source_relative_path": item.relative_path,
                    "label": item.label,
                    "review_reason": reason,
                    "review_status": "pending_human_or_expert_review",
                }
    review_rows = sorted(
        queue_by_path.values(),
        key=lambda row: (
            0 if "possible_collage_seam" in row["review_reason"] else 1,
            0 if "possible_border_overlay" in row["review_reason"] else 1,
            row["label"],
            row["source_relative_path"],
        ),
    )
    _write_csv(
        output_root / "review" / "manual_review_queue.csv",
        ["source_relative_path", "label", "review_reason", "review_status"],
        review_rows,
    )
    contact_sheet_count = _write_contact_sheets(source_root, output_root, review_rows)

    for split in ("all", "train", "validation", "test"):
        rows = manifest_rows if split == "all" else [row for row in manifest_rows if row["split"] == split]
        _write_csv(output_root / "manifests" / f"{split}.csv", MANIFEST_FIELDS, rows)
    _write_json(output_root / "manifests" / "class_mapping.json", class_to_id)

    archive_sha256 = None
    if config.source_archive is not None:
        archive_path = Path(config.source_archive).resolve()
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        archive_sha256 = _sha256_file(archive_path)
    _write_json(
        output_root / "audit" / "metadata_recovery.json",
        {
            "local_archive_contains_fixed_split": False,
            "local_archive_contains_metadata": False,
            "local_archive_contains_license_file": False,
            "label_provenance": "immediate_parent_directory",
            "filename_source_engine_is_original_source_id": False,
            "official_project_url": "https://tqwei05.github.io/PlantWild/",
            "official_baseline_repository": "https://github.com/tqwei05/MVPDR",
            "official_baseline_commit_inspected": "fe7082d486d65ca1eac8b1a5eae1c9af87b3280b",
            "official_loader_declared_fractions": {"train": 0.70, "validation": 0.10, "test": 0.20},
            "official_loader_reproducibility_issue": "seeded sampling is applied to unsorted os.listdir results",
            "official_training_validity_issue": "main.py selects the best epoch using test features",
            "audit_split_adopted": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "audit_split_reason": "fixed duplicate-aware development protocol with validation separated from sealed test",
        },
    )
    _write_json(
        output_root / "audit" / "support_policy.json",
        {
            "minimum_independent_components_per_class": config.minimum_components_per_class,
            "policy_basis": "predeclared support-only rule applied before model training",
            "excluded_classes": excluded_classes,
            "source_taxonomy_preserved_in_inventory": True,
            "excluded_source_images_deleted": False,
        },
    )
    _write_json(
        output_root / "manifests" / "access_policy.json",
        {
            "train": "model fitting allowed",
            "validation": "checkpoint selection and development comparisons allowed",
            "test": "sealed until architecture, recipe, and hyperparameters are frozen",
            "test_accessed_during_audit": False,
        },
    )

    summary = {
        "source_images": len(items),
        "source_classes": len(source_counts),
        "unreadable_images": len(unreadable),
        "file_exact_duplicate_edges": sum(
            "identical_file_sha256" in edge.evidence for edge in exact_edges
        ),
        "decoded_pixel_duplicate_edges": sum(
            "identical_decoded_pixels" in edge.evidence for edge in exact_edges
        ),
        "exact_duplicate_edges_union": len(exact_edges),
        "confirmed_near_duplicate_edges": len(confirmed_near),
        "probable_near_duplicate_pairs_for_review": len(probable_near),
        "duplicate_components": sum(size > 1 for size in component_sizes),
        "component_count": len(components),
        "excluded_conflicting_components": len(conflicts),
        "excluded_conflicting_images": sum(row["member_count"] for row in conflicts),
        "canonical_images_before_support_filter": len(canonical_before_support),
        "excluded_under_supported_classes": len(excluded_classes),
        "excluded_under_supported_images": sum(excluded_classes.values()),
        "retained_images": len(manifest_rows),
        "retained_classes": len(class_to_id),
        "quality_flagged_images": len(flagged),
        "manual_review_queue_images": len(review_rows),
        "contact_sheets": contact_sheet_count,
        "cross_label_duplicate_edges": len(cross_label_rows),
        "cross_label_duplicate_contact_sheets": duplicate_pair_sheet_count,
        "seed": config.seed,
        "fractions": {
            "train": config.split_fractions[0],
            "validation": config.split_fractions[1],
            "test": config.split_fractions[2],
        },
        "split_counts": dict(sorted(Counter(row["split"] for row in manifest_rows).items())),
        "minimum_components_per_class": config.minimum_components_per_class,
        "source_tree_inventory_sha256": source_digest,
        "source_archive_sha256": archive_sha256,
        "near_duplicate_thresholds": {
            "confirmed_phash_hamming": config.confirmed_phash_hamming,
            "confirmed_rgb_mad": config.confirmed_rgb_mad,
            "confirmed_gray_correlation": config.confirmed_gray_correlation,
            "confirmed_aspect_relative_delta": config.confirmed_aspect_relative_delta,
            "probable_phash_hamming": config.probable_phash_hamming,
            "probable_rgb_mad": config.probable_rgb_mad,
            "probable_gray_correlation": config.probable_gray_correlation,
            "probable_aspect_relative_delta": config.probable_aspect_relative_delta,
        },
    }
    _write_json(output_root / "audit" / "audit_summary.json", summary)
    validation = _validate(config, items, manifest_rows, conflicts)
    validation["source_tree_unchanged"] = _tree_inventory_digest(items) == source_digest
    validation["source_archive_sha256"] = archive_sha256
    if not validation["source_tree_unchanged"]:
        validation["errors"].append("source inventory changed during audit")
        validation["ok"] = False
    _write_json(output_root / "audit" / "validation.json", validation)
    if not validation["ok"]:
        raise ValueError(f"audit validation failed: {validation['errors'][:5]}")

    readme = f"""# PlantWild V2 Audit V1

This directory is a non-destructive audit of `{source_root}`. It does not copy,
rename, recompress, or delete source images.

## Result

- Source images/classes: {summary['source_images']} / {summary['source_classes']}
- Unreadable images: {summary['unreadable_images']}
- Exact/confirmed-near edges: {summary['exact_duplicate_edges_union']} / {summary['confirmed_near_duplicate_edges']}
- Conflicting components/images excluded: {summary['excluded_conflicting_components']} / {summary['excluded_conflicting_images']}
- Support-filtered classes/images: {summary['excluded_under_supported_classes']} / {summary['excluded_under_supported_images']}
- Benchmark images/classes: {summary['retained_images']} / {summary['retained_classes']}
- Split counts: {summary['split_counts']}
- Manual semantic/quality review queue: {summary['manual_review_queue_images']} images
- Cross-label duplicate edges/review sheets: {summary['cross_label_duplicate_edges']} / {summary['cross_label_duplicate_contact_sheets']}

## Interpretation

Exact and conservatively confirmed near-copy components cannot cross splits.
Every multi-label component is excluded. Quality and semantic heuristics are
review flags only; they do not alter the benchmark. The test manifest is sealed
by policy until all model-development decisions are frozen.

The local release has no fixed split, original URLs, or embedded license file.
See `audit/metadata_recovery.json` and `audit/validation.json`. Human or domain-
expert review of `review/manual_review_queue.csv` remains required before any
semantic relabeling or image exclusion.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")

    checksum_lines = []
    for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
        if path.name == "artifact_checksums.sha256":
            continue
        checksum_lines.append(f"{_sha256_file(path)}  {path.relative_to(output_root).as_posix()}")
    (output_root / "artifact_checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--minimum-components-per-class", type=int, default=20)
    parser.add_argument("--disable-near-duplicates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_audit(
        AuditConfig(
            source_root=args.source_root,
            output_root=args.output_root,
            source_archive=args.source_archive,
            minimum_components_per_class=args.minimum_components_per_class,
            enable_near_duplicates=not args.disable_near_duplicates,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
