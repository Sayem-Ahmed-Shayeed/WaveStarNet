"""Deterministic duplicate resolution and splitting for AgriVision 4."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class ImageRecord:
    index: int
    archive_path: str
    label: str
    crop: str
    sha256: str
    md5: str
    byte_size: int
    width: int
    height: int
    perceptual_hash: int
    rgb_thumbnail: np.ndarray = field(repr=False, compare=False)

    @property
    def pixel_count(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class DuplicateEdge:
    left: int
    right: int
    kind: Literal["exact", "near"]
    hamming: int = 0
    rgb_mad: float = 0.0


@dataclass(frozen=True)
class RetainedImage:
    component_id: str
    record: ImageRecord


@dataclass(frozen=True)
class ComponentExclusion:
    component_id: str
    member_indices: tuple[int, ...]
    labels: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class ComponentMembership:
    component_id: str
    record_index: int
    canonical: bool
    retained: bool
    reason: str


def normalized_rgb_thumbnail(image: Image.Image, size: int = 32) -> np.ndarray:
    transposed = ImageOps.exif_transpose(image).convert("RGB")
    resized = transposed.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def perceptual_signature(image: Image.Image, size: int = 32) -> int:
    transposed = ImageOps.exif_transpose(image).convert("L")
    values = np.asarray(
        transposed.resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    threshold = float(np.median(values))
    signature = 0
    for enabled in (values >= threshold).ravel():
        signature = (signature << 1) | int(enabled)
    return signature


def exact_duplicate_edges(records: list[ImageRecord]) -> list[DuplicateEdge]:
    groups: dict[str, list[int]] = defaultdict(list)
    for record in records:
        groups[record.sha256].append(record.index)

    edges: list[DuplicateEdge] = []
    for indices in groups.values():
        ordered = sorted(indices)
        for right in ordered[1:]:
            edges.append(DuplicateEdge(ordered[0], right, "exact"))
    return sorted(edges, key=lambda edge: (edge.left, edge.right))


def near_duplicate_edges(
    records: list[ImageRecord],
    max_hamming: int = 4,
    max_mad: float = 0.001,
) -> list[DuplicateEdge]:
    edges: list[DuplicateEdge] = []
    for left_position, left in enumerate(records):
        for right_position in range(left_position + 1, len(records)):
            right = records[right_position]
            if left.sha256 == right.sha256:
                continue
            hamming = (left.perceptual_hash ^ right.perceptual_hash).bit_count()
            if hamming > max_hamming:
                continue
            rgb_mad = float(np.mean(np.abs(left.rgb_thumbnail - right.rgb_thumbnail)))
            if rgb_mad <= max_mad:
                edges.append(
                    DuplicateEdge(
                        left.index,
                        right.index,
                        "near",
                        hamming=hamming,
                        rgb_mad=rgb_mad,
                    )
                )
    return sorted(edges, key=lambda edge: (edge.left, edge.right))


def connected_components(
    records: list[ImageRecord], edges: list[DuplicateEdge]
) -> list[tuple[int, ...]]:
    record_indices = {record.index for record in records}
    parent = {index: index for index in record_indices}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        if left not in parent or right not in parent:
            raise ValueError("duplicate edge references an unknown record")
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for edge in edges:
        union(edge.left, edge.right)

    components: dict[int, list[int]] = defaultdict(list)
    for index in sorted(record_indices):
        components[find(index)].append(index)
    return sorted((tuple(indices) for indices in components.values()), key=lambda x: x[0])


def _component_id(records: list[ImageRecord]) -> str:
    payload = "\n".join(sorted(record.archive_path for record in records))
    return "cmp_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def resolve_components(
    records: list[ImageRecord],
    components: list[tuple[int, ...]],
) -> tuple[list[RetainedImage], list[ComponentExclusion], list[ComponentMembership]]:
    by_index = {record.index: record for record in records}
    retained: list[RetainedImage] = []
    exclusions: list[ComponentExclusion] = []
    memberships: list[ComponentMembership] = []

    seen: set[int] = set()
    for member_indices in components:
        if not member_indices:
            raise ValueError("empty duplicate component")
        if seen.intersection(member_indices):
            raise ValueError("record appears in multiple duplicate components")
        seen.update(member_indices)
        members = [by_index[index] for index in member_indices]
        component_id = _component_id(members)
        labels = tuple(sorted({record.label for record in members}))

        if len(labels) != 1:
            reason = "conflicting_labels"
            exclusions.append(
                ComponentExclusion(component_id, tuple(member_indices), labels, reason)
            )
            for record in members:
                memberships.append(
                    ComponentMembership(component_id, record.index, False, False, reason)
                )
            continue

        canonical = min(
            members,
            key=lambda record: (
                -record.pixel_count,
                -record.byte_size,
                record.archive_path,
            ),
        )
        retained.append(RetainedImage(component_id, canonical))
        for record in members:
            is_canonical = record.index == canonical.index
            memberships.append(
                ComponentMembership(
                    component_id,
                    record.index,
                    is_canonical,
                    is_canonical,
                    "canonical" if is_canonical else "duplicate_of_canonical",
                )
            )

    missing = set(by_index) - seen
    if missing:
        raise ValueError(f"records missing from components: {sorted(missing)}")

    return (
        sorted(retained, key=lambda item: item.record.archive_path),
        sorted(exclusions, key=lambda item: item.component_id),
        sorted(memberships, key=lambda item: (item.component_id, item.record_index)),
    )


def _split_counts(
    total: int, fractions: tuple[float, float, float]
) -> tuple[int, int, int]:
    if total < 0:
        raise ValueError("class support cannot be negative")
    if len(fractions) != 3 or any(value < 0 for value in fractions):
        raise ValueError("split fractions must contain three non-negative values")
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        raise ValueError("split fractions must sum to one")

    raw = [total * value for value in fractions]
    counts = [math.floor(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(3), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1

    if total >= 3:
        for required_index in (1, 2):
            if counts[required_index] == 0:
                donor = max(range(3), key=lambda index: (counts[index], -index))
                if counts[donor] <= 1:
                    raise ValueError("cannot satisfy minimum split coverage")
                counts[donor] -= 1
                counts[required_index] += 1
    return counts[0], counts[1], counts[2]


def stratified_split(
    retained: list[RetainedImage],
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[str, list[RetainedImage]]:
    component_ids = [item.component_id for item in retained]
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("retained component IDs must be unique")

    by_label: dict[str, list[RetainedImage]] = defaultdict(list)
    for item in retained:
        by_label[item.record.label].append(item)

    split = {"train": [], "validation": [], "test": []}
    split_names = tuple(split)
    for label in sorted(by_label):
        values = sorted(by_label[label], key=lambda item: item.component_id)
        seed_payload = f"{seed}:{label}".encode("utf-8")
        class_seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
        random.Random(class_seed).shuffle(values)
        counts = _split_counts(len(values), fractions)
        offset = 0
        for name, count in zip(split_names, counts):
            split[name].extend(values[offset : offset + count])
            offset += count

    for name in split:
        split[name] = sorted(
            split[name], key=lambda item: (item.record.label, item.component_id)
        )

    assigned = [item.component_id for values in split.values() for item in values]
    if sorted(assigned) != sorted(component_ids):
        raise ValueError("split assignment is incomplete or overlapping")
    return split


def filter_supported_classes(
    retained: list[RetainedImage], minimum_components: int
) -> tuple[list[RetainedImage], dict[str, int]]:
    if minimum_components < 1:
        raise ValueError("minimum_components must be at least one")
    counts = Counter(item.record.label for item in retained)
    dropped = {
        label: count for label, count in sorted(counts.items()) if count < minimum_components
    }
    kept = [item for item in retained if item.record.label not in dropped]
    return kept, dropped
