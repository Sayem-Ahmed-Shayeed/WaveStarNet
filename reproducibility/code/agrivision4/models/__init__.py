"""Canonical model implementations used by controlled experiments."""

from .wavestarnet_family import (
    MODEL_REGISTRY,
    architecture_fingerprint,
    architecture_manifest,
    backbone_schema_fingerprint,
    create_wavestarnet,
)

__all__ = [
    "MODEL_REGISTRY",
    "architecture_fingerprint",
    "architecture_manifest",
    "backbone_schema_fingerprint",
    "create_wavestarnet",
]
