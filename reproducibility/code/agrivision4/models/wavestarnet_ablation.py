"""Controlled WavestarNet V2 architecture variants for ablation studies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

try:
    from .wavestarnet_family import (
        TransferS3StarBlock,
        WaveDown,
        WaveletGate,
        WavestarNetBigTransferV2,
        architecture_fingerprint,
    )
except (ImportError, ModuleNotFoundError):
    # Generated notebooks define the canonical symbols in the preceding cell.
    pass


ABLATION_KEYS = (
    "full_v2",
    "no_learned_gates",
    "all_blocks_gated",
    "conventional_downsampling",
    "stage4_only_head",
    "se_attention",
)

CANONICAL_MASKS = ((1, 0), (1, 0, 0), (1, 0, 0, 0, 1), (1,))
ZERO_MASKS = ((0, 0), (0, 0, 0), (0, 0, 0, 0, 0), (0,))
ALL_MASKS = ((1, 1), (1, 1, 1), (1, 1, 1, 1, 1), (1,))


@dataclass(frozen=True)
class AblationDescriptor:
    key: str
    display_name: str
    masks: tuple[tuple[int, ...], ...]
    spectral_operator: str
    downsampling: str
    head: str


ABLATION_DESCRIPTORS = {
    "full_v2": AblationDescriptor(
        "full_v2",
        "WavestarNet V2 (full control)",
        CANONICAL_MASKS,
        "selective_wavelet_gate",
        "wave_down",
        "multistage_s2_s3_s4",
    ),
    "no_learned_gates": AblationDescriptor(
        "no_learned_gates",
        "WavestarNet V2 (no learned gates)",
        ZERO_MASKS,
        "fixed_half_scale_bypass",
        "wave_down",
        "multistage_s2_s3_s4",
    ),
    "all_blocks_gated": AblationDescriptor(
        "all_blocks_gated",
        "WavestarNet V2 (all blocks gated)",
        ALL_MASKS,
        "all_wavelet_gate",
        "wave_down",
        "multistage_s2_s3_s4",
    ),
    "conventional_downsampling": AblationDescriptor(
        "conventional_downsampling",
        "WavestarNet V2 (conventional downsampling)",
        CANONICAL_MASKS,
        "selective_wavelet_gate",
        "depthwise_pointwise_stride2",
        "multistage_s2_s3_s4",
    ),
    "stage4_only_head": AblationDescriptor(
        "stage4_only_head",
        "WavestarNet V2 (stage-4-only head)",
        CANONICAL_MASKS,
        "selective_wavelet_gate",
        "wave_down",
        "stage4_only",
    ),
    "se_attention": AblationDescriptor(
        "se_attention",
        "WavestarNet V2 (standard channel SE attention)",
        CANONICAL_MASKS,
        "selective_spatial_se",
        "wave_down",
        "multistage_s2_s3_s4",
    ),
}


class ChannelSE(nn.Module):
    """Standard channel squeeze-and-excitation with reduction ratio eight."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class ConventionalDownsample(nn.Module):
    """Efficient spatial downsampling control preserving stage dimensions."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            input_channels,
            input_channels,
            3,
            stride=2,
            padding=1,
            groups=input_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            input_channels,
            output_channels,
            1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class WavestarNetV2Ablation(nn.Module):
    WIDTHS = (48, 96, 192, 384)
    DEPTHS = (2, 3, 5, 1)

    def __init__(
        self,
        descriptor: AblationDescriptor,
        num_classes: int,
        dropout: float,
        max_drop_path: float,
    ) -> None:
        super().__init__()
        self.ablation_key = descriptor.key
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.Hardswish(inplace=True),
        )
        rates = torch.linspace(0, max_drop_path, sum(self.DEPTHS)).tolist()
        stages: list[nn.Module] = []
        input_channels = 24
        rate_index = 0
        for output_channels, depth, mask in zip(
            self.WIDTHS,
            self.DEPTHS,
            descriptor.masks,
        ):
            downsample: nn.Module
            if descriptor.downsampling == "depthwise_pointwise_stride2":
                downsample = ConventionalDownsample(
                    input_channels,
                    output_channels,
                )
            else:
                downsample = WaveDown(input_channels, output_channels)
            stage: list[nn.Module] = [downsample]
            for block_index in range(depth):
                use_gate = bool(mask[block_index])
                block = TransferS3StarBlock(
                    output_channels,
                    use_wavelet=(
                        use_gate
                        and descriptor.spectral_operator != "selective_spatial_se"
                    ),
                    drop_path=rates[rate_index],
                )
                if use_gate and descriptor.spectral_operator == "selective_spatial_se":
                    block.spec = ChannelSE(output_channels)
                stage.append(block)
                rate_index += 1
            stages.append(nn.Sequential(*stage))
            input_channels = output_channels
        self.stages = nn.Sequential(*stages)
        pooled_channels = (
            self.WIDTHS[-1]
            if descriptor.head == "stage4_only"
            else sum(self.WIDTHS[1:])
        )
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_channels),
            nn.Dropout(dropout),
            nn.Linear(pooled_channels, num_classes),
        )
        self.head_type = descriptor.head

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return tuple(features)

    def forward_head(self, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError("forward_head requires four stage features")
        selected = (
            (features[-1],)
            if self.head_type == "stage4_only"
            else features[1:]
        )
        pooled = torch.cat(
            tuple(feature.float().mean(dim=(-2, -1)) for feature in selected),
            dim=1,
        )
        pooled = self.head[0](pooled).to(features[-1].dtype)
        return self.head[2](self.head[1](pooled))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))

    def classifier_module(self) -> nn.Module:
        return self.head[2]


def _descriptor(ablation_key: str) -> AblationDescriptor:
    try:
        return ABLATION_DESCRIPTORS[ablation_key]
    except KeyError as error:
        raise ValueError(f"unknown ablation key: {ablation_key}") from error


def create_wavestarnet_v2_ablation(
    ablation_key: str,
    num_classes: int,
    *,
    dropout: float = 0.15,
    max_drop_path: float = 0.08,
) -> nn.Module:
    descriptor = _descriptor(ablation_key)
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    if descriptor.key == "full_v2":
        return WavestarNetBigTransferV2(
            num_classes,
            dropout,
            max_drop_path,
        )
    return WavestarNetV2Ablation(
        descriptor,
        num_classes,
        dropout,
        max_drop_path,
    )


def _ablation_state_schema(model: nn.Module) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in sorted(model.state_dict().items())
    ]


def _ablation_hash_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ablation_fingerprint(ablation_key: str, num_classes: int) -> str:
    descriptor = _descriptor(ablation_key)
    if descriptor.key == "full_v2":
        return architecture_fingerprint(
            "wavestarnet_big_transfer_v2",
            num_classes,
        )
    model = create_wavestarnet_v2_ablation(ablation_key, num_classes)
    return _ablation_hash_json(
        {
            "ablation": asdict(descriptor),
            "num_classes": num_classes,
            "parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "state_schema": _ablation_state_schema(model),
        }
    )


def ablation_model_audit(
    model: nn.Module,
    ablation_key: str,
) -> dict[str, Any]:
    descriptor = _descriptor(ablation_key)
    wavelet_locations = [
        name
        for name, module in model.named_modules()
        if isinstance(module, WaveletGate)
    ]
    se_locations = [
        name
        for name, module in model.named_modules()
        if isinstance(module, ChannelSE)
    ]
    downsampling_locations = [
        name
        for name, module in model.named_modules()
        if isinstance(module, (WaveDown, ConventionalDownsample))
    ]
    return {
        "ablation_key": descriptor.key,
        "display_name": descriptor.display_name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "wavelet_gate_count": len(wavelet_locations),
        "wavelet_gate_locations": wavelet_locations,
        "se_attention_count": len(se_locations),
        "se_attention_locations": se_locations,
        "downsampling_type": descriptor.downsampling,
        "downsampling_locations": downsampling_locations,
        "head_type": descriptor.head,
        "spectral_operator": descriptor.spectral_operator,
        "state_schema_sha256": _ablation_hash_json(
            _ablation_state_schema(model)
        ),
    }
