"""The two permitted PlantWild WavestarNet WaveDown variants."""

from __future__ import annotations

import torch
import torch.nn as nn

from agrivision4.models.wavestarnet_family import (
    TransferS3StarBlock,
    WaveDown,
    WavestarNetBigTransferV2,
)

from .experiment_core import ABLATION_KEYS


class CapacityMatchedConventionalDownsample(nn.Module):
    """Conventional spatial reduction matched within 1,100 parameters of WaveDown."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            input_channels, input_channels, 3, padding=1, groups=input_channels, bias=False
        )
        self.reduction = nn.Conv2d(
            input_channels, output_channels, 2, stride=2, bias=False
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.reduction(self.depthwise(x))))


class ConventionalDownsamplingWavestarNet(nn.Module):
    """Canonical WavestarNet with only WaveDown replaced."""

    widths = (48, 96, 192, 384)
    depths = (2, 3, 5, 1)
    masks = ((1, 0), (1, 0, 0), (1, 0, 0, 0, 1), (1,))

    def __init__(self, num_classes: int, dropout: float, max_drop_path: float) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.Hardswish(inplace=True),
        )
        rates = torch.linspace(0, max_drop_path, sum(self.depths)).tolist()
        rate_index = 0
        stages: list[nn.Module] = []
        input_channels = 24
        for output_channels, depth, mask in zip(self.widths, self.depths, self.masks):
            stage: list[nn.Module] = [
                CapacityMatchedConventionalDownsample(input_channels, output_channels)
            ]
            for block_index in range(depth):
                stage.append(
                    TransferS3StarBlock(
                        output_channels,
                        use_wavelet=bool(mask[block_index]),
                        drop_path=rates[rate_index],
                    )
                )
                rate_index += 1
            stages.append(nn.Sequential(*stage))
            input_channels = output_channels
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.LayerNorm(sum(self.widths[1:])),
            nn.Dropout(dropout),
            nn.Linear(sum(self.widths[1:]), num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return tuple(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        pooled = torch.cat(
            tuple(feature.float().mean(dim=(-2, -1)) for feature in features[1:]), dim=1
        )
        pooled = self.head[0](pooled).to(features[-1].dtype)
        return self.head[2](self.head[1](pooled))


def create_wavedown_ablation(
    ablation_key: str,
    num_classes: int = 105,
    *,
    dropout: float = 0.15,
    max_drop_path: float = 0.08,
) -> nn.Module:
    if ablation_key not in ABLATION_KEYS:
        raise ValueError(f"unknown WaveDown ablation: {ablation_key}")
    if ablation_key == "full_wavestarnet":
        return WavestarNetBigTransferV2(num_classes, dropout, max_drop_path)
    return ConventionalDownsamplingWavestarNet(num_classes, dropout, max_drop_path)
