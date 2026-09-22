"""Canonical, self-contained WavestarNet architecture family."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


E11_V1_CHECKPOINT_SHA256 = (
    "9d305718b37a02fa075df80b473812dd6874d4102fec1b6b7442e1938394fd75"
)
TRANSFER_V2_PARAMETER_ELEMENTS = 2_941_100
V1_ENCODER_PARAMETER_ELEMENTS = 4_422_748


def _autocast_disabled(x: torch.Tensor):
    if x.is_cuda:
        return torch.amp.autocast(device_type="cuda", enabled=False)
    return nullcontext()


def _finite_clamp(x: torch.Tensor, value: float) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=value, neginf=-value).clamp(-value, value)


def _validate_probability(value: float, name: str, *, inclusive_one: bool) -> float:
    value = float(value)
    upper_ok = value <= 1.0 if inclusive_one else value < 1.0
    if value < 0.0 or not upper_ok:
        boundary = "[0, 1]" if inclusive_one else "[0, 1)"
        raise ValueError(f"{name} must be in {boundary}")
    return value


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = _validate_probability(
            probability, "DropPath probability", inclusive_one=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class Lifting1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if channels <= 0 or kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("channels and odd kernel_size must be positive")
        pad = kernel_size // 2
        self.predict = nn.Conv1d(
            channels, channels, kernel_size, padding=pad, groups=channels, bias=False
        )
        self.update = nn.Conv1d(
            channels, channels, kernel_size, padding=pad, groups=channels, bias=False
        )
        with torch.no_grad():
            self.predict.weight.zero_()
            self.update.weight.zero_()
            self.predict.weight[:, :, pad] = 1.0
            self.update.weight[:, :, pad] = 0.5

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] % 2 == 1:
            x = F.pad(x, (0, 1), mode="replicate")
        even, odd = x[..., 0::2], x[..., 1::2]
        detail = odd - self.predict(even)
        approximation = even + self.update(detail)
        return approximation, detail

    def inverse(
        self, approximation: torch.Tensor, detail: torch.Tensor, out_len: int | None = None
    ) -> torch.Tensor:
        even = approximation - self.update(detail)
        odd = detail + self.predict(even)
        output = torch.stack((even, odd), dim=-1).flatten(-2)
        return output if out_len is None else output[..., :out_len]


class DWT2D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.lw = Lifting1D(channels, kernel_size)
        self.lh = Lifting1D(channels, kernel_size)

    def _width(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, height, width = x.shape
        rows = x.permute(0, 2, 1, 3).contiguous().reshape(batch * height, channels, width)
        low, high = self.lw(rows)
        low = low.reshape(batch, height, channels, -1).permute(0, 2, 1, 3).contiguous()
        high = high.reshape(batch, height, channels, -1).permute(0, 2, 1, 3).contiguous()
        return low, high

    def _width_inverse(
        self, low: torch.Tensor, high: torch.Tensor, out_width: int | None
    ) -> torch.Tensor:
        batch, channels, height, half_width = low.shape
        low_rows = low.permute(0, 2, 1, 3).contiguous().reshape(
            batch * height, channels, half_width
        )
        high_rows = high.permute(0, 2, 1, 3).contiguous().reshape(
            batch * height, channels, half_width
        )
        output = self.lw.inverse(low_rows, high_rows, out_width)
        return output.reshape(batch, height, channels, -1).permute(0, 2, 1, 3).contiguous()

    def _height(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, height, width = x.shape
        columns = x.permute(0, 3, 1, 2).contiguous().reshape(
            batch * width, channels, height
        )
        low, high = self.lh(columns)
        low = low.reshape(batch, width, channels, -1).permute(0, 2, 3, 1).contiguous()
        high = high.reshape(batch, width, channels, -1).permute(0, 2, 3, 1).contiguous()
        return low, high

    def _height_inverse(
        self, low: torch.Tensor, high: torch.Tensor, out_height: int | None
    ) -> torch.Tensor:
        batch, channels, half_height, width = low.shape
        low_columns = low.permute(0, 3, 1, 2).contiguous().reshape(
            batch * width, channels, half_height
        )
        high_columns = high.permute(0, 3, 1, 2).contiguous().reshape(
            batch * width, channels, half_height
        )
        output = self.lh.inverse(low_columns, high_columns, out_height)
        return output.reshape(batch, width, channels, -1).permute(0, 2, 3, 1).contiguous()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        low, high = self._width(x)
        ll, lh = self._height(low)
        hl, hh = self._height(high)
        return ll, lh, hl, hh

    def inverse(
        self,
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
        out_hw: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        out_height = out_width = None
        if out_hw is not None:
            out_height, out_width = out_hw
        low = self._height_inverse(ll, lh, out_height)
        high = self._height_inverse(hl, hh, out_height)
        return self._width_inverse(low, high, out_width)


class WaveletGate(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, reduction: int = 8) -> None:
        super().__init__()
        self.dwt = DWT2D(channels, kernel_size)
        hidden = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 4, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        with _autocast_disabled(x):
            x32 = x.float()
            bands = torch.cat(self.dwt(x32), dim=1)
            gate = torch.nan_to_num(
                self.gate(bands), nan=0.5, posinf=1.0, neginf=0.0
            ).clamp(1e-4, 1.0)
            bands = _finite_clamp(bands * gate, 20.0)
            output = self.dwt.inverse(
                *torch.split(bands, x32.shape[1], dim=1), out_hw=x32.shape[2:]
            )
        return _finite_clamp(output, 20.0).to(output_dtype)


class S3StarBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dw_kernel: int = 7,
        expand: int = 2,
        layerscale: float = 1e-4,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        expanded = channels * expand
        self.dw = nn.Conv2d(
            channels, channels, dw_kernel, padding=dw_kernel // 2,
            groups=channels, bias=False,
        )
        self.bn_s = nn.BatchNorm2d(channels)
        self.spec = WaveletGate(channels)
        self.bn_f = nn.BatchNorm2d(channels)
        self.proj_s = nn.Conv2d(channels, expanded, 1, bias=False)
        self.proj_f = nn.Conv2d(channels, expanded, 1, bias=False)
        self.pbn_s = nn.BatchNorm2d(expanded)
        self.pbn_f = nn.BatchNorm2d(expanded)
        self.act = nn.Hardswish(inplace=True)
        self.proj_out = nn.Conv2d(expanded, channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(channels)
        self.gamma = nn.Parameter(layerscale * torch.ones(1, channels, 1, 1))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.bn_s(self.dw(x))
        spectral = self.bn_f(self.spec(x))
        a = self.act(self.pbn_s(self.proj_s(spatial)))
        b = self.act(self.pbn_f(self.proj_f(spectral)))
        with _autocast_disabled(a):
            fused = _finite_clamp(a.float() * b.float(), 8.0).to(a.dtype)
        output = self.gamma * self.bn_out(self.proj_out(fused))
        return x + self.drop_path(output)


class TransferS3StarBlock(nn.Module):
    """E11-compatible S3 block with an optional lightweight spectral bypass."""

    def __init__(
        self,
        channels: int,
        use_wavelet: bool,
        dw_kernel: int = 7,
        expand: int = 2,
        layerscale: float = 1e-4,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        expanded = channels * expand
        self.dw = nn.Conv2d(
            channels,
            channels,
            dw_kernel,
            padding=dw_kernel // 2,
            groups=channels,
            bias=False,
        )
        self.bn_s = nn.BatchNorm2d(channels)
        if use_wavelet:
            self.spec = WaveletGate(channels)
        self.bn_f = nn.BatchNorm2d(channels)
        self.proj_s = nn.Conv2d(channels, expanded, 1, bias=False)
        self.proj_f = nn.Conv2d(channels, expanded, 1, bias=False)
        self.pbn_s = nn.BatchNorm2d(expanded)
        self.pbn_f = nn.BatchNorm2d(expanded)
        self.act = nn.Hardswish(inplace=True)
        self.proj_out = nn.Conv2d(expanded, channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(channels)
        self.gamma = nn.Parameter(layerscale * torch.ones(1, channels, 1, 1))
        self.drop_path = DropPath(drop_path)

    def spectral_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "spec"):
            return self.spec(x)
        return x * 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.bn_s(self.dw(x))
        spectral = self.bn_f(self.spectral_features(x))
        a = self.act(self.pbn_s(self.proj_s(spatial)))
        b = self.act(self.pbn_f(self.proj_f(spectral)))
        with _autocast_disabled(a):
            fused = _finite_clamp(a.float() * b.float(), 8.0).to(a.dtype)
        output = (self.gamma * self.bn_out(self.proj_out(fused))).to(x.dtype)
        return x + self.drop_path(output)


class WaveDown(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.dwt = DWT2D(input_channels, 3)
        self.proj = nn.Conv2d(input_channels * 4, output_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(output_channels)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        with _autocast_disabled(x):
            bands = _finite_clamp(torch.cat(self.dwt(x.float()), dim=1), 20.0)
        return self.act(self.bn(self.proj(bands.to(output_dtype))))


class WavestarNetV1(nn.Module):
    def __init__(self, num_classes: int, dropout: float, max_drop_path: float) -> None:
        super().__init__()
        stem_channels = 24
        widths = (48, 96, 192, 384)
        depths = (2, 3, 6, 2)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.Hardswish(inplace=True),
        )
        stages: list[nn.Module] = []
        input_channels = stem_channels
        block_index = 0
        total_blocks = sum(depths)
        for output_channels, depth in zip(widths, depths):
            stage: list[nn.Module] = [WaveDown(input_channels, output_channels)]
            for _ in range(depth):
                probability = max_drop_path * block_index / max(1, total_blocks - 1)
                stage.append(S3StarBlock(output_channels, drop_path=probability))
                block_index += 1
            stages.append(nn.Sequential(*stage))
            input_channels = output_channels
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(widths[-1], 1536),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1536, num_classes),
        )

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
        return self.head(features[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))

    def classifier_module(self) -> nn.Module:
        return self.head


class WavestarNetBigTransferV2(nn.Module):
    """Compact V1-compatible model designed for exact E11 weight transfer."""

    WIDTHS = (48, 96, 192, 384)
    DEPTHS = (2, 3, 5, 1)
    MASKS = ((1, 0), (1, 0, 0), (1, 0, 0, 0, 1), (1,))

    def __init__(self, num_classes: int, dropout: float, max_drop_path: float) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.Hardswish(inplace=True),
        )
        rates = torch.linspace(0, max_drop_path, sum(self.DEPTHS)).tolist()
        stages: list[nn.Module] = []
        input_channels = 24
        rate_index = 0
        for output_channels, depth, mask in zip(self.WIDTHS, self.DEPTHS, self.MASKS):
            stage: list[nn.Module] = [WaveDown(input_channels, output_channels)]
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
        pooled_channels = sum(self.WIDTHS[1:])
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_channels),
            nn.Dropout(dropout),
            nn.Linear(pooled_channels, num_classes),
        )

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
        pooled = torch.cat(
            tuple(feature.float().mean(dim=(-2, -1)) for feature in features[1:]),
            dim=1,
        )
        pooled = self.head[0](pooled).to(features[-1].dtype)
        return self.head[2](self.head[1](pooled))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))

    def classifier_module(self) -> nn.Module:
        return self.head[2]


class WaveStem(nn.Module):
    def __init__(self, output_channels: int, wavelet_clamp: float = 20.0) -> None:
        super().__init__()
        self.dwt = DWT2D(3)
        self.mix = nn.Sequential(
            nn.Conv2d(12, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Hardswish(inplace=True),
            nn.Conv2d(
                output_channels, output_channels, 3, padding=1,
                groups=output_channels, bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.Hardswish(inplace=True),
        )
        self.wavelet_clamp = float(wavelet_clamp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        with _autocast_disabled(x):
            bands = _finite_clamp(torch.cat(self.dwt(x.float()), dim=1), self.wavelet_clamp)
        return self.mix(bands.to(output_dtype))


class ConfiguredWaveDown(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, wavelet_clamp: float = 20.0
    ) -> None:
        super().__init__()
        self.dwt = DWT2D(input_channels)
        self.proj = nn.Conv2d(4 * input_channels, output_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(output_channels)
        self.act = nn.Hardswish(inplace=True)
        self.wavelet_clamp = float(wavelet_clamp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        with _autocast_disabled(x):
            bands = _finite_clamp(torch.cat(self.dwt(x.float()), dim=1), self.wavelet_clamp)
        return self.act(self.bn(self.proj(bands.to(output_dtype))))


class IdentityWaveletGate(nn.Module):
    def __init__(
        self, channels: int, reduction: int = 8, wavelet_clamp: float = 20.0
    ) -> None:
        super().__init__()
        self.dwt = DWT2D(channels)
        hidden = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4 * channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 4 * channels, 1),
        )
        self.wavelet_clamp = float(wavelet_clamp)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        with _autocast_disabled(x):
            x32 = x.float()
            bands = torch.cat(self.dwt(x32), dim=1)
            gain = 1.0 + 0.5 * torch.tanh(self.gate(bands))
            gated = _finite_clamp(bands * gain, self.wavelet_clamp)
            output = self.dwt.inverse(
                *gated.chunk(4, dim=1), out_hw=(x32.shape[-2], x32.shape[-1])
            )
            output = _finite_clamp(output, self.wavelet_clamp)
        return output.to(output_dtype)


class PointwiseBNAct(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Hardswish(inplace=True),
        )


class SelectiveWaveStarBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        use_wavelet: bool,
        kernel: int,
        layer_scale: float,
        drop_path: float,
        fuse_clamp: float = 8.0,
    ) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel, padding=kernel // 2,
                groups=channels, bias=False,
            ),
            nn.BatchNorm2d(channels),
        )
        self.spectral = IdentityWaveletGate(channels) if use_wavelet else None
        self.bn_f = nn.BatchNorm2d(channels)
        self.proj_a = PointwiseBNAct(channels, 2 * channels)
        self.proj_b = PointwiseBNAct(channels, 2 * channels)
        self.out = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gamma = nn.Parameter(layer_scale * torch.ones(1, channels, 1, 1))
        self.drop_path = DropPath(drop_path)
        self.fuse_clamp = float(fuse_clamp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.proj_a(self.spatial(x))
        spectral = self.spectral(x) if self.spectral is not None else x
        b = self.proj_b(self.bn_f(spectral))
        with _autocast_disabled(a):
            fused = _finite_clamp(a.float() * b.float(), self.fuse_clamp).to(a.dtype)
        residual = self.gamma * self.out(fused)
        return x + self.drop_path(residual)


def _avg_rms_pool(x: torch.Tensor) -> torch.Tensor:
    x32 = x.float()
    average = x32.mean(dim=(-2, -1))
    rms = x32.square().mean(dim=(-2, -1)).add(1e-6).sqrt()
    return torch.cat((average, rms), dim=1)


class SelectiveWaveStarNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        stem_channels: int,
        widths: tuple[int, ...],
        depths: tuple[int, ...],
        kernels: tuple[int, ...],
        masks: tuple[tuple[int, ...], ...],
        head_dim: int,
        dropout: float,
        max_drop_path: float,
        layer_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        if not (len(widths) == len(depths) == len(kernels) == len(masks) == 4):
            raise ValueError("widths, depths, kernels, and masks must have four stages")
        if any(value <= 0 for value in (*widths, *depths, *kernels)):
            raise ValueError("channel, depth, and kernel values must be positive")
        if any(len(mask) != depth for mask, depth in zip(masks, depths)):
            raise ValueError("each wavelet mask length must equal its stage depth")
        if any(value not in (0, 1) for mask in masks for value in mask):
            raise ValueError("wavelet masks must contain only zero or one")
        rates = torch.linspace(0, max_drop_path, sum(depths)).tolist()
        self.stem = WaveStem(stem_channels)
        self.stages = nn.ModuleList()
        rate_id = 0
        input_channels = stem_channels
        for output_channels, depth, kernel, mask in zip(widths, depths, kernels, masks):
            blocks: list[nn.Module] = [ConfiguredWaveDown(input_channels, output_channels)]
            for block_id in range(depth):
                blocks.append(
                    SelectiveWaveStarBlock(
                        output_channels,
                        bool(mask[block_id]),
                        kernel,
                        layer_scale,
                        rates[rate_id],
                    )
                )
                rate_id += 1
            self.stages.append(nn.Sequential(*blocks))
            input_channels = output_channels
        pooled_dim = 2 * sum(widths[1:])
        self.head_norm = nn.LayerNorm(pooled_dim)
        self.head_fc1 = nn.Linear(pooled_dim, head_dim)
        self.head_act = nn.Hardswish(inplace=True)
        self.head_drop = nn.Dropout(dropout)
        self.head_fc2 = nn.Linear(head_dim, num_classes)

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
        pooled = torch.cat(tuple(_avg_rms_pool(x) for x in features[1:]), dim=1)
        pooled = self.head_norm(pooled).to(features[-1].dtype)
        pooled = self.head_drop(self.head_act(self.head_fc1(pooled)))
        return self.head_fc2(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))

    def classifier_module(self) -> nn.Module:
        return self.head_fc2


@dataclass(frozen=True)
class WavestarNetSpec:
    key: str
    display_name: str
    stem_channels: int
    widths: tuple[int, int, int, int]
    depths: tuple[int, int, int, int]
    kernels: tuple[int, int, int, int]
    masks: tuple[tuple[int, ...], ...]
    head_dim: int
    dropout: float
    max_drop_path: float
    layer_scale: float
    supports_e11: bool


MODEL_REGISTRY: dict[str, WavestarNetSpec] = {
    "wavestarnet_big_v1": WavestarNetSpec(
        "wavestarnet_big_v1", "WavestarNet-Big V1", 24,
        (48, 96, 192, 384), (2, 3, 6, 2), (7, 7, 7, 7),
        ((1, 1), (1, 1, 1), (1, 1, 1, 1, 1, 1), (1, 1)),
        1536, 0.25, 0.0, 1e-4, True,
    ),
    "wavestarnet_big_acc": WavestarNetSpec(
        "wavestarnet_big_acc", "WavestarNet-Big-Acc", 24,
        (48, 96, 192, 384), (2, 4, 6, 1), (7, 7, 7, 7),
        ((1, 0), (1, 0, 0, 1), (1, 0, 0, 0, 0, 1), (1,)),
        448, 0.20, 0.10, 1e-3, False,
    ),
    "wavestarnet_big_lite": WavestarNetSpec(
        "wavestarnet_big_lite", "WavestarNet-Big-Lite", 20,
        (40, 80, 160, 256), (2, 3, 5, 1), (5, 5, 7, 7),
        ((1, 0), (1, 0, 0), (1, 0, 0, 0, 0), (1,)),
        256, 0.15, 0.08, 1e-3, False,
    ),
    "wavestarnet_big_transfer_v2": WavestarNetSpec(
        "wavestarnet_big_transfer_v2", "WavestarNet-Big-Transfer-V2", 24,
        (48, 96, 192, 384), (2, 3, 5, 1), (7, 7, 7, 7),
        ((1, 0), (1, 0, 0), (1, 0, 0, 0, 1), (1,)),
        672, 0.15, 0.08, 1e-4, True,
    ),
}


def create_wavestarnet(
    model_key: str,
    num_classes: int,
    *,
    dropout: float | None = None,
    max_drop_path: float | None = None,
) -> nn.Module:
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"unknown model key: {model_key}")
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    spec = MODEL_REGISTRY[model_key]
    dropout_value = spec.dropout if dropout is None else _validate_probability(
        dropout, "dropout", inclusive_one=False
    )
    drop_path_value = spec.max_drop_path if max_drop_path is None else _validate_probability(
        max_drop_path, "drop path", inclusive_one=False
    )
    if model_key == "wavestarnet_big_v1":
        return WavestarNetV1(num_classes, dropout_value, drop_path_value)
    if model_key == "wavestarnet_big_transfer_v2":
        return WavestarNetBigTransferV2(num_classes, dropout_value, drop_path_value)
    return SelectiveWaveStarNet(
        num_classes=num_classes,
        stem_channels=spec.stem_channels,
        widths=spec.widths,
        depths=spec.depths,
        kernels=spec.kernels,
        masks=spec.masks,
        head_dim=spec.head_dim,
        dropout=dropout_value,
        max_drop_path=drop_path_value,
        layer_scale=spec.layer_scale,
    )


def _state_schema(model_key: str, num_classes: int, *, backbone_only: bool) -> list[dict[str, Any]]:
    model = create_wavestarnet(model_key, num_classes)
    if model_key == "wavestarnet_big_v1":
        excluded = ("head.5.",)
    elif model_key == "wavestarnet_big_transfer_v2":
        excluded = ("head.2.",)
    else:
        excluded = ("head_fc2.",)
    schema = []
    for name, value in sorted(model.state_dict().items()):
        if backbone_only and name.startswith(excluded):
            continue
        schema.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)})
    return schema


def architecture_manifest(model_key: str, num_classes: int) -> dict[str, Any]:
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"unknown model key: {model_key}")
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    model = create_wavestarnet(model_key, num_classes)
    return {
        "model_key": model_key,
        "num_classes": num_classes,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "spec": asdict(MODEL_REGISTRY[model_key]),
        "state_schema": _state_schema(model_key, num_classes, backbone_only=False),
    }


def _hash_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def architecture_fingerprint(model_key: str, num_classes: int) -> str:
    return _hash_json(architecture_manifest(model_key, num_classes))


def backbone_schema_fingerprint(model_key: str) -> str:
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"unknown model key: {model_key}")
    return _hash_json(
        {
            "model_key": model_key,
            "spec": asdict(MODEL_REGISTRY[model_key]),
            "state_schema": _state_schema(model_key, 21, backbone_only=True),
        }
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duplicate_encoder_aliases(model: nn.Module) -> list[list[str]]:
    aliases: list[list[str]] = []
    for values in (
        model.named_parameters(remove_duplicate=False),
        model.named_buffers(remove_duplicate=False),
    ):
        names_by_identity: dict[int, list[str]] = {}
        for name, value in values:
            if name.startswith(("stem.", "stages.")):
                names_by_identity.setdefault(id(value), []).append(name)
        aliases.extend(names for names in names_by_identity.values() if len(names) > 1)
    return aliases


def load_e11_transfer_v2(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str = E11_V1_CHECKPOINT_SHA256,
) -> dict[str, Any]:
    """Load the exact retained V1 encoder tensors into Transfer V2, fail closed."""

    if not isinstance(model, WavestarNetBigTransferV2):
        raise ValueError("E11 Transfer V2 loader requires WavestarNet-Big-Transfer-V2")
    duplicate_aliases = _duplicate_encoder_aliases(model)
    if duplicate_aliases:
        raise ValueError(f"duplicate target encoder aliases: {duplicate_aliases}")

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"E11 checkpoint not found: {path}")
    observed_sha256 = _sha256_file(path)
    if observed_sha256.lower() != str(expected_sha256).lower():
        raise ValueError(
            "E11 checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("E11 checkpoint must contain a tensor state mapping")
    wrappers = [key for key in ("model_state_dict", "state_dict") if key in payload]
    if len(wrappers) > 1:
        raise ValueError(f"ambiguous E11 checkpoint wrappers: {wrappers}")
    source_state = payload[wrappers[0]] if wrappers else payload
    if not isinstance(source_state, Mapping):
        raise ValueError("E11 checkpoint state must be a mapping")
    if any(not isinstance(key, str) or not torch.is_tensor(value)
           for key, value in source_state.items()):
        raise ValueError("E11 checkpoint contains non-tensor state values")

    target_state = model.state_dict()
    encoder_keys = sorted(
        key for key in target_state if key.startswith(("stem.", "stages."))
    )
    matching_keys = [key for key in encoder_keys if key in source_state]
    if not matching_keys:
        raise ValueError("no E11 tensors match the Transfer V2 encoder")
    missing_keys = sorted(set(encoder_keys) - set(source_state))
    if missing_keys:
        raise ValueError(f"unexpected target keys absent from E11: {missing_keys}")

    shape_errors = [
        (key, tuple(source_state[key].shape), tuple(target_state[key].shape))
        for key in encoder_keys
        if tuple(source_state[key].shape) != tuple(target_state[key].shape)
    ]
    if shape_errors:
        raise ValueError(f"E11 retained tensor shape mismatch: {shape_errors}")
    dtype_errors = [
        (key, str(source_state[key].dtype), str(target_state[key].dtype))
        for key in encoder_keys
        if source_state[key].dtype != target_state[key].dtype
    ]
    if dtype_errors:
        raise ValueError(f"E11 retained tensor dtype mismatch: {dtype_errors}")

    head_before = {
        key: value.detach().clone()
        for key, value in target_state.items()
        if key.startswith("head.")
    }
    updated_state = dict(target_state)
    for key in encoder_keys:
        updated_state[key] = source_state[key].detach().clone()
    model.load_state_dict(updated_state, strict=True)

    loaded_state = model.state_dict()
    exact_tensor_equality = all(
        torch.equal(loaded_state[key].cpu(), source_state[key].cpu())
        for key in encoder_keys
    )
    head_preserved = all(
        torch.equal(loaded_state[key].cpu(), value.cpu())
        for key, value in head_before.items()
    )
    if not exact_tensor_equality:
        raise RuntimeError("E11 transfer failed exact tensor equality validation")
    if not head_preserved:
        raise RuntimeError("E11 transfer modified the fresh classifier head")

    parameter_elements = {
        key: parameter.numel() for key, parameter in model.named_parameters()
    }
    loaded_parameter_elements = sum(
        parameter_elements.get(key, 0) for key in encoder_keys
    )
    if loaded_parameter_elements != TRANSFER_V2_PARAMETER_ELEMENTS:
        raise ValueError(
            "unexpected transferred parameter count: "
            f"expected {TRANSFER_V2_PARAMETER_ELEMENTS}, observed {loaded_parameter_elements}"
        )
    total_parameter_elements = sum(parameter.numel() for parameter in model.parameters())
    stage_coverage = {}
    for stage_index in range(4):
        prefix = f"stages.{stage_index}."
        target_elements = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith(prefix)
        )
        loaded_elements = sum(
            parameter_elements.get(key, 0)
            for key in encoder_keys
            if key.startswith(prefix)
        )
        stage_coverage[f"stage_{stage_index + 1}"] = {
            "loaded_parameter_elements": loaded_elements,
            "target_parameter_elements": target_elements,
            "coverage": loaded_elements / target_elements,
        }
    skipped_keys = sorted(
        key for key in source_state
        if key.startswith(("stem.", "stages.")) and key not in encoder_keys
    )
    all_skipped_source_keys = sorted(set(source_state) - set(encoder_keys))
    newly_initialized_target_keys = sorted(
        key for key in target_state if key not in encoder_keys
    )
    num_classes = model.head[2].out_features
    return {
        "target_model_key": "wavestarnet_big_transfer_v2",
        "target_num_classes": num_classes,
        "target_architecture_sha256": architecture_fingerprint(
            "wavestarnet_big_transfer_v2", num_classes
        ),
        "target_backbone_schema_sha256": backbone_schema_fingerprint(
            "wavestarnet_big_transfer_v2"
        ),
        "source_checkpoint": str(path),
        "source_sha256": observed_sha256,
        "expected_sha256": str(expected_sha256).lower(),
        "loaded_keys": encoder_keys,
        "loaded_key_count": len(encoder_keys),
        "loaded_tensor_count": len(encoder_keys),
        "loaded_state_elements": sum(target_state[key].numel() for key in encoder_keys),
        "skipped_source_encoder_keys": skipped_keys,
        "skipped_source_keys": all_skipped_source_keys,
        "newly_initialized_target_keys": newly_initialized_target_keys,
        "loaded_parameter_elements": loaded_parameter_elements,
        "target_parameter_elements": total_parameter_elements,
        "target_parameter_coverage": loaded_parameter_elements / total_parameter_elements,
        "original_e11_encoder_parameter_elements": V1_ENCODER_PARAMETER_ELEMENTS,
        "original_e11_encoder_utilization": (
            loaded_parameter_elements / V1_ENCODER_PARAMETER_ELEMENTS
        ),
        "per_stage_coverage": stage_coverage,
        "exact_tensor_equality": exact_tensor_equality,
        "head_preserved": head_preserved,
    }
