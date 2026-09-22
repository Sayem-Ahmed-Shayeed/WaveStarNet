"""Pure constants and validation for the PlantWild WaveDown ablation."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence


ABLATION_KEYS = (
    "full_wavestarnet",
    "parameter_matched_conventional_downsampling",
)
ABLATION_PARAMETER_COUNTS = {
    "full_wavestarnet": 3_013_109,
    "parameter_matched_conventional_downsampling": 3_012_029,
}
SEEDS = (42, 2024, 3407)


def active_seeds(run_mode: str, seeds: Sequence[int] = SEEDS) -> list[int]:
    if tuple(seeds) != SEEDS:
        raise ValueError("seeds must be exactly [42, 2024, 3407]")
    if run_mode in {"smoke", "screening"}:
        return [42]
    if run_mode == "development":
        return list(SEEDS)
    raise ValueError("run_mode must be smoke, screening, or development")


@dataclass(frozen=True)
class AblationConfig:
    ablation_key: str
    run_mode: str
    num_classes: int = 105
    seeds: tuple[int, ...] = SEEDS
    image_size: int = 224
    train_batch_size: int = 32
    eval_batch_size: int = 64
    max_epochs: int = 100
    smoke_epochs: int = 1
    warmup_epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.05
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    min_lr_ratio: float = 0.05
    dropout: float = 0.15
    max_drop_path: float = 0.08
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 0.001
    early_stopping_start_epoch: int = 10

    def __post_init__(self) -> None:
        if self.ablation_key not in ABLATION_KEYS:
            raise ValueError(f"unknown WaveDown ablation: {self.ablation_key}")
        active_seeds(self.run_mode, self.seeds)
        if self.num_classes != 105 or self.image_size != 224:
            raise ValueError("PlantWild ablation requires 105 classes at 224 px")
        if (self.max_epochs, self.smoke_epochs, self.warmup_epochs) != (100, 1, 5):
            raise ValueError("PlantWild ablation requires 100/1 epochs and five warmups")
        if self.gradient_accumulation_steps != 1:
            raise ValueError("PlantWild ablation requires one optimizer step per batch")

    @property
    def model_key(self) -> str:
        """Compatibility name used by the shared scratch runtime."""
        return self.ablation_key


def model_spec(ablation_key: str) -> SimpleNamespace:
    if ablation_key not in ABLATION_KEYS:
        raise ValueError(f"unknown WaveDown ablation: {ablation_key}")
    return SimpleNamespace(
        key=ablation_key,
        display_name=(
            "WavestarNet"
            if ablation_key == "full_wavestarnet"
            else "WavestarNet (parameter-matched conventional downsampling)"
        ),
        expected_parameters=ABLATION_PARAMETER_COUNTS[ablation_key],
        pretrained=False,
        kind="wavestarnet",
    )


ablation_model_spec = model_spec
