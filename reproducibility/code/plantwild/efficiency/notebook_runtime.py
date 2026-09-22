"""Dataset-free model-only efficiency benchmark used by the Kaggle notebook."""

from __future__ import annotations

import gc
import hashlib
import io
import json
import os
import platform
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from agrivision4.models.wavestarnet_family import create_wavestarnet
except ModuleNotFoundError:
    # Generated notebooks define this function in the embedded model cell.
    pass


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    timm_name: str | None


BENCHMARK_MODEL_REGISTRY: dict[str, ModelSpec] = {
    "wavestarnet_big_transfer_v2": ModelSpec(
        "WavestarNet-Big-Transfer-V2", None
    ),
    "efficientnet_b0": ModelSpec("EfficientNet-B0", "efficientnet_b0"),
    "efficientnet_b3": ModelSpec("EfficientNet-B3", "efficientnet_b3"),
    "mobilenetv3_large_100": ModelSpec(
        "MobileNetV3-Large-100", "mobilenetv3_large_100"
    ),
    "mobilevit_s": ModelSpec("MobileViT-S", "mobilevit_s"),
    "swin_tiny": ModelSpec("Swin-Tiny", "swin_tiny_patch4_window7_224"),
    "convnext_tiny": ModelSpec("ConvNeXt-Tiny", "convnext_tiny"),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    protocol_id: str = "plantwild_model_only_t4_efficiency_v1"
    num_classes: int = 105
    input_resolution: int = 224
    gpu_batch_sizes: tuple[int, ...] = (1, 32)
    gpu_precisions: tuple[str, ...] = ("fp32", "fp16")
    gpu_warmup_iterations: int = 50
    gpu_measurement_iterations: int = 500
    run_cpu: bool = True
    cpu_threads: int = 4
    cpu_warmup_iterations: int = 10
    cpu_measurement_iterations: int = 50
    require_t4: bool = True
    fail_on_error: bool = True

    def __post_init__(self) -> None:
        if self.num_classes != 105 or self.input_resolution != 224:
            raise ValueError("locked benchmark requires 105 classes at 224 px")
        if any(size <= 0 for size in self.gpu_batch_sizes):
            raise ValueError("GPU batch sizes must be positive")
        if any(value not in {"fp32", "fp16"} for value in self.gpu_precisions):
            raise ValueError("GPU precision must be fp32 or fp16")
        if bool(self.gpu_batch_sizes) != bool(self.gpu_precisions):
            raise ValueError("GPU batch sizes and precisions must both be enabled")
        if self.gpu_warmup_iterations < 0 or self.gpu_measurement_iterations <= 0:
            raise ValueError("GPU iteration budgets are invalid")
        if self.cpu_threads <= 0:
            raise ValueError("CPU thread count must be positive")
        if self.cpu_warmup_iterations < 0 or self.cpu_measurement_iterations <= 0:
            raise ValueError("CPU iteration budgets are invalid")
        if not self.run_cpu and not self.gpu_batch_sizes:
            raise ValueError("at least one CPU or GPU benchmark must be enabled")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(payload: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def create_model(model_key: str, *, num_classes: int = 105) -> nn.Module:
    if model_key not in BENCHMARK_MODEL_REGISTRY:
        raise KeyError(f"unknown benchmark model: {model_key}")
    spec = BENCHMARK_MODEL_REGISTRY[model_key]
    if spec.timm_name is None:
        return create_wavestarnet(model_key, num_classes)
    import timm

    return timm.create_model(spec.timm_name, pretrained=False, num_classes=num_classes)


def architecture_schema_sha256(model: nn.Module) -> str:
    schema = [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        for name, tensor in sorted(model.state_dict().items())
    ]
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def count_forward_flops(model: nn.Module, inputs: torch.Tensor) -> int:
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        output = model(inputs)
    if not isinstance(output, torch.Tensor):
        raise TypeError("benchmark model must return one logits tensor")
    total = int(counter.get_total_flops())
    if total <= 0:
        raise ValueError("native forward FLOP counter returned no operations")
    return total


def summarize_latencies(
    latencies_ms: Iterable[float], *, batch_size: int
) -> dict[str, float | int]:
    values = np.asarray(tuple(float(value) for value in latencies_ms), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("latencies must be a finite nonempty sequence")
    if (values <= 0.0).any() or batch_size <= 0:
        raise ValueError("latencies and batch size must be positive")
    mean = float(values.mean())
    return {
        "iterations": int(values.size),
        "latency_mean_ms": mean,
        "latency_std_ms": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "latency_median_ms": float(np.median(values)),
        "latency_p95_ms": float(np.percentile(values, 95)),
        "throughput_images_s": float(batch_size * 1000.0 / mean),
    }


def measure_complexity(
    model_key: str, model: nn.Module, config: BenchmarkConfig
) -> dict[str, Any]:
    model = model.cpu().float().eval()
    inputs = torch.zeros(1, 3, config.input_resolution, config.input_resolution)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    buffer_bytes = sum(
        buffer.numel() * buffer.element_size() for buffer in model.buffers()
    )
    serialized = io.BytesIO()
    torch.save(model.state_dict(), serialized)
    flops = count_forward_flops(model, inputs)
    return {
        "model_key": model_key,
        "model": BENCHMARK_MODEL_REGISTRY[model_key].display_name,
        "num_classes": config.num_classes,
        "input_resolution": config.input_resolution,
        "parameters": int(parameters),
        "trainable_parameters": int(trainable),
        "fp32_parameter_bytes": int(parameter_bytes),
        "fp32_parameter_mb": float(parameter_bytes / 1_000_000),
        "buffer_bytes": int(buffer_bytes),
        "serialized_state_dict_bytes": int(serialized.getbuffer().nbytes),
        "serialized_state_dict_mb": float(serialized.getbuffer().nbytes / 1_000_000),
        "forward_flops": flops,
        "forward_gflops": float(flops / 1_000_000_000),
        "fma_equivalent_macs": float(flops / 2.0),
        "architecture_schema_sha256": architecture_schema_sha256(model),
        "flop_counter": "torch.utils.flop_counter.FlopCounterMode",
    }


def _cpu_latency(
    model: nn.Module, config: BenchmarkConfig
) -> dict[str, Any]:
    model = model.cpu().float().eval()
    inputs = torch.zeros(1, 3, config.input_resolution, config.input_resolution)
    with torch.inference_mode():
        for _ in range(config.cpu_warmup_iterations):
            model(inputs)
        latencies = []
        for _ in range(config.cpu_measurement_iterations):
            started = time.perf_counter_ns()
            model(inputs)
            latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return {
        "device": "cpu",
        "precision": "fp32",
        "batch_size": 1,
        "warmup_iterations": config.cpu_warmup_iterations,
        "timing_method": "time.perf_counter_ns",
        "cpu_threads": config.cpu_threads,
        **summarize_latencies(latencies, batch_size=1),
        "peak_gpu_allocated_mb": None,
        "peak_gpu_reserved_mb": None,
    }


def _gpu_latency(
    model: nn.Module,
    config: BenchmarkConfig,
    *,
    batch_size: int,
    precision: str,
) -> dict[str, Any]:
    device = torch.device("cuda")
    model = model.to(device).eval()
    inputs = torch.zeros(
        batch_size,
        3,
        config.input_resolution,
        config.input_resolution,
        device=device,
    )
    def precision_context():
        return (
            torch.autocast("cuda", dtype=torch.float16)
            if precision == "fp16"
            else nullcontext()
        )

    with torch.inference_mode():
        for _ in range(config.gpu_warmup_iterations):
            with precision_context():
                output = model(inputs)
        del output
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        event_pairs = []
        for _ in range(config.gpu_measurement_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with precision_context():
                output = model(inputs)
            end.record()
            event_pairs.append((start, end))
        del output
        torch.cuda.synchronize()
    latencies = [float(start.elapsed_time(end)) for start, end in event_pairs]
    return {
        "device": "cuda",
        "precision": precision,
        "batch_size": batch_size,
        "warmup_iterations": config.gpu_warmup_iterations,
        "timing_method": "cuda_events_synchronized",
        "cpu_threads": None,
        **summarize_latencies(latencies, batch_size=batch_size),
        "peak_gpu_allocated_mb": float(torch.cuda.max_memory_allocated() / 1_000_000),
        "peak_gpu_reserved_mb": float(torch.cuda.max_memory_reserved() / 1_000_000),
    }


def environment_manifest(config: BenchmarkConfig) -> dict[str, Any]:
    import timm

    cuda_available = torch.cuda.is_available()
    return {
        "created_at": utc_now(),
        "protocol_id": config.protocol_id,
        "benchmark_scope": "model_only_synthetic_input",
        "dataset_accessed": False,
        "checkpoint_required": False,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32 if cuda_available else None,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32 if cuda_available else None,
    }


def write_artifact_inventory(output_directory: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output_directory.iterdir()):
        if path.is_file() and path.name != "artifact_inventory.json":
            files.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    payload = {
        "valid": bool(files),
        "file_count": len(files),
        "files": files,
    }
    write_json(payload, output_directory / "artifact_inventory.json")
    return payload


def run_benchmark(
    *,
    output_directory: str | Path,
    config: BenchmarkConfig,
    model_keys: Iterable[str] = tuple(BENCHMARK_MODEL_REGISTRY),
) -> Path:
    keys = tuple(model_keys)
    if not keys or len(set(keys)) != len(keys) or any(
        key not in BENCHMARK_MODEL_REGISTRY for key in keys
    ):
        raise ValueError("model_keys must be unique registered models")
    if config.gpu_batch_sizes:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the configured GPU benchmark")
        device_name = torch.cuda.get_device_name(0)
        if config.require_t4 and "T4" not in device_name.upper():
            raise RuntimeError(f"locked benchmark requires NVIDIA T4, found {device_name}")

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    original_threads = torch.get_num_threads()
    torch.set_num_threads(config.cpu_threads)
    write_json(asdict(config), output / "benchmark_config.json")
    write_json(environment_manifest(config), output / "environment.json")

    complexity_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    try:
        for model_index, model_key in enumerate(keys, start=1):
            spec = BENCHMARK_MODEL_REGISTRY[model_key]
            print(
                json.dumps(
                    {
                        "event": "model_benchmark_started",
                        "model_index": model_index,
                        "model_count": len(keys),
                        "model_key": model_key,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            try:
                cpu_model = create_model(model_key, num_classes=config.num_classes)
                complexity_rows.append(measure_complexity(model_key, cpu_model, config))
                if config.run_cpu:
                    cpu_row = _cpu_latency(cpu_model, config)
                    performance_rows.append(
                        {"model_key": model_key, "model": spec.display_name, **cpu_row}
                    )
                del cpu_model
                gc.collect()
            except Exception as error:
                errors.append(
                    {
                        "model_key": model_key,
                        "stage": "complexity_or_cpu",
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )
                continue

            for precision in config.gpu_precisions:
                for batch_size in config.gpu_batch_sizes:
                    try:
                        gpu_model = create_model(
                            model_key, num_classes=config.num_classes
                        )
                        gpu_row = _gpu_latency(
                            gpu_model,
                            config,
                            batch_size=batch_size,
                            precision=precision,
                        )
                        performance_rows.append(
                            {
                                "model_key": model_key,
                                "model": spec.display_name,
                                **gpu_row,
                            }
                        )
                        del gpu_model
                        gc.collect()
                        torch.cuda.empty_cache()
                    except Exception as error:
                        errors.append(
                            {
                                "model_key": model_key,
                                "stage": f"gpu_{precision}_batch_{batch_size}",
                                "error_type": type(error).__name__,
                                "message": str(error),
                            }
                        )
            print(
                json.dumps(
                    {"event": "model_benchmark_completed", "model_key": model_key},
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        torch.set_num_threads(original_threads)

    pd.DataFrame(complexity_rows).to_csv(output / "model_complexity.csv", index=False)
    pd.DataFrame(performance_rows).to_csv(
        output / "efficiency_results.csv", index=False
    )
    if errors:
        write_json(errors, output / "errors.json")
    write_artifact_inventory(output)
    if errors and config.fail_on_error:
        raise RuntimeError(
            f"benchmark recorded {len(errors)} errors; inspect {output / 'errors.json'}"
        )
    print(
        json.dumps(
            {
                "event": "benchmark_complete",
                "output_directory": str(output),
                "models": len(complexity_rows),
                "measurements": len(performance_rows),
                "errors": len(errors),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return output


def run_from_globals(namespace: dict[str, Any] | None = None) -> Path:
    values = globals() if namespace is None else namespace
    model_source = values["EMBEDDED_MODEL_SOURCE_BYTES"]
    runtime_source = values["EMBEDDED_RUNTIME_SOURCE_BYTES"]
    if hashlib.sha256(model_source).hexdigest() != values["EXPECTED_MODEL_SOURCE_SHA256"]:
        raise ValueError("embedded WavestarNet source hash mismatch")
    if hashlib.sha256(runtime_source).hexdigest() != values["EXPECTED_RUNTIME_SOURCE_SHA256"]:
        raise ValueError("embedded benchmark runtime hash mismatch")
    if tuple(values["MODEL_KEYS"]) != tuple(BENCHMARK_MODEL_REGISTRY):
        raise ValueError("benchmark model registry mismatch")
    config = BenchmarkConfig(**dict(values["BENCHMARK_CONFIG"]))
    output = Path("/kaggle/working") / str(values["OUTPUT_SAFE_NAME"])
    return run_benchmark(output_directory=output, config=config, model_keys=values["MODEL_KEYS"])


if __name__ == "__main__":
    run_from_globals()
