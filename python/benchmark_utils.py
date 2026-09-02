from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def validate_timing_config(
    warmup: int,
    repeats: int,
) -> None:
    values = {
        "warmup": warmup,
        "repeats": repeats,
    }

    for name, value in values.items():
        # bool 是 int 的子类，所以需要单独排除。
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{name} must be an integer, "
                f"got {type(value).__name__}"
            )

    if warmup < 0:
        raise ValueError(
            f"warmup must be at least 0, got {warmup}"
        )

    if repeats <= 0:
        raise ValueError(
            f"repeats must be greater than 0, got {repeats}"
        )


def _percentile(
    samples: Sequence[float],
    quantile: float,
) -> float:
    """Calculate a percentile with linear interpolation."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError(
            f"quantile must be in [0, 1], got {quantile}"
        )

    ordered = sorted(samples)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return ordered[lower_index]

    fraction = position - lower_index
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return lower + (upper - lower) * fraction


def summarize(samples_ms: Sequence[float]) -> dict[str, float | int]:
    if not samples_ms:
        raise ValueError("samples_ms must not be empty")

    samples = [float(sample) for sample in samples_ms]

    if any(not math.isfinite(sample) for sample in samples):
        raise ValueError(
            "samples_ms must contain finite values"
        )

    if any(sample < 0 for sample in samples):
        raise ValueError(
            "samples_ms must contain non-negative values"
        )

    return {
        "count": len(samples),
        "min_ms": min(samples),
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": _percentile(samples, 0.95),
        "max_ms": max(samples),
    }


def time_cpu(
    fn: Callable[[], Any],
    warmup: int = 10,
    repeats: int = 100,
) -> list[float]:
    """Measure a synchronous CPU callable."""

    validate_timing_config(warmup=warmup, repeats=repeats)

    for _ in range(warmup):
        fn()

    samples_ms: list[float] = []

    for _ in range(repeats):
        start_ns = time.perf_counter_ns()
        fn()
        end_ns = time.perf_counter_ns()
        samples_ms.append((end_ns - start_ns) / 1_000_000.0)

    return samples_ms


def time_cuda(
    fn: Callable[[], Any],
    warmup: int = 10,
    repeats: int = 100,
) -> list[float]:
    """Measure CUDA work with CUDA Events."""

    validate_timing_config(warmup=warmup, repeats=repeats)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    samples_ms: list[float] = []

    for _ in range(repeats):
        start_event.record()
        fn()
        end_event.record()
        end_event.synchronize()
        samples_ms.append(float(start_event.elapsed_time(end_event)))

    return samples_ms


def save_result(
    result: dict[str, Any],
    path: str | Path,
) -> Path:
    """Save benchmark result as UTF-8 JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    output_path.write_text(serialized + "\n", encoding="utf-8")
    return output_path
