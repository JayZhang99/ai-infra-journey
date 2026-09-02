from __future__ import annotations

import platform
from datetime import datetime
from pathlib import Path
from typing import Any
import argparse
import torch

from .benchmark_utils import (
    DTYPE_MAP,
    save_result,
    summarize,
    time_cpu,
    time_cuda,
)

BOUNDARY = "matmul_compute_only"

def validate_config(
    size: int,
    warmup: int,
    repeats: int,
):
    values = {
        "size": size,
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

    if size <= 0:
        raise ValueError(
            f"size must be greater than 0, got {size}"
        )

    if warmup < 0:
        raise ValueError(
            f"warmup must be at least 0, got {warmup}"
        )

    if repeats <= 0:
        raise ValueError(
            f"repeats must be greater than 0, got {repeats}"
        )

def run_matmul_benchmark(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    warmup: int = 10,
    repeats: int = 100,
    seed: int = 7,
) -> dict[str, Any]:
    """Benchmark [size, size] @ [size, size]."""

    validate_config(
        size=size,
        warmup=warmup,
        repeats=repeats,
    )

    if not isinstance(dtype, torch.dtype):
        raise TypeError(
            f"dtype must be torch.dtype, "
            f"got {type(dtype).__name__}"
        )

    if not dtype.is_floating_point:
        raise TypeError(
            "MatMul benchmark requires a "
            f"floating dtype, got {dtype}"
        )

    device = torch.device(device)

    if device.type not in {"cpu", "cuda"}:
        raise ValueError(
            "Only cpu and cuda are supported, "
            f"got {device.type}"
        )

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

        if device.index is None:
            device = torch.device(
                "cuda",
                torch.cuda.current_device(),
            )

    torch.manual_seed(seed)

    # 输入必须在计时区域之外创建。
    left = torch.randn(
        size,
        size,
        device=device,
        dtype=dtype,
    )

    right = torch.randn(
        size,
        size,
        device=device,
        dtype=dtype,
    )

    def matmul() -> torch.Tensor:
        # 计时区域只包含 MatMul。
        return torch.matmul(left, right)

    with torch.inference_mode():
        # 先验证正确性，再开始计时。
        output = matmul()

        output_shape = list(output.shape)

        output_is_finite = bool(
            torch.isfinite(output).all().item()
        )

        output_requires_grad = (
            output.requires_grad
        )

        if device.type == "cuda":
            # 保证 Event 建立在目标 CUDA 设备上。
            with torch.cuda.device(device):
                samples_ms = time_cuda(
                    matmul,
                    warmup=warmup,
                    repeats=repeats,
                )

            timer = "cuda_event"

        else:
            samples_ms = time_cpu(
                matmul,
                warmup=warmup,
                repeats=repeats,
            )

            timer = "perf_counter_ns"

    summary = summarize(samples_ms)

    median_ms = float(
        summary["median_ms"]
    )

    # 方阵 MatMul 的近似浮点运算量。
    flops_per_matmul = 2 * size**3

    if median_ms > 0:
        estimated_tflops = (
            flops_per_matmul
            / (median_ms / 1000.0)
            / 1_000_000_000_000.0
        )
    else:
        estimated_tflops = 0.0

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(
            device
        )
    else:
        device_name = (
            platform.processor()
            or platform.machine()
            or "CPU"
        )

    return {
        "schema_version": 1,
        "timestamp": (
            datetime.now()
            .astimezone()
            .isoformat(timespec="seconds")
        ),

        # 测量对象和边界
        "operation": "matmul",
        "boundary": BOUNDARY,

        # 输入和输出
        "shape_a": [size, size],
        "shape_b": [size, size],
        "output_shape": output_shape,
        "dtype": str(dtype).removeprefix("torch."),

        # 环境
        "device": str(device),
        "device_name": device_name,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "torch_num_threads": (
            torch.get_num_threads()
        ),

        # 实验协议
        "seed": seed,
        "warmup": warmup,
        "repeats": repeats,
        "timer": timer,

        # 原始结果
        "samples_ms": samples_ms,

        # 统计结果
        **summary,

        # 计算量和估算性能
        "flops_per_matmul": flops_per_matmul,
        "estimated_tflops": estimated_tflops,

        # 正确性证据
        "output_is_finite": output_is_finite,
        "output_requires_grad": (
            output_requires_grad
        ),
    }

def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a PyTorch MatMul benchmark."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sizes", nargs="+", type=int,
    required=True
    )
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser

def main() -> None:
    args = build_parser().parse_args()
    dtype = DTYPE_MAP[args.dtype]
    results = []
    for size in args.sizes:
        results.append(
            run_matmul_benchmark(
                size = size,
                device = args.device,
                dtype= dtype,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        )
    
    payload = {
        "schema_version": 1,
        "benchmark": "square_matmul",
        "results": results,
    }
    save_result(payload, args.output)



if __name__ == "__main__":
    main()
