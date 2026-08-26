from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
)


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def validate_device(device: torch.device) -> torch.device:
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(
            f"Only cpu and cuda are supported, got {device}"
        )

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

        if device.index is None:
            device = torch.device(
                "cuda",
                torch.cuda.current_device(),
            )

    return device


def profile_matmul(
    *,
    size: int,
    device: str,
    dtype: torch.dtype,
    warmup: int,
    steps: int,
    trace_path: Path,
    top5_path: Path,
) -> None:
    if size <= 0:
        raise ValueError("size must be greater than 0")

    if warmup < 0:
        raise ValueError("warmup must be at least 0")

    if steps <= 0:
        raise ValueError("steps must be greater than 0")

    resolved_device = validate_device(
        torch.device(device)
    )

    torch.manual_seed(7)

    # 输入创建不进入 Profiler。
    left = torch.randn(
        size,
        size,
        device=resolved_device,
        dtype=dtype,
    )

    right = torch.randn(
        size,
        size,
        device=resolved_device,
        dtype=dtype,
    )

    def matmul() -> torch.Tensor:
        return torch.matmul(left, right)

    trace_path.parent.mkdir(
    parents=True,
    exist_ok=True,)

    top5_path.parent.mkdir(
    parents=True,
    exist_ok=True,
    )

    sort_by = (
        "self_cuda_time_total"
        if resolved_device.type == "cuda"
        else "self_cpu_time_total"
    )


    def trace_handler(prof) -> None:
        top5 = prof.key_averages(
            group_by_input_shape=True,
        ).table(
            sort_by=sort_by,
            row_limit=5,
        )

        top5_path.write_text(
            top5 + "\n",
            encoding="utf-8",
        )

        prof.export_chrome_trace(
            str(trace_path)
        )

        print(top5)
        print(f"Top 5 saved to: {top5_path}")
        print(f"Trace saved to: {trace_path}")

    with torch.inference_mode():
        # Warmup 不进入 Profiler。
        for _ in range(warmup):
            output = matmul()

        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)

        activities = [ProfilerActivity.CPU]

        if resolved_device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        profiler_wait = 1
        profiler_warmup = 1

        schedule = torch.profiler.schedule(
            wait=profiler_wait, warmup=profiler_warmup, active=steps, repeat=1
        )

        total_profiler_steps = (
            profiler_wait
            + profiler_warmup
            + steps
            )

        with profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for _ in range(total_profiler_steps):
                with record_function(
                    "matmul_compute_only"
                ):
                    output = matmul()
                prof.step()

        # 正确性检查放在 Profiler 区域之外。
        output_is_finite = bool(
            torch.isfinite(output).all().item()
        )

    if not output_is_finite:
        raise RuntimeError(
            "MatMul output contains non-finite values"
        )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile PyTorch MatMul."
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPE_MAP),
        default="float32",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--trace",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--top5",
        type=Path,
        required=True,
    )

    return parser

def trace_handler(prof):
    output_dir = Path("benchmarks/profiler")
    output_dir.mkdir(parents=True, exist_ok=True)

    sort_by = (
        "self_cuda_time_total"
        if torch.cuda.is_available()
        else "self_cpu_time_total"
    )

    top5 = prof.key_averages(
        group_by_input_shape=True
    ).table(sort_by=sort_by, row_limit=5)
    top5_path = output_dir / "matmul_top5.txt"
    top5_path.write_text(top5 + "\n", encoding="utf-8")

    trace_path = output_dir / "matmul_trace.json"
    prof.export_chrome_trace(str(trace_path))

    print(top5)
    print(f"Top 5 saved to: {top5_path}")
    print(f"Trace saved to: {trace_path}")

def main() -> None:
    args = build_parser().parse_args()

    profile_matmul(
        size=args.size,
        device=args.device,
        dtype=DTYPE_MAP[args.dtype],
        warmup=args.warmup,
        steps=args.steps,
        trace_path=args.trace,
        top5_path=args.top5,
    )


if __name__ == "__main__":
    main()