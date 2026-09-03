from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
)

from .benchmark_h2d import H2DPipeline
from .benchmark_utils import DTYPE_MAP

def profile_h2d(
    *,
    rows: int,
    inner: int,
    out_features: int,
    chunks: int,
    active_steps: int,
    trace_path: Path,
    top_path: Path,
    capture_mode: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    if capture_mode not in {"full","light"}:
        raise ValueError("invalid capture mode")

    cuda_device = torch.device(device)

    if cuda_device.type != "cuda":
        raise ValueError("device must be CUDA")
    
    full = capture_mode == "full"

    trace_path.parent.mkdir(
        parents=True,
        exist_ok=True,)

    top_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.manual_seed(17)

    host_chunks = [
        torch.randn(
            rows,
            inner,
            dtype=dtype,
            pin_memory=True,
        )
        for _ in range(chunks)
    ]

    device_slots = [
        torch.empty(
            rows,
            inner,
            dtype=dtype,
            device=cuda_device,
        )
        for _ in range(2)
    ]

    weight = torch.randn(
        inner,
        out_features,
        dtype=dtype,
        device=cuda_device,
    )

    outputs = [
        torch.empty(
            rows,
            out_features,
            dtype=dtype,
            device=cuda_device,
        )
        for _ in range(chunks)
    ]

    pipeline = H2DPipeline(
        host_chunks=host_chunks,
        device_slots=device_slots,
        outputs=outputs,
        weight=weight,
    )

    for _ in range(3):
        pipeline.elapsed_ms()

    torch.cuda.synchronize(cuda_device)

    full = capture_mode == "full"

    def trace_handler(profiler) -> None:
            # 只能在这里导出一次 trace。
            profiler.export_chrome_trace(
                str(trace_path)
            )

            table = profiler.key_averages(
                group_by_input_shape=full,
            ).table(
                sort_by="self_cuda_time_total",
                row_limit=10,
            )

            top_path.write_text(
                table,
                encoding="utf-8",
            )

    profiler_wait = 1
    profiler_warmup = 1

    profiler_schedule = torch.profiler.schedule(
        wait=profiler_wait,
        warmup=profiler_warmup,
        active=active_steps,
        repeat=1,
    )

    total_steps = (
        profiler_wait
        + profiler_warmup
        + active_steps
        )
    
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=profiler_schedule,
        on_trace_ready=trace_handler,
        record_shapes=full,
        profile_memory=full,
        with_stack=False,
    ) as prof:
        for _ in range(total_steps):
            with record_function(
                "h2d_compute_pipeline"
            ):
                pipeline.elapsed_ms()

            # 告诉 Profiler：
            # 一次逻辑迭代已经结束。
            prof.step()

        print(f"Trace: {trace_path}")
        print(f"Top operators: {top_path}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile H2D compute pipeline"
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )
    parser.add_argument(
        "--rows",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--inner",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--out-features",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPE_MAP.keys(),
        default="float32",
    )
    parser.add_argument(
        "--active-steps",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--capture-mode",
        choices=["full", "light"],
        default="light",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--top",
        type=Path,
        required=True,
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    profile_h2d(
        rows=args.rows,
        inner=args.inner,
        out_features=args.out_features,
        chunks=args.chunks,
        dtype=DTYPE_MAP[args.dtype],
        device=args.device,
        active_steps=args.active_steps,
        capture_mode=args.capture_mode,
        trace_path=args.trace,
        top_path=args.top,
    )


if __name__ == "__main__":
    main()