from __future__ import annotations

from pathlib import Path
from typing import Callable
import argparse
import torch

from .benchmark_utils import(
    DTYPE_MAP,
    save_result,
    summarize,
)

def validate_config(
    rows: int,
    inner: int,
    out_features: int,
    chunks: int,
    warmup: int,
    repeats: int,
    dtype: torch.dtype,
) -> None:
    sizes = {
        "rows": rows,
        "inner": inner,
        "out_features": out_features,
        "chunks": chunks,
        "warmup": warmup,
        "repeats": repeats,
    }

    for name, value in sizes.items():
        if isinstance(value, bool):
            raise TypeError(f"{name} must be int")

        if not isinstance(value, int):
            raise TypeError(f"{name} must be int")

    if rows <= 0:
        raise ValueError("rows must be positive")

    if inner <= 0:
        raise ValueError("inner must be positive")

    if out_features <= 0:
        raise ValueError(
            "out_features must be positive"
        )

    if chunks < 2:
        raise ValueError(
            "pipeline requires at least 2 chunks"
        )

    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    if repeats <= 0:
        raise ValueError("repeats must be > 0")

    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be torch.dtype")

    if not dtype.is_floating_point:
        raise TypeError(
            "dtype must be floating point"
        )

def time_on_stream(
    fn: Callable[[], None],
    stream: torch.cuda.Stream,
    warmup: int,
    repeats: int,
) -> list[float]:

    for _ in range(warmup):
        with torch.cuda.stream(stream):
            fn()

    stream.synchronize()

    start = torch.cuda.Event(
        enable_timing=True
    )
    end = torch.cuda.Event(
        enable_timing=True
    )

    samples_ms = []

    for _ in range(repeats):
        with torch.cuda.stream(stream):
            start.record()
            fn()
            end.record()

        end.synchronize()

        samples_ms.append(
            float(start.elapsed_time(end))
        )

    return samples_ms

class H2DPipeline:
    def __init__(
        self,
        host_chunks: list[torch.Tensor],
        device_slots: list[torch.Tensor],
        outputs: list[torch.Tensor],
        weight: torch.Tensor,
    ) -> None:
        if len(device_slots) != 2:
            raise ValueError(
                "pipeline requires two device slots"
            )

        self.host_chunks = host_chunks
        self.device_slots = device_slots
        self.outputs = outputs
        self.weight = weight

        self.copy_stream = torch.cuda.Stream()
        self.compute_stream = torch.cuda.Stream()

        self.copy_done = [
            torch.cuda.Event(),
            torch.cuda.Event(),
        ]

        self.compute_done = [
            torch.cuda.Event(),
            torch.cuda.Event(),
        ]

        self.start = torch.cuda.Event(
            enable_timing=True
        )

        self.end = torch.cuda.Event(
            enable_timing=True
        )
    
    def elapsed_ms(self) -> float:
        self.start.record(self.copy_stream)

        self.compute_stream.wait_event(
            self.start
        )

        for step, host_chunk in enumerate(
            self.host_chunks
        ):
            slot = step % 2

            if step >= 2:
                self.copy_stream.wait_event(
                    self.compute_done[slot]
                )
            
            with torch.cuda.stream(
                self.copy_stream
            ):
                self.device_slots[slot].copy_(
                    host_chunk,
                    non_blocking=True,
                )

                self.copy_done[slot].record()
            
            with torch.cuda.stream(
                self.compute_stream
            ):
                self.compute_stream.wait_event(
                    self.copy_done[slot]
                )

                torch.mm(
                    self.device_slots[slot],
                    self.weight,
                    out = self.outputs[step],
                )

                self.compute_done[slot].record()
        self.end.record(self.compute_stream)
        self.end.synchronize()

        return float(
            self.start.elapsed_time(self.end)
        )

def time_pipeline(
    pipeline: H2DPipeline,
    warmup: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmup):
        pipeline.elapsed_ms()
    
    return [
        pipeline.elapsed_ms()
        for _ in range(repeats)
    ]

def run_h2d_benchmark(
    rows: int,
    inner: int,
    out_features: int,
    chunks: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    warmup: int = 5,
    repeats: int = 30,
    seed: int = 17,
) -> dict:
    validate_config(
        rows,
        inner,
        out_features,
        chunks,
        warmup,
        repeats,
        dtype,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    device = torch.device(device)

    if device.type != "cuda":
        raise ValueError(
            "H2D benchmark requires CUDA"
        )

    torch.manual_seed(seed)

    with torch.cuda.device(device):
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
                device=device,
            )
            for _ in range(2)
        ]

        weight = torch.randn(
            inner,
            out_features,
            dtype=dtype,
            device=device,
        )

        outputs = [
            torch.empty(
                rows,
                out_features,
                dtype=dtype,
                device=device,
            )
            for _ in range(chunks)
        ]

        resident_chunks = [
            chunk.to(device)
            for chunk in host_chunks
        ]

        compute_outputs = [
            torch.empty_like(outputs[0])
            for _ in range(chunks)
        ]

        sequential_outputs = [
            torch.empty_like(outputs[0])
            for _ in range(chunks)
        ]

        torch.cuda.synchronize()

        copy_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.Stream()
        sequential_stream = torch.cuda.Stream()

        def copy_all() -> None:
            for step, host_chunk in enumerate(
                host_chunks
            ):
                slot = step % 2
            
                device_slots[slot].copy_(
                    host_chunk,
                    non_blocking=True
                )

        def compute_all() -> None:
            for step, device_chunk in enumerate(
                resident_chunks
            ):
                torch.mm(
                    device_chunk,
                    weight,
                    out=compute_outputs[step],
                )

        def sequential_all() -> None:
            device_buffer = device_slots[0]

            for step, host_chunk in enumerate(
                host_chunks
            ):
                device_buffer.copy_(
                    host_chunk,
                    non_blocking=True,
                )

                torch.mm(
                    device_buffer,
                    weight,
                    out=sequential_outputs[step],
                )

        pipeline = H2DPipeline(
            host_chunks=host_chunks,
            device_slots=device_slots,
            outputs=outputs,
            weight=weight
        )

        pipeline.elapsed_ms()

        expected = [
            chunk @ weight
            for chunk in resident_chunks
        ]

        for actual, reference in zip(
            outputs,
            expected,
        ):
            torch.testing.assert_close(
                actual,
                reference,
            )
        
        copy_samples = time_on_stream(
            copy_all,
            copy_stream,
            warmup,
            repeats
        )

        compute_samples = time_on_stream(
                compute_all,
                compute_stream,
                warmup,
                repeats,
            )

        sequential_samples = time_on_stream(
            sequential_all,
            sequential_stream,
            warmup,
            repeats,
        )

        pipeline_samples = time_pipeline(
            pipeline,
            warmup,
            repeats,
        )
    
    chunk_bytes = (
        rows
        * inner
        * torch.empty(
            (),
            dtype=dtype,
        ).element_size()
    )

    total_bytes = chunks * chunk_bytes
    copy_summary = summarize(copy_samples)
    compute_summary = summarize(
        compute_samples
    )
    sequential_summary = summarize(
        sequential_samples
    )
    pipeline_summary = summarize(
        pipeline_samples
    )

    copy_median_ms = float(
        copy_summary["median_ms"]
    )

    bandwidth_gbps = (
        total_bytes
        / (copy_median_ms / 1000.0)
        / 1_000_000_000.0
    )

    pipeline_speedup = (
        float(sequential_summary["median_ms"])
        / float(pipeline_summary["median_ms"])
    )

    return {
        "schema_version": 1,
        "benchmark": "h2d_pipeline",
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
        ),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "dtype": str(dtype).removeprefix(
            "torch."
        ),
        "rows": rows,
        "inner": inner,
        "out_features": out_features,
        "chunks": chunks,
        "chunk_bytes": chunk_bytes,
        "total_bytes": total_bytes,
        "warmup": warmup,
        "repeats": repeats,
        "effective_h2d_gbps": bandwidth_gbps,
        "pipeline_speedup": pipeline_speedup,
        "results": [
            {
                "case": "copy_only",
                "samples_ms": copy_samples,
                **copy_summary,
            },
            {
                "case": "compute_only",
                "samples_ms": compute_samples,
                **compute_summary,
            },
            {
                "case": "sequential",
                "samples_ms": (
                    sequential_samples
                ),
                **sequential_summary,
            },
            {
                "case": "pipeline",
                "samples_ms": (
                    pipeline_samples
                ),
                **pipeline_summary,
            },
        ],
    }

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark H2D pipeline"
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
        "--warmup",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    result = run_h2d_benchmark(
        rows=args.rows,
        inner=args.inner,
        out_features=args.out_features,
        chunks=args.chunks,
        dtype=DTYPE_MAP[args.dtype],
        device=args.device,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    save_result(result, args.output)


if __name__ == "__main__":
    main()
