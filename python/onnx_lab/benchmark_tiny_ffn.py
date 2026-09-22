from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import onnxruntime as ort
import torch

from python.benchmark_utils import (
    save_result,
    summarize,
    time_cpu,
)
from python.onnx_lab.run_ort import (
    assert_ort_parity,
    load_reference_model,
    make_cpu_session,
)


def parse_shape(text: str) -> tuple[int, int, int]:
    parts = text.lower().split("x")

    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "shape must use BxTxD, for example 1x128x8"
        )

    shape = tuple(int(value) for value in parts)

    if any(value <= 0 for value in shape):
        raise argparse.ArgumentTypeError(
            "all shape dimensions must be positive"
        )

    return shape


def benchmark(
    *,
    model_path: Path,
    state_dict_path: Path,
    shapes: list[tuple[int, int, int]],
    warmup: int,
    repeats: int,
    threads: int,
) -> dict:
    torch.set_num_threads(threads)

    reference_model = load_reference_model(
        state_dict_path,
        d_model=8,
        d_ff=16,
    )

    session_start_ns = time.perf_counter_ns()

    session = make_cpu_session(
        model_path,
        threads=threads,
    )

    session_creation_ms = (
        time.perf_counter_ns() - session_start_ns
    ) / 1_000_000.0

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    cases = []

    for case_index, shape in enumerate(shapes):
        torch.manual_seed(1000 + case_index)
        x = torch.randn(shape, dtype=torch.float32)
        x_numpy = x.numpy()

        # Benchmark 之前先验证正确性。
        assert_ort_parity(
            reference_model,
            session,
            x,
        )

        with torch.inference_mode():
            pytorch_samples = time_cpu(
                lambda: reference_model(x),
                warmup=warmup,
                repeats=repeats,
            )

        ort_samples = time_cpu(
            lambda: session.run(
                [output_name],
                {input_name: x_numpy},
            ),
            warmup=warmup,
            repeats=repeats,
        )

        pytorch_summary = summarize(pytorch_samples)
        ort_summary = summarize(ort_samples)

        cases.append(
            {
                "shape": list(shape),
                "pytorch_cpu": {
                    "summary_ms": pytorch_summary,
                    "samples_ms": pytorch_samples,
                },
                "onnxruntime_cpu": {
                    "summary_ms": ort_summary,
                    "samples_ms": ort_samples,
                },
                "ort_over_pytorch_median": (
                    ort_summary["median_ms"]
                    / pytorch_summary["median_ms"]
                ),
            }
        )

    return {
        "schema_version": 1,
        "experiment": "tiny_ffn_pytorch_vs_ort_cpu",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
            "available_providers": ort.get_available_providers(),
            "selected_providers": session.get_providers(),
            "threads": threads,
        },
        "configuration": {
            "warmup": warmup,
            "repeats": repeats,
            "model_path": str(model_path),
            "session_creation_ms": session_creation_ms,
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--state-dict",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--shapes",
        type=parse_shape,
        nargs="+",
        default=[
            (1, 1, 8),
            (1, 128, 8),
            (8, 128, 8),
        ],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    result = benchmark(
        model_path=args.model,
        state_dict_path=args.state_dict,
        shapes=args.shapes,
        warmup=args.warmup,
        repeats=args.repeats,
        threads=args.threads,
    )

    save_result(result, args.output)

    for case in result["cases"]:
        pt = case["pytorch_cpu"]["summary_ms"]
        ort_result = case["onnxruntime_cpu"]["summary_ms"]

        print(
            f"shape={case['shape']} "
            f"PyTorch median={pt['median_ms']:.6f} ms "
            f"ORT median={ort_result['median_ms']:.6f} ms "
            f"ratio={case['ort_over_pytorch_median']:.3f}"
        )


if __name__ == "__main__":
    main()