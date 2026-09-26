from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import onnxruntime as ort

from python.benchmark_utils import (
    save_result,
    summarize,
)


LEVEL_ORDER = (
    "disable",
    "basic",
    "extended",
    "all",
)

LEVELS = {
    "disable":
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic":
        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended":
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all":
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}

def make_level_options(
    level: str,
    threads: int,
) -> ort.SessionOptions:
    if level not in LEVELS:
        raise ValueError(
            f"unknown optimization level: {level}"
        )

    if (
        isinstance(threads, bool)
        or not isinstance(threads, int)
        or threads <= 0
    ):
        raise ValueError(
            "threads must be a positive integer"
        )

    options = ort.SessionOptions()

    options.graph_optimization_level = (
        LEVELS[level]
    )

    options.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )

    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1

    # 必须显式关闭。
    options.enable_profiling = False

    # 不设置 optimized_model_filepath，
    # 避免模型写盘污染 Session 创建时间。

    return options


def make_level_session(
    model_path: Path,
    level: str,
    threads: int,
) -> ort.InferenceSession:
    if (
        "CPUExecutionProvider"
        not in ort.get_available_providers()
    ):
        raise RuntimeError(
            "CPUExecutionProvider is unavailable"
        )

    return ort.InferenceSession(
        str(model_path),
        sess_options=make_level_options(
            level,
            threads,
        ),
        providers=["CPUExecutionProvider"],
    )


def time_sample(
    run_once: Callable[[], Any],
    runs_per_sample: int,
) -> float:
    if (
        isinstance(runs_per_sample, bool)
        or not isinstance(runs_per_sample, int)
    ):
        raise TypeError(
            "runs_per_sample must be an integer"
        )

    if runs_per_sample <= 0:
        raise ValueError(
            "runs_per_sample must be positive"
        )

    start_ns = time.perf_counter_ns()

    for _ in range(runs_per_sample):
        run_once()

    elapsed_ms = (
        time.perf_counter_ns() - start_ns
    ) / 1_000_000.0

    return elapsed_ms / runs_per_sample

def benchmark_ort_levels(
    *,
    model_path: Path,
    shape: tuple[int, int, int],
    threads: int,
    warmup: int,
    repeats: int,
    runs_per_sample: int,
    seed: int,
    run_id: str | None = None,
) -> dict[str, Any]:
    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    if (
        len(shape) != 3
        or any(dim <= 0 for dim in shape)
    ):
        raise ValueError(
            "shape must be a positive rank-3 shape"
        )

    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    if repeats <= 0:
        raise ValueError("repeats must be positive")

    if runs_per_sample <= 0:
        raise ValueError(
            "runs_per_sample must be positive"
        )

    rng = np.random.default_rng(seed)
    input_array = (
        rng.standard_normal(shape)
        .astype(np.float32)
    )

    sessions: dict[
        str,
        ort.InferenceSession,
    ] = {}

    session_creation_ms: dict[str, float] = {}

    # 每个优化级别使用独立 Session。
    for level in LEVEL_ORDER:
        start_ns = time.perf_counter_ns()

        sessions[level] = make_level_session(
            model_path,
            level,
            threads,
        )

        session_creation_ms[level] = (
            time.perf_counter_ns() - start_ns
        ) / 1_000_000.0

    reference_session = sessions["disable"]

    inputs = reference_session.get_inputs()
    outputs = reference_session.get_outputs()

    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            "benchmark expects one input "
            "and one output"
        )

    input_name = inputs[0].name
    output_name = outputs[0].name
    feeds = {input_name: input_array}

    # -------------------------
    # 正确性检查
    # -------------------------

    reference = reference_session.run(
        [output_name],
        feeds,
    )[0]

    parity: dict[str, dict[str, float]] = {}

    for level in LEVEL_ORDER:
        actual = sessions[level].run(
            [output_name],
            feeds,
        )[0]

        np.testing.assert_allclose(
            actual,
            reference,
            rtol=1e-4,
            atol=1e-5,
        )

        error = np.abs(actual - reference)

        parity[level] = {
            "max_abs_error":
                float(error.max(initial=0.0)),
            "mean_abs_error":
                float(error.mean()),
        }

    # -------------------------
    # 每个 Session 独立 Warmup
    # -------------------------

    for level in LEVEL_ORDER:
        session = sessions[level]

        for _ in range(warmup):
            session.run([output_name], feeds)

    samples = {
        level: []
        for level in LEVEL_ORDER
    }

    sample_orders: list[list[str]] = []

    # -------------------------
    # 正式轮转采样
    # -------------------------

    for sample_index in range(repeats):
        offset = (
            sample_index % len(LEVEL_ORDER)
        )

        order = list(
            LEVEL_ORDER[offset:]
            + LEVEL_ORDER[:offset]
        )

        sample_orders.append(order)

        for level in order:
            session = sessions[level]

            # 默认参数绑定当前 Session，
            # 避免 lambda/闭包晚绑定问题。
            def run_once(
                current=session,
            ) -> None:
                current.run(
                    [output_name],
                    feeds,
                )

            latency_ms = time_sample(
                run_once,
                runs_per_sample,
            )

            samples[level].append(latency_ms)

    level_results = {}

    for level in LEVEL_ORDER:
        level_results[level] = {
            "session_creation_ms":
                session_creation_ms[level],

            "selected_providers":
                sessions[level].get_providers(),

            "parity":
                parity[level],

            "samples_ms_per_run":
                samples[level],

            "summary_ms_per_run":
                summarize(samples[level]),
        }

    return {
        "schema_version": 1,
        "experiment":
            "ort_optimization_levels_cpu",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "run_id": run_id,
        "profile_enabled": False,

        "sample_unit": (
            "mean_ms_per_ort_run_"
            "within_batched_sample"
        ),

        "model": {
            "path": str(model_path),
            "size": model_path.stat().st_size,
            "sha256": sha256_file(model_path),
        },

        "environment": {
            "onnxruntime": ort.__version__,
            "available_providers":
                ort.get_available_providers(),
            "selected_providers": [
                "CPUExecutionProvider"
            ],
        },

        "configuration": {
            "shape": list(shape),
            "dtype": "float32",
            "threads": threads,
            "execution_mode": "sequential",
            "warmup": warmup,
            "repeats": repeats,
            "runs_per_sample":
                runs_per_sample,
            "seed": seed,
            "level_order_policy":
                "rotating_start",
            "rtol": 1e-4,
            "atol": 1e-5,
        },

        "sample_orders": sample_orders,
        "levels": level_results,
    }

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()

def parse_shape(
    text: str,
) -> tuple[int, int, int]:
    try:
        shape = tuple(
            int(part)
            for part in text.lower().split("x")
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "shape must use BxTxD integers"
        ) from exc

    if (
        len(shape) != 3
        or any(dim <= 0 for dim in shape)
    ):
        raise argparse.ArgumentTypeError(
            "shape must be a positive "
            "rank-3 BxTxD shape"
        )

    return shape


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark four ONNX Runtime "
            "CPU optimization levels"
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--shape",
        type=parse_shape,
        default=(1, 128, 8),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--runs-per-sample",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)

    result = benchmark_ort_levels(
        model_path=args.model,
        shape=args.shape,
        threads=args.threads,
        warmup=args.warmup,
        repeats=args.repeats,
        runs_per_sample=args.runs_per_sample,
        seed=args.seed,
        run_id=args.run_id,
    )

    output = save_result(
        result,
        args.output,
    )

    print(f"result: {output}")

    for level in LEVEL_ORDER:
        summary = result[
            "levels"
        ][level]["summary_ms_per_run"]

        print(
            f"{level:8s} "
            f"median="
            f"{summary['median_ms']:.6f} ms "
            f"p95="
            f"{summary['p95_ms']:.6f} ms"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
