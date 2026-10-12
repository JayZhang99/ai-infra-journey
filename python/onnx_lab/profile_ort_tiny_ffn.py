from __future__ import annotations 

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any
import numpy as np
from python.benchmark_utils import save_result
import onnx
import onnxruntime as ort

LEVELS = {
    'disable': ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    'basic': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    'extended': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    'all': ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}



def parse_shape(text: str) -> tuple[int, int, int]:
    parts = text.lower().split("x")

    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "shape must use BxTxD, for example 1x128x8"
        )

    try:
        shape = tuple(int(value) for value in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "shape dimensions must be integers"
        ) from exc

    if any(value <= 0 for value in shape):
        raise argparse.ArgumentTypeError(
            "shape dimensions must be positive"
        )

    return shape


def make_providers(
    device: str,
    device_id: int = 0,
):
    available = list(ort.get_available_providers())

    if device == "cpu":
        if "CPUExecutionProvider" not in available:
            raise RuntimeError("CPUExecutionProvider is unavailable")
        return ["CPUExecutionProvider"]

    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable; "
                f"available providers: {available}"
            )

        return [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": device_id,
                    "do_copy_in_default_stream": True,
                },
            ),
            # 保留 CPU EP，允许 ORT 对 CUDA 不支持的节点 fallback。
            "CPUExecutionProvider",
        ]

    raise ValueError(f"unsupported device: {device!r}")

def make_options(*, level, threads, trace_prefix, optimized):
    if level not in LEVELS:
        raise ValueError(
            f"unknown optimization level: {level}"
        )

    if isinstance(threads, bool) or threads <= 0:
        raise ValueError("threads must be a positive integer")

    trace_prefix.parent.mkdir(parents=True, exist_ok=True)

    options = ort.SessionOptions()
    options.graph_optimization_level = LEVELS[level]
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.enable_profiling = True
    options.profile_file_prefix = str(trace_prefix)

    if optimized is not None:
        optimized.parent.mkdir(parents=True, exist_ok=True)
        options.optimized_model_filepath = str(optimized)

    return options

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()

def summarize_node_assignments(
    node_events: list[dict[str, Any]],
) -> dict[str, Any]:
    by_node: dict[str, dict[str, Any]] = {}

    for event in node_events:
        args = event.get("args") or {}

        provider = str(
            args.get("provider", "unknown")
        )
        op_name = str(
            args.get("op_name", "unknown")
        )
        node_index = args.get("node_index")
        event_name = str(
            event.get("name", "unknown")
        )
        duration_us = float(
            event.get("dur", 0.0)
        )

        # node_index 是跨多个 step 识别同一节点的首选字段。
        if node_index is not None:
            node_key = f"index:{node_index}"
        else:
            # 某些 ORT 版本可能没有 node_index。
            node_key = f"name:{event_name}"

        record = by_node.get(node_key)

        if record is None:
            by_node[node_key] = {
                "node_key": node_key,
                "node_index": (
                    str(node_index)
                    if node_index is not None
                    else None
                ),
                "node_name": event_name,
                "op_name": op_name,
                "provider": provider,
                "event_count": 1,
                "total_profile_dur_us": duration_us,
            }
            continue

        # 同一个图节点不应在不同 step 被分配给不同 Provider。
        if record["provider"] != provider:
            raise RuntimeError(
                "inconsistent provider assignment for "
                f"{node_key}: "
                f"{record['provider']} versus {provider}"
            )

        if record["op_name"] != op_name:
            raise RuntimeError(
                "inconsistent op name for "
                f"{node_key}: "
                f"{record['op_name']} versus {op_name}"
            )

        record["event_count"] += 1
        record["total_profile_dur_us"] += duration_us

    assignments = sorted(
        by_node.values(),
        key=lambda item: (
            int(item["node_index"])
            if (
                item["node_index"] is not None
                and item["node_index"].isdigit()
            )
            else 1 << 30,
            item["node_name"],
        ),
    )

    unique_provider_counts = Counter(
        item["provider"]
        for item in assignments
    )

    cpu_fallback_nodes = [
        item
        for item in assignments
        if item["provider"] == "CPUExecutionProvider"
    ]

    cuda_nodes = [
        item
        for item in assignments
        if item["provider"] == "CUDAExecutionProvider"
    ]

    return {
        "node_assignments": assignments,
        "unique_provider_counts": dict(
            sorted(unique_provider_counts.items())
        ),
        "cpu_fallback_nodes": cpu_fallback_nodes,
        "assignment_summary": {
            "unique_node_count": len(assignments),
            "cuda_unique_node_count": len(cuda_nodes),
            "cpu_unique_node_count": len(cpu_fallback_nodes),
            "all_unique_nodes_on_cuda": (
                bool(assignments)
                and not cpu_fallback_nodes
                and len(cuda_nodes) == len(assignments)
            ),
        },
    }


def summarize_events(
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    node_events = [
        event
        for event in events
        if event.get("cat") == "Node"
        and event.get("ph") == "X"
    ]

    assignment_result = summarize_node_assignments(
        node_events)

    session_events = [
        event
        for event in events
        if event.get("cat") == "Session"
        and event.get("ph") == "X"
    ]

    provider_counts: Counter[str] = Counter()
    op_counts: Counter[str] = Counter()
    op_total_dur_us: Counter[str] = Counter()
    session_event_counts: Counter[str] = Counter()

    for event in node_events:
        args = event.get("args") or {}
        provider = str(args.get("provider") or "unknown")
        op_name = str(args.get("op_name") or "unknown")
        duration_us = float(event.get("dur") or 0.0)

        provider_counts[provider] += 1
        op_counts[op_name] += 1
        op_total_dur_us[op_name] += duration_us

    for event in session_events:
        name = str(event.get("name") or "unknown")
        session_event_counts[name] += 1

    top_ops = sorted(
        op_total_dur_us.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:5]

    return {
        "event_count": len(events),
        "node_event_count": len(node_events),
        "session_event_count": len(session_events),
        "session_event_counts": dict(session_event_counts),
        "provider_counts": dict(provider_counts),
        "op_counts": dict(op_counts),
        "op_total_dur_us": dict(op_total_dur_us),
        "top_ops_by_profile_dur_us": [
            {
                "op_name": op_name,
                "dur_us": duration_us,
            }
            for op_name, duration_us in top_ops
        ],
        **assignment_result,
    }

def profile_model(
    *,
    model_path: Path,
    shape: tuple[int, ...],
    steps: int,
    threads: int,
    level: str,
    trace_path: Path,
    summary_path: Path,
    optimized_model_path: Path | None,
    seed: int = 0,
    device: str = "cpu",
    device_id: int = 0,
) -> dict[str, Any]:
    model_path = Path(model_path)
    trace_path = Path(trace_path)
    summary_path = Path(summary_path)

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("shape must be a positive rank-3 shape")

    if shape[-1] != 8:
        raise ValueError(
            f"TinyFFN hidden dimension must be 8, got {shape[-1]}"
        )

    if isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive integer")

    if isinstance(threads, bool) or threads <= 0:
        raise ValueError("threads must be a positive integer")

    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CPUExecutionProvider is unavailable")

    # Session 创建前确认输入模型本身有效。
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model, full_check=True)

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    trace_prefix = (
        trace_path.parent
        / f"{trace_path.stem}_generated"
    )

    options = make_options(
        level=level,
        threads=threads,
        trace_prefix=trace_prefix,
        optimized=optimized_model_path,
    )

    session_start_ns = time.perf_counter_ns()

    providers = make_providers(
        device=device,
        device_id=device_id,
    )

    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=providers,
    )

    selected_providers = list(session.get_providers())

    if device == "cuda":
        if not selected_providers:
            raise RuntimeError("ORT Session returned no providers")

        if selected_providers[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                "CUDAExecutionProvider is not the highest-priority provider: "
                f"{selected_providers}"
            )

    session_creation_ms = (
        time.perf_counter_ns() - session_start_ns
    ) / 1_000_000.0

    inputs = session.get_inputs()
    outputs_meta = session.get_outputs()

    if len(inputs) != 1 or len(outputs_meta) != 1:
        raise RuntimeError(
            "TinyFFN profile expects exactly one input and one output"
        )

    input_meta = inputs[0]
    output_meta = outputs_meta[0]

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32)

    outputs = None

    # 即使 session.run 失败，也明确结束 profiler；
    # 但异常发生时不会继续生成伪 summary。
    try:
        for _ in range(steps):
            outputs = session.run(
                [output_meta.name],
                {input_meta.name: x},
            )
    finally:
        generated_trace = Path(session.end_profiling())

    if outputs is None:
        raise RuntimeError("ORT did not produce an output")

    actual = np.asarray(outputs[0])

    if actual.shape != shape:
        raise RuntimeError(
            f"unexpected output shape: {actual.shape}, expected {shape}"
        )

    if not np.isfinite(actual).all():
        raise RuntimeError("ORT produced non-finite values")

    if not generated_trace.is_file():
        raise RuntimeError(
            f"ORT did not create its trace: {generated_trace}"
        )

    if generated_trace.resolve() != trace_path.resolve():
        generated_trace.replace(trace_path)

    payload = json.loads(
        trace_path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, list):
        raise TypeError(
            "ORT trace must be a JSON event list"
        )

    event_summary = summarize_events(payload)

    if event_summary["node_event_count"] <= 0:
        raise RuntimeError(
            "trace contains no complete Node events"
        )

    optimized_artifact = None

    if optimized_model_path is not None:
        if not optimized_model_path.is_file():
            raise RuntimeError(
                "ORT did not save the optimized model"
            )

        optimized_artifact = {
            "path": str(optimized_model_path),
            "size": optimized_model_path.stat().st_size,
            "sha256": sha256_file(optimized_model_path),
        }

    result = {
        "schema_version": 1,
        "experiment": f"tiny_ffn_ort_{device}_profile",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "environment": {
            "onnxruntime": ort.__version__,
            "available_providers": ort.get_available_providers(),
            "selected_providers": session.get_providers(),
        },
        "configuration": {
            "shape": list(shape),
            "dtype": "float32",
            "steps": steps,
            "threads": threads,
            "optimization_level": level,
            "seed": seed,
            "device": device,
            "device_id": device_id if device == "cuda" else None,
        },
        "session": {
            "creation_ms": session_creation_ms,
            "input_name": input_meta.name,
            "input_type": input_meta.type,
            "input_shape": input_meta.shape,
            "output_name": output_meta.name,
            "output_type": output_meta.type,
            "output_shape": output_meta.shape,
        },
        "artifacts": {
            "model": {
                "path": str(model_path),
                "size": model_path.stat().st_size,
                "sha256": sha256_file(model_path),
                "original_node_count": len(model.graph.node),
            },
            "trace": {
                "path": str(trace_path),
                "size": trace_path.stat().st_size,
                "sha256": sha256_file(trace_path),
            },
            "optimized_model": optimized_artifact,
        },
        **event_summary,
    }

    # 所有机械检查完成后才写 summary。
    save_result(result, summary_path)
    return result

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile TinyFFN with ONNX Runtime CPU/CUDA EP"
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--shape",
        type=parse_shape,
        default=(1, 128, 8),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--optimization",
        choices=sorted(LEVELS),
        default="all",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--optimized-model",
        type=Path,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )

    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    result = profile_model(
        model_path=args.model,
        shape=args.shape,
        steps=args.steps,
        threads=args.threads,
        level=args.optimization,
        trace_path=args.trace,
        summary_path=args.summary,
        optimized_model_path=args.optimized_model,
        seed=args.seed,
        device=args.device,
        device_id=args.device_id,
    )

    print(f"trace: {result['artifacts']['trace']['path']}")
    print(f"providers: {result['environment']['selected_providers']}")
    print(f"node events: {result['node_event_count']}")
    print(f"top ops: {result['top_ops_by_profile_dur_us']}")


if __name__ == "__main__":
    main()





