from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Result file does not exist: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported schema_version in {path}: "
            f"{payload.get('schema_version')}"
        )

    if payload.get("benchmark") != "square_matmul":
        raise ValueError(
            f"Unexpected benchmark in {path}: "
            f"{payload.get('benchmark')}"
        )

    results = payload.get("results")

    if not isinstance(results, list) or not results:
        raise ValueError(f"No benchmark results found in {path}")

    return payload


def get_size(result: dict[str, Any]) -> int:
    shape_a = result.get("shape_a")
    shape_b = result.get("shape_b")

    if (
        not isinstance(shape_a, list)
        or len(shape_a) != 2
        or shape_a[0] != shape_a[1]
    ):
        raise ValueError(f"Invalid shape_a: {shape_a}")

    if shape_b != shape_a:
        raise ValueError(
            f"Expected equal square matrices, got "
            f"shape_a={shape_a}, shape_b={shape_b}"
        )

    return int(shape_a[0])


def result_key(
    result: dict[str, Any],
) -> tuple[int, str, str]:
    return (
        get_size(result),
        str(result["dtype"]),
        str(result["boundary"]),
    )


def build_index(
    payload: dict[str, Any],
    expected_device_type: str,
) -> dict[tuple[int, str, str], dict[str, Any]]:
    index: dict[
        tuple[int, str, str],
        dict[str, Any],
    ] = {}

    for result in payload["results"]:
        device = str(result.get("device", ""))
        device_type = device.split(":", maxsplit=1)[0]

        if device_type != expected_device_type:
            raise ValueError(
                f"Expected {expected_device_type} result, "
                f"got device={device}"
            )

        key = result_key(result)

        if key in index:
            raise ValueError(f"Duplicate result: {key}")

        index[key] = result

    return index


def compare_results(
    cpu_payload: dict[str, Any],
    cuda_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    cpu_index = build_index(cpu_payload, "cpu")
    cuda_index = build_index(cuda_payload, "cuda")

    cpu_keys = set(cpu_index)
    cuda_keys = set(cuda_index)

    if cpu_keys != cuda_keys:
        only_cpu = sorted(cpu_keys - cuda_keys)
        only_cuda = sorted(cuda_keys - cpu_keys)

        raise ValueError(
            "CPU and CUDA result grids do not match. "
            f"Only CPU: {only_cpu}; "
            f"only CUDA: {only_cuda}"
        )

    comparisons: list[dict[str, Any]] = []

    for key in sorted(cpu_keys):
        cpu_result = cpu_index[key]
        cuda_result = cuda_index[key]

        cpu_median_ms = float(cpu_result["median_ms"])
        cuda_median_ms = float(cuda_result["median_ms"])

        if cuda_median_ms <= 0:
            raise ValueError(
                f"CUDA median_ms must be positive, got "
                f"{cuda_median_ms}"
            )

        speedup = cpu_median_ms / cuda_median_ms

        comparisons.append(
            {
                "size": key[0],
                "dtype": key[1],
                "boundary": key[2],
                "cpu_median_ms": cpu_median_ms,
                "cuda_median_ms": cuda_median_ms,
                "cpu_p95_ms": float(cpu_result["p95_ms"]),
                "cuda_p95_ms": float(cuda_result["p95_ms"]),
                "cpu_tflops": float(
                    cpu_result["estimated_tflops"]
                ),
                "cuda_tflops": float(
                    cuda_result["estimated_tflops"]
                ),
                "speedup": speedup,
            }
        )

    return comparisons


def print_table(
    comparisons: list[dict[str, Any]],
) -> None:
    header = (
        f"{'size':>6} "
        f"{'dtype':>10} "
        f"{'CPU median':>12} "
        f"{'CUDA median':>12} "
        f"{'CPU p95':>10} "
        f"{'CUDA p95':>10} "
        f"{'speedup':>9}"
    )

    print(header)
    print("-" * len(header))

    for result in comparisons:
        print(
            f"{result['size']:>6} "
            f"{result['dtype']:>10} "
            f"{result['cpu_median_ms']:>10.4f}ms "
            f"{result['cuda_median_ms']:>10.4f}ms "
            f"{result['cpu_p95_ms']:>8.4f}ms "
            f"{result['cuda_p95_ms']:>8.4f}ms "
            f"{result['speedup']:>8.2f}x"
        )


def save_comparison(
    comparisons: list[dict[str, Any]],
    cpu_path: Path,
    cuda_path: Path,
    output_path: Path,
) -> None:
    payload = {
        "schema_version": 1,
        "comparison": "cpu_vs_cuda_square_matmul",
        "cpu_source": str(cpu_path),
        "cuda_source": str(cuda_path),
        "results": comparisons,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare CPU and CUDA MatMul results."
    )

    parser.add_argument(
        "--cpu",
        type=Path,
        required=True,
        help="CPU benchmark JSON.",
    )

    parser.add_argument(
        "--cuda",
        type=Path,
        required=True,
        help="CUDA benchmark JSON.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Optional comparison JSON output.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    cpu_payload = load_payload(args.cpu)
    cuda_payload = load_payload(args.cuda)

    comparisons = compare_results(
        cpu_payload,
        cuda_payload,
    )

    print_table(comparisons)

    if args.output is not None:
        save_comparison(
            comparisons=comparisons,
            cpu_path=args.cpu,
            cuda_path=args.cuda,
            output_path=args.output,
        )

        print(f"\nSaved comparison to: {args.output}")


if __name__ == "__main__":
    main()