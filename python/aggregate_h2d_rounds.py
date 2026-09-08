from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any



CASES = (
    "copy_only",
    "compute_only",
    "sequential",
    "pipeline",
)

FIXED_FIELDS = (
    "schema_version",
    "benchmark",
    "device",
    "device_name",
    "torch_version",
    "cuda_version",
    "dtype",
    "inner",
    "out_features",
    "total_bytes",
    "warmup",
    "repeats",
)

DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
}

NAME_PATTERN = re.compile(
    r"r(?P<round>\d+)_chunks_(?P<chunks>\d+)"
)

def percentile(
    samples: list[float],
    quantile: float,
) -> float:
    """使用与 benchmark_utils.py 相同的线性插值。"""

    if not samples:
        raise ValueError("samples must not be empty")

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


def assert_close(
    actual: float,
    expected: float,
    *,
    field: str,
    path: Path,
) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=1e-7,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"{path}: invalid {field}: "
            f"{actual} != {expected}"
        )

def parse_file_name(path: Path) -> tuple[int, int]:
    match = NAME_PATTERN.fullmatch(path.stem)

    if match is None:
        raise ValueError(
            f"invalid file name: {path.name}; "
            "expected r01_chunks_2.json"
        )

    round_id = int(match.group("round"))
    chunks = int(match.group("chunks"))

    return round_id, chunks


def validate_case(
    case: dict[str, Any],
    *,
    repeats: int,
    path: Path,
) -> None:
    case_name = case.get("case")

    if case_name not in CASES:
        raise ValueError(
            f"{path}: unsupported case {case_name!r}"
        )

    samples = case.get("samples_ms")

    if not isinstance(samples, list):
        raise TypeError(
            f"{path}: {case_name}.samples_ms "
            "must be a list"
        )

    if len(samples) != repeats:
        raise ValueError(
            f"{path}: {case_name} contains "
            f"{len(samples)} samples, expected {repeats}"
        )

    converted: list[float] = []

    for sample in samples:
        if (
            isinstance(sample, bool)
            or not isinstance(sample, (int, float))
        ):
            raise TypeError(
                f"{path}: invalid sample {sample!r}"
            )

        sample = float(sample)

        if not math.isfinite(sample) or sample < 0:
            raise ValueError(
                f"{path}: invalid sample {sample}"
            )

        converted.append(sample)

    expected_values = {
        "count": len(converted),
        "min_ms": min(converted),
        "mean_ms": statistics.fmean(converted),
        "median_ms": statistics.median(converted),
        "p95_ms": percentile(converted, 0.95),
        "max_ms": max(converted),
    }

    if case.get("count") != expected_values["count"]:
        raise ValueError(
            f"{path}: invalid {case_name}.count"
        )

    for field in (
        "min_ms",
        "mean_ms",
        "median_ms",
        "p95_ms",
        "max_ms",
    ):
        assert_close(
            float(case[field]),
            float(expected_values[field]),
            field=f"{case_name}.{field}",
            path=path,
        )


def load_run(path: Path) -> dict[str, Any]:
    round_id, chunks_from_name = parse_file_name(path)

    with path.open("r", encoding="utf-8") as file:
        document = json.load(file)

    if document.get("benchmark") != "h2d_pipeline":
        raise ValueError(
            f"{path}: not an h2d_pipeline result"
        )

    chunks = document.get("chunks")

    if chunks != chunks_from_name:
        raise ValueError(
            f"{path}: filename chunks={chunks_from_name}, "
            f"JSON chunks={chunks}"
        )

    dtype = document.get("dtype")

    if dtype not in DTYPE_BYTES:
        raise ValueError(
            f"{path}: unsupported dtype {dtype!r}"
        )

    rows = int(document["rows"])
    inner = int(document["inner"])
    chunk_bytes = int(document["chunk_bytes"])
    total_bytes = int(document["total_bytes"])

    expected_chunk_bytes = (
        rows * inner * DTYPE_BYTES[dtype]
    )

    if chunk_bytes != expected_chunk_bytes:
        raise ValueError(
            f"{path}: chunk_bytes mismatch: "
            f"{chunk_bytes} != {expected_chunk_bytes}"
        )

    if total_bytes != chunks * chunk_bytes:
        raise ValueError(
            f"{path}: total_bytes mismatch"
        )

    results_list = document.get("results")

    if not isinstance(results_list, list):
        raise TypeError(
            f"{path}: results must be a list"
        )

    case_map = {
        result["case"]: result
        for result in results_list
    }

    if (
        len(case_map) != len(results_list)
        or set(case_map) != set(CASES)
    ):
        raise ValueError(
            f"{path}: expected exactly these cases: "
            f"{CASES}"
        )

    repeats = int(document["repeats"])

    for case in case_map.values():
        validate_case(
            case,
            repeats=repeats,
            path=path,
        )

    sequential_median = float(
        case_map["sequential"]["median_ms"]
    )
    pipeline_median = float(
        case_map["pipeline"]["median_ms"]
    )

    expected_speedup = (
        sequential_median / pipeline_median
    )

    assert_close(
        float(document["pipeline_speedup"]),
        expected_speedup,
        field="pipeline_speedup",
        path=path,
    )

    expected_bandwidth = (
        total_bytes
        / (
            float(case_map["copy_only"]["median_ms"])
            / 1000.0
        )
        / 1_000_000_000.0
    )

    assert_close(
        float(document["effective_h2d_gbps"]),
        expected_bandwidth,
        field="effective_h2d_gbps",
        path=path,
    )

    document["_path"] = path
    document["_round"] = round_id
    document["_cases"] = case_map

    return document

def validate_protocol(
    runs: list[dict[str, Any]],
    *,
    expected_chunks: set[int],
    rounds: int,
) -> None:
    if not runs:
        raise ValueError("no benchmark files found")

    baseline = runs[0]

    for run in runs[1:]:
        path = run["_path"]

        for field in FIXED_FIELDS:
            if run.get(field) != baseline.get(field):
                raise ValueError(
                    f"{path}: protocol field {field!r} "
                    f"changed from {baseline.get(field)!r} "
                    f"to {run.get(field)!r}"
                )

    # 比较 chunk 粒度时，总数据行数必须保持不变。
    expected_total_rows = (
        baseline["chunks"] * baseline["rows"]
    )

    for run in runs:
        actual_total_rows = (
            run["chunks"] * run["rows"]
        )

        if actual_total_rows != expected_total_rows:
            raise ValueError(
                f"{run['_path']}: chunks * rows changed: "
                f"{actual_total_rows} != "
                f"{expected_total_rows}"
            )

    actual_pairs = {
        (run["_round"], run["chunks"])
        for run in runs
    }

    expected_pairs = {
        (round_id, chunks)
        for round_id in range(1, rounds + 1)
        for chunks in expected_chunks
    }

    missing = expected_pairs - actual_pairs
    extra = actual_pairs - expected_pairs

    if missing or extra:
        raise ValueError(
            f"experiment grid mismatch; "
            f"missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    if len(actual_pairs) != len(runs):
        raise ValueError(
            "duplicate round/chunks result detected"
        )

def summarize_values(
    values: list[float],
) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }

def aggregate_case(
    runs: list[dict[str, Any]],
    case_name: str,
) -> dict[str, Any]:
    run_medians = [
        float(run["_cases"][case_name]["median_ms"])
        for run in runs
    ]

    run_p95s = [
        float(run["_cases"][case_name]["p95_ms"])
        for run in runs
    ]

    return {
        "median_ms_across_rounds": (
            summarize_values(run_medians)
        ),
        "p95_ms_across_rounds": (
            summarize_values(run_p95s)
        ),
    }

def aggregate(
    runs: list[dict[str, Any]],
    expected_chunks: list[int],
) -> dict[str, Any]:
    baseline = runs[0]
    groups: list[dict[str, Any]] = []

    for chunks in expected_chunks:
        chunk_runs = sorted(
            (
                run
                for run in runs
                if run["chunks"] == chunks
            ),
            key=lambda run: run["_round"],
        )

        per_round = []

        for run in chunk_runs:
            per_round.append(
                {
                    "round": run["_round"],
                    "source": run["_path"].name,
                    "rows": run["rows"],
                    "chunk_bytes": run["chunk_bytes"],
                    "pipeline_speedup": (
                        run["pipeline_speedup"]
                    ),
                    "effective_h2d_gbps": (
                        run["effective_h2d_gbps"]
                    ),
                    "cases": {
                        case_name: {
                            "median_ms": float(
                                run["_cases"][
                                    case_name
                                ]["median_ms"]
                            ),
                            "p95_ms": float(
                                run["_cases"][
                                    case_name
                                ]["p95_ms"]
                            ),
                        }
                        for case_name in CASES
                    },
                }
            )

        groups.append(
            {
                "chunks": chunks,
                "rows_per_chunk": chunk_runs[0]["rows"],
                "round_count": len(chunk_runs),
                "cases": {
                    case_name: aggregate_case(
                        chunk_runs,
                        case_name,
                    )
                    for case_name in CASES
                },
                "pipeline_speedup_across_rounds": (
                    summarize_values(
                        [
                            float(
                                run["pipeline_speedup"]
                            )
                            for run in chunk_runs
                        ]
                    )
                ),
                "effective_h2d_gbps_across_rounds": (
                    summarize_values(
                        [
                            float(
                                run["effective_h2d_gbps"]
                            )
                            for run in chunk_runs
                        ]
                    )
                ),
                "per_round": per_round,
            }
        )

    best_group = min(
        groups,
        key=lambda group: group["cases"]["pipeline"][
            "median_ms_across_rounds"
        ]["median"],
    )

    return {
        "schema_version": 1,
        "benchmark": "h2d_pipeline_five_round_aggregate",
        "aggregation_rule": (
            "median of per-round medians; "
            "samples from different rounds are not pooled"
        ),
        "protocol": {
            field: baseline[field]
            for field in FIXED_FIELDS
        },
        "protocol_total_rows": (
            baseline["chunks"] * baseline["rows"]
        ),
        "groups": groups,
        "observed_best_chunks": best_group["chunks"],
        "limitation": (
            "Observed winner applies only to this GPU, "
            "dtype, shape and protocol."
        ),
    }

def print_summary(result: dict[str, Any]) -> None:
    print(
        f"{'chunks':>8} "
        f"{'pipeline MoM(ms)':>18} "
        f"{'round range(ms)':>18} "
        f"{'speedup median':>16}"
    )

    for group in result["groups"]:
        pipeline = group["cases"]["pipeline"][
            "median_ms_across_rounds"
        ]
        speedup = group[
            "pipeline_speedup_across_rounds"
        ]

        print(
            f"{group['chunks']:>8d} "
            f"{pipeline['median']:>18.4f} "
            f"{pipeline['range']:>18.4f} "
            f"{speedup['median']:>16.3f}"
        )

    print(
        "\nObserved best chunks:",
        result["observed_best_chunks"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate five-round H2D benchmark results"
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        default=[2, 8, 32],
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    paths = sorted(
        args.input_dir.glob("r*_chunks_*.json")
    )

    runs = [
        load_run(path)
        for path in paths
    ]

    validate_protocol(
        runs,
        expected_chunks=set(args.chunks),
        rounds=args.rounds,
    )

    result = aggregate(
        runs,
        expected_chunks=args.chunks,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print_summary(result)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()



