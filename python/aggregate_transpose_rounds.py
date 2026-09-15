from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


KINDS = (
    "copy",
    "naive",
    "tiled",
    "padding",
)

FIXED_FIELDS = (
    "schema_version",
    "benchmark",
    "git_revision",
    "nvidia_driver_version",
    "storage_dtype",
    "accumulator_dtype",
    "measurement",
    "timer",
    "time_unit",
    "logical_bytes_rule",
    "warmup_batches",
    "repeats",
    "launches_per_sample",
    "block",
)


def require_finite_positive(
    value: Any,
    *,
    name: str,
    path: Path,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise TypeError(
            f"{path}: {name} must be numeric"
        )

    converted = float(value)

    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(
            f"{path}: invalid {name}={converted}"
        )

    return converted


def summarize(
    values: list[float],
) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize empty values")

    median = statistics.median(values)
    deviations = [
        abs(value - median)
        for value in values
    ]

    return {
        "count": len(values),
        "median": median,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "median_abs_deviation": statistics.median(
            deviations
        ),
    }


def case_key(
    result: dict[str, Any],
) -> tuple[int, int, str]:
    return (
        int(result["height"]),
        int(result["width"]),
        str(result["kind"]),
    )


def load_run(
    path: Path,
) -> dict[str, Any]:
    document = json.loads(
        path.read_text(encoding="utf-8")
    )

    if document.get("benchmark") != "transpose_f32":
        raise ValueError(
            f"{path}: expected benchmark=transpose_f32"
        )

    repeats = int(document["repeats"])
    results = document.get("results")

    if not isinstance(results, list):
        raise TypeError(
            f"{path}: results must be a list"
        )

    case_map: dict[
        tuple[int, int, str],
        dict[str, Any],
    ] = {}

    for result in results:
        key = case_key(result)
        height, width, kind = key

        if kind not in KINDS:
            raise ValueError(
                f"{path}: unsupported kind={kind!r}"
            )

        if height <= 0 or width <= 0:
            raise ValueError(
                f"{path}: invalid shape {height}x{width}"
            )

        if key in case_map:
            raise ValueError(
                f"{path}: duplicate case {key}"
            )

        median_ms = require_finite_positive(
            result.get("median_ms"),
            name=f"{key}.median_ms",
            path=path,
        )
        p95_ms = require_finite_positive(
            result.get("p95_batch_mean_ms"),
            name=f"{key}.p95_batch_mean_ms",
            path=path,
        )
        effective_gbps = require_finite_positive(
            result.get("effective_gbps"),
            name=f"{key}.effective_gbps",
            path=path,
        )

        samples = result.get("samples_ms")

        if not isinstance(samples, list):
            raise TypeError(
                f"{path}: {key}.samples_ms "
                "must be a list"
            )

        if len(samples) != repeats:
            raise ValueError(
                f"{path}: {key} has {len(samples)} "
                f"samples, expected {repeats}"
            )

        for index, sample in enumerate(samples):
            require_finite_positive(
                sample,
                name=f"{key}.samples_ms[{index}]",
                path=path,
            )

        logical_bytes = 2 * height * width * 4
        expected_gbps = (
            logical_bytes / (median_ms * 1_000_000.0)
        )

        if not math.isclose(
            effective_gbps,
            expected_gbps,
            rel_tol=1e-5,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"{path}: {key} effective_gbps "
                f"mismatch: {effective_gbps} "
                f"!= {expected_gbps}"
            )

        case_map[key] = {
            **result,
            "median_ms": median_ms,
            "p95_batch_mean_ms": p95_ms,
            "effective_gbps": effective_gbps,
        }

    document["_path"] = path
    document["_cases"] = case_map

    return document


def validate_protocol(
    runs: list[dict[str, Any]],
    *,
    expected_rounds: int,
) -> None:
    if len(runs) != expected_rounds:
        raise ValueError(
            f"expected {expected_rounds} runs, "
            f"found {len(runs)}"
        )

    run_ids = [
        str(run["run_id"])
        for run in runs
    ]

    if len(set(run_ids)) != len(run_ids):
        raise ValueError(
            f"duplicate run_id: {run_ids}"
        )

    baseline = runs[0]
    expected_cases = set(baseline["_cases"])

    for run in runs[1:]:
        path = run["_path"]

        for field in FIXED_FIELDS:
            if run.get(field) != baseline.get(field):
                raise ValueError(
                    f"{path}: protocol field "
                    f"{field!r} changed from "
                    f"{baseline.get(field)!r} to "
                    f"{run.get(field)!r}"
                )

        actual_cases = set(run["_cases"])

        if actual_cases != expected_cases:
            missing = expected_cases - actual_cases
            extra = actual_cases - expected_cases

            raise ValueError(
                f"{path}: case grid changed; "
                f"missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )


def aggregate(
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    runs = sorted(
        runs,
        key=lambda run: str(run["run_id"]),
    )

    baseline = runs[0]
    keys = set(baseline["_cases"])
    shapes = sorted({
        (height, width)
        for height, width, _ in keys
    })

    shape_groups: list[dict[str, Any]] = []

    for height, width in shapes:
        kind_mom: dict[str, float] = {}

        for kind in KINDS:
            key = (height, width, kind)

            run_medians = [
                float(run["_cases"][key]["median_ms"])
                for run in runs
            ]

            kind_mom[kind] = statistics.median(
                run_medians
            )

        naive_mom = kind_mom["naive"]
        result_groups: list[dict[str, Any]] = []

        for kind in KINDS:
            key = (height, width, kind)

            run_medians = [
                float(run["_cases"][key]["median_ms"])
                for run in runs
            ]
            run_p95s = [
                float(
                    run["_cases"][key][
                        "p95_batch_mean_ms"
                    ]
                )
                for run in runs
            ]

            mom_ms = statistics.median(run_medians)
            logical_bytes = 2 * height * width * 4
            effective_gbps = (
                logical_bytes
                / (mom_ms * 1_000_000.0)
            )

            per_run = []

            for run in runs:
                current = run["_cases"][key]
                naive = run["_cases"][
                    (height, width, "naive")
                ]

                per_run.append({
                    "run_id": run["run_id"],
                    "source": run["_path"].name,
                    "median_ms": current["median_ms"],
                    "p95_batch_mean_ms": (
                        current["p95_batch_mean_ms"]
                    ),
                    "effective_gbps": (
                        current["effective_gbps"]
                    ),
                    "speedup_vs_naive": (
                        float(naive["median_ms"])
                        / float(current["median_ms"])
                    ),
                })

            result_groups.append({
                "kind": kind,
                "run_median_ms": summarize(
                    run_medians
                ),
                "run_p95_batch_mean_ms": summarize(
                    run_p95s
                ),
                "effective_gbps_from_mom": (
                    effective_gbps
                ),
                "speedup_vs_naive_from_mom": (
                    naive_mom / mom_ms
                ),
                "per_run": per_run,
            })

        fastest_kind = min(
            KINDS,
            key=lambda kind: kind_mom[kind],
        )

        shape_groups.append({
            "height": height,
            "width": width,
            "logical_bytes": 2 * height * width * 4,
            "observed_fastest_kind": fastest_kind,
            "results": result_groups,
        })

    return {
        "schema_version": 1,
        "benchmark": (
            "transpose_f32_five_round_aggregate"
        ),
        "aggregation_rule": (
            "median of per-run medians; "
            "raw samples from different runs are not pooled"
        ),
        "round_count": len(runs),
        "source_files": [
            run["_path"].name
            for run in runs
        ],
        "protocol": {
            field: baseline.get(field)
            for field in FIXED_FIELDS
        },
        "shapes": shape_groups,
        "limitation": (
            "Observed results apply only to this GPU, "
            "driver, git revision and benchmark protocol."
        ),
    }


def print_summary(
    aggregate_result: dict[str, Any],
) -> None:
    print(
        f"{'shape':>12} "
        f"{'kind':>10} "
        f"{'MoM ms':>12} "
        f"{'range ms':>12} "
        f"{'GB/s':>12} "
        f"{'vs naive':>12}"
    )

    for shape in aggregate_result["shapes"]:
        shape_name = (
            f"{shape['height']}x{shape['width']}"
        )

        for result in shape["results"]:
            timing = result["run_median_ms"]

            print(
                f"{shape_name:>12} "
                f"{result['kind']:>10} "
                f"{timing['median']:>12.6f} "
                f"{timing['range']:>12.6f} "
                f"{result['effective_gbps_from_mom']:>12.3f} "
                f"{result['speedup_vs_naive_from_mom']:>12.3f}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate multi-run transpose benchmark JSON"
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--pattern",
        default="2026-09-14_f32_r*.json",
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
        args.input_dir.glob(args.pattern)
    )

    runs = [
        load_run(path)
        for path in paths
    ]

    validate_protocol(
        runs,
        expected_rounds=args.rounds,
    )

    result = aggregate(runs)

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