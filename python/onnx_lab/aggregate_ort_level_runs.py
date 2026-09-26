from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


LEVEL_ORDER = (
    "disable",
    "basic",
    "extended",
    "all",
)

CONFIG_FIELDS = (
    "shape",
    "dtype",
    "threads",
    "execution_mode",
    "warmup",
    "repeats",
    "runs_per_sample",
    "seed",
    "level_order_policy",
    "rtol",
    "atol",
)


def percentile(
    values: list[float],
    quantile: float,
) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]

    fraction = position - lower
    return (
        ordered[lower]
        + (ordered[upper] - ordered[lower]) * fraction
    )


def summarize(values: list[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("samples must not be empty")
    if any(not math.isfinite(value) for value in samples):
        raise ValueError("samples must be finite")
    if any(value < 0 for value in samples):
        raise ValueError("samples must be non-negative")

    return {
        "count": len(samples),
        "min_ms": min(samples),
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "max_ms": max(samples),
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


def assert_close(
    actual: float,
    expected: float,
    *,
    field: str,
) -> None:
    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"stored {field} does not match raw samples: "
            f"stored={actual}, recomputed={expected}"
        )


def contract_signature(data: dict[str, Any]) -> dict[str, Any]:
    config = data["configuration"]
    environment = data["environment"]
    return {
        "model_sha256": data["model"]["sha256"],
        "model_size": data["model"]["size"],
        "onnxruntime": environment["onnxruntime"],
        "selected_providers": environment[
            "selected_providers"
        ],
        "sample_unit": data["sample_unit"],
        **{
            field: config[field]
            for field in CONFIG_FIELDS
        },
    }


def validate_run(data: dict[str, Any]) -> None:
    if data.get("schema_version") != 1:
        raise ValueError("unsupported schema_version")
    if data.get("experiment") != "ort_optimization_levels_cpu":
        raise ValueError("unexpected experiment")
    if data.get("profile_enabled") is not False:
        raise ValueError("benchmark must run with profiling disabled")

    run_id = data.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")

    config = data["configuration"]
    repeats = config["repeats"]
    if isinstance(repeats, bool) or not isinstance(repeats, int):
        raise ValueError("repeats must be an integer")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    levels = data["levels"]
    if set(levels) != set(LEVEL_ORDER):
        raise ValueError("levels must contain disable/basic/extended/all")

    orders = data["sample_orders"]
    if len(orders) != repeats:
        raise ValueError("sample_orders count does not match repeats")

    for index, order in enumerate(orders):
        expected = list(
            LEVEL_ORDER[index % len(LEVEL_ORDER):]
            + LEVEL_ORDER[:index % len(LEVEL_ORDER)]
        )
        if order != expected:
            raise ValueError(
                f"sample order mismatch at sample {index}"
            )

    selected = data["environment"]["selected_providers"]
    for level in LEVEL_ORDER:
        row = levels[level]
        if row["selected_providers"] != selected:
            raise ValueError(
                f"provider mismatch for level {level}"
            )

        samples = row["samples_ms_per_run"]
        if len(samples) != repeats:
            raise ValueError(
                f"sample count mismatch for level {level}"
            )

        recomputed = summarize(samples)
        stored = row["summary_ms_per_run"]
        if stored["count"] != recomputed["count"]:
            raise ValueError(
                f"stored count mismatch for level {level}"
            )

        for field in (
            "min_ms",
            "mean_ms",
            "median_ms",
            "p95_ms",
            "max_ms",
        ):
            assert_close(
                stored[field],
                recomputed[field],
                field=f"{level}.{field}",
            )


def validate_compatible(
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = contract_signature(runs[0])
    for run in runs[1:]:
        current = contract_signature(run)
        changed = [
            key
            for key in baseline
            if baseline[key] != current[key]
        ]
        if changed:
            raise ValueError(
                f"incompatible run {run['run_id']}: "
                + ", ".join(changed)
            )
    return baseline


def aggregate_runs(
    runs: list[dict[str, Any]],
    *,
    sources: list[dict[str, Any]] | None = None,
    expected_rounds: int = 5,
) -> dict[str, Any]:
    if len(runs) != expected_rounds:
        raise ValueError(
            f"expected {expected_rounds} runs, got {len(runs)}"
        )

    for run in runs:
        validate_run(run)

    run_ids = [run["run_id"] for run in runs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("duplicate run_id")

    contract = validate_compatible(runs)
    winner_counts: Counter[str] = Counter()
    round_results: list[dict[str, Any]] = []

    for run in runs:
        per_level = {
            level: summarize(
                run["levels"][level]["samples_ms_per_run"]
            )
            for level in LEVEL_ORDER
        }
        best_median = min(
            row["median_ms"]
            for row in per_level.values()
        )
        winners = [
            level
            for level in LEVEL_ORDER
            if math.isclose(
                per_level[level]["median_ms"],
                best_median,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ]
        winner_counts.update(winners)
        round_results.append({
            "run_id": run["run_id"],
            "winner_levels": winners,
            "levels": per_level,
        })

    aggregated_levels: dict[str, Any] = {}
    for level in LEVEL_ORDER:
        medians = [
            row["levels"][level]["median_ms"]
            for row in round_results
        ]
        p95s = [
            row["levels"][level]["p95_ms"]
            for row in round_results
        ]
        means = [
            row["levels"][level]["mean_ms"]
            for row in round_results
        ]

        aggregated_levels[level] = {
            "round_medians_ms": medians,
            "median_of_medians_ms": statistics.median(medians),
            "median_min_ms": min(medians),
            "median_max_ms": max(medians),
            "median_range_ms": max(medians) - min(medians),
            "round_p95_ms": p95s,
            "median_of_p95_ms": statistics.median(p95s),
            "p95_min_ms": min(p95s),
            "p95_max_ms": max(p95s),
            "p95_range_ms": max(p95s) - min(p95s),
            "median_of_means_ms": statistics.median(means),
            "winner_count": winner_counts[level],
        }

    baseline = aggregated_levels["disable"][
        "median_of_medians_ms"
    ]
    for level in LEVEL_ORDER:
        value = aggregated_levels[level][
            "median_of_medians_ms"
        ]
        delta = value - baseline
        aggregated_levels[level][
            "delta_vs_disable_ms"
        ] = delta
        aggregated_levels[level][
            "delta_vs_disable_percent"
        ] = (
            delta / baseline * 100.0
            if baseline != 0.0
            else None
        )

    best_level = min(
        LEVEL_ORDER,
        key=lambda level: aggregated_levels[level][
            "median_of_medians_ms"
        ],
    )

    return {
        "schema_version": 1,
        "experiment": "ort_optimization_levels_cpu_5round_aggregate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_count": len(runs),
        "comparison_contract": contract,
        "source_files": sources or [],
        "run_ids": run_ids,
        "round_results": round_results,
        "levels": aggregated_levels,
        "best_level_by_median_of_medians": best_level,
        "interpretation": {
            "statement": (
                "The best level is descriptive for these runs only; "
                "it is not proof that the optimization level caused "
                "the latency difference."
            ),
            "next_test": (
                "Repeat on a larger workload and compare the observed "
                "delta with cross-round drift."
            ),
        },
    }


def aggregate_files(
    paths: Sequence[Path],
    *,
    expected_rounds: int = 5,
) -> dict[str, Any]:
    normalized = [Path(path) for path in paths]
    resolved = [path.resolve() for path in normalized]
    if len(set(resolved)) != len(resolved):
        raise ValueError("duplicate input path")

    runs: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in normalized:
        if not path.is_file():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"expected JSON object: {path}")
        runs.append(data)
        sources.append({
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "run_id": data.get("run_id"),
        })

    return aggregate_runs(
        runs,
        sources=sources,
        expected_rounds=expected_rounds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate repeated ORT optimization-level benchmarks"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--expected-rounds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = aggregate_files(
        args.inputs,
        expected_rounds=args.expected_rounds,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    print(f"aggregate: {args.output}")
    print(
        "best level by median-of-medians: "
        f"{result['best_level_by_median_of_medians']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
