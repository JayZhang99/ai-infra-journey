from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import pytest

from python.aggregate_transpose_rounds import (
    KINDS,
    aggregate,
    load_run,
    main,
    summarize,
    validate_protocol,
)


BASE_PROTOCOL = {
    "schema_version": 2,
    "benchmark": "transpose_f32",
    "git_revision": "abc1234",
    "nvidia_driver_version": "591.86",
    "storage_dtype": "float32",
    "accumulator_dtype": None,
    "measurement": "batched_average_per_operation",
    "timer": "cuda_event_same_stream",
    "time_unit": "ms",
    "logical_bytes_rule": "2*height*width*sizeof(float)",
    "warmup_batches": 2,
    "repeats": 3,
    "launches_per_sample": 10,
    "block": [32, 8, 1],
}


def make_result(
    *,
    height: int,
    width: int,
    kind: str,
    median_ms: float,
    samples_ms: list[float] | None = None,
) -> dict:
    if samples_ms is None:
        samples_ms = [
            median_ms * 0.9,
            median_ms,
            median_ms * 1.1,
        ]

    logical_bytes = 2 * height * width * 4
    effective_gbps = logical_bytes / (median_ms * 1_000_000.0)

    return {
        "height": height,
        "width": width,
        "kind": kind,
        "median_ms": median_ms,
        "p95_batch_mean_ms": max(samples_ms),
        "effective_gbps": effective_gbps,
        "samples_ms": samples_ms,
    }


def write_run(
    directory: Path,
    *,
    run_id: str,
    medians: dict[str, float],
    protocol_changes: dict | None = None,
    samples_by_kind: dict[str, list[float]] | None = None,
    drop_kind: str | None = None,
) -> Path:
    protocol = dict(BASE_PROTOCOL)
    protocol.update(protocol_changes or {})

    results = []
    for kind in KINDS:
        if kind == drop_kind:
            continue
        results.append(
            make_result(
                height=256,
                width=256,
                kind=kind,
                median_ms=medians[kind],
                samples_ms=(samples_by_kind or {}).get(kind),
            )
        )

    document = {
        **protocol,
        "run_id": run_id,
        "results": results,
    }
    path = directory / f"run-{run_id}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def constant_medians(value: float = 2.0) -> dict[str, float]:
    return {kind: value for kind in KINDS}


def test_summarize_reports_median_range_and_mad():
    result = summarize([1.0, 2.0, 3.0, 4.0, 5.0])

    assert result == {
        "count": 5,
        "median": 3.0,
        "min": 1.0,
        "max": 5.0,
        "range": 4.0,
        "median_abs_deviation": 1.0,
    }


def test_load_run_builds_case_index(tmp_path):
    path = write_run(
        tmp_path,
        run_id="r01",
        medians=constant_medians(),
    )

    run = load_run(path)
    case = run["_cases"][(256, 256, "naive")]

    assert run["_path"] == path
    assert len(run["_cases"]) == 4
    assert case["median_ms"] == pytest.approx(2.0)
    assert case["effective_gbps"] == pytest.approx(
        2 * 256 * 256 * 4 / (2.0 * 1_000_000.0)
    )


def test_load_run_rejects_bad_effective_gbps(tmp_path):
    path = write_run(
        tmp_path,
        run_id="r01",
        medians=constant_medians(),
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    document["results"][0]["effective_gbps"] *= 2
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="effective_gbps mismatch"):
        load_run(path)


def test_load_run_rejects_duplicate_case(tmp_path):
    path = write_run(
        tmp_path,
        run_id="r01",
        medians=constant_medians(),
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    document["results"].append(dict(document["results"][0]))
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case"):
        load_run(path)


def test_protocol_rejects_changed_driver(tmp_path):
    paths = [
        write_run(
            tmp_path,
            run_id=f"r{index:02d}",
            medians=constant_medians(),
            protocol_changes=(
                {"nvidia_driver_version": "changed"}
                if index == 5
                else None
            ),
        )
        for index in range(1, 6)
    ]
    runs = [load_run(path) for path in paths]

    with pytest.raises(ValueError, match="protocol field"):
        validate_protocol(runs, expected_rounds=5)


def test_protocol_rejects_case_grid_drift(tmp_path):
    paths = [
        write_run(
            tmp_path,
            run_id=f"r{index:02d}",
            medians=constant_medians(),
            drop_kind="padding" if index == 5 else None,
        )
        for index in range(1, 6)
    ]
    runs = [load_run(path) for path in paths]

    with pytest.raises(ValueError, match="case grid changed"):
        validate_protocol(runs, expected_rounds=5)


def test_aggregate_uses_median_of_run_medians(tmp_path):
    medians_by_run = [
        {"copy": 6.0, "naive": 1.0, "tiled": 4.0, "padding": 0.5},
        {"copy": 7.0, "naive": 2.0, "tiled": 5.0, "padding": 1.0},
        {"copy": 8.0, "naive": 3.0, "tiled": 6.0, "padding": 1.5},
        {"copy": 9.0, "naive": 4.0, "tiled": 7.0, "padding": 2.0},
        {"copy": 10.0, "naive": 5.0, "tiled": 8.0, "padding": 2.5},
    ]

    paths = []
    pooled_naive_samples = []
    for index, medians in enumerate(medians_by_run, 1):
        samples_by_kind = {
            kind: [median, median, 100.0]
            for kind, median in medians.items()
        }
        pooled_naive_samples.extend(samples_by_kind["naive"])
        paths.append(
            write_run(
                tmp_path,
                run_id=f"r{index:02d}",
                medians=medians,
                samples_by_kind=samples_by_kind,
            )
        )

    runs = [load_run(path) for path in paths]
    validate_protocol(runs, expected_rounds=5)
    result = aggregate(runs)

    shape = result["shapes"][0]
    by_kind = {
        item["kind"]: item
        for item in shape["results"]
    }

    assert result["round_count"] == 5
    assert "raw samples from different runs are not pooled" in result[
        "aggregation_rule"
    ]
    assert shape["logical_bytes"] == 2 * 256 * 256 * 4
    assert shape["observed_fastest_kind"] == "padding"

    assert by_kind["naive"]["run_median_ms"]["median"] == 3.0
    assert by_kind["naive"]["run_median_ms"]["range"] == 4.0
    assert statistics.median(pooled_naive_samples) == 4.0

    padding = by_kind["padding"]
    assert padding["run_median_ms"]["median"] == 1.5
    assert padding["speedup_vs_naive_from_mom"] == pytest.approx(2.0)
    assert padding["effective_gbps_from_mom"] == pytest.approx(
        (2 * 256 * 256 * 4) / (1.5 * 1_000_000.0)
    )


def test_cli_writes_summary_json(tmp_path, monkeypatch):
    for index in range(1, 6):
        write_run(
            tmp_path,
            run_id=f"r{index:02d}",
            medians=constant_medians(float(index)),
        )

    output = tmp_path / "summary" / "aggregate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_transpose_rounds.py",
            "--input-dir",
            str(tmp_path),
            "--pattern",
            "run-*.json",
            "--rounds",
            "5",
            "--output",
            str(output),
        ],
    )

    main()

    assert output.is_file()
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["round_count"] == 5
    assert len(document["source_files"]) == 5
    assert len(document["shapes"]) == 1
