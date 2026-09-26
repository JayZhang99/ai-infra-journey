from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from python.onnx_lab import aggregate_ort_level_runs as target


def make_identity(*, file_sha: str = "a" * 64) -> dict:
    return {
        "schema_version": 2,
        "capture_point": "before_measurement",
        "code_fingerprint": {
            "git_revision": "1" * 40,
            "source_dirty": False,
            "source_status_sha256": "2" * 64,
            "source_diff_sha256": "3" * 64,
            "source_files": [
                {
                    "path": "python/onnx_lab/benchmark_ort_levels.py",
                    "size_bytes": 1234,
                    "sha256": file_sha,
                }
            ],
        },
    }


def make_run(
    run_id: str,
    *,
    schema_version: int = 1,
    identity: dict | None = None,
    base_ms: float = 1.0,
) -> dict:
    repeats = 4
    providers = ["CPUExecutionProvider"]
    level_offsets = {
        "disable": 0.30,
        "basic": 0.20,
        "extended": 0.10,
        "all": 0.00,
    }

    levels = {}
    for level, offset in level_offsets.items():
        samples = [
            base_ms + offset + index * 0.01
            for index in range(repeats)
        ]
        levels[level] = {
            "selected_providers": providers,
            "samples_ms_per_run": samples,
            "summary_ms_per_run": target.summarize(samples),
        }

    run = {
        "schema_version": schema_version,
        "experiment": "ort_optimization_levels_cpu",
        "run_id": run_id,
        "profile_enabled": False,
        "model": {
            "path": "artifacts/onnx/tiny_ffn/tiny_ffn.onnx",
            "size": 100,
            "sha256": "f" * 64,
        },
        "environment": {
            "onnxruntime": "1.23.2",
            "selected_providers": providers,
        },
        "configuration": {
            "shape": [1, 128, 8],
            "dtype": "float32",
            "threads": 1,
            "execution_mode": "sequential",
            "warmup": 2,
            "repeats": repeats,
            "runs_per_sample": 3,
            "seed": 0,
            "level_order_policy": "rotating",
            "rtol": 1e-5,
            "atol": 1e-6,
        },
        "sample_unit": "milliseconds_per_session_run",
        "sample_orders": [
            list(
                target.LEVEL_ORDER[offset:]
                + target.LEVEL_ORDER[:offset]
            )
            for offset in range(repeats)
        ],
        "levels": levels,
    }

    if schema_version == 2:
        run["source_identity"] = (
            copy.deepcopy(identity)
            if identity is not None
            else None
        )

    return run


def write_run(tmp_path: Path, run: dict) -> Path:
    path = tmp_path / f"{run['run_id']}.json"
    path.write_text(
        json.dumps(run, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def aggregator_identity() -> dict:
    identity = make_identity(file_sha="9" * 64)
    identity["capture_point"] = "before_aggregation"
    return identity


@pytest.fixture
def stub_aggregator_identity(
    monkeypatch: pytest.MonkeyPatch,
    aggregator_identity: dict,
) -> dict:
    def fake_collect_source_identity(**kwargs) -> dict:
        assert kwargs["capture_point"] == "before_aggregation"
        assert kwargs["require_clean"] is False
        return copy.deepcopy(aggregator_identity)

    monkeypatch.setattr(
        target,
        "collect_source_identity",
        fake_collect_source_identity,
    )
    return aggregator_identity


def test_aggregate_runs_uses_per_run_samples() -> None:
    runs = [
        make_run("round-01", base_ms=1.0),
        make_run("round-02", base_ms=2.0),
    ]

    result = target.aggregate_runs(runs, expected_rounds=2)

    expected = [
        target.summarize(
            run["levels"]["all"]["samples_ms_per_run"]
        )["median_ms"]
        for run in runs
    ]
    assert result["round_count"] == 2
    assert result["levels"]["all"]["round_medians_ms"] == expected
    assert result["best_level_by_median_of_medians"] == "all"


def test_legacy_v1_files_are_accepted(
    tmp_path: Path,
    stub_aggregator_identity: dict,
) -> None:
    paths = [
        write_run(tmp_path, make_run(f"round-{index:02d}"))
        for index in range(1, 3)
    ]

    result = target.aggregate_files(paths, expected_rounds=2)

    assert result["schema_version"] == 2
    assert result["producer_identity_status"] == "legacy_missing"
    assert result["producer_source_identity"] is None
    assert result["aggregator_source_identity"] == stub_aggregator_identity


def test_v2_files_with_same_identity_are_verified(
    tmp_path: Path,
    stub_aggregator_identity: dict,
) -> None:
    producer = make_identity()
    paths = [
        write_run(
            tmp_path,
            make_run(
                f"round-{index:02d}",
                schema_version=2,
                identity=producer,
            ),
        )
        for index in range(1, 3)
    ]

    result = target.aggregate_files(paths, expected_rounds=2)

    assert result["producer_identity_status"] == "verified"
    assert result["producer_source_identity"] == producer
    assert result["aggregator_source_identity"] == stub_aggregator_identity
    assert len(result["source_files"]) == 2
    assert all(source["sha256"] for source in result["source_files"])


def test_mixed_legacy_and_v2_files_are_rejected(
    tmp_path: Path,
    stub_aggregator_identity: dict,
) -> None:
    paths = [
        write_run(tmp_path, make_run("round-01")),
        write_run(
            tmp_path,
            make_run(
                "round-02",
                schema_version=2,
                identity=make_identity(),
            ),
        ),
    ]

    with pytest.raises(
        ValueError,
        match="cannot mix runs with and without source_identity",
    ):
        target.aggregate_files(paths, expected_rounds=2)


def test_v2_source_identity_drift_is_rejected(
    tmp_path: Path,
    stub_aggregator_identity: dict,
) -> None:
    paths = [
        write_run(
            tmp_path,
            make_run(
                "round-01",
                schema_version=2,
                identity=make_identity(file_sha="a" * 64),
            ),
        ),
        write_run(
            tmp_path,
            make_run(
                "round-02",
                schema_version=2,
                identity=make_identity(file_sha="b" * 64),
            ),
        ),
    ]

    with pytest.raises(
        ValueError,
        match="source identity drifted at run round-02",
    ):
        target.aggregate_files(paths, expected_rounds=2)


def test_duplicate_input_path_is_rejected(
    tmp_path: Path,
    stub_aggregator_identity: dict,
) -> None:
    path = write_run(tmp_path, make_run("round-01"))

    with pytest.raises(ValueError, match="duplicate input path"):
        target.aggregate_files([path, path], expected_rounds=2)


def test_v2_null_source_identity_is_rejected(
    tmp_path: Path,
    stub_aggregator_identity: dict,
) -> None:
    paths = [
        write_run(
            tmp_path,
            make_run(
                f"round-{index:02d}",
                schema_version=2,
                identity=None,
            ),
        )
        for index in range(1, 3)
    ]

    with pytest.raises(
        ValueError,
        match="schema v2 requires source_identity object",
    ):
        target.aggregate_files(paths, expected_rounds=2)


def test_v2_malformed_source_identity_is_rejected_cleanly(
    tmp_path: Path,
    stub_aggregator_identity: dict,
) -> None:
    paths = [
        write_run(
            tmp_path,
            make_run(
                f"round-{index:02d}",
                schema_version=2,
                identity={},
            ),
        )
        for index in range(1, 3)
    ]

    with pytest.raises(ValueError, match="source_identity"):
        target.aggregate_files(paths, expected_rounds=2)

