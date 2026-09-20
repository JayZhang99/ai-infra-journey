from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


HELPER_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(HELPER_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(HELPER_DIR),
    )


from validate_build_experiment import (
    ValidationError,
    load_manifest,
    validate_experiment,
)


PHASE_TIMESTAMPS = {
    "cold": "2026-09-20T00:00:00+00:00",
    "warm": "2026-09-20T00:01:00+00:00",
    "flag_change": (
        "2026-09-20T00:02:00+00:00"
    ),
    "flag_warm": (
        "2026-09-20T00:03:00+00:00"
    ),
}


def make_record(
    phase: str,
) -> dict:
    changed = phase in {
        "flag_change",
        "flag_warm",
    }

    probe = (
        "JAY_PROBE_V2"
        if changed
        else None
    )

    cflags = ["-O2"]
    cuda_cflags = ["-O2"]

    if probe is not None:
        cflags.append(
            f"-D{probe}"
        )
        cuda_cflags.append(
            f"-D{probe}"
        )

    # Cold/Warm 指纹完全相同。
    # Flag-change/Flag-warm 指纹完全相同。
    #
    # 两组使用相同 SHA，专门验证：
    # 重编译发生不代表二进制 SHA 必须变化。
    mtime_ns = (
        2_000
        if changed
        else 1_000
    )

    return {
        "schema_version": 1,
        "created_at_utc": (
            PHASE_TIMESTAMPS[phase]
        ),
        "run_id": (
            "2026-09-20-wsl-test"
        ),
        "phase": phase,
        "repo_root": (
            "/home/jay/ai-infra-journey"
        ),
        "cache_dir": (
            "/tmp/jay_ops_test"
        ),
        "git": {
            "revision": "abc1234",
            "dirty": False,
            "dirty_entry_count": 0,
            "status_sha256": (
                "empty-status-sha"
            ),
        },
        "environment": {
            "python": {
                "version": "3.10.12",
                "executable": (
                    "/usr/bin/python3"
                ),
            },
            "platform": {
                "system": "Linux",
                "release": "test",
                "machine": "x86_64",
            },
            "pytorch": {
                "version": "test",
                "cxx11_abi": True,
            },
            "compiler": {
                "command": "c++",
                "diagnostic": {
                    "available": True,
                    "returncode": 0,
                    "stdout": "g++ test",
                    "stderr": "",
                },
            },
            "cuda": {
                "torch_cuda_version": "13.0",
                "cuda_home": (
                    "/usr/local/cuda-13.0"
                ),
                "torch_cuda_available": True,
                "nvcc": {
                    "available": True,
                    "returncode": 0,
                    "stdout": "nvcc test",
                    "stderr": "",
                },
                "driver": {
                    "available": True,
                    "returncode": 0,
                    "stdout": (
                        "RTX 4070 Ti, test, 8.9"
                    ),
                    "stderr": "",
                },
                "devices": [
                    {
                        "index": 0,
                        "name": "RTX 4070 Ti",
                        "total_memory_bytes": (
                            12_000_000_000
                        ),
                        "compute_capability": [
                            8,
                            9,
                        ],
                    }
                ],
            },
        },
        "build": {
            "name": "jay_ops",
            "with_cuda": True,
            "sources": [
                "cpp/pytorch-extension-lab/"
                "csrc/scale_add_cpu.cpp",
                "cpp/pytorch-extension-lab/"
                "csrc/scale_add_cuda.cu",
            ],
            "include_paths": [
                "cpp/cuda-kernel-lab/include"
            ],
            "extra_cflags": cflags,
            "extra_cuda_cflags": (
                cuda_cflags
            ),
            "probe_define": probe,
        },
        "artifact": {
            "path": (
                "/tmp/jay_ops_test/"
                "jay_ops/jay_ops.so"
            ),
            "size_bytes": 4096,
            "mtime_ns": mtime_ns,
            "mtime_utc": (
                "2026-09-20T00:00:00"
                "+00:00"
            ),
            "sha256": "same-binary-sha",
        },
        "smoke": {
            "passed": True,
            "operator": (
                "jay_ops::scale_add"
            ),
            "alpha": 2.0,
            "input": {
                "shape": [2],
                "dtype": "torch.float32",
                "device": "cpu",
            },
            "output": {
                "shape": [2],
                "dtype": "torch.float32",
                "device": "cpu",
            },
            "max_abs_error": 0.0,
        },
    }


@pytest.fixture
def valid_records() -> list[dict]:
    return [
        make_record("cold"),
        make_record("warm"),
        make_record("flag_change"),
        make_record("flag_warm"),
    ]


def run_validation(
    records: list[dict],
    *,
    allow_dirty: bool = False,
    log_paths: dict[str, Path]
    | None = None,
    require_logs: bool = False,
) -> dict:
    return validate_experiment(
        records,
        allow_dirty=allow_dirty,
        log_paths=(
            {}
            if log_paths is None
            else log_paths
        ),
        require_logs=require_logs,
    )


def write_logs(
    tmp_path: Path,
    contents: dict[str, str],
) -> dict[str, Path]:
    paths = {}

    for phase, text in contents.items():
        path = tmp_path / f"{phase}.log"
        path.write_text(
            text,
            encoding="utf-8",
        )
        paths[phase] = path

    return paths

def test_valid_four_phase_experiment(
    valid_records,
):
    summary = run_validation(
        valid_records
    )

    assert summary["run_id"] == (
        "2026-09-20-wsl-test"
    )
    assert summary["logs_validated"] is False

    # Probe 没有进入实际源码，
    # 因此允许重编译后 SHA 保持不变。
    assert (
        summary["binary_sha_changed"]
        is False
    )

def test_manifest_order_does_not_matter(
    valid_records,
):
    reordered = [
        valid_records[2],
        valid_records[0],
        valid_records[3],
        valid_records[1],
    ]

    summary = run_validation(
        reordered
    )

    assert summary["run_id"] == (
        "2026-09-20-wsl-test"
    )

def test_duplicate_phase_is_rejected(
    valid_records,
):
    records = copy.deepcopy(
        valid_records
    )

    records[3]["phase"] = (
        "flag_change"
    )

    with pytest.raises(
        ValidationError,
        match="duplicate phases",
    ):
        run_validation(records)


def test_cache_drift_is_rejected(
    valid_records,
):
    records = copy.deepcopy(
        valid_records
    )

    records[1]["cache_dir"] = (
        "/tmp/a-different-cache"
    )

    with pytest.raises(
        ValidationError,
        match="cache_dir",
    ):
        run_validation(records)

def test_git_revision_drift_is_rejected(
    valid_records,
):
    records = copy.deepcopy(
        valid_records
    )

    records[2]["git"]["revision"] = (
        "different-commit"
    )

    with pytest.raises(
        ValidationError,
        match="git revision drifted",
    ):
        run_validation(records)

def test_warm_artifact_change_is_rejected(
    valid_records,
):
    records = copy.deepcopy(
        valid_records
    )

    records[1]["artifact"]["sha256"] = (
        "unexpected-new-sha"
    )

    with pytest.raises(
        ValidationError,
        match="cold/warm artifact changed",
    ):
        run_validation(records)

def test_flag_change_must_rewrite_artifact(
    valid_records,
):
    records = copy.deepcopy(
        valid_records
    )

    cold_mtime = (
        records[0]
        ["artifact"]
        ["mtime_ns"]
    )

    records[2]["artifact"]["mtime_ns"] = (
        cold_mtime
    )
    records[3]["artifact"]["mtime_ns"] = (
        cold_mtime
    )

    with pytest.raises(
        ValidationError,
        match="mtime did not change",
    ):
        run_validation(records)

def test_source_drift_is_rejected(
    valid_records,
):
    records = copy.deepcopy(
        valid_records
    )

    records[2]["build"]["sources"].append(
        "unexpected_source.cpp"
    )
    records[3]["build"]["sources"].append(
        "unexpected_source.cpp"
    )

    with pytest.raises(
        ValidationError,
        match="base build identity changed",
    ):
        run_validation(records)

def test_missing_cuda_probe_flag_is_rejected(
    valid_records,
):
    records = copy.deepcopy(
        valid_records
    )

    records[2]["build"][
        "extra_cuda_cflags"
    ] = ["-O2"]

    records[3]["build"][
        "extra_cuda_cflags"
    ] = ["-O2"]

    with pytest.raises(
        ValidationError,
        match=(
            "missing from extra_cuda_cflags"
        ),
    ):
        run_validation(records)

def test_dirty_repo_is_rejected_by_default(
    valid_records,
):
    records = copy.deepcopy(
        valid_records
    )

    for record in records:
        record["git"]["dirty"] = True
        record["git"][
            "dirty_entry_count"
        ] = 1
        record["git"][
            "status_sha256"
        ] = "same-dirty-state"

    with pytest.raises(
        ValidationError,
        match="working tree is dirty",
    ):
        run_validation(records)


def test_stable_dirty_repo_can_be_allowed(
    valid_records,
):
    records = copy.deepcopy(
        valid_records
    )

    for record in records:
        record["git"]["dirty"] = True
        record["git"][
            "dirty_entry_count"
        ] = 1
        record["git"][
            "status_sha256"
        ] = "same-dirty-state"

    summary = run_validation(
        records,
        allow_dirty=True,
    )

    assert summary["run_id"] == (
        "2026-09-20-wsl-test"
    )

def test_dirty_status_drift_is_rejected(
    valid_records,
):
    records = copy.deepcopy(
        valid_records
    )

    for record in records:
        record["git"]["dirty"] = True
        record["git"][
            "status_sha256"
        ] = "state-a"

    records[2]["git"][
        "status_sha256"
    ] = "state-b"

    with pytest.raises(
        ValidationError,
        match=(
            "working-tree status changed"
        ),
    ):
        run_validation(
            records,
            allow_dirty=True,
        )

def test_complete_log_evidence_passes(
    tmp_path,
    valid_records,
):
    log_paths = write_logs(
        tmp_path,
        {
            "cold": (
                "Building extension module jay_ops\n"
                "[1/2] c++ -O2 -c scale_add_cpu.cpp\n"
                "[2/2] Linking jay_ops.so\n"
            ),
            "warm": (
                "ninja: no work to do.\n"
            ),
            "flag_change": (
                "Building extension module jay_ops\n"
                "[1/2] nvcc -O2 "
                "-DJAY_PROBE_V2 "
                "-c scale_add_cuda.cu\n"
                "[2/2] Linking jay_ops.so\n"
            ),
            "flag_warm": (
                "ninja: no work to do.\n"
            ),
        },
    )

    summary = run_validation(
        valid_records,
        log_paths=log_paths,
        require_logs=True,
    )

    assert (
        summary["logs_validated"]
        is True
    )

def test_warm_without_no_work_is_rejected(
    tmp_path,
    valid_records,
):
    log_paths = write_logs(
        tmp_path,
        {
            "cold": (
                "Building extension module\n"
                "[1/2] c++ -O2 -c source.cpp\n"
            ),
            "warm": (
                "Building extension module\n"
                "[1/2] c++ -O2 -c source.cpp\n"
            ),
            "flag_change": (
                "Building extension module\n"
                "nvcc -DJAY_PROBE_V2 "
                "-c source.cu\n"
            ),
            "flag_warm": (
                "ninja: no work to do.\n"
            ),
        },
    )

    with pytest.raises(
        ValidationError,
        match=(
            "warm: Ninja no-work "
            "evidence not found"
        ),
    ):
        run_validation(
            valid_records,
            log_paths=log_paths,
            require_logs=True,
        )

def test_load_manifest_rejects_invalid_json(
    tmp_path,
):
    path = tmp_path / "broken.json"

    path.write_text(
        "{not-json",
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationError,
        match="invalid JSON",
    ):
        load_manifest(path)

