from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest

from python.experiment_identity import (
    SourceIdentityError,
    collect_source_identity,
    fingerprint,
    require_same_fingerprint,
    require_same_source_identity,
    sha256_file,
)


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Identity Test")
    git(tmp_path, "config", "user.email", "identity@example.invalid")

    (tmp_path / "producer.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    (tmp_path / "unrelated.txt").write_text(
        "original\n",
        encoding="utf-8",
    )

    git(tmp_path, "add", "--", "producer.py", "unrelated.txt")
    git(tmp_path, "commit", "-q", "-m", "initial")
    return tmp_path


def capture(
    repo: Path,
    *,
    capture_point: str,
    require_clean: bool = False,
) -> dict:
    return collect_source_identity(
        start=repo,
        source_files=[repo / "producer.py"],
        require_clean=require_clean,
        capture_point=capture_point,
    )


def test_empty_source_files_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(
        SourceIdentityError,
        match="source_files must not be empty",
    ):
        collect_source_identity(
            start=tmp_path,
            source_files=[],
            capture_point="before_measurement",
        )


def test_capture_records_stable_file_identity(git_repo: Path) -> None:
    source = git_repo / "producer.py"
    identity = capture(
        git_repo,
        capture_point="before_measurement",
    )

    assert identity["schema_version"] == 2
    assert identity["capture_point"] == "before_measurement"

    code = identity["code_fingerprint"]
    assert code["git_revision"] == git(git_repo, "rev-parse", "HEAD")
    assert code["source_dirty"] is False
    assert code["source_files"] == [
        {
            "path": "producer.py",
            "size_bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        }
    ]


def test_capture_point_is_not_part_of_code_fingerprint(
    git_repo: Path,
) -> None:
    before = capture(
        git_repo,
        capture_point="before_measurement",
    )
    after = capture(
        git_repo,
        capture_point="after_measurement",
    )

    assert before["capture_point"] != after["capture_point"]
    assert fingerprint(before) == fingerprint(after)
    assert require_same_fingerprint([before, after]) is before


def test_unrelated_dirty_file_does_not_dirty_source_scope(
    git_repo: Path,
) -> None:
    (git_repo / "unrelated.txt").write_text(
        "changed outside the experiment source set\n",
        encoding="utf-8",
    )

    identity = capture(
        git_repo,
        capture_point="before_measurement",
        require_clean=True,
    )

    assert identity["code_fingerprint"]["source_dirty"] is False


def test_result_artifact_does_not_change_producer_fingerprint(
    git_repo: Path,
) -> None:
    before = capture(
        git_repo,
        capture_point="before_measurement",
    )

    # 模拟 benchmark 在仓库内写出 JSON；它不是 source_files 的成员。
    (git_repo / "result.json").write_text(
        '{"median_ms": 1.23}\n',
        encoding="utf-8",
    )

    after = capture(
        git_repo,
        capture_point="after_measurement",
    )

    assert fingerprint(before) == fingerprint(after)
    assert require_same_fingerprint([before, after]) is before


def test_dirty_source_is_recorded_and_can_be_rejected(
    git_repo: Path,
) -> None:
    (git_repo / "producer.py").write_text(
        "VALUE = 2\n",
        encoding="utf-8",
    )

    identity = capture(
        git_repo,
        capture_point="before_measurement",
    )
    assert identity["code_fingerprint"]["source_dirty"] is True

    with pytest.raises(
        SourceIdentityError,
        match="clean source files",
    ):
        capture(
            git_repo,
            capture_point="before_measurement",
            require_clean=True,
        )


def test_source_change_between_captures_is_rejected(
    git_repo: Path,
) -> None:
    before = capture(
        git_repo,
        capture_point="before_measurement",
    )
    (git_repo / "producer.py").write_text(
        "VALUE = 99\n",
        encoding="utf-8",
    )
    after = capture(
        git_repo,
        capture_point="after_measurement",
    )

    with pytest.raises(
        SourceIdentityError,
        match="source changed during benchmark",
    ):
        require_same_fingerprint([before, after])


def test_run_records_must_share_producer_identity(
    git_repo: Path,
) -> None:
    identity = capture(
        git_repo,
        capture_point="before_measurement",
    )
    drifted = copy.deepcopy(identity)
    drifted["code_fingerprint"]["source_files"][0]["sha256"] = "0" * 64

    records = [
        {"run_id": "round-01", "source_identity": identity},
        {"run_id": "round-02", "source_identity": drifted},
    ]

    with pytest.raises(
        ValueError,
        match="source identity drifted at run round-02",
    ):
        require_same_source_identity(records)