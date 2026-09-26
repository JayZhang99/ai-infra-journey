from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Sequence


class SourceIdentityError(RuntimeError):
    """Source identity cannot be captured reliably."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _git(
    repo_root: Path,
    *args: str,
) -> bytes:
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                *args,
            ],
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        raise SourceIdentityError(
            "git command failed: "
            f"git {' '.join(args)}"
        ) from exc

    return completed.stdout


def find_repo_root(start: Path) -> Path:
    resolved = start.expanduser().resolve()

    cwd = (
        resolved
        if resolved.is_dir()
        else resolved.parent
    )

    output = _git(
        cwd,
        "rev-parse",
        "--show-toplevel",
    )

    return Path(
        output.decode("utf-8").strip()
    ).resolve()


def collect_source_identity(
    *,
    start: Path,
    source_files: Sequence[Path],
    require_clean: bool = False,
    capture_point: str,
) -> dict[str, Any]:
    if not source_files:
        raise SourceIdentityError(
            "source_files must not be empty"
        )
    repo_root = find_repo_root(start)

    revision = _git(
        repo_root,
        "rev-parse",
        "HEAD",
    ).decode("ascii").strip()

    # 1. 先解析、验证源码文件。
    resolved_files: list[Path] = []
    relative_paths: list[str] = []

    for raw_path in source_files:
        path = raw_path.expanduser().resolve()

        if not path.is_file():
            raise SourceIdentityError(
                f"source file does not exist: {path}"
            )

        try:
            relative = path.relative_to(repo_root)
        except ValueError as exc:
            raise SourceIdentityError(
                f"source file is outside repo: {path}"
            ) from exc

        resolved_files.append(path)
        relative_paths.append(
            relative.as_posix()
        )

    # 保证不同调用顺序得到稳定结果。
    paired = sorted(
        zip(relative_paths, resolved_files),
        key=lambda item: item[0],
    )
    relative_paths = [
        relative
        for relative, _ in paired
    ]
    resolved_files = [
        path
        for _, path in paired
    ]

    # 2. 只检查参与实验的源码路径。
    source_status_bytes = _git(
        repo_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        *relative_paths,
    )

    source_dirty = bool(
        source_status_bytes
    )

    if require_clean and source_dirty:
        raise SourceIdentityError(
            "formal benchmark requires "
            "clean source files"
        )

    # 3. diff 同样只覆盖这些源码文件。
    source_diff_bytes = _git(
        repo_root,
        "diff",
        "--binary",
        "HEAD",
        "--",
        *relative_paths,
    )

    # 4. 单独计算每个源码文件的内容身份。
    files: list[dict[str, Any]] = []

    for relative, path in zip(
        relative_paths,
        resolved_files,
    ):
        files.append({
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    return {
        "schema_version": 2,
        "capture_point":
            capture_point,

        "code_fingerprint": {
            "git_revision": revision,

            "source_dirty":
                source_dirty,

            "source_status_sha256":
                sha256_bytes(
                    source_status_bytes
                ),

            "source_diff_sha256":
                sha256_bytes(
                    source_diff_bytes
                ),

            "source_files": files,
        },
    }

def fingerprint(identity: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise ValueError(
            "source_identity must be an object"
        )

    code_fingerprint = identity.get("code_fingerprint")
    if not isinstance(code_fingerprint, dict):
        raise ValueError(
            "source_identity.code_fingerprint "
            "must be an object"
        )

    return code_fingerprint

def require_same_source_identity(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError(
            "records must not be empty"
        )

    baseline = fingerprint(records[0]["source_identity"])

    for record in records[1:]:
        current = fingerprint(record['source_identity'])
        if current != baseline:
            raise ValueError(
                "source identity drifted at run "
                f"{record.get('run_id')}"
            )

    return records[0]["source_identity"]

def require_same_fingerprint(
    identities: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not identities:
        raise ValueError(
            "identities must not be empty"
        )

    baseline = fingerprint(
        identities[0]
    )

    for identity in identities[1:]:
        if fingerprint(identity) != baseline:
            raise SourceIdentityError(
                "source changed during benchmark"
            )

    return identities[0]