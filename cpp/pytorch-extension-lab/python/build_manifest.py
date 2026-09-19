from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import CUDA_HOME

from artifact_fingerprint import fingerprint_artifact


SCHEMA_VERSION = 1

VALID_PHASES = {
    "cold",
    "warm",
    "flag_change",
    "flag_warm",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def command_output(
    args: list[str],
    *,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """
    执行只读诊断命令。

    使用参数列表，不使用 shell=True，
    从而避免 shell quoting 和命令注入问题。
    """
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except Exception as exc:
        return {
            "available": False,
            "error": repr(exc),
        }

    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def require_cache_dir() -> Path:
    """
    要求实验显式指定 TORCH_EXTENSIONS_DIR。

    这可以防止：
    - 忘记 export；
    - 拼写为 TORCH_ENTENSIONS_DIR；
    - cold/warm 意外落入默认缓存。
    """
    raw = os.environ.get(
        "TORCH_EXTENSIONS_DIR"
    )

    if not raw:
        raise RuntimeError(
            "TORCH_EXTENSIONS_DIR must be set "
            "for a reproducible build experiment"
        )

    cache_dir = Path(raw).expanduser().resolve()
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return cache_dir


def collect_git_info(
    repo_root: Path,
) -> dict[str, Any]:
    revision_result = command_output(
        [
            "git",
            "rev-parse",
            "--short",
            "HEAD",
        ],
        cwd=repo_root,
    )

    status_result = command_output(
        [
            "git",
            "status",
            "--porcelain",
        ],
        cwd=repo_root,
    )

    if revision_result.get("returncode") != 0:
        raise RuntimeError(
            "failed to read git revision: "
            f"{revision_result}"
        )

    if status_result.get("returncode") != 0:
        raise RuntimeError(
            "failed to read git status: "
            f"{status_result}"
        )

    status_text = status_result["stdout"]

    return {
        "revision": revision_result["stdout"],
        "dirty": bool(status_text),
        "dirty_entry_count": (
            len(status_text.splitlines())
            if status_text
            else 0
        ),
        # 不把文件名直接写进 Manifest，
        # 只保存状态文本的指纹。
        "status_sha256": sha256_text(
            status_text
        ),
    }


def collect_cuda_info() -> dict[str, Any]:
    nvcc: dict[str, Any]

    if CUDA_HOME is None:
        nvcc = {
            "available": False,
            "reason": "CUDA_HOME is not set",
        }
    else:
        nvcc_path = (
            Path(CUDA_HOME)
            / "bin"
            / "nvcc"
        )

        nvcc = command_output(
            [
                str(nvcc_path),
                "--version",
            ]
        )

        nvcc["path"] = str(nvcc_path)

    driver = command_output(
        [
            "nvidia-smi",
            "--query-gpu="
            "name,driver_version,"
            "compute_cap",
            "--format=csv,noheader",
        ]
    )

    devices = []

    if torch.cuda.is_available():
        for index in range(
            torch.cuda.device_count()
        ):
            properties = (
                torch.cuda.get_device_properties(
                    index
                )
            )

            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": (
                        properties.total_memory
                    ),
                    "compute_capability": [
                        properties.major,
                        properties.minor,
                    ],
                }
            )

    return {
        "torch_cuda_version": (
            torch.version.cuda
        ),
        "cuda_home": (
            str(CUDA_HOME)
            if CUDA_HOME is not None
            else None
        ),
        "torch_cuda_available": (
            torch.cuda.is_available()
        ),
        "nvcc": nvcc,
        "driver": driver,
        "devices": devices,
    }


def collect_environment() -> dict[str, Any]:
    cxx = os.environ.get("CXX", "c++")

    cxx_result = command_output(
        [
            cxx,
            "--version",
        ]
    )

    return {
        "python": {
            "version": platform.python_version(),
            "executable": str(
                Path(sys.executable).resolve()
            ),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "pytorch": {
            "version": torch.__version__,
            "cxx11_abi": bool(
                getattr(
                    torch._C,
                    "_GLIBCXX_USE_CXX11_ABI",
                    False,
                )
            ),
        },
        "compiler": {
            "command": cxx,
            "diagnostic": cxx_result,
        },
        "cuda": collect_cuda_info(),
    }


def relative_to_repo(
    path: str | Path,
    repo_root: Path,
) -> str:
    resolved = Path(path).resolve()

    try:
        return str(
            resolved.relative_to(repo_root)
        )
    except ValueError:
        return str(resolved)


def validate_artifact_location(
    *,
    artifact: Path,
    cache_dir: Path,
) -> None:
    artifact = artifact.resolve()
    cache_dir = cache_dir.resolve()

    if (
        artifact != cache_dir
        and cache_dir not in artifact.parents
    ):
        raise RuntimeError(
            "compiled artifact is outside the "
            "requested TORCH_EXTENSIONS_DIR: "
            f"artifact={artifact}, "
            f"cache_dir={cache_dir}"
        )


def create_manifest(
    *,
    run_id: str,
    phase: str,
    repo_root: Path,
    cache_dir: Path,
    artifact: Path,
    load_kwargs: dict[str, Any],
    probe_define: str | None,
    smoke: dict[str, Any],
) -> dict[str, Any]:
    if phase not in VALID_PHASES:
        raise ValueError(
            f"invalid phase: {phase}"
        )

    if not run_id.strip():
        raise ValueError(
            "run_id must not be empty"
        )

    validate_artifact_location(
        artifact=artifact,
        cache_dir=cache_dir,
    )

    sources = [
        relative_to_repo(
            source,
            repo_root,
        )
        for source
        in load_kwargs["sources"]
    ]

    include_paths = [
        relative_to_repo(
            path,
            repo_root,
        )
        for path
        in load_kwargs.get(
            "extra_include_paths",
            [],
        )
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "run_id": run_id,
        "phase": phase,
        "repo_root": str(
            repo_root.resolve()
        ),
        "cache_dir": str(
            cache_dir.resolve()
        ),
        "git": collect_git_info(
            repo_root
        ),
        "environment": (
            collect_environment()
        ),
        "build": {
            "name": load_kwargs["name"],
            "with_cuda": bool(
                load_kwargs["with_cuda"]
            ),
            "sources": sources,
            "include_paths": include_paths,
            "extra_cflags": list(
                load_kwargs.get(
                    "extra_cflags",
                    [],
                )
            ),
            "extra_cuda_cflags": list(
                load_kwargs.get(
                    "extra_cuda_cflags",
                    [],
                )
            ),
            "probe_define": probe_define,
        },
        "artifact": (
            fingerprint_artifact(
                artifact
            )
        ),
        "smoke": smoke,
    }


def save_manifest(
    *,
    manifest: dict[str, Any],
    output: Path,
) -> None:
    output = output.expanduser().resolve()

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = output.with_suffix(
        output.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # 同一文件系统内原子替换，
    # 避免留下半份 JSON。
    temporary.replace(output)