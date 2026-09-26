from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROBE_SCHEMA_VERSION = 1
PROBE_NAME = "wsl_ort_cuda_environment"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def command_result(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(args),
            cwd=str(cwd) if cwd is not None else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "status": "failed",
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
        }

    return {
        "status": (
            "passed"
            if completed.returncode == 0
            else "failed"
        ),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def installed_distribution(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_repository(repo_root: Path) -> dict[str, Any]:
    revision = command_result(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
    )
    status = command_result(
        ["git", "status", "--porcelain=v1"],
        cwd=repo_root,
    )

    return {
        "root": str(repo_root),
        "git_revision": (
            revision["stdout"]
            if revision["status"] == "passed"
            else None
        ),
        "dirty": (
            bool(status["stdout"])
            if status["status"] == "passed"
            else None
        ),
        "revision_probe": revision,
        "status_probe": {
            "status": status["status"],
            "returncode": status["returncode"],
            "stderr": status["stderr"],
        },
    }


def collect_platform() -> dict[str, Any]:
    release = platform.release()
    return {
        "system": platform.system(),
        "release": release,
        "machine": platform.machine(),
        "is_wsl": (
            "microsoft" in release.lower()
            or "WSL_INTEROP" in os.environ
        ),
    }


def collect_model(model_path: Path) -> dict[str, Any]:
    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        return {
            "path": str(model_path),
            "exists": False,
            "size_bytes": None,
            "sha256": None,
        }

    return {
        "path": str(model_path),
        "exists": True,
        "size_bytes": model_path.stat().st_size,
        "sha256": sha256_file(model_path),
    }


def collect_torch() -> tuple[dict[str, Any], Any | None]:
    try:
        import torch
    except Exception as exc:
        return (
            {
                "import_status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            None,
        )

    cuda_available = bool(torch.cuda.is_available())
    device_count = (
        int(torch.cuda.device_count())
        if cuda_available
        else 0
    )

    devices = []
    for index in range(device_count):
        devices.append({
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(
                torch.cuda.get_device_capability(index)
            ),
        })

    return (
        {
            "import_status": "passed",
            "version": torch.__version__,
            "distribution": installed_distribution("torch"),
            "cuda_version": torch.version.cuda,
            "cuda_available": cuda_available,
            "device_count": device_count,
            "devices": devices,
        },
        torch,
    )


def collect_onnxruntime() -> tuple[dict[str, Any], Any | None]:
    try:
        import onnxruntime as ort
    except Exception as exc:
        return (
            {
                "import_status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "distributions": {
                    name: version
                    for name in (
                        "onnxruntime",
                        "onnxruntime-gpu",
                    )
                    if (
                        version := installed_distribution(name)
                    ) is not None
                },
                "available_providers": [],
                "cuda_ep_available": False,
            },
            None,
        )

    providers = list(ort.get_available_providers())
    distributions = {
        name: version
        for name in (
            "onnxruntime",
            "onnxruntime-gpu",
        )
        if (
            version := installed_distribution(name)
        ) is not None
    }

    return (
        {
            "import_status": "passed",
            "version": ort.__version__,
            "distributions": distributions,
            "available_providers": providers,
            "cuda_ep_available": (
                "CUDAExecutionProvider" in providers
            ),
        },
        ort,
    )


def run_session_smoke(
    *,
    ort: Any | None,
    ort_probe: dict[str, Any],
    model_probe: dict[str, Any],
) -> dict[str, Any]:
    if ort is None:
        return {
            "status": "skipped",
            "reason": "onnxruntime import failed",
        }

    if not ort_probe["cuda_ep_available"]:
        return {
            "status": "skipped",
            "reason": "CUDAExecutionProvider unavailable",
        }

    if not model_probe["exists"]:
        return {
            "status": "blocked",
            "reason": (
                "model does not exist: "
                f"{model_probe['path']}"
            ),
        }

    requested = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    try:
        session = ort.InferenceSession(
            model_probe["path"],
            providers=requested,
        )
        selected = list(session.get_providers())
        cuda_selected_first = (
            bool(selected)
            and selected[0] == "CUDAExecutionProvider"
        )

        return {
            "status": (
                "passed"
                if cuda_selected_first
                else "blocked"
            ),
            "requested_providers": requested,
            "selected_providers": selected,
            "provider_options": session.get_provider_options(),
            "reason": (
                None
                if cuda_selected_first
                else (
                    "CUDA EP was requested but was not "
                    "selected first"
                )
            ),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "requested_providers": requested,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def build_probe(
    *,
    repo_root: Path,
    model_path: Path,
) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve()
    model_probe = collect_model(model_path)
    torch_probe, _ = collect_torch()
    ort_probe, ort = collect_onnxruntime()

    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "probe": PROBE_NAME,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "repository": collect_repository(repo_root),
        "platform": collect_platform(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "nvidia_smi": command_result([
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]),
        "nvcc": command_result(["nvcc", "--version"]),
        "model": model_probe,
        "torch": torch_probe,
        "onnxruntime": ort_probe,
        "session_smoke": run_session_smoke(
            ort=ort,
            ort_probe=ort_probe,
            model_probe=model_probe,
        ),
    }


def save_probe(
    payload: dict[str, Any],
    output: Path,
) -> Path:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description=(
            "Capture WSL, PyTorch CUDA and ONNX Runtime "
            "CUDA Execution Provider evidence as JSON."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Probe JSON output path.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=(
            repo_root
            / "artifacts/onnx/tiny_ffn/tiny_ffn.onnx"
        ),
        help="ONNX model used for CUDA Session creation smoke.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Git repository root recorded by the probe.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_probe(
        repo_root=args.repo_root,
        model_path=args.model,
    )
    output = save_probe(payload, args.output)

    print(f"probe: {output}")
    print(
        "CUDA EP available:",
        payload["onnxruntime"]["cuda_ep_available"],
    )
    print(
        "session status:",
        payload["session_smoke"]["status"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
