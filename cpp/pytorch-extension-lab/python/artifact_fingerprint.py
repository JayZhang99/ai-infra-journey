from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def fingerprint_artifact(path: Path) -> dict:
    """
    返回动态库的文件指纹。

    合同：
    - path 必须是一个存在的普通文件。
    - size_bytes 使用字节。
    - mtime_ns 使用纳秒时间戳。
    - sha256 根据完整文件内容计算。
    """
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"artifact does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"artifact is not a file: {path}"
        )

    stat = path.stat()

    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "mtime_utc": datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat(),
        "sha256": sha256_file(path),
    }


def save_fingerprint(
    artifact: Path,
    output: Path,
) -> dict:
    result = fingerprint_artifact(artifact)

    output = output.resolve()
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record size, mtime and SHA256 "
            "for a compiled extension artifact."
        )
    )

    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
        help="Path to the generated shared library.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    result = save_fingerprint(
        artifact=args.artifact,
        output=args.output,
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()