from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1

REQUIRED_PHASES = {
    "cold",
    "warm",
    "flag_change",
    "flag_warm",
}

TOP_LEVEL_FIELDS = {
    "schema_version",
    "created_at_utc",
    "run_id",
    "phase",
    "repo_root",
    "cache_dir",
    "git",
    "environment",
    "build",
    "artifact",
    "smoke",
}

BUILD_FIELDS = {
    "name",
    "with_cuda",
    "sources",
    "include_paths",
    "extra_cflags",
    "extra_cuda_cflags",
    "probe_define",
}

ARTIFACT_FIELDS = {
    "path",
    "size_bytes",
    "mtime_ns",
    "sha256",
}


class ValidationError(RuntimeError):
    """Four-phase build experiment violates its contract."""


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise ValidationError(message)


def load_manifest(
    path: Path,
) -> dict[str, Any]:
    path = path.expanduser().resolve()

    if not path.exists():
        raise ValidationError(
            f"manifest does not exist: {path}"
        )

    if not path.is_file():
        raise ValidationError(
            f"manifest is not a file: {path}"
        )

    try:
        result = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"invalid JSON in {path}: {exc}"
        ) from exc

    if not isinstance(result, dict):
        raise ValidationError(
            f"manifest root must be an object: {path}"
        )

    # 仅用于错误信息，不参与合同比较。
    result["_manifest_path"] = str(path)

    return result


def validate_schema(
    record: dict[str, Any],
) -> None:
    path = record.get(
        "_manifest_path",
        "<unknown>",
    )

    missing = TOP_LEVEL_FIELDS - set(record)

    require(
        not missing,
        (
            f"{path}: missing top-level fields: "
            f"{sorted(missing)}"
        ),
    )

    require(
        record["schema_version"]
        == SCHEMA_VERSION,
        (
            f"{path}: unsupported schema_version: "
            f"{record['schema_version']}"
        ),
    )

    require(
        record["phase"]
        in REQUIRED_PHASES,
        (
            f"{path}: invalid phase: "
            f"{record['phase']!r}"
        ),
    )

    require(
        isinstance(record["run_id"], str)
        and bool(record["run_id"].strip()),
        f"{path}: run_id must be non-empty",
    )

    require(
        isinstance(record["git"], dict),
        f"{path}: git must be an object",
    )

    for field in (
        "revision",
        "dirty",
        "status_sha256",
    ):
        require(
            field in record["git"],
            f"{path}: missing git.{field}",
        )

    require(
        isinstance(record["build"], dict),
        f"{path}: build must be an object",
    )

    missing_build = (
        BUILD_FIELDS
        - set(record["build"])
    )

    require(
        not missing_build,
        (
            f"{path}: missing build fields: "
            f"{sorted(missing_build)}"
        ),
    )

    require(
        isinstance(
            record["artifact"],
            dict,
        ),
        f"{path}: artifact must be an object",
    )

    missing_artifact = (
        ARTIFACT_FIELDS
        - set(record["artifact"])
    )

    require(
        not missing_artifact,
        (
            f"{path}: missing artifact fields: "
            f"{sorted(missing_artifact)}"
        ),
    )

    require(
        record["smoke"].get("passed")
        is True,
        f"{path}: smoke test did not pass",
    )


def index_by_phase(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    phases = [
        record["phase"]
        for record in records
    ]

    require(
        len(phases) == len(set(phases)),
        f"duplicate phases: {phases}",
    )

    indexed = {
        record["phase"]: record
        for record in records
    }

    actual = set(indexed)

    require(
        actual == REQUIRED_PHASES,
        (
            "phase set mismatch: "
            f"expected={sorted(REQUIRED_PHASES)}, "
            f"actual={sorted(actual)}"
        ),
    )

    return indexed


def require_same_field(
    records: list[dict[str, Any]],
    field: str,
) -> None:
    values = {
        json.dumps(
            record[field],
            ensure_ascii=False,
            sort_keys=True,
        )
        for record in records
    }

    require(
        len(values) == 1,
        (
            f"field drifted across phases: "
            f"{field}"
        ),
    )


def artifact_fingerprint(
    record: dict[str, Any],
) -> tuple[Any, ...]:
    artifact = record["artifact"]

    return (
        artifact["path"],
        artifact["size_bytes"],
        artifact["mtime_ns"],
        artifact["sha256"],
    )


def remove_probe_flag(
    flags: list[str],
    probe: str | None,
) -> list[str]:
    if probe is None:
        return list(flags)

    expected = f"-D{probe}"

    return [
        flag
        for flag in flags
        if flag != expected
    ]


def normalized_build(
    record: dict[str, Any],
) -> dict[str, Any]:
    """
    返回去掉 Probe 变量后的基础构建身份。

    四个阶段的基础构建身份必须相同。
    """
    result = copy.deepcopy(
        record["build"]
    )

    probe = result.pop(
        "probe_define",
        None,
    )

    result["extra_cflags"] = (
        remove_probe_flag(
            result.get(
                "extra_cflags",
                [],
            ),
            probe,
        )
    )

    result["extra_cuda_cflags"] = (
        remove_probe_flag(
            result.get(
                "extra_cuda_cflags",
                [],
            ),
            probe,
        )
    )

    return result


def validate_probe_in_flags(
    record: dict[str, Any],
) -> None:
    build = record["build"]
    probe = build["probe_define"]

    if probe is None:
        return

    expected = f"-D{probe}"

    require(
        expected
        in build["extra_cflags"],
        (
            f"{record['phase']}: "
            f"{expected} missing from "
            "extra_cflags"
        ),
    )

    if build["with_cuda"]:
        require(
            expected
            in build[
                "extra_cuda_cflags"
            ],
            (
                f"{record['phase']}: "
                f"{expected} missing from "
                "extra_cuda_cflags"
            ),
        )


def validate_timestamps(
    phases: dict[str, dict[str, Any]],
) -> None:
    order = [
        "cold",
        "warm",
        "flag_change",
        "flag_warm",
    ]

    try:
        timestamps = [
            datetime.fromisoformat(
                phases[phase][
                    "created_at_utc"
                ]
            )
            for phase in order
        ]
    except ValueError as exc:
        raise ValidationError(
            "invalid created_at_utc timestamp"
        ) from exc

    require(
        timestamps
        == sorted(timestamps),
        (
            "phase timestamps are not ordered: "
            f"{list(zip(order, timestamps))}"
        ),
    )


def validate_identity(
    records: list[dict[str, Any]],
    *,
    allow_dirty: bool,
) -> None:
    for field in (
        "run_id",
        "repo_root",
        "cache_dir",
        "environment",
    ):
        require_same_field(
            records,
            field,
        )

    revisions = {
        record["git"]["revision"]
        for record in records
    }

    require(
        len(revisions) == 1,
        (
            "git revision drifted: "
            f"{sorted(revisions)}"
        ),
    )

    dirty_values = {
        record["git"]["dirty"]
        for record in records
    }

    require(
        len(dirty_values) == 1,
        (
            "git dirty state changed "
            "between phases"
        ),
    )

    if not allow_dirty:
        require(
            dirty_values == {False},
            (
                "working tree is dirty; "
                "commit/stash changes or use "
                "--allow-dirty explicitly"
            ),
        )
    else:
        status_hashes = {
            record["git"][
                "status_sha256"
            ]
            for record in records
        }

        require(
            len(status_hashes) == 1,
            (
                "working-tree status changed "
                "between phases"
            ),
        )


def validate_build_identity(
    records: list[dict[str, Any]],
    phases: dict[str, dict[str, Any]],
) -> None:
    normalized = [
        json.dumps(
            normalized_build(record),
            ensure_ascii=False,
            sort_keys=True,
        )
        for record in records
    ]

    require(
        len(set(normalized)) == 1,
        (
            "base build identity changed "
            "between phases"
        ),
    )

    cold = phases["cold"]
    warm = phases["warm"]
    flag_change = phases["flag_change"]
    flag_warm = phases["flag_warm"]

    require(
        cold["build"] == warm["build"],
        (
            "cold and warm build inputs "
            "are different"
        ),
    )

    require(
        flag_change["build"]
        == flag_warm["build"],
        (
            "flag_change and flag_warm "
            "build inputs are different"
        ),
    )

    cold_probe = (
        cold["build"]["probe_define"]
    )
    warm_probe = (
        warm["build"]["probe_define"]
    )
    changed_probe = (
        flag_change[
            "build"
        ]["probe_define"]
    )
    changed_warm_probe = (
        flag_warm[
            "build"
        ]["probe_define"]
    )

    require(
        cold_probe == warm_probe,
        (
            "warm does not reuse "
            "the cold probe"
        ),
    )

    require(
        changed_probe is not None,
        (
            "flag_change must define "
            "a probe"
        ),
    )

    require(
        changed_probe != cold_probe,
        (
            "flag_change probe is the "
            "same as the cold probe"
        ),
    )

    require(
        changed_warm_probe
        == changed_probe,
        (
            "flag_warm does not reuse "
            "the flag_change probe"
        ),
    )

    for record in records:
        validate_probe_in_flags(record)


def validate_artifacts(
    phases: dict[str, dict[str, Any]],
) -> None:
    cold = phases["cold"]
    warm = phases["warm"]
    flag_change = phases["flag_change"]
    flag_warm = phases["flag_warm"]

    require(
        artifact_fingerprint(cold)
        == artifact_fingerprint(warm),
        (
            "cold/warm artifact changed; "
            "warm build was not stable"
        ),
    )

    require(
        artifact_fingerprint(flag_change)
        == artifact_fingerprint(flag_warm),
        (
            "flag_change/flag_warm "
            "artifact changed"
        ),
    )

    cold_artifact = cold["artifact"]
    changed_artifact = (
        flag_change["artifact"]
    )

    # Flag 变化后应当重新编译并重新链接，
    # 因此最终产物的 mtime 应发生变化。
    require(
        cold_artifact["mtime_ns"]
        != changed_artifact["mtime_ns"],
        (
            "artifact mtime did not change "
            "after probe flag changed"
        ),
    )

    # 不要求 SHA 必须变化。
    # 如果 Probe 没被源码实际引用，
    # 两次二进制内容可能完全相同。


def validate_smoke(
    records: list[dict[str, Any]],
) -> None:
    operators = {
        record["smoke"]["operator"]
        for record in records
    }

    require(
        len(operators) == 1,
        (
            "different smoke operators "
            "were used across phases"
        ),
    )

    for record in records:
        smoke = record["smoke"]

        require(
            smoke["passed"] is True,
            (
                f"{record['phase']}: "
                "smoke did not pass"
            ),
        )

        require(
            smoke["max_abs_error"] == 0.0,
            (
                f"{record['phase']}: "
                "unexpected smoke error: "
                f"{smoke['max_abs_error']}"
            ),
        )


def parse_phase_paths(
    values: list[str],
    *,
    option_name: str,
) -> dict[str, Path]:
    result: dict[str, Path] = {}

    for value in values:
        if "=" not in value:
            raise ValidationError(
                f"{option_name} expects "
                "PHASE=PATH, got: "
                f"{value!r}"
            )

        phase, raw_path = value.split(
            "=",
            maxsplit=1,
        )

        require(
            phase in REQUIRED_PHASES,
            (
                f"{option_name}: "
                f"invalid phase {phase!r}"
            ),
        )

        require(
            phase not in result,
            (
                f"{option_name}: duplicate "
                f"phase {phase!r}"
            ),
        )

        result[phase] = (
            Path(raw_path)
            .expanduser()
            .resolve()
        )

    return result


def read_log(
    path: Path,
) -> str:
    if not path.exists():
        raise ValidationError(
            f"log does not exist: {path}"
        )

    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def contains_build_activity(
    text: str,
) -> bool:
    markers = (
        "Building extension module",
        "nvcc ",
        "c++ ",
        "clang++ ",
        "g++ ",
        "Linking",
        "[1/",
    )

    return any(
        marker in text
        for marker in markers
    )


def validate_logs(
    phases: dict[str, dict[str, Any]],
    log_paths: dict[str, Path],
    *,
    require_logs: bool,
) -> bool:
    if not log_paths:
        require(
            not require_logs,
            (
                "--require-logs was set, "
                "but no --log arguments "
                "were provided"
            ),
        )

        return False

    require(
        set(log_paths)
        == REQUIRED_PHASES,
        (
            "log phase set mismatch: "
            f"expected={sorted(REQUIRED_PHASES)}, "
            f"actual={sorted(log_paths)}"
        ),
    )

    logs = {
        phase: read_log(path)
        for phase, path
        in log_paths.items()
    }

    for phase in (
        "warm",
        "flag_warm",
    ):
        require(
            "no work to do"
            in logs[phase].lower(),
            (
                f"{phase}: Ninja no-work "
                "evidence not found"
            ),
        )

    for phase in (
        "cold",
        "flag_change",
    ):
        require(
            "no work to do"
            not in logs[phase].lower(),
            (
                f"{phase}: unexpectedly "
                "reported no work"
            ),
        )

        require(
            contains_build_activity(
                logs[phase]
            ),
            (
                f"{phase}: compile/link "
                "activity not found"
            ),
        )

    changed_probe = (
        phases["flag_change"]
        ["build"]
        ["probe_define"]
    )

    expected_flag = (
        f"-D{changed_probe}"
    )

    require(
        expected_flag
        in logs["flag_change"],
        (
            "flag_change log does not "
            f"contain {expected_flag}"
        ),
    )

    return True


def validate_experiment(
    records: list[dict[str, Any]],
    *,
    allow_dirty: bool,
    log_paths: dict[str, Path],
    require_logs: bool,
) -> dict[str, Any]:
    require(
        len(records) == 4,
        (
            "exactly four manifests "
            "are required"
        ),
    )

    for record in records:
        validate_schema(record)

    phases = index_by_phase(records)

    validate_timestamps(phases)

    validate_identity(
        records,
        allow_dirty=allow_dirty,
    )

    validate_build_identity(
        records,
        phases,
    )

    validate_artifacts(phases)
    validate_smoke(records)

    logs_validated = validate_logs(
        phases,
        log_paths,
        require_logs=require_logs,
    )

    cold_sha = (
        phases["cold"]
        ["artifact"]
        ["sha256"]
    )
    changed_sha = (
        phases["flag_change"]
        ["artifact"]
        ["sha256"]
    )

    return {
        "run_id": phases["cold"]["run_id"],
        "cache_dir": (
            phases["cold"]["cache_dir"]
        ),
        "git_revision": (
            phases["cold"]
            ["git"]
            ["revision"]
        ),
        "logs_validated": logs_validated,
        "probe_changed": (
            phases["flag_change"]
            ["build"]
            ["probe_define"]
        ),
        "binary_sha_changed": (
            cold_sha != changed_sha
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a four-phase "
            "PyTorch extension build experiment."
        )
    )

    parser.add_argument(
        "manifests",
        type=Path,
        nargs=4,
        help=(
            "Four manifest JSON files. "
            "Order does not matter."
        ),
    )

    parser.add_argument(
        "--log",
        action="append",
        default=[],
        metavar="PHASE=PATH",
        help=(
            "Build log for one phase. "
            "Repeat for all four phases."
        ),
    )

    parser.add_argument(
        "--require-logs",
        action="store_true",
        help=(
            "Fail unless complete build-log "
            "evidence is provided."
        ),
    )

    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Allow a dirty repository only "
            "when its status hash is stable "
            "across all phases."
        ),
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        records = [
            load_manifest(path)
            for path in args.manifests
        ]

        log_paths = parse_phase_paths(
            args.log,
            option_name="--log",
        )

        summary = validate_experiment(
            records,
            allow_dirty=args.allow_dirty,
            log_paths=log_paths,
            require_logs=args.require_logs,
        )

    except ValidationError as exc:
        print(
            f"FAILED: {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        "PASS schema: "
        "four valid schema-v1 manifests"
    )
    print(
        "PASS identity: "
        "run/cache/git/environment are stable"
    )
    print(
        "PASS warm: "
        "artifact fingerprint is unchanged"
    )
    print(
        "PASS flag_change: "
        "probe reached the build inputs"
    )
    print(
        "PASS flag_warm: "
        "artifact fingerprint is unchanged"
    )

    if summary["logs_validated"]:
        print(
            "PASS logs: "
            "cold/change built and "
            "warm phases reported no work"
        )
    else:
        print(
            "SKIP logs: "
            "manifest contracts passed, "
            "but no-work evidence was "
            "not validated"
        )

    print(
        "INFO binary_sha_changed:",
        summary["binary_sha_changed"],
    )

    print(
        "RESULT: build experiment passed"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())