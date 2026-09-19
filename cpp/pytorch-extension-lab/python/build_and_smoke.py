from pathlib import Path
import argparse
import torch
import os
from torch.utils.cpp_extension import load
from extension_build import make_load_kwargs
from runtime_registrations import (
    register_runtime_kernels
)
from artifact_fingerprint import save_fingerprint

from build_manifest import (
    create_manifest,
    require_cache_dir,
    save_manifest,
)

import re


PROBE_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*$"
)


def validate_probe(
    probe: str | None,
) -> None:
    if probe is None:
        return

    if not PROBE_PATTERN.fullmatch(probe):
        raise ValueError(
            "JAY_PROBE_DEFINE must be a "
            "valid C/C++ macro identifier"
        )

def dump_dispatch_tables(
    *,
    stage: str,
) -> None:
    operators = (
        "jay_ops::scale_add",
        "jay_ops::reduce_sum",
    )

    print()
    print(f"===== Dispatcher: {stage} =====")

    for operator in operators:
        dispatch_table = (
            torch._C._dispatch_dump_table(
                operator
            )
        )

        if not dispatch_table.strip():
            raise RuntimeError(
                "Dispatcher table is empty for "
                f"{operator} at stage={stage}"
            )

        print()
        print(f"--- {operator} ---")
        print(dispatch_table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-id",
        required=True,
        help=(
            "Identity shared by cold, warm, "
            "flag_change and flag_warm."
        ),
    )

    parser.add_argument(
        "--phase",
        required=True,
        choices=[
            "cold",
            "warm",
            "flag_change",
            "flag_warm",
        ],
    )

    parser.add_argument(
        "--manifest-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--fingerprint-output",
        type=Path,
        default=None,
        help=(
            "Optional legacy standalone "
            "fingerprint JSON."
        ),
    )

    parser.add_argument(
        "--dump-dispatch",
        action="store_true",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    cache_dir = require_cache_dir()
    repo_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    kwargs = make_load_kwargs(
        name="jay_ops",
        verbose=True,
    )

    probe = os.environ.get(
        "JAY_PROBE_DEFINE"
    )
    validate_probe(probe)

    if probe:
        kwargs["extra_cflags"].append(
            f"-D{probe}"
        )

        if kwargs["with_cuda"]:
            kwargs[
                "extra_cuda_cflags"
            ].append(
                f"-D{probe}"
            )

    library_path = Path(
        load(**kwargs)
    ).resolve()

    if args.dump_dispatch:
        dump_dispatch_tables(
            stage="after_library_load",
        )

    register_runtime_kernels()

    if args.dump_dispatch:
        dump_dispatch_tables(
            stage=(
                "after_runtime_registration"
            ),
        )

    x = torch.tensor(
        [1.0, 2.0],
        dtype=torch.float32,
    )

    actual = (
        torch.ops.jay_ops.scale_add(
            x,
            2.0,
        )
    )
    expected = x * 2.0 + 1.0

    torch.testing.assert_close(
        actual,
        expected,
    )

    smoke = {
        "passed": True,
        "operator": (
            "jay_ops::scale_add"
        ),
        "alpha": 2.0,
        "input": {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
        },
        "output": {
            "shape": list(
                actual.shape
            ),
            "dtype": str(
                actual.dtype
            ),
            "device": str(
                actual.device
            ),
        },
        "max_abs_error": float(
            (
                actual - expected
            )
            .abs()
            .max()
            .item()
        ),
    }

    manifest = create_manifest(
        run_id=args.run_id,
        phase=args.phase,
        repo_root=repo_root,
        cache_dir=cache_dir,
        artifact=library_path,
        load_kwargs=kwargs,
        probe_define=probe,
        smoke=smoke,
    )

    save_manifest(
        manifest=manifest,
        output=args.manifest_output,
    )

    if (
        args.fingerprint_output
        is not None
    ):
        save_fingerprint(
            artifact=library_path,
            output=(
                args.fingerprint_output
            ),
            metadata={
                "run_id": args.run_id,
                "phase": args.phase,
                "probe_define": probe,
            },
        )


if __name__ == "__main__":
    main()