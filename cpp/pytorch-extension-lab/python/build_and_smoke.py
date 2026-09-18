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
        "--fingerprint-output",
        type=Path,
        required=True,
        help="Path used to save the artifact fingerprint.",
    )

    parser.add_argument(
        "--dump-dispatch",
        action="store_true",
        help=(
            "Print Dispatcher tables before and "
            "after Python runtime registration."
        ),
)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    kwargs = make_load_kwargs(
                name="jay_ops",
                verbose=True,
            )

    probe = os.environ.get("JAY_PROBE_DEFINE")
    if probe:
        kwargs["extra_cflags"].append(f"-D{probe}")
        if kwargs["with_cuda"]:
            kwargs["extra_cuda_cflags"].append(f"-D{probe}")

    library_path = Path(
        load(
            **kwargs
        )
    )

    if args.dump_dispatch:
        dump_dispatch_tables(
            stage="after_library_load",
        )

    register_runtime_kernels()

    if args.dump_dispatch:
        dump_dispatch_tables(
            stage="after_runtime_registration",
        )

    # Smoke test
    x = torch.tensor(
        [1.0, 2.0],
        dtype=torch.float32,
    )

    actual = torch.ops.jay_ops.scale_add(
        x,
        2.0,
    )
    expected = x * 2.0 + 1.0

    torch.testing.assert_close(
        actual,
        expected,
    )

    save_fingerprint(
        artifact=library_path,
        output=args.fingerprint_output,
        metadata={
            "probe_define": probe,
            "with_cuda": kwargs["with_cuda"],
            "extra_cflags": kwargs["extra_cflags"],
            "extra_cuda_cflags": kwargs.get(
                "extra_cuda_cflags",
                [],
            ),
        },
    )


if __name__ == "__main__":
    main()