from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME


@dataclass(frozen=True)
class ExtensionBuildSpec:
    build_cuda: bool
    sources: tuple[str, ...]
    include_paths: tuple[str, ...]
    extra_cflags: tuple[str, ...]
    extra_cuda_cflags: tuple[str, ...]


def get_extension_build_spec() -> ExtensionBuildSpec:
    helper_path = Path(__file__).resolve()

    # .../cpp/pytorch-extension-lab
    extension_root = helper_path.parents[1]

    # .../cpp
    cpp_root = extension_root.parent

    # .../cpp/cuda-kernel-lab
    cuda_lab = cpp_root / "cuda-kernel-lab"

    csrc = extension_root / "csrc"

    scale_add_cpu = csrc / "scale_add_cpu.cpp"
    reduce_sum_cpu = csrc / "reduce_sum_cpu.cpp"

    scale_add_cuda = csrc / "scale_add_cuda.cu"
    reduce_sum_cuda = csrc / "reduce_sum_cuda.cu"

    reduction_cuda = (
        cuda_lab
        / "src"
        / "reduction.cu"
    )

    cuda_lab_include = (
        cuda_lab
        / "include"
    )

    # 判断当前 PyTorch 和本地 Toolkit 是否支持编译 CUDA。
    build_cuda = (
        CUDA_HOME is not None
        and torch.version.cuda is not None
    )

    sources: list[Path] = [
        scale_add_cpu,
        reduce_sum_cpu,
    ]

    include_paths: list[Path] = []

    if build_cuda:
        sources.extend(
            [
                scale_add_cuda,
                reduce_sum_cuda,
                reduction_cuda,
            ]
        )

        include_paths.append(cuda_lab_include)

    # 只检查本次真正参与构建的路径。
    required_paths = [
        *sources,
        *include_paths,
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        formatted = "\n".join(
            f"  - {path}"
            for path in missing_paths
        )

        raise FileNotFoundError(
            "Required extension build paths "
            f"do not exist:\n{formatted}"
        )

    return ExtensionBuildSpec(
        build_cuda=build_cuda,
        sources=tuple(
            str(path)
            for path in sources
        ),
        include_paths=tuple(
            str(path)
            for path in include_paths
        ),
        extra_cflags=("-O2",),
        extra_cuda_cflags=("-O2",),
    )

def make_load_kwargs(
    *,
    name: str,
    verbose: bool,
) -> dict:
    spec = get_extension_build_spec()

    kwargs = {
        "name": name,
        "sources": list(spec.sources),
        "extra_include_paths": list(
            spec.include_paths
        ),
        "extra_cflags": list(
            spec.extra_cflags
        ),
        "with_cuda": spec.build_cuda,
        "verbose": verbose,
        "is_python_module": False,
    }

    if spec.build_cuda:
        kwargs["extra_cuda_cflags"] = list(
            spec.extra_cuda_cflags
        )

    return kwargs