from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME,load


@pytest.fixture(scope="session", autouse=True)
def load_jay_ops():
    lab_root = Path(__file__).resolve().parents[1]
    source_cpu = lab_root / "csrc" / "scale_add_cpu.cpp"
    source_cuda = lab_root / "csrc" / "scale_add_cuda.cu"

    build_cuda = (
    CUDA_HOME is not None
    and torch.version.cuda is not None
    )

    sources = [str(source_cpu)]
    if build_cuda:
        sources.append(str(source_cuda))

    load(
        name="jay_ops_test",
        sources=sources,
        with_cuda=build_cuda,
        is_python_module=False,
    )

    yield