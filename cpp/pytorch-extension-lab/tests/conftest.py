from pathlib import Path

import pytest
from torch.utils.cpp_extension import load


@pytest.fixture(scope="session", autouse=True)
def load_jay_ops():
    lab_root = Path(__file__).resolve().parents[1]
    source = lab_root / "csrc" / "scale_add_cpu.cpp"

    load(
        name="jay_ops_cpu_test",
        sources=[str(source)],
        extra_cflags=["-O2"],
        verbose=False,
        is_python_module=False,
    )

    yield