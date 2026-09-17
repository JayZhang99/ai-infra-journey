from pathlib import Path
import sys

import pytest
from torch.utils.cpp_extension import load


lab_root = Path(__file__).resolve().parents[1]
helper_directory = lab_root / "python"

if str(helper_directory) not in sys.path:
    sys.path.insert(
        0,
        str(helper_directory),
    )

from extension_build import make_load_kwargs
from runtime_registrations import (
    register_runtime_kernels
)



@pytest.fixture(scope="session", autouse=True)
def load_jay_ops():
    load(
        **make_load_kwargs(
            name="jay_ops_test",
            verbose=False,
        )
    )
    register_runtime_kernels()
    yield
