from pathlib import Path
import torch
from torch.utils.cpp_extension import load
from extension_build import make_load_kwargs
from runtime_registrations import (
    register_runtime_kernels
)


load(
    **make_load_kwargs(
        name="jay_ops",
        verbose=True,
    )
)

register_runtime_kernels()