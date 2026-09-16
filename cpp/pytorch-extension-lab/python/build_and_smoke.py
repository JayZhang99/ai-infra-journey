from pathlib import Path
import torch
from torch.utils.cpp_extension import CUDA_HOME,load
from extension_build import make_load_kwargs


load(
    **make_load_kwargs(
        name="jay_ops",
        verbose=True,
    )
)