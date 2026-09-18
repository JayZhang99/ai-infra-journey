from pathlib import Path
import torch
from torch.utils.cpp_extension import load
from extension_build import make_load_kwargs
from runtime_registrations import (
    register_runtime_kernels
)
from artifact_fingerprint import save_fingerprint


library_path = Path(
    load(
        **make_load_kwargs(
            name="jay_ops",
            verbose=True,
        )
    )
)


register_runtime_kernels()

# 构建产物指纹
record = save_fingerprint(
    artifact=library_path,
    output=Path(
        "benchmarks/builds/"
        "2026-09-18_mac_cold.json"
    ),
)

print(record)