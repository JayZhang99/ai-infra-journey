from pathlib import Path
import torch
from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parents[1]
source = root / "csrc/scale_add_cpu.cpp"

load(
    name = "jay_ops_cpu",
    sources = [str(source)],
    extra_cflags = ["-O2"],
    verbose = True,
    is_python_module = False,
)

x = torch.tensor([1.0, 2.0])
print(torch.ops.jay_ops.scale_add(x, 3.0))