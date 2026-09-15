from pathlib import Path
import torch
from torch.utils.cpp_extension import CUDA_HOME,load

root = Path(__file__).resolve().parents[1]
source_cpu = root / "csrc/scale_add_cpu.cpp"
source_cuda = root / "csrc/scale_add_cuda.cu"

build_cuda = (
    CUDA_HOME is not None
    and torch.version.cuda is not None
)

sources = [str(source_cpu)]

if build_cuda:
    sources.append(str(source_cuda))

load(
    name = "jay_ops_cpu",
    sources=sources,
    extra_cflags = ["-O2"],
    extra_cuda_cflags = ["-O2"],
    with_cuda=build_cuda,
    verbose = True,
    is_python_module = False,
)

x = torch.tensor([1.0, 2.0])
print(torch.ops.jay_ops.scale_add(x, 3.0))