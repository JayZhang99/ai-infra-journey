from __future__ import annotations

from math import prod
from typing import Any, Sequence

import numpy as np
import torch


def tensor_nbytes(shape: Sequence[int], element_size: int) -> int:
    """Return storage size in bytes for a dense tensor."""
    dimensions = tuple(shape)

    if any(not isinstance(dim, int) or isinstance(dim, bool) for dim in dimensions):
        raise TypeError("every dimension must be an integer")

    if any(dim < 0 for dim in dimensions):
        raise ValueError("tensor dimensions must be non-negative")

    if not isinstance(element_size, int) or isinstance(element_size, bool):
        raise TypeError("element_size must be an integer")

    if element_size <= 0:
        raise ValueError("element_size must be positive")

    return prod(dimensions) * element_size

def describe_tensor(x:torch.Tensor) -> dict[str, Any]:
    '''Return shape, stride, dtype, device and contiguity.'''
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    return {
        "shape": tuple(x.shape),
        "stride": tuple(x.stride()),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "layout": str(x.layout),
        "is_contiguous": x.is_contiguous(),
        "numel": x.numel(),
        "element_size": x.element_size(),
        "logical_nbytes": x.numel() * x.element_size(),
        "storage_offset": x.storage_offset(),
    }

def broadcasting_demo() -> dict[str, torch.Tensor]:
    '''Return [8,64,128] from two broadcastable inputs.'''
    left = torch.arange(
        8 * 1 * 128,
        dtype = torch.float32,
    ).reshape(8, 1, 128)

    right = torch.ones(
        1,
        64,
        128,
        dtype = torch.float32,
    )

    output = left + right

    return {
        "left": left,
        "right": right,
        "output": output,
    }
def non_contiguous_demo() -> dict[str, Any]:
    '''Show transpose, failed view and safe fixes.'''
    original = torch.arange(24).reshape(2, 3, 4)
    transposed = original.transpose(1,2)

    view_failed = False
    view_eroor = ""

    try:
        transposed.view(-1)
    except RuntimeError as error:
        view_failed = True
        view_eroor = str(error)

    #reshape 会在必要时复制
    reshaped = transposed.reshape(-1)

    contiguous = transposed.contiguous()
    contiguous_view = contiguous.view(-1)

    return {
        "original": original,
        "transposed": transposed,
        "view_failed": view_failed,
        "view_error": view_eroor,
        "reshaped": reshaped,
        "contiguous": contiguous,
        "contiguous_view": contiguous_view,
    }

def cpu_cuda_roundtrip() -> dict[str, Any]:
    '''Run CPU→CUDA→CPU when CUDA is available.'''
    if not torch.cuda.is_available():
        return {
            "available": False,
            "reason": "CUDA is not available in the current PyTorch environment",
        }
    
    device = torch.device("cuda:0")

    cpu_input = torch.arange(8, dtype = torch.float32)
    cuda_input = cpu_input.to(device)

    cuda_output = cuda_input + 1.0
    cpu_output = cuda_output.cpu()

    expected = cpu_input + 1.0

    if not torch.equal(cpu_output, expected):
        raise AssertionError("CPU/CUDA roundtrip produced an incorrect result")

    properties = torch.cuda.get_device_properties(device)

    return {
        "available": True,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "total_memory_bytes": properties.total_memory,
        "cpu_input": cpu_input,
        "cuda_input_device": str(cuda_input.device),
        "cpu_output": cpu_output,
        "expected": expected
    }

def numpy_sharing_demo() -> dict[str, Any]:
    """Demonstrate shared memory between NumPy and a CPU Tensor."""
    array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    tensor = torch.from_numpy(array)

    tensor[0] = 99.0

    return {
        "array": array,
        "tensor": tensor,
        "shared_memory": array[0] == tensor[0].item() == 99.0,
    }

def reduction_and_matmul_demo() -> dict[str, torch.Tensor]:
    """Demonstrate reduction and matrix multiplication."""
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    reduced_sum = x.sum(dim=1)
    reduced_mean = x.mean(dim=-1)

    left = torch.ones(2, 3)
    right = torch.full((3, 4), 2.0)
    matmul_output = left @ right

    return {
        "input": x,
        "sum_dim_1": reduced_sum,
        "mean_last_dim": reduced_mean,
        "matmul_output": matmul_output
    }

def main() -> None:
    x = torch.arange(24).reshape(2, 3, 4)
    print("Original:")
    print(describe_tensor(x))

    y = x.transpose(1, 2)
    print("\nTransposed:")
    print(describe_tensor(y))

    broadcasting = broadcasting_demo()
    print("\nBroadcasting:")
    print("left:", tuple(broadcasting["left"].shape))
    print("right:", tuple(broadcasting["right"].shape))
    print("output:", tuple(broadcasting["output"].shape))

    non_contiguous = non_contiguous_demo()
    print("\nNon-contiguous:")
    print("view failed:", non_contiguous["view_failed"])
    print("error:", non_contiguous["view_error"])
    print("reshape shape:", tuple(non_contiguous["reshaped"].shape))

    print("\nMemory:")
    print(
        "[32, 2048, 4096] FP16:",
        tensor_nbytes((32, 2048, 4096), element_size=2),
        "bytes",
    )

    numpy_result = numpy_sharing_demo()
    print("\nNumPy sharing:")
    print("shared_memory:", numpy_result["shared_memory"])

    cuda_result = cpu_cuda_roundtrip()
    print("\nCUDA:")
    if cuda_result["available"]:
        print("device:", cuda_result["device_name"])
        print("memory:", cuda_result["total_memory_bytes"])
        print("result:", cuda_result["cpu_output"])
    else:
        print("SKIP:", cuda_result["reason"])


if __name__ == "__main__":
    main()

