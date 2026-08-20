import numpy as np
import pytest
import torch

from python.tensor_playground import (
    broadcasting_demo,
    cpu_cuda_roundtrip,
    describe_tensor,
    non_contiguous_demo,
    tensor_nbytes,
    reduction_and_matmul_demo,
)

def test_tensor_nbytes_fp16():
    result = tensor_nbytes((32, 2048, 4096), element_size=2)

    assert result == 536_870_912
    assert result / 1024**2 == 512
    assert result / 1024**3 == 0.5

def test_contiguous_shape_and_stride():
    x = torch.arange(24).reshape(2, 3, 4)
    description = describe_tensor(x)

    assert description["shape"] == (2, 3, 4)
    assert description["stride"] == (12, 4, 1)
    assert description["is_contiguous"] is True
    assert description["numel"] == 24

def test_transpose_shape_and_stride():
    x = torch.arange(24).reshape(2, 3, 4)
    y = x.transpose(1, 2)
    description = describe_tensor(y)

    assert description["shape"] == (2, 4, 3)
    assert description["stride"] == (12, 1, 4)
    assert description["is_contiguous"] is False

def test_permute_shape():
    x = torch.arange(24).reshape(2, 3, 4)
    y = x.permute(1, 0, 2)

    assert tuple(y.shape) == (3, 2, 4)
    assert tuple(y.stride()) == (4, 12, 1)

def test_broadcasting_output_shape():
    result = broadcasting_demo()

    assert tuple(result["left"].shape) == (8, 1, 128)
    assert tuple(result["right"].shape) == (1, 64, 128)
    assert tuple(result["output"].shape) == (8, 64, 128)

def test_incompatible_broadcast_raises():
    left = torch.zeros(2, 3)
    right = torch.zeros(4, 3)

    with pytest.raises(RuntimeError):
        _ = left + right

def test_non_contiguous_view_raises():
    result = non_contiguous_demo()

    assert result["view_failed"] is True
    assert result["transposed"].is_contiguous() is False
    assert result["contiguous"].is_contiguous() is True
    assert torch.equal(
        result["reshaped"],
        result["contiguous_view"],
    )


def test_reduction_and_matmul_shape():
    result = reduction_and_matmul_demo()

    assert tuple(result["sum_dim_1"].shape) == (2,4)
    assert tuple(result["mean_last_dim"].shape) == (2,3)
    assert tuple(result["matmul_output"].shape) == (2, 4)
    assert torch.equal(result["matmul_output"], torch.full((2, 4), 6.0))

def test_numpy_and_tensor_share_memory():
    array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    tensor = torch.from_numpy(array)

    tensor[0] = 99.0

    assert array[0] == 99.0



@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_cpu_cuda_roundtrip():
    result = cpu_cuda_roundtrip()

    assert result["available"] is True
    assert result["cuda_input_device"] == "cuda:0"
    assert torch.equal(result["cpu_output"], result["expected"])