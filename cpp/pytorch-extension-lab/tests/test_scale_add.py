# conftest.py or module-level loader
import pytest
import torch

@pytest.mark.parametrize(
    "shape",
    [(0,), (1,), (2,3), (4,5,6)],
)
def test_scale_add_matches_reference(shape):
    x = torch.randn(shape, dtype=torch.float32)
    actual = torch.ops.jay_ops.scale_add(x, 2.5)
    expected = x * 2.5 + 1.0
    torch.testing.assert_close(actual, expected)
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert actual.device == x.device

def test_scale_add_is_functional():
    x = torch.tensor([1.0, 2.0])
    before = x.clone()

    out = torch.ops.jay_ops.scale_add(x, 3.0)

    torch.testing.assert_close(x, before)
    assert out.data_ptr() != x.data_ptr()

def test_rejects_float64():
    x = torch.ones(3, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="float32"):
        torch.ops.jay_ops.scale_add(x, 2.0)

def test_rejects_non_contiguous():
    x = torch.randn(2,3).t()
    assert not x.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        torch.ops.jay_ops.scale_add(x, 2.0)

def test_rejects_non_finite_alpha():
    x = torch.ones(3)
    with pytest.raises(RuntimeError, match="finite"):
        torch.ops.jay_ops.scale_add(x, float("nan"))

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason = "CUDA is unavailable",
)
def test_cuda_has_no_registered_kernel():
    x = torch.ones(3, device="cuda")
    with pytest.raises(RuntimeError):
        torch.ops.jay_ops.scale_add(x, 2.0)