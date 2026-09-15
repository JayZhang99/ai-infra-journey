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


def test_scale_add_cpu():
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y = torch.ops.jay_ops.scale_add(x, 3.0)

    torch.testing.assert_close(
        y, torch.tensor([4.0, 7.0])
    )
    assert y.device.type == "cpu"
    assert y.dtype == x.dtype
    assert y.shape == x.shape
    assert y.data_ptr()!= x.data_ptr()

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason = "CUDA is unavailable",
)
def test_scale_add_cuda():
    x = torch.tensor([1.0, 2.0], device = "cuda:0", dtype=torch.float32)
    y = torch.ops.jay_ops.scale_add(x, 3.0)

    assert y.device == x.device
    assert y.dtype == x.dtype
    assert y.shape == x.shape
    torch.testing.assert_close(
        y.cpu(), torch.tensor([4.0, 7.0])
    )


DEVICE_CASES = [
    pytest.param(
        "cpu",
        id="cpu",
    ),
    pytest.param(
        "cuda:0",
        id="cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(),
            reason="CUDA is unavailable",
        ),
    ),
]

@pytest.mark.parametrize(
    "device",
    DEVICE_CASES,
)
def test_rdtype_non_contiguous(device):
    scale_add_op = torch.ops.jay_ops.scale_add

    with pytest.raises(RuntimeError, match="float32"):
        scale_add_op(torch.ones(3, device=device, dtype=torch.float64), 2.0)
    
    x = torch.randn(2,3, device=device).t()
    assert not x.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        scale_add_op(x, 2.0)

@pytest.mark.parametrize(
    "device",
    DEVICE_CASES,
)
def test_empty_non_finite(device):
    scale_add_op = torch.ops.jay_ops.scale_add

    x = torch.empty((0,3), device = device)
    y = scale_add_op(x,2.0)
    assert y.shape == (0,3)
    assert y.numel() == 0

    for alpha in [float("nan"), float("inf")]:
        with pytest.raises(RuntimeError, match="finite"):
            scale_add_op(torch.ones(3, device=device), alpha)

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason = "CUDA is unavailable",
)

def test_API_CUDA_CPU():
    scale_add_op = torch.ops.jay_ops.scale_add

    cpu_out = scale_add_op(torch.ones(4), 2.0)
    cuda_out = scale_add_op(torch.ones(3, device="cuda"), 2.0)

    assert cpu_out.device.type == "cpu"
    assert cuda_out.device.type == "cuda"

    print(torch._C._dispatch_dump_table("jay_ops::scale_add"))

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason = "CUDA is unavailable",
)
def test_curret_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.arange(1024, device="cuda", dtype=torch.float32)
        y = torch.ops.jay_ops.scale_add(x, 2.0)
        done = torch.cuda.Event()
        done.record()
    
    done.synchronize()
    torch.testing.assert_close(y, x * 2.0 + 1.0)

    