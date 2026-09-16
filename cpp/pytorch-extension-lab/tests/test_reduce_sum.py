import pytest
import torch


@pytest.mark.parametrize(
    "dev",[
        "cpu", "cuda:0"
    ]
)
def test_reduce_sum_reference(dev):
    if dev.startswith("cuda"):
        pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
    x = torch.randn(4,33,
                    device = dev,
                    dtype = torch.float32)
    actual = torch.ops.jay_ops.reduce_sum(x)
    expected = torch.sum(x)
    torch.testing.assert_close(
        actual, expected, rtol=1e-4, atol=1e-5
    )
    assert actual.shape == torch.Size([])
    assert actual.dtype == x.dtype
    assert actual.device == x.device

def test_reduce_sum_contracts():
    op = torch.ops.jay_ops.reduce_sum
    with pytest.raises(RuntimeError,match="float32"):
        op(torch.ones(3, dtype = torch.float64))

    x = torch.ones(2,3).t()
    assert not x.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        op(x)

def test_reduce_sum_empty():
    y = torch.ops.jay_ops.reduce_sum(
        torch.empty(0, dtype=torch.float32)
    )              
    assert y.shape == torch.Size([])
    assert y.item() == 0.0

def test_reduce_sum_schema():
    x = torch.randn(17, dtype=torch.float32)
    result = torch.library.opcheck(
        torch.ops.jay_ops.reduce_sum.default,
        (x,),
        test_utils=("test_schema",),
    )
    assert result["test_schema"] == "SUCCESS"
    
def test_reduce_sum_current_stream():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.empty(1<<20, device="cuda")
        x.fill_(2.0)
        y = torch.ops.jay_ops.reduce_sum(x)
        z = y * 3.0
    stream.synchronize()
    expected = float((1<<20) * 6)
    assert z.item() == pytest.approx(
        expected, rel=1e-4
    )
