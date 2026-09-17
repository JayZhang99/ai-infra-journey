import pytest
import torch

def test_reduce_sum_backward_cpu():
    x = torch.randn(2,3,
        dtype=torch.float32,
        requires_grad=True)

    y = torch.ops.jay_ops.reduce_sum(x)
    (3.0 * y).backward()

    torch.testing.assert_close(
        x.grad,
        torch.full_like(x, 3.0)
    )


def test_reduce_sum_opcheck():
    x = torch.randn(3, 4,
        dtype=torch.float32,
        requires_grad=True)
    result = torch.library.opcheck(
        torch.ops.jay_ops.reduce_sum.default,
        (x,),
        test_utils=(
            "test_schema",
            "test_autograd_registration",
            "test_faketensor",
            "test_aot_dispatch_dynamic",
        ),
        atol=1e-4, rtol=1e-4
    )
    assert set(result.values()) == {"SUCCESS"}


def test_reduce_sum_gradcheck():
    x = torch.randn(3, 4,
        dtype=torch.float32,
        requires_grad=True)
    
    assert torch.autograd.gradcheck(
        lambda z: torch.ops.jay_ops.reduce_sum(
            z
        ),
        (x,),
        eps=1e-3,
        atol=1e-2,
        rtol=1e-2,
        fast_mode=True
    )