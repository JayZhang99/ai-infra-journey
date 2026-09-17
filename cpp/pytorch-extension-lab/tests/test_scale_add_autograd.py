import pytest
import torch


def test_scale_add_backward_cpu():
    x = torch.randn(3, 4,
        dtype=torch.float32,
        requires_grad=True)
    alpha = 2.5
    y = torch.ops.jay_ops.scale_add(x, alpha)
    y.sum().backward()

    torch.testing.assert_close(
        x.grad,
        torch.full_like(x, alpha)
    )

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable"
)
@pytest.mark.parametrize(
    "op_name", ["scale_add", "reduce_sum"]
)
def test_backward_cuda(op_name):
    x = torch.randn(2,3, device="cuda", requires_grad=True)
    if op_name == "scale_add":
        y = torch.ops.jay_ops.scale_add(x, 2.0)
        expected = torch.full_like(x, 2.0)
    else:
        y = torch.ops.jay_ops.reduce_sum(x)
        expected = torch.ones_like(x)
    y.sum().backward()
    torch.testing.assert_close(x.grad, expected)

def test_scale_add_full_opcheck():
    x = torch.randn(3, 4,
        dtype=torch.float32,
        requires_grad=True)
    result = torch.library.opcheck(
        torch.ops.jay_ops.scale_add.default,
        (x, 2.5),
        test_utils=(
            "test_schema",
            "test_autograd_registration",
            "test_faketensor",
            "test_aot_dispatch_dynamic",
        ),
        atol=1e-4, rtol=1e-4
    )
    assert set(result.values()) == {"SUCCESS"}


def test_scale_add_gradcheck():
    x = torch.randn(3, 4,
        dtype=torch.float32,
        requires_grad=True)
    
    assert torch.autograd.gradcheck(
        lambda z: torch.ops.jay_ops.scale_add(
            z, 2.5
        ),
        (x,),
        eps=1e-3,
        atol=1e-2,
        rtol=1e-2,
        fast_mode=True
    )

