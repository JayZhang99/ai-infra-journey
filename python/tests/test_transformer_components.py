from __future__ import annotations

import math
import pytest
import torch

from python.transformer_components import RMSNorm
from python.transformer_components import SwiGLU
from python.transformer_components import TinyFFNBlock
from python.transformer_components import MultiHeadAttention



def rmsnorm_reference(x, weight, eps):
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x.float() * torch.rsqrt(variance + eps)
    return y.to(x.dtype) * weight.to(dtype=x.dtype)

def test_rmsnorm_mathes_reference():
    torch.manual_seed(0)
    x = torch.randn(2,3,8)
    mod = RMSNorm(8)
    expected = rmsnorm_reference(x, mod.weight, mod.eps)
    actual = mod(x)
    torch.testing.assert_close(actual, expected)

def test_rmsnorm_contract():
    x = torch.randn(2,3,8)
    mod = RMSNorm(8)
    y = mod(x)

    assert y.dtype == x.dtype
    assert y.shape == x.shape
    assert y.device == x.device
    assert dict(mod.named_parameters())["weight"].shape ==(8,)

def test_rmsnorm_rejects_bad_last_dim():
    with pytest.raises(ValueError):
        RMSNorm(8)(torch.randn(2,3,7))

def tset_swiglu_matches_reference():
    torch.manual_seed(0)
    mod = SwiGLU(d_model=8, d_ff=16, bias=False)
    x = torch.rand(2, 3, 8)

    gate = F.Linear(x, mod.gate_proj.weight)
    up = F.Linear(x, mod.up_proj.weight)
    expected = F.linear(
        F.silu(gate) * up,
        mod.down_proj.weight
    )
    actual = mod(x)
    torch.testing.assert_close(actual, expected)

def tset_swiglu_parameters_and_gradients():
    mod = SwiGLU(8, 16)
    x = torch.randn(2, 3, 8, requires_grad=True)
    y = mod(x)
    assert y.shape == (2,3,8)
    assert torch.isfinite(y).all()

    y.square().mean.backward()
    assert x.grad is not None
    for parameter in mod.parameters():
        assert parameter.grad is not None
    
    assert mod.gate_proj.weight.shape ==(16, 8)
    assert mod.down_proj.weight.shape ==(8, 16)

def tset_tiny_ffn_block_matches_compositino():
    torch.manual_seed(0)
    block = TinyFFNBlock(8, 16)
    x = torch.randn(2,3,8)
    expected = x + block.ffn(block.norm(x))
    actual = block(x)
    torch.testing.assert_close(actual, expected)

def test_tiny_ffn_block_preserves_constract():
    block = TinyFFNBlock(8, 16)
    x = torch.randn(2,3,8, requires_grad = True)
    y = block(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None

def test_split_merge_roundtrip():
    mod = MultiHeadAttention(d_model=8, num_heads=2)
    x = torch.randn(2,4,8)
    heads = mod._split_heads(x)
    assert heads.shape == (2,2,4,4)
    restored = mod._merge_heads(heads)
    torch.testing.assert_close(restored, x)

def test_mha_smoke():
    torch.manual_seed(0)
    mod = MultiHeadAttention(8,2)
    x = torch.randn(2,4,8)
    output ,weights = mod(x)

    assert output.shape == (2,4,8)
    assert weights.shape == (2,2,4,4)
    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones(2,2,4)
    )


def test_mha_causal_mask():
    mod = MultiHeadAttention(8,2)
    x = torch.randn(2,4,8)
    mask = torch.tril(
        torch.ones(4,4,dtype=torch.bool)
    )
    _,weights = mod(x, mask)

    blocked = ~mask.view(1,1,4,4)
    actual = weights.masked_select(
        blocked.expand_as(weights)
    )
    torch.testing.assert_close(
        actual, torch.zeros_like(actual)
    )

def test_mha_matches_reference():
    torch.manual_seed(0)
    mod =MultiHeadAttention(8,2)
    x = torch.randn(2,4,8)

    q = mod._split_heads(mod.q_proj(x))
    k = mod._split_heads(mod.k_proj(x))
    v = mod._split_heads(mod.v_proj(x))
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(mod.head_dim)
    expected_w = torch.softmax(scores, dim=-1)
    expected_y = mod.out_proj(mod._merge_heads(expected_w @ v))

    actual_y, actual_w = mod(x)
    torch.testing.assert_close(actual_w, expected_w)
    torch.testing.assert_close(actual_y, expected_y)

def test_mha_rejects_invalid_config():
    with pytest.raises(ValueError):
        MultiHeadAttention(10, 3)

def test_mha_rejects_bad_rank():
    mod = MultiHeadAttention(8,2)
    with pytest.raises(ValueError):
        mod(torch.randn(4,8))

def test_mha_rejects_fully_masked_row():
    mod = MultiHeadAttention(8,2)
    x = torch.randn(1,4,8)
    mask = torch.tril(torch.ones(4,4, dtype = torch.bool))
    mask[2] = False
    with pytest.raises(ValueError):
        mod(x, mask)

@pytest.mark.parametrize("device",[
    "cpu",
    pytest.param(
        "cuda",
        marks = pytest.mark.skipif(
            not torch.cuda.is_available(),
            reason="CUDA unavailable",
        ),
    ),
])

def test_mha_device(device):
    mod = MultiHeadAttention(8,2).to(device)
    x = torch.randn(2,4,8, device = device)
    y,w = mod(x)
    assert y.device.type == device
    assert w.device.type == device


