from __future__ import annotations

import math
import pytest
import torch

from python.attention_playground import(
    stable_softmax,
    make_causal_mask,
    scaled_dot_product_attention,
    append_kv_cache,
    validate_new_pair,
    validate_cache_pair,
)


def test_stable_softmax():
    with pytest.raises(
        TypeError,
        match="x must be a torch.Tensor",
    ):
        stable_softmax(1.0)


    intTensor = torch.tensor([1], dtype = torch.int)
    with pytest.raises(
        TypeError,
        match="x must be a floating-point Tensor"
    ):
        stable_softmax(intTensor)
    

    x = torch.tensor(1.0, dtype=torch.float32)

    assert x.ndim == 0
    with pytest.raises(ValueError):
        stable_softmax(x)

    test_Tensor = torch.Tensor([1.0,2.0])
    with pytest.raises(
        IndexError,
        match="dim is out of range: -3"
    ):
        stable_softmax(test_Tensor, -3)

    nanTensor = torch.Tensor([torch.nan])
    
    with pytest.raises(
        ValueError,
        match="x must not contain NaN"
    ):
        stable_softmax(nanTensor)

    expected = torch.tensor([0.5, 0.5], dtype = torch.float32)
    test_input = torch.tensor([1.0 , 1.0], dtype = torch.float32)
    output = stable_softmax(test_input)
    assert torch.equal(output, expected)
    assert output.dtype == torch.float32
    assert output.shape == (2,)

def test_make_caual_mask():
    acutal = make_causal_mask(4)
    expected = torch.tensor([
        [1,0,0,0],
        [1,1,0,0],
        [1,1,1,0],
        [1,1,1,1],
    ],dtype = torch.bool)

    assert torch.equal(acutal, expected)
    assert acutal.dtype == torch.bool
    assert acutal.shape == (4,4)


def test_attention_contract():
    q = torch.randn(2,3,8)
    k = torch.randn(2,4,8)
    v = torch.randn(2,4,6)

    output, weights = scaled_dot_product_attention(q,k,v)

    assert weights.shape ==(2,3,4)
    assert output.shape == (2,3,6)

    torch.testing.assert_close(
        weights.sum(-1),
        torch.ones(2,3)
    )
    assert torch.isfinite(output).all()

def test_causal_weights_are_zero():
    q = torch.randn(2,4,8)
    k = torch.randn(2,4,8)
    v = torch.randn(2,4,8)
    mask = make_causal_mask(4)

    _,weights = scaled_dot_product_attention(q,k,v,mask)
    assert torch.equal(
        weights.masked_select(~mask),
        torch.zeros(12)
    )


def test_softmax_extreme_values():
    x = torch.tensor([
        [1000.,1001.,1002.],
        [-1000.,-1001.,-1002.],
    ])
    actual = stable_softmax(x)
    expected = torch.softmax(x,dim=-1)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()

def test_reject_fully_masked_row():
    q = torch.randn(1,2,4)
    k = torch.randn(1,2,4)
    v = torch.randn(1,2,4)
    mask = torch.tensor([
        [True,False],
        [False,False]
    ])
    with pytest.raises(ValueError):
        scaled_dot_product_attention(q,k,v,mask)

def test_Dk_not_same():
    q = torch.randn(1,2,3)
    k = torch.randn(1,2,4)
    v = torch.randn(1,2,4)
    
    with pytest.raises(ValueError):
        scaled_dot_product_attention(q,k,v)

def test_Tk_not_same():
    q = torch.randn(1,2,4)
    k = torch.randn(1,2,4)
    v = torch.randn(1,1,4)

    with pytest.raises(ValueError):
        scaled_dot_product_attention(q,k,v)

def test_int_input():
    q = torch.tensor([[
        [1,2],
        [2,3]
    ]
    ],dtype = torch.int)
    k = torch.tensor([[
        [1,2],
        [2,3]
    ]
    ],dtype = torch.int)
    v = torch.tensor([[
        [1,2],
        [2,3]
    ]
    ],dtype = torch.int)

    with pytest.raises(TypeError):
        scaled_dot_product_attention(q,k,v)
    
def test_wrong_dtype():

    q = torch.tensor([[
        [1,2],
        [2,3]
    ]
    ],dtype = torch.float16)
    k = torch.tensor([[
        [1,2],
        [2,3]
    ]
    ],dtype = torch.float16)
    v = torch.tensor([[
        [1,2],
        [2,3]
    ]
    ],dtype = torch.float32)

    with pytest.raises(TypeError):
        scaled_dot_product_attention(q,k,v)


def test_new_pair_batch_dismatch():
    k_new = torch.randn(1,2,3)
    v_new = torch.randn(2,3,1)

    with pytest.raises(ValueError):
        validate_new_pair(k_new, v_new)

def test_new_pair_token_dismatch():
    k_new = torch.randn(1,2,3)
    v_new = torch.randn(1,3,2)

    with pytest.raises(ValueError):
        validate_new_pair(k_new, v_new)

def test_new_pair_dtype_mismatch():
    k = torch.tensor([[
        [1,2],
        [2,3]
    ]
    ],dtype = torch.float16)
    v = torch.tensor([[
        [1,2],
        [2,3]
    ]
    ],dtype = torch.float32)

    with pytest.raises(TypeError):
        validate_new_pair(k, v)

def test_cache_pair_batch():
    k_cache = torch.randn(1,2,3) 
    v_cache = torch.randn(1,2,3)
    k_new = torch.randn(1,2,3)
    v_new = torch.randn(2,3,1)

    with pytest.raises(ValueError):
        validate_cache_pair(k_cache, v_cache, k_new, v_new)

def test_cache_pair_dimention():
    k_cache = torch.randn(1,2,3) 
    v_cache = torch.randn(1,2,3)
    k_new = torch.randn(1,2,3)
    v_new = torch.randn(1,3,2)

    with pytest.raises(ValueError):
        validate_cache_pair(k_cache, v_cache, k_new, v_new)

def test_cache_pair_dtype():
    k_cache = torch.randn(1,2,3)
    k_new = torch.randn(1,2,3)
    v_cache = torch.tensor([[
        [1,2],
        [2,3]
    ]
    ],dtype = torch.float16)

    v_new = torch.tensor([[
        [1,2],
        [2,3]
    ]
    ],dtype = torch.float32)

    with pytest.raises(TypeError):
        validate_cache_pair(k_cache, v_cache, k_new, v_new)

def test_append_kv_cache_None():
    k_cache = None
    v_cache = None
    k_new = torch.randn(1,2,3)
    v_new = torch.randn(1,2,3)

    k,v = append_kv_cache(k_cache, v_cache, k_new, v_new)
    torch.testing.assert_close(
        k, k_new
    )

    torch.testing.assert_close(
        v, v_new
    )

def test_append_kv_cache_incompkete():
    k_cache = None
    v_cache = torch.randn(1,2,3)

    k_new = torch.randn(1,2,3)
    v_new = torch.randn(1,2,3)

    with pytest.raises(ValueError):
        append_kv_cache(k_cache, v_cache, k_new, v_new)

def test_append_kv_cache():
    k_cache = torch.randn(1,2,3)
    v_cache = torch.randn(1,2,3)
    k_new = torch.randn(1,2,3)
    v_new = torch.randn(1,2,3)

    k_out, v_out = append_kv_cache(k_cache, v_cache, k_new, v_new)

    assert k_out.shape[-2] == k_cache.shape[-2] + k_new.shape[-2]
    k_now =  k_out[:, -k_new.shape[-2]:, :] 
    assert torch.equal(k_now, k_new)
    k_pre = k_out[:,:-k_new.shape[-2],:]
    assert torch.equal(k_cache,k_cache)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)

def test_not_same_device():
    q = torch.tensor([[
        [1,2],
        [2,3]
    ]
    ],device = torch.device("cpu"))
    k = torch.tensor([[
        [1,2],
        [2,3]
    ]
    ],device = torch.device("cpu"))
    v = torch.tensor([[
        [1,2],
        [2,3]
    ]
    ],device = torch.device("cuda:0"))

    with pytest.raises(ValueError):
        scaled_dot_product_attention(q,k,v)

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
        
def test_cache_pair_device():
    k_cache = torch.randn(1,2,3)
    k_new = torch.randn(1,2,3)
    v_cache = torch.tensor([[
        [1,2],
        [2,3]
    ]
    ],device = torch.device("cpu"))

    v_new = torch.tensor([[
        [1,2],
        [2,3]
    ]
    ],device = torch.device("cuda:0"))

    with pytest.raises(ValueError):
        validate_cache_pair(k_cache, v_cache, k_new, v_new)

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)

def test_new_pair_device_mismatch():
    k = torch.tensor([[
        [1,2],
        [2,3]
    ]
    ],device = torch.device("cpu"))
    v = torch.tensor([[
        [1,2],
        [2,3]
    ]
    ],device = torch.device("cuda:0"))

    with pytest.raises(ValueError):
        validate_new_pair(k, v)


def test_cached_last_mathes_full():
    torch.manual_seed(7)
    q = torch.randn(2,4,8)
    k = torch.randn(2,4,8)
    v = torch.randn(2,4,6)

    full, _ = scaled_dot_product_attention(q,k,v,make_causal_mask(4))
    cached, _ = scaled_dot_product_attention(q[:,-1:,:],k,v)

    torch.testing.assert_close(
        cached,
        full[:,-1:,:]
    )

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)

def test_attention_cuda_smoke():
    device = torch.device("cuda")
    q = torch.randn(2,4,8, device = device)
    k = torch.randn(2,4,8, device = device)
    v = torch.randn(2,4,6, device = device)
    mask = make_causal_mask(4, device=device)

    output, weights = scaled_dot_product_attention(q,k,v,mask)

    assert output.is_cuda
    assert weights.is_cuda
    assert torch.isfinite(output).all()

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_reject_device_mismatch():
    q = torch.randn(1,2,4, device = "cpu")
    k = torch.randn(1,2,4, device = "cpu")
    v = torch.randn(1,2,4, device = "cuda")

    with pytest.raises(
        ValueError,
        match = "device mismatch"
    ):
        scaled_dot_product_attention(q,k,v)

