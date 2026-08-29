from __future__ import annotations

import torch
import math

def stable_softmax(
    x:torch.Tensor,
    dim:int = -1,
)-> torch.Tensor:
    
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            "x must be a torch.Tensor"
        )

    if not x.is_floating_point():
        raise TypeError(
            "x must be a floating-point Tensor"
        )

    if x.ndim == 0:
        raise ValueError(
            "x must have at least one dimension"
        )

    normalized_dim = dim
    if normalized_dim < 0 :
        normalized_dim += x.ndim

    if not 0 <= normalized_dim < x.ndim:
        raise IndexError(
            f"dim is out of range: {dim}"
        )

    if x.shape[normalized_dim] == 0:
        raise ValueError(
            "Softmax dimension must not be empty"
        )

    if torch.isnan(x).any():
        raise ValueError(
            "x must not contain NaN"
        )

    if torch.isposinf(x).any():
        raise ValueError(
            "x must not contain positive infinity"
        )

    has_finite_value = torch.isfinite(x).any(
        dim=normalized_dim,
        keepdim=True,
    )

    if not bool(has_finite_value.all().item()):
        raise ValueError(
            "Every Softmax row must contain "
            "at least one finite value"
        )

 # 第一步：找到每行最大值。
    maximum = x.amax(
        dim=normalized_dim,
        keepdim=True,
    )

    # 第二步：减去最大值。
    # Softmax 具有平移不变性，因此结果不变。
    shifted = x - maximum

    # 第三步：逐元素计算指数。
    exp_values = shifted.exp()

    # 第四步：计算每行指数和。
    denominator = exp_values.sum(
        dim=normalized_dim,
        keepdim=True,
    )

    # 第五步：归一化为概率。
    return exp_values / denominator

def make_causal_mask(
    query_length: int,
    key_length: int | None = None,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:

    lengths = {
        "query_length": query_length,
        "key_length": (
            query_length
            if key_length is None
            else key_length
        ),
    }

    for name, value in lengths.items():
        # bool 是 int 的子类，需要单独排除。
        if isinstance(value, bool):
            raise TypeError(
                f"{name} must be an integer, "
                "not bool"
            )

        if not isinstance(value, int):
            raise TypeError(
                f"{name} must be an integer, "
                f"got {type(value).__name__}"
            )

        if value <= 0:
            raise ValueError(
                f"{name} must be greater than 0, "
                f"got {value}"
            )

        resolved_key_length = lengths["key_length"]

    # 今天只实现方阵 self-attention。
    if query_length != resolved_key_length:
        raise ValueError(
            "Only square self-attention masks "
            "are supported, got "
            f"query_length={query_length}, "
            f"key_length={resolved_key_length}"
        )

    resolved_device = torch.device(device)

    # Query 位置：[T, 1]
    query_positions = torch.arange(
        query_length,
        device=resolved_device,
    ).unsqueeze(1)

    # Key 位置：[1, T]
    key_positions = torch.arange(
        resolved_key_length,
        device=resolved_device,
    ).unsqueeze(0)

    # 第 i 个 Query 只能关注 j <= i 的 Key。
    #
    # [T,1] >= [1,T]
    # 通过广播得到 [T,T]。
    return query_positions >= key_positions

def validate_rank3_floating(x, name):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"{name} must be a Tensor")
        
        if x.ndim != 3:
            raise ValueError(f"{name} must be rank-3")
        
        if not x.is_floating_point():
            raise TypeError(f"{name} must be floating")

def validate_attention_inputs(
    q,k,v,mask = None
):
    validate_rank3_floating(q,"q")
    validate_rank3_floating(k,"k")
    validate_rank3_floating(v,"v")

    B_q, T_q, D_q = q.shape
    B_k, T_k, D_k = k.shape
    B_v, T_v, D_v = v.shape

    if not (B_q == B_k == B_v):
        raise IndexError(f"batch mismatch: q={B_q}, k={B_k}, v={B_v}")
    
    # Q 和 K 的点积维度必须相同
    if D_q != D_k:
        raise ValueError(
            f"q/k feature mismatch: q={D_q}, k={D_k}"
        )

    # 每个 Key 必须有对应的 Value
    if T_k != T_v:
        raise ValueError(
            f"k/v token mismatch: k={T_k}, v={T_v}"
        )


    if not (q.dtype == k.dtype == v.dtype):
        raise TypeError(
            f"dtype mismatch: q={q.dtype}, "
            f"k={k.dtype}, v={v.dtype}"
        )

    if not (q.device == k.device == v.device):
        raise ValueError(
            f"device mismatch: q={q.device}, "
            f"k={k.device}, v={v.device}"
        )
        
    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            raise TypeError("mask must be a Tensor")

        if mask.dtype != torch.bool:
            raise TypeError("mask must be bool")

        if mask.device != q.device:
            raise ValueError(
                f"mask device {mask.device} "
                f"does not match q device {q.device}"
            )
        

        expected_shape = (B_q, T_q, T_k)
        try:
            torch.broadcast_shapes(
                mask.shape,
                expected_shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} cannot "
                f"broadcast to {expected_shape}"
            ) from exc
        expanded_mask = torch.broadcast_to(
            mask,
            expected_shape,
        )

        if not expanded_mask.any(dim=-1).all().item():
            raise ValueError("mask contains a fully masked query row")
    


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
)-> tuple[torch.Tensor, torch.Tensor]:
    validate_attention_inputs(q,k,v,mask)

    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        if not bool(mask.any(dim=-1).all().item()):
            raise ValueError("Fully masked now")
        scores = scores.masked_fill(
            ~mask,
            float("-inf")
        )
    
    weights = stable_softmax(scores, dim=-1)
    output = weights @ v
    return output, weights

def validate_new_pair(k_new, v_new):
    validate_rank3_floating(k_new, "k_new")
    validate_rank3_floating(v_new, "v_new")
    b_k, t_k , _ = k_new.shape
    b_v, t_v, _ = v_new.shape

    if not b_k == b_v:
        raise ValueError("Batch mismatchs")

    if not t_k == t_v:
        raise ValueError("Token mismatchs")

    if not k_new.dtype == v_new.dtype:
        raise TypeError("dtype mismatches")
    
    if not k_new.device == v_new.device:
        raise ValueError("k_new device mismatches v_new device")

def validate_cache_pair(
    k_cache, v_cache,
    k_new, v_new
):
    k_cache_b, k_cache_t, k_cache_d = k_cache.shape
    v_cache_b, v_cache_t, v_cache_d = v_cache.shape
    k_new_b, k_new_t, k_new_d = k_new.shape
    v_new_b, v_new_t, v_new_d = v_new.shape

    if not (k_cache_b == k_new_b and v_cache_b == v_new_b):
        raise ValueError("Batch dismatches")
    
    if not(k_cache_d == k_new_d and v_cache_d == v_new_d):
        raise ValueError("D dismatches")

    if not k_cache.dtype == k_new.dtype == v_cache.dtype == v_new.dtype:
        raise TypeError("dtype mismatches")
    
    if not k_cache.device == k_new.device == v_cache.device == v_new.device:
        raise ValueError("k_new device mismatches v_new device")
    

def append_kv_cache(
    k_cache, v_cache, k_new, v_new,
):
    validate_new_pair(k_new, v_new)
    
    if k_cache is None and v_cache is None:
        return k_new, v_new

    if(k_cache is None) != (v_cache is None):
        raise ValueError("incomplete cache pair")
    
    validate_cache_pair(k_cache, v_cache, k_new, v_new,)

    return (
        torch.cat([k_cache, k_new], dim = -2),
        torch.cat([v_cache, v_new], dim = -2),
    )
    