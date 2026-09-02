from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError("x must be floating point")
        if x.ndim < 1 or x.shape[-1] != self.d_model:
            raise ValueError("last dimention must equal d_model")
        scale = x.float().pow(2).mean(-1,keepdim = True)
        normalized = x.float() * torch.rsqrt(scale + self.eps)

        return normalized.to(x.dtype) * self.weight.to(dtype=x.dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff: int, bias: bool = False):
        super().__init__()
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("dimentions must be positive")
        self.d_model = d_model
        self.d_ff = d_ff
        self.gate_proj = nn.Linear(d_model, d_ff, bias = bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias = bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias = bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2 or x.shape[-1] != self.d_model:
            raise ValueError("invalid input shape")
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(hidden)

    
class TinyFFNBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
    
    def forward(self, x = torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads :int, bias=False):
        super().__init__()
        if d_model <=0 or num_heads <=0:
            raise ValueError("dimensions must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def _merge_heads(self, x:torch.Tensor) -> torch.Tensor:
        bsz, heads, seq_len, head_dim = x.shape
        if heads != self.num_heads or head_dim != self.head_dim:
            raise ValueError("invalid head shape")
        return (
            x.transpose(1,2)
            .contiguous()
            .view(bsz, seq_len, self.d_model)
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, width = x.shape
        if width != self.d_model:
            raise ValueError("width must equal d_model")
        return (
            x.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1,2)
        )    

    def _apply_mask(self, scores, mask):
        if mask is None:
            return scores
        if mask.dtype != torch.bool:
            raise TypeError("mask must be bool")
        try:
            mask = torch.broadcast_to(mask, scores.shape)
        except RuntimeError as exc:
            raise ValueError("mask is not broadcastable") from exc
        if (~mask).all(dim=-1).any():
            raise ValueError("a query row is fully masked")
        return scores.masked_fill(~mask, float("-inf"))
    
    def forward(self, x:torch.Tensor, mask=None):
        if not x.is_floating_point():
            raise TypeError("x must be floating point")
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("x must have shape [B,T, d_model]")
        
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        scores = self._apply_mask(scores, mask)
        weights = torch.softmax(scores, dim=-1)
        context = weights @ v
        output = self.out_proj(self._merge_heads(context))
        return output, weights
    
    