from __future__ import annotations

import pytest 
import torch
import math
import json

from python.benchmark_attention import (
    estimate_kv_cache_bytes,
    run_attention_case,
)

def test_estimate_kv_cache_bytes_fp16():
    actual = estimate_kv_cache_bytes(
        layers=32,
        batch=1,
        kv_heads=8,
        sequence=4096,
        head_dim=128,
        dtype=torch.float16,
    )
    assert actual == 536_870_912
    assert actual / 1024 **2 == 512

def test_kv_cache_bool_type():
    with pytest.raises(TypeError):
        estimate_kv_cache_bytes(
        layers=32,
        batch=True,
        kv_heads=8,
        sequence=4096,
        head_dim=128,
        dtype=torch.float16,)

def test_kv_cache_zero():
    with pytest.raises(ValueError):
        estimate_kv_cache_bytes(
        layers=32,
        batch=0,
        kv_heads=8,
        sequence=4096,
        head_dim=128,
        dtype=torch.float16,)

def test_kv_cache_negative():
    with pytest.raises(ValueError):
        estimate_kv_cache_bytes(
        layers=32,
        batch=-1,
        kv_heads=8,
        sequence=4096,
        head_dim=128,
        dtype=torch.float16,)

def test_kv_cache_wrongdtype():
    with pytest.raises(TypeError):
        estimate_kv_cache_bytes(
        layers=32,
        batch=1,
        kv_heads=8,
        sequence=4096,
        head_dim=128,
        dtype=int,)

def test_attention_benchmark_cpu_smoke():
    result = run_attention_case(
        case="decode_cached",
        batch=1,
        heads=2,
        sequence=8,
        head_dim=4,
        device="cpu",
        dtype=torch.float32,
        warmup=0,
        repeats=2,
    )
    assert result["case"] == "decode_cached"
    assert result["repeats"] == 2
    assert result["output_is_finite"] is True
    assert len(result["sample_ms"]) == 2

def test_unknown_case():
    with pytest.raises(ValueError):
        run_attention_case(
        case="unknown case",
        batch=1,
        heads=2,
        sequence=8,
        head_dim=4,
        device="cpu",
        dtype=torch.float32,
        warmup=0,
        repeats=2,
    )

def test_sequence_zero():
    with pytest.raises(ValueError):
        run_attention_case(
        case="decode_cached",
        batch=1,
        heads=2,
        sequence=0,
        head_dim=4,
        device="cpu",
        dtype=torch.float32,
        warmup=0,
        repeats=2,
    )

def test_repeats_zero():
    with pytest.raises(ValueError):
        run_attention_case(
        case="decode_cached",
        batch=1,
        heads=2,
        sequence=8,
        head_dim=4,
        device="cpu",
        dtype=torch.float32,
        warmup=0,
        repeats=0,
    )

def test_bool_size():
    with pytest.raises(TypeError):
        run_attention_case(
        case="decode_cached",
        batch=True,
        heads=2,
        sequence=8,
        head_dim=4,
        device="cpu",
        dtype=torch.float32,
        warmup=0,
        repeats=1,
    )

def test_int_dtype():
    with pytest.raises(TypeError):
        run_attention_case(
        case="decode_cached",
        batch=1,
        heads=2,
        sequence=8,
        head_dim=4,
        device="cpu",
        dtype= int,
        warmup=0,
        repeats=1,
    )

def test_prefill_output_shape():
    batch = 1
    heads = 2
    sequence = 8

    result = run_attention_case(
        case="prefill",
        batch=1,
        heads=2,
        sequence=8,
        head_dim=4,
        device="cpu",
        dtype=torch.float32,
        warmup=0,
        repeats=2,
    )
    output_shape = result["output_shape"]
    batch_size = output_shape[0]  # 2
    seq_len = output_shape[1]     # 4
    hidden_dim = output_shape[2]  # 8

    assert batch_size == result["batch"] * result["heads"]
    assert seq_len == result["sequence"]
    assert hidden_dim == result["head_dim"]

def test_cached_shape():
    result = run_attention_case(
        case="decode_cached",
        batch=1,
        heads=2,
        sequence=8,
        head_dim=4,
        device="cpu",
        dtype=torch.float32,
        warmup=0,
        repeats=2,
    )
    output_shape = result["output_shape"]
    batch_size = output_shape[0]  # 2
    seq_len = output_shape[1]     # 4
    hidden_dim = output_shape[2]  # 8
    
    assert batch_size == result["batch"] * result["heads"]
    assert seq_len == 1
    assert hidden_dim == result["head_dim"]
    