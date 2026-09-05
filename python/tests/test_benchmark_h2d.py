from __future__ import annotations

import pytest 
import torch
import math
import json


from python.benchmark_h2d import(
    run_h2d_benchmark,
    validate_config,
)

@pytest.mark.parametrize(
    "field,value,error",
    [
        ("rows", True, TypeError),
        ("inner", 0, ValueError),
        ("out_features", -1, ValueError),
        ("chunks", 1, ValueError),
        ("warmup", -1, ValueError),
        ("repeats", 0, ValueError),
    ],
)

def test_h2d_invalid_size(field, value, error):

    config = {
        "rows": 8,
        "inner": 8,
        "out_features": 8,
        "chunks": 2,
        "warmup": 0,
        "repeats": 1,
        "dtype": torch.float32,
    }

    # 每轮只修改一个字段
    config[field] = value

    with pytest.raises(error):
        validate_config(**config)

def test_h2d_dtype_must_be_torch_dtype():
    with pytest.raises(TypeError):
        validate_config(
            rows=8, inner=8,
            out_features=8, chunks=2,
            warmup=0, repeats=1,
            dtype="float32"
        )

def test_h2d_dtype_must_be_floating():
    with pytest.raises(TypeError):
        validate_config(
            rows=8, inner=8,
            out_features=8, chunks=2,
            warmup=0, repeats=1,
            dtype=torch.int64,
        )

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)

def test_h2d_pipeline_smoke():
    result = run_h2d_benchmark(
        rows=8,
        inner=8,
        out_features=8,
        chunks=2,
        warmup=0,
        repeats=1,
    )

    assert len(result["results"]) == 4
    assert result["chunk_bytes"] == 256

    cases = {
        item["case"]
        for item in result["results"]
    }

    assert cases == {
        "copy_only",
        "compute_only",
        "sequential",
        "pipeline",
    }

    by_case = {
        item["case"] : item
        for item in result["results"]
    }

    for item in by_case.values():
        assert item["count"] == 1
        assert len(item["samples_ms"]) == 1
        assert math.isfinite(item["median_ms"])

    assert math.isfinite(result["pipeline_speedup"])