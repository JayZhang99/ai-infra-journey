from __future__ import annotations

import pytest 
import torch
import math
import json


from python.benchmark_h2d import(
    run_h2d_benchmark,
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