import json
import math

import pytest
import torch

from python.benchmark_matmul import (
    run_matmul_benchmark,
    validate_config,
)
from python.benchmark_utils import (
    save_result,
    summarize,
    time_cpu,
)

def test_validate_config():
    validate_config(
        size=8,
        warmup=0,
        repeats=1,
    )

    with pytest.raises(ValueError):
        validate_config(
            size=0,
            warmup=0,
            repeats=1,
        )

    with pytest.raises(ValueError):
        validate_config(
            size=8,
            warmup=-1,
            repeats=1,
        )

    with pytest.raises(ValueError):
        validate_config(
            size=8,
            warmup=0,
            repeats=0,
        )

def test_summarize():
    result = summarize([1,2,3,4])
    assert result["count"] == 4
    assert result["median_ms"] == 2.5
    assert result["min_ms"] == 1.0
    assert result["max_ms"] == 4.0

def test_time_cpu():
    samples = time_cpu(
        lambda: sum(range(10)),
        warmup=1,
        repeats=3,
    )

    assert len(samples) == 3

    assert all(
        math.isfinite(sample)
        for sample in samples
    )

    assert all(
        sample >= 0
        for sample in samples
    )

def test_run_matmul_benchmark():
    result = run_matmul_benchmark(
        size=8,
        device="cpu",
        warmup=1,
        repeats=3,
    )

    assert result["operation"] == "matmul"
    assert result["boundary"] == (
        "matmul_compute_only"
    )

    assert result["shape_a"] == [8, 8]
    assert result["shape_b"] == [8, 8]
    assert result["output_shape"] == [8, 8]

    assert result["dtype"] == "float32"
    assert result["device"] == "cpu"
    assert result["timer"] == "perf_counter_ns"

    assert result["count"] == 3
    assert len(result["samples_ms"]) == 3

    assert result["output_is_finite"] is True
    assert result["output_requires_grad"] is False

def test_save_result(tmp_path):
    result = run_matmul_benchmark(
        size=4,
        warmup=0,
        repeats=2,
    )

    output_path = save_result(
        result,
        tmp_path / "result.json",
    )

    restored = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert restored["shape_a"] == [4, 4]
    assert restored["repeats"] == 2
    assert len(restored["samples_ms"]) == 2

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_run_matmul_benchmark_cuda():
    result = run_matmul_benchmark(
        size=8,
        device="cuda:0",
        warmup=1,
        repeats=3,
    )

    assert result["device"] == "cuda:0"
    assert result["timer"] == "cuda_event"
    assert result["count"] == 3
    assert result["output_is_finite"] is True
