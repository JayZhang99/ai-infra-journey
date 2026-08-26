from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import json

from python.compare_cuda_cpu import (
    compare_results,
    build_index
)

def make_result(
    *,
    size: int,
    device: str,
    median_ms: float,
    p95_ms: float = 1.5,
    dtype: str = "float32",
    boundary: str = "matmul_compute_only",
    estimated_tflops: float = 1.0,
) -> dict[str, Any]:
    """构造一条最小 benchmark 结果。"""

    return {
        "shape_a": [size, size],
        "shape_b": [size, size],
        "device": device,
        "dtype": dtype,
        "boundary": boundary,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "estimated_tflops": estimated_tflops,
    }

def make_payload(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """构造 compare_results 所需的 payload。"""

    return {
        "schema_version": 1,
        "benchmark": "square_matmul",
        "results": results,
    }



def test_compare_results():
    cpu_payload = make_payload(
        [
            make_result(
                size=128,
                device="cpu",
                median_ms=4.0,
                p95_ms=5.0,
                estimated_tflops=0.1,
            )
        ]
    )

    cuda_payload = make_payload(
        [
            make_result(
                size=128,
                device="cuda:0",
                median_ms=1.0,
                p95_ms=1.2,
                estimated_tflops=0.4,
            )
        ]
    )
    comparisons = compare_results(
        cpu_payload,
        cuda_payload,
    )

    assert len(comparisons) == 1
    result = comparisons[0]

    assert result["size"] == 128
    assert result["dtype"] == "float32"
    assert result["boundary"] == "matmul_compute_only"

    assert result["cpu_median_ms"] == 4.0
    assert result["cuda_median_ms"] == 1.0

    assert result["cpu_p95_ms"] == 5.0
    assert result["cuda_p95_ms"] == 1.2

    assert result["cpu_tflops"] == 0.1
    assert result["cuda_tflops"] == 0.4

    # speedup = CPU median / CUDA median
    assert result["speedup"] == pytest.approx(4.0)

def test_compare_results_rejects_mismatched_grid():
    cpu_payload = make_payload(
        [
            make_result(
                size=128,
                device="cpu",
                median_ms=4.0,
            )
        ]
    )

    cuda_payload = make_payload(
        [
            make_result(
                size=512,
                device="cuda:0",
                median_ms=1.0,
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="CPU and CUDA result grids do not match",
    ):
        compare_results(
            cpu_payload,
            cuda_payload,
        )

def test_build_index_rejects_duplicate_key():
    first = make_result(
        size=128,
        device="cpu",
        median_ms=4.0,
    )

    second = deepcopy(first)
    second["median_ms"] = 5.0

    payload = make_payload([first, second])

    with pytest.raises(
        ValueError,
        match="Duplicate result",
    ):
        build_index(
            payload,
            expected_device_type="cpu",
        )
    
@pytest.mark.parametrize(
    ("expected_device_type", "actual_device"),
    [
        ("cpu", "cuda:0"),
        ("cuda", "cpu"),
    ],
)

def test_build_index_rejects_wrong_device(
    expected_device_type: str,
    actual_device: str,
):
    payload = make_payload(
        [
            make_result(
                size=128,
                device=actual_device,
                median_ms=1.0,
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match=f"Expected {expected_device_type} result",
    ):
        build_index(
            payload,
            expected_device_type,
        )
    
def test_compare_results_rejects_zero_cuda_median():
    cpu_payload = make_payload(
        [
            make_result(
                size=128,
                device="cpu",
                median_ms=4.0,
            )
        ]
    )

    cuda_payload = make_payload(
        [
            make_result(
                size=128,
                device="cuda:0",
                median_ms=0.0,
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="CUDA median_ms must be positive",
    ):
        compare_results(
            cpu_payload,
            cuda_payload,
        )

