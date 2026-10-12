import numpy as np
import onnxruntime as ort
import pytest

from python.onnx_lab.run_ort import (
    compare_cpu_cuda,
    make_cpu_session,
    make_cuda_session,
)


CUDA_AVAILABLE = (
    "CUDAExecutionProvider"
    in ort.get_available_providers()
)


@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDAExecutionProvider unavailable",
)
def test_cpu_cuda_parity_smoke(artifacts):
    cpu_session = make_cpu_session(
        artifacts.model_path,
    )
    cuda_session = make_cuda_session(
        artifacts.model_path,
    )

    rng = np.random.default_rng(0)
    x = rng.standard_normal(
        (1, 2, 8),
    ).astype(np.float32)

    result = compare_cpu_cuda(
        cpu_session,
        cuda_session,
        x,
    )

    assert result["passed"] is True
    assert result["shape"] == [1, 2, 8]
    assert result["max_abs_error"] >= 0