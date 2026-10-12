from pathlib import Path

import onnxruntime as ort
import torch

from typing import Sequence
import numpy as np
from python.transformer_components import TinyFFNBlock


def make_session_options(
    *,
    threads: int = 1,
) -> ort.SessionOptions:
    if threads <= 0:
        raise ValueError("threads must be positive")

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.enable_profiling = False
    return options


def load_reference_model(
    state_dict_path: Path,
    *,
    d_model: int = 8,
    d_ff: int = 16,
) -> TinyFFNBlock:
    model = TinyFFNBlock(
        d_model=d_model,
        d_ff=d_ff,
    ).cpu().eval()

    state_dict = torch.load(
        state_dict_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    return model


def make_cpu_session(
    model_path: Path,
    *,
    threads: int = 1,
) -> ort.InferenceSession:
    available = ort.get_available_providers()

    if "CPUExecutionProvider" not in available:
        raise RuntimeError(
            f"CPUExecutionProvider unavailable: {available}"
        )

    return ort.InferenceSession(
        str(model_path),
        sess_options=make_session_options(
            threads=threads,
        ),
        providers=["CPUExecutionProvider"],
    )

def make_cuda_session(
    model_path: Path,
    *,
    device_id: int = 0,
    threads: int = 1,
) -> ort.InferenceSession:
    available = ort.get_available_providers()

    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDAExecutionProvider unavailable: "
            f"{available}"
        )

    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": device_id,
                "do_copy_in_default_stream": True,
            },
        ),
        "CPUExecutionProvider",
    ]

    session = ort.InferenceSession(
        str(model_path),
        sess_options=make_session_options(
            threads=threads,
        ),
        providers=providers,
    )

    selected = session.get_providers()

    if (
        not selected
        or selected[0] != "CUDAExecutionProvider"
    ):
        raise RuntimeError(
            "CUDAExecutionProvider was not selected first: "
            f"{selected}"
        )

    return session

def run_ort_numpy(
    session: ort.InferenceSession,
    x: np.ndarray,
) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a numpy array")

    if x.dtype != np.float32:
        raise ValueError("x must have dtype float32")

    if x.ndim != 3:
        raise ValueError("x must be rank-3")

    x = np.ascontiguousarray(x)

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    output = session.run(
        [output_meta.name],
        {
            input_meta.name: x,
        },
    )[0]

    return output

def compare_cpu_cuda(
    cpu_session: ort.InferenceSession,
    cuda_session: ort.InferenceSession,
    x: np.ndarray,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> dict:
    cpu_output = run_ort_numpy(
        cpu_session,
        x,
    )
    cuda_output = run_ort_numpy(
        cuda_session,
        x,
    )

    np.testing.assert_allclose(
        cuda_output,
        cpu_output,
        rtol=rtol,
        atol=atol,
    )

    abs_diff = np.abs(
        cuda_output - cpu_output
    )

    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "max_abs_error": float(abs_diff.max()),
        "mean_abs_error": float(abs_diff.mean()),
        "rtol": rtol,
        "atol": atol,
        "passed": True,
    }


def run_ort(
    session: ort.InferenceSession,
    x: torch.Tensor,
) -> torch.Tensor:
    if x.device.type != "cpu":
        raise ValueError("CPU ORT path requires a CPU tensor")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    ort_output = session.run(
        [output_name],
        {
            input_name: x.detach().contiguous().numpy(),
        },
    )[0]

    return torch.from_numpy(ort_output)


def assert_ort_parity(
    reference_model: TinyFFNBlock,
    session: ort.InferenceSession,
    x: torch.Tensor,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> None:
    with torch.inference_mode():
        expected = reference_model(x)

    actual = run_ort(session, x)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
    )