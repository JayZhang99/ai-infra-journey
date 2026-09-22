from pathlib import Path

import onnxruntime as ort
import torch

from python.transformer_components import TinyFFNBlock


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

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


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