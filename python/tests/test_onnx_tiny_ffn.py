import json

import pytest
import torch

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxscript")
pytest.importorskip("onnxruntime")

from onnxruntime.capi.onnxruntime_pybind11_state import (
    InvalidArgument as OrtInvalidArgument,
)

from python.onnx_lab.export_tiny_ffn import export_tiny_ffn
from python.onnx_lab.run_ort import (
    assert_ort_parity,
    load_reference_model,
    make_cpu_session,
    run_ort,
)


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("tiny_ffn_onnx")

    return export_tiny_ffn(
        output_dir=output_dir,
        d_model=8,
        d_ff=16,
        opset=18,
        seed=0,
    )


@pytest.fixture(scope="session")
def reference_model(artifacts):
    return load_reference_model(
        artifacts.state_dict_path,
        d_model=8,
        d_ff=16,
    )


@pytest.fixture(scope="session")
def ort_session(artifacts):
    return make_cpu_session(
        artifacts.model_path,
        threads=1,
    )


def test_onnx_checker_accepts_model(artifacts):
    model = onnx.load(str(artifacts.model_path))
    onnx.checker.check_model(model, full_check=True)


def test_graph_contract(artifacts):
    summary = json.loads(
        artifacts.graph_summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert summary["node_count"] > 0
    assert summary["inputs"][0]["name"] == "input"
    assert summary["outputs"][0]["name"] == "output"

    input_shape = summary["inputs"][0]["shape"]
    output_shape = summary["outputs"][0]["shape"]

    assert input_shape[-1] == 8
    assert output_shape[-1] == 8

    assert isinstance(input_shape[0], str)
    assert isinstance(input_shape[1], str)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 8),
        (1, 7, 8),
        (2, 3, 8),
        (4, 11, 8),
    ],
)
def test_ort_matches_pytorch(
    shape,
    reference_model,
    ort_session,
):
    torch.manual_seed(123)
    x = torch.randn(shape, dtype=torch.float32)

    assert_ort_parity(
        reference_model,
        ort_session,
        x,
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "bad_shape",
    [
        (2, 3, 7),  # hidden dimension 错误
        (3, 8),     # rank 错误
    ],
)
def test_ort_rejects_invalid_shape(
    bad_shape,
    ort_session,
):
    x = torch.randn(bad_shape, dtype=torch.float32)

    with pytest.raises(OrtInvalidArgument):
        run_ort(ort_session, x)


def test_ort_rejects_float64(ort_session):
    x = torch.randn(
        2,
        3,
        8,
        dtype=torch.float64,
    )

    with pytest.raises(OrtInvalidArgument):
        run_ort(ort_session, x)