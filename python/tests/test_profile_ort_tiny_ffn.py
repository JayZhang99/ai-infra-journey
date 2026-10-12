import json

import pytest
import argparse

pytest.importorskip("onnx")
pytest.importorskip("onnxscript")
ort = pytest.importorskip("onnxruntime")

from python.onnx_lab.export_tiny_ffn import export_tiny_ffn
from python.onnx_lab.profile_ort_tiny_ffn import (
    parse_shape,
    profile_model,
    summarize_events,
)

@pytest.fixture(scope="session")
def tiny_ffn_artifacts(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("profile_tiny_ffn")

    return export_tiny_ffn(
        output_dir=output_dir,
        d_model=8,
        d_ff=16,
        opset=18,
        seed=0,
    )


def test_parse_shape():
    assert parse_shape("1x128x8") == (1, 128, 8)

    with pytest.raises(argparse.ArgumentTypeError):
        parse_shape("1xabcx8")

    with pytest.raises(argparse.ArgumentTypeError):
        parse_shape("1x0x8")


def test_summarize_events_groups_provider_and_op():
    events = [
        {
            "cat": "Session",
            "ph": "X",
            "name": "model_run",
            "dur": 20,
        },
        {
            "cat": "Node",
            "ph": "X",
            "dur": 10,
            "args": {
                "op_name": "MatMul",
                "provider": "CPUExecutionProvider",
                "name": "matmul_kernel_time",
                "node_index": "0",  
            },
        },
        {
            "cat": "Node",
            "ph": "X",
            "dur": 4,
            "args": {
                "op_name": "Add",
                "provider": "CPUExecutionProvider",
                "name": "add_kernel_time",
                "node_index": "1",
            },
        },
    ]

    result = summarize_events(events)

    assert result["node_event_count"] == 2
    assert result["session_event_count"] == 1
    assert result["op_counts"] == {
        "MatMul": 1,
        "Add": 1,
    }
    assert result["op_total_dur_us"]["MatMul"] == 10
    assert result["provider_counts"] == {
        "CPUExecutionProvider": 2
    }


def test_profile_cpu_smoke(
    tmp_path,
    tiny_ffn_artifacts,
):
    trace = tmp_path / "trace.json"
    summary = tmp_path / "summary.json"
    optimized = tmp_path / "optimized.onnx"

    result = profile_model(
        model_path=tiny_ffn_artifacts.model_path,
        shape=(1, 2, 8),
        steps=1,
        threads=1,
        level="all",
        trace_path=trace,
        summary_path=summary,
        optimized_model_path=optimized,
        seed=0,
    )

    assert trace.stat().st_size > 0
    assert summary.stat().st_size > 0
    assert optimized.stat().st_size > 0

    assert result["environment"]["selected_providers"] == [
        "CPUExecutionProvider"
    ]
    assert result["node_event_count"] > 0
    assert set(result["provider_counts"]) == {
        "CPUExecutionProvider"
    }

    saved = json.loads(summary.read_text(encoding="utf-8"))
    assert saved["configuration"]["steps"] == 1

def make_node_event(
    node_index: int,
    op_name: str,
    provider: str,
    duration_us: float = 10.0,
):
    return {
        "cat": "Node",
        "ph": "X",
        "dur": duration_us,
        "name": f"node_{node_index}_{op_name}_kernel_time",
        "args": {
            "node_index": str(node_index),
            "op_name": op_name,
            "provider": provider,
        },
    }


def test_summarize_unique_node_assignments():
    events = [
        make_node_event(
            0, "MatMul", "CUDAExecutionProvider"
        ),
        make_node_event(
            1, "Add", "CPUExecutionProvider"
        ),
        # 第二个 step。
        make_node_event(
            0, "MatMul", "CUDAExecutionProvider"
        ),
        make_node_event(
            1, "Add", "CPUExecutionProvider"
        ),
    ]

    result = summarize_events(events)

    assert result["provider_counts"] == {
        "CPUExecutionProvider": 2,
        "CUDAExecutionProvider": 2,
    }

    assert result["unique_provider_counts"] == {
        "CPUExecutionProvider": 1,
        "CUDAExecutionProvider": 1,
    }

    assert result["assignment_summary"] == {
        "unique_node_count": 2,
        "cuda_unique_node_count": 1,
        "cpu_unique_node_count": 1,
        "all_unique_nodes_on_cuda": False,
    }

    assert len(result["cpu_fallback_nodes"]) == 1
    assert (
        result["cpu_fallback_nodes"][0]["op_name"]
        == "Add"
    )
    assert (
        result["cpu_fallback_nodes"][0]["event_count"]
        == 2
    )

CUDA_EP_AVAILABLE = (
    "CUDAExecutionProvider"
    in ort.get_available_providers()
)


@pytest.mark.skipif(
    not CUDA_EP_AVAILABLE,
    reason="CUDAExecutionProvider unavailable",
)
def test_cuda_profile_smoke(
    tiny_ffn_artifacts,
    tmp_path,
):
    result = profile_model(
        model_path=tiny_ffn_artifacts.model_path,
        shape=(1, 2, 8),
        steps=1,
        threads=1,
        trace_path=tmp_path / "cuda_trace.json",
        summary_path=tmp_path / "cuda_summary.json",
        optimized_model_path=None,
        seed=0,
        device="cuda",
        device_id=0,
        level="all",
    )

    assert (
        result["environment"]["selected_providers"][0]
        == "CUDAExecutionProvider"
    )

    assert (
        result["assignment_summary"][
            "cuda_unique_node_count"
        ]
        > 0
    )

    assert (
        sum(result["unique_provider_counts"].values())
        == result["assignment_summary"][
            "unique_node_count"
        ]
    )