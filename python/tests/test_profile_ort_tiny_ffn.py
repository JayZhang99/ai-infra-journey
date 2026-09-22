import json

import pytest
import argparse

pytest.importorskip("onnx")
pytest.importorskip("onnxscript")
pytest.importorskip("onnxruntime")

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
            },
        },
        {
            "cat": "Node",
            "ph": "X",
            "dur": 4,
            "args": {
                "op_name": "Add",
                "provider": "CPUExecutionProvider",
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