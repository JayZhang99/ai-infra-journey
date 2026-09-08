from __future__ import annotations

import pytest 
import torch
import math
import json


from python.profile_h2d import(
    profile_h2d,
    validate_config,
)

def valid_profile_config():
    return {
        "capture_mode": "light",
        "active_steps": 1,
        "rows": 8,
        "inner": 8,
        "out_features": 8,
        "chunks": 2,
        "device": "cuda:0",
    }

@pytest.mark.parametrize(
    "filed, value, error":
    [
        ("active_steps", 0, ValueError),
        ("chunks", 0, ValueError),
        ("rows", True, TypeError),
    ]
)

def test_profile_invalid(filed, value, error):
    config = valid_profile_config()
    config[filed] = value
    with pytest.raises(error):
        validate_config(config)

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason = "CUDA is unavailable",
)
def test_profile_h2d_smoke(tmp_path):
    trace = tmp_path/ "trace.json"
    top = tmp_path/ "top.ext"

    profile_h2d(
        rows=8, inner=8,
        out_features=8, chunks=2,
        active_steps=1,
        trace_path=trace,
        top_path= top,
        capture_mode="light",
    )

    assert trace.exists()
    assert top.exists()
    assert trace.stat().st_size > 0
    assert top.stat().st_size > 0

    text = top.read_text(encoding="utf-8")
    assert "aten::mm" in text
    assert "aten::copy" in text


    payload = json.loads(
        trace.read_text(encoding="utf-8")
    )
    assert "traceEvents" in payload
    assert isinstance(payload["traceEvents"], list)

    names = {
        str(event.get("name", ""))
        for event in payload["traceEvents"]
    }

    assert "h2d_compute_pipeline" in names
    assert any(
        "Memcpy HtoD" in name
        for name in names
    )
    assert any(
        name == "aten::mm" or "gemm" in name.lower()
        for name in names
    )

    def index_cases(payload):
        return {
            item["case"]: item
            for item in payload["results"]
        }
    cases = index_cases(payload)
    row = {
        "chunks": payload["chunks"],
        "rows": payload["rows"],
        "bytes": payload["bytes"],
        "seq_ms": cases["sequential"]["median_ms"],
        "pipe_ms": cases["pipeline"]["median_ms"],
        "speedup": payload["pipeline_speedup"]
    }


