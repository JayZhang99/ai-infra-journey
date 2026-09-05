from __future__ import annotations

import pytest 
import torch
import math
import json


from python.profile_h2d import(
    profile_h2d,
)

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