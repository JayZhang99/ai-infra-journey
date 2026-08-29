from __future__ import annotations

import pytest 
import torch
import json

from python.profile_matmul import(
    profile_matmul,
)

def test_cpu_profile_smoke(tmp_path):
    trace = tmp_path/"trace.json"
    top5 = tmp_path/"top5.txt"

    profile_matmul(
        size=16, device="cpu",
        dtype=torch.float32,
        warmup=1, steps=2,
        trace_path=trace,
        top5_path=top5,
        capture_mode="full"
    )

    trace_payload = json.loads(
    trace.read_text(encoding="utf-8")
)
    event_names = {
    event.get("name")
    for event in trace_payload["traceEvents"]}

    assert "matmul_compute_only" in event_names
    assert "aten::mm" in event_names

def test_size_eqaul_zero(tmp_path):
    trace = tmp_path/"trace.json"
    top5 = tmp_path/"top5.txt"

    with pytest.raises(
        ValueError,
        match="size must be greater than 0",
    ):
        profile_matmul(
        size=0, device="cpu",
        dtype=torch.float32,
        warmup=1, steps=2,
        trace_path=trace,
        top5_path=top5
    )

def test_warmup_less_zero(tmp_path):
    trace = tmp_path/"trace.json"
    top5 = tmp_path/"top5.txt"

    with pytest.raises(
        ValueError,
        match="warmup must be at least 0",
    ):
        profile_matmul(
        size=1, device="cpu",
        dtype=torch.float32,
        warmup=-1, steps=2,
        trace_path=trace,
        top5_path=top5
    )

def test_steps_eqaul_zero(tmp_path):
    trace = tmp_path/"trace.json"
    top5 = tmp_path/"top5.txt"

    with pytest.raises(
        ValueError,
        match="steps must be greater than 0",
    ):
        profile_matmul(
        size=1, device="cpu",
        dtype=torch.float32,
        warmup=0, steps=0,
        trace_path=trace,
        top5_path=top5
    )

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)

def test_cuda_smoke(tmp_path):
    trace = tmp_path/"trace.json"
    top5 = tmp_path/"top5.txt"

    profile_matmul(
        size=16, device="cuda:0",
        dtype=torch.float32,
        warmup=1, steps=2,
        trace_path=trace,
        top5_path=top5
    )

    trace_payload = json.loads(
    trace.read_text(encoding="utf-8")
)
    event_names = {
    event.get("name")
    for event in trace_payload["traceEvents"]}

    assert "matmul_compute_only" in event_names
    assert "aten::mm" in event_names
    
