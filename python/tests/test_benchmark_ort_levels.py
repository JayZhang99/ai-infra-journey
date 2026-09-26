from __future__ import annotations

import pytest 
import torch
import math
import json

import onnxruntime as ort

from python.onnx_lab.export_tiny_ffn import (
    export_tiny_ffn,
)

from python.onnx_lab.benchmark_ort_levels import (
    make_level_options,
    time_sample,
    LEVEL_ORDER,
    benchmark_ort_levels,
)

def test_make_options_disables_profile():
    options = make_level_options('all', threads=1)
    assert options.enable_profiling is False
    assert options.execution_mode == (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )
    assert options.intra_op_num_threads == 1

def test_time_sample_calls_exactly_n_times():
    calls = 0
    def run_once():
        nonlocal calls
        calls +=1
    value = time_sample(run_once, 4)
    assert calls ==4
    assert value >= 0.0

def rotated_order(sampe_index):
    offset = sampe_index % len(LEVEL_ORDER)
    return list(
        LEVEL_ORDER[offset:] + LEVEL_ORDER[:offset]
    )

def test_rotated_order():
    assert rotated_order(0)[0] == 'disable'
    assert rotated_order(1)[0] == 'basic'
    assert rotated_order(4)[0] == 'disable'


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp(
        "benchmark_ort_levels"
    )

    artifacts = export_tiny_ffn(
        output_dir=output_dir,
        d_model=8,
        d_ff=16,
        opset=18,
        seed=0,
    )

    return artifacts.model_path

def test_benchmark_cpu_smoke(model_path):
    result = benchmark_ort_levels(
        model_path=model_path,
        shape = (1,2,8),
        threads=1, warmup=0, repeats=1,
        runs_per_sample=1, seed=0,
    )
    assert result['profile_enabled'] is False
    assert set(result['levels']) == set(LEVEL_ORDER)
    assert all(
        len(v['samples_ms_per_run']) == 1
        for v in result['levels'].values()
    )