from __future__ import annotations 

import json
from pathlib import Path
from typing import Any
import pytest

from python.onnx_lab.analyze_ort_profiles import (
    build_comparison,
    detect_quickgelu_boundary,
    structure_row,
    validate_summary,
    LEVEL_ORDER,
)

def make_summary(level='disable'):
    quick = level in {'extended', 'all'}
    ops = {'MatMul': 15, 'Mul': 15 if quick else 20}
    ops['QuickGelu' if quick else 'Sigmoid'] = 5
    return {
        'schema_version': 1,
        'configuration': {
            'optimization_level': level,
            'steps': 5, 'shape': [1, 128, 8],
            'dtype': 'float32', 'threads': 1,
        },
        'environment': {
            'selected_providers':['CPUExecutionProvider']
        },
        'op_counts': ops,
        'node_event_count': sum(ops.values()),
        'provider_counts': {
            'CPUExecutionProvider': sum(ops.values())
        },
        'session_event_counts': {'model_run': 5},
        'session': {'creation_ms': 1.0},
        'artifacts': {'model': {'sha256': 'same'}},
    }

def test_structure_row_uses_per_run_counts():
    row = structure_row(make_summary('basic'))
    assert row['level'] == 'basic'
    assert row['ops_per_run']['Sigmoid'] == 1
    assert row['ops_per_run']['Mul'] == 4
    assert row['selected_providers'] == [
        'CPUExecutionProvider'
    ]

def test_rejects_op_count_mismatch():
    data = make_summary()
    data['op_counts']['MatMul'] += 1
    with pytest.raises(ValueError, match ='op count'):
        validate_summary(data)

def test_rejects_provider_mismatch():
    data = make_summary()
    data['provider_counts'] = {'CPUExecutionProvider': 400}
    with pytest.raises(ValueError, match='provider count mismatch'):
        validate_summary(data)
    
def test_rejevts_duplicate_level():
    items = [
        make_summary('disable'),
        make_summary('basic'),
        make_summary('extended'),
        make_summary('basic'),
    ]
    with pytest.raises(ValueError, match='duplicate'):
        build_comparison(items)

def test_rejects_incompatible_shape():
    items = [make_summary(level) for level in LEVEL_ORDER]
    items[2]['configuration']['shape'] = [1, 512, 8]
    with pytest.raises(ValueError, match='shape'):
        build_comparison(items)

def test_rejects_incompatible_SHA():
    items = [make_summary(level) for level in LEVEL_ORDER]
    items[2]['artifacts']['model']['sha256'] = 'notsame'
    with pytest.raises(ValueError, match='sha256'):
        build_comparison(items)

def test_rejects_incompatible_dtype():
    items = [make_summary(level) for level in LEVEL_ORDER]
    items[2]['configuration']['dtype'] = 'float64'
    with pytest.raises(ValueError, match='dtype'):
        build_comparison(items)

def test_rejects_incompatible_steps():
    items = [make_summary(level) for level in LEVEL_ORDER]
    items[2]['configuration']['steps'] = 15
    with pytest.raises(ValueError, match='model_run count mismatch'):
        build_comparison(items)

def test_rejects_incompatible_threads():
    items = [make_summary(level) for level in LEVEL_ORDER]
    items[2]['configuration']['threads'] = 2
    with pytest.raises(ValueError, match='threads'):
        build_comparison(items)

def test_rejects_incompatible_providers():
    items = [make_summary(level) for level in LEVEL_ORDER]
    items[2]['environment']['selected_providers'] = ['CUDAExtensionProvider']
    with pytest.raises(ValueError, match='unexpected providers'):
        build_comparison(items)

def test_boundary_does_not_false_positive():
    levels = {
        'basic': {'ops_per_run': {'Sigmoid': 1}},
        'extended': {'ops_per_run': {'Sigmoid': 1}},
    }
    result = detect_quickgelu_boundary(levels)
    assert result['observed'] is False
    assert result['observed_from'] is None