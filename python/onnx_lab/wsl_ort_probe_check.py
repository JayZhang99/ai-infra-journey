import json
from pathlib import Path

p = Path('benchmarks/onnx/probes/2026-09-26_wsl_ort_cuda_probe.json')
d = json.loads(p.read_text())
assert d['onnxruntime']['cuda_ep_available'] is True
assert d['model']['exists'] is True
assert d['session_smoke']['status'] == 'passed'
assert d['session_smoke']['selected_providers'][0] == '"CUDAExecutionProvider"'
print('probe gate passed')