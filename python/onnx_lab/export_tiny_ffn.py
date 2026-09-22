from __future__ import annotations 

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import onnx
import torch
from onnx import TensorProto, numpy_helper, shape_inference

from python.transformer_components import TinyFFNBlock

@dataclass(frozen=True)
class ExportArtifacts:
    model_path: Path
    state_dict_path: Path
    inferred_model_path: Path
    graph_summary_path: Path
    manifest_path: Path

class ExportableTinyFFN(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
    
    def forward(self, x):
        return self.block(x)
    
def sha256_file(path: Path) ->str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def git_metadata() -> dict:
    def run(*arg: str) -> str:
        result = subprocess.run(
            ["git", *arg],
            check = True,
            text = True,
            capture_output= True,
        )
        return result.stdout.strip()
    
    try:
        status = run("status", "--porcelain=v1")
        return {
            "revision": run("rev-parse", "--short", "HEAD"),
            "dirty": bool(status),
            "status_sha256": hashlib.sha256(
                status.encode("utf-8")
            ).hexdigest()
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "revision": None,
            "dirty": None,
            "status_sha256": None,
        }

def value_info_to_dict(value_info) -> dict:
    tensor_type = value_info.type.tensor_type

    shape: list[int | str] =[]

    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.dim_param:
            shape.append(dim.dim_param)
        else:
            shape.append("?")
    
    return {
        "name" : value_info.name,
        "dtype" : TensorProto.DataType.Name(tensor_type.elem_type),
        "shape" : shape,
    }

def build_graph_summary(model: onnx.ModelProto) -> dict:
    operator_counts = Counter(
        node.op_type
        for node in model.graph.node
    )

    initializer_bytes = sum(
        numpy_helper.to_array(initializer).nbytes
        for initializer in model.graph.initializer
    )

    return {
        "graph_name": model.graph.name,
        "opsets": {
            item.domain or  "ai.onnx": item.version
            for item in model.opset_import
        },
        "node_count": len(model.graph.node),
        "operator_counts": dict(sorted(operator_counts.items())),
        "inputs":[
            value_info_to_dict(value)
            for value in model.graph.input
        ],
        "outputs":[
            value_info_to_dict(value)
            for value in model.graph.output
        ],
        "initializer_count": len(model.graph.initializer),
        "initializer_bytes": initializer_bytes
    }

def export_tiny_ffn(
    output_dir: Path,
    *,
    d_model: int = 8,
    d_ff: int = 16,
    opset: int = 18,
    seed: int = 0,
) -> ExportArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "tiny_ffn.onnx"
    state_dict_path = output_dir / "tiny_ffn_state_dict.pt"
    inferred_model_path = output_dir / "tiny_ffn_inferred.onnx"
    graph_summary_path = output_dir / "tiny_ffn_graph_summary.json"
    manifest_path = output_dir / "tiny_ffn_manifest.json"

    torch.manual_seed(seed)

    block = TinyFFNBlock(
        d_model=d_model,
        d_ff=d_ff,
    ).cpu().eval()

    export_model = ExportableTinyFFN(block).eval()

    example = torch.randn(
        2,
        3,
        d_model,
        dtype=torch.float32,
    )

    batch_dim = torch.export.Dim("batch", min=1)
    sequence_dim = torch.export.Dim("sequence", min=1)

    onnx_program = torch.onnx.export(
        export_model,
        (example,),
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={
            # 这里用 Python forward 参数名 x，不是 ONNX 输入名 input。
            "x": {
                0: batch_dim,
                1: sequence_dim,
            }
        },
        opset_version=opset,
        dynamo=True,
        external_data=False,
        optimize=True,
    )

    onnx_program.save(str(model_path))
    torch.save(block.state_dict(), state_dict_path)

    onnx_model = onnx.load(str(model_path))

    # 结构、类型和节点合法性检查。
    onnx.checker.check_model(
        onnx_model,
        full_check=True,
    )

    inferred_model = shape_inference.infer_shapes(
        onnx_model,
        check_type=True,
        strict_mode=False,
    )

    onnx.save(inferred_model, str(inferred_model_path))

    graph_summary = build_graph_summary(inferred_model)
    graph_summary["model_sha256"] = sha256_file(model_path)

    graph_summary_path.write_text(
        json.dumps(
            graph_summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_metadata(),
        "versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
        },
        "model": {
            "name": "TinyFFNBlock",
            "d_model": d_model,
            "d_ff": d_ff,
            "dtype": "float32",
            "seed": seed,
            "opset": opset,
            "example_shape": list(example.shape),
            "dynamic_dimensions": ["batch", "sequence"],
            "static_dimensions": {
                "hidden": d_model,
            },
        },
        "artifacts": {
            "onnx": {
                "path": str(model_path),
                "size": model_path.stat().st_size,
                "sha256": sha256_file(model_path),
            },
            "state_dict": {
                "path": str(state_dict_path),
                "size": state_dict_path.stat().st_size,
                "sha256": sha256_file(state_dict_path),
            },
            "graph_summary": str(graph_summary_path),
            "inferred_model": str(inferred_model_path),
        },
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return ExportArtifacts(
        model_path=model_path,
        state_dict_path=state_dict_path,
        inferred_model_path=inferred_model_path,
        graph_summary_path=graph_summary_path,
        manifest_path=manifest_path,
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/onnx/tiny_ffn"),
    )
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=16)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    artifacts = export_tiny_ffn(
        output_dir=args.output_dir,
        d_model=args.d_model,
        d_ff=args.d_ff,
        opset=args.opset,
        seed=args.seed,
    )

    print(f"ONNX:    {artifacts.model_path}")
    print(f"state:   {artifacts.state_dict_path}")
    print(f"summary: {artifacts.graph_summary_path}")
    print(f"manifest:{artifacts.manifest_path}")


if __name__ == "__main__":
    main()