from __future__ import annotations

import math
import platform
from datetime import datetime
from pathlib import Path
from typing import Any
import argparse
import torch

from .benchmark_utils import (
    DTYPE_MAP,
    save_result,
    summarize,
    time_cpu,
    time_cuda,
)

BOUNDARY = "qk_softmax_v_only"
BENCHMARK = "attention_core"
CASE_LIST = ["prefill", "decode_no_cache", "decode_cached"]
def estimate_kv_cache_bytes(
    layers, batch, kv_heads, sequence, head_dim, dtype
):
    values = [layers, batch, kv_heads, sequence, head_dim]
    if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
        raise TypeError("sizes must be interges")
    if any(v<=0 for v in values):
        raise ValueError("sizes must be positive")
    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be torch.dtype")
    
    element_size = torch.empty((), dtype=dtype).element_size()
    return 2 * layers * batch * kv_heads * sequence * head_dim * element_size

def attention_core_unchecked(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v

def run_attention_case(
    case: str,
    batch: int,
    heads: int,
    sequence : int,
    head_dim : int,
    kv_heads: int,
    device : str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    layers: int = 1,
    warmup: int = 10,
    repeats: int =100,
    seed: int = 11,
) -> dict[str, Any]:
    if heads % kv_heads != 0:
        raise ValueError(
            "heads must be divisible by kv_heads"
        )
    
    if kv_heads != heads:
        raise NotImplementedError(
            "GQA compute is not implemented"
        )
    kv_cache_bytes = estimate_kv_cache_bytes(layers,batch,heads,sequence,head_dim,dtype)
    device = torch.device(device)

    if device.type not in {"cpu", "cuda"}:
        raise ValueError(
            "Only cpu and cuda are supported ",
            f"got {device.type}"
        )
    
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        
        if device.index is None:
            device = torch.device(
                "cuda",
                torch.cuda.current_device(),
            )

    torch.manual_seed(seed)
    bh = batch * heads

    q = torch.randn(
        bh, sequence, head_dim,
        device=device, dtype=dtype,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    mask = torch.tril(torch.ones(
        sequence, sequence,
        dtype = torch.bool,
        device = device,
    ))

    if case == "prefill":
        fn = lambda: attention_core_unchecked(q, k, v, mask)
    elif case == "decode_no_cache":
        fn = lambda: attention_core_unchecked(q, k, v, mask)[:,-1:]
    elif case == "decode_cached":
        q_new = q[:, -1:, :]
        fn = lambda: attention_core_unchecked(q_new, k, v, None)
    else:
        raise ValueError("unknown case")

    output = fn()
    output_is_finite = bool(
        torch.isfinite(output).all().item()
    )
    
    full = attention_core_unchecked(q, k, v, mask)
    cached = attention_core_unchecked(q[:,-1:,:], k, v, None)

    torch.testing.assert_close(
        full[:,-1:,:],
        cached
    )

    with torch.inference_mode():
        output = fn()
        output_shape = list(output.shape)
        output_is_finite = bool(torch.isfinite(output).all().item())

        if device.type == "cuda":
            samples_ms = time_cuda(fn, warmup, repeats)
            timer = "cuda_event"
        else:
            samples_ms = time_cpu(fn, warmup, repeats)
            timer = "perf_counter_ns"

    summary = summarize(samples_ms)

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(
            device
        )
    else:
        device_name = (
            platform.processor()
            or platform.machine()
            or "CPU"
        )

    return {
        "schema_version": 1,
        "timestamp":(
            datetime.now()
            .astimezone()
            .isoformat(timespec="seconds")
        ),
        "benchmark": BENCHMARK,
        "boundary": BOUNDARY,
        "case": case,
        "batch": batch,
        "heads": heads,
        "sequence": sequence,
        "head_dim": head_dim,
        "dtype":str(dtype).removeprefix("torch."),
        "device": str(device),
        "device_name": device_name,
        "torch_version": str(torch.__version__),
        "cuda_vesion": torch.version.cuda,
        "torch_num_threads":(
            torch.get_num_threads()
        ),
        "seed": seed,
        "warmup": warmup,
        "repeats": repeats,
        "sample_ms": samples_ms,
        "median_ms": summary["median_ms"],
        "p95_ms": summary["p95_ms"],
        "output_is_finite": output_is_finite,
        "output_shape":output_shape
    }

def build_parser():
    parser = argparse.ArgumentParser(
        description = "Run a PyTorch attention benchmark."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float16" )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--sequences", nargs="+", type=int, required=True)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=32)
    return parser

def main()-> None:
    args = build_parser().parse_args()
    dtype = DTYPE_MAP[args.dtype]
    results = []
    for sequence in args.sequences:
        for case_mode in CASE_LIST:
            results.append(
                run_attention_case(
                    batch=args.batch,
                    device = args.device,
                    dtype=dtype,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    heads=args.heads,
                    sequence=sequence,
                    head_dim=args.head_dim,
                    case = case_mode,
                    layers=args.layers,
                    kv_heads=args.kv_heads,       
            ))
    payload = {
        "schema_version":1,
        "benchmark": BENCHMARK,
        "results": results
    }
    save_result(payload, args.output)

if __name__ == "__main__":
    main()
