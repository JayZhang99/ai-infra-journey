# Benchmark artifacts

本目录只保存可复现实验产生的性能证据，按工作负载和证据类型组织。

## Layout

```text
benchmarks/
├── matmul/
│   ├── runs/          # MatMul 基准测试原始 JSON
│   ├── profiles/      # PyTorch Profiler trace 与 Top 表
│   └── summaries/     # CPU/CUDA 对比等派生结果
├── attention/
│   └── runs/
├── h2d/
│   ├── runs/          # 单轮与多轮 H2D 实验原始 JSON
│   ├── profiles/      # H2D trace 与 Top 表
│   └── summaries/     # 跨轮聚合结果
├── softmax/
│   ├── runs/
│   └── profiles/      # Nsight Systems 报告及配套运行结果
├── rmsnorm/
│   └── runs/
└── reduction/
    └── runs/          # 首次运行 reduction benchmark 时自动创建
```

## Artifact roles

- `runs/`：Benchmark 程序直接产生的原始结果；后续分析不能覆盖它。
- `profiles/`：Profiler、Perfetto 或 Nsight 产生的 trace、统计表和数据库。
- `summaries/`：由一个或多个原始结果计算得到的对比、聚合和结论数据。

## Naming convention

使用以下顺序命名：

```text
YYYY-MM-DD_[platform-or-device]_[variant]_[runNN].[ext]
```

规则：

1. 日期使用完整的 `YYYY-MM-DD`，不再使用 `9.3` 或 `0829`。
2. 同一协议的独立重复实验使用 `run01`、`run02`，不得覆盖旧文件。
3. 文件所在目录已经表达算子名称，因此文件名不必重复算子名。
4. `chunks`、`batched200`、`full`、`light` 等会影响解释的协议信息保留在文件名中。
5. Trace 与统计表共享相同前缀，只用 `_trace.json`、`_top.txt` 等后缀区分。

## Writing new results

示例：

```bash
# RMSNorm 原始结果
./cpp/build-cuda130/cuda-kernel-lab/rmsnorm_bench \
  benchmarks/rmsnorm/runs/2026-09-13_f32_batched200_run02.json

# Softmax 原始结果
./cpp/build-cuda130/cuda-kernel-lab/softmax_bench \
  benchmarks/softmax/runs/batched200/2026-09-13_f32_run01.json

# MatMul Profiler
python3 -m python.profile_matmul \
  --device cuda:0 --size 512 --dtype float32 \
  --warmup 5 --steps 10 --capture_mode light \
  --trace benchmarks/matmul/profiles/2026-09-13_cuda_light_trace.json \
  --top5 benchmarks/matmul/profiles/2026-09-13_cuda_light_top5.txt
```

## Interpretation rules

- 原始结果与 Profiler 结果分开；Profiler 开销不能当作正常运行延迟。
- `batched_average_per_launch` 的 p95 是“批平均值的 p95”，不是单次 Kernel 尾延迟。
- 报告带宽时必须说明使用的是逻辑字节数还是 Profiler 测得的 DRAM 字节数。
- 聚合多轮实验时保留每轮文件，汇总文件记录聚合方法，不直接合并全部样本。
