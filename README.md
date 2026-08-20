# ai-infra-journey

一个面向 C++ 开发者的 AI Infra 学习与工程仓库。目标是在三个月内建立从 PyTorch、CUDA Kernel、Inference Runtime 到 C++ Serving 和 LLM Serving 的完整实践能力。

## Hardware

- RTX 4070 Ti 12GB：CUDA、TensorRT、GPU Benchmark 与 Serving。
- MacBook Pro M1 Pro：理论学习、PyTorch CPU 实验、跨平台 C++、测试与报告。

## Learning path

```text
AI Infra 基础与性能指标
  -> PyTorch Tensor 与训练闭环
  -> Transformer 与 KV Cache
  -> CUDA 与 Kernel 优化
  -> PyTorch 自定义算子
  -> ONNX Runtime / TensorRT
  -> C++ 推理服务
  -> Docker / Kubernetes / 分布式
  -> LLM Serving 与求职项目
```

## Repository structure

```text
.
├── notes/
│   ├── concept/        # 概念资料
│   └── daily/          # 日报、闭卷复测与错题
├── python/
│   ├── tensor_playground.py
│   └── tests/
├── cpp/
│   ├── CMakeLists.txt
│   ├── src/
│   └── tests/
├── benchmarks/         # 原始性能数据
├── reports/            # 可复现报告
└── environment.md      # 双机环境基线
```

## Current progress

状态更新时间：2026-08-20。

| 模块 | 当前证据 | 状态 |
|---|---|---|
| 仓库与 Git 基线 | README、`.gitignore`、目录结构和日报已建立 | 已完成 |
| AI Infra 四类方向 | 已完成第一轮学习和闭卷；8.19 约 `4/10` | 已学习，待复测 |
| 推理链路与性能指标 | 已学习 Latency、QPS、Queue Time、p99、TTFT、TPOT、ITL 等；8.19 部分得分 `6/10` | 已学习，待复测 |
| 双机环境 | Windows/WSL、4070 Ti、Mac 与 PyTorch 信息已有初版记录 | 部分完成，待订正 |
| Tensor Playground | shape、stride、dtype、device、broadcasting、reduction、matmul、非连续布局和设备往返已有实现 | Mac/CPU 通过 |
| Tensor 自动测试 | 2026-08-20 在 Mac 实测 `9 passed, 1 skipped` | CPU 通过，CUDA 待复测 |
| C++20/CMake/CTest | 文件已预留，当前仍为空 | 未完成 |
| 两层 MLP 训练闭环 | 已列入 8.20 任务，仓库尚无实现和测试 | 未完成 |

详细验收与订正见 [8.19 日报](notes/daily/8.19日报.md)。

## Completed Tensor practice

`python/tensor_playground.py` 当前覆盖：

- 稠密 Tensor 逻辑字节数计算；
- shape、stride、dtype、device、layout、contiguous 和 storage offset 描述；
- Broadcasting：`[8,1,128] + [1,64,128] -> [8,64,128]`；
- `transpose()` 后的非连续布局；
- 非连续 Tensor 上 `view()` 失败，以及 `reshape()`、`contiguous().view()` 修复；
- Reduction 与 Matrix Multiplication；
- NumPy/Tensor 共享内存；
- CUDA 可用时的 CPU -> CUDA -> CPU 冒烟验证。

## Quick start

前置条件：Python、PyTorch 和 pytest 已安装。

```bash
python python/tensor_playground.py
python -m pytest -q
```

当前 Mac 参考结果：

```text
9 passed, 1 skipped
```

CUDA 测试在不支持 CUDA 的机器上应当 Skip。只有在 RTX 4070 Ti 主机上实际运行成功后，才记录为 CUDA Pass。

## Known gaps

1. `reduction_and_matmul_demo()` 返回键名为 `mean_last_dim`，当前实现却使用 `mean(dim=1)`，命名与计算维度需要对齐。
2. Reduction/MatMul 测试尚未直接调用 `reduction_and_matmul_demo()`，需要补充函数级测试。
3. `view_eroor` 需要改名为 `view_error`；`tensor_nbytes()` 的 Docstring 需要明确描述逻辑字节数而非底层 Storage。
4. RTX 4070 Ti 的 VRAM、Windows/WSL CMake 和 Mac MPS 字段需要重新采集并订正。
5. RTX 4070 Ti 上的完整 pytest 与 CPU -> CUDA -> CPU 冒烟测试尚未留下本轮复测证据。
6. `cpp/CMakeLists.txt` 与 `cpp/src/main.cpp` 仍为空，C++20/CMake/CTest 尚未通过验收。

## Next milestone

G0 基础工程的下一步：

- 修复 Tensor 已知问题并补直接测试；
- 完成 C++20 Hello World、CMake target 和至少 1 个 CTest；
- 实现两层 MLP、合成数据与最小训练循环；
- 验证 Loss 下降，完成至少 6 个 MLP 测试；
- 保存并加载 `state_dict`，固定输入最大绝对误差小于 `1e-6`；
- 在 RTX 4070 Ti 上补齐 CUDA 复测证据。

## Safety

- 不提交 Token、私钥、密码、公司数据或个人敏感信息。
- 不提交模型权重、TensorRT Engine、构建目录、缓存和本地环境文件。
- Benchmark 必须记录硬件、软件、shape、dtype、Batch Size、Warmup、同步方式和重复次数。
- README 和日报只记录实际运行结果，不用计划或空文件表示完成。
