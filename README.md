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
│   ├── tensor_playground.py  # Tensor、布局、广播与设备往返
│   ├── mlp_demo.py           # 两层 MLP、训练与 state_dict
│   └── tests/                # CPU/CUDA 自动测试
├── cpp/
│   ├── CMakeLists.txt        # C++20 target 与 CTest
│   └── src/main.cpp
├── benchmarks/         # 原始性能数据
├── reports/            # 可复现报告
└── environment.md      # 双机环境基线
```

## Current progress

状态更新时间：2026-08-21。

| 模块 | 当前证据 | 状态 |
|---|---|---|
| 仓库与 Git 基线 | README、`.gitignore`、目录结构和日报已建立 | 已完成 |
| AI Infra 基础理论 | 8.20 定点复测 `3/5`；Linear/Module 闭卷 `6.5/10` | 已学习，错题待复测 |
| 推理链路与性能指标 | 已覆盖 Latency、QPS、Throughput、Concurrency、Queue Time、p99、TTFT、TPOT、ITL | 已学习，Concurrency/有界队列待巩固 |
| 双机环境 | Mac M1 Pro + WSL/RTX 4070 Ti；Mac 全量 pytest `15 passed, 2 skipped`，WSL `17 passed` | CPU/CUDA 双端基线通过 |
| Tensor Playground | shape、stride、dtype、device、broadcasting、reduction、matmul、非连续布局和设备往返已有实现 | Mac CPU 与 WSL CUDA 通过 |
| Tensor 自动测试 | Mac `9 passed, 1 skipped`；WSL CUDA 测试包含在全量 `17 passed` 中 | 已完成当前验收 |
| C++20/CMake/CTest | C++20 target 已建立，实际编译含 `-std=c++20`；Mac CTest `1/1 Passed` | Mac 基线通过，WSL 日志待留存 |
| 两层 MLP 训练闭环 | 354 个参数；Loss `13.4590 -> 0.0194`；state_dict 误差 `0.0`；WSL CUDA smoke 通过 | CPU/CUDA 通过 |

详细验收与订正见 [8.19 日报](notes/daily/8.19日报.md) 和 [8.20 日报](notes/daily/8.20日报.md)。

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

当前验收：

- Mac：`9 passed, 1 skipped`，CUDA roundtrip 按预期跳过；
- WSL/RTX 4070 Ti：CUDA roundtrip 通过，并包含在全量 `17 passed` 中。

## Completed MLP practice

`python/mlp_demo.py` 当前覆盖：

- `Linear(8, 32) -> ReLU -> Linear(32, 2)` 两层 MLP；
- `[512, 8] -> [512, 2]` 的合成回归数据；
- MSE Loss、Adam、`zero_grad()`、`backward()`、`step()` 训练闭环；
- 模型参数注册与参数量检查，总参数量为 `354`；
- `train()`、`eval()` 和 `inference_mode()` 的最小验证；
- 保存与加载 `state_dict`，固定输入输出最大绝对误差为 `0.0`；
- 模型、输入和 target 同时迁移到 `cuda:0` 的 CUDA smoke。

当前参考结果：

```text
Mac / CPU loss: 13.459022521972656 -> 0.019438406452536583
Mac / MLP pytest: 6 passed, 1 skipped
WSL / full pytest: 17 passed in 3.65s
```

## Completed C++20 baseline

`cpp/` 当前覆盖：

- CMake C++20 target，强制标准且关闭编译器扩展；
- `std::vector`、`std::accumulate` 和返回码自检；
- 至少一个由 CTest 驱动的可执行程序测试。

Mac 实测环境与结果：

```text
CMake 4.4.2
AppleClang 14.0.0
compile flag: -std=c++20
CTest: 1/1 Passed
program: C++20 baseline OK
```

## Quick start

前置条件：Python、PyTorch、pytest、CMake 和 C++ 编译器已安装。

Python：

```bash
python python/tensor_playground.py
python python/mlp_demo.py
python -m pytest -q
```

C++：

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure
```

双机全量 pytest 参考结果：

```text
Mac: 15 passed, 2 skipped
WSL: 17 passed in 3.65s
```

Mac 上两个 CUDA 测试应当 Skip；WSL/RTX 4070 Ti 上两个 CUDA 测试均已实际通过。

## Known gaps

1. `non_contiguous_demo()` 内部变量 `view_eroor` 仍需改名为 `view_error`；`tensor_nbytes()` Docstring 应明确它计算的是稠密 Tensor 的逻辑字节数，不是任意 Tensor 的实际底层 Storage 大小。
2. `environment.md` 中 WSL CMake 字段存在文字错误；需要用实际命令重新采集 CMake 版本，并保存 WSL CMake/CTest 原始输出。
3. RTX 4070 Ti 的精确 VRAM 应使用 `nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv` 留证，不能只记录产品标称值。
4. 理论测试当前为 `9.5/15（63.3%）`，Parameter 注册、`state_dict`、`eval()`/Autograd、Concurrency 和有界队列需要闭卷复测。
5. 当前训练仍是全量 CPU/GPU 功能验证，尚未建立包含 Warmup、CUDA 同步、重复次数和分位数的正式 Benchmark。

## Next milestone

G0 基础工程的 Tensor、C++20 和 MLP CPU/CUDA 功能闭环已经通过。下一里程碑进入 Autograd 与训练性能基础：

- 闭卷复测 Parameter、`state_dict`、`eval()`、`no_grad()`、`inference_mode()`、Concurrency 和有界队列，目标正确率 `>= 80%`；
- 实现 Autograd Graph、Leaf/Non-leaf Tensor、`grad_fn`、梯度累积、`detach()` 的最小实验与自动测试；
- 记录一次前向、反向传播和优化器更新中的 shape、device、梯度与参数变化；
- 修正 `environment.md`，补齐 RTX 4070 Ti VRAM、PyTorch/CUDA 和 WSL CMake/CTest 原始证据；
- 建立第一个可复现 CPU/CUDA Benchmark，明确 shape、dtype、Batch Size、Warmup、同步方式和重复次数。

## Safety

- 不提交 Token、私钥、密码、公司数据或个人敏感信息。
- 不提交模型权重、TensorRT Engine、构建目录、缓存和本地环境文件。
- Benchmark 必须记录硬件、软件、shape、dtype、Batch Size、Warmup、同步方式和重复次数。
- README 和日报只记录实际运行结果，不用计划或空文件表示完成。
