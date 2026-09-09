### 1. 找座位（1 分）

一个 block 有 256 个线程。

- 一共有多少个 warp？
- `threadIdx.x == 70` 属于第几个 warp？
- 它在该 warp 中的 `lane_id` 是多少？

### 2. 32 人小队（1 分）

判断下面哪句话更准确，并说明理由：

A. 一个 warp 的 32 个线程就是一个线程，只是同时处理 32 份数据。
B. 一个 warp 包含 32 个独立线程，但 GPU 通常以 warp 为单位选择和发射指令。
C. 一个 warp 的 32 个线程任何时候都必须执行完全相同的代码。

### 3. 分支发散（1.5 分）

一个 warp 执行：

```
int lane = threadIdx.x % 32;

if (lane < 8) {
    y = expensive_a(x);
} else {
    y = expensive_b(x);
}
```

回答：

- GPU 是否会同时只执行其中一条路径？
- 执行 `expensive_a` 时哪些 lane 有效？
- 执行 `expensive_b` 时哪些 lane 有效？
- 为什么这通常比 32 个 lane 全走同一路径更慢？

### 4. Mask 是点名册（1.5 分）

只有 `lane 0~19` 需要参加一次 Shuffle。

- `mask` 表示“当前正在运行的线程”，还是“必须参加这次协作的线程集合”？
- 为什么直接使用 `0xffffffff` 可能错误？
- 如果参加者恰好是 `lane 0~19`，对应的低 20 位 mask 可以写成什么十六进制值？

### 5. Shuffle 搬运什么（2 分）

假设每个 lane 初始保存：

```
lane 0 保存 1
lane 1 保存 2
...
lane 31 保存 32
```

执行：

```
value += __shfl_down_sync(
    0xffffffff,
    value,
    16
);
```

回答：

- `lane 0` 从哪个 lane 读取值？结果是多少？
- `lane 10` 从哪个 lane 读取值？结果是多少？
- Shuffle 搬运的是 Shared Memory 中的数据，还是 lane 寄存器中的值？
- 它能不能让 `warp 0` 直接读取 `warp 1` 的寄存器？

### 6. Warp Reduction（1.5 分）

解释下面代码为什么最终可以让 `lane 0` 得到一个 warp 的总和：

```
for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(
        0xffffffff,
        value,
        offset
    );
}
```

请至少说明：

- 为什么 offset 是 `16、8、4、2、1`？
- 每轮之后，有效部分大约缩小成什么样？
- 如果输入是 `1~32`，最终 `lane 0` 应是多少？

### 7. 跨 Warp 与 Occupancy（1.5 分）

一个 block 有 256 个线程，也就是 8 个 warp。每个 warp 已经算出了一个局部和。

- 为什么不能直接用 Shuffle 把这 8 个结果合并？
- 一种常见的跨 warp 合并过程是什么？
- 把 Occupancy 从 50% 提升到 100%，Kernel 是否一定会快？为什么？

请直接按下面格式回答：

```
1. 8个warp，第三个warp,lane_id = 6
2. B,但是可能会执行到代码的不同分支
3. 不会都走同一路径，0-7 走expensive_a ， 8-31 走expensive_b，增加判断损耗
4. 必须参加这次协作的线程集合，0x 000fffff
5. lane 16 ，结果 17， lan 26,结果38， lane寄存器，不能
6. offset每次循环/=2， 16，8，4，2，1， 528
7. Shuffle 同warp内，   ， 
```





`lane 3` 的值为 4，`lane 19` 的值为 20。执行 `value += __shfl_down_sync(mask, value, 16)` 后，`lane 3` 等于多少？

24

为什么分支发散的核心成本不是一次 `if` 判断？

主要是有路径等待 分支A 执行路径A，分支B 执行路径B，成本则为路径A + 路径B

8 个 Warp 的局部和如何变成一个 Block 总和？

warp shuffle -> lane 0 写到 shared memory -> __syncthreads() -> warp shuffle

Occupancy 从 50% 提升到 100%，为什么可能完全没有加速？

Occupancy 100%可能导致内存溢出反而速度劣化