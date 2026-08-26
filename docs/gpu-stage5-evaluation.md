# GPU 阶段 5 评估：RK4/边界/内存

## 结论

阶段 5 的两项实现性实验均通过短窗数值校验，但没有通过端到端性能门槛，因此没有合入主线。当前主线仍为阶段 4 提交 `1a6023b`，阶段 4 的三次 `t=5` 基准为 `89.150199 +/- 0.301396 s`。

阶段 5 实验得到的主要结论是：RK4 kernel 的 launch 数确实可以减少，但 RK4 本身只占总 GPU 时间约 1%，新增的 device pointer-table 维护和跨字段访存抵消了收益；通用显存池还增加了初始化/清零和显存驻留成本。阶段 5 不应继续围绕这两个方向堆叠微优化。

## 实验口径

- GPU 环境、输入、编译参数和 checker 与前几阶段一致：A100 80GB PCIe 的 MIG `1g.10gb`，单 MPI rank，OpenMP OFF，`sm_80`，`-O3`。
- TwoPuncture 使用固定 cache；只比较 ABEGPU 端到端 `This Program Cost`。
- 所有候选先完成构建和 `t=5` correctness，再进行 profile。短窗只用于开发门禁，未将其外推为最终 `t=100` 成绩。

## 实验 A：RK4 跨变量批处理

### 实现

候选代码曾在 `rungekutta4_rout_gpu.cu` 增加一个二维 launch：`x` 维为空间点，`y` 维为 24 个状态变量；kernel 通过 Block 的 device pointer table 取得 `State/Pre(or Cor)/RHS` 三组字段，一次完成所有变量的 RK4 更新。为保持旧路径的 pointer swap 语义，`Block::swapList` 维护 host pointer 数组，下一次批处理前把变化延迟同步到 device table；细层 Sommerfeld 的 post-boundary 操作仍在 RK4 之后执行，未改变 producer-consumer 顺序。

### 正确性与端到端

| 候选 | Job / artifact | `This Program Cost` | checker |
| --- | --- | ---: | --- |
| 原始 pointer-swap kernel 版本 | 168759 / `profile/gpu-benchmark-stage5-rk2/gpu-benchmark-20260826T104200Z-63` | 90.155483 s | PASS |
| 延迟 table 同步版本 | 168843 / `profile/gpu-benchmark-stage5-rk3/gpu-benchmark-20260826T105655Z-61` | 90.176349 s | PASS |
| 阶段 4 主线基准 | `profile/gpu-benchmark-20260826T093400Z-65` | 89.150199 +/- 0.301396 s | 3/3 PASS |

延迟同步版本相对阶段 4 基准慢约 1.15%，超过基准波动范围，故不合入。两次候选都只跑了一次 `t=5`，但差异方向一致；最终决策以阶段 4 的三次基准为准。

### Nsys 证据

候选延迟同步版本的 `t=0..4` profile 为 `profile/gpu-nsys-stage5-rk3/gpu-nsys-20260826T105931Z-64`。为便于与阶段 4 的 `t=0..1` profile（`profile/gpu-nsys-20260826T092137Z-66`）比较，下面按物理时间单位归一化：

| 指标 | 阶段 4 profile | RK4 批处理 profile | 变化解释 |
| --- | ---: | ---: | --- |
| 旧 RK4 kernel | 0.2188 s / unit，9408 calls / unit | 不再出现 | launch 数减少 |
| 批处理 RK4 kernel | 不适用 | 0.2015 s / unit，392 calls / unit | kernel 约快 8% |
| `cudaLaunchKernel` | 44926 calls / unit | 35903 calls / unit | 延迟 pointer swap 后减少约 20% |
| `cudaMemcpyAsync` | 81 calls / unit | 362.5 calls / unit | table 同步显著增加调用次数；总 API 时间约 0.21 -> 0.22 s / unit |

第一版原始 pointer-swap 实现每次 `swapList` 都发射一个小 kernel，profile 中总 launch 达到 `145179`（`t=0..4`），因此被立即淘汰。延迟同步修复了这个问题，但 table copy 的 API/host 代价仍大于 RK4 的可节省时间。候选 profile 的批处理 kernel 为 `1568` calls、总 `0.8058 s`；这是 4 个物理单位的总量。

### NCU 证据

artifact：`profile/gpu-ncu-stage5-rk2/gpu-ncu-20260826T104939Z-64/ncu-details.csv`。

- block `(256,1,1)`，grid `(125,24,1)`。
- 20 registers/thread，无 local/shared spill。
- 理论 occupancy `100%`，实际 occupancy `91.10%`，所以寄存器不是批处理变慢的原因。
- Memory Throughput `90.28%`，L1/L2 hit `30.90%/25.14%`；NCU 指出 global sector 利用率约 `29.3/32 bytes`，主要等待 L1TEX scoreboard（约 90.7% warp stall samples）。

这说明把 24 个字段同时交错执行破坏了原来逐变量 kernel 的 cache/访存局部性；仅减少 launch 不能带来端到端收益。

## 实验 B：通用显存复用池

### 实现

`GPUManager` 原有按 `num_elements` 分桶的 `memory_pool` 被临时启用：allocate 从空闲桶取回并清零，free 只归还到桶，程序结束时统一 `cudaFree`。此实验没有改变 kernel 算法，也没有把 Block 的持久字段放入池中。

### 结果

Job 168917，artifact：`profile/gpu-benchmark-stage5-pool1/gpu-benchmark-20260826T110953Z-64`。

- `This Program Cost = 93.564903 s`，相对阶段 4 主线慢约 4.96 s（约 5.0%）。
- trajectory RMS 为 0，constraints 和 FINAL 均 PASS。
- 运行日志显示显存驻留从约 `1.8 GB` 增加到约 `2.1 GB`，并且每步时间逐渐上升。

池化在当前 workload 中没有消除主要等待，反而保留了大量临时分配、在复用时同步清零，并增加 allocator 的显存压力。该实现已撤回。若后续仍需做内存优化，应先按 buffer 生命周期和 stream ownership 设计持久 scratch，而不是启用全局无生命周期信息的通用池。

## 未继续的方向

Sommerfeld 和 lower-bound kernel 的单次成本很小；阶段 4 profile 中 Sommerfeld 约 `0.346 s / unit`，RK4 约 `0.206 s / unit`，lower-bound 仅约 `0.010 s / unit`。把 lower-bound 融入 RK4 的理论节省只有约 10 ms / unit，不能改变端到端结论。Sommerfeld 与 RK4 fusion 还会把边界插值的高寄存器局部数组带入 RK4 热路径，风险明显高于收益，故未盲目合入。

Pinned host staging 和 CUDA Graph 也没有在本阶段合入：前者需要先按 MPI/analysis buffer 的真实生命周期改造接口，后者受 AMR 拓扑和动态 transfer segment 影响，必须在执行图稳定后再做。阶段 5 的 profile 结果不支持继续做小 kernel batching；后续若要追求 `t=100 <= 330 s`，应回到阶段 1/2 的 RHS 数据流和阶段 3 的同步执行图，而不是优化当前只占约 1% 的 RK4。

## 提交状态

源码实验均已撤回，工作树只新增本报告；主线保持阶段 4 的 `1a6023b`。本报告提交后，阶段 5 的最终状态为“候选完成评估但不合入”。
