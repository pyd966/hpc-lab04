# GPU 阶段 3 扩展：stream 与同步范围联合实验

## 结论

本轮没有保留运行时代码，最终仍以 `caf593c` 为当前最优版本。原因不是依赖
无法表达，而是剩余同步点后没有足够的独立 GPU 工作：把正常路径剩余的 215 次
`cudaDeviceSynchronize` 全部改成 stream/event 依赖后，实际多 kernel 重叠比例
仍为约 15.5%，三次 `t=5` 的官方端到端均值反而从 `93.258103 s` 变为
`94.127087 s`（`+0.93%`）。这不满足阶段 3 的性能门槛。

逐 segment 和逐 buffer 的 pack/unpack event 方案也都被淘汰。它们在语义上
正确，但让 unpack 与尚未完成的 prolong 争用 MIG 的显存带宽和 L2，分别把
`t=1` 推高到 `13.227 s` 和 `13.436 s`。

本轮只保留 profile/benchmark 脚本的 artifact-root 参数和日志摘要输出。这些改动
不改变默认运行方式，解决了集群 home quota 紧张时无法完成 profile 的问题。

## 真实依赖图

基于 `Step_GPU`、AMR transfer、插值和曲面积分调用链，剩余边界可以写成：

```text
每个 Block stream:
  State -> RHS/RK -> ghost pack -> MPI/本地 transfer -> unpack -> 下一 RK stage

coarse/fine:
  coarse Future/State -> 时间插值 -> restrict -> Sync_GPU
  -> prolong pack -> unpack fine State -> fine step

analysis:
  各 Block State -> global interpolation -> host/MPI reduction
  -> normalize -> host error/result

surface integral:
  各 Block admmass scratch -> global interpolation
  -> surface reduction -> host/MPI result
```

各边界的判断如下。

| 边界 | 能否缩小 | 原因 |
| --- | --- | --- |
| 同一 Block 的时间插值到 restrict | 可以用同 stream 顺序 | producer 和下游 consumer 都提交到该 Block stream |
| Block State 到 analysis stream | 可以用 event handoff | analysis 只需等待实际提供 State 的 Block |
| analysis 到 host/MPI | 不能异步越过 | CPU/MPI 会立即读取插值、权重或错误标志 |
| 曲面积分结果到 host/MPI | 不能异步越过 | 后续立即做 host 缩放和 `MPI_Allreduce` |
| 同节点 prolong pack 到 unpack | 必须保证整个 packed buffer 就绪 | 多个 producer 写同一临时 buffer；过早 unpack 会与 producer 争用带宽 |
| unpack 到临时 buffer 释放 | 必须等待所有 consumer | 否则 `cudaFree` 前仍有在途读取 |
| 跨节点 CPU staging | 必须等待参与通信的 stream | D2H 完成后 MPI 才能读取 host buffer |
| NaN/error dump | 保留 device-wide wait | 异常路径要冻结并完整导出所有 device-backed state |

这里最重要的区别是：event 能表达依赖，不等于存在可获益的并行窗口。当前
analysis 的 host/MPI 消费、AMR 临时 buffer 生命周期和下一 RK stage 都紧跟在
producer 后面；缩小等待集合后，host 仍然在同一临界路径上等待。

## 实验实现

候选实现做了三类改动，均在最终版本中回退。

1. 为每条 parent stream 增加 timing-disabled handoff event。producer record，
   consumer `cudaStreamWaitEvent`，同 stream 自动省略 event。
2. 时间插值依赖同一 Block stream 的顺序；analysis 和 surface 路径只等待参与的
   Block/结果 stream，正常路径不再调用 `synchronize_all()`。
3. 对单 rank 的同节点 AMR transfer 分别测试：
   - segment 级：每个 pack segment 完成后允许对应 unpack 立即开始；
   - buffer 级：所有 producer event 汇聚为 buffer-ready event，再放行 consumer。

segment 级暴露了最多重叠，但也最早引入 prolong/unpack 带宽竞争。buffer 级避免
读未完成的区域，却仍改变了 producer/consumer 的执行节奏。event API 本身只有
毫秒级总开销，不是退化来源。

## Nsight Systems

环境统一为 A100 80GB PCIe 的 `1g.10gb` MIG、1 个 MPI rank、ABEGPU OpenMP
关闭、`t=0..1`；TwoPuncture 在采样区外。

| 指标 | 保留版本 `caf593c` | 全 stream/event 候选 |
| --- | ---: | ---: |
| Total Evolve | 12.7421 s | 12.7367 s |
| kernel duration sum | 15.168282 s | 15.198260 s |
| kernel 时间并集 | **12.789788 s** | 12.810122 s |
| 并发数 >= 2 的墙钟 | 1.982553 s | 1.989156 s |
| 并发墙钟占 kernel 并集 | 15.50% | 15.53% |
| 平均并发 kernel 数 | 1.1860 | 1.1864 |
| `cudaDeviceSynchronize` | 215 次 / 3.212 s | 0 |
| `cudaStreamSynchronize` | 472 次 / 4.848 s | 648 次 / 8.063 s |
| `cudaStreamWaitEvent` | 2,120 次 / 2.06 ms | 2,168 次 / 2.19 ms |

候选把 device-wide wait 完整换成了局部 wait，但两类等待合计和实际重叠都没有
改善；kernel 并集还增加约 20 ms。因此单次 Total Evolve 的 5 ms 差异只是噪声，
不能作为收益。

pack/unpack 两个额外实验：

| 方案 | Total Evolve | `prolong3_kernel` duration sum | 结果 |
| --- | ---: | ---: | --- |
| 保留版本参考 | 12.819 s | 3.152633 s | 当前三路 prolong |
| segment event | 13.227 s | 3.298613 s | 淘汰 |
| buffer-ready event | 13.436 s | 3.304370 s | 淘汰 |

segment 方案新增的 event 调用总开销不足 10 ms，却退化约 0.4 s；这进一步支持
“资源竞争而非 event 开销”的判断。

当前可在 Nsight Systems GUI 中直接打开：

- 保留版本：`profile/gpu-nsys-20260826T054302Z-65/nsys.nsys-rep`
- 最初 baseline：`profile/gpu-nsys-20260824T120352Z-65/nsys.nsys-rep`

保留版本对应作业 166974；候选、segment、buffer 实验分别对应作业 166919、
166806、166833。按 profile 清理约定，只持久保存 baseline 和当前保留版本。

## 端到端与正确性

`t=5` 三次结果：

| 版本 | This Program Cost | 均值 / 样本标准差 |
| --- | --- | ---: |
| 保留版本 | 93.255 / 93.333 / 93.186 s | **93.258103 +/- 0.073732 s** |
| 全 event 候选 | 94.081 / 93.798 / 94.502 s | 94.127087 +/- 0.354602 s |

只看演化段，保留版本均值为 `64.3432 s`，候选为 `64.6332 s`（`+0.45%`）。
候选的三次 trajectory RMS 都为 0，Hamiltonian/momentum constraints 和
`FINAL: PASS` 全部通过。因此淘汰原因纯粹是性能，不是数值错误。

## VTune 与 NCU

候选 VTune 作业 166908 的 CPU time 为 `18.250 s`，旧版为 `18.810 s`。
`cuStreamSynchronize` 从 `4.444 s` 上升到 `7.197 s`，`cuMemcpyHtoD_v2`
从 `4.576 s` 降到 `4.198 s`。这与 Nsys 一致：等待被重新分类和局部化，
但 host 临界路径没有缩短。

本轮没有重跑 NCU。所有候选都没有修改 kernel 源码、block/grid、寄存器数或
编译属性；问题是 kernel 之间的时间关系，Nsys 比重复采集 NCU 更直接。现有
RHS/prolong occupancy 结论仍适用。

## 后续建议

1. 不再单独消除剩余 215 次 device sync。只有消费者也能延后时，event handoff
   才可能产生收益。
2. 优先进入阶段 4 的跨变量 AMR batching：减少 11,955 次 prolong launch，并
   让 pack/prolong/unpack 以完整批次产生真正独立的工作，而不是让相同带宽热点
   强行重叠。
3. analysis 若要并行，必须先把 host/MPI reduction 从每次调用的立即消费者改成
   双缓冲任务；在不改变输出时序和数值语义前不能直接跨 step。
4. 后续每个异步方案同时比较 kernel duration sum、时间并集和端到端。只减少
   `cudaDeviceSynchronize` 次数不构成优化。

## Profile 脚本

`hpc_gpu_profile.sh` 新增 `AMSS_PROFILE_ROOT`，
`hpc_gpu_benchmark.sh` 新增 `AMSS_BENCHMARK_ROOT`。两者默认仍写入
`profile/`；指定 `/tmp` 时可绕过 home quota，Nsys/VTune 摘要会输出到作业
日志，即使临时 artifact 随 pod 结束消失也能保留关键结果。
