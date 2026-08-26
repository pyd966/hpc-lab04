# GPU 阶段 3 扩展：真实依赖图与跨变量 stream 并发

## 结论

本轮改动保留。它基于 `6fc58e2` 的全流程依赖图，只并行化确认读写不相交的
AMR prolong 变量；RHS、RK 层级边界和 ghost exchange 的真实依赖保持不变。

三次 `t=5` 端到端均值由 `97.693319 +/- 0.285257 s` 降到
`93.258103 +/- 0.073732 s`，改善 `4.54%`。三次 checker 均为
`FINAL: PASS`，trajectory RMS 为 0，constraint 上限通过。

## 全流程真实依赖图

省略 TwoPuncture 后，一次 coarse step 的主要执行关系为：

```text
可选 analysis(读当前 State，产出 Psi4/ADM/monitor)
  -> level predictor
     每个 Block: enforce -> RHS -> Sommerfeld/RK/lower-bound
  -> ghost/buffer exchange(SynchList_pre)
  -> 3 x corrector
     每个 Block: enforce -> RHS -> Sommerfeld/RK/lower-bound
     -> ghost/buffer exchange(SynchList_cor)
  -> 递归推进 fine level
  -> fine-to-coarse restrict / 时间插值 / coarse-to-fine prolong
  -> 下一 coarse step
```

依赖边界如下。

| 范围 | 可并行部分 | 必须等待的消费者 |
| --- | --- | --- |
| 同一 RK 子步、不同 Block | State/RHS/scratch 都属于各自 Block，可并行 | ghost exchange 必须看到所有参与 source Block 的结果 |
| 同一 Block 的 RHS | 存在若干公式分支 | 下游复用 RHS 与几何 scratch，必须在边界处 join |
| 同一变量的 RK stages | 下一 stage 读取上一 stage 的 state/ghost | 不能跨 RK stage 或 ghost exchange |
| 不同 AMR 变量的 prolong | source、target 和 packed-buffer offset 均不相交 | unpack/下一层计算前必须全部完成 |
| coarse/fine levels | 同层 Block 可并行 | subcycling、restrict、时间插值和 prolong 是真实因果链 |
| analysis 与 evolution | analysis 多数只读 State | 当前分析会复用临时场并立即回传 host；Step 不能提前覆盖 State |

当前 4 条原始 stream 对应 `Block` 的轮转分配。同一 Block 的操作依靠 stream
顺序；不同 Block 原本已经可以并发。因此，简单增加 Block stream 数不会创造
新的工作。

### RHS 内部依赖

源码中可还原出如下局部 DAG：

```text
geometry
  -> {ricci_a, gamma_derivatives}
  -> {Gamma-x/y/z seed}
  -> {beta/Gamma prepare -> beta/Gamma, metric second derivatives}
  -> {Ricci diagonal, Ricci off-diagonal}
  -> {metric source, chi Hessian}
  -> {chi Ricci, physical Gamma}
  -> lapse Hessian -> trace source
  -> {Aij diagonal, Aij off-diagonal}
  -> gauge -> advection -> optional constraints
```

这些分支在数学上可以用 event 表达，但本轮没有把 RHS 放到更多 stream：热点
grid 通常有 343--896 个 block，而 MIG 只有 14 个 SM，单 kernel 已有足够 block
覆盖整块 GPU。104-register kernel 每个 SM 已驻留两个 256-thread block；换成
两个 grid 各驻留一个 block 不会增加 active warp，只会增加 cache/带宽竞争。
原 Nsys 中 RHS `39.550 s` duration sum 的并集仍为 `38.095 s`，也说明跨 Block
stream 只能在 kernel 尾部形成有限重叠。RHS 的后续收益仍应来自减少单点工作量
和访问等待，而不是增加 stream。

### 选择 prolong 的原因

原 Nsys 中 `47,307` 次 `prolong3_kernel` 总时间和并集同为 `8.578469 s`，即
调用之间完全没有 overlap。该 kernel 为 66 registers/thread、256 threads/block，
NCU 既有结果为理论 occupancy 37.5%、实际 26.77%；典型 launch 只有约 54 个
block。14 个 SM 每个最多驻留 3 个此类 block，一次 launch 的最后一波利用率低，
适合把独立变量的 block 混合调度来填补尾部。

restrict 使用 183 registers/thread，通常只能驻留一个 block/SM；Sommerfeld
使用 156 registers/thread 且 grid 足够大。它们即使拆到更多 stream，也不会
增加同时驻留的同类 block。analysis kernel 在本次 `t=1` profile 中合计远低于
RHS/AMR 主路径，因此也没有为低收益增加复杂依赖。

## 实现

`GPUManager` 为 4 条原始 parent stream 各建立 2 条专用 auxiliary stream，并为
每组建立 timing-disabled fork/join event。API 行为是：

```text
parent 上已有 producer
  -> record fork event
     -> parent: variables 0,3,6,...
     -> aux 0: variables 1,4,7,...
     -> aux 1: variables 2,5,8,...
  -> 每条 aux record join event
  -> parent wait 两个 join event
  -> 原有 touched-stream wait / unpack / 下游消费者
```

`Parallel::gpu_data_packer()` 仅在 `PACK && type == prolong && variable_count > 1`
时启用三路轮转。每个变量仍使用原来的 `prolong3_kernel`、SoA/parity 和目标
buffer offset；没有修改插值公式、block/grid、MPI 分支或变量顺序。非 pool
stream 无法取得 companion 时自动退化为原单 stream 路径。

event 的 join 回到原 Block stream，因此调用方仍只需等待原 parent；这保持了
既有 `Sync_GPU()` 和 coarse/fine 消费者的接口语义。

## 实验与 profile

### Nsys 两路/三路对照

固定 `t=0..1`、11,955 次 prolong：

| 指标 | 两路 | 三路 |
| --- | ---: | ---: |
| prolong duration sum | 2.870782 s | 3.152633 s |
| prolong 时间并集 | 1.551393 s | **1.497268 s** |
| 双路 overlap wall time | 1.319389 s | 0.895281 s |
| 三路 overlap wall time | 0 | 0.380042 s |
| 全部 GPU kernel 时间并集 | 12.917585 s | **12.872666 s** |
| Total Evolve | 12.893 s | **12.819 s** |

并发会让单个 prolong 变慢，因此 duration sum 上升；正确比较对象是时间并集和
端到端时间。三路在这两个指标上都略优，最终保留三路。按原 t=4 profile 的
单次均值缩放到 11,955 次，串行 prolong 约为 `2.169 s`；三路并集下降约 31%。

最终 Nsys 的所有 kernel duration sum 为 `15.271865 s`，并集为
`12.872666 s`。并发墙钟分布为：单 kernel `10.872483 s`、双 kernel
`1.607826 s`、三 kernel `0.385695 s`、四 kernel `0.006661 s`。相较原 profile
约 5.3% 的多 kernel 墙钟占比，本轮达到约 15.5%。

新增 1,590 次 `cudaEventRecord` 和 2,120 次 `cudaStreamWaitEvent`，Nsys API
时间分别约 `1.64 ms` 和 `2.06 ms`，不是新瓶颈。VTune 的主要 host 等待仍是
`cuMemcpyHtoD_v2 4.576 s`、`cuStreamSynchronize 4.444 s` 和
`cuCtxSynchronize_v2 2.898 s`；event API 未进入热点列表。

没有重跑 NCU：本轮没有改变任何 kernel 源码、launch shape 或编译属性；Nsys
确认 `prolong3_kernel` 仍为 66 registers/thread。因此既有 NCU occupancy 结论仍
适用，本轮需要验证的是 kernel 间时间关系，Nsys 比重复 NCU 更直接。

### 正确性与端到端性能

最终三路方案在 `t=0..5` 独立执行三次：

| 版本 | 三次用时 | 均值 | 样本标准差 |
| --- | --- | ---: | ---: |
| 修改前 `6fc58e2` | 97.863 / 97.354 / 97.863 s | 97.693319 s | 0.285257 s |
| 本轮三路方案 | 93.255 / 93.333 / 93.186 s | **93.258103 s** | 0.073732 s |

端到端改善 `4.435216 s`，即 `4.54%`。三次均通过 `FINAL: PASS`，trajectory
RMS 为 0，Hamiltonian/momentum constraint 检查通过。

将之前 `t=100` 的约 `1389.9 s` 按本轮比例缩放，短窗口外推约为 `1326.8 s`。
这只能用于判断量级，不能替代真正的长程测试；距 `330 s` 目标仍很远，说明仅靠
stream tail filling 无法完成总目标。

### 可复现实验产物

- 修改前 Nsys：`profile/gpu-nsys-20260826T022152Z-64`
- 修改前端到端：`profile/gpu-benchmark-20260826T022748Z-65`
- 最终三路 Nsys：`profile/gpu-nsys-20260826T032334Z-65`
- 两路对照 Nsys：`profile/gpu-nsys-20260826T032736Z-64`
- 最终 VTune：`profile/gpu-vtune-20260826T033817Z-64`
- 最终三次端到端：`profile/gpu-benchmark-20260826T033210Z-61`

## 后续方向

本轮已覆盖全流程中最明确、成本最低的 stream 并行机会。继续增加 RHS 或 restrict
stream 不会突破寄存器决定的驻留上限。下一阶段应优先减少 AMR 的大量小调用和
host 同步：把同层多个变量的 pack/prolong/unpack 合为批次，减少 launch、拷贝与
`cudaStreamSynchronize` 次数；随后重新测量 RHS 和 AMR 在长程运行中的占比，再
决定是否值得对 RHS 做更深的公式融合或数据流重构。
