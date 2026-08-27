# GPU 优化到 370 秒的 Profile 报告

**日期：** 2026-08-27
**目标：** 官方 `t=100` 运行时间 `<=370s`  
**当前版本：** P2-A 持久化 transfer buffer + validity-aware 初始化实现

## 1. 结论

当前主线还没有接近 `370s`。阶段 4 基线的 `t=5` 三次结果为
`89.150199 +/- 0.301396s`；P0-A 为 `84.722587s`，P0-B 为
`83.529860 +/- 0.283895s`，P1-A 为 `82.166688 +/- 0.084628s`。按当前演化阶段近似线性外推，P1-A 的
`t=100` 约为 `1206s`；目标要求演化部分从约 `11.97s/单位时间` 降到约 `3.4s/单位时间`，
约需要 `3.5x` 的整体演化加速。

因此，继续优化 RK4 launch、单纯降低寄存器数或增加 stream 都不足以达到目标。
主攻方向必须是：

1. 避免约束输出重复执行完整 RHS；
2. 消除 stencil helper 产生的 local-memory 流量和重复边界计算；
3. 继续减少 RHS/AMR 的中间数据搬运；
4. 只在确认有独立工作和安全 buffer 生命周期时减少同步。

## 2. Profile 方法与结果

profile 在 HPC A100 MIG（`1g.10gb`，SM80）上执行，正确性检查通过。

- Nsight Systems：job `171518`，`profile/gpu-nsys-20260826T155727Z-65`
- Nsight Compute：job `171582`，`profile/gpu-ncu-20260826T160405Z-64`
- P0-B Nsight Systems：job `174898`，`profile/gpu-nsys-20260827T022226Z-65`
- P0-B benchmark：job `174882`，`profile/gpu-benchmark-20260827T021603Z-66`
- profile 覆盖当前代码的 `t=0..1`；绝对时间以生产构建 benchmark 为准，Nsight 结果用于确定瓶颈结构。

### 2.1 GPU kernel 热点

当前 `t=1` 的演化阶段约 `12.21s`，主要 kernel 为：

| Kernel | 时间 | 占演化阶段 |
|---|---:|---:|
| `rhs_advection_kernel` | 3.33s | 26.0% |
| `rhs_evolution_kernel` | 1.60s | 12.5% |
| `rhs_geometry_kernel` | 0.90s | 7.0% |
| `rhs_beta_gamma_prepare_kernel` | 0.86s | 6.7% |
| `prolong3_batch_kernel` | 0.82s | 6.4% |
| `rhs_constraints_kernel` | 0.55s | 4.3% |
| `restrict3_batch_kernel` | 0.52s | 4.0% |

`rhs_advection_kernel` 是绝对第一热点，但其它 RHS kernel 合计仍占大部分时间，
不能只优化 advection 后期待达到 3.5x。

### 2.2 CUDA API 与同步

- `cudaMemcpy`：3053 次，host API 时间 `7.88s`
- `cudaDeviceSynchronize`：215 次，`3.24s`
- `cudaStreamSynchronize`：472 次，`1.55s`
- `cudaLaunchKernel`：44926 次，只有 `0.23s`

`cudaMemcpy` 的 host 时间包含阻塞等待，不能和 GPU kernel 时间直接相加；它说明
host/device 工作仍有串行化。launch 数量本身已经不是首要瓶颈。

### 2.3 Advection 的 Nsight Compute 证据

当前 kernel 使用 `(8,8,4)` block，首个采样 launch 的指标为：

- 66 registers/thread，理论 occupancy `37.5%`，实际 occupancy `32.5%`；
- `No Eligible` warp `61.7%`，L1TEX scoreboard 等待约占 `51.8%`；
- local memory sector 约占全部 L1TEX sector 的 `85%`；
- local load utilization 约 `28.6%`；
- 没有 compiler spill request；
- L2 hit `96.1%`，DRAM throughput 约 `27.9%`。

结论是 local array、helper 的局部数据访问、重复 stencil/边界计算造成了等待。
这不是单纯的寄存器溢出或分支发散问题，继续追求更高理论 occupancy 不是第一优先级。

## 3. 优先级优化方向

### P0-A：消除 RHS stencil helper 的 local memory

相关代码：

- [`diff_new_gpu.cu`](/home/h3250106394/lab04/src/diff_new_gpu.cu:12)
- [`lopsidediff_gpu.cu`](/home/h3250106394/lab04/src/lopsidediff_gpu.cu:8)
- [`kodiss_gpu.cu`](/home/h3250106394/lab04/src/kodiss_gpu.cu:8)

当前 helper 使用 `double SoA[3]` 等局部数组，传入数组形式的坐标/边界元数据，
且没有 `__forceinline__`。这些 helper 位于不同 `.cu` 文件，而工程启用了
`-rdc=true`，跨 TU device inline 需要单独验证。

建议按以下顺序做 A/B 实验：

1. 将热点 helper 放入 `.cuh`，使用 `__device__ __forceinline__`；
2. 将 `ex[3]`、边界和 spacing 改为标量或 compile-time 参数；
3. 在 kernel 入口一次计算 spacing、边界范围、parity 和坐标因子；
4. 让 lopsided derivative 和 KO 共享这些结果；
5. 保持 advection 与 KO 在同一 kernel，避免额外 global-memory 往返。

验收指标应包括 local load/store sectors、excessive global sectors、eligible warps、
kernel wall time 和 `t=5` 端到端时间，不能只看 register 数。

#### P0-A 实测结果（2026-08-26）

保留的实现将 `d_fderivs_point`、`d_fdderivs_point`、`d_lopsided_point` 和
`d_kodis_point` 中的 `double SoA[3]`/lambda 替换为同一 TU 内的
`__device__ __forceinline__ rhs_symmetry_load`，并把 `ex[3]` 缓存在标量寄存器中。
跨 TU 的完整 helper 强制 inline 没有保留，因为它增加寄存器压力后端到端变慢。

| 版本 | `t=5` program mean | 结果 |
|---|---:|---|
| 阶段 4 主线 | `89.150199 +/- 0.301396s` | 3/3 PASS |
| 仅 lopsided/KO accessor | `86.797500 +/- 0.131332s` | 3/3 PASS |
| accessor + 两个导数 helper | `84.738476 +/- 0.765486s` | 3/3 PASS |
| 最终回退无效 extent 实验后的单次复验 | `84.722587s` | PASS |

对应 artifact 为 `gpu-benchmark-20260826T173115Z-64`、
`gpu-benchmark-20260826T180848Z-66` 和
`gpu-benchmark-20260826T184449Z-66`。最终 NCU（
`gpu-ncu-20260826T181436Z-64`）显示 local-memory sector 占比从约 `85%`
降到 `59%`，但寄存器从 `66` 增到 `107/thread`，理论 occupancy 为 `25%`；
因此该结果应按端到端收益接受，不能继续以“降低寄存器数”为单一目标。

额外的跨函数完整强制 inline 版本为 `90.637607s`，advection kernel 的 NCU
显示 `66` registers/thread、理论 occupancy `37.5%`，但整体变慢，已回退。
只把 advection 的 `dims[3]` 改为三个标量的实验为 `86.719597s`，没有稳定收益，
也已回退。

### P0-B：为约束输出建立 constraint-only RHS 路径

在 [`bssn_gpu_class.C`](/home/h3250106394/lab04/src/bssn_gpu_class.C:2819) 的
约束输出、约束插值和初始化约束计算中，原来都调用完整的
`gpu_compute_rhs_bssn_launch`，因此会重复启动 source/gauge 和 advection/KO。
本轮在 [`bssn_rhs.h`](/home/h3250106394/lab04/src/bssn_rhs.h:38) 定义
`RHS_CONSTRAINT_ONLY = -1`，复用同一 launcher 的接口切换调度模式：

- 保留 geometry、Ricci-A、Gamma derivative/seed、beta-Gamma、evolution 和 Ricci connection producer；
- 跳过全部 `rhs_source_*`、`rhs_source_gauge_kernel` 和 `rhs_advection_kernel`；
- 仍运行 `rhs_constraints_kernel`，并让它接受 `co=-1`；
- 正常 RK4 的 `co=0/1` 路径保持不变。

约束 kernel 直接读取 state、物理 Christoffel/Ricci producer 输出和物质变量，
不读取被跳过的 RHS source/advection 结果，因此该依赖裁剪不改变数值定义。

#### P0-B 实测结果（2026-08-27）

| 版本 | `t=5` program mean | 结果 |
|---|---:|---|
| P0-A 最终版本 | `84.722587s`（单次复验） | PASS |
| P0-B constraint-only | `83.529860 +/- 0.283895s` | 3/3 PASS |

P0-B 相对 P0-A 的 program cost 降低约 `1.4%`。benchmark artifact 为
`gpu-benchmark-20260827T021603Z-66`，三次 program cost 分别为
`83.282611s`、`83.467078s`、`83.839890s`；轨迹 RMS 为 `0`，
约束 checker 的 level-0 最大值为 `Ham=0.025628463`、`Px=0.012645773`、
`Py=0.012757900`、`Pz=0.025120159`。

Nsys artifact `gpu-nsys-20260827T022226Z-65`（`t=1`）验证了调度裁剪：
P0-A 的 source/advection 各为 `420` 次，P0-B 各为 `392` 次，而约束 kernel
仍为 `126` 次。P0-B 的关键总 GPU 时间为 `rhs_advection=2.475s/392`
（正常 RHS 调用部分）、`rhs_evolution=1.699s/420`、`rhs_constraints=0.368s/126`；
profile checker 也为 PASS。

### P1-A：融合 Gamma 导数与 seed 的 RHS 数据流

本轮将 `rhs_gamma_derivatives_kernel` 和三个 Gamma seed kernel 合并为
`rhs_gamma_seed_fused_kernel`（[`bssn_rhs_gpu.cu`](/home/h3250106394/lab04/src/bssn_rhs_gpu.cu:565)）。
每个线程在寄存器中计算 `Lap`/`trK` 的三方向导数，随后直接生成
`Gamx_rhs`、`Gamy_rhs`、`Gamz_rhs`，删除 3 次 seed launch，以及 6 个中间 scratch
写入和读取。实现保留旧式 Gamma-y 系数 `Gamyzz * Rzz`，并验证未发生隐含
scratch 依赖。

P1-A 实测结果（2026-08-27）：

| 版本 | `t=5` program mean | 结果 |
|---|---:|---|
| P0-B | `83.529860 +/- 0.283895s` | 3/3 PASS |
| P1-A | `82.166688 +/- 0.084628s` | 3/3 PASS |

P1-A 相对 P0-B 降低 `1.363172s`（`1.63%`）。artifact 为
`gpu-benchmark-20260827T034330Z-64`；Nsys artifact 为 `gpu-nsys-20260827T035243Z-62`，t=1 中 fused kernel 运行 420 次、总计 `0.168677s`；旧版 derivative + 三个 seed kernel 合计约 `0.311601s`，该 producer 时间减少约 `45.8%`。三次 program cost 均通过 checker，trajectory RMS
均为 `0`，level-0 约束最大值为 `Ham=0.025628463`、`Px=0.012645773`、
`Py=0.012757900`、`Pz=0.025120159`。单次去除 scratch 发布的对照 artifact
`gpu-benchmark-20260827T034007Z-63` 同样 PASS，说明这些写入不是必需副作用。

该融合只减少 RHS producer 的 launch 和中间全局流量，没有引入空间 tile 或跨点
数据复用；因此它是低风险的 P1-A 子集，后续仍需按变量组 batching、
`d_fderivs_point` tile 和 Ricci consumer 顺序继续 profile。

不要一次性物化完整 18 个 Christoffel 分量；已有实验显示 DRAM 流量会抵消寄存器收益。

### P1-B：降低 AMR batch kernel 的 local-memory 流量

跨变量 batching 已经有效，但 [`prolongrestrict_cell_gpu.cu`](/home/h3250106394/lab04/src/prolongrestrict_cell_gpu.cu:313)
仍使用 `double tmp2[6][6]`、`double tmp1[6]`。NCU 已显示 batch prolong 有明显
local-memory 流量，说明目前只完成了减少 launch，未完成中间数组优化。

可尝试固定阶数的标量累加器、模板展开和 shared tile；同时单独优化
`restrict3_batch_kernel`，因为它仍依赖较通用的限制插值 helper。

### P1-C：围绕 buffer ownership 减少同步

当前同步已经比 baseline 好很多，但仍有 215 次 device sync 和 472 次 stream sync。
后续应：

- 用 event 表示 producer/consumer 依赖；
- 将多个 touched stream 汇合为 join event；
- pack、compute、unpack 围绕 buffer 生命周期延迟等待；
- descriptor 使用双缓冲，避免复用时提前同步；
- 保留跨 rank/MPI 真正需要的等待。

此前“全部同步直接替换成 event”的实验变慢，因此不能只追求 API 数量下降，必须证明
存在可重叠的独立工作。

### P1-D：使用持久 scratch，避免重复分配

当前 `t=1` 仍有约 2513 次 `cudaMalloc` 和 2505 次 `cudaFree`。通用 memory pool
实验反而变慢，建议改用按 block/level/用途划分的持久 scratch：拓扑或尺寸未变时复用，
按 stream 记录 owner，并用 event 延迟释放。active-index、descriptor、坐标和 shell
buffer 都适合优先持久化。

### P2：优化分析/约束插值

[`Constraint_Out`](/home/h3250106394/lab04/src/bssn_gpu_class.C:3009) 会反复分配
坐标、active index 和 shell buffer，并对 7 个约束变量分别 launch。可评估：

- 将 7 个变量加入 batch 维度；
- 持久化坐标、active index、shell buffer；
- 缓存 level/block 索引映射；
- 只同步 CPU reduction 真正需要的结果。

该方向对 `t=100` 可能比 `t=1` 更重要，应使用 `t=5` 或更长 profile 确认占比。

### P3：编译和参数整理

做一个去 RDC/同 TU forced-inline 的对照构建，并在确认 alias 安全后给只读输入增加
`const __restrict__`。巨大的 RHS kernel 参数列表可以整理为结构体以改善维护性，
但当前 kernel launch 仅 `0.23s`，这不是主攻方向。除非数值 checker 允许，否则不要
把 `fast-math` 当作默认优化。


### P2-A：单进程 GPU transfer 的 device-side staging

P2-A 针对 profile 中反复出现的 transfer staging 做了多轮 A/B，而不是只依据
`cudaMemcpy` 的 API 时间推断收益。当前 benchmark 是单 MPI rank，因此没有真正的
MPI peer；原路径仍会在每次 `gpu_transfer` 中重新 `cudaMalloc/cudaFree` 一个 packed
buffer，再执行 pack、host/device staging 和 unpack。

保留的实现位于 [`gpu_manager.cu`](/home/h3250106394/lab04/src/gpu_manager.cu:85)、
[`gpu_manager.h`](/home/h3250106394/lab04/src/gpu_manager.h:34) 和
[`Parallel_GPU.cpp`](/home/h3250106394/lab04/src/Parallel_GPU.cpp:294)：按进程维护可增长的
单个 device transfer buffer，单 rank 的 pack/unpack 复用该 buffer；容量变更前由
`gpu_data_packer` 的 stream synchronization 保证旧 buffer 不再被使用。多 rank 的
MPI/CUDA-aware 分支保持原有行为。

初始化阶段另外增加了 `ensure_on_gpu` 语义：只保证 GPU 副本存在，不把仍然有效的
CPU 副本错误地标记为 stale。这修正了 `move_to_gpu()` 初始化上传的 ownership 语义，
但对总时间的影响很小。

#### 实测结果（2026-08-27）

| 候选 | `t=5` program mean | 相对 P1-A | 结果 |
|---|---:|---:|---|
| P1-A | `82.166688 +/- 0.084628s` | - | 3/3 PASS |
| 持久 transfer buffer | `81.966115 +/- 0.0144s` | `-0.24%` | 3/3 PASS |
| buffer + validity-aware 初始化 | `82.130736 +/- 0.162994s` | `-0.04%` | 3/3 PASS |
| 清理后最终复验 | `82.211364 +/- 0.678122s` | `+0.05%` | 3/3 PASS，与 P1-A 持平 |
| 仅收窄同步到 touched streams | `82.833139 +/- 0.120439s` | `+0.81%` | 3/3 PASS，回退 |
| same-level direct copy + AMR packed | `84.510283 +/- 1.580207s` | `+2.85%` | 3/3 PASS，回退 |
| 完整 direct-device（含 AMR direct batch） | `85.414677 +/- 1.150887s` | `+3.95%` | 3/3 PASS，回退 |
| AMR-only direct、type1 保持 packed | `82.201285 +/- 0.089987s` | `+0.04%` | 3/3 PASS，回退 |

持久 buffer 的主要结果 artifact 为 `gpu-benchmark-20260827T042314Z-63`；初始化
validity 版本为 `gpu-benchmark-20260827T052304Z-65`；同步对照为
`gpu-benchmark-20260827T050249Z-65`；direct 实验分别为
`gpu-benchmark-20260827T061906Z-65` 和 `gpu-benchmark-20260827T063144Z-66`。
清理实验代码后，保留版本的短窗复验为 `gpu-benchmark-20260827T064353Z-65`，
`t=1` checker PASS；最终三次复验为 `gpu-benchmark-20260827T065604Z-64`，三次
program cost 为 `81.782679s`、`81.858237s`、`82.993176s`，checker 3/3 PASS。

#### 为什么 direct 路径没有带来预期收益

Nsys（持久 buffer / direct 对照：`gpu-nsys-20260827T042902Z-62`、
`gpu-nsys-20260827T062505Z-64`、AMR-only `gpu-nsys-20260827T064029Z-65`）显示：

- `cudaMemcpy` 的 host API self time 约 `6.3--6.5s`，但真正 GPU H2D/D2H 流量只有
  约 `70--240ms`；大部分 API 时间是等待此前 kernel 完成，不能按字节传输时间估算
  可节省的端到端时间。
- direct-all 将 kernel launch 从约 `4.3万` 降到 `2.9万`，但 AMR batch kernel
  仍占约 `0.82s`，同步边界仍有 `215` 次 `cudaDeviceSynchronize` 和约 `1.43s`
  `cudaStreamSynchronize`。少掉的 unpack/pack kernel 只占几十毫秒，无法覆盖
  direct 访问目标完整字段时的调度和内存访问代价。
- 当前 workload 的 transfer 列表主要是 type2/3 AMR；自定义 same-level copy kernel
  实际不是主路径。因此“把所有 segment 都 direct 化”是错误的放大方向。

结论：isolated 持久 buffer 实验曾出现约 `0.24%` 收益，但最终复验与 P1-A 在统计上
持平，不能宣称端到端加速。保留它是因为它消除了反复分配并为后续持久化调度提供
基础设施；validity-aware 初始化则是 ownership 语义修正。同步收窄和 direct-device
实验均已用 checker 和重复 benchmark 排除。P2-A 不能贡献目标所需的 `3.5x` 加速，
下一步应回到 P1-B（prolong/restrict local-memory）、RHS 空间复用/融合及更激进的
constraint/analysis 调度。

## 4. 已有优化审计

| 已有工作 | 判断 | 未完成部分 |
|---|---|---|
| RHS dataflow/fission | 有效 | P1-A 已完成 Gamma 导数/seed 融合并降 1.63%；尚无完整空间 tile reuse，旧 batch 方案不应直接恢复 |
| Advection common factors | 有效但不彻底 | KO 仍重复坐标/边界计算；helper/local array 未处理 |
| Ricci derivative fission | 有效 | 继续做 contraction/load order，避免完整 Christoffel 物化 |
| Stream sync reduction | 有效 | 仍有大量串行 stream wait；需要 buffer-level scheduling |
| AMR cross-variable batching | 有效 | prolong/restrict 的局部数组和通用 helper 仍是瓶颈 |
| 3-way prolong streams | 基本完成 | MIG 资源有限，继续加 stream 预期收益很小 |
| RK4 batch | 正确回滚 | 仅约 `0.22s/t=1`，不是目标路径 |
| Generic memory pool | 正确回滚 | 应改为按用途持久 scratch |
| Pinned/async staging | 尚未证明有效 | 只应按 batch 异步传输，避免每段单独 copy |

## 5. 建议的实验与验收流程

1. 固定当前版本，保存生产构建 `t=5` 三次基线和当前 Nsys/NCU 指标。
2. P0 路径已验证；下一轮优先做 P1-B prolong/restrict local-memory 实验，再回到 helper scalar/inline/local-array。
3. 每个候选先跑 `t=1` NCU，确认 local-memory、scoreboard 和 kernel wall time 的变化。
4. 通过后用生产构建跑 `t=5` 三次，要求所有 checker、轨迹 RMS 和约束输出一致。
5. 对同步、AMR 和 scratch 改动分别做端到端 A/B，不以 API 调用数作为唯一成功标准。
6. 只有当分阶段结果外推到 `t=100 <=370s` 后，才提交完整官方 `t=100` 测试。

当前最值得立即尝试的是：**P1-B prolong/restrict local-memory 消除 + RHS helper local-memory 实验**。
P0-A/P0-B/P1-A 已分别覆盖 local memory、约束重复计算和 Gamma producer launch；要继续接近
`370s`，需要把 AMR batch 与 RHS stencil 的实际 memory 等待降下来，而不是继续增加 stream。

## 6. 达标判断与激进路线

### 6.1 当前路线是否足够

不能把“肯定做不到”理解为程序本身无解，但可以明确判断：**只沿着 P0-A、P0-B、P1-A
这类局部 RHS 优化继续推进，达不到 `370s`**。当前 P1-A 的 `t=5` program cost 为
`82.166688s`，按阶段 4 的近似外推，`t=100` 约为 `1206s`，仍需要约 `3.26x`
端到端加速。P1-A 已经把目标 producer 的 GPU 时间减少约 `45.8%`，端到端却只下降
`1.63%`，说明这条路线的单个 kernel 收益很快会被调度、同步和其他阶段吞掉。

当前 `t=1` Nsys 还给出了更重要的结构性证据：CUDA API trace 的 self-time 中，
`cudaMemcpy`、`cudaDeviceSynchronize`、`cudaStreamSynchronize` 分别占约 `56.2%`、
`25.2%`、`12.7%`；同时有 `43,386` 次 kernel launch、`215` 次 device synchronize、
`472` 次 stream synchronize。这里的百分比是 API trace 时间占比，不等价于 GPU kernel
占用率，但足以说明“再融合一个小 kernel”不是主要矛盾。

### 6.2 P2-A：实测结论与保留范围

P2-A 已完成并经过多轮 A/B。isolated 的单 rank 持久化 device transfer buffer 曾测得
`81.966115s`（相对 P1-A 约 `-0.24%`），但清理后的最终三次复验为
`82.211364 +/- 0.678122s`，相对 P1-A 的 `82.166688 +/- 0.084628s` 没有统计显著收益。
初始化 `ensure_on_gpu` 仍然保留，因为它修正了 CPU/GPU ownership 语义。

direct-device、收窄同步和 AMR-only direct 都已经做过重复 benchmark。它们要么回退
（`82.83--85.41s`），要么与基线持平（`82.20s`）。Nsys 说明 `cudaMemcpy` 的 host
API 时间主要是等待 kernel 的阻塞时间，真实 H2D/D2H 传输只有几十到几百毫秒；因此
减少 staging API 或 launch 数并不会自动转化为端到端收益。当前 transfer 列表主要是
AMR type2/3，direct 写完整目标字段还会改变内存访问和调度形态。

结论是：P2-A 是低风险的基础设施整理和一次关键证伪，不是已证实的性能改进，更不是
达到 `370s` 的主路径。后续只应把它作为 P2-B/P2-C 的基础，继续保持可增长 buffer、
明确 ownership 和多 rank 原有 MPI 语义；不要再投入“所有 segment direct 化”这条
已被实测排除的路线。

### 6.3 P2-B：重写时间步调度，做跨变量 batching 和 CUDA Graph

[`bssn_step_gpu.C`](/home/h3250106394/lab04/src/bssn_step_gpu.C:130) 在 predictor 和三个
corrector 中都逐变量发射 RK4 和边界 kernel。可以把同一 block 的独立状态变量组织成
持久化的 device pointer table，一次 batch launch 完成一组 RK4 更新；同理评估 level-0
Sommerfeld 和 level>0 边界修正的跨变量 batch。这样优化的是 launch 参数准备和 host 调度，
不是改变 RK4 数学步骤。

在 batch 依赖稳定后，再按 level/stage 捕获 CUDA Graph，把固定的 enforce -> RHS -> RK4 ->
boundary -> lower-bound 序列变成 replay。ghost exchange、MPI 和会改变拓扑的 AMR 操作
留在 graph 外，或使用 graph update 节点更新指针/标量。Graph 不能解决错误的同步依赖，
但有机会消化当前数万次 launch 的 CPU 开销。

已有 RK4 batch 实验只有约 `0.22s/t=1` 收益，因此“只改 RK4 batch”不应作为达标方案；
它需要和跨变量 boundary batch、基于真实依赖的 event/graph 一起评估。P2-A 已经说明
不能先验假定 device-only 或窄同步会带来收益。主要风险是指针表更新成本、不同变量的
边界属性，以及 graph capture 对动态 BH/AMR 状态的限制。

### 6.4 P2-C：针对 dominant RHS 的空间 tile，而不是继续做小范围 fission

P1-A 的 Nsys 中，`rhs_advection_kernel` 为 `2.48s/t=1`，`rhs_evolution_kernel` 为
`1.69s/t=1`，二者远高于刚刚融合的 Gamma producer。`d_fderivs_point`、
`d_fdderivs_point` 和 lopsided/KO helper 会对半径为 2 的邻域反复 global load；P0-A
虽降低了 local-memory sector，但没有消除跨线程的邻域重复读取。

建议先只为 advection + KO 做半径 2 的 shared-memory tile 或 x-line rolling cache，
测量 register、shared-memory、occupancy 和 DRAM sector；成功后再把同样的 tile 机制移植
到 evolution 中实际占时最高的一组导数。不要直接把完整 BSSN RHS 合成一个巨型 kernel：
Christoffel/Ricci 中间值的生命周期会推高寄存器和 spill，已有实验也表明完整物化会被
DRAM 流量抵消。

这是高工作量但最可能提供 GPU 算力级收益的方向。必须先用 NCU 证明 excessive global
load/local-memory load 下降，再用 `t=5` checker 验收；shared memory 版本如果只提高
occupancy、却增加同步或降低实际 kernel wall time，应判定为失败。

### 6.5 P2-D：AMR 交换从“每段/每变量”改成持久化批调度

当前 Nsys 中 `prolong3_batch`、`restrict3_batch`、`global_interp` 分别约为
`0.81s`、`0.51s`、`0.49s/t=1`，且 `PatList_Interp_Points_GPU` 仍逐变量发射
`global_interp`，函数中还会分配、清零、同步和释放 shell/weight buffer。P1-B 应先解决
batch kernel 的 local array；更激进的版本则是按 level 汇总所有 patch/variable 的
descriptor，一次提交连续 batch，复用 shell/weight/active-index，并把同一 level 的
restrict、同步和 prolong 依赖编排成少量 event。

这个方向通常不单独提供 `3x`；P2-A 留下的持久 buffer/ownership 机制可以作为其资源
生命周期基础，但不应再为组合收益额外计入未经证实的 device-only 加速。

### 6.6 可行性组合与不应采用的“捷径”

P2-A 已经实测排除了此前对 device-only/persistent path 的 `1.3--1.8x` 乐观估算。
其余 dominant stencil tile、batched schedule/graph 和 AMR 持久化 batch 仍只有工程
假设，不能据此承诺组合后必然达到 `370s`。现在唯一严谨的判断是：目标仍然可能，
但必须由 P2-C 首先提供显著的 RHS kernel 级收益，再由 P2-B/P2-D 减少调度与 AMR 工作；
如果首个 tile 和跨变量 batch 原型都不能明显降低 `t=5`，就需要进一步重构按 level/block
批处理的数据布局和跨 RHS 的导数复用，而不是继续做 producer 级微优化。

不建议把以下手段作为主路线：降低浮点精度、减少 RK 阶段或 AMR subcycling、跳过必要
输出、修改网格/演化时间、让多个 MPI rank 争用同一 MIG。文档明确要求物理问题和数值
结果等价；TwoPuncture 初值阶段即使全部消除，也不足以填补当前约 `3.26x` 的缺口。

### 6.7 推荐实施顺序

1. 保留 P2-A 的单 rank persistent transfer buffer 和 validity-aware ownership 修正；
   不再继续推进已经回退的“收窄同步/全量 direct-device”路径。
2. 在此基础上做 per-block persistent scratch 和 descriptor，避免每个 pack/analysis
   调用重新分配，并把同步点按真实数据依赖重新归类。
3. 选择 `rhs_advection_kernel` 做第一个 shared tile 原型，分别跑 NCU、`t=1` checker
   和 `t=5` 三次 A/B。
4. 再做跨变量 RK4/boundary batch；若 host launch 仍是主要空洞，捕获每 level/stage
   CUDA Graph。
5. 最后把 AMR prolong/restrict/interpolation 纳入同一套持久化 batch 调度。只有短跑外推
   已低于 `370s` 后，才提交完整 `t=100` 官方计时。
