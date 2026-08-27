# GPU 优化到 370 秒的 Profile 报告

**日期：** 2026-08-27
**目标：** 官方 `t=100` 运行时间 `<=370s`  
**当前版本：** P2-C 赤道对称边界专用 compact tiled advection + P2-A 基础设施

## 1. 结论

P2-C 已经取得本项目迄今最大的单项收益。最终独立复验的 `t=5` program cost 为
`74.633372 +/- 0.093387s`，相对 P1-A 的 `82.166688 +/- 0.084628s` 下降
`9.17%`；`Total Evolve` 从 `53.366067 +/- 0.012849s` 降至
`45.080233 +/- 0.043878s`，下降 `15.53%`。三次 checker 全部 PASS，trajectory RMS
均为 `0`。更早的同配置三次复验为 `74.881219 +/- 1.340158s`，其中一次节点慢点
同时拉高 program 和 evolve；该组也 3/3 PASS，本文不删除或隐藏这组波动数据。

Nsys 显示 advection kernel family 的 `t=1` aggregate GPU 时间从 P1-A 的
`2.4808s` 降到 `0.8289s`，下降 `66.59%`，即 `2.993x` kernel-family 加速。
这明显超过“20% kernel 收益”，但不是整程序 `20%`：固定初值和后处理会稀释短窗
program cost，剩余 evolution、beta-gamma 和 AMR kernel 也形成 Amdahl 上限。

按 `P(T) = Program5 - Evolve5 + T/5 * Evolve5` 做仅供排优先级使用的线性模型，
当前 `t=100` 约为 `931s`，仍需约 `2.52x` 整体模型加速，或让演化部分再加速约
`2.65x`。因此 P2-C 证明了空间 tile 路线有效，但单独不足以达到 `370s`。下一步必须
把 compact stencil 机制迁移到 `rhs_evolution_kernel`/beta-gamma 导数，并并行处理
prolong/restrict 的 local array 和 AMR 持久化批调度。

## 2. Profile 方法与结果

profile 在 HPC A100 MIG（`1g.10gb`，SM80）上执行，正确性检查通过。

- 早期 Nsight Systems：job `171518`，`profile/gpu-nsys-20260826T155727Z-65`
- 早期 Nsight Compute：job `171582`，`profile/gpu-ncu-20260826T160405Z-64`
- P1-A Nsight Systems：`profile/p1a-nsys/gpu-nsys-20260827T035243Z-62`
- P2-C 首版 full-tile Nsys：`profile/gpu-nsys-20260827T075056Z-66`（`t=0..4`）
- P2-C 最终 hybrid NCU：job `177066`，`profile/gpu-ncu-20260827T083353Z-64`
- P2-C 最终 benchmark：job `177015`、`177089`，artifact 分别为
  `profile/gpu-benchmark-20260827T082733Z-64` 和
  `profile/gpu-benchmark-20260827T083647Z-63`
- 绝对时间以生产构建 benchmark 为准；Nsight 结果用于定位结构和比较同配置 kernel。

### 2.1 GPU kernel 热点

P2-C 首版 full-tile Nsys 覆盖 `t=0..4`。按 4 个演化时间单位归一化后，当前主要
kernel family 为：

| Kernel | GPU 时间/单位演化时间 | GPU kernel time share |
|---|---:|---:|
| `rhs_evolution_kernel` | `1.646s` | `18.3%` |
| `rhs_beta_gamma_prepare_kernel` | `0.847s` | `9.4%` |
| `rhs_advection_equatorial_compact_kernel` | `0.829s` | `9.2%` |
| `prolong3_batch_kernel` | `0.797s` | `8.9%` |
| `restrict3_batch_kernel` | `0.546s` | `6.1%` |

P2-C 前的 P1-A advection 为 `2.4808s/单位演化时间`，占 GPU kernel time
`22.9%`；P2-C 后它已不再是第一热点。瓶颈已转移到 evolution、beta-gamma 和 AMR，
继续只压 advection 的 Amdahl 上限很低。

### 2.2 CUDA API 与同步

- `cudaMemcpy`：3053 次，host API 时间 `7.88s`
- `cudaDeviceSynchronize`：215 次，`3.24s`
- `cudaStreamSynchronize`：472 次，`1.55s`
- `cudaLaunchKernel`：44926 次，只有 `0.23s`

`cudaMemcpy` 的 host 时间包含阻塞等待，不能和 GPU kernel 时间直接相加；它说明
host/device 工作仍有串行化。launch 数量本身已经不是首要瓶颈。

### 2.3 Advection 的 Nsight Compute 证据

重构前的 legacy advection 使用 `(8,8,4)` block，早期采样显示 66
registers/thread、实际 occupancy `32.5%`、`No Eligible=61.7%`，且 local-memory
sector 约占全部 L1TEX sector 的 `85%`。这证明 local array、helper 和跨线程重复
stencil load 是主因，而不是 compiler spill request 或单纯 occupancy 不足。

最终 compact kernel 的同类采样为：

- 单 launch `491.62us`，执行指令 `15.770M`；
- 78 registers/thread，static shared memory `15.73KiB/block`；
- 理论/实际 occupancy 为 `37.50%/35.28%`；
- `No Eligible=57.06%`，L1TEX scoreboard stall 约占 `36.3%`；
- L2 hit `79.40%`。

与相同 grid/block/时钟的首版 compact NCU 相比，最终 hybrid 从 `502.94us` 降到
`491.62us`（`-2.25%`），执行指令从 `16.441M` 降到 `15.770M`
（`-4.08%`），寄存器和 shared memory 不变。occupancy 不是接受依据；同配置 kernel
wall time、Nsys aggregate 时间和生产 benchmark 共同支持保留该版本。

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
实验均已用 checker 和重复 benchmark 排除。P2-A 不能贡献当前模型仍需的 `2.52x`
加速；P2-C 已验证 RHS 空间复用有效，下一步应迁移到 evolution/beta-gamma，并推进
P1-B/P2-D 的 prolong/restrict local-memory 与持久化批调度。

## 4. 已有优化审计

| 已有工作 | 判断 | 未完成部分 |
|---|---|---|
| RHS dataflow/fission | 有效 | P1-A 已完成 Gamma 导数/seed 融合并降 1.63%；旧巨型 batch 方案不应直接恢复 |
| Advection compact tile | 显著有效 | P2-C 将 advection family 降 `66.59%`，Evolve 降 `15.53%`；同样的空间复用尚未迁移到 evolution/beta-gamma |
| Ricci derivative fission | 有效 | 继续做 contraction/load order，避免完整 Christoffel 物化 |
| Stream sync reduction | 有效 | 仍有大量串行 stream wait；需要 buffer-level scheduling |
| AMR cross-variable batching | 有效 | prolong/restrict 的局部数组和通用 helper 仍是瓶颈 |
| 3-way prolong streams | 基本完成 | MIG 资源有限，继续加 stream 预期收益很小 |
| RK4 batch | 正确回滚 | 仅约 `0.22s/t=1`，不是目标路径 |
| Generic memory pool | 正确回滚 | 应改为按用途持久 scratch |
| Pinned/async staging | 尚未证明有效 | 只应按 batch 异步传输，避免每段单独 copy |

## 5. 建议的实验与验收流程

1. 以最终 P2-C 的 `t=5` 两组三次结果、Nsys 和 NCU 作为新基线。
2. 下一轮先做 `rhs_evolution_kernel` 的变量组 compact tile；每个候选用 NCU 验证
   wall time、register/shared-memory、scoreboard 和指令数。
3. 候选通过短跑后用生产构建跑 `t=5` 三次，要求 checker、trajectory RMS 和约束输出一致。
4. 同时推进 P1-B/P2-D：消除 prolong/restrict local array，并复用 descriptor、
   shell/weight 和 active-index buffer。
5. 调度、graph、同步和 scratch 改动分别做端到端 A/B，不以 API 或 launch 数为唯一指标。
6. 只有短窗分阶段模型进入 `t=100 <=370s` 区间后，才提交完整官方 `t=100` 测试。

当前最值得立即尝试的是：**把已经在 advection 上验证的 compact tile 迁移到
`rhs_evolution_kernel` 的高占时导数组，同时并行处理 prolong/restrict local array**。
继续增加 stream 或只融合小 producer 无法填补剩余约 `2.52x` 的模型时间差距。

## 6. 达标判断与激进路线

### 6.1 当前路线是否足够

不能把“当前仍高于 `370s`”理解为程序无解，但 P2-C 的结果已经给出更严格的边界。
最终 `t=5` program/evolve 为 `74.633372s/45.080233s`；扣除约 `29.55s` 的短窗
固定部分再线性外推，`t=100` 约为 `931s`。该模型只用于排序，完整运行中的 regrid、
移动 patch、分析和晚期物理阶段都可能改变单位成本，不能把 `931s` 当成正式成绩。

即使如此，达到 `370s` 仍要求整体模型再加速约 `2.52x`，或将演化单位成本从
约 `9.016s` 降至约 `3.404s`，即演化部分再加速约 `2.65x`。P2-C 已把原第一热点
advection 的 aggregate GPU 时间削减 `66.59%`，而它现在只占 GPU kernel time
`9.2%`；继续只优化该 kernel 不可能补齐差距。必须处理新的第一热点 evolution、
beta-gamma，以及合计约 `15%` 的 prolong/restrict。

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
达到 `370s` 的主路径。后续只应把它作为 P2-B/P2-D 的资源生命周期基础，继续保持
可增长 buffer、明确 ownership 和多 rank 原有 MPI 语义；不要再投入“所有 segment
direct 化”这条已被实测排除的路线。

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

P2-C 已实现并保留。新 kernel 位于
[`advection_compact_gpu.cuh`](/home/h3250106394/lab04/src/advection_compact_gpu.cuh:1)，
调度入口位于
[`bssn_rhs_gpu.cu`](/home/h3250106394/lab04/src/bssn_rhs_gpu.cu:1616)。实现仅在
`symmetry == 1` 时启用，其它 symmetry mode 继续使用 legacy kernel：

- block 为 `(8,8,4)`；每个字段协作加载半径 3 的 `14x14x10` full tile，
  static shared memory 为 `15.73KiB`；
- 24 个演化字段顺序复用同一个 shared buffer，在 tile 内同时完成 lopsided advection
  与 KO，保留对已有 RHS 的累加语义；
- interior block 使用无边界检查快路径；边界 block 精确保留 legacy 的
  `q < 0 -> -q - 1` 反射、x/y/z parity、赤道 `kmin=-3` 和降阶 stencil；
- cooperative load 保持线性均衡和 coalescing，预计算 center index 后 stencil 访问变成
  常量 shared-memory offset；launcher 自行按 compact block 计算 grid。

正式结果如下：

| 版本 | `t=5` Program | `Total Evolve` | 正确性 |
|---|---:|---:|---|
| P1-A | `82.166688 +/- 0.084628s` | `53.366067 +/- 0.012849s` | 3/3 PASS |
| 首版 full-tile 性能原型 | `74.495520 +/- 0.213883s` | `45.343833 +/- 0.062073s` | 3/3 PASS；随后补全三轴 parity |
| 最终 hybrid，第一组 | `74.881219 +/- 1.340158s` | `45.523033 +/- 1.146557s` | 3/3 PASS；含一次整轮慢点 |
| 最终 hybrid，独立复验 | `74.633372 +/- 0.093387s` | `45.080233 +/- 0.043878s` | 3/3 PASS |

最终独立复验相对 P1-A 的 Program/Evolve 分别下降 `9.17%/15.53%`。两组最终
corrected-run 共 6 次 checker 全部 PASS，trajectory RMS 均为 `0`；第一组慢点未从
统计中剔除。Nsys 中 P1-A legacy advection 为 `2.480760s/392 launches/t=1`；
首版 compact 在 4 个演化单位中为 `3.315689s/1568 launches`，归一化后
`0.828922s/t=1`，即下降 `66.59%`、加速 `2.993x`。这与 Evolve 的实测收益符合
Amdahl 预期，但不能写成整程序 `2.993x`。

本轮没有在第一个有效版本上停止，还做了以下受控实验：

| 候选 | 关键观测 | 决策 |
|---|---|---|
| 两字段 full-tile group | barrier 减半，但 `t=1` step 从约 `8.57--8.63s` 增至 `8.85s` | 回退 |
| 四字段 cross tile | load 数减少，但 142 registers、`32.82KiB` shared、实际 occupancy `12.43%`，单 launch `870.53us` | 回退 |
| nested strided loader | 62 registers、occupancy `43.62%`，但单 launch `518.78us` | 回退 |
| linear loader + center-index hybrid | 78 registers、`15.73KiB` shared、occupancy `35.28%`，单 launch `491.62us` | 保留 |

cross tile 的 CPU 穷举映射和所有短窗 checker 都通过；它失败是资源压力和 latency hiding，
不是公式错误。这个反例也说明寄存器更少或 occupancy 更高都不是充分条件，接受标准必须是
kernel wall time、aggregate profile、生产 benchmark 和 checker 的组合。

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

P2-C 已经把“dominant stencil tile 是否有效”从工程假设变成肯定答案，但也同时暴露了
Amdahl 上限：advection 降 `66.59%` 后，短窗 Program 只降 `9.17%`、Evolve 只降
`15.53%`，剩余模型时间仍需约 `2.52x`。因此不能承诺把 P2-B/P2-D/P2-C 的乐观倍数
直接相乘后必然达到 `370s`。

可行组合必须是：先将已验证的 compact tile 迁移到 evolution 和 beta-gamma 的导数组，
再消除 prolong/restrict local array 并做 AMR 持久化 batch，最后才评估 boundary batch
和 CUDA Graph。若这三类 dominant workload 的 `t=5` 累计结果仍不能进入目标区间，
就需要进一步重构按 level/block 的数据布局和跨 RHS 导数复用，而不是继续做小 producer
融合。

不建议把以下手段作为主路线：降低浮点精度、减少 RK 阶段或 AMR subcycling、跳过必要
输出、修改网格/演化时间、让多个 MPI rank 争用同一 MIG。文档要求物理问题和数值结果
等价；TwoPuncture 初值阶段即使全部消除，也不足以填补当前约 `2.52x` 的模型缺口。

### 6.7 推荐实施顺序

1. 保留最终 P2-C linear-loader/center-index hybrid 和 P2-A ownership 修正；不再恢复
   group/cross/strided loader 或全量 direct-device。
2. 对 `rhs_evolution_kernel` 做变量组 compact tile，优先复用同一组半径 2/3 导数；
   不把完整 BSSN RHS 合成一个寄存器生命周期不可控的巨型 kernel。
3. 对 `rhs_beta_gamma_prepare_kernel` 做同样的导数 tile/dataflow 审计，评估能否与
   evolution 的部分 producer 共享中间量而不物化完整 Christoffel。
4. 并行推进 prolong/restrict 固定阶数标量化和 P2-D 持久 descriptor/buffer batch。
5. dominant kernel 降下来后，再做跨变量 boundary batch；只有 Nsys 仍显示 host launch
   空洞时才捕获 per-level/stage CUDA Graph。
6. 每个阶段保持 NCU、Nsys、`t=5 x3` 和 checker 四层验收；短跑模型进入
   `t=100 <=370s` 区间后，再提交完整官方 `t=100`。
