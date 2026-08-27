# GPU 优化到 370 秒的 Profile 报告

**日期：** 2026-08-27
**目标：** 官方 `t=100` 运行时间 `<=370s`  
**当前版本：** P0-B 实验实现（本次提交）

## 1. 结论

当前主线还没有接近 `370s`。阶段 4 基线的 `t=5` 三次结果为
`89.150199 +/- 0.301396s`；P0-A 为 `84.722587s`，P0-B 当前为
`83.529860 +/- 0.283895s`。按当前演化阶段近似线性外推，`t=100` 约为
`1226s`；目标要求演化部分从约 `11.97s/单位时间` 降到约 `3.4s/单位时间`，
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

### P1-A：继续改进 RHS 的空间数据流

当前已经将 RHS 拆为多个 kernel，但主要仍是同一网格上逐变量计算，并没有完整的
spatial tile/shared-memory reuse。历史 batch kernel 实验不能直接恢复，应重新评估：

- metric/shift Hessian 按变量组 batch；
- 对 `d_fderivs_point` 使用空间 tile；
- 重排 geometry、Gamma derivative、Ricci consumer 的 producer/consumer 顺序；
- 只物化约束和后续 kernel 实际使用的中间量。

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

## 4. 已有优化审计

| 已有工作 | 判断 | 未完成部分 |
|---|---|---|
| RHS dataflow/fission | 有效 | 尚无完整空间 tile reuse；旧 batch 方案不应直接恢复 |
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
2. 先做 constraint-only RHS，再做 helper scalar/inline/local-array 实验。
3. 每个候选先跑 `t=1` NCU，确认 local-memory、scoreboard 和 kernel wall time 的变化。
4. 通过后用生产构建跑 `t=5` 三次，要求所有 checker、轨迹 RMS 和约束输出一致。
5. 对同步、AMR 和 scratch 改动分别做端到端 A/B，不以 API 调用数作为唯一成功标准。
6. 只有当分阶段结果外推到 `t=100 <=370s` 后，才提交完整官方 `t=100` 测试。

当前最值得立即尝试的组合是：**constraint-only RHS + RHS helper local-memory 消除**。
这两项分别减少重复计算和单 kernel memory 等待，成功概率和潜在收益都高于继续调整
RK4、增加 stream 或恢复已回滚的通用 memory pool。
