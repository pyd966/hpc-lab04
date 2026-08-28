# P5 阶段报告：lopsided advection 双字段批量化实验

日期：2026-08-24  
阶段：P5，`lopsided` 公共 beta 系数与双字段 SIMD 批量化  
结论：候选路径正确，但端到端无收益，默认关闭

## 1. 目标与基线

当前 `compute_rhs_bssn` 中有多次独立的 `lopsided` 调用。每次调用都使用同一组
`betax/betay/betaz`，但会重新计算六个 beta 的正负系数，并分别遍历网格。这个阶段
尝试把相邻的 metric 和 Aij 字段两两合并，在同一个 SIMD 网格循环中共享 beta 系数。

生产基线保持：单 MPI 进程、30 个 OpenMP worker、`OMP_PLACES=cores`、
`OMP_PROC_BIND=close`、静态/移动层 `24/30` 个 block、`dynamic,1` 调度、P2 direct
same-level Sync、`-O3 -g -fno-omit-frame-pointer`，测量 `t=0..4`。

## 2. 实现和第一次失败

新增 CMake 选项 `AMSS_ENABLE_LOPSIDEDIFF_BATCH`，默认 `OFF`。开启时，metric 的
六次调用变为三个 `lopsided2`，Aij 的六次调用也变为三个 `lopsided2`。新内核：

1. 两个字段各自调用 `symmetry_bd`，产生带对称 ghost 的 `fh1/fh2`；
2. 内部区域用 `!$omp simd`，对每个网格点只加载一次 beta 正负系数，然后依次更新
   两个字段；
3. 边界 shell 仍使用原有分支逻辑，保证低阶边界模板和对称规则不变；
4. 内部三个方向按原实现逐次累加，避免改变浮点加法顺序。

最初把 `shell_only` 设计成 optional 参数。由于 `lopsided` 是没有显式 Fortran
interface 的外部过程，这会改变调用方和被调用方对 optional presence 标志的 ABI
约定。HPC 上 ON 版本在第 3 个时间步开始出现 NaN。这个问题不是数学公式本身，
而是接口不安全。随后将参数改为所有调用都显式传入的必传 `logical`，并把批量路径
的边界调用固定为 `.true.`；修复后完整演化不再出现 NaN。

## 3. 第一轮 A/B：先验证正确性和收益

作业 159743，运行顺序 `off, on, on, off`。四次运行均通过 course check，ON
输出与 OFF 逐文件逐位一致。

| 版本 | Evolve 平均 (s) | Total 平均 (s) | 平均活跃 CPU | IPC | L1D miss | dTLB miss |
|---|---:|---:|---:|---:|---:|---:|
| OFF | 29.9175 | 32.2257 | 23.320 | 1.465 | 4.14% | 3.63% |
| ON | 30.2563 | 32.5752 | 23.402 | 1.450 | 4.21% | 3.58% |

ON 比 OFF 慢约 **1.13%**。初步 profile 显示：`lopsided2` 自身约 2.40%，其内部
调用的两个 shell-only `lopsided` 又占约 2.19%；同时 `symmetry_bd` 被每个字段
重复执行。也就是说，省下的 beta 系数计算被 ghost 准备和边界函数开销抵消了。

## 4. 第二轮：消除重复 ghost 准备

根据上述 profile，把 `src/lopsidediff.f90` 拆成两层：

- 兼容 wrapper `lopsided` 负责准备一次 `fh`；
- `lopsided_core` 接收已准备的 ghost，并负责内部 SIMD 或边界 shell。

`lopsided2` 先为两个字段各准备一次 ghost，再直接调用两个 `lopsided_core`，不再
重复调用 `symmetry_bd`。这一步只改变调用结构，不改变模板和边界条件。

第二轮作业 159783 的结果：

| 版本 | Evolve 平均 (s) | Total 平均 (s) | 平均活跃 CPU | IPC | L1D miss | LLC miss |
|---|---:|---:|---:|---:|---:|---:|
| OFF | 29.6986 | 34.0187 | 22.058 | 1.48 | 4.11% | 49.3% |
| ON | 30.1641 | 34.7412 | 21.805 | 1.47 | 4.12% | 49.4% |

ON 仍慢约 **1.57%**，但四次运行全部 `FINAL: PASS`。这说明重复 ghost 是一个真实
的额外开销，却不是唯一瓶颈。

## 5. 第三轮：降低 SIMD 寄存器压力

批量循环最初同时保存两个字段的六个临时量
`ax1/ay1/az1/ax2/ay2/az2`。这可能让 AArch64 128-bit SIMD 循环产生 spill，
因此又测试了低寄存器版本：只保留 `ax/ay/az`，先完成字段 1，再完成字段 2；beta
正负系数仍在两个字段之间共享。

第三轮作业 159796：

| 版本 | Evolve 平均 (s) | Total 平均 (s) | 平均活跃 CPU | IPC | L1D miss | LLC miss |
|---|---:|---:|---:|---:|---:|---:|
| OFF | 29.8704 | 32.2712 | 23.274 | 1.475 | 4.09% | 49.3% |
| ON | 30.2353 | 32.6525 | 23.226 | 1.460 | 4.12% | 49.2% |

ON 仍慢约 **1.22%**，两次 ON 均 bitwise/course PASS。编译器仍报告批量内部循环
使用 16-byte vectors，因此不是“没有向量化”导致的失败。

## 6. 最终 profile 的证据

最终候选 profile：`profile/abe-20260824T212852Z-14`，作业 159803，包含
`perf stat` 和 `perf record`，丢样本数为 0。

候选 flat profile：

| 函数 | cycles |
|---|---:|
| `compute_rhs_bssn_` | 49.62% |
| `__memcpy_sve` | 8.83% |
| `__memset_sve_zva64` | 6.70% |
| `lopsided_core_` | 6.64% |
| `lopsided2_` | 2.94% |
| `prolong3_` | 4.47% |
| `fdderivs_` | 4.35% |
| `fderivs_` | 2.35% |

因此批量相关路径合计约 `6.64 + 2.94 = 9.58%`。P4 基线中全部
`lopsided_` 为 8.34%。批量化并没有降低 lopsided 总成本，反而增加了约 1.2 个
百分点。候选硬件计数为 IPC 1.46、L1D miss 4.12%、LLC miss 48.97%、dTLB
miss 3.58%；没有异常 branch miss 或线程迁移。调用图还显示批量函数下仍有
`lopsided_core` 的 shell 工作，说明共享 beta 系数不能消除两套字段的主要 stencil
访存。

## 7. 结论与保留状态

这个方向经过三轮实验后拒绝作为生产优化：

- 数值结果和边界覆盖已经正确；
- SIMD 指令确实生成；
- 消除重复 ghost、降低寄存器压力都无法使端到端变快；
- 当前 `lopsided` 更接近内存/缓存受限的 stencil，少算几次 `max/min` 的收益太小，
  不足以支付双字段同时存活带来的缓存和线程任务吞吐损失。

代码、CMake 选项和 sweep/profile 脚本保留为可复现实验，但
`AMSS_ENABLE_LOPSIDEDIFF_BATCH` 继续默认关闭。后续不再扩大这个 batch（例如一次
合并 6 或 12 个字段），因为这会进一步增加局部数组和缓存压力。更有希望的方向是
减少 RHS 其他公共 ghost/memcpy/memset，或在 block 内做真正的数据布局/分块优化，
而不是继续增加同时处理的字段数。

## 8. 可复现实验

- A/B 脚本：[`hpc_abe_lopsided_batch_sweep.sh`](../hpc_abe_lopsided_batch_sweep.sh)
- 候选 profile 脚本：[`hpc_abe_profile.sh`](../hpc_abe_profile.sh)
- 初轮 profile：`profile/abe-20260824T210710Z-14`
- 最终 profile：`profile/abe-20260824T212852Z-14`

