# P4：OpenMP 线程与 block 任务几何报告

日期：2026-08-24
代码：`64f7ce0`（P2 direct Sync 已开启；P3 direct AMR 默认关闭）

## 1. 本轮目标

前几轮已经把主要 OpenMP 区域移到 block 层，并使用 `dynamic,1` 调度。本轮专门
验证两个容易混淆的因素：

1. 把 OpenMP worker 从 30 扩到 60 是否能利用分配到的 60 个逻辑 CPU；
2. 在 30 个 worker 下增加 block 数，是否能用更多任务填补 block 长尾。

所有运行均使用同一个 `t=0..4` 输入、`OMP_PLACES=cores`、
`OMP_PROC_BIND=close`、`-O3 -g -fno-omit-frame-pointer`，并开启 P2 direct
same-level Sync。每个候选按交错顺序重复，所有输出都逐文件比较并通过 course check。

## 2. 节点约束

HPC 作业提供 60 个 processing units，但节点拓扑显示：

- 2 个 SMT thread/core；本作业 cpuset 中实际是 30 个物理核、60 个逻辑 CPU；
- 当前 cpuset 为一个 NUMA node（本轮 profile 为 node 3 的 192--251）；
- TaiShan-v120，最高 2.9 GHz，支持 ASIMD/NEON、SVE、SVE2 相关扩展和矩阵/i8mm
  扩展；本轮没有改变数值编译语义，也没有使用 `-Ofast`。

因此 60-thread 候选是“每个物理核同时放两个 OpenMP worker”的 SMT 实验，不是
增加了 60 个独立计算核心。

## 3. 实验 A：30 与 60 个 OpenMP worker

脚本：`hpc_abe_thread_geometry_sweep.sh`，作业 159581，顺序
`30t, 60t, 60t, 30t`。30-thread 候选使用静态/移动 `24/30` 个线程和
`24/30` block；60-thread 候选使用 `48/60` 个线程和 `48/60` block。

| 候选 | Evolve 平均 (s) | Total 平均 (s) | 平均活跃逻辑 CPU | IPC | dTLB miss |
|---|---:|---:|---:|---:|---:|
| 30 worker，24/30 block | 29.9830 | 32.3612 | 23.229 / 30 | 1.47 | 3.62% |
| 60 worker，48/60 block | 40.9949 | 43.8290 | 42.577 / 60 | 0.76 | 10.46% |

60 worker 慢 36.7%。它虽然增加了活跃逻辑 CPU 数，但 IPC 几乎减半，dTLB miss
约为 30-worker 的 2.9 倍。block 也被切得更小，边界、ghost、Sync 和 OpenMP
阶段等待在每个 block 上重复，SMT 线程还竞争同一物理核的执行资源。因此这不是
绑核失败：`perf stat` 没有 CPU migration，`OMP_PLACES=cores` 正常生效；是 SMT
和过细任务粒度共同造成的吞吐下降。

## 4. 实验 B：30 worker 下增加 block 目标

脚本：`hpc_abe_block_target_sweep.sh`，作业 159596，静态/移动 block 目标分别
为 `24/30`、`60/60`、`90/90`，线程始终为 `24/30`。结果：

| 候选 | Evolve 平均 (s) | Total 平均 (s) | 平均活跃 CPU | IPC | LLC miss |
|---|---:|---:|---:|---:|---:|
| 24/30 block | 29.8992 | 34.1633 | 21.982 / 30 | 1.48 | 49.45% |
| 60/60 block | 31.8635 | 36.6625 | 20.462 / 30 | 1.82 | 36.42% |
| 90/90 block | 35.5108 | 40.3474 | 18.576 / 30 | 1.97 | 30.89% |

更多 block 的 IPC/LLC 指标表面上变好，但端到端变慢：计算块变小后，
`prolong/restrict`、边界处理、Sync 和调度调用次数增加，且每个阶段更容易出现
短 block 尾部。硬件计数的改善不能抵消额外的非计算工作。

## 5. 实验 C：只增加移动层 block

为了排除静态层小任务对结果的干扰，脚本 `hpc_abe_moving_block_sweep.sh`（作业
159617）固定静态层为 24 block/24 threads，只比较移动层 30 和 60 block：

| 候选 | Evolve 平均 (s) | Total 平均 (s) | 平均活跃 CPU |
|---|---:|---:|---:|
| 静态 24、移动 30 | 29.6122 | 33.9409 | 22.089 / 30 |
| 静态 24、移动 60 | 31.1219 | 35.7639 | 20.804 / 30 |

只增加移动层 block 仍慢 5.1%，说明问题确实位于 block 过细和阶段开销，而不是
静态层的少量工作。三组实验所有二进制输出均 bitwise 一致，course check 均为
`PASS`。

## 6. P4 最终 profile

profile 目录：`profile/abe-20260824T202835Z-14`，作业 159629。配置为
30 worker、静态/移动 24/30 block 和 threads、`dynamic,1`、P2 direct Sync，
`-O3 -g -fno-omit-frame-pointer`；`perf record` 丢样本数为 0。

### 时间和硬件计数

- `perf stat` Evolve：`29.878 s`；Total：`32.277 s`；record pass Evolve：
  `30.385 s`；
- task-clock 对应平均 `23.268 / 30` 个逻辑 CPU；
- IPC `1.47`；branch miss `0.50%`；L1D miss `4.15%`；LLC load miss
  `49.04%`；dTLB miss `3.61%`；CPU migration 和 context switch 均为 0；
- 这些指标说明主要问题是访存/数据复用和阶段性空转，不是异常分支预测或线程
  迁移。

### 函数与调用路径

无子调用的 flat profile：

| 函数/路径 | cycles |
|---|---:|
| `compute_rhs_bssn_` | 50.50% |
| `__memcpy_sve` | 8.92% |
| `lopsided_` | 8.34% |
| `__memset_sve_zva64` | 7.04% |
| `prolong3_` | 4.45% |
| `fdderivs_` | 4.11% |
| `kodis_` | 2.34% |
| `fderivs_` | 2.31% |
| `rungekutta4_rout_` | 1.86% |
| `restrict3_` | 1.30% |

调用图显示：

- `Step` 的主 block worker 占约 57.21%，其中 `compute_rhs_bssn_` 占其 54.04%；
- `compute_rhs_bssn_` 内部的 `lopsided_`、`fdderivs_`、`fderivs_` 和 `kodis_`
  是主要导数/耗散路径；
- `omp_local_transfer` 约 8.01%，主要是 `prolong3_` 5.09% 和 `restrict3_`
  2.10%；
- P2 `omp_execute_cached_sync` 约 6.15%，其中 `copy_`/`memcpy` 约 4.96%；
- `libgomp` 约 1.97%，没有变成新的主要热点。

带符号的源代码热点集中在 `bssn_rhs.f90:618/657/696`、
`lopsidediff.f90:112/330`、`diff_new.f90:634` 和
`prolongrestrict_cell.f90:2136/2143`。这给下一阶段提供了可直接对应的代码行。

### 层级利用率

诊断字段 `utilization`（有效 block 工作 / team capacity）为：

| level | blocks | utilization | balance |
|---:|---:|---:|---:|
| 0 | 9 | 0.30 | 0.30 |
| 1--4 | 24 | 0.79--0.80 | 0.79--0.81 |
| 5 | 30 | 0.78 | 0.80 |
| 6 | 30 | 0.84 | 0.84 |
| 7 | 30 | 0.85 | 0.85 |
| 8 | 30 | 0.86 | 0.86 |

移动层的利用率已经明显高于静态层，剩余全程平均 CPU 下降来自 level 0 的 9 个
block、静态层使用 24 threads、AMR level 间的顺序依赖和每个 RK/Sync 阶段的
barrier。继续把 block 数加到 60/90 不能消除这些依赖，反而增加每个阶段的固定成本。

## 7. 结论和后续方向

P4 不保留新的代码路径，只保留已经验证的 `dynamic,1`、30 worker、静态/移动
`24/30` block/threads。60 OMP worker、60/90 block 以及只增加移动层 block 的
方案均应淘汰；新增 sweep 脚本保留在仓库，方便复现实验。

当前最有价值的优化顺序是：

1. 继续优化 `compute_rhs_bssn_` 内部的数组复用、临时数组写入和 SIMD；
2. 针对 `lopsided_`、`fdderivs_`、`fderivs_` 中的清零/边界复制，减少
   `memset/memcpy` 和重复 ghost-zone 触碰；
3. 再研究 RHS 内部的安全 phase 融合或 block 内任务拆分，但必须以数据依赖和
   数值一致性为前提；
4. 暂不继续尝试 SMT、盲目增加 block、`guided` 调度或 `-Ofast`。这些方向在本轮
   已有明确的负收益或风险证据。
