# ABE OpenMP 调度重设计报告

日期：2026-08-24
代码基线：`46d59d7`（RHS array-pass 优化之后）

## 1. 结论

本轮只处理 ABE 的 OpenMP 调度和任务划分，没有修改 BSSN 方程、差分公式或
编译器浮点语义。最终保留的策略是：

- `Step` 的四类 block 循环使用 `schedule(runtime)`，由 `OMP_SCHEDULE` 选择调度；
- 正式 CPU 入口默认 `OMP_SCHEDULE=dynamic,1`；
- 30 物理核测试节点使用静态层 `24 threads`、移动层 `30 threads`；
- 当前已验证的 block 目标为 `60`。评测节点有 60 个物理核时仍使用 60，避免
  未验证的 120-block 分解改变数值 block 边界；
- OpenMP 绑核保持 `OMP_PLACES=cores`、`OMP_PROC_BIND=close`。

严格使用 24/30 线程的同节点 sweep 中，`dynamic,1 + 60 blocks` 是最稳定的候选：
移动 level 7/8 的 block balance 从约 `0.85/0.89` 提高到 `0.92/0.91`，Evolve
从 `29.57 s` 降到 `28.66 s`，约 **3.05%**。30-block 下换成 dynamic/guided
几乎没有收益；90 blocks 已开始回退。所有候选的关键输出都逐文件一致。

这不是把所有 OpenMP 区域改成动态队列，也不是删除算法必需的 barrier。只对
独立 block 循环增加可调度任务，并保留每个 RHS、Sync、swap 之间的隐式 barrier。

## 2. 我先看了什么

`bssn_class::Step(lev, YN)` 先建立本 rank 的 `vector<Patch*, Block*>`，然后顺序执行：

1. predictor：每个 block 做边界约束、`compute_rhs_bssn`、RK4 更新；
2. `Parallel::Sync`；
3. 三次 corrector，每次都是同样的 RHS/RK block 循环和 Sync；
4. 中间时间层 swap 和最终 State/OldState swap。

不同 AMR level 仍然必须由 `RecursiveStep` 按层级依赖顺序推进，不能把所有 level
扔进一个全局任务队列。`Sync` 前后的 barrier 也不能删：前一阶段必须先完成所有
block 的读写，下一阶段才能读取 ghost/buffer 数据。

基线 OpenMP 诊断（`profile/abe-20260824T063606Z-14`）显示：

| level | 实际 block | phase utilization | balance |
|---:|---:|---:|---:|
| 5 | 30 | 0.842 | 0.846 |
| 6 | 30 | 0.851 | 0.854 |
| 7 | 30 | 0.851 | 0.853 |
| 8 | 30 | 0.884 | 0.886 |

level 7/8 是主要计算量，存在约 11%--15% 的 block 尾部。level 0 只有 9 个 block，
但总时间很短，不值得为它牺牲移动层并行度。

## 3. 尝试过的方案

### 3.1 静态层减少到 9 个线程

这是最早的 sanity check：静态层从 24 threads 降到 9，移动层保持 30。短程
Evolve 从基线约 `29.78 s` 变为 `31.21 s`，回退约 4.8%。原因是 level 1--4
虽然 block 少，但每层仍有大量 RK phase；节省不了足够的等待，反而直接损失计算
并行度。该方案淘汰。

### 3.2 只换 schedule，保持 30 blocks

在同一份二进制中比较 `static`、`static,1`、`dynamic,1`、`guided,1`。30 block
与 30 worker 时每个 worker 基本只有一个大任务，调度器没有第二个任务可以在尾部
重新分配，因此四种策略差异小于运行噪声。这验证了“仅换 schedule”不够。

### 3.3 增加 block 数，给队列更多任务

在固定线程数下测试 60 和 90 block，并比较 static/dynamic/guided。严格 24/30
线程 sweep 的 Evolve 结果如下（同一节点、同一输入、每个 case 顺序运行）：

| case | Evolve (s) | level 7 balance | level 8 balance |
|---|---:|---:|---:|
| static, 24/30 blocks | 29.568 | 0.851 | 0.889 |
| dynamic,1, 24/30 blocks | 29.587 | 0.866 | 0.887 |
| guided,1, 24/30 blocks | 29.456 | 0.872 | 0.882 |
| static, 60/60 blocks | 29.896 | 0.856 | 0.885 |
| **dynamic,1, 60/60 blocks** | **28.665** | **0.925** | **0.915** |
| guided,1, 60/60 blocks | 29.282 | 0.869 | 0.894 |
| dynamic,1, 90/90 blocks | 29.679 | 0.941 | 0.882 |

这里 60/60 表示 block 目标，不表示把静态层线程改成 60；所有 case 都是静态层
24 threads、移动层 30 threads。90 block 虽然 level 7 的 balance 更高，但 level
6/8 和 Sync/transfer 的任务量增加，整体回退。

### 3.4 跨整个 RK4 的持久 OpenMP team

我实现并实际测试过“一个 team 覆盖 predictor、三次 corrector、swap 和 Sync”的
版本。它需要额外的 `single`、`barrier`、团队内 Sync pack/unpack 协作。数值输出
逐文件一致，但在此前独立实验中 Evolve 稳定回退约 10%--12%，平均 CPU 几乎不变。
原因是 RK/Sync 的阶段依赖仍然存在，持久 team 不能填补 level 依赖和 block 尾部，
反而让整支 team 更久地等待。它没有进入生产入口。

## 4. 最终 profile

最终严格 profile：`profile/abe-20260824T081218Z-14`，配置为 `-O3 -g
-fno-omit-frame-pointer`、24/30 threads、60/60 block target、`dynamic,1`，
`perf record` 无丢样本。

| 指标 | perf stat |
|---|---:|
| Evolve | 28.9607 s |
| ABE total | 33.7791 s |
| task-clock | 772.05 s |
| 平均 CPU | 22.65 / 30 |
| IPC | 1.80 |
| branch miss | 0.43% |
| L1D miss | 3.86% |
| LLC load miss | 37.35% |
| dTLB miss | 3.41% |

record pass 为 Evolve `28.7611 s`，与 stat pass 接近；四个输出文件的数值行在
stat/record 两次运行之间完全一致。

热点仍然是数值内核和大数组搬运，而不是 OpenMP runtime：

| 函数/路径 | cycles self |
|---|---:|
| `compute_rhs_bssn_` | 41.47% |
| `__memcpy_sve` | 12.57% |
| `lopsided_` | 8.54% |
| `prolong3_` | 4.90% |
| `fdderivs_` | 4.64% |
| `__memset_sve_zva64` | 5.82% |
| `rungekutta4_rout_` | 2.45% |
| `fderivs_` | 2.31% |
| `kodis_` | 2.10% |

带 `-g` 的代码行 profile 仍集中在 `bssn_rhs.f90:609/687/648`、
`lopsidediff.f90:277/302`、`diff_new.f90:556` 和
`prolongrestrict_cell.f90:2136/2143`。调度没有把计算热点转移成异常的
`libgomp`、branch 或同步热点。

## 5. 为什么平均 CPU 仍不是 30

`perf stat` 的 22.65 是整个 t=0..4 窗口平均值，不是移动 level 7/8 的瞬时值。
它包含：

- level 0 只有 9 block；
- level 1--4 的 24-thread team；
- `RecursiveStep` 的层间顺序和每次 Sync/transfer；
- constraint、regrid 和输出等非 block 主循环工作。

最终诊断中，level 7/8 的 block phase utilization 已达到约 0.923/0.906，说明
主计算阶段大部分 worker 已有工作；剩下的平均值下降主要是算法层级和同步结构，
不是 affinity 丢失。作业 cpuset 和 OpenMP 绑定仍保持 cores/close。

## 6. 接入和边界

`hpc_cpu.sh` 现在默认导出：

```text
OMP_SCHEDULE=dynamic,1
AMSS_OMP_STATIC_BLOCK_TARGET=max(60, worker_count)
AMSS_OMP_MOVING_BLOCK_TARGET=max(60, worker_count)
AMSS_OMP_STATIC_THREADS=80% of worker_count
AMSS_OMP_MOVING_THREADS=100% of worker_count
```

所有变量都能在提交时覆盖，便于评测或回退。生产编译仍是 `-O3`；本轮没有使用
`-Ofast`、`-ffast-math` 或 MPI 多进程。调度实验脚本为
`hpc_abe_schedule_sweep.sh`，它将候选顺序运行在同一节点并检查数值一致性。

## 7. 下一步

调度层面的低风险空间已基本处理完。下一步应回到真正占周期的部分：

1. `compute_rhs_bssn` 内部 phase 的 SIMD/数据复用；
2. `memcpy/memset` 和 Sync/AMR transfer 的 scratch、布局及批量化；
3. 只有在这些内核优化后仍有明显尾部，才考虑 block 内部的更细 task 划分。

继续盲目增加 block 数会增加 ghost/boundary 和 transfer 成本；继续切换
`dynamic/guided` 在 30 block 上也没有可重复收益。
