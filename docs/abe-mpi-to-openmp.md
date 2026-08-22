# ABE 从 MPI 执行模型转换为单进程 OpenMP：实现与性能报告

## 1. 本阶段目标

本阶段不是简单地把启动命令从 `mpiexec -n 30` 改成
`OMP_NUM_THREADS=30`，而是把原先由 MPI rank 所隐含的并行工作真正交给
OpenMP 线程执行。固定条件如下：

- 1 个 ABE 进程，30 个 OpenMP 线程；
- `OMP_PLACES=cores`，`OMP_PROC_BIND=close`；
- CPU ABE 使用 `-O3 -g -fno-omit-frame-pointer`，没有使用
  `-Ofast` 或 `-ffast-math`；
- 课程固定网格与物理参数，正式测试演化区间为 `t=0..4`；
- TwoPuncture 使用缓存，计时与 profile 只覆盖 ABE。

## 2. 为什么上一版 OpenMP 并行度低

上一版已经不链接 MPI 库，但只把 `Step` 中的 RHS 块循环、
`Compute_Psi4` 和部分插值循环并行化。程序中原来依靠多个 MPI rank
并发执行的网格同步和 AMR 层间传输，在单进程下却退化为主线程串行执行：

1. `Sync`、`Restrict`、`Prolong` 等函数先生成源/目标网格段列表；
2. `transfer` 调用 `data_packer`，串行执行
   `copy/restrict3/prolong3`；
3. 全部数据打包完成后，再由主线程串行写入目标网格；
4. `prepare_inter_time_level` 和 `Step` 末尾的状态交换也串行遍历 block。

逐线程 perf 证实了这一点。上一版主线程占 24.50% 的周期样本，普通工作
线程大多只占约 2%；`prolong3/restrict3/copy` 几乎只出现在主线程。
所以问题不是 OpenMP 不能并行，而是原 MPI 数据所有权已经取消，相关工作
却没有重新分发给 OpenMP 线程。

## 3. 如何转换

### 3.1 运行时不再使用 MPI

OpenMP-only 目标不链接 `libmpi`，`ldd ABE` 只显示
`libgomp`，不显示 MPI 库。源码中仍保留一部分 `MPI_Comm_rank`、
`MPI_Allreduce` 等名字，是为了不重写整个类接口；在该构建中它们由
`omp_only_mpi.h` 提供单进程语义：

- rank 恒为 0，size 恒为 1；
- reduce/allreduce 退化为本地复制；
- barrier 是空操作；
- send/recv 路径不会进入。

因此这些名字是兼容接口，不代表运行时还有 MPI 进程、通信或 MPI barrier。

### 3.2 AMR 数据传输改为 OpenMP 两阶段执行

单进程 `transfer/transfermix` 现在把每个“网格段 × 变量”建立为一个
任务。每个任务记录源 block、目标 block、源/目标变量和临时缓冲区偏移。

执行仍严格保留原 MPI 路径的依赖次序：

1. 第一阶段并行 PACK：执行 `copy`、`restrict3` 或 `prolong3`，
   只读源网格并写各自的临时缓冲区；
2. OpenMP parallel-for 末尾的隐式 barrier 保证所有读取都已完成；
3. 第二阶段并行 UNPACK：不同变量并行写回；同一变量的网格段按原顺序
   写回，避免相邻边界段重叠时产生写冲突。

PACK 使用 `schedule(dynamic,1)`，因为不同网格段的大小不同；
UNPACK 按变量静态分配。这样既保持原来的 pack-before-unpack 数值语义，
又让最重的 prolong/restrict 工作不再集中在主线程。

### 3.3 其他串行 block 循环

另外并行化了两处明确独立的循环：

- `prepare_inter_time_level`：每个 block 的时间层平均只访问该 block
  自己的数组，因此按 block 使用静态 OpenMP 循环；
- `Step` 最终状态交换：复用本次 Step 已经建立的本地 block 向量，
  不再由主线程串行遍历所有 patch/block。

`RecursiveStep` 的层级递归顺序没有并行化。粗层推进、细层递归推进、
限制/延拓和下一次迭代之间存在真实的时间积分依赖，直接把这些步骤并发
会改变算法，而不是无痛的并行优化。

## 4. 正确性验证

进行了两类验证：

- 同一新版本的 perf-stat 与 perf-record 两次运行之间，
  `bssn_ADMQs.dat`、`bssn_BH.dat`、`bssn_constraint.dat` 和
  `bssn_psi4.dat` 的数值行逐字节一致；
- 新版本完整 `t=0..4` 输出与修改前 P1 在相同配置下逐字节一致。

OpenMP-only 与 MPI 兼容构建也都完成了全量编译。当前测试没有发现数值
变化或旧 MPI 构建被破坏。

## 5. 性能结果

正式结果取 perf-stat pass 的 ABE 内部计时：

| 版本 | 进程 × 线程 | Evolve (s) | Total (s) | 相对当前版本 |
|---|---:|---:|---:|---:|
| 原始 MPI baseline | 30 × 1 | 173.669 | 177.580 | 当前快 2.81 倍 |
| 上一版 P1 OpenMP | 1 × 30 | 542.358 | 550.343 | 当前快 8.76 倍 |
| 当前共享内存转换 | 1 × 30 | **61.883** | **69.670** | 1.00 |

当前 perf-record pass 的 Evolve 为 60.553s，与 stat pass 接近，说明结果
不是一次偶然的快跑。内存峰值约 3.2 GiB，远低于 100 GiB 限额。

### 5.1 并行度与线程平衡

| 指标 | 上一版 P1 | 当前版本 |
|---|---:|---:|
| 平均使用 CPU 数 | 3.226 | **16.371** |
| 主线程周期样本占比 | 24.50% | **4.78%** |
| 普通线程典型占比 | 约 2% | **约 2.7%--3.3%** |

平均并行度没有达到 30，主要是算法粒度造成的：

- `RecursiveStep` 的不同 AMR level 必须按依赖顺序推进；
- 最粗 level 只有 9 个 block，RHS 阶段最多只能有效使用约 9 个线程；
- RK 子步之间必须完成 ghost-zone 同步；
- 分析、内存分配和少量控制逻辑仍是串行或低并行度工作。

但主线程长尾已基本消除，负载均衡比上一版明显改善。

### 5.2 硬件计数器

| 指标 | 上一版 P1 | 当前版本 | 判断 |
|---|---:|---:|---|
| IPC | 2.53 | 2.23 | 略降，来自更多线程同时运行和带宽竞争 |
| L1D miss | 2.13% | 2.07% | 正常 |
| LLC load miss | 46.29% | 47.74% | 比例高但没有明显恶化 |
| dTLB miss | 2.50% | 1.73% | 改善 |
| branch miss | 0.38% | 0.36% | 正常 |

LLC miss 比例看起来高，但绝对 LLC miss 数约为 29.8G，与上一版约 30.2G
接近；它主要反映网格数组的流式访问，不是这次改写引入的新异常。
当前主要瓶颈仍是计算和内存访问，不是通信等待。

## 6. 新热点

当前周期热点如下：

| 函数/操作 | 周期占比 |
|---|---:|
| `compute_rhs_bssn` | 36.93% |
| `polint` | 8.32% |
| `memcpy` | 7.91% |
| `kodis` | 7.20% |
| `fdderivs` | 6.71% |
| `lopsided` | 5.88% |
| `memset` | 4.56% |
| `prolong3` | 3.52% |
| `malloc + free` | 4.75% |
| `restrict3` | 1.08% |
| `copy` | 0.55% |

`prolong3/restrict3/copy` 仍会消耗 CPU 时间，因为数学工作没有消失；
关键变化是它们已经分散到多个线程，不再形成串行墙钟瓶颈。
OpenMP runtime 本身约 2%，目前不值得优先手写 worker pool。

## 7. 30 与 60 线程短测

用同一 `t=0..0.5` 窗口补测：

| OpenMP 线程数 | Evolve (s) |
|---:|---:|
| 30 | **14.745** |
| 60 | 19.902 |

60 线程慢约 35%。它同时把部分 level 的 block 划分从 30 目标提高到 60
目标，而课程网格实际只能生成 9 或约 32 个有效 block，导致更多边界、
调度和同步开销；当前 core binding 下还会增加同一 core place 内的竞争。
因此保留 `OMP_NUM_THREADS=30`、`OMP_PLACES=cores` 和
`OMP_PROC_BIND=close` 是当前更好的选择，不需要用单 MPI rank 帮助绑核。

## 8. 结论与下一步

这次转换已经解决“OMP 线程只做 RHS，MPI 风格网格传输仍由主线程执行”
这一根因。单进程 OpenMP 不仅摆脱了真实 MPI 通信，而且在固定课程用例上
明显快于 30-rank MPI baseline。

下一轮不应继续盲目扩大线程数。更有价值的方向按优先级是：

1. 处理 `polint` 中的频繁 `malloc/free`，减少分配器开销和线程竞争；
2. 检查 `memcpy/memset` 对应的数据布局和不必要的全数组清零/复制；
3. 对 `compute_rhs_bssn`、`fdderivs`、`kodis` 做向量化报告和带宽分析；
4. 如果继续优化 AMR 传输，复用临时缓冲区，避免每次 transfer 分配大数组；
5. 在更大网格下重新测试 30/60 线程，因为最佳线程数取决于 block 数和内存带宽。
