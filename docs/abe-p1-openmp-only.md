# ABE P1：单进程 OpenMP 实验报告

记录日期：2026-08-22。本文记录 ABE 的 P1 实验：将 ABE 从“30 个 MPI rank、每个 rank 一个线程”切换为“一个进程、30 个 OpenMP 线程”。本阶段的目标是验证架构切换和正确性，不假定切换后一定更快。

## 1. 实验定义与实现

P1 采用纯 OpenMP 的 ABE 启动方式。`AMSS_ENABLE_OMP_ONLY=ON` 时，CMake 不再把 `MPI::MPI_CXX` 链接到 ABE；`src/omp_only_mpi.h` 只保留 MPI 的类型声明，并把已有调用替换为单进程语义：`Comm_size=1`、`Comm_rank=0`，`Allreduce` 退化为本地复制，`Bcast` 和点对点通信为空操作。这样 ABE 不会初始化 MPI，也不需要 `mpiexec` 启动。编译阶段仍使用 MPI 头文件，因为旧代码的函数签名依赖 `MPI_Comm` 等类型；这是编译依赖，不是运行时通信依赖。`ldd ABE` 只显示 `libgomp`，没有 `libmpi`。

OpenMP 绑定由 profile 脚本设置：

```text
OMP_NUM_THREADS=30
OMP_PLACES=cores
OMP_PROC_BIND=close
```

ABE 的 measured region 直接执行 `./ABE`，脚本中的 `mpiexec` 只用于课程原有的输入准备阶段（生成/缓存 `Ansorg.psid`），不在演化计时中。作业分配到的 CPU 列表为 `0-63`，每个物理核有两个 SMT 逻辑 CPU。

本阶段并行的具体位置如下：

1. `bssn_class::Step` 预先建立当前 refinement level 的 block 列表，用 OpenMP 并行执行 predictor、每个 corrector 和交换阶段的 block 工作；`Compute_Psi4` 也按 block 并行。
2. `Patch::Interp_Points` 按插值点并行。每个点只写自己的 `shellf` 和 `weight` 槽位，几何临时变量改成线程私有，避免线程间写冲突。
3. 单进程的 `Parallel::transfer/transfermix` 跳过 MPI request 和 `Waitall`，改走本地 pack/unpack 路径。
4. OMP-only 构建增加 block 切分数量，使 block 可以作为 OpenMP 工作单元；普通 MPI 构建不启用这条切分逻辑。

## 2. 测量方法

代码基线为提交 `7ecaa99`（ABE profile 后、实质修改前）。使用课程固定输入，复用已验证的 TwoPuncture 缓存，只把 ABE 副本的 `ABE::TotalTime` 设为 `0..4`。ABE 分别运行 `perf stat` 和 `perf record` 两遍，保留 `-O3 -g -fno-omit-frame-pointer`，没有使用 `-Ofast`。

比较基线：`profile/abe-20260821T071401Z-14`，30 MPI rank、`OMP_NUM_THREADS=1`。P1 结果：`profile/abe-20260822T001732Z-181597`，1 进程、30 OpenMP 线程。

| 配置 | 初始化 | 演化 `t=0..4` | 进程总计 | 每步演化（约） |
|---|---:|---:|---:|---:|
| MPI baseline | 3.91 s | 173.67 s | 177.58 s | 43.3 s |
| P1 `perf stat` | 7.98 s | 542.36 s | 550.34 s | 440.9 s |
| P1 `perf record` | 6.58 s | 541.86 s | 548.44 s | 440.5 s |

因此，P1 当前演化时间约为 baseline 的 **3.12 倍**。两次 P1 pass 的差异小于 0.1%，说明这个结论不是采样开销造成的。P1 的目标是去除 MPI 运行时，而不是本阶段就取得加速；在当前并行粒度下，单进程 OpenMP 反而暴露了大量串行阶段和不足的工作量。

四个输出文件在 `perf stat` 与 `perf record` 两次运行间逐行一致：`bssn_ADMQs.dat`、`bssn_BH.dat`、`bssn_constraint.dat`、`bssn_psi4.dat` 均通过数值一致性检查。

## 3. P1 profile 结果

### 3.1 热点和调用路径

`perf record` 共获得约 194K 个周期样本，没有丢样本。平坦热点为：

| 函数/符号 | 周期样本 |
|---|---:|
| `compute_rhs_bssn_` | 25.10% |
| `polint_` | 10.09% |
| `fdderivs_` | 9.33% |
| `kodis_` | 8.39% |
| `lopsided_` | 7.86% |
| `__memcpy_sve` | 7.01% |
| `prolong3_` | 6.55% |
| `fderivs_` | 3.70% |
| `malloc` + `cfree` | 约 6.3% |

调用图显示，OpenMP worker 中最大的路径是 `Step -> compute_rhs_bssn_`（一个 worker clone 的 children 约 41.4%，其中 RHS 约 39.5%）；插值路径 `Patch::Interp_Points -> global_interp_ -> polin3_ -> polint_` 约 17.5%。另一个 Step worker clone 约 14.9%。这说明 P1 的并行区域确实覆盖了 RHS、RK block 工作和插值，但核心数利用率没有跟上。

### 3.2 计算、访存和 OpenMP 利用率

硬件计数器（`perf stat`）给出：

- `task-clock` 约 1776.8 CPU-s，墙钟约 550.8 s，平均只使用 **3.23 个 CPU**；
- IPC 为 2.53，branch miss 0.38%，L1 data miss 2.13%；
- LLC load miss 46.29%，dTLB miss 2.50%；
- context switch 和 CPU migration 都为 0。

branch miss 不异常，IPC 也不低；但 LLC miss 较高，且平均只有 3.23 个 CPU 在工作。当前主要问题不是 MPI 通信等待（P1 已无 MPI DSO），而是**并行工作量不足/串行阶段过多，同时伴随明显的内存访问和复制成本**。尤其是粗层实际只有约 9 个 block，远少于 30 个线程；插值的 `polint_` 仍在频繁分配、释放和复制临时数据。

单进程时不存在 MPI rank 间负载均衡问题。对应的负载均衡问题已经转化为 OpenMP 层面的任务粒度问题：线程经常等待下一个 refinement、halo 或分析阶段，导致 30 个线程没有持续满载。

### 3.3 DSO 和通信结论

周期样本按 DSO 分布为：ABE 80.00%、libc 17.54%、libgomp 1.22%、libm 1.19%。baseline 中占大头的 `libmpi`/`libopen-pal` 已完全消失；libc 主要对应 `memcpy`、`memset`、`malloc/free`，不是通信库。

## 4. 结论与下一步

P1 已验证“单进程 OpenMP 可以绑定核并完全绕过 MPI 运行时”：ABE 的 `Comm_size` 为 1，直接启动，不链接 `libmpi`，且双 pass 数值一致。但仅把 MPI rank 级工作改成 block/插值点级 OpenMP 并不足以达到性能目标，当前比 30-rank baseline 慢约 3.1 倍。

下一步建议按以下顺序推进：

1. 在保持一个 OpenMP 进程的前提下，把长时间的 Fortran RHS、差分、耗散和 prolong/restrict 循环纳入持久并行区，减少每个 RK/level 反复创建 parallel region 的开销；必要时使用显式 worker pool。
2. 先解决任务粒度：按 block 数、refinement level 和数组 tile 做更细的 work queue，不能只依赖粗层的 9 个 block。
3. 优化 `global_interp/polint` 的临时数组和 allocator 热点；当前约 6.3% 周期样本在 `malloc/free`，且 `memcpy` 约 7.0%。
4. 把单进程 transfer 的临时 pack buffer 改为可复用的线程/level workspace，再进行 NUMA first-touch 和线程数（1、4、8、16、30、60）扫描。
5. 继续使用 `-O3 -g` 作为诊断基线；在并行粒度和内存布局稳定后再单独评估 `-mcpu=native` 或 `-Ofast`，不要用编译参数掩盖当前的并行性问题。

本阶段暂不建议删除普通 MPI 构建路径。OMP-only shim 只适用于 `Comm_size=1`，多进程时必须保留原 MPI 实现；后续若验证纯 OpenMP 算法稳定，再考虑把 MPI 类型和调用从公共代码中彻底清理。
