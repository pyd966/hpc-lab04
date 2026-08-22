# ABE 并行阶段报告：MPI 到 OpenMP 及共享内存优化

> 本文记录当时只完成 P1/transfer 的阶段状态，其中“未完成事项”已经过时。
> P0/P1/P2 的最终实现和结果见
> [abe-openmp-p0-p1-p2-final.md](abe-openmp-p0-p1-p2-final.md)。

## 1. 本阶段结论

本阶段完成并验证了两件不同性质的工作：

1. ABE 的执行模型已经是单进程 OpenMP。运行时不再启动 MPI
   rank，也不再发生跨进程通信。
2. 在这个 OpenMP-only 模型中，AMR 的本地网格传输使用一个 OpenMP
   team 完成 PACK 和 UNPACK 两个阶段，并复用 packed 数据缓冲区。

完整 HPC profile 的 t=0..4 结果为：

| 版本 | Evolve (s) | Total (s) |
|---|---:|---:|
| 修改前 P1 OpenMP | 61.883 | 69.670 |
| 本阶段 OpenMP | 61.477 | 68.761 |

两次运行环境和编译选项均为 1 个进程、30 个 OpenMP 线程、
OMP_PLACES=cores、OMP_PROC_BIND=close、-O3 -g
-fno-omit-frame-pointer。本阶段的变化只有小幅收益，不能据此声称
有显著加速；主要价值是减少了 transfer 的线程队伍切换和临时大数组
分配，并为后续缓存 transfer plan 留下了正确的结构。

profile 目录：
profile/abe-20260822T045940Z-14

## 2. 从 MPI 转为 OpenMP

### 2.1 运行时模型

OMP-only 构建通过 AMSS_ENABLE_OMP_ONLY=ON 启用 src/omp_only_mpi.h。
源码中保留 MPI 类型和函数名，是为了避免一次性重写全部旧接口，但这些
调用已经没有 MPI 运行时语义：

- MPI_Comm_size 固定返回 1；
- MPI_Comm_rank 固定返回 0；
- MPI_Init、MPI_Finalize 不启动或关闭 MPI world；
- MPI_Allreduce 在单进程中退化为本地复制；
- send、receive、wait 和 barrier 路径不会进行跨进程通信；
- OMP-only 的 ABE 不链接 libmpi，只使用 OpenMP runtime。

因此，源码中出现 MPI_Allreduce 名称不代表仍然有网络通信；它们是为了
维持旧 C++ 接口而保留的兼容调用。

### 2.2 原 MPI 并行区域如何重新分配

MPI 版本中由 rank 所有权隐含的并行工作，改由 OpenMP 线程执行：

- bssn_class::Step 中同一 RK 阶段的本地 block RHS、代数约束、RK 更新
  和边界处理按 block 使用 OpenMP；
- Compute_Psi4 的本地 block 计算按 block 并行；
- Patch::Interp_Points 的曲面积分点按 point 并行，每个 point 写入
  不重叠的 shellf/weight 槽位；
- prepare_inter_time_level 的 block 平均按 block 并行；
- Step 末尾的 block 状态交换按 block 并行；
- AMR transfer 的 copy、restrict3、prolong3 和 prolongmix3 由 OpenMP
  执行。

RecursiveStep 的层级递归顺序没有强行并发化。粗层推进、细层递归、
restrict/prolong、ghost 同步和下一次 RK 阶段之间有真实数据依赖；
直接把这些控制步骤同时执行会改变数值算法。

## 3. OpenMP 相对 MPI 新增加的算法优化

### 3.1 transfer 的任务粒度

单进程 transfer 先把每一个“源 grid segment、目标 grid segment、变量”
组成一个 OmpLocalTransferOp。每个 operation 记录源和目标 block、源和
目标变量、packed 缓冲区 offset，以及要处理的数据长度。

这样，原来由不同 MPI rank 分担的 segment/variable 工作，在共享地址空间
中可以直接交给 OpenMP 调度。

### 3.2 PACK 和 UNPACK 的依赖

transfer 仍然严格保持原 MPI 路径的先后关系：

1. 第一阶段 OpenMP omp for 并行执行 PACK。每个 operation 只读取源网格
   并写自己不重叠的 packed 区间。
2. 第一阶段末尾的隐式 barrier 保证所有源数据读取完成。
3. 第二阶段 OpenMP omp for 执行 UNPACK。不同变量可以并行；同一变量
   内的 segment 保持原有顺序，以避免相邻边界段覆盖同一点。

两个阶段现在共享一个 OpenMP team。只有 operation 数大于 1 时才真正
创建多线程 team；单 operation transfer 保持单线程，避免小任务反复
创建 30 个线程。

### 3.3 缓冲区复用

原实现每次 transfer 都执行：

    new double[total_size]
    PACK
    UNPACK
    delete[]

现在 OMP-only 路径保留一个进程内 vector<double> workspace。容量不足
时才扩容，后续 transfer 复用同一块 packed 存储。transfer 调用位于
同步边界，由控制线程顺序进入，因此不会有多个 transfer 同时使用这块
workspace。

这项改动只复用 packed data buffer，没有声称已经缓存所有 grid segment
列表、operation 元数据或每个变量的其它临时数组；那些属于后续 P2。

## 4. 正确性和 profile 结果

### 4.1 正确性

OMP-only 构建和 MPI 兼容构建都成功编译。完整 profile 的 stat run 与
record run 中，以下文件去掉头部后逐字节一致：

- bssn_ADMQs.dat
- bssn_BH.dat
- bssn_constraint.dat
- bssn_psi4.dat

### 4.2 硬件计数器

完整 stat run 的主要结果：

| 指标 | 结果 | 判断 |
|---|---:|---|
| 平均使用 CPU | 16.558 / 30 | 受 AMR 依赖和 block 数限制 |
| IPC | 2.24 | 正常，主要是计算和流式访存 |
| branch miss | 0.33% | 不异常 |
| L1D miss | 2.06% | 不异常 |
| LLC load miss | 48.04% | 网格数组流式访问导致，仍是访存压力 |
| dTLB miss | 1.73% | 与此前相当 |
| elapsed | 69.212s | 与程序内部计时一致 |

这说明本阶段的主要问题不是 MPI 等待。OpenMP team 的运行时样本约
1%，而 RHS 和网格数据访问占绝大多数时间。

### 4.3 当前热点

flat profile 的主要热点为：

| 函数或操作 | 周期占比 |
|---|---:|
| compute_rhs_bssn | 36.80% |
| polint | 8.32% |
| memcpy | 7.78% |
| kodis | 7.32% |
| fdderivs | 6.86% |
| lopsided | 5.84% |
| memset | 4.34% |
| prolong3 | 3.55% |
| fderivs | 2.86% |
| malloc + free | 4.72% |

调用树中，omp_local_transfer 的 children 约 12.18%，其中主要是 copy
5.41%、prolong3 4.02% 和 restrict3 1.67%。这表示传输计算本身仍然
重要，但本阶段消除的是 transfer 管理开销，不会消除 copy/prolong/
restrict 必须搬运的数据。

polint 仍然通过 global_interp -> polin3 -> polint 占约 13.8% 的调用树
时间，并且其内部仍能看到 malloc/free。下一步若要明显加速，应优先
处理插值权重、点的 block 所有权和 polint 临时数组，而不是继续微调
transfer 的 OpenMP pragma。

## 5. 未完成事项和建议

本阶段没有实现以下内容：

- P0 的直接局部积分归约；当前分析路径仍会生成完整 shellf；
- Sync grid segment 列表和 operation 元数据缓存；
- 所有分析归约的统一打包；
- 持久 OpenMP worker pool；
- RHS Fortran kernel 内部的 tile/loop 级并行。

建议的下一步顺序是：

1. 先做 P2 的 transfer plan 缓存，复用 gridseg 列表、ops 元数据和更细
   粒度的 per-thread scratch；
2. 再做 P0：预计算曲面积分点的 block/index/插值权重，并在插值后直接
   累积最终积分量，删除完整 shellf 中间数组；
3. 对 compute_rhs_bssn、fdderivs 和 kodis 做 block 内 tile/循环并行，
   解决当前平均只有约 16.6 个 CPU 在工作的限制；
4. 每一步继续用相同的 t=0..4、30 核绑定、符号 profile 和输出比对
   评估，避免把减少科学工作量误判为优化。
