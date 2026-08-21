# ABE CPU baseline profile 与优化建议

## 结论

本轮 ABE baseline 首先是通信/同步瓶颈，不是某个 BSSN 数值内核单独过慢：

1. libopen-pal 与 libmpi 合计占 82.25% 的 CPU 周期样本，主要是 collective 通信中的进度轮询。30 个核几乎一直处于运行态，但很多核是在忙等。
2. 最大等待来自分析阶段。调用栈显示 AnalysisStuff -> surf_MassPAng/surf_Wave -> Interp_Points -> MPI_Allreduce 是主路径；AnalysisStuff 的 inclusive 占比为 72.55%，PMPI_Allreduce 为 74.52%。
3. rank 负载明显不均衡。除 rank 4 外，其余 rank 有 81.4%--89.2% 的样本位于 MPI/OpenPAL；rank 4 有 61.3% 位于 ABE 本体。运行日志也报告 level 0 实际只能有效使用约 9/30 个进程。
4. 第一优先级应是改变分析阶段的归约方式；第二优先级是按既定方向改成单进程 OpenMP，并让 block 数量不再受 MPI rank 数量约束。单独调编译参数、换数学库或只优化 RHS，端到端收益上限很低。

## 1. 实验方法

### 1.1 环境和编译

有效集群任务为 129782，运行节点为 zjusct-920b-3，采样产物位于：

    profile/abe-20260821T071401Z-14/

本轮只分析 CPU ABE：

- CPU：HiSilicon TaiShan-v120，最高 2.9 GHz；
- 分配：60 个逻辑 CPU，即 30 个物理核，CPU 列表 64-123；
- NUMA：完全位于 NUMA node 1，没有跨 NUMA 访问；
- ISA：支持 NEON/ASIMD 和 SVE；
- 并行：30 个 MPI rank，OMP_NUM_THREADS=1；
- 绑核：每个 rank 绑定一个物理核，对应两个 SMT 逻辑 CPU；
- 编译器：GCC/GFortran 14.2.0；
- ABE 编译参数：-O3 -g -fno-omit-frame-pointer -fopt-info-vec-optimized；
- 未使用 -Ofast，也没有给 ABE 添加 -march=native。

-g 和 frame pointer 使 perf 能还原函数、源文件、代码行和调用路径。TwoPuncture
准备过程不在 perf 测量区间内。

### 1.2 输入和测量窗口

网格、物理参数、8 个探测半径、分析频率等均来自课程固定输入。为了在一个
作业中分别运行硬件计数器与调用栈采样，仅把 staging 副本中的 ABE::TotalTime
从 40 缩短到 4，其余参数不变。

做了两个独立进程：

1. perf stat -d -d：采集 IPC、cache、TLB 和分支事件；
2. perf record -F 99 --call-graph fp：采集 99 Hz 调用栈。

两轮 evolution 时间只相差 0.09%。忽略文件创建时间头后，
bssn_ADMQs.dat、bssn_BH.dat、bssn_constraint.dat 和 bssn_psi4.dat 的数值行
逐字节一致，说明采样具有良好的可重复性。

## 2. ABE 当前运行流程

程序主干可以简化为：

~~~text
读入 TwoPuncture 初值与 9 层 AMR 网格
  -> 初始化变量、同步边界和诊断器
  -> Evolve
       -> RecursiveStep(level 0)
            -> Step(level)
                 -> level 0 上执行 AnalysisStuff
                 -> RK4 predictor
                 -> 3 个 RK4 corrector
                 -> 每个 RK 子步：RHS + NaN 全局检查 + halo Sync
            -> 递归推进更细层
            -> Restrict/Prolong + Sync
       -> Constraint_Out
       -> 输出/检查点（达到设定时间时）
~~~

### 2.1 AMR 时间细分

当前有 9 层，时间细分从 level 3 开始。一次物理时间步内，各层 Step 调用数为：

| level | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Step 次数 | 1 | 1 | 1 | 1 | 2 | 4 | 8 | 16 | 32 |

合计 66 次 Step。每次 Step 有 4 个 RK 子步，因此每个物理时间单位至少发生：

- 264 次 compute_rhs_bssn；
- 264 次 RK 后的错误标志 MPI_Allreduce；
- 264 次 RK halo Parallel::Sync；
- Restrict/Prolong 再增加 130 次 Parallel::Sync。

所以不计约束和分析通信，已有至少 394 次 halo 同步。固定 t=0..40 运行至少
包含 10,560 次 RHS/错误归约和 15,760 次 halo 同步。

### 2.2 分析阶段的通信规模

AnalysisStuff 在 level 0 上运行。本输入 AnalysisTime=0.1，而粗层每次推进 1
个物理时间单位，因此每个粗层步都会做一次分析。

曲面积分参数为 N=96。在当前 z 对称设置下：

- N_phi=384；
- N_theta=96；
- 每个半径有 36,864 个积分点；
- 一共有 8 个探测半径。

每个半径分别计算：

- 波形：插值 2 个变量；
- 质量、线动量和角动量：插值 17 个变量。

Interp_Points 已经把一个半径的所有点放进一次函数调用，问题不是每个点调用
一次 Allreduce。问题是它随后对完整插值数组做两次 MPI_Allreduce：一次归约
数值，一次归约 point ownership 的 weight。

每个半径的静态 collective 次数为：

| 工作 | 次数 |
|---|---:|
| 波形插值：值 + weight | 2 |
| 波形积分：实部 + 虚部 | 2 |
| 质量/动量插值：值 + weight | 2 |
| 质量、3 个线动量、3 个角动量 | 7 |
| 每个半径合计 | 13 |

8 个半径即每次分析约 104 次 MPI_Allreduce。17 变量插值数组每个半径约
5.01 MB；加上 wave 和 weight，8 个半径每次分析要把约 47.2 MB 的中间数组
交给 collective。47.2 MB 是调用输入大小总和，实际网络/共享内存流量还会
随 MPI 算法和 rank 数增加。

## 3. Profile 结果

### 3.1 时间

perf stat 一轮结果：

| 阶段 | 时间 |
|---|---:|
| Evolution 前初始化 | 3.911 s |
| t=0..4 evolution | 173.669 s |
| 整个 ABE 进程 | 177.580 s |
| 每个物理时间单位 | 43.085--43.497 s |

线性估算固定 t=0..40 的 evolution 约需 28.9 分钟。这不是完整 40 步的实测值，
最终验收仍应实际跑满 40。

### 3.2 模块和调用路径

| DSO | 周期样本 |
|---|---:|
| libopen-pal.so | 69.48% |
| ABE | 14.33% |
| libmpi.so | 12.77% |
| libc.so | 3.25% |
| libm.so | 0.14% |

OpenPAL + MPI 合计 82.25%。这部分主要是 collective/等待期间的进度轮询，
不是有效的 BSSN 浮点计算。

调用栈 inclusive 热点为：

| 函数 | inclusive 样本 |
|---|---:|
| bssn_class::Step | 88.47% |
| PMPI_Allreduce | 74.52% |
| Patch::Interp_Points | 73.87% |
| bssn_class::AnalysisStuff | 72.55% |
| compute_rhs_bssn | 11.86% |

这些数字包含子调用，彼此重叠，不能相加。最大的实际路径是：

~~~text
Evolve -> RecursiveStep(0) -> Step(0) -> AnalysisStuff
      -> surf_MassPAng / surf_Wave -> Patch::Interp_Points
      -> PMPI_Allreduce -> OpenPAL progress polling
~~~

普通 halo transfer 的 MPI_Waitall 也能在调用栈中看到，但远小于上述分析归约。
每个 RK 子步的错误标志 Allreduce 合计约 2.85% inclusive，也不是第一热点。

### 3.3 有效计算内核

去掉 MPI/OpenPAL 后，平坦函数热点主要是：

| 函数 | self 样本 | 含义 |
|---|---:|---|
| compute_rhs_bssn | 6.80% | BSSN 右端项 |
| polint | 1.44% | 通用多项式插值 |
| kodis | 1.28% | Kreiss-Oliger 数值耗散 |
| fdderivs | 1.20% | 二阶/混合空间导数 |
| lopsided | 1.04% | 按 shift 方向选择的偏置导数 |
| prolong3 | 0.63% | AMR prolongation |
| fderivs | 0.50% | 一阶空间导数 |
| malloc + free | 0.87% | 动态分配器本身 |

一次 RHS 静态调用 21 次 fderivs、11 次 fdderivs、24 次 lopsided 和 24 次
kodis。这些函数反复扫描同一 block。compute_rhs_bssn 还声明了大量与整个
3D block 同尺寸的临时数组；各差分函数又创建扩展的 fh 临时数组。因此通信
问题解决后，下一层问题很可能是临时工作集过大和重复全数组遍历。

GCC 报告显示不少循环只生成 16-byte 向量，即 128-bit NEON；本轮 ABE 没有启用
面向本机的 SVE 编译。Fortran 循环以第一维 i 为内层，符合列主序连续访问。

### 3.4 硬件计数器

| 指标 | 结果 | 判断 |
|---|---:|---|
| 平均活跃 CPU | 29.867 / 30 | 核都在运行，但大量是 MPI 忙等 |
| IPC | 1.89 | 混入 MPI 轮询，不能代表 RHS IPC |
| branch miss | 1.07% | 正常，不是首要问题 |
| L1D miss | 0.37% | 很低 |
| LLC miss / LLC load | 50.91% | 条件比例高，但 LLC load 总量较小 |
| dTLB miss / dTLB load | 7.53% | 偏高，应在纯计算 profile 中复查 |
| iTLB miss | 约 0% | 正常 |

不能只凭这组计数器把程序定性为内存带宽瓶颈：本轮没有 DRAM 带宽事件，
而且 82.25% 的样本位于 MPI/OpenPAL。更准确的结论是：

- 端到端 baseline：通信/同步瓶颈；
- 有效计算部分：有大临时工作集、重复流式扫描和 dTLB 压力的迹象；
- 删除通信热点后，应重新只对 RHS/差分区间采样，再判断带宽还是计算瓶颈。

硬件事件有 57%--64% 的 multiplex 覆盖率，perf 对结果做了缩放。这些比例
适合判断量级，不应解释到小数点后的细微差异。

## 4. MPI rank 负载均衡

每个 rank 的全局样本都约 3.33%，表面上很均匀。但按样本所在模块拆开：

| rank | 全局样本 | ABE 本体占该 rank | MPI/OpenPAL 占该 rank | libc |
|---:|---:|---:|---:|---:|
| 4 | 3.33% | 61.3% | 8.7% | 29.7% |
| 16 | 3.33% | 15.9% | 81.4% | 2.7% |
| 1 | 3.33% | 10.8% | 86.8% | 2.4% |
| 29 | 3.33% | 8.7% | 89.2% | 2.1% |

除 rank 4 外，其余 rank 的 ABE 本体占比只有 8.7%--15.9%。rank 4 承担了
远多于其他 rank 的实际插值/内存工作，其他 rank 在 collective 中等待它。
所以 29.867 核利用率不等于 29.867 核有效加速。

成因有两层：

1. Parallel::distribute 按空间 block 分配工作，但粗网格只有 40x40x20，
   日志明确提示 level 0 只能有效使用约 9 个进程；
2. 曲面积分点的实际插值只由持有对应 block 的 rank 完成，随后所有 30 个
   rank 进入全局归约，空间所有权不均衡直接变成 collective 等待。

## 5. 推荐的优化顺序

### P0：重写分析阶段的归约方式

这是最可能带来数量级收益的改动。当前算法先生成完整 shellf 插值数组，再做
积分。更合适的方式是：

1. 每个执行单元只处理本地 block 拥有的球面积分点；
2. 插值后立即累积本地 wave mode、质量、线动量和角动量；
3. 不再全局归约点乘变量的中间数组；
4. 8 个半径的最终小结果打包，最后做一次归约。

最终结果只有每个半径 42 个 wave double 和 7 个质量/动量 double，与几十 MB
中间数组相比非常小。monitor 只在 rank 0 写文件，CPU 代码中 ADMMass 也没有
下游读取，因此 MPI 版本可以优先考虑 MPI_Reduce 到 rank 0；若未来某路径需要
所有 rank 使用结果，再广播这个很小的最终数组。

这一步可同时做三项低风险优化：

- level 0 和探测半径固定，预计算积分点所属 block、网格索引和 6 阶插值权重，
  避免每步搜索 block 并重复通用 polint；
- 预计算 wave 中只依赖球面点和 (l,m) 的 Wigner/trigonometric 系数；
- 复用 pox、shellf、weight 和局部积分缓冲区。

不要通过调大 AnalysisTime、减少探测器或降低球面分辨率获得成绩，因为这会改变
科学输出。应在输出语义不变的前提下改算法。

### P1：单进程 OpenMP，而不是 MPI + OpenMP 叠加

建议使用 OpenMP runtime 的持久 worker，不必手写线程池：

~~~text
一个长期 omp parallel
  -> single 线程推进 AMR 时间递归和建立依赖
  -> 每个 RK 阶段对 block 创建 task/并行循环
  -> taskgroup/barrier 只放在 RHS、halo、restrict/prolong 的真实依赖边界
~~~

必须先解决一个结构问题：当前 Parallel::distribute 用 MPI rank 数决定 block
切分。若简单改成一个进程，只会产生很少 block，OpenMP 没有足够任务。应把
空间 tile 数和进程数解耦，让每层至少有与线程数同量级的 tile，再由 OpenMP
调度。

具体并行区域：

- Step 中同一 RK 阶段的各 block RHS 相互独立，可按 block 并行；
- enforce_ga、Runge-Kutta 更新和边界处理跟随同一 block task；
- halo 在共享地址空间内改为直接 copy，完成后做阶段 barrier；
- Restrict/Prolong 按不重叠目标 block 并行；
- 曲面点插值和积分按 point/tile 并行，每线程维护局部积分数组，末尾合并。

绑核建议：

~~~bash
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=30
~~~

当前 allocation 只有 30 个物理核。先用 30 线程建立基准，再测 60 个 SMT 线程；
不要默认 SMT 一定更快。大数组应在并行区内 first-touch，使页保持在唯一允许
的 NUMA node 1。未来跨 NUMA 时，再按 block 做 first-touch 和本地调度。

### P2：减少高频同步和通信准备

在 MPI 过渡版本中：

- 4 个 RK 子步每次都对 NaN/error 标志做 MPI_Allreduce。release 模式可改为
  每个完整 Step 检查一次，或把 isfinite 检查融合进 RHS；debug 模式保留原频率；
- Parallel::Sync 每次重建 gridseg 链表、request/status 数组和每个 peer 的
  收发缓冲区，随后销毁。网格不变期间应缓存传输计划并复用缓冲区；
- 7 个标量质量/动量 Allreduce 可先打包成一个长度 7 的归约，wave 实部和
  虚部也可打包。

纯 OpenMP 版本中，error 归约改为线程级 OR reduction，halo 改为共享内存 copy；
MPI packing、request 和 collective 应退出主循环。

### P3：再优化 RHS、差分和临时数组

MPI 热点移除后重新 profile，然后依次尝试：

1. 为每个线程长期复用 RHS scratch，避免在 80 次差分/耗散调用中反复建立大型
   扩展数组；
2. 融合能共享输入的导数与耗散扫描，减少整块数据往返 cache/内存；
3. 将 interior 与少量 boundary 分开，interior 使用无分支连续 i 循环和
   omp simd；
4. 用小规模 kernel benchmark 测量 GB/s、IPC 和 dTLB，再决定 tile 尺寸。

不要让所有线程共用一份 scratch；应采用 per-thread 或 per-task 缓冲区，避免
数据竞争和 false sharing。

### P4：最后做编译参数和数学库实验

现在不建议更换数学库：

- libm 只有 0.14%；
- 主热点是自定义 MPI、插值和 stencil，不是 BLAS/LAPACK/FFT；
- 换库不能解决 82.25% 的同步等待。

也不建议现在把 -Ofast 作为 baseline。保留严格的 -O3，在 P0/P1 完成后做小型
参数矩阵：

1. -O3；
2. -O3 -mcpu=native 或集群确认后的 TaiShan 目标；
3. 在热点循环显式 omp simd 后复查 GCC vectorization report 和汇编。

本机支持 SVE，而当前主要热点只看到 128-bit 向量，-mcpu=native 值得实验，但
不是当前第一优先级。按 Amdahl 定律，即便把 6.80% self 的 RHS 变成零耗时，
端到端上限也约为 1.073x；即便整个 14.33% ABE DSO 变成零耗时，上限也只有
约 1.167x。先消除大比例 MPI 等待才合理。

## 6. 下一轮实验

1. 给 AnalysisStuff、两个 surf_*、Interp_Points 和每个 collective 增加调用次数
   与 wall-time 计时，验证静态计数并建立 P0 小基准；
2. 先合并 wave 的两个结果归约和 7 个标量归约，验证数值完全一致；
3. 实现本地点直接累积最终积分，删除大 shellf Allreduce；
4. 跑满相同 t=0..4，记录时间、调用栈、MPI 占比和 rank 工作占比；
5. 通过后再进入单进程 OpenMP block/task 改造；
6. 每个阶段最终都用固定 t=0..40 做性能与数值验收。

## 7. 可复现脚本和限制

采样脚本为仓库根目录的 hpc_abe_profile.sh。它会使用课程固定流程准备 ABE
输入，只缩短 profile 副本的 evolution 时间，分别运行 perf stat 和 perf record，
保存 rank 到 PID/CPU affinity 映射，并生成平坦热点、调用栈、源代码行、DSO、
rank/DSO 报告和双跑数值一致性检查。

本报告回答的是当前 30-rank baseline 在稳定时间步中的热点。它没有测量 DRAM
带宽，也没有把完整 t=0..40 放进 perf，因此不能把 28.9 分钟估算当作最终成绩；
但两轮稳定计时、52.6 万个无丢失周期样本、调用栈与静态源码计数相互印证了
“分析 collective 是第一热点”的结论。
