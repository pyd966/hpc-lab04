# Lab04 CPU/GPU 优化实验报告

日期：2026-08-26

本报告按照“最开始的 baseline profile -> TwoPuncture 优化 -> ABE 优化”
组织。短程 t=0..4 用于快速筛选和交错 A/B，最终版本使用完整 t=0..40
验收。每项重要优化都同时检查运行时间、profile 指标和数值正确性。

## 1. 运行环境

### CPU 环境

实验使用 lab4 队列的 HiSilicon TaiShan-v120/AArch64 节点。作业申请
60 个逻辑 CPU 和 100 GiB 内存，实际 cpuset 对应 30 个物理核，每核有
两个 SMT sibling。整机有 4 个 NUMA node，但本次作业的 CPU 和内存均位于
一个 NUMA node，因此没有跨 NUMA 访问。

CPU 支持 128-bit NEON/ASIMD 和 SVE，当前进程的 SVE vector length 为
256 bit。每个物理核有 64 KiB L1D、64 KiB L1I 和 1280 KiB L2，每个 NUMA
node 有 56 MiB L3。实际最高频率约为 2.9 GHz。

最开始 baseline 的配置为：

~~~text
ABE：30 MPI rank x 1 thread
绑定：mpiexec --map-by core --bind-to core
编译：-O3 -g -fno-omit-frame-pointer
TwoPuncture：单进程单线程，-O3 -g -fno-omit-frame-pointer
~~~

最终 CPU 配置为：

~~~text
ABE：1 process x 30 OpenMP threads，不启动 MPI runtime（rank 语义为 1）
OMP_NUM_THREADS=30
OMP_PLACES=cores
OMP_PROC_BIND=close
OMP_SCHEDULE=dynamic,1
静态层：24 blocks / 24 threads
移动层：30 blocks / 30 threads
ABE production：-O3
TwoPuncture production：-O3 -march=native
~~~

profile 版本额外保留 -g -fno-omit-frame-pointer，并使用 perf stat、
perf record -g 和 GCC vectorization report 定位函数、调用路径和源代码行。
最终 production 没有使用 -Ofast、fast-math 或全局 SVE 编译参数。

### GPU 环境

GPU 路径使用 NVIDIA A100 MIG 分区，一个 1g.10gb MIG 实例，16 个 CPU，
24 GiB 资源；主机工具链为 GNU 13、OpenMPI 5 和 CUDA 12.4。

~~~text
程序：ABEGPU + TwoPunctureABE
启动：1 MPI rank，mpiexec --bind-to core
OpenMP：关闭
CUDA architecture：sm_80
CUDA：-rdc=true，separable compilation，-lineinfo
AMSS_MPI_CUDA_AWARE=0，默认使用 host staging
~~~

本轮主要完成 CPU 优化。GPU 路径完成了构建、短程 benchmark 以及 Nsight
Systems/Compute 采样入口验证，但没有形成经过完整 t=100 A/B 的 GPU 优化
提交。因此这里只给出 GPU 最终运行配置，不宣称没有充分数据支持的 GPU 加速。

## 2. 最开始的 baseline 和 profile

### 2.1 程序总体流程

程序的主流程可以省略为：

~~~text
run.sh / Python driver
  -> 生成参数和运行目录
  -> TwoPuncture
       -> Newton + BiCGSTAB 求解黑洞初值
       -> 输出 Ansorg.psid
  -> ABE
       -> 建立 9 层 AMR 网格并读入初值
       -> Evolve
            -> RecursiveStep 按层级和时间细分递归
                 -> Step
                      -> predictor + 3 个 corrector
                      -> 每个 RK 阶段计算 RHS、更新状态、处理边界和 Sync
                 -> restrict/prolong、层间 Sync 和必要的 regrid
            -> AnalysisStuff / Constraint_Out / 输出
  -> Python 整理结果和 checker
~~~

RecursiveStep 的层级顺序和 Step 内 Sync 前后的 barrier 都是数值依赖的一部分，
不能直接删除。可以并行的是同一阶段中彼此独立的 block、谱线、网格点和分析点。

### 2.2 baseline 性能和正确性

最初未修改算法的作业使用 30 个 MPI rank。TwoPuncture 初值阶段实测约
294.6 s；ABE 从 t=0 演化到 t=33 后触及 30 分钟硬限制。ABE 每个物理时间
单位平均约 43.1 s，因此线性外推：

| 部分 | 最初 baseline |
|---|---:|
| TwoPuncture | 294.6 s |
| ABE Before Evolve | 3.72 s |
| ABE Evolve t=0..40 | 约 1724 s，线性外推 |
| 完整流程 | 约 2020 s，线性外推 |

已经完成的 33 个时间点中，波形匹配 33/100，trajectory RMS 为 0，约束检查
33 个时间组 x 9 个 AMR level 全部 PASS。作业没有完成是性能问题，不是数值
错误。因为 baseline 的 t=40 没有真正跑完，后文会明确区分“baseline 外推”
和“最终完整实测”。

### 2.3 TwoPuncture baseline profile

TwoPuncture 实际是单线程程序，没有 MPI 通信，也没有 OpenMP parallel region。
带调试符号的独立 profile 用时 286.5 s，平均只使用 0.999 个 CPU：

| 指标 | baseline |
|---|---:|
| IPC | 2.60 |
| branch miss | 1.22% |
| L1D miss | 2.51% |
| LLC miss | 0.05% |
| dTLB miss | 0.35% |
| 峰值内存 | 约 81 MB |

cache、TLB 和分支指标都没有异常，LLC miss 很低。因此这里不是通信或主存
带宽瓶颈，首先是串行计算问题。

主要调用路径为：

~~~text
Solve
  -> Newton
    -> bicgstab                         97.12%
      -> relax                          66.85%
        -> LineRelax_be / LineRelax_al
          -> ThomasAlgorithm
      -> J_times_dv                     27.03%
        -> Derivatives_AB3              25.86%
          -> Chebyshev/Fourier transform
~~~

cos() 自身约占 23.17%，热点 line solve 中的 malloc/free 至少占 6.07%。
这两个现象分别说明谱变换在重复计算固定系数，热点路径还在反复申请短数组。
因此先处理这两项，再根据新的 profile 决定是否改编译参数和并行化。

### 2.4 ABE baseline profile

ABE baseline 的第一瓶颈不是 BSSN 数值公式，而是单节点 MPI 通信和等待：

| 热点 | cycles/sample |
|---|---:|
| mca_btl_sm_poll_handle_frag | 58.98% |
| 其他 MPI/OpenPAL progress 路径 | 约 17% |
| compute_rhs_bssn | 6.10% |
| polint / memcpy | 1.54% / 1.17% |

调用图进一步显示：

~~~text
Evolve
  -> RecursiveStep(0)
    -> Step(0)
      -> AnalysisStuff
        -> surf_MassPAng / surf_Wave
          -> Interp_Points
            -> PMPI_Allreduce
~~~

PMPI_Allreduce 的 inclusive 样本约 74.52%，AnalysisStuff 约 72.55%。分析
阶段对 8 个半径反复归约完整的插值中间数组，每次分析交给 collective 的输入
总量约 47.2 MB。level 0 又只有约 9 个 rank 持有有效 block，因此少数 rank
做插值，其余 rank 在归约中等待。

perf stat 表面上显示平均 29.9/30 个 CPU 活跃，但大量时间是在 MPI progress
中忙等。IPC 1.89、branch miss 1.07%、LLC load miss 约 50.5%、dTLB miss
约 7.6%；这些指标混入了 MPI 自旋，不能用于判断 RHS 的纯计算效率。

这个 profile 决定了 ABE 的优化顺序：先消除 MPI/Allreduce 和 rank 负载
不均衡，再重新 profile 数值内核；如果一开始只优化占 6.1% 的 RHS，端到端
收益上限很低。

## 3. TwoPuncture 优化

### 3.1 消除热点临时分配

针对 chebft、fourft、Derivatives_AB3、F_of_v、J_times_dv、LineRelax
和 ThomasAlgorithm 的短数组，建立可复用的 TransformWorkspace、
PointWorkspace 和 LineWorkspace。workspace 是线程私有的，也为后续 OpenMP
避免了 scratch 数据竞争。

修改后 allocator 符号从热点 profile 中消失，instructions 下降 3.65%，cycles
下降 2.08%。但该次作业平均频率低 2.84%，wall time 从 286.505 s 变成
288.720 s，反而慢 0.77%。所以这一步不能宣称 wall-clock 加速，只能说明
确实减少了工作量，集群频率波动掩盖了收益。

第一次实现还因为 Thomas 每条线重新查询 TLS、别名关系不清楚而更慢。把 scratch
显式传给 Thomas 并标记 no-alias 后，Thomas 热点恢复正常。最终 BiCGSTAB
迭代、残差、参数文件和去时间戳后的 Ansorg.psid 均与 baseline 逐位一致。

### 3.2 预计算并缓存谱变换系数

固定 nA=50、nB=50、nphi=26 后，Chebyshev/Fourier 变换的角度只由网格尺寸和
循环下标决定。因此初始化时生成 Chebyshev forward/inverse cosine 表和
Fourier sine/cosine 表，后续直接读取。求和顺序没有改变，也没有改成 FFT。

| 指标 | 内存复用后 | 系数缓存后 | 变化 |
|---|---:|---:|---:|
| wall time | 288.720 s | 198.918 s | -31.10% |
| cycles | 808.832 B | 569.511 B | -29.59% |
| instructions | 2067.696 B | 1233.737 B | -40.33% |
| cos() self | 23.65% | <0.5% | 热点基本消失 |

相对最初独立 baseline 286.505 s，严格 -O3 版本累计快约 30.6%。这是
TwoPuncture 中最大的单项串行优化。

第一次实现把 1./M 和 Pi*fac 改写成了数学等价但浮点顺序不同的表达式，
导致场文件末位差异。恢复原运算顺序后，所有迭代信息和输出逐位一致。这说明
即使只是缓存系数，也必须保留浮点表达式顺序并做完整回归。

### 3.3 编译参数实验

缓存 cos 后，profile 的约 93% 集中到 LineRelax_be/al 和 ThomasAlgorithm，
不再有值得更换 libm、BLAS 或 FFT 库的热点。

同一作业内测试发现：

- -O3 -mcpu=native 基本持平；
- 额外 -funroll-loops 没有收益；
- -Ofast -march=native 比 -O3 快约 5.29%；
- -Ofast 的 Ansorg.psid 最大绝对差约 4e-15，不再逐位一致。

考虑到后续还要修改并行和数值内核，最终没有采用 -Ofast，只保留严格
-O3 -march=native，避免把 fast-math 的误差和代码优化混在一起。

### 3.4 OpenMP 并行化

新的 profile 已经非常集中，所以按调用路径寻找可证明独立的工作：

- Derivatives_AB3 在每个方向内部按独立谱线并行；A、B、phi 三阶段之间保留
  barrier，因为后一个方向会读取前一个方向的导数。
- F_of_v 和 J_times_dv 按独立网格点并行，每个线程使用私有 PointWorkspace。
- LineRelax 原本已有红黑/颜色 phase，同一 phase 的线互不依赖，可以并行；
  不同 phase 之间保留 barrier。
- 单条长度约 50 的 Thomas 递推仍串行，通过同时处理多条线获得并行度。
- 原先连续调用 200 次 relax，现在放进同一个 parallel region，避免 200 次
  fork/join；没有必要手写 worker pool。

线程 sweep 为：

| OpenMP threads | time | 相对 1 thread |
|---:|---:|---:|
| 1 | 215.162 s | 1.00x |
| 4 | 63.013 s | 3.42x |
| 8 | 31.114 s | 6.92x |
| 16 | 16.499 s | 13.04x |
| 30 | 10.375 s | 20.74x |
| 60 | 10.363 s | 20.76x |

严格 -O3 -march=native 的最终复测中，30/60 线程分别约 11.399/11.091 s。
60 个逻辑线程只比 30 个物理核快约 1.6%，却消耗接近两倍 task-clock，所以
最终选择 30 线程。相对最初 286.5 s baseline，TwoPuncture 最终约加速 25 倍。

最终 profile 中 LineRelax_be、LineRelax_al 和 Thomas 的计算样本合计约 55%，
libgomp barrier/work-sharing 约 41%。IPC 3.34，branch miss 0.28%，L1D miss
1.96%，LLC miss 0.06%，dTLB miss 0.25%。热点已经从重复 cos 和串行计算
转移到 line relaxation 必需的 phase 同步，继续简单增加线程意义不大。

## 4. ABE 优化

### 4.1 从 MPI 转为 OpenMP

baseline profile 显示单节点 MPI 和 Allreduce 占据主要时间，因此采用一个进程、
30 个 OpenMP 线程，而不是 MPI+OpenMP。源码中保留少量 MPI 类型兼容接口，但
OMP-only 构建不启动 MPI runtime、不链接 libmpi；单进程下的 rank、size 和
reduce 退化为本地语义。

第一次只并行部分 block/RHS 循环时，t=0..4 Evolve 从 MPI baseline 的
173.669 s 变成 542.358 s，平均只使用 3.23/30 个 CPU。这次失败说明“删掉 MPI”
本身不会自动得到并行：原来由不同 rank 执行的 Sync、copy、restrict、prolong、
分析和约束计算仍然落在主线程上。

后续把原 MPI 所有权代表的工作显式展开成 OpenMP 工作：

- Step 的 predictor 和三个 corrector 按 block 并行 RHS、RK 更新和边界处理；
- Sync、restrict3、prolong3、copy 按 grid segment x variable 建立 operation；
- pack 阶段并行读取源 block，barrier 后再按变量顺序写目标，避免重叠区域竞争；
- RecursiveStep 仍按 AMR 层级和时间依赖串行组织，不强行并发不同 level；
- 分析和 constraint 的 block/球面点计算也交给 OpenMP。

性能过程如下：

| 版本 | 进程 x 线程 | Evolve t=0..4 |
|---|---:|---:|
| 最初 MPI baseline | 30 x 1 | 173.669 s |
| 只并行部分 block 的初版 | 1 x 30 | 542.358 s |
| 完成 block 和 transfer 转换 | 1 x 30 | 61.883 s |
| transfer team/workspace | 1 x 30 | 61.477 s |

完整转换后平均 CPU 从 3.23 提高到约 16.4，说明 MPI 隐含的主要工作已经被
OpenMP 接管。所有改动均保持输出一致。

### 4.2 重写 AnalysisStuff 的数据流

MPI baseline 的最热调用路径是 AnalysisStuff -> Interp_Points ->
PMPI_Allreduce。旧代码对 8 个半径生成完整 pox/shellf，反复搜索 block、调用
通用 polint，并对大中间数组归约。

在单地址空间下，为每个半径预先缓存球面点所属 block、六阶插值下标/权重和
wave 系数。线程处理自己负责的球面点，插值后立刻累加到线程私有的小结果，
最后只做 OpenMP reduction，不再生成和归约几十 MB shellf。

| 版本 | Evolve t=0..4 | ABE Total |
|---|---:|---:|
| 完整 OpenMP transfer | 61.477 s | 68.761 s |
| AnalysisStuff 重写后 | 43.869 s | 51.249 s |

Evolve 下降约 28.6%，是 ABE 早期最大的单项优化。相对最初 MPI baseline，
此时 Evolve 已经快约 4 倍。新旧分析路径的演化状态一致，Psi4 仅在理论零项
出现约 1.48e-22 的舍入差异，课程 checker 通过。

### 4.3 消除隐藏的串行 constraint 路径

完成 MPI 到 OpenMP 后，cycles profile 已经以 RHS 为主，但全程平均 CPU 仍偏低。
普通 perf record 没有明显显示原因，因为一个串行函数运行 7 秒，只产生一个核
的 cycles，在 30 核 profile 中占比会被稀释。

因此增加 phase wall-time，比较总 wall、Step、Sync、transfer 和 constraint 的
时间，发现 Constraint_Out 每个物理时间单位都会在主线程上逐 block 重算 RHS，
短程约占 7 秒。随后把高频 Constraint_Out，以及进入 Evolve 前的
Compute_Constraint、Interp_Constraint RHS 重算按 block 并行：

| 指标 | 修改前 | 修改后 | 变化 |
|---|---:|---:|---:|
| Evolve t=0..4 | 37.682 s | 30.171 s | -19.9% |
| ABE Total | 44.405 s | 35.579 s | -19.9% |
| 平均 CPU | 17.07 | 21.45 | +25.7% |

高频 constraint 自身从 6.999 s 降到 0.793 s，约加速 8.82 倍；两条初始
constraint 路径约加速 9 倍。IPC/cache/TLB 基本不变，说明收益来自消除串行
wall time，而不是单核指令变快。输出逐位一致，checker PASS。

### 4.4 OpenMP 调度和任务几何

诊断显示移动 level 5--8 才承担主要计算，level 0 只有 9 个 block。主 RK block
phase 的利用率其实已有约 85%，所以“全程只用 16 个 CPU”并不表示 RHS 主区也只
用了 16 个线程。

我们先尝试仅更换 static/dynamic/guided，但当 30 blocks 对应 30 workers 时，
每个线程只有一个大任务，调度器无法接管其他线程的尾部。中间版本把 block 增加
到 60 并使用 dynamic,1，曾得到约 3.05% 收益；但是在后续版本重新校准几何时：

| 几何 | Evolve t=0..4 |
|---|---:|
| 24/30 blocks | 29.899 s |
| 60/60 blocks | 31.864 s |
| 90/90 blocks | 35.511 s |

block 过多会增加 ghost zone、边界、Sync 和调度工作，所以最终恢复静态层
24、移动层 30，并保留 dynamic,1 处理不同 block 的不均衡。

60 个 OpenMP worker 使用 SMT sibling 时比 30 worker 慢约 36.7%，IPC 约从
1.47 降到 0.76，dTLB miss 从约 3.62% 升到 10.46%。因此最终使用 30 个物理核，
而不是追求表面上的 60 个活跃硬件线程。

最终诊断中 level 1--4 利用率约 79%--80%，level 5--8 约 84%--89%，level 0
约 30%。剩余空转主要来自 AMR 层级依赖、粗层 block 上限和阶段间 Sync，不是
绑核失败。跨整个 RK4 的持久 parallel team 也实际测试过，但回退约 10%--12%，
因为真实 barrier 仍存在，整个 team 反而一起等待，所以没有保留。

### 4.5 stencil SIMD

通信热点消除后，profile 的主要热点转移到 compute_rhs_bssn、kodis、fdderivs、
lopsided、fderivs 和 memcpy/memset。GCC 报告显示 kodis 和导数循环中的边界
分支阻碍自动向量化。

优化方法是把规则内点和边界薄层分开，只对 Fortran 连续存储的 i 方向内点使用
!$omp simd；边界公式和每个点的浮点运算顺序不变。

| 函数 | profile 契机 | Evolve 收益 | 正确性 |
|---|---|---:|---|
| kodis | 8.13% 热点，原循环无向量指令 | 7.43% | 逐位一致 |
| fdderivs | 8.03%，混合二阶导数热行 | 2.94%--3.21% | PASS |
| fderivs | 一阶导数仍为热点 | 1.31% | 逐位一致 |
| lopsided | 符号分支阻碍 SIMD | 约 +0.006%，持平 | PASS |

kodis 优化后 self samples 从 8.13% 降到约 2.29%，fdderivs 从 8.03% 降到
4.13%，说明确实命中了目标热点。lopsided 虽然生成了 SIMD，但同时计算正负
两套 stencil，额外访问抵消了向量收益，所以不能把它写成有效加速。

### 4.6 减少数据搬运和 TLB 压力

新的 profile 中 memcpy、memset、Sync 和 AMR transfer 变成第二类热点。每项
改动都先证明读写区域是否重叠，并保留安全 fallback：

| 优化 | 针对瓶颈 | 收益/指标变化 | 决定 |
|---|---|---|---|
| symmetry ghost 去掉冗余清零 | memset | Evolve 0.60%，Total 1.04% | 保留 |
| 同级 Sync direct copy | 两次 pack/unpack 和 memcpy | Evolve 约 1.02% | 保留 |
| prolong3 相邻点复用 | AMR 插值重复输入 | Evolve 约 1.27% | 保留 |
| block field arena | 分散字段和 dTLB | Evolve 0.65%，dTLB 3.7% -> 1.4% | 保留 |
| direct AMR transfer | pack/unpack | 约 0.24%，低于噪声 | 关闭 |

同级 Sync 只有在几何证明源/目标不重叠时才直接复制，存在读后写风险的 operation
仍使用旧 pack-before-unpack。arena 只改变持久字段的分配位置和生命周期，不改变
数组索引或数学计算。上述保留项均逐位一致。

### 4.7 RHS 局部复用：从失败融合到 Ricci 行分块

后期 profile 中 compute_rhs_bssn 占 41%--51%，六个 Ricci 大表达式合计约
16.76%，是最值得处理的计算热点。编译器已经能够沿 i 方向生成 128-bit NEON，
问题不是“完全没有向量化”，而是六次完整三维遍历反复读取相同度规、连接系数
和导数，输入在不同分量之间难以留在近端缓存。

先测试了多种大范围融合：

- 18 个 connection 一次融合：回退约 2.0%；
- 分组 connection 融合：仍回退约 0.7%；
- Aij 六字段融合：回退约 0.76%；
- chi/Ricci producer-consumer 融合：回退约 0.37%--1.47%；
- 三字段 first-kind connection 融合：回退约 0.56%。

这些实验减少了循环次数，却扩大了 live set、地址流和寄存器压力。由此把方案
改成小范围复用：只融合两个 first-kind connection，Evolve 提升约 1.13%；
六个 Ricci 不合成一个巨型循环，而是按 j=1 行分块：

~~~text
k
  -> 一个 j 行
       -> Rxx / Ryy / Rzz / Rxy / Rxz / Ryz
            -> 连续 i SIMD
~~~

交错 ABBAAB 结果为：

| 版本 | Evolve 均值 |
|---|---:|
| 原始六次完整数组遍历 | 28.9043 s |
| j=1 行分块 | 26.8057 s |

Evolve 下降 7.26%。instructions 只增加 0.14%，IPC 从约 1.43 升到 1.55，
cycles 下降 8.11%，LLC load misses 下降 15.88%，六个 Ricci 热行的样本从
16.76% 降到 9.55%。因此收益来自缓存复用，而不是少算公式或增加线程。j=2/4/8
逐渐变慢，说明 tile 变大后工作集和复用距离反而增加，最终只保留 j=1。

六个表达式和每个点的浮点顺序没有改变，所有 A/B 输出逐位一致，课程检查 PASS。
RHS/RK4 融合没有实施：RHS 必须跨 RK stage 保存，而 RK routine 只占约 2%，
融合仍需保存 RHS 并会扩大 live set，没有可信收益。

### 4.8 没有采用的实验

失败尝试也反映了对瓶颈的判断过程：

- fderivs 小批量慢约 0.07%，lopsided batch 慢 1.1%--1.6%；
- 导数只清零边界虽然正确，但慢 1%--2%；完全跳过清零会导致数值检查失败；
- THP 在独立字段上慢 1.45%，在 arena 上仍慢 0.35%；
- 64-byte 对齐慢 0.86%，LTO 只快 0.32%，-mtune=native 基本持平；
- 显式 SVE 编译参数慢约 13.5%，IPC 明显下降；
- direct AMR、prolong3 显式 SIMD 和多种大 RHS 融合低于噪声或回退；
- 没有更换数学库，因为最终热点是项目内部 stencil/RHS，不是 BLAS、FFT 或 libm。

这些结果说明当前程序并非只要“更多向量、更大 batch、更多线程”就会更快。大数组
工作集和 AMR 边界成本会让一些理论上减少循环的改动在实际机器上变慢。

## 5. 最终 profile、性能和正确性

最终 profile 使用当前提交、严格 -O3 -g、单进程 30 OpenMP 线程、24/30
block 和 dynamic,1。perf record 采集 66,284 个样本，无丢样：

| 最终热点 | cycles/sample |
|---|---:|
| compute_rhs_bssn | 47.08% |
| memcpy | 10.42% |
| lopsided_core | 9.78% |
| memset | 5.89% |
| fdderivs | 4.58% |
| kodis | 2.82% |
| prolong3_pair_kernel | 2.75% |
| fderivs | 2.60% |

最终硬件指标为 IPC 1.54、branch miss 0.49%、L1D miss 3.97%、LLC load miss
48.29%、dTLB miss 1.44%，CPU migration 和 context switch 均为 0。分支预测、
TLB 和绑核没有异常；剩余主要瓶颈是 RHS 数组流、memcpy/memset 以及不可避免的
AMR 层级同步。

最终完整 t=40 作业 166233：

| 指标 | 最终结果 |
|---|---:|
| ABE Evolve | 271.451 s |
| ABE Total Running | 275.067 s |
| This Program Cost | 286.162 s |
| 外层 wall | 295 s |
| 峰值内存 | 约 3.84 GiB |
| trajectory RMS | 0 |
| constraints | 40 个时间组 x 9 个 level 全部 PASS |
| checker | FINAL: PASS |

## 6. 与最开始 baseline 的最终比较

最开始 baseline 没有在时限内完成 t=40，所以最终比较必须注明哪些是实测、
哪些是线性外推：

| 部分 | 最开始 baseline | 最终版本 | 变化 |
|---|---:|---:|---:|
| TwoPuncture 独立 profile | 286.5 s，1 CPU | 约 11.4 s，30 OMP threads | 约 25x |
| ABE t=0..4 | 173.669 s，30 MPI ranks | 26.538 s，1 x 30 OMP | 约 6.55x |
| ABE t=0..40 | 约 1724 s，线性外推 | 271.451 s，完整实测 | 约 6.35x |
| 完整流程 | 约 2020 s，线性外推 | 286.162 s，完整实测 | 约 7.1x |

最终完整流程相对最开始 baseline 的粗略时间下降约 86%。这个 7.1x 是由
baseline 已完成的 33 个时间单位外推得到，不能写成同一次完整 A/B；各项优化的
因果收益应以前文同节点交错实验为准。最终版本则是真正跑满 t=40，并通过全部
正确性检查，低于 330 s 目标。

## 7. 总结

优化过程发生了三次清晰的瓶颈转移：

1. 最开始 TwoPuncture 是单线程、重复 cos 和 line solve 热点；ABE 则主要在
   MPI shared-memory progress、AnalysisStuff 的 Allreduce 和 rank 等待；
2. TwoPuncture 缓存系数并行化、ABE 改为完整 OpenMP 后，热点转移到串行
   constraint、RHS/stencil 和 AMR 数据搬运；
3. constraint 并行、SIMD、direct Sync、arena 和 Ricci 行分块之后，剩余成本
   集中在 compute_rhs_bssn 的数组流、memcpy/memset 和真实 AMR 层级依赖。

最终保留的优化都有 profile 证据、可重复的时间方向和正确性回归。没有为了
追求更高表面 CPU 使用率而删除必要 barrier，也没有采用不稳定的 -Ofast、显式
SVE、过量 block 或大范围公式融合。
