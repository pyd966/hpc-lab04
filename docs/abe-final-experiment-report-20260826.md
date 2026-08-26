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

## 4. ABE 优化：沿着 profile 结果逐步转移瓶颈

ABE 的优化并不是同时尝试很多无关方案，而是每解决一个主要瓶颈就重新 profile，
再决定下一步：

~~~text
MPI communication / Allreduce / rank imbalance
  -> 单进程 OpenMP，去掉单节点 MPI overhead
  -> profile 暴露串行 analysis、constraint 和任务长尾
  -> 扩大 OpenMP 覆盖范围，重新设计任务粒度和调度
  -> profile 转移到 RHS、导数和耗散 stencil
  -> SIMD 向量化
  -> profile 转移到重复数组流、memcpy/memset 和 AMR transfer
  -> 减少内存搬运、改善缓存和 TLB 局部性
  -> 用编译参数实验确认没有遗漏低成本收益
~~~

因此下面的标题按优化方法组织，而不是按某个函数名组织。

### 4.1 并行模型优化：从 MPI 转为单进程 OpenMP

最开始 profile 中 MPI/OpenPAL 占据约 82% 的 cycles，最热路径是
AnalysisStuff -> Interp_Points -> PMPI_Allreduce。level 0 只有约 9 个 rank
有实际 block，其余 rank 主要在 collective 和 progress 中等待。这说明首先
应该降低单节点 MPI 通信、同步和静态 rank 所有权的开销，而不是先优化只占
6.1% 的 compute_rhs_bssn。

因此 ABE 改为一个进程、30 个 OpenMP 线程。OMP-only 构建不启动 MPI runtime，
也不链接 libmpi；保留的少量 MPI 类型接口只执行单进程本地语义。

但是第一次转换失败了：只把部分 RHS/block 循环套上 OpenMP 后，t=0..4
Evolve 从 MPI baseline 的 173.669 s 增加到 542.358 s，平均只使用
3.23/30 个 CPU。原因是 MPI 版本中的并行不只存在于 RHS：不同 rank 还同时
执行 Sync、copy、restrict、prolong 和分析工作。只删除 MPI 后，这些工作全部
集中到主线程，原来的 rank 并行没有自动变成线程并行。

完整转换因此包含：

- Step 的 predictor 和三个 corrector 按 block 并行 RHS、RK 更新和边界处理；
- 将 Sync、copy、restrict3、prolong3 展开为 grid segment x variable
  operation，由 OpenMP 线程处理；
- pack 阶段并行读取全部源数据，barrier 后再写目标，保持原 MPI 版本的
  先读后写语义；
- RecursiveStep 仍按 AMR 层级和时间细分顺序执行，不并发存在真实依赖的 level；
- 错误标志等小 collective 改为 OpenMP reduction 或本地操作。

| 版本 | 进程 x 线程 | Evolve t=0..4 |
|---|---:|---:|
| 最初 MPI baseline | 30 x 1 | 173.669 s |
| 不完整的 OpenMP 初版 | 1 x 30 | 542.358 s |
| 完成 block 和 transfer 转换 | 1 x 30 | 61.883 s |
| 复用 transfer team/workspace | 1 x 30 | 61.477 s |

完整转换后平均 CPU 从 3.23 提高到约 16.4，输出保持一致。此时重新 profile，
MPI/OpenPAL 热点已经消失；新的主要问题是分析阶段数据量过大、一些 constraint
路径仍然串行，以及不同 AMR level/block 的工作量不均衡。所以下一步仍然属于
OpenMP 优化，但目标从“替代 MPI”变成“扩大并行覆盖并改善任务调度”。

### 4.2 任务级并行优化：扩大 OpenMP 覆盖范围和任务粒度

这一阶段把所有能够证明独立、同时又具有足够工作量的任务纳入 OpenMP。这里的
AnalysisStuff 改动确实属于 OpenMP 并行化；所谓“数据流重写”是为了让点级并行
不再依赖大数组 Allreduce，并不是一个与 OpenMP 无关的优化方向。

#### 4.2.1 增大并行粒度：从 rank/block 所有权改为点级任务

MPI 版本的 AnalysisStuff 按 block/rank 所有权工作。球面积分点只落在少数
block 上，因此只有少数 rank 做有效插值，之后所有 rank 对完整 shellf 中间
数组执行 Allreduce。

OpenMP 版本把并行单位改成球面点：

~~~text
全部球面点
  -> OpenMP 线程领取独立点
  -> 使用预计算的所属 block、插值下标和权重
  -> 插值后立即累加到线程私有 wave/ADM 结果
  -> 最后只归约很小的线程私有结果
~~~

这里同时做了两件互相依赖的事情：第一，使用 OpenMP 让球面点并行；第二，
改变数据流，不再生成和归约几十 MB 的 pox/shellf。缓存插值计划和 wave 系数
则减少每个时间步重复搜索 block、调用通用 polint 和重算三角系数。

| 版本 | Evolve t=0..4 | ABE Total |
|---|---:|---:|
| 完整 OpenMP transfer | 61.477 s | 68.761 s |
| 点级 OpenMP + 局部归约 | 43.869 s | 51.249 s |

Evolve 下降约 28.6%，相对最初 MPI baseline 已约快 4 倍。演化状态保持一致；
Psi4 只在理论零项出现约 1.48e-22 的舍入差异，课程 checker 通过。

#### 4.2.2 补齐并行覆盖：处理 cycles profile 容易忽略的串行区

完成上述改动后，全程平均 CPU 仍偏低，但普通 cycles profile 没有显示一个足够
大的串行函数。这是因为一个串行函数运行 7 秒只产生一个核的 cycles，在同时
存在 30 核并行区的 profile 中占比会被稀释。

为此增加 phase wall-time，分别统计 Step、Sync、transfer、regrid 和 constraint。
结果发现 Constraint_Out 每个物理时间单位都会在主线程上逐 block 重算 RHS，
短程约占 7 秒。于是把高频 Constraint_Out，以及 Evolve 前的
Compute_Constraint、Interp_Constraint RHS 重算按 block 并行：

| 指标 | 修改前 | 修改后 | 变化 |
|---|---:|---:|---:|
| Evolve t=0..4 | 37.682 s | 30.171 s | -19.9% |
| ABE Total | 44.405 s | 35.579 s | -19.9% |
| 全程平均 CPU | 17.07 | 21.45 | +25.7% |

高频 constraint 从 6.999 s 降到 0.793 s，约加速 8.82 倍；两条初始
constraint 路径也分别加速约 9 倍。IPC、cache 和 TLB 基本不变，证明收益来自
消除串行 wall-time，而不是改变单核计算效率。

这部分在 MPI 版本中存在跨 rank 的 block 并行，所以严格地说既包含“恢复原
MPI 并行”，也包含“单进程内更细粒度的 OpenMP 调度”。真正属于 MPI 版本没有
充分使用的新粒度，主要是球面点级 AnalysisStuff 并行和 operation 级 transfer。

#### 4.2.3 调度优化：让已有任务尽量均衡，但不过度切分

OpenMP 诊断显示主 RK block phase 的综合利用率已经约 85%；全程平均 CPU 较低
还包含 level 0 只有 9 个 block、层间递归、Sync 和输出。因此不能只根据
“平均不到 30”就继续增加线程。

先比较 static、dynamic 和 guided。30 blocks 对应 30 workers 时，每个线程只有
一个大任务，换 schedule 几乎没有收益。中间版本增加到 60 blocks 并使用
dynamic,1，曾在当时的代码上改善 level 7/8 长尾约 3.05%。但是后续工作量变化
后重新校准发现：

| 最终几何 sweep | Evolve t=0..4 |
|---|---:|
| 24/30 blocks | 29.899 s |
| 60/60 blocks | 31.864 s |
| 90/90 blocks | 35.511 s |

更多 block 虽然提供更多任务，却增加 ghost zone、边界、Sync 和调度工作。
所以最终配置是静态层 24 blocks/threads、移动层 30 blocks/threads，并使用
dynamic,1 处理同一层中不同 block 的工作量差异。

60 个 OpenMP worker 使用 SMT sibling 时比 30 worker 慢约 36.7%，IPC 从约
1.47 降到 0.76，dTLB miss 从约 3.62% 升到 10.46%。最终使用 30 个物理核，
而不是追求表面上 60 个活跃硬件线程。

跨整个 RK4 的持久 OpenMP team 也做过实验，但稳定回退约 10%--12%。RK、Sync
和层间递归中的真实 barrier 仍然存在，持久 team 只是让整个线程组一起等待，
没有创造新的可执行任务，因此没有保留。

这一阶段结束后的 profile 中，MPI 和明显串行区已经不再是主要问题。
compute_rhs_bssn、kodis、fdderivs、lopsided 和 fderivs 成为主要热点，
所以优化方法自然转向 SIMD。

### 4.3 指令级并行优化：对规则 stencil 做 SIMD 向量化

GCC vectorization report 显示，kodis 和导数循环的三维边界判断阻碍自动
向量化。Fortran 数组第一维 i 连续，固定 j/k 后的内点之间没有写依赖，因此
把规则内点和边界薄层分开，只对连续 i 方向内点添加 !$omp simd。边界公式、
有效点集合和每个点的浮点运算顺序保持不变。

| 函数 | profile 证据 | Evolve 收益 | 热点变化 |
|---|---|---:|---|
| kodis | 8.13%，原目标文件无向量浮点指令 | 7.43% | self 约 8.13% -> 2.29% |
| fdderivs | 8.03%，混合二阶导数热行 | 2.94%--3.21% | self 约 8.03% -> 4.13% |
| fderivs | 一阶导数仍为热点 | 1.31% | 明显下降 |
| lopsided | 符号分支阻碍 SIMD | 约 +0.006%，持平 | 不宣称收益 |

前三个保留版本都由编译器生成 128-bit NEON 双精度向量指令，并通过逐位输出和
课程 checker。没有手写 AArch64 intrinsic，因为编译器在规则内点上已经生成
正确向量代码，额外的 C/Fortran 接口没有 profile 支持。

lopsided 是重要的反例：虽然新循环生成了 SIMD，但它同时计算正、负两套
stencil，再根据 shift 符号选择，额外访存抵消了向量收益。说明“成功向量化”
不等于“端到端一定加速”。

SIMD 之后，分支 miss 仍约 0.5%，不是问题；RHS 总体、memcpy/memset 和 AMR
transfer 的占比相对上升。下一步因此不是继续随意加 SIMD pragma，而是减少
数组搬运并提高缓存局部性。

### 4.4 内存访问优化：减少搬运并提高缓存、TLB 局部性

这一阶段把所有“减少内存流量或缩短数据复用距离”的方案合并在一起。profile
显示 memcpy/memset、Sync/AMR transfer 和 compute_rhs_bssn 的重复数组流已经
成为主要成本。

#### 4.4.1 减少不必要的数据写入和中间复制

| 方法 | 针对瓶颈 | 收益/指标变化 | 决定 |
|---|---|---|---|
| 去掉 symmetry ghost 冗余清零 | memset | Evolve 0.60%，Total 1.04% | 保留 |
| 同级 Sync direct copy | 两次 pack/unpack 和 memcpy | Evolve 约 1.02% | 保留 |
| prolong3 相邻点复用 | AMR 插值重复读取 | Evolve 约 1.27% | 保留 |
| direct AMR transfer | pack/unpack | 约 0.24%，低于噪声 | 关闭 |

同级 Sync 只有在几何证明源、目标不重叠时才直接复制；存在读后写风险的
operation 仍回退到旧的 pack-before-unpack。它利用了 OpenMP 单地址空间，
但主要收益来自减少中间内存搬运，而不是增加线程数。

导数数组只清零边界也做过实验。它虽然减少写入字节，却慢约 1%--2%，原因是
多个不连续小 memset 和额外控制流比一次连续写更差；完全跳过清零则破坏边界
零值语义，正确性检查失败。因此最终保留整数组连续清零。

#### 4.4.2 改善字段布局和 TLB 局部性

把一个 block 的多个持久字段放入连续 arena，代替大量分散分配。该方法不改变
数组索引和数学计算，只缩短字段地址距离并减少页表工作。

Evolve 改善约 0.65%，dTLB miss 从约 3.7% 降到 1.4%，所有输出逐位一致。
在 arena 之后继续启用 transparent huge page 反而慢约 0.35%；单独字段使用
THP 慢约 1.45%，64-byte 对齐也慢约 0.86%。说明 arena 已经解决了主要 TLB
问题，更激进的页和对齐策略没有额外收益。

#### 4.4.3 用循环分块和小范围融合提高 RHS 缓存复用

后期 profile 中 compute_rhs_bssn 占 41%--51%，六个 Ricci 大表达式合计约
16.76%。这些循环已经沿 i 方向生成 NEON，所以目标不是再次“强制 SIMD”，
而是减少六次完整三维遍历对相同度规、连接系数和导数的重复读取。

最先尝试大范围 loop fusion，但结果均回退：

- 18 个 connection 一次融合：约慢 2.0%；
- 分组 connection 融合：仍慢约 0.7%；
- Aij 六字段融合：慢约 0.76%；
- chi/Ricci producer-consumer 融合：慢约 0.37%--1.47%；
- 三字段 first-kind connection 融合：慢约 0.56%。

这些方案虽然减少数组遍历，却同时扩大 SIMD live set、地址流和寄存器压力。
因此改用更保守的局部复用：

1. 只融合两个 first-kind connection 字段，Evolve 提升约 1.13%；
2. 六个 Ricci 不合成一个巨型循环，而是按 j=1 行分块；每个分量仍使用独立
   连续 i SIMD 循环，一行的六个分量完成后才进入下一行。

~~~text
k
  -> 一个 j 行
       -> Rxx / Ryy / Rzz / Rxy / Rxz / Ryz
            -> 连续 i SIMD
~~~

| 版本 | Evolve 均值 |
|---|---:|
| 原始六次完整数组遍历 | 28.9043 s |
| j=1 行分块 | 26.8057 s |

行分块使 Evolve 下降 7.26%。instructions 只增加 0.14%，IPC 从约 1.43
升到 1.55，cycles 下降 8.11%，LLC load misses 下降 15.88%，六个 Ricci
热行的样本从 16.76% 降到 9.55%。这证明收益来自缩短输入复用距离，而不是
少算公式。j=2/4/8 逐渐变慢，因此最终只保留 j=1。

RHS/RK4 融合没有实施：RHS 需要跨 RK stage 保存，而 RK routine 只占约 2%；
融合后仍要保存完整 RHS，还会扩大 live set，没有可信的收益上限。

### 4.5 编译器和 ISA 参数实验：验证是否存在低成本尾部收益

在并行结构、SIMD 和内存访问都稳定以后，最后统一测试编译器和 ISA 参数，而
不是在瓶颈尚未明确时依赖“魔法 flags”：

| 实验 | 结果 | 最终决定 |
|---|---:|---|
| LTO | 约快 0.32%，低于波动阈值 | 关闭 |
| -mtune=native | 基本持平 | 关闭 |
| 显式 SVE 架构参数 | 慢约 13.5%，IPC 明显下降 | 关闭 |
| -Ofast / fast-math | 浮点语义风险，无 ABE profile 必要性 | 不使用 |
| 更换 BLAS/FFT/libm | profile 中没有对应热点 | 不更换 |

显式 SVE 失败是因为编译器把原本稳定的短 NEON 循环改成 variable-length SVE
路径，当前数组长度、边界和地址流无法摊薄额外开销。最终 ABE 使用严格 -O3；
热点是否得到改善由源码级 A/B 决定，而不是由更激进的全局 flags 决定。

### 4.6 ABE 优化主线总结

从 profile 的变化看，ABE 的优化顺序是连贯的：

| 阶段 | profile 判断 | 采用的方法 | 瓶颈转移 |
|---|---|---|---|
| baseline | MPI/OpenPAL 约 82%，Allreduce 等待 | MPI -> 单进程 OpenMP | 转向串行 analysis/constraint |
| OMP 初版 | 平均仅 3.23 CPU，transfer 串行 | 补齐 block/operation 级并行 | 转向任务粒度和负载均衡 |
| OMP 稳定版 | analysis 大归约、constraint 串行 wall | 点级并行、线程私有归约、动态调度 | 转向 RHS/stencil |
| 计算热点阶段 | kodis/derivative 未向量化 | 规则内点 SIMD | 转向数组流和数据搬运 |
| 内存热点阶段 | RHS、memcpy/memset、LLC/dTLB | direct copy、arena、局部融合和行分块 | 剩余 RHS/AMR 固有成本 |

所有保留改动都经过短程 A/B 和数值回归。没有收益的小批量导数、lopsided
batch、THP、过度对齐、大范围 RHS 融合、显式 SVE 和持久 RK team 均被关闭。

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
