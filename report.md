# Lab 4 AMSS-NCKU CPU/GPU 优化实验报告（技术参考草稿）

> 本文用于汇总真实代码、profile 和实验数据，方便后续整理正式报告。课程文档明确要求报告和思考题由本人完成，因此提交 PDF 前必须逐项核对，并按自己的理解与实际经历改写，尤其不要直接提交本节思考题文字。

## 1. 实验目标与方法

本实验包含两条固定路线：

- CPU：AArch64 节点，演化时间 t=0..40，目标是降低端到端时间。
- GPU：NVIDIA A100 MIG 1g.10gb，16 CPU、24 GiB host memory、sm_80，演化时间 t=0..100，目标是 This Program Cost <= 370 s。
- 两条路线都使用课程固定物理参数、9 层 AMR 网格和 FP64。
- 短程 t=0..4 或 t=0..5 只用于 profile、A/B 筛选和估算；最终结论来自完整规定输入。
- 每个保留优化都检查 checker；GPU 最终版本额外做了三次无 profiler 完整运行。

优化过程按瓶颈转移组织：

~~~text
建立 baseline 和正确性基线
  -> 先解决并行模型与调度问题
  -> 再解决最热数值 kernel
  -> 再减少重复访存、计算和 launch
  -> 最后做有数据依赖依据的小范围融合
  -> 完整输入多次验收
~~~

这种顺序的目的，是避免在旧热点只占很小端到端比例时过早做微优化，也避免把多个改动混在一次实验中而无法判断因果。

## 2. 最终配置与结果

### 2.1 CPU

实际作业 cpuset 对应 30 个物理核，位于同一个 NUMA node。最终配置为：

~~~text
MPI_processes = 1
OMP_NUM_THREADS = 30
OMP_PLACES = cores
OMP_PROC_BIND = close
OMP_SCHEDULE = dynamic,1
静态层：24 blocks / workers
移动层：30 blocks / workers
ABE：-O3
TwoPuncture：-O3 -march=native
~~~

最初 30 MPI rank baseline 的完整程序只能外推到约 2020 s；最终完整 t=40 结果为：

| 指标 | 最终结果 |
|---|---:|
| TwoPuncture | 约 11.4 s |
| ABE Evolve | 271.451 s |
| ABE Total Running | 275.067 s |
| This Program Cost | 286.162 s |
| checker | PASS |
| trajectory RMS | 0 |

相对最初 baseline 的外推值，端到端约加速 7.1 倍。baseline 没有完整跑完，所以 7.1 倍只能写成“基于已完成时间段的外推比较”，不能伪装成完整同作业 A/B。

### 2.2 GPU

最终配置为：

~~~text
MPI_processes = 1
OpenMP for ABEGPU = OFF
TwoPuncture OpenMP threads = 16
CUDA architecture = sm_80
CUDA separable compilation = ON
CUDA flags = -O3 -rdc=true -lineinfo
MIG = A100 1g.10gb，14 visible SM
~~~

正式 t=100 三次结果：

| Run | Program Cost (s) | Evolve (s) | checker |
|---:|---:|---:|---|
| 1 | 360.864865 | 331.528 | PASS |
| 2 | 359.853815 | 330.663 | PASS |
| 3 | 363.817126 | 333.419 | PASS |
| mean | 361.511935 | 331.870 | 3/3 PASS |
| sample stddev | 2.059365 | - | - |

最慢一次 363.817 s，仍低于 370 s。三次均完成 100/100 trajectory，trajectory RMS 为 0；596 个轨迹点与 9 层输出完整。Grid Level 0 的最大约束值为 Ham 0.03726、Px 0.01725、Py 0.01736、Pz 0.03760，远低于阈值 2.0。峰值显存约 2063 MiB。

## 3. 程序热点性质

AMSS-NCKU 不是以稠密 BLAS 为主的程序。TwoPuncture 包含谱变换和线性迭代，但主演化 ABE/ABEGPU 的核心是三维有限差分 stencil、BSSN 点上代数、AMR prolong/restrict、边界同步和分析插值。

CPU baseline 的第一热点却不是 stencil，而是通信和等待：MPI/OpenPAL 约占 82%，mca_btl_sm_poll_handle_frag 单项为 58.98%，PMPI_Allreduce inclusive 为 74.52%，compute_rhs_bssn 当时仅 6.10%。这说明旧 CPU 版本首先受错误的单节点并行模型和负载不均衡限制。

GPU baseline 中，Nsys 的 kernel 时间分布为：

| Kernel family | t=0..4 时间 | GPU kernel 占比 |
|---|---:|---:|
| rhs_kernel | 56.906 s | 77.3% |
| prolong3 | 8.870 s | 12.1% |
| restrict3 | 2.918 s | 4.0% |
| global interpolation | 2.009 s | 2.7% |

因此 GPU 主体确实是 stencil/点上数据流；同时存在显著调度问题。VTune 中 cuCtxSynchronize_v2 约占 host CPU 时间 81.5%，Nsys 观察到约 20 万次 kernel launch。单 rank MPI_Allreduce 只占 host 时间约 0.1%，不是 GPU 路线主瓶颈。

## 4. CPU 优化主线

### 4.1 TwoPuncture：先消除重复工作，再并行

#### 4.1.1 热点临时内存复用

启发证据：profile 中热点调用路径的 malloc/free 约占 6%，chebft、fourft、Derivatives_AB3、F_of_v、J_times_dv、LineRelax 和 ThomasAlgorithm 反复分配短数组。

目的与修改：建立线程私有 TransformWorkspace、PointWorkspace 和 LineWorkspace，复用 scratch；这既减少 allocator 工作，也为后续 OpenMP 避免共享 scratch race。

结果：instructions 下降 3.65%，cycles 下降 2.08%，但该次作业 CPU 频率低 2.84%，wall time 从 286.505 s 变成 288.720 s。这个结果不能写成 wall-clock 加速，只能说明工作量下降而集群波动掩盖了收益。allocator 从新 profile 热点中消失，输出逐位一致，因此保留。

#### 4.1.2 缓存三角与谱变换系数

启发证据：cos 自身占 TwoPuncture 23.17%，而固定 nA、nB、nphi 后角度只依赖网格尺寸和循环下标。

目的与修改：初始化时预计算 Chebyshev forward/inverse cosine 表和 Fourier sine/cosine 表，后续读取缓存；保持原求和顺序，不改成 FFT。

结果：

| 指标 | 优化前 | 优化后 |
|---|---:|---:|
| wall time | 288.720 s | 198.918 s |
| cycles | 808.832 B | 569.511 B |
| instructions | 2067.696 B | 1233.737 B |
| cos self | 23.65% | <0.5% |

cos 热点基本消失，说明优化确实命中了预期瓶颈。第一次实现曾改变等价表达式的浮点顺序，导致末位差异；恢复原表达式后输出逐位一致。

#### 4.1.3 OpenMP 并行

启发证据：缓存系数后约 93% profile 集中在 LineRelax、Thomas 和 Derivatives_AB3，串行计算成为绝对瓶颈。

目的与修改：

- Derivatives_AB3 按独立谱线并行，三个方向之间保留依赖 barrier。
- F_of_v、J_times_dv 按网格点并行，scratch 线程私有。
- LineRelax 利用已有红黑 phase；同 phase 的线并行，phase 之间保留 barrier。
- 单条 Thomas 递推仍串行，通过同时处理多条线获得并行度。
- 把连续 200 次 relax 放进同一个 parallel region，减少 fork/join。

线程扫描中，1、4、8、16、30、60 线程分别约为 215.162、63.013、31.114、16.499、10.375、10.363 s。60 个逻辑线程相对 30 个物理核没有可靠端到端收益，因此最终选 30。最终严格 O3 复测约 11.4 s，相对最初约 286.5 s 加速约 25 倍。

### 4.2 ABE：MPI 到 OpenMP

#### 4.2.1 更换单节点并行模型

启发证据：MPI/OpenPAL 约 82%，level 0 只有约 9 个 rank 持有有效 block，其他 rank 在 collective/progress 中等待。

目的：在单 NUMA、共享地址空间内消除 rank 私有内存、pack/unpack、Allreduce 和 MPI progress 开销，同时保留 AMR/RK 的真实依赖。

修改范围：

- Step predictor 和三个 corrector 按 block 并行。
- Sync、copy、restrict、prolong 展开为 segment x variable operation。
- 先并行 pack 全部源数据，barrier 后写目标，保持旧 MPI 的先读后写语义。
- 小 collective 改为 OpenMP reduction 或单进程本地语义。
- RecursiveStep 的 AMR 层级与时间细分顺序不变。

重要反例：只并行 RHS 的不完整 OpenMP 版本从 173.669 s 退化到 542.358 s，因为 MPI 旧版还隐含并行了 transfer、Sync 和分析。补齐全部区域后，t=0..4 Evolve 降到 61.477 s。这个失败说明“删掉 MPI”本身不是优化，必须把原 rank 并行覆盖完整地迁移到线程。

#### 4.2.2 AnalysisStuff 点级并行

启发证据：PMPI_Allreduce inclusive 74.52%，分析阶段每次把约 47.2 MB 中间数组交给 collective，而球面积分点只落在少数 block。

目的与修改：并行单位从 block/rank 改成球面点；预计算点所属 block、插值下标和权重；每个线程直接累计私有 wave/ADM 小结果，最后只归约小数组。

结果：t=0..4 Evolve 从 61.477 s 降到 43.869 s，下降 28.6%。大 Allreduce 与 MPI 热点消失；Psi4 只有理论零项约 1.48e-22 的舍入差，checker PASS。

#### 4.2.3 Constraint 并行与 OpenMP 调度

启发证据：phase wall-time 发现 Constraint_Out 每个时间单位都在主线程串行重算 RHS，约占短程 7 s；普通 cycles profile 会被 30 核并行区稀释，难以显示这个串行 wall-time。

修改：把 Constraint_Out、Compute_Constraint 和 Interp_Constraint 按 block 并行。Evolve 从 37.682 s 降到 30.171 s，高频 constraint 从 6.999 s 降到 0.793 s，全程平均 CPU 从 17.07 提高到 21.45。

随后扫描任务几何：

| 几何（线程仍为 24/30） | Evolve t=0..4 |
|---|---:|
| 24/30 blocks | 29.899 s |
| 60/60 blocks | 31.864 s |
| 90/90 blocks | 35.511 s |

更多 block 增加 ghost cell、边界、Sync 和 AMR 操作，负收益。因此最终静态层 24、移动层 30，dynamic,1 只处理有限长尾。60 个 SMT worker 也使 IPC 和 dTLB 指标恶化，未采用。

### 4.3 SIMD 与内存访问

GCC vectorization report 指出 kodis、fdderivs、fderivs 的三维边界判断阻碍连续 i 方向自动向量化。把内点与边界薄层分开，并对内点使用 OpenMP SIMD 后：

- kodis 端到端提升约 7.43%，self 约 8.13% 降到 2.29%。
- fdderivs 提升约 3%。
- fderivs 提升约 1.31%。
- lopsided 虽成功生成 SIMD，却因同时计算正负 stencil 增加访存而基本持平，未把它宣称为收益。

后续 profile 中 memcpy、memset、AMR transfer 和 RHS 数组流相对上升，因此做了：

- 去掉 symmetry ghost 冗余清零：Evolve 约提升 0.60%。
- 几何证明不重叠时做同级 Sync direct copy：约提升 1.02%。
- prolong3 相邻点复用：约提升 1.27%。
- block 多字段连续 arena：Evolve 约提升 0.65%，dTLB miss 从约 3.7% 降到 1.4%。
- Ricci 按 j=1 行分块，缩短六个分量复用同一输入的距离：Evolve 28.9043 -> 26.8057 s，下降 7.26%；IPC 1.43 -> 1.55，LLC load misses 下降 15.88%。

大范围 loop fusion、THP、64-byte 强制对齐、显式 SVE、LTO、持久 OpenMP team 等方案没有稳定收益，均未作为默认优化。最终 CPU profile 中 compute_rhs_bssn 47.08%、memcpy 10.42%、lopsided 9.78%、memset 5.89%，说明瓶颈已从通信转移为真实 stencil 与内存流。

## 5. GPU 优化主线

### 5.1 baseline：寄存器压力决定先拆 RHS

baseline 短程 Program 约 122 s、Evolve 约 92 s。NCU 对原 rhs_kernel 的关键指标为：

| 指标 | baseline |
|---|---:|
| registers/thread | 250 |
| theoretical occupancy | 12.5% |
| achieved occupancy | 11.04% |
| No Eligible | 75.30% |
| compute utilization | 22.26% |
| DRAM utilization | 3.97% |
| L1TEX scoreboard stall | 38.71% |
| fixed-latency dependency stall | 34.26% |

DRAM 只有 3.97%，所以问题不是简单的 HBM 带宽饱和，而是高寄存器压力限制并发，无法隐藏访存与执行依赖。由此确定第一步不是盲目加 shared memory，而是按自然数据边界拆分 RHS，缩短 live range。

### 5.2 RHS fission 与数据流重构

#### 5.2.1 按自然边界拆 kernel

目的：把 metric/shift Hessian、advection、constraints 等相对独立阶段移出巨型 RHS，降低单 kernel live set，为后续专门优化创造边界。

结果：RHS family 从 13.9386 s 降到 10.4985 s，下降 24.7%；早期短程 Program 约到 101.667 s。拆分不是目的，寄存器生命周期和后续可优化性才是目的。

失败粒度也很重要：

- 强制拆成 9 个输出 kernel 仍有 170--201 regs/thread，并增加 scratch 和 launch，端到端慢约 4.6%。
- 过粗的另一种拆分引入更多中间量，慢约 13.2%。

结论是 fission 应尊重 producer-consumer 边界；仅按输出数量机械拆分不能解决真正的 live set。

#### 5.2.2 advection 公共量提升

代码观察发现 24 个变量重复计算同一坐标偏移和插值系数。把公共因子提到变量循环外后，寄存器从 85 降到 66，achieved occupancy 从 21.31% 提升到 32.51%，短程 Program 从 104.729 s 降到 100.297 s。

#### 5.2.3 Ricci producer-consumer 重构

两个 consumer 重复计算 9 个复杂一阶导数。增加小 producer kernel 一次写出导数后复用，使 consumer 从 240 降到 126 regs/thread，理论 occupancy 从 12.5% 提升到 25%；三者合计 2.191 ms -> 1.084 ms，下降 50.5%。

### 5.3 执行图、同步、stream 与 batching

#### 5.3.1 精细同步

启发证据：约 4110 次 cudaDeviceSynchronize，host 大量时间阻塞；但等待时间与 GPU kernel 时间重叠，不能直接相加。

修改：只让有真实依赖的 stream/event 等待，减少 device-wide barrier。

结果：device sync 次数约 1044 -> 215；device-sync 时间 13.955 -> 3.408 s，全部同步合计 13.955 -> 9.145 s。短程端到端约下降 4%。

#### 5.3.2 多 stream 和跨变量 batching

prolong3 有 47307 次小 launch，单次 grid 约 54 blocks，在 14 SM MIG 上存在尾部。对独立 patch/segment 使用多 stream 后，prolong 阶段约 2.87 -> 1.49 s。

多个物理变量执行完全相同 prolong/restrict，因此把变量维加入 grid，一次 launch 处理同一 segment 多个字段。AMR launch 数下降约 95%，短程再次下降约 4.4%。stream 解决 overlap，batching 解决 launch 粒度，两者目的不同。

### 5.4 shared-memory stencil tile

#### 5.4.1 compact advection

advection 半径为 3，邻点重复读取严重。使用 block (8,8,4)，协作加载 14x14x10 tile，shared memory 约 15.73 KiB/block。

结果：aggregate 2.480760 -> 0.828922 s，2.993 倍；短程 Program 82.166688 -> 74.633372 s，Evolve 53.366067 -> 45.080233 s。最终 NCU 为 78 regs/thread、achieved occupancy 35.28%。

#### 5.4.2 evolution、beta 与 prolong tile

针对各 consumer 只搬运真正需要的数据，并在可能时先 contraction 再进 shared memory：

| Kernel | 优化前 | 优化后 | 关键 NCU 变化 |
|---|---:|---:|---|
| evolution | 1.03 ms | 107.36 us | regs 140 -> 64，occupancy 约 45.79% |
| beta | 528.45 us | 95.04 us | regs 140 -> 72，occupancy 10.93% -> 34.61% |
| prolong | 432.86 us | 86.85 us | regs 59 -> 33，occupancy 45.49% -> 64.83% |

shared memory 并非越多越好。一次跨字段大 tile 使用 32.82 KiB shared memory、142 regs/thread，achieved occupancy 只有 12.43%，duration 约 870.53 us，未获得稳定收益，因此淘汰。保留的 evolution tile 虽有 3--4 way bank conflict，但端到端收益显著，说明 occupancy 或 bank conflict 单项都不能替代实际测量。

### 5.5 删除无用计算与专用插值

演化每次只需要两个黑洞位置的三个场，却调用通用 global interpolation，经历完整 H2D/D2H、同步和 reduction。实现 point3 专用 kernel 后，端到端约快 1%。

通用插值原来有 ya[216] 线程局部数组，被放入 local memory。改为 streaming 标量算法后，local memory sectors 98.82 -> 61.35，kernel 2.87 -> 1.77 ms；虽然 registers 64 -> 236、occupancy 降到约 12.5%，仍然更快。这是“高 occupancy 不是唯一目标”的另一证据。

Constraint_Out 只需要约束所依赖的几何与导数，却走完整 RHS 链；增加 constraints-only 路径约快 1.4%。另一个 predictor 产生的 constraint 在消费前会被覆盖，删除该未消费 producer 后，对应 kernel 时间下降约 89%。

### 5.6 在 fission 后有选择地 fusion

前期拆分是为了降低 live range 和看清数据流；当各阶段稳定后，再只融合直接 producer-consumer，减少中间显存流量与 launch：

- gamma derivative + seed：kernel family 下降约 45%，短程约下降 1.6%。
- chi/lapse derivative + source algebra：原聚合 2.208929 -> 0.874053 s，单位时间成本下降约 60.43%。
- compact beta + Gamma、geometry + Ricci-A、gauge + advection 等小范围融合继续降低中间流量。

最后 trace/metric/A 的融合在三次短程 Program 均值上只改善 0.100884 s，而样本标准差为 0.305619 s，不能宣称统计显著。保留依据是明确的数据依赖证明和 Evolve 指标方向，而不是夸大端到端数字。最终正式版本没有追加一轮完整 Nsys，所以报告也不能虚构“最终 profile 已证明所有热点消失”。

## 6. 正确性与浮点误差判定

优化后出现小差异时，按以下顺序判断：

1. 先确认优化是否改变运算集合、顺序、编译数学语义或并行归约顺序。纯地址复用理论上不应产生误差，而 reduction 重排通常会产生舍入差。
2. 同时看绝对误差、相对误差和 ULP，避免参考值接近 0 时只看相对误差。
3. 检查误差随时间是否平滑、有界，还是突然跳变、指数增长或出现 NaN/Inf。
4. 重复运行；同输入结果不稳定通常提示 data race、未初始化数据或错误同步。
5. 检查 trajectory、波形、所有 AMR level 约束和课程 checker，不能只比较一个文件。
6. 对高风险改动使用小网格/短时间 sanitizer、边界样例和 producer-consumer 检查。

本实验保留版本都通过 checker。TwoPuncture 的 Ofast 曾带来约 4e-15 最大差异，量级可由重排解释，但为了不把 fast-math 与源码优化混淆，最终仍使用严格 O3。

## 7. 思考题参考要点

### 7.1 热点更接近哪一类

结论随阶段和平台变化。CPU baseline 首先是通信/调度：MPI/OpenPAL 约 82%，而 RHS 仅 6.10%；改成完整 OpenMP 后，最终才转为 RHS stencil、memcpy/memset 和 AMR。GPU 从一开始就是 stencil/点上数据流为主体，rhs、prolong、restrict 合计超过 93%，同时伴随过多同步与小 launch。没有 profile 证据表明稠密线性代数是主导热点。

### 7.2 MPI rank 与 OpenMP thread 的权衡

最佳 CPU 配置是 1 process x 30 OpenMP threads。2 MPI x 15 OpenMP 的实测约 227.7 s，比 1 x 30 慢约 31%；30 MPI x 1 的旧版又被 Allreduce、rank 等待和重复内存拖慢。

纯 MPI 或纯 OpenMP 都可以实现并行，但适用边界不同。单 NUMA 共享内存内，纯 OpenMP 避免显式通信和数据副本；跨 NUMA 或跨节点时，MPI 提供必要的地址空间和网络通信，典型混合配置是每个 NUMA/domain 一个或少数 MPI rank，rank 内用 OpenMP。是否混合不是原则问题，应由内存拓扑、任务粒度和 profile 决定。本实验单节点用纯 OpenMP 更合理。

### 7.3 程序中的并行层级

优化版本使用或识别了以下层级：

- MPI/domain 层：可用于跨节点或跨地址空间，本实验 CPU 最终退化为一个 rank 语义。
- AMR block/patch 层：同一 RK phase 的独立 block 由 OpenMP 或 CUDA grid 并行。
- operation 层：Sync、restrict、prolong 按 segment x variable 并行。
- 分析点层：球面积分点独立插值与线程私有归约。
- 网格点层：规则 stencil 在 CPU 上沿连续 i 方向 SIMD，在 GPU 上由 CUDA threads 处理。
- 字段层：GPU batching 把多个物理变量加入同一 launch。
- TwoPuncture 谱线与红黑 phase：同 phase 的线并行，方向/颜色之间保留 barrier。
- CUDA stream 层：相互独立的 patch、AMR 操作和传输重叠。

AMR level 递归、RK stage 和 halo producer-consumer 之间存在真实依赖，不能为了“更多并行”删除 barrier。

### 7.4 合理浮点误差还是程序错误

判断依据不是“误差很小就一定正确”，而是来源可解释、量级合理、随时间有界、重复运行稳定，并通过物理量和 checker。归约顺序或 FMA 变化可能产生数个 ULP；未改算术却出现随机差异、误差突增、不同运行不一致、约束发散或 NaN，更像 race、越界或同步错误。

### 7.5 单 MIG 上多个 MPI 进程

正式优化没有把 MPI_processes > 1 作为保留实验，因此不能编造实测现象。理论上每个 rank 会建立独立 CUDA context、复制设备状态并竞争同一 14 SM、L2、显存和带宽；该程序约 2 GiB/rank 的状态还会提高 OOM 风险。普通多 context 调度还可能产生切换和串行化。

NVIDIA 为合作式多进程提供 CUDA MPS，可让不同进程的 kernel/memcpy 更有效重叠并减少 context switching/storage；MPS 也能运行在 MIG 实例之上。MIG 是硬件资源隔离，MPS 是同一设备/实例上的软件多进程复用，两者用途不同。

参考：[NVIDIA MPS Architecture](https://docs.nvidia.com/deploy/mps/latest/architecture.html)；[NVIDIA MIG Concepts](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/concepts.html)。

### 7.6 Shared Memory 是否一定更快

不一定。shared memory 要先协作搬运、同步，再读取；它消耗每个 block 的 shared-memory 配额，可能减少同时 resident 的 blocks，还可能产生 bank conflict、边界分支和冗余 halo 搬运。如果复用不足，额外成本会超过节省的 global load。

本实验 compact advection 的 15.73 KiB tile 得到约 2.99 倍 kernel family 加速；但跨字段 32.82 KiB tile 把 occupancy 压到约 12.43%，没有稳定收益。正确做法是根据 stencil 半径和 consumer 集合只搬真正复用的数据，并同时看端到端、duration、register、shared memory、occupancy 和 stall。

### 7.7 A100 40GB 与 80GB 的理论差异

NVIDIA 数据表显示，A100 40GB PCIe 使用 HBM2、带宽 1555 GB/s、TDP 250 W；A100 80GB PCIe 使用 HBM2e、带宽 1935 GB/s、TDP 300 W。两者 FP64 峰值同为 9.7 TFLOPS，所以主要差异是容量、显存技术、带宽和功耗，不是 FP64 core 峰值。SXM 80GB 带宽还可到 2039 GB/s。

MIG 对应容量也不同：40GB 卡常见 1g.5gb，80GB 卡对应 1g.10gb；两者都是约 1/7 SM slice。AMSS 最终只用约 2 GiB 显存，baseline NCU DRAM utilization 仅 3.97%，计算更受寄存器、依赖和 launch 限制，所以本实验未观察到明显差异是合理的，不能据此说硬件没有理论差异。

参考：[NVIDIA A100 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)；[NVIDIA Supported MIG Profiles](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html)。

### 7.8 实验体验与建议

完整科学计算程序比孤立 kernel 更能体现优化的真实约束：局部 kernel 变快不一定改善端到端，提高 occupancy 也可能因额外 launch/中间量而变慢；必须反复在 profile、代码依赖、正确性和完整运行之间闭环。这部分是实验最有价值的地方。

主要困难是工具和任务体量同时很大。perf、VTune、Nsys、NCU 的指标都需要准确理解，而初次使用者容易把 host 等待与 GPU kernel 时间相加、把 occupancy 当成唯一目标，或不知道 profiler 覆盖了哪个子进程。建议课程增加一个小型的 profiler 专项练习和指标解读模板，再进入完整 AMSS；同时提供统一的 JSON/TSV benchmark 元数据、固定 smoke test、checker 说明，以及 CPU/GPU OJ 构建模式的明确约定。失败实验也应允许并鼓励以简短表格保留，因为它们最能说明为什么最终路线合理。

## 8. 结论

CPU 路线的主线是：TwoPuncture 消除重复 cos 并行化；ABE 用完整 OpenMP 替代单节点 MPI；再补齐分析/constraint 并行，最后做 SIMD 和内存局部性。瓶颈由通信等待转移为真实 RHS/stencil，端到端最终 286.162 s。

GPU 路线的主线是：先按数据边界拆巨型 RHS，降低寄存器 live set，再优化同步、stream 和 AMR batching；随后用受控 shared-memory tile 减少 stencil 重读，删除未消费计算；最后只对直接 producer-consumer 做小范围 fusion。正式三次均值 361.512 s、最慢 363.817 s，3/3 checker PASS，达成端到端 <= 370 s。
