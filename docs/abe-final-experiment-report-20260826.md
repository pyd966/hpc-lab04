# AMSS-NCKU CPU/GPU 优化实验报告草稿

日期：2026-08-26  
最终 CPU 提交：`d01e5d7`（`Tile RHS Ricci updates by row`）

## 1. 实验目标和报告口径

本实验优化的是一条完整的科学计算流水线：Python 参数生成和启动、
TwoPuncture 初值求解、BSSN/AMR 时间演化、结果输出和正确性检查。所有保留的
优化都同时满足两个条件：

1. 端到端时间或目标子阶段有可重复的改善；
2. 课程检查通过，主要数值输出在忽略时间戳后逐位一致。

短程 `t=0..4` 用于筛选候选，采用同一作业内交错 A/B 运行；最终结果使用完整
`t=0..40` 验证。`perf stat` 用于硬件计数器，`perf record -g` 用于函数、调用路径
和源代码行定位；OpenMP 诊断额外记录每个 level 的 block 利用率、最长 block、
Sync、AMR transfer 和 regrid 时间。

## 2. 运行环境和最终配置

### 2.1 CPU 平台

- 节点：HiSilicon TaiShan-v120 / AArch64；作业实际 cpuset 为 60 个逻辑 CPU，
  对应 30 个物理核、每核 2 个 SMT sibling。
- 全机有 4 个 NUMA node；本次作业的 CPU 和内存限制在一个 NUMA node，未出现
  跨 NUMA 访问。
- CPU 支持 ASIMD/NEON、SVE、FP16/BF16、dot-product 和矩阵相关扩展；应用的
  浮点内核最终使用严格 `-O3`，没有全局 `-mcpu=native`、SVE 或 fast-math。
- 资源：`lab4`，60 CPU、100 GiB、30 分钟。
- 演化配置：单进程 OpenMP-only ABE，`OMP_NUM_THREADS=30`，
  `OMP_PLACES=cores`，`OMP_PROC_BIND=close`，`OMP_SCHEDULE=dynamic,1`。
- 静态层：24 blocks / 24 threads；移动层：30 blocks / 30 threads。
- ABE production 编译：`-O3`；profile 编译：
  `-O3 -g -fno-omit-frame-pointer`。
- TwoPuncture production 编译：`-O3 -march=native`，并使用 OpenMP；没有使用
  `-Ofast`。

最终运行脚本为 [`hpc_cpu.sh`](../hpc_cpu.sh)，会根据作业内 CPU/core 映射自动计算
物理核数，设置 OpenMP 线程和 block 几何。当前公开队列只能提供 30 个物理核，
因此 60 物理核评测拓扑尚未在同一队列完成 sweep。

### 2.2 GPU 平台

GPU 路径使用课程的 A100 MIG 分区：一个 `1g.10gb` MIG 实例、16 个 CPU、24 GiB
内存；主机为 x86_64，默认工具链为 GNU 13、OpenMPI 5 和 CUDA 12.4。

- 可执行程序：`ABEGPU`；初值程序仍为 `TwoPunctureABE`。
- CUDA 架构：`CMAKE_CUDA_ARCHITECTURES=80`（`sm_80`）。
- GPU profile/benchmark 使用 1 个 MPI rank，通过 `mpiexec --bind-to core` 启动；
  OpenMP 关闭。
- CUDA 编译使用 `-rdc=true`、`CUDA_SEPARABLE_COMPILATION=ON` 和 `-lineinfo`。
- `AMSS_MPI_CUDA_AWARE=0`，因此默认通信路径是 host staging；没有在没有 MPI
  能力证明的情况下强行启用 CUDA-aware MPI。

本轮主要开发和验收集中在 CPU ABE。GPU 路径完成了构建、短程 benchmark 以及
Nsight Systems/Compute 采集入口的核对，但没有形成经过完整 `t=100` 端到端 A/B
验证的 GPU 优化提交。因此报告只把上述内容作为 GPU 最终运行配置，不宣称 GPU
端到端加速。

## 3. Baseline 和第一次 profile

### 3.1 早期 MPI baseline：为什么改变并行模型

最初的 ABE 使用 30 个 MPI rank、每个 rank 一个线程。早期完整 profile 中，
Open MPI shared-memory BTL 的 `mca_btl_sm_poll_handle_frag` 约占 **58.98%**，
另外约 11% 来自 MPI 内部地址和 progress 路径；`compute_rhs_bssn_` 只有约
6.10%。level 0 实际只有约 9 个有工作的 block，其他 rank 在同步点等待。

这说明当时的第一瓶颈不是某一条 Fortran 公式，而是单节点 MPI 的共享内存通信、
progress 和层级负载不均衡。因此先将 ABE 改成单进程 OpenMP-only，并保留旧 MPI
接口的类型兼容层。转换后 ABE 不再链接 MPI runtime：collective 退化成本地操作，
点对点通信在单进程语义下为空操作。

### 3.2 朴素 OpenMP 转换暴露的问题

第一次 P1 实验使用 1 个进程、30 个 OpenMP 线程。虽然去掉了 MPI，但 `t=0..4`
Evolve 从 MPI 版本约 173.7 s 变成约 542 s，慢约 3.12 倍；平均只有 **3.23/30**
个 CPU 在运行。profile 显示 `compute_rhs_bssn_`、`polint_`、导数和插值路径已经
被放进 OpenMP worker，但粗层 block 数太少，层间递归、Sync 和插值仍然串行，
线程创建/汇合及临时分配开销被放大。

这个失败实验很重要：它证明“删掉 MPI”本身不是优化，必须把并行区放在 block/patch
等足够外层，并重新设计任务几何。

### 3.3 可比较的 ABE P0 基线

经过外层 block 并行、绑核和调度重构后，得到后续 A/B 的 P0 基线：

| 指标 | P0 基线均值 |
|---|---:|
| ABE Evolve `t=0..40` | `298.655 s` |
| ABE Total | `301.339 s` |
| `This Program Cost` | `313.763 s` |
| 外层 wall | `322.5 s` |
| 平均内存 | 约 4 GiB |

P0 短程 profile 的主要热点为：`compute_rhs_bssn_` 约 **41.47%**，
`__memcpy_sve` **12.57%**，`lopsided_` **8.54%**，
`__memset_sve_zva64` **5.82%**，`prolong3_` **4.90%**，`fdderivs_` **4.64%**。
平均 CPU 约 22.65/30，IPC 1.80，branch miss 0.43%，L1D miss 3.86%，
LLC load miss 37.35%，dTLB miss 3.41%。

因此问题被判断为：RHS 是最大的计算/访存热点；`memcpy`、`memset` 和 AMR transfer
是第二类数据搬运热点；平均 CPU 偏低主要来自层级依赖和 block 数量，而不是线程
迁移或异常分支预测。

## 4. TwoPuncture 初值阶段

TwoPuncture 同时服务 CPU 和 GPU，因此先处理其独立 profile 发现的热点，再回到
ABE 演化。

### 4.1 消除高频临时分配

profile 发现 `chebft`、`fourft`、`Derivatives_AB3`、`F_of_v`、`J_times_dv`、
`LineRelax` 和 Thomas 求解器反复申请短数组。为每个线程建立独立的
`TransformWorkspace`、`PointWorkspace` 和 `LineWorkspace`，复用 scratch，避免
在热点路径 `malloc/free`；Thomas scratch 显式传参并标记 no-alias。

结果是指令约减少 3.65%、cycles 约减少 2.08%，但该轮节点频率较低，wall time
没有稳定下降。因此结论是“确实减少了工作量，但不能仅凭一次 wall time 宣称加速”。
正确性方面，BiCGSTAB 迭代、残差、`puncture_parameters_new.txt` 和去时间戳后的
`Ansorg.psid` 均与基线一致。

### 4.2 预计算谱变换系数

网格尺寸固定，Chebyshev/Fourier 变换中的三角函数输入只由网格下标决定。因此在
每个 workspace 中缓存实际出现的 Chebyshev cosine 表和 Fourier sine/cosine 表，
消除变换内重复的 `cos/sin` 调用；没有改变求和顺序，也没有改成 FFT。

这是 TwoPuncture 中收益最大的单项：相对前一阶段 wall time 约下降 **31.10%**，
cycles 约下降 29.59%，`cos` 从约 23.65% 的 self samples 降到不可见水平。这个
优化的本质是把重复的 libm 计算变成小型只读表访问，而不是提前知道整个演化过程。

### 4.3 编译参数和 OpenMP

`-Ofast -march=native` 在 TwoPuncture standalone benchmark 中比严格 `-O3` 快约
5.29%，但放宽浮点语义可能改变求解结果，因此最终配置没有采用 `-Ofast`。随后对
谱导数、逐点方程和红黑/颜色线松弛进行 OpenMP 并行：不同谱线、不同网格点和同一
颜色 phase 的线彼此独立，Thomas 三对角递推仍保持单线串行；200 次 relax 迭代放在
一个持久 parallel region 中，避免重复 fork/join。

严格 `-O3` 下 standalone benchmark 的 30/60 线程结果约为 11.399/11.091 s，
但当前 SMT 节点上 60 线程只比 30 线程快约 1.6%，因此生产脚本选择 30 个物理核。
最终 TwoPuncture 只保留严格 `-O3 -march=native` 和 OpenMP。详细阶段记录见
[`TwoPuncture 阶段 1`](twopuncture-stage1-allocations.md)、
[`阶段 2`](twopuncture-stage2-coefficients.md)、
[`阶段 3`](twopuncture-stage3-compiler-flags.md) 和
[`阶段 4`](twopuncture-stage4-openmp.md)。

## 5. ABE 的主要优化过程

### 5.1 先做可证明的 SIMD 优化

早期源码行 profile 显示 `kodis`、`fdderivs`、`fderivs` 和 `lopsided` 的规则内点
循环存在边界分支，编译器没有稳定生成向量循环。做法是把规则内点与边界薄层拆开，
只对连续的 `i` 方向内点加 `!$omp simd`，边界和 symmetry 逻辑保持原样。

- `kodis`：显式移出三维边界判断后，NEON 双精度向量指令出现，短程 Evolve
  **7.43%**，逐位一致，默认开启。
- `fdderivs`：规则内点 SIMD，Evolve 约 **2.94%–3.21%**，默认开启。
- `fderivs`：规则内点 SIMD，交错 A/B Evolve 约 **1.31%**，默认开启。
- `lopsided`：虽然生成了 SIMD，但同时计算正、负两套 stencil，再用符号选择，
  额外访存抵消收益，Evolve 基本持平（约 `+0.006%`）。代码保留但不能把它写成
  性能收益。

这些优化都只改变循环组织，不改变边界公式和浮点表达式顺序；A/B 输出逐位一致。

### 5.2 重新设计 OpenMP 外层任务和几何

在 `bssn_class::Step` 外层按 block 建立工作列表，覆盖 predictor、corrector、
RHS、RK 更新和 block 级分析；`RecursiveStep` 仍负责 AMR 层之间的依赖顺序。移动
层使用 `dynamic,1`，静态层减少到 24 个 worker，避免把只有少量 block 的层分给 30
个线程。Sync、prolong/restrict 和 regrid 在每个阶段边界保留必要同步。

实验表明：

- 60 OpenMP worker 在当前 SMT 节点慢 **36.7%**，IPC 约减半，dTLB miss 增加；
- 30 worker 下把 block 目标增到 60/90 反而慢，虽然 LLC/IPC 表面改善，但边界、
  ghost、Sync 和调度次数增加；
- 只增加移动层 block 也慢约 5.1%。

所以最终不是追求“平均 CPU 数最大”，而是保持 30 worker、24/30 block/thread 和
`dynamic,1`。最终 OpenMP 诊断中 level 5--8 利用率约 84%–89%，level 0 只有 9 个
block，剩余空转主要是算法层级依赖，不是绑核失败。

### 5.3 减少确定存在的数据搬运

这一步针对 profile 中的 `memcpy/memset`，每次都先做地址重叠证明和 A/B。

| 优化 | 关键位置 | 端到端结果 | 决定 |
|---|---|---:|---|
| symmetry ghost 冗余清零 | `symmetry_bd` | Evolve 约 `0.60%`，Total 约 `1.04%` | 保留、默认 ON |
| 同级 Sync 直接复制 | `omp_execute_cached_sync` | Evolve 约 `1.02%` | 保留、默认 ON |
| block field arena | `Block` 字段分配 | Evolve 约 `0.65%`；dTLB 约 3.7% 降到 1.4% | 保留、默认 ON |
| prolong3 相邻点复用 | `prolong3_pair_kernel` | Evolve 约 `1.27%` | 保留、默认 ON |
| direct AMR 输出 | `omp_local_transfer` | 约 `0.24%`，低于噪声 | 默认 OFF |

同级 Sync 的直接路径只对几何证明互不重叠的矩形生效，其他操作仍回退到旧的
pack/unpack；因此它不会破坏有读后写风险的 transfer。连续 arena 只改变字段的
分配位置和生命周期，未改变数组索引或数学计算。

### 5.4 RHS 局部数据复用：从失败融合到成功分块

在 P0 profile 中，`compute_rhs_bssn` 是第一热点。先尝试把多个完整数组表达式合成
一个循环，但 profile 发现活跃向量值、地址流和寄存器压力增加，IPC 下降：

- 18 个 second-kind connection 一次融合：约回退 2.0%；拆成小组后仍回退约 0.7%；
- Aij 六字段融合：回退约 0.76%；
- chi derivative/Ricci producer-consumer 融合：回退约 0.37%–1.47%；
- 三个 first-kind connection 全融合：回退约 0.56%。

这些失败实验说明“少一次数组遍历”不一定抵消更大的 SIMD live set；当前内核受缓存
和寄存器供给限制，而不是单纯受循环控制开销限制。

之后保留了两个小范围方案：

1. 只融合两个 first-kind connection 字段，避免扩大活跃输出集，Evolve 提升
   **1.13%**，默认开启；
2. 将六个 Ricci 更新按 `j=1` 行分块：最内层仍是连续 `i` SIMD 循环，六个分量
   逐个计算但共享同一行的输入缓存。交错 A/B 中 Evolve 从约 28.90 s 降到
   26.81 s，提升 **7.26%**；指令基本不变，IPC 约 1.43 升到 1.55，LLC misses
   下降。这是最终版本收益最大的 ABE 内核优化。

### 5.5 小批量、清零和编译器尾部实验

这些实验用于验证 profile 中“看起来可以减少工作”的方向，但没有全部进入生产：

- `fderivs` 双字段 batch：慢约 `0.067%`；lopsided 双字段 batch 三轮分别慢
  `1.13%`、`1.57%`、`1.22%`。原因是每个字段仍要单独 ghost/清零，批处理增加
  活跃数组和寄存器压力，默认关闭。
- `fderivs/fdderivs` 只清零边界：数值正确但慢 `1.0%–2.2%`；完全跳过清零会破坏
  下游边界零值语义，逐位检查失败。保留原始整数组清零。
- lopsided ghost-only：第一次版本出现 NaN；修正边界后仍慢约 `0.23%`，默认关闭。
- huge page：单独字段提示慢约 `1.45%`；arena 上启用 THP 仍慢约 `0.35%`。说明
  arena 已经解决主要 dTLB 压力，THP 没有额外收益。
- 64B 对齐慢约 `0.86%`；LTO 只快 `0.32%`；`-mtune=native` 与基线持平；显式
  SVE 慢约 `13.5%`。因此最终使用空的 ABE 架构 flags 和严格 `-O3`。
- `rungekutta4` 只占约 2%，而 RHS 数据必须跨 RK stage 保存；RHS/RK 融合会增加
  生命周期和寄存器风险，没有实现。
- 没有更换 BLAS/FFT/数学库：当前 profile 没有 BLAS/FFT 热点，`pow` 约只有
  `0.6%`，而真正主热点是项目内部 RHS、stencil 和数据搬运。

## 6. 最终 profile 和最终性能

最终 profile 作业 `166216` 使用当前提交和带符号 `-O3`，`perf record` 采集 66,284
个样本，无丢样。短程 `t=0..4` 的最终热点为：

| 函数 | cycles |
|---|---:|
| `compute_rhs_bssn_` | `47.08%` |
| `__memcpy_sve` | `10.42%` |
| `lopsided_core_` | `9.78%` |
| `__memset_sve_zva64` | `5.89%` |
| `fdderivs_` | `4.58%` |
| `kodis_` | `2.82%` |
| `prolong3_pair_kernel_` | `2.75%` |
| `fderivs_` | `2.60%` |

硬件计数器为 IPC `1.54`、branch miss `0.49%`、L1D miss `3.97%`、LLC load miss
`48.29%`、dTLB miss `1.44%`；CPU migration 和 context switch 均为 0。RHS 仍是
第一热点，但其六个 Ricci 行的绝对样本已经明显下降；热点占比上升的部分是总执行
时间缩短后的重新归一化，不代表这些函数变慢。

最终 profile 工件：[`perf stat`](../profile/abe-20260826T020426Z-14/perf-stat.txt)、
[`flat profile`](../profile/abe-20260826T020426Z-14/perf-report-flat.txt)、
[`source-line profile`](../profile/abe-20260826T020426Z-14/perf-report-lines.txt)。

最终正式 `t=40` 作业 `166233`：

| 指标 | 最终结果 |
|---|---:|
| ABE Evolve | `271.451 s` |
| ABE Total Running | `275.067 s` |
| `This Program Cost` | **`286.162 s`** |
| 外层 wall | `295 s` |
| 峰值内存 | 约 `3.84 GiB` |
| trajectory RMS | `0` |
| constraints | 40 个时间组 × 9 个 level，全部 PASS |

相对 P0 的 `This Program Cost = 313.763 s`，最终端到端改善约 **8.8%**；Evolve
改善约 **9.1%**。最终检查结果见 [`check.txt`](../profile/baseline-20260826T020719Z-$/check.txt)。

## 7. 结论

整个优化过程可以概括为三次瓶颈转移：

1. 原始 MPI 版本首先受共享内存通信、progress 和 rank 负载不均衡限制；
2. OpenMP-only 版本稳定后，瓶颈转移到规则网格 RHS、stencil 以及 memcpy/memset；
3. SIMD、外层 block 调度、直接 Sync、arena 和局部 RHS 分块后，剩余主要成本是
   `compute_rhs_bssn` 内尚未消除的数组流，以及不可避免的层级同步和 AMR 数据搬运。

最终版本没有依赖不安全的 fast-math、未经验证的 SVE、盲目增加线程或大范围公式
融合；每一项保留的改动都有对应 profile 证据和数值回归结果。当前唯一重要的
环境不确定性是评测平台可能提供 60 个独立物理核，需在该拓扑上重新校准 block
几何；这不影响当前提交在公开 30 物理核节点上的正确性和 330 秒目标结果。
