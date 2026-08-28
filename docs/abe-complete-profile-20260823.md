# ABE 完整 profile 报告（2026-08-23）

## 1. 结论

本轮对当前提交 `3559f23` 做了完整的 `TwoPuncture + Evolve(40)` profile。
正式运行使用单进程 OpenMP、`OMP_NUM_THREADS=30`、`OMP_PLACES=cores`、
`OMP_PROC_BIND=close`，静态层使用 `24 Block / 24 线程`，移动层使用
`30 Block / 30 线程`，编译参数为严格的 `-O3 -g -fno-omit-frame-pointer`。

最重要的结论是：当前版本已经没有 MPI 通信热点。ABE 的时间主要花在 BSSN
右端项和差分/耗散 stencil 上，其次是整块数组复制、清零以及 AMR
prolong/restrict。当前节点只有 30 个物理核，平均有效 CPU 为约 19.3/30，
说明仍有 OpenMP 层级之间的等待和工作量不均衡，但首要优化对象已经是数值
内核和访存，而不是 MPI。

端到端 Python 计时为 `496.623 s`（stat）和 `502.971 s`（record），均通过
课程检查。当前节点距离 330 秒目标还差约 1.50 倍；最终评测的 60 物理核
节点可能提供额外缩放，但不能假设会线性加速。

## 2. 测量方法

### 2.1 作业和配置

正式端到端作业如下：

| 作业 | 内容 | 节点 | 结果 |
|---|---|---|---|
| `142364` | `hpc_cpu.sh stat`，完整 `t=0..40` | `zjusct-920b-3` | 成功，`FINAL: PASS` |
| `142448` | `hpc_cpu.sh record`，完整 `t=0..40` | `zjusct-920b-1` | 成功，`FINAL: PASS` |

原始文件保存在：

    profile/stat-20260823T023259Z-$
    profile/record-20260823T024522Z-$

两次运行的输出轨迹均与 golden 的前 40 个时间点逐字一致，RMS 为 0；9 个
AMR level 的约束检查全部通过。record 的 `perf.data` 有 484K 个 cycles
样本，`Total Lost Samples: 0`。

### 2.2 集群硬件

本轮实际作业 cpuset 为 60 个逻辑 CPU，位于一个 NUMA node；该节点显示
每个物理核有 2 个硬件线程，因此对应 30 个物理核。处理器为 HiSilicon
TaiShan-v120，支持 ASIMD/NEON 和 SVE；L1D/L1I 为 64 KiB/core，L2 为
1.3 MiB/core，单 NUMA node 的 L3 为 56 MiB。程序的 ABE 目标没有显式加入
`-march=native`，但 libc 的 `memcpy`/`memset` 使用了本机的 SVE 实现。

### 2.3 采样口径

- `perf stat -d -d` 覆盖整个 `./run.sh`，包括 TwoPuncture、ABE 和 Python
  驱动收尾；
- `perf record -F 49 --call-graph fp` 覆盖同样的完整流程；
- `-g` 和 frame pointer 使 ABE 函数能对应到源文件和代码行；
- 两次运行使用不同节点，绝对时间有约 1%--2% 的正常波动，因此热点比例
  和趋势比单次小数点差异更可靠。

## 3. 一次完整运行流程

可以把当前程序压缩成下面几步：

1. `run.sh` 设置无限栈、输出目录和 OpenMP 环境，调用
   `AMSS_NCKU_Program.py`。
2. Python driver 读取固定课程输入，保存调度器给出的完整 CPU affinity，
   生成 AMR patch、参数文件和两个可执行程序。导入 matplotlib 等本地库
   后会恢复 affinity，避免再次把进程缩到单个物理核。
3. `TwoPunctureABE` 用 Ansorg 求解器生成 `Ansorg.psid`。本轮端到端 record
   中 TwoPuncture 只占 2.31% 的 cycles，因此不是当前主要瓶颈。
4. Python 启动单进程、OpenMP-only 的 `ABE`。源码仍保留 MPI API 形状，
   但 `omp_only_mpi.h` 将 collectives 变成本地复制，将点对点通信变成空操作，
   没有 MPI rank 间网络或共享内存通信。
5. ABE 建立 9 层 AMR 网格。粗层到细层的时间细分为
   `1, 1, 1, 1, 2, 4, 8, 16, 32`，所以一个物理时间单位包含 66 次
   `Step`；每次 `Step` 有 4 个 RK 子步。
6. `RecursiveStep(level)` 递归推进更细层；`Step(level)` 在 OpenMP block
   上执行 RK4。每个子步的主要路径是：

       Step -> compute_rhs_bssn
            -> fderivs / fdderivs / lopsided / kodis
            -> RK4 更新、边界处理
            -> 本地 OpenMP sync

   层间还会执行 `prolong3`、`restrict3` 和数组复制。当前日志确认 level 0
   实际只有 9 个 block，level 1--4 为 24 个，移动 level 5--8 为 30 个。
7. 每个粗层步执行分析和输出；当前 OpenMP surface analysis 已经很小，
   `omp_cached_interpolate` 只有约 0.16% 的 flat samples，不应作为第一优化目标。

## 4. 端到端时间和硬件计数器

### 4.1 时间

`perf stat` 作业 `142364` 的关键数据：

| 指标 | 结果 |
|---|---:|
| ABE `Before Evolve` | 7.835 s |
| ABE `Total Evolve` | 475.831 s |
| ABE `Total Running` | 483.666 s |
| Python `This Program Cost` | 496.623 s |
| `perf stat` wall time | 506.769 s |
| 平均有效 CPU | 19.260 / 30 |

日志中的 `Computer used` 是所有 OpenMP 线程累计的 CPU 时间，每个时间步约
230--245 CPU 秒；它不是墙钟时间。实际 Evolve 墙钟约 12 秒/物理时间单位。
把累计 CPU 秒直接当成运行时间会高估约 19 倍。

### 4.2 计数器

| 指标 | 结果 | 判断 |
|---|---:|---|
| task-clock | 9760.56 s | 平均 19.26 个 CPU 在运行 |
| IPC | 1.94 | 中等，包含不同层级和 runtime 开销 |
| branch miss | 0.48% | 正常，不是瓶颈 |
| L1D miss | 2.73% | L1 层有一定重用，但并不完美 |
| LLC load miss | 48.43% | 较高，说明大数组访问经常落到更低层级 |
| dTLB miss | 2.17% | 有明显页表压力，但不是唯一原因 |
| iTLB miss | 约 0% | 正常 |
| context switch / migration | 0 / 0 | 没有调度抖动迹象 |

perf 的计数器覆盖率约 57%--64%，硬件事件发生了 multiplex，因此这些数值
适合判断量级，不应比较到小数点后的微小差别。由于本轮没有采集 DRAM 带宽
事件，不能直接宣称已经达到内存带宽上限；更准确的说法是“计算内核伴随
明显的缓存/TLB 和大块内存流动压力”。

## 5. Profile 热点

### 5.1 DSO 和大类

正式 record `142448` 的 cycles 分布为：

| DSO | cycles |
|---|---:|
| `ABE` | 80.23% |
| `libc.so.6` | 14.10% |
| `libgomp.so` | 2.66% |
| `TwoPunctureABE` | 2.31% |
| `libm.so.6` | 0.59% |

这和早期 MPI baseline 完全不同：本轮没有 `libmpi` 或 `libopen-pal` 热点。
`libc` 主要是大块 `memcpy` 和 `memset`，不是 Python I/O。

### 5.2 Flat profile

以下为正式生产配置的 flat samples；这些数字是 self time，不能相加为完整
运行时间，但可直接比较热点优先级：

| 函数 | self samples | 作用 |
|---|---:|---|
| `compute_rhs_bssn_` | 42.38% | BSSN 方程右端项，包含大量点对点代数 |
| `__memcpy_sve` | 8.64% | AMR/同步工作区和数组复制 |
| `kodis_` | 8.13% | Kreiss--Oliger 数值耗散 stencil |
| `fdderivs_` | 7.64% | 二阶、混合空间导数 |
| `lopsided_` | 6.40% | shift 方向相关的偏置导数 |
| `__memset_sve_zva64` | 5.16% | 临时数组/边界工作区清零 |
| `prolong3_` | 4.06% | 细网格 prolongation |
| `fderivs_` | 3.12% | 一阶空间导数 |
| `enforce_ga_` | 1.55% | 代数约束/规范处理 |
| `symmetry_bd_` | 1.49% | 对称边界填充 |
| `rungekutta4_rout_` | 1.34% | RK4 数组更新 |
| `restrict3_` | 1.32% | 粗细网格 restriction |

仅前 8 项已经覆盖约 85.5% 的 samples。这里的 `memcpy + memset` 合计约
13.8%，`prolong3 + restrict3 + copy_` 约 5%--6%；它们与计算 kernel 共同
构成访存压力。

### 5.3 调用路径

最大路径是：

    Evolve
      -> RecursiveStep(0)
        -> Step(level)
          -> OpenMP worker
            -> compute_rhs_bssn
              -> fdderivs / kodis / fderivs / lopsided

另一条重要路径是：

    Step
      -> omp_local_transfer
        -> prolong3 / restrict3 / copy_
      -> omp_execute_cached_sync
        -> copy_ -> __memcpy_sve

在 inclusive call graph 中，`Step` 的两个 OpenMP loop clone 合计占大部分
worker samples；`omp_local_transfer` 约 7.7%，cached sync 约 7.0%。这些
百分比包含子调用，不能和 flat 表直接相加。

### 5.4 源代码行热点

带 `-g` 的源代码行 profile 把热点进一步定位到：

- `src/kodiss.f90:111`：三方向六阶耗散组合式，7.64%；
- `src/bssn_rhs.f90:550`、`:589`、`:511`：Ricci/二阶导数相关的大型代数表达式，
  单行约 3.1%--3.7%；
- `src/lopsidediff.f90:280`：偏置导数内层 stencil，约 2.76%；
- `src/diff_new.f90:574`、`:582`、`:578`：混合/二阶导数 stencil，单行约
  1.2%--1.8%；
- `src/prolongrestrict_cell.f90:2136`、`:2143`、`:2149`：六点插值，合计约
  3.7%；
- `src/prolongrestrict_cell.f90:2459`：restriction 插值，约 1.08%；
- `src/Parallel.C:2951`：cached sync 中将 block 数据打包到工作区。

`compute_rhs_bssn` 在 `src/bssn_rhs.f90:68-84` 声明了大量与整个 block 同尺寸
的自动数组，包括一阶导数、二阶导数、逆度规和中间量。随后它在
`src/bssn_rhs.f90:140` 以后多次调用不同导数例程。这个结构解释了为什么
profile 同时出现大量 Fortran stencil 和 `memset`：每个 RHS 调用不仅做浮点
计算，还要反复扫过同一块网格和管理大工作集。

## 6. 并行和负载均衡判断

### 6.1 通信/同步

当前编译为 `AMSS_OMP_ONLY`，没有 MPI rank、`MPI_Reduce` 或网络通信样本。
`libgomp` 只有 2.66%，说明 OpenMP runtime 不是最大开销；剩余的并行损失主要
来自不同 AMR 层 block 数量不同、层间 barrier 和 block 内存流动。

层级工作量并不均匀：level 0 只有 9 个实际 block，level 1--4 有 24 个，
level 5--8 才有 30 个。因而即使移动层能够填满 30 个 worker，粗层仍会有
线程没有迭代或在 barrier 等待。平均 CPU 19.26/30 正是这种现象的作业级
证据，而不是 affinity 丢失。

record 的 PID/TID profile 中，主要 30-thread worker 的样本约在 2.9%--4.1%
之间，最大值约为平均值的 1.24 倍，没有一个线程长期垄断计算。报告中还会
看到许多短生命周期 TID，这是不同 OpenMP parallel 区域和 24/30 团队切换
产生的 runtime 线程记录，不能把它们误当成 MPI rank。

### 6.2 计算还是访存

当前应定性为“计算内核 + 访存混合瓶颈”：

- `compute_rhs_bssn` 和 stencil 函数占绝大多数 ABE self samples，说明有真实
  的浮点/邻域计算；
- `memcpy`、`memset`、AMR transfer 合计占比较大，LLC miss 约 48%、dTLB miss
  约 2.17%，说明大数组搬运和缓存层次也明显限制吞吐；
- branch miss 仅 0.48%，不是分支预测问题；
- 没有 DRAM 带宽事件，不能把它进一步细化为“已经带宽饱和”。下一轮应在
  RHS-only 区间补采 `uncore/DRAM bandwidth`（若节点提供）或内存吞吐事件。

## 7. 推荐的下一步优化顺序

### P0：复用 RHS 和 AMR 工作区，先减少清零/复制

优先处理 `bssn_rhs.f90:68-84` 的整块临时数组以及 `Parallel.C:2938-2951`
的 transfer workspace：

1. 为每个 OpenMP worker 或每个 block 建立可复用 scratch workspace；
2. 在 `Step`/演化生命周期内保留其容量，不在每次 RHS 调用重新申请、释放或
   隐式创建；
3. 只写入确实需要的 interior/ghost 区域，避免整数组 `memset`；
4. 检查 `copy_` 的打包和解包是否可以直接写入目标 block，减少中间 buffer。

这是低风险且与数值公式无关的改动。`memcpy + memset` 理论上占约 13.8%，
即使只能消掉其中一半，端到端也可能得到约 7% 的收益；它单独不足以达到
330 秒目标，但会同时降低 LLC/TLB 压力。

### P1：优化导数/耗散 stencil 的循环结构

目标函数为 `compute_rhs_bssn`、`fdderivs`、`fderivs`、`lopsided` 和 `kodis`：

- 保持 Fortran 第一维 `i` 为连续内层，按 block 做 cache tiling；
- 把每次调用中相同的 `dX/dY/dZ` 倒数、边界范围和系数移到循环外；
- 对 interior 无边界分支的循环尝试 `!$omp simd` 或 `do concurrent`，再用
  `-fopt-info-vec` 验证实际生成的向量宽度；
- 评估把多个同一输入数组的导数合并到一次扫描，减少重复读取，但必须先做
  `t=4` 数值对照，因为改变求和顺序可能影响 bitwise 结果；
- 不要直接把 `-Ofast` 或 `-ffast-math` 放进正式版本，它们可能改变约束和
  轨迹的浮点结果。

P1 的收益上限明显高于继续优化 `AnalysisStuff`，因为 RHS 及其差分路径已经
占据约 70% 以上的热点样本。

### P2：AMR transfer 的批量化和线程调度

`prolong3/restrict3/copy_` 总体约 5%--6%，`omp_local_transfer` 和 cached
sync inclusive 约 7% 左右。可以：

- 按变量/相邻 block 合并小 copy，减少大量短调用；
- 对固定几何复用索引和插值权重，避免每次重新计算切片边界；
- 比较 `schedule(static)`、按 transfer 大小分桶的静态调度和当前动态调度；
- 让 transfer workspace 按 NUMA first-touch 由实际 worker 初始化。

这一阶段应在 P0/P1 之后进行，因为 transfer 的收益上限小于 RHS，且更容易
受 AMR 层级分布影响。

### P3：只做有数据支撑的线程/编译器实验

- 先在最终 60 物理核节点用同一版本实测缩放，再决定是否需要更多并行改动；
- 继续保留 `24/24 + 30/30`，不要把所有层都降到 24；移动层已有 30 个
  block，降低它会直接损失有效并行度；
- 之前的 `-mcpu=native` 实验在 ABE 端到端回归约 9%，因此不能盲目启用。应
  对 P1 修改后的 RHS 单独做 `-O3`、本机 SVE 和向量化报告 A/B；
- persistent OpenMP team 之前实测回归约 10.5%，暂不重做同类改动。只有在
  P0/P1 降低内核时间后，runtime/barrier 重新成为主要占比时，才值得再研究
  worker pool。

## 8. 建议的验证方式

每个优化阶段都应保持同一流程：

1. 先跑 `t=0..4`，检查四个 binary output 的数值行和 `check.sh`；
2. 用 `perf stat` 比较 wall、task-clock、IPC、LLC/dTLB 和平均 CPU；
3. 用 `perf record -g` 确认热点是否真的下降，而不是转移到 `memcpy` 或
   barrier；
4. 最后跑完整 `t=0..40`，确认轨迹 RMS、9 层约束和最终 PASS。

当前最合理的下一轮是 P0：先只改工作区生命周期和复制路径，再做一次短窗口
正确性 + profile；不要同时改公式、编译参数和线程策略。
