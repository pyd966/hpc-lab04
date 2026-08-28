# CPU ABE `compute_rhs_bssn` 内部优化报告

日期：2026-08-24

## 结论先行

本轮没有继续盲目扩大并行区，而是先把 `compute_rhs_bssn` 拆成数据流阶段，确认哪些阶段确实是在重复扫描整块网格。结果如下：

- `compute_rhs_bssn_` 仍是第一热点，占最终 profile 的 **49.66%** cycles。
- 主要的非 RHS 热点是 `memcpy` **9.99%**、`lopsided_` **7.71%**、`memset` **6.95%**、`fdderivs_` **4.29%**、`fderivs_` **2.27%**、`kodis_` **2.22%**。
- 编译器已经对 RHS 的大多数数组表达式和导数内层循环生成了 AArch64 SIMD（当前节点支持 SVE，部分普通循环也能生成 16-byte 向量）。因此本轮最有希望的方向不是再加一遍 `simd`，而是减少完整数组的读写遍数。
- 已保留三项有实验证据的改动：
  1. 将 `Lap+1`、`chi+1`、三个对角度规加一合并为一次网格遍历；
  2. 将六个 `gij_rhs` 合并为一次网格遍历；
  3. 将三个 `Gamma_rhs` 合并为一次网格遍历。
- 逆度规融合和 lopsided 三个累加项融合没有稳定收益，已经撤回。默认关闭 RHS 的全数组 NaN 扫描，因为它每次 RHS 调用都会额外扫描二十多个大数组；调试时仍可打开。

## 图在哪里

正式的 `t=40` 运行输出在：

`profile/runs/baseline-20260823T185911Z-$/GW250118/figure/`

代表性图表：

- [BH_Trajectory_XY.pdf](/home/h3250106394/hpc-lab04/profile/runs/baseline-20260823T185911Z-$/GW250118/figure/BH_Trajectory_XY.pdf)
- [BH_Trajectory_3D.pdf](/home/h3250106394/hpc-lab04/profile/runs/baseline-20260823T185911Z-$/GW250118/figure/BH_Trajectory_3D.pdf)
- [BH_Position_R.pdf](/home/h3250106394/hpc-lab04/profile/runs/baseline-20260823T185911Z-$/GW250118/figure/BH_Position_R.pdf)
- [ADM_Constraint_Grid_Level_0.pdf](/home/h3250106394/hpc-lab04/profile/runs/baseline-20260823T185911Z-$/GW250118/figure/ADM_Constraint_Grid_Level_0.pdf)
- [Initial_Grid.jpeg](/home/h3250106394/hpc-lab04/profile/runs/baseline-20260823T185911Z-$/GW250118/Initial_Grid.jpeg)

绘图脚本默认保存 PDF；`Initial_Grid.jpeg` 是初始网格的栅格图。

## 一次 RHS 调用的有效流程

下面省略边界拷贝等无关细节，只保留热点相关的数据依赖。所有数组都是三维网格，最内层 `i` 连续，因此本轮循环都沿 `i` 使用 SIMD。

1. **准备派生场。** 可选的 NaN sanity scan 先检查输入；随后构造 `alpn1=Lap+1`、`chin1=chi+1` 和倾斜度规的三个对角分量。接着对 shift、`chi`、`dxx/gxy/gxz/dyy/gyz/dzz` 调用 `fderivs`，得到一阶导数。
2. **先算一批局部 RHS。** 用 shift 散度、lapse、`trK` 和 `Aij` 得到 `chi_rhs` 与六个 `gij_rhs`。六个 `gij_rhs` 彼此只读输入、互不写入，所以适合合并遍历。
3. **度规与连接。** 由倾斜度规逐点计算逆度规 `gup**`，再计算第一类/第二类 Christoffel 量。这里是大量乘加，数组表达式已经能被编译器向量化；但它们的临时结果在后续 Ricci 计算中仍被复用，不能随意删掉。
4. **Gamma 方程。** 先算无 shift 的 Gamma RHS；再对三个 shift 分量调用 `fdderivs`，形成 `fxx/fxy/fxz` 和 `Gamxa/Gamya/Gamza`，最后更新三个 `Gam*_rhs`。三个分量在同一个网格点上独立，因此可合并为一个 SIMD 遍历。
5. **Ricci 张量。** 将 Christoffel 与度规组合为第一类连接，随后对 `dxx/dyy/dzz/gxy/gxz/gyz` 各调用一次 `fdderivs`，并执行六个 Ricci 分量的大型点乘加表达式。这部分对应 profile 中 `bssn_rhs.f90` 的 500--690 行，是 RHS 内最重的纯计算区。
6. **其余物理项和耗散。** 对 `chi`、`Lap` 等做二阶导数/协变导数，更新 `Aij`、`trK`、lapse、shift 和 gauge RHS。随后对 24 个场调用 `lopsided`，再调用 `kodis` 做耗散。`co=0` 时还会计算约束残差；预测/校正步骤 `co=0/1` 的主 RHS 结构相同，但约束只在 `co=0` 做。

## 如何判断哪些地方值得融合

我先检查了 `fderivs`、`fdderivs`、`lopsidediff`、`kodis` 的循环和编译器向量化报告。它们的内层循环已经有 `!$omp simd` 或由数组语句自动向量化，说明“再加一个 SIMD 指令”本身不会减少内存流量。然后检查 `compute_rhs_bssn` 中的数组赋值：如果几个赋值满足以下条件，就有融合价值：

- 对每个 `(i,j,k)` 都是独立的点运算；
- 赋值之间没有读后写或写后读依赖；
- 输出数组在后续阶段仍按完整网格使用，因此一次遍历可以直接减少多次读写；
- 融合后的表达式规模不会显著增加寄存器压力。

按这个标准，metric 初始化、六个 `gij_rhs` 和三个 `Gamma_rhs` 可以安全融合；Ricci 的六个巨大表达式不适合一次性拼成一个超大循环，原因是活跃变量太多，可能反而降低寄存器利用率并增加 spill，所以暂不改。

## 实验设置

每个候选都重新编译并在 HPC 上跑 `t=0..4`，使用 `-O3 -g -fno-omit-frame-pointer -fopt-info-vec-optimized`，单进程 OpenMP，`OMP_NUM_THREADS=30`、`OMP_PLACES=cores`、`OMP_PROC_BIND=close`。每个作业分别做 `perf stat` 和 `perf record`，并比较数值输出摘要。

绝对时间必须注意节点差异。基线、metric、Gamma、sanity A/B 都在 `zjusct-920b-1`；组合版本的两个作业在其它节点，不能用来声称同等幅度的单项加速。

| 实验 | `perf stat` Evolve | `perf record` Evolve | 结果 |
|---|---:|---:|---|
| 基线（所有新开关关闭） | 33.1165 s | 33.5806 s | 参考 |
| 逆度规融合 | 33.2247 s | 33.4058 s | 无稳定收益，撤回 |
| lopsided 三项累加融合 | 33.2701 s | 33.1336 s | 两次方向相反，视为噪声，撤回 |
| metric 初始化 + `gij_rhs` 融合 | 32.8478 s | 33.3798 s | 同节点约 0.6--0.8% 改善，保留 |
| Gamma RHS 融合 | 32.7105 s | 32.7155 s | 同节点约 1.2--2.6% 改善，保留 |
| 关闭 RHS NaN 全数组扫描 | 33.0241 s | 33.0169 s | 消除调试扫描开销，保留为生产默认 |

各候选的数值输出摘要与基线一致。组合开关在 `zjusct-920b-2` 的最终 profile 中，`perf stat` Evolve 为 29.9273 s；该节点整体更快，只能说明组合没有引入明显回归，不能把 29.9 s 与基线节点的 33.1 s 直接相减当作代码加速。

## 当前 profile 的含义

最终组合版本的硬件计数器为：

- 22.04 个逻辑 CPU 的平均利用率（30 个线程并不等于 30 个 CPU 始终满载）；
- IPC 1.49；branch miss 0.44%；L1D miss 4.19%；LLC load miss 49.70%；dTLB miss 3.56%。

branch miss 不异常，L1D 也不高；LLC miss 接近一半，结合 `memcpy/memset` 和大量三维临时数组，说明 RHS 更像“计算量很大且受到内存层次限制”的混合瓶颈，而不是分支或同步瓶颈。`compute_rhs_bssn` 内最重的行落在 Ricci/连接的乘加（例如约 514、609、648、687 行）；`lopsidediff.f90:112`、`diff_new.f90:560` 等是下一层独立热点。

## 本轮代码开关

生产默认值已经写入 CMake：

- `AMSS_ENABLE_RHS_METRIC_FUSION=ON`
- `AMSS_ENABLE_RHS_GAMMA_FUSION=ON`
- `AMSS_ENABLE_RHS_SANITY_CHECK=OFF`

调试或回归定位时，可以用 `-DAMSS_ENABLE_RHS_SANITY_CHECK=ON` 恢复原来的 NaN 扫描；这不是数值算法开关，只是诊断开关。profile 脚本也支持环境变量 `AMSS_RHS_METRIC_FUSION`、`AMSS_RHS_GAMMA_FUSION` 和 `AMSS_RHS_SANITY_CHECK` 做 A/B。

## 下一步建议（暂不实施）

1. 先针对 `fdderivs` 和 `lopsided` 做“批量输入/输出”实验，重点减少函数调用间的临时数组复制；这比继续拼接 Ricci 大表达式风险低。
2. 对现有 OpenMP 区域采集每个 block 的耗时，再决定是否采用任务队列或 `schedule(dynamic)`；RHS 内部融合不能解决跨 block 的负载不均衡。
3. 只有在确认临时数组生命周期后，才考虑复用缓冲区或改变布局。当前 LLC miss 较高，但贸然复用可能改变 Fortran 别名和边界语义。
4. 暂不换数学库或打开 `-Ofast`：当前主要问题是数组流量和调度，`-Ofast` 还可能改变浮点重排和结果可重复性。下一轮应保留 `-O3`，一次只测一个内存/调度候选。

本轮到此暂停，等待审查后再进入 `fdderivs/lopsided` 批处理或 OpenMP 调度实验。
