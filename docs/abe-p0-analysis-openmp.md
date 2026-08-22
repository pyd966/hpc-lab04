# ABE P0 阶段报告：球面分析的缓存与 OpenMP 归约

## 1. 本阶段结论

P0 已在单进程 OpenMP 路径完整实现。它不是简单地把原来的 MPI
Allreduce 合并，而是重写了曲面分析的数据流：

1. 首次遇到某个提取半径时，缓存每个球面点所属的 Block、六阶插值
   模板和插值权重；
2. 波形分析额外缓存 Wigner 函数和三角函数组合后的线性系数；
3. 每个 OpenMP 线程对自己负责的球面点插值，并立即累加到线程私有的
   最终物理量；
4. 线程结束后只合并很小的结果数组，不再建立或归约完整球面 shellf。

在相同 HPC 节点、输入、绑核和编译参数下，t=0..4 的结果为：

| 版本 | Evolve (s) | Total (s) |
|---|---:|---:|
| P1 + transfer 优化 | 61.4773 | 68.7607 |
| P0 | 43.8688 | 51.2490 |
| 加速比 | 1.401x | 1.342x |
| 时间减少 | 28.64% | 25.47% |

profile 目录：

    profile/abe-20260822T074932Z-14

## 2. 原分析路径的问题

每个分析时刻会遍历 8 个提取半径。每个半径先调用
Patch::Interp_Points，把球面上的所有变量插值到一个完整 shellf 数组，
然后 surf_Wave 或 surf_MassPAng 再遍历 shellf 做积分。

这条路径在每次调用中重复：

- 分配 pox、shellf、weight 和插值临时数组；
- 生成相同半径上的球面坐标；
- 对每个点扫描 Block 列表以确定所有者；
- 重算相同的六阶插值下标和系数；
- 通过 global_interp -> polin3 -> polint 做 3D 插值；
- 在每个点、每个球谐模式上重算 cos、sin 和 Wigner_d_function；
- 对完整 shellf 做 Allreduce，再对最终积分结果做 Allreduce。

在 OMP-only 模式中 Allreduce 只是本地复制，没有网络等待，但完整中间
数组、重复计算和 polint 的动态临时数组仍然是真实开销。上一阶段
profile 中 polint 自身占 8.32%，整条插值调用路径约占 13.8%，
malloc 和 free 合计约占 4.72%。

## 3. P0 的实现

### 3.1 球面插值计划

surface_integral 对每个 Patch、level 和半径保存一个 sphere plan。
计划包含每个球面点的：

- 所属 Block 指针；
- 三个方向各 6 个网格下标；
- 对称边界反射标记；
- 三个方向各 6 个 Lagrange 插值系数。

计划首次使用时构造。如果 Block 指针列表变化，缓存失效并重建，避免
网格重构后使用旧所有权。当前分析在固定的 level 0 上执行，因此正常
运行中每个半径只构造一次。

缓存插值器保持原 global_interp 的六阶模板、边界夹取和对称奇偶规则。
运行时只读取当前场数组并执行 z、y、x 三层张量插值，不再调用 polint，
也不分配插值临时数组。

### 3.2 波形分析

波形积分对每个球面点和每个 l,m 模式预计算四个系数，分别表示：

- 实部场对输出实部的贡献；
- 虚部场对输出实部的贡献；
- 实部场对输出虚部的贡献；
- 虚部场对输出虚部的贡献。

这些系数已经包含对称副本、Wigner 函数、cos、sin 和 theta 权重。
它们与演化场值无关，所以整个运行只需计算一次。

OpenMP 按球面点静态划分工作。每个线程只写自己的 2*NN 个局部结果，
循环结束后按固定线程编号顺序合并，因此没有原子操作，也没有并发写
冲突。原来的完整 psi4 shellf 和两次 MPI Allreduce 均不再需要。

### 3.3 ADM 质量、动量和角动量

f_admmass_bssn 的 Block 准备阶段按 Block 并行。之后每个线程负责一组
球面点，对 17 个场使用缓存插值计划，得到一个点的值后立即累加质量、
三维线动量和三维角动量。

每线程只保存 7 个 double 的局部积分量。最后按固定顺序合并这 7 个量，
替代完整的 17*n_tot shellf 和七次标量归约。

## 4. 正确性

perf stat 和 perf record 两次独立运行中，下列四个结果文件的数值行
逐字节一致：

- bssn_ADMQs.dat
- bssn_BH.dat
- bssn_constraint.dat
- bssn_psi4.dat

与 P0 之前的版本比较，ADMQs、BH 和 constraint 逐字节一致。psi4 的
非零输出在文件打印精度下相同；差异只出现在按对称性应为零的模式。
旧路径因浮点累加留下约 1e-22 到 1e-25 的残差，新路径给出精确 0，
最大绝对差为 1.4750232e-22。这不会反馈到演化状态。

缓存增加了约 93 MiB 常驻内存，主要是波形基函数系数。相对于 100 GiB
作业内存和约 3.3 GiB 实际峰值，这个交换是合理的。

## 5. Profile 变化

P0 后的主要 self 热点为：

| 函数或操作 | 周期占比 |
|---|---:|
| compute_rhs_bssn | 43.86% |
| memcpy | 8.81% |
| kodis | 8.08% |
| fdderivs | 7.53% |
| lopsided | 6.71% |
| memset | 5.42% |
| prolong3 | 4.00% |
| fderivs | 3.16% |
| cached interpolation | 0.28% |
| malloc + free | 0.21% |

polint 已从热点表中消失。compute_rhs 等比例上升不是它们变慢，而是
分析阶段被缩短后，它们占剩余总时间的比例自然增加。

平均使用 16.240 个 CPU，IPC 为 1.90，branch miss 0.41%，L1D miss
2.62%，LLC load miss 48.08%，dTLB miss 2.14%。没有出现异常分支或
缓存退化；剩余主要瓶颈仍是 RHS 计算以及 AMR copy/prolong/restrict
的数据访问。
