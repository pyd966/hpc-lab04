# P6 阶段报告：导数小批量实验

日期：2026-08-24
阶段：P6，导数/有限差分小批量处理
代码状态：实验开关保留，默认关闭；未改变默认生产路径

## 1. 目标和基线

P6 的目标是检查多个互相独立、但使用相同网格坐标的导数场，能否合并一次网格遍历，减少循环控制和部分公共数据访问。先选择 `compute_rhs_bssn` 中连续的六个 metric `fderivs` 调用，合并为三个两字段调用：

- `dxx` 与 `gxy`；
- `gxz` 与 `dyy`；
- `gyz` 与 `dzz`。

每个新调用 `fderivs2` 仍分别构造自己的 ghost 区域，分别执行原有的对称边界处理，并对两个字段执行原有的四阶内部模板和二阶边界模板。也就是说，这一版只合并网格遍历，没有改变数学公式、边界规则或字段依赖。

默认 profile 使用当前生产配置：单 MPI 进程、30 个 OpenMP 线程、`OMP_PLACES=cores`、`OMP_PROC_BIND=close`、`-O3 -g -fno-omit-frame-pointer`，演化窗口为 `t=0..4`。结果目录为 [`profile/abe-20260824T141547Z-14`](../profile/abe-20260824T141547Z-14)。

## 2. 实现方式

新增 CMake 选项 `AMSS_ENABLE_FDERIVS_BATCH`，默认是 `OFF`。开启后，`src/bssn_rhs.f90` 的六次 `fderivs` 调用改为三次 `fderivs2`；关闭时编译器仍看到原来的六次调用。

`fderivs2` 放在 `src/diff_new.f90`，其关键点是：

1. 两个输入场各自调用 `symmetry_bd`，保持原有 SoA 对称符号；
2. 两个输出场都先按原实现清零；
3. 内部区域仍使用 `!$omp simd`，编译器报告使用 16-byte 向量；
4. 外壳区域仍使用原来的二阶模板；
5. 两个字段的输出顺序与各自原始 `fderivs` 相同。

因此这不是“把所有临时内存消掉”，也不是把六个字段放进一个大数组。新内核仍有两个 ghost buffer、两次对称边界处理和各自的初始化，这一点直接限制了收益。

## 3. 正确性和 A/B 性能

在同一 HPC 作业、同一节点上交错运行 `full batch batch full`，每次都运行相同的 `t=0..4` 输入。四次运行的结果文件逐位比较一致，课程检查均为 `PASS`。

| 序列 | 版本 | Evolve (s) | Total (s) | 平均 CPU | IPC | LLC miss |
|---:|---|---:|---:|---:|---:|---:|
| 1 | full | 29.7479 | 34.2387 | 22.237 | 1.49 | 49.70% |
| 2 | batch | 29.7040 | 34.0919 | 22.343 | 1.48 | 49.60% |
| 3 | batch | 29.7382 | 34.0921 | 22.383 | 1.48 | 49.71% |
| 4 | full | 29.6545 | 34.0964 | 22.310 | 1.49 | 49.88% |

两次 full 的平均 Evolve 为 29.7012 s，两次 batch 的平均值为 29.7211 s，batch 反而慢 0.0199 s，即 **+0.067%**。这个差异远小于当前作业噪声，不能视为性能提升。平均 CPU、IPC 和 LLC miss 也没有向更有利的方向移动。

编译器确实向量化了新内部循环，但 `objdump` 显示 `fderivs2_` 仍然包含两次 `malloc`、两次 `symmetry_bd`，并有多次 `memset`；函数栈帧约为 0x3a0 字节。合并循环节省的控制开销被额外的局部状态、ghost 初始化和更高寄存器/内存压力抵消了。

## 4. 默认路径 fresh profile

P6 候选拒绝后，重新用默认 `AMSS_ENABLE_FDERIVS_BATCH=OFF` 的生产路径进行 profile。两个 profile run 的 Evolve 时间为 29.0928 s 和 29.2783 s，四个输出文件（`bssn_ADMQs.dat`、`bssn_BH.dat`、`bssn_constraint.dat`、`bssn_psi4.dat`）逐位一致。

硬件计数器（stat run）：

- task-clock：772753.23 ms，平均约 24.047 个 CPU 正在工作；
- IPC：1.79；
- branch miss：0.52%；
- L1D load miss：3.84%；
- LLC load miss：37.09%；
- dTLB load miss：3.42%。

这些指标说明主要问题仍是数据访问和工作量分布，而不是异常的分支预测或 TLB 抖动。

perf flat 热点如下：

| 函数 | 周期占比 | 说明 |
|---|---:|---|
| `compute_rhs_bssn_` | 40.74% | BSSN RHS 总入口，包含多个导数和代数阶段 |
| `__memcpy_sve` | 12.84% | 网格/层间数据搬运 |
| `lopsided_` | 8.69% | beta advection 的有限差分 |
| `__memset_sve_zva64` | 6.03% | 输出和 ghost 临时区初始化 |
| `prolong3_` | 4.92% | AMR prolongation |
| `fdderivs_` | 4.61% | Ricci/二阶导数相关导数核 |
| `rungekutta4_rout_` | 2.57% | RK4 外层推进 |
| `fderivs_` | 2.37% | 普通一阶导数 |
| `kodis_` | 2.12% | Kreiss-Oliger dissipation |
| `symmetry_bd_` | 1.89% | ghost 对称边界填充 |
| `restrict3_` | 1.67% | AMR restriction |

调用图显示 `compute_rhs_bssn_` 主要由 `bssn_class::Step` 的 OpenMP worker 调用；在其子树中，`lopsided` 约 7.22%，`fdderivs` 约 5.26%，`fderivs` 约 4.36%。这与 flat profile 一致：P6 选择的 `fderivs` 确实是热点，但只占约 2.4% 的 flat 周期，因此即使该核获得 20% 加速，端到端理论上也只有约 0.5% 的上限。

带源码行的 profile 还显示：`lopsidediff.f90` 的深部 SIMD/边界路径分别占若干 1% 左右的采样，`fdderivs` 的清零和对称边界子调用也很明显。它支持下一步优先检查公共 beta 访问、ghost 初始化和 AMR 搬运，而不是继续扩大 `fderivs` 批大小。

## 5. 结论

本阶段没有可接受的端到端性能提升，因此 **不启用 `AMSS_ENABLE_FDERIVS_BATCH`**。实验代码和脚本保留在默认关闭路径下，方便后续回归；生产结果与 P6 前保持一致。

这次实验得到的实际结论是：

- 仅合并独立导数调用，不足以抵消每个字段仍然存在的 ghost 分配、边界填充和清零；
- `fderivs` 的算法结构可以 SIMD，但它不是当前最值得投入的端到端杠杆；
- 当前更大的可测收益空间在 `memcpy`、`memset`、`lopsided`、`fdderivs` 和 `compute_rhs_bssn` 外层调度；
- 任何进一步批处理都应先做寄存器/栈访问检查，并从两个字段的小批开始，不能直接把 24 个字段合成一个“大 kernel”。

## 6. 可复现实验和提交

A/B 脚本：[`hpc_abe_fderivs_batch_sweep.sh`](../hpc_abe_fderivs_batch_sweep.sh)。
实验 profile：`profile/abe-fderivs-batch-20260824T134347Z-14`。
默认 fresh profile：`profile/abe-20260824T141547Z-14`。

本阶段提交只包含默认关闭的实验开关、实现、脚本和本报告；不会改变默认编译配置。
