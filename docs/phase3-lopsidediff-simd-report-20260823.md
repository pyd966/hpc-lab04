# 阶段：lopsidediff 内部模板 SIMD

日期：2026-08-23

本阶段只处理 ABE 的 `lopsidediff.f90`，没有改任务划分、线程数、缓存布局或其他热点函数。基线是已完成 `fdderivs` SIMD 的提交 `b2cd29c`；编译仍使用 `-O3 -g -fno-omit-frame-pointer`，没有使用 `-Ofast`。

## 1. 目标和基线

`lopsided` 在每个网格点沿 x/y/z 三个方向计算带符号的四阶迎风差分。原始最内层 i 循环同时包含：速度正负判断、距离边界的判断，以及四阶/中心/低阶模板选择。编译器的诊断是：该循环包含不支持的控制流，整个 `lopsided` 函数自动向量化循环数为 0。

当前提交前的 profile：

| 指标 | fdderivs SIMD 基线 |
|---|---:|
| ABE `Total Evolve`（perf stat） | 38.7123 s |
| `lopsided_` 样本占比 | 7.28% |
| `compute_rhs_bssn_` 样本占比 | 49.83% |
| 平均 CPU | 17.005 / 30 |
| IPC | 1.61 |
| L1D miss | 3.52% |
| LLC miss | 48.28% |
| dTLB miss | 2.83% |

原始热点行主要是 z 正向四阶模板（2.29%）、z 负向四阶模板（1.24%）和 y 正向四阶模板（1.10%）。这表明热点确实集中在迎风模板，而不是 `symmetry_bd` 或边界初始化。

## 2. 实现

新增 CMake 选项 `AMSS_ENABLE_LOPSIDEDIFF_SIMD`，默认 ON；它只给 CPU ABE 定义 `AMSS_LOPSIDEDIFF_SIMD`，关闭该选项即可回到原循环。

Fortran 中按每个方向取出深内部区域：

```text
ibegin = max(1, imin+3), iend = min(ex(1)-1, imax-3)
```

y、z 方向使用相同规则。这个区域中的任意点都满足正向和负向四阶模板的全部下标条件，所以不再需要边界阶数分支。最内层 i 循环加 `!$omp simd`。

为了让 GNU Fortran 生成 SIMD 掩码，符号选择写成：

```text
max(v,0) * forward_stencil - min(v,0) * backward_stencil
```

它与原逻辑等价：正速度只保留正向模板，负速度只保留负向模板，零速度贡献为零。x、y、z 仍按原顺序分别更新 `f_rhs`，避免不必要的浮点运算顺序变化。深内部点从原循环跳过，边界薄层继续使用原来的分支代码，因此对称边界和低阶边界模板没有改变。

本地 `gfortran -O3 -fopenmp -cpp -DAMSS_LOPSIDEDIFF_SIMD -fopt-info-vec-optimized` 诊断确认：新内部循环被向量化，使用 16-byte vectors；关闭宏的 fallback 也能独立编译。

## 3. HPC 验证

完整 profile 工件：`profile/abe-20260823T124919Z-1/`。使用与基线相同的固定输入、1 个 OpenMP-only 进程、30 个线程、`OMP_PLACES=cores`、`OMP_PROC_BIND=close`，并分别运行 `perf stat` 和 `perf record`。

| 指标 | 基线 | lopsided SIMD | 变化 |
|---|---:|---:|---:|
| `Total Evolve`（stat） | 38.7123 s | 38.7145 s | +0.006% |
| `Total Evolve`（record） | 38.3733 s | 38.7265 s | 独立采样波动 |
| `lopsided_` 样本占比 | 7.28% | 7.48% | +0.20 个百分点 |
| `compute_rhs_bssn_` | 49.83% | 50.04% | 基本不变 |
| 平均 CPU | 17.005 | 16.986 | 基本不变 |
| IPC | 1.61 | 1.60 | 略降 |
| L1D miss | 3.52% | 3.95% | 升高 |
| LLC miss | 48.28% | 48.52% | 基本不变 |
| dTLB miss | 2.83% | 2.94% | 基本不变 |

四个关键输出文件与标量基线逐行一致：`bssn_ADMQs.dat`、`bssn_BH.dat`、`bssn_constraint.dat`、`bssn_psi4.dat`。课程检查通过，`perf` 没有丢样本，rank/线程配置保持 1 进程 30 线程。

本次作业节点为 TaiShan-v120，CPU 可见 60 个逻辑 CPU，任务绑定在 NUMA node 0；整机为 4 NUMA 节点、每节点 64 个逻辑 CPU，支持 SVE。基线和候选都采用同样的绑定方式，但 profile 仍是不同作业，时间差应按独立作业波动解释。

## 4. 结果解释

这次没有获得可测的端到端加速，原因不是 SIMD 没有生成，而是改写后的每个 SIMD lane 同时计算正向和负向两个 stencil，再通过 `max/min` 选择有效结果。原标量代码按符号只计算其中一个模板；因此新路径减少了分支，却增加了另一套 stencil 的加载和乘加。`lopsided` 只占总程序约 7.3%，即使它本身有明显改善，端到端收益上限也有限；当前额外访存把收益完全抵消。

因此本阶段结论是：

1. “内部区域 + SIMD”在编译层面可行，数值结果保持一致。
2. 当前无条件计算双模板不是合适的最终优化，不能宣称性能提升。
3. 下一步若继续研究，应优先尝试按速度符号生成正/负点索引或分块分类，使 SIMD 循环只计算实际方向；但索引表的构造和额外访存也必须用 profile 验证。
4. 在没有更好符号分类方案前，不应把 lopsided SIMD 当作主要性能收益来源；更大的收益仍应从 `compute_rhs_bssn_`、访存布局和任务划分中寻找。
