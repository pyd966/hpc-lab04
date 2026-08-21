# TwoPuncture 阶段 1：消除热点内存分配

记录日期：2026-08-21。

## 修改范围

本阶段只处理 profile 已确认的高频临时分配，不改数值算法和迭代顺序：

- `chebft_Zeros()`、`chebft_Extremes()`、`fourft()` 和
  `Derivatives_AB3()` 的短向量改为可复用的 `TransformWorkspace`；
- `F_of_v()`、`J_times_dv()` 和 `JFD_times_dv()` 中长度为 `nvar`
  的临时导数数组改为 `PointWorkspace`；
- `LineRelax_be/al()` 的五组 line buffer 和 `ThomasAlgorithm()` 的四组
  三对角求解 buffer 改为 `LineWorkspace`；
- 三类 workspace 均为 `thread_local`，为后续 OpenMP 做准备：每个 worker
  拥有独立 scratch，不会并发覆盖；
- Thomas scratch 由 `LineRelax` 显式传入，并用 GCC/Clang 的
  `__restrict` 告诉编译器这些数组互不重叠。

Newton、BiCGSTAB、矩阵和最终输出等长生命周期数组仍按原有边界分配和释放。
它们每次求解只创建少数几次，并不是本阶段的热点。

## 测量方法

最终作业号为 `128484`，运行节点为 TaiShan-v120，输入仍为
`nA=50, nB=50, nphi=26`。编译参数保持 baseline 的：

```text
-O3 -g -fno-omit-frame-pointer
```

脚本先用 `perf stat -d -d` 完整运行一次，再用
`perf record -F 99 --call-graph fp` 独立运行一次。原始数据位于：

```text
profile/twopuncture-20260821T032358Z-13/
```

## 性能结果

| 指标 | baseline | 阶段 1 | 变化 |
| --- | ---: | ---: | ---: |
| wall time | 286.505 s | 288.720 s | +0.77% |
| cycles | 826.013 B | 808.832 B | -2.08% |
| instructions | 2146.051 B | 2067.696 B | -3.65% |
| IPC | 2.60 | 2.56 | -1.54% |
| 测量平均频率 | 2.886 GHz | 2.804 GHz | -2.84% |
| L1D miss rate | 2.51% | 2.11% | -0.40 pp |
| LLC miss rate | 0.05% | 0.08% | +0.03 pp |
| dTLB miss rate | 0.35% | 0.29% | -0.06 pp |
| branch miss rate | 1.22% | 1.38% | +0.16 pp |

结论需要分两层看：

1. 这一次 wall time 没有得到可声称的加速，反而慢了 0.77%。不能把它写成
   wall-clock speedup。
2. 在完整收敛路径和输出完全相同的前提下，执行指令减少 3.65%，cycles 减少
   2.08%。本次节点平均频率比 baseline 低 2.84%，把代码工作量的下降遮住了。
   因此合理结论是“分配工作已经去掉，约有 2% 的 cycle 收益，但一次 wall time
   落在集群频率波动范围内”，而不是宣称稳定的端到端加速。

## Profile 结果

最终 profile 收集约 28K 个 cycles 样本且无丢样。主要调用路径为：

| 调用路径/函数 | inclusive 或 self 占比 |
| --- | ---: |
| `bicgstab()` inclusive | 97.02% |
| `relax()` inclusive | 66.85% |
| `LineRelax_be()` inclusive | 37.96% |
| `LineRelax_al()` inclusive | 28.87% |
| `ThomasAlgorithm()` self | 13.61% |
| `J_times_dv()` inclusive | 28.09% |
| `Derivatives_AB3()` inclusive | 26.86% |
| `cos()` self | 23.65% |

baseline 中 allocator 相关符号合计至少占 6.07% samples；最终 flat profile
在 0.5% 展示阈值以上已经没有 `malloc/free/new/delete`。百分比不能直接解释为
6.07% wall-clock 加速，因为热点被删除后其余函数的相对占比会重新归一化，而且
缓存、别名分析和节点频率都会影响总 cycles。

第一次实现（作业 `128401`）只把 scratch 放入 TLS vector，Thomas 每条线又查找
一次 TLS，并使数组别名关系对编译器不清楚。其 wall time 为 290.869 s，
Thomas self 升到 17.99%。最终版本将 scratch 显式传给 Thomas 并标注 no-alias，
Thomas self 恢复到 baseline 的 13.60% 附近。这次失败重试说明：
“少了 malloc”不自动等于更快，必须重新 profile 生成代码所处的热点。

## 正确性

- 6 次 BiCGSTAB 的迭代数和逐步打印残差与 baseline 一致；
- `puncture_parameters_new.txt` 与 baseline 逐字节一致；
- `Ansorg.psid` 去掉首行生成时间后与 baseline 逐字节一致；
- 最终仍得到 `mp=0.576976`、`mm=0.378578`、总 ADM mass `0.983557`。

