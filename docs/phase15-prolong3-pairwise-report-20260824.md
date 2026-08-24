# Phase 15: `prolong3` 成对插值复用

## 为什么继续改写

Phase 14 的显式 SIMD 没有让主 `i` 循环真正向量化，原因是每个 fine 点都
重复计算同一 coarse x 索引对应的 6x6 z/y 插值。对于相邻的两个 fine 点，
若第一个点的 fine 坐标为偶数，它们共享 `cxI(1)`，区别只在最后一维 x
插值使用正向还是反向 6 点权重。因此本阶段把这两个输出合并：z/y 插值只
做一次，再分别计算两个 x 加权结果。

实现位于 `src/prolongrestrict_cell.f90` 的 `prolong3_pair_kernel`，由
`AMSS_ENABLE_PROLONG3_PAIRWISE` 控制，默认 OFF。若输出区从奇数 fine 坐标
开始，每个 `(j,k)` 行的第一个点调用一次 `prolong3_pair_point`，之后从偶数
fine 坐标开始按步长 2 成对处理；末尾不足一对时只写一个点。这样没有改变
插值系数或边界数据，只减少重复算术和临时数组赋值。

## A/B 测试

HPC job `160068`，目录
`profile/abe-prolong3-pair-20260824T230322Z-14`。固定 30 个 OpenMP
线程、静态层 24/24、移动层 30/30、`dynamic,1`，交错执行 `OFF ON ON OFF`。

| 版本 | 平均 Evolve (s) | 平均总时间 (s) | 平均 CPU | 结果 |
|---|---:|---:|---:|---|
| OFF | 29.581650 | 33.812250 | 22.106 | 2/2 PASS |
| ON  | 29.204900 | 33.392100 | 22.016 | 2/2 PASS |

Evolve 加速约 1.27%，总时间加速约 1.24%。四次运行均 bitwise 一致，课程
校验均 PASS。IPC 从约 1.45 降到 1.42，但 L1D miss（4.05%→4.14%）、LLC
miss（49.23%→49.34%）和 dTLB miss（3.69%→3.72%）只发生小幅波动，说明
收益主要来自少算了一遍插值，而不是缓存或调度变化。

## 正式 profile

HPC job `160074`，目录 `profile/abe-20260824T230753Z-14`，使用带调试
符号的 `-O3`，并执行 `perf stat`/`perf record`。主要热点为：

| 函数 | 周期占比 |
|---|---:|
| `compute_rhs_bssn_` | 50.85% |
| `__memcpy_sve` | 10.61% |
| `lopsided_core_` | 8.03% |
| `__memset_sve_zva64` | 5.30% |
| `fdderivs_` | 4.27% |
| `prolong3_pair_kernel_` | 2.34% |
| `prolong3_pair_point_` | 0.55% |

调用图中 `prolong3_`（含新内核）约 3.41%，相对之前约 4.4% 的
`prolong3` 函数占比下降约四分之一，与成对复用的预期一致。新的主要瓶颈
已经回到 `compute_rhs_bssn_`、内存复制和 lopsided/导数计算，而不是 AMR
插值本身。

硬件计数器：IPC 1.44、L1D miss 4.16%、LLC load miss 49.26%、dTLB miss
3.69%、branch miss 0.36%，平均约 21.9 个逻辑 CPU。没有出现异常的分支或
TLB 退化。

## 结论

这是目前对 `prolong3` 最有效且数值风险可控的改写，建议在当前 AArch64
配置中保持 `AMSS_ENABLE_PROLONG3_PAIRWISE=ON`。它不能改变 RHS 占据约一半
周期这一事实；后续优化重点应转向 RHS 内部数据复用/内存访问，以及剩余的
`restrict3` 和复制热点。
