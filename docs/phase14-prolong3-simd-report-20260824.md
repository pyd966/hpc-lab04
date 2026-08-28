# Phase 14: `prolong3` 显式 SIMD 实验

## 目标与基线

当前 ABE 的 AMR 粗到细传递由 `src/prolongrestrict_cell.f90` 中
`ghost_width=3` 的 `prolong3` 完成。正式 profile 中它约占 4.4% 的周期，
热点行是 6 点插值的 `tmp2`/`tmp1` 数组表达式（约 4.4% 是函数级占比）。
本阶段只在最内层 `i` 循环增加
`!$omp simd private(tmp1,tmp2,cxI,ii,jj,kk)`，没有改变数值公式和循环结构。
开关为 `AMSS_ENABLE_PROLONG3_SIMD`，默认 OFF。

## A/B 测试

HPC job `160032`，目录
`profile/abe-prolong3-simd-20260824T224926Z-14`。两种版本均使用 1 个
OpenMP-only 进程、30 个线程、静态层 24 个 worker、移动层 30 个 worker，
`dynamic,1`，并交错执行 `OFF ON ON OFF`。

| 版本 | 平均 Evolve (s) | 平均总时间 (s) | 平均 CPU | 结果 |
|---|---:|---:|---:|---|
| OFF | 29.431650 | 33.668900 | 22.066 | 2/2 PASS |
| ON  | 29.363800 | 33.681750 | 21.954 | 2/2 PASS |

Evolve 表面上快约 0.23%，但总时间反而慢约 0.04%，低于同一节点运行噪声。
四次运行输出均 bitwise 一致，课程校验均 PASS。

## 正式 profile

HPC job `160040`，目录
`profile/abe-20260824T225351Z-13`，使用 `-O3 -g -fno-omit-frame-pointer`
并执行 `perf stat` 与 `perf record`。主要热点为：

| 函数/路径 | 周期占比 |
|---|---:|
| `compute_rhs_bssn_` | 49.93% |
| `__memcpy_sve` | 10.48% |
| `lopsided_core_` | 8.29% |
| `__memset_sve_zva64` | 5.31% |
| `fdderivs_` | 4.39% |
| `prolong3_` | 4.14% |

调用图中 `prolong3_`（含 ghost 准备）约 4.57%。编译器只报告了
`prolong3.f90:2149` 的 basic-block 向量化，没有报告整个 `i` 循环被向量化；
这是因为相邻 fine 点通过整数除法映射到重复的 coarse 点，并且每点都有奇偶分支。

硬件计数器为 IPC 1.45、L1D miss 4.14%、LLC load miss 49.41%、dTLB miss
3.69%、branch miss 0.36%，平均 CPU 利用率约 21.9 个逻辑 CPU。SIMD 开关没有
改善这些瓶颈指标。

## 结论

直接给原循环加 SIMD 不是有效的端到端优化：它没有消除重复的 6x6 插值，
也没有形成真正的跨点向量循环。因此保留 `AMSS_ENABLE_PROLONG3_SIMD` 作为
可回退实验开关，但默认关闭；下一阶段改为按 coarse x 索引成对复用 z/y
插值，直接减少算术和临时数组操作。
