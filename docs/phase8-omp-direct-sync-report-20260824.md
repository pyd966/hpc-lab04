# ABE P2：同级 Sync 直拷贝优化报告

日期：2026-08-24

## 1. 阶段目标

当前单进程 OpenMP 的同级同步仍然是两步：

1. 每个源片段先由 `f_copy` 写入共享 workspace；
2. 所有源片段完成后，再由 `f_copy` 从 workspace 写入目标 ghost 区。

这保证了 pack-before-unpack 的语义，但同级数据至少搬运两次。P2 只处理 `type == 1` 的同级复制；restrict/prolong 和混合 AMR transfer 仍使用原路径。

## 2. 实现

- CMake 新增 `AMSS_ENABLE_OMP_DIRECT_SYNC`，默认 OFF，便于回退到原实现。
- `OmpCachedSyncTransfer` 在构建缓存计划时计算 `direct_safe`。
- 安全判断不再比较整个 Block 的数组，而是复现 `f_copy` 的整数索引公式，为每个操作建立源读矩形和目标写矩形。
- 只有以下冲突全部不存在时才使用 direct path：
  - 当前操作的源、目标矩形在同一数组上重叠；
  - 两个操作的目标矩形重叠；
  - 一个操作的目标矩形覆盖另一个操作的源读矩形。
- direct path 在一个 OpenMP team 中直接执行 `f_copy(source, destination)`，不申请或写入 workspace。任何非同级计划、索引计算失败或存在冲突的计划都回退到原来的 pack/unpack。

## 3. 失败的初版与修正

第一版用整块数组地址范围判断是否重叠。HPC 诊断显示所有计划都是 `direct_safe=0`，原因是同一目标 Block 的多个 ghost 面共享同一整块数组，即使实际矩形不相交也被保守拒绝。该作业在完成无效的第一轮后取消，没有把它当成性能结论。

第二版改为实际三维索引矩形判断。HPC 作业 159142 中记录了 178 个 direct 计划，全部为 `direct_safe=1`，覆盖了包含 8,016、6,144、1,824 个操作的主要同步计划。

## 4. 固定输入 A/B 结果

作业：159142；60 个 CPU，30 个 OpenMP 线程，`OMP_PLACES=cores`、`OMP_PROC_BIND=close`；静态层 24 线程，移动层 30 线程；演化窗口 t=0..4。运行顺序为 pack/direct/direct/pack。

| 路径 | evolve 平均 (s) | total 平均 (s) | 平均 CPU | 正确性 |
|---|---:|---:|---:|---|
| pack（原路径） | 30.1226 | 34.2127 | 22.143 | PASS |
| direct（矩形安全路径） | 29.7234 | 33.9382 | 22.067 | PASS |

按同一作业内交错均值计算，evolve 减少约 1.33%，ABE total 减少约 0.80%（单次总时间的原始值分别为 34.1721/33.9886 和 34.2532/33.8877，作业结果文件为 `profile/abe-sync-direct-20260824T183528Z-15/results.tsv`）。四次运行的 course check 均为 PASS，逐项输出比较均为 bitwise yes。

## 5. 优化后 profile

优化前 profile：`profile/abe-20260824T141547Z-14`；优化后 profile：`profile/abe-20260824T184630Z-14`。两次 profile 都使用 `-O3 -g -fno-omit-frame-pointer`，并对 t=0..4 做 `perf stat` 和 `perf record`。

| 函数/路径 | 优化前 samples | 优化后 samples |
|---|---:|---:|
| `compute_rhs_bssn_` | 40.74% | 41.74% |
| `__memcpy_sve` | 12.84% | 11.14% |
| `lopsided_` | 8.69% | 9.23% |
| `prolong3_` | 4.92% | 4.91% |
| `fdderivs_` | 4.61% | 5.31% |
| `fderivs_` | 2.37% | 4.37% |
| cached Sync worker（children） | 约 12.51% | 10.41% |

优化后 `perf` 调用图中，Sync worker 的主要子项是 `copy_` 7.08% 和 `__memcpy_sve` 6.40%；原 profile 中 cached Sync children 约 12.51%，且 `__memcpy_sve` 为 12.84%。这与“去掉一份 workspace 搬运”相符。绝对 profile 时间受采样开销和节点负载影响，因此性能结论以同一作业内交错 A/B 为准。

## 6. 结论与边界

P2 已完成并值得保留：收益约 1%，数值结果不变，且不会改变 RHS、AMR 插值或线程任务划分。收益没有达到 10% 级别是合理的，因为同步只是总时间的一部分，且 direct path 只消除同级复制的一次中间搬运。

下一阶段应继续看 `omp_local_transfer` 中的 prolong/restrict workspace，以及 RHS/导数热点；不要再把 P2 的同级 copy 优化误认为已经解决了主要 `compute_rhs_bssn_` 计算瓶颈。
