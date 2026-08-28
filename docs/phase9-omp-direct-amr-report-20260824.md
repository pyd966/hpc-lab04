# ABE P3：AMR prolong/restrict direct transfer 报告

日期：2026-08-24

## 1. 目标与范围

P2 只消除了同级 Sync 的 workspace 搬运。本阶段继续检查 `omp_local_transfer` 的 type 2/3 AMR transfer：当前路径先把 `restrict3/prolong3` 的结果写入 workspace，再用 `f_copy` 写入目标 Block。优化只针对非 mixed transfer；`prolongcopy3/prolongmix3` 的 mixed 语义保持原路径。

## 2. 实现

新增三个默认关闭的 CMake 开关：

- `AMSS_ENABLE_OMP_DIRECT_AMR_TRANSFER`：直接把 type 2/3 插值结果写入目标 Block；
- `AMSS_ENABLE_OMP_DIRECT_AMR_SPLIT`：对同一目标数组中索引矩形不相交的片段拆成独立任务；
- P2 的 `AMSS_ENABLE_OMP_DIRECT_SYNC` 在本阶段固定为 ON，避免把两个阶段混在一起。

第一版按“目标数组”分组，组间 OpenMP 动态调度，组内按原 segment 顺序串行，保证多个片段写同一数组时不改变顺序。第二版复现目标矩形的整数索引范围；不相交片段各自成为任务，有重叠片段仍作为一个串行组。两版都只在 type 2/3、非 mixed 路径执行，默认 OFF 时完全回退原 pack/unpack。

## 3. 第一版 A/B

作业 159326 因家目录磁盘满在 CMake 临时测试阶段失败；清理本轮生成的临时目录后，在作业 159364 重试成功。固定输入为 t=0..4、30 个 OpenMP 线程、静态/移动层 24/30 线程、cores 绑定；P2 direct Sync 保持开启，运行顺序 base/amr/amr/base。

| 路径 | evolve 平均 (s) | total 平均 (s) | 平均 CPU | 正确性 |
|---|---:|---:|---:|---|
| base（P2 + 原 AMR） | 29.7605 | 33.9848 | 22.068 | PASS |
| amr（P2 + direct AMR） | 29.6886 | 33.9812 | 21.924 | PASS |

所有运行 course check 和逐项 bitwise 比较均通过。type 2/3 诊断显示每个 transfer 计划都实际走了 direct 分支，例如 2,160/2,664 个操作被分到 360/648 个目标数组组。evolve 只快约 0.24%，total 基本不变，且平均活跃 CPU 略低，说明组内串行抵消了省掉一次 copy 的收益。

## 4. 第二版 split A/B

作业 159470 比较 direct AMR 的两种调度，顺序 group/split/split/group：

| 路径 | evolve 平均 (s) | total 平均 (s) | 平均 CPU | 正确性 |
|---|---:|---:|---:|---|
| group（组内串行） | 29.6573 | 33.8244 | 22.052 | PASS |
| split（不相交矩形拆任务） | 29.5830 | 33.8525 | 22.045 | PASS |

split 的 evolve 平均约快 0.25%，但 total 约慢 0.08%，平均 CPU 没有提高。这个差异小于当前短窗口的节点/调度噪声，不能作为端到端收益。

## 5. Profile 证据

第一版 direct AMR profile：`profile/abe-20260824T192447Z-14`；第二版 split profile：`profile/abe-20260824T195229Z-14`；对照为 P2 profile `profile/abe-20260824T184630Z-14`。

| 热点 | P2 对照 | AMR direct | AMR split |
|---|---:|---:|---:|
| `compute_rhs_bssn_` | 41.74% | 41.76% | 42.57% |
| `__memcpy_sve` | 11.14% | 10.39% | 10.48% |
| `prolong3_` | 4.91% | 5.07% | 4.99% |
| `restrict3_` | 1.61% | 1.63% | 1.61% |
| `copy_` | 0.76% | 未列入前列 | 0.68% |

调用图中 `omp_local_transfer` children 从 P2 的约 9.22% 降到 direct AMR 的约 8.48%，说明 workspace-to-target copy 确实被消除了；但这部分只占总时间的一小部分，且分组调度改变了并行粒度，所以端到端没有可重复的正收益。split profile 的平均 CPU 约 21.5，硬件计数器为 IPC 1.82、branch miss 0.46%、L1D miss 3.81%、LLC miss 36.52%、dTLB miss 3.52%，没有出现新的异常瓶颈。

## 6. 结论

P3 的 direct AMR 两版都通过数值验证，证明“插值结果直接写目标数组”在非 mixed type 2/3 transfer 上语义可行；但当前问题规模下，收益约为 0%，不应默认开启。代码保留为 OFF 的实验开关，生产配置继续使用 P2 direct Sync + 原 AMR pack/unpack，避免增加默认调度复杂度。

下一步更值得投入的是 60 个物理核上的任务几何/线程配置和 `compute_rhs_bssn` 内部的访存、向量化；继续微调 AMR workspace 预计无法带来目标级加速。
