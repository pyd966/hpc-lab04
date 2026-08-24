# P6 阶段报告：消除 `symmetry_bd` 的冗余整块清零

日期：2026-08-24  
阶段：P6，减少边界准备阶段的无效内存写入  
结论：实现正确，收益 modest 但稳定，生产默认开启

## 1. 为什么检查这段代码

前一轮正式 profile 中，`__memset_sve_zva64` 约占 6.95% cycles，是除
`compute_rhs_bssn`、`memcpy` 和 `lopsided` 之外最大的单一运行时符号。调用路径主要是
`fderivs/fdderivs -> symmetry_bd`。因此这一步先检查清零是否真的覆盖了后续不会写到的
区域，而不是直接把所有 allocation 或初始化删除。

`symmetry_bd` 的输出数组范围是
`(-ord+1:extc(1), -ord+1:extc(2), -ord+1:extc(3))`。它先把正区域
`1:extc(1),1:extc(2),1:extc(3)` 从输入 `func` 拷贝过来，然后分别按 x、y、z 方向
写入全部低端 ghost plane：

```fortran
do i=0,ord-1
  funcc(-i,1:...,1:...) = ...
enddo
do i=0,ord-1
  funcc(:,-i,...) = ...
enddo
do i=0,ord-1
  funcc(:,:,-i) = ...
enddo
```

这三个循环覆盖所有负索引 plane，包括边、角和坐标平面交叉处；输出没有高端 ghost
区域。因此在它们之前的 `funcc = 0.d0` 不会留下任何需要被读取的未初始化元素，属于
完整数组的冗余写入。需要注意的是，这个结论只适用于 `symmetry_bd`，不能据此删除
其它数组的初始化。

## 2. 实现

新增 CMake 选项 `AMSS_ENABLE_SYMMETRY_BD_NO_CLEAR`，现在默认 `ON`。在
`src/fmisc.f90:symmetry_bd` 中用预处理宏包住整块清零：

```fortran
#ifndef AMSS_SYMMETRY_BD_NO_CLEAR
  funcc = 0.d0
#endif
```

关闭该选项仍保留原始实现，便于回归和定位。`hpc_abe_profile.sh` 与独立 sweep 脚本
都能通过环境变量或 CMake 参数切换。此次没有改变 stencil、边界条件、线程划分或浮点
计算顺序。

## 3. A/B 时间和正确性

HPC 作业 159841，运行顺序 `OFF, ON, ON, OFF`，每次均为单 MPI 进程、30 个 OpenMP
线程、`OMP_PLACES=cores`、`OMP_PROC_BIND=close`，静态/移动层 block 目标为 24/30，
`dynamic,1` 调度，测量 `t=0..4`。

| 版本 | Evolve 平均 (s) | Total 平均 (s) | 平均活跃 CPU | 结果 |
|---|---:|---:|---:|---|
| OFF | 29.6251 | 34.0572 | 22.003 | 两次均 PASS |
| ON | 29.4476 | 33.7040 | 22.032 | 两次均 PASS，输出逐位一致 |

ON 相对 OFF 的 Evolve 平均改善约 **0.60%**，Total 平均改善约 **1.04%**。这个幅度
小于单次运行的节点噪声，但两个 ON 都比对应的 OFF 快，且平均活跃 CPU 没有下降，说明
不是因为线程数或调度改变造成的假收益。

## 4. 正式 profile

候选 profile：`profile/abe-20260824T215011Z-14`，作业 159854，`perf record` 无丢样本。
比较基线 profile `profile/abe-20260824T213634Z-14`：

| 符号 | 基线 | P6 ON |
|---|---:|---:|
| `compute_rhs_bssn_` | 50.31% | 51.28% |
| `__memcpy_sve` | 8.82% | 9.46% |
| `__memset_sve_zva64` | 6.95% | 5.27% |
| `lopsided_core_` | 8.35% | 8.42% |
| `prolong3_` | 4.49% | 4.49% |
| `fdderivs_` | 4.30% | 4.37% |
| `fderivs_` | 2.28% | 2.34% |
| `symmetry_bd_` | 1.58% | 1.33% |

硬件计数器也没有出现异常变化：P6 ON 的 IPC 1.45，L1D miss 4.06%，LLC miss
49.22%，dTLB miss 3.69%，branch miss 0.39%，平均活跃 CPU 22.04。清零省下的时间主要
被其它内存流量占据，所以 `memcpy` 和主要计算在 flat profile 中的比例上升；这不是
优化失效，而是典型的热点占比重新归一化。

## 5. 结论和后续方向

这项修改值得保留：代码证明了清零在该函数中没有语义作用，端到端有小幅稳定收益，且
没有改变数值结果。它不能单独解决性能目标，因为剩余的内存流量仍由 `symmetry_bd` 的
正区复制、导数临时数组以及 RHS 其它数组扫描构成。下一步应继续针对这些真实写入和
读取做实验，例如复用明确生命周期的 ghost 缓冲区、减少完整数组复制，或优化 block
级任务分配；不应再删除没有覆盖证明的初始化。

## 6. 可复现实验

- A/B：[`hpc_abe_symmetry_clear_sweep.sh`](../hpc_abe_symmetry_clear_sweep.sh)
- profile：[`hpc_abe_profile.sh`](../hpc_abe_profile.sh)
- 候选 profile：`profile/abe-20260824T215011Z-14`
