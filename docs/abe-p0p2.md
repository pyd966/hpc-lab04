# ABE P0/P2：分析归约与通信阶段报告

## 阶段目标

本阶段把 P0 和 P2 合并实现。P0 针对分析路径中大量小规模归约，P2 针对同一路径中可以在不改变数学结果的情况下减少的集体通信调用。没有修改 P1 的 OpenMP block 调度，也没有开始 P3（更深的内核级并行/线程池重构）。

## 代码修改

### 1. 波形实部/虚部合并归约

`src/surface_integral.C:34` 新增 `reduce_wave_pair`。原来 `surf_Wave` 的三个 overload 都分别对 `RP_out` 和 `IP_out` 调用一次 `MPI_Allreduce`，现在把它们暂存为连续的 `2*NN` 双精度数组，只做一次 `MPI_Allreduce`，再拆回 `RP` 和 `IP`。通信操作仍是逐元素 `MPI_SUM`，所以数学运算顺序和结果定义不变。

### 2. ADM 七个量合并归约

`src/surface_integral.C:1033` 和 `:1267` 将质量、三维动量、三维角动量组成长度为 7 的数组，一次完成 `MPI_Allreduce`。原实现是质量 1 次、角动量 3 次、动量 3 次，共 7 次标量 collective；现在变成 1 次长度 7 的 collective。之后按原有顺序缩放并写入 `Rout[0..6]`。

### 3. 插值 shell 值与 ownership weight 合并

`src/MPatch.C:16` 新增 `reduce_interp_values`。`Patch::Interp_Points` 需要同时归约 `NN*num_var` 个 shell 值和 `NN` 个整数 ownership weight；CPU 路径将 weight 转为 double，和 shell 值放入一个 buffer，做一次归约后再转回整数。ownership count 很小且可以被 double 精确表示，因此不会改变除法和多重 owner 检查。GPU 分支保持原来的设备路径。

这个修改减少了 collective 次数，但会带来一次 5 MB 级临时 buffer 的打包/拆包；因此它是需要实测验证的通信优化，不假定一定加速。

### 4. 未修改的通信

RK4 阶段间的 error `MPI_Allreduce`、AMR 的 `Parallel::Sync`、`MPI_Isend/Irecv/Waitall` 仍保持原有时序。它们涉及 ghost 数据和时间层状态，不能仅因为调用次数多就删除。程序主路径中仍没有 `MPI_Barrier` 或 `MPI_Reduce`。

## 正式实验

正式 job：`131930`，profile 目录：
`profile/abe-20260821T134140Z-14-p0p2-retry/`。

配置为 `2 MPI ranks * 15 OpenMP threads`，rank 绑定到两段不重叠的 15-core 区域。输入和 P1 相同，演化区间为 `t=0..4`，编译为 `-O3 -g -fno-omit-frame-pointer`；为了避免此前 cluster `perf record` 缓冲写满，第二遍 profile 使用 `19 Hz` 采样。两遍运行都完成。

### 时间

| 配置 | `Total Evolve Time` | 说明 |
|---|---:|---|
| P1，2 MPI × 15 OpenMP | `227.911 s` | 前一阶段 record |
| P0/P2，2 MPI × 15 OpenMP | `227.731 s` | 正式 record |
| P0/P2，2 MPI × 15 OpenMP | `228.166 s` | 正式 perf stat |

P0/P2 相对 P1 的变化约 `-0.08%`，在节点噪声范围内，不能宣称有可见端到端加速。另有一个 30 MPI × 1 的隔离 stat job（`132003`）：`Total Evolve Time = 174.05 s`，与原始 baseline 的约 `173.67 s` 相差约 `0.2%`，说明归约合并没有引入明显回归。该 job 的 `perf record` 在第一 timestep 后因集群磁盘写入失败，未用于调用图结论。

### 正确性

正式 job 的 `stat` 和 `record` 两次输出逐行比较（忽略文件的前两行时间戳头部）：

- `bssn_ADMQs.dat`：numerical rows identical
- `bssn_BH.dat`：numerical rows identical
- `bssn_constraint.dat`：numerical rows identical
- `bssn_psi4.dat`：numerical rows identical

这至少验证了本阶段的归约布局、weight 转换和后处理没有改变固定输入的数值结果。

## P0/P2 profile 结果

正式 P0/P2 `perf stat` 计数器为：IPC `2.20`，branch miss `0.30%`，L1D miss `1.68%`，LLC miss `47.53%`，dTLB miss `1.39%`。与 P1 的 IPC `2.19`、branch miss `0.33%`、LLC miss `47.51%` 基本相同，因此本阶段没有改变 RHS 的计算/访存性质。

DSO 样本分布为 ABE `69.18%`、libc `12.90%`、libmpi `10.49%`、libopen-pal `6.09%`、libgomp `0.58%`。主要 flat 热点仍是：`compute_rhs_bssn_` `32.96%`、`polint_` `7.14%`、`kodis_` `6.71%`、`fdderivs_` `5.49%`、`lopsided_` `5.16%`，以及 MPI/内存拷贝相关地址。

调用图给出的重点是：

- `compute_rhs_bssn_` inclusive `59.52%`，说明 P1 并行的 RHS 计算仍是绝对主热点；
- `surface_integral::surf_MassPAng` `22.28%`；
- `PMPI_Allreduce` `12.55%`；
- `reduce_interp_values` `12.47%`；
- `Parallel::transfer` `10.98%`；
- `surf_Wave` `3.16%`。

这里的 `reduce_interp_values` 是新 helper 的 inclusive 样本，包含其中的 MPI 调用和 buffer 打包/拆包。它与 `PMPI_Allreduce` 的比例接近，说明在 2 rank 配置下，shell 数组复制成本已经和 collective latency 同量级；这解释了为什么端到端时间没有明显改善。ADM 七个标量归约的调用次数确实从 7 次降到 1 次，但它们在完整演化中占比太小，无法抵消 RHS 和插值 shell 的成本。

## 结论

P0/P2 已完成编译、固定输入时间测试、硬件计数器、调用图 profile 和数值复现。它消除了分析路径中 7 次标量 ADM 归约，并将波形与插值归约分别合并，通信语义保持不变；但在当前 2×15 混合配置下端到端收益低于测量噪声。30×1 stat 也没有出现明显回归。

当前最重要的结论不是继续堆叠更多小型 `MPI_Allreduce` 合并，而是：

1. `compute_rhs_bssn_` 及其 `kodis/fdderivs/lopsided` 内层循环仍占约 60% 的调用图时间，是计算主瓶颈；
2. 插值 shell 的大消息归约仍约占 12%，且临时打包会抵消部分 collective 次数收益；后续应考虑持久化缓冲区或直接改变 shell 数据布局，而不是反复 `new/delete`；
3. `Parallel::transfer` 约 11%，AMR halo 的构造、打包和 `Waitall` 仍有优化空间，但需要单独验证数据依赖后再改；
4. P3 暂不实施。下一轮若继续，优先级应是减少 `Interp_Points` 的临时分配/复制，或在不改变 MPI rank 数的前提下把 RHS 的点循环并行化；同时必须保持现有数值复现检查。
