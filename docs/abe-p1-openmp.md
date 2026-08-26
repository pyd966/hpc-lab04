# ABE P1：OpenMP block 并行阶段报告

## 目标与范围

这一阶段只做 P1：把 ABE 中可以独立处理的本地网格 block 交给 OpenMP 线程。MPI 的职责没有被移除：rank 仍然拥有自己的 block，`MPI_Allreduce`、halo `Parallel::Sync`、RK4 阶段之间的同步仍由原来的主线程顺序执行。这样可以先验证线程并行是否保持数值结果，再单独处理 P0/P2 的通信和分析归约。

修改位置如下：

- `src/Parallel.C:124` 的 `Parallel::distribute` 在 OpenMP 开启时把目标切分数提高到 `MPI ranks * OMP threads`，使一个 rank 可以拥有多个较小的 block；MPI rank 编号和 block 的 owner 没有改变。
- `src/bssn_class.C:1777` 的 `bssn_class::Step` 在进入 RK4 前建立本 rank 的 `(Patch*, Block*)` 工作表。predictor、三个 corrector 以及中间的 list swap 使用 `omp parallel for schedule(static)`。
- `src/bssn_class.C:2464` 的 `Compute_Psi4` 只并行本地 `f_getnp4` 计算，随后仍按原顺序调用 `Parallel::Sync`。

分析积分 `AnalysisStuff -> surf_Wave/surf_MassPAng -> Interp_Points` 没有在本阶段改动，因为这些函数内部包含 MPI 归约；把它们放进 OpenMP 区域会让 MPI 调用与线程级工作混在一起，风险高且不利于判断收益。

## 测试方法

远端 job：`131676`，profile 目录：
`profile/abe-20260821T130325Z-14-p1-openmp/`。

两次运行使用同一个固定输入，只把 `ABE::TotalTime` 改为 `4.0`；第一次 `perf stat`，第二次 `perf record -F 99 --call-graph fp`。P1 使用 `2 MPI ranks * 15 OpenMP threads`，绑定信息为 rank 0 的 core 64--93、rank 1 的 core 94--123。编译使用 `-O3 -g -fno-omit-frame-pointer`，没有使用 `-Ofast`。

## 性能结果

| 配置 | 演化时间 `t=0..4` | 相对 baseline |
|---|---:|---:|
| baseline，30 MPI × 1 thread | 约 `173.7 s` | 1.00x |
| P1，2 MPI × 15 OpenMP | `227.659 s` | `0.76x`（慢约 31%） |

`perf stat` 的 P1 运行耗时为 `233.497 s`（包含初始化和启动开销）；`perf record` 的 `Total Evolve Time` 为 `227.911 s`，两次差异约 `0.1%`。四个主要输出文件在两次运行中逐行比较（忽略时间戳头部）均完全一致：`bssn_ADMQs.dat`、`bssn_BH.dat`、`bssn_constraint.dat`、`bssn_psi4.dat`。

P1 的硬件计数器：IPC `2.19`，branch miss `0.33%`，L1D miss `1.69%`，LLC miss `47.51%`，dTLB miss `1.39%`。与 baseline 的 IPC `1.89`、LLC miss `50.91%`、dTLB miss `7.53%` 相比，单个计算线程的局部访存指标没有恶化；真正的问题是并行度没有被填满：`task-clock` 对应的平均 CPU 使用量只有 `4.65` 个 CPU。

## 热点和调用路径

P1 的 DSO 样本分布为：ABE `69.37%`、libc `12.93%`、libmpi `10.13%`、Open MPI 的 libopen-pal `6.27%`、libgomp `0.54%`。调用图中最重要的路径是：

1. `bssn_class::Evolve -> RecursiveStep -> Step`；
2. `Step` 的 OpenMP predictor/corrector worker 中调用 `compute_rhs_bssn_`，该 Fortran RHS 内核本身占 flat samples `33.41%`，从 `Step` 路径向下累计约 `59.88%`；
3. `Step -> AnalysisStuff -> surf_MassPAng` 累计约 `22.00%`，其中 `PMPI_Allreduce` 约 `12.37%`；
4. `RestrictProlong -> OutBdLow2Hi -> Parallel::transfer` 累计约 `8.98%`，`MPI_Waitall` 约 `3.81%`，说明 halo/AMR 同步仍是第二类成本。

这说明 P1 没有改变程序的主要数学热点，RHS 计算仍是计算部分的核心；但在本题的固定网格上，分析归约和同步的固定成本不会随 OpenMP 线程数降低。

## 为什么本阶段没有加速

`Parallel::distribute` 的细分受到 ghost/buffer 最小宽度和三维网格形状限制。日志显示 level 0 即使目标切分数是 30，也只实际使用约 9 个有效 block；更细的移动层只有两个 patch，能并行的 block 数更少。使用两个 MPI rank 时，每个 rank 往往只有 4--5 个可运行 block，15 个线程中大部分时间没有工作。baseline 的 30 个 rank 虽然也存在 level 0 的 9-way 限制，但它避免了 OpenMP 区域开销，并且 MPI rank 的布局正好覆盖了不同 level 的 block。

因此本阶段得到的是一个正确的 OpenMP 执行骨架，而不是最终的性能配置。继续单纯增加线程数没有意义；要获得收益，必须同时减少分析中的集体通信，并重新设计 block 粒度或把线程并行推进到 block 内的点循环。后者会扩大 P1 的范围，暂不在本阶段混入。

## 结论与下一阶段

P1 通过了编译、固定输入运行、数值复现和 `perf` 采样，但在当前网格和 2×15 配置下比 baseline 慢约 31%。下一阶段按要求把 P0 与 P2 合并：先把 `surf_Wave`/`surf_MassPAng` 的多个标量 `MPI_Allreduce` 合并为少量向量归约，再减少分析路径上不必要的通信/同步开销；每次修改后仍使用相同的 `t=0..4` 输入、时间测试、profile 和输出比对。
