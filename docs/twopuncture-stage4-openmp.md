# TwoPuncture 第四阶段：OpenMP 并行化

> 后续说明：提交 `f7814c7` 已把默认优化级别从 `-Ofast` 恢复为 `-O3`。
> 使用当前严格 `-O3 -march=native` 配置复测（作业 129616），30/60 线程分别为
> 11.399 s 和 11.091 s，输出逐位一致。下文 10.504 s 的阶段结果是此前
> `-Ofast -march=native` 配置的历史数据。

## 结论

本阶段把 TwoPuncture 的谱导数、逐点方程和预条件器线松弛改为 OpenMP，并把线程绑定到
core place。最终在课程 HPC 节点的 30 个物理核上用时 10.504 s，相对第三阶段最佳串行
结果 187.091 s 加速 17.81 倍，用时下降 94.39%。相对最初 baseline 的 286.505 s，四个
阶段累计加速 27.28 倍。

当前节点给作业 60 个硬件线程，但它们实际是 30 个物理核、每核 2 个线程。60 线程复测
为 10.334 s，只比 30 线程快 1.62%，task-clock 和周期数却接近翻倍。因此脚本选择“进入
最快值 2% 区间的最少线程数”，本节点推荐 30 线程。若评测 allocation 确实包含 60 个
不同物理核，脚本会从 affinity 和 CPU/core 映射识别到 60，并同时测试半数和全部物理核。

## 并行了什么，为什么可以并行

### 1. 谱导数 `Derivatives_AB3`

程序依次沿 A、B、phi 三个方向做一维谱变换。固定另外两个坐标后，每条一维线只读取
自己的输入并写自己的导数位置，所以同一方向的线之间没有写冲突。每个 OpenMP 线程使用
自己的 `thread_local TransformWorkspace`，临时数组和系数缓存也不会互相覆盖。

三个方向不能全部混在一起：B 方向需要读取已完成的 A 导数，phi 方向又需要读取 A/B
导数。因此实现只在每个方向内部用 `omp for collapse(3) schedule(static)` 分线，三个
work-sharing loop 末尾保留隐式 barrier，维持原来的 A -> B -> phi 数据依赖。

### 2. `F_of_v` 和 `J_times_dv`

谱导数准备好以后，一个 `(i,j,k)` 网格点的坐标变换和方程值只读取该点数据，结果也只写
回该点。因此两个函数都按三维网格点并行。每个线程使用独立的 `PointWorkspace`，避免
`values`、`U`、`dU` 这些小型临时向量发生数据竞争。调试文件路径虽然默认关闭，仍用
OpenMP critical 保护，避免以后打开调试输出时并发写文件。

### 3. `relax`、`LineRelax_be/al` 和 Thomas 求解

这是最重要的部分。第三阶段约 93.4% 的串行样本位于这条路径。稀疏 JFD 模板只连接
`i/j/k` 各方向的当前点和相邻点。原代码已经按偶/奇 `k`、偶/奇 `i` 或 `j` 分成红黑式
phase；同一 phase 的线不直接相邻，因此可以同时求解，不同 phase 之间则必须同步。

实现对每个 phase 使用 `omp for collapse(2) schedule(static)`，并保留 loop 末尾的隐式
barrier。`LineRelax_be/al` 内部的 Thomas 求解仍是单条线上的串行递推，因为长度只有 50，
跨线并行比拆开这条短递推更合适。每个线程有独立的 `LineWorkspace`。

原来 BiCGStab 对同一个预条件器连续调用 `relax()` 200 次。如果简单在 `relax()` 内加
parallel for，就会进入 200 次并行区。现在接口一次接收迭代次数，在一个 parallel region
里完成全部 200 轮；OpenMP runtime 负责复用 worker，不需要维护手写线程池。每轮的 8 个
phase barrier 仍然是算法正确性所需。

### 4. 暂时没有并行的部分

`SetMatrix_JFD` 通过逐列扰动并更新多个稀疏行的 `ncols/cols/Matrix`，直接并行外层循环会
竞争同一行，不能只加一个 pragma。BiCGStab 的点积和范数也暂留串行，以保持浮点归约顺序。
最终 profile 中这些部分都低于 0.5% 的单项显示阈值，当前没有必要为它们引入复杂同步。

## 性能扩展与正确性

完整线程扫描（作业 129362）如下：

| OpenMP 线程 | 用时 / s | 相对 OpenMP 1 线程加速 | task-clock / s | 输出 |
|---:|---:|---:|---:|---|
| 1 | 215.162 | 1.00x | 214.919 | 参考 |
| 4 | 63.013 | 3.42x | 251.755 | 逐位一致 |
| 8 | 31.114 | 6.92x | 248.425 | 逐位一致 |
| 16 | 16.499 | 13.04x | 263.052 | 逐位一致 |
| 30 | 10.375 | 20.74x | 309.460 | 逐位一致 |
| 60 | 10.363 | 20.76x | 617.275 | 逐位一致 |

最终 30/60 复测（作业 129425）分别为 10.504 s 和 10.334 s，并选择 30 线程进行 profile。
30 线程相对同一实现的 1 线程加速 20.48 倍，并行效率为 68.3%；相对第三阶段更快的纯串行
二进制则是 17.81 倍、效率 59.4%。后一个口径更适合作为最终收益，因为 OpenMP 1 线程有
work-sharing 开销，且循环 phase 的排列与第三阶段不同。

所有线程数的 `puncture_parameters_new.txt` 和去时间戳后的 `Ansorg.psid` 逐位一致。
OpenMP 1 线程与第三阶段最终结果的参数文件一致，场数据最大绝对差 7.51e-16；求解全部
正常收敛。缩小网格的 1/4 线程冒烟测试也逐位一致。

## 最终 profile

30 线程的主要自耗时样本为：

| 位置 | 样本比例 | 含义 |
|---|---:|---|
| `LineRelax_be` | 23.07% | B 方向线松弛计算 |
| `LineRelax_al` | 20.13% | A 方向线松弛计算 |
| `ThomasAlgorithm` | 11.67% | 两种线松弛中的短三对角求解 |
| `libgomp` 内部路径合计 | 约 40.8% | work-sharing barrier、等待与调度 |

串行时约 93% 都是线松弛计算；并行后它的实际计算部分降至约 55%，OpenMP 同步成为新的
主要限制。说明热点确实被并行加速了，也说明继续简单增加线程不会线性提速。

硬件计数为 IPC 3.34、分支失误率 0.28%、L1D miss 1.96%、LLC miss 0.06%、dTLB miss
0.25%，仍没有 cache、TLB 或分支异常。30 个线程使用了 29.85 个 CPU 的 task-clock，运行
中无 CPU migration。各线程总周期样本占比为 3.29%--3.37%；只看线松弛和 Thomas 的计算
样本时为 3.10%--3.56%，负载没有明显拖尾。

affinity 输出确认 30 个线程依次绑定到 `0-1, 2-3, ..., 58-59` 这 30 个 core place；该作业
的 CPU 与内存都限制在 NUMA node 0，不涉及跨 NUMA 访问。这里每个 place 含同一物理核的
两个硬件线程，但 30 线程时每个 place 只放一个 OpenMP 线程。

## 运行方式与下一步

`hpc_twopuncture_openmp.sh` 会自动识别当前 affinity 内的物理核，设置
`OMP_DYNAMIC=FALSE`、`OMP_PLACES=cores`、`OMP_PROC_BIND=close` 和
`OMP_WAIT_POLICY=ACTIVE`，扫描线程数、校验结果并 profile。可用
`AMSS_OMP_CANDIDATES="30 60"` 限定复测集合。

最终数据位于 `profile/twopuncture-openmp-20260821T055638Z-16/`。若继续优化 TwoPuncture，
最高优先级已不是数学库或通信，而是减少 `relax` 的必要 barrier/调度成本，例如研究能否
在不破坏红黑依赖的前提下合并 phase，或调整线/网格数据布局减少同步期间的缓存一致性成本。
