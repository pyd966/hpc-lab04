# Phase 29：`-mtune=native` 作用范围实验

日期：2026-08-25

## 1. 为什么仍然值得测试

此前已经确认：全 ABE 使用 `-mcpu=native` 回退约 9%，显式启用 SVE 回退约
13.5%。这两项不仅改变 CPU 调度模型，也改变可用指令集并使大量循环从 128-bit
Neon 变成可变长 SVE，因此不能回答一个更窄的问题：只根据 TaiShan 核心调整指令
排序、但仍保持当前指令集，是否能改善 RHS。

`-mtune=native` 只影响调度和 cost model，不启用新的 ISA，也不允许浮点重排。本阶段
分别测试：

1. 只给 `src/bssn_rhs.f90` 添加 `-mtune=native`；
2. 给整个 CPU ABE 添加 `-mtune=native`；
3. 不添加调度参数的当前生产配置。

这样可以区分“RHS 局部调度可能有效”和“其它 stencil/控制路径抵消收益”。本阶段
保持严格 `-O3 -g -fno-omit-frame-pointer`，没有使用 `-Ofast`、fast-math、
`-mcpu=native` 或 SVE 参数。

## 2. 实验设置

HPC job `164930`，artifact：
`profile/abe-tune-scope-20260825T164021Z-14`。

节点仍提供 60 个逻辑 CPU，对应 30 个物理核。运行配置为单进程 OpenMP、30 个
绑核 worker、静态层 24、移动层 30、`dynamic,1`。三个二进制按
`OFF -> RHS -> ALL -> RHS -> ALL -> OFF` 交错运行到 `t=4`，每轮采集
`perf stat -d -d`。

CMake 生成的 verbose 编译命令确认，RHS-only 版本的 `-mtune=native` 只出现在
`bssn_rhs.f90`，相邻的 `diff_new.f90` 等文件没有收到该参数。三版
`compute_rhs_bssn` 函数大小都为 `0x199ec`，但反汇编逐字节不同，说明调度参数实际
改变了机器码，并非被编译器忽略。

## 3. 时间结果

| 候选 | Evolve 两次结果（s） | 均值（s） | 相对 baseline | 平均活跃 CPU |
|---|---:|---:|---:|---:|
| baseline | 28.4778, 28.4695 | 28.47365 | - | 22.675 |
| RHS-only tune | 28.4962, 28.4379 | 28.46705 | -0.023% | 22.631 |
| whole-ABE tune | 28.4498, 28.4659 | 28.45785 | -0.055% | 22.770 |

两个候选都远低于 1% 保留门槛。RHS-only 的两次结果分别位于 baseline 两侧，
全 ABE 的绝对优势也只有 0.016 秒；这些差值不能从当前短测噪声中区分出来。

## 4. 硬件计数器与正确性

三组 IPC 都在 1.41--1.42，L1D miss 都在 4.14%--4.16%，LLC load miss 都在
49.23%--49.28%。每轮总周期约 `2.093--2.096e12`，总指令约
`2.962--2.969e12`，没有与候选一致的下降方向。

所有六轮输出在忽略前两行时间戳后逐位一致，课程检查全部 PASS，trajectory RMS
均为 0。因此候选在数值上没有问题，拒绝原因只有性能收益不可重复。

## 5. 决策

临时 RHS-only CMake 参数和 sweep 脚本已撤回，生产 ABE 继续使用空
`AMSS_ARCH_FLAGS`。结合已有实验，现在可以关闭通用编译参数路线：

- `-mcpu=native` 和显式 SVE 有稳定回退；
- LTO 只有约 0.32%，低于门槛；
- `-mtune=native` 无论局部还是全局都与 baseline 相同；
- `-Ofast` 会改变浮点语义，用户已明确要求不使用。

下一步应回到有 16.76% 行级样本上限的 Ricci 大表达式，测试不增加单循环寄存器
活跃集的小 tile。该方案改善的是同一 tile 内多个分量之间的缓存复用，与本阶段只
调整机器码调度属于不同机制。
