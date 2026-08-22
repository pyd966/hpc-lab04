# ABE 阶段三报告：编译参数与向量化实验

记录日期：2026-08-22。本阶段在不使用 -Ofast、不启用 fast-math 的前提
下，比较通用 O3、TaiShan 本机指令集和循环展开。结论是继续保留严格 O3：
-mcpu=native 确实把热点编译成 SVE，但端到端 Evolve 稳定回归约 9%，
-funroll-loops 没有补救。生产脚本现已显式传入空 AMSS_ARCH_FLAGS，
避免环境变量或旧 CMake cache 意外启用失败参数。

## 1. 为什么测试这些参数

正式节点是 AArch64 TaiShan-v120，支持 128-bit Neon 和 SVE，当前进程的
SVE 向量长度为 256 bit。此前 ABE 只使用通用 -O3，理论上
-mcpu=native 可以让 GCC 14.2 为 256-bit SVE 选指令并按本机调度。
热点 compute_rhs_bssn、fdderivs、kodis 和 lopsided 又包含大量连续的
Fortran i 内层循环，所以 native 值得实测。

本阶段没有测试 -Ofast 或 -ffast-math。这些参数允许重排浮点表达式、改变
NaN/Inf 和有符号零语义，当前目标是先分离“指令集选择”本身的效果。
数学库也没有更换：正式 profile 中 libm 只有约 0.63%，程序主要时间在
自定义 stencil 和数据移动，不是 BLAS、FFT 或超越函数库。

## 2. 实验设计

脚本 hpc_abe_flag_sweep.sh 在同一个作业内顺序构建并运行三组：

| 名称 | 优化参数 | 架构参数 |
|---|---|---|
| o3 | -O3 -g -fno-omit-frame-pointer | 无 |
| native | 同上 | -mcpu=native |
| native-unroll | 同上 | -mcpu=native -funroll-loops |

所有候选固定为单进程 OpenMP、30 个物理核心、静态层 24 Block/24 线程、
移动层 30/30，并演化 t=0..4。每一组都独立构建 ABE，运行 perf stat -d -d，
执行课程 check.sh，并将 ADMQs、BH、constraint 和 psi4 与 O3 逐字节比较。

第一次 sweep 的前两组完成后，home 配额耗尽，第三组被 SIGPIPE 中断。
这不是候选自身失败。释放空间后重新完整运行三组，正式比较只采用重试作业：

    profile/abe-flag-sweep-20260822T152620Z-14

两个未被报告引用的早期失败 block sweep 已从 home 移出，其可恢复副本位于
/tmp/abe-archived-profiles-20260822/。

## 3. 时间与正确性

| 候选 | Evolve / s | Total / s | 相对 O3 | 平均 CPU | 正确性 |
|---|---:|---:|---:|---:|---|
| o3 | 42.9353 | 50.1099 | 基准 | 16.317 | bitwise，PASS |
| native | 46.8343 | 54.1446 | +9.08% | 16.239 | bitwise，PASS |
| native-unroll | 46.8075 | 54.2700 | +9.02% | 16.189 | bitwise，PASS |

两种 native 输出与 O3 的四类数值行逐字节一致，轨迹 RMS 为 0，
constraint 全部通过。因此它们是数值正确但性能更差，不存在“因正确性
门槛太严而放弃更快版本”的问题。第一次未完成 sweep 中，o3 为 43.3604s、
native 为 47.4901s，也得到相同方向的 9.5% 回归，说明结论可重复。

## 4. 硬件计数器说明了什么

| 指标 | o3 | native | native-unroll |
|---|---:|---:|---:|
| task-clock / s | 821.81 | 883.30 | 882.46 |
| cycles / 万亿 | 2.354 | 2.529 | 2.525 |
| instructions / 万亿 | 4.390 | 4.007 | 3.988 |
| IPC | 1.86 | 1.58 | 1.58 |
| branch miss | 0.42% | 0.54% | 0.46% |
| L1D miss | 2.65% | 2.93% | 2.92% |
| LLC load miss | 48.02% | 48.14% | 47.90% |
| dTLB miss | 2.23% | 1.60% | 1.56% |

native 的动态指令数减少约 8.7%，说明更宽向量并非完全无效；但 IPC 从
1.86 降到 1.58，cycles 增加约 7.4%，最终 wall time 变慢。L1D miss rate
也从 2.65% 增至约 2.92%。dTLB 指标改善并不足以抵消执行吞吐和一级缓存
访问的退化。

反汇编对整个 ABE 二进制作静态计数：

| 二进制 | SVE z-register 行 | Neon v-register 行 | text 大小 |
|---|---:|---:|---:|
| o3 | 0 | 4232 | 790736 B |
| native | 4418 | 319 | 710166 B |
| native-unroll | 4417 | 359 | 873314 B |

这证明 native 确实触发了 SVE，而不是参数没有生效。它也说明“更宽的
向量化”不能单独作为优化成功的判断：这些 stencil 会同时受到加载、
跨方向访问、边界分支、缓存带宽和核心执行资源约束。循环展开将 text
增大约 23%，但几乎没有改变时间或 IPC，所以不保留。

## 5. 胜出配置的正式 Profile

胜出的 O3 配置又进行了独立的 perf stat 和 perf record：

    profile/abe-20260822T153432Z-14

perf stat 的 Evolve/Total 为 43.7150s/51.0260s，perf record 为
42.9676s/50.1917s。record 时间与 sweep 的 42.9353s 基本一致；stat
这一遍略慢，属于本节点短测试的运行波动。两遍输出逐字节一致，且与阶段一
最终 O3 输出逐字节一致，课程检查 PASS。

正式 flat profile 仍为：

| 热点 | 周期占比 |
|---|---:|
| compute_rhs_bssn | 44.22% |
| memcpy | 8.88% |
| kodis | 8.31% |
| fdderivs | 7.79% |
| lopsided | 6.69% |
| memset | 5.44% |
| prolong3 | 3.98% |
| fderivs | 3.20% |

IPC 1.88、branch miss 0.41%、L1D miss 2.65%、LLC load miss 47.80%、
dTLB miss 2.22%。这些结果与阶段一 O3 profile 一致，说明第三阶段没有
隐藏的性能退化，也没有改变剩余热点。

## 6. 最终决定

生产 ABE 保持严格 -O3，不添加 -mcpu=native、-funroll-loops 或 -Ofast。
hpc_cpu.sh 显式设置空 AMSS_ARCH_FLAGS，以确保这个选择可复现。
hpc_abe_flag_sweep.sh 保留为后续更换编译器或 CPU 时的回归工具。

第三阶段也排除了“只换全局编译参数就能解决热点”的路线。下一步若继续
优化，应该针对热点循环做局部改造和局部 vectorization report，而不是
对整个程序强制 SVE：先分离无分支 interior 与 boundary，检查 contiguous
i 循环的别名和临时数组，再逐 kernel 做严格 A/B。这样可以保留 O3 在
控制流和 AMR 路径上的优势，同时只对有证据受益的内层循环启用 SIMD。
