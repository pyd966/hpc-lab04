# 阶段一报告：kodis 热点循环 SIMD 向量化

日期：2026-08-23

## 结论

本阶段只优化了 CPU ABE 的 kodis（Kreiss-Oliger 人工耗散）内层循环，没有修改 compute_rhs_bssn 的数学表达式，也没有开始第二阶段的任务重新划分或第三阶段的内存布局优化。

在同一节点、相同输入、相同 24/30 分层 OpenMP 配置下，4 次交错短程 A/B 的结果为：

| 版本 | 次数 | Evolve 平均时间 | 范围 | 平均 CPU 数 | 正确性 |
| --- | ---: | ---: | ---: | ---: | --- |
| scalar（原循环） | 2 | 43.0179 s | 42.9751--43.0607 s | 16.466 | PASS |
| simd（本阶段） | 2 | 39.82365 s | 39.7905--39.8568 s | 16.700 | PASS |

按交错均值计算，Evolve 加速为 7.43%。两组内部波动约为 0.1%，小于收益。SIMD 两次课程检查均 PASS；四个关键输出文件与 scalar 版本逐字节一致。实验完整工件在 profile/abe-kodis-simd-20260823T052738Z-14/。

## 为什么选 kodis

正式基线的 ABE record（24/30 OpenMP、-O3 -g）中，主要 self 热点为：

| 函数 | 基线占比 |
| --- | ---: |
| compute_rhs_bssn_ | 42.38% |
| __memcpy_sve | 8.64% |
| kodis_ | 8.13% |
| fdderivs_ | 7.64% |
| lopsided_ | 6.40% |
| __memset_sve_zva64 | 5.16% |
| prolong3_ | 4.06% |
| fderivs_ | 3.12% |

编译器的 -fopt-info-vec-all 诊断显示：

- 原 kodis 三层循环（src/kodiss.f90:67-69）因循环体内的三维边界 if 被报告为 unsupported control flow；
- 原目标文件中找不到 vN.2d/vN.4s 等 NEON 浮点向量指令，只有标量双精度运算；
- fdderivs 和 lopsided 也有边界/符号分支，但它们的重构需要拆分内部点与边界点，改动面更大；
- 因此先选择可将判断完全移出循环、且不改变公式和数组布局的 kodis，便于做单因素实验。

## 做了什么

### 1. 有效迭代域显式化

原代码遍历所有 i,j,k，每个点都检查 i-3 >= imin 且 i+3 <= imax、j-3 >= jmin 且 j+3 <= jmax、k-3 >= kmin 且 k+3 <= kmax。

新路径先计算：

- ibegin=max(1,imin+3)，iend=min(ex(1),imax-3)
- jbegin=max(1,jmin+3)，jend=min(ex(2),jmax-3)
- kbegin=max(1,kmin+3)，kend=min(ex(3),kmax-3)

然后只遍历这个有效矩形。有效点集合和原 if 完全相同，原有 eps/cof 三个方向的计算顺序保持不变。

### 2. 声明内层循环可 SIMD

在 j、k 固定时，i 循环的每个点只读 fh 的邻域并更新自己的 f_rhs(i,j,k)，不存在跨 i 的写依赖。因此在该循环前加入：

~~~fortran
!$omp simd
do i=ibegin,iend
~~~

编译器报告由：

~~~text
src/kodiss.f90:67-69: missed: unsupported control flow in loop
~~~

变为：

~~~text
src/kodiss.f90:126:58: optimized: loop vectorized using 16 byte vectors
~~~

AArch64 汇编对照中，scalar 目标没有 v*.2d 指令；simd 目标出现大量 v*.2d NEON 双精度向量指令，表示每条指令同时处理两个 double。没有手写 AArch64 intrinsic：当前 Fortran 循环已经能由 GCC 生成 NEON，手写 intrinsic 需要额外的 C/Fortran 接口和数据封装，第一阶段收益与维护成本不匹配。

### 3. 保留可回退开关

CMakeLists.txt 增加 AMSS_ENABLE_KODIS_SIMD，默认 ON，并只对 CPU ABE target 定义 AMSS_KODIS_SIMD。OFF 时编译原循环，便于后续复现实验和快速回退；GPU target 不受该 CPU 开关影响。

## 正式 t=40 验证

保留 SIMD 版本在生产配置下完成正式运行：

- 配置：单进程 OpenMP，OMP_NUM_THREADS=30，OMP_PLACES=cores，OMP_PROC_BIND=close
- 静态层：24 blocks / 24 threads
- 移动层：30 blocks / 30 threads
- 编译：ABE=-O3（SIMD 开关 ON），TwoPuncture -O3 -march=native
- Evolve=395.318 s，Total Running=401.792 s
- perf stat wall：422.462736980 s
- 课程检查：FINAL: PASS，40/100 golden 时间点，轨迹 RMS=0

正式 stat 计数器：

| 指标 | SIMD 正式值 |
| --- | ---: |
| 平均 CPU | 19.657 / 30 |
| IPC | 1.81 |
| branch miss | 0.46% |
| L1D load miss | 2.98% |
| LLC load miss | 48.75% |
| dTLB load miss | 1.48% |

与旧基线不同节点的绝对时间不能直接归因于本改动：旧基线频率约 2.454 GHz，本次约 2.868 GHz。因此正式 stat 只用于完整运行和计数器确认，7.43% 的因果收益采用同节点交错 A/B。

正式 perf record 结果位于 profile/record-20260823T055929Z-$/：

- 446K cycles samples，Total Lost Samples: 0
- DSO：ABE 79.48%，libc 14.92%，libgomp 2.51%，TwoPuncture 2.40%
- 主要 self 热点：compute_rhs_bssn_ 45.63%，__memcpy_sve 9.11%，fdderivs_ 8.09%，lopsided_ 6.81%，__memset_sve_zva64 5.51%，prolong3_ 4.23%，fderivs_ 3.33%，kodis_ 2.29%
- kodis_ 从 8.13% 降到 2.29%，下降约 71.8%；剩余占比主要是 symmetry_bd 和非向量化边界/调用开销，说明热点已成功向其他 RHS/访存部分迁移
- 新源码行热点：bssn_rhs.f90:550 3.99%、:589 3.38%、:511 3.31%；lopsidediff.f90:280 2.96%；kodiss.f90:126 2.15%；diff_new.f90:574 1.92%、:582 1.89%；prolongrestrict_cell.f90:2136 1.46%、:2143 1.43%

## 对当前瓶颈的解释

这不是“所有计算都已经 SIMD 化”。本阶段只消除了 kodis 的控制流阻碍：

- kodis 的算术热点显著下降；
- compute_rhs_bssn 的总占比上升，是因为它成为剩余工作的相对大头，不表示它变慢；
- __memcpy_sve、__memset_sve_zva64 仍合计约 14.6%，表明通信/同步缓冲和临时数组初始化仍是重要的访存成本；
- IPC 从旧 record 的 1.94 降到本次正式 stat 的 1.81，不能单独解释为退化：两次运行节点频率、采样方式不同，且 SIMD 改变了指令混合。更可信的判断是总时间和同节点 A/B；
- branch miss 仍约 0.5%，没有异常；LLC miss 约 49%，继续显示工作集受内存层级限制，后续内存访问优化仍有必要；
- 平均 CPU 约 16--20 / 30，说明下一阶段的任务划分/同步等待仍可能比单纯 SIMD 更大的收益来源。

## 下一步建议（暂不执行）

第二阶段应只处理任务划分和已有 OpenMP 区域：

1. 以 perf-report-children.txt 的 Step(...)._omp_fn.0/.1、omp_local_transfer、omp_execute_cached_sync 为入口，统计各 level/block 的线程工作量和等待时间；
2. 先实验 static/moving level 的 block-to-thread 映射、OMP_SCHEDULE 或现有分块目标，避免把静态层的少量 block 分给过多线程；
3. 检查天然独立的 patch/grid level/analysis task 是否被串行调用，优先扩大已有并行区而不是再创建大量短生命周期 parallel region；
4. 每次只改一种划分策略，保持 24/30、绑核和输入不变，做短程交错 A/B，随后再做完整 t=40 profile。

第三阶段再处理内存访问：

- 减少 fdderivs/fderivs 的整数组初始化和临时数组；
- 检查 copy_、prolong3_、restrict3_ 的缓冲区布局与 NUMA 首触；
- 用线程私有、cache-line 对齐的计数器/临时缓冲避免 false sharing；
- 只有在 profile 证明收益后再调整 Fortran 数组布局或生命周期。

本阶段没有更换数学库、没有使用 -Ofast、没有改变 MPI/OpenMP 架构，也没有修改第二、三阶段代码。
