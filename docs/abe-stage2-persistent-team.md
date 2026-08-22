# ABE 阶段二报告：持久 OpenMP 团队实验

记录日期：2026-08-22。本阶段验证的假设是：把 Step 内多个短 OpenMP
区域合成一个覆盖完整 RK4 的并行区，能否减少 fork/join 和 barrier
管理开销。结果是否定的。实现保持数值完全一致，但 Evolve 反而变慢
10%--12%，因此实验代码已撤回，当前生产源码仍采用阶段一的独立
parallel for 结构。

## 1. 为什么做这个实验

阶段一 profile 显示平均使用 16.427/30 个 CPU。Step 的每个 RK4 子步
依次执行 Block RHS/RK 更新、ghost/buffer 同步和时间层交换。旧实现为
RHS、Sync 的 pack/unpack 和 swap 分别进入 OpenMP 区域。从源码表面看，
反复进入并行区可能有额外开销，所以本阶段直接验证“让一支团队跨越整个
Step”这个建议，而不凭经验假定它一定有效。

## 2. 实验实现

实验版只改了 OpenMP-only 路径，传统 MPI 路径保持原样：

- Step 建立一次 omp parallel，四次 RHS/RK 计算改用团队内的 omp for；
- 三次中间 swap 和一次最终 swap 也改用同一团队的 omp for；
- 新增集体调用的 SyncOmpTeam。缓存查询和 workspace 扩容由
  omp single 完成，pack 与 unpack 仍分别用 omp for；
- pack 到 unpack、Sync 到下一 RK 子步之间的 barrier 全部保留，
  因为这些是数值依赖，不能删除；
- 错误标志改成原子写，错误检查由 omp single 执行。

因此实验只改变团队生命周期，没有改变 RK4 顺序、Block 划分、线程数、
变量次序或浮点 kernel。本地 OpenMP-only 和传统 MPI 两种构建均以
-O3 -g -fno-omit-frame-pointer 通过。

## 3. 测试配置和正确性

正式产物位于 profile/abe-20260822T144812Z-14。配置与阶段一最终
profile 相同：单进程 OpenMP、30 个物理核心；静态层 24 Block/24
线程，移动层 30/30；演化 t=0..4；严格 -O3 -g，没有 -Ofast。

perf stat 与 perf record 两遍运行的 ADMQs、BH、constraint、psi4
数值行逐字节一致。实验版与阶段一最终版的这四类输出也逐字节一致。
课程检查为 PASS，轨迹 RMS 为 0，constraint 全部满足阈值。

## 4. 时间结果

| 版本 | Evolve / s | Total / s | 平均 CPU |
|---|---:|---:|---:|
| 阶段一 perf stat | 42.9766 | 49.8645 | 16.427 |
| 持久团队 perf stat | 47.4987 | 55.2696 | 16.428 |
| 阶段一 perf record | 43.1286 | 49.9341 | - |
| 持久团队 perf record | 48.2458 | 56.1387 | - |

按 perf stat 比较，Evolve 回归 10.52%，Total 回归 10.84%；独立的
perf record 比较分别回归 11.86% 和 12.43%。两遍方向一致，足以排除
一次运行抖动。

最重要的是平均 CPU 数从 16.427 变为 16.428，基本完全不变。持久团队
没有填上 AMR 层级依赖、level 0 只有 9 个 Block 和各阶段尾部不均衡
形成的空洞；它只改变了线程在这些空洞里的等待方式。

## 5. Profile 结果

| 指标 | 阶段一 | 持久团队 |
|---|---:|---:|
| task-clock / s | 823.38 | 915.46 |
| 平均 CPU | 16.427 | 16.428 |
| IPC | 1.86 | 1.95 |
| 估算平均频率 | 2.865 GHz | 2.465 GHz |
| branch miss | 0.48% | 0.44% |
| L1D miss | 2.65% | 2.61% |
| LLC load miss | 47.99% | 47.74% |
| dTLB miss | 2.22% | 2.14% |

cache、TLB 和 branch 指标都没有恶化，指令数也只从约 4.389 万亿变为
4.407 万亿，增加约 0.4%。显著变化是 task-clock 增加 11.2%，同时
cycles / task-clock 对应的平均频率下降约 14%。这与更长时间维持整支
活跃团队、让空闲 worker 在必要同步点等待的行为一致。这里的降频原因是
根据计数器作出的推断；“平均并行度未提高且 wall time 增加”则是直接测量。

实验版 flat profile 的热点仍为：

| 热点 | 周期占比 |
|---|---:|
| compute_rhs_bssn | 43.09% |
| kodis | 8.63% |
| memcpy | 8.53% |
| fdderivs | 8.10% |
| lopsided | 6.86% |
| memset | 5.31% |
| prolong3 | 4.18% |
| fderivs | 3.40% |

阶段一中几个 libgomp self 项合计约 1.73%，实验版约 1.86%，并未下降。
原来的 parallel for 虽然语义上有 fork/join，libgomp 已经复用底层
worker，并不是每次重新创建操作系统线程，所以可消除的开销本来就很小。

## 6. 结论和后续方向

本阶段否决“用一个并行区包住整个 Step”：RK4/Sync 的 barrier 是算法
需要；合并后平均并行度仍是 16.4/30；runtime 自身仅占约 2%；实测出现
稳定的 10%--12% 回归。因此实验实现已撤回，不进入生产代码。

下一阶段不再优化 OpenMP 外壳，而进入真正占周期的 compute_rhs_bssn、
fdderivs、kodis 和 lopsided。先用编译器向量化诊断和严格 A/B 编译
参数测试确认哪些内层循环能利用本机 SVE，再决定是否需要改循环或布局。
这直接覆盖约 60%--70% 的周期，收益上限明显高于继续压缩 fork/join。
