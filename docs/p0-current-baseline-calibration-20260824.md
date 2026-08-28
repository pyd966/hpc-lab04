# P0 报告：当前端到端基线与 CPU 拓扑校准

日期：2026-08-24

代码基线：`ffc6ebd`，报告提交前 HEAD 为 `fc5a125`。两者之间只有文档提交，
production 二进制代码相同。

## 1. 结论

P0 没有修改计算代码。它完成了两件事：

1. 在相同节点、相同 cpuset 上重复两次正式 `TwoPuncture + Evolve(40)`；
2. 核对公开 `lab4` 队列能够提供的真实物理核数，以及现有短程 profile 是否已经
   有足够重复证据。

两次正式长测都通过课程检查，关键数值行逐字节一致：

| 指标 | 作业 155072 | 作业 155131 | 均值 |
|---|---:|---:|---:|
| ABE Before Evolve | 2.699 s | 2.668 s | 2.684 s |
| ABE Evolve | 298.972 s | 298.338 s | **298.655 s** |
| ABE Total Running | 301.671 s | 301.006 s | **301.339 s** |
| `This Program Cost` | 313.983 s | 313.543 s | **313.763 s** |
| `run.sh` wall | 323 s | 322 s | **322.5 s** |
| 峰值内存 | 约 3.999 GB | 约 3.999 GB | 约 4.0 GB |

Evolve 两次相差 0.634 秒，即均值的 0.212%；`This Program Cost` 相差
0.440 秒，即 0.140%。当前时间具有良好的同节点可重复性，后续优化不应使用单次
最快值，而应以 `298.655 s / 313.763 s` 为 P0 长程参考。

按课程核心端到端口径，最慢一遍 `This Program Cost=313.983 s`，低于 330 秒
约 16.0 秒。即使把绘图和 shell 收尾也包括在外层 wall 中，最慢一遍为 323 秒，
仍低于 330 秒 7 秒。

## 2. 测试配置

两次作业都使用：

- partition：`lab4`；
- 调度资源：60 CPU、100 GiB、30 分钟；
- 节点：`zjusct-920b-1`；
- cpuset：`64-123`；
- OpenMP：30 threads，`OMP_PLACES=cores`、`OMP_PROC_BIND=close`；
- 静态层：60 block target / 24 threads；
- 移动层：60 block target / 30 threads；
- block schedule：`dynamic,1`；
- ABE：严格 `-O3`，没有 `-Ofast`、fast-math 或全局 `-mcpu=native`；
- 单进程 OpenMP-only ABE，重新运行 TwoPuncture，没有复用初值缓存；
- 演化区间：`t=0..40`。

工件：

- 作业 155072：`profile/baseline-20260824T093511Z-$/`；
- 作业 155131：`profile/baseline-20260824T094218Z-$/`；
- 输出分别在对应的 `profile/runs/baseline-.../` 目录。

## 3. 正确性与可重复性

两遍结果均为：

- trajectory：匹配 40/100 golden 时间点；
- trajectory RMS：0；
- constraints：40 组时间、每组 9 levels，全部低于阈值；
- `FINAL: PASS`。

直接比较完整 `.dat` 文件时，第一、第二行的运行时间戳不同。跳过两行文件头后，
以下四个文件的全部数值字节一致：

- `bssn_ADMQs.dat`；
- `bssn_BH.dat`；
- `bssn_constraint.dat`；
- `bssn_psi4.dat`。

因此当前 OpenMP dynamic 调度没有引入运行间数值不确定性。

## 4. `This Program Cost` 的准确含义

重新阅读 `AMSS_NCKU_Program.py` 后确认：

1. `start_time` 在 TwoPuncture 之前记录；
2. TwoPuncture、输入更新和 ABE 都包含在计时内；
3. ABE 返回后立刻计算 `elapsed_time`；
4. 随后的文件复制和绘图不包含在 `This Program Cost` 中；
5. 因为打印语句位于绘图之后，所以日志顺序容易让人误以为绘图也被计入。

因此 `This Program Cost` 是当前最接近 `TwoPuncture + ABE` 的应用内端到端口径。
两遍中它比 ABE Total 多 12.31 秒和 12.54 秒，主要是 TwoPuncture、输入生成和
进程切换。外层 wall 再多约 8--9 秒，主要是绘图和 shell 收尾。

## 5. 与上一版正式长测比较

历史作业 149666 的结果是：

| 指标 | 历史版本 | 当前 P0 均值 | 改善 |
|---|---:|---:|---:|
| ABE Evolve | 305.589 s | 298.655 s | 2.27% |
| ABE Total | 310.505 s | 301.339 s | 2.95% |
| `This Program Cost` | 321.915 s | 313.763 s | 2.53% |
| 外层 wall | 331 s | 322.5 s | 2.57% |

这段改善包括历史长测之后保留的 SIMD、RHS array-pass 和 block scheduling 等改动，
不能全部归因于最后一项调度修改。由于两个版本不是同一个二进制，表格只用于说明
累计进展，不作为某个单项优化的 A/B 结果。

## 6. 短测外推为什么偏乐观

当前配置已有四个独立 `t=4` 结果：

- 两次完整 schedule sweep：28.6506 s、28.6649 s；
- 最终 strict profile：28.9607 s（stat）、28.7611 s（record）。

四次均值为 28.7593 秒，范围为 28.6506--28.9607 秒。两次 sweep 分别位于
`zjusct-920b-1` 和 `zjusct-920b-2`，当前候选在两台节点上都稳定胜过 24/30 block
和 90 block 候选，关键输出均一致。

长测 Evolve 均值与短测均值的比例是 10.3846，而不是简单的 10。这是因为
`t=4` 不能完整代表后续 regrid、层级变化和分析工作。因此后续可以用短测筛选候选，
但不能再直接把 `t=4` 乘 10 当作最终成绩；保留候选必须用 `t=40` 验证。

## 7. 公开队列无法测试 60 个独立物理核

`hpc limits` 显示 `lab4` 最多只能申请 60 CPU。当前 TaiShan-v120 节点每个物理核
有两个硬件线程，作业得到的 60 个调度 CPU 实际是 30 个物理核的两个 SMT sibling。
脚本通过 CPU topology 去重，正确启动 30 个 OpenMP worker。

`hpc` 没有关闭 SMT 或要求“一线程一物理核”的资源选项；申请 60 CPU 已经达到
个人上限，也不能申请 120 个逻辑 CPU 来覆盖 60 个物理核。因此公开队列中无法
完成真实 60 物理核的 block/thread sweep。

正式评测据课程说明会提供 60 个物理核、无超线程。当前脚本会自动识别为 60 worker，
但仍有一个待验证问题：

- 当前 30 worker / 60 blocks 时，每个 worker 平均有两个 block，`dynamic,1` 可以
  在尾部继续取任务；
- 评测 60 worker / 60 blocks 时，每个 worker 只有一个 block，动态队列没有第二个
  任务可偷取；
- 120 blocks 可能恢复这一优势，但会增加 ghost、Sync 和 AMR transfer，不能在
  30 物理核结果上直接断言更快。

因此 production 默认暂时保持已验证的 60 block，不把未经测试的 120 block 写成
评测默认值。脚本保留环境变量覆盖；若获得真实 60 物理核，应优先测试 60、90、120、
150 blocks，再决定评测参数。

## 8. 当前 profile 基线

最新 strict profile 为 `profile/abe-20260824T081218Z-14/`：

| 指标 | 结果 |
|---|---:|
| `t=4` Evolve | 28.9607 s |
| 平均 CPU | 22.65 / 30 |
| IPC | 1.80 |
| branch miss | 0.43% |
| L1D miss | 3.86% |
| LLC load miss | 37.35% |
| dTLB miss | 3.41% |

主要 flat 热点：

| 热点 | cycles self |
|---|---:|
| `compute_rhs_bssn_` | 41.47% |
| `__memcpy_sve` | 12.57% |
| `lopsided_` | 8.54% |
| `__memset_sve_zva64` | 5.82% |
| `prolong3_` | 4.90% |
| `fdderivs_` | 4.64% |
| `rungekutta4_rout_` | 2.45% |
| `fderivs_` | 2.31% |

这仍支持总计划的下一步：先减少 `fderivs/fdderivs` 整数组清零，再处理 Sync 的
中间 pack/unpack。P0 没有发现需要回滚当前 block 调度或重新引入 MPI 的证据。

## 9. P0 结论与下一阶段入口

P0 到此完成：

- 正式端到端成绩不再是外推，而是两遍同节点实测；
- 核心口径稳定在约 313.8 秒，当前已低于 330 秒；
- 长程自然波动远小于 1%，后续 1% 以上的同节点改善可以被分辨；
- 当前内存仅约 4 GB，容量充足，但 profile 证明瓶颈是数据流量而不是容量；
- 真实 60 物理核实验受公开队列限制，必须作为评测前单独校准项保留。

下一阶段按总计划进入 P1：分别实验 `fderivs` 和 `fdderivs` 的整数组清零消除。
两个例程必须分开提交、交错 A/B、重新 profile；如果第一种实现没有收益，应检查
边界 shell 写入和缓存流量，而不是只试一个循环版本就结束阶段。
