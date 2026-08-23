# ABE 阶段 5：OpenMP 调度与空转分析

日期：2026-08-23

## 1. 结论

本阶段没有简单地把现有 `schedule(static)` 改为 `dynamic`，而是先用 wall-time
诊断找到了更重要的问题：`Step` 中的 RK4 block 计算已经并行，但三条 constraint
路径仍在逐 block 串行调用 `compute_rhs_bssn`。其中 `Constraint_Out` 每个物理时间步
都会执行，因而让其他 29 个线程持续空转约 7 秒（短程 `t=4`）。

保留的改动如下：

1. `Constraint_Out` 的高频 RHS 重算按 block 并行；
2. 初始数据之后的 `Compute_Constraint` 按 block 并行；
3. Evolve 开始前 `Interp_Constraint(true)` 中的 RHS 重算按 block 并行；
4. 各层仍按原顺序执行，层间 `Sync`、restriction/prolongation、输出顺序均未改变；
5. 静态层继续使用 24 blocks/24 threads，移动层继续使用 30 blocks/30 threads，
   并使用 `OMP_PLACES=cores`、`OMP_PROC_BIND=close` 绑核。

两轮交叉 A/B 的综合结果是：

| 指标（`t=4`） | 本阶段开始 | 最终候选 | 变化 |
| --- | ---: | ---: | ---: |
| Evolve | 37.682 s | 30.171 s | -19.9% |
| ABE total | 44.405 s | 35.579 s | -19.9% |
| 全程平均使用 CPU | 17.07 | 21.45 | +25.7% |

正式 `t=40` 无采样运行中，ABE Evolve 为 **305.589 s**，ABE total 为
**310.505 s**。驱动报告的 `This Program Cost` 为 **321.915 s**；它还包含
TwoPuncture 和结果绘图，因此是 `TwoPuncture + Evolve` 的保守上界，仍低于 330 秒。

## 2. 为什么普通 profile 漏掉了这个热点

`perf record` 默认按 CPU cycles 采样。一个函数如果只占用 1 个核运行 7 秒，在同一段
时间内 30 核并行区可以产生约 30 倍的 cycles，因此这个串行函数在调用图中的比例会
显得很小。旧 profile 中 `Constraint_Out -> compute_rhs_bssn` 只有约 1% cycles，
但专门加入的 wall-time 计时显示它实际占了约 7 秒，即短程 Evolve 的 18.6%。

因此本阶段同时使用了两类证据：

- `perf stat/record`：识别真正消耗 CPU cycles 的函数和硬件行为；
- OpenMP phase wall-time：记录每层、每个 RK phase、Sync、transfer、regrid 和
  constraint 的等待时间及各 block 的工作时间。

这也是找到新增并行区域的方法：先从“总 wall time 减去已解释的 Step/Sync/transfer”
得到约 7--9 秒未解释时间，再沿 `Evolve -> Constraint_Out` 阅读调用路径，最终发现
每个时间步都存在一个串行的全层、全 block RHS 重算。

## 3. 原有并行区的实际利用率

修改前的诊断运行（`t=4`）得到：

- 所有 RK block phase 共 23.669 s；
- 这些 phase 的总有效工作为 603.378 CPU-s；
- 等价于 phase 运行期间平均有 25.49 个 CPU 在做 block 工作；
- 相对 24/30 的层级线程配置，综合利用率约 85.5%；
- moving level 5--8 的利用率约 82.0%--87.4%；
- level 0 只有 9 个 block，利用率约 30%，但其全部 Step 只有约 0.53 s，
  不是值得牺牲其他层性能的主矛盾；
- RHS 占 block 有效工作约 94.5%，`enforce_ga` 和 RK update 只占很小一部分。

这说明 `Step` 已经不是“只有 16 个线程工作”。旧的全程平均值约 17 个 CPU，把初始
串行工作、高频串行 constraint、递归控制、同步和 block phase 混在了一起。只观察
全程平均值会错误地判断 RK 主并行区本身也只有 16 个线程在运行。

### 为什么没有改成 dynamic/guided

移动层每个 RK phase 正好有 30 个 block 和 30 个线程，每个线程只领取一个大任务。
无论 `static`、`dynamic` 还是 `guided`，都没有第二个任务可供先完成的线程领取，
因此调度策略本身无法消除最后 12%--18% 的 block 长尾。静态层则有 24 个 block 和
24 个线程，同理如此。

真正消除这部分长尾需要增加任务数量，例如拆分 block 内的 RHS。可是
`compute_rhs_bssn` 内的导数、几何量、演化方程和耗散之间存在阶段依赖，不能简单把
Fortran 循环任意切成 OpenMP tasks。此前增加物理 block 数的实验还会增加 ghost-zone
和 AMR 边界工作，并改变浮点轨迹，收益也为负，因此本阶段没有重复该方案。

## 4. 实现方式

### 4.1 高频 `Constraint_Out`

原实现按 linked list 逐个访问每层的 block，并在主线程上调用
`f_compute_rhs_bssn`。新实现先按原遍历顺序建立本进程的 `vector<Block *>`，然后：

```cpp
#pragma omp parallel for schedule(static)
for (int block_index = 0; block_index < constraint_blocks.size(); ++block_index)
  f_compute_rhs_bssn(... constraint_blocks[block_index] ...);
```

不同 block 拥有独立的 `fgfs` 数组，RHS 只读写当前 block，因此 block 之间没有数据
竞争。parallel-for 的隐式 barrier 保证所有 block 完成后才进入本层原有的
`Parallel::Sync`。每层使用 `OmpThreadScope`，从而沿用静态层 24、移动层 30 的策略。

该路径由 `AMSS_ENABLE_OMP_CONSTRAINT_PARALLEL` 控制，默认开启；关闭时保留相同的
vector 遍历但不创建 OpenMP team，便于精确 A/B 和回退。

### 4.2 初始 constraint 两条路径

`Compute_Constraint` 在读取 TwoPuncture 初值后执行一次；`Interp_Constraint(true)`
在进入递归演化之前执行一次。它们原来复制了与 `Constraint_Out` 相同的串行 block
循环。本阶段使用同样的 block 级并行方法，但用独立开关
`AMSS_ENABLE_OMP_INITIAL_CONSTRAINT_PARALLEL` 控制，因此第二轮实验可以只测这两处。

没有并行化 `Interp_Constraint` 后面的 1000 点插值，也没有并行化 L2Norm。新 profile
显示它们不是当前主要 cycles 来源；插值内部还会访问共享层级搜索结构，在没有专门的
线程安全验证前不应仅为了增加 pragma 而并行。

### 4.3 诊断设施

`AMSS_ENABLE_OMP_DIAGNOSTICS` 默认为 OFF。开启时记录：

- 每层 Step/RK phase/Sync/transfer/regrid wall time；
- 每个 block 的 enforce/RHS/update 时间以及最长 block；
- 三条 constraint 路径的调用次数和 wall time。

正式构建不会包含这些 per-block 计时和临时 timing vector，因此最终性能数据不受该
诊断开销影响。

## 5. 分阶段实验

每组 A/B 都在同一个 60 logical-CPU allocation（30 个可用物理核）内，以单进程
30 OpenMP threads 运行。编译参数为 `-O3 -g -fno-omit-frame-pointer`，没有使用
`-Ofast`。每轮顺序为 A/B/B/A，用于抵消节点温度和运行顺序影响。

### 5.1 只并行高频 `Constraint_Out`

| 候选 | Evolve 均值 | Constraint wall | 平均 CPU | 关键输出逐位一致 | check |
| --- | ---: | ---: | ---: | --- | --- |
| serial | 37.682 s | 6.999 s | 17.07 | yes | PASS |
| parallel | 31.438 s | 0.793 s | 19.92 | yes | PASS |

结果：constraint 重算加速 **8.82x**，Evolve 减少 **16.6%**。IPC、cache、TLB
指标基本不变，说明收益来自消除串行 wall-time，而不是改变单核指令效率。

### 5.2 再并行初始两条路径

两边都已开启高频 `Constraint_Out` 并行，本轮只切换初始 constraint 开关。

| 候选 | Evolve | ABE total | `Compute_Constraint` | 初始 `Interp_Constraint` | 平均 CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| initial-off | 31.452 s | 38.193 s | 1.763 s | 1.594 s | 20.04 |
| initial-on | 30.171 s | 35.579 s | 0.190 s | 0.176 s | 21.45 |

结果：两条一次性路径分别加速约 **9.30x** 和 **9.04x**。Evolve 再减少 4.1%，
ABE total 再减少 6.8%。四次运行的 trajectory RMS 都为 0，约束检查均 PASS，四个
关键输出文件去掉时间戳头之后逐位一致。

原始结果位于：

- `profile/abe-omp-constraint-20260823T183314Z-15/`
- `profile/abe-omp-initial-constraint-20260823T184459Z-14/`

## 6. 修改后的完整 profile

完整 `perf stat + perf record` 目录：
`profile/abe-20260823T185102Z-14/`，共 83,649 个 cycles 样本，无丢样。

### 6.1 CPU cycles 热点

| 路径/函数 | 占比 |
| --- | ---: |
| RK Step 两个 OpenMP worker 路径（inclusive） | 约 79.0% |
| AMR local transfer worker（inclusive） | 8.5% |
| cached Sync worker（inclusive） | 7.9% |
| `Constraint_Out` worker（inclusive） | 1.28% |
| `compute_rhs_bssn` self | 49.47% |
| `memcpy` / `memset` self 合计 | 16.17% |
| `lopsided` self | 8.08% |
| `prolong3` self | 4.57% |
| `fdderivs` self | 4.35% |
| `fderivs` self | 2.36% |
| `kodis` self | 2.26% |

约束路径不再是明显的串行 wall-time 缺口。热点重新集中到预期的 RHS 数值计算和 AMR
数据移动。

### 6.2 硬件计数器

| 指标 | 当前值 | 判断 |
| --- | ---: | --- |
| 平均使用 CPU | 22.98 | 比本阶段前明显提高，但仍受 block 长尾和 Sync/transfer 限制 |
| IPC | 1.59 | 正常，没有异常低 IPC |
| branch miss | 0.43% | 很低，不是分支瓶颈 |
| L1D miss | 4.09% | 可接受 |
| LLC load miss | 49.46% | 偏高，与大数组流式访问及复制相符 |
| dTLB miss | 3.06% | 有成本，但不是本轮调度的首要问题 |

`perf record` 会带来采样开销，所以其中 33--35 秒的短程绝对时间不用于比较性能；
性能结论来自无 record 的交叉 A/B。两个 profile pass 的四个关键数值文件完全一致。

## 7. `t=40` 正式结果

作业：`149666`，使用 `hpc_cpu.sh`，实际重新运行 TwoPuncture，未复用初值缓存。

| 指标 | 结果 |
| --- | ---: |
| ABE Before Evolve | 4.916 s |
| ABE Evolve | 305.589 s |
| ABE Total Running | 310.505 s |
| Python driver `This Program Cost` | 321.915 s |
| 峰值内存 | 约 3.50 GiB |
| trajectory RMS | 0 |
| constraint check | PASS |
| 最终课程检查 | PASS |

`This Program Cost` 在 TwoPuncture、ABE 和绘图全部结束之后输出，因此它是要求中的
`TwoPuncture + Evolve` 的保守上界。外层脚本的 331 秒还包括绘图、Python/进程启动
和退出等非计算流程，不应用作题目要求的核心计算时间。

## 8. 剩余空间与建议

本阶段已经完成可低风险实施的 OpenMP 调度修正。继续提高 22.98 的全程平均 CPU，
不应再盲目切换 schedule，而应按下面顺序推进：

1. **RHS 内部任务分解**：把一个 block 的 `compute_rhs_bssn` 拆成有明确依赖的若干
   phase，在 block 长尾出现时让空闲线程协助最后几个 block。这是最直接的调度方向，
   但需要先画出 Fortran 数组的读写依赖并做单独 A/B；错误拆分会产生数据竞争。
2. **Sync/transfer 的 unpack 粒度**：当前 pack 已用 `dynamic,1`，unpack 为按变量的
   static 循环。状态变量数少于 30 时会有一部分线程空闲。可尝试把一个变量的互不
   重叠 segment 再细分，但必须先证明边界 segment 不会写同一点。
3. **内存访问优化**：新 profile 中 `memcpy/memset` 自身已占 16.17%，LLC miss 约
   49.5%。下一阶段的布局、清零范围、复制合并和 first-touch/NUMA 实验，比继续调整
   OpenMP schedule 更有可能得到稳定收益。

当前已经低于 330 秒目标，所以下一步应优先保证结果稳定并以小开关做 A/B，而不应
为了追求“30/30 平均占用”破坏数值依赖或增加 ghost-zone 工作。
