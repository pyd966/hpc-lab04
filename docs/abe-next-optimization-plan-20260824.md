# ABE CPU 后续优化总计划

日期：2026-08-24
当前基线：`ffc6ebd`（OpenMP block 调度重设计之后）
主要证据：`profile/abe-20260824T081218Z-14/`

## 1. 结论

当前版本已经解决了最明显的 MPI、约束计算和 block 尾部问题，但还没有到
“只能微调”的阶段。后续仍有三块值得投入的空间：

1. 减少导数、边界同步和 AMR transfer 中不必要的整数组写入；
2. 让单进程 OpenMP 路径真正使用直接的内存传递，而不是保留 MPI 时代的
   pack/unpack 数据路径；
3. 在不增加寄存器 spill 的前提下，对 RHS 做小规模批处理、数据复用和局部融合。

按当前 30 物理核 profile 外推，ABE `Evolve(40)` 约为 `292--296 s`，应用内
端到端约为 `306--313 s`。其中：

- 低到中风险阶段如果都得到正收益，预计可再减少 Evolve **6%--11%**，即端到端
  大约到 **276--294 s**；
- 如果后续 RHS 批处理和局部融合也成功，累计 Evolve 收益可能达到
  **12%--18%**，对应端到端约 **255--277 s**；
- 第二个区间是努力目标，不是承诺。RHS、memcpy 和 memset 共同争用缓存与带宽，
  各阶段收益不能直接相加。

正式评测使用 60 个物理核，而当前 profile 的 cpuset 是 60 个逻辑 CPU、每核
2 线程，实际对应 30 个物理核。因此必须先做一次 60 物理核校准；当前时间外推
不能代替评测拓扑上的实测。

## 2. 当前 profile 告诉了我们什么

### 2.1 热点构成

最终 `perf record` 没有丢样本，flat profile 为：

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
| `kodis_` | 2.10% |
| `restrict3_` | 1.59% |
| `copy_` | 1.50% |

`compute_rhs_bssn`、三个导数/耗散例程和 `kodis` 的 self 样本合计约 59%。这还
没有把它们调用的 symmetry、memset 和 memcpy 算进去，因此完整 RHS 路径实际
超过 60%。RHS 仍然是第一主线，但不是唯一主线。

调用图进一步把数据搬运分成了两类：

- `omp_execute_cached_sync` 路径约占 12.69%，其中 `copy_` 约 9.40%，最终落到
  `memcpy` 的约 8.10%；
- `omp_local_transfer` 路径约占 9.11%，其中 `prolong3` 约 5.49%，`restrict3`
  约 2.59%。

这说明目前的主要问题已经不是频繁 `malloc/free`。flat profile 中 `malloc` 只有
0.18%，`free` 约 0.13%，而同步 workspace 也已经缓存。真正昂贵的是缓存好的
workspace 仍被反复读写，即“没有重新分配，但仍搬了两遍数据”。

### 2.2 CPU 和硬件计数器

| 指标 | 当前值 | 判断 |
|---|---:|---|
| 平均 CPU | 22.65 / 30 | 全窗口约 75.5%，仍有层级和同步空洞 |
| level 7/8 block phase utilization | 92.3% / 90.6% | 主移动层已经较均衡 |
| IPC | 1.80 | 不低到异常，也没有达到纯计算内核的高吞吐 |
| branch miss | 0.43% | 分支预测不是全局问题 |
| L1D miss | 3.86% | 一级缓存表现正常 |
| LLC load miss | 37.35% | 大工作集和数组流量仍然明显 |
| dTLB miss | 3.41% | 值得做 huge-page 实验，但不是第一优先级 |
| CPU migrations | 0 | 绑核有效 |

22.65/30 不能解释成“RHS 中一直有 7.35 个核没工作”。它包含 level 0 的 9 个
block、level 1--4 的 24-thread team、层间递归、Sync/transfer、regrid、constraint
和输出。移动 level 7/8 的主 block 阶段已经达到约 90%，所以继续只换
`static/dynamic/guided` 的收益上限很低。

本次 cpuset 和内存都位于 NUMA node 1，`Mems_allowed_list` 也是 1。当前没有
跨 NUMA 远程访存证据。NUMA 优化应在 60 物理核分配确认跨节点之后再启用，不能
预先把它当作主要收益来源。

### 2.3 如何理解收益上限

以下比例是 CPU cycles 样本，不是可以直接兑换的 wall time，但能给出上限：

- 即使 `compute_rhs_bssn` self 完全消失，上限也只有约 41.5%；若只让它快一倍，
  全局理论收益约 20.7%；
- 即使整个 cached Sync 免费，上限也约 12.7%；实际目标应是减少一半左右的数据
  搬运，而不是假设同步可以消失；
- 即使 local transfer 免费，上限也约 9.1%；prolong/restrict 的数学计算仍然必须做；
- 当前每减少 1% Evolve，大约节省 `2.9 s` 的 `t=40` 时间。

所以后面的收益区间都按“热点内部能减少多少工作量”估计，而不是把热点占比直接
当作最终收益。

## 3. 分阶段路线

### P0：建立 30/60 物理核正式基线

目标不是加速，而是消除评测拓扑的不确定性。

具体工作：

1. 用当前提交做一次 production `t=40`，得到真实的 Evolve、ABE total 和
   `This Program Cost`，替换目前的短程外推；
2. 当前 30 物理核环境做同节点三次 `t=4`，记录自然波动；
3. 如果队列允许，申请真实 60 物理核，核对 `OMP_PLACES` 展开、线程 affinity、
   cpuset、NUMA node 和每个线程的物理 core；
4. 60 核至少比较 60、90、120 个 block。当前 30 核使用 60 block 时每核有两个
   任务，可以动态窃取；评测 60 核若仍只有 60 block，每核只有一个任务，当前
   3.05% 的调度收益不一定能保留；
5. profile 版保留 `-g -fno-omit-frame-pointer`，production 关闭诊断计时，二者
   都保持严格 `-O3`。

预期直接收益：**0%**。这一阶段决定后面所有数字是否可信。

### P1：消除导数例程的整数组清零

证据：`memset` 占 5.82%。调用图显示 `fderivs` 和 `fdderivs` 内有明显的
memset 子路径。源码中 `fderivs` 在计算前清零三个完整输出数组，`fdderivs`
清零六个完整输出数组，随后 SIMD 内点循环又覆盖绝大多数元素。

具体改法：

1. 先增加仅用于 profile 的字节计数，分别统计 `fderivs`、`fdderivs`、`kodis`
   和其它路径清零了多少字节；
2. 改写内点与边界 shell，使每个有效输出点只写一次；
3. 对确实要求为零、但 stencil 没有覆盖的最外层平面或角点，只清零薄边界，
   不再清零整个三维体积；
4. `fderivs` 和 `fdderivs` 分成两个独立 A/B，不能一次提交，以便判断六输出版本
   是否因额外分支而抵消收益；
5. 保留 CMake 开关和标量 fallback，并对无对称、等面对称、八分体对称分别做
   单元级逐位比较。

预期 Evolve 收益：**1.5%--3%**，约 **4--9 s**。调用图中可直接归到
`fderivs/fdderivs` 的 memset 约占 3.7%，而 symmetry 和其它初始化仍然需要清零，
所以不能用整个 5.82% 作为本阶段的可实现收益。

风险：边界上原本依赖“初始化为零”的点很容易遗漏。验收必须逐位比较完整输出，
不能只看 trajectory RMS。

### P2：为单进程 Sync 增加直接复制快路径

证据：cached Sync 路径约 12.69%，其中约 8.10% 最终在 `memcpy`。当前 OpenMP-only
路径虽然不发送 MPI 消息，仍然执行：

```text
源 block -> packed workspace -> 目标 ghost/buffer
```

workspace 已经复用，因此继续优化 allocation 几乎无收益；应减少中间搬运。

具体改法：

1. 从缓存的 sync geometry 中导出每个 op 的源地址、目标地址和区间，证明哪些
   same-level owned-to-ghost 复制不存在地址重叠；
2. 对无重叠 op 直接从源 block 写到目标 ghost/buffer，跳过 packed workspace；
3. 如果多个 op 的目标相交，按变量和区域着色，保证同一批内互不冲突；
4. 如果存在读后写环，只为形成环的少数 op 使用 scratch，其余仍走直接路径；
5. 原 pack-before-unpack 实现保留为 fallback，用开关做同二进制 A/B；
6. 新增 Sync wall time 和实际搬运字节统计，验证收益确实来自字节数下降。

预期 Evolve 收益：**3%--6%**，约 **9--18 s**。若只能覆盖部分 same-level copy，
结果可能靠近下界；整个 12.69% Sync 路径不可能全部删除。

风险：ghost 区域可能相交，错误的并行直接写会产生数据竞争。地址区间和依赖关系
必须在实施前验证，不能仅凭“单进程”假设安全。

### P3：去掉 AMR transfer 的中间落地，并优化 prolong/restrict

证据：`omp_local_transfer` 占约 9.11%，其中 `prolong3` 5.49%、`restrict3` 2.59%。
当前非 mixed 路径先让 prolong/restrict 写 packed workspace，再用 `f_copy` 写目标。

分成两个实验：

1. **P3a，直接输出。** 为 prolong/restrict 增加接收目标 stride/bounds 的版本，
   让插值或限制结果直接写入最终目标区域。mixed transfer 和不能证明无冲突的
   情况继续走旧路径；
2. **P3b，内核循环。** 对 `prolongrestrict_cell.f90:2136/2143` 的热点循环检查
   vectorization report，提升连续 `i` 方向的 SIMD，外提重复的权重、索引和边界判断；
3. 每个子阶段单独 profile。直接输出和 stencil 算法优化不能放进同一个提交；
4. 记录 transfer 的输入、输出字节数和各 level 调用次数，防止只优化一次性路径。

预期 Evolve 收益：P3a **0.5%--1.2%**，P3b **1%--2%**；合计合理区间
**1.5%--3%**，约 **4--9 s**。P3a 能直接删除的 `copy_` 子路径约为 0.73%，
额外收益只能来自缓存流量下降，因此不应按整个 9.11% transfer 路径估算。

风险：AMR 边界的目标布局不是简单连续数组，直接写版本必须完整保留 ghost width、
symmetry 和 mixed prolong 的语义。

### P4：针对 60 物理核重新确定任务几何

这不是重新做已经完成的 30 核 schedule sweep，而是解决评测核数变化。

具体实验：

1. 在 60 核上比较 block target 60、90、120、150，记录每个 level 的
   `phase utilization`、balance、Sync 和 transfer wall；
2. 对 level 5--8 使用记录过的 block wall time，将重 block 放到队列前部，
   比较“原顺序 dynamic,1”和“heavy-first dynamic,1”；
3. 分 level 选择策略：移动层优先 dynamic，短小静态层只有在 profile 支持时
   使用 static/guided；
4. 同时比较静态层 48/54/60 threads，移动层保持 60。当前 30 核的 24/30 比例
   不能未经验证直接套到 60 核；
5. block 增多后必须同时检查 ghost/buffer 总点数。balance 更高但 transfer 变大
   的候选应淘汰。

预期收益：当前 30 核上只剩 **0.5%--2%**；60 核评测环境可能有 **1%--5%**。
也可能只有稳定性收益而没有加速。

明确不重试：

- 9-thread 静态层曾回退约 4.8%；
- 30 block 下单独切 schedule 没有效果；
- 跨整个 RK4 的 persistent OpenMP team 曾稳定回退 10%--12%；
- 当前 30 核上 90 block 已经回退。

只有新的 60 核证据才能推翻这些结论。

### P5：小批量处理 lopsided 和导数

单函数继续加 `!$omp simd` 的空间已经不大：`fdderivs` 和 `fderivs` 已有 SIMD；
原 lopsided SIMD 同时计算正、负两套 stencil，端到端没有收益。下一步应减少
多次调用之间的重复读取，而不是重复旧实验。

具体实验：

1. 把使用相同 `betax/betay/betaz` 的 lopsided 场按 2、3、4 个一组处理；每个
   网格点只读取一次 shift、只判断一次符号，然后对这一小组场使用相同方向的
   stencil；
2. 不一次批 24 个场。批量过大会增加活跃指针和寄存器数，容易 spill；
3. 对 `fderivs` 的六个 metric 场、`fdderivs` 的 Ricci 输入分别试 2/3-field
   batch，复用网格步长、边界分类和循环地址计算；
4. 每个 batch size 都检查反汇编中的向量宽度、栈 load/store 和编译器
   vectorization missed 原因；
5. 如果 batch 后内核自身没有至少 15% 改善，立即停止，因为它无法覆盖更复杂
   接口和寄存器压力。

预期 Evolve 收益：**2%--5%**，约 **6--15 s**，置信度中等偏低。

### P6：RHS 热行的局部数据复用和分块

证据：`compute_rhs_bssn` self 仍占 41.47%，热点集中在
`bssn_rhs.f90:609/648/687` 的 Ricci、连接和 Aij 组合。已有 metric、gij RHS 和
Gamma RHS 小规模融合有收益，但逆度规融合和 lopsided 累加融合没有稳定收益。

执行顺序：

1. 为三段热点分别生成 optimized/missed vectorization report 和反汇编，确认
   当前向量宽度、别名检查、寄存器 spill 和重复 load；
2. 先在单点循环内引入少量标量临时量，消除编译器没有处理掉的重复倒数、乘积
   和同数组重复 load；
3. 再测试小范围 producer-consumer 融合，只融合生命周期相邻、共享输入明显的
   两个 pass；
4. 测试 `j/k` 小 tile，使刚生成的连接或导数临时量尽可能留在 L2；tile size
   必须 sweep，不能只试一个；
5. 只有通用 O3 循环仍未充分向量化时，才对单一 kernel 测试局部 SVE 版本。
   全程序 `-mcpu=native` 已经稳定回退约 9%，不能重新作为全局选项；
6. 每次只保留一个局部改动。六个 Ricci 大表达式不能一次拼成超大循环，否则
   很可能因寄存器不足产生 spill。

预期 Evolve 收益：**3%--7%**，约 **9--21 s**。这是剩余空间最大的单项之一，
也是最容易出现“函数内更快、端到端无收益”的高风险阶段。

### P7：RHS 与 RK4 状态更新的融合

`rungekutta4_rout` self 只有 2.45%，单独向量化它的上限很低。真正可能有价值的
方案是减少 RHS 数组的落地：某个 RHS 场完成最后一次 lopsided/kodis 更新后，
直接进入对应 RK 更新，避免完整 RHS 数组再读一次。

实施前必须回答：

1. 哪些 RHS 输出在 RK 更新之外还被 constraint、analysis 或下一场方程读取；
2. predictor 和三个 corrector 的 RK4 次序是否允许逐 field 完成；
3. State、SynchList 和 RHSList 的 swap 是否依赖完整数组同时存在。

只有确认某组场是单生产者、单消费者，才做 2--4 个场的小型融合原型。预期
Evolve 收益 **1%--3%**，风险高，放在 RHS 局部优化之后。

### P8：TLB、对齐和编译器尾部实验

这些方向有依据，但收益上限较小：

1. **Huge page。** dTLB miss 为 3.41%，可对长期存在的大 `fgfs` 存储做
   `madvise(MADV_HUGEPAGE)` A/B，并检查实际 huge-page 命中；预期 **0%--2%**；
2. **对齐和 first touch。** 对大数组做 cache-line 对齐和并行 first touch。
   当前内存、CPU 已在同一 NUMA node，预期 **0%--1%**；只有 60 核跨 NUMA 时
   才可能更高；
3. **编译链接参数。** 可单独测试 LTO 或仅热点文件的架构参数，预期 **0%--1.5%**。
   不启用 `-Ofast/fast-math`；
4. **数学库。** 当前没有 BLAS/FFT 热点，`libm pow` 仅约 0.64%，更换数学库不会
   带来可见端到端收益；
5. **false sharing。** 当前没有迁核或 OpenMP runtime 异常热点。先用 cache-line
   争用工具证明存在共享写，再改线程私有计数器或 padding，不能凭猜测重排布局。

## 4. TwoPuncture 和非计算尾部

当前 TwoPuncture 约 10 秒，已远小于 ABE Evolve。即使再加速 20%，端到端也只
减少约 2 秒，不到 1%。因此在 ABE 仍有 6% 以上可信空间时，不应把主要开发时间
转回 TwoPuncture。

TwoPuncture、ABE 初始化、绘图和 Python driver 合计约 16 秒；扣除约 10 秒的
TwoPuncture 后，其余固定流程约 6 秒。它们影响外层 wall time，但课程的核心目标
是 TwoPuncture + Evolve；除非 Evolve 已经稳定低于目标且评分口径明确包含这些
流程，否则只做低风险的 I/O 精简。

## 5. 建议的实际执行顺序

建议按下面顺序推进，每个保留阶段完成后暂停审查：

| 顺序 | 阶段 | 预期 Evolve 收益 | 风险 |
|---:|---|---:|---|
| 0 | 30/60 核正式基线 | 0% | 低 |
| 1 | 导数输出避免整数组清零 | 1.5%--3% | 中低 |
| 2 | OpenMP Sync 直接复制 | 3%--6% | 中 |
| 3 | AMR transfer 直接输出 | 0.5%--1.2% | 中 |
| 4 | prolong/restrict 内核 | 1%--2% | 中 |
| 5 | 60 核 block/thread 几何 | 1%--5% | 中 |
| 6 | lopsided/导数小批量 | 2%--5% | 中高 |
| 7 | RHS 热行局部复用/分块 | 3%--7% | 高 |
| 8 | RHS/RK 小规模融合 | 1%--3% | 高 |
| 9 | huge page、对齐、局部编译参数 | 0%--2% | 低到中 |

P1--P4 都在减少确定存在的内存流量，证据最强，应先做。P5 解决的是评测硬件
变化，不能省略。P6--P8 有更高理论上限，但需要更多原型和失败实验，不适合与
前面阶段混在一个提交中。

## 6. 每阶段的验收协议

每个阶段统一执行：

1. 修改前保留同节点 baseline；
2. 候选使用编译期开关，可在同一源码切换；
3. 同一作业内交错运行 `baseline -> candidate -> candidate -> baseline`，至少两次；
4. 先跑 `t=4`，比较 Evolve、Total、task-clock、IPC、cache/TLB、线程利用率和
   目标子阶段 wall time；
5. 比较 `ADMQs/BH/constraint/psi4` 全部数值行，默认要求逐位一致，并运行课程
   `check.sh`；
6. 候选方向稳定后再做一次 `perf record`，要求无丢样本，并确认目标热点的绝对
   样本或子阶段 wall time下降；
7. 每个保留阶段写报告、提交并 push；未获益候选撤回代码，但在报告中保留配置、
   作业号和失败原因；
8. 累计提升超过 2% 或涉及 Sync/AMR/RK 语义时，补一次正式 `t=40`；
9. 完成一个阶段后暂停，等待审查，不把多个高风险改动一次合并。

短测保留门槛建议为：同节点交错均值至少改善 1%，或者目标内核明显加速且端到端
方向在全部重复中一致。低于门槛的结果按噪声处理，不能用单次最快值宣称收益。

## 7. 目前不建议做的事

- 不恢复 MPI 或 MPI+OpenMP。单节点单进程已经去掉真正的 MPI 通信，当前热点是
  本地数组计算和搬运；
- 不再实现通用 worker pool。跨 RK4 的 persistent team 已实测回退；
- 不全局启用 `-mcpu=native`、`-funroll-loops`、`-Ofast` 或 fast-math；
- 不更换 BLAS/数学库，程序没有相应热点；
- 不把所有临时数组简单永久保留。allocation 已不是热点，永久数组不能减少必须的
  读写，反而可能扩大工作集；
- 不再次使用“正负 stencil 全算一遍”的 lopsided SIMD；
- 不一次融合全部 Ricci 表达式，也不一次批处理全部 24 个场；
- 不为了提高平均 CPU 数删除算法必需的 barrier。正确目标是减少 barrier 前的尾部
  和 barrier 之间的数据搬运，而不是破坏阶段依赖。

## 8. 推荐的下一步

先完成 P0 的当前提交 `t=40` 实测和 60 物理核任务几何校准；如果 60 核暂时无法
申请，则直接进入 P1，分别处理 `fderivs` 和 `fdderivs` 的整数组清零。P1 的数学
语义最清楚，profile 证据明确，也能为后续 P5/P6 的批处理减少干扰。
