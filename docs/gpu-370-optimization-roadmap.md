# GPU 端到端 370 秒优化路径

> 更新日期：2026-08-27
> 正式目标：A100 MIG 1g.10gb、16 CPU、单 MPI rank、`t=100`，`This Program Cost <= 370s`
> 当前基线：commit `6ab1f2c`，`t=5` 三次均值 `74.633372 +/- 0.093387s`

## 结论先行

当前版本按短窗线性外推约为 `931s`，距离 `370s` 不是再做几轮 launch、stream 或 memcpy 微调就能补上的差距。`compact advection` 已经把旧 advection 热点加速约 `3x`，但主演化仍需再加速 `2.65x`。

从最新 Nsys 的 Amdahl 分布看，能达到目标的路线必须同时完成三件事：

1. 把 RHS family 从当前约 `6.50s/物理时间单位` 降到 `2.0-2.2s/unit`；核心是 tiled Hessian 和选择性导数复用，而不是继续拆更多 algebra kernel。
2. 把 prolong + restrict 从约 `1.34s/unit` 降到 `<=0.50s/unit`（首轮止损线 `0.60s/unit`）；核心是改变 `6x6x6` 插值的数据复用方式，而不是再做 descriptor/launch batching。
3. 把高频黑洞位置插值和通用 global interpolation 从约 `0.42-0.44s/unit` 降到 `<=0.16s/unit`（首轮止损线 `0.20s/unit`），同时清掉其大量小分配、同步和单 rank 下无意义的归约。

一个只有 `RHS 3x + AMR/interp 3x` 的方案线性外推仍约 `378s`，会失败。建议工程目标是 `RHS 3.5-4x + AMR/interp 2.5x 以上`，并把短窗斜率压到 `<=3.2s/unit`，为长程 AMR 增长留出余量。

这条路线在 Amdahl 上可行，但它要求结构性重写；现有 profile 不能证明这些重写一定达到预估。文档中的每一阶段都设置了止损门槛，避免在不能支撑 `370s` 的方向上持续投入。

## 1. 口径、当前版本和可复现性

正式约束来自 `AGENT.md` 和实验说明：

- 只走 GPU 路线，目标机器是 A100 MIG 1g.10gb，16 CPU、24 GiB。
- 正式时间是 `t=100` 的 `This Program Cost`；它包含 TwoPuncture 和 ABEGPU，不能拿 profiler 内的 kernel sum 代替。
- 网格、物理时间、输出、四阶差分和 RK4 不能改变，关键路径不能降精度，也不能使用预计算答案。
- 每个候选都必须通过 checker；短窗只用于筛选，不能替代完整 `t=100` 验收。

当前最好结果来自：

- benchmark：`profile/gpu-benchmark-20260827T083647Z-63/`
- Nsys：`profile/gpu-nsys-20260827T075056Z-66/`
- compact NCU：`profile/gpu-ncu-20260827T083353Z-64/`
- evolution NCU：`profile/gpu-ncu-20260827T081526Z-65/`
- beta-prepare NCU：`profile/gpu-ncu-20260827T081703Z-64/`
- prolong NCU：`profile/gpu-ncu-20260827T082407Z-65/`
- restrict NCU：`profile/gpu-ncu-20260827T083505Z-65/`
- global-interp NCU：`profile/gpu-ncu-20260827T084542Z-63/`

这些运行发生在 compact 提交之前，因此 job log 只记录基准提交 `92a3e131c1f99993dcbad3e2da7eed1c98558147`；当前实现已由 `6ab1f2c` 冻结。当前提交中的源码指纹为：

```text
980c5abc9b010f919128a4839738a9fd04ba8aad2df29321caed8149f4e4591b  src/bssn_rhs_gpu.cu
b89272cd992eb704655c3fa14e3073e6b535860b89aa1a777ee420dcfca28f10  src/advection_compact_gpu.cuh
```

源码现在已经可定位，但 artifact 本身没有记录当时的 dirty patch hash。第一步仍应在 clean `6ab1f2c` 上复现三跑，把提交号和性能结果真正绑定。

## 2. 从 74.63 秒到 370 秒的预算

三次 `t=5` 日志给出：

| 指标 | 当前值 |
|---|---:|
| Program Cost 均值 | `74.633372s` |
| Evolve 均值 | `45.080233s` |
| 固定成本 `Program - Evolve` | `29.553139s` |
| 当前演化斜率 | `9.016047s/unit` |
| 当前线性 `t=100` 外推 | `931.158s` |

目标约束是：

```text
演化预算 = 370 - 29.553139 = 340.446861s
目标斜率 = 340.446861 / 100 = 3.404469s/unit
仍需加速 = 9.016047 / 3.404469 = 2.648298x
```

如果固定成本不变，理论短窗硬门槛是：

| 窗口 | 刚好对应 370s | 建议的约 350s 余量线 |
|---|---:|---:|
| `t=5` Program | `46.58s` | `45.58s` |
| `t=10` Program | `63.60s` | `61.60s` |
| `t=20` Program | `97.64s` | `93.64s` |
| Evolve 斜率 | `3.404s/unit` | `3.204s/unit` |

不能直接相信 `931s` 或表中的线性值。已有正式长程运行表明后段 AMR 工作量会增加；正确做法是在 `t=10/20` 记录每层 block、cell、prolong/restrict 和插值次数，按事件数量重新拟合 `t=100`，而不是只把 `t=5` 乘二十。

## 3. 当前热点和受限因素

最新 Nsys 覆盖 `t=0..4`，Evolve 为 `36.4204s`，GPU kernel sum 为 `35.9119s`。kernel 时间并集约 `34.7104s`，多 kernel 并发只有约 `3.35%`；这说明当前首先是 GPU 计算量问题，而不是缺少 stream。

### 3.1 Kernel 排名

| Kernel/组 | t=0..4 GPU 时间 | 占 kernel sum | 约 s/unit |
|---|---:|---:|---:|
| `rhs_evolution_kernel` | `6.5828s` | `18.3%` | `1.646` |
| `rhs_beta_gamma_prepare_kernel` | `3.3893s` | `9.4%` | `0.847` |
| compact advection | `3.3157s` | `9.2%` | `0.829` |
| `prolong3_batch_kernel` | `3.1867s` | `8.9%` | `0.797` |
| `restrict3_batch_kernel` | `2.1838s` | `6.1%` | `0.546` |
| `rhs_geometry_kernel` | `2.1552s` | `6.0%` | `0.539` |
| `global_interp_kernel` | `1.6719s` | `4.7%` | `0.418` |
| `rhs_source_lapse_kernel` | `1.5254s` | `4.2%` | `0.381` |
| `rhs_source_chi_hessian_kernel` | `1.4682s` | `4.1%` | `0.367` |
| Sommerfeld | `1.3923s` | `3.9%` | `0.348` |
| constraints | `1.3641s` | `3.8%` | `0.341` |

精确分组为：RHS `72.35%`，prolong/restrict/global-interp `19.61%`，其余 `8.04%`。Nsys 各 stream 的 kernel duration 会重叠，所以该表用于排序和 Amdahl 预算，不能把 API 时间再次加到它上面。

### 3.2 NCU 说明了什么

| Kernel | registers/thread | 理论/实际 occupancy | 关键证据 |
|---|---:|---:|---|
| evolution | `140` | `12.5% / 11.0%` | `No Eligible 70.68%`，约 `26%` excessive global sectors，无 compiler spill |
| beta prepare | `140` | `12.5% / 10.93%` | `No Eligible 70.96%`，约 `23%` excessive global sectors，无 compiler spill |
| compact advection | `78` | `37.5% / 35.28%` | `No Eligible 57.06%`，约 `34%` excessive shared wavefronts，无 spill |
| prolong | `93` | `25% / 22.12%` | `No Eligible 57.3%`，NCU local-memory 规则估计约 `31.8%` 潜在收益 |
| restrict | `191` | `12.5% / 11.42%` | `No Eligible 73.34%`，`52%` excessive global sectors，local-memory 规则约 `21.0%` |
| global interp | `64` | `50% / 43.69%` | memory throughput `86.15%`，`No Eligible 79.79%`，local-memory 规则约 `33.5%` |

evolution 和 beta prepare 都不是 DRAM 带宽打满：它们约 `28-29%` compute throughput、`12-14%` memory throughput。真正问题是 `140` 个寄存器导致只有一个 256-thread block 驻留、长依赖链，以及通用 stencil 的重复/不合并访问。盲目限制寄存器很可能制造 spill，必须通过缩小活动值集合和专用 producer 自然降寄存器。

RHS/AMR NCU 主要取到最早的 launch，例如 RHS 是 `(5,5,5)` grid；global-interp 则是首个 `(144,1,1)` 大分析 launch。它们适合定位代码形态，但不能代表所有细层 patch；正式决策前必须用 `launch-skip` 补采代表性细层。

### 3.3 API 时间不能重复计入

Nsys 报告的 `cudaMemcpy 19.55s`、`cudaDeviceSynchronize 9.32s` 和 `cudaStreamSynchronize 5.50s` 大多是 CPU 阻塞等待 GPU。真实 H2D + D2H 只有约 `0.435s`，launch API 约 `0.790s/t=4`。kernel sum 已接近整个 Evolve，这些 API 时间不能与 kernel 时间相加。

不过，高频黑洞插值确实制造了大量不必要的调用：`normalize_shellf_kernel` 有 `320` 次，即约 `80` 次/unit；其通用路径每次还会分配、同步、拷贝和做 MPI 归约。这是一个具体、可消除的控制路径，不等价于泛化地“优化所有 memcpy”。

## 4. Amdahl 可行性

以当前 kernel 时间归一化：

```text
RHS                    = 0.723472
AMR + global_interp    = 0.196104
其他                    = 0.080425
```

| RHS 加速 | AMR/interp 加速 | 其他不变时的线性 t=100 | 判断 |
|---:|---:|---:|---|
| `3x` | `3x` | `~378.4s` | 明确不够 |
| `3.5x` | `2.5x` | `~359.2s` | 理论达标，余量很小 |
| `3x` | `4x` | `~363.7s` | 理论达标，依赖 AMR 大幅收益 |
| `4x` | `2x` | `~353.5s` | 理论达标 |
| `4x` | `2.5x` | `~335.9s` | 推荐的余量区间 |

两个直接结论：

- compact advection 现在只有 `9.2%`。即使它变成零成本，整体也最多加速约 `1.10x`，不能继续作为主路线。
- 如果 AMR/interp 完全不动，仅靠 RHS 达标需要 RHS 约 `7.3x`，同样不现实。

建议以如下每单位预算指导实现：

| 成本组 | 当前约值 | 370 硬预算 | 有余量的开发预算 |
|---|---:|---:|---:|
| RHS family | `6.50s` | `2.10s` | `2.00s` |
| prolong + restrict | `1.34s` | `0.50s` | `0.45s` |
| global/BH interpolation | `0.42-0.44s` | `0.16s` | `0.15s` |
| RK/Sommerfeld/enforce/pack 等 | `~0.69s` | `~0.55s` | `~0.52s` |
| 临界路径空隙 | `~0.14s` | `~0.10s` | `~0.08s` |

硬预算合计约 `3.41s/unit`（舍入后对应 370 秒），开发预算合计约 `3.20s/unit`。

## 5. 实施路线

### M0：冻结当前 compact 基线

目标不是优化，而是建立可信的比较点。

1. 以 clean `6ab1f2c` 作为 M0 候选；后续 artifact 必须同时记录 `HEAD`、clean/dirty 状态和编译参数。
2. 用完全相同的源码重新跑 `t=5` 三次。要求均值与 `74.6334s` 的差异不超过约 `1%`，三次 checker 全部通过。
3. 跑一次 `t=10` 和 `t=20`，输出逐步 wall time、每层 block/cell 数、prolong/restrict/global-interp 次数。
4. 为 evolution、beta prepare、prolong、restrict 和 global interpolation 各采“第一个粗层 + 一个代表性细层”两组 NCU。

若三跑不能复现，就以新三跑为 M0，不能继续引用 `74.63s` 作为收益基准。

### M1：先消掉高频控制路径和无用工作

#### M1-A：黑洞位置专用插值 fast path

当前调用链是：

```text
Step_GPU (每个 finest-level step)
  -> compute_Porg_rhs (2 个 BH)
    -> 3 次标量 H2D 坐标拷贝/BH
    -> PatList_Interp_Points_GPU
      -> 3 次坐标 D2H 做 host block selection
      -> 每变量 global_interp launch
      -> device sync + D2H + 2 次 MPI_Allreduce
      -> H2D + normalize + device sync
    -> 3 值 D2H/BH
```

代码入口：`src/bssn_step_gpu.C:29`、`src/bssn_gpu_class.C:2699`、`src/Parallel_GPU.cpp:744`、`src/fmisc_gpu.cu:156`。

实现方案：

1. 坐标本来就在 host 的 `Porg0` 中，直接在 host 选择 level/patch/block，不再 H2D 后又 D2H。
2. 第一版只为正式的 `nprocs == 1 && NN == 1` 建 fast path：对每个 BH 直接选中本地 block，并在各自的 `Block::stream` 上执行；多 rank 和其它通用插值保留原路径。
3. 第一版保持 `global_interp_device` 的数学不变，只按 BH 批量处理三个 field；若两个 owner stream 不同，用 event 汇合到专用 interpolation stream。使用持久的 6-double device/pinned-host 输出，每个 finest step 只做一次 D2H 和一次最终 stream 等待，并跳过 weight、normalize、device-wide sync 和 `MPI_Allreduce`。
4. 第一版稳定后再建立两个 BH 的 descriptor：三个 field 指针、基准 index、x/y/z 各 6 个固定阶 Lagrange 权重、parity 和 level；一个小 kernel 直接做 `2 BH x 3 fields` tensor-product contraction，删除 `ya[216]` 和动态 Neville 临时数组。
5. owner 不唯一、level 未命中或非正式配置时立即走原 fallback；fast path 不改变通用 API 的语义。

验收：逐 step 对比新旧 `Porg_rhs/Porg1`，随后 checker 全过。调度版应把 device-wide sync 降到 `<70 次/unit`，并至少节省 `0.15s/unit`；达不到时先检查 producer-stream 依赖和 fallback 命中率。直接权重版再以 BH fast-path NVTX 区间 `<=0.08-0.12s/unit` 为目标；浮点结合顺序改变后必须重新比较完整 trajectory。

#### M1-B：约束计算去重

`rhs_constraints_kernel` 在 Nsys 中调用 `447` 次。普通 RHS 只有 predictor (`co == 0`) 计算约束，而 `Constraint_Out` 又明确对 `lev > 0` 使用 `RHS_CONSTRAINT_ONLY` 重算，见 `src/bssn_gpu_class.C:2826`。

先画清楚从 predictor 到 `Constraint_Out` 之间是否有任何 lev>0 consumer；若没有：

1. 普通 predictor 只在 lev0 保留 constraint，lev>0 设置为不计算。
2. 保留 `Constraint_Out` 的 lev>0 重算，确保移动网格后的数据正确。
3. 对 `RHS_CONSTRAINT_ONLY` 建 producer/consumer 表，只启动真正生成 `Gam*`、`R*` 和 constraint 输入的 kernel；当前它仍会经过 geometry、Ricci、Gamma seed、beta Hessian 和 evolution 等完整前半链，其中有些输出不被 constraint 消费。
4. 将 7 个 `L2Norm_GPU` 合为一次 7-variable reduction 和一次同步，见 `src/bssn_gpu_class.C:2898`。

这是删除冗余工作的候选，不允许凭注释直接删。必须比较全部输出时刻的 7 个约束场和 checker；任何数据依赖不清楚时保留原路径。

#### M1-C：每 kernel 的 block shape 小扫参

当前 launcher 对所有 RHS kernel 固定使用 `(8,8,4)`，见 `src/bssn_rhs_gpu.cu:1582`。先将 evolution/beta 的 launch shape 与其他 kernel 解耦，测试：

```text
(8,8,4)  256 threads，现基线
(8,8,2)  128 threads
(8,4,4)  128 threads
(16,4,2) 128 threads，保持 x 连续访问
```

`140 registers/thread` 下，128-thread block 至少有机会增加驻留 block 数；但只按 kernel duration 选，不按 occupancy 数字选。单 kernel 没有 `>=5%` 收益就停止，不要把全 launcher 一起改形状。

M1 结束建议门槛：Evolve 斜率 `<=8.5s/unit`，约对应 `t=5 Program <=72.1s`。

### M2：重写 Hessian 热点的数据访问

这是最高优先级的计算主线。

#### M2-A：evolution 的 contracted Hessian producer

`rhs_evolution_kernel` 在 `src/bssn_rhs_gpu.cu:84` 连续对 6 个 metric field 调用 `d_fdderivs_point`。每次通用 helper 计算六个二阶/混合导数，consumer 随即只保留：

```text
g^xx f_xx + g^yy f_yy + g^zz f_zz
+ 2 (g^xy f_xy + g^xz f_xz + g^yz f_yz)
```

实现一个 equatorial 专用、legacy fallback 保留的 radius-2 tiled producer：

1. 按 field 顺序加载 shared tile；混合导数需要 xy/xz/yz 平面 halo，不能照搬只含轴向 halo 的 advection tile。
2. 直接累加 contracted Hessian，不让六个 Hessian 和所有 stencil 值同时存活。
3. 每次只驻留 1 个、最多 2 个 field，避免 shared memory 和寄存器随 6 个 field 成倍增长。
4. 将无边界的 interior 与 symmetry/outer-boundary 路径分开；official equatorial 模式使用编译期 parity，legacy 模式继续调用通用 helper。
5. 联合测试 128-thread shapes；目标通过缩短 live range 把寄存器自然压到 `<=128`，而不是用 `maxrregcount` 强压。

首个 tiled prototype 的 go/no-go 门槛是 evolution 从 `1.646` 降到 `<=0.8s/unit`，无 local spill 且 checker 通过；为了满足最终 RHS 预算，M3 结束时它还应进入约 `0.45-0.60s/unit`。

#### M2-B：beta、chi 和 lapse Hessian

按相同 primitive 依次替换：

- `rhs_beta_gamma_prepare_kernel` 的三个 beta Hessian，代码在 `src/bssn_rhs_gpu.cu:153`；只输出三个 Laplacian 和 divergence-Hessian 所需分量。
- `rhs_source_chi_hessian_kernel` 的 chi gradient + Hessian，代码在 `src/bssn_rhs_gpu.cu:1044`。
- `rhs_source_lapse_kernel` 的 lapse gradient + Hessian，代码在 `src/bssn_rhs_gpu.cu:1138`。

先做 evolution，再复用已经验证的 tile loader；不要一开始写一个覆盖所有 parity/所有 derivative 的巨型模板。beta-prepare prototype 的门槛是 `<=0.4s/unit`，最终 beta-prepare + beta-gamma 合计目标也是 `<=0.4s/unit`。evolution + beta 合计若没有至少 `35%` kernel 收益，或 `t=5` 没有节省 `>=5s`，停止 shape/padding 微调，直接进入 M3 的两阶段 producer/consumer 设计。

### M3：选择性导数复用，而不是全量物化

当前代码已经暴露出明确重复：

| 导数 | 当前重复位置 | 建议发布内容 |
|---|---|---|
| beta 一阶导 | geometry、A-diag、A-offdiag | 9 个 beta gradient，供一个受控的 metric/A algebra consumer 使用 |
| chi gradient | geometry、chi-hessian、chi-Ricci、physical-Gamma、constraints | 3 个 gradient，保持到最后一个正常 RHS consumer；constraints 单独处理 |
| lapse gradient/Hessian | lapse 及其下游 | 只发布 covariant Hessian/trace 真正需要的 6+1 个量 |
| metric gradient | geometry、constraints | 普通 predictor 复用 geometry 结果，或只发布 contracted conformal Gamma |
| trK/A gradient | constraints | 只在 constraint schedule 中生产，不进入每个普通 RK stage |

实现顺序：

1. 先画出 `gxx_rhs...Gmz_Res` 每个 scratch slot 的 producer、最后 consumer 和覆盖点。
2. 先做低风险 chi 复用：geometry 已把三个 chi gradient 写入 `chi_rhs/trK_rhs/Lap_rhs`。将 `source_metric` 移到 `physical_gamma` 之后，并让 chi-hessian、chi-Ricci、physical-Gamma 读取这三个 cache，可删除三次重复 `d_fderivs_point`。
3. beta 的 9 个 gradient 不能靠简单移动 `source_metric` 全部复用：source_metric 会覆盖 6 个 metric-RHS slot，而任一 A kernel 写回也会破坏另一个消费者。先融合 A-diag + A-offdiag 以省一组导数；更高收益版本应把 metric + 六个 A 分量做成受控 algebra consumer，先 load 全部 derivative，再写 12 个 RHS，或使用独立的 9-value cache。
4. constraint schedule 与普通 RHS 分离，避免为了低频输出让所有 RK stage 物化 A/trK derivative；chi/metric gradient 仅在确有 consumer 时延长生命周期。
5. 只有被两个以上重 kernel 消费、且新增全局写读小于省掉的 stencil 读取时，才建立 cache。

仅做跨 kernel cache 很可能仍不足以把 RHS 压到 `2.0-2.3s/unit`。M3-B 应原型化 field-centric spatial producer：同一个 field 的 shared tile 驻留期间，一次完成它需要的所有空间算子，而不是按方程重复加载。

- metric group：一次 tile 生成 metric gradient、contracted Hessian，以及该 field 的 advection/KO。
- shift group：一次 tile 生成 9 个一阶导、Laplacian/divergence-Hessian summary，以及 advection/KO。
- chi/lapse group：一次 tile 生成 gradient、需要的 Hessian contraction/分量，以及 advection/KO。
- A/trK/Gamma/dtSf 等只需 advection 的 field 继续走缩小后的 compact producer。
- 全局 scratch 只保存低维 summary 和确有多 consumer 的值；每次仍只驻留 1-2 个 field，algebra consumer 保持低寄存器。

这与已经失败的 equation-centric fission 不同：旧方案让多个方程 kernel 分别重读 stencil，并物化大 tensor；field-centric 方案的判定标准是“一个 tile load 覆盖该 field 的全部空间算子”。M2 的独立 Hessian prototype 后续可以并入这个 producer。

不要恢复“完整 18 个 Christoffel/全导数张量物化”的旧方案。既有 Ricci fission 已证明额外全局 scratch 会把 consumer 推向 DRAM，完整 fission 也因为重复读取而回退。

M2 + M3 的第一止损线是整个 RHS family `<=2.3s/unit`，370 硬预算是 `<=2.1s/unit`，有余量目标是 `<=2.0s/unit`。如果 field-centric prototype 后 RHS 仍大于 `3.0s/unit`，则当前设计不足以支撑 `370s`，不应继续靠小 kernel fusion 外推达标。

M2/M3 结束建议门槛：总 Evolve `<=4.8-5.0s/unit`，约对应 `t=5 Program <=53.6-54.6s`。

### M4：改写 AMR 插值核心

Stage 4 已把 AMR launch 数降低约 `95.6%`，当前剩下的 `1.34s/unit` 主要是插值数学和数据复用问题。

`src/prolongrestrict_cell_gpu.cu:327` 的 prolong 以及 `:260` 的 restrict 都有 `tmp2[6][6] + tmp1[6]`。每个输出点独立读取至多 `6^3=216` 个源值，相邻输出却没有合作复用。prolong 新 NCU 为 93 registers、实际 occupancy 22.1%，且 local-memory 规则给出约 31.8% 的改进提示。

restrict 更差：batch kernel 在 `src/prolongrestrict_cell_gpu.cu:539` 内仍逐线程构造多组 3 元 geometry 数组，再调用通用 `d_restrict3_device`。新 NCU 显示 191 registers、实际 occupancy 11.42%、`No Eligible 73.34%`、约 52% excessive global sectors。它应先完成与 prolong 对称的 host-precomputed scalar path，再进入共享 tile 重写。

按以下顺序原型化：

1. restrict P0：新增 `d_restrict3_precomputed`，将 base、lbc/lbf、有效范围和 coarse-to-fine 映射参数放进 host descriptor；device 只收 scalar，删除每线程 `arr_llb*` 和 geometry 循环。要求 registers 明显下降且 kernel 至少 `10%` 收益。
2. 短实验：把固定 6 阶循环模板化/展开，流式收缩，确认显式 local array 是否消失。没有额外 `10%` 收益就不在标量化上继续投入。
3. prolong 主方案：coarse-cell-centric block 合作加载共享 coarse tile，一次生成 8 个 fine children；8 种 even/odd 权重组合共享同一个 `6x6x6` 邻域。
4. restrict 主方案：多个相邻 coarse output 合作加载重叠的 fine tile，再做三维 6-tap contraction。
5. 备选方案：x/y/z 三个 separable pass，变量维度批处理，使用按 stream 持久中间 buffer。只有三遍全局流量仍低于当前重复 216-load 路径时才保留。
6. 继续保留当前跨变量 batching；descriptor 可持久化，但它是收尾，不是主要收益来源。

必须保持现有 `C_PROLONG/C_RESTRICT`、parity、边界语义和四阶方案。逐点累加顺序改变可能造成 FP 差异，先对随机 interior、symmetry plane、outer boundary 做 GPU/旧实现 A/B，再跑 checker。

M4 第一止损线是 prolong + restrict `<=0.60s/unit`，最终硬预算是 `<=0.50s/unit`。完成后总 Evolve 应接近 `<=3.5s/unit`；若高于该值，不应直接跑 `t=100`。

### M5：通用 interpolation、analysis 和尾部整理

黑洞 fast path 之外，固定的波形/约束采样点仍可优化：

新 global-interp NCU 的首个大 launch 为 `(144,1,1)` grid：64 registers、实际 occupancy 43.69%、memory throughput 86.15%，NCU 对显式 local-memory 数组给出约 33.5% 的优化提示。它说明瓶颈是 `ya[216]`/Neville 中间数据流和逐变量重复读取，不是 occupancy；该样本不是两个 BH 的小 launch，BH fast path 仍应单独加 NVTX 或用 launch-skip 采样。

1. 对不随时间变化的点，缓存 owner block、基准 index、parity 和 6-tap 权重，只有 regrid 时更新。
2. 将 `src/Parallel_GPU.cpp:860` 的逐变量 launch 改为 `(point, variable)` 二维 batch；7 个 constraint field 共享位置与权重。
3. 唯一 owner 已确定时直接写结果，避免 atomic weight/normalize；多 rank fallback 保留原归约语义。
4. 将 `global_interp_device` 的 `ya[216]`、`yatmp[36]` 和多层 Neville 算法替换为固定 6-tap tensor contraction。
5. 将同层 7 个 L2 norm 合并成一次批量 reduction 和一次 D2H/sync。

当 Evolve 已进入 `3.7-3.8s/unit` 后，再考虑：

- 按边界属性批量 Sommerfeld；当前约 `0.348s/unit`。
- purpose-specific 的持久 scratch/descriptor。
- 对真正稳定的 level/stage 子图做 CUDA Graph，并重新用 Nsys 证明 CPU bubble 已成为瓶颈。

compact advection 只保留一轮有止损线的可选优化：shared x-stride padding、去掉不需要的角落 halo、128-thread block shape、interior/boundary 分离。要求 compact kernel 自身至少 `10%`、`t=5` E2E 至少 `1%` 才保留；否则冻结当前版本。它不在 `370s` 的关键路径上。

M5 最终门槛：`t=5 Evolve <=16.0s`、`Program <=45.6s`，即斜率 `<=3.2s/unit`。

## 6. 里程碑和决策门

| 里程碑 | 必须完成的工作 | Evolve 斜率门槛 | t=5 Program 参考 |
|---|---|---:|---:|
| M0 | clean `6ab1f2c` 三跑复现、补 t=20 | `~9.02s/unit` | `~74.63s` |
| M1 | BH fast path、约束去重、局部 shape sweep | `<=8.5s/unit` | `<=72.1s` |
| M2/M3 | Hessian tile + 选择性 derivative reuse | `<=4.8-5.0s/unit` | `<=53.6-54.6s` |
| M4 | prolong/restrict 结构重写 | `<=3.5s/unit` | `<=47.1s` |
| M5 | interpolation/analysis/尾部整理 | `<=3.2s/unit` | `<=45.6s` |

这些是 go/no-go 门槛，不是把各项“预计收益”相加后的承诺。每个里程碑结束都重跑 Nsys，按新的热点重新排序；某项下降后，旧百分比立即失效。

建议实验接受规则：

- 数学/数据流改动：checker 必须通过，且 `t=5` 三跑均值必须改善；不接受只改善单个 NCU 指标。
- 小优化：E2E 收益至少 `1%` 或明显大于两组噪声区间；否则回退候选。
- 大 kernel：粗层和代表性细层都不能出现 spill/灾难性 occupancy 回退。
- 每次只改变一个假设，保留 control/candidate 两个可复现实验点。

## 7. 验证流程

每个候选按固定顺序验证：

1. 远端 A100 构建；本机 x86 只做源码检查，不能代表 CUDA 性能。
2. `t=1` 或最短可用 checker，覆盖 parity/boundary；数值不通过立即停止。
3. NCU 各采一个粗层和代表性细层 launch，记录 registers、local memory、occupancy、stall、global/shared sectors。
4. 无 profiler 的 `t=5` 三跑，记录 Program、Evolve、每层事件计数和 checker。
5. 里程碑候选跑 `t=10/20`，用事件计数模型外推；要求预测 `t=100 <=350-360s`。
6. 最终候选在正式配置跑完整、无 profiler 的 `t=100`，以 `This Program Cost <=370s` 和完整 checker 为唯一验收。

当前 compact 的 checker 只覆盖了 `5/100` 时间点，不能据此宣布正式正确。

## 8. 不要重复的方向

以下方向已有负收益或 Amdahl 上不可能补足差距：

| 方向 | 结论 |
|---|---|
| Ricci 按输出分量拆分 | 约 `+5.1%` 回退，重复读取/写 scratch |
| geometry/equation 粗拆、六段/九段完整 fission | 回退，增加全局数据流并未解决 stencil 成本 |
| 完整 forced-inline | 从约 `84.72s` 回退到约 `90.64s` |
| 全 event 化、窄同步、segment/buffer event | 持平或更慢，当前不是同步 API 主导 |
| full direct-device transfer | 多个候选慢约 `0.8-4%`；API memcpy 大多是等待 |
| async AMR descriptor copy | 明显回退；launch batching 已基本完成 |
| 独立 RK4 batch | E2E 约 `+1.15%` |
| 通用显存池 | 约 `+5%`；只允许按用途持久 buffer |
| 更多 stream 或更多 MPI rank | 重 kernel 已占满 14 SM，kernel overlap 很小 |
| 完整 18 Christoffel/全 derivative tensor 物化 | consumer 变成 DRAM 受限 |
| fast-math、降精度、改 RK/网格/输出 | 违反实验约束或数值风险不可接受 |

`docs/gpu-optimization-report.md` 已随 `6ab1f2c` 更新，适合作为完整实验台账；本文在其基础上补充新的 evolution/beta/restrict/global-interp NCU、BH 高频调用链和精确的 370 秒 Amdahl 预算。历史章节中的 P1/P2 基线不应再当作当前热点排序。

## 9. 推荐执行顺序

```text
冻结并复现 compact
  -> BH 专用插值 + 约束去重
  -> evolution tiled contracted Hessian
  -> beta/chi/lapse tiled derivative
  -> 选择性 derivative lifetime/cache
  -> prolong/restrict shared-tile 或 separable 重写
  -> 固定点 global interpolation + analysis batching
  -> 仅按新 profile 做 Sommerfeld/Graph/compact 收尾
  -> t=20 事件模型
  -> 完整 t=100 正式验收
```

真正的关键路径不是“继续把某一个 kernel 磨快”，而是让同一个网格邻域只被合理加载一次，并让 AMR 的重叠插值点合作复用输入。只有 RHS、AMR/interp 两条主线都达到阶段预算，`<=370s` 才有可执行的数学基础。
