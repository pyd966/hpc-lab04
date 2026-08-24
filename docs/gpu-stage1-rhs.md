# GPU 阶段 1：RHS 数据流重构与实验报告

> 日期：2026-08-24  
> 对照基线：`d571ff8`（RHS 源码与回滚点 `0951bc3` 相同）  
> 平台：NVIDIA A100 80GB PCIe，MIG `1g.10gb`，CUDA 13.3，`sm_80`，1 个 MPI rank，ABEGPU OpenMP OFF  
> 范围：只修改 `src/bssn_rhs_gpu.cu`；TwoPuncture、AMR、RK、边界和同步路径均未改动。

## 1. 结论

这一轮得到了一版正确且有明确收益的 RHS 重构，但**没有通过阶段 1 的完整硬门槛**：

- `t=0..1` Nsys 中，RHS kernel family 从基线 `13.9386 s` 降到 `10.4985 s`，下降 `24.7%`，即 `1.33x` 加速。
- 同一次采样中，Total Evolve 从 `17.5389 s` 降到 `14.2263 s`，下降 `18.9%`。未修改的 `prolong3_kernel` 为 `2.1597 s`，与基线 `2.1468 s` 接近，说明差异不只是 GPU 频率波动。
- 新拆出的 metric Hessian、shift Hessian、advection/KO kernel 均达到理论 occupancy `25%`，且 NCU 的 compiler spill 指标为 0。
- 核心 `rhs_kernel` 仍为 255 registers/thread，理论/实际 occupancy 为 `12.50%/11.03%`。虽然其 spill 从实验中间版本的 107 KB 降到 38 KB，但它仍未跨过 occupancy 档位。
- 三次 `t=5` 独立运行全部通过 checker，程序计时均值为 `101.667 s`，样本标准差 `0.583 s`。同节点交错 A/B 中，候选端到端均值为 `75.237 s`，基线为 `95.360 s`，下降 `21.1%`（`1.27x`）。

阶段门槛要求 RHS 至少 `2x`，或证明后续还有清晰的 `3x` 空间。本版只有 `1.33x`，而最大的两个剩余 kernel 仍受寄存器/长延迟 stencil 限制，所以不能把阶段 1 标记为完成，也不能直接进入只调 block/grid 的阶段 2。

## 2. 数据流分析与保留的实现

原始 `rhs_kernel` 在一个线程生命周期内同时保留一、二阶导数、逆度规、Christoffel、Ricci、24 个方程源项、advection/KO 和约束所需变量。寄存器峰值不是由某一条表达式造成，而是多个物理阶段的 live range 重叠。因此本轮沿真实 producer-consumer 依赖拆分，不改变每个 patch 的计算域和 stream 顺序：

```text
metric/shift fields
        |
        +--> batched Hessian scratch --+
                                       v
                              geometry + equation core
                                       |
                  +--------------------+-------------------+
                  v                                        v
       24-field advection/KO batch              co==0 constraints
```

### 2.1 Metric Hessian batch

`rhs_metric_hessian_batch_kernel` 一次 launch 处理 6 个 metric 分量，`grid.y=3`，每个线程处理两个字段。它计算并收缩各分量的 Hessian，只物化 Ricci 后续真正消费的 6 个标量。

为了不引入每个 patch 的额外分配，结果暂存到原有 `Rxx..Rzz` scratch。随后同一 stream 上的 core kernel 顺序读取这些值，并在 `co==0` 时才用最终 Ricci 覆盖该 scratch。这个复用依赖同一 stream 的有序执行，不存在跨 stream 共享。

### 2.2 Shift Hessian batch

`rhs_shift_hessian_batch_kernel` 对 3 个 shift 分量批量计算 6 个二阶导数，`grid.y=3`，每线程一个字段。输出暂存于 18 个已有 Christoffel scratch 数组；core 消费完成后，在 `co==0` 路径将其覆盖为约束需要的物理 Christoffel。

这种生命周期复用避免新增持久显存，但也限定了 kernel 顺序。未来做异步执行图时，必须把这些 scratch 的 producer-consumer event 一并纳入依赖，不能让相同 patch 的两次 RHS 重叠使用它们。

### 2.3 跨变量 advection/KO batching

`rhs_advection_ko_batch_kernel` 用紧凑 metadata 描述 24 个演化字段的输入、RHS 输出和三轴 parity。`grid.y=6`，每个线程顺序处理 4 个变量。它把所有方程共有的 lopsided advection 和 Kreiss-Oliger dissipation 从 core 的长 live range 中移出，同时把逐变量逻辑合并为 6 个变量批次，而不是 24 次 launch。

对每线程 4、8、24 个变量做了实测。8 个变量相对 4 个变量的归一化时间约慢 `0.6%`；24 个变量约慢 `8.8%`。原因是更大的字段循环没有增加空间复用，却延长单线程依赖链并增加 metadata/local-memory 压力，因此保留 4 个变量一组。

### 2.4 约束路径

Hamiltonian 与三个 momentum 分量从 core 中独立出来，并只在 `co==0` 发射。三个 momentum kernel 通过编译期方向模板复用公式，使 parity、张量分量和导数方向可在编译期消解。这消除了普通 corrector 中无用约束局部量的生命周期。

### 2.5 Core live-range code motion

core 中与 beta、A、Gamma 和最终方程装配相关的加载被推迟到 Ricci 计算之后。寄存器数仍是 255，说明跨过 `12.5% -> 25%` occupancy 档位需要结构性缩短几何/Ricci 生命周期；但 NCU spill 从重排前的 107 KB 降到 38 KB，因此保留该改动。

## 3. 实验筛选

| 实验 | 结果 | 决策 |
| --- | --- | --- |
| 单独拆 constraints + advection/KO | RHS family 约 `12.12 s` | 保留，验证独立 consumer 有收益 |
| 增加 metric Hessian batch | RHS family 约 `11.43 s` | 保留；新 kernel 104 regs、25% theoretical occupancy |
| 增加 shift Hessian batch | 继续降低 core 时间 | 保留；新 kernel 104 regs、25% theoretical occupancy |
| beta 一阶导数额外物化 | core 仍 250 regs，并产生约 22 KB spill，整体更慢 | 淘汰；写回/重读成本没有换来 occupancy 档位变化 |
| advection 每线程 4/8/24 变量 | 4 变量最快；8/24 分别约慢 0.6%/8.8% | 保留 4 变量 |
| geometry 与 equation 粗粒度再拆分 | 同一 MIG 上 core 组从约 `5.84 s` 增到 `6.61 s`，约慢 13.2% | 淘汰；中间量全局往返大于寄存器收益 |
| 既往九段细拆 | 自然达到各 kernel >=25% occupancy，但同节点 `t=5` 慢约 4.6% | 不采用；重复 stencil/字段加载和 launch 太多 |

粗粒度 geometry/equation fission 的 NCU 也解释了失败原因：geometry kernel 仍使用 216 registers/thread，理论/实际 occupancy 仍只有 `12.5%/10.95%`。也就是说，它既支付了物化中间量的代价，又没有跨过 occupancy 档位。

## 4. Nsight Systems

固定 `t=0..1`，两版均为 420 次 RHS 调用：

| 指标 | 基线 | 阶段 1 | 变化 |
| --- | ---: | ---: | ---: |
| RHS family | 13.9386 s | 10.4985 s | -24.7% |
| Total Evolve | 17.5389 s | 14.2263 s | -18.9% |
| Total Running | 20.7719 s | 18.0850 s | -12.9% |
| `prolong3_kernel` | 2.1468 s | 2.1597 s | +0.6% |
| `cudaDeviceSynchronize` | 1,044 calls | 1,044 calls | 不变 |
| `cudaLaunchKernel` | 50,110 calls | 51,874 calls | +3.5% |

阶段 1 RHS family 的构成：

| Kernel | 总时间 | 占 RHS family |
| --- | ---: | ---: |
| advection/KO batch | 4.2459 s | 40.4% |
| core `rhs_kernel` | 3.1229 s | 29.7% |
| metric Hessian batch | 1.6729 s | 15.9% |
| shift Hessian batch | 0.8190 s | 7.8% |
| momentum x/y/z | 0.6226 s | 5.9% |
| Hamiltonian | 0.0150 s | 0.1% |

launch 增加 1,764 次，但 RHS family 净省 3.44 s，当前拆分粒度的收益显著大于 launch 开销。同步次数完全没变符合阶段范围：本轮只建立同一 stream 内的数据流，没有提前实施阶段 3 的同步改造。

## 5. Nsight Compute 与 VTune

NCU 采样同一类 `(5,5,5)` patch；“spill”是 NCU 的 compiler spill request，不等同于运行时索引 metadata/局部数组形成的 local-memory 访问：

| Kernel | regs/thread | 理论 occupancy | 实际 occupancy | No Eligible | compiler spill | DRAM | L1 / L2 hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 基线 core | 250 | 12.50% | 11.04% | 75.27% | 4 KB | 4.04% | 92.34% / 96.95% |
| 阶段 1 core | 255 | 12.50% | 11.03% | 76.90% | 38 KB | 12.85% | 82.70% / 92.02% |
| metric Hessian | 104 | 25.00% | 22.53% | 57.89% | 0 | 21.05% | 77.38% / 96.26% |
| shift Hessian | 104 | 25.00% | 22.39% | 59.38% | 0 | 19.85% | 77.65% / 98.95% |
| advection/KO | 85 | 25.00% | 23.08% | 64.37% | 0 | 9.91% | 86.85% / 96.79% |

新 kernel 已把 active warps 提高一倍附近，但 scheduler 仍有 58%--64% 周期没有 eligible warp，主要是 stencil 的 scoreboard/L1TEX 依赖。下一步若继续这条路线，重点应是跨变量共享邻域加载或分阶段 stencil scratch，而不是仅靠更小 block；block 不会减少每线程寄存器需求，也不能减少必须覆盖的 grid。

VTune 中 `cuCtxSynchronize_v2` 为 `12.648 s / 62.8% CPU time`，基线是 `15.635 s / 65.7%`。调用结构未改，因此下降来自 GPU RHS 变快；host 仍以等待 GPU 为主。`cuMemcpyHtoD_v2` 为 `0.661 s / 3.3%`，单 rank MPI 仍不是当前主瓶颈。

## 6. 端到端与正确性

未使用 profiler 的三次 `t=5` 测试如下；每次包含当前约 11 s 的 TwoPuncture 和 Python 后处理，因此是完整 program 口径：

| Run | This Program Cost | 外层 wall time | checker |
| --- | ---: | ---: | --- |
| 1 | 100.994 s | 109.937 s | PASS |
| 2 | 101.996 s | 110.902 s | PASS |
| 3 | 102.012 s | 110.959 s | PASS |
| 均值 / 样本标准差 | 101.667 / 0.583 s | 110.599 / 0.578 s | 3/3 PASS |

三个运行均匹配 golden 的 5/100 个时间点，trajectory RMS 为 0。level 0 constraint 最大值为 `Ham=0.22822817`、`Px=0.026833543`、`Py=0.010600207`、`Pz=0.017673173`，全部通过阈值。

为排除不同 MIG 实例和频率状态的偏差，又在一个 allocation 内按 `base -> candidate -> candidate -> base` 交错运行。两版都从对应 git revision 全新构建，均使用默认 `-O3`，不设置 `AMSS_CUDA_MAX_REGCOUNT`；两版命中同一个 TwoPuncture cache，以隔离 ABEGPU 路径：

| 顺序 | 版本 | This Program Cost | Total Evolve | 外层 wall time | checker |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | base | 96.807 s | 93.256 s | 105.570 s | PASS |
| 2 | candidate | 75.053 s | 71.494 s | 83.997 s | PASS |
| 3 | candidate | 75.421 s | 71.944 s | 84.367 s | PASS |
| 4 | base | 93.914 s | 90.394 s | 102.806 s | PASS |

| Paired 均值 | base | candidate | 变化 |
| --- | ---: | ---: | ---: |
| This Program Cost | 95.360 s | 75.237 s | -21.1%，`1.27x` |
| Total Evolve | 91.825 s | 71.719 s | -21.9%，`1.28x` |
| 外层 wall time | 104.188 s | 84.182 s | -19.2%，`1.24x` |

第二个 base 虽处在末尾且比第一个 base 快，仍比两个 candidate 慢约 18.5 s，说明收益不是热身或单向频率漂移造成。四次都匹配 5/100 golden 时间点、trajectory RMS 为 0，constraint 最大值相同。

## 7. 判断与后续方向

当前最值得继续处理的是两个结构问题，而不是立即进入常规微调：

1. advection/KO 已成为最大单项，占 RHS 40.4%。其 24 个字段做相同 stencil，但当前 batching 只合并控制流，没有共享邻域数据。需要比较“按 stencil 操作分批物化导数”与“按空间 tile 共享坐标/邻域”的真实流量，目标应是同时降低 local-memory 压力和 scoreboard stall。
2. core 仍是 255 registers。下一次 fission 必须比 geometry/equation 边界更细且更有针对性，例如先建立持久、紧凑的 derivative/geometry scratch，再让多个低寄存器 consumer 顺序读取；否则只会重复本轮被淘汰的全局中间量往返。

按 `14.23 s/物理时间单位` 粗略线性外推，演化本身仍约 1,423 s，离 `t=100 <=330 s` 很远，而且 AMR 后期成本并不严格线性。因此本轮是有效的阶段性降时，不是通往目标的充分方案。进入下一阶段前，应先重新设计剩余 RHS 的 scratch 布局并做小规模原型；如果无法展示 RHS 再降到约 2--3 s/单位的路径，就应重新评估 330 s 目标的可达性。

## 8. Artifacts

- 基线：`profile/gpu-nsys-20260824T071953Z-57/`、`profile/gpu-ncu-20260824T072559Z-66/`、`profile/gpu-vtune-20260824T073534Z-65/`
- 最终 Nsys：`profile/gpu-nsys-20260824T093805Z-65/`
- 最终 NCU core：`profile/gpu-ncu-20260824T091154Z-57/`
- 最终 NCU metric/shift/advection：`profile/gpu-ncu-20260824T085341Z-57/`、`profile/gpu-ncu-20260824T094052Z-64/`、`profile/gpu-ncu-20260824T094215Z-65/`
- 最终 VTune：`profile/gpu-vtune-20260824T095355Z-66/`
- 三次 benchmark：`profile/gpu-benchmark-20260824T094429Z-64/`
- 同节点交错 A/B：`profile/gpu-rhs-pair-20260824T100451Z-60/`
- 被淘汰的 geometry/equation Nsys/NCU：`profile/gpu-nsys-20260824T092501Z-61/`、`profile/gpu-ncu-20260824T093142Z-66/`

所有 GPU 测试均经 `hpc` 提交；Nsys/NCU/VTune 使用 `hpc_gpu_profile.sh`，普通三次测试使用 `hpc_gpu_benchmark.sh`，交错对照使用 `hpc_gpu_rhs_pair.sh`。profile 目录包含 build log、原始报告、程序日志和 checker 输出。
