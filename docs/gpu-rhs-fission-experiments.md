# RHS register-live-range fission experiments

## 目标与测量方法

本轮只针对 GPU RHS，不改 TwoPuncture、MPI 配置或物理公式。目标是确认“拆分”是否真的缩短 kernel 内的寄存器 live range，而不是只增加 kernel 数量。所有版本都使用 A100 MIG 1g.10gb、`sm_80`、`block=(8,8,4)`、单 MPI rank；t=1 的 Nsys 运行先用 `check.sh` 做正确性检查，t=5 benchmark 使用相同输入并检查最终输出。

作为参照，原始单体 `rhs_kernel` 的 NCU 结果为 255 registers/thread、理论 occupancy 12.5%、达到 occupancy 11.05%、No Eligible 76.88%，并有约 38 KiB compiler spill。当前版本没有依靠降低 block 大小或强制 `maxrregcount`，先验证数据流拆分的收益。

## 实验 A：完整 RHS 多阶段拆分

实验 A 使用已经验证过的九阶段 source fission，将原 RHS 按数据依赖排成：

`geometry -> ricci_a -> gamma seeds -> beta/gamma -> evolution -> Ricci connection -> source metric/chi/gamma/lapse/trace/Aij/gauge -> advection -> constraints`。

这一步证明了公式和 scratch 依赖可以拆开，但没有充分缩短两个热点的内部 live range。实验 A 的 t=1 Nsys RHS kernel 总时间为 **12.372124 s**。NCU 代表性结果如下：

| kernel | regs/thread | 理论 occupancy | 达到 occupancy | No Eligible |
|---|---:|---:|---:|---:|
| `rhs_geometry` | 72 | 37.5% | 31.90% | 67.02% |
| `rhs_gam{ x,y,z }_seed` | 160 | 12.5% | 10.78--10.91% | 80.55--80.71% |
| `rhs_beta_gamma` | 177 | 12.5% | 10.89% | 75.65% |
| `rhs_evolution` | 104 | 25.0% | 21.37% | 58.38% |
| `rhs_ricci_connection_{diag,offdiag}` | 239/241 | 12.5% | 10.89/10.93% | 78.63/78.42% |

所有 sampled kernel 的 NCU local-memory spill 为 0；但 gamma seed、beta/gamma 和 Ricci 仍然是高寄存器热点。A 的结果说明“按 RHS 的大功能段拆分”还不够。

## 实验 B：真正的 live-range 拆分

实验 B 保留 A 的阶段顺序，只对两个高寄存器区做 producer/consumer 拆分。

1. `rhs_gamma_derivatives_kernel`（`src/bssn_rhs_gpu.cu:534`）只计算 `Lap` 和 `trK` 的一阶导数，写入六个已有 scratch 数组。三个 gamma seed kernel 读取对应的六个标量，不再各自重算导数，也不再同时持有完整导数局部变量。
2. `rhs_beta_gamma_prepare_kernel`（`src/bssn_rhs_gpu.cu:141`）集中计算三个 shift 的 Hessian、三个 contracted beta Laplacian，以及 contracted conformal Gamma，写入 scratch；`rhs_beta_gamma_kernel`（`:194`）只读取摘要值并完成最终 RHS 更新。
3. 计算顺序保持在同一个 CUDA stream 中，因而没有引入错误的跨 stream 读写；每个 producer/consumer 之间由同一 stream 的顺序保证依赖。

NCU 结果直接验证了寄存器下降：

| kernel | regs/thread | 理论 occupancy | 达到 occupancy | No Eligible |
|---|---:|---:|---:|---:|
| 原 A `rhs_gam{ x,y,z }_seed` | 160 | 12.5% | 10.78--10.91% | 约 80.6% |
| B `rhs_gamma_derivatives` | 72 | 37.5% | 32.32% | 65.29% |
| B `rhs_gam{ x,y,z }_seed` | 78 | 37.5% | 31.13--31.50% | 约 95.3% |
| 原 A `rhs_beta_gamma` | 177 | 12.5% | 10.89% | 75.65% |
| B `rhs_beta_gamma_prepare` | 104 | 25.0% | 21.46% | 60.43% |
| B `rhs_beta_gamma` consumer | 32 | 100.0% | 71.40% | 95.56% |

B 的 Nsys RHS 总时间为 **11.379526 s**，比 A 减少约 **8.0%**；t=5 端到端 benchmark 为 **104.729061 +/- 0.692102 s**（3 次，三次 checker 均 PASS）。这证明 B 是本轮唯一明确实现了“降低 RHS 寄存器占用”的拆分。

## 实验 C：按 Ricci 输出分量拆分

实验 C 将两个 Ricci connection kernel 改为六个 compile-time component kernel（3 个 diagonal + 3 个 off-diagonal），希望每个 kernel 只保留一个 Ricci 分量。

结果不满足目标：diag 三个实例为 **170/183/183 regs/thread**，offdiag 三个实例均为 **201 regs/thread**；六个 kernel 的理论 occupancy 都只有 **12.5%**，达到 occupancy 约 **10.79--10.87%**，No Eligible **78.80--79.15%**。NCU 没有报告 spill，但寄存器并没有进入 25% occupancy 区间。t=5 端到端为 **110.114760 s**（PASS），比 B 慢约 **5.1%**。原因是每个分量仍然先加载完整的 18 个 Christoffel 分量并构造完整中间量；拆的是输出，不是导致寄存器峰值的计算依赖，同时还增加了六次 launch 和全局 scratch 读写。

## 最终选择与后续方向

当前源码保留实验 B，已恢复 Ricci 为两个 kernel（`src/bssn_rhs_gpu.cu:228`、`:321`，launch 在 `:1506--1507`）。实验 C 的组件拆分不保留。B 的收益是明确的寄存器 live-range 降低和约 8% 的 A->B RHS profile 改善，但它不是相对于当前 Stage3 端到端版本的最终加速：现有 Stage3 t=5 记录为 100.018310 s，而 B 为 104.729061 s。这个差距说明 launch/scratch 成本和其他 kernel 仍需单独优化，不能只追求 occupancy 数字。

下一步若继续压 Ricci，应按计算依赖拆分：先生成可复用的 `dGamma`/Christoffel contraction 摘要，再由 diagonal/off-diagonal consumer 完成公式；不能再次只按输出分量复制完整公式。每次拆分都必须同时检查 scratch 写入成本、同一 stream 的依赖和端到端时间。

## 可复现实验产物

- 实验 A：`profile/gpu-nsys-20260825T050722Z-66`、`profile/gpu-ncu-20260825T050924Z-67`
- 实验 B：`profile/gpu-nsys-20260825T052202Z-56`、`profile/gpu-ncu-20260825T052417Z-57`、`profile/gpu-benchmark-20260825T052553Z-65`
- 实验 C：`profile/gpu-nsys-20260825T053827Z-65`、`profile/gpu-ncu-20260825T054445Z-67`、`profile/gpu-benchmark-20260825T054845Z-57`
- 参照 Stage3：`profile/gpu-benchmark-20260825T031112Z-57`、`profile/gpu-nsys-20260825T030934Z-66`
- 最终 B 源码复验：`profile/gpu-nsys-20260825T162415Z-60`（编译成功，checker PASS）
