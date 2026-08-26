# RHS Ricci dataflow fission

## 结论

本阶段保留。核心改动不是继续按输出分量拆 Ricci，而是把两个 Ricci consumer 重复执行的 `Gamx/Gamy/Gamz` 九个一阶导数抽成共同数据流。最终将 producer 融入已存在的 `rhs_beta_gamma_kernel` 尾部，随后由 `rhs_ricci_connection_diag_kernel` 和 `rhs_ricci_connection_offdiag_kernel` 消费。

- Ricci 两个 consumer：约 `240 registers/thread -> 126 registers/thread`，理论 occupancy 从 `12.5%` 提升到 `25%`。
- Nsys 中 Ricci 导数加两个 connection consumer 的每次 RHS 合计时间从 `2.191 ms` 降到 `1.084 ms`，下降 `50.5%`。
- 包含原 `beta_gamma` 的完整局部链从 `2.380 ms` 降到 `1.277 ms`，下降 `46.3%`。
- 三次 `t=5` 端到端均值从 `100.296712 s` 降到 `97.693319 s`，提升 `2.60%`，三次 checker 均 PASS。

## 依赖分析

原始 diag/offdiag kernel 都执行相同的三次 `d_fderivs_point`，分别对 `Gamx`、`Gamy`、`Gamz` 计算九个空间导数。它们只读 BSSN 状态，不依赖 `rhs_evolution_kernel` 的输出；两个 Ricci kernel 对这些导数的消费也都是只读。

已有临时数组在 `rhs_beta_gamma_kernel` 返回后已经完成旧值消费；`rhs_evolution_kernel` 不读取这些数组，而后续 source kernel 会重新覆盖它们。因此九个导数可以安全映射到现有槽位，不增加显存分配：

| 导数 | 临时槽 | 导数 | 临时槽 |
| --- | --- | --- | --- |
| `dGamxx` | `ham_Res` | `dGamyx` | `movz_Res` |
| `dGamxy` | `movx_Res` | `dGamyy` | `Gmx_Res` |
| `dGamxz` | `movy_Res` | `dGamyz` | `Gmy_Res` |
| `dGamzx` | `Gmz_Res` | `dGamzy` | `Ayy_rhs` |
| `dGamzz` | `Ayz_rhs` |  |  |

所有 launch 仍在同一 CUDA stream。producer 写入、独立的 evolution、两个 Ricci consumer 之间由 stream 顺序保证可见性，不需要新增 event 或 device synchronize。

## 实现

1. 在 `rhs_beta_gamma_kernel` 完成原有 Gamma RHS 写回后计算九个 Gamma 一阶导数，并写入上述 scratch。
2. diag/offdiag kernel 删除各自三次重复 stencil，直接读取 scratch；读取表达式按 Rxx/Ryy/Rzz 或 Rxy/Rxz/Ryz 的使用位置展开，避免重新建立九个长生命周期局部变量。
3. 没有改变 Ricci 公式、浮点运算顺序中的 contraction 部分、block/grid、stream、边界 parity 或物理输出。

## 两组实验

硬件均为 A100 MIG 1g.10gb，`sm_80`，`block=(8,8,4)`。

### A：独立 derivative producer

先用独立 kernel 验证数据流边界。Nsys 共观察 1623 次对应 launch：

| kernel | 平均时间/launch |
| --- | ---: |
| derivative producer | `0.581865 ms` |
| Ricci diag | `0.262318 ms` |
| Ricci offdiag | `0.240174 ms` |
| 三者合计 | `1.084357 ms` |

上一提交的 Ricci diag/offdiag 分别为 `1.092017/1.098640 ms`，合计 `2.190657 ms`。实验 A 因而减少 `50.5%`，并以 `t=4` checker 的 trajectory RMS 0、constraints PASS 验证正确性。

### B：producer 融入 beta_gamma

producer 与 `beta_gamma` 尾部没有 live-data 冲突。融合后 1623 次 `beta_gamma` 总时间为 `1.256346 s`；实验 A 的 `beta_gamma + producer` 为 `1.255487 s`，差异仅 `0.07%`。两个 Ricci consumer 也只在噪声范围变化。`t=4` Total Evolve 为 `54.573 s`，独立 producer 为 `54.392 s`，不能据此宣称融合有额外速度收益。

最终仍保留融合形态，因为它不增加 kernel 时间，并且每次 RHS 少一个 launch；报告中的主要收益来自数据流 fission，而非这一步 fusion。

## Nsight Compute

最终 NCU 对同一粗层 `grid=(5,5,5)` 的结果：

| kernel | regs/thread | theoretical occ. | achieved occ. | No Eligible | DRAM throughput | spill requests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fused beta_gamma/producer | 72 | 37.50% | 33.17% | 72.79% | 56.39% | 0 |
| Ricci diag | 126 | 25.00% | 22.00% | 91.27% | 72.51% | 0 |
| Ricci offdiag | 126 | 25.00% | 22.18% | 88.76% | 79.57% | 0 |

改动前 diag/offdiag 分别为 `239/241 registers/thread`，理论 occupancy 均为 `12.5%`，实际为 `10.89%/10.93%`。本阶段把两个 consumer 的寄存器约减半，并达到此前要求的 25% 理论 occupancy。

`No Eligible` 百分比上升不代表本轮变慢：重复 stencil 和大量算术被移走后，剩余 consumer 更短且更偏向全局读依赖，绝对 kernel 时间已经显著下降。NCU 同时显示 DRAM throughput 达到 `72.51%/79.57%`，所以目前不适合把 18 个 lowered-Christoffel 再物化到全局 scratch；那会新增 18 次写和两个 consumer 的 36 次读，用更多内存流量换已经足够的 occupancy。

## 端到端

| run | This Program Cost | checker |
| --- | ---: | --- |
| 1 | `97.694569 s` | PASS |
| 2 | `97.977948 s` | PASS |
| 3 | `97.407439 s` | PASS |
| mean +/- sample stddev | `97.693319 +/- 0.285257 s` | 3/3 PASS |

上一提交为 `100.296712 +/- 0.088852 s`，因此端到端下降 `2.603393 s`，即 `2.60%`。三次 trajectory RMS 均为 0，所有 9 层 constraints 均通过。

三次 evolution-only 均值从 `70.332267 s` 降到 `68.012667 s`，下降 `3.30%`。沿用短窗的“固定开销 + 线性 evolution”模型：

```text
fixed ~= 97.693319 - 68.012667 = 29.680652 s
evolution/unit ~= 68.012667 / 5 = 13.602533 s
t=100 extrapolation ~= 29.680652 + 100 * 13.602533 = 1389.934 s
```

即约 `23 min 10 s`。这是开发期乐观外推，后期 AMR 成本并不严格线性，不能替代完整 `t=100`；它仍远高于 `330 s` 目标。

## 后续判断

本阶段后，继续以 occupancy 为唯一目标拆 Ricci 的收益已经很低。若再次处理 Ricci，优先方向应是重新组织 contraction 的加载/累加顺序，减少 consumer 的 global-memory scoreboard 与 LG throttle；只有能避免全局物化中间量时才值得实验。下一阶段应先重新做全局 Nsys hotspot 排名，再决定是否转向同步执行图或 AMR 跨变量批处理。

## Artifacts

- 独立 producer Nsys：`profile/gpu-nsys-20260826T021304Z-65`
- 独立 producer NCU：`profile/gpu-ncu-20260826T021617Z-65`
- 融合版 Nsys：`profile/gpu-nsys-20260826T022152Z-64`
- 最终 NCU：`profile/gpu-ncu-20260826T022528Z-64`
- 最终三次 benchmark：`profile/gpu-benchmark-20260826T022748Z-65`

