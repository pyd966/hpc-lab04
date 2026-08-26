# GPU 阶段 4：AMR prolong/restrict 跨变量批处理

## 结论

本阶段保留最终版本。对同一 coarse/fine transfer segment，将变量列表改为一个连续的 device descriptor 表，并把变量映射到 `grid.y`，一次 launch 同时完成该 segment 的多个变量 prolong 或 restrict。随后只对 batch kernel 使用 host 预计算的对齐整数，去掉每个输出点重复的几何数组、浮点除法和边界索引构造。

固定 A100 80GB PCIe 的 `1g.10gb` MIG、单 MPI rank、GPU OpenMP 关闭和 TwoPuncture cache，t=5 三次 `This Program Cost` 为 `89.150199 +/- 0.301396 s`。阶段 3 保留版本为 `93.258103 +/- 0.073732 s`，端到端改善 `4.40%`。三次 trajectory RMS 均为 0，constraints 和 `FINAL: PASS` 全部通过。

## 依赖与边界

本改动只覆盖 `PACK` 路径的 AMR prolong/restrict，且要求变量数大于 1。coarse/fine 时间对齐、ghost exchange、MPI staging、UNPACK 以及同层 copy 仍保持原顺序；这些边界不能跨越。每个 segment 的 source field、destination packed range 和 descriptor range 都不相交，因此可以在一个二维 launch 中并行处理。batch 完成后仍等待实际访问的 stream，调用方接口和 buffer 生命周期不变。

## 实现

- `Prolong3BatchVar` 保存 source/destination device pointer 和三维 SoA/parity 元数据；`GPUManager` 持久化 descriptor device buffer，按需扩容。
- `Parallel::gpu_data_packer()` 先遍历 segment 和变量，按原变量顺序计算 packed offset，再发射一个 `prolong3_batch_kernel` 或 `restrict3_batch_kernel`。单变量、UNPACK 和非 AMR 路径自动回退旧实现。
- batch kernel 使用 `block=(256,1,1)`、`grid=(ceil(points/256), nvars, 1)`；因此计算点数没有减少，只减少了 launch 和重复的 host dispatch。
- batch prolong 的 host launcher 预计算 `CD/FD/base/lbc/lbf`。专用 device helper 使用标量 extents/alignment，保留原 5th-order interpolation、SoA symmetry 和 column-major index；legacy kernel 未改动。
- descriptor 采用一次同步 H2D copy。逐 segment `cudaMemcpyAsync` 实验在 t=5 反而为 `96.423 s`，因此没有合入。

## 结果与对照

| 版本 | prolong calls / time | restrict calls / time | AMR 合计 |
| --- | ---: | ---: | ---: |
| 阶段 3 | 11,955 / 3.129587 s | 2,391 / 0.653454 s | 3.783041 s |
| 仅跨变量 batch | 530 / 1.572992 s | 106 / 0.511528 s | 2.084520 s |
| 最终 batch + 几何预计算 | 530 / 0.815499 s | 106 / 0.511638 s | 1.327137 s |

最终 AMR kernel 合计比阶段 3 少 `64.92%`，launch 数量少 `95.6%`。这满足阶段计划中对 2.796 s AMR 成本至少减半的门槛；主要收益来自 geometry/index 计算和 kernel launch，而不是减少数值工作。

## Profile

最终 Nsight Systems：`profile/gpu-nsys-20260826T092137Z-66/nsys.nsys-rep`，作业 168276，t=0..1。`prolong3_batch_kernel` 为 530 次、`815.499 ms`；`restrict3_batch_kernel` 为 106 次、`511.638 ms`。RHS kernel 次数和时间与阶段 3 基本一致，说明改动没有污染 RHS 热路径。最终 profile 中 `cudaDeviceSynchronize` 仍为 215 次；本阶段没有改变同步语义。

最终 Nsight Compute：`profile/gpu-ncu-stage4-precomputed/full.ncu-rep`，直接 profile 一次 batch prolong launch（作业 168458）。launch 为 256 threads/block、54 x 7 blocks；93 registers/thread，理论 occupancy 25%，实际 occupancy 22.17%，local spill requests 为 0，kernel duration 553.12 us。可见 occupancy 已在门槛附近且没有 spill；继续压寄存器不是本阶段的主要收益点。

NCU 报告同时显示 scheduler eligible warps 为 0.63、active warps 为 3.56，说明单个短 segment 仍有尾部空闲；跨 segment 融合会受 coarse/fine 和 buffer 生命周期限制，不能简单把全部 transfer 合成一个 kernel。

## 正确性与可复现产物

最终 t=1 Nsys 和 t=1 benchmark 均为 `FINAL: PASS`；t=5 三次 checker 均通过，trajectory RMS 为 0，level-0 constraint maxima 为 Ham=0.025628463、Px=0.012645773、Py=0.0127579、Pz=0.025120159。

- 阶段 3 benchmark：`profile/gpu-benchmark-20260826T033210Z-61`
- 阶段 4 最终 benchmark：`profile/gpu-benchmark-20260826T093400Z-65`
- 阶段 4 Nsys：`profile/gpu-nsys-20260826T092137Z-66`
- 阶段 4 NCU：`profile/gpu-ncu-stage4-precomputed/full.ncu-rep`

## 后续建议

阶段 4 已完成 AMR transfer 的结构性低风险优化，但端到端仍主要由 RHS 和同步/host memcpy 主导。下一步应优先按真实 producer-consumer 关系减少 RHS 的单点工作量，或在阶段 5 对重复的 RK/boundary 小 kernel 做批处理；不要继续单独调 batch kernel 的 occupancy。完整 t=100 仍需最终整合后重新测量，当前 t=5 比例不能替代长程验收。
