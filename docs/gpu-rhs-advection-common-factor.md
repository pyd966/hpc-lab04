# RHS advection common-factor optimization

## 目标

本阶段针对 `rhs_advection_kernel` 做低风险优化。上一版 B 的 Nsys profile 中，advection 是 RHS 最大耗时 kernel，时间为 **4.021589 s**，占 RHS profile 总时间约 35%；NCU 显示它使用 85 registers/thread，理论 occupancy 25.0%，实际 occupancy 21.31%。

## 修改内容

`d_lopsided_point` 之前在每个变量调用中重复完成以下工作：

- 读取当前点的 `betax/betay/betaz`；
- 读取坐标并计算 `dX/dY/dZ` 和三阶 stencil 系数；
- 计算边界范围和 symmetry 对应的 `imin/jmin/kmin`；
- 接收但不使用 `f_rhs`。

现在由 `rhs_advection_kernel` 在每个 thread 中只计算一次这些公共量，并通过 `RHS_ADVECTION_LOPSIDED` 宏传递给 17 个变量的 stencil 调用。`d_lopsided_point` 只保留 stencil 和 symmetry boundary lookup，不再读取 shift 数组或坐标数组。修改位置：

- `src/bssn_rhs_gpu.cu:1202--1224`：公共量预计算和调用宏；
- `src/lopsidediff.h:7`：helper 新接口；
- `src/lopsidediff_gpu.cu:7`：helper 删除重复初始化。

stencil 系数、变量处理顺序、symmetry 规则和 KO dissipation 都没有改变；没有增加 kernel，也没有改变 stream 或 scratch 数据依赖。

## Profile 结果

硬件为 A100 MIG 1g.10gb，`sm_80`，`block=(8,8,4)`。

| 指标 | 上一版 B | 本阶段 | 变化 |
|---|---:|---:|---:|
| advection Nsys kernel time | 4.021589 s | 3.225269 s | -19.8% |
| registers/thread | 85 | 66 | -22.4% |
| theoretical occupancy | 25.0% | 37.5% | +12.5 percentage points |
| achieved occupancy | 21.31% | 32.51% | +11.20 percentage points |
| No Eligible | 65.17% | 62.27% | -2.90 percentage points |

NCU 没有显示 compiler spill。t=1 Nsys correctness 为 `FINAL: PASS`，轨迹 RMS 为 0。

## 端到端结果

同一 t=5 配置运行 3 次，均通过 checker：

```text
100.336780 s  PASS
100.194882 s  PASS
100.358473 s  PASS
mean = 100.296712 s
sample stddev = 0.088852 s
```

上一版 B 为 `104.729061 +/- 0.692102 s`，因此本阶段相对 B 提升约 **4.23%**。与之前 Stage3 的 `100.018310 s` 记录相比仍慢约 **0.28%**，说明 advection 之外还有其它端到端开销，不能把本阶段的收益解释为已经达到最终最佳。

## 结论与下一步

这次公共量提取同时降低了寄存器和 advection 时间，应保留。下一步可继续检查 advection 的 KO stencil 是否能复用相同的坐标/边界公共量；但不应直接拆成独立 kernel，因为这会增加 RHS 全局内存读写。之后再处理 Ricci 时，应按 contraction/dataflow 拆分，而不是再次按输出分量复制完整公式。

Profile artifacts：

- Nsys：`profile/gpu-nsys-20260825T172034Z-64`
- NCU：`profile/gpu-ncu-20260825T172314Z-64`
- benchmark：`profile/gpu-benchmark-20260825T172447Z-64`

