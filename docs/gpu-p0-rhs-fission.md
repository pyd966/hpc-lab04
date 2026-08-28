# GPU P0：拆分 RHS 平流/耗散阶段

> 日期：2026-08-22  
> 测试平台：A100 80GB PCIe MIG `1g.10gb`，单 GPU、单 MPI rank  
> 基线：`docs/gpu-baseline-profile.md`；P0 的作业以 `0951bc3` 为源码基点并带本阶段工作树修改

## 1. 结论

P0 将 BSSN 主 RHS 中 24 个变量的 upwind advection 和 Kreiss-Oliger dissipation 拆为同一 CUDA stream 上的第二个 kernel。短窗 `t=5` 三次端到端均值从 122.492 s 降至 107.109 s，提升 **12.56%**；Nsight Systems 中两段 RHS 合计从 56.906 s 降至 44.043 s，提升 **22.60%**。三次 checker 全部通过，轨迹与 golden 前缀的 RMS 为 0。

这一轮没有彻底解决主 RHS 的寄存器压力：主 kernel 仍为 250 registers/thread，且采样 launch 有 10 KB local-memory spilling。收益来自缩短主 kernel 的指令/依赖路径，并把平流与耗散隔离到一个 85 registers/thread、无 spill、可驻留 2 blocks/SM 的 kernel。

## 2. 实现

修改集中在 `src/bssn_rhs_gpu.cu`：

1. 新增 `add_advection_dissipation()` device inline helper，保持每个变量原有的 lopsided derivative、KO dissipation 和 parity 参数。
2. 新增 `rhs_advection_dissipation_kernel()`，一次处理原来融合在 RHS 尾部的 24 个状态变量。
3. `rhs_kernel()` 保留几何量、BSSN RHS 主体和 constraint 计算；平流/耗散不再夹在 RHS 主体和 constraint 之间。
4. wrapper 在原有 `rhs_kernel` 后、同一 stream 上提交新 kernel，不增加 host/device 临时数组，也不引入显式同步。

同 stream 的提交顺序保证新 kernel 读取已经写好的 RHS，并按原顺序执行 `lopsided + kodiss` 累加。constraint 只依赖状态和主 RHS 中已计算的局部几何量，不依赖这些 RHS 累加项，因此提前到主 kernel 尾部不改变数据依赖。

## 3. 测试方法

端到端测试使用 `hpc_gpu_benchmark.sh`，严格 `-O3`，每次重新运行 TwoPuncture，不使用 cache；终点缩短为 `t=5`，用于与 baseline 的同口径趋势比较。构建不计入程序时间，每次运行后执行 checker。

Profile 使用缓存好的初值并只采集 ABEGPU：

- Nsight Systems：`t=4`，统计 GPU kernel、CUDA API、传输和 launch 数。
- Nsight Compute：采集第一个匹配 `rhs_.*kernel` 的主 RHS 与新拆分 kernel，grid `(5,5,5)`、block `(8,8,4)`。MIG 环境不允许锁定时钟，使用 `--clock-control none`。
- NCU 在采完两个 launch 后终止目标进程；正确性由完整 benchmark 和 Nsys run 验证。

## 4. 端到端性能

| Run | Program Cost (s) | Evolve (s) | Running (s) | Outer wall (s) | Check |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 106.900424 | 77.0420 | 80.4684 | 115.666 | PASS |
| 2 | 107.298256 | 77.4935 | 80.9306 | 116.244 | PASS |
| 3 | 107.126953 | 77.3750 | 80.7918 | 116.012 | PASS |
| Mean | **107.108544** | **77.3035** | **80.7303** | **115.974** | 3/3 PASS |
| Program Cost sample stddev | **0.199554** | - | - | - | - |

| 指标 | Baseline | P0 | 改善 |
| --- | ---: | ---: | ---: |
| Program Cost mean | 122.491884 s | 107.108544 s | **12.56%** |
| ABEGPU Evolve mean | 92.1168 s | 77.3035 s | **16.08%** |
| ABEGPU Running mean | 95.6112 s | 80.7303 s | **15.56%** |
| Outer wall mean | 131.478 s | 115.974 s | **11.79%** |

Program Cost 仍包含 GPU allocation 上的 TwoPuncture 与初始化，因此 Evolve 的改善更直接反映本阶段 GPU 修改。

## 5. Nsight Systems

P0 profile 的 ABEGPU Evolve/Running 为 60.577/64.202 s，checker 通过。

| Kernel | Baseline 总时间 | P0 总时间 | P0 calls | P0 平均 |
| --- | ---: | ---: | ---: | ---: |
| 原融合 `rhs_kernel` | 56.906 s | - | 1623 | 35.062 ms |
| P0 主 `rhs_kernel` | - | 28.306 s | 1623 | 17.441 ms |
| P0 advection/dissipation | - | 15.737 s | 1623 | 9.696 ms |
| P0 两段 RHS 合计 | 56.906 s | **44.043 s** | 3246 launches | 27.137 ms/调用对 |

RHS 合计减少 12.863 s，即 **22.60%**。拆分额外增加 1623 次 launch；总 launch 从 201694 增至 203317，但 launch API 时间反而由 0.907 s 降至 0.857 s，说明新增 launch 开销远小于 kernel 缩短收益。

P0 尚未处理同步：`cudaDeviceSynchronize` 仍是 **4110 次**，与 baseline 完全相同；host API 等待随 GPU 计算缩短，从 71.573 s 降至 57.998 s。H2D/D2H 的调用数和字节量也不变，因此性能变化不是由减少传输或同步造成的。

## 6. Nsight Compute

| 指标 | Baseline 融合 RHS | P0 主 RHS | P0 advection/dissipation |
| --- | ---: | ---: | ---: |
| Duration | 7.37 ms | **5.14 ms** | 1.81 ms |
| Registers/thread | 250 | **250** | **85** |
| Register block limit | 1 block/SM | 1 block/SM | **2 blocks/SM** |
| Theoretical occupancy | 12.50% | 12.50% | **25.00%** |
| Achieved occupancy | 11.04% | 11.08% | **21.86%** |
| No eligible scheduler cycles | 75.30% | 75.58% | **65.26%** |
| Eligible warps/scheduler | 0.28 | 0.27 | **0.48** |
| Local-memory spilling | - | **10 KB** | **0 B** |
| Compute throughput | 22.26% | 22.06% | 30.93% |
| Memory throughput | 28.14% | 22.52% | 55.19% |
| Branch efficiency | 99.47% | 99.42% | 99.59% |

第一个小 grid 的两段 NCU duration 合计为 6.95 ms，比 baseline 的 7.37 ms 低 5.7%。它低于 Nsys 全工作负载的 22.6% 改善，原因是 NCU 只采样第一个较小 launch，并受 metric replay 和未锁定时钟影响；端到端与 Nsys 聚合值更适合判断总体收益。

P0 的机制不是“让整个 RHS occupancy 翻倍”。主 kernel 仍被 250 个寄存器限制；真正改变的是独立 advection/dissipation kernel 的资源形态，以及主 RHS 不再执行这段长链。下一轮若继续针对 RHS，应进一步拆主 kernel 的几何中间量或减少局部数组，但这不属于当前 P1 范围。

## 7. 正确性与结论边界

三次 `t=5` checker 均通过：

- trajectory RMS：0；
- level 0 Hamiltonian 最大值：0.22822817；
- momentum x/y/z 最大值：0.026833543 / 0.010600207 / 0.017673173；
- 9 层 constraint 全部低于阈值。

短窗不能替代最终 `t=100` 验收。P0-P3 完成后统一执行正式 `t=100`，以免每个中间阶段消耗一次接近任务上限的完整运行。

## 8. 复现材料

- Benchmark：作业 `137671`，artifact `profile/gpu-benchmark-20260822T091918Z-65`
- Nsight Systems：作业 `137866`，artifact `profile/gpu-nsys-20260822T092948Z-66`
- Nsight Compute：作业 `137907`，artifact `profile/gpu-ncu-20260822T093501Z-66`

Profile 原始文件位于 gitignored 的 `profile/`，仓库提交本报告和实现源码。
