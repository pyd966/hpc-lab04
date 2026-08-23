# 阶段一（fdderivs SIMD）优化报告

日期：2026-08-23

本阶段只处理 ABE CPU 路径中 `fdderivs` 的规则内点循环。没有修改
`compute_rhs_bssn` 的公式、OpenMP block 划分、同步策略、编译优化级别或
`lopsidediff`。下一阶段的任务划分优化尚未开始。

## 1. 优化前证据

优化前基线为提交 `8ad31d1`，HPC 作业 `145839`，profile 目录为：

`profile/abe-20260823T112529Z-1/`

配置与候选版本相同：单进程 OpenMP，30 个线程，`OMP_PLACES=cores`、
`OMP_PROC_BIND=close`，静态层 24 个 block/线程，移动层 30 个 block/线程，
演化区间为 `t=0..4`，ABE 使用 `-O3 -g -fno-omit-frame-pointer`。

基线的主要结果：

| 指标 | 基线 |
|---|---:|
| Evolve 时间（perf stat） | 39.8851 s |
| Evolve 时间（perf record） | 39.6456 s |
| 平均 CPU 使用 | 16.786 / 30 |
| IPC | 1.81 |
| L1D miss | 2.88% |
| LLC miss | 48.02% |
| dTLB miss | 2.40% |

`perf record` 的 flat profile 中，`fdderivs_` 占 8.03%。源码行热点包括
`diff_new.f90:574`（1.99%）和 `diff_new.f90:582`（1.82%）。这些行对应
混合二阶导数的四阶 stencil。编译器对原始循环报告了边界分支造成的
unsupported control flow，不能把整个循环直接向量化。

## 2. 代码修改

`fdderivs` 的现有有效分支要求 `i/j/k` 三个方向都距离边界至少两点才使用
四阶公式，否则使用二阶公式。修改位于 `src/diff_new.f90:480` 附近：

1. 预先计算 `ibegin..iend`、`jbegin..jend`、`kbegin..kend`，表示三维规则
   内点区域。
2. 对规则内点单独执行六个导数的四阶 stencil，沿连续的 `i` 方向添加
   `!$omp simd`。这个循环内部没有边界判断。
3. 对规则内点之外的薄边界壳保留二阶公式和原来的 symmetry ghost-cell
   条件。输出数组此前统一清零，因此不满足二阶条件的点仍保持零值。
4. 增加 CMake 选项 `AMSS_ENABLE_FDDERIVS_SIMD`，默认开启，仅对 CPU `ABE`
   target 定义 `AMSS_FDDERIVS_SIMD`。关闭该选项时仍编译原始循环，便于复现
   和回退。

这个拆分没有改变四阶/二阶的选择条件，也没有改变浮点表达式的运算顺序；
它只把原来在每个网格点执行的边界判断移出规则内点循环。

## 3. 正确性

候选作业为 `146062`，profile 目录为：

`profile/abe-20260823T115307Z-1/`

候选的 stat 和 record 两次运行产生的四个输出文件逐位一致。课程检查结果：

```text
Trajectory RMS: 0 (0.000000%)
Constraints: PASS
FINAL: PASS
```

本轮只运行了前 4 个时间单位，因此检查报告明确说明是匹配 golden 的前缀
（4/100），这与本阶段短 profile 的设计一致。

## 4. 性能结果

| 指标 | 基线 | fdderivs SIMD | 变化 |
|---|---:|---:|---:|
| Evolve（perf stat） | 39.8851 s | 38.7123 s | -2.94% |
| Evolve（perf record） | 39.6456 s | 38.3733 s | -3.21% |
| perf wall time（stat） | 46.4943 s | 45.2237 s | -2.73% |
| 平均 CPU 使用 | 16.786 | 17.005 | +1.30% |
| `fdderivs_` self samples | 8.03% | 4.13% | -48.6% |

两次作业落在同一 TaiShan-v120 集群的不同 NUMA 节点（基线 node 3，候选
node 2），因此不能把 2.94% 当作严格的同节点 A/B 精确值；不过两次的 CPU
上限、频率范围、绑核和输入一致，且候选运行频率略低（2.854 GHz 对比
2.861 GHz），所以方向和数量级是可信的。后续如需发布最终数字，应再做同一
节点交错 A/B。

## 5. Profile 变化

候选 flat profile 的主要热点为：

| 部分 | 基线 | 候选 |
|---|---:|---:|
| `compute_rhs_bssn_` | 47.67% | 49.83% |
| `__memcpy_sve` | 9.17% | 9.68% |
| `fdderivs_` | 8.03% | 4.13% |
| `lopsided_` | 6.95% | 7.28% |
| `__memset_sve_zva64` | 5.96% | 6.43% |
| `prolong3_` | 4.42% | 4.27% |

`fdderivs` 的绝对样本明显下降，剩余时间被其他热点重新占比，说明修改
确实命中了目标函数，而不是只改变了采样分布。候选的 LLC miss 为 48.28%，
与基线 48.02% 基本相同；L1D miss 从 2.88% 增到 3.52%，dTLB miss 从
2.40% 增到 2.83%。这说明 stencil 的主要限制仍是访存，SIMD 消除了部分
分支/标量指令，但没有改善数据布局或缓存复用。

候选 IPC 为 1.61，低于基线 1.81；同时总指令数从约 `4.05e12` 降到
`3.53e12`。在相近频率下，减少的指令数和 `fdderivs` 样本下降与耗时下降
一致。IPC 单独下降并不表示优化失败，因为该阶段减少了计算指令，而 LLC
miss 基本不变，程序整体仍然不是纯计算瓶颈。

编译器的 `-fopt-info-vec-all` 诊断确认新内点循环（当前源代码
`diff_new.f90:498`）生成了 `16 byte vectors`。同时报告了复杂数组访问，
这是 Fortran 三维 stencil 的地址计算成本；它没有阻止该循环生成向量版本。

## 6. 结论和边界

本阶段达到预期目标：只改 `fdderivs`，通过了数值检查，函数自身样本约减半，
端到端 Evolve 获得约 3% 的稳定方向收益。收益没有达到更高水平的主要原因是：

- `fdderivs` 只占约 8% 的总时间，Amdahl 上限有限；
- stencil 读取多个邻居，整体受缓存/内存带宽约束；

因此本阶段不继续扩大到 `lopsidediff` 或 `compute_rhs_bssn`。下一阶段再单独
研究 block 任务的加权划分、schedule 和 barrier 等并行负载均衡问题，保持
本阶段代码作为可比较基线。

本阶段不包含第二个任务划分优化，也不包含后续的内存布局优化；这些实验需要
分别 profile，避免把不同原因的收益混在一起。
