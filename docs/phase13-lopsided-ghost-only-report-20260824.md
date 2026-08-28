# P7 阶段报告：lopsided ghost-only 内存复制实验

日期：2026-08-24  
阶段：P7，尝试减少 lopsided stencil 的完整 ghost 数组复制  
结论：数值正确的修正版仍无收益，默认关闭

## 1. 动机

当前 profile 中 `lopsided_core_` 约占 8.4%，每次 `lopsided` 先调用
`symmetry_bd(3,...)`，把整个输入场复制到带有 `-2:0` ghost 的自动数组，再由深部
SIMD stencil 读取这个副本。理论上深部点的所有索引都是正的，可以直接读取原始 `f`，只
为边界 shell 和低侧对称面准备少量 ghost，从而减少一次完整数组 copy。

这里要区分两类 memcpy：正式 profile 的调用图显示约 4.7% 的 memcpy 来自 OpenMP
same-level Sync 的 `copy_`，那是跨 block 边界的必要数据传输；本阶段只修改
`lopsided -> symmetry_bd` 这条约占 0.4--0.6% 的 memcpy，不能把 Sync copy 一起删掉。

## 2. 实现思路

新增默认关闭的 `AMSS_ENABLE_LOPSIDEDIFF_GHOST_ONLY`：

1. 新增 `symmetry_bd_partial`，只复制低、高边界 slab 和低侧 ghost 平面，不复制未被
   shell 读取的深部正区域；小数组仍回退到原始 full copy。
2. `lopsided_core` 增加原始输入 `f` 参数。开启候选时，深部 SIMD 区域改为直接读取
   `f`；边界 shell 保留原来的 `fh`、一侧模板、二阶模板和对称反射顺序。
3. `shell_only=.true.` 的 lopsided2 实验路径继续使用原始 full ghost，避免把本阶段和
   已拒绝的双字段 batch 实验耦合。
4. 首次用 4 层 slab 时，单步即出现 NaN。检查 stencil 后发现 shell 点 `i=3` 的
   一侧模板会读取 `i+3=6`，因此修正版把 slab 扩为 6 层。修正版没有改变浮点运算
   顺序，只改变输入数组的来源。

独立数组测试证明：在 ghost 和边界 slab 的合法读取区域内，`symmetry_bd_partial` 与
原始 `symmetry_bd` 逐元素一致；初版 NaN 的原因是 slab 覆盖不足，而不是反射符号错误。

## 3. HPC A/B 结果

第一次作业 159946 用 4 层 slab，OFF 正常但 ON 在第一个时间步产生 NaN，课程检查
失败；该作业在确认失败后取消。

修正版作业 159970（节点 `zjusct-920b-1`），运行顺序 `OFF, ON, ON, OFF`，30 个
绑定 OpenMP worker，静态/移动层 24/30，`dynamic,1`，测量 `t=0..4`：

| 版本 | Evolve 平均 (s) | Total 平均 (s) | 平均 CPU | 结果 |
|---|---:|---:|---:|---|
| OFF | 29.7272 | 32.1065 | 23.192 | 两次 PASS |
| ON（6 层） | 29.7944 | 32.2758 | 23.170 | 两次 PASS，逐位一致 |

ON 比 OFF 慢约 **0.23%**。两次 ON 分别为 29.8672 s 和 29.7216 s，差异方向与
运行顺序和节点噪声相符，不能视为稳定加速；平均 CPU 也没有改善。

## 4. 正式 profile 及失败原因

候选 profile：`profile/abe-20260824T222812Z-13`，作业 159982；`perf record` 捕获
74766 个样本且无丢样本。与 P6 ON 基线 `profile/abe-20260824T215011Z-14` 比较：

| 符号 | P6 基线 | ghost-only ON |
|---|---:|---:|
| `compute_rhs_bssn_` | 51.28% | 49.76% |
| `lopsided_core_` | 8.42% | 7.84% |
| `symmetry_bd_` | 1.33% | 1.02% |
| `symmetry_bd_partial_` | 0 | 0.36% |
| `__memcpy_sve` | 9.46% | 11.10% |
| `__memset_sve_zva64` | 5.27% | 5.45% |

调用图进一步显示，lopsided 路径中的 partial 准备和 memcpy 约占 1.16%，已经高于
基线 full `symmetry_bd` 的对应成本；Sync 路径的 `copy_` 仍然约占 5.57%，没有因为
本实验改变。

根本原因是当前课程网格的 block 很薄：典型形状为 `40x40x20`、`48x60x24`、
`48x96x24`。为了保持一侧模板的正确性，6 层 slab 在 z 方向覆盖了 20/24 个点中的
大部分区域；把这些 slab 分成多次 Fortran section assignment，比一次连续 full copy
更难让 libc 使用高效的连续 SVE copy。`lopsided_core` 节省的深部数组读取不足以抵消
partial 准备的写入和地址计算。

候选硬件计数器为 IPC 1.45、L1D miss 4.16%、LLC miss 48.77%、dTLB miss 3.61%、
branch miss 0.42%、平均活跃 CPU 23.18。没有出现同步或分支异常，说明回归确实来自
内存准备路径，而不是线程调度波动。

## 5. 为什么不继续扩大 slab 或强行启用

把 slab 扩大到 6 层已经是正确性所需的最小安全宽度；继续扩大只会接近甚至超过 full
copy 的字节数。按 block 形状估算，只有三维尺寸都远大于 12 层时 partial 才可能减少
足够多的内存，但当前固定评测网格的 z 维不满足这个条件。把 full copy 改成“根据 block
尺寸自适应”只能避免回归，无法带来收益；而要再进一步，必须让 shell 对正索引直接读
`f`、只对负索引做符号映射，这会把大量边界分支引入一个本来已经很重的 stencil，风险和
复杂度都明显上升。

因此本阶段结论是：ghost-only 的一般想法成立，但不适合当前 block 几何；代码和脚本
保留为默认关闭的实验，不进入生产配置。后续应优先优化真实占比更大的 Sync/AMR copy
调度或 RHS 代数/数据布局，而不是继续切分这个已经很薄的 ghost 数组。

## 6. 可复现实验

- A/B：[`hpc_abe_lopsided_ghost_sweep.sh`](../hpc_abe_lopsided_ghost_sweep.sh)
- profile：[`hpc_abe_profile.sh`](../hpc_abe_profile.sh)，设置
  `AMSS_ENABLE_LOPSIDEDIFF_GHOST_ONLY=ON`
- 初次错误 profile：`profile/abe-lopsided-ghost-20260824T221629Z-15`
- 修正版 A/B：`profile/abe-lopsided-ghost-20260824T222406Z-13`
- 正式 profile：`profile/abe-20260824T222812Z-13`
