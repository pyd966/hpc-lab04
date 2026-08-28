# P1 阶段报告：导数输出初始化

日期：2026-08-24
范围：`fderivs` / `fdderivs` 输出数组初始化
基线：`cb637c6`，ABE 单进程 OpenMP，30 线程，24/30 threads，t=0..4

## 1. 为什么先做这个

当前 profile 中 `memset` 约占 6%，而 `fderivs` 和 `fdderivs` 在进入差分循环前分别对 3 个和 6 个完整三维输出数组执行 `= 0.d0`。活动 BAM 路径随后只覆盖有效 stencil 点：内部区域用四阶模板，边界 shell 用二阶模板；最外层未满足 stencil 条件的点依赖初始化保持为零。

因此先检查两个问题：

1. 能否只清零未覆盖的边界平面；
2. 如果这些点确实不会被后续使用，能否完全跳过初始化。

两种候选都使用独立的 CMake 开关，默认关闭；原始路径可以随时回退。

## 2. 第一版：只清零边界平面

新增：

- `AMSS_ENABLE_FDERIVS_BOUNDARY_INIT`
- `AMSS_ENABLE_FDDERIVS_BOUNDARY_INIT`

打开后不再对整个输出数组赋零，而是清零上界平面，以及没有 symmetry ghost 时的下界平面。四阶 interior、二阶 shell、`symmetry_bd` 和输出顺序均未改变。

在同一 HPC 作业内交错运行 `full fderivs fdderivs both both fdderivs fderivs full`，四次前半段均逐位一致且课程检查通过。由于作业在第五次写输出时遇到磁盘空间不足，初版只得到前四个完整统计样本；这不是代码运行错误。

| 版本 | Evolve (s) | 相对 full |
|---|---:|---:|
| full | 29.6223 | 基准 |
| fderivs boundary | 29.9203 | +1.0% |
| fdderivs boundary | 30.1197 | +1.7% |
| both boundary | 30.2705 | +2.2% |

汇编是关键证据：原始 `fderivs` 约 3 次 `memset@plt`，边界版约 12 次。Fortran 数组切片中的跨步平面赋值没有变成一次廉价的连续写入，而是生成多组分片清零/地址计算。减少字节数没有抵消增加的调用和非连续访问成本。

## 3. 第二版：完全跳过初始化

为了排除“薄平面写入本身”的影响，又测试了：

- `AMSS_ENABLE_FDERIVS_SKIP_INIT`
- `AMSS_ENABLE_FDDERIVS_SKIP_INIT`

它完全跳过输出数组初始化，只依靠差分循环写入有效点。脚本在每轮比较后删除非参考的 52 MB 二进制输出，避免实验工件耗尽工作目录空间。

| 版本 | Evolve (s) | 相对 full | 数值结果 |
|---|---:|---:|---|
| full | 29.4356 | 基准 | PASS |
| fderivs skip | 30.7132 | +4.3% | 逐位不一致，FAIL |
| fdderivs skip | 30.0598 | +2.1% | 逐位不一致，FAIL |
| both skip | 31.1812 | +5.9% | 逐位不一致，FAIL |

两次 full 分别为 29.3217 s 和 29.5494 s；所有 skip 版本的 course check 都失败。说明最外层/无 stencil 点并非完全死数据，至少有下游路径读取这些原本由零初始化保证的值。这个候选不应保留在生产编译入口。

## 4. 结论

P1 已经尝试了两种方向，并分别找到了失败机制：

- “薄边界清零”数值正确，但 SVE `memset` 已经非常高效，多个非连续分片写入更慢；
- “完全跳过清零”虽然理论上减少最多内存写，但破坏了下游依赖的零边界语义，不能接受。

因此默认生产路径继续使用原始整数组清零。代码中仅保留默认关闭的薄边界实验开关，作为正确性/汇编对照；skip-init 开关和实现已撤回。P1 不产生可接受的性能提升，不应把“减少 memset”简单当作后续优化目标。

下一步应转向 profile 中更大的 `memcpy`/Sync/AMR transfer 路径：那里可以减少真正的数据搬运，而不改变导数输出的边界契约。

## 5. 工件

- 初版前四轮结果：`profile/p1-derivative-init-158707-results.tsv`
- skip-init 完整摘要：`profile/p1-derivative-skip-158776-results.tsv`、`profile/p1-derivative-skip-158776-summary.tsv`
- 默认 fresh profile：`profile/abe-20260824T141547Z-14`
