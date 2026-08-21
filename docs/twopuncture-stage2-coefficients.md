# TwoPuncture 阶段 2：预计算并缓存谱变换系数

记录日期：2026-08-21。

## 修改内容

固定谱网格下，变换中的三角函数输入只由网格尺寸和循环下标决定：

```text
Chebyshev forward: cos((Pi/n) * j * (k+0.5))
Chebyshev inverse: cos((Pi/n) * (j+0.5) * k)
Fourier:           cos/sin((Pi/M) * l * k)
```

本阶段为每个实际出现的网格尺寸建立：

- Chebyshev zeros 的 forward/inverse 两张 `n x n` cosine 表；
- Chebyshev extremes 的一张 `n x n` cosine 表；
- Fourier forward/inverse 各自的 cosine/sine 表。

表格属于 `thread_local TransformWorkspace`。当前串行运行只生成一份；后续
OpenMP 中每个 worker 第一次使用某个尺寸时生成自己的只读表，因此没有锁和共享
写竞争。以本次 `n=50`、`Nphi=26` 计，每个线程只需几十 KiB。

变换仍按原有 `j/k/l` 顺序逐项累加，没有改成 FFT，也没有改变求和顺序。缓存
消除的是每次变换内重复调用 libm，而不是省略任何乘加。

## 测量方法

最终作业号 `128687`，节点为 TaiShan-v120。编译参数和输入保持：

```text
-O3 -g -fno-omit-frame-pointer
nA=50, nB=50, nphi=26
```

`perf stat` 和 `perf record` 各完整运行一次。原始结果位于：

```text
profile/twopuncture-20260821T040007Z-14/
```

## 性能结果

| 指标 | 阶段 1 | 阶段 2 | 变化 |
| --- | ---: | ---: | ---: |
| wall time | 288.720 s | 198.918 s | -31.10% |
| cycles | 808.832 B | 569.511 B | -29.59% |
| instructions | 2067.696 B | 1233.737 B | -40.33% |
| IPC | 2.56 | 2.17 | -15.23% |
| 测量平均频率 | 2.804 GHz | 2.865 GHz | +2.18% |
| L1D miss rate | 2.11% | 2.86% | +0.75 pp |
| LLC miss rate | 0.08% | 0.13% | +0.05 pp |
| dTLB miss rate | 0.29% | 0.32% | +0.03 pp |
| branch miss rate | 1.38% | 1.06% | -0.32 pp |

相对原始 baseline 286.505 s，累计 wall-time 加速为 30.57%，即约 `1.44x`。
cycles 与 instructions 同时大幅下降，说明这不是节点频率造成的假象。

IPC 下降并不代表优化失败：原先大量时间在 libm 的紧凑计算代码里，IPC 较高；
现在改成读取 coefficient table 后，总指令数大幅减少，但 load 比例上升，
L1D miss rate 也从 2.11% 升到 2.86%。绝对 LLC miss rate 仍只有 0.13%，
系数表没有把程序变成 DRAM 带宽瓶颈。

## Profile 结果

| 调用路径/函数 | 阶段 1 | 阶段 2 |
| --- | ---: | ---: |
| `relax()` inclusive | 66.85% | 93.36% |
| `LineRelax_be()` inclusive | 37.96% | 51.71% |
| `LineRelax_al()` inclusive | 28.87% | 41.62% |
| `ThomasAlgorithm()` self | 13.61% | 19.50% |
| `J_times_dv()` inclusive | 28.09% | 4.34% |
| `Derivatives_AB3()` inclusive | 26.86% | 2.51% |
| `cos()` self | 23.65% | <0.5% |

这些是相对占比，不能把 `relax` 的 93.36% 误解为它变慢了。总 cycles 已下降
29.59%，而谱导数被大幅压缩后，原本基本未变的 line relaxation 自然占据更大
比例。新的绝对主热点非常集中：预条件器中的两类 line solve 和 Thomas 递推。

## 正确性与失败重试

第一次实现（作业 `128641`）把原来的 `fac=1./M; Pi_fac=Pi*fac` 写成数学等价的
`Pi/M`。浮点舍入并不保证等价，细小差异被迭代过程放大，虽然最终参数相同，
`Ansorg.psid` 却出现大量末位差异，因此没有接受该结果。

最终版本严格保留原表达式的运算顺序：

```cpp
double const fac = 1. / M;
double const Pi_fac = Pi * fac;
```

验证结果：

- 所有 BiCGSTAB iteration 和打印残差与阶段 1 一致；
- `puncture_parameters_new.txt` 逐字节一致；
- `Ansorg.psid` 去掉首行时间后逐字节一致；
- 最终 bare mass 和 ADM mass 与 baseline 一致。

阶段 3 应在这个已消除 libm 热点的版本上测试架构编译参数；当前最值得关注的是
line relaxation 的间接 stencil 访问和 Thomas 的串行递推。
