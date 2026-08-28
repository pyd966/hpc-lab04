# TwoPuncture 并行区域与工具链选择说明

记录日期：2026-08-21。本文补充说明如何从 profile 和源码中找到 OpenMP 并行
区域、这些函数究竟有多热，以及现阶段是否值得更换数学库或尝试编译参数。

## 1. 并行区域是怎样找到的

并行区域不是根据“这个循环看起来很长”猜出来的，而是经过两步筛选：先用动态
profile 找值得处理的代码，再对这些代码做数据依赖检查。只有同时满足“耗时足够大”
和“跨迭代没有不能消除的依赖”，才把它列为 OpenMP 候选。

### 1.1 先用 profile 缩小范围

本次基线以 `-O3 -g -fno-omit-frame-pointer` 编译，在 HPC 节点上分别运行：

```text
perf stat -d -d ./TwoPunctureABE
perf record -F 99 --call-graph fp ./TwoPunctureABE
perf report
perf annotate
```

这些工具回答不同问题：

1. `perf stat` 判断程序总体是单线程、计算、访存还是分支问题。本次平均只使用
   `0.999 CPU`，没有 MPI 通信；IPC 2.60，LLC miss 0.05%，所以最显眼的问题是
   大量串行计算，而不是进程通信或主存带宽。
2. `perf report` 的调用图找到从 `Solve()` 向下最重的调用路径。例如
   `bicgstab -> relax` 占 66.85%，`bicgstab -> J_times_dv -> Derivatives_AB3`
   占 25.86%。
3. `perf report --no-children` 区分函数自身时间和子函数时间。例如
   `Derivatives_AB3()` 自身循环控制很薄，但它调用的 `cos()` 和变换很热，因此它
   适合作为组织并行工作的入口，而不是要优化的单条算术指令。
4. `perf annotate` 把采样映射到源码行。最热的两行是
   `TwoPunctures.C:2000` 和 `TwoPunctures.C:2245`，都是 line relaxation 中
   `JFD[Ic][m] * dv[col]` 的稀疏 stencil 更新。这说明并行化必须覆盖许多条 line
   solve，不能只处理外层 Newton 流程。

`-g` 的作用是保留函数、文件和行号；仍使用 `-O3`，所以得到的是优化后程序的真实
热点，而不是 `-O0` 调试版本的热点。由于内联、常量传播和指令调度，一条采样到的
源码行可能代表附近多条语句，因此行级结果用于定位，不应被解释为精确计时器。

### 1.2 再对候选循环做数据依赖检查

对每一个热点循环，检查以下五件事：

1. **工作单元是什么**：一个网格点、一条一维谱线，还是一条三对角线？
2. **每个工作单元读什么、写什么**：把共享数组的下标写出来。
3. **不同单元是否写同一位置**：若 `W(a) intersect W(b)` 非空，会发生写写竞争。
4. **一个单元是否读取另一个正在写的位置**：若 `W(a) intersect R(b)` 非空，
   就有顺序依赖；需要重排、着色或 barrier，而不能直接 `parallel for`。
5. **临时变量是否真正私有**：即使输出元素互不重叠，共用 `p/dp/U/values` 等
   scratch 也会造成竞争，必须改为每线程一份。

这里的核心判断可概括为：对同一并行阶段内任意两个不同工作单元 `a,b`，应满足

```text
W(a) intersect W(b) = empty
W(a) intersect R(b) = empty
W(b) intersect R(a) = empty
```

允许的例外是显式 reduction，以及被阶段 barrier 隔开的生产者和消费者。

### 1.3 `Derivatives_AB3()` 的判定过程

这个函数包含三个阶段：

| 阶段 | 一个工作单元 | 数量 | 读取 | 写入 |
| --- | --- | ---: | --- | --- |
| A 导数 | 固定 `(k,j)` 的 A 方向线 | `26*50=1300` | `v.d0` | 该线的 `v.d1,d11` |
| B 导数 | 固定 `(k,i)` 的 B 方向线 | `26*50=1300` | `v.d0,d1` | 该线的 `v.d2,d22,d12` |
| phi 导数 | 固定 `(i,j)` 的 phi 方向线 | `50*50=2500` | `v.d0,d1,d2` | 该线的 `v.d3,d33,d13,d23` |

同一阶段内，两条不同的线写入不重叠的元素，所以线与线之间可以分给不同线程。
但是 B 阶段会读取 A 阶段生成的 `d1`，phi 阶段又会读取 A/B 阶段的 `d1,d2`，因此
正确结构是：

```text
parallel region
    omp for: all A lines
    barrier
    omp for: all B lines
    barrier
    omp for: all phi lines
```

每条线内部的 Chebyshev/Fourier 变换仍按原顺序执行。原函数只有一套
`p/dp/d2p/q/dq/r/dr/indx`，并行后必须给每个线程一套 scratch，否则不同线虽然输出
不冲突，临时数组仍会互相覆盖。

### 1.4 `F_of_v()` 和 `J_times_dv()` 的判定过程

这两个函数都先完成 `Derivatives_AB3()`，随后遍历 65,000 个 `(i,j,k)` 点。每个点
只读取已经完成的导数和物理参数，并写自己对应的 `F/u` 或 `Jdv` 元素。不同点的输出
下标不同，因此点循环可以使用 `omp for collapse(3) schedule(static)`。

需要处理的是当前在循环外复用的 `U/dU/values`：它们必须放在线程私有 workspace
中。导数结束和点循环开始之间还要有 barrier。这里任务规则、每点工作量和网格边界
基本一致，`schedule(static)` 比 dynamic 调度更合适，额外调度开销也更小。

### 1.5 `relax()` 为什么不能直接并行最外层循环

`relax()` 是 Gauss-Seidel 风格的 line relaxation。`LineRelax_be()` 固定 `i,k`
并沿 `j` 更新整条 B 线，`LineRelax_al()` 固定 `j,k` 并沿 `i` 更新整条 A 线。
每条线会读取 stencil 邻点的最新 `dv`，所以任意并行所有线会改变算法并产生数据竞争。

源码已经把 `k`、`i` 和 `j` 按奇偶顺序访问。这提供了可并行的“颜色”：

- 同一 `k` 奇偶组中，两条相同 `i` 奇偶的 B 线相差至少 2；当前 stencil 只跨到
  `i +/- 1`，所以它们不读取对方本阶段写入的线。
- 两条相同 `j` 奇偶的 A 线同理。
- 同一 `k` 奇偶组内的平面相差至少 2，而 stencil 只跨到 `k +/- 1`。

因此可以并行同一个颜色中的 `(k,line)`，但颜色间必须按原算法顺序保留 barrier：

```text
even k: B-even-i -> B-odd-i -> A-odd-j -> A-even-j
odd  k: B-even-i -> B-odd-i -> A-odd-j -> A-even-j
```

每个箭头都是 barrier。一个颜色约有 312--325 条独立线，足以分给 30 或 60 个
worker。不能把 barrier 去掉，也不能让每个线程独立执行一份完整 `relax()`。

### 1.6 明确不能直接并行的部分

- `ThomasAlgorithm()` 的 LU、前代和回代都有 `i` 对 `i-1` 或 `i+1` 的递推依赖。
  单条长度 50 的线内部保持串行，线程级并行放在线与线之间。
- `SetMatrix_JFD()` 当前按“列”构造矩阵。不同列会向同一个 `row` 的
  `ncols[row]`、`cols[row]`、`Matrix[row]` 追加数据，直接并行列循环会发生竞争。
  它需要先改成按行构造，或使用线程局部暂存后合并。
- BiCGSTAB 的 `alpha/beta/omega/rho` 有算法顺序，不能并行不同 iteration；但每个
  iteration 内长度为 65,000 的向量更新、点积和范数可以用 `omp for`/`reduction`。

## 2. 这些热点究竟有多热

本次 `perf stat` 总时间为 286.50 秒。下表把采样比例换算成近似秒数，便于形成直观
认识。调用路径占比包含子函数，自身占比不包含子函数，两者不能相加；例如
`ThomasAlgorithm` 已经包含在 `LineRelax` 和 `relax` 的调用路径中。

| 函数或路径 | 类型 | 占比 | 约合时间 | 含义 |
| --- | --- | ---: | ---: | --- |
| `Solve -> bicgstab` | 调用路径 | 97.12% | 278.25 s | 几乎全部求解时间 |
| `bicgstab -> relax` | 调用路径 | 66.85% | 191.53 s | 最大优化对象，预条件器 |
| `relax -> LineRelax_be` | 调用路径 | 37.35% | 107.01 s | B 方向所有工作及 Thomas |
| `LineRelax_be` | 自身 | 27.17% | 77.84 s | 稀疏扫描、索引和数组处理 |
| `relax -> LineRelax_al` | 调用路径 | 29.15% | 83.52 s | A 方向所有工作及 Thomas |
| `LineRelax_al` | 自身 | 19.09% | 54.69 s | A 方向稀疏扫描等 |
| `ThomasAlgorithm` | 调用路径 | 17.47% | 50.05 s | 三对角解和其中的分配 |
| `ThomasAlgorithm` | 自身 | 13.60% | 38.96 s | 串行递推本身 |
| `bicgstab -> J_times_dv` | 调用路径 | 27.03% | 77.44 s | 真实 Jacobian-vector product |
| `J_times_dv -> Derivatives_AB3` | 调用路径 | 25.86% | 74.09 s | 谱导数及其变换 |
| `libm::__cos` | 自身 | 23.17% | 66.38 s | 重复计算固定变换系数 |
| `fourft` 两版本合计 | 调用路径 | 约 5.51% | 15.79 s | phi 方向直接 Fourier 变换 |
| allocator | 自身下限 | 至少 6.07% | 至少 17.39 s | `malloc/free/new/delete` |
| `JFD_times_dv` | 调用路径 | 2.12% | 6.07 s | 构造有限差分 Jacobian 的点计算 |

另外，`perf annotate` 显示全程序最热源码行为：

| 源码行 | 全程序 cycles 占比 | 内容 |
| --- | ---: | --- |
| `TwoPunctures.C:2000` | 18.76% | B 线的非线内 stencil 累加 |
| `TwoPunctures.C:2245` | 15.69% | A 线的对应累加 |
| `TwoPunctures.C:2323` | 4.95% | Thomas 回代 |
| `TwoPunctures.C:2307` | 3.70% | Thomas LU 递推 |
| `TwoPunctures.C:821` | 2.23% | Fourier 逆变换内层 |

这意味着建议的并行区域覆盖的不是边角代码：同色 line relaxation 覆盖约三分之二
的运行路径，谱线和点循环覆盖约四分之一。即使理想地让所有其它代码无限快，仅
`relax` 保持串行，程序加速仍不可能超过约 `1/0.6685 = 1.50x`；所以最终必须处理
line relaxation，而不能只并行容易改的 `F_of_v()`。

这些百分比是 cycles 采样估计。约 27,999 个样本且没有丢样，足以判断十几个百分点
的主热点，但不应把 0.1% 级差异当成精确结论。修改后必须重新 profile，因为热点会
转移。

## 3. 是否需要更换数学库

### 3.1 现在不应先替换 `libm`

当前二进制链接 glibc `libm.so.6`，`__cos` 自身占 23.17%。这看起来像数学库问题，
但源码中的 Chebyshev 角度只依赖固定的 `n,j,k`。对 `n=50`，正逆两张 cosine 表
只有约 40 KiB。初始化时用同一个 `cos()` 生成一次，后续查表，可以把热点循环中的
这些 `cos()` 调用全部消除。

因此两种方案解决的问题不同：

```text
替换 libm:       每一次重复 cos 算得更快
预计算系数表:    相同 cos 只算一次，之后不再调用数学库
```

后者更直接、可控，也更容易维持原来的 double 值。Arm Performance Libraries 的
`libamath` 确实提供 AArch64、Neon 和 SVE 优化的数学函数，适合做对照实验，但它
不应排在预计算之前。官方功能说明见
[Arm Performance Libraries](https://developer.arm.com/tools-and-software/arm-performance-libraries)。

如果课程环境已经安装 ArmPL，且只允许非常小的链接修改，可以测一次
glibc libm 与 libamath；必须重新检查 BiCGSTAB 迭代、残差和最终输出，因为不同
数学库不保证最后一位完全相同。不建议为了这一个实验在提交脚本中临时下载安装库，
否则可移植性和评测环境可用性会成为新风险。

### 3.2 FFTW 有潜力，但不是第一优先级

`fourft()` 是手写的 `O(N^2)` 直接变换，`N=26`；FFTW 可以提供 FFT/DCT 实现，
Chebyshev 的正逆变换也可映射到带缩放和符号调整的 DCT-II/DCT-III。FFTW 的实偶
变换采用未归一化约定，因此替换时必须显式核对缩放，见
[FFTW real even/odd transforms](https://fftw.org/fftw3_doc/Real-even_002fodd-DFTs-_0028cosine_002fsine-transforms_0029.html)。

但当前 `fourft` 调用路径只有约 5.51%，即使整段变成零成本，单独带来的理论加速
上限也只有约 1.058x。长度 26/50 很短，plan 和调用开销也不能忽略。合理顺序是：

1. 先做系数表和 OpenMP 外层线并行；
2. 再把很多条线批处理，避免每条短线单独进入库；
3. 最后比较手写表驱动变换、FFTW batch 和矩阵乘形式。

### 3.3 BLAS 只有在重构为批量矩阵乘后才值得

预计算后，每条 Chebyshev 变换本质上是固定 `50x50` 矩阵乘长度 50 向量。单条
`DGEMV` 很小，函数调用和多线程启动成本可能抵消收益；但把 1,300 或 2,500 条线
整理成矩阵，可以形成 BLAS-3 `DGEMM`，这时 ArmPL/OpenBLAS 才更可能发挥 SVE 和
cache blocking 的优势。

这属于第二阶段设计，不是简单换链接选项。还必须避免 OpenMP 外层和多线程 BLAS
同时开线程造成 oversubscription：要么外层 OpenMP + 单线程 BLAS，要么把整个批量
矩阵交给多线程 BLAS，不能默认两层都使用全部核心。

## 4. 是否值得尝试编译参数

值得，但应做小规模、可解释的参数矩阵，而不是排列组合所有 flag。当前 baseline
只有通用 `-O3`，而 profile 节点支持 256-bit SVE，因此最有依据的第一项是目标
架构参数。

### 4.1 本地静态向量化诊断

在与项目一致的 AArch64 GCC 14.2 上，使用

```text
-O3 -march=native -fopt-info-vec-all
```

只编译 `TwoPunctures.C` 得到以下结果。这是编译器诊断，不是 HPC 性能测量：

| 循环 | 诊断 |
| --- | --- |
| BiCGSTAB 初始残差 `:1516` | 使用可变长度向量，即 SVE |
| LineRelax 解向量回写 `:2024/:2263` | 使用可变长度向量 |
| Chebyshev `cos` 内层 `:667/:680` | 未向量化，`cos` 调用不受支持 |
| Fourier `cos` 内层 `:821` | 未向量化 |
| 稀疏 stencil `:1996/:2241` | 未向量化，间接访问和控制流复杂 |
| Thomas `:2304/:2316/:2322` | 未向量化，递推依赖 |

`-Ofast -march=native` 的第二次诊断仍没有把上述 `cos`、稀疏扫描和 Thomas 热循环
向量化。因此不能把 `-Ofast` 当成“自动解决 23% cos 热点”的办法。

### 4.2 推荐测试的参数层级

| 优先级 | 配置 | 预期与风险 |
| --- | --- | --- |
| 基准 | `-O3` | 当前 release baseline |
| 高 | `-O3 -mcpu=native` | 允许并针对作业节点 ISA 调度；AArch64 GCC 中，在没有其它 `-mcpu/-mtune` 时，`-march=native` 等价地选择本机 CPU，见 [GCC AArch64 options](https://gcc.gnu.org/onlinedocs/gcc/AArch64-Options.html) |
| 中 | `-O3 -mcpu=native -flto` | 跨 `TwoPunctureABE.C`/`TwoPunctures.C` 内联和全程序优化；主要热点本来就在同一源文件，收益可能有限。GCC 要求编译和链接都使用 `-flto`，见 [GCC optimize options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html) |
| 中后期 | PGO：`-fprofile-generate` 后 `-fprofile-use` | 输入固定、分支分布稳定，可能改善布局和内联；训练必须使用代表性完整输入，流程比单个 flag 复杂 |
| 低且高风险 | `-Ofast`/`-ffast-math` | 允许重排浮点、忽略部分 IEEE 语义，可能改变点积、残差和收敛；只能作为独立实验 |

`-O3` 已经开启普通 loop/SLP vectorization，所以重复添加 `-ftree-vectorize` 没有意义。
`-funroll-loops`、prefetch 等孤立 flag 可能使代码变大或变慢，应在编译器报告指出具体
循环后单独测试。可以用 GCC 官方的
[`-fopt-info-vec-optimized/-missed`](https://gcc.gnu.org/onlinedocs/gcc/Developer-Options.html)
查看编译器实际做了什么。

profile 版本保留 `-g -fno-omit-frame-pointer` 以得到可靠调用栈；正式计时版本可去掉
`-fno-omit-frame-pointer`。`-g` 主要增加调试信息和文件体积，本身通常不改变热路径
指令，不必为了计时强制移除。

### 4.3 参数实验怎样才有效

建议只保留四个构建点：

```text
A: -O3
B: -O3 -mcpu=native
C: -O3 -mcpu=native -flto
D: -Ofast -mcpu=native       # 单独标为数值风险实验
```

每个点至少在 HPC 上运行 3 次，报告中使用中位数，同时记录：

- TwoPuncture 总时间和 CPU time；
- 六次 BiCGSTAB 各自 iteration、最终 residual；
- `puncture_parameters_new.txt` 和忽略时间行后的 `Ansorg.psid` 差异；
- `perf stat` 的 instructions、cycles、IPC；
- 实际编译命令和 CPU affinity。

不要把编译参数实验与 OpenMP、预计算表或 `NRELAX` 修改混在同一次提交，否则无法
判断收益来自哪里。`-mcpu=native` 生成的二进制也只应在相同 CPU/ISA 的评测节点使用；
若编译节点和运行节点不同，应改成明确且由评测节点支持的架构参数。

## 5. 推荐结论

对当前版本，优化优先级应是：

1. 复用 scratch，消除热点内存分配；
2. 预计算 Chebyshev/Fourier 系数；
3. 用 OpenMP 并行独立谱线、点循环和同色 line relaxation，并正确绑核；
4. 测试 `-O3 -mcpu=native`，再单独测试 LTO；
5. 热点转移后再评估批量 BLAS/FFTW、PGO；
6. 最后才测试 ArmPL `libamath` 或 `-Ofast`，并做完整数值验证。

理由不是数学库和编译器不重要，而是目前最大的限制来自源码结构：重复计算常量、
串行处理独立线、递推依赖和间接 stencil。库和编译参数可以放大结构良好的代码，
但不能自动消除这些结构问题。
