# TwoPuncture baseline profile 与优化建议

记录日期：2026-08-19。本文只讨论 CPU/GPU 两条路径共用的
`TwoPunctureABE` 初值求解阶段，不包含后续 `ABE` 演化。

## 1. 测量方法

测量在 `lab4` 的 TaiShan-v120 计算节点上完成，作业号为 `117235`。编译选项为：

```text
-O3 -g -fno-omit-frame-pointer
```

输入与 baseline 相同：`nA=50`、`nB=50`、`nphi=26`，共 65000 个谱配置点；
Newton tolerance 为 `5e-12`，ADM mass tolerance 为 `1e-8`。作业完整运行两次
求解器：一次由 `perf stat -d -d` 统计硬件事件，一次由
`perf record -F 99 --call-graph fp` 采样调用栈。第二次得到 27999 个 cycles 样本，
没有丢样。

正式端到端 baseline 日志中的 TwoPuncture 阶段约为 294.6 秒，本次带调试信息的
独立 `perf stat` 运行是 286.5 秒，相差约 2.8%，属于不同作业间的正常波动；两者
数量级和收敛路径一致。`-g` 提供符号与源码行，不会像 `-O0` 那样关闭优化。

profile 输出的 `puncture_parameters_new.txt` 与正式 baseline 逐字节相同；
`Ansorg.psid` 只有首行生成时间不同，去掉时间行后内容相同。两次 profile 都收敛到：

```text
mp = 0.576976, mm = 0.378578
Mp = 0.598837, Mm = 0.401163
total ADM mass = 0.983557
```

原始数据保存在本地 `profile/twopuncture-20260819T071854Z-$/`，不提交到 Git。
可使用仓库根目录的 `hpc_twopuncture_profile.sh` 重复采集。

## 2. 程序流程

从外层看，TwoPuncture 的工作并不复杂：

1. Python 根据 `AMSS_NCKU_Input.py` 生成 `TwoPunctureinput.par`。
2. `TwoPunctureABE.C` 读取双黑洞质量、位置、动量、谱网格和误差阈值，构造
   `TwoPunctures`，依次调用 `Solve()` 和 `Save()`。
3. 因输入中的 bare mass 为负值，`Solve()` 先进入外层质量校准循环。每轮用一次
   Newton 更新场变量，再根据得到的 ADM mass 修正两个 bare mass，直到满足
   `1e-8` 误差。本次运行做了 5 轮质量校准。
4. 质量校准后，再做一次最终 Newton 求解。本次一共调用了 6 次 BiCGSTAB，合计
   88 个 BiCGSTAB iteration。
5. 每个 Newton step 都在求解线性化方程 `J * dv = F`。代码使用 BiCGSTAB；其中
   `J_times_dv()` 计算矩阵向量乘，`relax()` 是 line-relaxation 预条件器。
6. `J_times_dv()` 先调用 `Derivatives_AB3()`。后者在 A/B 方向做 Chebyshev 变换，
   在 phi 方向做 Fourier 变换，再把导数转换到物理坐标并计算线性化方程。
7. `Save()` 写出 `Ansorg.psid` 和 puncture 参数，供后续 ABE/ABEGPU 读取。

关键代码入口：

- `src/TwoPunctureABE.C:83`：参数读取和主程序；
- `src/TwoPunctures.C:71`：`Solve()`；
- `src/TwoPunctures.C:1228`：Newton；
- `src/TwoPunctures.C:1470`：BiCGSTAB；
- `src/TwoPunctures.C:1848`：矩阵向量乘；
- `src/TwoPunctures.C:1923`：line relaxation；
- `src/TwoPunctures.C:1122`：谱导数。

## 3. 当前并行方式

TwoPuncture 当前实际上是串行程序：

- 源码没有 `MPI_Init`、MPI 通信或 collective；它只是因为构建系统而链接 MPI。
- Python 直接运行 `./TwoPunctureABE`，没有通过 `mpiexec` 启动。
- baseline 没有启用 OpenMP，源码中也没有 OpenMP parallel region。
- `perf stat` 显示平均使用 `0.999 CPU`，0 次 CPU migration。

因此这里不存在“MPI rank 之间是否均衡”的问题，因为只有一个进程、一个线程，
甚至没有 MPI rank。`-c60` 分配提供的其余 CPU 在 TwoPuncture 阶段处于空闲状态。
这也是它与 ABE 最大的结构差异：ABE 已经是 MPI 程序，而 TwoPuncture 尚未并行化。

## 4. 总体硬件指标

`perf stat` 测得单次求解时间为 286.50 秒：

| 指标 | 结果 | 判断 |
| --- | ---: | --- |
| CPU 利用率 | 0.999 CPU | 完全串行 |
| 平均频率 | 2.886 GHz | 接近节点 2.9 GHz 上限 |
| IPC | 2.60 | 不低，执行流水线总体有足够工作 |
| branch miss | 1.22% | 正常，没有明显分支预测问题 |
| L1D load miss | 2.51% | 有一定局部性损失，但不异常 |
| LLC load miss | 0.05% | 很低，不像 DRAM 带宽瓶颈 |
| dTLB load miss | 0.35% | 不高，不是主要问题 |
| L1I / iTLB miss | 接近 0 | 指令侧没有压力 |
| 峰值内存 | 约 81 MB | 容量远不是限制 |

由于同时采集的硬件事件多于 PMU counter 数量，各事件运行覆盖率约为 57%--64%，
`perf` 已按覆盖时间缩放计数。比例仍适合做粗粒度判断，但不应把绝对事件数当作精确
的内存带宽测量。

结论是：TwoPuncture 不是通信/同步瓶颈，也没有证据表明它受主存带宽限制。
当前主要问题是单核串行计算，以及在热点循环中重复执行大量三角函数、间接数组访问
和短生命周期内存分配。`LineRelax` 的不规则访问会造成部分 L1 miss，但数据大多在
到达 LLC 之前已经命中，因此更接近单核计算和访问延迟问题，而不是 DRAM 吞吐问题。

## 5. 函数与调用路径热点

下面的“调用路径占比”包含该函数调用的子函数，“自身占比”只表示函数本身。二者
不能直接相加。

| 调用路径 | 调用路径占比 | 自身占比 | 主要工作 |
| --- | ---: | ---: | --- |
| `Solve -> bicgstab` | 97.12% | 0.08% | 整个线性求解 |
| `bicgstab -> relax` | 66.85% | 0.02% | line-relaxation 预条件器 |
| `relax -> LineRelax_be` | 37.35% | 27.17% | B 方向线更新 |
| `relax -> LineRelax_al` | 29.15% | 19.09% | A 方向线更新 |
| `ThomasAlgorithm` | 17.47% | 13.60% | 三对角方程前向/回代 |
| `bicgstab -> J_times_dv` | 27.03% | 0.39% | 线性化矩阵向量乘 |
| `J_times_dv -> Derivatives_AB3` | 25.86% | 约 0.08% | 谱导数 |
| `cos()` | 23.17% | 23.17% | Chebyshev 变换内重复计算角度 |
| 两个 `fourft()` 版本合计 | 约 5.51% | 约 3.87% | phi 方向 Fourier 变换 |
| `malloc/free` | - | 至少 6.07% | 热点内反复创建临时数组 |

最热的具体源码行是：

| 源码行 | cycles 占比 | 含义 |
| --- | ---: | --- |
| `TwoPunctures.C:2000` | 18.76% | B 方向 line relaxation 的稀疏 stencil 更新 |
| `TwoPunctures.C:2245` | 15.69% | A 方向对应更新 |
| `TwoPunctures.C:2323` | 4.95% | Thomas 回代 |
| `TwoPunctures.C:2307` | 3.70% | Thomas LU 分解 |
| `TwoPunctures.C:821` | 2.23% | Fourier 变换内层计算 |

这条调用链解释了几乎全部运行时间：

```text
Solve
  -> Newton
    -> bicgstab                         97.12%
      -> relax                          66.85%
        -> LineRelax_be / LineRelax_al
          -> ThomasAlgorithm
      -> J_times_dv                     27.03%
        -> Derivatives_AB3              25.86%
          -> Chebyshev/Fourier transforms
```

`NRELAX` 当前固定为 200。每个 BiCGSTAB iteration 至少对一个搜索方向执行 200 次
完整 relaxation，通常还会对第二个方向再执行 200 次。每次完整 relaxation 又遍历
所有 A/B 线。因此当前实现会进行数千万次短 line solve。

每次 `LineRelax_be/al()` 都 `new[]` 五个长度约 50 的数组；其内部调用的
`ThomasAlgorithm()` 又 `new[]` 四个数组，结束后全部释放。如此细粒度的 heap 操作
没有必要，也与 profile 中至少 6% 的 allocator 开销吻合。

Chebyshev 变换的角度只由固定的 `n=50` 和循环下标决定，但当前在每次变换中重新调用
`cos()`。这就是 `cos()` 单独占到 23.17% 的原因。它是重复计算，不是物理模型要求
每次重新求值。

## 6. 推荐优化顺序

### 第一优先级：复用 line solve 临时空间

把 `diag/e/f/b/x` 和 Thomas 所需的 `l/u/d/y` 提升为可复用 workspace，或让 Thomas
在已有数组上原位计算，避免每条线的 9 次 `new[]` 和 9 次 `delete[]`。这是局部、低
风险修改，不改变迭代顺序。仅按 allocator 自身样本估计，上限约为 6%，同时还能
减少 allocator 元数据访问和 cache 污染。

### 第二优先级：预计算谱变换系数

对固定的 `nA/nB/nphi` 预计算 Chebyshev 的 cosine 表和 Fourier 的 sine/cosine 表，
后续变换直接做乘加。`cos()` 自身占 23.17%，理论收益空间明显大于小规模循环微调。
第一版应使用同一 `libm` 在初始化阶段生成表，以尽量保持数值一致；之后再评估 FFTW
一类 DCT/FFT 实现。每次修改都要比较输出和 BiCGSTAB iteration 数，不能只看时间。

### 第三优先级：为共享内存并行整理循环

不建议先引入 MPI：问题只有 65000 个点，谱变换带有全局方向操作，MPI 拆分会增加
通信和实现复杂度。OpenMP 更适合 CPU 与 GPU-host 两条路径共用。

可先并行 `Derivatives_AB3()` 中相互独立的多条一维变换，每个线程使用自己的临时
buffer。line relaxation 的收益潜力更大，但它具有 Gauss-Seidel/red-black 更新依赖，
不能直接在最外层套一个 `omp parallel for`。需要证明同一颜色内的线彼此独立，并在
颜色之间保留 barrier；还应把并行区放在大量线更新之外，避免为长度 50 的单条线
反复创建线程。

### 第四优先级：改善稀疏数据布局

`JFD` 和 `cols` 当前是 `double**`/`int**`，每行单独分配；热点行通过 `cols[Ic][m]`
间接访问 `dv`。可改成固定 `StencilSize=19` 的连续二维存储，并把 `ncols`、index 和
不随 Newton iteration 变化的网格几何量连续保存。这会降低指针追踪和 L1 miss，也
更利于编译器生成 Neon/SVE 向量代码。

### 第五优先级：再评估算法参数

预条件器占 66.85%，根因之一是每个方向固定做 `NRELAX=200` 次。可以系统测试较小
的 `NRELAX` 或更强的预条件器，但不能直接把 200 改小：单次 iteration 会变快，
BiCGSTAB iteration 数和收敛稳定性也可能恶化。应记录完整求解时间、6 次线性求解
各自的 iteration 数、最终 residual 和输出误差，再选择参数。

最后才建议测试 `-march=native`/SVE。当前大量时间位于 `cos()`、短三对角递推和间接
访问，这些代码不会仅靠一个编译选项自动获得理想向量化。先去掉重复三角函数、整理
连续数据和独立循环，再看编译器 vectorization report，会更可靠。

## 7. 下一轮建议实验

建议每次只改变一个因素，按以下顺序做小步验证：

1. workspace 复用：确认输出不变，观察 malloc/free 是否从热点消失；
2. cosine/Fourier 表：确认 `cos()` 占比和总时间下降；
3. 连续化 `JFD/cols`：观察 L1D miss 与 line-relaxation 时间；
4. OpenMP 并行谱变换，再谨慎并行同色 line relaxation；
5. 对 `NRELAX` 做参数扫描，而不是凭经验直接修改。

每轮至少保留三项证据：端到端 TwoPuncture 时间、收敛 iteration/residual、与 baseline
输出的数值比较。这样可以区分真正加速与“少算了但结果已变化”。
