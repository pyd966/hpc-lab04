# Lab04-CPU

为什么是 30 physical cores, 60 logical cores 啊。。。实验文档里不是说 arm 机器无 SMT 吗。

下面所有都是绑在同一个 NUMA 上运行得到的结果。

## baseline

通过在计算节点上提交 baseline 计算请求并进行初步 profile，发现 twopuncture 需要运行 286s，ABE 部分每个时间单位需要运行 43.1s，在当前时间预算下无法完成整个演化过程，最多只能推算 33 个时间单位。线性外推，baseline 需要 2020s。

到这里，我认为对端到端直接进行采样分析意义不大，我们应该对 twopuncture 和 ABE 两部分分开进行分析和优化。

## twopuncture

使用编译参数 `-O3 -g -fno-omit-frame-pointer`，进行 `perf stat` 和 `perf record` profiling。

结合 profiling 结果大概阅读一下代码，我们可以发现 twopuncture 整体工作流程是这样的：

读取输入数据，然后生成一个 twopuncture 对象，依次调用 `Solve()` 和 `Save()`。在 `Solve()` 中，会先不断调用 Newton 修正质量，最后再额外调用一次 Newton。每个 Newton 内部调用 bicgstab，先依次进行两个方向的 relax，再进行 J_times_dv。

上述整个流程都是串行执行的，最终把热点的调用关系整理如下（后面标的百分比代表时间占比）：

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

采样结果还发现 `cos()` 函数本身占用了 23.17% 的时间，因此我们可以尝试对重复的 `cos()` 计算进行缓存，或者换用更快的计算 cos 的方式。

整体能带来最大收益的，肯定是去做 OpenMP 并行化，我们需要找可以并行的区域。或者可以尝试向量化。

此外，一个显而易见的小优化是，因为我们有整整 128GiB 内存，我们可以把代码中频繁的内存 malloc/free 删掉，改成统一提前声明并且复用 buffer。事实上 profile 结果也支持这一点，我们有 6% 的 allocator 开销。

另一个可能收益比较大的优化是，修改预条件器，调整参数一类的。但是这个我不太懂有点玄学，留到最后再说。

#### 减少内存声明与释放

对于我们上述分析出的热点函数，其中有很多无意义的内存声明与释放，直接使用一个可以复用的内存空间 `TransformerWorkerSpace` 来替代。对于那些不在热点上的函数则没有进行这个优化。

最终从 286s->288s，反而变慢了，但是观察 cycles 和 instructions 都有约 2-3% 的下降，但是本次提交集群 CPU 的频率也有 2% 左右的下降，说明此次提交应该是集群波动掩盖了本次优化的效果。总之，本次优化依然被采纳。

#### 缓存三角函数值

发现只要网格尺寸固定，那么调用 cos 函数时的自变量就是一系列确定的数。因此，为了减少反复计算，我们可以在读入网格尺寸之后，立刻把它们全部计算缓存下来。主要涉及到 chebeft_zeros, fourft 这些过程。

最终从 288s->198s，有 1.4x 的提升。同时采样发现 cos 函数自身的占比从 23%->低于 0.5%，说明这个热点已经被我们完全解决。并且后续更换数学库，或者更换其他的三角函数计算方法也不再有必要。

#### 编译指令

试了一下加 `-march=native`，反而慢了一点点，但是这应该是波动。后面又尝试加了 `-Ofast`，198s->187s，变快了 1.05x 左右，但是结果不再逐字节匹配，最大误差约 4e-15，我觉得在可以接受的范围内，但是因为后面还有一整个阶段要优化，所以我这里想先不加 Ofast。等最后了再一起考虑要不要加。因此这项优化暂时没有采用。

#### OMP 并行化

重头戏。

观察热点，发现两个 LineRelax，以及 J_times_dv 特别值得看。

先看简单的 J_times_dv（以及完全类似的 F_of_v），它们都是对每个坐标单独访问、单独计算、单独存储，所以三重循环可以直接完全并行。

再看 `Derivatives_AB3`。发现其内部有依赖关系，需要分三个阶段执行。但是每个阶段内部每次枚举一行，每行之间彼此是独立的，因此可以并行。

再看两个 LineRelax。可以发现原本的代码已经将其进行红黑染色，每一步分两个 phase 完成，每个 phase 内部是独立可以并行的，phase 之间则有执行顺序。所以我们对 phase 内部进行并行即可。

最终测试发现，OMP 开 30 线程 11.399s，开 60 线程 11.091s，相比于 baseline，我们从 286s->11s，目前得到了 25.8x 的加速。

继续优化 twopuncture 目前已经没有意义，我们应该去优化 ABE 了。

## ABE

使用同样的参数进行 profiling，

```text
读入 TwoPuncture 初值与 9 层 AMR 网格
  -> 初始化变量、同步边界和诊断器
  -> Evolve
       -> RecursiveStep(level 0)
            -> Step(level)
                 -> level 0 上执行 AnalysisStuff
                 -> RK4 predictor
                 -> 3 个 RK4 corrector
                 -> 每个 RK 子步：RHS + NaN 全局检查 + halo Sync
            -> 递归推进更细层
            -> Restrict/Prolong + Sync
       -> Constraint_Out
       -> 输出/检查点（达到设定时间时）
```

这里 RecursiveStep 基本就是一个用来调用 Step 的壳子。一次 RecursiveStep 会调用 66 次 Step。目前来看最热点区域都集中在 Step 里。

profile 结果表明，最大瓶颈是 MPI 通信与等待（mca_btl_sm_poll_handle_frag 58.98%），而最 heavy 的计算 compute_rhs_bssn 只有 6.10%。

显然，这个项目完全没必要用到 MPI，我们可以使用更加轻量级的 OMP。

因此第一步应该是去掉 MPI，改用 OMP，之后大概率是优化计算部分。

接下来的演化时间只考虑 ABE 部分，并且只做 5 个时间步，这部分的 baseline 时 173s。

### MPI -> OMP

这应该是绝大多数 speedup 的来源，应该认真做一做。不过 MPI 其实已经发现很多可以并行的区域了，这部分我们基本能无痛转成 OMP，主要包括 Step 的 predictor, corrector 按 block 并行 rhs 等等，不再赘述了。

仅仅是把 MPI 转成 OMP 就已经做到 61s，得到 2.81x 优化。重新 profile 表明 MPI 热点已经完全消失，但是我们还可以从多核中榨出一些性能，需要进一步分析哪些地方可以并行。

首先是 AnalysisStuff，之前是按 block 所有工作的，那这就导致只有少数 block 有积分点有工作可做。所以我们 OMP 版本改成以点为单位并行，并且最后不再对 shellf 进行规约，而是对最终结果进行规约。

现在变成 43s，额外 1.41x。

进行 profile，并且观察代码进一步找串行区域，发现 Constraint_Out 每个时间单位都会在主线程上串行重算 rhs。所以我们改成按 block 并行，最终变成 30s，额外 1.43x。

这时 profile 结果表明，CPU 平均使用已经达到 21.45/30。分析原因发现是因为工作量不够，level 0 只有 9 个 block，很多 OMP 线程在空转，导致 CPU 使用率并不高。这意味着接下来我们需要优化 OMP 调度。

分析发现，在静态层（除了第 0 层）中我们有 24 blocks，移动层有 30 threads。

emm 似乎没什么太好优化的。因为我们有 30 threads，除非多切出一些 blocks 出来。

尝试了一下发现是负优化，所以干脆直接保留原来的版本好了。

### SIMD

上面做了这么久 OMP，多核基本也差不多了。接下来还能从 SIMD 榨出性能。

这个可以看 GCC vectorization report，发现可以对 kodis, fdderivs, fderivs 进一步做 SIMD 优化。

最终得到约 1.16x 收益。

### 内存

profile 显示 memcpy/memset 等占比不低，我们为了追求性能应该进行一些优化。

这里都是很多边边角角优化了。比方说减少不必要的内存访问（symmetry ghost 的冗余 memset, prolong3 相邻点复用），改善连续性（把 block 的多个字段放到一起），提高 cache 复用（rhs 局部 fusion）

最终测试发现，ABE 部分完整演化耗时 271s，程序总耗时 286s，已经达成了我们的目标。相比于 baseline 达到了 7.1x 的优化。