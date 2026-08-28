# Lab4-gpu report

首先我们直接沿用 cpu 部分优化的 twopuncture。因为 cpu 核数减小了，所以 twopuncture 在当前配置下需要 30s 跑完。也许会成为最后的瓶颈，但是我们还是先说 ABEGPU 部分该如何优化。

## baseline

依旧只能跑 t=5 短测，一共花费 122s，在 evolve 部分花费 92s。

进行 nsys，发现 rhs_kernel 占比 77.3%，接下来是 prolong3_kernel, restrict3_kernel。

显然我们应该把重点放在 rhs_kernel 上。

对其进行 NCU，可以看到它是 250 regs/thread，理论 occupancy 12.5%，No Eligible 75.3%。因此它是一个 latency bound，主要问题在于 reg 占用太高导致实际并行度太低了，无法掩盖 latency。

所以我们的第一优先级是先把 rhs 给拆了，尽量提高 occupancy。

VTune 发现 cuCtxSynchronize_v2 占 81.5%，nsys 发现有约 20 万次 kernel launch，所以第二优先级是处理同步问题以及 kernel launch 问题。

## kernel fission

拆 kernel 不是目的，缩短 reg 生命周期，减少 reg 占用，提高 occupancy 才是目的。

阅读代码发现 metric/shift Hessian, advection, constraints 存在明显边界，应该把这些相对独立的工作移出。

最终发现 rhs family 从 13.9s->10.4s，并且 evolve 从 91.8s->71.7s，metric/shift Hessian, advection 理论 occupancy 都达到 25%，然而我们的 rhs core 还是 255 regs/thread，说明我们虽然得到了部分提速，但是还是没完全做完这部分工作。

进一步深入源码分析，发现 advection 中重复的坐标偏移与插值系数的计算被算了 24 次，我们提到循环外面就好了。

效果是这个 kernel 的 reg 从 85->66，理论 occupancy 进一步上升，总时间下降 4%。

继续分析，发现 Ricci 有两个 consumer，它们重复计算了九个一阶导数，并且这些导数的计算比较复杂，导致寄存器数量上升。

方法就是加一个小 kernel，它算九个一阶导数，算完之后再给两个 consumer 直接用。

两个 consumer 的 reg 占用从 240->126 regs/thread，理论 occupancy 提升到 12.5%。三者用时从原来 2.191ms->1.084ms，程序总用时下降 2.6%。

## 调度优化

之前 profile 发现，有过多 device sync，以及过多小 kernel launch。并且在 nsys 上也能看出，我们 gpu 几乎是串行执行的，很少有多个 kernel 同时运行的情况，这就是由于调度太烂了。

### 精细化同步

只让真正需要 sync 的 stream 进行 wait，而不是随随便便就整个 device。

profile 表明，device sync 降到 215 次，并且总体 sync time 从 13.9s->9.14s，t=5 时间从 104.9s->100s，下降约 4%。

### 增加 stream

其实就是为了减少 gpu 空转的时间。

以 prolong3_kernel 为例，profile 表明它被调用了 47307 次，调用之间完全没有 overlap，并且它的 grid 有 54 blocks，在我们 14SM 的 MIG 上会有尾部。

所以我们增加 stream，具体来说在 parent stream 外建两条 stream，直接三路轮转，时间从 2.87s->1.49s。

t=5 时间下降 4.54%。

### 跨变量 batching

用了三路 stream 并不会减少 launch 数，prolong/restrict 分别有几万次 launch，有不少 overhead。

并且这些小 kernel 是对几十个物理变量做完全相同的插值，那为什么不把它们合并到同一个 kernel 一起启动？

做完之后，AMR 部分 launch 数下降 95%，t=5 时间再次下降 4.4%，来到 89s 左右。

## 访存优化

profile 表明，GPU kernel 已经几乎填满 evolve 了，也就是说 GPU 总在干活，但是还是很慢，这就说明 GPU 干的活中有重复的无用功。

最显然的无用功就是，因为我们用 stencil 计算模式，在计算一个点的时候要读取前后左右上下相邻的很多点。如果每个点不考虑彼此，独立计算，那么每个点会被读取很多很多次。

那咋办？分块好了，一趟把一个块的周围全读回来，大家一起用。

### advection

这是最热的热点区域，我们先看这一部分，这里每个点要读半径为 3 的邻居。

我们选择 block 大小 (8, 8, 4)，最开始让一些线程去 HBM 把 (14, 14, 10) 的 tile 搬到 shared mem 里。

测试发现，单个 advection kernel 从 2.48s->0.82s，获得 2.99x 优化。并且 t=5 也下降了 9.17%。

后面尝试了更加激进的数据搬运策略，发现 shared mem 暴涨，导致 occupancy 掉到 12.5%，反而变差。所以我们保留这一版。

### evolution

现在 evolution 是第一热点。

同样的方法，不过这次 consumer 只需要 contracted Hessian，所以我们先 contract 再搬运进 shared mem。

profile 发现该 kernel occupancy 从 12.5%->50%（来自于 reg 下降），耗时从 1.03ms->0.107ms，提升了 10.3x，t=5 也下降了 13%。

### beta/chi/lapse

基本一样。

大约下降 17%。

这一部分做完之后，我们基本 t=5 做到了约 55s。

## 计算优化

继续观察 nsys，发现 GPU 占用仍然几乎都是满的。

我们之前优化掉了不必要的内存访问，接下来该优化不必要的计算了。

### global_interp_kernel

占 1.67s。

审查代码发现，在演化阶段明明只是算两个黑洞位置的三个场（一共 6 个标量结果），却要仍调用这个 kernel，要经历完整的 H2D, D2H, 同步, MPI reduction 等等步骤，overhead 太大了。

写一个专用 kernel（global_interp_point3_kernel）就好了，总程序快约 1%。

此外，发现每个线程定义了一个巨大的数组 `ya[216]`，被编译器扔到了 local memory。而且每次读取 32 byte 只使用约 1 byte，这很坏。

我们直接删掉这个数组，重写算法，采用 streaming 方法，数据沿着 z->y->z 方向流动，只用几个标量寄存器来保存暂时的结果，算完一层就丢掉一层，不存储中间矩阵。

虽然寄存器数量猛增 64->236，occupancy 50%->12.5%，但是耗时快了 38%

### constraints-only mode

输出 constraints 报告的时候，源程序仍然完整跑所有的物理方程。但是事实上我们只需要几何数据、导数数据，别的根本不需要。所以不算就好了。

快了 1.4%。

### 删掉未被消费的生产者

源代码有些愚蠢。

constraint kernel 中有一个 predictor，会先算 constraints，但是这些结果在没有被读取前就会被重新计算并覆盖，也就是说这是彻头彻尾的多余计算。

删掉就好了。

constraint kernel 用时下降 89%。

最终，t=5 从之前约 55s 下降到了约 49s，我们接近了。

## kernel fusion

首尾呼应了属于是。

我们把 kernel 切开，基本把该做的优化都做完了，整个流程也基本固定下来了，那接下来就是看一看有没有适合做 kernel fusion 的，这样可以减少内存访问以及 kernel launch overhead。

### gamma derivative & seed

本来是 gamma derivative 算出 6 个导数，写入显存，然后 consumer 读这 6 个导数，算出 seed。那它们完全没必要分开嘛，直接一起做好了。

融合之后，这两个 kernel 下降了 45%，t=5 下降了 1.6%

### chi/lapse derivative 与下游 source algebra

source-metric, physical-Gamma, chi-Ricci, chi-Hessian, lapse 这些 kernel 直接消费，所以也没必要分开。

合并之后发现这些 kernel 的耗时下降 60%，t=5 总时间下降 6%。

### 零零碎碎的

beta compact prepare & Gamma consumer.

metric & A-diagonal & A-offdiagonal.

gauge & advection.

做完这一堆 fusion 之后，我们的 t=5 来到了 45s，外推已经 <= 370s。

最终实际测试约 360s。

## 失败的尝试

挺多的，介绍最有意思的一组。

在对 rhs 进行 fission 时曾尝试强制拆成 9 个 kernel（为了 occupancy），但是反而慢了 4.6%，说明其实 occupancy 不是最重要的东西。

然而又尝试过粗粒度的 fission，发现慢了 13.2%，这是因为引入了额外中间量与 launch 的 overhead 大于 occupancy 收益。

这部分告诉我，kernel fission 该进行到什么粒度其实是一个挺困难的问题，因为你也不知道这里多拆一点会不会方便后面优化。我目前的理解是，可以在尊重算法本身的数据边界的前提下，多拆一点，大不了到最后再 fusion 回来。

## 思考题

### 1

在 GPU 版本中，瓶颈主要是 规则 stencil，加上调度开销，加上重复访存以及计算。

profile 表明 rhs_kernel, prolong3, restrict3 等等分别占据 77.3%, 12.1%, 4.0%，这些 kernel 都是进行网格点上的计算，也就是规则 stencil 部分。

profile 同时表明，有太多次 device sync，以及过多的 kernel launch。

上述优化过程中我们也看到了，明明 GPU 一直打满，但是还是能找到优化。这就是因为 GPU 进行了无用的内存访问/计算。

CPU 上就不一样了。最开始 MPI 相关代码善举 82%，说明最开始版本是通信开销。

### 2

在我们这个 CPU 程序中，根本没有权衡的必要性。我们就一个节点，你干啥要用 MPI？

正经分析，MPI 底层的模型是，每个 worker 都有独立的内存空间，因此通信需要显式进行，开销更大。

而 OMP 底层的模型是，每个 worker 共享一块内存空间，通信是隐式进行的（不过会需要 lock 等机制来保证一致性），开销低非常多。

像我们这种在单 NUMA 上跑的程序，所有 core 本来访问的都是同一个物理内存，既然可以使用 OMP 的共享内存模型来降低开销，我们干嘛要用 MPI？

我自己实测也是 1 MPI * 30 OMP 最好。

当然可以用单一方法并行，比方说我这个程序就是。

问题是很多时候你无法使用 OMP，比方说不同 NUMA 可能访问的根本就不是同一块物理内存，你咋共享。

混合模式可能会出现在多台服务器的多个 NUMA 中，每个 NUMA 内部用 OMP，NUMA 之间、服务器之间用 MPI。

### 3

并行是分很多层级的。

最高层应该是 MPI，跨节点级别，当然我们没有这么做。

第二层是 block 层级的并行。AMR 把空间分成很多 block，在 CPU 版本中这些被分配给 OMP 不同 thread，GPU 中这就对应一个 block。block 内部是由顺序的，你要按 level 先把粗网格算完才能算更细的。

进一步在 block 内部，每个 thread 负责处理一个网格点，这些点之间也是并行的。

进行过跨变量 batching 后，同一个位置的 24 个物理变量也是并行的。

twopuncture 内部，在单条线上是串行的，但是不同线、不同方向之间可以并行。

并且我们通过 stream 实现了数据传输与计算的并行。

### 4

因为浮点运算不满足交换律、分配律等等，所以在优化过程中，如果我们修改顺序，经常会遇到结果不完全一致的情况。

精细判断是非常复杂的事情，所以我们介绍一些粗略判断的方法。

首先看误差是否够小，通常而言我们看相对误差。

其次看误差是否能解释，比方说我们交换了顺序，那可能会有误差。但是要是仅仅做了访存优化，就不该产生新的误差。

然后要看运行结果是否稳定，如果不稳定可能有 race。

最后你可以写一个 checker，在运行过程中自动帮你检查误差是否够小。

### 5

没试过。

首先这么做是可以跑起来的，问题也很明显。每个 rank 都会有自己的 context，会竞争 SM、L2 等等资源，更容易 OOM，通信要走 MPI 消耗更大。

其实是有解决方案的，有硬件和软件两个。

硬件方面，可以用 MIG 进行切分，有最强的隔离，但是是静态的。

软件方面，可以用 MPS，也就是一个软件调度器，但是隔离性弱。

二者可以结合起来一起使用。

### 6

不一定。

我们之前那个例子就很好。我们使用了 shared mem，但是 occupancy 下降了。虽然那个例子最终加速了，结果是好的，但是不一定总是这样。

这个例子表明，如果你一味提高 shread mem 占用，很可能导致 occupancy 下降，这不一定划算。而且还会带来一点额外的 sync 同步开销。也会多一点最开始搬运到 shared mem 的开销。

这就是为啥 infra 这么复杂，很多东西是不能一概而论的，都要结合具体情况做 trade-off。

### 7

80GB 不仅比 40GB 显存更大，而且是 HBM2e，运行频率高，带宽大。不过计算核心是完全相同的。

此前提交的时候已经注意到提交到 40G 80G 这两个不同版本上会带来明显的性能波动。应该就是由带宽导致的。

### 8

我觉得这个实验本身挺好的。所以我对 HPC101 这门课整体（包括 lab）发表一些不成熟的建议。

这门课体量太大了，学完之后整体感觉有点，囫囵吞枣。

比方说 profile 部分，我们常用的 profile 工具其实就那么几个，perf, VTune, Nsys, NCU。它们非常重要，在 lab 中被反复用到。但是当时讲 profile 那门课更多地是作为一个导论，没有那么深入。实验文档中也没有对这些工具进行进一步展开介绍，也没有给出优质的学习资料链接。我第一次打开 Nsys 的时候有点懵，每个部分我大概能猜出来大概是什么意思，但是毕竟不是 100% 确认（我甚至无法确定哪部分是 CPU 哪部分是 GPU），而 HPC 又比较在意精准理解，就做起来很难受。如果能有一个从界面到每个指标说的是什么的全面一些的新手指导我觉得会好很多。

此外实验部分也是如此，感觉上来就丢给大家太多太 heavy 的 work 了。我觉得 ai 时代，HPC 课程的学习确实也要不太一样，我们要学会使用 ai，这是对的。但是我觉得我们在对 HPC 能使用什么优化、这些优化的底层原理（乃至细节，很多时候细节非常重要）还不清楚的前提下，就想要指挥 ai 是很困难的事情。我此前学过 CSAPP，并且这个暑假看了一部分 Stanford CS149，按理说我对很多概念，很多优化都不陌生。但是饶是如此，还是觉得接触太多太多新概念了，有点知识过载。我的建议是，可以从我们这些 lab 中抽离出一些专项训练，这些专项训练让同学们自己手写（或者起码要看懂 ai 给的代码在干什么，搞懂每个细节），然后做完一堆专项训练之后，再开始给同学们丢这种比较综合、比较 heavy、更加切合实际的 lab。这样的话同学们就能学会看 profile 结果，并且看到每个结果也知道该用什么优化。当然不用手写这些优化，但是最起码要能解释为什么要用这些优化，它们在做什么，有没有什么底层小细节，它们优化什么部分，最后预期什么结果。

好像说得有点多了。不过总体来讲我觉得 HPC 还是一门非常硬的好课，助教 ggjj 们辛苦了，希望 HPC 越来越好！