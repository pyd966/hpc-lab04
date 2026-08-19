# TwoPuncture 优化前补充报告

记录日期：2026-08-19。本文回答正式改代码前的五个问题，范围是
`src/TwoPunctures.C` 的 TwoPuncture 初值求解器，不讨论后续 ABE 演化。
结论基于带 `-O3 -g -fno-omit-frame-pointer` 的 HPC profile，以及对当前源码的
逐函数检查；本文本身不改变数值算法。

## 先说结论

1. 128 GiB 内存足以把求解阶段的工作区在初始化时一次申请，并在求解器结束时释放。
   但不应该把“完全不释放”当成目标。最值得消除的是热点循环中的短生命周期
   `new[]/delete[]`、`malloc/free`，而不是所有动态内存。
2. “预计算 cos”不是预知物理过程中的所有余弦，而是把只依赖网格尺寸和循环下标的
   角度表提前算一次。`chebft_Zeros()` 中的这类 `cos()` 在本次 profile 中占
   23.17% cycles，适合做第一批低风险优化。
3. OpenMP 可以覆盖大部分点循环、谱变换中的独立一维线，以及同一颜色的 line
   relaxation 线；三对角线内部的 Thomas 递推仍然是顺序的。并行区应包住一批
   `NRELAX`，不能为每一条长度 50 的线创建一次线程队伍。OpenMP runtime 本身已经
   维护线程队伍，第一版不需要另写 worker pool。
4. 当前 `JFD`/`cols` 并不是“每行单独 malloc”。`dmatrix()`/`imatrix()` 为行指针表
   和一整块连续 payload 各申请一次。当前真正的问题是 `double**/int**` 的间接访问、
   固定 19 项行格式没有被显式表达，以及 B 方向按网格布局跨步访问。
5. 预条件器就是 BiCGSTAB 每一步中先近似解 `JFD * z = rhs` 的那部分。可以改变
   relaxation 次数或换算法，但收益和收敛性必须一起测量。`NRELAX=200` 直接改小
   可能使每步更快，也可能让 BiCGSTAB 需要更多步甚至不收敛。

## 1. 内存：128 GiB 是否意味着可以去掉所有 allocation？

### 正确的目标

应采用“初始化时分配、循环中复用、结束时释放”的工作区设计：

```text
构造/初始化
    allocate persistent solver workspace
Newton/BiCGSTAB/relax/F_of_v
    only reuse workspace; no hot-loop heap allocation
求解结束/析构
    free workspace
```

内存容量解决的是“能否常驻”，不解决 allocator 的调用成本、cache 污染、NUMA
first-touch 位置和程序生命周期管理。永久不 `delete/free` 会造成泄漏，也会使多次
调用求解器或测试不同网格时无法可靠复用。

因此，如果“完全去掉”指 `Solve()` 的稳态热路径，答案是可以：网格尺寸在构造后
固定，所有最大 buffer 长度都已知，可以做到进入求解循环后零 heap allocation。
如果是字面上的“整个程序一次也不申请、最后也不释放”，则既没有必要，也不利于
资源管理。

### 当前各类分配的生命周期

以 profile 网格 `n1=n2=50, n3=26, nvar=1` 为例，`ntotal=65000`，一个 double
向量约 0.50 MiB，一个 `derivs`（10 个 double 向量）约 4.96 MiB。

| 位置 | 当前行为 | 建议 |
| --- | --- | --- |
| 构造函数 `TwoPunctures::TwoPunctures()` | `F`、`u`、`v` 长期保存，约 10.42 MiB | 继续常驻，在析构时释放 |
| `Newton()` | 每次 Newton 调用申请 `F`、`dv`、`u`，用完释放 | 提升为 `Solve` 级 workspace；六次线性求解之间复用 |
| `bicgstab()` | `JFD`、`cols`、`ncols` 和多个向量每次申请；矩阵内容随当前 `u` 变化 | 数组容量常驻，进入下一次 BiCGSTAB 前覆盖内容 |
| `F_of_v()` | 每次申请 `values`、长度为 1 的 `U`，并 `calloc` 一整块全零 `sources` | `values/U` 用栈或线程私有 workspace；当前 `if (0)` 分支下 `sources` 可移除，若以后启用源项则保留一个常驻源项数组 |
| `Derivatives_AB3()` | 每次申请 7 个长度约 50 的数组和一个索引数组 | 每个 OpenMP worker 一份固定大小 scratch，反复复用 |
| `LineRelax_be/al()` | 每条线申请 `diag/e/f/b/x` 五个数组 | 每个 worker 准备一套按 `max(n1,n2)` 配置的 scratch，A/B 阶段复用 |
| `ThomasAlgorithm()` | 每条线再申请 `l/u/d/y` 四个数组 | 改成传入已有 scratch，或在已有数组上原位完成 |
| `JFD_times_dv()` | 被构造有限差分矩阵时大量调用；每次申请两个长度为 `nvar` 的 `derivs` | 改成传入调用者的 scratch；不能在该函数内 heap allocate |

`LineRelax` 的一条线只有约 50 个未知量。即使给 60 个 worker 各准备一套约
3.6 KiB 的 line/Thomas 数组，也只需要约 0.21 MiB。即使把一个完整 BiCGSTAB 的
矩阵和向量全部常驻，按当前规模也只有几十 MiB，远低于 128 GiB。

需要注意的是，内存峰值很小并不代表分配无害。本次 profile 看到 allocator 自身
至少占 6.07% samples；此外 `SetMatrix_JFD()` 中约 65,000 个列、每列最多 27 个
邻域点都会调用 `JFD_times_dv()`，所以小数组分配的调用次数远多于峰值内存所能反映的
数量。

根据本次日志的 88 个 BiCGSTAB iteration，每个 iteration 最多对两个方向各做一次
预条件器应用，所以应用次数上限是 176；考虑最后一次的 early check，合理估计约为
170--176 次。本报告用中间值 174 估算：每次应用执行 `NRELAX=200`，即约 34,800
个完整 relaxation。按当前 A/B 线循环，一次 relaxation 调用 2,587 条 line solver，
所以总计约 90,027,600 条线。每条线的 `LineRelax` 和 `ThomasAlgorithm` 合计有
9 组 `new[]/delete[]`，源码级估算约为 8.10 亿组小数组申请/释放。这不是峰值内存
问题，而是同一小块 scratch 被反复创建的问题；也解释了为什么只占几十 MiB 的程序
仍会在 allocator 上耗时。

### 建议的工作区层次

第一阶段不要一次重写整个类，可以分三层逐步做：

1. `LineWorkspace`：每个线程保存 `diag/e/f/b/x`、`l/u/d/y`，长度按最大 A/B
   网格配置申请一次。
2. `TransformWorkspace`：每个线程保存 Chebyshev/Fourier 的输入、导数、索引
   临时数组；同一个 worker 依次处理 A、B、phi 三个阶段。
3. `SolverWorkspace`：保存 `F`、`JFD`、`cols`、`ncols`、BiCGSTAB 向量和
   Newton 临时 `derivs`。一次求解完成后只清空/覆盖，不释放再申请。

这种设计不是为了“把 128 GiB 填满”，而是为了让分配次数与网格大小和线程数相关，
而不再与数百万条短线相关。数组应在绑定好的并行区内 first-touch，以便未来跨 NUMA
运行时避免工作线程访问远端内存。

## 2. “预计算 cos”究竟预计算什么？

### 可以预计算的部分

`chebft_Zeros()` 的正变换使用

```text
cos((Pi/n) * j * (k + 0.5))
```

逆变换使用

```text
cos((Pi/n) * (j + 0.5) * k)
```

这里的 `n`、`j`、`k` 只由谱网格决定，与数组 `u[]` 的数值无关。以 `n=50` 为例，
正变换表和逆变换表各 50 x 50 个 double，总共约 40 KiB。每次变换仍然要做乘加，
只是把重复的 libm `cos()` 调用改成数组读取。

`fourft()` 也一样：`cos((Pi/M) * l * k)` 和 `sin((Pi/M) * l * k)` 只依赖
`N`、`l`、`k`。`N=26` 时两张表合计约 6 KiB。它们的收益小于 Chebyshev 表，
但实现方式相同。

还可以在后续阶段缓存网格几何量，例如

```text
A[i] = -cos(Pih * (2*i+1) / n1)
B[j] = -cos(Pih * (2*j+1) / n2)
sin(phi[k]), cos(phi[k])
```

这些也只依赖网格下标。进一步检查代码后，`par_b`、动量和自旋在一次 `Solve()` 中
也不变化，因此各网格点的 `X/R/x/r/y/z`、`r_plus/r_minus`，甚至
`BY_KKofxyz(x,y,z)` 都有可能缓存。不能缓存的是依赖当前 `v/u` 和本轮 bare mass 的
`psi`、方程残差和 Jacobian 作用结果。几何缓存应作为单独实验，不要和 transform
表同时改，以便判断收益和验证数值。

### 不能预计算的部分

不能预先列出“整个运行过程中所有 cos 的结果”。凡是角度由当前场、插值位置或运行
时才确定的表达式，都不能仅凭网格下标查表。例如任意位置插值的 `phi=atan2(z,y)`
由查询点决定，不能用固定的 `phi[k]` 代替。预计算的边界是：表达式的所有输入都在
初始化时确定。

### 数值正确性

第一版表格应在初始化时使用同一个 `libm` `cos()`/`sin()` 生成，并在变换中保持原有
`j`、`k` 的累加顺序。这样同一个进程中查表得到的 double 通常与原来重复调用的
double 完全相同；真正改变求和顺序的 OpenMP reduction 或 FFT 替换则可能产生末位
差异。每次实验都要比较 Newton/BiCGSTAB 的迭代数、最终残差和输出文件，而不只比较
运行时间。

## 3. OpenMP：哪些热点可以并行？并行区放哪里？

### 当前基线不是 MPI 程序

源码没有 `MPI_Init`、通信或 collective；TwoPuncture 直接由一个进程、一个线程
执行。本次 profile 的 `CPU utilized=0.999`，因此不存在 MPI rank 负载均衡问题。
这里最自然的是一个共享内存 OpenMP team，而不是把谱方向拆成 MPI 子域。

OpenMP 工作共享循环由已经存在的 team 分配迭代，并在没有 `nowait` 时带隐式 barrier；
这正适合将独立的一维变换分给线程，同时明确保留导数阶段之间的屏障。参见
[OpenMP 5.2 execution model](https://www.openmp.org/spec-html/5.2/openmpse3.html)
和
[worksharing-loop 语义](https://www.openmp.org/spec-html/5.2/openmpse66.html)。

### 按热点函数划分

| 函数/区域 | 能否并行 | 依赖和实现建议 |
| --- | --- | --- |
| `F_of_v()` 最后的三重网格循环 | 可以 | 每个 `(i,j,k)` 写不同的 `F/u` 元素；`U`、`values` 必须是线程私有。当前 `sources` 先全部置零且源项分支是 `if (0)`，可在保留功能开关的前提下去掉这次无意义清零。 |
| `J_times_dv()` 计算 `Jdv` 的点循环 | 可以 | 先完成 `Derivatives_AB3()`，再对 `(i,j,k)` 做 `omp for collapse(3)`；每个线程使用自己的 `dU/U/values`。 |
| `Derivatives_AB3()` A 阶段 | 可以 | `(k,j)` 对应的 A 方向线互不写同一输出；每条线内部的 transform 和 `chder` 顺序保留。 |
| `Derivatives_AB3()` B 阶段 | 可以 | 等 A 阶段结束后，对 `(k,i)` 的 B 线分工；它读取 A 导数，所以需要阶段 barrier。 |
| `Derivatives_AB3()` phi 阶段 | 可以 | 等 B 阶段结束后，对 `(i,j)` 的 Fourier 线分工；需要自己的 phi scratch。 |
| `relax()` 的同一颜色线 | 可以，但有条件 | 同一 `k` 奇偶、同一 `i`/`j` 奇偶的线更新点不重叠，且 19 点 stencil 只读相反颜色的邻点；可以并行该颜色，颜色之间必须 barrier。 |
| `ThomasAlgorithm()` 单条线 | 基本不能 | LU、前代、回代各自有前后依赖；应并行许多条线，而不是拆一条线。 |
| BiCGSTAB 向量更新、范数、点积 | 可以 | 用 `omp for` 和 `reduction`；标量 `alpha/beta/omega/rho` 的更新仍要按算法顺序进行。 |
| `SetMatrix_JFD()` 当前按列构造 | 不能直接并行 | 不同列会同时向同一行 `ncols[row]`、`cols[row]`、`Matrix[row]` 追加，存在写冲突；需要先改为行中心或线程局部 buffer，再考虑并行。 |

### line relaxation 的依赖为什么允许“同色并行”

`LineRelax_be()` 固定 `i,k`，沿 `j` 解一条三对角线；`LineRelax_al()` 固定 `j,k`，
沿 `i` 解一条三对角线。线内部的三个对角项被 Thomas 算法处理，线外项读取邻接
点。`Index()` 采用 `i` 最快、随后 `j`、最后 `k` 的布局。

因此，两条 `i` 相差 2 的 B 线不会通过当前 19 点 stencil 读取对方刚写的数据；两条
`j` 相差 2 的 A 线同理。`k` 方向也按奇偶分组，偶数平面更新完后，奇数平面才读取
这些结果。一个可行的逻辑顺序是：

```text
for each plane-relax pass:
    even-k: B-even-i -> barrier -> B-odd-i -> barrier
            A-odd-j  -> barrier -> A-even-j -> barrier
    odd-k:  B-even-i -> barrier -> B-odd-i -> barrier
            A-odd-j  -> barrier -> A-even-j -> barrier
```

同色阶段可以用 `omp for collapse(2) schedule(static)` 分配 `(k,line)`。这不是把
最外层 `k` 循环简单加上 `omp parallel for`：如果忽略上述颜色和阶段顺序，会把
Gauss-Seidel 更新错误地改成有竞争的异步更新。第一版必须用小网格对照串行结果，
并检查每个 barrier 后的 residual。

### 并行区和 worker pool 的选择

若直接改造，最坏的组织方式是“每条线一个 parallel region”；这会把线程管理成本
放进约 9,000 万条短线中。推荐按以下顺序实现：

1. 先让 `Derivatives_AB3()`、`F_of_v()`、`J_times_dv()` 各自有一个较粗粒度并行
   区，线程私有 scratch 在区内复用。
2. 对预条件器，把一整个 `NRELAX` 批次包在一个 `#pragma omp parallel` 中；批次
   内的每个线程依次参加所有颜色阶段，不能让每个线程独立调用完整 `relax()`。
   这样并行区次数从“每次 relaxation”降到“每次预条件器应用”，同时保留所需 barrier。
3. 若 profile 显示 fork/join 仍明显，再把 BiCGSTAB 的向量操作和预条件器批次合并
   到更大的 persistent team，并用 `single`/`reduction` 管理标量。此阶段改动较大，
   不应和第一次数值改动混在一起。

OpenMP runtime 已经是一个线程队伍管理器；`parallel` 区域会创建 team，工作共享循环
使用该 team 中已经存在的线程。另写锁队列式 worker pool 会增加任务协议、停止条件、
异常处理和数值屏障的复杂度，未必比 libgomp 的复用更快。只有在 profile 证明 OpenMP
fork/join 或 barrier 成为主要热点时，才值得考虑定制 pool。

### 绑核、线程数和 NUMA

HPC profile 节点显示 `TaiShan-v120`、4 个 NUMA node、每个 core 2 个 hardware
thread；这次作业的 `Cpus_allowed_list=64-123`、`Mems_allowed_list=1`，即只在
NUMA node 1 上运行。课程评测说明中的“60 physical cores、无超线程”是评测资源
语义，不能用本次节点的 `60 logical CPUs = 30 cores` 机械替代。因此脚本应根据
调度器分配和 `lscpu -e=CPU,CORE,NODE` 计算实际可用物理核数；不要把 30 或 60 永久
写死。

建议的运行环境是：

```bash
export OMP_DYNAMIC=FALSE
export OMP_NUM_THREADS=<allocated physical core count>
export OMP_PLACES=cores
export OMP_PROC_BIND=close
```

调试绑定时再临时设置 `OMP_DISPLAY_ENV=VERBOSE` 和
`OMP_DISPLAY_AFFINITY=TRUE`，检查每个线程是否落在分配的 core 上；正式计时时关闭
详细输出。GNU libgomp 的说明明确了 `OMP_PLACES=cores` 的 place 含义、
`OMP_PROC_BIND` 的不可迁移策略，以及 affinity display 选项，参见
[OMP_PLACES](https://gcc.gnu.org/onlinedocs/libgomp/OMP_005fPLACES.html)、
[环境变量](https://gcc.gnu.org/onlinedocs/libgomp/Environment-Variables.html)
和
[OMP_DISPLAY_AFFINITY](https://gcc.gnu.org/onlinedocs/gcc-14.1.0/libgomp/OMP_005fDISPLAY_005fAFFINITY.html)。

本次 profile 的内存许可只包含一个 NUMA node，所以 first-touch 后没有跨 node
访问问题。若以后申请跨多个 NUMA node 的 60 物理核，应让线程绑定与数据初始化
保持一致，并按 node 或 tile 分配大数组；这不是当前单 node TwoPuncture 的第一优先级。

## 4. “稀疏数据布局”是什么意思？

### 先更正原报告的一句话

原报告中“当前每行单独分配”说得不准确。`dmatrix()` 的实际逻辑是：

```text
一块 double payload: ntotal * maxcol
一张 double* row pointer table: ntotal
每个 row pointer 指向 payload 中对应的固定宽度位置
```

`imatrix()` 对 `cols` 做同样的事情。因此数据本体已经连续，改成 flat array 的
收益会比“把分散 malloc 合并”小；仍然可以去除 row pointer 间接层。

### 为什么叫稀疏

`JFD` 是有限差分 Jacobian。每个方程只依赖本点及其附近的 19 个位置，矩阵一行最多
保存 `StencilSize=19` 个非零项，`cols[row][m]` 保存列号，`JFD[row][m]` 保存值，
`ncols[row]` 保存实际数量。它不是 65,000 x 65,000 的全矩阵。

对当前规模，稠密 double 矩阵需要约 `65000^2*8 = 33.8 GB`（31.5 GiB），而
固定 19 项的 `JFD` payload 约 9.9 MB（9.4 MiB）、`cols` payload 约 4.9 MB
（4.7 MiB），另加行指针和 `ncols`。
128 GiB 虽然容得下稠密矩阵，但稠密矩阵还会使每次矩阵向量乘从约百万级非零项变成
42 亿项，不能因为“内存够”就改成稠密。

### 当前布局真正的性能问题

`Index()` 中 `i` 是最内层布局。于是 `LineRelax_al()` 固定 `j,k`、沿 `i` 走时
访问较连续；`LineRelax_be()` 固定 `i,k`、沿 `j` 走时，每个点相隔 `n1=50` 个
double，JFD 行还相隔约 `50*19` 个 double。profile 中 B 方向 `LineRelax_be`
自身 27.17%，A 方向 `LineRelax_al` 自身 19.09%，这个方向差异与跨步访问和矩阵
邻项 gather 是一致的线索，但仍应以改版后的 profile 验证。

### 推荐的数据结构改动

低风险候选是把当前二维接口改成固定宽度的一维数组：

```cpp
value[row * StencilSize + m]
column[row * StencilSize + m]
```

这仍然是同一稀疏格式，只是没有 `double**/int**` 的 row pointer。`ncols` 可以保留。
更进一步，可以在矩阵构造后把每行的 A/B 三对角项单独提取，line solver 不必每次
扫描 19 项并比较 `col == Ip/Ic/Im`；其余非线项再用连续数组保存。由于 B 线的
网格跨步仍然存在，这一改动属于第二阶段，优先级低于消除分配和并行颜色线。

## 5. 预条件器是什么？可以换吗？收益多大？

### 用一个不抽象的比喻说明

Newton 在每一步要解线性方程 `J * dv = F`，BiCGSTAB 是主迭代器。它不直接精确
求解这个大方程，而是先用一个便宜的近似解法得到方向，再用真实的 `J_times_dv()`
修正方向。这个“先把问题变得好解一些”的近似解法就是预条件器。

当前代码中：

```text
SetMatrix_JFD()  构造稀疏有限差分 Jacobian JFD
relax()          对 JFD*z = rhs 做 NRELAX 次 line relaxation
J_times_dv()     用谱导数计算真实 J*z
BiCGSTAB        根据残差和点积更新 dv
```

所以 `relax()` 并不是最终物理解，也不是把 `JFD` 一次精确分解；它是每个 BiCGSTAB
方向上的近似求解器。profile 中 `relax` 调用路径占 66.85%，是最值得研究的算法
热点。

### 可以调哪些东西？

当前最直接的参数是：

- `NRELAX=200`：每次预条件器应用重复多少个完整 relaxation pass；
- `N_PlaneRelax=1`：一个平面颜色序列重复多少次；
- line solver 的顺序和颜色方案；
- 更换为 Jacobi/block-Jacobi、加性 Schwarz、ILU(0) 或针对椭圆型问题的 multigrid
  等预条件器。

可以改，但必须把“每次应用成本”和“BiCGSTAB 需要的应用次数”一起看。假设其它
工作不变，`relax` 占 66.85%：如果将其成本理想地减半，Amdahl 上限约为 1.50x；
减到四分之一，上限约为 2.01x。实际若 `NRELAX` 减半导致 BiCGSTAB 迭代数翻倍，
收益可能消失；过弱还可能触发 `rho/omega` breakdown 或在 `itmax=100` 内不收敛。

反过来，更强的预条件器通常每次更贵，但可能显著减少迭代。它不会改变“收敛到同一
残差时的连续数学问题”，但有限精度、并行更新顺序和停止阈值会改变中间路径，输出
的末位也可能变化。因此不能只看最终程序退出码。

### 建议的参数实验

第一轮只扫描 `NRELAX`，例如 `200, 100, 50, 25`，保持其余代码和线程数不变。每个
点记录：

1. 六次 Newton/BiCGSTAB 的各自 iteration 数和总数（baseline 总数为 88）；
2. 每次线性求解的最终 `normres`、是否触发 breakdown；
3. TwoPuncture 总时间以及 `relax`、`J_times_dv` 的 profile 占比；
4. `puncture_parameters_new.txt` 和去掉时间行后的 `Ansorg.psid` 数值比较。

若较少 relaxation 已经足够，收益可能很大；若收敛明显恶化，则应保留 200，先通过
同色 OpenMP、scratch 复用和数据布局降低单次成本。更换预条件器属于第二个实验系列，
因为它同时改变算法行为和并行依赖，不能与内存重构一起提交。

## 推荐的实施顺序

正式优化时建议每次只改变一个因素：

1. 先复用 `LineRelax`/`Thomas`/`JFD_times_dv` scratch，验证输出和 iteration 数；
2. 加 Chebyshev/Fourier 表，确认 `cos()` 和 allocator 热点下降；
3. OpenMP 并行 `Derivatives_AB3`、`F_of_v`、`J_times_dv` 的独立点/线循环，使用
   `OMP_PLACES=cores` 和 `OMP_PROC_BIND=close`，做线程数扫描；
4. 在保留颜色 barrier 的前提下并行 line relaxation，并检查不同线程数的残差；
5. 最后再试 flat sparse layout、line-oriented coefficients 和 `NRELAX`/新预条件器。

每一步的验收标准都应包括：端到端时间、CPU 利用率、线程 affinity、六次线性求解的
迭代/残差，以及与 baseline 的输出比较。这样才能区分真实加速、数值路径改变和
偶然的 HPC 节点波动。
