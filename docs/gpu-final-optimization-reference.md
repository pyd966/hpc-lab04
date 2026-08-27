# ABEGPU 端到端优化参考报告

> **用途说明：** 本文只用于帮助理解实验过程、核对数据和组织自己的报告。课程文档明确
> 禁止直接提交 AI 生成的报告，因此请结合你本人实际做过的实验、截图、代码阅读和理解，
> 用自己的语言重新撰写；本文不代写思考题。
>
> **覆盖范围：** NVIDIA A100 80GB PCIe 的 MIG `1g.10gb`、单 MPI rank、GPU 路线。
> 最终代码提交为 `0e4551a`，370 秒结果的 roadmap 记录提交为 `57f89f9`。

## 1. 摘要

本次优化对象不是单个 CUDA kernel，而是
`TwoPunctureABE + ABEGPU` 的完整科学计算流水线。最初的 `t=5` 三次生产测试中，
`This Program Cost` 为 `122.491884 +/- 1.557112s`，其中 Evolve 均值为
`92.1168s`。正式 `t=100` baseline 无法在 30 分钟时限内完成；运行到第 9 个物理
时间步后已经显示单步约 `18--20s`，短窗乐观外推约 `1872.71s`。

最终短窗结果为 Program `45.658880 +/- 0.044578s`、Evolve `16.9007s`，相对最初
baseline 分别加速 `2.6828x` 和 `5.4505x`。最终在规定输入、无 profiler、禁用
TwoPuncture cache 的正式 `t=100` 配置下重复三次：

| Run | Total Evolve Time | This Program Cost | checker |
|---:|---:|---:|---|
| 1 | `331.528s` | `360.864865s` | PASS，trajectory RMS `0` |
| 2 | `330.663s` | `359.853815s` | PASS，trajectory RMS `0` |
| 3 | `333.419s` | `363.817126s` | PASS，trajectory RMS `0` |
| mean | `331.870s` | **`361.511935s`** | 3/3 PASS |

Program Cost 的样本标准差为 `2.059365s`，最慢一次仍低于 `370s`。三次都匹配
`100/100` 个轨迹时刻、`596` 个有效比较项，且 9 层约束输出完整。

优化主线可以概括为：

1. 先用 profile 找到巨型 RHS 的寄存器生命周期问题，通过有边界的数据流拆分使各阶段可调优；
2. 再根据全流程依赖图缩小同步范围，并提高 AMR 调度粒度；
3. 用 shared-memory tile 解决规则 stencil 和 AMR 插值中的跨线程重复读取；
4. 消除高频插值中的通用控制路径、显式 local array 和稍后必然被覆盖的计算；
5. 当各 producer 已经足够小后，再按清晰的 producer-consumer 关系做受控 fusion，减少中间
   global-memory 往返。

这里的核心不是“kernel 越拆越好”或“kernel 越少越好”。拆分用于缩短寄存器 live range，
融合用于消除已经识别出的中间数据流；两者都必须由 NCU、Nsys、端到端时间和正确性共同决定。

## 2. 实验要求、环境与测量口径

### 2.1 不能改变的内容

所有保留版本均保持课程规定的物理问题和数值方法：

- GPU 模式、单 MPI rank、A100 MIG `1g.10gb`；
- 正式演化终点 `t=100`、Courant factor `0.5`、9 层 AMR；
- equatorial symmetry、四阶有限差分、项目配置字符串 `runge-kutta-45`；GPU 路径原有
  四个 RK4 substep 完整保留；
- FP64 关键路径、原输出间隔和全部必要输出；
- 不读取预计算答案，不减少 RK stage，不缩小网格，不跳过必要分析。

相对 GPU baseline `0951bc3`，性能实现改动集中在 `src/`，同时修改/新增了
`hpc_gpu_benchmark.sh`、`hpc_gpu_profile.sh`、`hpc_gpu_rhs_pair.sh` 等测量 helper
和内部文档；正式输入、Python driver、checker、绘图逻辑与 CMake 精度配置未改变。

### 2.2 硬件与软件

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA A100 80GB PCIe，MIG `1g.10gb`，14 SM，compute capability 8.0 |
| Host | Intel Xeon Gold 5320，allocation 提供 16 logical CPU，24 GiB 内存 |
| MPI | Open MPI 5.0.7，`mpiexec -n 1`，绑定 core |
| 编译器 | GCC/GFortran 14.2.0，CUDA 13.3 |
| CUDA 目标 | `sm_80` |
| 生产构建 | C++/Fortran `-O3`；CUDA 显式参数见 2.4；未使用 fast-math |
| Profiler | Nsight Systems 2026.1.3、Nsight Compute 2026.2.1、VTune 2026.3 |

所有性能测试都通过 `hpc submit` 在远程计算节点执行；x86 DevPod 只用于源码编辑和
只读分析。

### 2.3 三种证据各自回答的问题

| 证据 | 用途 | 不能推出什么 |
|---|---|---|
| VTune | 看 host 调用链、同步等待和 MPI/CUDA API 热点 | 不能解释 kernel 内部寄存器和访存原因 |
| Nsight Systems | 看全程序 GPU 时间线、kernel aggregate、调用数、同步和 overlap | profiler 下的绝对 Program Cost 不能当正式成绩 |
| Nsight Compute | 看一个代表性 launch 的 registers、occupancy、stall、local/global/shared memory | 单 launch 加速不能直接等同于端到端加速 |
| 生产 benchmark | `t=5 x3` 筛选候选，`t=100 x3` 最终验收 | 短窗不能替代正式 `t=100` |

需要特别注意：`cudaDeviceSynchronize` 或阻塞 `cudaMemcpy` 的 host API 时间通常是在
等待已经计入 GPU kernel 时间的工作，二者不能相加。正式评分口径始终是 Python driver
打印的 `This Program Cost`，不是 outer wall、kernel sum 或短窗线性外推。

### 2.4 最终并行与 kernel launch 配置

课程要求给出最终配置，而不能只写硬件型号。正式 artifact 中的实际配置如下：

| 项目 | 最终设置 |
|---|---|
| MPI | `mpiexec -n 1 --bind-to core`，Open MPI 5.0.7 |
| OpenMP | `OMP_NUM_THREADS=16`；ABEGPU 构建为 `AMSS_ENABLE_OPENMP=OFF`，TwoPuncture 为 `AMSS_ENABLE_TWOPUNCTURE_OPENMP=ON` |
| NUMA/绑定 | 单 rank 绑定到 core；未额外使用 `numactl` 或手工 memory binding |
| GPU/MPI 数据路径 | `AMSS_MPI_CUDA_AWARE=0`，保留 host staging |
| CMake | `AMSS_ENABLE_GPU=ON`、`CMAKE_CUDA_ARCHITECTURES=80`、`AMSS_OPT=-O3` |
| ABEGPU C++ | `-O3 -fno-strict-aliasing -Wno-deprecated -Dfortran3 -Dnewc` |
| ABEGPU Fortran | `-O3 -fno-strict-aliasing -cpp` |
| CUDA/NVCC | C++14、`compute_80,sm_80`、`-rdc=true -lineinfo`；没有额外 `-O` flag |
| TwoPuncture C++ | `-O3 -march=native -fopenmp`，并继承 common safety/define flags |
| 浮点/编译约束 | FP64；未使用 `-Ofast`、`-ffast-math` 或 `--use_fast_math` |

ABEGPU 没有一个对所有 kernel 通用的固定 grid，因为每个 AMR patch 的 extent 不同。最终主要
launch 几何为：

- RHS、compact Hessian/advection、Sommerfeld boundary 和 tiled prolong 采用 block
  `(8,8,4)`，即 256 threads；grid 按每个 patch 的三维 extent 向上取整；
- batched prolong 仍用 `(8,8,4)`，把空间 tile 展平到 `grid.x`，变量 batch 放在
  `grid.y=nvars`；
- batched restrict、RK 和多数一维 pointwise kernel 使用 256-thread block，
  `grid.x=ceil(n/256)`；batched restrict 同样用 `grid.y=nvars`；
- 通用 analysis interpolation 使用 256-thread block、`grid.x=ceil(NN/256)`，固定 BH
  point3 快路径使用单 block、3 threads。

因此报告 block/grid 时应按 kernel family 描述，不能把某一次 NCU 采样的 grid 数写成全程序
固定配置。

## 3. Baseline 与优化路线的形成

### 3.1 最初 baseline

最初 `t=5` 三次结果为：

| 指标 | baseline |
|---|---:|
| This Program Cost | `122.491884 +/- 1.557112s` |
| Total Evolve | `92.1168s` |
| checker | 3/3 PASS，trajectory RMS `0` |

初始 Nsys（`profile/gpu-nsys-20260821T*`）覆盖 `t=0..4`：

| Kernel family | 总时间 | 占 kernel time | Calls |
|---|---:|---:|---:|
| 单体 `rhs_kernel` | `56.9060s` | `77.3%` | 1623 |
| `prolong3_kernel` | `8.8696s` | `12.1%` | 47307 |
| `restrict3_kernel` | `2.9178s` | `4.0%` | 10047 |
| `global_interp_kernel` | `2.0093s` | `2.7%` | 1376 |

初始 RHS 的 NCU 结果为 250 registers/thread，理论/实际 occupancy
`12.50%/11.04%`，`No Eligible=75.30%`。Compute throughput 只有 `22.26%`，
DRAM throughput 只有 `3.97%`；因此它既不是 FP64 峰值饱和，也不是 HBM 带宽饱和，
而是高寄存器压力下只有很少 warp 驻留，无法隐藏 L1TEX scoreboard 和执行依赖。

VTune 中 `cuCtxSynchronize_v2` 占 host CPU time 的 `81.5%`，Nsys 同时观察到
4110 次 `cudaDeviceSynchronize` 和约 20 万次 kernel launch。MPI Allreduce 只有
约 `0.1%` host CPU time。由此得到三个优先级：

1. 最大收益必须来自 RHS 计算量和数据访问；
2. 同步和 AMR 小粒度调度是第二层问题；
3. 单 rank MPI、PCIe 字节传输和小型 RK4 kernel 不是当前第一瓶颈。

### 3.2 从热点比例到阶段预算

compact advection 完成后的新 Nsys 中，RHS family 占 `72.35%`，
prolong/restrict/global interpolation 占 `19.61%`，其余只有 `8.04%`。当时
`t=5` 的 Evolve 为 `45.0802s`，即约 `9.016s/unit`；要在固定成本约
`29.55s` 下进入 `370s`，演化斜率需要下降到约 `3.404s/unit`。

这说明只优化一个 kernel 不可能达标。后续路线必须同时覆盖 RHS stencil、
AMR/interpolation 和重复数据流，并在每次热点迁移后重新排序。

## 4. 主线一：先拆开不可调优的巨型 RHS

这一阶段的目的不是追求更多 kernel，而是缩短不同数学阶段的寄存器生命周期，使真正的
热点可以分别优化。

### 4.1 拆出 Hessian、advection/KO 和 constraints

**触发证据。** 单体 RHS 占 GPU kernel 时间 `77.3%`，250 registers/thread 将 occupancy
限制在 `12.5%`。源码阅读还发现 metric Hessian、shift Hessian、advection/KO 和
constraints 在数据流上存在明确边界。

**优化目的。** 将能独立消费的导数和约束工作从巨型 kernel 中移出，缩短 core 内部
中间量 live range，同时避免把完整几何张量全部物化。

**实现。** 提交 `3b8706a`、`f90ea68` 和最终 `81cf518`：

- metric Hessian 和 shift Hessian 分别批量生产紧凑结果；
- 24 个字段的 advection 与 KO 建立跨变量处理路径；
- constraints 成为独立 consumer；
- scratch 复用原有 `Rxx..Rzz` 等数组，并依靠同一 CUDA stream 的顺序保证生命周期。

**结果。**

- Nsys 的 RHS family 从 `13.9386s` 降至 `10.4985s`，下降 `24.7%`；
- profile 中 Total Evolve 从 `17.5389s` 降至 `14.2263s`，下降 `18.9%`；
- metric/shift Hessian 和 advection/KO 的理论 occupancy 达到 `25%`；
- 无 profiler 的 `t=5` Program 为 `101.667 +/- 0.583s`，3/3 PASS。

在同一 allocation 的 base/candidate 交错测试中，Evolve 从 `91.825s` 降至
`71.719s`，下降 `21.9%`。但 core 本身仍是 255 registers/thread，说明宽泛 fission
只能移走独立阶段，无法自动解决 Ricci 等内部 live range。

### 4.2 有针对性的 live-range fission 与 advection 公共量提取

**触发证据。** 第一轮拆分后，Gamma seed、beta/Gamma 和 Ricci consumer 仍有高寄存器；
advection 已成为 RHS 内最大项。NCU 还显示某些“按大功能块拆分”的 kernel 仍未跨过
`12.5% -> 25%` occupancy 档位。

**优化目的。** 拆真正造成寄存器峰值的 producer-consumer，而不是按输出分量复制整套公式；
同时把 advection 对 24 个字段重复计算的坐标、边界和 shift 公共量提到线程级只计算一次。

**实现。** `2efeb02` 对高寄存器区做 producer/consumer fission；`6b5385e` 在
`rhs_advection_kernel` 中一次计算 `betax/y/z`、`dX/dY/dZ`、stencil 系数和
symmetry bounds，再传给各字段 stencil。

**结果与 profile。**

| 指标 | 修改前 | 修改后 |
|---|---:|---:|
| advection Nsys time | `4.021589s` | `3.225269s` |
| registers/thread | 85 | 66 |
| theoretical occupancy | `25.0%` | `37.5%` |
| achieved occupancy | `21.31%` | `32.51%` |
| No Eligible | `65.17%` | `62.27%` |

相对 live-range candidate，`t=5` Program 从 `104.729061s` 降到
`100.296712 +/- 0.088852s`，改善 `4.23%`，checker 3/3 PASS。

**失败对照。** 把 Ricci 按六个输出分量拆成六个 kernel 后，每个分量仍加载完整
Christoffel，中间量没有缩短；register 仍为 `170--201/thread`、occupancy 仍为
`12.5%`，Program 反而为 `110.114760s`。这证明 fission 必须沿数据依赖切，而不是沿
输出列表切。

### 4.3 Ricci 导数共享数据流

**触发证据。** 两个 Ricci connection consumer 各自重复计算
`Gamx/Gamy/Gamz` 的九个一阶导数，分别约 239/241 registers/thread，实际 occupancy
约 `10.9%`。

**优化目的。** 将重复 stencil 变成一个紧凑 producer，使两个 consumer 只保留各自的
contraction，从而同时降低计算量和寄存器生命周期。

**实现。** `6fc58e2` 最终把九个 Gamma 导数 producer 融入已有
`rhs_beta_gamma_kernel` 的尾部，由 Ricci diagonal/off-diagonal 两个 consumer 读取。
没有物化 18 个 lowered Christoffel，因为那会增加 18 次写和两个 consumer 的大量全局读。

**结果。**

- 两个 consumer 从约 240 降到 126 registers/thread，理论 occupancy 从 `12.5%`
  提升到 `25%`；
- derivative + 两个 consumer 每次 RHS 从 `2.191ms` 降到 `1.084ms`，下降
  `50.5%`；
- `t=5` Program 从 `100.296712s` 降到
  `97.693319 +/- 0.285257s`，下降 `2.60%`，3/3 PASS。

独立 producer 与融合到 beta/Gamma 尾部的版本几乎同速；保留融合形态是因为它少一次
launch，而不是宣称这一步本身又有额外加速。

## 5. 主线二：按真实依赖改善调度和 AMR 粒度

### 5.1 用局部 stream wait 替代无条件 device-wide barrier

**触发证据。** 初期 Nsys 有 4110 次 device sync。Stage 1 后同一短 profile 仍有 1044 次，
而不同 Block 的 State/RHS/scratch 实际相互独立；很多长度查询甚至没有发射 kernel 却调用
`cudaDeviceSynchronize`。

**优化目的。** 不改变数学流程，只让 host 等待真正参与当前 ghost exchange、copy 或
analysis 的 stream，释放无关 Block 的并发。

**实现。** `fcbb1bd` 增加 `GPUManager::synchronize_streams()`，由
`gpu_data_packer()` 记录本次 pack/restrict/prolong/unpack 触碰的唯一 stream；
删除 predictor/corrector 后、真正 ghost exchange 前的冗余全设备等待。

**Nsys 证据。**

| 指标 | baseline | 局部同步 | 变化 |
|---|---:|---:|---:|
| device sync calls | 1044 | 215 | `-79.4%` |
| device sync time | `13.955s` | `3.408s` | `-75.6%` |
| device+stream sync time | `13.955s` | `9.145s` | `-34.5%` |
| kernel duration sum | `14.434s` | `14.429s` | 基本不变 |

`t=5` Program 从 `104.962209s` 降到
`100.018310 +/- 0.064500s`，下降 `4.71%`，3/3 PASS。kernel sum 不变正是预期：
收益来自等待范围，而不是算术变少。

### 5.2 用辅助 stream 填充 prolong 小 grid 的尾波

**触发证据。** 原 `prolong3_kernel` 有 47307 次调用，调用之间完全没有 overlap；
典型 grid 约 54 blocks，而 MIG 只有 14 SM，单次 launch 存在明显 partial-wave 尾部。
相反，RHS grid 已足够大且寄存器受限，多开 RHS stream 只会竞争 cache/带宽。

**优化目的。** 只并发 source、destination 和 packed offset 都不相交的不同 AMR 变量，
填充小 prolong launch 的尾波；不跨越 RK、ghost exchange 或 coarse/fine 因果边界。

**实现。** `caf593c` 为每条 parent stream 建两条 auxiliary stream，以 fork/join event
将变量按三路轮转，完成后 join 回原 parent。

**结果。**

- 三路 prolong 的 duration sum 因竞争从 `2.87s` 增到 `3.15s`，但时间并集降到
  `1.497s`；相对串行估计约下降 `31%`；
- 多 kernel overlap 的 wall-time 占比约从 `5.3%` 提高到 `15.5%`；
- `t=5` Program 从 `97.693319s` 降到
  `93.258103 +/- 0.073732s`，下降 `4.54%`，3/3 PASS。

这里必须比较 kernel 时间并集而不是 duration sum；并发使单 kernel 变慢并不表示端到端失败。

### 5.3 AMR 跨变量 batching

**触发证据。** 即使已有三路 stream，prolong/restrict 仍分别有数万次 launch，同一 transfer
segment 的各变量使用相同几何映射，却重复执行 host dispatch 和 index 计算。

**优化目的。** 提高 launch 粒度，将变量维加入 `grid.y`，一次提交同一 segment 的多个
变量；在 host 预计算不随输出点变化的对齐整数。

**实现。** `1a6023b` 建立连续 device descriptor table，
`block=(256,1,1)`、`grid=(ceil(points/256), nvars, 1)`。单变量、UNPACK 和非 AMR
路径保留 fallback。

**结果与 Nsys。**

- AMR launch 数下降 `95.6%`；
- prolong + restrict aggregate time 下降 `64.92%`；
- profile 中 prolong batch 为 `815.499ms/530 calls`，restrict batch 为
  `511.638ms/106 calls`；
- `t=5` Program 从 `93.258103s` 降到
  `89.150199 +/- 0.301396s`，下降 `4.40%`，3/3 PASS。

这一步主要减少调度和重复 geometry/index 工作，尚未消除每线程的
`tmp2[6][6]`、`tmp1[6]`，因此为后续 AMR 数学重写留下了清晰热点。

### 5.4 持久 transfer buffer：作为基础设施保留，不计性能收益

**触发证据。** 单 rank 路径仍频繁 malloc/free packed buffer，Nsys 中阻塞
`cudaMemcpy` API 时间看起来很高。

**优化目的。** 复用 transfer storage，减少 allocator/API 固定成本，并为后续 direct/batched
数据路径建立可控的 buffer ownership；是否能缩短关键路径仍由 E2E 决定。

**实现。** `92a3e13` 按进程维护可增长的 device transfer buffer，并修正
`ensure_on_gpu` 的 ownership 语义。

**结果。** isolated 版本最好为 `81.966115s`，但清理后三次复验为
`82.211364 +/- 0.678122s`，与 `82.166688 +/- 0.084628s` 基线统计持平。
因此它只作为资源生命周期基础设施保留，不能写成端到端加速；候选正确性检查通过。

Nsys 解释了原因：`cudaMemcpy` host API self time 约 `6.3--6.5s`，真实 H2D/D2H
device 时间只有约 `70--240ms`，多数 API 时间在等待前序 kernel。same-level direct、
AMR direct、全 direct 和仅缩小 touched-stream 等候都更慢，后文失败实验表列出。

## 6. 主线三：用空间 tile 消除 stencil 重复读取

前两条主线解决了资源边界和调度粒度，但 profile 仍显示 GPU kernel 几乎填满 Evolve。
真正达标必须降低每个网格点的 stencil 工作量。策略是让一个 thread block 合作加载带 halo
的邻域，然后让相邻输出点从 shared memory 复用；不是简单把所有字段一次装入共享内存。

### 6.1 先在 advection/KO 上验证 compact tile

**触发证据。** Gamma derivative+seed checkpoint（`14f32a9`）后，legacy advection 为
`2.480760s/unit`，是第一热点。NCU 显示
局部数组和 stencil helper 使 local-memory sector 占 L1TEX 的约 `85%`。

**优化目的。** 先在最热、规则且拥有清晰 parity/boundary fallback 的 stencil family 上验证
shared tile 骨架，同时消除相邻线程的重复 radius-3 field load。

**前置低风险优化。** `f244410` 先把 `SoA[3]` 等 helper 局部数组换为标量/同 TU
accessor，使 local-memory sector 占比从约 `85%` 降到 `59%`；Program 从
`89.150199s` 降到约 `84.738476s`。但 registers 从 66 增到 107，说明“local memory
减少”和“registers 更少”不能同时假定。

**tile 实现。** `6ab1f2c` 新增 equatorial 专用 compact kernel：

- block `(8,8,4)`，shared tile `14x14x10`，NCU 报告约 `15.73 KB/block`
  （约 `15.36 KiB`）；
- 24 个字段依次复用同一 tile，在 tile 内完成 lopsided advection 与 KO；
- interior 使用无边界快路径，边界保留 parity、赤道反射和原降阶规则；
- 非 equatorial 模式继续走 legacy 实现。

**结果与 profile。**

| 指标 | legacy | compact |
|---|---:|---:|
| advection aggregate / unit | `2.480760s` | `0.828922s` |
| family speedup | - | `2.993x` |
| Program `t=5` | `82.166688s` | `74.633372 +/- 0.093387s` |
| Evolve `t=5` | `53.366067s` | `45.080233 +/- 0.043878s` |

Program/Evolve 分别下降 `9.17%/15.53%`。最终 NCU 为 78 registers/thread、
`15.73 KB` shared memory、实际 occupancy `35.28%`、单 launch `491.62us`；
三次 checker 均 PASS、trajectory RMS 为 `0`。

**关键反例。** 四字段 cross-tile 把 shared memory 增到 `32.82KiB`、registers 增到
142，实际 occupancy 降到 `12.43%`，单 launch 反而为 `870.53us`；两字段 group、
padding、nested loader 也没有端到端收益。这说明 shared memory 只有在复用收益大于
barrier、bank conflict、容量和寄存器代价时才有效。

### 6.2 Evolution contracted Hessian tile

**触发证据。** advection 降下来后，`rhs_evolution_kernel` 成为第一热点：
`6.5828s/t=0..4`，占 kernel sum `18.3%`。NCU 为 140 registers/thread、
`12.50%/11.00%` occupancy，单个粗层 launch `1.03ms`。

源码中对 6 个 metric field 分别调用通用 `d_fdderivs_point`，每次先构造完整 Hessian，
但 consumer 最终只需要 contracted Hessian。

**优化目的。** 合作加载 radius-2 邻域，按字段直接累加 contraction，避免完整六分量
Hessian 长期存活和相邻线程重复读取。

**实现。** `d4866f1` 在 `hessian_compact_gpu.cuh` 建立
`rhs_evolution_equatorial_compact_kernel`；一个 shared tile 顺序处理字段，legacy
路径保留。

**结果。**

- Nsys aggregate 从 `6.5911s` 降到 `0.6412s`，约 `10.3x`；
- 同一粗层 NCU duration 从 `1.03ms` 降到 `107.36us`；
- registers 从 140 降到 64，理论/实际 occupancy 从 `12.5%/11.0%` 提升到
  `50%/45.79%`；
- `t=5` Evolve 从 `44.4299s` 降到 `38.4472s`，下降 `13.47%`；
- Program 从 `73.715278s` 降到 `67.504175 +/- 0.064103s`，3/3 PASS。

NCU 同时显示 shared load/store 存在约三到四路 bank conflict，但绝对 duration 已大幅下降，
因此没有为了单个规则提示立即加入 padding；实际时间比“指标看起来更漂亮”优先。

### 6.3 Beta/Gamma Hessian tile

**触发证据。** evolution 优化后，beta prepare 变为首要 RHS stencil：
`3.4561s/t=0..4`，NCU 为 140 registers/thread、实际 occupancy `10.93%`、
粗层 duration `528.45us`。

**优化目的。** 复用已经验证的 radius-2 loader，只生成 Gamma consumer 真正需要的
Laplacian、divergence Hessian 和 contraction，缩短完整 Hessian 的 live range。

**实现。** `ebe5f5a` 复用相同 loader，但只发布三个 beta Laplacian、divergence Hessian
和 contracted Gamma 所需摘要，避免全量 Hessian。

**结果。**

- Nsys aggregate 从 `3.4561s` 降到 `0.6717s`，下降 `80.6%`；
- NCU duration 从 `528.45us` 降到 `95.04us`；
- registers 140 -> 72，实际 occupancy `10.93% -> 34.61%`；
- `t=5` Evolve `38.4472 -> 31.8655s`，下降 `17.12%`；
- Program `67.504175 -> 61.457379 +/- 0.056998s`，3/3 PASS。

这里 `No Eligible` 百分比没有随 occupancy 同步下降，说明单一比例指标不能替代 kernel
duration；总指令和重复 stencil 大幅减少后，剩余 kernel 的 stall 构成会改变。

### 6.4 Chi 与 lapse 导数 tile

**触发证据。** beta/evolution 降低后，`rhs_source_chi_hessian` 和
`rhs_source_lapse` 合计约 `3.08s/t=0..4`，成为新的可复制 stencil 模式。

**优化目的。** 把 chi/lapse 的相邻点重复读取转为 block 内复用，并保留精确的对称性和边界
降阶语义，为后续与 source algebra 融合准备紧凑 producer。

**实现。** `a624d51` 为 chi gradient/Hessian 和 lapse gradient/Hessian 建立两个
equatorial compact kernel，使用相同四阶/边界二阶 fallback 和 parity。

**结果。**

| Kernel | 优化前 aggregate | 优化后 aggregate |
|---|---:|---:|
| chi Hessian | `1.5219s` | `0.2704s` |
| lapse | `1.5585s` | `0.3059s` |

合计下降约 `81.3%`。优化后 NCU 分别为 49/51 registers/thread、约 `40%` achieved
occupancy，单 launch `41.98/49.09us`。`t=5` Evolve 从 `25.9956s` 降到
`24.5479s`，Program 从 `54.749915s` 降到
`53.493903 +/- 0.160339s`，3/3 PASS。

### 6.5 AMR contraction 标量化与 prolong shared tile

**触发证据。** 初始 batch prolong/restrict 内仍有
`tmp2[6][6] + tmp1[6]`，NCU 对 prolong 为 93 registers/thread、实际 occupancy
`22.12%`，且提示显式 local-memory 数据流；相邻 fine 输出重复访问高度重叠的
`6x6x6` coarse 邻域。

**优化目的。** 第一阶段删除每线程大临时数组，第二阶段让相邻 fine 输出共享 coarse halo；
二者分别针对 local-memory spill 和跨线程重复 global load。

**第一步：流式 contraction。** `66b560a` 将 prolong 的二维/一维临时数组改为
z->y->x 流式标量收缩，并为 restrict 建专用 fixed-6 scalar helper；host 预计算 coarse/fine
对齐整数，删除每线程 geometry 数组。Nsys 中：

- restrict `2.3327 -> 0.6128s`，下降 `73.7%`；
- prolong `3.4065 -> 2.5296s`，下降 `25.7%`；
- `t=5` Evolve `31.8655 -> 29.3828s`，Program
  `61.457379 -> 58.018595 +/- 0.200208s`。

**第二步：coarse tile。** `0878a0f` 改为 block `(8,8,4)`，合作加载
`10x10x8` coarse tile（x pitch 11），再为 fine 输出做原六点 separable contraction。

- Nsys prolong `2.5296 -> 0.5689s`，下降 `77.5%`；
- 同 workload NCU duration `432.86 -> 86.85us`；
- registers `59 -> 33`，实际 occupancy `45.49% -> 64.83%`；
- `t=5` Evolve `29.3828 -> 25.9956s`，Program
  `58.018595 -> 54.749915 +/- 0.233883s`。

两步均保持原 `C_PROLONG/C_RESTRICT` 系数、even/odd 权重、SoA parity 和边界语义；
两个保留 checkpoint 的三次短窗 checker 均 PASS、trajectory RMS 为 `0`。

## 7. 主线四：裁掉高频通用控制、局部数组和被覆盖的工作

完成主要 stencil tile 后，单纯继续压缩浮点指令的边际收益开始下降。此时 profile 中出现的
热点有一个共同点：它们不一定承担大量物理计算，却在每个 RK stage 或每个变量上反复执行
通用路径、分配、同步、局部数组读写，甚至计算随后会被覆盖的结果。因此下一阶段的目标不是
“把同一计算写得更快”，而是证明哪些控制和计算在当前合法配置下可以不做。

### 7.1 两个黑洞、三个场的 point interpolation 快路径

**触发证据。** compact advection 后的 Nsys 中，`global_interp_kernel` 在
`t=0..4` 仍占 `1.6719s`。源码审查发现，演化阶段每次只查询两个黑洞位置上的三个场，
但通用路径仍执行坐标 H2D、结果 D2H、host owner 选择、临时分配、同步和 MPI reduction。
正式环境只有一个 MPI rank，这些固定成本相对于 6 个标量结果过大。

**优化目的。** 为高频、固定形状的调用建立窄接口，同时保留通用 interpolation 供分析输出
使用；优化的是控制流和调用开销，不改变插值阶数或插值公式。

**实现。** `d861ef3` 在 host 已知坐标上确定本地 owner，复用持久的 6-double device/host
缓冲区，并用 `global_interp_point3_kernel` 一次处理 `2 BH x 3 fields`。非本地 owner、
多 rank 或不满足固定形状的调用仍回退到原路径。

**结果与证据。** Nsys 中通用 `global_interp_kernel` launch 从 `1376` 次降到 `608` 次；
新增 point3 kernel 为 `256` 次、合计仅 `69.44ms`。同日无 profiler control 的 Program 为
`75.818901 +/- 0.605s`，快路径为 `73.715278 +/- 0.830s`；若与较稳定的上一阶段
`74.633372s` 比较，则净改善 `0.918s`、约 `1.23%`。Evolve 从 `45.0802s` 降到
`44.4299s`。由于节点波动会影响这一量级的 Program 差值，报告中应同时给出 launch/profile
证据，不把单次 E2E 差值当作唯一依据。快路径三次 checker 均 PASS、trajectory RMS 为 `0`。

### 7.2 六阶 analysis interpolation 的流式收缩和标量 Neville

**触发证据。** NCU 对原 `global_interp_kernel` 的一次代表性采样为：

- duration `2.87ms`，64 registers/thread，实际 occupancy `43.69%`；
- `98.82%` 的 L1TEX sectors 来自 local memory；
- local load/store 每个 32-byte sector 只使用约 1 byte，表明 `ya[216]`、
  `yatmp[36]` 等线程私有数组已落入 local memory，并产生极差的 transaction 利用率。

**优化目的。** 保持原六点、三维 separable interpolation 与 Neville 递推顺序，删除线程私有
大数组造成的 local-memory 流量。这里的“streaming”指 z->y->x 流式数值收缩，不是 CUDA
stream 并发。

**实现。** `11c6356` 先将固定六阶路径改为逐维流式收缩，只保存当前维需要的结果；
`f0e1d91` 又把六点 Neville helper 展开为显式标量 `c/d` 状态并控制 inlining，消除动态索引
数组。对称性、边界选点和 fallback 均保持不变。

**结果与解释。** 第一版 duration 降到 `2.02ms`，但 registers 上升到 164，实际 occupancy
降到约 `12%`；标量版进一步降到 `1.77ms`，相对原始 `2.87ms` 下降 `38.3%`，但仍有
236 registers/thread、`12.5%` 理论 occupancy。local-memory sector 占比降到 `61.35%`，
local load 利用率提升到 `12.4/32 bytes`。这再次说明 occupancy 不是最终目标：即使 occupancy
下降，只要消除的低效 local-memory transaction 更多，kernel 仍可更快。

这两步没有各自独立的三次无 profiler `t=5` 归因。包含 interpolation 与 7.4 所述 predictor
constraint 裁剪的联合 checkpoint 从 Program `51.584440s` 降到 `49.954786s`，Evolve
从 `22.9206s`
降到 `19.8344s`。因此报告应把 kernel 级 NCU 作为 interpolation 成功的直接证据，不能把
整个 E2E 收益全部归给它。流式和标量实现均通过短窗 checker；联合 checkpoint 3/3 PASS。

### 7.3 裁剪 output-only RHS 的无关阶段

**触发证据。** 输出 constraints 时原代码仍执行完整 RHS pipeline；但约束公式只依赖
geometry、Ricci、Gamma/beta derivative 等 producer，不消费 source、gauge、advection 或
演化 RHS 的最终值。Nsys 也显示输出点会额外重复这些 kernel。

**优化目的。** 为“只需要 constraints”这一调用建立显式 mode，让依赖图决定需要执行的
producer，避免先产生再丢弃整组 RHS。

**实现。** `05f0df8` 引入 constraint-only RHS mode：保留构造 constraints 所需的 geometry、
Ricci、Gamma、beta 和 derivative 阶段，跳过与约束无关的 source/gauge/advection 计算；
普通 RK stage 不受影响。

**结果。** Nsys 中 source/advection 类调用数从 `420` 降到 `392`，constraint 调用仍为
`126`；Program 从 `84.7226s` 降到 `83.529860s`，约 `1.4%`，3/3 PASS。后续
`adeca02` 是这一思路的进一步扩展：不仅裁掉 output-only RHS 的无关 producer，还裁掉
会被 output-only RHS 覆盖的 predictor constraint。

### 7.4 跳过随后必然被覆盖的 refined predictor constraints

**触发证据。** Nsys 中 constraint kernel 在 `t=0..4` 有 `447` 次调用、耗时 `1.376s`。
沿调用链检查后发现：predictor `co=0` 会为所有 refinement level 计算 constraints，但 refined
level 的这些结果在没有 consumer 读取前，就会被前一节 `Constraint_Out` 的 constraint-only
RHS 重新计算并覆盖。

**优化目的。** 删除可以由数据流证明无观察者的工作，而不是放宽或跳过正式约束检查。

**实现。** `adeca02` 在 predictor 阶段只为 level 0 计算 constraints；输出阶段仍对所有 refined
levels 运行原 constraint-only 路径。正式输出的 9 个 AMR level 和约束文件保持完整。

**结果。** 同一 profile 区间内 constraint 调用从 `447` 降到 `59`，耗时从 `1.376s`
降到 `0.152s`，约下降 `89%`；后续 `t=1` profile 中为 29 次。它与 analysis
interpolation 优化共同包含在 Program `51.584440 -> 49.954786s`、Evolve
`22.9206 -> 19.8344s` 的 checkpoint 中，所以只报告联合 E2E 收益。该 checkpoint
3/3 checker PASS；最终正式运行的 9 层约束输出结构完整，checker 对 level 0 的四类约束
均判定 PASS，证明没有把评分工作删掉。

### 7.5 跨变量批处理 corrected AMR boundary copies

**触发证据。** Sommerfeld corrected boundary 对每个 field 单独 dispatch，而同一 AMR block
的区域、索引和多数 metadata 相同。tile 优化后，这些小 kernel 的 launch/control 开销开始
具有可见比例。

**优化目的。** 共享边界遍历和 launch 固定成本，同时保留每个 field 自己的传播速度、边界
属性和不兼容情况 fallback。

**实现。** `75543e3` 将兼容变量的指针和 metadata 组成 batch，由
`sommerfeld_correct_batch_kernel` 一次处理；无法组成同质 batch 时仍调用原实现。

**结果。** `t=5` Evolve 从 `24.5479s` 降到 `23.2508s`，下降 `5.28%`；Program 从
`53.493903s` 降到 `51.875333 +/- 0.203747s`，下降 `3.03%`，3/3 PASS。该提交没有
紧邻的独立 Nsys before/after；后续 fusion 前 profile 中 batch kernel 为 `388` 次、
`51.299ms/t=0..4`。因此可以确认最终 batch 成本很小，但不能从这一后测值精确反推原 launch
成本，报告中不应制造不存在的 profile 对照。

## 8. 主线五：在数据流已经变窄后做受控融合

前四条主线先降低寄存器、重复 stencil、launch 数和无效工作，最后才适合融合。融合的统一
判据不是“kernel 越少越好”，而是相邻 producer/consumer 是否满足：同一线程同一 `idx`
消费、无跨 block 依赖、可消除明确的全局 scratch 往返，并且融合后 live range 可控。每次均
保留 legacy/fallback 路径，且用 Evolve 与 profile 判断是否出现寄存器回退。

### 8.1 融合 Gamma derivative 与 seed

**触发证据。** Gamma derivative producer 与三个 seed consumer 严格相邻，consumer 只读取
同一 `idx` 的六个导数 scratch；旧路径合计约 `0.311601s/unit`。

**实现与目的。** `14f32a9` 在同一线程内直接把六个导数传给 seed algebra，消除六次 scratch
写和六次读，同时保持 derivative 公式、parity 和边界 fallback。

**结果。** 融合后为 `0.168677s/unit`，下降 `45.8%`；Program
`83.529860 -> 82.166688 +/- 0.0846s`，下降 `1.63%`；对应 Evolve 为
`53.366067s`，全部 checker PASS。

### 8.2 融合 metric 与 A 方程

**触发证据。** tile 后旧 metric、A-diagonal、A-offdiagonal 三组 kernel 在 `t=0..4`
分别为 `0.2607s`、`0.7842s`、`0.7799s`，合计 `1.8248s`，即约
`0.4562s/unit`。三组都重复读取 9 个 beta gradient、metric 与 A 分量。

**优化目的。** 在一次加载中完成 metric 和 A 的更新，消除重复全局读取和中间 scratch；
同时避免把更早的 geometry/Ricci 也并入，控制寄存器生命周期。

**实现。** `557d397` 新增 equatorial fused path。为避免 in-place 写 metric 影响后续 A 公式，
先加载并保存全部必要 derivative，再发布结果；其他 symmetry 走 legacy 路径。

**结果。** 后续 profile 中 fused kernel 为 `0.5125s/t=0..4`，即 `0.1281s/unit`，相对
旧三组下降 `71.9%`。Evolve `23.2508 -> 22.9206s`，下降 `1.42%`；Program
`51.875333 -> 51.584440 +/- 0.091932s`，下降 `0.56%`，3/3 PASS。kernel 收益没有
一比一反映到 Program，是因为这一阶段 Program 已含约 29 秒固定分析/输出成本。

### 8.3 融合 chi/lapse derivative 与下游 source algebra

**触发证据。** compact tile 后 source-metric、physical-Gamma、chi-Ricci、chi-Hessian、
lapse 五组相邻 kernel 合计 `2.208929s/t=0..4`，约 `0.552232s/unit`。下游代数马上
消费 chi/lapse gradient 和 Hessian 摘要，写入全局 scratch 后再读没有长期价值。

**优化目的。** 在 shared tile 中完成 chi/lapse derivative 后直接执行 source algebra，避免
中间结果落到全局内存，同时共享 inverse metric 等公共量。

**实现。** `b6217ff` 建立受控 fused equatorial kernel，沿用 compact helper 的四阶内部点、
二阶边界 fallback 和 parity；不满足条件时回退旧 pipeline。

**结果。** fused aggregate 为 `0.874053s/t=0..4`，即 `0.218513s/unit`，相对完整五组
下降 `60.43%`。
相对上一生产 checkpoint，Program `49.954786 -> 46.898728 +/- 0.533196s`，下降
`6.12%`；Evolve `19.8344 -> 18.1178s`，下降 `8.65%`，3/3 PASS。这是后期少数
在 kernel 和 E2E 两侧都非常清晰的融合。

### 8.4 融合 geometry 与 Ricci-A

**触发证据。** 原 geometry 与 Ricci-A 在 `t=0..4` 分别约 `2.3071s` 和 `0.2191s`，
合计约 `0.6316s/unit`。Ricci-A 立即消费 geometry 阶段产生的 metric gradient、connection
和 Ricci 摘要。

**优化目的。** 让 shared tile 中已经得到的 metric derivative 直接进入 geometry/Ricci-A
代数，删除不必要的全局摘要往返，并把 connection 的 live range 限制在一个阶段内。

**实现。** `d4a05db` 建立 compact fused path，按 field 顺序复用 tile，最终 profile 中为
`0.31435s/unit`，相对旧组合下降约 `50.2%`。

该提交没有独立三次无 profiler `t=5`；它与下一节首版 beta/Gamma fusion 共同形成 checkpoint。
因此只能把 `0.31435s/unit` 的 profile 降幅直接归给 geometry/Ricci-A，不能拆分联合 E2E
差值。联合 checkpoint 3/3 checker PASS、trajectory RMS 为 `0`。

### 8.5 融合 beta producer/Gamma consumer，并继续 tile Gamma derivative

**触发证据。** beta compact prepare 与 Gamma consumer 严格相邻，consumer 只读取同一
`idx` 的 9 个摘要；旧 prepare `0.15999s/unit`、consumer `0.23692s/unit`，合计
`0.39691s/unit`。源码依赖审计确认 9 个 producer 摘要由同一 `idx` 的 consumer 使用；
consumer 自身仍会为 Gamma 邻点导数执行 stencil。consumer 写回的 9 个 Gamma derivatives
才是后续 Ricci 真正需要的 scratch。

**优化目的。** 先消除同线程 producer/consumer 的 9 组 scratch 往返，再复用已有 beta tile
生成三个 Gamma field 导数；分两步控制寄存器回退和正确性定位范围。

**第一步。** `bcb1c39` 将 beta 的 Laplacian、divergence Hessian、contracted Gamma 放入
block shared memory 的 `prepared[9][tid]` 每线程独占 slot，直接送入 consumer，消除
producer/consumer 的中间全局往返，但先保留三个 Gamma field 的原导数 helper。包含前一节
geometry/Ricci-A 的 checkpoint 为：Program
`46.898728 -> 46.473755 +/- 0.116s`，Evolve `18.1178 -> 17.8601s`。由于包含两个
提交，不能把全部收益归给 beta fusion。

**第二步。** `793768d` 复用 beta shared tile，依次计算 `Gamx/Gamy/Gamz` 一阶导数；每次
换 field 前所有线程共同同步，parity 保持 `(-,+,+)/(+,-,+)/(+,+,-)`。两步完成后的 fused
group profile 为 `0.215236s/unit`，相对旧两组下降 `45.8%`。这是
`bcb1c39 + 793768d` 的累计结果，且 before 使用 `t=0..4`、after 使用 `t=0..1` 后按
unit 归一化，不能解释成 `793768d` 单步的独立 profile 收益。

Evolve 从 `17.8601s` 降到 `17.4100s`，下降 `2.52%`；Program 却从
`46.473755s` 波动到 `46.709760s`。这一步被保留的依据是重复 Evolve 与 Nsys kernel
证据，而不是一次 Program 均值。它是报告中应明确说明的例子：固定分析阶段的噪声可能掩盖
演化段的真实改善。首版联合 checkpoint 与 tiled Gamma checkpoint 均为 3/3 checker PASS、
trajectory RMS `0`。

### 8.6 将 gauge 基值折叠进 advection

**触发证据。** fusion 前 profile 中独立 gauge 为 `0.034784s/unit`。其输出的 7 个 gauge
RHS 随后被 advection 逐点读取并加上 transport/KO；两者之间没有其他 consumer。

**优化目的。** 不再把 7 个 gauge 基值先写到全局内存再读回，而是在 advection 已加载本点
状态时直接生成。重点是消除数据往返，单纯省一个 launch 并不是主要收益。

**实现。** `fae118c` 在进入 Gamma advection 前保存必要的 pre-advection Gamma RHS，随后在
compact advection 内生成 fields 17--23 的 7 个 gauge bases。`valid && !active` 边界仍显式
写 gauge RHS；非 equatorial symmetry 保留独立 kernel。

**结果。** Evolve `17.4100 -> 17.1363s`，下降 `1.57%`；Program
`46.709760 -> 45.759764 +/- 0.305619s`，下降 `2.03%`，3/3 PASS。该提交之后没有单独
Nsys 重采样，因此直接证据是优化前热点消失的依赖证明加 E2E；报告不能声称有不存在的
post-profile 数值。

### 8.7 将 trace/source 代数折叠进 metric/A

**触发证据。** fusion 前 trace kernel 为 `0.061723s/unit`，它与后续 metric/A 重复读取
inverse metric 和 A 分量；beta gradients 只由 metric/A 使用，不属于两者的共享输入。
前面 metric/A fusion 已把相关输入集中到单个 kernel，具备最后一次窄融合条件。

**优化目的。** 复用同一组 `Aij` contraction，一次产生 `trA2`、stress、`f_trace`、`trK`
以及 metric/A RHS，删除独立 trace launch 和下游对 `f_trace` 的全局回读，同时不把
geometry 或其他 stencil 再并入；为兼容 scratch 生命周期仍写回 `Gmz_Res[idx]`。

**实现。** `0e4551a` 用 shared volatile `terms[6][256]` 分段保存公共 contraction，控制编译器
寄存器 live range，先读取会被覆盖的 9 个 beta gradient 和相关 scratch，再发布所有结果；
legacy 路径保留。

**结果。** Evolve `17.1363 -> 16.9007s`，下降 `1.38%`；Program
`45.759764 -> 45.658880 +/- 0.044578s`，均值改善仅 `0.22%`，低于前测标准差
`0.305619s`，不能声称 Program 收益具有统计显著性。保留依据主要是稳定的 Evolve 改善与
依赖证明，三次 checker 均 PASS。正式 `t=100` 的代码就是该提交；最终没有追加一轮 Nsys，
因此不虚构 post-profile。

## 9. 重新组织后的性能演进

上文按技术主线重组优化；下表则按保留版本的实际性能演进排序，方便追踪数字，并尽量选用
相邻、三次无 profiler 的生产 run checkpoint。一些试验在同一 checkpoint 内完成，所以用
“联合”明确
标注。Program 是完整
`This Program Cost`，Evolve 是其中演化阶段。`-` 表示没有可可靠配对的阶段值，不用其他
日期或不同机器的数字补齐。

| 阶段 | 代表提交/检查点 | Program (`t=5`, s) | Evolve (`s`) | 主要结论 |
|---|---|---:|---:|---|
| 初始 GPU baseline | baseline | `122.491884 +/- 1.557112` | `92.1168` | RHS 占 77.3% |
| 拆分巨型 RHS | `81cf518` | `101.667` | `-` | RHS family -24.7% |
| 收窄同步 | `fcbb1bd` | `100.018` | `-` | device sync 13.955 -> 3.408s |
| Ricci 共享数据流 | `6fc58e2` | `97.693` | `-` | 240 -> 126 regs，kernel -50.5% |
| prolong 尾波并发 | `caf593c` | `93.258` | `-` | critical-path union 缩短 |
| AMR 跨变量 batch | `1a6023b` | `89.150` | `-` | AMR kernel time -64.9% |
| helper 标量化 | `f244410` | `84.738` | `-` | local sectors 85% -> 59% |
| constraint-only RHS | `05f0df8` | `83.529860` | `-` | 裁掉输出无关阶段 |
| Gamma derivative+seed | `14f32a9` | `82.166688` | `53.3661` | group -45.8% |
| compact advection | `6ab1f2c` | `74.633372` | `45.0802` | family 2.99x |
| BH interpolation 快路径 | `d861ef3` | `73.715278` | `44.4299` | 通用 launch 1376 -> 608 |
| evolution Hessian tile | `d4866f1` | `67.504175` | `38.4472` | kernel 10.3x |
| beta Hessian tile | `ebe5f5a` | `61.457379` | `31.8655` | aggregate -80.6% |
| AMR scalar contraction | `66b560a` | `58.018595` | `29.3828` | restrict -73.7% |
| prolong shared tile | `0878a0f` | `54.749915` | `25.9956` | prolong -77.5% |
| chi/lapse tile | `a624d51` | `53.493903` | `24.5479` | 两组合计 -81.3% |
| corrected-boundary batch | `75543e3` | `51.875333` | `23.2508` | Program -3.03% |
| metric+A fusion | `557d397` | `51.584440` | `22.9206` | kernel group -71.9% |
| analysis + constraint 裁剪（联合） | `f0e1d91` | `49.954786` | `19.8344` | 不能拆分归因 |
| chi/lapse/source fusion | `b6217ff` | `46.898728` | `18.1178` | Evolve -8.65% |
| geometry/Ricci + beta/Gamma（联合） | `bcb1c39` | `46.473755` | `17.8601` | 联合 checkpoint |
| tiled Gamma derivative | `793768d` | `46.709760` | `17.4100` | Evolve 改善，Program 有噪声 |
| gauge -> advection | `fae118c` | `45.759764` | `17.1363` | Evolve -1.57% |
| trace -> metric/A | `0e4551a` | `45.658880 +/- 0.044578` | `16.9007` | 最终 `t=5` 候选 |
| 正式 `t=100` | `0e4551a` | `361.511935 +/- 2.059365` | `331.870` | 3/3 PASS，达到目标 |

初始到最终 `t=5`，Program 加速 `2.6828x`、降低 `62.725%`；Evolve 加速
`5.4505x`、降低 `81.653%`。不能用这组 `t=5` 比率替代正式 `t=100` 成绩。最初完整
`t=100` baseline 因单次超过 30 分钟没有完成，文档中的约 `1872.71s` 只是基于短程比例的
乐观外推；若要与最终正式结果比较，必须写成“估算约 `5.18x`”，不能写成实测 speedup。

## 10. Profile 闭环：热点如何被逐层压平

初始 Nsys 的 GPU 时间非常集中：RHS `56.906s/77.3%`、prolong
`8.8696s/12.1%`、restrict `2.9178s/4.0%`、global interpolation
`2.0093s/2.7%`。初始 RHS 的 NCU 为 250 registers/thread，理论/实际 occupancy
`12.5%/11.04%`，`No Eligible` `75.3%`，而 compute 与 DRAM throughput 仅
`22.26%/3.97%`。这说明问题不是带宽饱和，而是巨型数据流、低可调度 warp、重复 stencil
和控制固定成本的组合。

在最后两次小融合之前的统一 `t=1` Nsys 中，主要演化 unit 已被压平为：

| Kernel/group | 每 unit 时间 |
|---|---:|
| compact advection/KO | `0.841744s` |
| analysis global interpolation | `0.360399s` |
| fused geometry/Ricci-A | `0.314353s` |
| RK update | `0.241955s` |
| fused chi/lapse/source | `0.220283s` |
| fused beta/Gamma | `0.215236s` |
| Gamma derivative/seed | `0.177602s` |
| evolution contracted Hessian | `0.167040s` |
| restrict | `0.144243s` |
| prolong | `0.142331s` |
| fused metric/A | `0.128130s` |
| constraints | `0.073921s` |
| trace（随后被融合） | `0.061723s` |
| gauge（随后被融合） | `0.034784s` |

这个终局 profile 有两层意义。第一，初始单个巨型 RHS 和 AMR 热点已变成一组相近的中小
kernel，不再存在一个能靠单点改写获得几十秒的瓶颈；第二，它直接给出最后两次融合的来源，
即独立 trace 和 gauge。由于 `fae118c`、`0e4551a` 后未重新采 Nsys，最终报告应把这张表称为
“最终两次融合前的 profile”，而不是“最终 profile”。正式 endpoint 的证据是 artifact 中
记录 HEAD 为 `0e4551a` 的三次完整运行。

## 11. 正式结果与正确性

正式 artifact 为 `profile/gpu-benchmark-20260827T152719Z-64`，代码提交为 `0e4551a`。
运行配置为 A100 80GB 的 MIG `1g.10gb` 实例、16 CPU、24 GiB memory、单 MPI rank、
`t=100`、9 层 AMR、Courant factor `0.5`、四阶差分、项目原
`runge-kutta-45` 配置（GPU 代码的四个 RK4 substep 未变）与 FP64。

| Run | Evolve (`s`) | This Program Cost (`s`) | Checker |
|---:|---:|---:|---|
| 1 | `331.528` | `360.8648648262` | PASS |
| 2 | `330.663` | `359.8538150787` | PASS |
| 3 | `333.419` | `363.8171262741` | PASS |
| Mean | `331.870` | `361.5119353930` | 3/3 PASS |
| Sample stddev | `1.409` | `2.059365` | - |

三次运行均匹配 `100/100` 个 trajectory 时刻，trajectory RMS 为 `0`，trajectory 有效
比较项为 `596`。约束文件的 9 层输出结构完整；checker 对 level 0 的 Hamiltonian、
momentum-x、momentum-y、momentum-z 做阈值判定并全部通过。代表性 level-0 最大值分别为
`0.037260575`、`0.017252719`、`0.017360407`、`0.037598783`，远低于 checker 阈值
`2`。四个必需输出文件均存在，峰值显存约 `2063 MiB`。

这些结果证明的是：同一正式输入、同一数值精度和同一检查流程下，三次完整 Program cost
都小于 `370s`。报告不应只写均值，因为目标是单次端到端阈值；这里最慢一次
`363.8171s` 仍有约 `6.18s` 余量。

## 12. 失败或不稳定尝试：它们如何收紧优化原则

失败实验不是附带噪声，而是路线形成的重要证据。下表只保留能改变后续决策的代表性试验。

| 尝试 | 观察结果 | 为什么拒绝 / 学到什么 |
|---|---|---|
| 按输出 component 拆 Ricci | Program `110.115s`，原为 `104.729s` | 寄存器未降，重复公共量；拆 kernel 必须同时缩短数据流 |
| geometry/equation 粗粒度 fission | 约慢 `13.2%` | 额外全局中间量和 launch 大于 occupancy 收益 |
| RHS 九阶段 fission | 约慢 `4.6%` | “kernel 更小”不等于关键路径更短 |
| 强制 inline 标量 helper | `90.638s`，原为 `84.738s` | inline 拉长 live range；编译器边界需要实测 |
| 交叉 field shared tile | 代表 kernel `870.53us`，明显回退 | shared memory 容量、同步和 bank conflict 可抵消复用 |
| RK 跨变量 batch | `90.155s`，原为 `89.150s` | 小 kernel launch 不是当时主瓶颈，batch 增加寄存器/分支 |
| 通用 mempool | `93.565s`，原为 `89.150s` | API time 下降不代表 GPU critical path 下降 |
| touched-level auxiliary streams | `82.833s`，原为 `82.167s` | 并发工作互相争抢，不能只看 overlap 图 |
| direct same/full/AMR-only 变体 | `84.510/85.415/82.201s`，均无稳定收益 | 去 staging 必须连同访问合并和生命周期一起设计 |
| 将 Lap/trK 额外写 scratch | Evolve `3.52228s`，基线 `3.51940s` | 虽正确且复用 tile，但 6 写+6 读抵消 stencil 收益 |
| advection padding/cross/slab 变体 | 无稳定改善或回退 | bank-conflict 提示不等于值得增加 shared footprint |
| persistent transfer buffer | 性能中性 | 可作基础设施保留，不能包装成性能优化 |

由此得到五条贯穿全文的原则：

1. occupancy 是诊断量，不是目标；最终看 duration、Evolve 和 Program。
2. 只有缩短关键路径的并发才有价值，kernel duration sum 甚至可能因竞争上升。
3. shared tile 必须以 consumer 所需摘要为边界，不能为了“复用”保存全量中间结果。
4. launch、allocation 或 API time 的减少只有在端到端可见时才计收益。
5. 小优化必须 A/B 重复测量；若 Program 噪声掩盖结果，至少要有 Evolve 和 kernel profile
   两种一致证据，否则回退。

## 13. 建议的最终实验报告叙事

正式报告篇幅有限时，不必逐个复述所有提交。建议保留以下主线：

1. 用 baseline Nsys/NCU 证明巨型 RHS、AMR 和低 occupancy/live range 是首要问题。
2. 用 RHS fission、同步范围缩小和 AMR batching 说明先建立可调度结构。
3. 把 compact advection、evolution/beta/chi/lapse tile 作为最大的一组数值 kernel 优化，给出
   registers、occupancy、kernel duration 和 E2E 四级证据。
4. 用 interpolation local-memory、constraint 覆盖关系说明后期瓶颈已从算术转为通用控制和
   冗余工作。
5. 把若干融合归为“producer-consumer 数据流融合”，重点讲 metric+A、chi/lapse/source 和
   beta/Gamma 三个有代表性的例子；小型 gauge/trace 融合可合并成一段。
6. 最后给三次正式结果、完整 checker、失败实验与剩余热点，说明结果可复现且没有通过更改
   数值问题、精度或检查器获得。

建议至少制作四张图或表：baseline GPU time breakdown、一个代表性 NCU before/after（推荐
evolution tile）、`t=5` Program/Evolve 阶段演进、正式三次 `t=100` 结果与 checker。图中应
标注测量区间和提交，避免混用 `t=1`、`t=5`、`t=100`。

## 14. Artifact 与复现索引

### 14.1 关键源码位置

| 优化范围 | 主要源码 |
|---|---|
| RHS 拆分、依赖裁剪和受控 fusion | `src/bssn_rhs_gpu.cu`、`src/bssn_rhs.h` |
| compact advection/KO 与 gauge folding | `src/advection_compact_gpu.cuh` |
| evolution/beta/chi/lapse/geometry tiles | `src/hessian_compact_gpu.cuh` |
| AMR batching、stream 调度 | `src/Parallel_GPU.cpp`、`src/gpu_manager.cu`、`src/gpu_manager.h` |
| prolong/restrict 数学与 tile | `src/prolongrestrict_cell_gpu.cu` |
| 通用/BH interpolation | `src/fmisc_gpu.cu`、`src/bssn_gpu_class.C`、`src/Parallel_GPU.cpp` |
| corrected boundary batching | `src/bssn_step_gpu.C`、`src/sommerfeld_rout_gpu.cu` |
| 构建和正式运行 | `CMakeLists.txt`、`hpc_gpu_benchmark.sh` |

### 14.2 主要 benchmark/profile artifact

下表用于核对正文数字。当前仍保留的 benchmark 目录内有三次 run log、`summary.txt` 和
`check-{1,2,3}.txt`；最早两阶段的部分大体积目录已清理，只保留阶段文档中的路径和汇总。
NCU/Nsys 的具体采样 launch 和窗口还应结合相应阶段文档阅读。

| 阶段/提交 | 生产 benchmark | profiler 或详细索引 |
|---|---|---|
| 初始 baseline | 历史路径 `profile/gpu-benchmark-20260821T133110Z-64`（目录已清理） | Nsys/NCU 历史路径及汇总见 `docs/gpu-baseline-profile.md` |
| RHS 拆分 `81cf518` | 历史路径 `profile/gpu-benchmark-20260824T094429Z-64`（目录已清理） | Nsys/多个 NCU 路径及汇总见 `docs/gpu-stage1-rhs.md` |
| Ricci/stream `6fc58e2..caf593c` | `gpu-benchmark-20260826T022748Z-65`、`gpu-benchmark-20260826T033210Z-61` | `docs/gpu-rhs-ricci-dataflow-fission.md`、`docs/gpu-stage3-execution-graph.md` |
| AMR batch `1a6023b` | `profile/gpu-benchmark-20260826T093400Z-65` | Nsys `gpu-nsys-20260826T092137Z-66`；NCU `gpu-ncu-stage4-precomputed/full.ncu-rep` |
| compact suite `6ab1f2c` | `profile/gpu-benchmark-20260827T083647Z-63` | Nsys `gpu-nsys-20260827T075056Z-66`；NCU `gpu-ncu-20260827T083353Z-64`；`docs/gpu-optimization-report.md` |
| BH/evolution/beta tiles | `gpu-benchmark-20260827T095539Z-62`、`T101248Z-64`、`T110137Z-65` | commits `d861ef3`、`d4866f1`、`ebe5f5a` |
| AMR/chi/lapse tiles | `gpu-benchmark-20260827T112116Z-61`、`T113948Z-63`、`T115638Z-63` | commits `66b560a`、`0878a0f`、`a624d51` |
| boundary/metric/analysis/chi fusion | `gpu-benchmark-20260827T121442Z-63`、`T122726Z-63`、`T130116Z-63`、`T134341Z-63` | commits `75543e3..b6217ff` |
| 最后三组 fusion | `gpu-benchmark-20260827T143825Z-64`、`T145005Z-66`、`T150636Z-63`、`T151810Z-64` | fusion 前统一 Nsys `gpu-nsys-20260827T145807Z-65`；commits `bcb1c39..0e4551a` |
| 正式 `t=100` | `profile/gpu-benchmark-20260827T152719Z-64` | artifact 记录 HEAD `0e4551a`，三次完整 checker |

表中同一日期的缩写 `T...` 仍位于 `profile/gpu-benchmark-20260827...` 下；实际引用时建议
写完整目录名。后期 NCU/Nsys 的单项数据和失败候选很多，完整台账保存在
`docs/gpu-370-optimization-roadmap.md` 与各阶段文档中，不在此重复所有文件。

### 14.3 总入口

- 正式完整运行：`profile/gpu-benchmark-20260827T152719Z-64`
- 最终源码提交：`0e4551a`
- 370 秒结果 roadmap 提交：`57f89f9`
- 本报告提交：运行 `git log -1 -- docs/gpu-final-optimization-reference.md` 查询
- baseline/profile 总结：`docs/gpu-baseline-profile.md`
- 优化路线与详细实验索引：`docs/gpu-370-optimization-roadmap.md`
- 实验规范：`docs/Lab4-AMSS-NCKU/index.md`

部分 profile artifact 是在候选代码仍为 dirty worktree、随后立即提交的流程中生成，因此 metadata
可能记录父提交。引用时以 artifact 时间戳、相应 commit 和输出内容三者共同定位。正式 artifact
记录的 HEAD 是 `0e4551a`；根据当时执行记录，该任务从 clean worktree 提交，但 artifact 脚本
本身没有保存 `git status` 或源码/executable hash，不能仅靠 artifact 自证 clean。后续复现实验
应补充这些字段。

---

本文件只作为内部技术参考。实验文档明确禁止直接提交 AI 生成的报告文本；请依据实际代码、
profile 和你自己的理解重新组织、重写，并自行完成实验文档中的思考题。
