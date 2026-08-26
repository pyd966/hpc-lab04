# ABEGPU 基准、Profile 与优化路线

> 日期：2026-08-21
> 基线源码：`7ecaa99fa4e74e464d207da7562b16d213423dda`，加上本报告配套的 GPU 运行脚本和输入切换
> 范围：单个 A100 80GB PCIe 的 MIG `1g.10gb` 实例，1 个 MPI rank；主演化流程使用 `ABEGPU`
> 排除项：本文不重复分析 TwoPuncture 的内部算法，只说明如何接入远程已优化版本以及它与端到端计时的边界。

## 1. 结论摘要

当前 GPU baseline 的主要问题不是 PCIe 带宽，也不是分支发散，而是以下三项叠加：

1. `rhs_kernel` 占 Nsight Systems 所见 GPU kernel 时间的 77.3%。它每线程使用 250 个寄存器，理论 occupancy 只有 12.5%，实测 11.04%。
2. 主机侧存在大量全设备同步。短窗 profile 中 `cudaDeviceSynchronize` 共 4110 次，API 等待时间 71.57 s，占 CUDA API 时间的 96.0%。这些时间主要是在等待前述 kernel，不应与 kernel 时间重复相加。
3. AMR 的 prolong/restrict、边界和 RK4 更新被拆成大量小 launch。短窗共观察到 201694 次 `cudaLaunchKernel`；其中 prolong 和 restrict 分别调用 47307 和 10047 次。

优先顺序应当是：先控制 RHS 的寄存器生命周期和依赖延迟，再缩小全局同步范围，然后批处理 AMR/逐变量小 kernel。传输异步化、内存池和编译组织属于后续工作。

正式输入要求演化到 `t=100`。当前基线在 30 分钟任务上限内无法完成：正式尝试运行到 `t=9` 后，第 1 至 9 个物理时间步分别耗时约 17.86 至 20.26 s。基于短窗三次均值的乐观线性外推约为 1873 s，即 31.2 分钟，且正式运行已经显示后续步变慢。因此本文把可复现的 `t=5` 三次结果作为优化前 baseline，同时保留正式任务脚本用于最终验收。

## 2. 远程 TwoPuncture 优化的接入结论

本地原先位于 `cf7e7a8`，远程 `origin/master` 已快进到 `7ecaa99`。与可执行行为直接相关的最小文件集是：

| 文件 | 必须一起保存的原因 |
| --- | --- |
| `src/TwoPunctures.C` | workspace 复用、谱系数缓存和 OpenMP 热路径都在此实现 |
| `src/TwoPunctures.h` | 新增的长期 workspace/缓存成员与析构接口必须匹配实现 |
| `CMakeLists.txt` | 独立的 `-O3 -march=native` 与 `AMSS_ENABLE_TWOPUNCTURE_OPENMP` 构建和链接设置 |

不能只复制 `TwoPunctures.C`：这样会造成类布局不匹配或未链接 OpenMP。若从旧分支挑提交，源码优化序列是 `aa897d0`、`90a3de1`、`281d65a`、`f61e652`、`f7814c7`；`b617fea` 只固定 benchmark 脚本的严格 `-O3`，不是运行时源码依赖。

远程严格 `-O3 -march=native` 的复测结果是：30/60 OpenMP 线程分别 11.399/11.091 s，输出逐位一致。这个约 11 s 的数字来自 30 个物理核、60 个逻辑 CPU 的 CPU allocation。GPU allocation 给出的 16 个逻辑 CPU 实际只覆盖 8 个物理核，因此不能期待相同时间。当前 GPU 短窗中，`This Program Cost - Total Running Time` 的均值约 26.88 s；它包含 TwoPuncture、参数更新和进程衔接，不能视为纯 TwoPuncture 精确计时，但足以说明 GPU 节点的 host 配额不同。

远程附带的 `docs/twopuncture-*.md` 和 `hpc_twopuncture_*.sh` 对复现有用，但直接运行优化版本并不依赖这些文档和专项 profile 脚本。

## 3. 固化的运行与分析入口

三个脚本把正式提交、重复计时和 profiler 分开，避免不同目的互相污染：

| 用途 | 命令 | 行为 |
| --- | --- | --- |
| 正式单次运行 | `hpc submit ./hpc_gpu_run.sh` | 构建后执行固定 `t=100`，禁用 TwoPuncture cache，运行 checker |
| 正式重复计时 | `hpc submit -e AMSS_BENCHMARK_RUNS=3 ./hpc_gpu_benchmark.sh` | 重复官方端到端计时，记录 TSV、均值和样本标准差 |
| 短窗 baseline | `hpc submit -e AMSS_BENCHMARK_RUNS=3,AMSS_BENCHMARK_TIME=5 ./hpc_gpu_benchmark.sh` | 只用于开发和趋势比较，checker 对 golden 前缀 |
| VTune | `hpc submit -e AMSS_PROFILE_TOOL=vtune ./hpc_gpu_profile.sh` | 缓存并在测量区外准备初值，只 profile ABEGPU host 侧 |
| Nsight Systems | `hpc submit -e AMSS_PROFILE_TOOL=nsys ./hpc_gpu_profile.sh` | 采集 CUDA/MPI/OS runtime/CPU sampling 时间线 |
| Nsight Compute | `hpc submit -e AMSS_PROFILE_TOOL=ncu,NCU_KERNEL_REGEX=rhs_kernel ./hpc_gpu_profile.sh` | 对一个 RHS launch 采集 full metric set |

脚本的共同约束：

- 1 个 MPI rank 驱动 1 个 MIG 实例，CUDA 架构固定为 `sm_80`。
- 正式计时不使用 TwoPuncture cache；profile 的初值准备和编译在测量区外。
- profiler 默认把演化终点缩短到 `t=4`。环境覆盖只允许 `0 < t <= 100`，正常提交不设置覆盖时仍是官方输入。
- 每个任务生成独立 artifact 目录；`profile/` 中原始大文件被 gitignore，报告只提交归纳结果。
- HPC 脚本使用 `--export=NONE`，需要的覆盖通过 `hpc submit -e K=V` 显式传入，避免把提交终端中无关环境变量带入计算任务。
- `scripts/collect_gpu_info.sh` 固化 allocation、CPU 拓扑、GPU、编译器和 profiler 版本。VTune 位于 oneAPI 目录而非默认 `PATH`，脚本会自动发现它。

### 计时边界

仓库里有三个容易混淆的时间：

- `This Program Cost`：Python driver 在启动 TwoPuncture 前开始、ABEGPU 返回后停止。它是评分最接近的口径，包含 TwoPuncture 和主演化，不包含后续复制和绘图。
- `Total Running Time`：ABEGPU 从 `MPI_Init` 后到演化结束的时间，包含网格/初值初始化与演化。
- `outer_wall_seconds`：围绕整个 `./run.sh`，还包含 Python 参数生成、结果复制和绘图，但构建仍在计时区外。

所有端到端比较都应以 `This Program Cost` 为主，同时用另两个时间解释变化来自求解器、初始化还是 Python 后处理。

## 4. 测试环境

| 项目 | 实测配置 |
| --- | --- |
| GPU | NVIDIA A100 80GB PCIe，MIG `1g.10gb`，compute capability 8.0 |
| Driver | 610.43.02 |
| CPU | Intel Xeon Gold 5320，任务 cpuset 为 16 logical CPU = 8 physical cores |
| NUMA | 分配位于单一 NUMA node |
| 内存 | 24 GiB |
| MPI | Open MPI 5.0.7，1 rank，`--bind-to core` |
| C/C++/Fortran | GCC/GFortran 14.2.0 |
| CUDA | nvcc 13.3，目标 `sm_80` |
| Nsight Systems | 2026.1.3 |
| Nsight Compute | 2026.2.1 |
| Intel VTune | oneAPI VTune 2026.3，`/opt/intel/oneapi/vtune/2026.3/bin64/vtune` |

GPU build 使用严格 `-O3`。profile build 额外加入 `-g -fno-omit-frame-pointer` 和 CUDA `-lineinfo`，没有启用 fast-math。

## 5. 除 TwoPuncture 外的完整流程

以下从 TwoPuncture 已产生 `Ansorg.psid` 和更新后的 puncture 参数开始。

### 5.1 Python driver 与输入装配

1. `run.sh` 设置无限栈、Open MPI 容器运行许可、build/output/cache 绝对路径，随后运行 `AMSS_NCKU_Program.py`。
2. Python driver 重建本次输出目录，保存输入快照，调用 `scripts/setup.py` 和 `scripts/numerical_grid.py` 生成物理参数、AMR 网格描述和 `AMSS-NCKU.input`。
3. driver 把 `ABEGPU` 和 `TwoPunctureABE` 从独立 build 目录复制进运行目录。
4. TwoPuncture 完成后，`renew_puncture_parameter.py` 把求得的 puncture 参数追加到主演化输入，并生成最终 `input.par`。
5. `makefile_and_run.py` 通过 `mpiexec -n 1` 启动 `ABEGPU`。

固定物理配置是 vacuum BSSN、四阶有限差分、cell-centered Patch 网格、equatorial symmetry、9 层 AMR、Courant factor 0.5。GPU 路径的正式终点为 `t=100`，analysis 间隔为 0.1；checkpoint 和大规模 dump 间隔为 1000，因此这次正式区间内不会触发周期 checkpoint/dump。

### 5.2 ABEGPU 初始化

`src/ABE.C` 完成以下工作：

1. 初始化 MPI，解析 `input.par`，构造 `bssn_class`。
2. `Initialize()` 创建变量列表、monitor、9 层 AMR hierarchy 和 patch/block，依据最粗层网格间距计算时间步长。
3. `Read_Ansorg()` 读取 `Ansorg.psid`，在 CPU 上把初始解插值到每层每个 block 的网格点，并构造 BSSN 初值。
4. `move_to_gpu()` 把 State、RHS、同步临时量、constraint 和 analysis 变量整体搬到 device。主演化随后保持 device-resident，只有通信、monitor/分析输出和部分控制路径回到 host。
5. `Before Evolve` 时间点在上述初始化和初始 H2D 之后；三次短窗中此阶段为 3.463 至 3.527 s。

### 5.3 AMR 递归与 RK4 演化

`Evolve()` 每个最粗层物理时间步调用 `RecursiveStep(0)`。递归过程依次推进更细层；超过 time-refinement 起始层后，每个父层步对应两个子层步，从而保持层间时间对齐。

每个 level 的 `Step_GPU()` 主要包含：

1. 在最细层通过场插值更新黑洞位置的 predictor/corrector 数据。
2. 到达 analysis level 且累计到 0.1 时，计算 `Psi4`、波区 surface integral、ADM 量和黑洞轨迹。
3. 执行 RK4 predictor 加 3 个 corrector 子步。每个子步按本 rank 的 patch/block 和各自 CUDA stream 提交：
   - BSSN 代数约束修正；
   - `rhs_kernel` 计算 BSSN RHS、几何中间量和约束量；
   - 最粗层 Sommerfeld 外边界或细层边界修正；
   - 各状态变量的 RK4 更新；
   - lapse lower-bound 修正。
4. 子步边界调用全设备同步，随后做错误归约和 ghost-zone 同步。
5. level 推进完成后执行 fine/coarse restrict/prolong，再根据黑洞位置重网格。

回到最粗层后，`Constraint_Out()` 按 analysis 间隔补算/同步各层 Hamiltonian 和 momentum constraints 并写 monitor。由于 interval 为 0.1，而最粗层每步推进到整数时间，这部分每个最粗步都会触发，是 baseline 的真实组成，不应在优化计时中关闭。

### 5.4 输出、绘图与正确性

ABEGPU 结束后，Python driver：

1. 把 setting、error、黑洞轨迹、ADM、`Psi4` 和 constraint 文件复制到上一层输出目录。
2. 调用两个 plotting 模块生成轨迹、距离、波形、ADM 和 constraint 图。绘图失败会给 warning，但不改变已经得到的求解结果。
3. 外部 `check.sh` 调用 `scripts/check_result.py`，按共同时间点比较轨迹 RMS，并检查 9 层 constraint 上限。

正式优化每一轮至少应保存 `This Program Cost`、checker 结果、Nsight Systems 的 launch/sync 变化；涉及 RHS 代码形态时还应复测 NCU 的 registers、occupancy、stall 和访存指标。

## 6. Baseline 性能

### 6.1 正式 `t=100` 尝试

作业 131769 使用正式输入、无 TwoPuncture cache。它运行到 `t=9`，在开始第 10 步后手动停止，以避免已确定会超过 30 分钟上限的任务继续占用资源。

| 物理时间步 | wall time (s) |
| ---: | ---: |
| 1 | 18.4032 |
| 2 | 17.8566 |
| 3 | 17.9680 |
| 4 | 18.6986 |
| 5 | 19.2590 |
| 6 | 20.1060 |
| 7 | 20.2606 |
| 8 | 20.0835 |
| 9 | 20.0992 |

不能把短窗均值简单当正式成绩。短窗的乐观线性估计是：

```text
pre-evolution driver stage       26.88 s
ABEGPU initialization             3.49 s
evolution 18.423 s/t * 100     1842.34 s
estimated This Program Cost     1872.71 s = 31.21 min
```

正式尝试后半段已经从约 18 s/t 上升到约 20 s/t，因此实际 `t=100` 很可能更慢。当前可靠结论是“超过 30 分钟任务上限”，不是一个伪造的完整 `t=100` 成绩。

### 6.2 可复现短窗 `t=5`，三次

作业 131838，构建在计时区外，每次都重新执行 TwoPuncture，未使用 cache。

| Run | This Program Cost (s) | ABEGPU Total Evolve (s) | ABEGPU Total Running (s) | Outer wall (s) | Check |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 123.004985 | 91.6950 | 95.2218 | 131.854 | PASS |
| 2 | 123.727695 | 93.8690 | 97.3626 | 132.823 | PASS |
| 3 | 120.742973 | 90.7865 | 94.2491 | 129.758 | PASS |
| Mean | **122.491884** | **92.1168** | **95.6112** | **131.478** | 3/3 PASS |
| Sample stddev | **1.557112** | - | - | - | - |

checker 在 5/100 个 golden 时间点的共同前缀上得到 trajectory RMS 0；9 层 constraint 均通过。level 0 最大值为：

- Hamiltonian: 0.22822817
- Momentum x/y/z: 0.026833543 / 0.010600207 / 0.017673173
- 允许上限: 2

短窗只用于 profile 和优化趋势，不能替代最终 `t=100` trajectory 完整校验。

## 7. Nsight Systems：全程序 GPU 时间线

作业 131989 profile `t=0..4` 的 ABEGPU，初值准备在采集区外；该次 checker 通过且 trajectory RMS 为 0。ABEGPU 报告 Total Evolve 74.066 s、Total Running 77.704 s。

### 7.1 GPU kernel 汇总

| Kernel | 总时间 (s) | GPU kernel 时间占比 | Calls | 平均每次 |
| --- | ---: | ---: | ---: | ---: |
| `rhs_kernel` | 56.9060 | 77.3% | 1623 | 35.062 ms |
| `prolong3_kernel` | 8.8696 | 12.1% | 47307 | 187.49 us |
| `restrict3_kernel` | 2.9178 | 4.0% | 10047 | 290.41 us |
| `global_interp_kernel` | 2.0093 | 2.7% | 1376 | 1.460 ms |
| Sommerfeld kernels | 1.4420 | 2.0% | 37248 | 38.71 us |
| RK4 kernels | 0.8349 | 1.1% | 37632 | 22.19 us |

RHS 是决定性热点。AMR prolong/restrict 合计 16.1%，且调用量极高，是第二组结构性热点。Sommerfeld 和 RK4 单次很小，但大量逐变量 launch 会增加主机调度和依赖管理成本。

### 7.2 CUDA API 与数据移动

| CUDA API | Host API 时间 | 占比 | Calls |
| --- | ---: | ---: | ---: |
| `cudaDeviceSynchronize` | 71.5725 s | 96.0% | 4110 |
| `cudaMemcpy` | 1.0844 s | 1.5% | 5544 |
| `cudaLaunchKernel` | 0.9072 s | 1.2% | 201694 |
| `cudaMalloc` | 0.3902 s | 0.5% | 7256 |
| `cudaFree` | 0.3696 s | 0.5% | 7224 |

GPU 数据移动统计：

| 类型 | Device 时间 | Calls | 数据量 |
| --- | ---: | ---: | ---: |
| H2D | 497.85 ms | 3194 | 1522.04 MB |
| D2H | 149.89 ms | 2647 | 438.15 MB |
| Memset | 30.88 ms | 6898 | 4609.24 MB |

解释时要避免重复计时：`cudaDeviceSynchronize` 的 71.57 s 是 host 等待 GPU 完成的时间，和 56.91 s RHS 等 kernel 已重叠。它说明提交路径被频繁同步串行化，而不是额外多出 71.57 s 计算。

## 8. Nsight Compute：`rhs_kernel`

作业 132076 对第一个匹配的 RHS launch 使用 full set，MIG 不允许 profiler 锁定 GPU clocks，因此使用 `--clock-control none`。报告完成后旧脚本的目标进程未正常退出，任务被取消；`.ncu-rep` 已完整导出，后续脚本已加入 `--kill yes`。同一 binary/input 在 baseline 和 Nsight Systems 运行中通过 checker。

被采样 launch 的配置为 grid `(5,5,5)`、block `(8,8,4)`，每 block 256 threads，duration 7.37 ms。它是最早的较小 RHS launch；不同 AMR level 和 block 尺寸仍需用 `NCU_LAUNCH_SKIP` 补采。

| 指标 | 数值 | 含义 |
| --- | ---: | --- |
| Registers per thread | **250** | 寄存器是 occupancy 的直接限制 |
| Block limit: registers | **1 block/SM** | 其余 shared memory/warp 限制都更宽松 |
| Theoretical occupancy | **12.50%** | 理论仅 8 active warps/SM |
| Achieved occupancy | **11.04%** | 实测 7.07 active warps/SM |
| No eligible scheduler cycles | **75.30%** | 大量周期没有可发射 warp |
| Eligible warps/scheduler | **0.28** | 延迟隐藏能力不足 |
| Compute throughput | 22.26% | 未接近 SM 峰值 |
| Memory throughput | 28.14% | 也未接近 memory pipe 峰值 |
| DRAM throughput | 3.97% | 不是简单的 HBM 带宽饱和 |
| L1/L2 hit rate | 92.33% / 97.00% | 大多数请求命中 cache |
| Branch efficiency | 99.47% | 分支发散不是主因 |

warp stall 规则把约 38.71% 的 issue 间隔归因于 L1TEX scoreboard dependency，34.26% 归因于 fixed-latency execution dependency。NCU 同时报告 2,782,280 个 excessive global sectors，约占 13,769,410 sectors 的 20%。因此 RHS 的问题是“高寄存器压力导致并发度过低，无法隐藏访存/执行依赖”，并伴随一部分不合并访问；不是单纯缺少 DRAM 带宽。

NCU 给出的 71.86% occupancy-limit 和 17.45% uncoalesced-access speedup 都是规则上界估计，不能直接当作端到端预期收益。

## 9. Intel VTune：Host 侧热点

VTune 2026.3 实际安装在 oneAPI 目录，但不在镜像默认 `PATH`。第一次 PATH-only 检查因此误报不可用；脚本已修正为自动发现 `/opt/intel/oneapi/vtune/*/bin64/vtune`。作业 132408 以 software sampling hotspots 模式包裹 `mpiexec -n 1 ./ABEGPU`，采集 `t=0..4`：VTune elapsed 82.009 s、CPU time 80.440 s；ABEGPU 自报 Total Evolve 74.094 s、Total Running 77.752 s。

| Host function | CPU time (s) | CPU time 占比 |
| --- | ---: | ---: |
| `cuCtxSynchronize_v2` | **65.576** | **81.5%** |
| `clock_gettime` | 6.704 | 8.3% |
| `__memset_evex_unaligned_erms` | 1.168 | 1.5% |
| Unknown stack frames | 0.860 | 1.1% |
| `cuMemcpyHtoD_v2` | 0.774 | 1.0% |
| `cuMemAlloc_v2` | 0.396 | 0.5% |
| `cuMemFree_v2` | 0.363 | 0.5% |
| `cuLaunchKernel` | 0.360 | 0.4% |
| `cuMemcpyDtoH_v2` | 0.288 | 0.4% |
| `PMPI_Allreduce` | 0.080 | 0.1% |

这个结果独立确认 host 主线程主要在等待 GPU：`cuCtxSynchronize_v2` 的 65.58 s 与 Nsight Systems 的 `cudaDeviceSynchronize` 71.57 s 结论一致。它不能解释 kernel 内部原因，需由 Nsys 的 RHS 时间和 NCU 的寄存器/occupancy 数据补全。单 rank 下 MPI Allreduce 不是瓶颈，H2D、allocation/free 和 launch 的单项 host 开销也都远小于同步等待。

当前采样通过 checker：匹配 4/100 个 golden 时间点，trajectory RMS 为 0，所有 constraint maxima 小于 2。保存的脚本已额外打开 VTune call-stack collection，供后续优化轮次从同步 API 追到具体调用边界。

## 10. 优化建议与验证顺序

### P0：降低 RHS 寄存器压力，针对依赖链做 kernel fission

依据最强：RHS 占 77.3%，250 registers/thread 把 occupancy 限到 12.5%，75.3% scheduler 周期无 eligible warp。

建议先做小范围、可回退的拆分实验：

1. 用 `nvcc --resource-usage` 或 `-Xptxas=-v` 固化拆分前的 registers、spill、stack 数据。
2. 沿真实数据依赖把 `bssn_rhs_gpu.cu` 拆成 2 至 3 个阶段，优先缩短大量几何中间量和导数临时变量的 live range。
3. 每个方案同时比较 RHS 总时间、额外 global traffic、launch 数、registers、occupancy 和端到端时间。
4. `__launch_bounds__` 或 `-maxrregcount` 只能作为受控实验。若产生 local-memory spill，occupancy 提升可能反而变慢。
5. NCU 已显示 cache hit 较高但存在 20% excessive sectors。先从 source view 定位不合并访问行，再决定数据布局或索引变换；不要在没有 tile 复用证据时把整个 3D stencil 直接搬入 shared memory。

### P1：把全设备同步缩小为依赖所需的 stream/event 同步

`GPUManager::synchronize_all()` 直接调用 `cudaDeviceSynchronize()`，而 `Step_GPU()`、`Parallel_GPU.cpp`、constraint/interpolation 路径中多次使用它。

建议：

1. 画出每个 patch stream 在 predictor、corrector、ghost exchange、restrict/prolong 前后的数据依赖。
2. 同一 stream 内依靠顺序语义；跨 stream 只在实际 producer/consumer 边界记录和等待 event。
3. MPI host staging 前只同步参与通信的 stream/buffer，不等待无关 patch。
4. 保留错误检查，但可将每个子步的全局错误检查合并到已有必要同步边界。
5. 用 Nsight Systems 验证 device idle gap、同步调用次数和 kernel overlap，不能只看 `cudaDeviceSynchronize` API 时间下降。

### P2：批处理 AMR 和逐变量小 kernel

prolong/restrict 合计 16.1% kernel 时间、57354 次 launch；RK4 和 Sommerfeld 又各有约 3.7 万次调用。

候选方案：

- 给 prolong/restrict 增加变量维度或指针表，一次 launch 处理同一 patch 上一组兼容变量。
- 对 RK4 逐变量更新做 batching；把每个变量的传播速度、SoA、边界属性作为紧凑 metadata。
- 只融合迭代域和依赖兼容的操作。不要跨 ghost exchange、RK4 stage 或 coarse/fine 时间对齐边界融合。
- 对批处理前后比较 launch 数、L2 命中、指针间接开销和寄存器变化。

### P3：复用 device 临时缓冲区

短窗出现 7256 次 `cudaMalloc` 和 7224 次 `cudaFree`。源码中 black-hole 插值、global interpolation 和 MPI staging 都有按调用分配路径。

建议按最大需要量在对象初始化或 regrid 时扩容，平时复用；只有拓扑/容量改变时重新分配。CUDA memory pool 可以作为低侵入对照，但显式 ownership 更容易保证生命周期和 MPI buffer 正确性。

### P4：减少 host staging 与同步 copy

短窗 H2D+D2H 约 1.96 GB，device copy 时间约 0.65 s，优先级低于 RHS/同步，但仍有优化空间：

- 对必须 host staging 的 MPI/monitor buffer 使用 pinned host memory 和 `cudaMemcpyAsync`。
- 将多个小字段打包为较大连续传输，并与不依赖这些数据的 patch 计算重叠。
- 单 rank baseline 中 MPI CUDA-aware 不会消除本地 AMR/analysis 的 host 往返，不能把启用宏当成主要优化。
- 任何 device-resident 改动都要检查 monitor 和 checker 输出，避免 host 读取旧数据。

### P5：检查 RDC 与 device helper 内联

当前 `CUDA_SEPARABLE_COMPILATION` 和 `-rdc=true` 对整个 ABEGPU 生效。对 RHS 高频小 helper，可尝试移入 `.cuh` 并使用受控的 `__forceinline__`，或比较不需要跨翻译单元 device 调用时去除 RDC 的构建。

这项必须排在寄存器实验之后：激进 inline 可能继续增加 RHS 的寄存器 live range 和代码体积。

## 11. 每轮优化的验收门槛

1. 先用 `AMSS_BENCHMARK_TIME=5` 重复至少 3 次，比较 `This Program Cost` 均值和标准差。
2. 每次运行 `check.sh`；trajectory RMS 必须不超过 0.001，所有 constraint maxima 不超过 2。
3. 若改变 RHS，至少复测一个相同 launch 和一个更大 AMR launch 的 NCU metrics。
4. 若改变 stream、同步或 batching，复测 Nsight Systems 的 kernel 总时间、launch 数、同步次数和 GPU idle gap。
5. 有明确收益后再跑正式 `t=100`。最终结果必须覆盖完整 100 个 golden 时间点；短窗 PASS 不能替代正式验收。
6. 记录硬件 MIG 类型、cpuset、commit、构建参数和任务 ID，避免把节点差异误判为代码收益。

## 12. 原始证据位置

原始 artifacts 保存在本地 `profile/` 并被 gitignore：

- baseline：`profile/gpu-benchmark-20260821T133110Z-64/`
- 正式尝试日志：`profile/hpc_amss-gpu-time_131769.log`
- Nsight Systems：`profile/gpu-nsys-20260821T135113Z-66/`
- Nsight Compute：`profile/gpu-ncu-20260821T141631Z-63/`
- NCU CSV 导出：`profile/gpu-ncu-20260821T141631Z-63/ncu-details.csv`
- VTune：`profile/gpu-vtune-20260821T153432Z-64/`

这些大文件不进入 git；本文中的表格是提交后可长期保留的可审计摘要。
