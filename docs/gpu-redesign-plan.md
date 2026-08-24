# ABEGPU 重新设计基线与分阶段优化方案

> 基线提交：`0951bc3`（已回滚到此前未进行 RHS fission/register 实验的版本）
> 目标：在不改变物理问题、网格、时间区间和输出语义的前提下，使官方 GPU 路径 `t=100` 的端到端 `This Program Cost` 不超过 330 s。
> 本报告范围：TwoPuncture 只作为已优化的初值阶段计入预算；重点是其后的 ABEGPU 全流程。

## 1. 先给结论

当前程序距离目标不是一个局部 kernel 调优问题，而是两个层次同时受限：

1. `rhs_kernel` 是计算主热点，但它把导数、几何、Ricci、方程组装、耗散和约束放进同一个线程生命周期。新鲜 NCU 结果为 250 registers/thread、理论 occupancy 12.50%、实际 occupancy 11.04%；75.27% 的 scheduler 周期没有 eligible warp。
2. host 调度路径几乎每个阶段都做全设备同步。Nsight Systems 中 1 个物理时间单位有 1,044 次 `cudaDeviceSynchronize`；VTune 中 `cuCtxSynchronize_v2` 占 host CPU time 65.7%。四条 stream 实际几乎没有重叠，不能把“有四条 stream”当成已有四路并行。

因此建议固定为 **6 个优化阶段**：

| 阶段 | 核心工作 | 进入下一阶段的硬门槛 |
| --- | --- | --- |
| 0. 基线与预算 | 固化输入、正确性、Nsys/NCU/VTune 口径 | 已完成；所有后续结果与本报告同口径 |
| 1. RHS 数据流重构 | 按真实依赖做四类 RHS kernel，并建立跨变量 stencil batching | RHS 不再出现 250-reg 巨核；每个新 kernel 无明显 spill，RHS 总时间有端到端收益 |
| 2. RHS 微观调优 | block/grid、布局、coalescing、helper inline/RDC、受控 launch bounds | RHS 理论 occupancy 至少 25%，且 E2E 不回退 |
| 3. 同步与执行图 | `cudaDeviceSynchronize` 改为 stream/event 依赖，保留必要的 MPI/ghost 边界 | correctness 不变；全设备同步显著减少，时间线出现可解释的跨 patch 重叠 |
| 4. AMR 跨变量批处理 | prolong/restrict 按 patch 和变量批次发射，复用几何映射与权重 | prolong/restrict launch 和总时间显著下降，粗细层边界语义不变 |
| 5. RK/边界/内存与收尾 | RK4、Sommerfeld、小 kernel batching；持久临时缓冲；最终 t=100 验收 | 短窗与完整 t=100 均通过；只保留有端到端收益的改动 |

阶段 1 是决定成败的阶段。若阶段 1 后 RHS 仍只能下降约 20%，就不应继续在 block 大小或寄存器上做边角实验，而应重新检查中间量物化和跨变量批处理设计。

## 2. 回滚后的新鲜基线

### 2.1 环境与运行口径

- GPU：NVIDIA A100 80GB PCIe，MIG `1g.10gb`，compute capability 8.0。
- 任务 CPU：16 logical CPU，单一 NUMA node；本次 1 个 MPI rank。
- MPI：Open MPI 5.0.7；`MPI_CUDA_AWARE=0`，通信分支使用 host staging。
- CUDA：nvcc 13.3，`sm_80`；ABEGPU 仍使用 `CUDA_SEPARABLE_COMPILATION ON` 和 `-rdc=true`。
- ABEGPU 的 OpenMP 为 OFF；TwoPuncture 的 OpenMP 只属于初值阶段，不增加 RHS kernel 的 host 并行度。
- profile 输入为固定课程输入的 `t=0..1`，TwoPuncture 初值准备在测量区外，cache 只用于 profile 减少准备噪声。缩短窗口只用于开发和 profile，不能替代 `t=100` 验收。

### 2.2 端到端单步

| 指标 | Nsight Systems | VTune |
| --- | ---: | ---: |
| Before Evolve | 3.23306 s | 3.58038 s |
| Total Evolve | 17.5389 s | 17.4886 s |
| Total Running | 20.7719 s | 21.0690 s |
| 正确性 | trajectory RMS 0，constraints PASS，FINAL PASS | trajectory RMS 0，constraints PASS，FINAL PASS |

若远程 TwoPuncture 约 11 s，且初始化/driver 固定开销按约 15 s 估算，则留给 100 个物理时间单位的演化预算约为 315 s，即约 **3.15 s/单位**。当前演化约 17.5 s/单位，至少需要约 **5.6 倍** 的演化加速；因此单独把 RHS occupancy 从 12.5% 调到 25% 不是目标本身，只是必要的中间条件。

### 2.3 Nsight Systems：kernel、API 和 stream

| Kernel | 总时间 | 占 kernel 时间 | 调用次数 |
| --- | ---: | ---: | ---: |
| `rhs_kernel` | 13.9386 s | 77.7% | 420 |
| `prolong3_kernel` | 2.1468 s | 12.0% | 11,955 |
| `restrict3_kernel` | 0.6493 s | 3.6% | 2,391 |
| `global_interp_kernel` | 0.4890 s | 2.7% | 344 |
| Sommerfeld kernels | 0.3457 s | 1.9% | 9,312 |
| RK4 kernels | 0.2062 s | 1.1% | 9,408 |

CUDA API 侧为：

| API | 时间 | 调用次数 |
| --- | ---: | ---: |
| `cudaDeviceSynchronize` | 17.4124 s | 1,044 |
| `cudaMemcpy` | 0.4625 s | 2,874 |
| `cudaLaunchKernel` | 0.2457 s | 50,110 |
| `cudaMalloc` | 0.1166 s | 2,509 |
| `cudaFree` | 0.0939 s | 2,501 |

GPU kernel duration 之和为 17.9320 s，时间并集为 17.6235 s。四条 stream 为 13、14、15、16；并发分布为：单流 17.3285 s、双流 0.2879 s、三流 0.00043 s、四流 0.00659 s。RHS 自身 13.9386 s 的并集为 13.8777 s，只有 60.9 ms 双流重叠。结论是：当前不同 stream 代表不同 Block 的轮转提交，并非独立的 RHS 工作队列；Block 内 kernel 仍按 host 循环顺序执行，阶段末的全局同步把依赖扩大到了整个 device。

### 2.4 Nsight Compute：两个真正热点

`rhs_kernel`（grid `(5,5,5)`，block `(8,8,4)`，256 threads/block）：

- 250 registers/thread；register 是 occupancy 的唯一紧限制，理论 occupancy 12.50%，实际 11.04%。
- Compute throughput 22.37%，memory throughput 28.27%，DRAM throughput 4.04%。L1/L2 hit rate 为 92.34%/96.95%。
- `No Eligible` 为 75.27%，eligible warps/scheduler 只有 0.28；warp latency 没有被足够多的独立线程隐藏。
- branch efficiency 99.47%，所以分支发散不是主要方向。
- 仍有约 20% excessive global sectors；需要定位具体索引后再决定布局或 tile，不应盲目把整个 RHS 搬到 shared memory。

`prolong3_kernel`（grid 54，block 256）：

- 66 registers/thread，无 local/shared spill；理论 occupancy 37.50%，实际 26.77%。
- Compute throughput 45.78%，DRAM throughput 4.85%，L1/L2 hit rate 87.84%/98.15%。
- 它不是和 RHS 同类的寄存器灾难；主要问题是每个变量重复 launch，并在每个输出点重复做对齐、奇偶和 6x6x6 张量积。

### 2.5 VTune：host 侧确认

新鲜 VTune `t=0..1` 的 top hotspots：

| Host function | CPU time | 占比 |
| --- | ---: | ---: |
| `cuCtxSynchronize_v2` | 15.635 s | 65.7% |
| `clock_gettime` | 1.828 s | 7.7% |
| `prte_init` | 1.091 s | 4.6% |
| `cuMemcpyHtoD_v2` | 0.718 s | 3.0% |

单 rank 下 `MPI_Allreduce` 不是主要瓶颈。host 等待时间不能和 GPU kernel 时间相加，但它清楚地说明当前提交路径被同步串行化。

原始 artifacts：

- Nsys：`profile/gpu-nsys-20260824T071953Z-57/`
- NCU RHS：`profile/gpu-ncu-20260824T072559Z-66/`
- NCU prolong：`profile/gpu-ncu-20260824T072859Z-58/`
- VTune：`profile/gpu-vtune-20260824T073534Z-65/`

## 3. 除 TwoPuncture 外的实际全流程

### 3.1 输入与初始化

Python driver 根据固定参数建立输出目录、9 层 Patch AMR 描述和 `input.par`，再用 `mpiexec -n 1` 启动 `ABEGPU`。`ABE.C` 初始化 MPI、变量列表、monitor、patch/block 和时间步；`Read_Ansorg()` 把 `Ansorg.psid` 插值到每个 block；`move_to_gpu()` 把状态、RHS、同步临时量、约束和分析变量搬到 device。主演化期间数据原则上保持 device-resident，通信、monitor 和部分分析路径才回到 host。

### 3.2 每个 level 的 RK4 step

`bssn_step_gpu.C` 对每个 level 执行 predictor 和 3 个 corrector：

1. 代数约束修正 `enforce_ga_kernel`。
2. 启动一个巨型 `rhs_kernel`，计算 BSSN 右端、几何中间量、Ricci、源项、耗散和约束。
3. 遍历 `StateList`，逐变量启动 Sommerfeld、RK4 更新和细层边界修正。
4. 对 lapse 做 lower-bound 修正。
5. 全设备同步、错误归约、`Parallel::Sync_GPU` ghost/buffer exchange，再进入下一 RK 子步。

细层推进结束后，递归的 coarse/fine 时间对齐会触发 restrict/prolong；analysis level 按 0.1 间隔计算波形、ADM、黑洞轨迹和 constraints。所有这些调用都在评分时间边界内，不能通过关闭 analysis 或减少物理计算来加速。

### 3.3 数据交换与输出

`Parallel_GPU.cpp` 为每个 transfer segment 调用 pack/restrict/prolong/unpack。默认单 rank 仍走同一套 transfer 逻辑；多 rank 分支还会分配 device buffer，并在 `MPI_CUDA_AWARE=0` 时执行 D2H、MPI host buffer、H2D。`GPUManager` 当前创建 4 条 stream，但 `allocate_device_memory`/`free_device_memory` 直接调用 `cudaMalloc`/`cudaFree`，缓存池代码被注释掉。ABEGPU 结束后，Python driver 整理 monitor、轨迹、Psi4 和 constraint，checker 再验证共同时间点和 constraint 上限。

## 4. 六阶段优化设计

### 阶段 0：基线与预算（已完成）

冻结 commit、MIG 类型、CPU cpuset、编译参数和固定输入；每轮保存：

- 3 次 `t=5` 端到端 benchmark（均值、标准差、This Program Cost）；
- 一次 `t=0..1` Nsys；
- RHS 和 AMR 热点的 NCU；
- 涉及 host 调度时的 VTune；
- checker 的 trajectory RMS 与 constraints。

开发门槛是短窗 PASS；进入最终验收前必须完整运行 `t=100`，覆盖全部 golden 时间点。

### 阶段 1：按真实依赖重构 RHS，并建立跨变量 stencil 批处理

这一阶段不是把原 kernel 任意切成许多小块，而是先确定四个有物理/数据意义的边界：

1. **Stencil batch**：按同一迭代域批量计算可复用的一阶/二阶导数。beta 的一阶和二阶导、chi/metric/Lap/trK/Gamma 的导数按字段批次处理；每个批次只保留该操作需要的局部量，不把 24 个最终方程的全部变量同时放进寄存器。
2. **Geometry/Ricci**：消费基础场和导数 scratch，计算逆度规、Christoffel、Ricci 和 gauge 几何项。几何输出只保留下一阶段真正复用的量，避免写回全部局部命名变量。
3. **Equation assembly**：计算 24 个演化变量的方程源项；对相同的 lopsided/advection 和 KO 操作引入变量维度批处理，变量属性（SoA、奇偶、传播速度）放入紧凑 metadata。
4. **Constraints**：`co==0` 才需要的 Gamma/约束残差独立处理。当前 kernel 在所有 RK 子步都携带这些局部生命周期，即使只有 predictor 写入约束，这是寄存器和依赖链的重要来源。

阶段 1 的 scratch 设计必须以“每个字段只写一次、下一阶段顺序读”为原则；不能为减少 launch 把所有导数一次性塞进单线程数组，也不能为了拆分把每个标量单独发射成数千个 kernel。第一版先验证 2、3、4 个 batch 粒度，比较 scratch bytes、L2 traffic、RHS 时间和端到端时间。

**阶段门槛：** 每个新 kernel 的理论 occupancy 至少 25%（优选 37.5% 以上），无明显 local spill；RHS 总时间至少下降 2 倍或能证明后续阶段仍有清晰的 3 倍以上空间。若只得到 10% 级收益，应回到数据流设计而不是继续调 `maxrregcount`。

### 阶段 2：RHS 微观调优与编译组织

在阶段 1 的 kernel 边界稳定后，按 NCU 逐项实验：

- 对每类 kernel 比较 `(8,8,4)`、`(8,8,2)`、`(16,4,4)` 等保持 coalescing 的 block 形状；grid 由有效点数决定，不能通过减少 block 改变计算域。
- 对 stencil 访问定位 excessive sectors，优先修正 x-fastest 布局、地址计算和只读 cache；只有有邻域复用证据时才加入 shared-memory tile。
- 将 `diff_new/lopsidediff/kodiss` 中真正小且高频的 device helper 迁移到 `.cuh` 做受控 `__forceinline__` 实验；同时比较去除 RHS 热路径 RDC 的专用构建。inline 可能增加寄存器，必须以 NCU 和 E2E 判定。
- 对已低于 100 registers 的 kernel 尝试 `__launch_bounds__`；`-maxrregcount` 只做对照，不作为默认方案。若 local spill 增加或内存 throughput 上升但 kernel 变慢，立即淘汰。

**门槛：** RHS 理论 occupancy >=25%，实际 occupancy 不低于阶段 1；scheduler `No Eligible` 明显下降；RHS 和 `This Program Cost` 同时改善。单独的 occupancy 数字不算成功。

### 阶段 3：缩小同步范围，重建 stream/event 执行图

根据 `Step_GPU` 和 `Parallel_GPU.cpp` 的 producer-consumer 关系建立依赖图：

- 同一 Block 的 kernel 继续依靠同一 stream 的顺序语义；不增加无必要 event。
- 不同 Block 的 RHS、边界和可独立的 analysis 保持各自 stream；只有 ghost exchange、restrict/prolong 或 RK stage 的真实消费者记录 event 并等待。
- 将 `GPUManager::synchronize_all()` 的全设备等待替换为参与该 buffer 的 stream/event 等待。MPI host staging 只同步参加通信的 stream，不等待无关 Block。
- 错误标志保留；把检查放到已有的 stage/通信边界，不能为了去掉同步而跳过 NaN 检测。

**门槛：** correctness 不变；`cudaDeviceSynchronize` 次数和 device idle gap 均下降；Nsys 的 kernel 并集时间与 host `This Program Cost` 一起改善。若只是 API 时间下降而 GPU 并集不变，说明只是隐藏了等待，不算收益。

### 阶段 4：AMR prolong/restrict 跨变量批处理

`prolong3_kernel` 已经有足够 occupancy，重点不是继续压寄存器，而是消除重复工作：

- 新增 `var` 维度，一次 launch 处理同一 coarse/fine segment 的一组变量；指针表和 SoA/parity metadata 使用连续 device arrays。
- 对同一 segment 预先计算 `CD/FD/base/lbf/lbc`、有效范围、奇偶和插值权重；这些几何量不应每个变量、每个输出点重复构造。
- 优先融合兼容变量的 prolong 与 pack/unpack；不能跨 coarse/fine 时间对齐或 ghost exchange 边界融合。
- 用 shared memory 或寄存器缓存小的 1D 权重/局部中间面板，但以 shared-memory occupancy 和 L1/L2 指标为准。

**门槛：** prolong/restrict launch 数显著下降，2.796 s 的合计 kernel 时间至少下降一半；AMR 之后 trajectory/constraints 与基线一致。若指针间接导致 coalescing 变差，则退回按变量组而非全量批处理。

### 阶段 5：RK、边界、临时内存和最终整合

- RK4 更新和 Sommerfeld 当前单次很小但各有约 9,000 次调用；按相同 patch/属性批处理，避免逐变量 launch。
- `GPUManager` 的显存池改为持久 scratch allocator，在初始化/regrid 时按峰值扩容；普通 step 不再反复 `cudaMalloc/cudaFree`。allocator 改动必须先通过生命周期和 stream ownership 检查。
- 对必要的 host staging 使用 pinned buffer 和 `cudaMemcpyAsync`；单 rank 下只把它作为补充，不把 CUDA-aware MPI 当作主要收益来源。
- 只在阶段 1--5 均有独立收益后启用 CUDA Graph 或固定 RK launch graph；AMR 拓扑变化时重建 graph。

**最终门槛：** 3 次 `t=5` benchmark 稳定改善，完整 `t=100` `This Program Cost <=330 s`，全部 100 个 golden 时间点和 constraint 检查通过。任何阶段若让总时间回退，即使单 kernel 更快，也不合入主线。

## 5. 预算与预期收益

以下是用于 go/no-go 的目标区间，不是对尚未实现代码的承诺。按当前单步 kernel 比例，达到 330 s 需要大致接近：

| 成本组 | 当前 t=1 GPU kernel | 目标区间 | 主要来源 |
| --- | ---: | ---: | --- |
| RHS | 13.94 s | 1.8--2.5 s | 阶段 1--2 的低寄存器依赖分解、跨变量 stencil、访问优化 |
| prolong/restrict | 2.80 s | 0.6--1.0 s | 阶段 4 的 segment/variable batching |
| interp/analysis/boundary/RK | 1.20 s 左右 | 0.4--0.7 s | 阶段 3、5 的批处理和等待缩小 |
| 同步/调度额外损失 | 约 0.3 s overlap 缺口及 host wait | 尽量接近 kernel 并集 | 阶段 3 的 event graph |

这要求 RHS 约 5.5--7.5 倍、AMR 约 3 倍、其余路径约 2 倍的综合改善，属于结构重写级目标。若阶段 1 后无法把 RHS 带到约 2--3 s 的方向，330 s 目标在当前算法/网格上不现实，应及时报告而不是堆叠低收益微优化。

## 6. 每阶段固定的实验协议

1. 先编译并运行 `t=0..1`，检查无 crash、NaN 和 kernel launch error。
2. 运行相同 binary 的 NCU；RHS 改动至少采一个粗层和一个细层 launch，记录 registers、理论/实际 occupancy、spill、scheduler、L1/L2/DRAM、excessive sectors。
3. 运行 Nsys，记录 kernel 总时间、时间并集、每条 stream 的时间、`cudaDeviceSynchronize`/launch/malloc/free 次数和 H2D/D2H。
4. 涉及 host 调度或同步时运行 VTune，确认 `cuCtxSynchronize_v2`、MPI 和 launch 的变化。
5. 用固定输入做 3 次 `t=5` benchmark，报告均值和样本标准差；所有结果必须带 commit、MIG 类型和任务 ID。
6. 只有独立收益通过后才进入下一阶段；阶段结束提交一次 git。最终再运行不使用 cache 的完整 `t=100`，做全量 checker。

## 7. 不能采用的“优化”

不能降低网格或演化时间、跳过 RHS/constraint/analysis、读取预计算答案、改变数值格式或借助输出缺失伪造加速。不能把 `cudaDeviceSynchronize` 的 host 等待时间与 kernel 时间直接相加，也不能把 NCU 的 rule-based speedup 上界当作端到端收益。所有 fusion、fission、batching 都必须以真实 producer-consumer 依赖和最终 checker 为准。
