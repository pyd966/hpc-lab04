# ABE P0/P1/P2 并行优化总结

## 1. 最终状态

原计划中的并行相关 P0、P1、P2 均已完成：

| 阶段 | 状态 | 核心内容 |
|---|---|---|
| P0 | 完成 | 球面点就地插值和积分、缓存插值计划与波形系数 |
| P1 | 完成 | 单进程 OpenMP、block/RK/分析/AMR transfer 并行 |
| P2 | 完成 | Sync 几何和 operation 缓存、workspace 复用、错误归约 |
| P3 | 未做 | RHS/差分 Fortran kernel 内部的 tile 级并行 |

最终运行方式是 1 个 ABE 进程、30 个 OpenMP 线程：

    OMP_NUM_THREADS=30
    OMP_PLACES=cores
    OMP_PROC_BIND=close

ABE 直接执行，不用 mpiexec，也不链接 libmpi。编译仍为
-O3 -g -fno-omit-frame-pointer，没有使用 -Ofast 或 fast-math。

## 2. 第一部分：从 MPI 转成 OpenMP

### 2.1 运行时模型

旧版本用 30 个 MPI rank，每个 rank 拥有一部分 Block。rank 所有权既
决定数据在哪里，也隐含决定由谁计算。改成一个进程后，所有 Block 都在
同一地址空间，原 MPI rank 分工必须显式重新分配给 OpenMP 线程。

OMP-only 构建用一个兼容头保留 MPI_Comm 等旧接口类型，但运行语义为：

- Comm_size 固定为 1，Comm_rank 固定为 0；
- 不初始化 MPI runtime；
- 不进入 send、recv、wait 或真实 barrier；
- 尚未移除的 Allreduce 只执行本地语义；
- 最终二进制有 libgomp，没有 libmpi。

保留这些接口是为了控制改动范围，不表示程序仍在通信。

### 2.2 Step 中的 block 工作

每次 Step 先建立该 AMR level 的本地 Block 向量。predictor 和三个
corrector 都按 Block 使用 OpenMP static schedule。每个线程在独立
Block 上执行：

- enforce algebraic constraints；
- compute_rhs_bssn；
- RK4 更新；
- Sommerfeld 边界处理；
- lower-bound 修正。

Step 末尾的状态交换、时间层准备和 Compute_Psi4 的 Block 计算也按
Block 并行。不同 Block 写不同数组，不需要原子操作。

### 2.3 MPI 数据传输改写

最初只并行 RHS 后，单进程版本反而很慢：Evolve 542.358s，平均只使用
3.23 个 CPU。原因是 Sync、Restrict、Prolong 的 copy/restrict3/
prolong3 仍由主线程串行执行；旧 MPI rank 的并行工作没有自动变成
OpenMP 工作。

完成的转换把每个“grid segment x variable”展开为 operation：

1. PACK 阶段按 operation 并行读取源 Block；
2. 隐式 barrier 保证所有源读取结束；
3. UNPACK 阶段按变量并行写目标 Block；
4. 同一变量的 segment 保持原顺序，避免重叠边界并发写。

不同大小的 PACK 采用 dynamic,1，同一变量的 UNPACK 采用 static。
这一步把平均 CPU 使用数从 3.23 提高到约 16.4，并消除了主线程长尾。

### 2.4 为什么 RecursiveStep 没有整体并行

RecursiveStep 的顺序是数值算法的一部分：粗层推进后，细层可能执行
多次子步；之后才能 restrict/prolong 和同步，再进入下一 RK 阶段。
这些步骤存在真实的时间层依赖。

因此并行化放在每个 Step、transfer、analysis 内部，而不是让不同递归
层无条件同时运行。强行并发不同 level 会读取尚未完成的数据，属于改变
算法，不是无痛的 MPI 到 OpenMP 翻译。

## 3. 第二部分：OpenMP 条件下新增的算法优化

### 3.1 P0：分析数据流重写

旧分析对 8 个半径反复生成完整 pox 和 shellf，逐点扫描 Block，调用
global_interp -> polin3 -> polint，并对每个点和模式重算 Wigner、cos
和 sin。polint 自身占 8.32%，整条插值路径约 13.8%，malloc/free
约 4.72%。

P0 为每个半径缓存：

- 球面点所属 Block；
- 三方向六阶插值下标、反射标记和权重；
- 波形模式的 Wigner/三角线性系数。

每个线程插值一个点后立即累加到线程私有的波形或 7 个 ADM 量，最后只
归约这些小数组。完整 shellf、点坐标临时数组和分析 Allreduce 均被删除。
P0 使 Evolve 从 61.4773s 降到 43.8688s，是后续阶段最大的一次收益。

### 3.2 共享内存 transfer 的附加优化

单地址空间允许程序在 PACK 完成后由任意线程直接写目标数组，不再组织
MPI request 和 rank buffer。PACK/UNPACK 共用一个 OpenMP team，并复用
进程内最大的 packed workspace，避免每个 RK/AMR transfer 都 new/delete。

它仍保留 packed 中间区，因为某些源区域和目标区域可能重叠；先完整读取
再写回是正确性边界，不能简单改成无缓冲 memcpy。

### 3.3 P2：同步计划缓存

Sync 的 ghost/buffer segment、owned segment 和几何交叠只取决于网格
拓扑。P2 缓存这些结构，并针对实际变量序列缓存 operation 的
source/destination/offset/size。

每次调用都会比较 Patch/Block 指针签名；拓扑变化时销毁旧计划并重建。
因此固定网格能复用计划，重网格也不会继续使用过期 Block。

实际签名同时包含 bbox、shape 等几何数值，并以无临时分配的顺序比较。
所以同一 Block 对象上的原地网格移动同样会触发重建，不会继续使用过期
几何。

Step 的 NaN 标志改为 OpenMP 位或归约。文本输出仍在 critical 中，
但正常路径不再写共享 ERROR，也不执行 OMP-only 的伪 MPI_Allreduce。

P2 最终使 Evolve 从 43.8688s 降到 43.4695s，约减少 0.91%。收益小是因为
它只消除计划准备，copy/prolong/restrict 的实际数据工作仍然存在。

## 4. 性能总览

所有正式比较都使用课程固定输入、t=0..4、30 个绑定线程和相同 -O3
符号构建：

| 版本 | 进程 x 线程 | Evolve (s) | Total (s) |
|---|---:|---:|---:|
| MPI baseline | 30 x 1 | 173.669 | 177.580 |
| P1 初始 block 并行 | 1 x 30 | 542.358 | 550.343 |
| P1 完整 transfer 转换 | 1 x 30 | 61.883 | 69.670 |
| transfer team/workspace | 1 x 30 | 61.477 | 68.761 |
| P0 | 1 x 30 | 43.869 | 51.249 |
| P2 最终 | 1 x 30 | 43.470 | 50.634 |

最终版本相对原 30-rank MPI baseline：

- Evolve 加速 4.00 倍；
- Total 加速 3.51 倍；
- 相对错误的“只改 block 循环”初版 OpenMP，Evolve 加速 12.48 倍。

这组结果说明去掉 MPI 本身不是主要收益。真正的收益来自把 MPI 隐含的
transfer 并行显式交给 OpenMP，以及利用固定分析几何重写数据流。

## 5. 正确性与最终热点

P2 的最终 perf stat 和 perf record 分别得到 Evolve 43.4695s 和 43.0478s。
两遍运行、P2 与 P0 之间，ADMQs、BH、constraint、psi4 的数值行全部
逐字节一致。P0 与旧插值路径相比也只有 psi4 理论零项的 1.48e-22
舍入差异，演化状态文件完全一致。

最终 flat profile 的主要热点是：

| 热点 | 周期占比 |
|---|---:|
| compute_rhs_bssn | 44.50% |
| memcpy | 8.72% |
| kodis | 8.40% |
| fdderivs | 7.63% |
| lopsided | 6.63% |
| memset | 5.33% |
| prolong3 | 3.94% |
| fderivs | 3.24% |

IPC 1.86、branch miss 0.42%、L1D miss 2.65%、LLC load miss 48.29%、
dTLB miss 2.22%。分支没有异常；主要剩余问题是 RHS/差分计算、流式
网格访存和 AMR 数据搬运。平均并行度约 16.27/30，受 AMR 层级依赖和
粗层 Block 数限制。最终 profile 位于 `profile/abe-20260822T084733Z-14`。

P0/P1/P2 到此已经闭环。若继续提高并行度，下一项应是此前暂缓的 P3：
在 compute_rhs_bssn、fdderivs、kodis 等 Fortran kernel 内按 tile 或
空间循环并行，使只有 9 个 Block 的粗层也能使用更多线程；这会触及
kernel 内部数据布局和并行区位置，风险明显高于本轮。
