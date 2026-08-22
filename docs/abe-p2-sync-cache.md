# ABE P2 阶段报告：Sync 计划缓存与线程错误归约

## 1. 本阶段结论

P2 完成了单进程 OpenMP 路径中剩余的同步准备优化：

- 缓存 Patch 内 ghost Sync 的 grid segment 和交叠关系；
- 缓存 Patch 间 buffer Sync 的 grid segment 和交叠关系；
- 按变量列表缓存展开后的 transfer operations；
- 所有 transfer 共用一个可增长的 packed data workspace；
- Step 的 NaN/error 状态使用 OpenMP 位或归约；
- OMP-only 路径不再执行没有意义的 MPI_Allreduce。

相同 HPC 配置下，t=0..4 的结果为：

| 版本 | Evolve (s) | Total (s) |
|---|---:|---:|
| P0 | 43.8688 | 51.2490 |
| P2 最终 | 43.4695 | 50.6337 |
| 加速比 | 1.009x | 1.012x |
| 时间减少 | 0.91% | 1.20% |

P2 的 perf record 独立运行得到 Evolve 43.0478s、Total 50.2960s，
与 perf stat 结果接近。前一轮 P2 也得到 43.2883s，因此可以把它判断为
约 1% 的小幅优化，不能描述成显著加速。

profile 目录：

    profile/abe-20260822T084733Z-14

## 2. Sync 原来重复做什么

每次 Parallel::Sync 都先从 Block/Patch 几何重新建立：

1. 目标 ghost 或 buffer grid segment 列表；
2. 源 owned grid segment 列表；
3. 源列表和目标列表之间的几何交叠；
4. 每个 segment 与每个变量组合成的 transfer operation；
5. packed data 缓冲区。

这些几何关系由网格拓扑决定，而不是由每一步的场值决定。固定网格在
每个 RK4 子步、每个 AMR 层反复建立和销毁相同列表，只增加控制开销。

## 3. 缓存设计

### 3.1 两级几何缓存

P2 分别保存 Patch 内和 Patch list 间的 SyncGeometry。每个对象拥有：

- source、destination grid segment；
- build_gstl 产生的 transfer source/destination；
- Patch 和 Block 指针签名；
- 针对不同变量列表的 transfer operation plans。

调用 Sync 时先比较当前 Patch/Block 签名。签名相同则复用；如果发生
重网格或 Block 被替换，旧对象会完整释放并重新构建。因而缓存没有把
固定网格作为无检查的永久假设。

实现中的签名还包含 bbox、shape 等几何数值，并以无临时分配的顺序比较
完成匹配。因此同一个 Block 对象原地移动时也会触发计划重建，而不只是
在 Block 指针发生替换时失效。

### 3.2 变量 operation 缓存

同一份几何计划可能用于 StateList、SynchList_pre、SynchList_cor 或
Psi4 等不同变量集合。缓存按实际 var 指针序列匹配，不依赖临时
MyList 节点的地址。

首次看到某个变量序列时，才展开

    segment x variable -> source/destination/offset/size

后续调用直接执行已有 operation。PACK 和 UNPACK 仍保持原有依赖：
PACK 并行结束处的隐式 barrier 保证所有读取完成，UNPACK 再按变量
并行，并保持同一变量的 segment 写入顺序。

### 3.3 错误状态

Step 的每个 Block 都可能报告 NaN。原 OpenMP 代码在 critical 中同时
打印和写共享 ERROR，之后还调用退化为本地复制的 MPI_Allreduce。

P2 只把错误文本输出保留在 critical 内，ERROR 通过
OpenMP reduction(|:ERROR) 合并。OMP-only 构建直接检查归约结果；
MPI-compatible 构建仍保留跨 rank Allreduce，所以旧运行模式没有改变。

## 4. 正确性与 profile

OMP-only 和 MPI-compatible 构建均通过。perf stat 与 perf record
两次运行之间，以及 P2 与 P0 之间，以下文件数值行全部逐字节一致：

- bssn_ADMQs.dat
- bssn_BH.dat
- bssn_constraint.dat
- bssn_psi4.dat

P2 的主要硬件指标为：

| 指标 | P2 |
|---|---:|
| 平均使用 CPU | 16.271 / 30 |
| IPC | 1.86 |
| branch miss | 0.42% |
| L1D miss | 2.65% |
| LLC load miss | 48.29% |
| dTLB miss | 2.22% |

这些数值与 P0 接近，没有发现缓存造成的硬件退化。malloc/free 以及
build_ghost_gsl、build_owned_gsl、build_gstl 已不在 0.1% 以上的 flat
热点中。cached Sync 的 children 仍约为 7.5%，但其中是必须执行的
copy/restrict/prolong 和 OpenMP 调度，不再是计划重建。

P2 只减少同步控制面工作，不减少实际搬运的网格数据，因此收益约 1%
是合理结果。继续优化 transfer 时，应针对数据布局、相邻 segment 合并
或 kernel 内存访问，而不是继续增加缓存层。
