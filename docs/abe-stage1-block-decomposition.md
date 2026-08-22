# ABE 优化阶段一：Block 分解实验与正式 OpenMP 入口

## 1. 本阶段结论

本阶段检验了“减少 Block，避免 30 线程执行 32 个任务时出现第二波”这一
假设。结论是：对课程固定网格，当前以线程数作为目标的分解仍是合法候选
中最快的方案。不能把目标简单改成 24 或 32：它们不但更慢，而且会使
课程轨迹检查失败。

因此，生产默认值仍是

    block target = OMP_NUM_THREADS

本阶段没有宣称分块算法带来性能提升。保留的改动是：

- `AMSS_OMP_BLOCK_TARGET` 实验开关；
- 每层打印 target 和实际生成的 Block 数；
- 一次编译、依次运行多个候选的 HPC 扫描脚本；
- 正式 `hpc_cpu.sh` 接入已经完成的 OMP-only ABE；
- 自动按作业 cpuset 统计物理核心，并设置 OpenMP 线程数和绑核。

最后一项是运行链路修正，不是本阶段 profile 二进制的额外加速。修正前，
正式脚本仍会编译 MPI 版本并启动 30 个 rank；修正后才会真正评测此前完成
的单进程 OpenMP 代码。

## 2. 为什么要实测 Block 数

`Parallel::distribute` 接收的是目标值，不保证最终数目。三维 Patch 只能
被切成整数个长方体，还受 ghost/buffer 最小宽度限制。30 线程、默认目标
30 时，实际结果是：

| AMR level | Patch 类型 | 实际 Block 数 |
|---|---|---:|
| 0 | 单个最粗静态 Patch | 9 |
| 1--4 | 单个静态 Patch | 32 |
| 5--8 | 两个移动 Patch 合计 | 30 |

这修正了此前“所有移动层也是 32 个 Block”的推断。真正的 32→30 尾波只
发生在静态 level 1--4；计算更频繁的移动层已经恰好是一轮 30 个任务。
所以把所有 level 的目标一起降到 24，会损失移动层的 30 路并行，收益
无法覆盖损失。

## 3. 实验设计

脚本 `hpc_abe_block_sweep.sh` 在同一作业中：

1. 用 `-O3 -g -fno-omit-frame-pointer` 编译一次 OMP-only ABE；
2. 复用同一份 TwoPuncture 初值；
3. 固定 1 进程、30 物理核心、`OMP_PLACES=cores`、`OMP_PROC_BIND=close`；
4. 每个候选演化到 t=4，并用 `perf stat` 计时；
5. 用课程 `check.sh` 检查轨迹和约束。

两轮共扫描 18、24、25--32、36。较有代表性的结果如下：

| Block 目标 | Evolve (s) | 平均 CPU | 课程检查 |
|---:|---:|---:|---|
| 18 | 48.7075 | 11.619 | PASS |
| 24 | 46.5486 | 13.594 | FAIL |
| 27 | 43.6174 | 16.101 | PASS |
| 28 | 43.3700 | 16.100 | PASS |
| 29 | 43.5043 | 16.051 | PASS |
| 30 | 43.0152 / 43.4148 | 16.350 / 16.223 | PASS |
| 31 | 47.9938 | 14.578 | FAIL |
| 32 | 48.0109 | 14.505 | FAIL |
| 36 | 50.2391 | 13.440 | PASS |

同一配置的两次 target=30 相差约 0.4 秒，说明 27--30 之间不足 0.4 秒的
差别不能稳定视为收益。30 的最好值最低，而且保持原课程输出；因此没有
理由换成更脆弱的新分解。

24/25/26/31/32 的约束上限仍通过，但轨迹 RMS 约 12.44%，远高于课程
0.1% 阈值。这说明 Block 边界会参与 ghost/boundary 数值路径，分解不是
可以只按线程利用率任意改变的纯调度参数。

## 4. 正式 profile

最终选定 target=30 后重新运行完整 `perf stat + perf record`：

    profile/abe-20260822T115654Z-14

| 指标 | 阶段一 | 前一版 P2 |
|---|---:|---:|
| Evolve | 43.3009 s | 43.4695 s |
| Total | 50.5116 s | 50.6337 s |
| 平均使用 CPU | 16.299 / 30 | 16.271 / 30 |
| IPC | 1.87 | 1.86 |
| branch miss | 0.43% | 0.42% |
| L1D miss | 2.65% | 2.65% |
| LLC load miss | 48.13% | 48.29% |
| dTLB miss | 2.21% | 2.22% |

0.39% 的 Evolve 差异处于运行波动范围。本阶段 profile 的价值是确认：
可配置逻辑和正式入口没有造成退化，剩余瓶颈也没有改变。

flat profile 的主要热点为：

| 热点 | 周期占比 |
|---|---:|
| `compute_rhs_bssn` 自身 | 43.92% |
| `memcpy` | 8.76% |
| `kodis` | 8.30% |
| `fdderivs` | 7.86% |
| `lopsided` | 6.73% |
| `memset` | 5.65% |
| `prolong3` | 4.06% |
| `fderivs` | 3.19% |

两遍 profile 的四类数值输出逐字节一致；课程检查轨迹 RMS 为 0，约束
通过。perf record 丢失样本为 0。

## 5. 正式运行入口如何改变

`hpc_cpu.sh` 现在显式启用 `AMSS_ENABLE_OPENMP=ON` 和
`AMSS_ENABLE_OMP_ONLY=ON`。Python 驱动收到 `AMSS_OMP_ONLY_RUN=1` 时
直接执行一次 `./ABE`，不再调用 `mpiexec -n 30`。

物理核心数不是直接使用 `nproc`：当前 lab4 作业给出 60 个逻辑 CPU，
实际是 30 个带 SMT 的物理核心；最终评测则是 60 个无 SMT 的物理核心。
新脚本读取当前进程 affinity mask，并按 socket/core id 去重，因此两种
环境会分别得到 30 和 60。随后设置：

    OMP_NUM_THREADS=<cpuset 中物理核心数>
    OMP_PLACES=cores
    OMP_PROC_BIND=close

这样不会在当前节点误用 60 个 SMT 线程，也不会在最终 60 核节点只使用
30 核。当前默认 Block 目标随线程数变化：本节点为 30，评测节点为 60。

## 6. 对下一阶段的含义

阶段一排除了“全局调低 Block 数”这条简单路线。平均 16.3/30 不能只用
32 个静态 Block 的尾波解释，因为高频移动层已经有 30 个 Block。下一步
必须在低 Block 数阶段或热点 kernel 内增加更细粒度的并行，同时避免在
已有 30 个 Block 的层上形成嵌套线程和过量并发。
