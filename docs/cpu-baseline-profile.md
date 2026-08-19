# CPU baseline、热点与集群硬件记录

记录日期：2026-08-19。除特别说明外，测量均使用 `lab4` CPU 队列的
`-c60 -m100Gi -t30m` 配额和 `AMSS_NCKU_Input.py`。

## 1. 运行方法

仓库根目录的 `hpc_cpu.sh` 是统一入口。不要在脚本名后加参数，否则 `hpc`
不会读取脚本头部的 `#HPC` 指令。通过环境变量选择模式：

```bash
hpc submit ./hpc_cpu.sh
AMSS_JOB_MODE=stat hpc submit ./hpc_cpu.sh
AMSS_JOB_MODE=record hpc submit ./hpc_cpu.sh
AMSS_JOB_MODE=topology hpc submit ./hpc_cpu.sh
```

脚本申请 60 个逻辑 CPU、100 GiB 内存和 30 分钟，使用 30 个 MPI rank，并以
`--map-by core --bind-to core` 绑定。每次运行使用独立输出目录；日志和 profile 原始
数据保存在 `profile/`，不进入 Git。`scripts/collect_cpu_info.sh` 可单独采集 CPU、
cache、NUMA 和编译器信息。

## 2. Baseline

完整、未修改算法的 baseline 作业为 116413。构建成功，TwoPuncture 初值阶段约
294.6 秒，ABE 演化从 `t=0` 到 `t=33`；队列在 30 分钟硬上限处终止作业。

| 项目 | 实测/推算 |
| --- | ---: |
| TwoPuncture | 294.6 s |
| ABE `Before Evolve` | 3.72 s |
| ABE 每个外层时间单位（`t=1..33`） | 平均约 43.1 s |
| ABE 演化到 `t=40` | 约 1724 s（线性推算） |
| 完整端到端 | 约 2020 s，超过 1800 s 队列限制 |

对已产生的 `t=1..33` 数据运行 checker：波形 33/100 组匹配，RMS error 为 0；
约束检查 33 组、9 个 refinement level 全部通过，最终结果为 PASS。因此超时是性能
问题，不是数值错误。另有一份使用已验证 TwoPuncture 初值缓存的完整 ABE 作业，
用于验证演化阶段；缓存运行不应当作为正式端到端 baseline 成绩。

## 3. 粗略 profile 与热点

`perf stat -d -d` 覆盖约 180.27 秒（接近 4 个演化时间单位）：

| 指标 | 结果 |
| --- | ---: |
| 平均活跃 CPU | 29.924 |
| cycles | 15.45e12，平均 2.864 GHz |
| instructions / IPC | 29.26e12 / 1.89 |
| branch miss | 1.07% |
| L1D miss | 0.36% |
| LLC load miss | 50.52% |
| dTLB load miss | 7.59% |

`perf record -m 1 -F 49 --call-graph fp` 采得 220120 个 userspace cycles 样本、
无丢样。主要 flat profile：

| 热点 | cycles 占比 |
| --- | ---: |
| Open MPI shared-memory BTL：`mca_btl_sm_poll_handle_frag` | 58.98% |
| `compute_rhs_bssn_` | 6.10% |
| 两个 `libmpi` 内部地址 | 11.13% |
| 其他 OpenPAL / `opal_progress` | 约 6% |
| `polint_` / `__memcpy_sve` | 1.54% / 1.17% |
| `kodis_` / `fdderivs_` / `lopsided_` | 1.13% / 1.06% / 0.89% |
| `prolong3_` | 0.56% |

主要瓶颈是 MPI 共享内存通信/progress 和同步等待，而不是单一 Fortran 算子。程序
输出还显示 level 0 只有约 9 个 rank 有网格块，30 个 rank 的负载不均衡；
`Parallel.C` 的 ghost-zone `Isend/Irecv/Waitall`、collective 和 AMR 层间同步会把
负载不均衡放大为 poll 时间。高 LLC miss 与 dTLB miss 说明大数组、通信缓冲和 AMR
离散访问也有明显代价。IPC 包含 MPI 自旋，不能代表纯计算 kernel 效率。

优化优先级应是：先检查网格块到 rank 的负载分配并减少或聚合同步；再针对
`compute_rhs_bssn_` 及 `fdderivs_`、`lopsided_`、`kodis_` stencil 检查循环顺序、
临时数组和自动向量化；最后处理 restriction/prolongation 和内存分配。直接先改占比
约 1% 的算子，整体收益上限很小。

## 4. 程序运行流程

1. `run.sh` 设置环境并调用 `AMSS_NCKU.py`。
2. Python driver 读取 `AMSS_NCKU_Input.py`，生成运行目录、网格/AMR 参数文件和
   两个可执行程序需要的 parfile。
3. `TwoPuncture` 用 Newton/BiCGSTAB 求解黑洞初值，输出 `Ansorg.psid` 和新的
   puncture 参数。
4. driver 合并参数、复制可执行文件，并以 30 个 MPI rank 启动 `ABE`。
5. `ABE` 建立 MPI、`cgh`、patch/block、BSSN 变量和初值。
6. `bssn_class::Evolve()` 调用 `RecursiveStep()` 做 AMR 递归子步；每层 `Step()`
   包含 predictor 加三次 corrector、BSSN RHS、RK4 更新、边界条件和 MPI halo 同步。
   RHS 内执行有限差分、偏置导数和 Kreiss-Oliger dissipation；层间执行 restriction、
   prolongation 和必要的 regrid。
7. 每个输出点计算约束、视界/ADM/Psi4 等诊断并写出 binary output。Python driver
   最后整理输出和绘图，checker 对波形与约束文件判分。

关键调用入口是 `src/bssn-evolution.C` 中的 `Evolve`、`RecursiveStep`、`Step`，
MPI halo 交换位于 `src/Parallel.C`，主要数值 RHS 位于 `src/bssn_rhs.f90`。

## 5. CPU、向量化与 NUMA

以下信息来自作业分配内部的 `lscpu`、`numactl -H`、sysfs、`prctl` 和 GCC target
查询。全机信息与单个作业配额必须分开看。

| 项目 | 实测 |
| --- | --- |
| 架构 / 型号 | AArch64，HiSilicon TaiShan-v120（队列节点名为 920B） |
| 全机拓扑 | 2 socket，128 physical core，256 logical CPU |
| 作业 `-c60` 实际 cpuset | 60 logical CPU = 30 physical core 的 SMT sibling |
| 全机 NUMA | 4 node，每 node 64 logical CPU / 32 core，约 128 GiB |
| 本作业内存节点 | 单一 NUMA node |
| L1D / L1I | 各 64 KiB/core，64-byte line |
| L2 | 1280 KiB/core，64-byte line |
| L3 | 56 MiB/NUMA node，128-byte coherency line |
| 频率 | 0.4--2.9 GHz；profile 平均 2.864 GHz |
| Neon / SVE | Neon 128-bit；当前进程 SVE VL = 256-bit |

NUMA node 的距离矩阵中，本地为 10；同一 socket 内另一 node 为 12；跨 socket 为
35--40。因 CPU 和内存都落在单一 NUMA node，当前 30-rank 作业不需要跨 socket
访存；但共享内存通信、first-touch 和 rank 绑定仍然重要。

CPU flags 和 `gcc -march=native -dM` 表明支持 ASIMD/Neon、SVE、FP16、BF16、
FP32/FP64 matrix multiply、dot-product、I8MM、原子操作等；未观察到 SVE2。
256-bit SVE 每条向量可容纳 4 个 FP64 或 8 个 FP32 元素，单条 FP64 vector FMA
完成 8 FLOP。仅凭公开资料和系统寄存器不能可靠推出 TaiShan-v120 的 FMA 发射端口
数，因此不虚构峰值 FLOP/cycle。当前 baseline 只用了通用 `-O3`，并未显式使用
`-march=native`，所以 libc 的 `__memcpy_sve` 出现在 profile 中不等于应用 kernel
已经生成 SVE 指令。

本课程 `AGENT.md` 写的是“60 个物理核、无超线程”，但当前节点实测的 CPU ID、
core ID 和 cache sibling 都显示 `-c60` 是 30 个物理核的两个硬件线程；课程网页也
将 CPU 作业描述为 30 physical cores。公开的早期 Kunpeng 920 FAQ 又写着每物理核
一个线程。这很可能是当前 920B/TaiShan-v120 节点与公开早期 920 型号或其拓扑呈现
方式不同。提交与绑定应以当前队列中的实测 cpuset 为准，而不是据旧型号资料猜测。

## 6. 资料来源

- Huawei Kunpeng 920 产品页：<https://www.hikunpeng.com/compute/kunpeng920>
- Huawei Kunpeng Programming and Tuning Guide：
  <https://www.hikunpeng.com/document/detail/en/perftuning/progtuneg/kunpengprogramming_05_0001.html>
- Huawei Kunpeng Hardware FAQ：
  <https://www.hikunpeng.com/document/detail/en/kunpengfaq/productfaq/hardwarefaq/hardware_faq_0001.html>
- Arm SVE Programmer's Guide：
  <https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/SVE%20programmers%20guide/102476_0001_00_en_introduction-to-sve.pdf>

原始日志、checker 输出、`perf stat` 和 `perf report` 保留在本地 `profile/` 目录。
