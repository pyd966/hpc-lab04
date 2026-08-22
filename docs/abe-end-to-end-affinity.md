# ABE 端到端补充报告：Python affinity 问题与 t=40 验证

## 1. 结论

正式运行入口曾经把 60 个逻辑 CPU 的作业错误收窄到 2 个逻辑 CPU。
这不是 ABE 的 OpenMP 分解失效，而是 Python 驱动导入 `matplotlib`
及其本地数值库后，主进程在 `OMP_PROC_BIND=close` 环境中只保留了
第一个物理核对应的两个硬件线程。TwoPuncture 和 ABE 都是 Python 的
子进程，因此继承了这个错误 affinity。

修复后，Python 在导入本地数值库前保存调度器分配的完整 cpuset，导入后
立即恢复，并在 TwoPuncture 结束后再次核验。最终作业在两个计算阶段前后
都保持 60 个逻辑 CPU，即当前节点上的 30 个物理核。

修复后的完整 `TwoPuncture + Evolve(40)` 数值检查通过，但当前 30
物理核节点上的 Python 计时为 493.738 秒，尚未达到 330 秒目标。

## 2. 问题如何暴露

短 profile 直接执行 ABE，`t=0..4` 通常约 43 秒，平均使用约 16 个核。
最初的完整作业却每个时间单位需要约 90--190 秒，调度器记录的平均 CPU
约 1.8 核、峰值约 2.0 核。输入、数值结果和每步累计 CPU 时间都正常，
只有墙钟时间异常。

在主驱动最早可观察的位置加入 `os.sched_getaffinity(0)` 后，作业
`140185` 给出了直接证据：

    调度器 cpuset: 64-123，共 60 个逻辑 CPU
    Python 导入后: [64, 65]，只剩 2 个逻辑 CPU

因此 30 个 OpenMP 线程虽然被创建出来，却全部只能在一个物理核上运行。
这也解释了为什么日志中的 Block 数和 `OMP_NUM_THREADS=30` 都正确，
实际 CPU 利用率却极低。

## 3. 排除过的其他因素

为避免把相关性误判为原因，使用同一份 `t=1` 输入做了四组受控 A/B：

| 对比 | A (s) | B (s) | 结论 |
|---|---:|---:|---|
| 默认栈 / unlimited 栈 | 13.595 | 13.332 | 栈限制不是原因 |
| profile / production 二进制 | 13.556 | 14.026 | `-g -fno-omit-frame-pointer` 不是原因 |
| 直接 exec / Python wrapper | 12.135 | 12.120 | `ABE | tee` 不是原因 |
| `TotalTime=1` / `40`，均只跑一步 | 12.132 | 12.378 | 结束时间不是原因 |

新生成的 `Ansorg.psid` 与 profile 缓存文件除首行生成时间外完全一致，
也排除了 TwoPuncture 初值差异。

## 4. 修复方式

`AMSS_NCKU_Program.py` 现在执行以下顺序：

1. 只导入 Python 标准库，立即保存调度器 affinity；
2. 导入 `matplotlib`，随后恢复完整 affinity；
3. 启动 TwoPuncture；
4. TwoPuncture 返回后再次检查 affinity，若有变化则恢复；
5. 以完整 affinity 启动单进程 OpenMP ABE。

另一个启动修复保留了外部 `OMP_NUM_THREADS`：输入文件中的旧值
`OMP_threads=1` 只作为交互运行的默认值，不能覆盖 HPC 脚本检测出的
30 或 60 个物理核配置。

## 5. 最终 t=40 结果

作业 `140226`，commit `bda108a`，当前节点为 30 个物理核、60 个
逻辑 CPU：

| 指标 | 结果 |
|---|---:|
| ABE Evolve | 473.573 s |
| ABE 内部总时间 | 480.947 s |
| Python `This Program Cost` | 493.738 s |
| `run.sh` 外层 wall time | 504 s |
| 调度器平均 / 峰值 CPU | 18.264 / 21.452 核 |
| 调度器峰值内存 | 3.78 GB |
| 轨迹 RMS | 0 |
| 约束检查 | PASS |
| 最终检查 | PASS |

Python 口径比 330 秒多 163.738 秒，即还需要约 1.50 倍端到端加速。
ABE 演化占 Python 计时约 95.9%，所以下一步应继续优化 ABE 热点，而不是
优先削减绘图或启动开销。

当前结果不能直接代表最终 60 物理核评测环境。30 到 60 物理核可能明显
缩短移动层计算，但最粗层只有 9 个 Block，静态层和内存带宽也不会线性
缩放。应在 60 物理核环境先跑同一份 `t=40` 验收，再决定还需多少内核
级优化；若仍有缺口，优先处理 `compute_rhs`、有限差分/耗散和大块
`memcpy/memset`，因为这些是当前 profile 中占比最高的路径。
