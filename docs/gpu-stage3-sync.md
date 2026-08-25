# GPU 阶段 3：缩小同步范围

## 结论

阶段 3 保留。它没有改变 RHS 或其他 kernel 的数学流程，只重建了已有
producer-consumer 边界上的等待范围。最终 `t=5` 端到端均值从 baseline 的
`104.962209 s` 降至 `100.018310 s`，下降 `4.71%`；三次运行全部
`FINAL: PASS`，样本标准差为 `0.064500 s`。

## 依赖分析

`Block` 构造时从 `GPUManager` 轮转取得四个 stream，同一个 Block 的 kernel
始终在同一个 stream 上按提交顺序执行。不同 Block 之间没有隐含的逐 kernel
依赖，真正的跨 Block 消费者是 ghost exchange、restrict/prolong、约束插值
和 RK 阶段的后续操作。

原实现把这些边界统一实现成 `cudaDeviceSynchronize()`。特别是
`Parallel::gpu_data_packer()` 在长度查询（没有发射 kernel）时也会触发全设备
同步，实际 pack/unpack 时则等待了不相关 Block 的 stream。

## 实现内容

1. 在 [gpu_manager.h](/home/h3250106394/lab04/src/gpu_manager.h) 和
   [gpu_manager.cu](/home/h3250106394/lab04/src/gpu_manager.cu) 增加
   `synchronize_streams()`，只同步传入的唯一 stream 集合。
2. 在 [Parallel_GPU.cpp](/home/h3250106394/lab04/src/Parallel_GPU.cpp) 的
   `gpu_data_packer()` 中记录实际发射 pack/restrict/prolong/unpack kernel 的
   source/destination stream。长度查询不再同步；CPU staging 只等待本次通信
   触碰的 stream。
3. 在 [bssn_step_gpu.C](/home/h3250106394/lab04/src/bssn_step_gpu.C) 中去掉
   predictor/corrector 后、ghost exchange 前的两次全设备等待。后续
   `Sync_GPU()` 的 pack/unpack 局部等待和同一 Block 的 stream 顺序分别承担
   跨 Block 与同 Block 依赖。
4. 在 [bssn_gpu_class.C](/home/h3250106394/lab04/src/bssn_gpu_class.C) 中：
   约束计算和约束插值进入 ghost exchange 前不再全局等待；约束插值的共享
   `d_shellf` 只等待实际有 active point 的 Block stream；单个 Block 的
   `move_to_cpu()` 只等待该 Block 的 stream。

没有修改 kernel 的 block/grid、寄存器限制或数值公式，也没有跳过已有的
checker/约束检查。

## 正确性测试

| 测试 | 结果 |
| --- | --- |
| `t=2` 短测试 | `57.264935 s`，trajectory RMS `0`，constraints PASS，FINAL PASS |
| Nsys `t=0..1` | trajectory RMS `0`，constraints PASS，FINAL PASS |
| `t=5` 三次 | `100.022230 / 100.080762 / 99.951940 s`，全部 FINAL PASS |

## Nsys 对照

对照 artifacts：

- baseline：[gpu-nsys-20260824T120352Z-65](/home/h3250106394/lab04/profile/gpu-nsys-20260824T120352Z-65)
- 阶段 3：[gpu-nsys-20260825T030934Z-66](/home/h3250106394/lab04/profile/gpu-nsys-20260825T030934Z-66)

| CUDA API / kernel | baseline | 阶段 3 | 变化 |
| --- | ---: | ---: | ---: |
| `cudaDeviceSynchronize` 次数 | 1044 | 215 | -79.4% |
| `cudaStreamSynchronize` 次数 | 63 | 472 | 局部等待替代全局等待 |
| `cudaDeviceSynchronize` 时间 | 13.955 s | 3.408 s | -75.6% |
| `cudaStreamSynchronize` 时间 | 0.0001 s | 5.737 s | 参与 stream 的真实等待 |
| 两类同步 API 合计 | 13.955 s | 9.145 s | -34.5% |
| kernel duration sum | 14.434 s | 14.429 s | -0.03% |

kernel 总时长基本不变是预期结果：阶段 3 的收益来自让 host 不再等待无关
Block，而不是减少算术工作。由于同步被推迟到真实的 host copy/staging 边界，
Nsys 中 `cudaMemcpy` API 时间会上升；端到端重复测试仍显示稳定的约 `4.7%`
收益，因此没有把 API 单项时间下降误认为 kernel 加速。

## 后续方向

阶段 3 仍有进一步空间：当前同节点 D2D pack/unpack 为了保证临时 buffer
生命周期仍有 host-side stream wait，可以用按 buffer 管理的 CUDA event
record/wait 消除这部分 host 阻塞；跨节点 CPU staging 则仍必须等待参与通信
的 source stream。下一阶段应优先处理 prolong/restrict 的跨变量批处理，避免
把同步优化误当成 kernel 算术优化。
