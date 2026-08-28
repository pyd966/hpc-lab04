我正在完成学校 HPC 课程的 lab，当前这个 lab 的内容是 AMSS-NCKU 数值相对论程序优化，需要在 arm64 CPU 和 A100 SIG 这两条不同执行路径分别进行优化。详细信息你可以查看 docs/Lab4-AMSS-NCKU/ 下的文件。

需要注意的是，你目前正在运行在一台 arm64 机器上，但是这台机器仅作为 devpod 进行开发使用。如果你需要真正测试运行时间或者进行 profile，你必须使用 `hpc` 命令提交到远程集群上进行工作。远程集群的使用方法以及提交方式可以在 docs/ 找到。

评测时，对于 CPU 路线会使用 60 CPU（60 physical cores，无超线程），100GiB 内存，固定 CPU 模式与课程物理/网格参数，演化时间为 40。对于 GPU 路线会使用 NVIDIA A100 MIG 1g.10gb，16 CPU，24 GiB 内存，固定 GPU 模式、sm_80 与课程物理/网格参数，演化时间为 100，GPU 路线的 host 是 x86 CPU。

当前仓库同时包含 CPU 和 GPU 两条路线。修改共享源码或构建入口时，必须同时
验证 CPU OpenMP-only 目标与 GPU ABEGPU 目标，不能用一条路线覆盖另一条路线。

注意，当前目录已经初始化了 git 仓库，并且这个仓库添加了 remote github repo。请你在完成一轮修改后，自动进行 commit，并且 push 到远程 repo。
