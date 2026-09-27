# 简介

HCCL Python语言的接口用于实现图模式下的框架适配，当前仅用于TensorFlow网络在NPU执行分布式优化。

## 相关概念

| 概念 | 介绍 |
| --- | --- |
| group | 指参与集合通信的进程组，包括：<br>  - hccl_world_group：默认的全局group，包含所有参与集合通信的rank，通过rank table文件创建。<br>  - 自定义group：hccl_world_group包含的进程组的子集，可以通过create_group接口将rank table中的rank定义成不同的group，并行执行集合通信算法。 |
| rank | group中的每个通信实体称为一个rank，每个rank都会分配一个介于0~n-1（n为NPU的数量）的唯一标识。 |
| rank size | - rank size，指整个group的rank数量。<br>  - local rank size，指group内进程在其所在Server内的rank数量。 |
| rank id | - rank id，指进程在group中对应的rank标识序号。范围：0~（rank size-1）。对于用户自定义group，rank在本group内从0开始进行重排；对于hccl_world_group，rank id和world rank id相同。<br>  - world rank id，指进程在hccl_world_group中对应的rank标识序号，范围：0~（rank size-1）。<br>  - local rank id，指group内进程在其所在Server内的rank编号，范围：0~（local rank size-1）。 |
