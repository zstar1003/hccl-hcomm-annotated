# hccl.split.api

hccl.split.api模块提供反向梯度切分策略设置接口，用于allreduce融合的性能调优。该模块所有接口必须在集合通信初始化完成之后调用。

## 接口列表

- [set_split_strategy_by_idx](set_split_strategy_by_idx.md)：基于梯度的索引id，在集合通信group内设置反向梯度切分策略。
- [set_split_strategy_by_size](set_split_strategy_by_size.md)：基于梯度数据量百分比，在集合通信group内设置反向梯度切分策略。
