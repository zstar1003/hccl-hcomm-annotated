# hccl.manage.api

hccl.manage.api模块提供集合通信group管理接口，包括group的创建与销毁、rank信息查询等能力。该模块所有接口必须在集合通信初始化完成之后调用。

## 接口列表

- [create_group](create_group.md)：创建以group为名字的集合通信group。
- [destroy_group](destroy_group.md)：销毁group。
- [get_rank_size](get_rank_size.md)：获取group内rank数量。
- [get_rank_id](get_rank_id.md)：获取device在group中对应的rank序号。
- [get_local_rank_size](get_local_rank_size.md)：获取group内device所在服务器内的local rank数量。
- [get_local_rank_id](get_local_rank_id.md)：获取group内device所在服务器内的local rank序号。
- [get_world_rank_from_group_rank](get_world_rank_from_group_rank.md)：根据进程在group中的rank id获取对应的world rank id。
- [get_group_rank_from_world_rank](get_group_rank_from_world_rank.md)：根据world rank id获取该进程在group中的group rank id。
