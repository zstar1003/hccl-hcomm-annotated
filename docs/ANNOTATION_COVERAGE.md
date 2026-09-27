# 首批注释覆盖清单

范围：固定版本的核心跨仓路径。全量上游文件10,048个，其中HCCL 1,828个、HCOMM 8,220个；这些数量含源码、测试和文档，不能解释为全部是函数或源码文件。

新增注释覆盖17个`.cc`文件、95个连续注释块、196行。**“注释块”不是“已完整注释的函数数”。** 某些函数内部有多块说明；某些文件仅补了入口说明。机器可校验的文件列表见[annotations.json](../annotations.json)。

| 文件 | 本批新增说明 | 块 / 行 |
|---|---|---|
| [all_to_all_v.cc](../hccl/src/ops/all_to_all_v/all_to_all_v.cc) | 入口、参数跨度/变长描述、调度分支、参数生命周期 | 8 / 18 |
| [all_gather.cc](../hccl/src/ops/all_gather/all_gather.cc) | 顶层入口语义与数量单位 | 1 / 3 |
| [all_reduce.cc](../hccl/src/ops/all_reduce/all_reduce.cc) | 顶层归约语义与入口分流 | 1 / 3 |
| [op_common.cc](../hccl/src/ops/op_common/op_common.cc) | 执行枢纽、拓扑缓存、资源复用、AICPU/CCU/AIV资源、Host下发 | 23 / 50 |
| [kernel_launch.cc](../hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc) | AICPU新流程入口、描述恢复、task缓存、编排与通知 | 7 / 10 |
| [ins_temp_all_to_all_v_mesh_1D.cc](../hccl/src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc) | Mesh1D资源需求、Peer轮次、Read/Write、切片与中转 | 8 / 19 |
| [alg_data_trans_wrapper.cc](../hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc) | 单向/双向收发协议、本地复制/归约、主从Thread同步 | 10 / 16 |
| [coll_comm_res_c_adpt.cc](../hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_res_c_adpt.cc) | 通道申请、V2/兼容分支、MyRank调用边界 | 2 / 6 |
| [exchange_info_c_adpt.cc](../hcomm/src/coll_communicator_mgr/api_c_adpt/exchange_info_c_adpt.cc) | 算子元信息登记、精确长度读取及消费、重置 | 3 / 6 |
| [channel_c_adpt.cc](../hcomm/src/coll_communicator_mgr/api_c_adpt/resource/channel_c_adpt.cc) | 通道通知数、远端CCL、远端注册内存查询 | 3 / 6 |
| [comm_mem_c_adpt.cc](../hcomm/src/coll_communicator_mgr/api_c_adpt/resource/comm_mem_c_adpt.cc) | 本地CCL获取、单Rank空返回、所有权 | 1 / 3 |
| [thread_c_adpt.cc](../hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc) | WithConfig/兼容申请、TS枚举转换、用户流包装 | 3 / 6 |
| [my_rank.cc](../hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/my_rank.cc) | 建链阶段、Endpoint与内存选择、资源复用和返回属性 | 5 / 12 |
| [endpoint_mgr.cc](../hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/endpoints/endpoint_mgr.cc) | Endpoint缓存、注册版本、按tag选句柄、释放先后 | 4 / 9 |
| [endpoint_pair.cc](../hcomm/src/base_comm/resources/endpoint_pairs/endpoint_pair.cc) | 按Engine/槽位新建或复用通道、更新附加内存 | 1 / 3 |
| [hcomm_channel_c_adpt.cc](../hcomm/src/base_comm/primitives/api_c_adpt/hcomm_channel_c_adpt.cc) | Collective与公开Create入口差异、内存描述更新 | 3 / 5 |
| [aicpu_ts_primitives_c_adpt.cc](../hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc) | 复制/读写/归约、Thread/Channel通知、提交模式、域引用 | 12 / 21 |

## 尚未逐函数补充的范围

- 通信域创建/销毁、完整RankGraph拓扑实现与底层驱动封装。
- AllGather/AllReduce的各算法executor/template，以及其他集合通信与点对点算子的完整路径。
- AllToAll的其他算法、所有外层分块executor、完整对称内存和图模式分支。
- CCU C++指令与Kernel生成、AIV核内数据面、全部Socket/Transport/协议状态机。
- legacy、experimental、全部异常恢复路径、UT/ST以及其他已有文档。

这些内容仍保留全量上游代码及原注释，但不能计入本次新增注释覆盖。已有的中文注释不会被改写成“本次新增”。

## 后续补注释的标准

先读完整函数和关键依赖，再写职责、参数单位、条件路径与资源归属。只在代码证据支持时写调用关系；没有运行验证就不写成实测结论。每批保持原始文本，更新机器清单、本文统计及验证记录，不把同义复述当作有效覆盖。
