# 注释覆盖清单

保留固定版本HCCL/HCOMM的全部10,048个上游文件。累计新增中文导读覆盖48个源码文件、8241个连续注释块、8605行。“注释块”不是函数数；未列明函数仍保留上游原注释。

本轮以AllReduce为例，新增7999条逐物理行说明，覆盖44个文件、305个选定完整函数/注册范围。主线为Ascend950、OPBASE、规则选择器、AICPU_TS、单层Mesh1D、FP32 SUM、普通内存、UB_CTP OneShot；选中函数内的其他条件分支也有逐行说明。空行及已有注释不重复标注。

- [AllReduce阅读指南](READING_GUIDE.zh-CN.md)：主调用树、分支条件、4Rank数据与通知示例。
- [精确函数与行范围](allreduce/COVERAGE.zh-CN.md)：固定提交源码行链接及停止展开的边界。
- [分阶段调用详表](allreduce/CALL_RELATIONS.zh-CN.md)：caller调用点、callee定义与分支。
- [机器逐行清单](allreduce/line-notes.json)：每行快照原文、功能说明、范围和固定源码提交。
- [原AllToAll阅读路线](ALLTOALL_READING_GUIDE.zh-CN.md)：此前功能小段导读。

新增说明统一为`// [中文导读]`独立注释，AllReduce逐行标记另带`[AllReduce逐行 S<快照行号>]`。长语句按物理行解释；遇宏续行或字符串等无法安全插入的位置，说明放在相应完整语句之前，不把注释塞入原语句。

| 文件 | 说明主题 | 累计块 / 行 | 本轮AllReduce逐行数 |
|---|---|---|---|
| [hccl/src/common/alg_env_config.cc](../hccl/src/common/alg_env_config.cc) | AllReduce主链相关函数与内部条件分支 | 21 / 21 | 21 |
| [hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc](../hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc) | AllReduce主链相关函数与内部条件分支 | 52 / 52 | 52 |
| [hccl/src/common/inconsistent_check.cc](../hccl/src/common/inconsistent_check.cc) | AllReduce主链相关函数与内部条件分支 | 149 / 149 | 149 |
| [hccl/src/ops/all_gather/all_gather.cc](../hccl/src/ops/all_gather/all_gather.cc) | AllGather入口、图资源、参数容量、快速路径与算法调度 | 24 / 26 | 0 |
| [hccl/src/ops/all_reduce/algorithm/executor/ins_v2_all_reduce_sole_executor.cc](../hccl/src/ops/all_reduce/algorithm/executor/ins_v2_all_reduce_sole_executor.cc) | AllReduce主链相关函数与内部条件分支 | 241 / 241 | 241 |
| [hccl/src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_one_shot.cc](../hccl/src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_one_shot.cc) | AllReduce主链相关函数与内部条件分支 | 233 / 233 | 233 |
| [hccl/src/ops/all_reduce/all_reduce.cc](../hccl/src/ops/all_reduce/all_reduce.cc) | AllReduce入口、类型/运算校验、参数与对称内存条件、算法调度 | 224 / 251 | 224 |
| [hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc](../hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc) | AllReduce主链相关函数与内部条件分支 | 612 / 612 | 612 |
| [hccl/src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc](../hccl/src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc) | Mesh1D模板资源需求、分片、Peer轮次及Read/Write路径 | 54 / 65 | 0 |
| [hccl/src/ops/all_to_all_v/all_to_all_v.cc](../hccl/src/ops/all_to_all_v/all_to_all_v.cc) | AllToAll/V入口、参数布局、公共调度 | 37 / 47 | 0 |
| [hccl/src/ops/op_common/algorithm/executor/channel/channel.cc](../hccl/src/ops/op_common/algorithm/executor/channel/channel.cc) | AllReduce主链相关函数与内部条件分支 | 146 / 146 | 146 |
| [hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc](../hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc) | AllReduce主链相关函数与内部条件分支 | 25 / 25 | 25 |
| [hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.h](../hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.h) | AllReduce主链相关函数与内部条件分支 | 7 / 7 | 7 |
| [hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc](../hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc) | 设备入口、上下文恢复、任务缓存与编排 | 486 / 523 | 482 |
| [hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.h](../hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.h) | AllReduce主链相关函数与内部条件分支 | 4 / 4 | 4 |
| [hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc](../hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc) | 收发协议、LocalCopy/Reduce、主从同步 | 550 / 581 | 525 |
| [hccl/src/ops/op_common/algorithm/topo_match/topo_match_one_level.cc](../hccl/src/ops/op_common/algorithm/topo_match/topo_match_one_level.cc) | AllReduce主链相关函数与内部条件分支 | 48 / 48 | 48 |
| [hccl/src/ops/op_common/op_common.cc](../hccl/src/ops/op_common/op_common.cc) | 公共调度、资源复用、AICPU/CCU/AIV分支 | 1318 / 1417 | 1294 |
| [hccl/src/ops/op_common/order_launch.cc](../hccl/src/ops/op_common/order_launch.cc) | AllReduce主链相关函数与内部条件分支 | 188 / 188 | 188 |
| [hccl/src/ops/op_common/selector/auto_selector_base.cc](../hccl/src/ops/op_common/selector/auto_selector_base.cc) | AllReduce主链相关函数与内部条件分支 | 337 / 337 | 337 |
| [hccl/src/ops/op_common/selector/execute_selector.cc](../hccl/src/ops/op_common/selector/execute_selector.cc) | AllReduce主链相关函数与内部条件分支 | 36 / 36 | 36 |
| [hccl/src/ops/op_common/selector/selector_engine.cc](../hccl/src/ops/op_common/selector/selector_engine.cc) | AllReduce主链相关函数与内部条件分支 | 258 / 258 | 258 |
| [hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc](../hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc) | 本地/远端数据原语、通知、批提交与域使用保护 | 445 / 487 | 430 |
| [hcomm/src/base_comm/primitives/api_c_adpt/hcomm_channel_c_adpt.cc](../hcomm/src/base_comm/primitives/api_c_adpt/hcomm_channel_c_adpt.cc) | 集合通信创建入口与公开创建入口的区别 | 76 / 82 | 65 |
| [hcomm/src/base_comm/primitives/launch_context.cc](../hcomm/src/base_comm/primitives/launch_context.cc) | AllReduce主链相关函数与内部条件分支 | 69 / 69 | 69 |
| [hcomm/src/base_comm/primitives/launch_context.h](../hcomm/src/base_comm/primitives/launch_context.h) | AllReduce主链相关函数与内部条件分支 | 17 / 17 | 17 |
| [hcomm/src/base_comm/resources/comm_engine_res/threads/aicpu_ts_thread.cc](../hcomm/src/base_comm/resources/comm_engine_res/threads/aicpu_ts_thread.cc) | AllReduce主链相关函数与内部条件分支 | 87 / 87 | 87 |
| [hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc](../hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc) | AllReduce主链相关函数与内部条件分支 | 48 / 48 | 48 |
| [hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc](../hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc) | AllReduce主链相关函数与内部条件分支 | 29 / 29 | 29 |
| [hcomm/src/base_comm/resources/endpoint_pairs/channels/channel.cc](../hcomm/src/base_comm/resources/endpoint_pairs/channels/channel.cc) | AllReduce主链相关函数与内部条件分支 | 97 / 97 | 97 |
| [hcomm/src/base_comm/resources/endpoint_pairs/channels/channel_process.cc](../hcomm/src/base_comm/resources/endpoint_pairs/channels/channel_process.cc) | AllReduce主链相关函数与内部条件分支 | 43 / 43 | 43 |
| [hcomm/src/base_comm/resources/endpoint_pairs/endpoint_pair.cc](../hcomm/src/base_comm/resources/endpoint_pairs/endpoint_pair.cc) | 按Engine及复用槽位创建或更新通道 | 35 / 40 | 27 |
| [hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc](../hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc) | AllReduce主链相关函数与内部条件分支 | 34 / 34 | 34 |
| [hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_res_c_adpt.cc](../hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_res_c_adpt.cc) | HcclChannelAcquire申请边界 | 205 / 221 | 203 |
| [hcomm/src/coll_communicator_mgr/api_c_adpt/exchange_info_c_adpt.cc](../hcomm/src/coll_communicator_mgr/api_c_adpt/exchange_info_c_adpt.cc) | 一致性元信息登记、读取和清理 | 8 / 11 | 0 |
| [hcomm/src/coll_communicator_mgr/api_c_adpt/resource/channel_c_adpt.cc](../hcomm/src/coll_communicator_mgr/api_c_adpt/resource/channel_c_adpt.cc) | 通道通知数、远端CCL/注册内存查询 | 73 / 82 | 67 |
| [hcomm/src/coll_communicator_mgr/api_c_adpt/resource/comm_mem_c_adpt.cc](../hcomm/src/coll_communicator_mgr/api_c_adpt/resource/comm_mem_c_adpt.cc) | 本地CCL缓冲区获取及生命周期 | 71 / 77 | 65 |
| [hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc](../hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc) | Thread申请、TS枚举转换、用户流包装 | 238 / 251 | 232 |
| [hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/endpoints/endpoint_mgr.cc](../hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/endpoints/endpoint_mgr.cc) | Endpoint缓存、内存注册、句柄选择与释放顺序 | 81 / 97 | 76 |
| [hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/my_rank.cc](../hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/my_rank.cc) | Socket、Endpoint、Channel创建/复用与一致性交换 | 543 / 574 | 536 |
| [hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc](../hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc) | AllReduce主链相关函数与内部条件分支 | 64 / 64 | 64 |
| [hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc](../hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc) | AllReduce主链相关函数与内部条件分支 | 200 / 200 | 200 |
| [hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.h](../hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.h) | AllReduce主链相关函数与内部条件分支 | 5 / 5 | 5 |
| [hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc](../hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc) | AllReduce主链相关函数与内部条件分支 | 193 / 193 | 193 |
| [hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc](../hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc) | AllReduce主链相关函数与内部条件分支 | 40 / 40 | 40 |
| [hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h](../hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h) | AllReduce主链相关函数与内部条件分支 | 71 / 71 | 71 |
| [hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/stream_lite.cc](../hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/stream_lite.cc) | AllReduce主链相关函数与内部条件分支 | 1 / 1 | 1 |
| [hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc](../hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc) | AllReduce主链相关函数与内部条件分支 | 418 / 418 | 418 |

## 覆盖边界

精确范围以AllReduce机器清单为准，不宣称全仓、全部AllReduce算法或所有递归依赖已逐行注释。其他executor/template、CCU/AIV设备算法、其他协议、通信域初始化、完整拓扑/内存/Socket状态机、通用配置/序列化/日志/cache工具和外部runtime/driver实现均有未展开范围，详见[AllReduce覆盖清单](allreduce/COVERAGE.zh-CN.md)。

本例的新HCOMM数据面确实调用legacy/ascend950中的UB轻量传输、连接和RTSQ；已列出的函数做了逐行说明。legacy目录名不等于一次算子调用进入了旧入口回退。

[HCCL目录导读](HCCL_DIRECTORY_GUIDE.zh-CN.md)与[HCOMM目录导读](HCOMM_DIRECTORY_GUIDE.zh-CN.md)及完整索引覆盖全部文件；目录导航与源码新增注释是不同的覆盖范围。这里没有改动原始代码、原注释或构建逻辑，也未在CANN设备环境编译运行。
