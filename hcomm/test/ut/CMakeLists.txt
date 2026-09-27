# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(HCCL_CODE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
message(STATUS "${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/common/ *******************************************")
set(HCCL_LLT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(UT_COMMON_INCLUDE_LIST
    ${HCCL_CODE_ROOT}
    ${HCCL_CODE_ROOT}/include
    ${HCCL_CODE_ROOT}/include/hccl
    ${HCCL_CODE_ROOT}/pkg_inc
    ${HCCL_CODE_ROOT}/pkg_inc/hccl
    ${HCCL_CODE_ROOT}/pkg_inc/hcomm
    ${HCCL_CODE_ROOT}/pkg_inc/hcomm/ccu
    ${HCCL_CODE_ROOT}/pkg_inc/legacy
    ${HCCL_CODE_ROOT}/pkg_inc/legacy/hccl
    ${HCCL_CODE_ROOT}/include/ccu

    # src/pub_inc 目录下头文件 (legacy/ascend910)
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/pub_inc/inner
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/pub_inc/aicpu
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/pub_inc/new

    # src/hccd 目录下头文件 (legacy/ascend910)
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/hccd

    # src/common 目录下头文件 (legacy/ascend910)
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/health
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/stream
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/launch_device
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/src
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/src/host
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/src/topo
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/debug/config
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/debug/profiling
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/debug/profiling/inc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/debug/profiling/inc/host
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/launch_aicpu
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common

    # src/framework 目录下头文件 (legacy/ascend910)
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/inc/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/inc/host
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/cluster_maintenance/health/heartbeat/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/cluster_maintenance/detect/detect_connect_anomalies/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/cluster_maintenance/recovery/operator_retry
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/cluster_maintenance/snapshot
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/cluster_maintenance/aclgraph
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/host
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/task
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/topo
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/exception
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/config
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/onesided_memory_management
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/order_launch
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/zero_copy
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/one_sided_service
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/resource_manager
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/independent_op
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/independent_op/channel/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/independent_op/channel/device
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/common
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/group_schedule_mgr
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/config_mgr
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/threads
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/threads/order_launch_thread_mgr
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/notify
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine

    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_mems
    ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res/launch
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/hcom/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/hcom/gradient_segment
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/op_base/src/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/group
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/nslbdp/

    # src/platform 目录下头文件 (legacy/ascend910)
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/common/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/common/unique/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/common/buffer_manager
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/common/unfold_cache
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/ping_mesh/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/inc/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/inc/adapter
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/error_manager
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/mem
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/notify/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/rma_buffer
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/transport/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/transport/device/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/transport/host/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/transport/heterog/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/transport/onesided
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/transport/onesided/device
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/socket/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/dispatcher_ctx
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/task/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/aiv_communication/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/debug/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/debug/dfx/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/debug/dfx/trace/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/task/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/task/rtsq_interact/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/typical

    # hccp (moved to base_comm/resources)
    ${HCCL_CODE_ROOT}/src/base_comm/resources/hccp/inc
    ${HCCL_CODE_ROOT}/src/base_comm/resources/hccp/inc/network
    ${HCCL_CODE_ROOT}/src/base_comm/resources/hccp/inc/private/network

    # hccl/algorithm 目录下头文件 (legacy/ascend910)
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/pub_inc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/inc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/temp_all_gather
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/temp_all_reduce
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/temp_alltoall
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/temp_alltoallv
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/temp_broadcast
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/temp_gather
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/temp_reduce
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/temp_reduce_scatter
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/temp_scatter
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/temp_send_recv
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/component
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/component/inc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/inc_all_reduce_deter
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/communicator
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/mc2_handler
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/inc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/legacy
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/legacy/operator
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/communicator/legacy
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/task
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/operator
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/operator/registry
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/resource_manager
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/registry
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/310P
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/310P
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v/310P
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/310P
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/310P
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_scatter
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_broadcast
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_broadcast/310P

    ${HCCL_CODE_ROOT}/test/hccl_api_single_thread_test

    # hccl/operator目录下头文件
    ${HCCL_CODE_ROOT}/operator

    # proto.h头文件
    ${CMAKE_BINARY_DIR}/proto/hccl_llt

    # /src/framework/device/目录下头文件 (legacy/ascend910)
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/device/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/device/inc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/device/aicpu_kfc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/device/aicpu_kfc/inc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/device/aicpu_kfc/common
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/device/common
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/device/utils
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/device/framework

    # legacy include (ascend950)
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/include
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/common/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/common/utils/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/common/exception/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/common/types/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/service/collective/primitive
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/resource_manager/socket/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/resource_manager/connection/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/communicator/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/communicator/aicpu/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/communicator/aicpu/daemon
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/communicator/aicpu/inc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/communicator/aicpu/daemon
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/topo/
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/common/
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_graph/
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_table_info/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/env_config/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/interface/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/service/collective/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/service/collective/primitive/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/external_system/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/socket/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/transport/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/transport/aicpu/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/notify/aicpu/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/stream/aicpu/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/buffer/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/buffer/aicpu/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/connection/aicpu/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/connection/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/pub_inc/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/common/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/task/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/notify/

    # tbe头文件 (legacy/ascend910)
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/legacy/hccl_tbe_task/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/legacy/common/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/legacy/graph_ctx_mgr/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/legacy/inc/

    # hcce头文件
    ${HCCL_CODE_ROOT}/src/engines/hcclengine/inc

    ${HCCL_CODE_ROOT}/test/ut/stub
    ${HCCL_CODE_ROOT}/test/ut/stub/legacy/
    ${HCCL_CODE_ROOT}/test/stub/depends/pkg_inc/
    ${HCCL_CODE_ROOT}/test/stub/depends/pkg_inc/aicpu/
    ${HCCL_CODE_ROOT}/test/stub/depends/pkg_inc/aicpu/common/
    ${HCCL_CODE_ROOT}/test/stub/depends/pkg_inc/runtime/
    ${HCCL_CODE_ROOT}/test/stub/depends/pkg_inc/profiling/
    ${HCCL_CODE_ROOT}/test/stub/depends/pkg_inc/base/
    ${HCCL_CODE_ROOT}/test/stub/depends/include/base/context_builder
    ${HCCL_CODE_ROOT}/test/stub/depends/pkg_inc/dump/
    ${HCCL_CODE_ROOT}/test/stub/depends/pkg_inc/trace/
    ${HCCL_CODE_ROOT}/test/stub/depends/pkg_inc/asc/include/adv_api/hccl/internal

    ${HCCL_CODE_ROOT}/test/stub/depends/include/
    ${HCCL_CODE_ROOT}/test/stub/depends/runtime/include/
    ${HCCL_CODE_ROOT}/test/stub/depends/include/mmpa/
    ${HCCL_CODE_ROOT}/test/stub/depends/include/driver/
    ${HCCL_CODE_ROOT}/test/stub/depends/include/acl/
    ${HCCL_CODE_ROOT}/test/stub/depends/include/ascendc/highlevel_api/
    ${HCCL_CODE_ROOT}/test/stub/depends/include/hcomm-legacy/
    ${HCCL_CODE_ROOT}/test/stub/depends/include/urma/

    # ascendc
    ${ASCEND_CANN_PACKAGE_PATH}/include/ascendc/
    ${ASCEND_CANN_PACKAGE_PATH}/include/ascendc/highlevel_api/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/asc/hccl/internal/

    # next (拆分到 base_comm 和 coll_communicator_mgr)
    ${HCCL_CODE_ROOT}/src/base_comm/
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/device
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/device/resource_mgr/local/my_rank/comm_engine_reses
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/device/resource_mgr/remote
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/device/c_adpt
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/remote/rank_pairs/
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/engine_ctxs
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_mems
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/endpoints
    ${HCCL_CODE_ROOT}/src/base_comm/common/
    ${HCCL_CODE_ROOT}/src/base_comm/common/debug
    ${HCCL_CODE_ROOT}/src/base_comm/common/device
    ${HCCL_CODE_ROOT}/src/base_comm/dfx/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoints/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoints/dfx
    ${HCCL_CODE_ROOT}/src/base_comm/resources/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/reged_mems/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoints/server_socket/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/sockets/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/channels/host/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res/engine_ctxs
    ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res/threads
    ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res/ccu
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/channels
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/channels/aicpu/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoints/net_dev
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/channels/aicpu/device
    ${HCCL_CODE_ROOT}/src/base_comm/primitives
    ${HCCL_CODE_ROOT}/src/base_comm/primitives/api_c_adpt
    ${HCCL_CODE_ROOT}/src/base_comm/primitives/api_c_adpt/nic_plugin
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoints/mgr
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/api_c_adpt/
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/ns_recovery
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/ns_recovery/aicpu

    ${HCCL_CODE_ROOT}/src/legacy/ascend950/interface/
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/cluster_monitor
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/ns_recovery
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/profiling
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/profiling/aicpu
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/profiling/host
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/taskException
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/taskException/aicpu
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/taskException/host
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/device/framework
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/dfx/task_exception
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/dfx/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/dfx/common
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/dfx/task_exception
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/dfx/profiling
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/dfx/aicpu
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/dfx/aicpu/task_exception
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/dfx/aicpu/profiling
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/dfx/aicpu/common
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/resource_manager/stream
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/stream
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/interface/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/dfx/aicpu/profiling
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/ccu/dfx/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/reps/common/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/ccu/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/ccu/ccu_device/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/ccu/ccu_device/ccu_component/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/ccu/ccu_device/ccu_component/ccu_channel/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/ccu/ccu_context/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/pub_inc/ccu/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/communicator/hostdpu/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/ccu/ccu_microcode/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/context/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/pub_inc/resource/transport/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/ccu_transport/

    ${HCCL_CODE_ROOT}/src/base_comm/resources/southbound_adpt/
    ${HCCL_CODE_ROOT}/src/base_comm/common/
    ${HCCL_CODE_ROOT}/src/base_comm/common/debug
    ${HCCL_CODE_ROOT}/src/base_comm/common/device
    ${HCCL_CODE_ROOT}/src/base_comm/dfx/

    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/pub_inc/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/ccu_comp/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/ccu_comp/ccu_channel/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/ccu_comp/ccu_channel/ccu_pfe/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/ccu_comp/ccu_channel/ccu_channel_ctx_v1/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/ccu_comp/ccu_channel/ccu_channel_ctx_v2/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_transport/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/channels/ccu/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/channels/aiv/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_kernel/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_instance/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_dfx
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_representation
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_representation/interface
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_representation/context
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_representation/reps
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_representation/reps/arithmetic
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_representation/reps/common
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_representation/reps/control
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_representation/reps/data
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_representation/reps/logical
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_representation/reps/loop
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_representation/reps/shift
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_representation/reps/sync
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_representation/reps/translator
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/common/loggers/
)

add_subdirectory(stub)
add_subdirectory(framework/next)
add_subdirectory(framework/next/coll_comms/dfx)
add_subdirectory(framework/next/comms/ccu)
add_subdirectory(common/prepare_ut_env)
add_subdirectory(common/binary_package)
add_subdirectory(common/debug/profiling)
add_subdirectory(experimental/base_comm/nic_plugin)
add_subdirectory(misc)
add_subdirectory(impl)
add_subdirectory(device)
add_subdirectory(platform/hcom)
add_subdirectory(platform/resource/dispatcher)
add_subdirectory(platform/resource/notify)
add_subdirectory(platform/resource/transport)
add_subdirectory(platform/ping_mesh)
add_subdirectory(platform/common/unfold_cache)
add_subdirectory(platform/common/device_capacity)
add_subdirectory(platform/common/mr_manager)
add_subdirectory(platform/task)
add_subdirectory(platform/task/task_logic)
add_subdirectory(framework/communicator)
add_subdirectory(framework/cluster_maintenance)
add_subdirectory(framework/op_base_api)
add_subdirectory(framework/hcom)
add_subdirectory(framework/group)
add_subdirectory(framework/communicator/impl)
add_subdirectory(framework/resource)
add_subdirectory(framework/communicator/impl/independent_op/data_api)
add_subdirectory(framework/communicator/impl/independent_op/data_api_cpu)
add_subdirectory(aicpu_kfc/aicpu)
add_subdirectory(aicpu_kfc/misc)
add_subdirectory(inter/comm)
add_subdirectory(inter/hcom)
add_subdirectory(platform/hccp)
add_subdirectory(dfx)
add_subdirectory(base_comm/primitives/aicpu/task_cache)
add_subdirectory(framework/nslbdp)
add_subdirectory(base_comm/primitives/api_c_adpt/nic_plugin)
add_subdirectory(framework/dpu_npu_sync)
